// SPDX-License-Identifier: GPL-2.0-or-later

//! Physical memory protection, a port of QEMU's `target/riscv/tcg/pmp.c`.
//!
//! The model is QEMU's default `rv64`: 16 PMP entries, a granularity of 4 bytes (G = 0),
//! and none of Smepmp (`mseccfg`), Smpmpmt or Smmpm. Each `pmpcfg` byte lives in its own
//! slot of [`CpuRiscvState::pmpcfg`], and the decoded address range of every entry in
//! `pmp_sa` and `pmp_ea`, as `pmp_state.addr[]` holds it in QEMU.
//!
//! Differences from QEMU:
//!
//! - A `pmpcfg` byte with W set and R clear (a reserved encoding without Smepmp) is not
//!   written; the entry keeps its old configuration. QEMU 11.1 stores the reserved value
//!   as written.
//! - The write functions return whether the TLB has to be flushed, and the CSR layer
//!   flushes it, where QEMU calls `tlb_flush()` from here.

use ruvm_jit::page;

use crate::cpu::{CpuRiscvState, PMP_REGIONS, PRV_M};

/// `PMP_READ`.
pub(crate) const PMP_READ: u32 = 1 << 0;
/// `PMP_WRITE`.
pub(crate) const PMP_WRITE: u32 = 1 << 1;
/// `PMP_EXEC`.
pub(crate) const PMP_EXEC: u32 = 1 << 2;
/// `PMP_AMATCH`: the address matching mode field of a `pmpcfg` byte.
const PMP_AMATCH: u64 = 3 << 3;
/// `PMP_MTMATCH`: the Smpmpmt memory type field, always zero here.
const PMP_MTMATCH: u64 = 3 << 5;
/// `PMP_LOCK`.
const PMP_LOCK: u64 = 1 << 7;

/// `PMP_AMATCH_OFF`: the entry is disabled.
const AMATCH_OFF: u64 = 0;
/// `PMP_AMATCH_TOR`: top of range.
const AMATCH_TOR: u64 = 1;
/// `PMP_AMATCH_NA4`: a naturally aligned four byte region.
const AMATCH_NA4: u64 = 2;
/// `PMP_AMATCH_NAPOT`: a naturally aligned power of two region.
const AMATCH_NAPOT: u64 = 3;

/// `TARGET_PAGE_SIZE`.
const PAGE_SIZE: u64 = 4096;

/// `pmp_get_a_field()`.
fn a_field(cfg: u64) -> u64 {
    (cfg & PMP_AMATCH) >> 3
}

/// `pmp_is_locked()`.
fn is_locked(st: &CpuRiscvState, i: usize) -> bool {
    st.pmpcfg[i] & PMP_LOCK != 0
}

/// `pmp_is_readonly()`: locked, as there is no `mseccfg.RLB`.
fn is_readonly(st: &CpuRiscvState, i: usize) -> bool {
    is_locked(st, i)
}

/// `pmp_decode_napot()`: the first and last address of the NAPOT range `a` encodes.
fn decode_napot(a: u64) -> (u64, u64) {
    let a = (a << 2) | 3;
    (a & a.wrapping_add(1), a | a.wrapping_add(1))
}

/// `pmp_update_rule_addr()`: decode the address range of entry `i`.
fn update_rule_addr(st: &mut CpuRiscvState, i: usize) {
    let this_cfg = st.pmpcfg[i];
    let this_addr = st.pmpaddr[i];
    let prev_addr = if i >= 1 { st.pmpaddr[i - 1] } else { 0 };
    // The granularity is 4 bytes (G = 0), so no address bits are ignored.
    let (sa, ea) = match a_field(this_cfg) {
        AMATCH_OFF => (0, u64::MAX),
        AMATCH_TOR => {
            if prev_addr >= this_addr {
                (0, 0)
            } else {
                (prev_addr << 2, (this_addr << 2).wrapping_sub(1))
            }
        }
        AMATCH_NA4 => {
            let sa = this_addr << 2;
            (sa, sa.wrapping_add(4).wrapping_sub(1))
        }
        AMATCH_NAPOT => decode_napot(this_addr),
        _ => (0, 0),
    };
    st.pmp_sa[i] = sa;
    st.pmp_ea[i] = ea;
}

/// `pmp_update_rule_nums()`: count the entries that are not off.
fn update_rule_nums(st: &mut CpuRiscvState) {
    st.pmp_num_rules = st.pmpcfg.iter().filter(|&&c| a_field(c) != AMATCH_OFF).count() as u64;
}

/// `pmp_update_rule_addr()` of every entry and `pmp_update_rule_nums()`: rebuild the
/// decoded rules from `pmpcfg` and `pmpaddr`, as after loading a saved state.
pub(crate) fn update_rules(st: &mut CpuRiscvState) {
    for i in 0..PMP_REGIONS {
        update_rule_addr(st, i);
    }
    update_rule_nums(st);
}

/// `pmp_is_in_range()`.
fn is_in_range(st: &CpuRiscvState, i: usize, addr: u64) -> bool {
    addr >= st.pmp_sa[i] && addr <= st.pmp_ea[i]
}

/// `pmp_priv_to_page_prot()`.
fn priv_to_page_prot(privs: u32) -> u32 {
    let mut prot = 0;
    if privs & PMP_READ != 0 {
        prot |= page::READ;
    }
    if privs & PMP_WRITE != 0 {
        prot |= page::WRITE;
    }
    if privs & PMP_EXEC != 0 {
        prot |= page::EXEC;
    }
    prot
}

/// `pmp_hart_has_privs_default()`: the privileges when no entry matches. M mode gets
/// everything; S and U mode get nothing, as the hart implements PMP.
fn has_privs_default(mode: u64) -> Option<u32> {
    (mode == PRV_M).then_some(PMP_READ | PMP_WRITE | PMP_EXEC)
}

/// `pmp_hart_has_privs()` and `pmp_priv_to_page_prot()`: the page protection PMP allows
/// for `size` bytes at `addr` in privilege `mode`, or `None` if an access needing `privs`
/// (`PMP_READ`, `PMP_WRITE` or `PMP_EXEC`) is denied. A `size` of 0 means the rest of
/// the page.
pub(crate) fn hart_has_privs(
    st: &CpuRiscvState,
    addr: u64,
    size: u64,
    privs: u32,
    mode: u64,
) -> Option<u32> {
    let allowed = if st.pmp_num_rules == 0 {
        has_privs_default(mode)?
    } else {
        rule_privs(st, addr, size, mode)?
    };
    (privs & allowed == privs).then(|| priv_to_page_prot(allowed))
}

/// The privileges of the first entry that matches `size` bytes at `addr`, or the default
/// ones. `None` if an entry matches only part of the access, or no entry matches an S or
/// U mode access.
fn rule_privs(st: &CpuRiscvState, addr: u64, size: u64, mode: u64) -> Option<u32> {
    // An unknown size covers everything from addr to the end of the page.
    let size = if size == 0 { (addr | !(PAGE_SIZE - 1)).wrapping_neg() } else { size };
    let last = addr.wrapping_add(size).wrapping_sub(1);
    // The entries have an implicit priority from low to high.
    for i in 0..PMP_REGIONS {
        let s = is_in_range(st, i, addr);
        let e = is_in_range(st, i, last);
        if s != e {
            // The access is partially inside the entry.
            return None;
        }
        if s && a_field(st.pmpcfg[i]) != AMATCH_OFF {
            let mut allowed = PMP_READ | PMP_WRITE | PMP_EXEC;
            if mode != PRV_M || is_locked(st, i) {
                allowed &= st.pmpcfg[i] as u32;
            }
            // A matching entry decides; there is no fall back to the default.
            return Some(allowed);
        }
    }
    has_privs_default(mode)
}

/// `pmp_write_cfg()`: write the configuration byte of entry `i`, returning whether it
/// changed.
fn write_cfg(st: &mut CpuRiscvState, i: usize, val: u8) -> bool {
    if i >= PMP_REGIONS {
        // Out of bounds writes are ignored.
        return false;
    }
    let mut val = u64::from(val);
    if st.pmpcfg[i] == val {
        return false;
    }
    if is_readonly(st, i) {
        // The entry is locked.
        return false;
    }
    let rw = u64::from(PMP_READ | PMP_WRITE);
    if val & rw == u64::from(PMP_WRITE) {
        // R = 0 and W = 1 is reserved without Smepmp; keep the old configuration.
        return false;
    }
    // There is no Smpmpmt, so the memory type field reads as zero.
    val &= !PMP_MTMATCH;
    st.pmpcfg[i] = val;
    update_rule_addr(st, i);
    true
}

/// `pmpcfg_csr_write()`: write `pmpcfg<reg_index>`, returning whether a rule changed and
/// the TLB has to be flushed.
pub(crate) fn pmpcfg_csr_write(st: &mut CpuRiscvState, reg_index: usize, val: u64) -> bool {
    let mut modified = false;
    // RV64 packs eight entries into each even numbered pmpcfg register.
    for i in 0..8 {
        let cfg = (val >> (8 * i)) as u8;
        modified |= write_cfg(st, reg_index * 4 + i, cfg);
    }
    if modified {
        update_rule_nums(st);
    }
    modified
}

/// `pmpcfg_csr_read()`: read `pmpcfg<reg_index>`.
pub(crate) fn pmpcfg_csr_read(st: &CpuRiscvState, reg_index: usize) -> u64 {
    (0..8).fold(0, |acc, i| {
        let e = reg_index * 4 + i;
        let cfg = if e < PMP_REGIONS { st.pmpcfg[e] & 0xff } else { 0 };
        acc | (cfg << (8 * i))
    })
}

/// `pmpaddr_csr_write()`: write `pmpaddr<i>`, returning whether a rule changed and the
/// TLB has to be flushed.
pub(crate) fn pmpaddr_csr_write(st: &mut CpuRiscvState, i: usize, val: u64) -> bool {
    if i >= PMP_REGIONS {
        // Out of bounds writes are ignored.
        return false;
    }
    if st.pmpaddr[i] == val {
        return false;
    }
    // In TOR mode the next entry uses this address too, so its lock applies.
    let mut is_next_cfg_tor = false;
    if i + 1 < PMP_REGIONS {
        is_next_cfg_tor = a_field(st.pmpcfg[i + 1]) == AMATCH_TOR;
        if is_readonly(st, i + 1) && is_next_cfg_tor {
            return false;
        }
    }
    if is_readonly(st, i) {
        return false;
    }
    st.pmpaddr[i] = val;
    update_rule_addr(st, i);
    if is_next_cfg_tor {
        update_rule_addr(st, i + 1);
    }
    true
}

/// `pmpaddr_csr_read()`: read `pmpaddr<i>`. With G = 0 no bits read differently from
/// what was written.
pub(crate) fn pmpaddr_csr_read(st: &CpuRiscvState, i: usize) -> u64 {
    if i < PMP_REGIONS { st.pmpaddr[i] } else { 0 }
}

/// `pmp_get_tlb_size()`: the size of the TLB entry the page at `addr` may use: a whole
/// page, or 1 (do not cache) if the first entry that touches the page covers only part
/// of it.
pub(crate) fn get_tlb_size(st: &CpuRiscvState, addr: u64) -> u64 {
    if st.pmp_num_rules == 0 {
        return PAGE_SIZE;
    }
    let tlb_sa = addr & !(PAGE_SIZE - 1);
    let tlb_ea = tlb_sa + (PAGE_SIZE - 1);
    for i in 0..PMP_REGIONS {
        if a_field(st.pmpcfg[i]) == AMATCH_OFF {
            continue;
        }
        let (sa, ea) = (st.pmp_sa[i], st.pmp_ea[i]);
        if sa <= tlb_sa && ea >= tlb_ea {
            return PAGE_SIZE;
        } else if (sa >= tlb_sa && sa <= tlb_ea) || (ea >= tlb_sa && ea <= tlb_ea) {
            return 1;
        }
    }
    PAGE_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{PRV_S, PRV_U};

    const R: u64 = 1;
    const W: u64 = 2;
    const X: u64 = 4;
    const TOR: u64 = AMATCH_TOR << 3;
    const NA4: u64 = AMATCH_NA4 << 3;
    const NAPOT: u64 = AMATCH_NAPOT << 3;
    const L: u64 = PMP_LOCK;
    const RWX: u32 = page::READ | page::WRITE | page::EXEC;

    fn st() -> CpuRiscvState {
        let mut s = CpuRiscvState::default();
        update_rules(&mut s);
        s
    }

    /// The pmpaddr value of a NAPOT region of `size` bytes at `base`.
    fn napot(base: u64, size: u64) -> u64 {
        (base >> 2) | ((size >> 3) - 1)
    }

    #[test]
    fn no_rules() {
        let s = st();
        assert_eq!(s.pmp_num_rules, 0);
        assert_eq!(hart_has_privs(&s, 0x8000_0000, 8, PMP_EXEC, PRV_M), Some(RWX));
        assert_eq!(hart_has_privs(&s, 0x8000_0000, 8, PMP_READ, PRV_S), None);
        assert_eq!(get_tlb_size(&s, 0x8000_0000), PAGE_SIZE);
    }

    #[test]
    fn napot_rule() {
        let mut s = st();
        assert!(pmpaddr_csr_write(&mut s, 0, napot(0x8000_0000, 0x1_0000)));
        assert!(pmpcfg_csr_write(&mut s, 0, NAPOT | R | X));
        assert_eq!(s.pmp_num_rules, 1);
        assert_eq!((s.pmp_sa[0], s.pmp_ea[0]), (0x8000_0000, 0x8000_ffff));
        let rx = page::READ | page::EXEC;
        assert_eq!(hart_has_privs(&s, 0x8000_1000, 4, PMP_READ, PRV_S), Some(rx));
        assert_eq!(hart_has_privs(&s, 0x8000_1000, 4, PMP_WRITE, PRV_U), None);
        // Outside the region S mode has nothing; M mode has everything.
        assert_eq!(hart_has_privs(&s, 0x8001_0000, 4, PMP_READ, PRV_S), None);
        assert_eq!(hart_has_privs(&s, 0x8001_0000, 4, PMP_WRITE, PRV_M), Some(RWX));
        // An unlocked rule does not apply to M mode.
        assert_eq!(hart_has_privs(&s, 0x8000_1000, 4, PMP_WRITE, PRV_M), Some(RWX));
        // An access straddling the end of the region fails, even for M mode.
        assert_eq!(hart_has_privs(&s, 0x8000_fffc, 8, PMP_READ, PRV_M), None);
        // The whole page is covered, so it may be cached.
        assert_eq!(get_tlb_size(&s, 0x8000_1234), PAGE_SIZE);
        // pmpaddr reads back as written.
        assert_eq!(pmpaddr_csr_read(&s, 0), napot(0x8000_0000, 0x1_0000));
        assert_eq!(pmpcfg_csr_read(&s, 0), NAPOT | R | X);
    }

    #[test]
    fn napot_whole_address_space() {
        let mut s = st();
        assert!(pmpaddr_csr_write(&mut s, 0, u64::MAX >> 10));
        assert!(pmpcfg_csr_write(&mut s, 0, NAPOT | R | W | X));
        assert_eq!(s.pmp_sa[0], 0);
        assert_eq!(hart_has_privs(&s, 0xffff_ffff_0000, 8, PMP_WRITE, PRV_U), Some(RWX));
    }

    #[test]
    fn tor_rule() {
        let mut s = st();
        // Entry 0 is off and only gives the bottom of entry 1's range.
        assert!(pmpaddr_csr_write(&mut s, 0, 0x1000 >> 2));
        assert!(pmpaddr_csr_write(&mut s, 1, 0x1800 >> 2));
        assert!(pmpcfg_csr_write(&mut s, 0, (TOR | R | W) << 8));
        assert_eq!(s.pmp_num_rules, 1);
        assert_eq!((s.pmp_sa[1], s.pmp_ea[1]), (0x1000, 0x17ff));
        let rw = page::READ | page::WRITE;
        assert_eq!(hart_has_privs(&s, 0x1000, 8, PMP_WRITE, PRV_S), Some(rw));
        assert_eq!(hart_has_privs(&s, 0x17f8, 8, PMP_EXEC, PRV_S), None);
        assert_eq!(hart_has_privs(&s, 0x1800, 8, PMP_READ, PRV_S), None);
        assert_eq!(hart_has_privs(&s, 0xff8, 8, PMP_READ, PRV_S), None);
        // The rule covers half the page, so the page must not be cached.
        assert_eq!(get_tlb_size(&s, 0x1000), 1);
        // Moving the bottom moves the TOR range.
        assert!(pmpaddr_csr_write(&mut s, 0, 0x800 >> 2));
        assert_eq!(s.pmp_sa[1], 0x800);
        // An empty TOR range matches nothing.
        assert!(pmpaddr_csr_write(&mut s, 0, 0x2000 >> 2));
        assert_eq!((s.pmp_sa[1], s.pmp_ea[1]), (0, 0));
    }

    #[test]
    fn na4_rule() {
        let mut s = st();
        assert!(pmpaddr_csr_write(&mut s, 3, 0x2004 >> 2));
        assert!(pmpcfg_csr_write(&mut s, 0, (NA4 | R) << 24));
        assert_eq!((s.pmp_sa[3], s.pmp_ea[3]), (0x2004, 0x2007));
        assert_eq!(hart_has_privs(&s, 0x2004, 4, PMP_READ, PRV_U), Some(page::READ));
        assert_eq!(hart_has_privs(&s, 0x2004, 8, PMP_READ, PRV_U), None);
        assert_eq!(hart_has_privs(&s, 0x2000, 4, PMP_READ, PRV_U), None);
        assert_eq!(get_tlb_size(&s, 0x2000), 1);
        assert_eq!(get_tlb_size(&s, 0x3000), PAGE_SIZE);
    }

    #[test]
    fn priority_is_lowest_entry_first() {
        let mut s = st();
        // Entry 0: 8 bytes at 0x8000_0008, read only. Entry 1: 4 KiB at 0x8000_0000, RWX.
        assert!(pmpaddr_csr_write(&mut s, 0, napot(0x8000_0008, 8)));
        assert!(pmpaddr_csr_write(&mut s, 1, napot(0x8000_0000, 0x1000)));
        assert!(pmpcfg_csr_write(&mut s, 0, (NAPOT | R) | ((NAPOT | R | W | X) << 8)));
        assert_eq!(hart_has_privs(&s, 0x8000_0008, 8, PMP_WRITE, PRV_S), None);
        assert_eq!(hart_has_privs(&s, 0x8000_0000, 8, PMP_WRITE, PRV_S), Some(RWX));
        assert_eq!(get_tlb_size(&s, 0x8000_0000), 1);
    }

    #[test]
    fn lock() {
        let mut s = st();
        assert!(pmpaddr_csr_write(&mut s, 0, napot(0x8000_0000, 0x1000)));
        assert!(pmpcfg_csr_write(&mut s, 0, NAPOT | R | L));
        // A locked rule applies to M mode too.
        assert_eq!(hart_has_privs(&s, 0x8000_0000, 8, PMP_WRITE, PRV_M), None);
        assert_eq!(hart_has_privs(&s, 0x8000_0000, 8, PMP_READ, PRV_M), Some(page::READ));
        // A locked entry ignores writes to its configuration and address.
        assert!(!pmpcfg_csr_write(&mut s, 0, NAPOT | R | W | X));
        assert_eq!(pmpcfg_csr_read(&s, 0), NAPOT | R | L);
        assert!(!pmpaddr_csr_write(&mut s, 0, 0));
        assert_eq!(pmpaddr_csr_read(&s, 0), napot(0x8000_0000, 0x1000));
        // The other bytes of the register are still written.
        assert!(pmpcfg_csr_write(&mut s, 0, (NAPOT | R | L) | ((NA4 | X) << 8)));
        assert_eq!(pmpcfg_csr_read(&s, 0), (NAPOT | R | L) | ((NA4 | X) << 8));
    }

    #[test]
    fn locked_tor_locks_previous_address() {
        let mut s = st();
        assert!(pmpaddr_csr_write(&mut s, 1, 0x2000 >> 2));
        assert!(pmpcfg_csr_write(&mut s, 0, (TOR | R | L) << 8));
        // pmpaddr0 is the bottom of locked entry 1, so it cannot change.
        assert!(!pmpaddr_csr_write(&mut s, 0, 0x1000 >> 2));
        assert_eq!(pmpaddr_csr_read(&s, 0), 0);
        // With entry 1 in NAPOT mode, pmpaddr0 is free again; check on a fresh state.
        let mut s = st();
        assert!(pmpcfg_csr_write(&mut s, 0, (NAPOT | R | L) << 8));
        assert!(pmpaddr_csr_write(&mut s, 0, 0x1000 >> 2));
    }

    #[test]
    fn reserved_write_only_encoding() {
        let mut s = st();
        assert!(!pmpcfg_csr_write(&mut s, 0, NAPOT | W));
        assert_eq!(pmpcfg_csr_read(&s, 0), 0);
        assert_eq!(s.pmp_num_rules, 0);
        // The memory type bits read as zero.
        assert!(pmpcfg_csr_write(&mut s, 0, NAPOT | R | PMP_MTMATCH));
        assert_eq!(pmpcfg_csr_read(&s, 0), NAPOT | R);
    }

    #[test]
    fn out_of_range_registers() {
        let mut s = st();
        // pmpcfg4 and up would hold entries 16 and up, which do not exist.
        assert!(!pmpcfg_csr_write(&mut s, 4, u64::MAX >> 1));
        assert_eq!(pmpcfg_csr_read(&s, 4), 0);
        assert!(!pmpaddr_csr_write(&mut s, 16, 5));
        assert_eq!(pmpaddr_csr_read(&s, 16), 0);
        // pmpcfg2 holds entries 8 to 15.
        assert!(pmpcfg_csr_write(&mut s, 2, (NA4 | R) << 56));
        assert_eq!(s.pmpcfg[15], NA4 | R);
    }

    #[test]
    fn update_rules_rebuilds_ranges() {
        let mut s = CpuRiscvState::default();
        s.pmpcfg[2] = NA4 | R;
        s.pmpaddr[2] = 0x40;
        update_rules(&mut s);
        assert_eq!(s.pmp_num_rules, 1);
        assert_eq!((s.pmp_sa[2], s.pmp_ea[2]), (0x100, 0x103));
        assert_eq!((s.pmp_sa[0], s.pmp_ea[0]), (0, u64::MAX));
    }
}
