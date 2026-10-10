// SPDX-License-Identifier: GPL-2.0-or-later

//! The Hyper-V LAPIC as a device, from target/i386/whpx/whpx-apic.c: the state page that
//! `WHvGet/SetVirtualProcessorInterruptControllerState2()` move, the APIC base transitions
//! and MSI delivery through `WHvRequestInterrupt()`.

use crate::msr::{APICBASE_BASE, APICBASE_BSP, APICBASE_ENABLE, APICBASE_EXTD};

/// `APIC_LVT_NB`.
pub const LVT_NB: usize = 6;
/// The size of `struct whpx_lapic_state`: 256 registers, each padded to 16 bytes.
pub const PAGE_SIZE: usize = 4096;

/// The fields of `APICCommonState` the state page carries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LapicState {
    /// `apicbase`.
    pub apicbase: u64,
    /// `id`.
    pub id: u8,
    /// `initial_apic_id`, which is the ID in x2APIC mode.
    pub initial_apic_id: u32,
    /// `version`.
    pub version: u8,
    /// `arb_id`, read back only.
    pub arb_id: u8,
    /// `tpr`.
    pub tpr: u8,
    /// `log_dest`.
    pub log_dest: u8,
    /// `dest_mode`.
    pub dest_mode: u8,
    /// `spurious_vec`.
    pub spurious_vec: u32,
    /// `isr`.
    pub isr: [u32; 8],
    /// `tmr`.
    pub tmr: [u32; 8],
    /// `irr`.
    pub irr: [u32; 8],
    /// `esr`.
    pub esr: u32,
    /// `icr`.
    pub icr: [u32; 2],
    /// `lvt`.
    pub lvt: [u32; LVT_NB],
    /// `initial_count`.
    pub initial_count: u32,
    /// `divide_conf`.
    pub divide_conf: u32,
    /// `count_shift`, derived from `divide_conf` on the way back.
    pub count_shift: u32,
}

/// A write to `IA32_APIC_BASE` that the APIC mode rules forbid, which is #GP.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BadApicBase;

fn put(page: &mut [u8; PAGE_SIZE], index: usize, val: u32) {
    page[index * 16..index * 16 + 4].copy_from_slice(&val.to_le_bytes());
}

fn get(page: &[u8; PAGE_SIZE], index: usize) -> u32 {
    let mut b = [0; 4];
    b.copy_from_slice(&page[index * 16..index * 16 + 4]);
    u32::from_le_bytes(b)
}

/// `count_shift` for a divide configuration register value.
pub fn count_shift(divide_conf: u32) -> u32 {
    let v = (divide_conf & 3) | ((divide_conf >> 1) & 4);
    (v + 1) & 7
}

impl LapicState {
    /// `whpx_put_apic_state()`.
    pub fn to_page(&self) -> [u8; PAGE_SIZE] {
        let mut p = [0; PAGE_SIZE];
        let id = if self.apicbase & APICBASE_EXTD != 0 {
            self.initial_apic_id
        } else {
            u32::from(self.id) << 24
        };
        put(&mut p, 0x2, id);
        put(&mut p, 0x3, u32::from(self.version) | ((LVT_NB as u32 - 1) << 16));
        put(&mut p, 0x8, u32::from(self.tpr));
        put(&mut p, 0xd, u32::from(self.log_dest) << 24);
        put(&mut p, 0xe, u32::from(self.dest_mode) << 28 | 0x0fff_ffff);
        put(&mut p, 0xf, self.spurious_vec);
        for i in 0..8 {
            put(&mut p, 0x10 + i, self.isr[i]);
            put(&mut p, 0x18 + i, self.tmr[i]);
            put(&mut p, 0x20 + i, self.irr[i]);
        }
        put(&mut p, 0x28, self.esr);
        put(&mut p, 0x30, self.icr[0]);
        put(&mut p, 0x31, self.icr[1]);
        for (i, &lvt) in self.lvt.iter().enumerate() {
            put(&mut p, 0x32 + i, lvt);
        }
        put(&mut p, 0x38, self.initial_count);
        put(&mut p, 0x3e, self.divide_conf);
        p
    }

    /// `whpx_get_apic_state()`, except for restarting the timer, which the APIC device does
    /// with the new `initial_count` and `count_shift`.
    pub fn from_page(&mut self, p: &[u8; PAGE_SIZE]) {
        // In x2APIC mode the ID is the initial APIC ID and cannot change.
        if self.apicbase & APICBASE_EXTD == 0 {
            self.id = (get(p, 0x2) >> 24) as u8;
        }
        self.tpr = get(p, 0x8) as u8;
        self.arb_id = get(p, 0x9) as u8;
        self.log_dest = (get(p, 0xd) >> 24) as u8;
        self.dest_mode = (get(p, 0xe) >> 28) as u8;
        self.spurious_vec = get(p, 0xf);
        for i in 0..8 {
            self.isr[i] = get(p, 0x10 + i);
            self.tmr[i] = get(p, 0x18 + i);
            self.irr[i] = get(p, 0x20 + i);
        }
        self.esr = get(p, 0x28);
        self.icr = [get(p, 0x30), get(p, 0x31)];
        for i in 0..LVT_NB {
            self.lvt[i] = get(p, 0x32 + i);
        }
        self.initial_count = get(p, 0x38);
        self.divide_conf = get(p, 0x3e);
        self.count_shift = count_shift(self.divide_conf);
    }

    /// `apic_set_base()`. `x2apic` says whether the CPU model has x2APIC. On success the
    /// result says whether CPUID's APIC bit turns on or off.
    pub fn set_base(&mut self, val: u64, x2apic: bool) -> Result<Option<bool>, BadApicBase> {
        let old = self.apicbase;
        let en = |v: u64| v & APICBASE_ENABLE != 0;
        let extd = |v: u64| v & APICBASE_EXTD != 0;
        if (!x2apic && extd(val))
            || (!en(val) && extd(val))
            || (!en(old) && !extd(old) && en(val) && extd(val))
            || (en(old) && extd(old) && en(val) && !extd(val))
        {
            return Err(BadApicBase);
        }
        let mut feature = None;
        self.apicbase = (val & APICBASE_BASE) | (old & (APICBASE_BSP | APICBASE_ENABLE));
        if !en(val) {
            self.apicbase &= !APICBASE_ENABLE;
            feature = Some(false);
        }
        if !en(self.apicbase) && en(val) {
            self.apicbase |= APICBASE_ENABLE;
            feature = Some(true);
        }
        if x2apic && !extd(self.apicbase) && extd(val) {
            self.apicbase |= APICBASE_EXTD;
        }
        Ok(feature)
    }
}

/// `WHV_INTERRUPT_CONTROL` for one MSI.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InterruptControl {
    /// The delivery mode, which `WHV_INTERRUPT_TYPE` numbers the same way.
    pub kind: u8,
    /// Logical destination mode.
    pub logical: bool,
    /// Level triggered.
    pub level: bool,
    /// The destination APIC ID.
    pub destination: u32,
    /// The vector.
    pub vector: u32,
}

impl InterruptControl {
    /// The first 64 bits of the struct, the bitfields.
    pub fn bits(&self) -> u64 {
        u64::from(self.kind) | u64::from(self.logical) << 8 | u64::from(self.level) << 12
    }
}

/// `whpx_send_msi()` up to the call. Vector 0 is dropped with QEMU's warning, which the caller
/// prints.
pub fn decode_msi(addr: u64, data: u32) -> Result<InterruptControl, &'static str> {
    let vector = data & 0xff;
    if vector == 0 {
        return Err("Ignoring request for interrupt vector 0");
    }
    Ok(InterruptControl {
        kind: ((data >> 8) & 7) as u8,
        logical: (addr >> 2) & 1 != 0,
        level: (data >> 15) & 1 != 0,
        destination: ((addr & 0xff000) >> 12) as u32,
        vector,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_round_trip() {
        let s = LapicState {
            apicbase: 0xfee0_0900,
            id: 3,
            version: 0x14,
            tpr: 2,
            log_dest: 1,
            dest_mode: 0xf,
            spurious_vec: 0x1ff,
            isr: [1, 0, 0, 0, 0, 0, 0, 0x8000_0000],
            irr: [0, 2, 0, 0, 0, 0, 0, 0],
            icr: [0x4500, 0x0300_0000],
            lvt: [0x10000, 0x10000, 0x700, 0x400, 0x10000, 0xfe],
            initial_count: 0x1000,
            divide_conf: 0xb,
            ..LapicState::default()
        };
        let p = s.to_page();
        assert_eq!(get(&p, 0x2), 3 << 24);
        assert_eq!(get(&p, 0x3), 0x5_0014);
        assert_eq!(get(&p, 0xe), 0xffff_ffff);
        let mut back =
            LapicState { apicbase: s.apicbase, version: s.version, ..LapicState::default() };
        back.from_page(&p);
        assert_eq!(back, LapicState { count_shift: 0, ..s });
        assert_eq!(count_shift(0b0011), 4);
        assert_eq!(count_shift(0b1000), 5);
    }

    #[test]
    fn x2apic_id() {
        let s = LapicState {
            apicbase: 0xfee0_0d00,
            id: 1,
            initial_apic_id: 260,
            ..LapicState::default()
        };
        assert_eq!(get(&s.to_page(), 0x2), 260);
    }

    #[test]
    fn base_transitions() {
        let mut s = LapicState { apicbase: 0xfee0_0900, ..LapicState::default() };
        assert_eq!(s.set_base(0xfee0_0d00, false), Err(BadApicBase));
        assert_eq!(s.set_base(0xfee0_0d00, true), Ok(None));
        assert_eq!(s.apicbase, 0xfee0_0d00);
        // x2APIC back to xAPIC is refused, disabling is fine.
        assert_eq!(s.set_base(0xfee0_0900, true), Err(BadApicBase));
        assert_eq!(s.set_base(0xfee0_0100, true), Ok(Some(false)));
        assert_eq!(s.apicbase, 0xfee0_0100);
        // Disabled straight to x2APIC is refused.
        assert_eq!(s.set_base(0xfee0_0d00, true), Err(BadApicBase));
        assert_eq!(s.set_base(0xfee0_0900, true), Ok(Some(true)));
    }

    #[test]
    fn msi() {
        let c = decode_msi(0xfee0_3004, 0x8031).unwrap();
        assert_eq!(
            c,
            InterruptControl { kind: 0, logical: true, level: true, destination: 3, vector: 0x31 }
        );
        assert_eq!(c.bits(), 0x1100);
        assert_eq!(decode_msi(0xfee0_0000, 0x430).unwrap().kind, 4);
        assert!(decode_msi(0xfee0_0000, 0x100).is_err());
    }
}
