// SPDX-License-Identifier: GPL-2.0-or-later

//! The GICv3 ITS, from hw/intc/arm_gicv3_its.c, hw/intc/arm_gicv3_its_common.c and the ITS
//! parts of hw/intc/gicv3_internal.h.
//!
//! The ITS has two MMIO frames. The control frame ([`GicV3Its::control_ops`]) holds GITS_CTLR,
//! GITS_TYPER, the command queue registers and the GITS_BASER<n> table registers. The
//! translation frame ([`GicV3Its::translation_ops`]) holds GITS_TRANSLATER, where a device
//! writes an event ID to raise an MSI. The device is identified by the requester ID of the
//! write, so a board that wires PCI devices to the ITS must have their MSI writes carry it.
//!
//! The device and collection tables, the interrupt translation tables and the command queue all
//! live in guest memory, the address space the GIC was given as `sysmem`. Nothing is cached, so
//! the commands that only exist to flush caches (SYNC, INV and INVALL) just make the
//! redistributors rescan their LPIs.
//!
//! Like QEMU, the model supports physical LPIs only: GITS_TYPER.Virtual is 0, the virtual
//! commands are skipped and GITS_BASER2 to GITS_BASER7 are unimplemented. GITS_TYPER.PTA is 0,
//! so a target redistributor is named by its processor number.
//!
//! # Differences from QEMU
//!
//! - The command queue is read with an ordinary memory read rather than `address_space_map()`,
//!   so a command in MMIO is read instead of stalling the queue.
//! - Guest errors that QEMU logs with `LOG_GUEST_ERROR` are silent.
//! - There is no VMState.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, MemResult, MemTxAttrs, MemTxResult,
    MmioOps,
};

use super::{
    Engaged, GICR_TYPER_PLPIS, GICV3_IIDR, GICV3_LPI_INTID_START, GicV3, deposit_half, engaged,
};

/// The size of the control frame.
pub const ITS_CONTROL_SIZE: u64 = 0x10000;
/// The size of the translation frame.
pub const ITS_TRANS_SIZE: u64 = 0x10000;
/// The size of the whole ITS, `ITS_SIZE`.
pub const ITS_SIZE: u64 = ITS_CONTROL_SIZE + ITS_TRANS_SIZE;

const GITS_CTLR: u64 = 0x0;
const GITS_IIDR: u64 = 0x4;
const GITS_TYPER: u64 = 0x8;
const GITS_CBASER: u64 = 0x80;
const GITS_CWRITER: u64 = 0x88;
const GITS_CREADR: u64 = 0x90;
const GITS_BASER: u64 = 0x100;
const GITS_IDREGS: u64 = 0xffd0;
const GITS_TRANSLATER: u64 = 0x40;

const GITS_CTLR_ENABLED: u32 = 1 << 0;
const GITS_CTLR_QUIESCENT: u32 = 1 << 31;
const GITS_CREADR_STALLED: u64 = 1 << 0;
const GITS_CWRITER_RETRY: u64 = 1 << 0;
/// The OFFSET field of GITS_CREADR and GITS_CWRITER, bits 5 to 19.
const GITS_CQ_OFFSET_SHIFT: u32 = 5;
const GITS_CQ_OFFSET_MASK: u64 = 0x7fff;

/// GITS_BASER.ENTRYSIZE, bits 48 to 52, and GITS_BASER.TYPE, bits 56 to 58, are read only.
const GITS_BASER_RO_MASK: u64 = (0x1f << 48) | (0x7 << 56);
const GITS_BASER_TYPE_DEVICE: u64 = 1;
const GITS_BASER_TYPE_COLLECTION: u64 = 4;
const GITS_BASER_PAGESIZE_64K: u64 = 2;
const GITS_BASER_VALID: u64 = 1 << 63;
const GITS_BASER_INDIRECT: u64 = 1 << 62;
const GITS_CBASER_VALID: u64 = 1 << 63;

const GITS_PAGE_SIZE_4K: u32 = 0x1000;
const GITS_PAGE_SIZE_16K: u32 = 0x4000;
const GITS_PAGE_SIZE_64K: u32 = 0x10000;
/// `GITS_DTE_SIZE` and `GITS_CTE_SIZE`.
const GITS_TABLE_ENTRY_SIZE: u64 = 8;
const L1TABLE_ENTRY_SIZE: u32 = 8;
const L2_TABLE_VALID: u64 = 1 << 63;

const GITS_CMDQ_ENTRY_SIZE: u64 = 32;
const GITS_CMD_MOVI: u8 = 0x01;
const GITS_CMD_INT: u8 = 0x03;
const GITS_CMD_CLEAR: u8 = 0x04;
const GITS_CMD_SYNC: u8 = 0x05;
const GITS_CMD_MAPD: u8 = 0x08;
const GITS_CMD_MAPC: u8 = 0x09;
const GITS_CMD_MAPTI: u8 = 0x0a;
const GITS_CMD_MAPI: u8 = 0x0b;
const GITS_CMD_INV: u8 = 0x0c;
const GITS_CMD_INVALL: u8 = 0x0d;
const GITS_CMD_MOVALL: u8 = 0x0e;
const GITS_CMD_DISCARD: u8 = 0x0f;
const GITS_CMD_VMOVI: u8 = 0x21;
const GITS_CMD_VMOVP: u8 = 0x22;
const GITS_CMD_VSYNC: u8 = 0x25;
const GITS_CMD_VMAPP: u8 = 0x29;
const GITS_CMD_VMAPTI: u8 = 0x2a;
const GITS_CMD_VMAPI: u8 = 0x2b;
const GITS_CMD_VINVALL: u8 = 0x2d;

/// `ITS_ITT_ENTRY_SIZE`: the bytes of one interrupt translation table entry.
const ITS_ITT_ENTRY_SIZE: u64 = 0xc;
/// `ITS_IDBITS`, `ITS_DEVBITS` and `ITS_CIDBITS`, each one less than the bits supported.
const ITS_IDBITS: u64 = 0xf;
const ITS_DEVBITS: u64 = 0xf;
const ITS_CIDBITS: u64 = 0xf;
const GICV3_PIDR0_ITS: u32 = 0x94;
/// `INTID_SPURIOUS`, the doorbell that means none.
const NO_DOORBELL: u32 = 1023;

const CMD_FIELD_VALID: u64 = 1 << 63;
/// MAPD's ITT_addr, bits 8 to 51.
const ITTADDR_MASK: u64 = ((1 << 44) - 1) << 8;
const ITTADDR_SHIFT: u32 = 8;

const ITE_INTTYPE_PHYSICAL: u8 = 1;

/// `intid_in_lpi_range()`.
fn intid_in_lpi_range(id: u32) -> bool {
    (GICV3_LPI_INTID_START..(1 << (ITS_IDBITS + 1))).contains(&id)
}

/// `baser_base_addr()`.
fn baser_base_addr(value: u64, page_sz: u32) -> u64 {
    match page_sz {
        GITS_PAGE_SIZE_4K | GITS_PAGE_SIZE_16K => ((value >> 12) & ((1 << 36) - 1)) << 12,
        GITS_PAGE_SIZE_64K => (((value >> 16) & 0xffff_ffff) << 16) | (((value >> 12) & 0xf) << 48),
        _ => 0,
    }
}

/// What a command did, `ItsCmdResult`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CmdResult {
    /// A memory error: stop the queue here.
    Stall,
    /// The command was bad and was skipped.
    Continue,
    /// The command worked.
    ContinueOk,
}

/// The flavour of an INT, CLEAR, DISCARD or GITS_TRANSLATER write, `ItsCmdType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CmdType {
    None,
    Interrupt,
    Clear,
    Discard,
}

/// One of the tables in guest memory, `TableDesc`.
#[derive(Clone, Copy, Debug, Default)]
struct TableDesc {
    indirect: bool,
    entry_sz: u16,
    page_sz: u32,
    num_entries: u32,
    base_addr: u64,
}

/// The command queue, `CmdQDesc`.
#[derive(Clone, Copy, Debug, Default)]
struct CmdQDesc {
    num_entries: u32,
    base_addr: u64,
}

/// A device table entry, `DTEntry`.
#[derive(Clone, Copy, Debug, Default)]
struct DtEntry {
    valid: bool,
    size: u8,
    ittaddr: u64,
}

/// A collection table entry, `CTEntry`.
#[derive(Clone, Copy, Debug, Default)]
struct CtEntry {
    valid: bool,
    rdbase: u32,
}

/// An interrupt translation table entry, `ITEntry`.
#[derive(Clone, Copy, Debug, Default)]
struct ItEntry {
    valid: bool,
    inttype: u8,
    intid: u32,
    icid: u16,
    vpeid: u16,
    doorbell: u32,
}

/// The guest visible state, the registers of `GICv3ITSState`.
#[derive(Debug)]
struct ItsState {
    ctlr: u32,
    cbaser: u64,
    cwriter: u64,
    creadr: u64,
    baser: [u64; 8],
    dt: TableDesc,
    ct: TableDesc,
    cq: CmdQDesc,
}

/// The `arm-gicv3-its` device.
pub struct GicV3Its {
    gic: Arc<GicV3>,
    typer: u64,
    state: Mutex<ItsState>,
}

impl fmt::Debug for GicV3Its {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GicV3Its").field("typer", &self.typer).finish_non_exhaustive()
    }
}

/// The ITS state together with what the commands need, for the length of one access.
struct Exec<'a> {
    gic: &'a GicV3,
    dma: Option<Arc<AddressSpace>>,
    s: &'a mut ItsState,
}

impl Exec<'_> {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemTxResult> {
        let Some(a) = &self.dma else {
            return Err(MemTxResult::DECODE_ERROR);
        };
        let r = a.read(addr, MemTxAttrs::UNSPECIFIED, buf);
        if r.is_ok() { Ok(()) } else { Err(r) }
    }

    fn write(&self, addr: u64, buf: &[u8]) -> Result<(), MemTxResult> {
        let Some(a) = &self.dma else {
            return Err(MemTxResult::DECODE_ERROR);
        };
        let r = a.write(addr, MemTxAttrs::UNSPECIFIED, buf);
        if r.is_ok() { Ok(()) } else { Err(r) }
    }

    /// `address_space_ldq_le()`.
    fn ldq(&self, addr: u64) -> Result<u64, MemTxResult> {
        let mut b = [0u8; 8];
        self.read(addr, &mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    /// `address_space_ldl_le()`.
    fn ldl(&self, addr: u64) -> Result<u32, MemTxResult> {
        let mut b = [0u8; 4];
        self.read(addr, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    /// `table_entry_addr()`: where entry `idx` of `td` is. `None` when the level 1 entry of an
    /// indirect table is not valid.
    fn table_entry_addr(&self, td: &TableDesc, idx: u32) -> Result<Option<u64>, MemTxResult> {
        let entry_sz = u32::from(td.entry_sz);
        if !td.indirect {
            return Ok(Some(td.base_addr.wrapping_add(u64::from(idx.wrapping_mul(entry_sz)))));
        }
        let l2idx = idx / (td.page_sz / L1TABLE_ENTRY_SIZE);
        let l2 =
            self.ldq(td.base_addr.wrapping_add(u64::from(l2idx.wrapping_mul(L1TABLE_ENTRY_SIZE))))?;
        if l2 & L2_TABLE_VALID == 0 {
            return Ok(None);
        }
        let num_l2_entries = td.page_sz / entry_sz;
        let off = (idx % num_l2_entries).wrapping_mul(entry_sz);
        Ok(Some((l2 & ((1 << 51) - 1)).wrapping_add(u64::from(off))))
    }

    /// `get_cte()`.
    fn get_cte(&self, icid: u16) -> Result<CtEntry, MemTxResult> {
        let ct = self.s.ct;
        let Some(addr) = self.table_entry_addr(&ct, u32::from(icid))? else {
            return Ok(CtEntry::default());
        };
        let v = self.ldq(addr)?;
        Ok(CtEntry { valid: v & 1 != 0, rdbase: ((v >> 1) & 0xffff) as u32 })
    }

    /// `update_cte()`: false on a memory error.
    fn update_cte(&self, icid: u16, cte: &CtEntry) -> bool {
        let v = if cte.valid { 1 | (u64::from(cte.rdbase & 0xffff) << 1) } else { 0 };
        let ct = self.s.ct;
        match self.table_entry_addr(&ct, u32::from(icid)) {
            Err(_) => false,
            // No level 2 table for this index: the write is dropped.
            Ok(None) => true,
            Ok(Some(addr)) => self.write(addr, &v.to_le_bytes()).is_ok(),
        }
    }

    /// `get_dte()`.
    fn get_dte(&self, devid: u32) -> Result<DtEntry, MemTxResult> {
        let dt = self.s.dt;
        let Some(addr) = self.table_entry_addr(&dt, devid)? else {
            return Ok(DtEntry::default());
        };
        let v = self.ldq(addr)?;
        Ok(DtEntry {
            valid: v & 1 != 0,
            size: ((v >> 1) & 0x1f) as u8,
            // The DTE holds bits 8 to 51 of the ITT address.
            ittaddr: ((v >> 6) & ((1 << 44) - 1)) << ITTADDR_SHIFT,
        })
    }

    /// `update_dte()`. `dte.ittaddr` is the address shifted right by 8, as MAPD gives it.
    fn update_dte(&self, devid: u32, dte: &DtEntry) -> bool {
        let v = if dte.valid {
            1 | (u64::from(dte.size & 0x1f) << 1) | ((dte.ittaddr & ((1 << 44) - 1)) << 6)
        } else {
            0
        };
        let dt = self.s.dt;
        match self.table_entry_addr(&dt, devid) {
            Err(_) => false,
            Ok(None) => true,
            Ok(Some(addr)) => self.write(addr, &v.to_le_bytes()).is_ok(),
        }
    }

    fn ite_addr(dte: &DtEntry, eventid: u32) -> u64 {
        dte.ittaddr.wrapping_add(u64::from(eventid) * ITS_ITT_ENTRY_SIZE)
    }

    /// `get_ite()`.
    fn get_ite(&self, eventid: u32, dte: &DtEntry) -> Result<ItEntry, MemTxResult> {
        let addr = Self::ite_addr(dte, eventid);
        let itel = self.ldq(addr)?;
        let iteh = self.ldl(addr.wrapping_add(8))?;
        Ok(ItEntry {
            valid: itel & 1 != 0,
            inttype: ((itel >> 1) & 1) as u8,
            intid: ((itel >> 2) & 0xff_ffff) as u32,
            icid: (itel >> 32) as u16,
            vpeid: (itel >> 48) as u16,
            doorbell: iteh & 0xff_ffff,
        })
    }

    /// `update_ite()`: false on a memory error.
    fn update_ite(&self, eventid: u32, dte: &DtEntry, ite: &ItEntry) -> bool {
        let addr = Self::ite_addr(dte, eventid);
        let (itel, iteh) = if ite.valid {
            (
                1 | (u64::from(ite.inttype & 1) << 1)
                    | (u64::from(ite.intid & 0xff_ffff) << 2)
                    | (u64::from(ite.icid) << 32)
                    | (u64::from(ite.vpeid) << 48),
                ite.doorbell & 0xff_ffff,
            )
        } else {
            (0, 0)
        };
        self.write(addr, &itel.to_le_bytes()).is_ok()
            && self.write(addr.wrapping_add(8), &iteh.to_le_bytes()).is_ok()
    }

    /// `lookup_ite()`: the valid ITE for `devid` and `eventid`, and the DTE it came from.
    fn lookup_ite(&self, devid: u32, eventid: u32) -> Result<(ItEntry, DtEntry), CmdResult> {
        if devid >= self.s.dt.num_entries {
            return Err(CmdResult::Continue);
        }
        let dte = self.get_dte(devid).map_err(|_| CmdResult::Stall)?;
        if !dte.valid {
            return Err(CmdResult::Continue);
        }
        let num_eventids = 1u64 << (dte.size + 1);
        if u64::from(eventid) >= num_eventids {
            return Err(CmdResult::Continue);
        }
        let ite = self.get_ite(eventid, &dte).map_err(|_| CmdResult::Stall)?;
        if !ite.valid {
            return Err(CmdResult::Continue);
        }
        Ok((ite, dte))
    }

    /// `lookup_cte()`: the valid CTE for `icid`, naming an existing redistributor.
    fn lookup_cte(&self, icid: u32) -> Result<CtEntry, CmdResult> {
        if icid >= self.s.ct.num_entries {
            return Err(CmdResult::Continue);
        }
        let cte = self.get_cte(icid as u16).map_err(|_| CmdResult::Stall)?;
        if !cte.valid || cte.rdbase as usize >= self.gic.num_cpu() {
            return Err(CmdResult::Continue);
        }
        Ok(cte)
    }

    /// `do_process_its_cmd()`: INT, CLEAR, DISCARD and GITS_TRANSLATER writes.
    fn do_process_its_cmd(&self, devid: u32, eventid: u32, cmd: CmdType) -> CmdResult {
        let (ite, dte) = match self.lookup_ite(devid, eventid) {
            Ok(v) => v,
            Err(r) => return r,
        };
        let level = !matches!(cmd, CmdType::Clear | CmdType::Discard);
        if ite.inttype != ITE_INTTYPE_PHYSICAL {
            // Only a guest writing the table memory behind our back gets a virtual ITE here.
            return CmdResult::Continue;
        }
        let cmdres = match self.lookup_cte(u32::from(ite.icid)) {
            Ok(cte) => {
                self.gic.with_state(|g| g.process_lpi(cte.rdbase as usize, ite.intid, level));
                CmdResult::ContinueOk
            }
            Err(r) => r,
        };
        if cmdres == CmdResult::ContinueOk && cmd == CmdType::Discard {
            // Remove the mapping from the interrupt translation table.
            return if self.update_ite(eventid, &dte, &ItEntry::default()) {
                CmdResult::ContinueOk
            } else {
                CmdResult::Stall
            };
        }
        // As in QEMU, a failed collection lookup does not fail the command.
        CmdResult::ContinueOk
    }

    /// `process_mapti()`, which is MAPI when `ignore_pintid`.
    fn process_mapti(&self, pkt: &[u64; 4], ignore_pintid: bool) -> CmdResult {
        let devid = (pkt[0] >> 32) as u32;
        let eventid = pkt[1] as u32;
        let icid = pkt[2] as u16;
        let pintid = if ignore_pintid { eventid } else { (pkt[1] >> 32) as u32 };

        if devid >= self.s.dt.num_entries {
            return CmdResult::Continue;
        }
        let Ok(dte) = self.get_dte(devid) else {
            return CmdResult::Stall;
        };
        let num_eventids = 1u64 << (dte.size + 1);
        if u32::from(icid) >= self.s.ct.num_entries
            || !dte.valid
            || u64::from(eventid) >= num_eventids
            || !intid_in_lpi_range(pintid)
        {
            return CmdResult::Continue;
        }
        let ite = ItEntry {
            valid: true,
            inttype: ITE_INTTYPE_PHYSICAL,
            intid: pintid,
            icid,
            doorbell: NO_DOORBELL,
            vpeid: 0,
        };
        if self.update_ite(eventid, &dte, &ite) { CmdResult::ContinueOk } else { CmdResult::Stall }
    }

    /// `process_mapc()`.
    fn process_mapc(&self, pkt: &[u64; 4]) -> CmdResult {
        let icid = pkt[2] as u16;
        let valid = pkt[2] & CMD_FIELD_VALID != 0;
        let rdbase = if valid { ((pkt[2] >> 16) & 0xffff) as u32 } else { 0 };
        if u32::from(icid) >= self.s.ct.num_entries
            || (valid && rdbase as usize >= self.gic.num_cpu())
        {
            return CmdResult::Continue;
        }
        if self.update_cte(icid, &CtEntry { valid, rdbase }) {
            CmdResult::ContinueOk
        } else {
            CmdResult::Stall
        }
    }

    /// `process_mapd()`.
    fn process_mapd(&self, pkt: &[u64; 4]) -> CmdResult {
        let devid = (pkt[0] >> 32) as u32;
        let dte = DtEntry {
            size: (pkt[1] & 0x1f) as u8,
            ittaddr: (pkt[2] & ITTADDR_MASK) >> ITTADDR_SHIFT,
            valid: pkt[2] & CMD_FIELD_VALID != 0,
        };
        if devid >= self.s.dt.num_entries || u64::from(dte.size) > ITS_IDBITS {
            return CmdResult::Continue;
        }
        if self.update_dte(devid, &dte) { CmdResult::ContinueOk } else { CmdResult::Stall }
    }

    /// `process_movall()`.
    fn process_movall(&self, pkt: &[u64; 4]) -> CmdResult {
        let rd1 = (pkt[2] >> 16) & ((1 << 36) - 1);
        let rd2 = (pkt[3] >> 16) & ((1 << 36) - 1);
        let num_cpu = self.gic.num_cpu() as u64;
        if rd1 >= num_cpu || rd2 >= num_cpu {
            return CmdResult::Continue;
        }
        if rd1 == rd2 {
            // A move to the same place has to succeed as a no-op.
            return CmdResult::ContinueOk;
        }
        self.gic.with_state(|g| g.movall_lpis(rd1 as usize, rd2 as usize));
        CmdResult::ContinueOk
    }

    /// `process_movi()`.
    fn process_movi(&self, pkt: &[u64; 4]) -> CmdResult {
        let devid = (pkt[0] >> 32) as u32;
        let eventid = pkt[1] as u32;
        let new_icid = pkt[2] as u16;
        let (mut ite, dte) = match self.lookup_ite(devid, eventid) {
            Ok(v) => v,
            Err(r) => return r,
        };
        if ite.inttype != ITE_INTTYPE_PHYSICAL {
            return CmdResult::Continue;
        }
        let old_cte = match self.lookup_cte(u32::from(ite.icid)) {
            Ok(c) => c,
            Err(r) => return r,
        };
        let new_cte = match self.lookup_cte(u32::from(new_icid)) {
            Ok(c) => c,
            Err(r) => return r,
        };
        if old_cte.rdbase != new_cte.rdbase {
            self.gic.with_state(|g| {
                g.mov_lpi(old_cte.rdbase as usize, new_cte.rdbase as usize, ite.intid);
            });
        }
        ite.icid = new_icid;
        if self.update_ite(eventid, &dte, &ite) { CmdResult::ContinueOk } else { CmdResult::Stall }
    }

    /// `process_inv()`.
    fn process_inv(&self, pkt: &[u64; 4]) -> CmdResult {
        let devid = (pkt[0] >> 32) as u32;
        let eventid = pkt[1] as u32;
        let (ite, _) = match self.lookup_ite(devid, eventid) {
            Ok(v) => v,
            Err(r) => return r,
        };
        if ite.inttype != ITE_INTTYPE_PHYSICAL {
            return CmdResult::Continue;
        }
        match self.lookup_cte(u32::from(ite.icid)) {
            Ok(cte) => {
                self.gic.with_state(|g| g.inv_lpi(cte.rdbase as usize));
                CmdResult::ContinueOk
            }
            Err(r) => r,
        }
    }

    /// `process_cmdq()`: run the commands from GITS_CREADR up to GITS_CWRITER. Everything
    /// runs to completion before the write that started it returns.
    fn process_cmdq(&mut self) {
        if self.s.ctlr & GITS_CTLR_ENABLED == 0 {
            return;
        }
        let wr_offset = ((self.s.cwriter >> GITS_CQ_OFFSET_SHIFT) & GITS_CQ_OFFSET_MASK) as u32;
        if wr_offset >= self.s.cq.num_entries {
            return;
        }
        let mut rd_offset = ((self.s.creadr >> GITS_CQ_OFFSET_SHIFT) & GITS_CQ_OFFSET_MASK) as u32;
        if rd_offset >= self.s.cq.num_entries {
            return;
        }

        while wr_offset != rd_offset {
            let addr =
                self.s.cq.base_addr.wrapping_add(u64::from(rd_offset) * GITS_CMDQ_ENTRY_SIZE);
            let mut raw = [0u8; GITS_CMDQ_ENTRY_SIZE as usize];
            if self.read(addr, &mut raw).is_err() {
                self.s.creadr |= GITS_CREADR_STALLED;
                break;
            }
            let mut pkt = [0u64; 4];
            for (i, w) in pkt.iter_mut().enumerate() {
                let mut b = [0u8; 8];
                b.copy_from_slice(&raw[i * 8..i * 8 + 8]);
                *w = u64::from_le_bytes(b);
            }

            let result = match pkt[0] as u8 {
                GITS_CMD_INT => self.process_its_cmd(&pkt, CmdType::Interrupt),
                GITS_CMD_CLEAR => self.process_its_cmd(&pkt, CmdType::Clear),
                GITS_CMD_DISCARD => self.process_its_cmd(&pkt, CmdType::Discard),
                // Every command completes before the next, so SYNC has nothing to wait for.
                GITS_CMD_SYNC => CmdResult::ContinueOk,
                GITS_CMD_MAPD => self.process_mapd(&pkt),
                GITS_CMD_MAPC => self.process_mapc(&pkt),
                GITS_CMD_MAPTI => self.process_mapti(&pkt, false),
                GITS_CMD_MAPI => self.process_mapti(&pkt, true),
                GITS_CMD_INV => self.process_inv(&pkt),
                GITS_CMD_INVALL => {
                    // Nothing is cached but the best pending LPI of each redistributor, so
                    // recompute those.
                    self.gic.with_state(|g| {
                        for cpu in 0..g.cpu.len() {
                            g.update_lpi(cpu);
                        }
                    });
                    CmdResult::ContinueOk
                }
                GITS_CMD_MOVI => self.process_movi(&pkt),
                GITS_CMD_MOVALL => self.process_movall(&pkt),
                // Without GITS_TYPER.Virtual the virtual commands are skipped.
                GITS_CMD_VSYNC | GITS_CMD_VMAPTI | GITS_CMD_VMAPI | GITS_CMD_VMAPP
                | GITS_CMD_VMOVP | GITS_CMD_VMOVI | GITS_CMD_VINVALL => CmdResult::Continue,
                _ => CmdResult::ContinueOk,
            };
            if result == CmdResult::Stall {
                self.s.creadr |= GITS_CREADR_STALLED;
                break;
            }
            rd_offset = (rd_offset + 1) % self.s.cq.num_entries;
            self.s.creadr = (self.s.creadr & !(GITS_CQ_OFFSET_MASK << GITS_CQ_OFFSET_SHIFT))
                | (u64::from(rd_offset) << GITS_CQ_OFFSET_SHIFT);
        }
    }

    /// `process_its_cmd()`: INT, CLEAR and DISCARD.
    fn process_its_cmd(&self, pkt: &[u64; 4], cmd: CmdType) -> CmdResult {
        self.do_process_its_cmd((pkt[0] >> 32) as u32, pkt[1] as u32, cmd)
    }

    /// `extract_table_params()`: decode GITS_BASER<n> into the device and collection tables.
    fn extract_table_params(&mut self, typer: u64) {
        for i in 0..8 {
            let value = self.s.baser[i];
            if value == 0 {
                continue;
            }
            let page_sz = match (value >> 8) & 3 {
                0 => GITS_PAGE_SIZE_4K,
                1 => GITS_PAGE_SIZE_16K,
                _ => GITS_PAGE_SIZE_64K,
            };
            let num_pages = (value & 0xff) as u32 + 1;
            let (td, idbits) = match (value >> 56) & 7 {
                GITS_BASER_TYPE_DEVICE => (&mut self.s.dt, ((typer >> 13) & 0x1f) + 1),
                GITS_BASER_TYPE_COLLECTION => {
                    // GITS_TYPER.CIL is set, so CIDbits gives the collection ID width.
                    (&mut self.s.ct, ((typer >> 32) & 0xf) + 1)
                }
                // TYPE is read only, so only the types reset put there can be seen.
                _ => unreachable!("GITS_BASER{i} has an unimplemented type"),
            };
            *td = TableDesc::default();
            // A table that is not valid is left with no entries, so every lookup in it fails.
            if value & GITS_BASER_VALID == 0 {
                continue;
            }
            td.page_sz = page_sz;
            td.indirect = value & GITS_BASER_INDIRECT != 0;
            td.entry_sz = ((value >> 48) & 0x1f) as u16 + 1;
            td.base_addr = baser_base_addr(value, page_sz);
            let entry_sz = u32::from(td.entry_sz);
            let num_entries = if td.indirect {
                (num_pages.wrapping_mul(page_sz) / L1TABLE_ENTRY_SIZE)
                    .wrapping_mul(page_sz / entry_sz)
            } else {
                num_pages.wrapping_mul(page_sz) / entry_sz
            };
            td.num_entries = u64::from(num_entries).min(1 << idbits) as u32;
        }
    }

    /// `extract_cmdq_params()`.
    fn extract_cmdq_params(&mut self) {
        let value = self.s.cbaser;
        let num_pages = (value & 0xff) + 1;
        self.s.cq = CmdQDesc::default();
        if value & GITS_CBASER_VALID != 0 {
            self.s.cq.num_entries =
                (num_pages * u64::from(GITS_PAGE_SIZE_4K) / GITS_CMDQ_ENTRY_SIZE) as u32;
            self.s.cq.base_addr = ((value >> 12) & ((1 << 40) - 1)) << 12;
        }
    }
}

impl GicV3Its {
    /// Realize the ITS of `gic`, `gicv3_arm_its_realize()`. The GIC must have LPIs. The ITS
    /// comes back reset.
    pub fn new(gic: &Arc<GicV3>) -> Result<Arc<GicV3Its>, String> {
        {
            let g = gic.lock();
            for (i, cs) in g.cpu.iter().enumerate() {
                if cs.gicr_typer & GICR_TYPER_PLPIS == 0 {
                    return Err(format!("Physical LPI not supported by CPU {i}"));
                }
            }
        }
        let typer = 1
            | ((ITS_ITT_ENTRY_SIZE - 1) << 4)
            | (ITS_IDBITS << 8)
            | (ITS_DEVBITS << 13)
            | (ITS_CIDBITS << 32)
            | (1 << 36);
        let its = GicV3Its {
            gic: gic.clone(),
            typer,
            state: Mutex::new(ItsState {
                ctlr: 0,
                cbaser: 0,
                cwriter: 0,
                creadr: 0,
                baser: [0; 8],
                dt: TableDesc::default(),
                ct: TableDesc::default(),
                cq: CmdQDesc::default(),
            }),
        };
        its.reset();
        Ok(Arc::new(its))
    }

    fn lock(&self) -> MutexGuard<'_, ItsState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Run `f` with the state locked, marking the thread as inside the ITS. The GIC lock is
    /// always taken after the ITS lock.
    fn with_exec<R>(&self, f: impl FnOnce(&mut Exec<'_>) -> R) -> R {
        let _engaged = Engaged::enter();
        let mut s = self.lock();
        let mut e = Exec { gic: &self.gic, dma: self.gic.dma(), s: &mut s };
        f(&mut e)
    }

    /// The device reset, `gicv3_its_common_reset_hold()` and `gicv3_its_reset_hold()`.
    pub fn reset(&self) {
        let mut s = self.lock();
        s.ctlr = GITS_CTLR_QUIESCENT;
        s.cbaser = 0;
        s.cwriter = 0;
        s.creadr = 0;
        s.baser = [0; 8];
        // GITS_BASER0 is the device table and GITS_BASER1 the collection table, both with 64K
        // pages and 8 byte entries. The others are unimplemented.
        let entrysize = (GITS_TABLE_ENTRY_SIZE - 1) << 48;
        let pagesize = GITS_BASER_PAGESIZE_64K << 8;
        s.baser[0] = (GITS_BASER_TYPE_DEVICE << 56) | pagesize | entrysize;
        s.baser[1] = (GITS_BASER_TYPE_COLLECTION << 56) | pagesize | entrysize;
    }

    /// GITS_TYPER.
    pub fn typer(&self) -> u64 {
        self.typer
    }

    /// The control frame, [`ITS_CONTROL_SIZE`] bytes.
    pub fn control_ops(self: &Arc<Self>) -> Arc<dyn MmioOps> {
        Arc::new(ItsControl { its: self.clone() })
    }

    /// The translation frame, [`ITS_TRANS_SIZE`] bytes.
    pub fn translation_ops(self: &Arc<Self>) -> Arc<dyn MmioOps> {
        Arc::new(ItsTranslation { its: self.clone() })
    }

    /// `its_readl()`. `None` for a reserved offset.
    fn readl(&self, s: &ItsState, offset: u64) -> Option<u64> {
        let lo = |v: u64| v & 0xffff_ffff;
        let hi = |v: u64| v >> 32;
        Some(match offset {
            GITS_CTLR => u64::from(s.ctlr),
            GITS_IIDR => u64::from(GICV3_IIDR),
            GITS_IDREGS..=0xffff => {
                u64::from(self.gic.lock().idreg(offset - GITS_IDREGS, GICV3_PIDR0_ITS))
            }
            GITS_TYPER => lo(self.typer),
            0x0c => hi(self.typer),
            GITS_CBASER => lo(s.cbaser),
            0x84 => hi(s.cbaser),
            GITS_CREADR => lo(s.creadr),
            0x94 => hi(s.creadr),
            GITS_CWRITER => lo(s.cwriter),
            0x8c => hi(s.cwriter),
            GITS_BASER..=0x13f => {
                let v = s.baser[((offset - GITS_BASER) / 8) as usize];
                if offset & 7 != 0 { hi(v) } else { lo(v) }
            }
            _ => return None,
        })
    }

    /// `its_readll()`.
    fn readll(&self, s: &ItsState, offset: u64) -> Option<u64> {
        Some(match offset {
            GITS_TYPER => self.typer,
            GITS_BASER..=0x13f => s.baser[((offset - GITS_BASER) / 8) as usize],
            GITS_CBASER => s.cbaser,
            GITS_CREADR => s.creadr,
            GITS_CWRITER => s.cwriter,
            _ => return None,
        })
    }

    /// `its_writel()`.
    fn writel(&self, e: &mut Exec<'_>, offset: u64, value: u64) {
        let enabled = e.s.ctlr & GITS_CTLR_ENABLED != 0;
        match offset {
            GITS_CTLR => {
                if value & u64::from(GITS_CTLR_ENABLED) != 0 {
                    e.s.ctlr |= GITS_CTLR_ENABLED;
                    e.extract_table_params(self.typer);
                    e.extract_cmdq_params();
                    e.process_cmdq();
                } else {
                    e.s.ctlr &= !GITS_CTLR_ENABLED;
                }
            }
            // GITS_CBASER and GITS_BASER<n> are read only while the ITS is enabled, an IMPDEF
            // choice.
            GITS_CBASER | 0x84 => {
                if !enabled {
                    e.s.cbaser = deposit_half(e.s.cbaser, offset & 4 != 0, value);
                    e.s.creadr = 0;
                }
            }
            GITS_CWRITER => {
                e.s.cwriter = deposit_half(e.s.cwriter, false, value & !GITS_CWRITER_RETRY);
                if e.s.cwriter != e.s.creadr {
                    e.process_cmdq();
                }
            }
            0x8c => e.s.cwriter = deposit_half(e.s.cwriter, true, value),
            GITS_CREADR | 0x94 => {
                // GITS_CREADR is only writable without security.
                if self.gic.lock().ds() {
                    let high = offset & 4 != 0;
                    let v = if high { value } else { value & !GITS_CREADR_STALLED };
                    e.s.creadr = deposit_half(e.s.creadr, high, v);
                }
            }
            GITS_BASER..=0x13f => {
                let index = ((offset - GITS_BASER) / 8) as usize;
                // An unimplemented GITS_BASER<n> is RAZ/WI.
                if !enabled && e.s.baser[index] != 0 {
                    let b = &mut e.s.baser[index];
                    if offset & 7 != 0 {
                        *b &= GITS_BASER_RO_MASK | 0xffff_ffff;
                        *b |= (value << 32) & !GITS_BASER_RO_MASK;
                    } else {
                        *b &= GITS_BASER_RO_MASK | 0xffff_ffff_0000_0000;
                        *b |= value & !GITS_BASER_RO_MASK;
                    }
                }
            }
            // GITS_IIDR, the ID registers and the reserved offsets ignore writes.
            _ => {}
        }
    }

    /// `its_writell()`.
    fn writell(&self, e: &mut Exec<'_>, offset: u64, value: u64) {
        let enabled = e.s.ctlr & GITS_CTLR_ENABLED != 0;
        match offset {
            GITS_BASER..=0x13f => {
                let index = ((offset - GITS_BASER) / 8) as usize;
                if !enabled && e.s.baser[index] != 0 {
                    let b = &mut e.s.baser[index];
                    *b = (*b & GITS_BASER_RO_MASK) | (value & !GITS_BASER_RO_MASK);
                }
            }
            GITS_CBASER => {
                if !enabled {
                    e.s.cbaser = value;
                    e.s.creadr = 0;
                }
            }
            GITS_CWRITER => {
                e.s.cwriter = value & !GITS_CWRITER_RETRY;
                if e.s.cwriter != e.s.creadr {
                    e.process_cmdq();
                }
            }
            GITS_CREADR if self.gic.lock().ds() => {
                e.s.creadr = value & !GITS_CREADR_STALLED;
            }
            _ => {}
        }
    }
}

/// The control frame, `gicv3_its_control_ops`.
struct ItsControl {
    its: Arc<GicV3Its>,
}

impl fmt::Debug for ItsControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ItsControl")
    }
}

impl MmioOps for ItsControl {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        if engaged() {
            return Err(MemTxResult::ACCESS_ERROR);
        }
        let its = &self.its;
        let v = its.with_exec(|e| match size.bytes() {
            4 => its.readl(e.s, offset),
            8 => its.readll(e.s, offset),
            _ => None,
        });
        // Reserved registers are RAZ/WI, so a bad access reads zero rather than aborting.
        Ok(v.unwrap_or(0))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        if engaged() {
            return Err(MemTxResult::ACCESS_ERROR);
        }
        let its = &self.its;
        its.with_exec(|e| match size.bytes() {
            4 => its.writel(e, offset, value),
            8 => its.writell(e, offset, value),
            _ => {}
        });
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 8)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 8)
    }
}

/// The translation frame, `gicv3_its_translation_ops`.
struct ItsTranslation {
    its: Arc<GicV3Its>,
}

impl fmt::Debug for ItsTranslation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ItsTranslation")
    }
}

impl MmioOps for ItsTranslation {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        // GITS_TRANSLATER is write only and the rest of the frame is RES0.
        Ok(0)
    }

    fn write(&self, cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        if offset != GITS_TRANSLATER {
            return Ok(());
        }
        if engaged() {
            return Err(MemTxResult::ACCESS_ERROR);
        }
        let devid = u32::from(cx.attrs.requester_id());
        let r = self.its.with_exec(|e| {
            if e.s.ctlr & GITS_CTLR_ENABLED == 0 {
                return CmdResult::ContinueOk;
            }
            e.do_process_its_cmd(devid, value as u32, CmdType::None)
        });
        if r == CmdResult::Stall { Err(MemTxResult::ERROR) } else { Ok(()) }
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(2, 4)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(2, 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baser_addresses() {
        let v = 0x8000_1234_5678_9000u64 | (0xf << 12);
        assert_eq!(baser_base_addr(v, GITS_PAGE_SIZE_4K), 0x1234_5678_f000);
        assert_eq!(baser_base_addr(v, GITS_PAGE_SIZE_16K), 0x1234_5678_f000);
        // With 64K pages, bits 12 to 15 are bits 48 to 51 of the address.
        assert_eq!(baser_base_addr(v, GITS_PAGE_SIZE_64K), 0xf_1234_5678_0000);
    }

    #[test]
    fn lpi_range() {
        assert!(!intid_in_lpi_range(8191));
        assert!(intid_in_lpi_range(8192));
        assert!(intid_in_lpi_range(65535));
        assert!(!intid_in_lpi_range(65536));
    }
}
