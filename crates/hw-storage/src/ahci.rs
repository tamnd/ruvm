// SPDX-License-Identifier: GPL-2.0-or-later

//! The AHCI host bus adapter, ported from QEMU's `hw/ide/ahci.c`.
//!
//! [`AhciState`] holds the generic host control registers and one [`AhciPort`] per port. Each
//! port has a single ATA or ATAPI drive, the `IDEState` that QEMU keeps on the port's one-device
//! IDE bus. The PCI wrapper in `ich.rs` owns the state behind a lock and turns the interrupt
//! level changes collected here into INTx or MSI once the lock is released.
//!
//! QEMU maps the command list and the received FIS area into host memory when the guest starts
//! the engines. Here the mapping is a check that the guest memory is readable at that point, and
//! each later access goes through [`DmaMemory`] at the address that was checked.

use std::fmt;
use std::sync::Arc;

use ruvm_mem::{AddressSpace, MemTxAttrs};

use crate::ide::{
    BUSY_STAT, DRQ_STAT, DriveConfig, DriveKind, ERR_STAT, IdeBusVmState, IdeDrive,
    IdeDriveVmState, IdeHost, READY_STAT, SEEK_STAT, SgList, WRERR_STAT,
};

/// Guest memory as the controller sees it for DMA.
pub trait DmaMemory: Send + Sync {
    /// Reads `buf.len()` bytes at `addr`. Returns false if any part failed.
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool;

    /// Writes `buf` at `addr`. Returns false if any part failed.
    fn dma_write(&self, addr: u64, buf: &[u8]) -> bool;
}

impl DmaMemory for AddressSpace {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.read(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok()
    }

    fn dma_write(&self, addr: u64, buf: &[u8]) -> bool {
        self.write(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok()
    }
}

pub(crate) const AHCI_MEM_BAR_SIZE: u64 = 0x1000;
pub(crate) const AHCI_MAX_CMDS: u32 = 32;

const AHCI_CMD_ATAPI: u16 = 1 << 5;
const AHCI_CMD_WRITE: u16 = 1 << 6;
const AHCI_CMD_CLR_BUSY: u16 = 1 << 10;

const HOST_CTL_RESET: u32 = 1 << 0;
const HOST_CTL_IRQ_EN: u32 = 1 << 1;
const HOST_CTL_AHCI_EN: u32 = 1 << 31;

const HOST_CAP_AHCI: u32 = 1 << 18;
const HOST_CAP_NCQ: u32 = 1 << 30;
const HOST_CAP_64: u32 = 1 << 31;

const PORT_CMD_LIST_ON: u32 = 1 << 15;
const PORT_CMD_FIS_ON: u32 = 1 << 14;
const PORT_CMD_FIS_RX: u32 = 1 << 4;
const PORT_CMD_POWER_ON: u32 = 1 << 2;
const PORT_CMD_SPIN_UP: u32 = 1 << 1;
const PORT_CMD_START: u32 = 1 << 0;
const PORT_CMD_ICC_MASK: u32 = 0xf << 28;
const PORT_CMD_RO_MASK: u32 = 0x007d_ffe0;

const PORT_IRQ_BIT_DHRS: u32 = 0;
const PORT_IRQ_BIT_PSS: u32 = 1;
const PORT_IRQ_BIT_SDBS: u32 = 3;
const PORT_IRQ_BIT_OFS: u32 = 24;
const PORT_IRQ_BIT_HBFS: u32 = 29;
const PORT_IRQ_BIT_TFES: u32 = 30;

const ATA_SRST: u8 = 1 << 2;

const SATA_SCR_SSTATUS_DET_DEV_PRESENT_PHY_UP: u32 = 0x3;
const SATA_SCR_SSTATUS_SPD_GEN1: u32 = 0x10;
const SATA_SCR_SSTATUS_IPM_ACTIVE: u32 = 0x100;
const AHCI_SCR_SCTL_DET: u32 = 0xf;

const SATA_FIS_TYPE_REGISTER_H2D: u8 = 0x27;
const SATA_FIS_REG_H2D_UPDATE_COMMAND_REGISTER: u8 = 0x80;
const SATA_FIS_TYPE_REGISTER_D2H: u8 = 0x34;
const SATA_FIS_TYPE_PIO_SETUP: u8 = 0x5f;
const SATA_FIS_TYPE_SDB: u8 = 0xa1;

const SATA_SIGNATURE_CDROM: u32 = 0xeb14_0101;
const SATA_SIGNATURE_DISK: u32 = 0x0000_0101;

const AHCI_GENERIC_HOST_CONTROL_REGS_MAX_ADDR: u64 = 0x2c;
const AHCI_PORT_REGS_START_ADDR: u64 = 0x100;
const AHCI_PORT_ADDR_OFFSET_MASK: u64 = 0x7f;
const AHCI_PORT_ADDR_OFFSET_LEN: u64 = 0x80;

const AHCI_NUM_COMMAND_SLOTS: u32 = 31;
const AHCI_SUPPORTED_SPEED: u32 = 20;
const AHCI_SUPPORTED_SPEED_GEN1: u32 = 1;
const AHCI_VERSION_1_0: u32 = 0x10000;
const AHCI_COMMAND_TABLE_ACMD: usize = 0x40;
const AHCI_PRDT_SIZE_MASK: u32 = 0x3f_ffff;

const READ_FPDMA_QUEUED: u8 = 0x60;
const WRITE_FPDMA_QUEUED: u8 = 0x61;
const NCQ_NON_DATA: u8 = 0x63;
const SEND_FPDMA_QUEUED: u8 = 0x64;
const RECEIVE_FPDMA_QUEUED: u8 = 0x65;

const RES_FIS_PSFIS: u64 = 0x20;
const RES_FIS_RFIS: u64 = 0x40;
const RES_FIS_SDBFIS: u64 = 0x58;

/// How many times the deferred "check the command list again" step may rerun after one guest
/// access. QEMU runs it from a bottom half; here it runs before the access returns.
const MAX_RECHECKS: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PortState {
    Run,
    Reset,
}

/// The port registers, `AHCIPortRegs`.
#[derive(Clone, Copy, Debug, Default)]
struct PortRegs {
    lst_addr: u32,
    lst_addr_hi: u32,
    fis_addr: u32,
    fis_addr_hi: u32,
    irq_stat: u32,
    irq_mask: u32,
    cmd: u32,
    tfdata: u32,
    sig: u32,
    scr_ctl: u32,
    scr_err: u32,
    scr_act: u32,
    cmd_issue: u32,
}

/// One port and its drive, `AHCIDevice`.
#[derive(Debug)]
pub(crate) struct AhciPort {
    regs: PortRegs,
    /// The guest address of the command list while the command engine is running.
    lst: Option<u64>,
    /// The guest address of the received FIS area while FIS receive is running.
    res_fis: Option<u64>,
    port_state: PortState,
    /// NCQ tags finished since the last Set Device Bits FIS.
    finished: u32,
    done_first_drq: bool,
    busy_slot: Option<u8>,
    init_d2h_sent: bool,
    /// The slot whose command is being processed, `cur_cmd`.
    cur_slot: Option<u8>,
    /// `check_bh` is scheduled.
    check_pending: bool,
    /// A device is plugged in, QEMU's `ifs[0].blk != NULL`.
    attached: bool,
    /// Present even on an empty port, as QEMU's `IDEState` is.
    drive: Option<IdeDrive>,
}

impl AhciPort {
    /// Whether a drive is plugged in.
    pub(crate) fn attached(&self) -> bool {
        self.attached
    }

    fn new() -> Self {
        AhciPort {
            regs: PortRegs::default(),
            lst: None,
            res_fis: None,
            port_state: PortState::Run,
            finished: 0,
            done_first_drq: false,
            busy_slot: None,
            init_d2h_sent: false,
            cur_slot: None,
            check_pending: false,
            attached: false,
            drive: Some(IdeDrive::new(&DriveConfig::hd(), None, String::new())),
        }
    }
}

/// The controller, `AHCIState`.
pub(crate) struct AhciState {
    pub(crate) ports: Vec<AhciPort>,
    cap: u32,
    ghc: u32,
    irqstatus: u32,
    impl_: u32,
    version: u32,
    idp_index: u32,
    idp_offset: u64,
    mem: Arc<dyn DmaMemory>,
    /// The level of the interrupt output after every `ahci_check_irq()`, oldest first. The PCI
    /// wrapper drains it once the lock is dropped.
    pub(crate) irq_events: Vec<bool>,
}

impl fmt::Debug for AhciState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AhciState")
            .field("cap", &self.cap)
            .field("ghc", &self.ghc)
            .field("irqstatus", &self.irqstatus)
            .field("ports", &self.ports)
            .finish_non_exhaustive()
    }
}

/// `vmstate_ncq_tfs`, one NCQ tag of a port.
///
/// Every NCQ command finishes inside the register write that issued it, so a saved tag is
/// always free and all zeros. QEMU keeps the fields of the last command in a free tag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NcqVmState {
    pub sector_count: u32,
    pub lba: u64,
    pub tag: u8,
    pub cmd: u8,
    pub slot: u8,
    pub used: bool,
    pub halt: bool,
}

/// `vmstate_ahci_device`, one port and its drive.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AhciPortVmState {
    pub port: IdeBusVmState,
    /// `port.ifs[0]`.
    pub ifs0: IdeDriveVmState,
    /// `STATE_RUN` (0) or `STATE_RESET` (1).
    pub port_state: u32,
    pub finished: u32,
    pub lst_addr: u32,
    pub lst_addr_hi: u32,
    pub fis_addr: u32,
    pub fis_addr_hi: u32,
    pub irq_stat: u32,
    pub irq_mask: u32,
    pub cmd: u32,
    pub tfdata: u32,
    pub sig: u32,
    /// QEMU only ever stores 0 here; reads of PxSSTS are computed.
    pub scr_stat: u32,
    pub scr_ctl: u32,
    pub scr_err: u32,
    pub scr_act: u32,
    pub cmd_issue: u32,
    pub done_first_drq: bool,
    /// The slot of the command in progress, -1 for none.
    pub busy_slot: i32,
    pub init_d2h_sent: bool,
    pub ncq_tfs: [NcqVmState; AHCI_MAX_CMDS as usize],
}

/// `vmstate_ahci`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AhciVmState {
    /// One per port, `ports` of them. The count is not in the stream.
    pub dev: Vec<AhciPortVmState>,
    pub cap: u32,
    pub ghc: u32,
    pub irqstatus: u32,
    /// `control_regs.impl`.
    pub impl_: u32,
    pub version: u32,
    pub idp_index: u32,
    /// `VMSTATE_UINT32_EQUAL`.
    pub ports: u32,
}

fn le16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn le32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn le64(b: &[u8]) -> u64 {
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

fn is_ncq(cmd: u8) -> bool {
    matches!(
        cmd,
        READ_FPDMA_QUEUED
            | WRITE_FPDMA_QUEUED
            | NCQ_NON_DATA
            | RECEIVE_FPDMA_QUEUED
            | SEND_FPDMA_QUEUED
    )
}

/// A command header, `AHCICmdHdr`.
#[derive(Clone, Copy, Debug)]
struct CmdHdr {
    opts: u16,
    prdtl: u16,
    tbl_addr: u64,
}

impl AhciState {
    /// `ahci_realize()` and `ahci_reg_init()`.
    pub(crate) fn new(nports: usize, idp_offset: u64, mem: Arc<dyn DmaMemory>) -> Self {
        let n = nports as u32;
        AhciState {
            ports: (0..nports).map(|_| AhciPort::new()).collect(),
            cap: (n - 1)
                | (AHCI_NUM_COMMAND_SLOTS << 8)
                | (AHCI_SUPPORTED_SPEED_GEN1 << AHCI_SUPPORTED_SPEED)
                | HOST_CAP_NCQ
                | HOST_CAP_AHCI
                | HOST_CAP_64,
            ghc: 0,
            irqstatus: 0,
            impl_: ((1u64 << n) - 1) as u32,
            version: AHCI_VERSION_1_0,
            idp_index: 0,
            idp_offset,
            mem,
            irq_events: Vec::new(),
        }
    }

    /// Plugs a drive into `port` and puts the port through a reset so the guest sees its
    /// signature.
    pub(crate) fn attach(&mut self, port: usize, drive: IdeDrive) {
        let p = &mut self.ports[port];
        p.drive = Some(drive);
        p.attached = true;
        self.reset_port(port);
    }

    /// Runs `f` with the drive of `port` taken out of the port, so both can be borrowed.
    fn with_drive<R>(&mut self, port: usize, f: impl FnOnce(&mut Self, &mut IdeDrive) -> R) -> R {
        let mut d = self.ports[port].drive.take().expect("drive is in use");
        let r = f(self, &mut d);
        self.ports[port].drive = Some(d);
        r
    }

    fn drive(&self, port: usize) -> &IdeDrive {
        self.ports[port].drive.as_ref().expect("drive is in use")
    }

    /// `ahci_check_irq()`.
    fn check_irq(&mut self) {
        self.irqstatus = 0;
        for (i, p) in self.ports.iter().enumerate() {
            if p.regs.irq_stat & p.regs.irq_mask != 0 {
                self.irqstatus |= 1 << i;
            }
        }
        let level = self.irqstatus != 0 && self.ghc & HOST_CTL_IRQ_EN != 0;
        self.irq_events.push(level);
    }

    /// `ahci_trigger_irq()`.
    fn trigger_irq(&mut self, port: usize, bit: u32) {
        self.ports[port].regs.irq_stat |= 1 << bit;
        self.check_irq();
    }

    fn port_read(&self, port: usize, offset: u64) -> u32 {
        let p = &self.ports[port];
        let r = &p.regs;
        match offset / 4 {
            0 => r.lst_addr,
            1 => r.lst_addr_hi,
            2 => r.fis_addr,
            3 => r.fis_addr_hi,
            4 => r.irq_stat,
            5 => r.irq_mask,
            6 => r.cmd,
            8 => r.tfdata,
            9 => r.sig,
            10 => {
                if p.attached {
                    SATA_SCR_SSTATUS_DET_DEV_PRESENT_PHY_UP
                        | SATA_SCR_SSTATUS_SPD_GEN1
                        | SATA_SCR_SSTATUS_IPM_ACTIVE
                } else {
                    0
                }
            }
            11 => r.scr_ctl,
            12 => r.scr_err,
            13 => r.scr_act,
            14 => r.cmd_issue,
            _ => 0,
        }
    }

    /// `map_page()`: whether `wanted` bytes at `addr` are guest memory the device can reach.
    fn probe(&self, addr: u64, wanted: usize) -> Option<u64> {
        let mut buf = vec![0; wanted];
        self.mem.dma_read(addr, &mut buf).then_some(addr)
    }

    fn map_clb_address(&mut self, port: usize) -> bool {
        let r = self.ports[port].regs;
        self.ports[port].cur_slot = None;
        let lst = self.probe(u64::from(r.lst_addr_hi) << 32 | u64::from(r.lst_addr), 1024);
        let p = &mut self.ports[port];
        p.lst = lst;
        if lst.is_some() {
            p.regs.cmd |= PORT_CMD_LIST_ON;
            true
        } else {
            p.regs.cmd &= !PORT_CMD_LIST_ON;
            false
        }
    }

    fn unmap_clb_address(&mut self, port: usize) {
        let p = &mut self.ports[port];
        if p.lst.is_none() {
            return;
        }
        p.regs.cmd &= !PORT_CMD_LIST_ON;
        p.lst = None;
    }

    fn map_fis_address(&mut self, port: usize) -> bool {
        let r = self.ports[port].regs;
        let fis = self.probe(u64::from(r.fis_addr_hi) << 32 | u64::from(r.fis_addr), 256);
        let p = &mut self.ports[port];
        p.res_fis = fis;
        if fis.is_some() {
            p.regs.cmd |= PORT_CMD_FIS_ON;
            true
        } else {
            p.regs.cmd &= !PORT_CMD_FIS_ON;
            false
        }
    }

    fn unmap_fis_address(&mut self, port: usize) {
        let p = &mut self.ports[port];
        if p.res_fis.is_none() {
            return;
        }
        p.regs.cmd &= !PORT_CMD_FIS_ON;
        p.res_fis = None;
    }

    /// `ahci_cond_start_engines()`.
    fn cond_start_engines(&mut self, port: usize) {
        let cmd = self.ports[port].regs.cmd;
        let cmd_start = cmd & PORT_CMD_START != 0;
        let cmd_on = cmd & PORT_CMD_LIST_ON != 0;
        let fis_start = cmd & PORT_CMD_FIS_RX != 0;
        let fis_on = cmd & PORT_CMD_FIS_ON != 0;

        if cmd_start && !cmd_on {
            if !self.map_clb_address(port) {
                self.ports[port].regs.cmd &= !PORT_CMD_START;
                return;
            }
        } else if !cmd_start && cmd_on {
            self.unmap_clb_address(port);
        }

        if fis_start && !fis_on {
            if !self.map_fis_address(port) {
                self.ports[port].regs.cmd &= !PORT_CMD_FIS_RX;
            }
        } else if !fis_start && fis_on {
            self.unmap_fis_address(port);
        }
    }

    fn port_write(&mut self, port: usize, offset: u64, val: u32) {
        match offset / 4 {
            0 => self.ports[port].regs.lst_addr = val,
            1 => self.ports[port].regs.lst_addr_hi = val,
            2 => self.ports[port].regs.fis_addr = val,
            3 => self.ports[port].regs.fis_addr_hi = val,
            4 => {
                self.ports[port].regs.irq_stat &= !val;
                self.check_irq();
            }
            5 => {
                self.ports[port].regs.irq_mask = val & 0xfdc0_00ff;
                self.check_irq();
            }
            6 => {
                let r = &mut self.ports[port].regs;
                if r.cmd & PORT_CMD_START != 0 && val & PORT_CMD_START == 0 {
                    r.scr_act = 0;
                    r.cmd_issue = 0;
                }
                // Read-only fields, LIST_ON and FIS_ON among them, keep their value. ICC state
                // changes are not supported, so the ICC field always reads as zero.
                r.cmd =
                    (r.cmd & PORT_CMD_RO_MASK) | (val & !(PORT_CMD_RO_MASK | PORT_CMD_ICC_MASK));
                self.cond_start_engines(port);
                // QEMU sends the initial D2H FIS only once, as soon as FIS receive is on.
                if self.ports[port].regs.cmd & PORT_CMD_FIS_ON != 0
                    && !self.ports[port].init_d2h_sent
                {
                    self.init_d2h(port);
                }
                self.check_cmd(port);
            }
            11 => {
                let r = self.ports[port].regs;
                if r.scr_ctl & AHCI_SCR_SCTL_DET == 1 && val & AHCI_SCR_SCTL_DET == 0 {
                    self.reset_port(port);
                }
                self.ports[port].regs.scr_ctl = val;
            }
            12 => self.ports[port].regs.scr_err &= !val,
            13 => self.ports[port].regs.scr_act |= val,
            14 => {
                self.ports[port].regs.cmd_issue |= val;
                self.check_cmd(port);
            }
            // TFD, SIG and SSTS are read-only; the rest is not implemented.
            _ => {}
        }
    }

    fn mem_read_32(&self, addr: u64) -> u32 {
        if addr < AHCI_GENERIC_HOST_CONTROL_REGS_MAX_ADDR {
            match addr / 4 {
                0 => self.cap,
                1 => self.ghc,
                2 => self.irqstatus,
                3 => self.impl_,
                4 => self.version,
                _ => 0,
            }
        } else if (AHCI_PORT_REGS_START_ADDR
            ..AHCI_PORT_REGS_START_ADDR + self.ports.len() as u64 * AHCI_PORT_ADDR_OFFSET_LEN)
            .contains(&addr)
        {
            self.port_read(
                ((addr - AHCI_PORT_REGS_START_ADDR) >> 7) as usize,
                addr & AHCI_PORT_ADDR_OFFSET_MASK,
            )
        } else {
            0
        }
    }

    /// `ahci_mem_read()`: unaligned 8, 16 and 32 bit reads, and aligned 64 bit reads.
    pub(crate) fn mem_read(&self, addr: u64, size: u32) -> u64 {
        let aligned = addr & !3;
        let ofst = (addr - aligned) as u32;
        let lo = u64::from(self.mem_read_32(aligned));
        let val = if ofst + size <= 4 {
            lo >> (ofst * 8)
        } else {
            let hi = u64::from(self.mem_read_32(aligned + 4));
            (hi << 32 | lo) >> (ofst * 8)
        };
        if size >= 8 { val } else { val & ((1u64 << (size * 8)) - 1) }
    }

    /// `ahci_mem_write()`. Only aligned writes are accepted.
    pub(crate) fn mem_write(&mut self, addr: u64, val: u64) {
        if addr & 3 != 0 {
            return;
        }
        if addr < AHCI_GENERIC_HOST_CONTROL_REGS_MAX_ADDR {
            match addr / 4 {
                1 => {
                    if val as u32 & HOST_CTL_RESET != 0 {
                        self.reset();
                    } else {
                        self.ghc = (val as u32 & 0x3) | HOST_CTL_AHCI_EN;
                        self.check_irq();
                    }
                }
                2 => {
                    self.irqstatus &= !(val as u32);
                    self.check_irq();
                }
                // CAP, PI and VS are read-only.
                _ => {}
            }
        } else if (AHCI_PORT_REGS_START_ADDR
            ..AHCI_PORT_REGS_START_ADDR + self.ports.len() as u64 * AHCI_PORT_ADDR_OFFSET_LEN)
            .contains(&addr)
        {
            self.port_write(
                ((addr - AHCI_PORT_REGS_START_ADDR) >> 7) as usize,
                addr & AHCI_PORT_ADDR_OFFSET_MASK,
                val as u32,
            );
        }
        self.run_pending_checks();
    }

    /// `ahci_idp_read()`.
    pub(crate) fn idp_read(&self, addr: u64, size: u32) -> u64 {
        if addr == self.idp_offset {
            u64::from(self.idp_index)
        } else if addr == self.idp_offset + 4 {
            self.mem_read(u64::from(self.idp_index), size)
        } else {
            0
        }
    }

    /// `ahci_idp_write()`.
    pub(crate) fn idp_write(&mut self, addr: u64, val: u64) {
        if addr == self.idp_offset {
            self.idp_index = val as u32 & ((AHCI_MEM_BAR_SIZE as u32 - 1) & !3);
        } else if addr == self.idp_offset + 4 {
            self.mem_write(u64::from(self.idp_index), val);
        }
    }

    /// Runs the `check_bh` bottom halves that command completion scheduled.
    fn run_pending_checks(&mut self) {
        for _ in 0..MAX_RECHECKS {
            let Some(port) = self.ports.iter().position(|p| p.check_pending) else {
                return;
            };
            self.ports[port].check_pending = false;
            self.check_cmd(port);
        }
    }

    /// `check_cmd()`.
    fn check_cmd(&mut self, port: usize) {
        let r = self.ports[port].regs;
        if r.cmd & PORT_CMD_START != 0 && r.cmd_issue != 0 {
            for slot in 0..32u8 {
                let ci = self.ports[port].regs.cmd_issue;
                if ci == 0 {
                    break;
                }
                if ci & (1 << slot) != 0 {
                    self.handle_cmd(port, slot);
                }
            }
        }
    }

    /// `ahci_init_d2h()`.
    fn init_d2h(&mut self, port: usize) {
        if self.ports[port].init_d2h_sent {
            return;
        }
        let sent = self.with_drive(port, |st, d| st.write_fis_d2h(port, d, true));
        if sent {
            let d = self.drive(port);
            let sig = u32::from(d.hcyl) << 24
                | u32::from(d.lcyl) << 16
                | u32::from(d.sector) << 8
                | (d.nsector & 0xff);
            let p = &mut self.ports[port];
            p.init_d2h_sent = true;
            p.regs.sig = sig;
        }
    }

    /// `ahci_reset_port()`.
    fn reset_port(&mut self, port: usize) {
        {
            let p = &mut self.ports[port];
            let d = p.drive.as_mut().expect("drive is in use");
            // ide_bus_reset()
            d.reset();
            d.ncq_queues = AHCI_MAX_CMDS;
            p.regs.scr_err = 0;
            p.regs.scr_act = 0;
            p.regs.tfdata = 0x7f;
            p.regs.sig = 0xffff_ffff;
            p.regs.cmd_issue = 0;
            p.busy_slot = None;
            p.init_d2h_sent = false;
            if !p.attached {
                return;
            }
            p.port_state = PortState::Run;
            let sig = if d.kind == DriveKind::Cd {
                d.status = SEEK_STAT | WRERR_STAT | READY_STAT;
                SATA_SIGNATURE_CDROM
            } else {
                d.status = SEEK_STAT | WRERR_STAT;
                SATA_SIGNATURE_DISK
            };
            // ahci_set_signature()
            d.hcyl = (sig >> 24) as u8;
            d.lcyl = (sig >> 16) as u8;
            d.sector = (sig >> 8) as u8;
            d.nsector = sig & 0xff;
            d.error = 1;
        }
        self.init_d2h(port);
    }

    /// `ahci_reset()`. As in QEMU the interrupt line is not recomputed here; the PCI reset that
    /// comes with it deasserts INTx.
    pub(crate) fn reset(&mut self) {
        self.irqstatus = 0;
        self.ghc = HOST_CTL_AHCI_EN;
        for i in 0..self.ports.len() {
            let r = &mut self.ports[i].regs;
            r.irq_stat = 0;
            r.irq_mask = 0;
            r.scr_ctl = 0;
            r.cmd = PORT_CMD_SPIN_UP | PORT_CMD_POWER_ON;
            self.reset_port(i);
        }
    }

    fn res_fis_enabled(&self, port: usize) -> Option<u64> {
        let p = &self.ports[port];
        if p.regs.cmd & PORT_CMD_FIS_RX == 0 {
            return None;
        }
        p.res_fis
    }

    /// `ahci_write_fis_sdb()`.
    fn write_fis_sdb(&mut self, port: usize) {
        let Some(res_fis) = self.res_fis_enabled(port) else {
            return;
        };
        let d = self.drive(port);
        let (status, error) = (d.status, d.error);
        let finished = self.ports[port].finished;
        let mut fis = [0u8; 8];
        fis[0] = SATA_FIS_TYPE_SDB;
        // The interrupt bit, always set for NCQ.
        fis[1] = 0x40;
        fis[2] = status & 0x77;
        fis[3] = error;
        fis[4..8].copy_from_slice(&finished.to_le_bytes());
        self.mem.dma_write(res_fis + RES_FIS_SDBFIS, &fis);

        let p = &mut self.ports[port];
        // Update the shadow registers, except BSY and DRQ.
        p.regs.tfdata = u32::from(error) << 8 | u32::from(status & 0x77) | (p.regs.tfdata & 0x88);
        p.regs.scr_act &= !finished;
        p.finished = 0;

        // TFES is raised whenever ERR is set, SDBS otherwise since the I bit is always set.
        if fis[2] & ERR_STAT != 0 {
            self.trigger_irq(port, PORT_IRQ_BIT_TFES);
        } else {
            self.trigger_irq(port, PORT_IRQ_BIT_SDBS);
        }
    }

    /// `ahci_write_fis_pio()`.
    fn write_fis_pio(&mut self, port: usize, s: &IdeDrive, len: u16, pio_fis_i: bool) {
        let Some(res_fis) = self.res_fis_enabled(port) else {
            return;
        };
        let mut fis = [0u8; 20];
        fis[0] = SATA_FIS_TYPE_PIO_SETUP;
        fis[1] = if pio_fis_i { 1 << 6 } else { 0 };
        fis[2] = s.status;
        fis[3] = s.error;
        fis[4] = s.sector;
        fis[5] = s.lcyl;
        fis[6] = s.hcyl;
        fis[7] = s.select;
        fis[8] = s.hob_sector;
        fis[9] = s.hob_lcyl;
        fis[10] = s.hob_hcyl;
        fis[12] = s.nsector as u8;
        fis[13] = (s.nsector >> 8) as u8;
        fis[15] = s.status;
        fis[16..18].copy_from_slice(&len.to_le_bytes());
        self.mem.dma_write(res_fis + RES_FIS_PSFIS, &fis);

        self.ports[port].regs.tfdata = u32::from(s.error) << 8 | u32::from(s.status);
        if fis[2] & ERR_STAT != 0 {
            self.trigger_irq(port, PORT_IRQ_BIT_TFES);
        }
    }

    /// `ahci_write_fis_d2h()`.
    fn write_fis_d2h(&mut self, port: usize, s: &IdeDrive, d2h_fis_i: bool) -> bool {
        let Some(res_fis) = self.res_fis_enabled(port) else {
            return false;
        };
        let mut fis = [0u8; 20];
        fis[0] = SATA_FIS_TYPE_REGISTER_D2H;
        fis[1] = if d2h_fis_i { 1 << 6 } else { 0 };
        fis[2] = s.status;
        fis[3] = s.error;
        fis[4] = s.sector;
        fis[5] = s.lcyl;
        fis[6] = s.hcyl;
        fis[7] = s.select;
        fis[8] = s.hob_sector;
        fis[9] = s.hob_lcyl;
        fis[10] = s.hob_hcyl;
        fis[12] = s.nsector as u8;
        fis[13] = (s.nsector >> 8) as u8;
        self.mem.dma_write(res_fis + RES_FIS_RFIS, &fis);

        self.ports[port].regs.tfdata = u32::from(s.error) << 8 | u32::from(s.status);
        if fis[2] & ERR_STAT != 0 {
            self.trigger_irq(port, PORT_IRQ_BIT_TFES);
        } else if d2h_fis_i {
            self.trigger_irq(port, PORT_IRQ_BIT_DHRS);
        }
        true
    }

    /// Reads the header of `slot`, `get_cmd_header()`.
    fn cmd_header(&self, port: usize, slot: u8) -> Option<CmdHdr> {
        let lst = self.ports[port].lst?;
        let mut h = [0u8; 16];
        if !self.mem.dma_read(lst + u64::from(slot) * 32, &mut h) {
            return None;
        }
        Some(CmdHdr { opts: le16(&h[0..]), prdtl: le16(&h[2..]), tbl_addr: le64(&h[8..]) })
    }

    fn cur_header(&self, port: usize) -> Option<CmdHdr> {
        let slot = self.ports[port].cur_slot?;
        self.cmd_header(port, slot)
    }

    /// The PRD byte count of `slot`, `AHCICmdHdr::status`.
    fn prdbc_addr(&self, port: usize, slot: u8) -> Option<u64> {
        Some(self.ports[port].lst? + u64::from(slot) * 32 + 4)
    }

    fn set_prdbc(&self, port: usize, slot: u8, val: u32) {
        if let Some(a) = self.prdbc_addr(port, slot) {
            self.mem.dma_write(a, &val.to_le_bytes());
        }
    }

    fn prdbc(&self, port: usize, slot: u8) -> u32 {
        let mut b = [0u8; 4];
        match self.prdbc_addr(port, slot) {
            Some(a) if self.mem.dma_read(a, &mut b) => u32::from_le_bytes(b),
            _ => 0,
        }
    }

    /// `ahci_populate_sglist()`: the part of the PRDT that starts `offset` bytes into the
    /// transfer, at most `limit` bytes of it.
    fn populate_sglist(&self, cmd: &CmdHdr, limit: u64, offset: u64) -> Option<SgList> {
        let prdtl = usize::from(cmd.prdtl);
        if prdtl == 0 {
            return None;
        }
        let mut prdt = vec![0u8; prdtl * 16];
        if !self.mem.dma_read(cmd.tbl_addr.wrapping_add(0x80), &mut prdt) {
            return None;
        }
        let entry = |i: usize| {
            let e = &prdt[i * 16..i * 16 + 16];
            (le64(&e[0..]), u64::from(le32(&e[12..]) & AHCI_PRDT_SIZE_MASK) + 1)
        };

        let mut sum = 0u64;
        let mut found = None;
        for i in 0..prdtl {
            let (_, size) = entry(i);
            if offset < sum + size {
                found = Some((i, offset - sum));
                break;
            }
            sum += size;
        }
        let (off_idx, off_pos) = found?;

        let mut sg = SgList::default();
        let (addr, size) = entry(off_idx);
        sg.add(addr.wrapping_add(off_pos), (size - off_pos).min(limit));
        let mut i = off_idx + 1;
        while i < prdtl && sg.size < limit {
            let (addr, size) = entry(i);
            sg.add(addr, size.min(limit - sg.size));
            i += 1;
        }
        Some(sg)
    }

    fn sg_to_guest(&self, sg: &SgList, data: &[u8]) {
        let mut done = 0usize;
        for &(addr, len) in &sg.entries {
            if done >= data.len() {
                break;
            }
            let n = (len as usize).min(data.len() - done);
            self.mem.dma_write(addr, &data[done..done + n]);
            done += n;
        }
    }

    fn sg_from_guest(&self, sg: &SgList, data: &mut [u8]) {
        let mut done = 0usize;
        for &(addr, len) in &sg.entries {
            if done >= data.len() {
                break;
            }
            let n = (len as usize).min(data.len() - done);
            self.mem.dma_read(addr, &mut data[done..done + n]);
            done += n;
        }
    }

    /// `ahci_clear_cmd_issue()`.
    fn clear_cmd_issue(&mut self, port: usize, slot: u8, status: u8) {
        if status & (ERR_STAT | BUSY_STAT | DRQ_STAT) == 0 {
            self.ports[port].regs.cmd_issue &= !(1 << slot);
        }
    }

    /// `handle_cmd()`.
    fn handle_cmd(&mut self, port: usize, slot: u8) {
        if self.drive(port).status & (BUSY_STAT | DRQ_STAT) != 0 {
            // The engine is busy; try again later.
            return;
        }
        if self.ports[port].lst.is_none() {
            return;
        }
        self.ports[port].cur_slot = Some(slot);
        if !self.ports[port].attached {
            return;
        }
        let Some(cmd) = self.cmd_header(port, slot) else {
            return;
        };

        let mut cmd_fis = [0u8; 0x80];
        if !self.mem.dma_read(cmd.tbl_addr, &mut cmd_fis) {
            // dma_memory_map() either fails outright or maps less than asked for.
            let mut probe = [0u8; 1];
            if self.mem.dma_read(cmd.tbl_addr, &mut probe) {
                self.trigger_irq(port, PORT_IRQ_BIT_HBFS);
            }
            return;
        }
        if cmd_fis[0] == SATA_FIS_TYPE_REGISTER_H2D {
            self.handle_reg_h2d_fis(port, slot, &cmd, &cmd_fis);
        }
    }

    /// `handle_reg_h2d_fis()`.
    fn handle_reg_h2d_fis(&mut self, port: usize, slot: u8, cmd: &CmdHdr, cmd_fis: &[u8; 0x80]) {
        // Port multiplier or reserved bits.
        if cmd_fis[1] & 0x0f != 0 || cmd_fis[1] & 0x70 != 0 {
            return;
        }

        if cmd_fis[1] & SATA_FIS_REG_H2D_UPDATE_COMMAND_REGISTER == 0 {
            // A control register update: the software reset sequence.
            match self.ports[port].port_state {
                PortState::Run => {
                    if cmd_fis[15] & ATA_SRST != 0 {
                        self.ports[port].port_state = PortState::Reset;
                        // Setting SRST sends no D2H FIS, so software sets "clear busy upon R_OK"
                        // to get the slot back.
                        if cmd.opts & AHCI_CMD_CLR_BUSY != 0 {
                            let status = self.drive(port).status;
                            self.clear_cmd_issue(port, slot, status);
                        }
                    }
                }
                PortState::Reset => {
                    if cmd_fis[15] & ATA_SRST == 0 {
                        self.reset_port(port);
                    }
                }
            }
            return;
        }

        if is_ncq(cmd_fis[2]) {
            self.process_ncq_command(port, slot, cmd, cmd_fis);
            return;
        }

        self.set_prdbc(port, slot, 0);
        self.ports[port].done_first_drq = false;
        self.ports[port].busy_slot = Some(slot);

        self.with_drive(port, |st, s| {
            s.feature = cmd_fis[3];
            s.sector = cmd_fis[4];
            s.lcyl = cmd_fis[5];
            s.hcyl = cmd_fis[6];
            s.select = cmd_fis[7];
            s.hob_sector = cmd_fis[8];
            s.hob_lcyl = cmd_fis[9];
            s.hob_hcyl = cmd_fis[10];
            s.hob_feature = cmd_fis[11];
            s.nsector = u32::from(cmd_fis[13]) << 8 | u32::from(cmd_fis[12]);
            s.hob_nsector = cmd_fis[13];

            // The ATAPI packet, if any, goes to the start of the I/O buffer.
            if cmd.opts & AHCI_CMD_ATAPI != 0 {
                s.io_buffer[..0x10].copy_from_slice(
                    &cmd_fis[AHCI_COMMAND_TABLE_ACMD..AHCI_COMMAND_TABLE_ACMD + 0x10],
                );
            }
            s.error = 0;

            let mut host = PortHost { st, port };
            s.exec_cmd(&mut host, cmd_fis[2]);
        });
    }

    /// `process_ncq_command()` and `execute_ncq_command()`. The transfer runs to completion
    /// right away, so a tag is never still in use when the guest issues it again.
    fn process_ncq_command(&mut self, port: usize, slot: u8, cmd: &CmdHdr, fis: &[u8; 0x80]) {
        let tag = fis[12] >> 3;
        let status = self.drive(port).status;
        // PxCI is cleared once the command is queued.
        self.clear_cmd_issue(port, slot, status);
        self.with_drive(port, |st, s| st.write_fis_d2h(port, s, false));

        let ncq_cmd = fis[2];
        let lba = u64::from(fis[10]) << 40
            | u64::from(fis[9]) << 32
            | u64::from(fis[8]) << 24
            | u64::from(fis[6]) << 16
            | u64::from(fis[5]) << 8
            | u64::from(fis[4]);
        // The sector count is in the feature fields.
        let mut sector_count = u64::from(fis[11]) << 8 | u64::from(fis[3]);
        if sector_count == 0 {
            sector_count = 0x10000;
        }
        let size = sector_count * 512;
        let sg = self.populate_sglist(cmd, size, 0).unwrap_or_default();

        if sg.size < size {
            // ncq_err()
            let d = self.ports[port].drive.as_mut().expect("drive is in use");
            d.error = crate::ide::ABRT_ERR;
            d.status = READY_STAT | ERR_STAT;
            self.trigger_irq(port, PORT_IRQ_BIT_OFS);
            return;
        }

        let blk = self.drive(port).blk.clone();
        let mut used = true;
        let ok = match (ncq_cmd, blk) {
            (READ_FPDMA_QUEUED, Some(blk)) => {
                let mut buf = vec![0u8; sg.size as usize];
                let ok = blk.read_at(lba * 512, &mut buf).is_ok();
                if ok {
                    self.sg_to_guest(&sg, &buf);
                }
                Some(ok)
            }
            (WRITE_FPDMA_QUEUED, Some(blk)) => {
                let mut buf = vec![0u8; sg.size as usize];
                self.sg_from_guest(&sg, &mut buf);
                Some(blk.write_at(lba * 512, &buf).is_ok())
            }
            (READ_FPDMA_QUEUED | WRITE_FPDMA_QUEUED, None) => Some(false),
            // Other NCQ commands are not supported: ncq_err() without a completion.
            _ => None,
        };
        let d = self.ports[port].drive.as_mut().expect("drive is in use");
        match ok {
            None => {
                d.error = crate::ide::ABRT_ERR;
                d.status = READY_STAT | ERR_STAT;
                return;
            }
            Some(true) => d.status = READY_STAT | SEEK_STAT,
            Some(false) => {
                // ncq_cb() with the "report" error policy.
                d.error = crate::ide::ABRT_ERR;
                d.status = READY_STAT | ERR_STAT;
                used = false;
            }
        }
        // ncq_finish(): a failed command gets no finished bit and keeps its PxSACT bit.
        if used {
            self.ports[port].finished |= 1 << tag;
        }
        self.write_fis_sdb(port);
    }
}

impl AhciPort {
    fn vmstate_save(&self) -> AhciPortVmState {
        let r = &self.regs;
        AhciPortVmState {
            port: IdeBusVmState::default(),
            ifs0: self.drive.as_ref().expect("drive is in use").vmstate_save(),
            port_state: match self.port_state {
                PortState::Run => 0,
                PortState::Reset => 1,
            },
            finished: self.finished,
            lst_addr: r.lst_addr,
            lst_addr_hi: r.lst_addr_hi,
            fis_addr: r.fis_addr,
            fis_addr_hi: r.fis_addr_hi,
            irq_stat: r.irq_stat,
            irq_mask: r.irq_mask,
            cmd: r.cmd,
            tfdata: r.tfdata,
            sig: r.sig,
            scr_stat: 0,
            scr_ctl: r.scr_ctl,
            scr_err: r.scr_err,
            scr_act: r.scr_act,
            cmd_issue: r.cmd_issue,
            done_first_drq: self.done_first_drq,
            busy_slot: self.busy_slot.map_or(-1, i32::from),
            init_d2h_sent: self.init_d2h_sent,
            ncq_tfs: [NcqVmState::default(); AHCI_MAX_CMDS as usize],
        }
    }
}

/// Why `v` cannot be loaded into a port: what `ahci_state_post_load()` refuses, and the busy
/// states this model has no place for.
fn port_vmstate_check(i: usize, v: &AhciPortVmState) -> Result<(), String> {
    if v.cmd & PORT_CMD_START == 0 && v.cmd & PORT_CMD_LIST_ON != 0 {
        return Err(format!("ahci: port {i}: the DMA engine is off but still running"));
    }
    if v.cmd & PORT_CMD_FIS_RX == 0 && v.cmd & PORT_CMD_FIS_ON != 0 {
        return Err(format!("ahci: port {i}: the FIS RX engine is off but still running"));
    }
    if v.ncq_tfs.iter().any(|t| t.used != t.halt) {
        return Err(format!("ahci: port {i}: NCQ commands in flight are not supported"));
    }
    if v.ncq_tfs.iter().any(|t| t.halt) {
        return Err(format!("ahci: port {i}: halted NCQ commands are not supported"));
    }
    if v.busy_slot != -1 {
        let slot = v.busy_slot;
        return Err(format!(
            "ahci: port {i}: a command in progress (slot {slot}) is not supported"
        ));
    }
    if v.port.error_status != 0 {
        return Err(format!("ahci: port {i}: a request waiting to be retried is not supported"));
    }
    if v.port.unit != 0 {
        return Err(format!("ahci: port {i}: unit {} on a one-drive bus", v.port.unit));
    }
    if v.port_state > 1 {
        return Err(format!("ahci: port {i}: bad port state {}", v.port_state));
    }
    v.ifs0.check().map_err(|e| format!("ahci: port {i}: {e}"))
}

impl AhciState {
    /// The `ahci` part of the stream.
    pub(crate) fn vmstate_save(&self) -> AhciVmState {
        AhciVmState {
            dev: self.ports.iter().map(AhciPort::vmstate_save).collect(),
            cap: self.cap,
            ghc: self.ghc,
            irqstatus: self.irqstatus,
            impl_: self.impl_,
            version: self.version,
            idp_index: self.idp_index,
            ports: self.ports.len() as u32,
        }
    }

    /// Loads the `ahci` part of the stream, then `ahci_state_post_load()`: the engines that
    /// were running are restarted, which fails if their buffers are no longer in guest memory,
    /// and the command list of every port is checked again.
    ///
    /// Only an idle controller loads: no NCQ tag in use, no command in progress, no PIO
    /// transfer, no request waiting to be retried. Anything else fails before the state
    /// changes.
    pub(crate) fn vmstate_load(&mut self, v: &AhciVmState) -> Result<(), String> {
        if v.ports as usize != self.ports.len() || v.dev.len() != self.ports.len() {
            let (n, here) = (v.dev.len(), self.ports.len());
            return Err(format!("ahci: {n} ports in the stream, {here} here"));
        }
        for (i, p) in v.dev.iter().enumerate() {
            port_vmstate_check(i, p)?;
        }

        self.cap = v.cap;
        self.ghc = v.ghc;
        self.irqstatus = v.irqstatus;
        self.impl_ = v.impl_;
        self.version = v.version;
        self.idp_index = v.idp_index;
        for (i, s) in v.dev.iter().enumerate() {
            let p = &mut self.ports[i];
            p.regs = PortRegs {
                lst_addr: s.lst_addr,
                lst_addr_hi: s.lst_addr_hi,
                fis_addr: s.fis_addr,
                fis_addr_hi: s.fis_addr_hi,
                irq_stat: s.irq_stat,
                irq_mask: s.irq_mask,
                // After a migration the engines are off and are restarted below.
                cmd: s.cmd & !(PORT_CMD_LIST_ON | PORT_CMD_FIS_ON),
                tfdata: s.tfdata,
                sig: s.sig,
                scr_ctl: s.scr_ctl,
                scr_err: s.scr_err,
                scr_act: s.scr_act,
                cmd_issue: s.cmd_issue,
            };
            p.lst = None;
            p.res_fis = None;
            p.port_state = if s.port_state == 0 { PortState::Run } else { PortState::Reset };
            p.finished = s.finished;
            p.done_first_drq = s.done_first_drq;
            p.busy_slot = None;
            p.init_d2h_sent = s.init_d2h_sent;
            p.cur_slot = None;
            p.check_pending = false;
            p.drive.as_mut().expect("drive is in use").vmstate_load(&s.ifs0)?;

            // ahci_cond_start_engines(), where a failure fails the load.
            let cmd = self.ports[i].regs.cmd;
            if cmd & PORT_CMD_START != 0 && !self.map_clb_address(i) {
                self.ports[i].regs.cmd &= !PORT_CMD_START;
                return Err(format!("ahci: port {i}: bad command list buffer address"));
            }
            if cmd & PORT_CMD_FIS_RX != 0 && !self.map_fis_address(i) {
                self.ports[i].regs.cmd &= !PORT_CMD_FIS_RX;
                return Err(format!("ahci: port {i}: bad FIS receive buffer address"));
            }
            // busy_slot is -1: look for commands that were issued but not started.
            self.check_cmd(i);
        }
        self.run_pending_checks();
        Ok(())
    }
}

/// The adapter side of a running command, `IDEDMAOps` of an AHCI port.
struct PortHost<'a> {
    st: &'a mut AhciState,
    port: usize,
}

impl IdeHost for PortHost<'_> {
    /// `ahci_pio_transfer()`.
    fn pio_transfer(&mut self, s: &mut IdeDrive, start: usize, len: usize) {
        let port = self.port;
        let opts = self.st.cur_header(port).map_or(0, |h| h.opts);
        let is_write = opts & AHCI_CMD_WRITE != 0;
        let is_atapi = opts & AHCI_CMD_ATAPI != 0;
        let done_first_drq = self.st.ports[port].done_first_drq;

        // The PIO Setup FIS comes before the data, but its interrupt only after. The I bit is
        // set for device to host transfers, and for host to device ones after the first DRQ.
        let pio_fis_i = done_first_drq || (!is_atapi && !is_write);
        self.st.write_fis_pio(port, s, len as u16, pio_fis_i);

        // The ATAPI packet is already in the I/O buffer.
        if !(is_atapi && !done_first_drq) {
            if let Some(sg) = self.sglist(len as u64, s.io_buffer_offset) {
                s.io_buffer_size = sg.size as usize;
                if len > 0 {
                    let buf = &mut s.io_buffer[start..start + len];
                    if is_write {
                        self.st.sg_from_guest(&sg, buf);
                    } else {
                        self.st.sg_to_guest(&sg, buf);
                    }
                }
            }
            s.dma_buf_commit(self, len as u32);
        }

        self.st.ports[port].done_first_drq = true;
        if pio_fis_i {
            self.st.trigger_irq(port, PORT_IRQ_BIT_PSS);
        }
    }

    fn sglist(&mut self, limit: u64, offset: u64) -> Option<SgList> {
        let cmd = self.st.cur_header(self.port)?;
        self.st.populate_sglist(&cmd, limit, offset)
    }

    fn write_guest(&mut self, sg: &SgList, data: &[u8]) {
        self.st.sg_to_guest(sg, data);
    }

    fn read_guest(&mut self, sg: &SgList, data: &mut [u8]) {
        self.st.sg_from_guest(sg, data);
    }

    /// `ahci_commit_buf()`.
    fn commit_buf(&mut self, tx_bytes: u32) {
        if let Some(slot) = self.st.ports[self.port].cur_slot {
            let v = self.st.prdbc(self.port, slot).wrapping_add(tx_bytes);
            self.st.set_prdbc(self.port, slot, v);
        }
    }

    /// `ahci_cmd_done()`.
    fn cmd_done(&mut self, s: &mut IdeDrive) {
        let port = self.port;
        if let Some(slot) = self.st.ports[port].busy_slot.take() {
            self.st.clear_cmd_issue(port, slot, s.status);
        }
        // PxCI really clears after the D2H FIS, but writing the FIS raises the interrupt, so
        // the order is reversed.
        self.st.write_fis_d2h(port, s, true);
        if s.status & ERR_STAT == 0 && self.st.ports[port].regs.cmd_issue != 0 {
            self.st.ports[port].check_pending = true;
        }
    }
}
