// SPDX-License-Identifier: GPL-2.0-or-later

//! The TPM Interface Specification FIFO interface, hw/tpm/tpm_tis_common.c, and its ISA front
//! end, hw/tpm/tpm_tis_isa.c.
//!
//! Five localities of 4 KiB each sit at 0xFED40000. A driver claims a locality through its
//! ACCESS register, writes a command into the data FIFO, sets `tpmGo` in STS and reads the
//! response back from the FIFO. The command goes to the backend when `tpmGo` is written and the
//! response is there when the write returns, so a guest polling STS never sees the command
//! executing.

use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::error_report;
use ruvm_hw_core::irq::IrqPin;
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

use crate::backend::{TpmBackend, TpmCmd};
use crate::util::{self, TpmVersion};

/// `TPM_TIS_ADDR_BASE`.
pub const TPM_TIS_ADDR_BASE: u64 = 0xfed4_0000;
/// `TPM_TIS_ADDR_SIZE`, five localities.
pub const TPM_TIS_ADDR_SIZE: u64 = 0x5000;
/// `TPM_TIS_IRQ`, the default of the `irq` property.
pub const TPM_TIS_IRQ: u32 = 5;
/// `TPM_TIS_BUFFER_MAX`.
pub const TPM_TIS_BUFFER_MAX: usize = 4096;

const NUM_LOCALITIES: u8 = 5;
const LOCALITY_SHIFT: u32 = 12;
const NO_LOCALITY: u8 = 0xff;

const REG_ACCESS: u16 = 0x00;
const REG_INT_ENABLE: u16 = 0x08;
const REG_INT_VECTOR: u16 = 0x0c;
const REG_INT_STATUS: u16 = 0x10;
const REG_INTF_CAPABILITY: u16 = 0x14;
const REG_STS: u16 = 0x18;
const REG_DATA_FIFO: u16 = 0x24;
const REG_INTERFACE_ID: u16 = 0x30;
const REG_DATA_XFIFO: u16 = 0x80;
const REG_DATA_XFIFO_END: u16 = 0xbc;
const REG_DID_VID: u16 = 0xf00;
const REG_RID: u16 = 0xf04;

const STS_TPM_FAMILY_MASK: u32 = 0x3 << 26;
const STS_TPM_FAMILY1_2: u32 = 0;
const STS_TPM_FAMILY2_0: u32 = 1 << 26;
const STS_RESET_ESTABLISHMENT_BIT: u32 = 1 << 25;
const STS_COMMAND_CANCEL: u32 = 1 << 24;
const STS_VALID: u32 = 1 << 7;
const STS_COMMAND_READY: u32 = 1 << 6;
const STS_TPM_GO: u32 = 1 << 5;
const STS_DATA_AVAILABLE: u32 = 1 << 4;
const STS_EXPECT: u32 = 1 << 3;
const STS_SELFTEST_DONE: u32 = 1 << 2;
const STS_RESPONSE_RETRY: u32 = 1 << 1;
const BURST_COUNT_SHIFT: u32 = 8;

const ACCESS_TPM_REG_VALID_STS: u8 = 1 << 7;
const ACCESS_ACTIVE_LOCALITY: u8 = 1 << 5;
const ACCESS_BEEN_SEIZED: u8 = 1 << 4;
const ACCESS_SEIZE: u8 = 1 << 3;
const ACCESS_PENDING_REQUEST: u8 = 1 << 2;
const ACCESS_REQUEST_USE: u8 = 1 << 1;

const INT_ENABLED: u32 = 1 << 31;
const INT_DATA_AVAILABLE: u32 = 1 << 0;
const INT_STS_VALID: u32 = 1 << 1;
const INT_LOCALITY_CHANGED: u32 = 1 << 2;
const INT_COMMAND_READY: u32 = 1 << 7;
const INT_POLARITY_MASK: u32 = 3 << 3;
const INT_POLARITY_LOW_LEVEL: u32 = 1 << 3;
const INTERRUPTS_SUPPORTED: u32 =
    INT_LOCALITY_CHANGED | INT_DATA_AVAILABLE | INT_STS_VALID | INT_COMMAND_READY;

const CAP_INTERFACE_VERSION1_3: u32 = 2 << 28;
const CAP_INTERFACE_VERSION1_3_FOR_TPM2_0: u32 = 3 << 28;
const CAP_DATA_TRANSFER_64B: u32 = 3 << 9;
const CAP_INTERRUPT_LOW_LEVEL: u32 = 1 << 4;
const CAPABILITIES_SUPPORTED1_3: u32 = CAP_INTERRUPT_LOW_LEVEL
    | CAP_DATA_TRANSFER_64B
    | CAP_INTERFACE_VERSION1_3
    | INTERRUPTS_SUPPORTED;
const CAPABILITIES_SUPPORTED2_0: u32 = CAP_INTERRUPT_LOW_LEVEL
    | CAP_DATA_TRANSFER_64B
    | CAP_INTERFACE_VERSION1_3_FOR_TPM2_0
    | INTERRUPTS_SUPPORTED;

const IFACE_ID_INTERFACE_TIS1_3: u32 = 0xf;
const IFACE_ID_CAP_5_LOCALITIES: u32 = 1 << 8;
const IFACE_ID_CAP_TIS_SUPPORTED: u32 = 1 << 13;
const IFACE_ID_INT_SEL_LOCK: u32 = 1 << 19;
const IFACE_ID_SUPPORTED_FLAGS1_3: u32 = IFACE_ID_INTERFACE_TIS1_3 | (!0u32 << 4);
const IFACE_ID_SUPPORTED_FLAGS2_0: u32 = IFACE_ID_CAP_5_LOCALITIES | IFACE_ID_CAP_TIS_SUPPORTED;

/// `TPM_TIS_TPM_DID`, `TPM_TIS_TPM_VID` (IBM) and `TPM_TIS_TPM_RID`.
const TPM_DID: u32 = 0x0001;
const TPM_VID: u32 = 0x1014;
const TPM_RID: u32 = 0x0001;
const NO_DATA_BYTE: u32 = 0xff;

/// `TPMTISState`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum TisState {
    #[default]
    Idle,
    Ready,
    Completion,
    Execution,
    Reception,
}

/// `TPMLocality`.
#[derive(Clone, Copy, Debug, Default)]
struct Locality {
    state: TisState,
    access: u8,
    sts: u32,
    iface_id: u32,
    inte: u32,
    ints: u32,
}

impl Locality {
    /// `tpm_tis_sts_set()`: keeps only the self test and family bits, then sets `flags`.
    fn sts_set(&mut self, flags: u32) {
        self.sts &= STS_SELFTEST_DONE | STS_TPM_FAMILY_MASK;
        self.sts |= flags;
    }
}

fn valid_locty(l: u8) -> bool {
    l < NUM_LOCALITIES
}

/// `TPMState`.
struct Tis {
    buffer: Box<[u8; TPM_TIS_BUFFER_MAX]>,
    rw_offset: u16,
    active_locty: u8,
    aborting_locty: u8,
    next_locty: u8,
    loc: [Locality; NUM_LOCALITIES as usize],
    irq_num: u32,
    be: Box<dyn TpmBackend>,
    be_version: TpmVersion,
    be_buffer_size: usize,
    had_startup_error: bool,
}

impl Tis {
    fn resp_len(&self) -> usize {
        (util::cmd_size(&self.buffer[..]) as usize).min(self.be_buffer_size)
    }

    /// `tpm_tis_raise_irq()`.
    fn raise_irq(&mut self, irq: &IrqPin, locty: u8, mask: u32) {
        if !valid_locty(locty) {
            return;
        }
        let l = &mut self.loc[locty as usize];
        if l.inte & INT_ENABLED != 0 && l.inte & mask != 0 {
            irq.raise();
            l.ints |= mask;
        }
    }

    /// `tpm_tis_check_request_use_except()`.
    fn request_use_except(&self, locty: u8) -> bool {
        (0..NUM_LOCALITIES)
            .any(|l| l != locty && self.loc[l as usize].access & ACCESS_REQUEST_USE != 0)
    }

    /// `tpm_tis_new_active_locality()`.
    fn new_active_locality(&mut self, irq: &IrqPin, new: u8) {
        let change = self.active_locty != new;
        if change && valid_locty(self.active_locty) {
            let is_seize = valid_locty(new) && self.loc[new as usize].access & ACCESS_SEIZE != 0;
            let mask = if is_seize {
                !ACCESS_ACTIVE_LOCALITY
            } else {
                !(ACCESS_ACTIVE_LOCALITY | ACCESS_REQUEST_USE)
            };
            let old = &mut self.loc[self.active_locty as usize];
            old.access &= mask;
            if is_seize {
                old.access |= ACCESS_BEEN_SEIZED;
            }
        }
        self.active_locty = new;
        if valid_locty(new) {
            let l = &mut self.loc[new as usize];
            l.access |= ACCESS_ACTIVE_LOCALITY;
            l.access &= !(ACCESS_REQUEST_USE | ACCESS_SEIZE);
        }
        if change {
            self.raise_irq(irq, self.active_locty, INT_LOCALITY_CHANGED);
        }
    }

    /// `tpm_tis_abort()`.
    fn abort(&mut self, irq: &IrqPin) {
        self.rw_offset = 0;
        if self.aborting_locty == self.next_locty {
            let l = &mut self.loc[self.aborting_locty as usize];
            l.state = TisState::Ready;
            l.sts_set(STS_COMMAND_READY);
            self.raise_irq(irq, self.aborting_locty, INT_COMMAND_READY);
        }
        self.new_active_locality(irq, self.next_locty);
        self.next_locty = NO_LOCALITY;
        self.aborting_locty = NO_LOCALITY;
    }

    /// `tpm_tis_prep_abort()`.
    fn prep_abort(&mut self, irq: &IrqPin, locty: u8, newlocty: u8) {
        self.aborting_locty = locty;
        self.next_locty = newlocty;
        if self.loc.iter().any(|l| l.state == TisState::Execution) {
            self.be.cancel();
            return;
        }
        self.abort(irq);
    }

    /// `tpm_tis_tpm_send()` and, as the backend answers right away,
    /// `tpm_tis_request_completed()`.
    fn tpm_send(&mut self, irq: &IrqPin, locty: u8) {
        self.loc[locty as usize].state = TisState::Execution;
        let len = self.be_buffer_size;
        let input = self.buffer[..self.rw_offset as usize].to_vec();
        let mut cmd =
            TpmCmd { locty, input: &input, output: &mut self.buffer[..len], selftest_done: false };
        if let Err(e) = self.be.handle_request(&mut cmd) {
            error_report(&e);
        }
        let selftest_done = cmd.selftest_done;
        self.request_completed(irq, locty, selftest_done);
    }

    /// `tpm_tis_request_completed()`.
    fn request_completed(&mut self, irq: &IrqPin, locty: u8, selftest_done: bool) {
        if selftest_done {
            for l in &mut self.loc {
                l.sts |= STS_SELFTEST_DONE;
            }
        }
        let l = &mut self.loc[locty as usize];
        l.sts_set(STS_VALID | STS_DATA_AVAILABLE);
        l.state = TisState::Completion;
        self.rw_offset = 0;
        if valid_locty(self.next_locty) {
            self.abort(irq);
        }
        self.raise_irq(irq, locty, INT_DATA_AVAILABLE | INT_STS_VALID);
    }

    /// `tpm_tis_data_read()`.
    fn data_read(&mut self, irq: &IrqPin, locty: u8) -> u32 {
        if self.loc[locty as usize].sts & STS_DATA_AVAILABLE == 0 {
            return NO_DATA_BYTE;
        }
        let len = self.resp_len();
        let ret = u32::from(self.buffer[self.rw_offset as usize]);
        self.rw_offset += 1;
        if self.rw_offset as usize >= len {
            self.loc[locty as usize].sts_set(STS_VALID);
            self.raise_irq(irq, locty, INT_STS_VALID);
        }
        ret
    }

    /// `tpm_tis_mmio_read()`.
    fn read(&mut self, irq: &IrqPin, addr: u64, mut size: u32) -> u32 {
        if self.had_startup_error {
            return 0;
        }
        let offset = (addr & 0xffc) as u16;
        let mut shift = (addr & 3) as u32 * 8;
        let locty = ((addr >> LOCALITY_SHIFT) & 7) as u8;
        let mut val = 0xffff_ffff;
        match offset {
            REG_ACCESS => {
                // SEIZE is internal; the pending flag is worked out on each read.
                val = u32::from(self.loc[locty as usize].access & !ACCESS_SEIZE);
                if self.request_use_except(locty) {
                    val |= u32::from(ACCESS_PENDING_REQUEST);
                }
                val |= u32::from(!self.be.established_flag());
            }
            REG_INT_ENABLE => val = self.loc[locty as usize].inte,
            REG_INT_VECTOR => val = self.irq_num,
            REG_INT_STATUS => val = self.loc[locty as usize].ints,
            REG_INTF_CAPABILITY => {
                val = match self.be_version {
                    TpmVersion::V1_2 => CAPABILITIES_SUPPORTED1_3,
                    TpmVersion::V2_0 => CAPABILITIES_SUPPORTED2_0,
                };
            }
            REG_STS => {
                if self.active_locty == locty {
                    let sts = self.loc[locty as usize].sts;
                    let burst = if sts & STS_DATA_AVAILABLE != 0 {
                        (self.resp_len() as u32).wrapping_sub(u32::from(self.rw_offset))
                    } else {
                        let avail = self.be_buffer_size as u32 - u32::from(self.rw_offset);
                        // A byte read must not show 0x00 for 0x100 bytes.
                        if size == 1 && avail > 0xff { 0xff } else { avail }
                    };
                    val = (burst << BURST_COUNT_SHIFT) | sts;
                }
            }
            REG_DATA_FIFO | REG_DATA_XFIFO..=REG_DATA_XFIFO_END => {
                if self.active_locty == locty {
                    size = size.min(4 - (addr & 3) as u32);
                    val = 0;
                    shift = 0;
                    while size > 0 {
                        let v = match self.loc[locty as usize].state {
                            TisState::Completion => self.data_read(irq, locty),
                            _ => NO_DATA_BYTE,
                        };
                        val |= v << shift;
                        shift += 8;
                        size -= 1;
                    }
                    shift = 0;
                }
            }
            REG_INTERFACE_ID => val = self.loc[locty as usize].iface_id,
            REG_DID_VID => val = (TPM_DID << 16) | TPM_VID,
            REG_RID => val = TPM_RID,
            _ => {}
        }
        val >> shift
    }

    /// `tpm_tis_mmio_write()`.
    fn write(&mut self, irq: &IrqPin, addr: u64, val: u64, mut size: u32) {
        let off = (addr & 0xffc) as u16;
        let shift = (addr & 3) as u32 * 8;
        let locty = ((addr >> LOCALITY_SHIFT) & 7) as u8;
        if locty == 4 || self.had_startup_error {
            return;
        }
        let mut mask: u32 = match size {
            1 => 0xff,
            2 => 0xffff,
            _ => !0,
        };
        let mut val = val as u32 & mask;
        if shift != 0 {
            val <<= shift;
            mask <<= shift;
        }
        mask ^= 0xffff_ffff;
        let li = locty as usize;

        match off {
            REG_ACCESS => self.write_access(irq, locty, val as u8),
            REG_INT_ENABLE => {
                let l = &mut self.loc[li];
                l.inte &= mask;
                l.inte |= val & (INT_ENABLED | INT_POLARITY_MASK | INTERRUPTS_SUPPORTED);
            }
            REG_INT_VECTOR => {}
            REG_INT_STATUS => {
                let l = &mut self.loc[li];
                if val & INTERRUPTS_SUPPORTED != 0 && l.ints & INTERRUPTS_SUPPORTED != 0 {
                    l.ints &= !val;
                    if l.ints == 0 {
                        irq.lower();
                    }
                }
                l.ints &= !(val & INTERRUPTS_SUPPORTED);
            }
            REG_STS => self.write_sts(irq, locty, val),
            REG_DATA_FIFO | REG_DATA_XFIFO..=REG_DATA_XFIFO_END => {
                if self.active_locty != locty {
                    return;
                }
                let state = self.loc[li].state;
                if matches!(state, TisState::Idle | TisState::Execution | TisState::Completion) {
                    return;
                }
                if state == TisState::Ready {
                    self.loc[li].state = TisState::Reception;
                    self.loc[li].sts_set(STS_EXPECT | STS_VALID);
                }
                val >>= shift;
                size = size.min(4 - (addr & 3) as u32);
                while self.loc[li].sts & STS_EXPECT != 0 && size > 0 {
                    if (self.rw_offset as usize) < self.be_buffer_size {
                        self.buffer[self.rw_offset as usize] = val as u8;
                        self.rw_offset += 1;
                        val >>= 8;
                        size -= 1;
                    } else {
                        self.loc[li].sts_set(STS_VALID);
                    }
                }
                // Once the length is in, see whether the whole command is.
                if self.rw_offset > 5 && self.loc[li].sts & STS_EXPECT != 0 {
                    let need_irq = self.loc[li].sts & STS_VALID == 0;
                    let len = util::cmd_size(&self.buffer[..]);
                    if len > u32::from(self.rw_offset) {
                        self.loc[li].sts_set(STS_EXPECT | STS_VALID);
                    } else {
                        self.loc[li].sts_set(STS_VALID);
                    }
                    if need_irq {
                        self.raise_irq(irq, locty, INT_STS_VALID);
                    }
                }
            }
            REG_INTERFACE_ID if val & IFACE_ID_INT_SEL_LOCK != 0 => {
                for l in &mut self.loc {
                    l.iface_id |= IFACE_ID_INT_SEL_LOCK;
                }
            }
            _ => {}
        }
    }

    /// The ACCESS case of `tpm_tis_mmio_write()`.
    fn write_access(&mut self, irq: &IrqPin, locty: u8, mut val: u8) {
        let li = locty as usize;
        let mut set_new_locty = true;
        if val & ACCESS_SEIZE != 0 {
            val &= !(ACCESS_REQUEST_USE | ACCESS_ACTIVE_LOCALITY);
        }
        let mut active_locty = self.active_locty;

        if val & ACCESS_ACTIVE_LOCALITY != 0 {
            if self.active_locty == locty {
                // Giving the locality up: does anybody want it?
                let newlocty = (0..NUM_LOCALITIES)
                    .rev()
                    .find(|&c| self.loc[c as usize].access & ACCESS_REQUEST_USE != 0)
                    .unwrap_or(NO_LOCALITY);
                if valid_locty(newlocty) {
                    set_new_locty = false;
                    self.prep_abort(irq, locty, newlocty);
                } else {
                    active_locty = NO_LOCALITY;
                }
            } else {
                // Not the owner: clear a pending request.
                self.loc[li].access &= !ACCESS_REQUEST_USE;
            }
        }

        if val & ACCESS_BEEN_SEIZED != 0 {
            self.loc[li].access &= !ACCESS_BEEN_SEIZED;
        }

        // A seize works when no locality is active or the active one is lower.
        if val & ACCESS_SEIZE != 0
            && (!valid_locty(self.active_locty) || locty > self.active_locty)
            && self.loc[li].access & ACCESS_SEIZE == 0
            && !(locty + 1..NUM_LOCALITIES).any(|l| self.loc[l as usize].access & ACCESS_SEIZE != 0)
        {
            for l in 0..li {
                self.loc[l].access &= !ACCESS_SEIZE;
            }
            self.loc[li].access |= ACCESS_SEIZE;
            set_new_locty = false;
            self.prep_abort(irq, self.active_locty, locty);
        }

        if val & ACCESS_REQUEST_USE != 0 && self.active_locty != locty {
            if valid_locty(self.active_locty) {
                self.loc[li].access |= ACCESS_REQUEST_USE;
            } else {
                // Nobody is active, so this one is now.
                active_locty = locty;
            }
        }

        if set_new_locty {
            self.new_active_locality(irq, active_locty);
        }
    }

    /// The STS case of `tpm_tis_mmio_write()`.
    fn write_sts(&mut self, irq: &IrqPin, locty: u8, mut val: u32) {
        let li = locty as usize;
        if self.active_locty != locty {
            return;
        }
        if self.be_version == TpmVersion::V2_0 {
            if val & STS_COMMAND_CANCEL != 0 && self.loc[li].state == TisState::Execution {
                self.be.cancel();
            }
            if val & STS_RESET_ESTABLISHMENT_BIT != 0 && (locty == 3 || locty == 4) {
                if let Err(e) = self.be.reset_established_flag(locty) {
                    error_report(&e);
                }
            }
        }
        val &= STS_COMMAND_READY | STS_TPM_GO | STS_RESPONSE_RETRY;
        match val {
            STS_COMMAND_READY => match self.loc[li].state {
                TisState::Ready => self.rw_offset = 0,
                TisState::Idle => {
                    self.loc[li].sts_set(STS_COMMAND_READY);
                    self.loc[li].state = TisState::Ready;
                    self.raise_irq(irq, locty, INT_COMMAND_READY);
                }
                TisState::Execution | TisState::Reception => {
                    self.prep_abort(irq, locty, locty);
                }
                TisState::Completion => {
                    self.rw_offset = 0;
                    // A shortcut to ready with commandReady set.
                    self.loc[li].state = TisState::Ready;
                    if self.loc[li].sts & STS_COMMAND_READY == 0 {
                        self.loc[li].sts_set(STS_COMMAND_READY);
                        self.raise_irq(irq, locty, INT_COMMAND_READY);
                    }
                    self.loc[li].sts &= !STS_DATA_AVAILABLE;
                }
            },
            STS_TPM_GO => {
                if self.loc[li].state == TisState::Reception && self.loc[li].sts & STS_EXPECT == 0 {
                    self.tpm_send(irq, locty);
                }
            }
            STS_RESPONSE_RETRY if self.loc[li].state == TisState::Completion => {
                self.rw_offset = 0;
                self.loc[li].sts_set(STS_VALID | STS_DATA_AVAILABLE);
            }
            _ => {}
        }
    }

    /// `tpm_tis_reset()` without the PPI part.
    fn reset(&mut self) -> Result<(), String> {
        self.be_version = self.be.version();
        self.be_buffer_size = self.be.buffer_size().min(TPM_TIS_BUFFER_MAX);
        self.had_startup_error = false;
        self.active_locty = NO_LOCALITY;
        self.next_locty = NO_LOCALITY;
        self.aborting_locty = NO_LOCALITY;
        let (sts, iface_id) = match self.be_version {
            TpmVersion::V1_2 => (STS_TPM_FAMILY1_2, IFACE_ID_SUPPORTED_FLAGS1_3),
            TpmVersion::V2_0 => (STS_TPM_FAMILY2_0, IFACE_ID_SUPPORTED_FLAGS2_0),
        };
        self.loc.fill(Locality {
            state: TisState::Idle,
            access: ACCESS_TPM_REG_VALID_STS,
            sts,
            iface_id,
            inte: INT_POLARITY_LOW_LEVEL,
            ints: 0,
        });
        self.rw_offset = 0;
        let r = self.be.startup(self.be_buffer_size);
        self.had_startup_error = r.is_err();
        r
    }
}

/// `-device tpm-tis`: the TIS registers over a backend.
pub struct TpmTis {
    state: Mutex<Tis>,
    irq: IrqPin,
}

impl std::fmt::Debug for TpmTis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TpmTis").finish_non_exhaustive()
    }
}

impl TpmTis {
    /// A TIS over `be` with interrupt vector `irq_num`. The registers make sense only after
    /// [`TpmTis::reset`].
    pub fn new(be: Box<dyn TpmBackend>, irq_num: u32) -> Arc<TpmTis> {
        let be_version = be.version();
        Arc::new(TpmTis {
            state: Mutex::new(Tis {
                buffer: Box::new([0; TPM_TIS_BUFFER_MAX]),
                rw_offset: 0,
                active_locty: NO_LOCALITY,
                aborting_locty: NO_LOCALITY,
                next_locty: NO_LOCALITY,
                loc: [Locality::default(); NUM_LOCALITIES as usize],
                irq_num,
                be,
                be_version,
                be_buffer_size: TPM_TIS_BUFFER_MAX,
                had_startup_error: false,
            }),
            irq: IrqPin::new(),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Tis> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The interrupt output, `s->irq`.
    pub fn irq(&self) -> &IrqPin {
        &self.irq
    }

    /// `tpm_tis_get_tpm_version()`: `None` after a failed startup.
    pub fn tpm_version(&self) -> Option<TpmVersion> {
        let s = self.lock();
        (!s.had_startup_error).then(|| s.be.version())
    }

    /// `tpm_tis_reset()`: resets the registers and starts the TPM, which QEMU does at every
    /// machine reset. QEMU exits when the TPM does not start.
    pub fn reset(&self) -> Result<(), String> {
        self.lock().reset()
    }

    /// A register read, `tpm_tis_read_data()`.
    pub fn read_data(&self, addr: u64, size: u32) -> u32 {
        self.lock().read(&self.irq, addr, size)
    }

    /// A register write, `tpm_tis_write_data()`.
    pub fn write_data(&self, addr: u64, val: u64, size: u32) {
        self.lock().write(&self.irq, addr, val, size)
    }
}

impl MmioOps for TpmTis {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.read_data(offset, size.bytes())))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        self.write_data(offset, value, size.bytes());
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }
}
