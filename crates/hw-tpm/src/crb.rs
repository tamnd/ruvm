// SPDX-License-Identifier: GPL-2.0-or-later

//! The Command Response Buffer interface, hw/tpm/tpm_crb.c.
//!
//! The control registers sit at 0xFED40000 and the command buffer follows them at 0xFED40080.
//! Only locality 0 is there. A driver requests the locality, puts a command into the buffer
//! and writes `Start`. The command goes to the backend during that write and the response
//! replaces the command in the buffer, so a driver polling `Start` finds it clear. Commands and
//! responses larger than the buffer go through it in chunks with `nextChunk`.

use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::error_report;
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

use crate::backend::{TpmBackend, TpmCmd};
use crate::util::{self, TPM_HDR_SIZE, TpmVersion};

/// `TPM_CRB_ADDR_BASE`.
pub const TPM_CRB_ADDR_BASE: u64 = 0xfed4_0000;
/// `TPM_CRB_ADDR_SIZE`: the registers and the command buffer.
pub const TPM_CRB_ADDR_SIZE: u64 = 0x1000;
/// `A_CRB_DATA_BUFFER`, where the command buffer starts and the size of the registers.
pub const TPM_CRB_DATA_BUFFER: u64 = 0x80;
/// `CRB_CTRL_CMD_SIZE`, the size of the command buffer.
pub const TPM_CRB_CMD_SIZE: usize = (TPM_CRB_ADDR_SIZE - TPM_CRB_DATA_BUFFER) as usize;

const R_LOC_STATE: usize = 0;
const R_LOC_STS: usize = 0x0c / 4;
const R_INTF_ID: usize = 0x30 / 4;
const R_INTF_ID2: usize = 0x34 / 4;
const R_CTRL_STS: usize = 0x44 / 4;
const R_CTRL_START: usize = 0x4c / 4;
const R_CTRL_CMD_SIZE: usize = 0x58 / 4;
const R_CTRL_CMD_LADDR: usize = 0x5c / 4;
const R_CTRL_RSP_SIZE: usize = 0x64 / 4;
const R_CTRL_RSP_ADDR: usize = 0x68 / 4;
/// `TPM_CRB_R_MAX`.
const R_MAX: usize = TPM_CRB_DATA_BUFFER as usize / 4;

const A_LOC_STATE: u64 = 0x00;
const A_LOC_CTRL: u64 = 0x08;
const A_CTRL_REQ: u64 = 0x40;
const A_CTRL_CANCEL: u64 = 0x48;
const A_CTRL_START: u64 = 0x4c;

// CRB_LOC_STATE
const LOC_STATE_LOC_ASSIGNED: u32 = 1 << 1;
const LOC_STATE_ACTIVE_LOCALITY_SHIFT: u32 = 2;
const LOC_STATE_TPM_REG_VALID_STS: u32 = 1 << 7;
// CRB_LOC_STS
const LOC_STS_GRANTED: u32 = 1 << 0;
const LOC_STS_BEEN_SEIZED: u32 = 1 << 1;
// CRB_CTRL_STS
const CTRL_STS_TPM_STS: u32 = 1 << 0;
const CTRL_STS_TPM_IDLE: u32 = 1 << 1;
// CRB_CTRL_START
const START_INVOKE: u32 = 1 << 0;
const START_RSP_RETRY: u32 = 1 << 1;
const START_NEXT_CHUNK: u32 = 1 << 2;

const LOC_CTRL_REQUEST_ACCESS: u64 = 1 << 0;
const LOC_CTRL_RELINQUISH: u64 = 1 << 1;
const LOC_CTRL_RESET_ESTABLISHMENT_BIT: u64 = 1 << 3;
const CTRL_REQ_CMD_READY: u64 = 1 << 0;
const CTRL_REQ_GO_IDLE: u64 = 1 << 1;
const CANCEL_INVOKE: u64 = 1 << 0;

/// `CRB_INTF_ID` after reset without the chunk capability: CRB active, CRB version 1, locality
/// 0 only, 64-byte transfers, CRB supported and selected.
const INTF_ID: u32 = 0b1 | 0b1 << 4 | 0b11 << 11 | 0b1 << 14 | 0b1 << 17;
/// `CapCRBChunk` in `CRB_INTF_ID`.
const INTF_ID_CAP_CRB_CHUNK: u32 = 1 << 10;
/// `PCI_VENDOR_ID_IBM`, the vendor in `CRB_INTF_ID2`.
const VID_IBM: u32 = 0x1014;

const NO_LOCALITY: u8 = 0xff;

/// `CRBState` without the memory regions.
struct Crb {
    be: Box<dyn TpmBackend>,
    regs: [u32; R_MAX],
    /// `cmdmem`, RAM in QEMU.
    cmdmem: Box<[u8; TPM_CRB_CMD_SIZE]>,
    be_buffer_size: usize,
    command_buffer: Vec<u8>,
    response_buffer: Vec<u8>,
    response_offset: usize,
    cap_chunk: bool,
}

impl Crb {
    /// `tpm_crb_clear_internal_buffers()`.
    fn clear_internal_buffers(&mut self) {
        self.response_buffer.clear();
        self.command_buffer.clear();
        self.response_offset = 0;
    }

    fn read(&mut self, addr: u64, size: u32) -> u32 {
        let reg = (addr & !3) as usize / 4;
        let mut val = self.regs.get(reg).copied().unwrap_or(0) >> (8 * (addr & 3));
        if addr == A_LOC_STATE {
            val |= u32::from(!self.be.established_flag());
        }
        if size < 4 { val & ((1 << (8 * size)) - 1) } else { val }
    }

    /// `tpm_crb_get_active_locty()`.
    fn active_locty(&self) -> u8 {
        let state = self.regs[R_LOC_STATE];
        if state & LOC_STATE_LOC_ASSIGNED == 0 {
            return NO_LOCALITY;
        }
        ((state >> LOC_STATE_ACTIVE_LOCALITY_SHIFT) & 7) as u8
    }

    /// `tpm_crb_append_command_request()`: adds what is in the command buffer to the command.
    fn append_command_request(&mut self) -> bool {
        let total = if self.command_buffer.is_empty() {
            let total = util::cmd_size(&self.cmdmem[..]) as usize;
            if total < TPM_HDR_SIZE {
                self.regs[R_CTRL_STS] |= CTRL_STS_TPM_STS;
                self.regs[R_CTRL_START] &= !(START_INVOKE | START_NEXT_CHUNK);
                self.clear_internal_buffers();
                error_report(&format!(
                    "Command size {total} less than TPM header size {TPM_HDR_SIZE}"
                ));
                return false;
            }
            total
        } else {
            util::cmd_size(&self.command_buffer) as usize
        };
        let total = total.min(self.be_buffer_size);
        if total > self.command_buffer.len() {
            let to_copy = (total - self.command_buffer.len()).min(TPM_CRB_CMD_SIZE);
            self.command_buffer.extend_from_slice(&self.cmdmem[..to_copy]);
        }
        true
    }

    /// `tpm_crb_fill_command_response()`: puts the next part of the response into the command
    /// buffer.
    fn fill_command_response(&mut self) {
        let remaining = self.response_buffer.len() - self.response_offset;
        let to_copy = remaining.min(TPM_CRB_CMD_SIZE);
        let from = self.response_offset;
        self.cmdmem[..to_copy].copy_from_slice(&self.response_buffer[from..from + to_copy]);
        self.cmdmem[to_copy..].fill(0);
        self.response_offset += to_copy;
    }

    fn write(&mut self, addr: u64, val: u64) {
        let locty = (addr >> 12) as u8;
        match addr {
            A_CTRL_REQ => match val {
                CTRL_REQ_CMD_READY => self.regs[R_CTRL_STS] &= !CTRL_STS_TPM_IDLE,
                CTRL_REQ_GO_IDLE => self.regs[R_CTRL_STS] |= CTRL_STS_TPM_IDLE,
                _ => {}
            },
            A_CTRL_CANCEL => {
                if val == CANCEL_INVOKE {
                    if self.regs[R_CTRL_START] & START_INVOKE != 0 {
                        self.be.cancel();
                    }
                    self.clear_internal_buffers();
                }
            }
            A_CTRL_START => {
                if self.active_locty() != locty || self.regs[R_CTRL_START] & START_INVOKE != 0 {
                    return;
                }
                if val & u64::from(START_INVOKE) != 0 {
                    if !self.append_command_request() {
                        return;
                    }
                    self.regs[R_CTRL_START] |= START_INVOKE;
                    self.deliver_request();
                } else if val & u64::from(START_NEXT_CHUNK) != 0 {
                    if !self.cap_chunk {
                        return;
                    }
                    // nextChunk goes both ways: a response not all read yet means the guest
                    // wants more of it, otherwise it is sending more of the command.
                    if !self.response_buffer.is_empty()
                        && self.response_offset < self.response_buffer.len()
                    {
                        self.fill_command_response();
                    } else if !self.append_command_request() {
                        return;
                    }
                    self.regs[R_CTRL_START] &= !START_NEXT_CHUNK;
                } else if val & u64::from(START_RSP_RETRY) != 0 {
                    if !self.cap_chunk {
                        return;
                    }
                    if !self.response_buffer.is_empty() {
                        self.response_offset = 0;
                        self.fill_command_response();
                    }
                    self.regs[R_CTRL_START] &= !(START_RSP_RETRY | START_NEXT_CHUNK);
                }
            }
            A_LOC_CTRL => match val {
                // Not for localities 3 and 4, the only ones that may reset it.
                LOC_CTRL_RESET_ESTABLISHMENT_BIT => {}
                LOC_CTRL_RELINQUISH => {
                    self.regs[R_LOC_STATE] &= !LOC_STATE_LOC_ASSIGNED;
                    self.regs[R_LOC_STS] &= !LOC_STS_GRANTED;
                }
                LOC_CTRL_REQUEST_ACCESS => {
                    self.regs[R_LOC_STS] |= LOC_STS_GRANTED;
                    self.regs[R_LOC_STS] &= !LOC_STS_BEEN_SEIZED;
                    self.regs[R_LOC_STATE] |= LOC_STATE_LOC_ASSIGNED;
                }
                _ => {}
            },
            _ => {}
        }
    }

    /// `tpm_backend_deliver_request()` and `tpm_crb_request_completed()`.
    fn deliver_request(&mut self) {
        self.response_buffer.resize(self.be_buffer_size, 0);
        let mut cmd = TpmCmd {
            locty: 0,
            input: &self.command_buffer,
            output: &mut self.response_buffer,
            selftest_done: false,
        };
        let ok = match self.be.handle_request(&mut cmd) {
            Ok(()) => true,
            Err(e) => {
                error_report(&e);
                false
            }
        };
        self.regs[R_CTRL_START] &= !START_INVOKE;
        if ok {
            let len = (util::cmd_size(&self.response_buffer) as usize).min(self.be_buffer_size);
            self.response_buffer.truncate(len);
            self.response_offset = 0;
        } else {
            // A fatal error.
            self.regs[R_CTRL_STS] |= CTRL_STS_TPM_STS;
            self.clear_internal_buffers();
        }
        // The first chunk now, the others when the guest asks with nextChunk.
        self.fill_command_response();
        self.regs[R_CTRL_START] &= !(START_NEXT_CHUNK | START_RSP_RETRY);
        self.command_buffer.clear();
    }

    /// `tpm_crb_reset()`.
    fn reset(&mut self) -> Result<(), String> {
        self.clear_internal_buffers();
        self.regs = [0; R_MAX];
        self.regs[R_LOC_STATE] = LOC_STATE_TPM_REG_VALID_STS;
        self.regs[R_CTRL_STS] = CTRL_STS_TPM_IDLE;
        self.regs[R_INTF_ID] = INTF_ID | if self.cap_chunk { INTF_ID_CAP_CRB_CHUNK } else { 0 };
        self.regs[R_INTF_ID2] = VID_IBM;
        let buffer = (TPM_CRB_ADDR_BASE + TPM_CRB_DATA_BUFFER) as u32;
        self.regs[R_CTRL_CMD_SIZE] = TPM_CRB_CMD_SIZE as u32;
        self.regs[R_CTRL_CMD_LADDR] = buffer;
        self.regs[R_CTRL_RSP_SIZE] = TPM_CRB_CMD_SIZE as u32;
        self.regs[R_CTRL_RSP_ADDR] = buffer;
        self.be_buffer_size = self.be.buffer_size();
        self.be.startup(self.be_buffer_size)
    }
}

/// `-device tpm-crb`: the CRB registers and command buffer over a backend.
pub struct TpmCrb {
    state: Mutex<Crb>,
}

impl std::fmt::Debug for TpmCrb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TpmCrb").finish_non_exhaustive()
    }
}

impl TpmCrb {
    /// A CRB over `be`. `cap_chunk` is the `cap-chunk` property. The registers make sense only
    /// after [`TpmCrb::reset`].
    pub fn new(be: Box<dyn TpmBackend>, cap_chunk: bool) -> Arc<TpmCrb> {
        Arc::new(TpmCrb {
            state: Mutex::new(Crb {
                be,
                regs: [0; R_MAX],
                cmdmem: Box::new([0; TPM_CRB_CMD_SIZE]),
                be_buffer_size: 0,
                command_buffer: Vec::new(),
                response_buffer: Vec::new(),
                response_offset: 0,
                cap_chunk,
            }),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Crb> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `tpm_crb_get_version()`.
    pub fn tpm_version(&self) -> TpmVersion {
        self.lock().be.version()
    }

    /// `tpm_crb_reset()`: resets the registers and starts the TPM, which QEMU does at every
    /// machine reset. QEMU exits when the TPM does not start.
    pub fn reset(&self) -> Result<(), String> {
        self.lock().reset()
    }

    /// The command buffer, `tpm-crb-cmd`, to map at [`TPM_CRB_DATA_BUFFER`] past the
    /// registers.
    pub fn cmd_region(self: &Arc<Self>) -> Arc<dyn MmioOps> {
        Arc::new(CrbCmd(Arc::clone(self)))
    }
}

/// `tpm_crb_memory_ops`, the registers, `tpm-crb-mmio`.
impl MmioOps for TpmCrb {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.lock().read(offset, size.bytes())))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.lock().write(offset, value);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }
}

/// The command buffer, RAM in QEMU.
struct CrbCmd(Arc<TpmCrb>);

impl MmioOps for CrbCmd {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        let s = self.0.lock();
        let mut v = [0; 8];
        let (o, n) = (offset as usize, size.bytes() as usize);
        let n = n.min(TPM_CRB_CMD_SIZE.saturating_sub(o));
        v[..n].copy_from_slice(&s.cmdmem[o..o + n]);
        Ok(u64::from_le_bytes(v))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        let mut s = self.0.lock();
        let (o, n) = (offset as usize, size.bytes() as usize);
        let n = n.min(TPM_CRB_CMD_SIZE.saturating_sub(o));
        s.cmdmem[o..o + n].copy_from_slice(&value.to_le_bytes()[..n]);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 8).allow_unaligned()
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 8).allow_unaligned()
    }
}
