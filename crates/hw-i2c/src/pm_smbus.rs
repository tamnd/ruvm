// SPDX-License-Identifier: GPL-2.0-or-later

//! The PC SMBus host controller, QEMU's `hw/i2c/pm_smbus.c`.
//!
//! This is the Intel style register block shared by the PIIX4 power management function, the
//! ICH9 SMBus function and the VIA south bridges. The guest loads an address, a command and data
//! into the host registers and writes the protocol and START to HST_CNT. The transaction runs
//! at once and the result lands in HST_STS. Block transfers either go a byte at a time through
//! HOST_BLOCK_DB with BYTE_DONE handshakes, or through the 32 byte buffer when AUX_CTL.E32B is
//! set.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, Endian, MemResult, MmioOps};

use crate::i2c::I2cBus;
use crate::smbus::{
    smbus_quick_command, smbus_read_block, smbus_read_byte, smbus_read_word, smbus_receive_byte,
    smbus_send_byte, smbus_write_block, smbus_write_byte, smbus_write_word,
};

/// `PM_SMBUS_MAX_MSG_SIZE`: the block buffer.
pub const PM_SMBUS_MAX_MSG_SIZE: usize = 32;

/// The size of the register region QEMU creates, `pm-smbus`. Only the first 14 bytes decode.
pub const PM_SMBUS_IO_SIZE: u64 = 64;

pub const SMBHSTSTS: u64 = 0x00;
pub const SMBHSTCNT: u64 = 0x02;
pub const SMBHSTCMD: u64 = 0x03;
pub const SMBHSTADD: u64 = 0x04;
pub const SMBHSTDAT0: u64 = 0x05;
pub const SMBHSTDAT1: u64 = 0x06;
pub const SMBBLKDAT: u64 = 0x07;
pub const SMBAUXCTL: u64 = 0x0d;

pub const STS_HOST_BUSY: u8 = 1 << 0;
pub const STS_INTR: u8 = 1 << 1;
pub const STS_DEV_ERR: u8 = 1 << 2;
pub const STS_BUS_ERR: u8 = 1 << 3;
pub const STS_FAILED: u8 = 1 << 4;
pub const STS_SMBALERT: u8 = 1 << 5;
pub const STS_INUSE_STS: u8 = 1 << 6;
pub const STS_BYTE_DONE: u8 = 1 << 7;

pub const CTL_INTREN: u8 = 1 << 0;
pub const CTL_KILL: u8 = 1 << 1;
pub const CTL_LAST_BYTE: u8 = 1 << 5;
pub const CTL_START: u8 = 1 << 6;
pub const CTL_PEC_EN: u8 = 1 << 7;
pub const CTL_RETURN_MASK: u8 = 0x1f;

pub const PROT_QUICK: u8 = 0;
pub const PROT_BYTE: u8 = 1;
pub const PROT_BYTE_DATA: u8 = 2;
pub const PROT_WORD_DATA: u8 = 3;
pub const PROT_PROC_CALL: u8 = 4;
pub const PROT_BLOCK_DATA: u8 = 5;
pub const PROT_I2C_BLOCK_READ: u8 = 6;

pub const AUX_PEC: u8 = 1 << 0;
pub const AUX_BLK: u8 = 1 << 1;
pub const AUX_MASK: u8 = 0x3;

/// Called with the new interrupt level after every register access, `PMSMBus::set_irq`.
pub type PmSmbusIrqFn = Arc<dyn Fn(bool) + Send + Sync>;

/// The register state of `PMSMBus`, everything QEMU migrates.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PmSmbusRegs {
    pub smb_stat: u8,
    pub smb_ctl: u8,
    pub smb_cmd: u8,
    pub smb_addr: u8,
    pub smb_data0: u8,
    pub smb_data1: u8,
    pub smb_data: [u8; PM_SMBUS_MAX_MSG_SIZE],
    pub smb_blkdata: u8,
    pub smb_auxctl: u8,
    pub smb_index: u32,
    /// HOSTC.I2C_EN on ICH9: block transfers leave out the length byte.
    pub i2c_enable: bool,
    /// The block transfer in progress is finished, so INTR can be raised at the right time.
    pub op_done: bool,
    pub in_i2c_block_read: bool,
    /// The AMIBIOS workaround, see [`PmSmbus`].
    pub start_transaction_on_status_read: bool,
}

/// A PM SMBus host controller, `PMSMBus`.
///
/// With HST_CNT.INTREN clear, START does not run the transaction. It sets HOST_BUSY and the
/// transaction runs on the next read of HST_STS. QEMU does this for an AMIBIOS that waits to see
/// HOST_BUSY before it polls for completion.
pub struct PmSmbus {
    bus: Arc<I2cBus>,
    regs: Mutex<PmSmbusRegs>,
    set_irq: Mutex<Option<PmSmbusIrqFn>>,
}

impl fmt::Debug for PmSmbus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PmSmbus").field("regs", &*self.lock()).finish_non_exhaustive()
    }
}

impl PmSmbus {
    /// `pm_smbus_init()`: a controller with its own I2C bus named `"i2c"`. `force_aux_blk` turns
    /// the 32 byte buffer on for good, which the VIA bridges want.
    pub fn new(force_aux_blk: bool) -> Arc<PmSmbus> {
        let regs = PmSmbusRegs {
            op_done: true,
            smb_auxctl: if force_aux_blk { AUX_BLK } else { 0 },
            ..PmSmbusRegs::default()
        };
        Arc::new(PmSmbus {
            bus: I2cBus::new("i2c"),
            regs: Mutex::new(regs),
            set_irq: Mutex::default(),
        })
    }

    fn lock(&self) -> MutexGuard<'_, PmSmbusRegs> {
        self.regs.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The I2C bus the controller masters.
    pub fn bus(&self) -> &Arc<I2cBus> {
        &self.bus
    }

    /// Sets the interrupt hook, `PMSMBus::set_irq`. The host bridge model decides whether that
    /// is an INTx pin or an SMI.
    pub fn set_irq_handler(&self, f: Option<PmSmbusIrqFn>) {
        *self.set_irq.lock().unwrap_or_else(|e| e.into_inner()) = f;
    }

    /// `PMSMBus::i2c_enable`.
    pub fn set_i2c_enable(&self, on: bool) {
        self.lock().i2c_enable = on;
    }

    /// A copy of the registers.
    pub fn regs(&self) -> PmSmbusRegs {
        self.lock().clone()
    }

    /// `pm_smbus_reset()`: ends any block transfer and clears the status. The other registers
    /// keep their values.
    pub fn reset(&self) {
        let mut s = self.lock();
        s.op_done = true;
        s.smb_index = 0;
        s.smb_stat = 0;
    }

    /// The MMIO callbacks for the register region.
    pub fn io_ops(self: &Arc<Self>) -> Arc<dyn MmioOps> {
        Arc::new(PmSmbusOps(Arc::clone(self)))
    }

    fn irq_value(s: &PmSmbusRegs) -> bool {
        s.smb_stat & !STS_HOST_BUSY != 0 && s.smb_ctl & CTL_INTREN != 0
    }

    fn update_irq(&self, level: bool) {
        let f = self.set_irq.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some(f) = f {
            f(level);
        }
    }

    /// `smb_transaction()`.
    fn transaction(&self, s: &mut PmSmbusRegs) {
        let prot = (s.smb_ctl >> 2) & 0x07;
        let read = s.smb_addr & 0x01 != 0;
        let cmd = s.smb_cmd;
        let addr = s.smb_addr >> 1;
        let bus = &*self.bus;

        // Nothing runs while DEV_ERR is still set.
        if s.smb_stat & STS_DEV_ERR != 0 {
            s.smb_stat |= STS_DEV_ERR;
            return;
        }

        let ok = match prot {
            PROT_QUICK => smbus_quick_command(bus, addr, read).is_ok(),
            PROT_BYTE if read => smbus_receive_byte(bus, addr).map(|v| s.smb_data0 = v).is_ok(),
            PROT_BYTE => smbus_send_byte(bus, addr, cmd).is_ok(),
            PROT_BYTE_DATA if read => {
                smbus_read_byte(bus, addr, cmd).map(|v| s.smb_data0 = v).is_ok()
            }
            PROT_BYTE_DATA => smbus_write_byte(bus, addr, cmd, s.smb_data0).is_ok(),
            PROT_WORD_DATA if read => smbus_read_word(bus, addr, cmd)
                .map(|v| {
                    s.smb_data1 = (v >> 8) as u8;
                    s.smb_data0 = v as u8;
                })
                .is_ok(),
            PROT_WORD_DATA => {
                let word = u16::from(s.smb_data1) << 8 | u16::from(s.smb_data0);
                smbus_write_word(bus, addr, cmd, word).is_ok()
            }
            PROT_I2C_BLOCK_READ => {
                // The Linux i801 driver may or may not set the read bit here (ICH5 says clear
                // it, Lynx Point with SPD write disable needs it set), so it is ignored.
                // HST_D1 holds the offset to read from.
                if bus.start_send(addr).is_err()
                    || bus.send(s.smb_data1).is_err()
                    || bus.start_recv(addr).is_err()
                {
                    s.smb_stat |= STS_DEV_ERR;
                    return;
                }
                s.in_i2c_block_read = true;
                s.smb_blkdata = bus.recv();
                s.op_done = false;
                s.smb_stat |= STS_HOST_BUSY | STS_BYTE_DONE;
                return;
            }
            PROT_BLOCK_DATA if read => {
                let send = !s.i2c_enable;
                let ret = smbus_read_block(bus, addr, cmd, &mut s.smb_data, send, send);
                let Ok(n) = ret else {
                    s.smb_stat |= STS_DEV_ERR;
                    return;
                };
                s.smb_index = 0;
                s.op_done = false;
                if s.smb_auxctl & AUX_BLK != 0 {
                    s.smb_stat |= STS_INTR;
                } else {
                    s.smb_blkdata = s.smb_data[0];
                    s.smb_stat |= STS_HOST_BUSY | STS_BYTE_DONE;
                }
                s.smb_data0 = n as u8;
                return;
            }
            PROT_BLOCK_DATA => {
                if s.smb_auxctl & AUX_BLK != 0 {
                    // The guest has filled the buffer through HOST_BLOCK_DB already.
                    if s.smb_index != u32::from(s.smb_data0) {
                        s.smb_index = 0;
                        s.smb_stat |= STS_DEV_ERR;
                        return;
                    }
                    s.smb_index = 0;
                    let len = usize::from(s.smb_data0).min(PM_SMBUS_MAX_MSG_SIZE);
                    let data = s.smb_data;
                    if smbus_write_block(bus, addr, cmd, &data[..len], !s.i2c_enable).is_err() {
                        s.smb_stat |= STS_DEV_ERR;
                        return;
                    }
                    s.op_done = true;
                    s.smb_stat |= STS_INTR;
                    s.smb_stat &= !STS_HOST_BUSY;
                } else {
                    s.op_done = false;
                    s.smb_stat |= STS_HOST_BUSY | STS_BYTE_DONE;
                    s.smb_data[0] = s.smb_blkdata;
                    s.smb_index = 0;
                }
                return;
            }
            _ => false,
        };
        s.smb_stat |= if ok { STS_INTR } else { STS_DEV_ERR };
    }

    /// `smb_transaction_start()`.
    fn transaction_start(&self, s: &mut PmSmbusRegs) {
        if s.smb_ctl & CTL_INTREN != 0 {
            self.transaction(s);
            s.start_transaction_on_status_read = false;
        } else {
            s.smb_stat |= STS_HOST_BUSY;
            s.start_transaction_on_status_read = true;
        }
    }

    /// `smb_byte_by_byte()`: a block transfer is going through HOST_BLOCK_DB one byte at a time.
    fn byte_by_byte(s: &PmSmbusRegs) -> bool {
        if s.op_done {
            return false;
        }
        if s.in_i2c_block_read {
            return true;
        }
        s.smb_auxctl & AUX_BLK == 0
    }

    /// The guest cleared BYTE_DONE in the middle of a byte by byte block transfer.
    fn next_block_byte(&self, s: &mut PmSmbusRegs) {
        // See the note on the read bit in PROT_I2C_BLOCK_READ.
        let read = s.smb_addr & 0x01 != 0 || s.in_i2c_block_read;

        s.smb_index += 1;
        if s.smb_index as usize >= PM_SMBUS_MAX_MSG_SIZE {
            s.smb_index = 0;
        }
        let index = s.smb_index as usize;
        if !read && s.smb_index == u32::from(s.smb_data0) {
            let prot = (s.smb_ctl >> 2) & 0x07;
            if prot == PROT_I2C_BLOCK_READ {
                s.smb_stat |= STS_DEV_ERR;
                return;
            }
            let len = usize::from(s.smb_data0).min(PM_SMBUS_MAX_MSG_SIZE);
            let data = s.smb_data;
            let addr = s.smb_addr >> 1;
            if smbus_write_block(&self.bus, addr, s.smb_cmd, &data[..len], !s.i2c_enable).is_err() {
                s.smb_stat |= STS_DEV_ERR;
                return;
            }
            s.op_done = true;
            s.smb_stat |= STS_INTR;
            s.smb_stat &= !STS_HOST_BUSY;
        } else if !read {
            s.smb_data[index] = s.smb_blkdata;
            s.smb_stat |= STS_BYTE_DONE;
        } else if s.smb_ctl & CTL_LAST_BYTE != 0 {
            s.op_done = true;
            if s.in_i2c_block_read {
                s.in_i2c_block_read = false;
                s.smb_blkdata = self.bus.recv();
                self.bus.nack();
                self.bus.end_transfer();
            } else {
                s.smb_blkdata = s.smb_data[index];
            }
            s.smb_index = 0;
            s.smb_stat |= STS_INTR;
            s.smb_stat &= !STS_HOST_BUSY;
        } else {
            s.smb_blkdata = if s.in_i2c_block_read { self.bus.recv() } else { s.smb_data[index] };
            s.smb_stat |= STS_BYTE_DONE;
        }
    }

    /// `smb_ioport_writeb()`.
    pub fn write_reg(&self, addr: u64, val: u8) {
        let level = {
            let mut s = self.lock();
            match addr {
                SMBHSTSTS => {
                    let clear_byte_done = s.smb_stat & val & STS_BYTE_DONE != 0;
                    s.smb_stat &= !(val & !STS_HOST_BUSY);
                    if clear_byte_done && Self::byte_by_byte(&s) {
                        self.next_block_byte(&mut s);
                    }
                }
                SMBHSTCNT => {
                    // START always reads back as 0.
                    s.smb_ctl = val & !CTL_START;
                    if val & CTL_START != 0 {
                        if !s.op_done {
                            s.smb_index = 0;
                            s.op_done = true;
                            if s.in_i2c_block_read {
                                s.in_i2c_block_read = false;
                                self.bus.end_transfer();
                            }
                        }
                        self.transaction_start(&mut s);
                    }
                    if s.smb_ctl & CTL_KILL != 0 {
                        s.op_done = true;
                        s.smb_index = 0;
                        s.smb_stat |= STS_FAILED;
                        s.smb_stat &= !STS_HOST_BUSY;
                    }
                }
                SMBHSTCMD => s.smb_cmd = val,
                SMBHSTADD => s.smb_addr = val,
                SMBHSTDAT0 => s.smb_data0 = val,
                SMBHSTDAT1 => s.smb_data1 = val,
                SMBBLKDAT => {
                    if s.smb_index as usize >= PM_SMBUS_MAX_MSG_SIZE {
                        s.smb_index = 0;
                    }
                    if s.smb_auxctl & AUX_BLK != 0 {
                        let i = s.smb_index as usize;
                        s.smb_data[i] = val;
                        s.smb_index += 1;
                    } else {
                        s.smb_blkdata = val;
                    }
                }
                SMBAUXCTL => s.smb_auxctl = val & AUX_MASK,
                _ => {}
            }
            Self::irq_value(&s)
        };
        self.update_irq(level);
    }

    /// `smb_ioport_readb()`.
    pub fn read_reg(&self, addr: u64) -> u8 {
        let (val, level) = {
            let mut s = self.lock();
            let val = match addr {
                SMBHSTSTS => {
                    // The value from before a deferred transaction runs.
                    let val = s.smb_stat;
                    if s.start_transaction_on_status_read {
                        s.start_transaction_on_status_read = false;
                        s.smb_stat &= !STS_HOST_BUSY;
                        self.transaction(&mut s);
                    }
                    val
                }
                SMBHSTCNT => s.smb_ctl & CTL_RETURN_MASK,
                SMBHSTCMD => s.smb_cmd,
                SMBHSTADD => s.smb_addr,
                SMBHSTDAT0 => s.smb_data0,
                SMBHSTDAT1 => s.smb_data1,
                SMBBLKDAT => {
                    if s.smb_auxctl & AUX_BLK != 0 && !s.in_i2c_block_read {
                        if s.smb_index as usize >= PM_SMBUS_MAX_MSG_SIZE {
                            s.smb_index = 0;
                        }
                        let val = s.smb_data[s.smb_index as usize];
                        s.smb_index += 1;
                        if !s.op_done && s.smb_index == u32::from(s.smb_data0) {
                            s.op_done = true;
                            s.smb_index = 0;
                            s.smb_stat &= !STS_HOST_BUSY;
                        }
                        val
                    } else {
                        s.smb_blkdata
                    }
                }
                SMBAUXCTL => s.smb_auxctl,
                _ => 0,
            };
            (val, Self::irq_value(&s))
        };
        self.update_irq(level);
        val
    }
}

/// `pm_smbus_ops`: byte accesses only.
#[derive(Debug)]
pub struct PmSmbusOps(pub Arc<PmSmbus>);

impl MmioOps for PmSmbusOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.0.read_reg(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.write_reg(offset, value as u8);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::exact(1)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::exact(1)
    }

    fn endianness(&self) -> Endian {
        Endian::Little
    }
}
