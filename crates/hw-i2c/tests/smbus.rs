// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests of the I2C core, the SMBus slave and master layers, the PM SMBus host controller, the
//! ICH9 SMBus function and the SMBus EEPROM. The controller tests drive the registers through
//! port I/O the way the Linux i2c-i801 driver does.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_hw_i2c::pm_smbus::*;
use ruvm_hw_i2c::smbus::*;
use ruvm_hw_i2c::smbus_eeprom::SMBUS_EEPROM_SIZE;
use ruvm_hw_i2c::smbus_ich9::*;
use ruvm_hw_i2c::*;
use ruvm_hw_pci::PciBus;
use ruvm_hw_pci::regs::*;
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemorySystem};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const SMB_IO: u64 = 0xb100;

/// An SMBus slave that logs what it sees and answers reads from a queue.
#[derive(Debug, Default)]
struct TestSlave {
    log: Mutex<Vec<String>>,
    replies: Mutex<VecDeque<u8>>,
}

impl TestSlave {
    fn reply(&self, bytes: &[u8]) {
        self.replies.lock().unwrap().extend(bytes);
    }

    fn take_log(&self) -> Vec<String> {
        std::mem::take(&mut *self.log.lock().unwrap())
    }
}

impl SmbusDevice for TestSlave {
    fn quick_cmd(&self, read: bool) {
        self.log.lock().unwrap().push(format!("quick {}", u8::from(read)));
    }

    fn write_data(&self, buf: &[u8]) {
        self.log.lock().unwrap().push(format!("write {buf:02x?}"));
    }

    fn receive_byte(&self) -> u8 {
        let b = self.replies.lock().unwrap().pop_front().unwrap_or(0xee);
        self.log.lock().unwrap().push(format!("recv {b:02x}"));
        b
    }
}

struct Env {
    io_as: Arc<AddressSpace>,
    bus: Arc<PciBus>,
    smb: Arc<Ich9Smbus>,
    eeproms: Vec<Arc<SmbusSlave<SmbusEeprom>>>,
    slave: Arc<SmbusSlave<TestSlave>>,
    levels: Arc<[AtomicI32; 4]>,
}

impl Env {
    /// A Q35 style SMBus function with its 8 EEPROMs and a test slave at 0x20, BAR 4 mapped at
    /// `SMB_IO` and HOSTC.HST_EN set, the way firmware leaves it.
    fn new() -> Env {
        let env = Env::bare();
        let dev = env.smb.device();
        dev.config_write(PCI_BASE_ADDRESS_4 as u32, SMB_IO as u32 | 1, 4);
        dev.config_write(PCI_COMMAND as u32, u32::from(PCI_COMMAND_IO), 2);
        dev.config_write(ICH9_SMB_HOSTC as u32, u32::from(ICH9_SMB_HOSTC_HST_EN), 1);
        env
    }

    fn bare() -> Env {
        let mem = Arc::new(MemorySystem::new());
        let sysmem = mem.new_container("system", 1 << 64).unwrap();
        let io = mem.new_container("io", 1 << 16).unwrap();
        let io_as = mem.address_space_init(io, "I/O").unwrap();
        let bus = PciBus::new_root("pcie.0", Arc::clone(&mem), sysmem, io, 0);
        let levels: Arc<[AtomicI32; 4]> = Arc::new(Default::default());
        let l = Arc::clone(&levels);
        bus.set_irqs(Arc::new(move |irq, level| l[irq as usize].store(level, Ordering::SeqCst)), 4);
        bus.set_map_irq(Arc::new(|_devfn, pin| pin));

        let (smb, eeproms) = ich9_smbus_q35_init(&bus).unwrap();
        let slave = SmbusSlave::new(Arc::new(TestSlave::default()));
        smb.smbus().attach(0x20, Arc::clone(&slave) as Arc<dyn I2cSlave>);
        Env { io_as, bus, smb, eeproms, slave, levels }
    }

    fn inb(&self, reg: u64) -> u8 {
        self.io_as.load(SMB_IO + reg, 1, Endian::Little, ATTRS).0 as u8
    }

    fn outb(&self, reg: u64, v: u8) {
        let _ = self.io_as.store(SMB_IO + reg, 1, v.into(), Endian::Little, ATTRS);
    }

    fn test(&self) -> &TestSlave {
        self.slave.device()
    }

    /// Starts a transaction with `prot` and waits for it the way i801 does: read HST_STS until
    /// HOST_BUSY clears.
    fn run(&self, addr: u8, read: bool, cmd: u8, prot: u8, ctl: u8) -> u8 {
        self.outb(SMBHSTSTS, 0xff);
        self.outb(SMBHSTADD, addr << 1 | u8::from(read));
        self.outb(SMBHSTCMD, cmd);
        self.outb(SMBHSTCNT, prot << 2 | CTL_START | ctl);
        let mut sts = self.inb(SMBHSTSTS);
        for _ in 0..4 {
            if sts & STS_HOST_BUSY == 0 {
                break;
            }
            sts = self.inb(SMBHSTSTS);
        }
        sts
    }
}

#[test]
fn pci_identity_and_bar() {
    let env = Env::bare();
    let dev = env.smb.device();
    assert_eq!(dev.devfn(), ICH9_SMB_DEVFN);
    assert_eq!(dev.devfn(), 0xfb);
    assert_eq!(dev.config_read(PCI_VENDOR_ID as u32, 2), 0x8086);
    assert_eq!(dev.config_read(PCI_DEVICE_ID as u32, 2), 0x2930);
    assert_eq!(dev.config_read(PCI_REVISION_ID as u32, 1), 0x02);
    assert_eq!(dev.config_read(PCI_CLASS_DEVICE as u32, 2), 0x0c05);
    assert_eq!(dev.config_read(PCI_INTERRUPT_PIN as u32, 1), 1);
    assert_eq!(dev.config_read(ICH9_SMB_HOSTC as u32, 1), 0);
    assert!(dev.is_multifunction());
    // BAR 4 is I/O. QEMU backs it with the 64 byte pm-smbus region, so that is its size.
    assert_eq!(dev.config_read(ICH9_SMB_SMB_BASE as u32, 4), 1);
    dev.config_write(ICH9_SMB_SMB_BASE as u32, 0xffff_ffff, 4);
    assert_eq!(dev.config_read(ICH9_SMB_SMB_BASE as u32, 4), 0xffff_ffc1);
    let bar = dev.bar_info(ICH9_SMB_SMB_BASE_BAR).unwrap();
    assert_eq!(bar.size, 64);
    assert_eq!(env.smb.smbus().addresses().len(), 9);
}

#[test]
fn hostc_gates_the_io_region() {
    let env = Env::bare();
    let dev = env.smb.device();
    dev.config_write(PCI_BASE_ADDRESS_4 as u32, SMB_IO as u32 | 1, 4);
    dev.config_write(PCI_COMMAND as u32, u32::from(PCI_COMMAND_IO), 2);
    // Like QEMU, the region starts enabled even though HOSTC reads 0.
    env.outb(SMBHSTCMD, 0x5a);
    assert_eq!(env.inb(SMBHSTCMD), 0x5a);

    dev.config_write(ICH9_SMB_HOSTC as u32, 0, 1);
    env.outb(SMBHSTCMD, 0x11);
    assert_ne!(env.inb(SMBHSTCMD), 0x11);
    assert_eq!(env.smb.pm().regs().smb_cmd, 0x5a);

    dev.config_write(ICH9_SMB_HOSTC as u32, u32::from(ICH9_SMB_HOSTC_HST_EN), 1);
    assert_eq!(env.inb(SMBHSTCMD), 0x5a);

    // A 4 byte write that covers HOSTC counts too.
    dev.config_write(ICH9_SMB_HOSTC as u32, u32::from(ICH9_SMB_HOSTC_I2C_EN), 4);
    assert!(env.smb.pm().regs().i2c_enable);
    assert_ne!(env.inb(SMBHSTCMD), 0x5a);
}

#[test]
fn hostc_soft_reset_self_clears() {
    let env = Env::new();
    assert_eq!(env.run(0x7f, false, 0, PROT_QUICK, 0) & STS_DEV_ERR, STS_DEV_ERR);
    let dev = env.smb.device();
    let v = ICH9_SMB_HOSTC_HST_EN | ICH9_SMB_HOSTC_SSRESET;
    dev.config_write(ICH9_SMB_HOSTC as u32, u32::from(v), 1);
    assert_eq!(dev.config_read(ICH9_SMB_HOSTC as u32, 1), u32::from(ICH9_SMB_HOSTC_HST_EN));
    assert_eq!(env.inb(SMBHSTSTS), 0);
}

#[test]
fn quick_command() {
    let env = Env::new();
    assert_eq!(env.run(0x20, false, 0, PROT_QUICK, CTL_INTREN), STS_INTR);
    assert_eq!(env.run(0x20, true, 0, PROT_QUICK, CTL_INTREN), STS_INTR);
    assert_eq!(env.test().take_log(), ["quick 0", "quick 1"]);
    // Every EEPROM answers.
    for a in 0x50..0x58 {
        assert_eq!(env.run(a, false, 0, PROT_QUICK, 0), STS_INTR);
    }
}

#[test]
fn missing_device_sets_dev_err_and_blocks_the_next_transaction() {
    let env = Env::new();
    env.outb(SMBHSTADD, 0x10 << 1);
    env.outb(SMBHSTCNT, PROT_QUICK << 2 | CTL_START);
    assert_eq!(env.inb(SMBHSTSTS), STS_HOST_BUSY);
    assert_eq!(env.inb(SMBHSTSTS), STS_DEV_ERR);

    // DEV_ERR still set: nothing reaches the bus.
    env.outb(SMBHSTADD, 0x20 << 1);
    env.outb(SMBHSTCNT, PROT_QUICK << 2 | CTL_START | CTL_INTREN);
    assert_eq!(env.inb(SMBHSTSTS), STS_DEV_ERR);
    assert!(env.test().take_log().is_empty());

    env.outb(SMBHSTSTS, STS_DEV_ERR);
    assert_eq!(env.inb(SMBHSTSTS), 0);
}

#[test]
fn deferred_start_until_status_read() {
    let env = Env::new();
    env.outb(SMBHSTADD, 0x20 << 1);
    env.outb(SMBHSTCNT, PROT_QUICK << 2 | CTL_START);
    // START reads back as 0 and nothing has run yet.
    assert_eq!(env.inb(SMBHSTCNT), PROT_QUICK << 2);
    assert!(env.test().take_log().is_empty());
    // The first read sees HOST_BUSY and runs the transaction.
    assert_eq!(env.inb(SMBHSTSTS), STS_HOST_BUSY);
    assert_eq!(env.test().take_log(), ["quick 0"]);
    assert_eq!(env.inb(SMBHSTSTS), STS_INTR);
}

#[test]
fn byte_and_word_transactions_with_test_slave() {
    let env = Env::new();
    assert_eq!(env.run(0x20, false, 0x42, PROT_BYTE, 0), STS_INTR);
    assert_eq!(env.test().take_log(), ["write [42]"]);

    env.test().reply(&[0x99]);
    assert_eq!(env.run(0x20, true, 0, PROT_BYTE, 0), STS_INTR);
    assert_eq!(env.inb(SMBHSTDAT0), 0x99);
    assert_eq!(env.test().take_log(), ["recv 99"]);

    env.outb(SMBHSTDAT0, 0x12);
    assert_eq!(env.run(0x20, false, 0x07, PROT_BYTE_DATA, 0), STS_INTR);
    assert_eq!(env.test().take_log(), ["write [07, 12]"]);

    env.test().reply(&[0x34]);
    assert_eq!(env.run(0x20, true, 0x08, PROT_BYTE_DATA, 0), STS_INTR);
    assert_eq!(env.inb(SMBHSTDAT0), 0x34);
    assert_eq!(env.test().take_log(), ["write [08]", "recv 34"]);

    env.outb(SMBHSTDAT0, 0xcd);
    env.outb(SMBHSTDAT1, 0xab);
    assert_eq!(env.run(0x20, false, 0x09, PROT_WORD_DATA, 0), STS_INTR);
    assert_eq!(env.test().take_log(), ["write [09, cd, ab]"]);

    env.test().reply(&[0x78, 0x56]);
    assert_eq!(env.run(0x20, true, 0x0a, PROT_WORD_DATA, 0), STS_INTR);
    assert_eq!((env.inb(SMBHSTDAT1), env.inb(SMBHSTDAT0)), (0x56, 0x78));
    assert_eq!(env.test().take_log(), ["write [0a]", "recv 78", "recv 56"]);

    // Process call is not implemented: DEV_ERR.
    assert_eq!(env.run(0x20, false, 0, PROT_PROC_CALL, 0), STS_DEV_ERR);
}

#[test]
fn eeprom_byte_and_word_reads_and_writes() {
    let env = Env::new();
    env.outb(SMBHSTDAT0, 0xa5);
    assert_eq!(env.run(0x50, false, 0x10, PROT_BYTE_DATA, 0), STS_INTR);
    env.outb(SMBHSTDAT0, 0x01);
    env.outb(SMBHSTDAT1, 0x02);
    assert_eq!(env.run(0x50, false, 0x11, PROT_WORD_DATA, 0), STS_INTR);
    assert_eq!(&env.eeproms[0].device().data()[0x10..0x13], [0xa5, 0x01, 0x02]);
    assert!(env.eeproms[0].device().accessed());
    assert!(!env.eeproms[1].device().accessed());

    assert_eq!(env.run(0x50, true, 0x10, PROT_BYTE_DATA, 0), STS_INTR);
    assert_eq!(env.inb(SMBHSTDAT0), 0xa5);
    assert_eq!(env.run(0x50, true, 0x11, PROT_WORD_DATA, 0), STS_INTR);
    assert_eq!((env.inb(SMBHSTDAT0), env.inb(SMBHSTDAT1)), (0x01, 0x02));
    // Receive byte continues from the pointer.
    assert_eq!(env.run(0x50, true, 0, PROT_BYTE, 0), STS_INTR);
    assert_eq!(env.inb(SMBHSTDAT0), 0x00);
    assert_eq!(env.eeproms[0].device().offset(), 0x14);

    // The eeproms are blank on Q35 and reset brings them back to that.
    env.smb.smbus().reset();
    assert_eq!(env.eeproms[0].device().data(), [0; SMBUS_EEPROM_SIZE]);
    assert_eq!(env.eeproms[0].device().offset(), 0);
}

#[test]
fn i2c_block_read_byte_by_byte_from_eeprom() {
    let env = Env::new();
    let data: Vec<u8> = (0..8).map(|i| 0x30 + i).collect();
    smbus_write_block(env.smb.smbus(), 0x51, 0x40, &data, false).unwrap();

    // HST_D1 holds the offset. The read bit is ignored.
    env.outb(SMBHSTDAT1, 0x40);
    let sts = env.run(0x51, false, 0, PROT_I2C_BLOCK_READ, 0);
    assert_eq!(sts, STS_HOST_BUSY | STS_BYTE_DONE);
    let mut got = vec![env.inb(SMBBLKDAT)];
    for i in 1..8 {
        if i == 7 {
            env.outb(SMBHSTCNT, PROT_I2C_BLOCK_READ << 2 | CTL_LAST_BYTE);
        }
        env.outb(SMBHSTSTS, STS_BYTE_DONE);
        let sts = env.inb(SMBHSTSTS);
        if i == 7 {
            assert_eq!(sts, STS_INTR);
        } else {
            assert_eq!(sts, STS_HOST_BUSY | STS_BYTE_DONE);
        }
        got.push(env.inb(SMBBLKDAT));
    }
    assert_eq!(got, data);
    assert!(!env.smb.smbus().busy());
    assert_eq!(env.eeproms[1].mode(), SmbusMode::Idle);
}

#[test]
fn block_write_byte_by_byte() {
    let env = Env::new();
    env.outb(SMBHSTDAT0, 3);
    env.outb(SMBBLKDAT, 0xa1);
    let sts = env.run(0x20, false, 0x05, PROT_BLOCK_DATA, 0);
    assert_eq!(sts, STS_HOST_BUSY | STS_BYTE_DONE);
    env.outb(SMBBLKDAT, 0xa2);
    env.outb(SMBHSTSTS, STS_BYTE_DONE);
    assert_eq!(env.inb(SMBHSTSTS), STS_HOST_BUSY | STS_BYTE_DONE);
    env.outb(SMBBLKDAT, 0xa3);
    env.outb(SMBHSTSTS, STS_BYTE_DONE);
    assert_eq!(env.inb(SMBHSTSTS), STS_HOST_BUSY | STS_BYTE_DONE);
    assert!(env.test().take_log().is_empty());
    // Clearing BYTE_DONE after the last byte sends the block, with its length since I2C_EN is
    // clear.
    env.outb(SMBHSTSTS, STS_BYTE_DONE);
    assert_eq!(env.inb(SMBHSTSTS), STS_INTR);
    assert_eq!(env.test().take_log(), ["write [05, 03, a1, a2, a3]"]);
}

#[test]
fn block_write_and_read_with_32_byte_buffer() {
    let env = Env::new();
    env.outb(SMBAUXCTL, 0xff);
    assert_eq!(env.inb(SMBAUXCTL), AUX_MASK);

    env.outb(SMBHSTDAT0, 4);
    for b in [1, 2, 3, 4] {
        env.outb(SMBBLKDAT, b);
    }
    assert_eq!(env.run(0x20, false, 0x33, PROT_BLOCK_DATA, 0), STS_INTR);
    assert_eq!(env.test().take_log(), ["write [33, 04, 01, 02, 03, 04]"]);

    // With I2C_EN set the length byte is left out.
    env.smb.device().config_write(
        ICH9_SMB_HOSTC as u32,
        u32::from(ICH9_SMB_HOSTC_HST_EN | ICH9_SMB_HOSTC_I2C_EN),
        1,
    );
    env.outb(SMBHSTDAT0, 2);
    env.outb(SMBBLKDAT, 9);
    env.outb(SMBBLKDAT, 8);
    assert_eq!(env.run(0x20, false, 0x34, PROT_BLOCK_DATA, 0), STS_INTR);
    assert_eq!(env.test().take_log(), ["write [34, 09, 08]"]);
    env.smb.device().config_write(ICH9_SMB_HOSTC as u32, u32::from(ICH9_SMB_HOSTC_HST_EN), 1);

    // A count that does not match what was queued fails.
    env.outb(SMBHSTDAT0, 3);
    env.outb(SMBBLKDAT, 1);
    assert_eq!(env.run(0x20, false, 0x35, PROT_BLOCK_DATA, 0), STS_DEV_ERR);
    env.outb(SMBHSTSTS, 0xff);

    // Block read: the slave sends the length first.
    env.test().take_log();
    env.test().reply(&[3, 0xd1, 0xd2, 0xd3]);
    assert_eq!(env.run(0x20, true, 0x36, PROT_BLOCK_DATA, 0), STS_INTR);
    assert_eq!(env.inb(SMBHSTDAT0), 3);
    let got: Vec<u8> = (0..3).map(|_| env.inb(SMBBLKDAT)).collect();
    assert_eq!(got, [0xd1, 0xd2, 0xd3]);
    assert!(env.smb.pm().regs().op_done);
}

#[test]
fn block_read_byte_by_byte() {
    let env = Env::new();
    env.test().reply(&[3, 0xc1, 0xc2, 0xc3]);
    let sts = env.run(0x20, true, 0x36, PROT_BLOCK_DATA, 0);
    assert_eq!(sts, STS_HOST_BUSY | STS_BYTE_DONE);
    assert_eq!(env.inb(SMBHSTDAT0), 3);
    let mut got = vec![env.inb(SMBBLKDAT)];
    env.outb(SMBHSTSTS, STS_BYTE_DONE);
    got.push(env.inb(SMBBLKDAT));
    env.outb(SMBHSTCNT, PROT_BLOCK_DATA << 2 | CTL_LAST_BYTE);
    env.outb(SMBHSTSTS, STS_BYTE_DONE);
    assert_eq!(env.inb(SMBHSTSTS), STS_INTR);
    got.push(env.inb(SMBBLKDAT));
    assert_eq!(got, [0xc1, 0xc2, 0xc3]);
}

#[test]
fn kill_and_inuse() {
    let env = Env::new();
    env.test().reply(&[3, 1, 2, 3]);
    env.run(0x20, true, 0, PROT_BLOCK_DATA, 0);
    assert!(!env.smb.pm().regs().op_done);
    env.outb(SMBHSTCNT, CTL_KILL);
    assert_eq!(env.inb(SMBHSTSTS), STS_BYTE_DONE | STS_FAILED);
    assert!(env.smb.pm().regs().op_done);
    env.outb(SMBHSTSTS, 0xff);
    env.outb(SMBHSTCNT, 0);

    // INUSE_STS has no semaphore behaviour in QEMU: it is never set by a read and a write of 1
    // just clears it.
    assert_eq!(env.inb(SMBHSTSTS) & STS_INUSE_STS, 0);
    env.outb(SMBHSTSTS, STS_INUSE_STS);
    assert_eq!(env.inb(SMBHSTSTS), 0);
    // HOST_BUSY cannot be cleared by the guest.
    env.outb(SMBHSTADD, 0x20 << 1);
    env.outb(SMBHSTCNT, CTL_START);
    env.outb(SMBHSTSTS, 0xff);
    assert_eq!(env.inb(SMBHSTSTS), STS_HOST_BUSY);
}

#[test]
fn intx_follows_intren() {
    let env = Env::new();
    let inta = || env.levels[0].load(Ordering::SeqCst);
    assert_eq!(env.run(0x20, false, 0, PROT_QUICK, 0), STS_INTR);
    assert_eq!(inta(), 0);
    assert!(!env.smb.irq_enabled());

    assert_eq!(env.run(0x20, false, 0, PROT_QUICK, CTL_INTREN), STS_INTR);
    assert_eq!(inta(), 1);
    assert!(env.smb.irq_enabled());
    env.outb(SMBHSTSTS, STS_INTR);
    assert_eq!(inta(), 0);

    // Errors interrupt too.
    env.run(0x11, false, 0, PROT_QUICK, CTL_INTREN);
    assert_eq!(inta(), 1);
    // Masking INTREN drops the line.
    env.outb(SMBHSTCNT, 0);
    assert_eq!(inta(), 0);
    let _ = &env.bus;
}

#[test]
fn standalone_controller_irq_hook() {
    // The hook a bridge model uses to turn the controller interrupt into an SMI or a pin.
    let pm = PmSmbus::new(true);
    assert_eq!(pm.read_reg(SMBAUXCTL), AUX_BLK);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s = Arc::clone(&seen);
    pm.set_irq_handler(Some(Arc::new(move |l| s.lock().unwrap().push(l))));
    let slave = SmbusSlave::new(Arc::new(TestSlave::default()));
    pm.bus().attach(0x21, Arc::clone(&slave) as Arc<dyn I2cSlave>);
    pm.write_reg(SMBHSTADD, 0x21 << 1);
    pm.write_reg(SMBHSTCNT, CTL_START | CTL_INTREN);
    assert_eq!(pm.read_reg(SMBHSTSTS), STS_INTR);
    pm.write_reg(SMBHSTSTS, STS_INTR);
    let seen = seen.lock().unwrap().clone();
    // Every access reports the level: ADD write, CNT write, STS read, STS write.
    assert_eq!(seen, [false, true, true, false]);
    assert_eq!(slave.device().take_log(), ["quick 0"]);
}

#[test]
fn i2c_core_addressing_and_broadcast() {
    let bus = I2cBus::new("i2c");
    let a = SmbusSlave::new(Arc::new(TestSlave::default()));
    let b = SmbusSlave::new(Arc::new(TestSlave::default()));
    bus.attach(0x30, Arc::clone(&a) as Arc<dyn I2cSlave>);
    bus.attach(0x31, Arc::clone(&b) as Arc<dyn I2cSlave>);
    assert_eq!(bus.start_send(0x32), Err(I2cNak));
    assert!(!bus.busy());

    // General call: both slaves get the byte, nobody answers a read.
    bus.start_send(I2C_BROADCAST).unwrap();
    bus.send(0x77).unwrap();
    assert_eq!(bus.recv(), 0xff);
    bus.end_transfer();
    assert_eq!(a.device().take_log(), ["write [77]"]);
    assert_eq!(b.device().take_log(), ["write [77]"]);

    // Moving a slave.
    let dyn_b = Arc::clone(&b) as Arc<dyn I2cSlave>;
    bus.set_address(&dyn_b, 0x40);
    assert!(smbus_quick_command(&bus, 0x31, false).is_err());
    smbus_quick_command(&bus, 0x40, true).unwrap();
    assert_eq!(b.device().take_log(), ["quick 1"]);
    bus.detach(&dyn_b);
    assert!(smbus_quick_command(&bus, 0x40, true).is_err());

    // Async sends are not supported by SMBus slaves.
    assert_eq!(bus.start_send_async(0x30), Err(I2cNak));
    assert!(!bus.busy());

    // A read with no command byte first leaves the slave confused until the stop.
    a.device().reply(&[5]);
    assert_eq!(smbus_receive_byte(&bus, 0x30), Ok(5));
    assert_eq!(a.mode(), SmbusMode::Idle);
    bus.start_recv(0x30).unwrap();
    bus.start_send(0x30).unwrap();
    assert_eq!(a.mode(), SmbusMode::Confused);
    bus.end_transfer();
    assert_eq!(a.mode(), SmbusMode::Idle);
}

#[test]
fn smbus_read_block_length_handling() {
    let bus = I2cBus::new("i2c");
    let a = SmbusSlave::new(Arc::new(TestSlave::default()));
    bus.attach(0x30, Arc::clone(&a) as Arc<dyn I2cSlave>);
    let mut buf = [0u8; 4];
    a.device().reply(&[9]);
    // A length that does not fit reads nothing.
    assert_eq!(smbus_read_block(&bus, 0x30, 1, &mut buf, true, true), Ok(0));
    a.device().reply(&[1, 2, 3, 4]);
    assert_eq!(smbus_read_block(&bus, 0x30, 1, &mut buf, false, false), Ok(4));
    assert_eq!(buf, [1, 2, 3, 4]);
    // Writes stop at 32 bytes.
    a.device().take_log();
    smbus_write_block(&bus, 0x30, 7, &[0x11; 40], true).unwrap();
    let log = a.device().take_log();
    assert_eq!(log.len(), 1);
    assert!(log[0].starts_with("write [07, 20, 11"));
    assert_eq!(log[0].matches("11").count(), 32);
}

#[test]
fn eeprom_init_splits_spd_data() {
    let bus = I2cBus::new("i2c");
    let mut spd = vec![0u8; 300];
    spd[0] = 0x80;
    spd[256] = 0x42;
    let e = smbus_eeprom_init(&bus, 3, &spd).unwrap();
    assert_eq!(bus.addresses(), [0x52, 0x51, 0x50]);
    assert_eq!(e[0].device().data()[0], 0x80);
    assert_eq!(e[1].device().data()[0], 0x42);
    assert_eq!(e[2].device().data(), [0; SMBUS_EEPROM_SIZE]);
    assert!(smbus_eeprom_init(&bus, 9, &[]).is_err());

    // The pointer wraps at 256.
    smbus_write_byte(&bus, 0x52, 0xff, 1).unwrap();
    assert_eq!(e[2].device().offset(), 0);
    assert_eq!(smbus_read_byte(&bus, 0x52, 0xff), Ok(1));
    assert_eq!(e[2].device().offset(), 0);
}

#[test]
fn spd_ddr2_contents() {
    let spd = spd_data_generate(SdramType::Ddr2, 1 << 30).unwrap();
    // 1 GiB: 2^10 MiB, split into 2 ranks of 512 MiB, density 512 MiB / 4 = 128.
    assert_eq!(&spd[..7], [128, 8, 8, 13, 10, 1, 64]);
    assert_eq!(spd[15], 0);
    assert_eq!(spd[19], 0);
    assert_eq!(spd[21], 0);
    assert_eq!(spd[31], 0x80);
    assert_eq!(spd[36], 13 << 2);
    let sum = spd[..63].iter().fold(0u8, |s, &b| s.wrapping_add(b));
    assert_eq!(spd[63], sum);
    assert!(spd[64..].iter().all(|&b| b == 0));

    // 32 GiB of DDR2 is 2 ranks of 16 GiB: density wraps into the low bits.
    let big = spd_data_generate(SdramType::Ddr2, 32 << 30).unwrap();
    assert_eq!(big[5], 1);
    assert_eq!(big[31], 0x10);
}

#[test]
fn spd_sdr_and_ddr_and_errors() {
    let sdr = spd_data_generate(SdramType::Sdr, 64 << 20).unwrap();
    assert_eq!((sdr[2], sdr[5], sdr[31]), (4, 2, 8));
    assert_eq!((sdr[15], sdr[19], sdr[21], sdr[36]), (1, 1, 0x20, 0));
    let ddr = spd_data_generate(SdramType::Ddr, 256 << 20).unwrap();
    assert_eq!((ddr[2], ddr[5], ddr[31]), (7, 2, 32));
    // The smallest size does not get split.
    let small = spd_data_generate(SdramType::Ddr, 32 << 20).unwrap();
    assert_eq!((small[5], small[31]), (1, 8));

    assert!(spd_data_generate(SdramType::Ddr2, 64 << 20).is_err());
    assert!(spd_data_generate(SdramType::Sdr, 3 << 20).is_err());
    assert!(spd_data_generate(SdramType::Sdr, (1 << 20) + 1).is_err());
    // 1 TiB of SDR is 8 ranks and a density that truncates to 0, as in QEMU.
    assert_eq!(spd_data_generate(SdramType::Sdr, 1 << 40).unwrap()[31], 0);
    // QEMU keeps the MiB count in 32 bits.
    assert!(spd_data_generate(SdramType::Sdr, 1 << 52).is_err());
}

#[test]
fn ich9_vmstate_round_trip() {
    let src = Env::new();
    src.outb(SMBHSTCMD, 0x5a);
    src.outb(SMBHSTDAT0, 0x12);
    // An interrupt-enabled transaction to a missing device leaves INTR and INTA up.
    assert_ne!(src.run(0x7f, false, 0, PROT_QUICK, CTL_INTREN) & STS_DEV_ERR, 0);
    let saved = src.smb.vmstate_save();
    assert!(saved.irq_enabled);
    assert_eq!(saved.dev.irq_state, [1, 0, 0, 0]);
    assert_eq!(saved.smb.smb_data0, 0x12);
    assert_eq!(saved.smb, src.smb.pm().regs());
    let counts = src.bus.irq_counts();

    let dst = Env::bare();
    dst.smb.vmstate_load(&saved).unwrap();
    dst.bus.set_irq_counts(&counts).unwrap();
    assert_eq!(dst.smb.vmstate_save(), saved);
    // BAR 4, the command register and HOSTC come with config space.
    assert_eq!(dst.inb(SMBHSTDAT0), 0x12);
    // Clearing the status drops INTA through the restored count.
    dst.outb(SMBHSTSTS, 0xff);
    assert!(!dst.smb.irq_enabled());
    assert_eq!(dst.bus.irq_count(0), 0);

    // With HOSTC.HST_EN clear in the stream the registers are hidden.
    let mut off = saved;
    off.dev.config[ICH9_SMB_HOSTC] = 0;
    dst.smb.vmstate_load(&off).unwrap();
    assert_ne!(dst.inb(SMBHSTDAT0), 0x12);
}
