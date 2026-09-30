// SPDX-License-Identifier: GPL-2.0-or-later

//! Runs tiny real mode guests. Each test is skipped, with a note on stderr, when the host has no
//! usable `/dev/kvm`, which is the case on macOS and on CI runners without nested virtualization.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use ruvm_accel_kvm::{KvmAccel, KvmError, KvmOptions, VcpuKick, VcpuStop, spawn_vcpu_thread};
use ruvm_mem::{AccessCtx, AccessSize, AddressSpace, MemResult, MemorySystem, MmioOps};

#[derive(Debug, Default)]
struct Recorder(Mutex<Vec<(u64, u64)>>);

impl MmioOps for Recorder {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(0x40 + offset)
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.lock().unwrap().push((offset, value));
        Ok(())
    }
}

struct Guest {
    accel: KvmAccel,
    _ms: MemorySystem,
    mem: Arc<AddressSpace>,
    io: Arc<AddressSpace>,
    port: Arc<Recorder>,
    mmio: Arc<Recorder>,
}

/// The guests here end in `hlt`. With the local APIC in the kernel KVM handles that itself and
/// waits for an interrupt that never comes, so these tests keep the irqchip in userspace, where
/// `hlt` exits to us.
fn open() -> Option<KvmAccel> {
    let opts = KvmOptions {
        kernel_irqchip: Some(ruvm_accel_kvm::KernelIrqchip::Off),
        ..KvmOptions::default()
    };
    match KvmAccel::new(&opts, false) {
        Ok(a) => Some(a),
        Err(e @ (KvmError::Open(_) | KvmError::Unavailable)) => {
            eprintln!("skipping: {e}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

/// 32 KiB of RAM at 0 holding `code` at 0x1000 and `data` at 0x1100, a recorder on port 0x3f8
/// and another on MMIO at 0xd000.
fn guest(code: &[u8], data: &[u8]) -> Option<Guest> {
    let accel = open()?;
    let ms = MemorySystem::new();
    let root = ms.new_container("system", 1 << 32).unwrap();
    let ram = ms.new_ram("pc.ram", 0x8000).unwrap();
    ms.add_subregion(root, 0, ram).unwrap();
    let mmio = Arc::new(Recorder::default());
    let dev = ms.new_io("dev", 0x1000, mmio.clone()).unwrap();
    ms.add_subregion(root, 0xd000, dev).unwrap();
    let ioroot = ms.new_container("io", 0x10000).unwrap();
    let port = Arc::new(Recorder::default());
    let serial = ms.new_io("serial", 8, port.clone()).unwrap();
    ms.add_subregion(ioroot, 0x3f8, serial).unwrap();
    let mem = ms.address_space_init(root, "memory").unwrap();
    let io = ms.address_space_init(ioroot, "I/O").unwrap();

    let block = ms.ram_block(ram).unwrap();
    block.write(0x1000, code).unwrap();
    block.write(0x1100, data).unwrap();
    let listener = accel.slot_listener();
    ms.register_listener(listener.clone(), &mem).unwrap();
    assert_eq!(listener.slots(), [(0, 0x8000, false)]);
    Some(Guest { accel, _ms: ms, mem, io, port, mmio })
}

fn reset_to(vcpu: &ruvm_accel_kvm::KvmVcpu, ip: u64) {
    let mut sregs = vcpu.fd().get_sregs().unwrap();
    for seg in [&mut sregs.cs, &mut sregs.ds, &mut sregs.es, &mut sregs.ss] {
        seg.base = 0;
        seg.selector = 0;
    }
    vcpu.fd().set_sregs(&sregs).unwrap();
    let mut regs = vcpu.fd().get_regs().unwrap();
    regs.rip = ip;
    regs.rflags = 2;
    regs.rsp = 0x7000;
    vcpu.fd().set_regs(&regs).unwrap();
}

#[test]
fn port_string_io_and_mmio_reach_the_address_spaces() {
    #[rustfmt::skip]
    let code = [
        0xba, 0xf8, 0x03,             // mov dx, 0x3f8
        0xb0, 0x41,                   // mov al, 'A'
        0xee,                         // out dx, al
        0xbe, 0x00, 0x11,             // mov si, 0x1100
        0xb9, 0x03, 0x00,             // mov cx, 3
        0xfc,                         // cld
        0xf3, 0x6e,                   // rep outsb
        0xc6, 0x06, 0x10, 0xd0, 0x5a, // mov byte [0xd010], 0x5a
        0xa0, 0x20, 0xd0,             // mov al, [0xd020]
        0xa2, 0x00, 0x12,             // mov [0x1200], al
        0xec,                         // in al, dx
        0xa2, 0x01, 0x12,             // mov [0x1201], al
        0xf4,                         // hlt
    ];
    let Some(g) = guest(&code, b"xyz") else { return };
    let mut vcpu = g.accel.create_vcpu(0).unwrap();
    reset_to(&vcpu, 0x1000);
    assert_eq!(vcpu.run(&g.io, &g.mem).unwrap(), VcpuStop::Halted);

    let bytes: Vec<u8> = g.port.0.lock().unwrap().iter().map(|&(_, v)| v as u8).collect();
    assert_eq!(bytes, b"Axyz");
    assert_eq!(*g.mmio.0.lock().unwrap(), [(0x10, 0x5a)]);
    let mut back = [0u8; 2];
    g.mem.read(0x1200, ruvm_mem::MemTxAttrs::UNSPECIFIED, &mut back);
    assert_eq!(back, [0x60, 0x40]);
}

#[test]
fn a_kick_stops_a_spinning_guest() {
    let code = [0xeb, 0xfe]; // jmp $
    let Some(g) = guest(&code, &[]) else { return };
    let mut vcpu = g.accel.create_vcpu(0).unwrap();
    reset_to(&vcpu, 0x1000);
    let flag = vcpu.exit_request();
    let (io, mem) = (g.io.clone(), g.mem.clone());
    let thread = spawn_vcpu_thread(0, move || {
        assert_eq!(std::thread::current().name(), Some("CPU 0/KVM"));
        vcpu.run(&io, &mem).unwrap()
    })
    .unwrap();
    let kick = VcpuKick::new(&thread, flag);
    std::thread::sleep(Duration::from_millis(50));
    kick.kick().unwrap();
    assert_eq!(thread.join().unwrap(), VcpuStop::Kicked);
}

#[test]
fn split_irqchip_and_bad_device() {
    if open().is_none() {
        return;
    }
    let opts = KvmOptions {
        kernel_irqchip: Some(ruvm_accel_kvm::KernelIrqchip::Split),
        ..KvmOptions::default()
    };
    let accel = KvmAccel::new(&opts, false).unwrap();
    assert_eq!(accel.kernel_irqchip(), ruvm_accel_kvm::KernelIrqchip::Split);

    let opts = KvmOptions { device: Some("/nonexistent/kvm".into()), ..KvmOptions::default() };
    let err = KvmAccel::new(&opts, false).unwrap_err();
    assert_eq!(err.to_string(), "Could not access KVM kernel module: No such file or directory");
}
