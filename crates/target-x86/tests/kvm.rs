// SPDX-License-Identifier: GPL-2.0-or-later

//! Puts a reset x86 CPU into a real KVM vCPU and runs a little real mode
//! code on it. Each test is skipped, with a note on stderr, when the host
//! has no usable `/dev/kvm`.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::sync::{Arc, Mutex};

use ruvm_accel_kvm::{KernelIrqchip, KvmAccel, KvmError, KvmOptions, VcpuStop};
use ruvm_mem::{AccessCtx, AccessSize, AddressSpace, MemResult, MemorySystem, MmioOps};
use ruvm_target_x86::cpuid::{Accel, X86Cpu};
use ruvm_target_x86::kvm::{X86KvmVcpu, host_cpuid, setup_vcpu};
use ruvm_target_x86::kvm_convert::PutLevel;
use ruvm_target_x86::state::{R_CS, R_DS, R_EAX, R_EBX, R_ECX, R_EDX, R_ES, R_SS};

#[derive(Debug, Default)]
struct Recorder(Mutex<Vec<(u64, u64)>>);

impl MmioOps for Recorder {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(0)
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.lock().unwrap().push((offset, value));
        Ok(())
    }
}

fn open(irqchip: KernelIrqchip) -> Option<KvmAccel> {
    let opts = KvmOptions { kernel_irqchip: Some(irqchip), ..KvmOptions::default() };
    match KvmAccel::new(&opts, false) {
        Ok(a) => Some(a),
        Err(e @ (KvmError::Open(_) | KvmError::Unavailable)) => {
            eprintln!("skipping: {e}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

fn kvm_cpu(accel: &KvmAccel, model: &str) -> X86Cpu {
    let host = host_cpuid(accel).unwrap();
    let mut cpu = X86Cpu::new(model, Accel::Kvm(host)).unwrap();
    cpu.realize().unwrap();
    cpu
}

#[test]
fn reset_state_reaches_the_vcpu() {
    let Some(accel) = open(KernelIrqchip::On) else { return };
    let cpu = kvm_cpu(&accel, "qemu64");
    let state = cpu.new_state(true);
    let vcpu = accel.create_vcpu(0).unwrap();
    let ctx = setup_vcpu(&accel, vcpu.fd(), &cpu, &state).unwrap();

    let sregs = vcpu.fd().get_sregs().unwrap();
    assert_eq!(sregs.cs.base, 0xffff_0000);
    assert_eq!(sregs.cs.selector, 0xf000);
    assert_eq!(sregs.cr0, state.cr0);
    let regs = vcpu.fd().get_regs().unwrap();
    assert_eq!(regs.rip, 0xfff0);
    assert_eq!(regs.rdx, state.regs[R_EDX]);

    let mut back = cpu.new_state(true);
    back.rip = 0;
    back.segs[R_CS].base = 0;
    ctx.get_registers(vcpu.fd(), &mut back).unwrap();
    assert_eq!(back.rip, 0xfff0);
    assert_eq!(back.segs[R_CS], state.segs[R_CS]);
    assert_eq!(back.segs[R_DS], state.segs[R_DS]);
    assert_eq!(back.cr0, state.cr0);
    assert_eq!(back.hflags & 0xffff, state.hflags & 0xffff);
    assert_eq!(back.fpuc, 0x37f);
    assert_eq!(back.mxcsr, state.mxcsr);
    assert_eq!(back.xcr0, state.xcr0);
    assert_eq!(back.pat, state.pat);
    assert_eq!(back.mp_state, state.mp_state);
    assert_eq!(back.dr[7], state.dr[7]);
    assert!(ctx.cpuid().iter().any(|e| e.function == 0x4000_0000));

    // The in-kernel APIC got the reset page: spurious vector 0xff, LINT0
    // masked.
    let lapic = vcpu.fd().get_lapic().unwrap();
    let reg = |r: usize| {
        let b = &lapic.regs[r << 4..(r << 4) + 4];
        u32::from_le_bytes([b[0] as u8, b[1] as u8, b[2] as u8, b[3] as u8])
    };
    assert_eq!(reg(0xf), 0xff);
    assert_eq!(reg(0x35), 1 << 16);
}

/// Real mode code at 0x1000: CPUID leaf 0, store the four registers at
/// 0x1100, write 'K' to port 0x3f8, halt.
#[rustfmt::skip]
const CODE: [u8; 28] = [
    0x66, 0x31, 0xc0,             // xor eax, eax
    0x66, 0x31, 0xc9,             // xor ecx, ecx
    0x0f, 0xa2,                   // cpuid
    0x66, 0xa3, 0x00, 0x11,       // mov [0x1100], eax
    0x66, 0x89, 0x1e, 0x04, 0x11, // mov [0x1104], ebx
    0xba, 0xf8, 0x03,             // mov dx, 0x3f8
    0xb0, 0x4b,                   // mov al, 'K'
    0xee,                         // out dx, al
    0xf4,                         // hlt
    0x90, 0x90, 0x90, 0x90,
];

#[test]
fn guest_cpuid_matches_the_model() {
    // With the APIC in the kernel, HLT blocks there instead of exiting.
    let Some(accel) = open(KernelIrqchip::Off) else { return };
    let cpu = kvm_cpu(&accel, "qemu64");

    let ms = MemorySystem::new();
    let root = ms.new_container("system", 1 << 32).unwrap();
    let ram = ms.new_ram("pc.ram", 0x8000).unwrap();
    ms.add_subregion(root, 0, ram).unwrap();
    let ioroot = ms.new_container("io", 0x10000).unwrap();
    let port = Arc::new(Recorder::default());
    let serial = ms.new_io("serial", 8, port.clone()).unwrap();
    ms.add_subregion(ioroot, 0x3f8, serial).unwrap();
    let mem: Arc<AddressSpace> = ms.address_space_init(root, "memory").unwrap();
    let io: Arc<AddressSpace> = ms.address_space_init(ioroot, "I/O").unwrap();
    ms.ram_block(ram).unwrap().write(0x1000, &CODE).unwrap();
    ms.register_listener(accel.slot_listener(), &mem).unwrap();

    let mut state = cpu.new_state(true);
    for seg in [R_CS, R_DS, R_ES, R_SS] {
        state.segs[seg].selector = 0;
        state.segs[seg].base = 0;
    }
    state.rip = 0x1000;
    let mut vcpu = accel.create_vcpu(0).unwrap();
    let ctx = X86KvmVcpu::init(&accel, vcpu.fd(), &cpu).unwrap();
    ctx.put_registers(vcpu.fd(), &state, PutLevel::Reset).unwrap();
    ctx.put_lapic_reset(vcpu.fd(), &state).unwrap();

    assert_eq!(vcpu.run(&io, &mem).unwrap(), VcpuStop::Halted);
    assert_eq!(*port.0.lock().unwrap(), [(0, u64::from(b'K'))]);

    let mut back = state.clone();
    ctx.get_registers(vcpu.fd(), &mut back).unwrap();
    let want = cpu.cpuid(0, 0);
    // AL and DX were overwritten after CPUID.
    assert_eq!(back.regs[R_EAX] as u32, (want[0] & !0xff) | u32::from(b'K'));
    assert_eq!(back.regs[R_EBX] as u32, want[1]);
    assert_eq!(back.regs[R_ECX] as u32, want[2]);
    assert_eq!(back.regs[R_EDX] as u32, (want[3] & 0xffff_0000) | 0x3f8);
    let mut stored = [0u8; 8];
    mem.read(0x1100, ruvm_mem::MemTxAttrs::UNSPECIFIED, &mut stored);
    assert_eq!(u32::from_le_bytes(stored[..4].try_into().unwrap()), want[0]);
    assert_eq!(u32::from_le_bytes(stored[4..].try_into().unwrap()), want[1]);
    // hlt leaves RIP after the instruction.
    assert_eq!(back.rip, 0x1000 + 24);
}
