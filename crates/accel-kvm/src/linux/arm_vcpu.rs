// SPDX-License-Identifier: GPL-2.0-or-later

//! The Arm vCPU sequences of `crate::arm::vcpu` on a real vCPU, and the VM queries they need:
//! the preferred target and the capabilities.

use std::io;
use std::mem::offset_of;
use std::os::fd::AsRawFd;

use kvm_bindings::{
    kvm_device_attr, kvm_mp_state, kvm_regs, kvm_vcpu_events, kvm_vcpu_init, user_fpsimd_state,
    user_pt_regs,
};

use super::{KvmAccel, KvmVcpu, os_error};
use crate::KvmError;
use crate::arm::vcpu::{self, ArmKvmCaps, ArmVcpuOps, SErrorEvents, core_offset};

// The pure code spells out the kernel ABI by hand so it builds on every host. Tie it to the
// bindings here.
const _: () = {
    use kvm_bindings as b;
    assert!(vcpu::KVM_REG_ARM64 == b::KVM_REG_ARM64);
    assert!(vcpu::KVM_REG_SIZE_SHIFT == b::KVM_REG_SIZE_SHIFT);
    assert!(vcpu::KVM_REG_SIZE_MASK == b::KVM_REG_SIZE_MASK);
    assert!(vcpu::KVM_REG_SIZE_U32 == b::KVM_REG_SIZE_U32);
    assert!(vcpu::KVM_REG_SIZE_U64 == b::KVM_REG_SIZE_U64);
    assert!(vcpu::KVM_REG_SIZE_U128 == b::KVM_REG_SIZE_U128);
    assert!(vcpu::KVM_REG_SIZE_U256 == b::KVM_REG_SIZE_U256);
    assert!(vcpu::KVM_REG_SIZE_U512 == b::KVM_REG_SIZE_U512);
    assert!(vcpu::KVM_REG_SIZE_U2048 == b::KVM_REG_SIZE_U2048);
    assert!(vcpu::KVM_REG_ARM_COPROC_MASK == b::KVM_REG_ARM_COPROC_MASK as u64);
    assert!(vcpu::KVM_REG_ARM_CORE == b::KVM_REG_ARM_CORE as u64);
    assert!(vcpu::KVM_REG_ARM_DEMUX == b::KVM_REG_ARM_DEMUX as u64);
    assert!(vcpu::KVM_REG_ARM64_SYSREG == b::KVM_REG_ARM64_SYSREG as u64);
    assert!(vcpu::KVM_REG_ARM_FW == b::KVM_REG_ARM_FW as u64);
    assert!(vcpu::KVM_REG_ARM64_SVE == b::KVM_REG_ARM64_SVE as u64);
    assert!(vcpu::KVM_REG_ARM_FW_FEAT_BMAP == b::KVM_REG_ARM_FW_FEAT_BMAP as u64);
    assert!(vcpu::KVM_ARM_VCPU_POWER_OFF == b::KVM_ARM_VCPU_POWER_OFF);
    assert!(vcpu::KVM_ARM_VCPU_EL1_32BIT == b::KVM_ARM_VCPU_EL1_32BIT);
    assert!(vcpu::KVM_ARM_VCPU_PSCI_0_2 == b::KVM_ARM_VCPU_PSCI_0_2);
    assert!(vcpu::KVM_ARM_VCPU_PMU_V3 == b::KVM_ARM_VCPU_PMU_V3);
    assert!(vcpu::KVM_ARM_VCPU_SVE == b::KVM_ARM_VCPU_SVE);
    assert!(vcpu::KVM_ARM_VCPU_PTRAUTH_ADDRESS == b::KVM_ARM_VCPU_PTRAUTH_ADDRESS);
    assert!(vcpu::KVM_ARM_VCPU_PTRAUTH_GENERIC == b::KVM_ARM_VCPU_PTRAUTH_GENERIC);
    assert!(vcpu::KVM_ARM_VCPU_HAS_EL2 == b::KVM_ARM_VCPU_HAS_EL2);
    assert!(vcpu::KVM_ARM_VCPU_PMU_V3_CTRL == b::KVM_ARM_VCPU_PMU_V3_CTRL);
    assert!(vcpu::KVM_ARM_VCPU_PMU_V3_IRQ == b::KVM_ARM_VCPU_PMU_V3_IRQ as u64);
    assert!(vcpu::KVM_ARM_VCPU_PMU_V3_INIT == b::KVM_ARM_VCPU_PMU_V3_INIT as u64);
    assert!(vcpu::KVM_MP_STATE_RUNNABLE == b::KVM_MP_STATE_RUNNABLE);
    assert!(vcpu::KVM_MP_STATE_STOPPED == b::KVM_MP_STATE_STOPPED);

    assert!(offset_of!(kvm_regs, regs) == 0);
    assert!(offset_of!(user_pt_regs, regs) as u64 == core_offset::X0);
    assert!(offset_of!(user_pt_regs, sp) as u64 == core_offset::SP);
    assert!(offset_of!(user_pt_regs, pc) as u64 == core_offset::PC);
    assert!(offset_of!(user_pt_regs, pstate) as u64 == core_offset::PSTATE);
    assert!(offset_of!(kvm_regs, sp_el1) as u64 == core_offset::SP_EL1);
    assert!(offset_of!(kvm_regs, elr_el1) as u64 == core_offset::ELR_EL1);
    assert!(offset_of!(kvm_regs, spsr) as u64 == core_offset::SPSR);
    let fp = offset_of!(kvm_regs, fp_regs) as u64;
    assert!(fp + offset_of!(user_fpsimd_state, vregs) as u64 == core_offset::VREGS);
    assert!(fp + offset_of!(user_fpsimd_state, fpsr) as u64 == core_offset::FPSR);
    assert!(fp + offset_of!(user_fpsimd_state, fpcr) as u64 == core_offset::FPCR);
};

/// `_IOWR(KVMIO, 0xb0, struct kvm_reg_list)`. The struct is its 8 byte count.
const KVM_GET_REG_LIST: u64 = (3 << 30) | (8 << 16) | (0xae << 8) | 0xb0;

impl KvmVcpu {
    /// `KVM_GET_REG_LIST` into `buf`, whose first word is the room for ids. On `E2BIG` the
    /// kernel has put the count it needs there. kvm-ioctls caps its list at 500 ids, which a
    /// kernel with SVE and the newer system registers can pass, so this sizes its own.
    fn get_reg_list_into(&self, buf: &mut [u64]) -> io::Result<()> {
        debug_assert!((buf[0] as usize) < buf.len());
        // SAFETY: `buf` is a `struct kvm_reg_list` whose `n` is no larger than the room after
        // it, so the kernel writes at most `n` ids inside the slice, and the slice outlives the
        // call.
        let ret =
            unsafe { libc::ioctl(self.fd().as_raw_fd(), KVM_GET_REG_LIST as _, buf.as_mut_ptr()) };
        if ret < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
    }
}

impl ArmVcpuOps for KvmVcpu {
    fn vcpu_init(&mut self, target: u32, features: &[u32; 7]) -> io::Result<()> {
        let init = kvm_vcpu_init { target, features: *features };
        self.fd().vcpu_init(&init).map_err(os_error)
    }

    fn vcpu_finalize(&mut self, feature: u32) -> io::Result<()> {
        self.fd().vcpu_finalize(&(feature as libc::c_int)).map_err(os_error)
    }

    fn reg_list(&mut self) -> io::Result<Vec<u64>> {
        // `kvm_arm_init_cpreg_list()`: ask with no room to learn the count, then fetch.
        let mut probe = [0u64; 1];
        match self.get_reg_list_into(&mut probe) {
            Ok(()) => return Ok(Vec::new()),
            Err(e) if e.raw_os_error() == Some(libc::E2BIG) => {}
            Err(e) => return Err(e),
        }
        let n = probe[0] as usize;
        let mut buf = vec![0u64; n + 1];
        buf[0] = n as u64;
        self.get_reg_list_into(&mut buf)?;
        let got = (buf[0] as usize).min(n);
        buf.truncate(got + 1);
        buf.remove(0);
        Ok(buf)
    }

    fn get_one_reg(&mut self, id: u64, data: &mut [u8]) -> io::Result<()> {
        self.fd().get_one_reg(id, data).map(drop).map_err(os_error)
    }

    fn set_one_reg(&mut self, id: u64, data: &[u8]) -> io::Result<()> {
        self.fd().set_one_reg(id, data).map(drop).map_err(os_error)
    }

    fn get_mp_state(&mut self) -> io::Result<u32> {
        self.fd().get_mp_state().map(|s| s.mp_state).map_err(os_error)
    }

    fn set_mp_state(&mut self, state: u32) -> io::Result<()> {
        self.fd().set_mp_state(kvm_mp_state { mp_state: state }).map_err(os_error)
    }

    fn get_vcpu_events(&mut self) -> io::Result<SErrorEvents> {
        let events = self.fd().get_vcpu_events().map_err(os_error)?;
        let e = events.exception;
        Ok(SErrorEvents {
            pending: e.serror_pending != 0,
            has_esr: e.serror_has_esr != 0,
            esr: e.serror_esr,
        })
    }

    fn set_vcpu_events(&mut self, events: &SErrorEvents) -> io::Result<()> {
        let mut raw = kvm_vcpu_events::default();
        raw.exception.serror_pending = u8::from(events.pending);
        raw.exception.serror_has_esr = u8::from(events.has_esr);
        raw.exception.serror_esr = events.esr;
        self.fd().set_vcpu_events(&raw).map_err(os_error)
    }

    fn has_device_attr(&mut self, group: u32, attr: u64, value: Option<i32>) -> io::Result<()> {
        let value = value.unwrap_or(0);
        let raw = device_attr(group, attr, &value);
        self.fd().has_device_attr(&raw).map_err(os_error)
    }

    fn set_device_attr(&mut self, group: u32, attr: u64, value: Option<i32>) -> io::Result<()> {
        let value = value.unwrap_or(0);
        let raw = device_attr(group, attr, &value);
        self.fd().set_device_attr(&raw).map_err(os_error)
    }
}

/// A vCPU attribute whose address points at `value`. The PMU attributes that take no value
/// ignore it, and QEMU passes a null address for them; a valid one is just as good.
fn device_attr(group: u32, attr: u64, value: &i32) -> kvm_device_attr {
    kvm_device_attr { flags: 0, group, attr, addr: value as *const i32 as u64 }
}

impl KvmAccel {
    /// `KVM_ARM_PREFERRED_TARGET`: the target `-cpu host` passes to `KVM_ARM_VCPU_INIT`.
    pub fn arm_preferred_target(&self) -> Result<u32, KvmError> {
        let mut init = kvm_vcpu_init::default();
        self.vm()
            .get_preferred_target(&mut init)
            .map_err(|e| KvmError::Ioctl("KVM_ARM_PREFERRED_TARGET", os_error(e)))?;
        Ok(init.target)
    }

    /// The capabilities the Arm vCPU code looks at.
    pub fn arm_caps(&self) -> ArmKvmCaps {
        let has = |cap: u32| self.vm().check_extension_raw(libc::c_ulong::from(cap)) > 0;
        use kvm_bindings as b;
        ArmKvmCaps {
            psci_0_2: has(b::KVM_CAP_ARM_PSCI_0_2),
            el1_32bit: has(b::KVM_CAP_ARM_EL1_32BIT),
            pmu_v3: has(b::KVM_CAP_ARM_PMU_V3),
            sve: has(b::KVM_CAP_ARM_SVE),
            ptrauth: has(b::KVM_CAP_ARM_PTRAUTH_ADDRESS) && has(b::KVM_CAP_ARM_PTRAUTH_GENERIC),
            el2: has(vcpu::KVM_CAP_ARM_EL2),
            mp_state: has(b::KVM_CAP_MP_STATE),
            vcpu_events: has(b::KVM_CAP_VCPU_EVENTS),
            inject_serror_esr: has(b::KVM_CAP_ARM_INJECT_SERROR_ESR),
            irq_line_layout_2: has(b::KVM_CAP_ARM_IRQ_LINE_LAYOUT_2),
        }
    }
}
