// SPDX-License-Identifier: GPL-2.0-or-later

//! `-accel kvm` for the virt board, on Linux on AArch64 hosts: the KVM parts of
//! `machvirt_init()` and `finalize_gic_version()`, then the board on a KVM VM.

use std::sync::Arc;

use ruvm_accel::tcg::TcgOptions;
use ruvm_accel_kvm::arm::{GicVersion, default_vm_type, finalize_gic_version, kvm_gics_supported};
use ruvm_accel_kvm::{KernelIrqchip, Kvm, KvmAccel, KvmError, arm_max_vm_ipa_size, vgic_probe};
use ruvm_base::Error;
use ruvm_chardev::Chardev;
use ruvm_machine_arm::tcg_run::VirtRunConfig;
use ruvm_machine_arm::virt::kvm::{KvmVirtMachine, prepare_config, virt_kvm_type};
use ruvm_machine_arm::virt::{VIRT_DEFAULT_RAM_SIZE, VirtGicVersion};

use super::{
    ArmArgs, ArmBoard, BoardOptions, BuiltVirt, Machine, Running, build_virt, dumpdtb,
    event_handler, kvm_cpu_model, lock,
};
use crate::vl::Vm;
use crate::x86::Located;

/// The `kvm_type` hook of the board of `opts`, which `kvm_init()` calls for the type of the VM
/// it creates: `virt_kvm_type()` for virt, the IPA size of its memory map, and
/// `kvm_arch_get_default_type()` for sbsa-ref, which has no hook.
pub(crate) fn kvm_vm_type(opts: &BoardOptions) -> impl Fn(&Kvm) -> Result<u64, KvmError> + use<> {
    let virt = opts.board == ArmBoard::Virt;
    let ram_size = opts.ram_size.unwrap_or(VIRT_DEFAULT_RAM_SIZE);
    let hm = opts.highmem;
    move |kvm| {
        let (max, fixed) = arm_max_vm_ipa_size(kvm);
        if virt {
            virt_kvm_type(ram_size, &hm, max, fixed)
        } else {
            Ok(default_vm_type(max, fixed))
        }
    }
}

/// `finalize_gic_version()` for KVM: checks the GIC the board asks for is one the host's KVM
/// can give the guest, with QEMU's errors.
fn check_gic_version(accel: &KvmAccel, opts: &BoardOptions) -> Result<(), String> {
    let in_kernel = accel.kernel_irqchip() != KernelIrqchip::Off;
    let probe = if in_kernel { vgic_probe(accel) } else { 0 };
    let (supported, name) = kvm_gics_supported(in_kernel, probe).map_err(|e| e.to_string())?;
    let version = match opts.gic_version {
        VirtGicVersion::V2 => GicVersion::V2,
        VirtGicVersion::V3 => GicVersion::V3,
    };
    finalize_gic_version(name, true, version, supported, opts.max_cpus)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Lets the runstate start and stop the vCPUs.
fn set_cpu_hook(vm: &Arc<Vm>, machine: &Arc<KvmVirtMachine>) {
    let weak = Arc::downgrade(machine);
    vm.runstate.set_cpu_hook(Some(Arc::new(move |run| {
        if let Some(m) = weak.upgrade() {
            if run {
                m.start();
            } else {
                m.pause();
            }
        }
    })));
}

/// [`super::start_board_tcg`] for KVM: builds the virt board, connects its UARTs to
/// `serial_hds` and puts it on the vCPUs of `accel`, stopped until `vm_start()`.
pub(crate) fn start_board_kvm(
    vm: &Arc<Vm>,
    accel: KvmAccel,
    opts: BoardOptions,
    args: &ArmArgs<'_>,
    serial_hds: &[Option<Arc<Chardev>>],
) -> Result<Running, Vec<Located>> {
    let one = |e: String| vec![Located(None, Error::generic(e))];
    if opts.board == ArmBoard::SbsaRef {
        return Err(one("sbsa-ref: KVM is NOT supported.".to_string()));
    }
    if args.semihosting.enabled {
        return Err(one("semihosting with KVM is not supported by ruvm yet".to_string()));
    }
    check_gic_version(&accel, &opts).map_err(one)?;
    let host_pmu = accel.arm_caps().pmu_v3;
    let cpu = || kvm_cpu_model(args.cpu, host_pmu).map_err(|e| vec![Located(None, e)]);
    let mut spis = None;
    let built = build_virt(vm, opts, args, serial_hds, cpu, |cfg| {
        spis = Some(prepare_config(&accel, cfg).map_err(one)?);
        Ok(())
    })?;
    let BuiltVirt { board, console, attachments, clocks, dumpdtb: dtb_path } = built;
    let spis = spis.expect("build_virt calls prepare");

    let cfg =
        VirtRunConfig { no_reboot: args.no_reboot, tcg: TcgOptions::default(), backend: None };
    let machine =
        KvmVirtMachine::new(accel, board, spis, clocks, &cfg, event_handler(vm)).map_err(one)?;
    if let Some(path) = &dtb_path {
        let fdt = lock(machine.board()).fdt().as_bytes().to_vec();
        dumpdtb(path, &fdt)?;
    }
    let machine = Arc::new(machine);
    set_cpu_hook(vm, &machine);
    Ok(Running { machine: Machine::Kvm(machine), console, _attachments: attachments })
}
