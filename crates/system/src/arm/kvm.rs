// SPDX-License-Identifier: GPL-2.0-or-later

//! `-accel kvm` for the virt board, on Linux on AArch64 hosts: the KVM parts of
//! `machvirt_init()` and `finalize_gic_version()`, then the board on a KVM VM.

use std::sync::Arc;

use ruvm_accel::tcg::TcgOptions;
use ruvm_accel_kvm::arm::{GicVersion, finalize_gic_version, kvm_gics_supported};
use ruvm_accel_kvm::{KernelIrqchip, KvmAccel, vgic_probe};
use ruvm_base::Error;
use ruvm_chardev::Chardev;
use ruvm_machine_arm::tcg_run::VirtRunConfig;
use ruvm_machine_arm::virt::VirtGicVersion;
use ruvm_machine_arm::virt::kvm::{KvmVirtMachine, prepare_config};

use super::{
    ArmArgs, ArmBoard, BoardOptions, BuiltVirt, Machine, Running, build_virt, dumpdtb,
    event_handler, kvm_cpu_model, lock,
};
use crate::vl::Vm;
use crate::x86::Located;

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
