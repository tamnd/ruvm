// SPDX-License-Identifier: GPL-2.0-or-later

//! The queries about the build and the machine: the accelerator ones from accel/accel-qmp.c
//! and hw/core/machine-qmp-cmds.c, the memory and CPU ones from the same file, and the ones
//! for devices and features ruvm does not have, which answer the way QEMU's stubs do.
//!
//! The machine queries are answered for machine `none`, the one QOM machine. The boards ruvm
//! builds outside QOM say they do not support the command yet rather than report a machine
//! without devices.

use std::sync::Arc;

use ruvm_base::{Error, ErrorClass, Result};
use ruvm_hw_core::Machine;
use ruvm_migration::default_parameters;
use ruvm_monitor::{Commands, MonitorQmp};
use ruvm_qapi::commands::*;
use ruvm_qapi::types::{
    Accelerator, AcceleratorInfo, CurrentMachineParams, DirtyRateInfo, DirtyRateMeasureMode,
    DirtyRateStatus, DumpGuestMemoryCapability, DumpQueryResult, DumpStatus, HumanReadableText,
    KvmInfo, MemoryInfo, MigrationCapability, MigrationCapabilityStatus, QemuTargetInfo,
    ReplayInfo, ReplayMode, SysEmuTarget, TimeUnit, UuidInfo, XDbgBlockGraph, YankInstance,
    YankInstanceBlockNode, YankInstanceChardev, YankInstanceU,
};

use crate::vl::{Vm, accels};

/// Machine `none`, or the error for a board that does not answer `cmd` yet.
fn none_machine<'a>(vm: &'a Vm, cmd: &str) -> Result<&'a Machine> {
    vm.machine.get().ok_or_else(|| {
        Error::generic(format!("{cmd} is not supported with this machine by ruvm yet"))
    })
}

fn text(s: impl Into<String>) -> HumanReadableText {
    HumanReadableText { human_readable_text: s.into() }
}

/// The accelerator `configure_accelerators()` picked.
fn current_accel(vm: &Vm) -> Result<Accelerator> {
    vm.accel.get().copied().ok_or_else(|| Error::generic("no accelerator has been set up yet"))
}

/// `CONFIG_MEM_DEVICE`, which the aarch64 and x86 boards select and the riscv64 ones do not.
fn has_mem_device(target: &str) -> bool {
    !crate::riscv::is_riscv(target)
}

/// Registers the queries with `vm`'s dispatcher. `target` is the one of the personality.
pub(crate) fn register(vm: &Arc<Vm>, target: &str, cmds: &mut Commands) {
    let present: Vec<Accelerator> =
        Accelerator::ALL.iter().copied().filter(|a| accels(target).contains(&a.as_str())).collect();
    let kvm_present = present.contains(&Accelerator::Kvm);
    let v = vm.clone();
    register_query_kvm(cmds, move |_: &MonitorQmp| {
        let enabled = v.accel.get() == Some(&Accelerator::Kvm);
        Ok(KvmInfo { enabled, present: kvm_present })
    });
    let v = vm.clone();
    register_query_accelerators(cmds, move |_: &MonitorQmp| {
        Ok(AcceleratorInfo { enabled: current_accel(&v)?, present: present.clone() })
    });
    let arch = SysEmuTarget::from_name(target);
    register_query_target(cmds, move |_: &MonitorQmp| {
        let arch = arch.ok_or_else(|| Error::generic("unknown target"))?;
        Ok(QemuTargetInfo { arch })
    });
    register_query_uuid(cmds, |_: &MonitorQmp| {
        Ok(UuidInfo { UUID: crate::display::qemu_uuid_string() })
    });
    // x-query-jit: ruvm keeps no statistics of its translation cache.
    let v = vm.clone();
    register_x_query_jit(cmds, move |_: &MonitorQmp| {
        if current_accel(&v)? != Accelerator::Tcg {
            return Err(Error::generic("JIT information is only available with accel=tcg"));
        }
        Err(Error::generic("x-query-jit is not supported by ruvm yet"))
    });
    // qtest keeps no statistics. TCG and KVM print theirs in QEMU, which ruvm does not yet.
    let v = vm.clone();
    register_x_accel_stats(cmds, move |_: &MonitorQmp| {
        if current_accel(&v)? != Accelerator::Qtest {
            return Err(Error::generic(
                "x-accel-stats is not supported with this accelerator by ruvm yet",
            ));
        }
        Ok(text(""))
    });
    let v = vm.clone();
    register_query_command_line_options(cmds, move |_: &MonitorQmp, arg| {
        crate::config_qmp::query_command_line_options(&v.registry, arg.option.as_deref())
    });
    // ruvm registers no statistics providers.
    register_query_stats_schemas(cmds, |_: &MonitorQmp, _| Ok(Vec::new()));
    // qmp_query_dirty_rate() before a measurement. ruvm has no calc-dirty-rate to start one.
    register_query_dirty_rate(cmds, |_: &MonitorQmp, arg| {
        Ok(DirtyRateInfo {
            dirty_rate: None,
            status: DirtyRateStatus::Unstarted,
            start_time: 0,
            calc_time: 0,
            calc_time_unit: arg.calc_time_unit.unwrap_or(TimeUnit::Second),
            sample_pages: 0,
            mode: DirtyRateMeasureMode::PageSampling,
            vcpu_dirty_rate: None,
        })
    });
    // ruvm has no record and replay and no -icount.
    register_query_replay(cmds, |_: &MonitorQmp| {
        Ok(ReplayInfo { mode: ReplayMode::None, filename: None, icount: 0 })
    });
    // Nor guest memory dumps.
    register_query_dump(cmds, |_: &MonitorQmp| {
        Ok(DumpQueryResult { status: DumpStatus::None, completed: 0, total: 0 })
    });
    register_query_dump_guest_memory_capability(cmds, |_: &MonitorQmp| {
        Ok(DumpGuestMemoryCapability { formats: Vec::new() })
    });

    // Objects and exports the system emulator has no types for.
    register_query_iothreads(cmds, |_: &MonitorQmp| Ok(Vec::new()));
    register_query_pr_managers(cmds, |_: &MonitorQmp| Ok(Vec::new()));
    register_query_cryptodev(cmds, |_: &MonitorQmp| Ok(Vec::new()));
    register_query_block_exports(cmds, |_: &MonitorQmp| Ok(Vec::new()));
    // qmp_query_vcpu_dirty_limit() outside a dirty limit, which ruvm does not set.
    register_query_vcpu_dirty_limit(cmds, |_: &MonitorQmp| Ok(Vec::new()));
    let v = vm.clone();
    register_query_block_jobs(cmds, move |_: &MonitorQmp| v.block.query_block_jobs());
    let v = vm.clone();
    register_query_memdev(cmds, move |_: &MonitorQmp| Ok(ruvm_hostmem::query_memdev(&v.registry)));
    // qmp_query_yank(): the instances in the order they were registered. Socket chardevs and
    // NBD nodes register one as they open; ruvm lists the chardevs first.
    let v = vm.clone();
    register_query_yank(cmds, move |_: &MonitorQmp| {
        let chardevs = v
            .chardevs
            .yank_instances()
            .into_iter()
            .map(|id| YankInstance { u: YankInstanceU::Chardev(YankInstanceChardev { id }) });
        let nodes = v.block.yank_nodes().into_iter().map(|node_name| YankInstance {
            u: YankInstanceU::BlockNode(YankInstanceBlockNode { node_name }),
        });
        Ok(chardevs.chain(nodes).collect())
    });

    // The migration settings before a machine that migrates has made its state.
    let v = vm.clone();
    register_query_migrate_capabilities(cmds, move |_: &MonitorQmp| match v.migration.get() {
        Some(m) => Ok(m.query_capabilities()),
        None => Ok(MigrationCapability::ALL
            .iter()
            .map(|&capability| MigrationCapabilityStatus { capability, state: false })
            .collect()),
    });
    let v = vm.clone();
    register_query_migrate_parameters(cmds, move |_: &MonitorQmp| match v.migration.get() {
        Some(m) => Ok(m.query_parameters()),
        None => Ok(ruvm_qapi::types::MigrationParameters {
            block_bitmap_mapping: None,
            cpr_exec_command: Some(Vec::new()),
            ..default_parameters()
        }),
    });

    // The stubs of features ruvm does not build.
    register_query_hv_balloon_status_report(cmds, |_: &MonitorQmp| {
        Err(Error::generic("hv-balloon device not enabled in this build"))
    });
    register_query_sev(cmds, |_: &MonitorQmp| {
        Err(Error::generic("SEV is not available in this QEMU"))
    });
    register_query_sev_capabilities(cmds, |_: &MonitorQmp| {
        Err(Error::generic("SEV is not available in this QEMU"))
    });
    register_query_sev_launch_measure(cmds, |_: &MonitorQmp| {
        Err(Error::generic("SEV is not available in this QEMU"))
    });
    register_query_sgx(cmds, |_: &MonitorQmp| {
        Err(Error::generic("SGX support is not compiled in"))
    });
    register_query_sgx_capabilities(cmds, |_: &MonitorQmp| {
        Err(Error::generic("SGX support is not compiled in"))
    });
    register_query_s390x_cpu_polarization(cmds, |_: &MonitorQmp| {
        Err(Error::generic("CPU polarization is not supported on this target"))
    });
    register_xen_event_list(cmds, |_: &MonitorQmp| {
        Err(Error::generic("Xen event channel emulation not enabled"))
    });

    register_machine(vm, target, cmds);
}

/// The queries about the machine and its devices.
fn register_machine(vm: &Arc<Vm>, target: &str, cmds: &mut Commands) {
    let v = vm.clone();
    register_query_current_machine(cmds, move |_: &MonitorQmp| {
        none_machine(&v, "query-current-machine")?;
        Ok(CurrentMachineParams { wakeup_suspend_support: false })
    });
    let v = vm.clone();
    let plugged = has_mem_device(target).then_some(0);
    register_query_memory_size_summary(cmds, move |_: &MonitorQmp| {
        let m = none_machine(&v, "query-memory-size-summary")?;
        Ok(MemoryInfo { base_memory: m.ram_size(), plugged_memory: plugged })
    });
    // Machine none has no PCI bus, no memory devices, no ACPI and no balloon.
    let v = vm.clone();
    register_query_pci(cmds, move |_: &MonitorQmp| {
        none_machine(&v, "query-pci")?;
        Ok(Vec::new())
    });
    let v = vm.clone();
    register_query_memory_devices(cmds, move |_: &MonitorQmp| {
        none_machine(&v, "query-memory-devices")?;
        Ok(Vec::new())
    });
    let v = vm.clone();
    register_query_hotpluggable_cpus(cmds, move |_: &MonitorQmp| {
        none_machine(&v, "query-hotpluggable-cpus")?;
        Err(Error::generic("machine does not support hot-plugging CPUs"))
    });
    let v = vm.clone();
    register_query_acpi_ospm_status(cmds, move |_: &MonitorQmp| {
        none_machine(&v, "query-acpi-ospm-status")?;
        Err(Error::generic("command is not supported, missing ACPI device"))
    });
    let v = vm.clone();
    register_query_balloon(cmds, move |_: &MonitorQmp| {
        none_machine(&v, "query-balloon")?;
        Err(Error::new(ErrorClass::DeviceNotActive, "No balloon device has been activated"))
    });
    let v = vm.clone();
    register_query_vm_generation_id(cmds, move |_: &MonitorQmp| {
        none_machine(&v, "query-vm-generation-id")?;
        Err(Error::generic("VM Generation ID device not found"))
    });
    let v = vm.clone();
    register_x_query_virtio(cmds, move |_: &MonitorQmp| {
        none_machine(&v, "x-query-virtio")?;
        Err(Error::generic("No virtio devices found"))
    });
    let v = vm.clone();
    register_x_query_usb(cmds, move |_: &MonitorQmp| {
        let m = none_machine(&v, "x-query-usb")?;
        if !m.object.property_get_bool("usb")? {
            return Err(Error::generic("USB support not enabled"));
        }
        Ok(text(""))
    });
    // Nothing on machine none keeps interrupt statistics or loads ROMs.
    let v = vm.clone();
    register_x_query_irq(cmds, move |_: &MonitorQmp| {
        none_machine(&v, "x-query-irq")?;
        Ok(text(""))
    });
    let v = vm.clone();
    register_x_query_interrupt_controllers(cmds, move |_: &MonitorQmp| {
        none_machine(&v, "x-query-interrupt-controllers")?;
        Ok(text(""))
    });
    let v = vm.clone();
    register_x_query_roms(cmds, move |_: &MonitorQmp| {
        none_machine(&v, "x-query-roms")?;
        Ok(text(""))
    });
    // Machine none has no drives and no NICs. Its block nodes are the ones blockdev-add made.
    let v = vm.clone();
    register_query_block(cmds, move |_: &MonitorQmp, _| {
        none_machine(&v, "query-block")?;
        Ok(Vec::new())
    });
    let v = vm.clone();
    register_query_blockstats(cmds, move |_: &MonitorQmp, arg| {
        none_machine(&v, "query-blockstats")?;
        if arg.query_nodes == Some(true) && !v.block.nodes().is_empty() {
            return Err(Error::generic(
                "query-blockstats of block nodes is not supported by ruvm yet",
            ));
        }
        Ok(Vec::new())
    });
    let v = vm.clone();
    register_query_named_block_nodes(cmds, move |_: &MonitorQmp, arg| {
        none_machine(&v, "query-named-block-nodes")?;
        v.block.query_named_block_nodes(arg.flat)
    });
    let v = vm.clone();
    register_x_debug_query_block_graph(cmds, move |_: &MonitorQmp| {
        none_machine(&v, "x-debug-query-block-graph")?;
        if !v.block.nodes().is_empty() {
            return Err(Error::generic(
                "x-debug-query-block-graph with block nodes is not supported by ruvm yet",
            ));
        }
        Ok(XDbgBlockGraph { nodes: Vec::new(), edges: Vec::new() })
    });
    let v = vm.clone();
    register_query_rx_filter(cmds, move |_: &MonitorQmp, arg| {
        none_machine(&v, "query-rx-filter")?;
        if arg.name.is_some() {
            return Err(Error::generic(
                "query-rx-filter of a net client is not supported by ruvm yet",
            ));
        }
        Ok(Vec::new())
    });
    // find_ovmf_log() looks only on the x86 and Arm virt boards.
    let v = vm.clone();
    register_query_firmware_log(cmds, move |_: &MonitorQmp, _| {
        none_machine(&v, "query-firmware-log")?;
        Err(Error::generic("firmware log buffer not found"))
    });
    // ruvm has no -numa.
    register_x_query_numa(cmds, |_: &MonitorQmp| Ok(text("0 nodes\n")));
    // ram_block_format() with no RAM blocks, the header alone. ruvm does not keep the
    // ram_addr_t offsets the rows show.
    let v = vm.clone();
    register_x_query_ramblock(cmds, move |_: &MonitorQmp| {
        let m = none_machine(&v, "x-query-ramblock")?;
        if m.ram_size() != 0 || !ruvm_hostmem::query_memdev(&v.registry).is_empty() {
            return Err(Error::generic(
                "x-query-ramblock is not supported with RAM blocks by ruvm yet",
            ));
        }
        Ok(text(format!(
            "{:>24} {:>8}  {:>18} {:>18} {:>18} {:>18} {:>3}\n",
            "Block Name", "PSize", "Offset", "Used", "Total", "HVA", "RO"
        )))
    });
}
