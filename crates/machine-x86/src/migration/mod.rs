// SPDX-License-Identifier: GPL-2.0-or-later

//! Migration of a q35 or microvm machine on TCG, in QEMU's stream format.
//!
//! [`x86_savevm`] registers the sections of the machine the way QEMU 11.1 lays them out for
//! `-M pc-q35-11.1` (and the older q35 versions, which are the same here) or `-M microvm`, in
//! QEMU's order so they get QEMU's section ids: the `timer` section, RAM, each vCPU with its
//! local APIC, then the board's devices (microvm's fw_cfg comes before the vCPUs). Each device section carries the
//! device's state field by field, as QEMU's `VMStateDescription` for it does. The 8237 DMA
//! controllers, which ruvm does not model, and kvmvapic, which is for KVM, are parsed and
//! dropped.

mod ahci;
mod apic;
mod cpu;
mod fw_cfg;
mod ged;
mod hpet;
mod input;
mod ioapic;
mod legacy;
mod lpc;
mod pci;
mod pic;
mod pit;
mod rtc;
mod serial;
mod skip;
mod smbus;
mod timer;
mod virtio;

use std::sync::{Arc, PoisonError, Weak};

use ruvm_accel::tcg::TcgVcpus;
use ruvm_jit::cputlb::tlb_flush;
use ruvm_migration::{
    EntryInfo, GlobalState, MachineConfig, RamHooks, RamSection, RamStats, SaveVm,
};

use ruvm_base::error_report;
use ruvm_hw_virtio::VirtioPci;
use ruvm_mem::{GLOBAL_DIRTY_MIGRATION, MemorySystem, RamBlock};

use crate::board::X86Board;
use crate::microvm::{Microvm, VIRTIO_MMIO_BASE, VIRTIO_MMIO_STRIDE};
use crate::q35::{MAX_ISA_SERIAL_PORTS, Q35};
use crate::tcg_run::TcgMachine;

/// The RAM hooks of a TCG machine. The dirty bitmaps are set by the TLB's not-dirty path, so
/// after bits are taken out the TLBs must forget which pages were already written. Device
/// writes go through the address spaces, which mark pages for migration only while global
/// dirty logging is on.
#[derive(Debug)]
struct TcgHooks {
    vcpus: Weak<TcgVcpus>,
    mem: Weak<MemorySystem>,
}

impl RamHooks for TcgHooks {
    fn log_start(&self) {
        if let Some(m) = self.mem.upgrade() {
            if let Err(e) = m.global_dirty_log_start(GLOBAL_DIRTY_MIGRATION) {
                error_report(&format!("cannot start dirty logging: {e}"));
            }
        }
    }

    fn log_stop(&self) {
        if let Some(m) = self.mem.upgrade() {
            m.global_dirty_log_stop(GLOBAL_DIRTY_MIGRATION);
        }
    }

    fn after_clear(&self) {
        if let Some(v) = self.vcpus.upgrade() {
            v.run_on_each(tlb_flush);
        }
    }

    fn load_done(&self) {
        if let Some(v) = self.vcpus.upgrade() {
            v.jit().tb_flush_exclusive_or_serial();
            v.run_on_each(tlb_flush);
        }
    }
}

/// The registered sections of a machine and the handles the migration code needs.
#[derive(Debug)]
pub struct X86Migration {
    /// The sections.
    pub savevm: SaveVm,
    /// The counters of the `ram` section.
    pub ram_stats: Arc<RamStats>,
    /// The `globalstate` section.
    pub global_state: Arc<GlobalState>,
}

/// Registers the sections of `machine`, a q35 or microvm board, for migration as machine type
/// `machine_type`.
pub fn x86_savevm(
    machine: &TcgMachine,
    machine_type: &str,
    uuid: Option<[u8; 16]>,
) -> Result<X86Migration, String> {
    let mut savevm = SaveVm::new(MachineConfig {
        name: machine_type.to_string(),
        page_bits: 12,
        legacy_page_bits: 12,
        uuid,
    });
    let board = machine.board().lock().unwrap_or_else(PoisonError::into_inner);
    let ram_stats = match &*board {
        X86Board::Q35(q35, devs) => register_q35(&mut savevm, machine, q35, devs),
        X86Board::Microvm(m) => register_microvm(&mut savevm, machine, m),
    };
    drop(board);
    let global_state = GlobalState::register(&mut savevm);
    Ok(X86Migration { savevm, ram_stats, global_state })
}

/// The sections every board starts with, from migration_object_init() and the timers: the
/// `timer` section, RAM and the dirty bitmaps. Gives the RAM counters.
fn register_common(
    savevm: &mut SaveVm,
    machine: &TcgMachine,
    mem: &Arc<MemorySystem>,
    blocks: Vec<Arc<RamBlock>>,
) -> Arc<RamStats> {
    timer::register(savevm, machine.virtual_clock(), machine.ticks());
    let ram = RamSection::new(
        blocks,
        TcgHooks { vcpus: Arc::downgrade(machine.vcpus()), mem: Arc::downgrade(mem) },
    );
    let ram_stats = ram.stats();
    savevm.register_live(EntryInfo::new("ram", 4).instance(0), ram);
    // "dirty-bitmap": ruvm has no block dirty bitmaps to migrate.
    savevm.reserve_section_id();
    ram_stats
}

/// Each vCPU with its APIC; the first APIC brings kvmvapic along.
fn register_cpus(savevm: &mut SaveVm, machine: &TcgMachine) {
    for (n, shared) in machine.vcpus().cpus().iter().enumerate() {
        let index = shared.cpu_index as u32;
        let (get, put) = (Arc::clone(shared), Arc::clone(shared));
        savevm.register_vmsd(
            "",
            Some(index),
            &cpu::VMSTATE_CPU_COMMON,
            move || cpu::get_common(&get),
            move |c| cpu::put_common(&put, c),
        );
        let side = cpu::SideStore::default();
        let (get, put) = (Arc::clone(shared), Arc::clone(shared));
        let get_side = Arc::clone(&side);
        savevm.register_vmsd(
            "",
            Some(index),
            &cpu::VMSTATE_X86_CPU,
            move || cpu::get_cpu(&get, &get_side),
            move |c| cpu::put_cpu(&put, &side, c),
        );
        if n == 0 {
            skip::register_vapic(savevm);
        }
        if let Some(a) = machine.apics().get(n) {
            apic::register(savevm, a);
        }
    }
}

/// The sections of a q35 board, in the order QEMU registers them so they get QEMU's section
/// ids: the common ones, the vCPUs, then the board.
fn register_q35(
    savevm: &mut SaveVm,
    machine: &TcgMachine,
    q35: &Q35,
    devs: &[VirtioPci],
) -> Arc<RamStats> {
    let ram_stats =
        register_common(savevm, machine, q35.memory_system(), q35.migratable_ram_blocks());
    register_cpus(savevm, machine);

    // pc_q35_init(): fw_cfg, the host bridge and its bus, then the ISA bridge and the devices
    // on it in the order pc_basic_device_init() and pc_q35_init() create them.
    let max_cpus = q35.max_cpus() as usize;
    fw_cfg::register(savevm, q35.fw_cfg());
    pci::register_mch(savevm, q35.host());
    pci::register_pci_host(savevm, q35.host().host_state());
    pci::register_pci_bus(savevm, q35.pci_bus());
    legacy::register_dma(savevm);
    rtc::register(savevm, q35.rtc());
    lpc::register(savevm, q35.lpc(), max_cpus);
    if let Some(pic) = q35.pic() {
        pic::register(savevm, pic);
    }
    ioapic::register(savevm, 0, q35.ioapic());
    if let Some(hpet) = q35.hpet() {
        hpet::register(savevm, hpet);
    }
    if let Some(pit) = q35.pit() {
        pit::register(savevm, pit);
    }
    if let Some(spk) = q35.pcspk() {
        pit::register_pcspk(savevm, spk);
    }
    for i in 0..MAX_ISA_SERIAL_PORTS {
        if let Some(s) = q35.serial(i) {
            serial::register(savevm, i as u32, s);
        }
    }
    if let Some(i8042) = q35.i8042() {
        input::register(savevm, i8042, q35.vmport());
    }
    if let Some(p) = q35.port92() {
        legacy::register_port92(savevm, p);
    }
    if let Some(ahci) = q35.ahci() {
        ahci::register(savevm, ahci);
    }
    if let Some(smb) = q35.smbus() {
        legacy::register_i2c_bus(savevm, smb.smbus());
        smbus::register(savevm, smb);
        // The eight smbus-eeprom devices: registered, never sent.
        for _ in 0..8 {
            savevm.reserve_section_id();
        }
    }
    // The -device virtio functions, then acpi_build from the machine_done notifier.
    for dev in devs {
        virtio::register_pci(savevm, dev);
    }
    let (get, set) = q35.acpi_patched();
    lpc::register_acpi_build(savevm, get, set);
    ram_stats
}

/// The sections of a microvm board, in QEMU's order: microvm_memory_init() creates fw_cfg
/// before the vCPUs, then microvm_devices_init() adds the IOAPICs, the GED, the 8259 pair, the
/// PIT, the RTC and the serial port. The virtio-mmio transports have no section of their own;
/// the devices plugged into them come last, as -device creates them after the board.
fn register_microvm(savevm: &mut SaveVm, machine: &TcgMachine, m: &Microvm) -> Arc<RamStats> {
    let ram_stats = register_common(savevm, machine, m.memory_system(), m.migratable_ram_blocks());
    fw_cfg::register(savevm, m.fw_cfg());
    register_cpus(savevm, machine);
    ioapic::register(savevm, 0, m.ioapic());
    if let Some(io2) = m.ioapic2() {
        ioapic::register(savevm, 1, io2);
    }
    if let Some(ged) = m.ged() {
        ged::register(savevm, ged);
    }
    if let Some(pic) = m.pic() {
        pic::register(savevm, pic);
    }
    if let Some(pit) = m.pit() {
        pit::register(savevm, pit);
    }
    if let Some(rtc) = m.rtc() {
        rtc::register(savevm, rtc);
    }
    if let Some(s) = m.serial() {
        serial::register(savevm, 0, s);
    }
    for i in 0..m.virtio_transport_count() {
        if !m.virtio_plugged(i) {
            continue;
        }
        if let Some(t) = m.virtio_transport(i) {
            virtio::register_mmio(savevm, VIRTIO_MMIO_BASE + i as u64 * VIRTIO_MMIO_STRIDE, t);
        }
    }
    ram_stats
}

#[cfg(test)]
mod tests {
    use ruvm_mem::{DirtyClient, MemTxAttrs};

    use super::*;

    #[test]
    fn device_writes_are_dirty_while_migrating() {
        let mem = Arc::new(MemorySystem::new());
        let root = mem.new_container("system", 1 << 20).unwrap();
        let ram = mem.new_ram("ram", 0x10000).unwrap();
        mem.add_subregion(root, 0, ram).unwrap();
        let space = mem.address_space_init(root, "memory").unwrap();
        let block = mem.ram_block(ram).unwrap();
        block.start_dirty_log(DirtyClient::Migration);
        let hooks = TcgHooks { vcpus: Weak::new(), mem: Arc::downgrade(&mem) };

        // What a virtio device does to a used ring, before and while the migration logs.
        let _ = space.write(0x1000, MemTxAttrs::UNSPECIFIED, &[1]);
        assert!(!block.get_dirty(0x1000, 1, DirtyClient::Migration));
        hooks.log_start();
        let _ = space.write(0x3000, MemTxAttrs::UNSPECIFIED, &[1]);
        assert!(block.get_dirty(0x3000, 1, DirtyClient::Migration));
        hooks.log_stop();
        block.stop_dirty_log(DirtyClient::Migration);
    }
}
