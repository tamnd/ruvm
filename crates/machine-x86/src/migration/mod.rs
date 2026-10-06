// SPDX-License-Identifier: GPL-2.0-or-later

//! Migration of a q35 machine on TCG, in QEMU's stream format.
//!
//! [`q35_savevm`] registers the sections of the machine the way QEMU 11.1 lays them out for
//! `-M pc-q35-11.1`: the vCPUs (`cpu_common` and `cpu`), RAM, `globalstate`, and the board's
//! devices. The vCPUs and RAM are carried for real. The device sections (APIC, PIC, IOAPIC, RTC,
//! ICH9, fw_cfg and the rest) are parsed from an incoming stream and dropped, and not sent; a
//! guest that relies on interrupt controller state does not survive the hop yet.

mod cpu;
mod skip;

use std::sync::{Arc, PoisonError, Weak};

use ruvm_accel::tcg::TcgVcpus;
use ruvm_jit::cputlb::tlb_flush;
use ruvm_migration::{
    EntryInfo, GlobalState, MachineConfig, RamHooks, RamSection, RamStats, SaveVm,
};

use crate::board::X86Board;
use crate::tcg_run::TcgMachine;

/// The RAM hooks of a TCG machine. The dirty bitmaps are set by the TLB's not-dirty path, so
/// after bits are taken out the TLBs must forget which pages were already written.
#[derive(Debug)]
struct TcgHooks {
    vcpus: Weak<TcgVcpus>,
}

impl RamHooks for TcgHooks {
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
pub struct Q35Migration {
    /// The sections.
    pub savevm: SaveVm,
    /// The counters of the `ram` section.
    pub ram_stats: Arc<RamStats>,
    /// The `globalstate` section.
    pub global_state: Arc<GlobalState>,
}

/// Registers the sections of `machine`, a q35 board, for migration as machine type
/// `machine_type`.
pub fn q35_savevm(
    machine: &TcgMachine,
    machine_type: &str,
    uuid: Option<[u8; 16]>,
) -> Result<Q35Migration, String> {
    let board = machine.board().lock().unwrap_or_else(PoisonError::into_inner);
    let X86Board::Q35(q35, _) = &*board else {
        return Err("migration is only supported on the q35 machine".to_string());
    };
    let blocks = q35.migratable_ram_blocks();
    let max_cpus = q35.max_cpus() as usize;
    let apic_ids = board.apic_ids();
    drop(board);

    let mut savevm = SaveVm::new(MachineConfig {
        name: machine_type.to_string(),
        page_bits: 12,
        legacy_page_bits: 12,
        uuid,
    });

    // The sections in the order QEMU registers them, so they get QEMU's section ids: the
    // timers, then RAM and the dirty bitmaps from migration_object_init(), then each vCPU with
    // its APIC (the first APIC brings kvmvapic along), then the board.
    skip::register_timer(&mut savevm);
    let vcpus = machine.vcpus();
    let ram = RamSection::new(blocks, TcgHooks { vcpus: Arc::downgrade(vcpus) });
    let ram_stats = ram.stats();
    savevm.register_live(EntryInfo::new("ram", 4).instance(0), ram);
    // "dirty-bitmap": ruvm has no block dirty bitmaps to migrate.
    savevm.reserve_section_id();

    for (n, shared) in vcpus.cpus().iter().enumerate() {
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
            skip::register_vapic(&mut savevm);
        }
        if let Some(&id) = apic_ids.get(n) {
            skip::register_apic(&mut savevm, id);
        }
    }
    skip::register_board(&mut savevm, max_cpus);
    let global_state = GlobalState::register(&mut savevm);
    Ok(Q35Migration { savevm, ram_stats, global_state })
}
