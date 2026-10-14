// SPDX-License-Identifier: GPL-2.0-or-later

//! The migration sections of an Arm board, in QEMU's stream format.
//!
//! [`virt_savevm`] registers the sections of `virt` the way QEMU 11.1 lays them out. So far
//! these are the `timer` section, RAM and `globalstate`; the CPUs, the GIC and the devices
//! are not registered yet.
//!
//! The device sections carry the state field by field, as QEMU 11.1's `VMStateDescription`
//! for it does: the PL011 UART (hw/char/pl011.c and hw/core/clock-vmstate.c), the PL031 RTC
//! (hw/rtc/pl031.c), the PL061 GPIO controller (hw/gpio/pl061.c) and the GPIO power key
//! (hw/gpio/gpio_key.c). The registers add no path prefix, as these are sysbus devices.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, PoisonError, Weak};

use ruvm_accel::tcg::TcgVcpus;
use ruvm_base::error_report;
use ruvm_hw_char::pl011::{PL011_FIFO_DEPTH, Pl011, Pl011VmState};
use ruvm_hw_core::timer::Clock;
use ruvm_hw_misc::gpio_key::GpioKey;
use ruvm_hw_misc::pl061::{Pl061, Pl061VmState};
use ruvm_hw_timer::pl031::{Pl031, Pl031VmState};
use ruvm_jit::cputlb::tlb_flush;
use ruvm_mem::{GLOBAL_DIRTY_MIGRATION, MemorySystem};
use ruvm_migration::timers::{TimersState, VMSTATE_TIMERS};
use ruvm_migration::{
    EntryInfo, GlobalState, MachineConfig, RamHooks, RamSection, RamStats, SaveVm,
};
use ruvm_vmstate::info::Timer;
use ruvm_vmstate::{VmStateDescription, VmStateField};

use crate::tcg_run::VirtTcgMachine;

/// The registered sections of a machine and the handles the migration code needs.
#[derive(Debug)]
pub struct ArmMigration {
    /// The sections.
    pub savevm: SaveVm,
    /// The counters of the `ram` section.
    pub ram_stats: Arc<RamStats>,
    /// The `globalstate` section.
    pub global_state: Arc<GlobalState>,
}

/// Registers the sections of `machine`, a virt board on TCG, for migration as machine type
/// `machine_type`.
pub fn virt_savevm(
    machine: &VirtTcgMachine,
    machine_type: &str,
    uuid: Option<[u8; 16]>,
) -> ArmMigration {
    // The virt board's pages are 4 KiB, and the smallest Arm pages 1 KiB.
    let mut savevm = SaveVm::new(MachineConfig {
        name: machine_type.to_string(),
        page_bits: 12,
        legacy_page_bits: 10,
        uuid,
    });
    let board = machine.board().lock().unwrap_or_else(PoisonError::into_inner);
    register_timer(&mut savevm, board.clock());
    let ram = RamSection::new(
        board.migratable_ram_blocks(),
        TcgHooks {
            vcpus: Arc::downgrade(machine.vcpus()),
            mem: Arc::downgrade(board.memory_system()),
        },
    );
    drop(board);
    let ram_stats = ram.stats();
    savevm.register_live(EntryInfo::new("ram", 4).instance(0), ram);
    // "dirty-bitmap": ruvm has no block dirty bitmaps to migrate.
    savevm.reserve_section_id();
    let global_state = GlobalState::register(&mut savevm);
    ArmMigration { savevm, ram_stats, global_state }
}

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

/// Registers the `timer` section over `clock`, the virtual clock. Nothing on an Arm board
/// reads `cpu_get_ticks()`, so `cpu_ticks_offset` goes out as it came in, 0 on a fresh start.
/// QEMU registers the section first, so it gets section id 0.
fn register_timer(savevm: &mut SaveVm, clock: &Arc<Clock>) {
    let ticks = Arc::new(AtomicI64::new(0));
    let (get_clock, put_clock) = (Arc::clone(clock), Arc::clone(clock));
    let get_ticks = Arc::clone(&ticks);
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_TIMERS,
        move || {
            Ok(TimersState {
                cpu_ticks_offset: get_ticks.load(Ordering::Relaxed),
                cpu_clock_offset: get_clock.get_ns(),
            })
        },
        move |s| {
            put_clock.set_ns(s.cpu_clock_offset);
            ticks.store(s.cpu_ticks_offset, Ordering::Relaxed);
            Ok(())
        },
    );
}

/// A `Clock` as `vmstate_clock` carries it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ClockVmState {
    period: u64,
    multiplier: u32,
    divider: u32,
}

/// `vmstate_muldiv`.
static VMSTATE_MULDIV: LazyLock<VmStateDescription<ClockVmState>> = LazyLock::new(|| {
    type S = ClockVmState;
    VmStateDescription::new("clock/muldiv")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &S| s.multiplier != 1 || s.divider != 1)
        .fields([
            VmStateField::scalar("multiplier", |s: &mut S| &mut s.multiplier),
            VmStateField::scalar("divider", |s: &mut S| &mut s.divider),
        ])
});

/// `vmstate_clock`.
static VMSTATE_CLOCK: LazyLock<VmStateDescription<ClockVmState>> = LazyLock::new(|| {
    type S = ClockVmState;
    VmStateDescription::new("clock")
        .version_id(0)
        .minimum_version_id(0)
        .pre_load(|s: &mut S| {
            s.multiplier = 1;
            s.divider = 1;
            0
        })
        .field(VmStateField::scalar("period", |s: &mut S| &mut s.period))
        .subsection(&VMSTATE_MULDIV)
});

/// The PL011 state with its clock, which the device keeps as a frequency and so always has
/// a multiplier and divider of 1.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Pl011Section {
    regs: Pl011VmState,
    clk: ClockVmState,
}

impl Pl011Section {
    fn new(regs: Pl011VmState) -> Self {
        let clk = ClockVmState { period: regs.clk_period, multiplier: 1, divider: 1 };
        Pl011Section { regs, clk }
    }
}

/// `vmstate_pl011_clock`, always sent since `migrate-clk` is on for every machine version ruvm
/// has.
static VMSTATE_PL011_CLOCK: LazyLock<VmStateDescription<Pl011Section>> = LazyLock::new(|| {
    type S = Pl011Section;
    VmStateDescription::new("pl011/clock")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|_: &S| true)
        .field(VmStateField::structure("clk", &VMSTATE_CLOCK, |s: &mut S| &mut s.clk))
});

/// `vmstate_pl011`. `pl011_post_load()` is [`Pl011::vmstate_load`].
static VMSTATE_PL011: LazyLock<VmStateDescription<Pl011Section>> = LazyLock::new(|| {
    type S = Pl011Section;
    VmStateDescription::new("pl011")
        .version_id(2)
        .minimum_version_id(2)
        .fields([
            VmStateField::unused(4),
            VmStateField::scalar("flags", |s: &mut S| &mut s.regs.flags),
            VmStateField::scalar("lcr", |s: &mut S| &mut s.regs.lcr),
            VmStateField::scalar("rsr", |s: &mut S| &mut s.regs.rsr),
            VmStateField::scalar("cr", |s: &mut S| &mut s.regs.cr),
            VmStateField::scalar("dmacr", |s: &mut S| &mut s.regs.dmacr),
            VmStateField::scalar("int_enabled", |s: &mut S| &mut s.regs.int_enabled),
            VmStateField::scalar("int_level", |s: &mut S| &mut s.regs.int_level),
            VmStateField::array::<u32, PL011_FIFO_DEPTH>("read_fifo", |s: &mut S| {
                &mut s.regs.read_fifo
            }),
            VmStateField::scalar("ilpr", |s: &mut S| &mut s.regs.ilpr),
            VmStateField::scalar("ibrd", |s: &mut S| &mut s.regs.ibrd),
            VmStateField::scalar("fbrd", |s: &mut S| &mut s.regs.fbrd),
            VmStateField::scalar("ifl", |s: &mut S| &mut s.regs.ifl),
            VmStateField::scalar("read_pos", |s: &mut S| &mut s.regs.read_pos),
            VmStateField::scalar("read_count", |s: &mut S| &mut s.regs.read_count),
            VmStateField::scalar("read_trigger", |s: &mut S| &mut s.regs.read_trigger),
        ])
        .subsection(&VMSTATE_PL011_CLOCK)
});

/// Registers the `pl011` section of `uart`, instance `instance` (0 for the first UART QEMU
/// creates).
pub fn register_pl011(savevm: &mut SaveVm, instance: u32, uart: &Arc<Pl011>) {
    let (get, put) = (Arc::clone(uart), Arc::clone(uart));
    savevm.register_vmsd(
        "",
        Some(instance),
        &VMSTATE_PL011,
        move || Ok(Pl011Section::new(get.vmstate_save())),
        move |s| put.vmstate_load(&Pl011VmState { clk_period: s.clk.period, ..s.regs }),
    );
}

/// `vmstate_pl031_tick_offset`.
static VMSTATE_PL031_TICK_OFFSET: LazyLock<VmStateDescription<Pl031VmState>> =
    LazyLock::new(|| {
        type S = Pl031VmState;
        VmStateDescription::new("pl031/tick-offset")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|_: &S| true)
            .post_load(|s: &mut S, _| {
                s.tick_offset_migrated = true;
                0
            })
            .field(VmStateField::scalar("tick_offset", |s: &mut S| &mut s.tick_offset))
    });

/// `vmstate_pl031`. `pl031_pre_save()` and `pl031_post_load()` are in [`Pl031::vmstate_save`]
/// and [`Pl031::vmstate_load`].
static VMSTATE_PL031: LazyLock<VmStateDescription<Pl031VmState>> = LazyLock::new(|| {
    type S = Pl031VmState;
    VmStateDescription::new("pl031")
        .version_id(1)
        .minimum_version_id(1)
        .pre_load(|s: &mut S| {
            s.tick_offset_migrated = false;
            0
        })
        .fields([
            VmStateField::scalar("tick_offset_vmstate", |s: &mut S| &mut s.tick_offset_vmstate),
            VmStateField::scalar("mr", |s: &mut S| &mut s.mr),
            VmStateField::scalar("lr", |s: &mut S| &mut s.lr),
            VmStateField::scalar("cr", |s: &mut S| &mut s.cr),
            VmStateField::scalar("im", |s: &mut S| &mut s.im),
            VmStateField::scalar("is", |s: &mut S| &mut s.is),
        ])
        .subsection(&VMSTATE_PL031_TICK_OFFSET)
});

/// Registers the `pl031` section of `rtc`. `virtual_clock` is the clock the old offset in the
/// main section is relative to.
pub fn register_pl031(savevm: &mut SaveVm, rtc: &Arc<Pl031>, virtual_clock: &Arc<Clock>) {
    let (get, put) = (Arc::clone(rtc), Arc::clone(rtc));
    let (get_clock, put_clock) = (Arc::clone(virtual_clock), Arc::clone(virtual_clock));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_PL031,
        move || Ok(get.vmstate_save(get_clock.get_ns())),
        move |s| {
            put.vmstate_load(&s, put_clock.get_ns());
            Ok(())
        },
    );
}

/// `vmstate_pl061`.
static VMSTATE_PL061: LazyLock<VmStateDescription<Pl061VmState>> = LazyLock::new(|| {
    type S = Pl061VmState;
    VmStateDescription::new("pl061").version_id(4).minimum_version_id(4).fields([
        VmStateField::scalar("locked", |s: &mut S| &mut s.locked),
        VmStateField::scalar("data", |s: &mut S| &mut s.data),
        VmStateField::scalar("old_out_data", |s: &mut S| &mut s.old_out_data),
        VmStateField::scalar("old_in_data", |s: &mut S| &mut s.old_in_data),
        VmStateField::scalar("dir", |s: &mut S| &mut s.dir),
        VmStateField::scalar("isense", |s: &mut S| &mut s.isense),
        VmStateField::scalar("ibe", |s: &mut S| &mut s.ibe),
        VmStateField::scalar("iev", |s: &mut S| &mut s.iev),
        VmStateField::scalar("im", |s: &mut S| &mut s.im),
        VmStateField::scalar("istate", |s: &mut S| &mut s.istate),
        VmStateField::scalar("afsel", |s: &mut S| &mut s.afsel),
        VmStateField::scalar("dr2r", |s: &mut S| &mut s.dr2r),
        VmStateField::scalar("dr4r", |s: &mut S| &mut s.dr4r),
        VmStateField::scalar("dr8r", |s: &mut S| &mut s.dr8r),
        VmStateField::scalar("odr", |s: &mut S| &mut s.odr),
        VmStateField::scalar("pur", |s: &mut S| &mut s.pur),
        VmStateField::scalar("pdr", |s: &mut S| &mut s.pdr),
        VmStateField::scalar("slr", |s: &mut S| &mut s.slr),
        VmStateField::scalar("den", |s: &mut S| &mut s.den),
        VmStateField::scalar("cr", |s: &mut S| &mut s.cr),
        VmStateField::scalar("amsel", |s: &mut S| &mut s.amsel).version(2),
    ])
});

/// Registers the `pl061` section of `gpio`.
pub fn register_pl061(savevm: &mut SaveVm, gpio: &Arc<Pl061>) {
    let (get, put) = (Arc::clone(gpio), Arc::clone(gpio));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_PL061,
        move || Ok(get.vmstate_save()),
        move |s| {
            put.vmstate_load(&s);
            Ok(())
        },
    );
}

/// `vmstate_gpio_key`: the release timer.
static VMSTATE_GPIO_KEY: LazyLock<VmStateDescription<i64>> = LazyLock::new(|| {
    VmStateDescription::new("gpio-key")
        .version_id(1)
        .minimum_version_id(1)
        .field(VmStateField::single("timer", &Timer, |s: &mut i64| s))
});

/// Registers the `gpio-key` section of `key`.
pub fn register_gpio_key(savevm: &mut SaveVm, key: &Arc<GpioKey>) {
    let (get, put) = (Arc::clone(key), Arc::clone(key));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_GPIO_KEY,
        move || Ok(get.vmstate_save()),
        move |t| {
            put.vmstate_load(t);
            Ok(())
        },
    );
}

#[cfg(test)]
mod tests {
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_load_state, vmstate_save_state};

    use super::*;

    fn save<T>(vmsd: &VmStateDescription<T>, mut s: T) -> Vec<u8> {
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, vmsd, &mut s).unwrap();
        f.into_inner()
    }

    fn load<T>(vmsd: &VmStateDescription<T>, bytes: &[u8], into: &mut T) {
        let mut b = bytes.to_vec();
        // The byte after the section, which is not a subsection.
        b.push(0);
        vmstate_load_state(&mut StreamReader::new(&b), vmsd, into, vmsd.version_id).unwrap();
    }

    fn be32(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_be_bytes()).collect()
    }

    fn subsection(name: &str, version: u32, body: &[u8]) -> Vec<u8> {
        let mut b = vec![0x05, name.len() as u8];
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&version.to_be_bytes());
        b.extend_from_slice(body);
        b
    }

    /// A PL011 out of reset, as QEMU's `virt` sends it: no clock is connected.
    #[test]
    fn pl011_layout_matches_qemu() {
        let uart = Pl011::new(None);
        let bytes = save(&VMSTATE_PL011, Pl011Section::new(uart.vmstate_save()));
        let mut want = be32(&[0, 0x90, 0, 0, 0x300, 0, 0, 0]);
        want.extend(be32(&[0; PL011_FIFO_DEPTH]));
        want.extend(be32(&[0, 0, 0, 0x12, 0, 0, 1]));
        want.extend(subsection("pl011/clock", 1, &0u64.to_be_bytes()));
        assert_eq!(bytes, want);

        let mut back = Pl011Section::new(Pl011VmState::default());
        back.clk.multiplier = 7;
        load(&VMSTATE_PL011, &bytes, &mut back);
        assert_eq!(back, Pl011Section::new(uart.vmstate_save()));
    }

    #[test]
    fn pl011_load_checks_the_fifo() {
        let uart = Pl011::new(None);
        let bad = Pl011VmState { read_pos: 16, ..uart.vmstate_save() };
        assert!(uart.vmstate_load(&bad).is_err());
        let bad = Pl011VmState { read_count: 17, ..uart.vmstate_save() };
        assert!(uart.vmstate_load(&bad).is_err());
        // A character of a disabled FIFO moves to the front.
        let mut v = Pl011VmState { read_pos: 3, read_count: 1, ..uart.vmstate_save() };
        v.read_fifo[3] = 0x41;
        uart.vmstate_load(&v).unwrap();
        let back = uart.vmstate_save();
        assert_eq!((back.read_pos, back.read_fifo[0]), (0, 0x41));
    }

    #[test]
    fn pl031_layout_matches_qemu() {
        let v = Pl031VmState {
            tick_offset_vmstate: 0x6acf133a,
            tick_offset: 0x6acf1000,
            mr: 5,
            tick_offset_migrated: true,
            ..Pl031VmState::default()
        };
        let bytes = save(&VMSTATE_PL031, v.clone());
        let mut want = be32(&[0x6acf133a, 5, 0, 0, 0, 0]);
        want.extend(subsection("pl031/tick-offset", 1, &be32(&[0x6acf1000])));
        assert_eq!(bytes, want);

        let mut back = Pl031VmState { tick_offset_migrated: true, ..Pl031VmState::default() };
        load(&VMSTATE_PL031, &bytes, &mut back);
        assert_eq!(back, v);
        // Without the subsection the destination converts the old offset.
        let mut back = Pl031VmState { tick_offset_migrated: true, ..Pl031VmState::default() };
        load(&VMSTATE_PL031, &bytes[..24], &mut back);
        assert!(!back.tick_offset_migrated);
    }

    #[test]
    fn pl031_offset_round_trip() {
        use std::time::{Duration, UNIX_EPOCH};

        let rtc_clock = Clock::manual(ruvm_base::ClockType::Host);
        rtc_clock.advance_to(5_000_000_000);
        let rtc = Pl031::new(rtc_clock, UNIX_EPOCH + Duration::from_secs(1000));
        // The virtual clock is 2 s behind the RTC clock.
        let v = rtc.vmstate_save(3_000_000_000);
        assert_eq!(v.tick_offset_vmstate, v.tick_offset + 2);
        let old = Pl031VmState { tick_offset_migrated: false, tick_offset: 0, ..v.clone() };
        rtc.vmstate_load(&old, 3_000_000_000);
        assert_eq!(rtc.tick_offset(), v.tick_offset);
    }

    #[test]
    fn pl061_layout_matches_qemu() {
        let gpio =
            Pl061::new(ruvm_hw_misc::pl061::Pl061Props { pullups: 0, pulldowns: 0xff }).unwrap();
        let bytes = save(&VMSTATE_PL061, gpio.vmstate_save());
        let mut want = vec![0; 21];
        (want[0], want[11], want[19]) = (1, 0xff, 0xff);
        assert_eq!(bytes, be32(&want));
        let mut back = Pl061VmState::default();
        load(&VMSTATE_PL061, &bytes, &mut back);
        assert_eq!(back, gpio.vmstate_save());
    }

    #[test]
    fn gpio_key_layout_matches_qemu() {
        assert_eq!(save(&VMSTATE_GPIO_KEY, -1), [0xff; 8]);
        let mut back = 0;
        load(&VMSTATE_GPIO_KEY, &1234i64.to_be_bytes(), &mut back);
        assert_eq!(back, 1234);
    }
}
