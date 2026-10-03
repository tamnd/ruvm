// SPDX-License-Identifier: GPL-2.0-or-later

//! The TCG plugin host, `plugins/` in QEMU: loading plugins written against QEMU 11.1's
//! `include/plugins/qemu-plugin.h` unmodified, and the 65 `qemu_plugin_*` functions they call.
//!
//! A front end uses it like `vl.c` and `accel/tcg` do:
//!
//! 1. Every `-plugin` option goes through [`qemu_plugin_opt_parse`].
//! 2. Before the vCPUs exist, [`qemu_plugin_load_list`] opens each plugin with `dlopen()`,
//!    checks `qemu_plugin_version` and calls `qemu_plugin_install()`.
//! 3. [`attach`] connects the loaded plugins to a [`Jit`](ruvm_jit::Jit). With no plugin
//!    loaded it does nothing, and the runtime emits no instrumentation at all.
//! 4. [`qemu_plugin_atexit_cb`] runs the `atexit` callbacks when the machine stops.
//!
//! The runtime side, instrumenting blocks and calling back at run time, is
//! [`ruvm_jit::plugin`]. The target side, registers, disassembly and symbols, comes from a
//! [`PluginTarget`].
//!
//! # Making the API visible to plugins
//!
//! The API functions are `#[no_mangle] extern "C-unwind"` functions in this crate. A plugin
//! leaves them undefined and the dynamic linker resolves them against the program that loads
//! it, so the program must put them in its dynamic symbol table:
//!
//! - On Linux and the BSDs, link the program with `-rdynamic` (`-C link-arg=-rdynamic`, or
//!   `cargo:rustc-link-arg-bins=-rdynamic` in a build script).
//! - On macOS, link with `-Wl,-export_dynamic`, and build plugins with
//!   `-undefined dynamic_lookup`, which is what QEMU's own build does there.
//!
//! The functions are referenced from the load path, so the linker keeps them even though
//! nothing in Rust calls them. This crate's build script does the above for its own tests.
//!
//! # Differences from QEMU
//!
//! - Only Unix hosts load plugins. On Windows, which needs QEMU's import library trick
//!   (`win32_linker.c`), [`qemu_plugin_load_list`] fails for any plugin.
//! - `qemu_plugin_set_pc()` leaves the callback by unwinding through the plugin's frames, where
//!   QEMU uses `siglongjmp()`. That needs the plugin to have unwind tables, which is the default
//!   for C on x86-64 and AArch64 hosts, and a build with `panic = "unwind"`. In a
//!   `panic = "abort"` build it aborts.
//! - `current_cpu` is only known while a callback runs. Functions that need it and are called
//!   elsewhere, on a vCPU thread outside a callback, panic like QEMU's assertion; uninstall and
//!   reset then remove the callbacks synchronously.
//! - The event mask is one for all vCPUs, where QEMU keeps a copy per vCPU that it updates
//!   asynchronously. The difference is only when a new registration is seen.
//! - The host address of an instruction is its `ram_addr` with bit 63 set, not a pointer into
//!   guest RAM. QEMU only promises it is a proxy for the address space and physical address.
//! - `qemu_plugin_hwaddr_is_io()` returns false for a NULL handle instead of crashing.
//! - Scoreboards are allocated with eight byte alignment. An inline operation on a field that
//!   is not eight byte aligned still works, but is not a single access.
//! - `qemu_plugin_outs()` goes to the sink given to [`set_log`]; there is none by default,
//!   which matches QEMU without `-d plugin`.
//! - `qemu_plugin_insn_disas()`, `qemu_plugin_insn_symbol()` and the register functions use
//!   the [`PluginTarget`]; the defaults give an empty string, no symbol and no registers.
//! - Callback lists are copied before they are walked, in place of RCU. A callback that
//!   unregisters another one of the same event during the walk does not stop it from running
//!   this once.
//! - Strings handed to plugins (register names, device names, symbols) are interned for the
//!   life of the process, like `g_intern_string()`.
//! - QEMU links glib; this crate does not. The few glib functions the API needs
//!   (`g_array_new()`, `g_byte_array_append()`, `g_strdup()` and so on) are looked up in the
//!   process and then in the loaded plugins, which all link glib, and that copy of glib is kept
//!   loaded from then on. A plugin built against a different glib than the one found first
//!   would get arrays from the wrong allocator.
//! - `qemu_plugin_write_memory_vaddr()` writes through the address space like any other write,
//!   so it cannot change ROM, where QEMU's debug access can.
//! - The host only does system emulation: `qemu_plugin_path_to_binary()` is NULL and the code
//!   ranges are 0, as in QEMU's system mode. The syscall callbacks run when a front end calls
//!   [`ruvm_jit::plugin::vcpu_syscall`] and its siblings.

use std::fmt;

use ruvm_jit::cputlb::tlb_plugin_lookup;
use ruvm_jit::{Cpu, MmuAccessType, Ra};

mod opts;

#[cfg(unix)]
mod api;
#[cfg(unix)]
mod host;
#[cfg(unix)]
mod sys;

pub use opts::{OptError, PLUGIN_HELP, PluginDesc, qemu_plugin_opt_parse};

#[cfg(unix)]
pub use host::{attach, loaded, qemu_plugin_atexit_cb, qemu_plugin_load_list, set_log};

/// `QEMU_PLUGIN_VERSION`: the API version this host implements.
pub const QEMU_PLUGIN_VERSION: i32 = 7;

/// `QEMU_PLUGIN_MIN_VERSION`: the oldest API version a plugin may ask for.
pub const QEMU_PLUGIN_MIN_VERSION: i32 = 7;

/// Where `qemu_plugin_outs()` writes.
pub type LogSink = std::sync::Arc<dyn Fn(&str) + Send + Sync>;

/// What `qemu_plugin_install()` is told about the emulator, `qemu_info_t`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QemuInfo {
    /// The target, for example `x86_64` or `aarch64`.
    pub target_name: String,
    /// Whether this is a full system emulation.
    pub system_emulation: bool,
    /// The initial number of vCPUs.
    pub smp_vcpus: i32,
    /// The most vCPUs there can be.
    pub max_vcpus: i32,
}

/// A register as the gdbstub describes it, `GDBRegDesc`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GdbReg {
    /// The gdb register number.
    pub num: u32,
    /// The name; registers without one are not shown to plugins.
    pub name: Option<String>,
    /// The name of the feature that has it.
    pub feature: String,
}

/// What the plugin API needs from the guest target.
pub trait PluginTarget: Send + Sync + fmt::Debug {
    /// `gdb_get_register_list()`.
    fn registers(&self, cpu: &Cpu<'_>) -> Vec<GdbReg> {
        let _ = cpu;
        Vec::new()
    }

    /// `gdb_read_register()`: append register `reg` to `buf` and return its size, 0 if there
    /// is no such register.
    fn read_register(&self, cpu: &mut Cpu<'_>, reg: u32, buf: &mut Vec<u8>) -> usize {
        let _ = (cpu, reg, buf);
        0
    }

    /// `gdb_write_register()`: set register `reg` from the start of `buf` and return its size,
    /// 0 if there is no such register.
    fn write_register(&self, cpu: &mut Cpu<'_>, reg: u32, buf: &[u8]) -> usize {
        let _ = (cpu, reg, buf);
        0
    }

    /// `plugin_disas()`: the instruction at `vaddr` made of `bytes`, or an empty string when
    /// there is no disassembler.
    fn disas(&self, cpu: &Cpu<'_>, vaddr: u64, bytes: &[u8]) -> String {
        let _ = (cpu, vaddr, bytes);
        String::new()
    }

    /// `lookup_symbol()`: the symbol `vaddr` is in.
    fn symbol(&self, vaddr: u64) -> Option<String> {
        let _ = vaddr;
        None
    }

    /// `cpu_get_phys_page_debug()`: the physical address of the page that holds `addr`. The
    /// default asks the softmmu TLB, filling it without faulting when the page is not there.
    fn phys_page_debug(&self, cpu: &mut Cpu<'_>, addr: u64) -> Option<u64> {
        phys_page_debug_from_tlb(cpu, addr)
    }

    /// `qemu_clock_advance_virtual_time()`, for `qemu_plugin_update_ns()`.
    fn advance_virtual_time(&self, ns: i64) {
        let _ = ns;
    }
}

/// The default [`PluginTarget::phys_page_debug`]: look `addr` up in the softmmu TLB of the
/// current MMU index, and probe the target's page tables with `tlb_fill` when it is missing.
pub fn phys_page_debug_from_tlb(cpu: &mut Cpu<'_>, addr: u64) -> Option<u64> {
    let ops = cpu.ops();
    let mmu_idx = ops.mmu_index(cpu, false);
    let page = addr & cpu.core.jit().page_mask();
    if let Some((phys, _)) = tlb_plugin_lookup(cpu, page, mmu_idx, false) {
        return Some(phys);
    }
    match ops.tlb_fill(cpu, page, 1, MmuAccessType::DataLoad, mmu_idx, true, Ra::None) {
        Ok(true) => tlb_plugin_lookup(cpu, page, mmu_idx, false).map(|(phys, _)| phys),
        _ => None,
    }
}

/// [`qemu_plugin_load_list`] for hosts that cannot load plugins: an error for the first one.
#[cfg(not(unix))]
#[allow(clippy::ptr_arg, reason = "the same signature as on Unix, where plugins are removed")]
pub fn qemu_plugin_load_list(
    head: &mut Vec<PluginDesc>,
    info: &QemuInfo,
) -> Result<(), ruvm_base::Error> {
    let _ = info;
    match head.first() {
        Some(desc) => Err(ruvm_base::Error::generic(format!(
            "Could not load plugin {}: plugins are not supported on this host",
            desc.path
        ))),
        None => Ok(()),
    }
}

/// [`attach`] for hosts that cannot load plugins: there is nothing to attach.
#[cfg(not(unix))]
pub fn attach(jit: &std::sync::Arc<ruvm_jit::Jit>, target: std::sync::Arc<dyn PluginTarget>) {
    let _ = (jit, target);
}

/// [`loaded`] for hosts that cannot load plugins.
#[cfg(not(unix))]
pub fn loaded() -> usize {
    0
}

/// [`qemu_plugin_atexit_cb`] for hosts that cannot load plugins.
#[cfg(not(unix))]
pub fn qemu_plugin_atexit_cb() {}

/// [`set_log`] for hosts that cannot load plugins.
#[cfg(not(unix))]
pub fn set_log(sink: Option<LogSink>) {
    let _ = sink;
}
