// SPDX-License-Identifier: GPL-2.0-or-later

//! The 65 functions of `qemu-plugin.h`, `plugins/api.c`, `plugins/api-system.c` and the
//! registration functions of `plugins/core.c`, exported under their C names.
//!
//! They are `extern "C-unwind"` so that `qemu_plugin_set_pc()` and a failed assertion can
//! unwind back through the plugin to the callback that called it.

use std::cell::Cell;
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::panic::resume_unwind;

use ruvm_base::report::error_report;
use ruvm_jit::Cpu;
use ruvm_jit::cputlb::tlb_plugin_lookup;
use ruvm_jit::plugin::{CbFlags, MEM_W, PluginInsn};
use ruvm_mem::{MemTxAttrs, MemTxResult};

use crate::host::{
    self, DisconCb, Entry, Func, HostCb, MemCb, SyscallCb, SyscallFilterCb, SyscallRetCb,
    TbTransCb, UdataCb, VcpuUdataCb, ev,
};
use crate::sys::{self, GArray};

/// `qemu_plugin_u64`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PluginU64 {
    score: *mut c_void,
    offset: usize,
}

/// `struct { uint64_t low; uint64_t high; } u128`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct U128 {
    low: u64,
    high: u64,
}

/// The `data` union of `qemu_plugin_mem_value`.
#[repr(C)]
#[derive(Clone, Copy)]
union MemData {
    u8: u8,
    u16: u16,
    u32: u32,
    u64: u64,
    u128: U128,
}

/// `qemu_plugin_mem_value`.
#[repr(C)]
#[derive(Clone, Copy)]
struct MemValue {
    ty: c_uint,
    data: MemData,
}

/// `qemu_plugin_reg_descriptor`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RegDesc {
    handle: *mut c_void,
    name: *const c_char,
    feature: *const c_char,
    is_readonly: bool,
}

/// `MO_SIZE`, `MO_SIGN` and `MO_BE` as `ruvm-jit-core` defines them.
const MO_SIZE: u32 = 7;
const MO_SIGN: u32 = 8;
const MO_BSWAP: u32 = 0x10;
const MO_BE: u32 = 0x10;

/// The `QEMU_PLUGIN_HWADDR_OPERATION_*` results.
const HWADDR_OK: c_uint = 0;
const HWADDR_ERROR: c_uint = 1;
const HWADDR_DEVICE_ERROR: c_uint = 2;
const HWADDR_ACCESS_DENIED: c_uint = 3;
const HWADDR_INVALID_ADDRESS: c_uint = 4;

// Opaque handles: plugins only ever pass these back, so they are addresses of statics.
static TB_TOKEN: u8 = 0;
static INSN_TOKENS: [u8; 16] = [0; 16];
static HWADDR_TOKEN: u8 = 0;
static TIME_TOKEN: u8 = 0;

/// The handle of the block being translated.
pub(crate) fn tb_handle() -> *mut c_void {
    (&raw const TB_TOKEN).cast_mut().cast()
}

fn insn_handle(idx: usize) -> *mut c_void {
    (&raw const INSN_TOKENS).cast::<u8>().cast_mut().wrapping_add(idx * 16).cast()
}

fn insn_index(insn: *const c_void) -> usize {
    let base = (&raw const INSN_TOKENS) as usize;
    (insn as usize).wrapping_sub(base) / 16
}

/// Run `f` on instruction `insn` of the block being translated.
fn with_insn<R>(insn: *const c_void, f: impl FnOnce(&mut PluginInsn, bool) -> R) -> R {
    let idx = insn_index(insn);
    host::with_tb(|tb| {
        let mem_only = tb.mem_only;
        match tb.insns.get_mut(idx) {
            Some(i) => f(i, mem_only),
            None => panic!("plugin: invalid instruction handle {insn:p}"),
        }
    })
}

fn entry(e: PluginU64) -> Entry {
    Entry { sb: host::scoreboard(e.score as usize), offset: e.offset }
}

fn inline_cb(op: c_uint, e: PluginU64, imm: u64) -> HostCb {
    match op {
        0 => HostCb::Inline { store: false, entry: entry(e), imm },
        1 => HostCb::Inline { store: true, entry: entry(e), imm },
        _ => panic!("code should not be reached"),
    }
}

thread_local! {
    /// `hwaddr_info`: the physical address, whether it is IO, and the device name.
    static HWADDR: Cell<(u64, bool, *const c_char)> =
        const { Cell::new((0, false, std::ptr::null())) };
}

/// Reference every API function from the load path, so that the linker keeps them in the
/// program for plugins to find.
pub(crate) fn keep_alive() {
    let table: [*const (); 65] = [
        qemu_plugin_uninstall as *const (),
        qemu_plugin_reset as *const (),
        qemu_plugin_register_vcpu_init_cb as *const (),
        qemu_plugin_register_vcpu_exit_cb as *const (),
        qemu_plugin_register_vcpu_idle_cb as *const (),
        qemu_plugin_register_vcpu_resume_cb as *const (),
        qemu_plugin_register_vcpu_discon_cb as *const (),
        qemu_plugin_register_vcpu_tb_trans_cb as *const (),
        qemu_plugin_register_vcpu_tb_exec_cb as *const (),
        qemu_plugin_register_vcpu_tb_exec_cond_cb as *const (),
        qemu_plugin_register_vcpu_tb_exec_inline_per_vcpu as *const (),
        qemu_plugin_register_vcpu_insn_exec_cb as *const (),
        qemu_plugin_register_vcpu_insn_exec_cond_cb as *const (),
        qemu_plugin_register_vcpu_insn_exec_inline_per_vcpu as *const (),
        qemu_plugin_tb_n_insns as *const (),
        qemu_plugin_tb_vaddr as *const (),
        qemu_plugin_tb_get_insn as *const (),
        qemu_plugin_insn_data as *const (),
        qemu_plugin_insn_size as *const (),
        qemu_plugin_insn_vaddr as *const (),
        qemu_plugin_insn_haddr as *const (),
        qemu_plugin_mem_size_shift as *const (),
        qemu_plugin_mem_is_sign_extended as *const (),
        qemu_plugin_mem_is_big_endian as *const (),
        qemu_plugin_mem_is_store as *const (),
        qemu_plugin_mem_get_value as *const (),
        qemu_plugin_get_hwaddr as *const (),
        qemu_plugin_hwaddr_is_io as *const (),
        qemu_plugin_hwaddr_phys_addr as *const (),
        qemu_plugin_hwaddr_device_name as *const (),
        qemu_plugin_register_vcpu_mem_cb as *const (),
        qemu_plugin_register_vcpu_mem_inline_per_vcpu as *const (),
        qemu_plugin_request_time_control as *const (),
        qemu_plugin_update_ns as *const (),
        qemu_plugin_register_vcpu_syscall_cb as *const (),
        qemu_plugin_register_vcpu_syscall_filter_cb as *const (),
        qemu_plugin_register_vcpu_syscall_ret_cb as *const (),
        qemu_plugin_insn_disas as *const (),
        qemu_plugin_insn_symbol as *const (),
        qemu_plugin_vcpu_for_each as *const (),
        qemu_plugin_register_flush_cb as *const (),
        qemu_plugin_register_atexit_cb as *const (),
        qemu_plugin_num_vcpus as *const (),
        qemu_plugin_outs as *const (),
        qemu_plugin_bool_parse as *const (),
        qemu_plugin_path_to_binary as *const (),
        qemu_plugin_start_code as *const (),
        qemu_plugin_end_code as *const (),
        qemu_plugin_entry_code as *const (),
        qemu_plugin_get_registers as *const (),
        qemu_plugin_read_register as *const (),
        qemu_plugin_write_register as *const (),
        qemu_plugin_set_pc as *const (),
        qemu_plugin_read_memory_vaddr as *const (),
        qemu_plugin_write_memory_vaddr as *const (),
        qemu_plugin_read_memory_hwaddr as *const (),
        qemu_plugin_write_memory_hwaddr as *const (),
        qemu_plugin_translate_vaddr as *const (),
        qemu_plugin_scoreboard_new as *const (),
        qemu_plugin_scoreboard_free as *const (),
        qemu_plugin_scoreboard_find as *const (),
        qemu_plugin_u64_add as *const (),
        qemu_plugin_u64_get as *const (),
        qemu_plugin_u64_set as *const (),
        qemu_plugin_u64_sum as *const (),
    ];
    std::hint::black_box(&table);
}

// Install and uninstall.

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_uninstall(id: u64, cb: Option<UdataCb>, userdata: *mut c_void) {
    host::reset_uninstall(id, cb, userdata as usize, false);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_reset(id: u64, cb: Option<UdataCb>, userdata: *mut c_void) {
    host::reset_uninstall(id, cb, userdata as usize, true);
}

// Event registration.

fn reg_vcpu(id: u64, e: usize, cb: Option<VcpuUdataCb>, userdata: *mut c_void) {
    host::register_cb(id, e, cb.map(Func::VcpuUdata), userdata as usize);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_init_cb(
    id: u64,
    cb: Option<VcpuUdataCb>,
    userdata: *mut c_void,
) {
    reg_vcpu(id, ev::VCPU_INIT, cb, userdata);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_exit_cb(
    id: u64,
    cb: Option<VcpuUdataCb>,
    userdata: *mut c_void,
) {
    reg_vcpu(id, ev::VCPU_EXIT, cb, userdata);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_idle_cb(
    id: u64,
    cb: Option<VcpuUdataCb>,
    userdata: *mut c_void,
) {
    reg_vcpu(id, ev::VCPU_IDLE, cb, userdata);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_resume_cb(
    id: u64,
    cb: Option<VcpuUdataCb>,
    userdata: *mut c_void,
) {
    reg_vcpu(id, ev::VCPU_RESUME, cb, userdata);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_discon_cb(
    id: u64,
    ty: c_int,
    cb: Option<DisconCb>,
    userdata: *mut c_void,
) {
    let ty = ty as u32;
    let u = userdata as usize;
    for (bit, e) in [(1, ev::VCPU_INTERRUPT), (2, ev::VCPU_EXCEPTION), (4, ev::VCPU_HOSTCALL)] {
        if ty & bit != 0 {
            host::register_cb(id, e, cb.map(Func::Discon), u);
        }
    }
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_tb_trans_cb(
    id: u64,
    cb: Option<TbTransCb>,
    userdata: *mut c_void,
) {
    host::register_cb(id, ev::VCPU_TB_TRANS, cb.map(Func::TbTrans), userdata as usize);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_syscall_cb(
    id: u64,
    cb: Option<SyscallCb>,
    userdata: *mut c_void,
) {
    host::register_cb(id, ev::VCPU_SYSCALL, cb.map(Func::Syscall), userdata as usize);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_syscall_filter_cb(
    id: u64,
    cb: Option<SyscallFilterCb>,
    userdata: *mut c_void,
) {
    let f = cb.map(Func::SyscallFilter);
    host::register_cb(id, ev::VCPU_SYSCALL_FILTER, f, userdata as usize);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_syscall_ret_cb(
    id: u64,
    cb: Option<SyscallRetCb>,
    userdata: *mut c_void,
) {
    host::register_cb(id, ev::VCPU_SYSCALL_RET, cb.map(Func::SyscallRet), userdata as usize);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_flush_cb(
    id: u64,
    cb: Option<UdataCb>,
    userdata: *mut c_void,
) {
    host::register_cb(id, ev::FLUSH, cb.map(Func::Udata), userdata as usize);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_atexit_cb(
    id: u64,
    cb: Option<UdataCb>,
    userdata: *mut c_void,
) {
    host::register_cb(id, ev::ATEXIT, cb.map(Func::Udata), userdata as usize);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_vcpu_for_each(
    id: u64,
    cb: Option<VcpuUdataCb>,
    userdata: *mut c_void,
) {
    if let Some(cb) = cb {
        host::vcpu_for_each(id, cb, userdata as usize);
    }
}

// Callbacks on translated code.

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_tb_exec_cb(
    _tb: *mut c_void,
    cb: Option<VcpuUdataCb>,
    flags: c_uint,
    userdata: *mut c_void,
) {
    let Some(f) = cb else { return };
    host::with_tb(|tb| {
        if !tb.mem_only {
            let cb = HostCb::Regular { f, udata: userdata as usize };
            tb.cbs.push(cb.dyn_cb(CbFlags::from_raw(flags), 0));
        }
    });
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_tb_exec_cond_cb(
    tb: *mut c_void,
    cb: Option<VcpuUdataCb>,
    flags: c_uint,
    cond: c_uint,
    entry_: PluginU64,
    imm: u64,
    userdata: *mut c_void,
) {
    if cond == 0 || host::with_tb(|t| t.mem_only) {
        return;
    }
    if cond == 1 {
        qemu_plugin_register_vcpu_tb_exec_cb(tb, cb, flags, userdata);
        return;
    }
    let Some(f) = cb else { return };
    let cb = HostCb::Cond { f, udata: userdata as usize, cond, entry: entry(entry_), imm };
    host::with_tb(|t| t.cbs.push(cb.dyn_cb(CbFlags::from_raw(flags), 0)));
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_tb_exec_inline_per_vcpu(
    _tb: *mut c_void,
    op: c_uint,
    entry_: PluginU64,
    imm: u64,
) {
    if host::with_tb(|t| t.mem_only) {
        return;
    }
    let cb = inline_cb(op, entry_, imm);
    host::with_tb(|t| t.cbs.push(cb.dyn_cb(CbFlags::NoRegs, 0)));
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_insn_exec_cb(
    insn: *mut c_void,
    cb: Option<VcpuUdataCb>,
    flags: c_uint,
    userdata: *mut c_void,
) {
    let Some(f) = cb else { return };
    with_insn(insn, |i, mem_only| {
        if !mem_only {
            let cb = HostCb::Regular { f, udata: userdata as usize };
            i.insn_cbs.push(cb.dyn_cb(CbFlags::from_raw(flags), 0));
        }
    });
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_insn_exec_cond_cb(
    insn: *mut c_void,
    cb: Option<VcpuUdataCb>,
    flags: c_uint,
    cond: c_uint,
    entry_: PluginU64,
    imm: u64,
    userdata: *mut c_void,
) {
    if cond == 0 || host::with_tb(|t| t.mem_only) {
        return;
    }
    if cond == 1 {
        qemu_plugin_register_vcpu_insn_exec_cb(insn, cb, flags, userdata);
        return;
    }
    let Some(f) = cb else { return };
    let cb = HostCb::Cond { f, udata: userdata as usize, cond, entry: entry(entry_), imm };
    with_insn(insn, |i, _| i.insn_cbs.push(cb.dyn_cb(CbFlags::from_raw(flags), 0)));
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_insn_exec_inline_per_vcpu(
    insn: *mut c_void,
    op: c_uint,
    entry_: PluginU64,
    imm: u64,
) {
    if host::with_tb(|t| t.mem_only) {
        return;
    }
    let cb = inline_cb(op, entry_, imm);
    with_insn(insn, |i, _| i.insn_cbs.push(cb.dyn_cb(CbFlags::NoRegs, 0)));
}

// Memory instrumentation is always planted, as it does not finish until after the access.

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_mem_cb(
    insn: *mut c_void,
    cb: Option<MemCb>,
    flags: c_uint,
    rw: c_uint,
    userdata: *mut c_void,
) {
    let Some(f) = cb else { return };
    let cb = HostCb::Mem { f, udata: userdata as usize };
    with_insn(insn, |i, _| i.mem_cbs.push(cb.dyn_cb(CbFlags::from_raw(flags), rw)));
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_register_vcpu_mem_inline_per_vcpu(
    insn: *mut c_void,
    rw: c_uint,
    op: c_uint,
    entry_: PluginU64,
    imm: u64,
) {
    let cb = inline_cb(op, entry_, imm);
    with_insn(insn, |i, _| i.mem_cbs.push(cb.dyn_cb(CbFlags::NoRegs, rw)));
}

// Blocks and instructions.

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_tb_n_insns(_tb: *const c_void) -> usize {
    host::with_tb(|t| t.insns.len())
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_tb_vaddr(_tb: *const c_void) -> u64 {
    host::with_tb(|t| t.vaddr)
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_tb_get_insn(_tb: *const c_void, idx: usize) -> *mut c_void {
    if idx >= host::with_tb(|t| t.insns.len()) { std::ptr::null_mut() } else { insn_handle(idx) }
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_insn_data(
    insn: *const c_void,
    dest: *mut c_void,
    len: usize,
) -> usize {
    with_insn(insn, |i, _| {
        let len = len.min(usize::try_from(i.len).unwrap_or(usize::MAX));
        match &i.data {
            Some(d) if d.len() >= len => {
                sys::copy_out(dest, &d[..len]);
                len
            }
            _ => 0,
        }
    })
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_insn_size(insn: *const c_void) -> usize {
    with_insn(insn, |i, _| usize::try_from(i.len).unwrap_or(usize::MAX))
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_insn_vaddr(insn: *const c_void) -> u64 {
    with_insn(insn, |i, _| i.vaddr)
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_insn_haddr(insn: *const c_void) -> *mut c_void {
    with_insn(insn, |i, _| match i.haddr {
        Some(h) => (h | (1 << 63)) as usize as *mut c_void,
        None => std::ptr::null_mut(),
    })
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_insn_disas(insn: *const c_void) -> *mut c_char {
    let (vaddr, bytes) = with_insn(insn, |i, _| (i.vaddr, i.data.clone().unwrap_or_default()));
    let target = host::target();
    let s = sys::with_cpu(|cpu| target.disas(cpu, vaddr, &bytes));
    match host::glib() {
        Some(g) => g.strdup(&s),
        None => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_insn_symbol(insn: *const c_void) -> *const c_char {
    let vaddr = with_insn(insn, |i, _| i.vaddr);
    match host::target().symbol(vaddr) {
        Some(s) if !s.is_empty() => host::intern(&s),
        _ => std::ptr::null(),
    }
}

// Memory accesses.

fn memop(info: u32) -> u32 {
    info >> 4
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_mem_size_shift(info: u32) -> c_uint {
    memop(info) & MO_SIZE
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_mem_is_sign_extended(info: u32) -> bool {
    memop(info) & MO_SIGN != 0
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_mem_is_big_endian(info: u32) -> bool {
    memop(info) & MO_BSWAP == MO_BE
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_mem_is_store(info: u32) -> bool {
    (info >> 16) & MEM_W != 0
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_mem_get_value(info: u32) -> MemValue {
    let (low, high) = host::mem_value();
    let (ty, data) = match qemu_plugin_mem_size_shift(info) {
        0 => (0, MemData { u8: low as u8 }),
        1 => (1, MemData { u16: low as u16 }),
        2 => (2, MemData { u32: low as u32 }),
        3 => (3, MemData { u64: low }),
        4 => (4, MemData { u128: U128 { low, high } }),
        _ => panic!("code should not be reached"),
    };
    MemValue { ty, data }
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_get_hwaddr(info: u32, vaddr: u64) -> *mut c_void {
    let mmu_idx = (info & 15) as usize;
    let is_store = (info >> 16) & MEM_W != 0;
    let found = sys::with_cpu(|cpu| {
        let (phys, is_io) = tlb_plugin_lookup(cpu, vaddr, mmu_idx, is_store)?;
        let name = if is_io { Some(device_name(cpu, phys)) } else { None };
        Some((phys, is_io, name.unwrap_or(std::ptr::null())))
    });
    match found {
        Some(h) => {
            HWADDR.with(|c| c.set(h));
            (&raw const HWADDR_TOKEN).cast_mut().cast()
        }
        None => {
            error_report("invalid use of qemu_plugin_get_hwaddr");
            std::ptr::null_mut()
        }
    }
}

fn device_name(cpu: &Cpu<'_>, phys: u64) -> *const c_char {
    let fv = cpu.core.address_space().flatview();
    match fv.lookup(phys) {
        Some(fr) if !fr.name().is_empty() => host::intern(fr.name()),
        Some(fr) => host::intern(&format!("anon{:08x}", std::ptr::from_ref(fr) as usize as u32)),
        None => host::intern("RAM"),
    }
}

fn hwaddr(h: *const c_void) -> Option<(u64, bool, *const c_char)> {
    if h.is_null() { None } else { Some(HWADDR.with(Cell::get)) }
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_hwaddr_is_io(h: *const c_void) -> bool {
    hwaddr(h).is_some_and(|h| h.1)
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_hwaddr_phys_addr(h: *const c_void) -> u64 {
    hwaddr(h).map_or(0, |h| h.0)
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_hwaddr_device_name(h: *const c_void) -> *const c_char {
    match hwaddr(h) {
        Some((_, true, name)) if !name.is_null() => name,
        _ => host::intern("RAM"),
    }
}

// Time control.

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_request_time_control() -> *const c_void {
    if host::request_time_control() { (&raw const TIME_TOKEN).cast() } else { std::ptr::null() }
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_update_ns(handle: *const c_void, time: i64) {
    if handle == (&raw const TIME_TOKEN).cast() {
        let target = host::target();
        // Need to execute out of cpu_exec.
        sys::with_cpu(|cpu| {
            cpu.shared().async_run_on_cpu(move |_| target.advance_virtual_time(time));
        });
    }
}

// Miscellaneous queries.

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_num_vcpus() -> c_int {
    c_int::try_from(host::num_vcpus()).unwrap_or(c_int::MAX)
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_outs(string: *const c_char) {
    if let Some(s) = sys::cstr(string) {
        host::outs(&s);
    }
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_bool_parse(
    name: *const c_char,
    value: *const c_char,
    ret: *mut bool,
) -> bool {
    if name.is_null() {
        return false;
    }
    let Some(value) = sys::cstr(value) else { return false };
    match ruvm_qapi::cutils::bool_parse(&value) {
        Some(b) => {
            sys::write_out(ret, b);
            true
        }
        None => false,
    }
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_path_to_binary() -> *const c_char {
    std::ptr::null()
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_start_code() -> u64 {
    0
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_end_code() -> u64 {
    0
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_entry_code() -> u64 {
    0
}

// Registers.

const PC_NAMES: [&str; 6] = ["pc", "eip", "rip", "pswa", "iaoq", "rpc"];

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_get_registers() -> *mut GArray {
    let target = host::target();
    let regs = sys::with_cpu(|cpu| target.registers(cpu));
    let descs: Vec<RegDesc> = regs
        .iter()
        .filter_map(|r| {
            // Skip registers without a name.
            let name = r.name.as_deref()?;
            let ro = PC_NAMES.contains(&name);
            Some(RegDesc {
                handle: ((r.num as usize) << 1 | usize::from(ro)) as *mut c_void,
                name: host::intern(name),
                feature: host::intern(&r.feature),
                is_readonly: ro,
            })
        })
        .collect();
    match host::glib() {
        Some(g) => g.array_of(&descs),
        None => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_read_register(reg: *mut c_void, buf: *mut GArray) -> bool {
    let target = host::target();
    let n = (reg as usize >> 1) as u32;
    let mut bytes = Vec::new();
    let r = sys::with_cpu(|cpu| {
        if host::cb_flags() == CbFlags::NoRegs {
            return 0;
        }
        target.read_register(cpu, n, &mut bytes)
    });
    if r > 0 {
        if let Some(g) = host::glib() {
            g.byte_array_append(buf, &bytes[..r.min(bytes.len())]);
        }
    }
    r > 0
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_write_register(reg: *mut c_void, buf: *mut GArray) -> bool {
    let target = host::target();
    // The read-only property is in the least significant bit.
    assert_eq!(reg as usize & 1, 0, "assertion failed: ((GPOINTER_TO_INT(reg) & 1) == 0)");
    let bytes = sys::byte_array_bytes(buf);
    sys::with_cpu(|cpu| {
        let flags = host::cb_flags();
        if bytes.is_empty() || (flags != CbFlags::RwRegs && flags != CbFlags::RwRegsPc) {
            return false;
        }
        target.write_register(cpu, (reg as usize >> 1) as u32, &bytes) > 0
    })
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_set_pc(vaddr: u64) -> ! {
    sys::with_cpu(|cpu| {
        assert!(
            host::cb_flags() == CbFlags::RwRegsPc,
            "assertion failed: (qemu_plugin_get_cb_flags() == QEMU_PLUGIN_CB_RW_REGS_PC)"
        );
        let ops = cpu.ops();
        ops.set_pc(cpu, vaddr);
        host::set_pending_exit(cpu.cpu_loop_exit());
    });
    resume_unwind(Box::new(host::SetPc))
}

// Memory.

/// `cpu_memory_rw_debug()`.
fn memory_rw_debug(cpu: &mut Cpu<'_>, addr: u64, buf: &mut [u8], is_write: bool) -> bool {
    let target = host::target();
    let mask = cpu.core.jit().page_mask();
    let page_size = (!mask).wrapping_add(1);
    let as_ = cpu.core.address_space().clone();
    let mut done = 0usize;
    while done < buf.len() {
        let a = addr.wrapping_add(done as u64);
        let page = a & mask;
        let Some(phys) = target.phys_page_debug(cpu, page) else { return false };
        let l = usize::try_from(page.wrapping_add(page_size).wrapping_sub(a))
            .unwrap_or(usize::MAX)
            .min(buf.len() - done);
        let p = (phys & mask) + (a & !mask);
        let chunk = &mut buf[done..done + l];
        let r = if is_write {
            as_.write(p, MemTxAttrs::UNSPECIFIED, chunk)
        } else {
            as_.read(p, MemTxAttrs::UNSPECIFIED, chunk)
        };
        if !r.is_ok() {
            return false;
        }
        done += l;
    }
    true
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_read_memory_vaddr(
    addr: u64,
    data: *mut GArray,
    len: usize,
) -> bool {
    assert!(sys::has_cpu(), "assertion failed: (current_cpu)");
    if len == 0 {
        return false;
    }
    let mut buf = vec![0u8; len];
    let ok = sys::with_cpu(|cpu| memory_rw_debug(cpu, addr, &mut buf, false));
    if let Some(g) = host::glib() {
        g.byte_array_fill(data, &buf);
    }
    ok
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_write_memory_vaddr(addr: u64, data: *mut GArray) -> bool {
    assert!(sys::has_cpu(), "assertion failed: (current_cpu)");
    let mut buf = sys::byte_array_bytes(data);
    if buf.is_empty() {
        return false;
    }
    sys::with_cpu(|cpu| memory_rw_debug(cpu, addr, &mut buf, true))
}

fn hwaddr_result(r: MemTxResult) -> c_uint {
    if r == MemTxResult::OK {
        HWADDR_OK
    } else if r == MemTxResult::ERROR {
        HWADDR_DEVICE_ERROR
    } else if r == MemTxResult::DECODE_ERROR {
        HWADDR_INVALID_ADDRESS
    } else if r == MemTxResult::ACCESS_ERROR {
        HWADDR_ACCESS_DENIED
    } else {
        HWADDR_ERROR
    }
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_read_memory_hwaddr(
    addr: u64,
    data: *mut GArray,
    len: usize,
) -> c_uint {
    if len == 0 {
        return HWADDR_ERROR;
    }
    let as_ = sys::with_cpu(|cpu| cpu.core.address_space().clone());
    let mut buf = vec![0u8; len];
    let r = as_.read(addr, MemTxAttrs::UNSPECIFIED, &mut buf);
    if let Some(g) = host::glib() {
        g.byte_array_fill(data, &buf);
    }
    hwaddr_result(r)
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_write_memory_hwaddr(addr: u64, data: *mut GArray) -> c_uint {
    let buf = sys::byte_array_bytes(data);
    if buf.is_empty() {
        return HWADDR_ERROR;
    }
    let as_ = sys::with_cpu(|cpu| cpu.core.address_space().clone());
    hwaddr_result(as_.write(addr, MemTxAttrs::UNSPECIFIED, &buf))
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_translate_vaddr(vaddr: u64, hwaddr: *mut u64) -> bool {
    let target = host::target();
    let phys = sys::with_cpu(|cpu| {
        let mask = cpu.core.jit().page_mask();
        target.phys_page_debug(cpu, vaddr).map(|p| (p & mask) | (vaddr & !mask))
    });
    match phys {
        Some(p) => {
            sys::write_out(hwaddr, p);
            true
        }
        None => false,
    }
}

// Scoreboards.

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_scoreboard_new(element_size: usize) -> *mut c_void {
    host::scoreboard_new(element_size) as *mut c_void
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_scoreboard_free(score: *mut c_void) {
    host::scoreboard_free(score as usize);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_scoreboard_find(
    score: *mut c_void,
    vcpu_index: c_uint,
) -> *mut c_void {
    host::check_vcpu_index(vcpu_index as usize);
    host::scoreboard(score as usize).ptr(vcpu_index as usize)
}

fn u64_entry(e: PluginU64, vcpu_index: c_uint) -> Entry {
    host::check_vcpu_index(vcpu_index as usize);
    entry(e)
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_u64_add(e: PluginU64, vcpu_index: c_uint, added: u64) {
    u64_entry(e, vcpu_index).add(vcpu_index as usize, added);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_u64_get(e: PluginU64, vcpu_index: c_uint) -> u64 {
    u64_entry(e, vcpu_index).get(vcpu_index as usize)
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_u64_set(e: PluginU64, vcpu_index: c_uint, val: u64) {
    u64_entry(e, vcpu_index).set(vcpu_index as usize, val);
}

#[unsafe(no_mangle)]
extern "C-unwind" fn qemu_plugin_u64_sum(e: PluginU64) -> u64 {
    let n = host::num_vcpus();
    let en = entry(e);
    (0..n).fold(0u64, |t, i| t.wrapping_add(en.get(i)))
}
