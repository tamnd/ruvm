// SPDX-License-Identifier: GPL-2.0-or-later

//! The plugin state and the runtime hooks: `plugins/core.c` and `plugins/loader.c`.
//!
//! The state is one mutex, `plugin.lock`. It is never held while a plugin function runs, so a
//! plugin can call any API function from any callback; QEMU gets the same from a recursive
//! mutex.

#![forbid(unsafe_code)]

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::{CStr, CString, c_char, c_uint, c_void};
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};

use ruvm_base::Error;
use ruvm_base::report::{error_report, warn_report};
use ruvm_jit::plugin::{CbFlags, DisconType, DynCb, PluginCb, PluginHooks, PluginMem, PluginTb};
use ruvm_jit::{Cpu, CpuLoopExit, Jit};

use crate::sys::{self, Handle, QemuInfoC};
use crate::{
    LogSink, PluginDesc, PluginTarget, QEMU_PLUGIN_MIN_VERSION, QEMU_PLUGIN_VERSION, QemuInfo,
};

/// `QEMU_PLUGIN_EV_*`.
pub(crate) mod ev {
    pub(crate) const VCPU_INIT: usize = 0;
    pub(crate) const VCPU_EXIT: usize = 1;
    pub(crate) const VCPU_TB_TRANS: usize = 2;
    pub(crate) const VCPU_IDLE: usize = 3;
    pub(crate) const VCPU_RESUME: usize = 4;
    pub(crate) const VCPU_SYSCALL: usize = 5;
    pub(crate) const VCPU_SYSCALL_RET: usize = 6;
    pub(crate) const FLUSH: usize = 7;
    pub(crate) const ATEXIT: usize = 8;
    pub(crate) const VCPU_INTERRUPT: usize = 9;
    pub(crate) const VCPU_EXCEPTION: usize = 10;
    pub(crate) const VCPU_HOSTCALL: usize = 11;
    pub(crate) const VCPU_SYSCALL_FILTER: usize = 12;
    pub(crate) const MAX: usize = 13;
}

/// `qemu_plugin_udata_cb_t`.
pub(crate) type UdataCb = extern "C-unwind" fn(*mut c_void);
/// `qemu_plugin_vcpu_udata_cb_t`.
pub(crate) type VcpuUdataCb = extern "C-unwind" fn(c_uint, *mut c_void);
/// `qemu_plugin_vcpu_discon_cb_t`.
pub(crate) type DisconCb = extern "C-unwind" fn(c_uint, c_uint, u64, u64, *mut c_void);
/// `qemu_plugin_vcpu_tb_trans_cb_t`.
pub(crate) type TbTransCb = extern "C-unwind" fn(*mut c_void, *mut c_void);
/// `qemu_plugin_vcpu_mem_cb_t`.
pub(crate) type MemCb = extern "C-unwind" fn(c_uint, u32, u64, *mut c_void);
/// `qemu_plugin_vcpu_syscall_cb_t`.
pub(crate) type SyscallCb =
    extern "C-unwind" fn(c_uint, i64, u64, u64, u64, u64, u64, u64, u64, u64, *mut c_void);
/// `qemu_plugin_vcpu_syscall_ret_cb_t`.
pub(crate) type SyscallRetCb = extern "C-unwind" fn(c_uint, i64, i64, *mut c_void);
/// `qemu_plugin_vcpu_syscall_filter_cb_t`.
pub(crate) type SyscallFilterCb = extern "C-unwind" fn(
    c_uint,
    i64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    *mut i64,
    *mut c_void,
) -> bool;

/// `union qemu_plugin_cb_sig`, tagged.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Func {
    Udata(UdataCb),
    VcpuUdata(VcpuUdataCb),
    Discon(DisconCb),
    TbTrans(TbTransCb),
    Syscall(SyscallCb),
    SyscallRet(SyscallRetCb),
    SyscallFilter(SyscallFilterCb),
}

/// `struct qemu_plugin_cb`.
#[derive(Clone, Copy, Debug)]
struct Cb {
    id: u64,
    func: Func,
    udata: usize,
}

/// `struct qemu_plugin_ctx`.
#[derive(Debug)]
struct Ctx {
    id: u64,
    handle: Handle,
    /// The arguments, kept alive with the plugin since it may hold on to them.
    _argv: Vec<CString>,
    _argv_ptrs: Vec<usize>,
    installing: bool,
    uninstalling: bool,
    resetting: bool,
}

/// `struct qemu_plugin_scoreboard`: `alloc_size` elements of `elem` bytes, in 64-bit words so
/// that inline operations can be atomic.
pub(crate) struct Scoreboard {
    elem: usize,
    words: RwLock<Box<[AtomicU64]>>,
}

impl fmt::Debug for Scoreboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scoreboard").field("elem", &self.elem).finish_non_exhaustive()
    }
}

fn new_words(bytes: usize) -> Box<[AtomicU64]> {
    (0..bytes.div_ceil(8).max(1)).map(|_| AtomicU64::new(0)).collect()
}

impl Scoreboard {
    fn new(elem: usize, n: usize) -> Scoreboard {
        Scoreboard { elem, words: RwLock::new(new_words(elem * n)) }
    }

    /// `g_array_set_size()`: grow to `n` elements, keeping the values.
    fn resize(&self, n: usize) {
        let mut w = self.words.write().unwrap_or_else(|e| e.into_inner());
        let new = new_words(self.elem * n);
        for (d, s) in new.iter().zip(w.iter()) {
            d.store(s.load(Ordering::Relaxed), Ordering::Relaxed);
        }
        *w = new;
    }

    /// The address of element `idx`, valid until the scoreboard grows.
    pub(crate) fn ptr(&self, idx: usize) -> *mut c_void {
        let w = self.words.read().unwrap_or_else(|e| e.into_inner());
        w.as_ptr().cast::<u8>().cast_mut().wrapping_add(idx * self.elem).cast()
    }

    fn get(&self, off: usize) -> u64 {
        let w = self.words.read().unwrap_or_else(|e| e.into_inner());
        let (i, sh) = (off / 8, off % 8);
        if sh == 0 {
            return w[i].load(Ordering::Relaxed);
        }
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&w[i].load(Ordering::Relaxed).to_ne_bytes());
        b[8..].copy_from_slice(&w[i + 1].load(Ordering::Relaxed).to_ne_bytes());
        u64::from_ne_bytes(b[sh..sh + 8].try_into().expect("eight bytes"))
    }

    fn set(&self, off: usize, v: u64) {
        let w = self.words.read().unwrap_or_else(|e| e.into_inner());
        let (i, sh) = (off / 8, off % 8);
        if sh == 0 {
            w[i].store(v, Ordering::Relaxed);
            return;
        }
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&w[i].load(Ordering::Relaxed).to_ne_bytes());
        b[8..].copy_from_slice(&w[i + 1].load(Ordering::Relaxed).to_ne_bytes());
        b[sh..sh + 8].copy_from_slice(&v.to_ne_bytes());
        w[i].store(u64::from_ne_bytes(b[..8].try_into().expect("eight bytes")), Ordering::Relaxed);
        w[i + 1]
            .store(u64::from_ne_bytes(b[8..].try_into().expect("eight bytes")), Ordering::Relaxed);
    }

    fn add(&self, off: usize, v: u64) {
        if off % 8 == 0 {
            let w = self.words.read().unwrap_or_else(|e| e.into_inner());
            w[off / 8].fetch_add(v, Ordering::Relaxed);
        } else {
            self.set(off, self.get(off).wrapping_add(v));
        }
    }
}

/// `qemu_plugin_u64` resolved.
#[derive(Clone, Debug)]
pub(crate) struct Entry {
    pub(crate) sb: Arc<Scoreboard>,
    pub(crate) offset: usize,
}

impl Entry {
    fn off(&self, idx: usize) -> usize {
        idx * self.sb.elem + self.offset
    }

    pub(crate) fn get(&self, idx: usize) -> u64 {
        self.sb.get(self.off(idx))
    }

    pub(crate) fn set(&self, idx: usize, v: u64) {
        self.sb.set(self.off(idx), v);
    }

    pub(crate) fn add(&self, idx: usize, v: u64) {
        self.sb.add(self.off(idx), v);
    }
}

/// The target hooks used when none was given.
#[derive(Debug)]
struct NoTarget;

impl PluginTarget for NoTarget {}

/// `struct qemu_plugin_state`.
struct State {
    /// Boxed so that a context keeps its address, which seeds its id.
    #[allow(clippy::vec_box)]
    ctxs: Vec<Box<Ctx>>,
    lists: [Vec<Cb>; ev::MAX],
    num_vcpus: usize,
    cpus: BTreeSet<usize>,
    alloc_size: usize,
    scoreboards: BTreeMap<usize, Arc<Scoreboard>>,
    target: Option<Arc<dyn PluginTarget>>,
    log: Option<LogSink>,
}

static STATE: Mutex<State> = Mutex::new(State {
    ctxs: Vec::new(),
    lists: [const { Vec::new() }; ev::MAX],
    num_vcpus: 0,
    cpus: BTreeSet::new(),
    alloc_size: 16,
    scoreboards: BTreeMap::new(),
    target: None,
    log: None,
});

/// `plugin.mask`, readable without the lock.
static MASK: AtomicU32 = AtomicU32::new(0);

static TIME_CONTROL: AtomicBool = AtomicBool::new(false);

fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn invalid_id(id: u64) -> ! {
    error_report(&format!("plugin: invalid plugin id {id}"));
    std::process::abort()
}

impl State {
    /// `plugin_id_to_ctx_locked()`.
    fn ctx(&mut self, id: u64) -> &mut Ctx {
        match self.ctxs.iter_mut().find(|c| c.id == id) {
            Some(c) => c,
            None => invalid_id(id),
        }
    }

    /// `plugin_unregister_cb__locked()`.
    fn unregister(&mut self, id: u64, e: usize) {
        let list = &mut self.lists[e];
        let Some(i) = list.iter().position(|c| c.id == id) else { return };
        list.remove(i);
        if list.is_empty() {
            MASK.fetch_and(!(1 << e), Ordering::AcqRel);
        }
    }

    fn handles(&self) -> Vec<Handle> {
        self.ctxs.iter().map(|c| c.handle).collect()
    }
}

/// `do_plugin_register_cb()`.
pub(crate) fn register_cb(id: u64, e: usize, func: Option<Func>, udata: usize) {
    let mut s = state();
    if s.ctx(id).uninstalling {
        return;
    }
    let Some(func) = func else {
        s.unregister(id, e);
        return;
    };
    let list = &mut s.lists[e];
    match list.iter_mut().find(|c| c.id == id) {
        Some(cb) => {
            cb.func = func;
            cb.udata = udata;
        }
        None => {
            list.insert(0, Cb { id, func, udata });
            MASK.fetch_or(1 << e, Ordering::AcqRel);
        }
    }
}

fn snapshot(e: usize) -> Vec<Cb> {
    if MASK.load(Ordering::Acquire) & (1 << e) == 0 {
        return Vec::new();
    }
    state().lists[e].clone()
}

/// The current target hooks.
pub(crate) fn target() -> Arc<dyn PluginTarget> {
    state().target.clone().unwrap_or_else(|| Arc::new(NoTarget))
}

/// The glib functions, from the process or a plugin.
pub(crate) fn glib() -> Option<sys::Glib> {
    let handles = state().handles();
    sys::Glib::get(&handles)
}

/// `plugin.num_vcpus`.
pub(crate) fn num_vcpus() -> usize {
    state().num_vcpus
}

/// `qemu_plugin_outs()`.
pub(crate) fn outs(s: &str) {
    let log = state().log.clone();
    if let Some(log) = log {
        log(s);
    }
}

/// `qemu_plugin_vcpu_for_each()`.
pub(crate) fn vcpu_for_each(id: u64, cb: VcpuUdataCb, udata: usize) {
    let cpus: Vec<usize> = {
        let mut s = state();
        s.ctx(id);
        s.cpus.iter().copied().collect()
    };
    for idx in cpus {
        cb(idx as c_uint, udata as *mut c_void);
    }
}

/// `qemu_plugin_request_time_control()`.
pub(crate) fn request_time_control() -> bool {
    !TIME_CONTROL.swap(true, Ordering::AcqRel)
}

/// `qemu_plugin_scoreboard_new()`: the handle.
pub(crate) fn scoreboard_new(elem: usize) -> usize {
    let mut s = state();
    let sb = Arc::new(Scoreboard::new(elem, s.alloc_size));
    let handle = Arc::as_ptr(&sb) as usize;
    s.scoreboards.insert(handle, sb);
    handle
}

/// `qemu_plugin_scoreboard_free()`.
pub(crate) fn scoreboard_free(handle: usize) {
    state().scoreboards.remove(&handle);
}

/// The scoreboard behind a handle.
///
/// # Panics
///
/// For a handle that is not a live scoreboard.
pub(crate) fn scoreboard(handle: usize) -> Arc<Scoreboard> {
    match state().scoreboards.get(&handle) {
        Some(sb) => sb.clone(),
        None => panic!("plugin: invalid scoreboard {handle:#x}"),
    }
}

/// Check `vcpu_index` like `qemu_plugin_scoreboard_find()`.
pub(crate) fn check_vcpu_index(idx: usize) {
    assert!(idx < num_vcpus(), "assertion failed: (vcpu_index < qemu_plugin_num_vcpus())");
}

/// `plugin_grow_scoreboards__locked()`.
fn grow_scoreboards(jit: &Jit, idx: usize) {
    let size = {
        let mut s = state();
        if idx < s.alloc_size {
            return;
        }
        let mut size = s.alloc_size;
        while idx >= size {
            size *= 2;
        }
        if s.scoreboards.is_empty() {
            // Just update the size for future scoreboards.
            s.alloc_size = size;
            return;
        }
        size
    };
    // The vCPUs must be stopped, as blocks might still use an existing scoreboard.
    jit.start_exclusive();
    let flush = {
        let mut s = state();
        // In case another vCPU was created before the exclusive section.
        if size > s.alloc_size {
            for sb in s.scoreboards.values() {
                sb.resize(size);
            }
            s.alloc_size = size;
            true
        } else {
            false
        }
    };
    if flush {
        jit.tb_flush_exclusive_or_serial();
    }
    jit.end_exclusive();
}

/// The value `qemu_plugin_set_pc()` unwinds with.
pub(crate) struct SetPc;

thread_local! {
    /// `cpu->neg.plugin_cb_flags`.
    static FLAGS: Cell<CbFlags> = const { Cell::new(CbFlags::NoRegs) };
    /// `plugin_mem_value_low` and `plugin_mem_value_high`.
    static MEM_VALUE: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
    /// The block being translated, while the translation callbacks run.
    static CUR_TB: RefCell<Option<PluginTb>> = const { RefCell::new(None) };
    /// The `cpu_loop_exit()` of a `qemu_plugin_set_pc()` on its way out.
    static PENDING_EXIT: RefCell<Option<CpuLoopExit>> = const { RefCell::new(None) };
}

/// `qemu_plugin_get_cb_flags()`.
pub(crate) fn cb_flags() -> CbFlags {
    FLAGS.with(Cell::get)
}

/// The value of the access a memory callback runs for.
pub(crate) fn mem_value() -> (u64, u64) {
    MEM_VALUE.with(Cell::get)
}

/// Record the exit of `qemu_plugin_set_pc()` before unwinding with [`SetPc`].
pub(crate) fn set_pending_exit(e: CpuLoopExit) {
    PENDING_EXIT.with(|p| *p.borrow_mut() = Some(e));
}

/// Run `f` on the block being translated.
///
/// # Panics
///
/// Outside a translation callback.
pub(crate) fn with_tb<R>(f: impl FnOnce(&mut PluginTb) -> R) -> R {
    CUR_TB.with(|t| {
        let mut t = t.borrow_mut();
        let tb = t.as_mut().expect("plugin: no block is being translated");
        f(tb)
    })
}

/// Puts the block back where the runtime keeps it, even when a callback unwinds.
struct TbGuard<'a>(&'a mut PluginTb);

impl Drop for TbGuard<'_> {
    fn drop(&mut self) {
        if let Some(tb) = CUR_TB.with(|t| t.borrow_mut().take()) {
            *self.0 = tb;
        }
    }
}

/// Resets the callback flags when dropped.
struct FlagsGuard(CbFlags);

impl Drop for FlagsGuard {
    fn drop(&mut self) {
        FLAGS.with(|f| f.set(self.0));
    }
}

/// Call into a plugin on `cpu` with `flags`, turning a `qemu_plugin_set_pc()` into the exit it
/// stands for.
fn call_cpu(cpu: &mut Cpu<'_>, flags: CbFlags, f: impl FnOnce()) -> Result<(), CpuLoopExit> {
    let _flags = FlagsGuard(FLAGS.with(|c| c.replace(flags)));
    match sys::enter_cpu(cpu, || catch_unwind(AssertUnwindSafe(f))) {
        Ok(()) => Ok(()),
        Err(payload) if payload.is::<SetPc>() => {
            Err(PENDING_EXIT.with(|p| p.borrow_mut().take()).expect("set_pc without an exit"))
        }
        Err(payload) => resume_unwind(payload),
    }
}

fn cpu_index(cpu: &Cpu<'_>) -> usize {
    cpu.core.shared().cpu_index
}

/// A callback attached to generated code, `struct qemu_plugin_dyn_cb`.
#[derive(Debug)]
pub(crate) enum HostCb {
    Regular { f: VcpuUdataCb, udata: usize },
    Cond { f: VcpuUdataCb, udata: usize, cond: u32, entry: Entry, imm: u64 },
    Inline { store: bool, entry: Entry, imm: u64 },
    Mem { f: MemCb, udata: usize },
}

impl HostCb {
    /// Wrap it for the runtime.
    pub(crate) fn dyn_cb(self, flags: CbFlags, rw: u32) -> DynCb {
        DynCb { flags, rw, cb: Arc::new(self) }
    }
}

/// `enum qemu_plugin_cond` on unsigned values.
fn cond_holds(cond: u32, v: u64, imm: u64) -> bool {
    match cond {
        1 => true,
        2 => v == imm,
        3 => v != imm,
        4 => v < imm,
        5 => v <= imm,
        6 => v > imm,
        7 => v >= imm,
        _ => false,
    }
}

impl PluginCb for HostCb {
    fn run(
        &self,
        cpu: &mut Cpu<'_>,
        flags: CbFlags,
        mem: Option<&PluginMem>,
    ) -> Result<(), CpuLoopExit> {
        let idx = cpu_index(cpu);
        match self {
            HostCb::Regular { f, udata } => {
                call_cpu(cpu, flags, || f(idx as c_uint, *udata as *mut c_void))
            }
            HostCb::Cond { f, udata, cond, entry, imm } => {
                if cond_holds(*cond, entry.get(idx), *imm) {
                    call_cpu(cpu, flags, || f(idx as c_uint, *udata as *mut c_void))
                } else {
                    Ok(())
                }
            }
            HostCb::Inline { store, entry, imm } => {
                if *store {
                    entry.set(idx, *imm);
                } else {
                    entry.add(idx, *imm);
                }
                Ok(())
            }
            HostCb::Mem { f, udata } => {
                let m = mem.copied().unwrap_or(PluginMem { meminfo: 0, vaddr: 0, low: 0, high: 0 });
                let prev = MEM_VALUE.with(|v| v.replace((m.low, m.high)));
                let r = call_cpu(cpu, flags, || {
                    f(idx as c_uint, m.meminfo, m.vaddr, *udata as *mut c_void)
                });
                MEM_VALUE.with(|v| v.set(prev));
                r
            }
        }
    }
}

/// The runtime hooks.
#[derive(Debug)]
struct Hooks;

fn udata_cbs(cpu: &mut Cpu<'_>, e: usize, flags: CbFlags) {
    let idx = cpu_index(cpu);
    for cb in snapshot(e) {
        if let Func::VcpuUdata(f) = cb.func {
            // A set_pc() from here has nowhere to go; the PC is set and the block goes on.
            let _ = call_cpu(cpu, flags, || f(idx as c_uint, cb.udata as *mut c_void));
        }
    }
}

impl PluginHooks for Hooks {
    fn tb_trans_enabled(&self, _cpu: &Cpu<'_>) -> bool {
        MASK.load(Ordering::Acquire) & (1 << ev::VCPU_TB_TRANS) != 0
    }

    fn tb_trans(&self, cpu: &mut Cpu<'_>, tb: &mut PluginTb) {
        let cbs = snapshot(ev::VCPU_TB_TRANS);
        if cbs.is_empty() {
            return;
        }
        CUR_TB.with(|t| *t.borrow_mut() = Some(std::mem::take(tb)));
        let _guard = TbGuard(tb);
        for cb in cbs {
            if let Func::TbTrans(f) = cb.func {
                let _ = call_cpu(cpu, CbFlags::RwRegs, || {
                    f(crate::api::tb_handle(), cb.udata as *mut c_void)
                });
            }
        }
    }

    fn flush(&self) {
        plugin_cb_udata(ev::FLUSH);
    }

    fn vcpu_init(&self, cpu: &mut Cpu<'_>) {
        let idx = cpu_index(cpu);
        {
            let mut s = state();
            s.num_vcpus = s.num_vcpus.max(idx + 1);
            s.cpus.insert(idx);
        }
        let jit = cpu.jit();
        grow_scoreboards(&jit, idx);
        udata_cbs(cpu, ev::VCPU_INIT, CbFlags::RwRegs);
    }

    fn vcpu_exit(&self, cpu: &mut Cpu<'_>) {
        udata_cbs(cpu, ev::VCPU_EXIT, CbFlags::RwRegs);
        state().cpus.remove(&cpu_index(cpu));
    }

    fn vcpu_idle(&self, cpu: &mut Cpu<'_>) {
        // Idle and resume may come before init; they are ignored then.
        if cpu_index(cpu) < num_vcpus() {
            udata_cbs(cpu, ev::VCPU_IDLE, CbFlags::RwRegsPc);
        }
    }

    fn vcpu_resume(&self, cpu: &mut Cpu<'_>) {
        if cpu_index(cpu) < num_vcpus() {
            udata_cbs(cpu, ev::VCPU_RESUME, CbFlags::RwRegsPc);
        }
    }

    fn discon(
        &self,
        cpu: &mut Cpu<'_>,
        ty: DisconType,
        from: u64,
        to: u64,
    ) -> Result<(), CpuLoopExit> {
        let e = match ty {
            DisconType::Interrupt => ev::VCPU_INTERRUPT,
            DisconType::Exception => ev::VCPU_EXCEPTION,
            DisconType::HostCall => ev::VCPU_HOSTCALL,
        };
        let idx = cpu_index(cpu);
        if idx >= num_vcpus() {
            return Ok(());
        }
        for cb in snapshot(e) {
            if let Func::Discon(f) = cb.func {
                call_cpu(cpu, CbFlags::RwRegsPc, || {
                    f(idx as c_uint, ty as c_uint, from, to, cb.udata as *mut c_void)
                })?;
            }
        }
        Ok(())
    }

    fn syscall(&self, cpu: &mut Cpu<'_>, num: i64, a: &[u64; 8]) -> Result<(), CpuLoopExit> {
        let idx = cpu_index(cpu) as c_uint;
        for cb in snapshot(ev::VCPU_SYSCALL) {
            if let Func::Syscall(f) = cb.func {
                call_cpu(cpu, CbFlags::RwRegsPc, || {
                    let u = cb.udata as *mut c_void;
                    f(idx, num, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], u)
                })?;
            }
        }
        Ok(())
    }

    fn syscall_filter(
        &self,
        cpu: &mut Cpu<'_>,
        num: i64,
        a: &[u64; 8],
    ) -> Result<Option<i64>, CpuLoopExit> {
        let idx = cpu_index(cpu) as c_uint;
        for cb in snapshot(ev::VCPU_SYSCALL_FILTER) {
            if let Func::SyscallFilter(f) = cb.func {
                let mut ret = 0i64;
                let mut filtered = false;
                call_cpu(cpu, CbFlags::RwRegsPc, || {
                    let u = cb.udata as *mut c_void;
                    let r = &raw mut ret;
                    filtered = f(idx, num, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], r, u);
                })?;
                if filtered {
                    return Ok(Some(ret));
                }
            }
        }
        Ok(None)
    }

    fn syscall_ret(&self, cpu: &mut Cpu<'_>, num: i64, ret: i64) -> Result<(), CpuLoopExit> {
        let idx = cpu_index(cpu) as c_uint;
        for cb in snapshot(ev::VCPU_SYSCALL_RET) {
            if let Func::SyscallRet(f) = cb.func {
                call_cpu(cpu, CbFlags::RwRegsPc, || f(idx, num, ret, cb.udata as *mut c_void))?;
            }
        }
        Ok(())
    }
}

/// `plugin_cb__udata()`.
fn plugin_cb_udata(e: usize) {
    for cb in snapshot(e) {
        if let Func::Udata(f) = cb.func {
            f(cb.udata as *mut c_void);
        }
    }
}

/// `qemu_plugin_atexit_cb()`: run the `atexit` callbacks. Call it when the machine stops.
pub fn qemu_plugin_atexit_cb() {
    plugin_cb_udata(ev::ATEXIT);
}

/// Where `qemu_plugin_outs()` writes, `-d plugin` in QEMU. `None` drops the output.
pub fn set_log(sink: Option<LogSink>) {
    state().log = sink;
}

/// How many plugins are loaded.
pub fn loaded() -> usize {
    state().ctxs.len()
}

/// Connect the loaded plugins to `jit`, with `target` for registers, disassembly and symbols.
/// Does nothing when no plugin is loaded. The vCPUs run their `vcpu_init` callbacks when they
/// next run their queued work.
pub fn attach(jit: &Arc<Jit>, target: Arc<dyn PluginTarget>) {
    let any = {
        let mut s = state();
        s.target = Some(target);
        !s.ctxs.is_empty()
    };
    if any {
        jit.set_plugin_hooks(Some(Arc::new(Hooks)));
    }
}

/// What `plugin_reset_destroy()` needs.
#[derive(Debug)]
struct ResetData {
    id: u64,
    cb: Option<UdataCb>,
    udata: usize,
    reset: bool,
}

/// `plugin_reset_destroy__locked()`.
fn reset_destroy(data: ResetData) {
    let handle = {
        let mut s = state();
        for e in 0..ev::MAX {
            s.unregister(data.id, e);
        }
        let ctx = s.ctx(data.id);
        if data.reset {
            assert!(ctx.resetting, "assertion failed: (ctx->resetting)");
            None
        } else {
            assert!(ctx.uninstalling, "assertion failed: (ctx->uninstalling)");
            // We cannot dlclose if we are going to return to plugin code.
            if ctx.installing {
                error_report(
                    "Calling qemu_plugin_uninstall from the install function is a bug. Instead, \
                     return !0 from the install function.",
                );
                std::process::abort();
            }
            let i = s.ctxs.iter().position(|c| c.id == data.id).expect("checked above");
            Some(s.ctxs.remove(i).handle)
        }
    };
    if let Some(cb) = data.cb {
        cb(data.udata as *mut c_void);
    }
    match handle {
        None => state().ctx(data.id).resetting = false,
        Some(h) => {
            if let Err(e) = sys::dlclose(h) {
                warn_report(&format!("plugin_reset_destroy__locked: {e}"));
            }
        }
    }
}

/// `plugin_reset_uninstall()`.
pub(crate) fn reset_uninstall(id: u64, cb: Option<UdataCb>, udata: usize, reset: bool) {
    {
        let mut s = state();
        let ctx = s.ctx(id);
        if ctx.uninstalling || (reset && ctx.resetting) {
            return;
        }
        ctx.resetting = reset;
        ctx.uninstalling = !reset;
    }
    let data = ResetData { id, cb, udata, reset };
    // Only flush the code cache if the vCPUs have been created. If so, current_cpu must be
    // set.
    if sys::has_cpu() {
        sys::with_cpu(|cpu| {
            cpu.shared().async_safe_run_on_cpu(move |cpu| {
                cpu.jit().tb_flush_exclusive_or_serial();
                sys::enter_cpu(cpu, || reset_destroy(data));
            });
        });
    } else {
        // There are no vCPU threads yet, so the callbacks can go synchronously.
        reset_destroy(data);
    }
}

/// From <https://en.wikipedia.org/wiki/Xorshift>.
fn xorshift64star(mut x: u64) -> u64 {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x.wrapping_mul(2685821657736338717)
}

/// `g_intern_string()`.
pub(crate) fn intern(s: &str) -> *const c_char {
    static STRINGS: Mutex<Option<HashMap<String, &'static CStr>>> = Mutex::new(None);
    let mut m = STRINGS.lock().unwrap_or_else(|e| e.into_inner());
    let m = m.get_or_insert_with(HashMap::new);
    if let Some(c) = m.get(s) {
        return c.as_ptr();
    }
    let c: &'static CStr =
        Box::leak(CString::new(s.replace('\0', "")).unwrap_or_default().into_boxed_c_str());
    m.insert(s.to_string(), c);
    c.as_ptr()
}

/// `plugin_load()`.
fn plugin_load(desc: &PluginDesc, info: &QemuInfoC) -> Result<(), Error> {
    let path = &desc.path;
    let fail = |msg: String| Error::generic(format!("Could not load plugin {path}: {msg}"));
    crate::api::keep_alive();

    let handle = sys::dlopen(path).map_err(&fail)?;
    let close = |msg: String| {
        let _ = sys::dlclose(handle);
        Err(fail(msg))
    };
    let install = match sys::dlsym(handle, "qemu_plugin_install") {
        Err(e) => return close(e),
        // The symbol was found; it could be NULL though.
        Ok(0) => return close("qemu_plugin_install is NULL".to_string()),
        Ok(a) => sys::install_fn(a),
    };
    let version = match sys::dlsym(handle, "qemu_plugin_version") {
        Err(e) => return close(format!("plugin does not declare API version {e}")),
        Ok(0) => return close("plugin does not declare API version ".to_string()),
        Ok(a) => sys::read_int(a),
    };
    if version < QEMU_PLUGIN_MIN_VERSION {
        return close(format!(
            "plugin requires API version {version}, but this QEMU supports only a minimum \
             version of {QEMU_PLUGIN_MIN_VERSION}"
        ));
    } else if version > QEMU_PLUGIN_VERSION {
        return close(format!(
            "plugin requires API version {version}, but this QEMU supports only up to version \
             {QEMU_PLUGIN_VERSION}"
        ));
    }

    let argv: Vec<CString> =
        desc.argv.iter().map(|a| CString::new(a.replace('\0', "")).unwrap_or_default()).collect();
    let mut argv_ptrs: Vec<usize> = argv.iter().map(|a| a.as_ptr() as usize).collect();
    argv_ptrs.push(0);
    let argc = i32::try_from(argv.len()).expect("too many plugin arguments");
    let argv_ptr = argv_ptrs.as_ptr().cast::<*const c_char>();
    let mut ctx = Box::new(Ctx {
        id: 0,
        handle,
        _argv: argv,
        _argv_ptrs: argv_ptrs,
        installing: true,
        uninstalling: false,
        resetting: false,
    });
    let id = {
        let mut s = state();
        // Find an unused random id with the address of the context as the seed.
        let mut id = std::ptr::from_ref::<Ctx>(&*ctx) as usize as u64;
        loop {
            id = xorshift64star(id);
            if !s.ctxs.iter().any(|c| c.id == id) {
                break;
            }
        }
        ctx.id = id;
        s.ctxs.push(ctx);
        id
    };
    let rc = install(id, info, argc, argv_ptr);
    state().ctx(id).installing = false;
    if rc != 0 {
        // We cannot rely on the plugin doing its own cleanup, so call a full uninstall if the
        // plugin did not yet call it.
        if !state().ctx(id).uninstalling {
            reset_uninstall(id, None, 0, false);
        }
        return Err(fail(format!("qemu_plugin_install returned error code {rc}")));
    }
    Ok(())
}

/// `qemu_plugin_load_list()`: load the plugins in `head` in order, removing each one that is
/// installed. Stops at the first that fails, leaving it and the rest in `head`.
pub fn qemu_plugin_load_list(head: &mut Vec<PluginDesc>, info: &QemuInfo) -> Result<(), Error> {
    let cinfo = QemuInfoC {
        target_name: intern(&info.target_name),
        version_min: QEMU_PLUGIN_MIN_VERSION,
        version_cur: QEMU_PLUGIN_VERSION,
        system_emulation: info.system_emulation,
        smp_vcpus: info.smp_vcpus,
        max_vcpus: info.max_vcpus,
    };
    while let Some(desc) = head.first() {
        plugin_load(desc, &cinfo)?;
        head.remove(0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoreboard_words() {
        let sb = Scoreboard::new(12, 4);
        let e = Entry { sb: Arc::new(sb), offset: 4 };
        e.set(0, 0x1122_3344_5566_7788);
        e.add(0, 1);
        e.set(1, 5);
        assert_eq!(e.get(0), 0x1122_3344_5566_7789);
        assert_eq!(e.get(1), 5);
        e.sb.resize(8);
        assert_eq!(e.get(0), 0x1122_3344_5566_7789);
        e.add(7, 3);
        assert_eq!(e.get(7), 3);
    }

    #[test]
    fn conditions() {
        assert!(!cond_holds(0, 1, 1));
        assert!(cond_holds(1, 1, 2));
        assert!(cond_holds(2, 3, 3));
        assert!(cond_holds(4, 1, u64::MAX));
        assert!(!cond_holds(6, 1, u64::MAX));
    }

    #[test]
    fn interned_strings_are_shared() {
        assert_eq!(intern("RAM"), intern("RAM"));
        assert_ne!(intern("pc"), intern("RAM"));
    }
}
