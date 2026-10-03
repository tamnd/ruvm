// SPDX-License-Identifier: GPL-2.0-or-later

//! The runtime half of TCG plugins: `accel/tcg/plugin-gen.c`, the plugin hooks of
//! `translator.c`, the memory callbacks of the softmmu helpers, and the vCPU event hooks that
//! `plugins/core.c` exposes to the rest of QEMU.
//!
//! The plugin host (`ruvm-plugin`) implements [`PluginHooks`] and installs it with
//! [`Jit::set_plugin_hooks`]. With no hooks installed the translator emits no markers and
//! generated code is exactly what it is without plugin support.
//!
//! During translation the translator emits `plugin_cb` markers at the start of the block, at
//! the start and end of every instruction, and records the bytes it fetched. At the end of the
//! block the host's `tb_trans` callback sees a [`PluginTb`] and attaches callbacks to it; then
//! [`inject`] replaces the markers with helper calls, exactly where `plugin-gen.c` puts them,
//! and adds a memory callback after every guest load and store of an instrumented instruction.
//!
//! Differences from QEMU:
//!
//! - The callbacks of a block live in a table owned by the runtime and the generated code names
//!   them by index. The table is emptied on `tb_flush`, which is when QEMU frees them too.
//! - Inline operations (`ADD`, `STORE`) and conditional callbacks are helper calls that do the
//!   operation or test the condition, instead of inline TCG ops. The counts they produce are
//!   the same.
//! - `jit-core`'s `qemu_ld` and `qemu_st` emitters do not emit `plugin_mem_cb` markers; the
//!   memory callbacks are attached to the `qemu_ld*`/`qemu_st*` ops of an instrumented
//!   instruction at injection time. The loaded or stored value is passed to the helper instead
//!   of being stored in `CPUState`.
//! - The memory callbacks used by target helpers (`cpu->neg.plugin_mem_cbs`) are a per thread
//!   value, cleared after every run of generated code as in `cpu_tb_exec()`.
//! - `probe_access` does not force the slow path while memory callbacks are enabled, so a
//!   target helper that writes through a host pointer is not seen by memory callbacks.
//! - The host address of an instruction is its `ram_addr`, as for the translator.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use ruvm_jit_core::ir::{Temp, TempI64};
use ruvm_jit_core::types::{PluginFrom, call_flags};
use ruvm_jit_core::{Func, HelperId, HelperInfo, HelperType, MemOpIdx, OpId, Opcode, Type};
use ruvm_jit_interp::{HelperEnv, HelperRegistry, Unwind};

use crate::cf;
use crate::cpu::{Cpu, CpuLoopExit, CpuShared};
use crate::jit::Jit;
use crate::translator::DisasContextBase;

/// `QEMU_PLUGIN_MEM_R`.
pub const MEM_R: u32 = 1;
/// `QEMU_PLUGIN_MEM_W`.
pub const MEM_W: u32 = 2;
/// `QEMU_PLUGIN_MEM_RW`.
pub const MEM_RW: u32 = 3;

/// What a callback may do with the guest registers, `enum qemu_plugin_cb_flags`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CbFlags {
    /// `QEMU_PLUGIN_CB_NO_REGS`.
    NoRegs = 0,
    /// `QEMU_PLUGIN_CB_R_REGS`.
    RRegs = 1,
    /// `QEMU_PLUGIN_CB_RW_REGS`.
    RwRegs = 2,
    /// `QEMU_PLUGIN_CB_RW_REGS_PC`.
    RwRegsPc = 3,
}

impl CbFlags {
    /// The flags for the C value, `NoRegs` for anything unknown.
    pub fn from_raw(v: u32) -> CbFlags {
        match v {
            1 => CbFlags::RRegs,
            2 => CbFlags::RwRegs,
            3 => CbFlags::RwRegsPc,
            _ => CbFlags::NoRegs,
        }
    }

    /// `plugin_cb_flags_to_tcg()`: the helper call flags.
    pub fn call_flags(self) -> u32 {
        match self {
            CbFlags::NoRegs => call_flags::NO_RWG,
            CbFlags::RRegs => call_flags::NO_WG,
            CbFlags::RwRegs | CbFlags::RwRegsPc => 0,
        }
    }

    /// What the callback sees from `qemu_plugin_get_cb_flags()` while it runs, which is
    /// `tcg_call_to_qemu_plugin_cb_flags()` of [`CbFlags::call_flags`].
    pub fn at_run_time(self) -> CbFlags {
        match self {
            CbFlags::NoRegs => CbFlags::NoRegs,
            CbFlags::RRegs => CbFlags::RRegs,
            CbFlags::RwRegs | CbFlags::RwRegsPc => CbFlags::RwRegsPc,
        }
    }
}

/// The memory access a memory callback runs for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PluginMem {
    /// `qemu_plugin_meminfo_t`: the `MemOpIdx` with the access direction in bits 16 and up.
    pub meminfo: u32,
    /// The guest virtual address.
    pub vaddr: u64,
    /// The low 64 bits of the value, `plugin_mem_value_low`.
    pub low: u64,
    /// The high 64 bits of a 128-bit value, `plugin_mem_value_high`.
    pub high: u64,
}

/// `make_plugin_meminfo()`.
pub fn make_plugin_meminfo(oi: MemOpIdx, rw: u32) -> u32 {
    oi.0 | (rw << 16)
}

/// One callback attached to generated code. The host decides what running it means: a C
/// call, an inline operation, or a conditional call.
pub trait PluginCb: Send + Sync + fmt::Debug {
    /// Run the callback on `cpu` with `flags` visible to the plugin. `mem` is the access for
    /// a memory callback. An `Err` leaves the block, which is how `qemu_plugin_set_pc()` ends.
    fn run(
        &self,
        cpu: &mut Cpu<'_>,
        flags: CbFlags,
        mem: Option<&PluginMem>,
    ) -> Result<(), CpuLoopExit>;
}

/// A callback with its call flags, `struct qemu_plugin_dyn_cb`.
#[derive(Clone, Debug)]
pub struct DynCb {
    /// What the callback may do with registers. Inline operations use
    /// [`CbFlags::NoRegs`].
    pub flags: CbFlags,
    /// For memory callbacks, the directions it runs for ([`MEM_R`], [`MEM_W`]). Ignored
    /// otherwise.
    pub rw: u32,
    /// The callback.
    pub cb: Arc<dyn PluginCb>,
}

/// An instruction of a block being translated, `struct qemu_plugin_insn`.
#[derive(Clone, Debug, Default)]
pub struct PluginInsn {
    /// The guest virtual address.
    pub vaddr: u64,
    /// The length in bytes.
    pub len: u64,
    /// The bytes, or `None` when they could not be read back.
    pub data: Option<Vec<u8>>,
    /// The `ram_addr` of the first byte, standing in for the host address.
    pub haddr: Option<u64>,
    /// Callbacks run before the instruction.
    pub insn_cbs: Vec<DynCb>,
    /// Callbacks run after every memory access of the instruction.
    pub mem_cbs: Vec<DynCb>,
    calls_helpers: bool,
    mem_helper: bool,
}

/// A block being translated, `struct qemu_plugin_tb`.
#[derive(Clone, Debug, Default)]
pub struct PluginTb {
    /// The guest virtual address of the first instruction.
    pub vaddr: u64,
    /// The instructions.
    pub insns: Vec<PluginInsn>,
    /// Callbacks run at the start of the block.
    pub cbs: Vec<DynCb>,
    /// `CF_MEMI_ONLY`: only memory callbacks may be added, because the instruction already
    /// ran and is being replayed for its IO access.
    pub mem_only: bool,
    mem_helper: bool,
}

/// The kinds of discontinuity, `enum qemu_plugin_discon_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DisconType {
    /// `QEMU_PLUGIN_DISCON_INTERRUPT`.
    Interrupt = 1,
    /// `QEMU_PLUGIN_DISCON_EXCEPTION`.
    Exception = 2,
    /// `QEMU_PLUGIN_DISCON_HOSTCALL`.
    HostCall = 4,
}

/// What the runtime asks of the plugin host. All methods but [`PluginHooks::tb_trans`] and
/// [`PluginHooks::tb_trans_enabled`] have empty defaults.
pub trait PluginHooks: Send + Sync + fmt::Debug {
    /// Whether `cpu` instruments the block it is about to translate, the
    /// `QEMU_PLUGIN_EV_VCPU_TB_TRANS` bit of its event mask.
    fn tb_trans_enabled(&self, cpu: &Cpu<'_>) -> bool;

    /// `qemu_plugin_tb_trans_cb()`: let the plugins attach callbacks to `tb`.
    fn tb_trans(&self, cpu: &mut Cpu<'_>, tb: &mut PluginTb);

    /// `qemu_plugin_flush_cb()`: the code buffer was flushed.
    fn flush(&self) {}

    /// `qemu_plugin_vcpu_init__async()`: `cpu` starts running.
    fn vcpu_init(&self, cpu: &mut Cpu<'_>) {
        let _ = cpu;
    }

    /// `qemu_plugin_vcpu_exit_hook()`.
    fn vcpu_exit(&self, cpu: &mut Cpu<'_>) {
        let _ = cpu;
    }

    /// `qemu_plugin_vcpu_idle_cb()`.
    fn vcpu_idle(&self, cpu: &mut Cpu<'_>) {
        let _ = cpu;
    }

    /// `qemu_plugin_vcpu_resume_cb()`.
    fn vcpu_resume(&self, cpu: &mut Cpu<'_>) {
        let _ = cpu;
    }

    /// The discontinuity callbacks, `plugin_vcpu_cb__discon()`.
    fn discon(
        &self,
        cpu: &mut Cpu<'_>,
        ty: DisconType,
        from: u64,
        to: u64,
    ) -> Result<(), CpuLoopExit> {
        let _ = (cpu, ty, from, to);
        Ok(())
    }

    /// `qemu_plugin_vcpu_syscall()`.
    fn syscall(&self, cpu: &mut Cpu<'_>, num: i64, args: &[u64; 8]) -> Result<(), CpuLoopExit> {
        let _ = (cpu, num, args);
        Ok(())
    }

    /// `qemu_plugin_vcpu_syscall_filter()`: `Some(ret)` when a plugin handled the call.
    fn syscall_filter(
        &self,
        cpu: &mut Cpu<'_>,
        num: i64,
        args: &[u64; 8],
    ) -> Result<Option<i64>, CpuLoopExit> {
        let _ = (cpu, num, args);
        Ok(None)
    }

    /// `qemu_plugin_vcpu_syscall_ret()`.
    fn syscall_ret(&self, cpu: &mut Cpu<'_>, num: i64, ret: i64) -> Result<(), CpuLoopExit> {
        let _ = (cpu, num, ret);
        Ok(())
    }
}

/// The callbacks generated code refers to, by `base + index`.
#[derive(Debug, Default)]
struct CbTable {
    base: u64,
    entries: Vec<Arc<[DynCb]>>,
}

/// The plugin state of a [`Jit`].
#[derive(Debug, Default)]
pub(crate) struct JitPlugin {
    active: AtomicBool,
    hooks: RwLock<Option<Arc<dyn PluginHooks>>>,
    table: RwLock<CbTable>,
}

impl JitPlugin {
    fn hooks(&self) -> Option<Arc<dyn PluginHooks>> {
        if !self.active.load(Ordering::Acquire) {
            return None;
        }
        self.hooks.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn register(&self, cbs: Arc<[DynCb]>) -> u64 {
        let mut t = self.table.write().unwrap_or_else(|e| e.into_inner());
        let id = t.base + t.entries.len() as u64;
        t.entries.push(cbs);
        id
    }

    fn entry(&self, id: u64) -> Option<Arc<[DynCb]>> {
        let t = self.table.read().unwrap_or_else(|e| e.into_inner());
        let i = id.checked_sub(t.base)?;
        t.entries.get(usize::try_from(i).ok()?).cloned()
    }
}

impl Jit {
    /// Install the plugin host, or remove it with `None`. vCPUs created before get their
    /// `vcpu_init` callback queued as work, as do vCPUs created later.
    pub fn set_plugin_hooks(&self, hooks: Option<Arc<dyn PluginHooks>>) {
        let on = hooks.is_some();
        *self.plugin.hooks.write().unwrap_or_else(|e| e.into_inner()) = hooks;
        let was = self.plugin.active.swap(on, Ordering::AcqRel);
        if on && !was {
            for cpu in self.cpu_list() {
                self.plugin_vcpu_created(&cpu);
            }
        }
    }

    /// The installed plugin host.
    pub fn plugin_hooks(&self) -> Option<Arc<dyn PluginHooks>> {
        self.plugin.hooks()
    }

    /// `qemu_plugin_vcpu_init_hook()`: the init callback runs once the vCPU runs its work.
    pub(crate) fn plugin_vcpu_created(&self, cpu: &Arc<CpuShared>) {
        if !self.plugin.active.load(Ordering::Acquire) {
            return;
        }
        cpu.async_run_on_cpu(|cpu| {
            if let Some(h) = cpu.core.jit.plugin.hooks() {
                h.vcpu_init(cpu);
            }
        });
    }

    /// `qemu_plugin_flush_cb()`, at the end of `tb_flush`.
    pub(crate) fn plugin_flush(&self) {
        {
            let mut t = self.plugin.table.write().unwrap_or_else(|e| e.into_inner());
            t.base += t.entries.len() as u64;
            t.entries.clear();
        }
        if let Some(h) = self.plugin.hooks() {
            h.flush();
        }
    }
}

/// `qemu_plugin_vcpu_exit_hook()`: call it when `cpu` is unrealized.
pub fn vcpu_exit(cpu: &mut Cpu<'_>) {
    if let Some(h) = cpu.core.jit.plugin.hooks() {
        h.vcpu_exit(cpu);
    }
}

/// `qemu_plugin_vcpu_idle_cb()`.
pub fn vcpu_idle(cpu: &mut Cpu<'_>) {
    if let Some(h) = cpu.core.jit.plugin.hooks() {
        h.vcpu_idle(cpu);
    }
}

/// `qemu_plugin_vcpu_resume_cb()`.
pub fn vcpu_resume(cpu: &mut Cpu<'_>) {
    if let Some(h) = cpu.core.jit.plugin.hooks() {
        h.vcpu_resume(cpu);
    }
}

/// Whether a plugin host is installed; cheap.
pub fn enabled(cpu: &Cpu<'_>) -> bool {
    cpu.core.jit.plugin.active.load(Ordering::Acquire)
}

fn discon(cpu: &mut Cpu<'_>, ty: DisconType, from: u64) -> Result<(), CpuLoopExit> {
    let Some(h) = cpu.core.jit.plugin.hooks() else { return Ok(()) };
    let ops = cpu.ops();
    let to = ops.get_pc(cpu);
    h.discon(cpu, ty, from, to)
}

/// `qemu_plugin_vcpu_interrupt_cb()`: the target took an interrupt at `from`; the PC is
/// already the handler's.
pub fn vcpu_interrupt(cpu: &mut Cpu<'_>, from: u64) -> Result<(), CpuLoopExit> {
    discon(cpu, DisconType::Interrupt, from)
}

/// `qemu_plugin_vcpu_exception_cb()`.
pub fn vcpu_exception(cpu: &mut Cpu<'_>, from: u64) -> Result<(), CpuLoopExit> {
    discon(cpu, DisconType::Exception, from)
}

/// `qemu_plugin_vcpu_hostcall_cb()`.
pub fn vcpu_hostcall(cpu: &mut Cpu<'_>, from: u64) -> Result<(), CpuLoopExit> {
    discon(cpu, DisconType::HostCall, from)
}

fn syscall_args(cpu: &Cpu<'_>, args: &[u64; 8]) -> [u64; 8] {
    let mut a = *args;
    if cpu.core.jit.config.target_long_bits == 32 {
        for v in &mut a {
            *v = u64::from(*v as u32);
        }
    }
    a
}

/// `qemu_plugin_vcpu_syscall()`.
pub fn vcpu_syscall(cpu: &mut Cpu<'_>, num: i64, args: &[u64; 8]) -> Result<(), CpuLoopExit> {
    let Some(h) = cpu.core.jit.plugin.hooks() else { return Ok(()) };
    let a = syscall_args(cpu, args);
    h.syscall(cpu, num, &a)
}

/// `qemu_plugin_vcpu_syscall_filter()`: `Some(ret)` when a plugin handled the call and the
/// guest should see `ret` without the call being made.
pub fn vcpu_syscall_filter(
    cpu: &mut Cpu<'_>,
    num: i64,
    args: &[u64; 8],
) -> Result<Option<i64>, CpuLoopExit> {
    let Some(h) = cpu.core.jit.plugin.hooks() else { return Ok(None) };
    let a = syscall_args(cpu, args);
    h.syscall_filter(cpu, num, &a)
}

/// `qemu_plugin_vcpu_syscall_ret()`.
pub fn vcpu_syscall_ret(cpu: &mut Cpu<'_>, num: i64, ret: i64) -> Result<(), CpuLoopExit> {
    let Some(h) = cpu.core.jit.plugin.hooks() else { return Ok(()) };
    h.syscall_ret(cpu, num, ret)
}

thread_local! {
    /// `cpu->neg.plugin_mem_cbs`.
    static MEM_HELPERS: RefCell<Option<Arc<[DynCb]>>> = const { RefCell::new(None) };
}

/// `qemu_plugin_disable_mem_helpers()`.
pub fn disable_mem_helpers() {
    MEM_HELPERS.with(|m| {
        if let Ok(mut m) = m.try_borrow_mut() {
            *m = None;
        }
    });
}

/// `qemu_plugin_vcpu_mem_cb()`: a target helper accessed memory. Runs the memory callbacks
/// of the current instruction when it was instrumented and calls helpers.
pub(crate) fn helper_mem_cb(
    cpu: &mut Cpu<'_>,
    vaddr: u64,
    low: u64,
    high: u64,
    oi: MemOpIdx,
    rw: u32,
) -> Result<(), CpuLoopExit> {
    let cbs = MEM_HELPERS.with(|m| m.try_borrow().ok().and_then(|m| m.clone()));
    let Some(cbs) = cbs else { return Ok(()) };
    let mem = PluginMem { meminfo: make_plugin_meminfo(oi, rw), vaddr, low, high };
    for cb in cbs.iter() {
        if rw & cb.rw != 0 {
            cb.cb.run(cpu, cb.flags.at_run_time(), Some(&mem))?;
        }
    }
    Ok(())
}

const CB_HELPERS: [&str; 3] = ["plugin_cb_no_regs", "plugin_cb_r_regs", "plugin_cb_rw_regs"];
const MEM_CB_HELPERS: [&str; 3] =
    ["plugin_mem_cb_no_regs", "plugin_mem_cb_r_regs", "plugin_mem_cb_rw_regs"];
const MEM_HELPERS_HELPER: &str = "plugin_mem_helpers";
const MEM_CB_ARGS: [HelperType; 5] =
    [HelperType::I64, HelperType::I32, HelperType::I64, HelperType::I64, HelperType::I64];

fn flags_index(flags: CbFlags) -> usize {
    match flags {
        CbFlags::NoRegs => 0,
        CbFlags::RRegs => 1,
        CbFlags::RwRegs | CbFlags::RwRegsPc => 2,
    }
}

fn run_cbs(h: &mut HelperEnv<'_>, id: u64, mem: Option<&PluginMem>) -> Result<u128, Unwind> {
    let Some(mut cpu) = Cpu::from_helper_env(h) else { return Ok(0) };
    let Some(cbs) = cpu.core.jit.plugin.entry(id) else { return Ok(0) };
    for cb in cbs.iter() {
        if let Err(e) = cb.cb.run(&mut cpu, cb.flags.at_run_time(), mem) {
            return Err(cpu.unwind(e));
        }
    }
    Ok(0)
}

fn helper_cb(h: &mut HelperEnv<'_>, args: &[u64]) -> Result<u128, Unwind> {
    run_cbs(h, args[0], None)
}

fn helper_mem_cb_call(h: &mut HelperEnv<'_>, args: &[u64]) -> Result<u128, Unwind> {
    let mem = PluginMem { meminfo: args[1] as u32, vaddr: args[2], low: args[3], high: args[4] };
    run_cbs(h, args[0], Some(&mem))
}

fn helper_mem_helpers(h: &mut HelperEnv<'_>, args: &[u64]) -> Result<u128, Unwind> {
    let Some(cpu) = Cpu::from_helper_env(h) else { return Ok(0) };
    let cbs = if args[0] == u64::MAX { None } else { cpu.core.jit.plugin.entry(args[0]) };
    MEM_HELPERS.with(|m| {
        if let Ok(mut m) = m.try_borrow_mut() {
            *m = cbs;
        }
    });
    Ok(0)
}

/// Add the plugin helpers to an interpreter helper registry.
pub fn register_helpers(helpers: &mut HelperRegistry) {
    for name in CB_HELPERS {
        helpers.register(name, HelperType::Void, &[HelperType::I64], helper_cb);
    }
    for name in MEM_CB_HELPERS {
        helpers.register(name, HelperType::Void, &MEM_CB_ARGS, helper_mem_cb_call);
    }
    helpers.register(MEM_HELPERS_HELPER, HelperType::Void, &[HelperType::I64], helper_mem_helpers);
}

/// The plugin state of a block being translated.
#[derive(Debug, Default)]
pub(crate) struct PluginGen {
    tb: PluginTb,
    /// Code bytes fetched by the translator, by guest address.
    bytes: BTreeMap<u64, u8>,
}

impl PluginGen {
    /// `record_save()`: remember code bytes for `qemu_plugin_insn_data()`.
    pub(crate) fn record(&mut self, pc: u64, pc_first: u64, buf: &[u8]) {
        if pc < pc_first {
            return;
        }
        for (i, b) in buf.iter().enumerate() {
            self.bytes.insert(pc.wrapping_add(i as u64), *b);
        }
    }
}

/// `plugin_gen_tb_start()`: whether the block is instrumented. Emits the block marker.
pub(crate) fn gen_tb_start(db: &mut DisasContextBase<'_>, cpu: &mut Cpu<'_>) -> bool {
    let Some(h) = cpu.core.jit.plugin.hooks() else { return false };
    if !h.tb_trans_enabled(cpu) {
        return false;
    }
    let tb = PluginTb {
        vaddr: db.pc_first,
        mem_only: db.tb.cflags & cf::MEMI_ONLY != 0,
        ..PluginTb::default()
    };
    db.plugin = Some(PluginGen { tb, bytes: BTreeMap::new() });
    db.tb.f.gen_plugin_cb(PluginFrom::Tb);
    true
}

/// `plugin_gen_insn_start()`.
pub(crate) fn gen_insn_start(db: &mut DisasContextBase<'_>) {
    let n = db.num_insns as usize;
    let vaddr = db.pc_next;
    let Some(p) = db.plugin.as_mut() else { return };
    p.tb.insns.truncate(n - 1);
    p.tb.insns.push(PluginInsn { vaddr, ..PluginInsn::default() });
    db.tb.f.gen_plugin_cb(PluginFrom::Insn);
}

/// `plugin_gen_insn_end()`.
pub(crate) fn gen_insn_end(db: &mut DisasContextBase<'_>) {
    let pc_next = db.pc_next;
    let Some(p) = db.plugin.as_mut() else { return };
    if let Some(insn) = p.tb.insns.last_mut() {
        insn.len = pc_next.wrapping_sub(insn.vaddr);
    }
    db.tb.f.gen_plugin_cb(PluginFrom::AfterInsn);
}

/// `plugin_gen_tb_end()`: fill in the instructions, call the `tb_trans` callbacks and turn the
/// markers into calls.
pub(crate) fn gen_tb_end(db: &mut DisasContextBase<'_>, cpu: &mut Cpu<'_>) {
    let Some(mut p) = db.plugin.take() else { return };
    let n = db.num_insns as usize;
    p.tb.insns.truncate(n);
    let page0_last = db.pc_first | !db.page_mask();
    let host = db.host_addrs();
    let jit = cpu.jit();
    for insn in &mut p.tb.insns {
        insn.haddr = if insn.vaddr <= page0_last {
            host[0].map(|h| h.wrapping_add(insn.vaddr.wrapping_sub(db.pc_first)))
        } else {
            host[1].map(|h| h.wrapping_add(insn.vaddr.wrapping_sub(page0_last.wrapping_add(1))))
        };
        let len = insn.len as usize;
        let mut data = Vec::with_capacity(len);
        for i in 0..len as u64 {
            match p.bytes.get(&insn.vaddr.wrapping_add(i)) {
                Some(b) => data.push(*b),
                None => break,
            }
        }
        if data.len() < len {
            data.resize(len, 0);
            let read = insn
                .haddr
                .and_then(|ra| jit.ram_block_from_addr(ra))
                .is_some_and(|(block, off)| block.read(off, &mut data).is_ok());
            insn.data = if read { Some(data) } else { None };
        } else {
            insn.data = Some(data);
        }
    }
    if let Some(h) = jit.plugin.hooks() {
        h.tb_trans(cpu, &mut p.tb);
    }
    inject(&jit, &mut db.tb.f, &mut p.tb);
}

/// The helpers injection calls, declared in the block's IR.
struct Helpers {
    cb: [Option<HelperId>; 3],
    mem_cb: [Option<HelperId>; 3],
    mem_helpers: Option<HelperId>,
}

impl Helpers {
    fn cb(&mut self, f: &mut Func, flags: CbFlags) -> HelperId {
        let i = flags_index(flags);
        *self.cb[i].get_or_insert_with(|| {
            f.helper(HelperInfo::new(
                CB_HELPERS[i],
                flags.call_flags(),
                HelperType::Void,
                &[HelperType::I64],
            ))
        })
    }

    fn mem_cb(&mut self, f: &mut Func, flags: CbFlags) -> HelperId {
        let i = flags_index(flags);
        *self.mem_cb[i].get_or_insert_with(|| {
            f.helper(HelperInfo::new(
                MEM_CB_HELPERS[i],
                flags.call_flags(),
                HelperType::Void,
                &MEM_CB_ARGS,
            ))
        })
    }

    fn mem_helpers(&mut self, f: &mut Func) -> HelperId {
        *self.mem_helpers.get_or_insert_with(|| {
            f.helper(HelperInfo::new(
                MEM_HELPERS_HELPER,
                call_flags::NO_RWG,
                HelperType::Void,
                &[HelperType::I64],
            ))
        })
    }
}

/// Move the ops emitted after `mark` to just before `target`, or leave them at the end for
/// `None`.
fn move_before(f: &mut Func, mark: Option<OpId>, target: Option<OpId>) {
    let Some(target) = target else { return };
    let mut id = match mark {
        Some(m) => f.next_op(m),
        None => f.first_op(),
    };
    let mut moved = Vec::new();
    while let Some(i) = id {
        moved.push(i);
        id = f.next_op(i);
    }
    for i in moved {
        let op = *f.op(i);
        let new = f.insert_before(target, op.opc, op.ty, op.nargs as usize);
        *f.op_mut(new) = op;
        f.remove_op(i);
    }
}

fn const_i64(f: &mut Func, v: u64) -> Temp {
    f.constant_i64(v as i64).temp()
}

/// Emit one call per callback, each to the helper for its flags.
fn gen_udata_cbs(jit: &Jit, f: &mut Func, h: &mut Helpers, cbs: &[DynCb]) {
    for cb in cbs {
        let id = jit.plugin.register(Arc::from([cb.clone()]));
        let helper = h.cb(f, cb.flags);
        let arg = const_i64(f, id);
        f.gen_call(helper, None, &[arg]);
    }
}

/// `gen_disable_mem_helper()`.
fn gen_disable_mem_helper(f: &mut Func, h: &mut Helpers) {
    let helper = h.mem_helpers(f);
    let arg = const_i64(f, u64::MAX);
    f.gen_call(helper, None, &[arg]);
}

/// `gen_enable_mem_helper()`.
fn gen_enable_mem_helper(jit: &Jit, f: &mut Func, h: &mut Helpers, ptb: &mut PluginTb, i: usize) {
    let insn = &mut ptb.insns[i];
    // Tracking memory accesses performed from helpers requires extra work. If an instruction
    // is emulated with helpers, we do two things: (1) copy the CB descriptors, and keep track
    // of it so that they can be freed later on, and (2) point the vCPU's memory callbacks to
    // the descriptors, so that we can read them at run-time (i.e. when the helper executes).
    // This run-time access is performed from qemu_plugin_vcpu_mem_cb.
    //
    // Note that plugin_gen_disable_mem_helpers undoes (2). Since it is unlikely that
    // a helper accesses memory, the disable is done after the instruction.
    if !insn.calls_helpers {
        return;
    }
    if insn.mem_cbs.is_empty() {
        insn.mem_helper = false;
        return;
    }
    insn.mem_helper = true;
    ptb.mem_helper = true;
    let id = jit.plugin.register(Arc::from(insn.mem_cbs.clone()));
    let helper = h.mem_helpers(f);
    let arg = const_i64(f, id);
    f.gen_call(helper, None, &[arg]);
}

/// The memory callbacks after one guest access: a copy of the address made before the op,
/// and the call made after it.
fn gen_mem_cbs(jit: &Jit, f: &mut Func, h: &mut Helpers, insn: &PluginInsn, op: OpId) {
    let o = *f.op(op);
    let (two, store) = match o.opc {
        Opcode::QemuLd => (false, false),
        Opcode::QemuSt => (false, true),
        Opcode::QemuLd2 => (true, false),
        _ => (true, true),
    };
    let rw = if store { MEM_W } else { MEM_R };
    let cbs: Vec<&DynCb> = insn.mem_cbs.iter().filter(|cb| rw & cb.rw != 0).collect();
    if cbs.is_empty() {
        return;
    }
    let (val_lo, val_hi, addr, oi) = if two {
        (o.arg_temp(0), Some(o.arg_temp(1)), o.arg_temp(2), MemOpIdx(o.args[3] as u32))
    } else {
        (o.arg_temp(0), None, o.arg_temp(1), MemOpIdx(o.args[2] as u32))
    };

    // plugin_maybe_preserve_addr(): the load may overwrite its own address.
    let mark = f.last_op();
    let copy: TempI64 = f.temp_ebb_new_i64();
    let addr_opc = if f.temp(addr).ty == Type::I32 { Opcode::ExtuI32I64 } else { Opcode::Mov };
    f.emit_op(addr_opc, Type::I64, &[copy.arg(), addr.arg()]);
    move_before(f, mark, Some(op));

    let next = f.next_op(op);
    let mark = f.last_op();
    let lo = if f.temp(val_lo).ty == Type::I32 {
        let t = f.temp_ebb_new_i64();
        f.emit_op(Opcode::ExtuI32I64, Type::I64, &[t.arg(), val_lo.arg()]);
        t.temp()
    } else {
        val_lo
    };
    let hi = match val_hi {
        Some(t) => t,
        None => const_i64(f, 0),
    };
    let meminfo = f.constant_i32(make_plugin_meminfo(oi, rw) as i32).temp();
    for cb in cbs {
        let id = jit.plugin.register(Arc::from([cb.clone()]));
        let helper = h.mem_cb(f, cb.flags);
        let arg = const_i64(f, id);
        f.gen_call(helper, None, &[arg, meminfo, copy.temp(), lo, hi]);
    }
    if lo != val_lo {
        f.temp_free(lo);
    }
    f.temp_free(copy);
    move_before(f, mark, next);
}

/// `plugin_gen_inject()`: replace the markers of an instrumented block with the calls the
/// plugins asked for.
fn inject(jit: &Jit, f: &mut Func, ptb: &mut PluginTb) {
    f.temp_ebb_reset_freed();

    // tcg_gen_callN() notes, per instruction, whether a helper that may touch guest state is
    // called. The plugin calls are not there yet, so every call counts.
    let mut insn_idx: isize = -1;
    for (_, op) in f.ops() {
        match op.opc {
            Opcode::InsnStart => insn_idx += 1,
            Opcode::Call if insn_idx >= 0 => {
                let info = f.helper_info(op.call_helper());
                if info.flags & call_flags::NO_SIDE_EFFECTS == 0 {
                    if let Some(insn) = ptb.insns.get_mut(insn_idx as usize) {
                        insn.calls_helpers = true;
                    }
                }
            }
            _ => {}
        }
    }

    let mut h = Helpers { cb: [None; 3], mem_cb: [None; 3], mem_helpers: None };
    let mut insn_idx: isize = -1;
    for op in f.op_ids() {
        if !f.is_linked(op) {
            continue;
        }
        let o = *f.op(op);
        match o.opc {
            Opcode::InsnStart => insn_idx += 1,
            Opcode::ExitTb | Opcode::GotoTb | Opcode::GotoPtr if insn_idx >= 0 => {
                // plugin_gen_disable_mem_helpers() puts an AFTER_TB marker before every way
                // out of the block once an instruction started.
                if ptb.mem_helper {
                    let mark = f.last_op();
                    gen_disable_mem_helper(f, &mut h);
                    move_before(f, mark, Some(op));
                }
            }
            Opcode::PluginCb => {
                let mark = f.last_op();
                match o.args[0] {
                    x if x == PluginFrom::AfterTb as u64 => {
                        if ptb.mem_helper {
                            gen_disable_mem_helper(f, &mut h);
                        }
                    }
                    x if x == PluginFrom::AfterInsn as u64 => {
                        let i = insn_idx as usize;
                        if ptb.insns.get(i).is_some_and(|insn| insn.mem_helper) {
                            gen_disable_mem_helper(f, &mut h);
                        }
                    }
                    x if x == PluginFrom::Tb as u64 => {
                        let cbs = std::mem::take(&mut ptb.cbs);
                        gen_udata_cbs(jit, f, &mut h, &cbs);
                        ptb.cbs = cbs;
                    }
                    x if x == PluginFrom::Insn as u64 => {
                        let i = insn_idx as usize;
                        if i < ptb.insns.len() {
                            gen_enable_mem_helper(jit, f, &mut h, ptb, i);
                            let cbs = std::mem::take(&mut ptb.insns[i].insn_cbs);
                            gen_udata_cbs(jit, f, &mut h, &cbs);
                            ptb.insns[i].insn_cbs = cbs;
                        }
                    }
                    _ => unreachable!("bad plugin_cb marker"),
                }
                move_before(f, mark, Some(op));
                f.remove_op(op);
            }
            Opcode::PluginMemCb => f.remove_op(op),
            Opcode::QemuLd | Opcode::QemuSt | Opcode::QemuLd2 | Opcode::QemuSt2
                if insn_idx >= 0 =>
            {
                if let Some(insn) = ptb.insns.get(insn_idx as usize) {
                    if !insn.mem_cbs.is_empty() {
                        let insn = insn.clone();
                        gen_mem_cbs(jit, f, &mut h, &insn, op);
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_follow_qemu() {
        assert_eq!(CbFlags::NoRegs.call_flags(), call_flags::NO_RWG);
        assert_eq!(CbFlags::RRegs.call_flags(), call_flags::NO_WG);
        assert_eq!(CbFlags::RwRegs.at_run_time(), CbFlags::RwRegsPc);
        assert_eq!(CbFlags::from_raw(7), CbFlags::NoRegs);
        let oi = MemOpIdx::new(ruvm_jit_core::MemOp(3), 1);
        assert_eq!(make_plugin_meminfo(oi, MEM_W) >> 16, MEM_W);
    }
}
