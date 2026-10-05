// SPDX-License-Identifier: GPL-2.0-or-later

//! The code generator contract, the part of `tcg/tcg.c` and `tcg-target.c.inc` the runtime
//! calls: `tcg_gen_code()`, `tb_target_set_jmp_target()` and the prologue that enters a
//! block, with [`InterpBackend`], an implementation on top of `ruvm-jit-interp`.
//!
//! Differences from QEMU:
//!
//! - A backend owns the whole of entering and leaving generated code, `cpu_tb_exec()`'s
//!   `tcg_qemu_tb_exec()` call, so it can follow `goto_tb` and `goto_ptr` itself.
//! - A block's code is a boxed value the backend downcasts, not a pointer into a buffer. The
//!   size charged to the code region is what the backend reports. [`InterpBackend`] charges
//!   16 bytes per op.
//! - [`InterpBackend`] replaces the interpreter's `lookup_tb_ptr` with the runtime's, so
//!   `lookup_and_goto_ptr` chains blocks as in QEMU.
//! - [`Backend::tb_created`] and [`Backend::tb_flush`] tell the backend about the block that
//!   owns code it made and about a flush, which QEMU's backends do not need because the code
//!   buffer and the `TranslationBlock` are laid out together.

use std::any::Any;
use std::fmt;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, Weak};

use ruvm_jit_core::{Func, HelperType};
use ruvm_jit_interp::{Exit, FaultKind, HelperEnv, HelperRegistry, Machine, Unwind};

use crate::ENV_ICOUNT_DECR_OFFSET;
use crate::cpu::{Cpu, CpuLoopExit, MmuAccessType, Ra};
use crate::cpu_exec;
use crate::tb::{Tb, lock};

/// Why [`Backend::gen_code`] failed, the negative returns of `tcg_gen_code()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GenCodeError {
    /// -1: the code buffer is full. The runtime flushes it and translates again.
    BufferFull,
    /// -2: the block is too large. The runtime translates again with half the instructions.
    TooLarge,
}

/// How a run of generated code ended, `tcg_qemu_tb_exec()`'s return value split in two.
#[derive(Clone, Debug)]
pub struct TbRet {
    /// The last block that ran, or `None` when the code returned 0.
    pub last_tb: Option<Arc<Tb>>,
    /// The `TB_EXIT_*` index in the low bits of the return value.
    pub exit: u64,
}

/// A code generator.
pub trait Backend: Send + Sync + fmt::Debug {
    /// `tcg_gen_code()`: turn the finished IR of block `id` into code and report the size to
    /// charge to the code region.
    fn gen_code(
        &self,
        f: Func,
        id: u64,
    ) -> Result<(Box<dyn Any + Send + Sync>, usize), GenCodeError>;

    /// `tb_set_jmp_target()`: point `goto_tb` slot `n` of `tb` at `dest`, or back at the code
    /// after the jump for `None`.
    fn set_jmp_target(&self, tb: &Arc<Tb>, n: usize, dest: Option<&Arc<Tb>>);

    /// Run `tb` and whatever it chains to until the code returns to the execution loop.
    fn exec(&self, cpu: &mut Cpu<'_>, tb: &Arc<Tb>) -> Result<TbRet, CpuLoopExit>;

    /// The block whose code [`Backend::gen_code`] made was created as `tb`, so that the code
    /// can name it. The default does nothing.
    fn tb_created(&self, tb: &Arc<Tb>) {
        let _ = tb;
    }

    /// `tcg_region_reset_all()`: every block was dropped, and none is running. Code made from
    /// now on is never chained to code made before. The default does nothing.
    fn tb_flush(&self) {}

    /// `TCG_TARGET_DEFAULT_MO`: the memory orders the host gives without barriers.
    fn target_default_mo(&self) -> u32 {
        0
    }

    /// How blocks of a guest whose memory order is `guest_mo` keep it on this host; see
    /// [`ruvm_jit_core::memory_model`]. QEMU's mapping by default.
    fn fence_mapping(&self, guest_mo: u32) -> ruvm_jit_core::FenceMapping {
        let _ = guest_mo;
        ruvm_jit_core::FenceMapping::Qemu
    }

    /// The helpers of an interpreter backend, so that the runtime can make a native backend
    /// with the same ones; `None` for other backends.
    fn interp_helpers(&self) -> Option<&HelperRegistry> {
        None
    }

    /// For a native backend, how many blocks were compiled to host code and how many fell
    /// back to the interpreter; `None` for other backends.
    fn native_stats(&self) -> Option<(u64, u64)> {
        None
    }
}

/// The code of a block for [`InterpBackend`].
struct InterpCode {
    func: Func,
    targets: Mutex<[Option<Weak<Tb>>; 2]>,
}

/// A backend that runs blocks with the IR interpreter, like QEMU's TCI.
pub struct InterpBackend {
    helpers: HelperRegistry,
}

impl fmt::Debug for InterpBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InterpBackend").field("helpers", &self.helpers.len()).finish()
    }
}

impl Default for InterpBackend {
    fn default() -> InterpBackend {
        InterpBackend::new()
    }
}

/// The runtime's `helper_lookup_tb_ptr()`.
pub(crate) fn lookup_tb_ptr(h: &mut HelperEnv<'_>, _args: &[u64]) -> Result<u128, Unwind> {
    let Some(mut cpu) = Cpu::from_helper_env(h) else { return Ok(0) };
    match cpu_exec::helper_lookup_tb_ptr(&mut cpu) {
        Ok(Some(tb)) => {
            let id = tb.id;
            cpu.core.goto_ptr_target = Some(tb);
            Ok(u128::from(id))
        }
        Ok(None) => Ok(0),
        Err(e) => Err(cpu.unwind(e)),
    }
}

impl InterpBackend {
    /// A backend with the interpreter's built-in helpers.
    pub fn new() -> InterpBackend {
        InterpBackend::with_helpers(HelperRegistry::new())
    }

    /// A backend with `helpers`, which should include the interpreter's built-in ones. The
    /// target's helpers go here. `lookup_tb_ptr` is replaced by the runtime's.
    pub fn with_helpers(mut helpers: HelperRegistry) -> InterpBackend {
        helpers.register("lookup_tb_ptr", HelperType::Ptr, &[HelperType::Ptr], lookup_tb_ptr);
        crate::plugin::register_helpers(&mut helpers);
        InterpBackend { helpers }
    }

    fn unwind(&self, cpu: &mut Cpu<'_>, u: Unwind) -> CpuLoopExit {
        unwind(cpu, u)
    }
}

/// Turn the reason generated code left with into the `cpu_loop_exit()` it stands for.
pub(crate) fn unwind(cpu: &mut Cpu<'_>, u: Unwind) -> CpuLoopExit {
    if let Some(e) = cpu.core.unwinding.take() {
        return e;
    }
    match u {
        Unwind::Mem(f) if f.kind == FaultKind::Unaligned => {
            let ops = cpu.ops();
            let at = if f.write { MmuAccessType::DataStore } else { MmuAccessType::DataLoad };
            ops.do_unaligned_access(cpu, f.addr, at, f.oi.mmu_idx() as usize, Ra::Tb)
        }
        Unwind::Mem(f) => {
            panic!("guest memory fault at {:#x} did not leave through the softmmu", f.addr)
        }
        Unwind::Exception(code) => cpu.raise_exception(code as i32, Ra::Tb),
        Unwind::ExitAtomic => cpu.cpu_loop_exit_atomic(Ra::Tb),
    }
}

/// Copy the shared `icount_decr` into `env`, where generated code reads it.
pub(crate) fn copy_icount_decr(cpu: &mut Cpu<'_>) {
    let decr = cpu.core.shared.icount_decr.load(Ordering::Acquire);
    let off = ENV_ICOUNT_DECR_OFFSET as usize;
    cpu.env[off..off + 4].copy_from_slice(&decr.to_le_bytes());
}

/// The `goto_tb` targets of a block that are still alive.
pub(crate) fn live_targets(targets: &Mutex<[Option<Weak<Tb>>; 2]>) -> [Option<Arc<Tb>>; 2] {
    let t = lock(targets);
    [t[0].as_ref().and_then(Weak::upgrade), t[1].as_ref().and_then(Weak::upgrade)]
}

fn interp_code(tb: &Tb) -> &InterpCode {
    tb.code().downcast_ref::<InterpCode>().expect("block was not generated by InterpBackend")
}

impl Backend for InterpBackend {
    fn gen_code(
        &self,
        f: Func,
        _id: u64,
    ) -> Result<(Box<dyn Any + Send + Sync>, usize), GenCodeError> {
        let size = f.nb_ops() * 16;
        if size > usize::from(u16::MAX) {
            return Err(GenCodeError::TooLarge);
        }
        let code = InterpCode { func: f, targets: Mutex::new([None, None]) };
        Ok((Box::new(code), size))
    }

    fn set_jmp_target(&self, tb: &Arc<Tb>, n: usize, dest: Option<&Arc<Tb>>) {
        lock(&interp_code(tb).targets)[n] = dest.map(Arc::downgrade);
    }

    fn interp_helpers(&self) -> Option<&HelperRegistry> {
        Some(&self.helpers)
    }

    fn exec(&self, cpu: &mut Cpu<'_>, tb: &Arc<Tb>) -> Result<TbRet, CpuLoopExit> {
        let mut tb = tb.clone();
        let r = loop {
            cpu.core.current_tb = Some(tb.clone());
            cpu.core.cur_insn = None;
            let decr = cpu.core.shared.icount_decr.load(Ordering::Acquire);
            let off = ENV_ICOUNT_DECR_OFFSET as usize;
            cpu.env[off..off + 4].copy_from_slice(&decr.to_le_bytes());
            let code = interp_code(&tb);
            let targets: [Option<Arc<Tb>>; 2] = {
                let t = lock(&code.targets);
                [t[0].as_ref().and_then(Weak::upgrade), t[1].as_ref().and_then(Weak::upgrade)]
            };
            let exit = {
                let mut m = Machine::new(&mut *cpu.env, &mut *cpu.core, &self.helpers);
                m.linked = [targets[0].is_some(), targets[1].is_some()];
                m.run(&code.func)
            };
            match exit {
                Ok(Exit::ExitTb(v)) => {
                    let id = v & !3;
                    if id == 0 {
                        break Ok(TbRet { last_tb: None, exit: v & 3 });
                    }
                    assert_eq!(id, tb.id, "exit_tb names a block that is not running");
                    break Ok(TbRet { last_tb: Some(tb), exit: v & 3 });
                }
                Ok(Exit::GotoTb(n)) => {
                    let [t0, t1] = targets;
                    let next = if n == 0 { t0 } else { t1 };
                    tb = next.expect("goto_tb on an unlinked slot");
                }
                Ok(Exit::GotoPtr(0)) => break Ok(TbRet { last_tb: None, exit: 0 }),
                Ok(Exit::GotoPtr(p)) => {
                    let next = cpu.core.goto_ptr_target.take().expect("goto_ptr without a lookup");
                    assert_eq!(next.id, p, "goto_ptr to a block lookup_tb_ptr did not return");
                    tb = next;
                }
                Ok(Exit::Unwind(u)) => break Err(self.unwind(cpu, u)),
                Err(e) => panic!("interpreter error in the block at pc {:#x}: {e}", tb.pc),
            }
        };
        cpu.core.current_tb = None;
        cpu.core.goto_ptr_target = None;
        r
    }
}
