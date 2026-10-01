// SPDX-License-Identifier: GPL-2.0-or-later

//! The generic translator loop, `accel/tcg/translator.c`: the contract between the runtime and
//! a guest front end ([`TranslatorOps`]), the block prologue and epilogue that check for exit
//! requests, `can_do_io` management, and the `translator_ld*` code loads with QEMU's page
//! crossing rules.
//!
//! Differences from QEMU:
//!
//! - The target's `DisasContext` is the [`TranslatorOps`] implementor itself, and the generic
//!   [`DisasContextBase`] is passed next to it instead of being embedded in it.
//! - A guest fault while fetching code is returned as a [`CpuLoopExit`] from the hook, instead
//!   of a longjmp out of the translator.
//! - The "host address" of a code page is its `ram_addr`, read through the runtime's RAM
//!   registry. There is no plugin byte recording.

use ruvm_jit_core::Opcode;
use ruvm_jit_core::ir::OpId;
use ruvm_jit_core::types::{Cond, Type, tb_exit};
use ruvm_mem::Endian;

use crate::cpu::{Cpu, CpuLoopExit, Ra};
use crate::cputlb;
use crate::translate::TbBuild;
use crate::{ENV_CAN_DO_IO_OFFSET, ENV_ICOUNT_DECR_OFFSET, OPC_MAX_SIZE, cf};

/// How translation of a block should continue, `DisasJumpType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DisasJumpType {
    /// `DISAS_NEXT`: translate the next instruction.
    Next,
    /// `DISAS_TOO_MANY`: stop; the block ends by going to the next instruction.
    TooMany,
    /// `DISAS_NORETURN`: stop; the code already left the block.
    NoReturn,
    /// `DISAS_TARGET_0` and up: target specific endings.
    Target(u32),
}

/// The generic part of the disassembly context, `DisasContextBase`.
#[derive(Debug)]
pub struct DisasContextBase<'a> {
    /// The block being built; its [`TbBuild::f`] is where ops go.
    pub tb: &'a mut TbBuild,
    /// The PC of the first instruction.
    pub pc_first: u64,
    /// The PC of the next instruction to translate.
    pub pc_next: u64,
    /// How to go on after the current instruction.
    pub is_jmp: DisasJumpType,
    /// Instructions translated so far, the current one included.
    pub num_insns: u32,
    /// The most instructions the block may hold.
    pub max_insns: u32,
    /// The `insn_start` op of the current instruction.
    pub insn_start: Option<OpId>,
    /// `ram_addr` of `pc_first`, and of the start of the second page once it is used.
    host_addr: [Option<u64>; 2],
    page_mask: u64,
}

impl DisasContextBase<'_> {
    /// `is_same_page()`: whether `addr` is on the page of the first instruction.
    pub fn is_same_page(&self, addr: u64) -> bool {
        (addr ^ self.pc_first) & self.page_mask == 0
    }

    /// `translator_use_goto_tb()`: whether a direct jump to `dest` may be chained.
    pub fn translator_use_goto_tb(&self, dest: u64) -> bool {
        // Suppress goto_tb if requested.
        if self.tb.cflags & cf::NO_GOTO_TB != 0 {
            return false;
        }
        // Check for the dest on the same page as the start of the TB.
        (self.pc_first ^ dest) & self.page_mask == 0
    }

    /// `translator_io_start()`: the current instruction does IO, so it must end the block.
    pub fn translator_io_start(&mut self) -> bool {
        // Ensure that this instruction will be the last in the TB. The target may override
        // this to something more forceful.
        if self.is_jmp == DisasJumpType::Next {
            self.is_jmp = DisasJumpType::TooMany;
        }
        true
    }

    /// `translator_access()`: the `ram_addr` of `len` bytes at `pc`, or `None` for the slow
    /// path.
    fn translator_access(
        &mut self,
        cpu: &mut Cpu<'_>,
        pc: u64,
        len: u64,
    ) -> Result<Option<u64>, CpuLoopExit> {
        // Use slow path if first page is MMIO.
        if self.tb.page_addr[0] == u64::MAX {
            return Ok(None);
        }
        let end = pc.wrapping_add(len - 1);
        let (host, base);
        if self.is_same_page(end) {
            host = self.host_addr[0];
            base = self.pc_first;
        } else {
            base = (self.pc_first & self.page_mask).wrapping_add(!self.page_mask + 1);
            if self.host_addr[1].is_none() {
                let new_page1 = cputlb::get_page_addr_code(cpu, base)?;
                // If the second page is MMIO, treat as if the first page was MMIO as well, so
                // that we do not cache the TB.
                if new_page1 == u64::MAX {
                    self.tb.page_addr[0] = u64::MAX;
                    // Require that this be the final insn.
                    self.max_insns = self.num_insns;
                    return Ok(None);
                }
                self.tb.page_addr[1] = new_page1;
                self.host_addr[1] = Some(new_page1);
            }
            host = self.host_addr[1];
            // Use slow path when crossing pages.
            if self.is_same_page(pc) {
                return Ok(None);
            }
        }
        debug_assert!(pc >= base);
        Ok(host.map(|h| h + (pc - base)))
    }

    /// Fetch `buf.len()` code bytes at `pc` in memory order, `translator_ld()` with the
    /// `cpu_ld*_code_mmu()` fallback.
    pub fn translator_ld(
        &mut self,
        cpu: &mut Cpu<'_>,
        pc: u64,
        buf: &mut [u8],
    ) -> Result<(), CpuLoopExit> {
        if let Some(ram_addr) = self.translator_access(cpu, pc, buf.len() as u64)? {
            let jit = cpu.jit();
            if let Some((block, off)) = jit.ram_block_from_addr(ram_addr) {
                if block.read(off, buf).is_ok() {
                    return Ok(());
                }
            }
        }
        let ops = cpu.ops();
        let mmu_idx = ops.mmu_index(cpu, true);
        cputlb::cpu_ld_code(cpu, pc, buf, mmu_idx, Ra::None)
    }

    /// `translator_ldub()`.
    pub fn translator_ldub(&mut self, cpu: &mut Cpu<'_>, pc: u64) -> Result<u8, CpuLoopExit> {
        let mut b = [0u8; 1];
        self.translator_ld(cpu, pc, &mut b)?;
        Ok(b[0])
    }

    /// `translator_lduw_end()`.
    pub fn translator_lduw(
        &mut self,
        cpu: &mut Cpu<'_>,
        pc: u64,
        endian: Endian,
    ) -> Result<u16, CpuLoopExit> {
        let mut b = [0u8; 2];
        self.translator_ld(cpu, pc, &mut b)?;
        Ok(match endian {
            Endian::Little => u16::from_le_bytes(b),
            Endian::Big => u16::from_be_bytes(b),
        })
    }

    /// `translator_ldl_end()`.
    pub fn translator_ldl(
        &mut self,
        cpu: &mut Cpu<'_>,
        pc: u64,
        endian: Endian,
    ) -> Result<u32, CpuLoopExit> {
        let mut b = [0u8; 4];
        self.translator_ld(cpu, pc, &mut b)?;
        Ok(match endian {
            Endian::Little => u32::from_le_bytes(b),
            Endian::Big => u32::from_be_bytes(b),
        })
    }

    /// `translator_ldq_end()`.
    pub fn translator_ldq(
        &mut self,
        cpu: &mut Cpu<'_>,
        pc: u64,
        endian: Endian,
    ) -> Result<u64, CpuLoopExit> {
        let mut b = [0u8; 8];
        self.translator_ld(cpu, pc, &mut b)?;
        Ok(match endian {
            Endian::Little => u64::from_le_bytes(b),
            Endian::Big => u64::from_be_bytes(b),
        })
    }

    /// Emit `st8 val, env, can_do_io` before `op`.
    fn set_can_do_io_before(&mut self, op: OpId, val: bool) {
        let f = &mut self.tb.f;
        let c = f.constant_i32(i32::from(val));
        let env = f.env();
        let id = f.insert_before(op, Opcode::St8, Type::I32, 3);
        let args = &mut f.op_mut(id).args;
        args[0] = c.arg();
        args[1] = env.arg();
        args[2] = ENV_CAN_DO_IO_OFFSET as u64;
    }
}

/// The front end hooks, `TranslatorOps`. The implementor is the target's `DisasContext`.
pub trait TranslatorOps {
    /// `init_disas_context`: set up the target state from the block. May lower
    /// `db.max_insns`.
    fn init_disas_context(&mut self, db: &mut DisasContextBase<'_>, cpu: &mut Cpu<'_>) {
        let _ = (db, cpu);
    }

    /// `tb_start`: emit code at the start of the block.
    fn tb_start(&mut self, db: &mut DisasContextBase<'_>, cpu: &mut Cpu<'_>) {
        let _ = (db, cpu);
    }

    /// `insn_start`: emit the `insn_start` op of the instruction at `db.pc_next`.
    fn insn_start(&mut self, db: &mut DisasContextBase<'_>, cpu: &mut Cpu<'_>);

    /// `translate_insn`: translate one instruction, advance `db.pc_next` and set `db.is_jmp`
    /// to end the block. A fault while fetching the instruction is returned.
    fn translate_insn(
        &mut self,
        db: &mut DisasContextBase<'_>,
        cpu: &mut Cpu<'_>,
    ) -> Result<(), CpuLoopExit>;

    /// `tb_stop`: emit the end of the block for `db.is_jmp`.
    fn tb_stop(&mut self, db: &mut DisasContextBase<'_>, cpu: &mut Cpu<'_>);
}

/// `translator_loop()`: translate the block `tb` describes with the front end `ops`.
pub fn translator_loop(
    cpu: &mut Cpu<'_>,
    tb: &mut TbBuild,
    ops: &mut dyn TranslatorOps,
) -> Result<(), CpuLoopExit> {
    let cflags = tb.cflags;
    let page_mask = cpu.core.jit.page_mask();
    let host0 = if tb.page_addr[0] == u64::MAX { None } else { Some(tb.page_addr[0]) };
    let pc = tb.pc;
    let max_insns = tb.max_insns;
    let mut db = DisasContextBase {
        tb,
        pc_first: pc,
        pc_next: pc,
        is_jmp: DisasJumpType::Next,
        num_insns: 0,
        max_insns,
        insn_start: None,
        host_addr: [host0, None],
        page_mask,
    };
    ops.init_disas_context(&mut db, cpu);
    // No early exit.
    debug_assert_eq!(db.is_jmp, DisasJumpType::Next);

    // Start translating.
    let exitreq_label = gen_tb_start(&mut db, cflags);
    ops.tb_start(&mut db, cpu);
    debug_assert_eq!(db.is_jmp, DisasJumpType::Next);

    let mut first_insn_start = None;
    loop {
        db.num_insns += 1;
        db.tb.icount = db.num_insns as u16;
        ops.insn_start(&mut db, cpu);
        db.insn_start = db.tb.f.last_op();
        if first_insn_start.is_none() {
            first_insn_start = db.insn_start;
        }
        debug_assert_eq!(db.is_jmp, DisasJumpType::Next);

        // Disassemble one instruction. The translate_insn hook should update db.pc_next and
        // db.is_jmp to indicate what should be done next: either exiting this loop or locate
        // the start of the next instruction.
        ops.translate_insn(&mut db, cpu)?;

        // Stop translation if translate_insn so indicated.
        if db.is_jmp != DisasJumpType::Next {
            break;
        }

        // Stop translation if the output buffer is full, or we have executed all of the
        // allowed instructions.
        if db.tb.f.nb_ops() >= OPC_MAX_SIZE || db.num_insns >= db.max_insns {
            db.is_jmp = DisasJumpType::TooMany;
            break;
        }
    }

    // Emit code to exit the TB, as indicated by db.is_jmp.
    ops.tb_stop(&mut db, cpu);
    if let Some(l) = exitreq_label {
        let id = db.tb.id;
        db.tb.f.gen_set_label(l);
        db.tb.f.gen_exit_tb(id, tb_exit::REQUESTED);
    }

    // Manage can_do_io for the translation block: set to false before the first insn and set
    // to true before the last insn.
    let last = db.insn_start.expect("a block has at least one instruction");
    if db.num_insns == 1 {
        debug_assert_eq!(first_insn_start, db.insn_start);
    } else {
        let first = first_insn_start.expect("a block has at least one instruction");
        debug_assert_ne!(first, last);
        db.set_can_do_io_before(first, false);
    }
    db.set_can_do_io_before(last, true);

    // The disas_log hook may use these values rather than recompute.
    db.tb.size = db.pc_next.wrapping_sub(db.pc_first) as u32;
    db.tb.icount = db.num_insns as u16;
    Ok(())
}

/// `gen_tb_start()`: unless the block is uninterruptible, leave it at once when the exit
/// request half of `icount_decr` is set.
fn gen_tb_start(db: &mut DisasContextBase<'_>, cflags: u32) -> Option<ruvm_jit_core::Label> {
    if cflags & cf::NOIRQ != 0 {
        return None;
    }
    let f = &mut db.tb.f;
    let count = f.temp_new_i32();
    let env = f.env();
    f.gen_ld_i32(count, env, ENV_ICOUNT_DECR_OFFSET);
    let l = f.new_label();
    f.gen_brcondi_i32(Cond::Lt, count, 0, l);
    Some(l)
}
