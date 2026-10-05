// SPDX-License-Identifier: GPL-2.0-or-later

//! The x86 translator: a port of the integer and system parts of `target/i386/tcg/translate.c`
//! and `emit.c.inc`.
//!
//! The decoder is the table free, one switch per opcode decoder of `disas_insn()` from before
//! QEMU moved to `decode-new.c.inc`; the code each instruction generates follows the current
//! emitters. The guest registers, EIP, the segment bases and the lazy flags state (`cc_dst`,
//! `cc_src`, `cc_src2` and `cc_op`) are TCG globals, as in QEMU; everything else in `env` is
//! loaded and stored by offset.
//!
//! This file holds the shared state, the flags machinery, the address and stack helpers and
//! the end of block code; [`insn`] holds the per opcode decoder and emitters, [`ext`] the
//! later extensions with their own decoding and [`sse`] the table driven decoder of the vector
//! instructions.
//!
//! Differences from QEMU:
//!
//! - A near `jcc` does not always end the block (superblocks). Up to `MAX_SIDE_EXITS` times
//!   per block, the block goes on with the fall-through path, and the taken edge branches to
//!   an exit emitted after the last instruction. The fall-through path keeps guest registers
//!   in host registers and the flags state across the branch, and skips the exit request
//!   check a new block would make. Control only moves forward inside a block, so it still
//!   cannot loop without passing that check. Not with icount, single step, an interrupt
//!   shadow or RF set. The exits get the two `goto_tb` slots in program order, and the
//!   others go through the jump cache.
//! - A near jump, call or return that leaves the CPU state as the block found it apart from
//!   EIP looks the next block up through an inline cache, `lookup_tb_ptr_ic`.

mod ext;
mod insn;
mod sse;
mod sse_tab;
mod x87;

use ruvm_jit::{Cpu, CpuLoopExit, DisasContextBase, DisasJumpType, TranslatorOps, cf};
use ruvm_jit_core::ir::{TempI32, TempI64, TempPtr};
use ruvm_jit_core::{Cond, Func, Label, MemOp, OpId, Temp};

use super::cc::{
    CC_OP_ADCOX, CC_OP_ADCX, CC_OP_ADDB, CC_OP_ADOX, CC_OP_BLSIB, CC_OP_BMILGB, CC_OP_CLR,
    CC_OP_DECB, CC_OP_DYNAMIC, CC_OP_EFLAGS, CC_OP_INCB, CC_OP_LOGICB, CC_OP_MULB, CC_OP_POPCNT,
    CC_OP_POPCNTB, CC_OP_SARB, CC_OP_SBB_SELFB, CC_OP_SHLB, CC_OP_SUBB, cc_op_size,
};
use super::env::{
    AC_MASK, CC_C, CC_DST, CC_O, CC_OP, CC_P, CC_S, CC_SRC, CC_SRC2, CC_Z, EFLAGS, EIP,
    HF_INHIBIT_IRQ_MASK, HF_SMAP_MASK, HFLAGS, RF_MASK, SEG_BASE, SEG_SELECTOR, TF_MASK, VM_MASK,
    reg, seg,
};
use super::helpers::{self, Def};
use super::{EXCP0D_GPF, EXCP06_ILLOP, MMU_KNOSMAP64_IDX, MMU_KSMAP64_IDX, MMU_USER64_IDX};
use crate::cpuid::X86Cpu;
use crate::state::{
    HF_ADDSEG_MASK, HF_CPL_MASK, HF_CS32_MASK, HF_CS64_MASK, HF_LMA_MASK, HF_PE_MASK, HF_SS32_MASK,
    R_ECX, R_ESP, R_SS,
};

/// Operand sizes, `MO_8` to `MO_64`.
const OT8: u32 = 0;
const OT16: u32 = 1;
const OT32: u32 = 2;
const OT64: u32 = 3;

const PREFIX_REPZ: u32 = 0x01;
const PREFIX_REPNZ: u32 = 0x02;
const PREFIX_LOCK: u32 = 0x04;
const PREFIX_DATA: u32 = 0x08;
const PREFIX_ADR: u32 = 0x10;
const PREFIX_REX: u32 = 0x40;
const PREFIX_VEX: u32 = 0x80;

/// Stop after this instruction with EIP already set: `DISAS_EOB_ONLY`.
const DISAS_EOB_ONLY: DisasJumpType = DisasJumpType::Target(0);
/// Stop after this instruction, which falls through: `DISAS_EOB_NEXT`.
const DISAS_EOB_NEXT: DisasJumpType = DisasJumpType::Target(1);
/// `DISAS_EOB_NEXT` that also inhibits interrupts for one instruction.
const DISAS_EOB_INHIBIT_IRQ: DisasJumpType = DisasJumpType::Target(2);
/// Stop and check TF again after SYSCALL and SYSRET: `DISAS_EOB_RECHECK_TF`.
const DISAS_EOB_RECHECK_TF: DisasJumpType = DisasJumpType::Target(3);
/// An indirect jump with EIP already set: `DISAS_JUMP`.
const DISAS_JUMP: DisasJumpType = DisasJumpType::Target(4);

const JCC_O: u32 = 0;
const JCC_B: u32 = 1;
const JCC_Z: u32 = 2;
const JCC_BE: u32 = 3;
const JCC_S: u32 = 4;
const JCC_P: u32 = 5;
const JCC_L: u32 = 6;
const JCC_LE: u32 = 7;

/// The longest instruction, `X86_MAX_INSN_LENGTH`.
const MAX_INSN_LENGTH: u64 = 15;

const REG_NAMES: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];
const SEG_NAMES: [&str; 6] = ["es_base", "cs_base", "ss_base", "ds_base", "fs_base", "gs_base"];

/// The CPUID features the decoder looks at.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Feat {
    popcnt: bool,
    abm: bool,
    bmi1: bool,
    cx8: bool,
    cx16: bool,
    rdtscp: bool,
    lahf_lm: bool,
    smap: bool,
    intel: bool,
    sse: bool,
    sse2: bool,
    sse42: bool,
    movbe: bool,
    adx: bool,
    bmi2: bool,
    rdrand: bool,
    rdseed: bool,
    rdpid: bool,
    xsave: bool,
    fsgsbase: bool,
    sse3: bool,
    ssse3: bool,
    sse41: bool,
    sse4a: bool,
    avx: bool,
    avx2: bool,
    fma: bool,
    f16c: bool,
    aes: bool,
    vaes: bool,
    pclmulqdq: bool,
    sha_ni: bool,
    cmpccxadd: bool,
    fxsr: bool,
    xsaveopt: bool,
    cmov: bool,
}

/// The TCG globals and the block wide temps.
#[derive(Clone, Copy, Debug)]
struct G {
    env: TempPtr,
    regs: [TempI64; 16],
    eip: TempI64,
    cc_dst: TempI64,
    cc_src: TempI64,
    cc_src2: TempI64,
    cc_op: TempI32,
    seg_base: [TempI64; 6],
    t0: TempI64,
    t1: TempI64,
    a0: TempI64,
    cc_srct: TempI64,
    tmp0: TempI64,
    tmp4: TempI64,
}

/// Why the decoder gave up on an instruction.
enum Stop {
    /// More than 15 bytes: #GP.
    TooLong,
    /// The instruction crosses into a new page and is not the first of the block: end the
    /// block before it.
    Retry,
    /// A fault while fetching the instruction.
    Exit(CpuLoopExit),
}

type R<T = ()> = Result<T, Stop>;

/// The per block translator state, `DisasContext`.
pub(crate) struct DisasContext {
    feat: Feat,
    g: Option<G>,
    // Block constants.
    flags: u32,
    cs_base: u64,
    pe: bool,
    code32: bool,
    code64: bool,
    ss32: bool,
    addseg: bool,
    vm86: bool,
    lma: bool,
    cpl: u32,
    iopl: u32,
    mem_index: u32,
    jmp_opt: bool,
    repz_opt: bool,
    // Lazy flags.
    cc_op: u32,
    cc_op_dirty: bool,
    // The current instruction.
    pc_start: u64,
    pc: u64,
    prefix: u32,
    aflag: u32,
    dflag: u32,
    rex_r: usize,
    rex_x: usize,
    rex_b: usize,
    rex_w: bool,
    vex_l: bool,
    vex_v: usize,
    vex_w: bool,
    override_seg: i32,
    rip_offset: u64,
    popl_esp_hack: i64,
    prev_insn_start: Option<OpId>,
    prev_insn_end: Option<OpId>,
    /// The `DISAS_JUMP` that ends the block is a near jump, call or return, which leaves the
    /// CPU state other than EIP as the block found it, so the next block may be cached where
    /// it is looked up (`gen_lookup_and_goto_ptr_ic`). Not in QEMU.
    jmp_ic: bool,
    /// The `goto_tb` exit slots this block has used, one bit per slot. Not in QEMU, which
    /// has at most two direct exits per block by construction; superblocks can have more
    /// exits than slots, and the rest go through the jump cache.
    goto_tb_used: u8,
    /// Conditional branches whose fall-through was translated in line, see
    /// [`SideExit`]. Not in QEMU.
    side_exits: Vec<SideExit>,
}

/// The taken edge of a conditional branch whose fall-through path the block continues
/// with (a superblock, not in QEMU). Its exit is emitted out of line when the block
/// ends, so the fall-through path keeps guest registers in host registers.
#[derive(Clone, Copy)]
struct SideExit {
    label: Label,
    new_eip: u64,
    new_pc: u64,
    flags: u32,
    /// The `goto_tb` slot kept for this exit.
    slot: Option<u64>,
}

impl Feat {
    /// The features of `model`, looked up by name once per CPU rather than per block.
    pub(crate) fn of(model: &X86Cpu) -> Feat {
        Feat {
            popcnt: model.has_feature("popcnt"),
            abm: model.has_feature("abm"),
            bmi1: model.has_feature("bmi1"),
            cx8: model.has_feature("cx8"),
            cx16: model.has_feature("cx16"),
            rdtscp: model.has_feature("rdtscp"),
            lahf_lm: model.has_feature("lahf-lm"),
            smap: model.has_feature("smap"),
            intel: model.is_intel(),
            sse: model.has_feature("sse"),
            sse2: model.has_feature("sse2"),
            sse42: model.has_feature("sse4.2"),
            movbe: model.has_feature("movbe"),
            adx: model.has_feature("adx"),
            bmi2: model.has_feature("bmi2"),
            rdrand: model.has_feature("rdrand"),
            rdseed: model.has_feature("rdseed"),
            rdpid: model.has_feature("rdpid"),
            xsave: model.has_feature("xsave"),
            fsgsbase: model.has_feature("fsgsbase"),
            sse3: model.has_feature("pni"),
            ssse3: model.has_feature("ssse3"),
            sse41: model.has_feature("sse4.1"),
            sse4a: model.has_feature("sse4a"),
            avx: model.has_feature("avx"),
            avx2: model.has_feature("avx2"),
            fma: model.has_feature("fma"),
            f16c: model.has_feature("f16c"),
            aes: model.has_feature("aes"),
            vaes: model.has_feature("vaes"),
            pclmulqdq: model.has_feature("pclmulqdq"),
            sha_ni: model.has_feature("sha-ni"),
            cmpccxadd: model.has_feature("cmpccxadd"),
            fxsr: model.has_feature("fxsr"),
            xsaveopt: model.has_feature("xsaveopt"),
            cmov: model.has_feature("cmov"),
        }
    }
}

impl DisasContext {
    /// A context for a block decoded with the features `feat`, see [`Feat::of`].
    pub(crate) fn new(feat: Feat) -> DisasContext {
        DisasContext {
            feat,
            g: None,
            flags: 0,
            cs_base: 0,
            pe: false,
            code32: false,
            code64: false,
            ss32: false,
            addseg: false,
            vm86: false,
            lma: false,
            cpl: 0,
            iopl: 0,
            mem_index: 0,
            jmp_opt: true,
            repz_opt: true,
            cc_op: CC_OP_DYNAMIC,
            cc_op_dirty: false,
            pc_start: 0,
            pc: 0,
            prefix: 0,
            aflag: OT32,
            dflag: OT32,
            rex_r: 0,
            rex_x: 0,
            rex_b: 0,
            rex_w: false,
            vex_l: false,
            vex_v: 0,
            vex_w: false,
            override_seg: -1,
            rip_offset: 0,
            popl_esp_hack: 0,
            prev_insn_start: None,
            prev_insn_end: None,
            jmp_ic: false,
            goto_tb_used: 0,
            side_exits: Vec::new(),
        }
    }
}

/// A prepared condition, `CCPrepare`: `cond(reg, reg2)` or `cond(reg, imm)`.
#[derive(Clone, Copy)]
struct Cc {
    cond: Cond,
    reg: TempI64,
    reg2: Option<TempI64>,
    imm: i64,
}

impl Cc {
    fn imm(cond: Cond, reg: TempI64, imm: i64) -> Cc {
        Cc { cond, reg, reg2: None, imm }
    }

    fn regs(cond: Cond, reg: TempI64, reg2: TempI64) -> Cc {
        Cc { cond, reg, reg2: Some(reg2), imm: 0 }
    }
}

/// A decoded memory operand, `AddressParts`.
#[derive(Clone, Copy)]
struct Addr {
    def_seg: i32,
    base: i32,
    index: i32,
    scale: u32,
    disp: i64,
}

/// The translator working on one instruction: the context, the block and the vCPU.
struct S<'a, 'b, 'c> {
    d: &'a mut DisasContext,
    b: &'a mut DisasContextBase<'b>,
    cpu: &'a mut Cpu<'c>,
    g: G,
}

fn mo(ot: u32) -> MemOp {
    MemOp(ot)
}

impl S<'_, '_, '_> {
    // Basic plumbing.

    fn f(&mut self) -> &mut Func {
        &mut self.b.tb.f
    }

    fn new64(&mut self) -> TempI64 {
        self.f().temp_new_i64()
    }

    fn new32(&mut self) -> TempI32 {
        self.f().temp_new_i32()
    }

    fn c64(&mut self, v: i64) -> TempI64 {
        self.f().constant_i64(v)
    }

    fn c32(&mut self, v: i32) -> TempI32 {
        self.f().constant_i32(v)
    }

    fn label(&mut self) -> Label {
        self.f().new_label()
    }

    fn set_label(&mut self, l: Label) {
        self.f().gen_set_label(l);
    }

    /// Declare and call a helper. Helpers that may look at the CPU state see the current
    /// `cc_op`.
    fn call(&mut self, d: &Def, ret: Option<Temp>, args: &[Temp]) {
        if d.flags == 0 {
            self.gen_update_cc_op();
        }
        let f = self.f();
        let h = f.helper(d.info());
        f.gen_call(h, ret, args);
    }

    fn env_call(&mut self, d: &Def, ret: Option<Temp>, args: &[Temp]) {
        let mut v: Vec<Temp> = Vec::with_capacity(args.len() + 1);
        v.push(self.g.env.into());
        v.extend_from_slice(args);
        self.call(d, ret, &v);
    }

    fn trunc32(&mut self, t: TempI64) -> TempI32 {
        let r = self.new32();
        self.f().gen_extrl_i64_i32(r, t);
        r
    }

    fn ld_env32(&mut self, off: usize) -> TempI64 {
        let t = self.new64();
        let env = self.g.env;
        self.f().gen_ld32u_i64(t, env, off as i64);
        t
    }

    fn ld_env64(&mut self, off: usize) -> TempI64 {
        let t = self.new64();
        let env = self.g.env;
        self.f().gen_ld_i64(t, env, off as i64);
        t
    }

    fn st_env32(&mut self, t: TempI64, off: usize) {
        let env = self.g.env;
        self.f().gen_st32_i64(t, env, off as i64);
    }

    fn st_env64(&mut self, t: TempI64, off: usize) {
        let env = self.g.env;
        self.f().gen_st_i64(t, env, off as i64);
    }

    /// `gen_ext_tl()`: extend `src` from `ot` into `dst`.
    fn ext(&mut self, ot: u32, dst: TempI64, src: TempI64, sign: bool) {
        let f = self.f();
        match (ot, sign) {
            (OT8, false) => f.gen_ext8u_i64(dst, src),
            (OT8, true) => f.gen_ext8s_i64(dst, src),
            (OT16, false) => f.gen_ext16u_i64(dst, src),
            (OT16, true) => f.gen_ext16s_i64(dst, src),
            (OT32, false) => f.gen_ext32u_i64(dst, src),
            (OT32, true) => f.gen_ext32s_i64(dst, src),
            _ => f.gen_mov_i64(dst, src),
        }
    }

    fn ext_new(&mut self, ot: u32, src: TempI64, sign: bool) -> TempI64 {
        let t = self.new64();
        self.ext(ot, t, src, sign);
        t
    }

    // Modes.

    fn code64(&self) -> bool {
        self.d.code64
    }

    /// `mo_pushpop()`.
    fn mo_pushpop(&self, ot: u32) -> u32 {
        if self.code64() { if ot == OT16 { OT16 } else { OT64 } } else { ot }
    }

    /// `mo_stacksize()`.
    fn mo_stacksize(&self) -> u32 {
        if self.code64() {
            OT64
        } else if self.d.ss32 {
            OT32
        } else {
            OT16
        }
    }

    /// `insn_const_size()`.
    fn insn_const_size(ot: u32) -> u64 {
        if ot <= OT32 { 1 << ot } else { 4 }
    }

    // Code fetch.

    fn advance_pc(&mut self, n: u64) -> R<u64> {
        let pc = self.d.pc;
        if self.b.num_insns > 1 && !self.b.is_same_page(pc.wrapping_add(n - 1)) {
            return Err(Stop::Retry);
        }
        self.d.pc = pc.wrapping_add(n);
        if self.d.pc.wrapping_sub(self.d.pc_start) > MAX_INSN_LENGTH {
            // If the 16th byte is on another page, a page fault there wins over the #GP.
            let last = self.d.pc.wrapping_sub(1);
            if (last ^ pc.wrapping_sub(1)) & !0xfff != 0 {
                self.b.translator_ldub(self.cpu, last & !0xfff).map_err(Stop::Exit)?;
            }
            return Err(Stop::TooLong);
        }
        Ok(pc)
    }

    fn ldub(&mut self) -> R<u64> {
        let pc = self.advance_pc(1)?;
        self.b.translator_ldub(self.cpu, pc).map(u64::from).map_err(Stop::Exit)
    }

    fn lduw(&mut self) -> R<u64> {
        let lo = self.ldub()?;
        let hi = self.ldub()?;
        Ok(lo | (hi << 8))
    }

    fn ldl(&mut self) -> R<u64> {
        let lo = self.lduw()?;
        let hi = self.lduw()?;
        Ok(lo | (hi << 16))
    }

    fn ldq(&mut self) -> R<u64> {
        let lo = self.ldl()?;
        let hi = self.ldl()?;
        Ok(lo | (hi << 32))
    }

    /// `insn_get()`: an immediate of size `ot`, at most 32 bits, zero extended.
    fn insn_get(&mut self, ot: u32) -> R<u64> {
        match ot {
            OT8 => self.ldub(),
            OT16 => self.lduw(),
            _ => self.ldl(),
        }
    }

    /// `insn_get_signed()`.
    fn insn_get_signed(&mut self, ot: u32) -> R<i64> {
        Ok(match ot {
            OT8 => self.ldub()? as u8 as i8 as i64,
            OT16 => self.lduw()? as u16 as i16 as i64,
            _ => self.ldl()? as u32 as i32 as i64,
        })
    }

    // EIP.

    fn eip_of(&self, pc: u64) -> u64 {
        let eip = pc.wrapping_sub(self.d.cs_base);
        if self.code64() {
            eip
        } else if self.d.code32 {
            eip & 0xffff_ffff
        } else {
            eip & 0xffff
        }
    }

    fn eip_next(&self) -> u64 {
        self.eip_of(self.d.pc)
    }

    fn eip_cur(&self) -> u64 {
        self.eip_of(self.d.pc_start)
    }

    fn cur_insn_len(&self) -> i64 {
        self.d.pc.wrapping_sub(self.d.pc_start) as i64
    }

    /// `gen_update_eip_cur()`.
    fn gen_update_eip_cur(&mut self) {
        let (eip, v) = (self.g.eip, self.eip_cur());
        self.f().gen_movi_i64(eip, v as i64);
    }

    /// `gen_update_eip_next()`.
    fn gen_update_eip_next(&mut self) {
        let (eip, v) = (self.g.eip, self.eip_next());
        self.f().gen_movi_i64(eip, v as i64);
    }

    /// `gen_op_jmp_v()`.
    fn gen_op_jmp_v(&mut self, dest: TempI64) {
        let eip = self.g.eip;
        self.f().gen_mov_i64(eip, dest);
    }

    // Lazy flags.

    fn set_cc_op_1(&mut self, op: u32, dirty: bool) {
        if self.d.cc_op == op {
            return;
        }
        self.d.cc_op_dirty = dirty;
        self.d.cc_op = op;
    }

    /// `set_cc_op()`.
    fn set_cc_op(&mut self, op: u32) {
        if op == CC_OP_DYNAMIC {
            // Unlike QEMU, spill a pending value here so no caller can lose it.
            self.gen_update_cc_op();
        }
        self.set_cc_op_1(op, op != CC_OP_DYNAMIC);
    }

    /// The `cc_op` global was just written by generated code: forget the static value.
    fn cc_op_now_dynamic(&mut self) {
        self.d.cc_op = CC_OP_DYNAMIC;
        self.d.cc_op_dirty = false;
    }

    /// `assume_cc_op()`: a helper already stored `op`.
    fn assume_cc_op(&mut self, op: u32) {
        self.set_cc_op_1(op, false);
    }

    /// `gen_update_cc_op()`.
    fn gen_update_cc_op(&mut self) {
        if self.d.cc_op_dirty {
            let (g, v) = (self.g.cc_op, self.d.cc_op as i32);
            self.f().gen_movi_i32(g, v);
            self.d.cc_op_dirty = false;
        }
    }

    fn cc_op_value(&mut self) -> TempI32 {
        if self.d.cc_op == CC_OP_DYNAMIC {
            self.g.cc_op
        } else {
            let v = self.d.cc_op as i32;
            self.c32(v)
        }
    }

    /// `gen_compute_eflags()`: every arithmetic flag into `cc_src`, `cc_op` becomes EFLAGS.
    fn gen_compute_eflags(&mut self) {
        if self.d.cc_op == CC_OP_EFLAGS {
            return;
        }
        let g = self.g;
        if self.d.cc_op == CC_OP_CLR {
            self.f().gen_movi_i64(g.cc_src, i64::from(CC_Z | CC_P));
            self.set_cc_op(CC_OP_EFLAGS);
            return;
        }
        let op = self.cc_op_value();
        self.call(
            &helpers::CC_COMPUTE_ALL,
            Some(g.cc_src.into()),
            &[g.cc_dst.into(), g.cc_src.into(), g.cc_src2.into(), op.into()],
        );
        self.set_cc_op(CC_OP_EFLAGS);
    }

    /// `gen_prepare_eflags_c()`.
    fn prepare_eflags_c(&mut self) -> Cc {
        let g = self.g;
        let op = self.d.cc_op;
        match op {
            CC_OP_POPCNTB..=CC_OP_POPCNT | CC_OP_CLR => Cc::imm(Cond::Never, g.cc_src, 0),
            CC_OP_EFLAGS | CC_OP_ADOX => Cc::imm(Cond::TstNe, g.cc_src, i64::from(CC_C)),
            CC_OP_ADCX | CC_OP_ADCOX => Cc::imm(Cond::Ne, g.cc_dst, 0),
            CC_OP_DYNAMIC => self.prepare_c_helper(),
            _ => {
                let size = cc_op_size(op);
                match op & !3 {
                    CC_OP_SUBB => {
                        let a = self.ext_new(size, g.cc_srct, false);
                        let b = self.ext_new(size, g.cc_src, false);
                        Cc::regs(Cond::Ltu, a, b)
                    }
                    CC_OP_ADDB => {
                        let a = self.ext_new(size, g.cc_dst, false);
                        let b = self.ext_new(size, g.cc_src, false);
                        Cc::regs(Cond::Ltu, a, b)
                    }
                    CC_OP_LOGICB => Cc::imm(Cond::Never, g.cc_src, 0),
                    CC_OP_INCB | CC_OP_DECB | CC_OP_SARB => Cc::imm(Cond::TstNe, g.cc_src, 1),
                    CC_OP_SHLB => Cc::imm(Cond::TstNe, g.cc_src, 1i64 << ((8 << size) - 1)),
                    CC_OP_MULB | CC_OP_BLSIB => Cc::imm(Cond::Ne, g.cc_src, 0),
                    CC_OP_BMILGB => Cc::imm(Cond::Eq, g.cc_src, 0),
                    CC_OP_SBB_SELFB => Cc::imm(Cond::Ne, g.cc_dst, 0),
                    _ => self.prepare_c_helper(),
                }
            }
        }
    }

    fn prepare_c_helper(&mut self) -> Cc {
        let g = self.g;
        let r = self.new64();
        let op = self.cc_op_value();
        self.call(
            &helpers::CC_COMPUTE_C,
            Some(r.into()),
            &[g.cc_dst.into(), g.cc_src.into(), g.cc_src2.into(), op.into()],
        );
        Cc::imm(Cond::Ne, r, 0)
    }

    /// `gen_prepare_eflags_p()`.
    fn prepare_eflags_p(&mut self) -> Cc {
        self.gen_compute_eflags();
        Cc::imm(Cond::TstNe, self.g.cc_src, i64::from(CC_P))
    }

    /// `gen_prepare_eflags_s()`.
    fn prepare_eflags_s(&mut self) -> Cc {
        let g = self.g;
        match self.d.cc_op {
            CC_OP_DYNAMIC => {
                self.gen_compute_eflags();
                Cc::imm(Cond::TstNe, g.cc_src, i64::from(CC_S))
            }
            CC_OP_EFLAGS | CC_OP_ADCX | CC_OP_ADOX | CC_OP_ADCOX => {
                Cc::imm(Cond::TstNe, g.cc_src, i64::from(CC_S))
            }
            CC_OP_CLR | CC_OP_POPCNTB..=CC_OP_POPCNT => Cc::imm(Cond::Never, g.cc_src, 0),
            op => {
                let size = cc_op_size(op);
                Cc::imm(Cond::TstNe, g.cc_dst, 1i64 << ((8 << size) - 1))
            }
        }
    }

    /// `gen_prepare_eflags_o()`.
    fn prepare_eflags_o(&mut self) -> Cc {
        let g = self.g;
        match self.d.cc_op {
            CC_OP_ADOX | CC_OP_ADCOX => Cc::imm(Cond::Ne, g.cc_src2, 0),
            CC_OP_CLR | CC_OP_POPCNTB..=CC_OP_POPCNT => Cc::imm(Cond::Never, g.cc_src, 0),
            op if op != CC_OP_DYNAMIC && op & !3 == CC_OP_MULB => Cc::imm(Cond::Ne, g.cc_src, 0),
            op if op != CC_OP_DYNAMIC && op & !3 == CC_OP_LOGICB => {
                Cc::imm(Cond::Never, g.cc_src, 0)
            }
            _ => {
                self.gen_compute_eflags();
                Cc::imm(Cond::TstNe, g.cc_src, i64::from(CC_O))
            }
        }
    }

    /// `gen_prepare_eflags_z()`.
    fn prepare_eflags_z(&mut self) -> Cc {
        let g = self.g;
        match self.d.cc_op {
            CC_OP_DYNAMIC => {
                self.gen_compute_eflags();
                Cc::imm(Cond::TstNe, g.cc_src, i64::from(CC_Z))
            }
            CC_OP_EFLAGS | CC_OP_ADCX | CC_OP_ADOX | CC_OP_ADCOX => {
                Cc::imm(Cond::TstNe, g.cc_src, i64::from(CC_Z))
            }
            CC_OP_CLR => Cc::imm(Cond::Always, g.cc_src, 0),
            CC_OP_POPCNTB..=CC_OP_POPCNT => Cc::imm(Cond::Eq, g.cc_dst, 0),
            op => {
                let size = cc_op_size(op);
                if size == OT64 {
                    Cc::imm(Cond::Eq, g.cc_dst, 0)
                } else {
                    Cc::imm(Cond::TstEq, g.cc_dst, ((1u64 << (8 << size)) - 1) as i64)
                }
            }
        }
    }

    /// `gen_prepare_cc()`: the condition of jcc opcode `b`, in `reg` when a value is
    /// computed.
    fn prepare_cc(&mut self, b: u32) -> Cc {
        let inv = b & 1 != 0;
        let jcc_op = (b >> 1) & 7;
        let g = self.g;
        let op = self.d.cc_op;
        let mut cc = None;
        if op != CC_OP_DYNAMIC && op & !3 == CC_OP_SUBB {
            // We optimize relational operators for the cmp/jcc case.
            let size = cc_op_size(op);
            match jcc_op {
                JCC_BE => {
                    let a = self.ext_new(size, g.cc_srct, false);
                    let b = self.ext_new(size, g.cc_src, false);
                    cc = Some(Cc::regs(Cond::Leu, a, b));
                }
                JCC_L | JCC_LE => {
                    let a = self.ext_new(size, g.cc_srct, true);
                    let b = self.ext_new(size, g.cc_src, true);
                    let c = if jcc_op == JCC_L { Cond::Lt } else { Cond::Le };
                    cc = Some(Cc::regs(c, a, b));
                }
                _ => {}
            }
        }
        let mut cc = match cc {
            Some(c) => c,
            None => match jcc_op {
                JCC_O => self.prepare_eflags_o(),
                JCC_B => self.prepare_eflags_c(),
                JCC_Z => self.prepare_eflags_z(),
                JCC_BE => {
                    self.gen_compute_eflags();
                    Cc::imm(Cond::TstNe, g.cc_src, i64::from(CC_Z | CC_C))
                }
                JCC_S => self.prepare_eflags_s(),
                JCC_P => self.prepare_eflags_p(),
                _ => {
                    // JCC_L and JCC_LE: S != O, or Z.
                    self.gen_compute_eflags();
                    let t = self.new64();
                    let f = self.f();
                    f.gen_shri_i64(t, g.cc_src, 4);
                    f.gen_xor_i64(t, t, g.cc_src);
                    let m = if jcc_op == JCC_L { CC_S } else { CC_S | CC_Z };
                    Cc::imm(Cond::TstNe, t, i64::from(m))
                }
            },
        };
        if inv {
            cc.cond = cc.cond.invert();
        }
        cc
    }

    fn cc_reg2(&mut self, cc: &Cc) -> TempI64 {
        match cc.reg2 {
            Some(r) => r,
            None => self.c64(cc.imm),
        }
    }

    /// `gen_setcc()`: 1 into `reg` when jcc condition `b` holds.
    fn gen_setcc(&mut self, b: u32, reg: TempI64) {
        let cc = self.prepare_cc(b);
        self.setcond_cc(&cc, reg);
    }

    fn setcond_cc(&mut self, cc: &Cc, reg: TempI64) {
        match cc.cond {
            Cond::Never => self.f().gen_movi_i64(reg, 0),
            Cond::Always => self.f().gen_movi_i64(reg, 1),
            c => {
                let r2 = self.cc_reg2(cc);
                self.f().gen_setcond_i64(c, reg, cc.reg, r2);
            }
        }
    }

    /// `gen_compute_eflags_c()`: CF as 0 or 1 into `reg`.
    fn gen_compute_eflags_c(&mut self, reg: TempI64) {
        let cc = self.prepare_eflags_c();
        self.setcond_cc(&cc, reg);
    }

    /// `gen_jcc()` without the jumps: branch to `l` when condition `b` holds, with the
    /// flags state written back first.
    fn gen_jcc1(&mut self, b: u32, l: Label) {
        let cc = self.prepare_cc(b);
        self.gen_update_cc_op();
        match cc.cond {
            Cond::Never => {}
            Cond::Always => self.f().gen_br(l),
            c => {
                let r2 = self.cc_reg2(&cc);
                self.f().gen_brcond_i64(c, cc.reg, r2, l);
            }
        }
    }

    // Registers.

    /// `byte_reg_is_xH()`.
    fn byte_reg_is_xh(&self, r: usize) -> bool {
        (4..8).contains(&r) && self.d.prefix & PREFIX_REX == 0
    }

    /// `gen_op_mov_reg_v()`: write `t` to register `r` with size `ot`.
    fn mov_reg_v(&mut self, ot: u32, r: usize, t: TempI64) {
        let regs = self.g.regs;
        let xh = ot == OT8 && self.byte_reg_is_xh(r);
        let f = self.f();
        match ot {
            OT8 if xh => f.gen_deposit_i64(regs[r - 4], regs[r - 4], t, 8, 8),
            OT8 => f.gen_deposit_i64(regs[r], regs[r], t, 0, 8),
            OT16 => f.gen_deposit_i64(regs[r], regs[r], t, 0, 16),
            OT32 => f.gen_ext32u_i64(regs[r], t),
            _ => f.gen_mov_i64(regs[r], t),
        }
    }

    /// `gen_op_mov_v_reg()`: register `r` with size `ot`, zero extended, into `t`.
    fn mov_v_reg(&mut self, ot: u32, t: TempI64, r: usize) {
        let regs = self.g.regs;
        if ot == OT8 && self.byte_reg_is_xh(r) {
            self.f().gen_extract_i64(t, regs[r - 4], 8, 8);
        } else {
            self.ext(ot, t, regs[r], false);
        }
    }

    /// `gen_op_add_reg_im()`.
    fn add_reg_im(&mut self, size: u32, r: usize, v: i64) {
        let t = self.new64();
        let rr = self.g.regs[r];
        self.f().gen_addi_i64(t, rr, v);
        self.mov_reg_v(size, r, t);
    }

    /// `gen_op_add_reg()`.
    fn add_reg(&mut self, size: u32, r: usize, v: TempI64) {
        let t = self.new64();
        let rr = self.g.regs[r];
        self.f().gen_add_i64(t, rr, v);
        self.mov_reg_v(size, r, t);
    }

    // Memory.

    fn ld_v(&mut self, ot: u32, t: TempI64, a: TempI64) {
        let idx = self.d.mem_index;
        self.f().gen_qemu_ld_i64(t, a, idx, mo(ot));
    }

    fn st_v(&mut self, ot: u32, t: TempI64, a: TempI64) {
        let idx = self.d.mem_index;
        self.f().gen_qemu_st_i64(t, a, idx, mo(ot));
    }

    /// `gen_lea_v_seg_dest()`: the linear address of `a0` in segment `def_seg` or the
    /// override `ovr_seg`, with address size `aflag`, into A0.
    fn lea_v_seg(&mut self, aflag: u32, a0: TempI64, def_seg: i32, ovr_seg: i32) {
        let dest = self.g.a0;
        let mut a0 = a0;
        let mut ovr = ovr_seg;
        match aflag {
            OT64 => {
                if ovr < 0 {
                    self.f().gen_mov_i64(dest, a0);
                    return;
                }
            }
            OT32 => {
                // 32-bit address.
                if ovr < 0 && self.d.addseg {
                    ovr = def_seg;
                }
                if ovr < 0 {
                    self.f().gen_ext32u_i64(dest, a0);
                    return;
                }
            }
            _ => {
                // 16-bit address.
                self.f().gen_ext16u_i64(dest, a0);
                a0 = dest;
                if ovr < 0 && self.d.addseg {
                    ovr = def_seg;
                }
                if ovr < 0 {
                    return;
                }
            }
        }
        let base = self.g.seg_base[ovr as usize];
        let code64 = self.code64();
        let f = self.f();
        if aflag == OT64 {
            f.gen_add_i64(dest, a0, base);
        } else if code64 {
            f.gen_ext32u_i64(dest, a0);
            f.gen_add_i64(dest, dest, base);
        } else {
            f.gen_add_i64(dest, a0, base);
            f.gen_ext32u_i64(dest, dest);
        }
    }

    /// `gen_add_A0_im()`.
    fn add_a0_im(&mut self, v: i64) {
        let a0 = self.g.a0;
        let code64 = self.code64();
        let f = self.f();
        f.gen_addi_i64(a0, a0, v);
        if !code64 {
            f.gen_ext32u_i64(a0, a0);
        }
    }

    /// `gen_lea_modrm_0()`: decode the memory operand of `modrm`.
    fn lea_modrm_0(&mut self, modrm: u32) -> R<Addr> {
        let mut def_seg = R_DS_I;
        let mut index = -1;
        let mut scale = 0;
        let mut disp: i64 = 0;
        let md = (modrm >> 6) & 3;
        let rm = (modrm & 7) as i32;
        let mut base = rm | self.d.rex_b as i32;
        if md == 3 {
            return Ok(Addr { def_seg, base, index, scale, disp });
        }
        match self.d.aflag {
            OT64 | OT32 => {
                let mut havesib = false;
                if rm == 4 {
                    let code = self.ldub()? as u32;
                    scale = (code >> 6) & 3;
                    index = (((code >> 3) & 7) as usize | self.d.rex_x) as i32;
                    if index == 4 {
                        index = -1;
                    }
                    base = ((code & 7) as usize | self.d.rex_b) as i32;
                    havesib = true;
                }
                match md {
                    0 => {
                        if base & 7 == 5 {
                            base = -1;
                            disp = self.ldl()? as u32 as i32 as i64;
                            if self.code64() && !havesib {
                                base = -2;
                                disp = disp
                                    .wrapping_add(self.d.pc as i64)
                                    .wrapping_add(self.d.rip_offset as i64);
                            }
                        }
                    }
                    1 => disp = self.ldub()? as u8 as i8 as i64,
                    _ => disp = self.ldl()? as u32 as i32 as i64,
                }
                // For correct popl handling with esp.
                if base == R_ESP as i32 && self.d.popl_esp_hack != 0 {
                    disp += self.d.popl_esp_hack;
                }
                if base == R_EBP_I || base == R_ESP as i32 {
                    def_seg = R_SS as i32;
                }
            }
            _ => {
                if md == 0 {
                    if rm == 6 {
                        base = -1;
                        disp = self.lduw()? as i64;
                        return Ok(Addr { def_seg, base, index, scale, disp });
                    }
                } else if md == 1 {
                    disp = self.ldub()? as u8 as i8 as i64;
                } else {
                    disp = self.lduw()? as u16 as i16 as i64;
                }
                let (b, i, ss) = match rm {
                    0 => (3, 6, false),
                    1 => (3, 7, false),
                    2 => (5, 6, true),
                    3 => (5, 7, true),
                    4 => (6, -1, false),
                    5 => (7, -1, false),
                    6 => (5, -1, true),
                    _ => (3, -1, false),
                };
                base = b;
                index = i;
                if ss {
                    def_seg = R_SS as i32;
                }
            }
        }
        Ok(Addr { def_seg, base, index, scale, disp })
    }

    /// `gen_lea_modrm_1()`: the effective address, without the segment.
    fn lea_modrm_1(&mut self, a: Addr) -> TempI64 {
        let regs = self.g.regs;
        let a0 = self.g.a0;
        let mut ea: Option<TempI64> = None;
        if a.index >= 0 {
            let mut e = regs[a.index as usize];
            if a.scale != 0 {
                self.f().gen_shli_i64(a0, e, i64::from(a.scale));
                e = a0;
            }
            if a.base >= 0 {
                self.f().gen_add_i64(a0, e, regs[a.base as usize]);
                e = a0;
            }
            ea = Some(e);
        } else if a.base >= 0 {
            ea = Some(regs[a.base as usize]);
        }
        match ea {
            None => {
                self.f().gen_movi_i64(a0, a.disp);
                a0
            }
            Some(e) if a.disp != 0 => {
                self.f().gen_addi_i64(a0, e, a.disp);
                a0
            }
            Some(e) => e,
        }
    }

    /// `gen_lea_modrm()`: decode the memory operand and put its linear address in A0.
    fn gen_lea_modrm(&mut self, modrm: u32) -> R {
        let a = self.lea_modrm_0(modrm)?;
        let ea = self.lea_modrm_1(a);
        let (aflag, ovr) = (self.d.aflag, self.d.override_seg);
        self.lea_v_seg(aflag, ea, a.def_seg, ovr);
        Ok(())
    }

    /// `gen_ldst_modrm()`: load the r/m operand into T0 (and register `reg`), or store T0
    /// (or register `reg`) into it.
    fn gen_ldst_modrm(&mut self, modrm: u32, ot: u32, reg: Option<usize>, is_store: bool) -> R {
        let md = (modrm >> 6) & 3;
        let rm = (modrm & 7) as usize | self.d.rex_b;
        let (t0, a0) = (self.g.t0, self.g.a0);
        if md == 3 {
            if is_store {
                if let Some(r) = reg {
                    self.mov_v_reg(ot, t0, r);
                }
                self.mov_reg_v(ot, rm, t0);
            } else {
                self.mov_v_reg(ot, t0, rm);
                if let Some(r) = reg {
                    self.mov_reg_v(ot, r, t0);
                }
            }
        } else {
            self.gen_lea_modrm(modrm)?;
            if is_store {
                if let Some(r) = reg {
                    self.mov_v_reg(ot, t0, r);
                }
                self.st_v(ot, t0, a0);
            } else {
                self.ld_v(ot, t0, a0);
                if let Some(r) = reg {
                    self.mov_reg_v(ot, r, t0);
                }
            }
        }
        Ok(())
    }

    /// `gen_op_st_rm_T0_A0()`: store T0 into register `d`, or memory at A0.
    fn st_rm_t0(&mut self, ot: u32, d: Option<usize>) {
        let (t0, a0) = (self.g.t0, self.g.a0);
        match d {
            Some(r) => self.mov_reg_v(ot, r, t0),
            None => self.st_v(ot, t0, a0),
        }
    }

    // Stack.

    /// `gen_push_v()`.
    fn gen_push_v(&mut self, val: TempI64) {
        let d_ot = self.mo_pushpop(self.d.dflag);
        let a_ot = self.mo_stacksize();
        let size = 1i64 << d_ot;
        let new_esp = self.new64();
        let esp = self.g.regs[R_ESP];
        self.f().gen_subi_i64(new_esp, esp, size);
        self.lea_v_seg(a_ot, new_esp, R_SS as i32, -1);
        let a0 = self.g.a0;
        self.st_v(d_ot, val, a0);
        self.mov_reg_v(a_ot, R_ESP, new_esp);
    }

    /// `gen_pop_T0()`: load the top of the stack into T0, without moving ESP.
    fn gen_pop_t0(&mut self) -> u32 {
        let d_ot = self.mo_pushpop(self.d.dflag);
        let a_ot = self.mo_stacksize();
        let esp = self.g.regs[R_ESP];
        self.lea_v_seg(a_ot, esp, R_SS as i32, -1);
        let (t0, a0) = (self.g.t0, self.g.a0);
        self.ld_v(d_ot, t0, a0);
        d_ot
    }

    /// `gen_stack_update()`.
    fn gen_stack_update(&mut self, addend: i64) {
        let a_ot = self.mo_stacksize();
        self.add_reg_im(a_ot, R_ESP, addend);
    }

    /// `gen_pop_update()`.
    fn gen_pop_update(&mut self, ot: u32) {
        self.gen_stack_update(1 << ot);
    }

    /// `gen_stack_A0()`.
    fn gen_stack_a0(&mut self) {
        let a_ot = self.mo_stacksize();
        let esp = self.g.regs[R_ESP];
        self.lea_v_seg(a_ot, esp, R_SS as i32, -1);
    }

    // Segments.

    /// `gen_op_movl_seg_real()`.
    fn movl_seg_real(&mut self, seg_reg: usize, src: TempI64) {
        let t = self.new64();
        self.f().gen_ext16u_i64(t, src);
        self.st_env32(t, seg(seg_reg) + SEG_SELECTOR);
        let base = self.g.seg_base[seg_reg];
        self.f().gen_shli_i64(base, t, 4);
    }

    /// `gen_movl_seg()`: load segment register `seg_reg` from the selector in `src`.
    fn gen_movl_seg(&mut self, seg_reg: usize, src: TempI64) {
        if self.d.pe && !self.d.vm86 {
            let sel = self.trunc32(src);
            let sr = self.c32(seg_reg as i32);
            self.env_call(&helpers::LOAD_SEG, None, &[sr.into(), sel.into()]);
            // For move to DS/ES/SS, the addseg or ss32 flags may change.
            if self.d.code32 && seg_reg < 4 {
                self.b.is_jmp = DISAS_EOB_NEXT;
            }
        } else {
            self.movl_seg_real(seg_reg, src);
        }
        // MOV or POP to SS inhibits interrupts (and single step traps) for one instruction.
        if seg_reg == R_SS {
            self.b.is_jmp = DISAS_EOB_INHIBIT_IRQ;
            self.d.flags &= !TF_MASK;
        }
    }

    // Exceptions.

    /// `gen_exception()`.
    fn gen_exception(&mut self, trapno: i32) {
        self.gen_update_cc_op();
        self.gen_update_eip_cur();
        let t = self.c32(trapno);
        self.env_call(&helpers::RAISE_EXCEPTION, None, &[t.into()]);
        self.b.is_jmp = DisasJumpType::NoReturn;
    }

    /// `gen_illegal_opcode()`: #UD.
    fn gen_illegal_opcode(&mut self) {
        self.gen_exception(EXCP06_ILLOP);
    }

    /// `gen_exception_gpf()`: #GP(0).
    fn gen_exception_gpf(&mut self) {
        self.gen_exception(EXCP0D_GPF);
    }

    /// `check_cpl0()`.
    fn check_cpl0(&mut self) -> bool {
        if self.d.cpl == 0 {
            return true;
        }
        self.gen_exception_gpf();
        false
    }

    /// `check_vm86_iopl()`.
    fn check_vm86_iopl(&mut self) -> bool {
        if !self.d.vm86 || self.d.iopl == 3 {
            return true;
        }
        self.gen_exception_gpf();
        false
    }

    /// `check_iopl()`.
    fn check_iopl(&mut self) -> bool {
        let ok = if self.d.vm86 { self.d.iopl == 3 } else { self.d.cpl <= self.d.iopl };
        if ok {
            return true;
        }
        self.gen_exception_gpf();
        false
    }

    /// `gen_interrupt()`: INT n.
    fn gen_interrupt(&mut self, intno: i32) {
        self.gen_update_cc_op();
        self.gen_update_eip_cur();
        let n = self.c32(intno);
        let len = self.cur_insn_len() as i32;
        let l = self.c32(len);
        self.env_call(&helpers::RAISE_INTERRUPT, None, &[n.into(), l.into()]);
        self.b.is_jmp = DisasJumpType::NoReturn;
    }

    // Flags in env.

    fn gen_set_hflag(&mut self, mask: u32) {
        if self.d.flags & mask == 0 {
            let t = self.ld_env32(HFLAGS);
            self.f().gen_ori_i64(t, t, i64::from(mask));
            self.st_env32(t, HFLAGS);
            self.d.flags |= mask;
        }
    }

    fn gen_reset_hflag(&mut self, mask: u32) {
        if self.d.flags & mask != 0 {
            let t = self.ld_env32(HFLAGS);
            self.f().gen_andi_i64(t, t, !i64::from(mask));
            self.st_env32(t, HFLAGS);
            self.d.flags &= !mask;
        }
    }

    fn gen_set_eflags(&mut self, mask: u32) {
        let t = self.ld_env64(EFLAGS);
        self.f().gen_ori_i64(t, t, i64::from(mask));
        self.st_env64(t, EFLAGS);
    }

    fn gen_reset_eflags(&mut self, mask: u32) {
        let t = self.ld_env64(EFLAGS);
        self.f().gen_andi_i64(t, t, !i64::from(mask));
        self.st_env64(t, EFLAGS);
    }

    // End of block.

    /// `gen_eob()`.
    fn gen_eob(&mut self, mode: DisasJumpType) {
        // A block can have more than one exit, so keep the flags each one starts from.
        let saved_flags = self.d.flags;
        self.gen_update_cc_op();
        // If several instructions disable interrupts, only the first does it.
        let had_inhibit = self.d.flags & HF_INHIBIT_IRQ_MASK != 0;
        if mode == DISAS_EOB_INHIBIT_IRQ && !had_inhibit {
            self.gen_set_hflag(HF_INHIBIT_IRQ_MASK);
        } else {
            self.gen_reset_hflag(HF_INHIBIT_IRQ_MASK);
        }
        if self.b.tb.flags & RF_MASK != 0 {
            self.gen_reset_eflags(RF_MASK);
        }
        let env = self.g.env;
        if mode == DISAS_EOB_RECHECK_TF {
            self.call(&helpers::RECHECKING_SINGLE_STEP, None, &[env.into()]);
            self.f().gen_exit_tb(0, 0);
        } else if self.d.flags & TF_MASK != 0 {
            self.call(&helpers::SINGLE_STEP, None, &[env.into()]);
        } else if mode == DISAS_JUMP && !had_inhibit && self.d.jmp_ic {
            let pc = self.linear_pc();
            self.f().gen_lookup_and_goto_ptr_ic(pc);
        } else if mode == DISAS_JUMP && !had_inhibit {
            // Give interrupts a chance to happen after an instruction that inhibited them.
            self.f().gen_lookup_and_goto_ptr();
        } else {
            self.f().gen_exit_tb(0, 0);
        }
        self.d.flags = saved_flags;
        self.d.jmp_ic = false;
        self.b.is_jmp = DisasJumpType::NoReturn;
    }

    /// The linear address of EIP, as `get_tb_cpu_state` computes the program counter, for a
    /// block end where CS is the one the block started with.
    fn linear_pc(&mut self) -> TempI64 {
        let eip = self.g.eip;
        if self.d.cs_base == 0 && self.code64() {
            return eip;
        }
        let pc = self.new64();
        let cs_base = self.d.cs_base as i64;
        let code64 = self.code64();
        let f = self.f();
        f.gen_addi_i64(pc, eip, cs_base);
        if !code64 {
            f.gen_ext32u_i64(pc, pc);
        }
        pc
    }

    /// The EIP and linear PC of the next instruction plus `diff`, with an operand size of
    /// `ot`.
    fn jmp_target(&mut self, ot: u32, diff: i64) -> (u64, u64) {
        let new_pc = self.d.pc.wrapping_add(diff as u64);
        let mask: u64 = if ot == OT16 {
            0xffff
        } else if !self.code64() {
            0xffff_ffff
        } else {
            !0
        };
        let new_eip = new_pc.wrapping_sub(self.d.cs_base) & mask;
        let mut new_pc = new_eip.wrapping_add(self.d.cs_base);
        if !self.code64() {
            new_pc &= 0xffff_ffff;
        }
        (new_eip, new_pc)
    }

    /// `gen_jmp_rel()`: jump to the next instruction plus `diff`, with an operand size of
    /// `ot`.
    fn gen_jmp_rel(&mut self, ot: u32, diff: i64, tb_num: u64) {
        let (new_eip, new_pc) = self.jmp_target(ot, diff);
        self.gen_update_cc_op();
        self.set_cc_op(CC_OP_DYNAMIC);
        self.gen_jmp_to(new_eip, new_pc, tb_num);
    }

    /// The `goto_tb` slot to use for an exit that would like slot `tb_num`: that one if it
    /// is free, else the other, else none.
    fn goto_tb_slot(&self, tb_num: u64) -> Option<u64> {
        [tb_num, tb_num ^ 1].into_iter().find(|&n| self.d.goto_tb_used & (1 << n) == 0)
    }

    /// The jump of [`Self::gen_jmp_rel`] once the target is known and the flags state is
    /// written back.
    fn gen_jmp_to(&mut self, new_eip: u64, new_pc: u64, tb_num: u64) {
        let eip = self.g.eip;
        let slot = self.goto_tb_slot(tb_num);
        let same_page = self.b.translator_use_goto_tb(new_pc);
        if let (true, true, Some(n)) = (self.d.jmp_opt, same_page, slot) {
            // Jump to the same page: we can use a direct jump.
            self.d.goto_tb_used |= 1 << n;
            let id = self.b.tb.id;
            let f = self.f();
            f.gen_goto_tb(n);
            f.gen_movi_i64(eip, new_eip as i64);
            f.gen_exit_tb(id, n);
            self.b.is_jmp = DisasJumpType::NoReturn;
        } else {
            self.f().gen_movi_i64(eip, new_eip as i64);
            if self.d.jmp_opt {
                // Jump to another page, or a block with no free direct exit.
                self.d.jmp_ic = true;
                self.gen_eob(DISAS_JUMP);
            } else {
                // Exit to the main loop.
                self.gen_eob(DISAS_EOB_ONLY);
            }
        }
    }

    /// Whether a conditional branch here may continue the block with its fall-through
    /// path rather than end it (a superblock, not in QEMU). Not with icount, which charges
    /// a whole block's instructions on entry, and not when the block end has to do more
    /// than jump (single step, an interrupt shadow, or RF to clear).
    fn can_inline_jcc(&self) -> bool {
        self.d.jmp_opt
            && self.b.tb.cflags & cf::USE_ICOUNT == 0
            && self.b.tb.flags & RF_MASK == 0
            && self.d.flags & (TF_MASK | HF_INHIBIT_IRQ_MASK) == 0
            && self.d.side_exits.len() < MAX_SIDE_EXITS
    }

    /// `jcc` with a relative target `diff` on condition `b`. When it can, the block goes on
    /// with the fall-through path and the taken edge becomes a [`SideExit`].
    fn gen_jcc_rel(&mut self, b: u32, diff: i64) {
        let l1 = self.label();
        if self.can_inline_jcc() {
            let ot = self.d.dflag;
            let (new_eip, new_pc) = self.jmp_target(ot, diff);
            self.gen_jcc1(b, l1);
            let flags = self.d.flags;
            // The earlier exits get the direct slots: every run of the block passes their
            // branch, not all reach the later ones.
            let mut slot = None;
            if self.b.translator_use_goto_tb(new_pc) {
                slot = self.goto_tb_slot(0);
                if let Some(n) = slot {
                    self.d.goto_tb_used |= 1 << n;
                }
            }
            self.d.side_exits.push(SideExit { label: l1, new_eip, new_pc, flags, slot });
            return;
        }
        self.gen_jcc1(b, l1);
        self.gen_jmp_rel_csize(0, 1);
        self.set_label(l1);
        let ot = self.d.dflag;
        self.gen_jmp_rel(ot, diff, 0);
    }

    /// Emit the exits of the conditional branches the block went past, after its last
    /// instruction. The flags state was written back before each branch.
    fn gen_side_exits(&mut self) {
        let exits = std::mem::take(&mut self.d.side_exits);
        let saved_flags = self.d.flags;
        for e in &exits {
            self.set_label(e.label);
            self.d.flags = e.flags;
            self.d.cc_op = CC_OP_DYNAMIC;
            self.d.cc_op_dirty = false;
            let mut n = 0;
            if let Some(k) = e.slot {
                self.d.goto_tb_used &= !(1 << k);
                n = k;
            }
            self.gen_jmp_to(e.new_eip, e.new_pc, n);
        }
        self.d.flags = saved_flags;
        self.d.side_exits = exits;
        self.d.side_exits.clear();
    }

    /// `gen_jmp_rel_csize()`.
    fn gen_jmp_rel_csize(&mut self, diff: i64, tb_num: u64) {
        let ot = if self.d.code32 { OT32 } else { OT16 };
        self.gen_jmp_rel(ot, diff, tb_num);
    }
}

/// The most conditional branches one block goes past, see [`SideExit`].
const MAX_SIDE_EXITS: usize = 8;

const R_DS_I: i32 = 3;
const R_EBP_I: i32 = 5;

impl TranslatorOps for DisasContext {
    fn init_disas_context(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        let flags = db.tb.flags;
        let cflags = db.tb.cflags;
        self.flags = flags;
        self.cs_base = db.tb.cs_base;
        self.pe = flags & HF_PE_MASK != 0;
        self.code32 = flags & HF_CS32_MASK != 0;
        self.code64 = flags & HF_CS64_MASK != 0;
        self.ss32 = flags & HF_SS32_MASK != 0;
        self.addseg = flags & HF_ADDSEG_MASK != 0;
        self.vm86 = flags & VM_MASK != 0;
        self.lma = flags & HF_LMA_MASK != 0;
        self.cpl = flags & HF_CPL_MASK;
        self.iopl = (flags >> 12) & 3;
        // cpu_mmu_index() from the block flags.
        let base = if self.cpl == 3 {
            MMU_USER64_IDX
        } else if flags & HF_SMAP_MASK == 0 || flags & AC_MASK != 0 {
            MMU_KNOSMAP64_IDX
        } else {
            MMU_KSMAP64_IDX
        };
        self.mem_index = (base + usize::from(!self.code64)) as u32;
        self.jmp_opt =
            flags & (TF_MASK | HF_INHIBIT_IRQ_MASK) == 0 && cflags & cf::SINGLE_STEP == 0;
        self.repz_opt = cflags & cf::USE_ICOUNT == 0;
        self.cc_op = CC_OP_DYNAMIC;
        self.cc_op_dirty = false;
        self.goto_tb_used = 0;
        self.side_exits.clear();

        let f = &mut db.tb.f;
        let env = f.env();
        let regs = std::array::from_fn(|r| f.global_mem_new_i64(env, reg(r) as i64, REG_NAMES[r]));
        let eip = f.global_mem_new_i64(env, EIP as i64, "eip");
        let cc_dst = f.global_mem_new_i64(env, CC_DST as i64, "cc_dst");
        let cc_src = f.global_mem_new_i64(env, CC_SRC as i64, "cc_src");
        let cc_src2 = f.global_mem_new_i64(env, CC_SRC2 as i64, "cc_src2");
        let cc_op = f.global_mem_new_i32(env, CC_OP as i64, "cc_op");
        let seg_base = std::array::from_fn(|s| {
            f.global_mem_new_i64(env, (seg(s) + SEG_BASE) as i64, SEG_NAMES[s])
        });
        let t0 = f.temp_new_i64();
        let t1 = f.temp_new_i64();
        let a0 = f.temp_new_i64();
        let cc_srct = f.temp_new_i64();
        let tmp0 = f.temp_new_i64();
        let tmp4 = f.temp_new_i64();
        self.g = Some(G {
            env,
            regs,
            eip,
            cc_dst,
            cc_src,
            cc_src2,
            cc_op,
            seg_base,
            t0,
            t1,
            a0,
            cc_srct,
            tmp0,
            tmp4,
        });
    }

    fn insn_start(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        self.prev_insn_start = db.insn_start;
        self.prev_insn_end = db.tb.f.last_op();
        let pc = db.pc_next;
        db.tb.f.gen_insn_start(&[pc, u64::from(self.cc_op), 0]);
    }

    fn translate_insn(
        &mut self,
        db: &mut DisasContextBase<'_>,
        cpu: &mut Cpu<'_>,
    ) -> Result<(), CpuLoopExit> {
        let pc_next = db.pc_next;
        let orig_cc_op = self.cc_op;
        let orig_dirty = self.cc_op_dirty;
        let orig_flags = self.flags;
        let g = self.g.expect("init_disas_context ran");
        self.pc_start = pc_next;
        self.pc = pc_next;
        let r = {
            let mut s = S { d: self, b: db, cpu, g };
            s.disas_insn()
        };
        match r {
            Ok(()) => {}
            Err(Stop::Exit(e)) => return Err(e),
            Err(Stop::TooLong) => {
                let start = db.insn_start;
                db.tb.f.remove_ops_after(start);
                self.cc_op = orig_cc_op;
                self.cc_op_dirty = orig_dirty;
                self.flags = orig_flags;
                db.is_jmp = DisasJumpType::Next;
                let mut s = S { d: self, b: db, cpu, g };
                s.gen_exception_gpf();
            }
            Err(Stop::Retry) => {
                // Restore the state that may affect the next instruction.
                self.pc = pc_next;
                self.cc_op = orig_cc_op;
                self.cc_op_dirty = orig_dirty;
                self.flags = orig_flags;
                db.num_insns -= 1;
                let end = self.prev_insn_end;
                db.tb.f.remove_ops_after(end);
                db.insn_start = self.prev_insn_start;
                db.is_jmp = DisasJumpType::TooMany;
                return Ok(());
            }
        }
        db.pc_next = self.pc;
        if db.is_jmp == DisasJumpType::Next {
            if self.flags & (TF_MASK | HF_INHIBIT_IRQ_MASK) != 0 {
                // In single step mode, or with interrupts inhibited, translate one
                // instruction and stop so the trap or the interrupt can happen.
                db.is_jmp = DISAS_EOB_NEXT;
            } else if !db.is_same_page(db.pc_next) {
                db.is_jmp = DisasJumpType::TooMany;
            }
        }
        Ok(())
    }

    fn tb_stop(&mut self, db: &mut DisasContextBase<'_>, cpu: &mut Cpu<'_>) {
        let g = self.g.expect("init_disas_context ran");
        self.pc = db.pc_next;
        self.pc_start = db.pc_next;
        let mut s = S { d: self, b: db, cpu, g };
        match s.b.is_jmp {
            DisasJumpType::NoReturn => {}
            DisasJumpType::Next | DisasJumpType::TooMany => {
                s.gen_update_cc_op();
                s.gen_jmp_rel_csize(0, 0);
            }
            m if m == DISAS_EOB_NEXT || m == DISAS_EOB_INHIBIT_IRQ => {
                s.gen_update_eip_cur();
                s.gen_eob(m);
            }
            m => s.gen_eob(m),
        }
        s.gen_side_exits();
    }
}
