// SPDX-License-Identifier: GPL-2.0-or-later

//! The vector instructions: a port of the MMX, SSE and AVX part of `decode-new.c.inc` and
//! `emit.c.inc`, for the 0F, 0F 38 and 0F 3A opcode maps with legacy and VEX encodings.
//!
//! The opcode tables live in [`super::sse_tab`], which is generated from QEMU's
//! `decode-new.c.inc` by `tests/data/gen_sse_tab.py`. Decoding follows `decode_insn()`: the
//! table entry, the prefix specific sub decoders, `validate_sse_prefix()`, the operand sizes
//! and then the operands themselves. Translation follows `disas_insn()`: the CPUID check,
//! `validate_vex()`, loading the memory operand into `xmm_t0` or `mmx_t0`, the emitter and
//! the writeback.
//!
//! The emitters call the single vector kernel of [`super::super::helpers::vec`] where QEMU
//! calls one helper per operation or a `tcg_gen_gvec_*` expansion; moves, inserts and
//! extracts are inline loads and stores, as in QEMU.
//!
//! Differences from QEMU:
//!
//! - The flag setting helpers (COMISS, UCOMISS, PTEST and VTESTPS/PD) return the flags and
//!   the translator copies them into `cc_src`, where QEMU's helpers write `CC_SRC` and
//!   `CC_OP` themselves.
//! - The gathers are emitted inline, element by element, where QEMU calls a helper; the
//!   result, including what a fault in the middle leaves behind, is the same.
//! - PCMPESTRI and PCMPESTRM read 64-bit lengths from RAX and RDX with REX.W (VEX.W), as
//!   hardware does. QEMU 11.1's new decoder lost the REX.W bit that `pcmp_elen()` checks
//!   and always reads EAX and EDX.
//! - These instructions decode but raise #UD when executed: EXTRQ and INSERTQ (SSE4a) and
//!   CMPccXADD.
//!
//! QEMU behaviors kept on purpose, although real hardware differs:
//!
//! - The class 0 MMX entries (EMMS, MOVQ2DQ, MOVDQ2Q, CVTPI2Px and friends) skip the CR0.TS,
//!   CR0.EM and CR4.OSFXSR checks.
//! - CVTPI2PS and CVTPI2PD switch the FPU to MMX mode even with a memory operand.
//! - VEX encoded MMX opcodes without a 66 prefix that QEMU emits with `gvec` work on the MMX
//!   registers instead of raising #UD.
//! - Masked loads (VMASKMOVPS, VMASKMOVPD and VPMASKMOV) read the whole vector from memory, so
//!   a fault on a masked out element is reported.
//! - VCVTDQ2PD loads a full vector from memory.
//! - The fourth register of VEX blends comes from bits 7:4 of the immediate, even outside of
//!   64-bit mode.
//! - VEX encoded LDMXCSR stores the value in MXCSR without checking the reserved bits, as
//!   the legacy form does.
//! - With MXCSR.DAZ set, MINPS, MAXPS and the other minimum and maximum instructions compare
//!   the flushed inputs but return the original denormal operand, where hardware returns
//!   zero.
//! - VCVTPH2PS sets MXCSR.DE for a denormal half precision input, which AMD hardware does
//!   not.

use ruvm_jit_core::ir::TempI128;

use super::super::EXCP07_PREX;
use super::super::env::{
    HF_AVX_EN_MASK, HF_OSFXSR_MASK, MMX_T0, MXCSR, XMM_T0, ZMM_SIZE, fpreg, zmm,
};
use super::super::helpers::vec::{self, K, sse_offs, sse_op};
use super::sse_tab::{self as tab, Dec, Ft, G as Gen};
use super::*;
use crate::state::{CPU_NB_REGS, HF_EM_MASK, HF_TS_MASK, R_DS, R_EAX, R_EDI, R_EDX};

/// The bits of an [`E`] entry's flags word.
pub(super) mod flags {
    /// The VEX exception class, `vex_class`.
    pub(crate) const VEX1: u64 = 1;
    pub(crate) const VEX2: u64 = 2;
    pub(crate) const VEX3: u64 = 3;
    pub(crate) const VEX4: u64 = 4;
    pub(crate) const VEX5: u64 = 5;
    pub(crate) const VEX6: u64 = 6;
    pub(crate) const VEX7: u64 = 7;
    pub(crate) const VEX11: u64 = 11;
    pub(crate) const VEX12: u64 = 12;
    pub(crate) const VEX13: u64 = 13;
    pub(crate) const CLASS: u64 = 0xf;

    /// `vex_special`.
    pub(crate) const REP_SCALAR: u64 = 1 << 4;
    pub(crate) const SSE_UNALIGNED: u64 = 2 << 4;
    pub(crate) const AVX2_256: u64 = 3 << 4;
    pub(crate) const VEX_SPECIAL: u64 = 3 << 4;

    /// `special`.
    pub(crate) const MMX: u64 = 1 << 8;
    pub(crate) const OP0_RD: u64 = 2 << 8;
    pub(crate) const OP2_RY: u64 = 3 << 8;
    pub(crate) const AVX_MOVX: u64 = 4 << 8;
    pub(crate) const XCHG: u64 = 5 << 8;
    pub(crate) const SPECIAL: u64 = 7 << 8;

    /// `valid_prefix`: bit 16 plus the REPZ, REPNZ and DATA prefix bits.
    pub(crate) const P_00: u64 = 1 << 16;
    pub(crate) const P_F3: u64 = 1 << 17;
    pub(crate) const P_F2: u64 = 1 << 18;
    pub(crate) const P_66: u64 = 1 << 24;

    /// `check`.
    pub(crate) const CHK_O64: u64 = 1 << 32;
    pub(crate) const CHK_W0: u64 = 1 << 34;
    pub(crate) const CHK_W1: u64 = 1 << 35;

    /// An 8-bit immediate follows the operands, `op3`.
    pub(crate) const OP3: u64 = 1 << 36;
}

use flags::*;

/// The operand types of `X86OpType` that the vector maps use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum T {
    None,
    /// The same operand as operand 0, `2op`.
    Op0,
    /// VEX.vvvv selects a general purpose register.
    B,
    /// The modrm r/m general purpose register or memory operand.
    E,
    /// The modrm reg field selects a general purpose register.
    G,
    /// VEX.vvvv selects a vector register; without VEX, the destination.
    H,
    /// An immediate.
    I,
    /// A memory operand, which the emitter loads or stores itself.
    M,
    /// The modrm r/m field selects an MMX register.
    N,
    /// The modrm reg field selects an MMX register.
    P,
    /// The modrm r/m MMX register or memory operand.
    Q,
    /// The modrm r/m field selects a vector register.
    U,
    /// The modrm reg field selects a vector register.
    V,
    /// The modrm r/m vector register or memory operand.
    W,
    /// A vector memory operand.
    Wm,
}

/// The operand sizes of `X86OpSize` that the vector maps use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Z {
    None,
    B,
    W,
    D,
    Q,
    /// 32 or 64 bits by operand size.
    Y,
    /// 128 bits, or 64 bits for MMX.
    Dq,
    /// 256 bits.
    Qq,
    /// 128 or 256 bits by VEX.L, or 64 bits for MMX.
    X,
    /// Half of `X`.
    Xh,
    Ss,
    Sd,
}

/// An opcode table entry, `X86OpEntry`.
#[derive(Clone, Copy)]
pub(super) struct E {
    g: Gen,
    dec: Dec,
    op: [T; 3],
    s: [Z; 3],
    flags: u64,
    cpuid: Ft,
}

pub(super) const fn e(g: Gen, dec: Dec, op: [T; 3], s: [Z; 3], flags: u64, cpuid: Ft) -> E {
    E { g, dec, op, s, flags, cpuid }
}

/// An empty entry: #UD.
pub(super) const E0: E = e(Gen::None, Dec::None, [T::None; 3], [Z::None; 3], 0, Ft::None);

/// Where an operand lives, `X86OpUnit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unit {
    Skip,
    Int,
    Imm,
    Mmx,
    Sse,
}

/// A decoded operand, `X86DecodedOp`.
#[derive(Clone, Copy)]
struct Op {
    unit: Unit,
    n: usize,
    ot: u32,
    ea: bool,
    /// The `env` offset of an MMX or vector operand.
    off: usize,
}

const OP_NONE: Op = Op { unit: Unit::Skip, n: 0, ot: 0, ea: false, off: 0 };

/// A decoded instruction, `X86DecodedInsn`.
struct Dx {
    e: E,
    /// The VEX class, which `validate_vex()` may change.
    class: u64,
    op: [Op; 3],
    modrm: Option<u32>,
    mem: Option<Addr>,
    imm: i64,
}

const REP: u32 = PREFIX_REPZ | PREFIX_REPNZ;
const SSE_PREFIX: u32 = PREFIX_REPZ | PREFIX_REPNZ | PREFIX_DATA;

/// The integer operations that work on operands 0, 1 and 2 with the vector length: the
/// kernel, its variant and whether QEMU emits them with `gen_binary_int_sse()`, which refuses
/// the VEX encoding of an MMX instruction.
fn int_op(g: Gen) -> Option<(K, u32, bool)> {
    Some(match g {
        Gen::Paddb => (K::Add, 0, false),
        Gen::Paddw => (K::Add, 1, false),
        Gen::Paddd => (K::Add, 2, false),
        Gen::Paddq => (K::Add, 3, false),
        Gen::Psubb => (K::Sub, 0, false),
        Gen::Psubw => (K::Sub, 1, false),
        Gen::Psubd => (K::Sub, 2, false),
        Gen::Psubq => (K::Sub, 3, false),
        Gen::Paddsb => (K::AddS, 0, false),
        Gen::Paddsw => (K::AddS, 1, false),
        Gen::Paddusb => (K::AddUS, 0, false),
        Gen::Paddusw => (K::AddUS, 1, false),
        Gen::Psubsb => (K::SubS, 0, false),
        Gen::Psubsw => (K::SubS, 1, false),
        Gen::Psubusb => (K::SubUS, 0, false),
        Gen::Psubusw => (K::SubUS, 1, false),
        Gen::Pcmpeqb => (K::CmpEq, 0, false),
        Gen::Pcmpeqw => (K::CmpEq, 1, false),
        Gen::Pcmpeqd => (K::CmpEq, 2, false),
        Gen::Pcmpeqq => (K::CmpEq, 3, false),
        Gen::Pcmpgtb => (K::CmpGt, 0, false),
        Gen::Pcmpgtw => (K::CmpGt, 1, false),
        Gen::Pcmpgtd => (K::CmpGt, 2, false),
        Gen::Pcmpgtq => (K::CmpGt, 3, false),
        Gen::Pminsb => (K::MinS, 0, false),
        Gen::Pminsw => (K::MinS, 1, false),
        Gen::Pminsd => (K::MinS, 2, false),
        Gen::Pminub => (K::MinU, 0, false),
        Gen::Pminuw => (K::MinU, 1, false),
        Gen::Pminud => (K::MinU, 2, false),
        Gen::Pmaxsb => (K::MaxS, 0, false),
        Gen::Pmaxsw => (K::MaxS, 1, false),
        Gen::Pmaxsd => (K::MaxS, 2, false),
        Gen::Pmaxub => (K::MaxU, 0, false),
        Gen::Pmaxuw => (K::MaxU, 1, false),
        Gen::Pmaxud => (K::MaxU, 2, false),
        Gen::Pand => (K::And, 3, false),
        Gen::Pandn => (K::AndN, 3, false),
        Gen::Por => (K::Or, 3, false),
        Gen::Pxor => (K::Xor, 3, false),
        Gen::Pmullw => (K::MulL, 1, false),
        Gen::Pmulld => (K::MulL, 2, false),
        Gen::Punpcklbw => (K::UnpckL, 0, true),
        Gen::Punpcklwd => (K::UnpckL, 1, true),
        Gen::Punpckldq => (K::UnpckL, 2, true),
        Gen::Punpcklqdq => (K::UnpckL, 3, true),
        Gen::Punpckhbw => (K::UnpckH, 0, true),
        Gen::Punpckhwd => (K::UnpckH, 1, true),
        Gen::Punpckhdq => (K::UnpckH, 2, true),
        Gen::Punpckhqdq => (K::UnpckH, 3, true),
        Gen::Packsswb => (K::PackSS, 0, true),
        Gen::Packssdw => (K::PackSS, 1, true),
        Gen::Packuswb => (K::PackUS, 0, true),
        Gen::Vpackusdw => (K::PackUS, 1, true),
        Gen::Pavgb => (K::Avg, 0, true),
        Gen::Pavgw => (K::Avg, 1, true),
        Gen::Pmaddwd => (K::MaddWd, 0, true),
        Gen::Pmulhuw => (K::MulHU, 1, true),
        Gen::Pmulhw => (K::MulHS, 1, true),
        Gen::Pmuludq => (K::MulUdq, 0, true),
        Gen::Pmuldq => (K::MulDq, 0, true),
        Gen::Psadbw => (K::SadBw, 0, true),
        Gen::PsllwR => (K::ShlR, 1, true),
        Gen::PslldR => (K::ShlR, 2, true),
        Gen::PsllqR => (K::ShlR, 3, true),
        Gen::PsrlwR => (K::ShrR, 1, true),
        Gen::PsrldR => (K::ShrR, 2, true),
        Gen::PsrlqR => (K::ShrR, 3, true),
        Gen::PsrawR => (K::SarR, 1, true),
        Gen::PsradR => (K::SarR, 2, true),
        Gen::Phaddw => (K::HAdd, 1, true),
        Gen::Phaddd => (K::HAdd, 2, true),
        Gen::Phaddsw => (K::HAddS, 1, true),
        Gen::Phsubw => (K::HSub, 1, true),
        Gen::Phsubd => (K::HSub, 2, true),
        Gen::Phsubsw => (K::HSubS, 1, true),
        Gen::Pmaddubsw => (K::MaddUbsw, 0, true),
        Gen::Pshufb => (K::ShufB, 0, true),
        Gen::Psignb => (K::Sign, 0, true),
        Gen::Psignw => (K::Sign, 1, true),
        Gen::Psignd => (K::Sign, 2, true),
        Gen::Pmulhrsw => (K::MulHRS, 1, true),
        Gen::Vpermilps => (K::PermilV, 2, false),
        Gen::Vpermilpd => (K::PermilV, 3, false),
        Gen::Vpermd => (K::PermD, 2, false),
        Gen::Vmaskmovps => (K::MaskLd, 2, false),
        Gen::Vmaskmovpd => (K::MaskLd, 3, false),
        _ => return None,
    })
}

/// The shifts by an immediate, on operand 1 into operand 0.
fn shift_imm_op(g: Gen) -> Option<(K, u32)> {
    Some(match g {
        Gen::PsllwI => (K::ShlI, 1),
        Gen::PslldI => (K::ShlI, 2),
        Gen::PsllqI => (K::ShlI, 3),
        Gen::PsrlwI => (K::ShrI, 1),
        Gen::PsrldI => (K::ShrI, 2),
        Gen::PsrlqI => (K::ShrI, 3),
        Gen::PsrawI => (K::SarI, 1),
        Gen::PsradI => (K::SarI, 2),
        Gen::PslldqI => (K::BslI, 0),
        Gen::PsrldqI => (K::BsrI, 0),
        _ => return None,
    })
}

/// The operations that only read operand 2: the kernel and its variant.
fn unary_op(g: Gen) -> Option<(K, u32)> {
    Some(match g {
        Gen::Pabsb => (K::Abs, 0),
        Gen::Pabsw => (K::Abs, 1),
        Gen::Pabsd => (K::Abs, 2),
        Gen::Vmovsldup => (K::MovSlDup, 0),
        Gen::Vmovshdup => (K::MovShDup, 0),
        Gen::Vmovddup => (K::MovDDup, 0),
        Gen::Vpmovsxbw => (K::MovSx, 1 << 2),
        Gen::Vpmovsxbd => (K::MovSx, 2 << 2),
        Gen::Vpmovsxbq => (K::MovSx, 3 << 2),
        Gen::Vpmovsxwd => (K::MovSx, 1 | 2 << 2),
        Gen::Vpmovsxwq => (K::MovSx, 1 | 3 << 2),
        Gen::Vpmovsxdq => (K::MovSx, 2 | 3 << 2),
        Gen::Vpmovzxbw => (K::MovZx, 1 << 2),
        Gen::Vpmovzxbd => (K::MovZx, 2 << 2),
        Gen::Vpmovzxbq => (K::MovZx, 3 << 2),
        Gen::Vpmovzxwd => (K::MovZx, 1 | 2 << 2),
        Gen::Vpmovzxwq => (K::MovZx, 1 | 3 << 2),
        Gen::Vpmovzxdq => (K::MovZx, 2 | 3 << 2),
        Gen::Vpbroadcastb => (K::Bcast, 0),
        Gen::Vpbroadcastw => (K::Bcast, 1),
        Gen::Vpbroadcastd => (K::Bcast, 2),
        Gen::Vpbroadcastq => (K::Bcast, 3),
        Gen::Vbroadcastx128 => (K::Bcast, 4),
        Gen::Vcvtdq2ps => (K::CvtDq2Ps, 0),
        Gen::Vcvtps2dq => (K::CvtPs2Dq, 0),
        Gen::Vcvttps2dq => (K::CvttPs2Dq, 0),
        Gen::Vcvtdq2pd => (K::CvtDq2Pd, 0),
        Gen::Vcvtpd2dq => (K::CvtPd2Dq, 0),
        Gen::Vcvttpd2dq => (K::CvttPd2Dq, 0),
        Gen::Vcvtps2pd => (K::CvtPs2Pd, 0),
        Gen::Vcvtpd2ps => (K::CvtPd2Ps, 0),
        Gen::Vcvtph2ps => (K::CvtPh2Ps, 0),
        _ => return None,
    })
}

/// The operations with an immediate that read operands 1 and 2.
fn imm_op(g: Gen) -> Option<(K, u32)> {
    Some(match g {
        Gen::Palignr => (K::AlignR, 0),
        Gen::Vblendps => (K::Blend, 2),
        Gen::Vblendpd => (K::Blend, 3),
        Gen::Vpblendw => (K::BlendW, 0),
        Gen::Vdpps => (K::Dpp, 0),
        Gen::Vdppd => (K::Dpp, 1),
        Gen::Vmpsadbw => (K::Mpsadbw, 0),
        Gen::Pclmulqdq => (K::Pclmul, 0),
        Gen::Vperm2x128 => (K::Perm2, 0),
        _ => return None,
    })
}

/// The operations with an immediate that only read operand 1.
fn imm_unary_op(g: Gen) -> Option<(K, u32)> {
    Some(match g {
        Gen::Pshufw => (K::PshufW, 0),
        Gen::Pshufd => (K::PshufD, 0),
        Gen::Pshufhw => (K::PshufHW, 0),
        Gen::Pshuflw => (K::PshufLW, 0),
        Gen::Vpermq => (K::PermQ, 0),
        Gen::VpermilpsI => (K::PermilI, 2),
        Gen::VpermilpdI => (K::PermilI, 3),
        _ => return None,
    })
}

/// The packed and scalar floating point operations, `FP_SSE`.
fn fp_op(g: Gen) -> Option<K> {
    Some(match g {
        Gen::Vadd => K::FAdd,
        Gen::Vsub => K::FSub,
        Gen::Vmul => K::FMul,
        Gen::Vdiv => K::FDiv,
        Gen::Vmin => K::FMin,
        Gen::Vmax => K::FMax,
        _ => return None,
    })
}

/// `softfloat`'s `float_muladd_negate_c` and `float_muladd_negate_product`.
const NEG_C: u32 = 1;
const NEG_P: u32 = 2;

/// The FMA instructions: the operand order (231, 213 or 132), the flags for even and odd
/// elements and whether the operation is scalar.
fn fma_op(g: Gen) -> Option<(u32, u32, u32, bool)> {
    let (order, even, odd, scalar) = match g {
        Gen::Vfmadd231px => (231, 0, 0, false),
        Gen::Vfmadd213px => (213, 0, 0, false),
        Gen::Vfmadd132px => (132, 0, 0, false),
        Gen::Vfmadd231sx => (231, 0, 0, true),
        Gen::Vfmadd213sx => (213, 0, 0, true),
        Gen::Vfmadd132sx => (132, 0, 0, true),
        Gen::Vfnmadd231px => (231, NEG_P, NEG_P, false),
        Gen::Vfnmadd213px => (213, NEG_P, NEG_P, false),
        Gen::Vfnmadd132px => (132, NEG_P, NEG_P, false),
        Gen::Vfnmadd231sx => (231, NEG_P, NEG_P, true),
        Gen::Vfnmadd213sx => (213, NEG_P, NEG_P, true),
        Gen::Vfnmadd132sx => (132, NEG_P, NEG_P, true),
        Gen::Vfmsub231px => (231, NEG_C, NEG_C, false),
        Gen::Vfmsub213px => (213, NEG_C, NEG_C, false),
        Gen::Vfmsub132px => (132, NEG_C, NEG_C, false),
        Gen::Vfmsub231sx => (231, NEG_C, NEG_C, true),
        Gen::Vfmsub213sx => (213, NEG_C, NEG_C, true),
        Gen::Vfmsub132sx => (132, NEG_C, NEG_C, true),
        Gen::Vfnmsub231px => (231, NEG_C | NEG_P, NEG_C | NEG_P, false),
        Gen::Vfnmsub213px => (213, NEG_C | NEG_P, NEG_C | NEG_P, false),
        Gen::Vfnmsub132px => (132, NEG_C | NEG_P, NEG_C | NEG_P, false),
        Gen::Vfnmsub231sx => (231, NEG_C | NEG_P, NEG_C | NEG_P, true),
        Gen::Vfnmsub213sx => (213, NEG_C | NEG_P, NEG_C | NEG_P, true),
        Gen::Vfnmsub132sx => (132, NEG_C | NEG_P, NEG_C | NEG_P, true),
        Gen::Vfmaddsub231px => (231, NEG_C, 0, false),
        Gen::Vfmaddsub213px => (213, NEG_C, 0, false),
        Gen::Vfmaddsub132px => (132, NEG_C, 0, false),
        Gen::Vfmsubadd231px => (231, 0, NEG_C, false),
        Gen::Vfmsubadd213px => (213, 0, NEG_C, false),
        Gen::Vfmsubadd132px => (132, 0, NEG_C, false),
        _ => return None,
    };
    Some((order, even, odd, scalar))
}

impl S<'_, '_, '_> {
    /// Decode and translate an instruction of the vector maps: map 1 is 0F, 2 is 0F 38 and
    /// 3 is 0F 3A. `b` is the opcode byte after the escape bytes.
    pub(super) fn sse_insn(&mut self, map: u32, b: u32) -> R {
        match self.sse_decode(map, b)? {
            Some(x) => self.sse_disas(x),
            None => {
                self.gen_illegal_opcode();
                Ok(())
            }
        }
    }

    fn get_modrm(&mut self, x: &mut Dx) -> R<u32> {
        if let Some(m) = x.modrm {
            return Ok(m);
        }
        let m = self.ldub()? as u32;
        x.modrm = Some(m);
        Ok(m)
    }

    /// No REP or 66 prefix on an MMX entry: the operands are MMX registers.
    fn mmx_noprefix(&self, e: &E) -> bool {
        e.flags & SPECIAL == MMX && self.d.prefix & SSE_PREFIX == 0
    }

    /// `vector_len()`.
    fn vec_len(&self, x: &Dx) -> u32 {
        if self.mmx_noprefix(&x.e) {
            8
        } else if self.d.vex_l {
            32
        } else {
            16
        }
    }

    /// `decode_insn()`. `None` means #UD.
    fn sse_decode(&mut self, map: u32, b: u32) -> R<Option<Dx>> {
        let e = match map {
            1 => tab::OPCODES_0F[b as usize],
            2 if b < 0xf0 => tab::OPCODES_0F38[b as usize],
            3 => tab::OPCODES_0F3A[b as usize],
            _ => E0,
        };
        let mut x = Dx { e, class: 0, op: [OP_NONE; 3], modrm: None, mem: None, imm: 0 };
        self.sse_subdecode(&mut x, b)?;
        x.class = x.e.flags & CLASS;
        let e = x.e;

        // validate_sse_prefix().
        let valid = (e.flags >> 16) & 0x1ff;
        if valid != 0 {
            if self.d.prefix & REP != 0 {
                // In SSE instructions, F3 and F2 cancel 66.
                self.d.prefix &= !PREFIX_DATA;
            }
            if valid & (1 << (self.d.prefix & SSE_PREFIX)) == 0 {
                return Ok(None);
            }
        }

        // The sizes first, so that the RIP relative addresses know the immediate's length.
        for i in 0..3 {
            if e.op[i] == T::None {
                continue;
            }
            let Some(ot) = self.sse_size(&e, e.s[i]) else {
                return Ok(None);
            };
            x.op[i].ot = ot;
            if e.op[i] == T::I {
                self.d.rip_offset += 1 << ot;
            }
        }
        if e.flags & OP3 != 0 {
            self.d.rip_offset += 1;
        }
        for i in 0..3 {
            if e.op[i] != T::None && !self.sse_decode_op(&mut x, i)? {
                return Ok(None);
            }
        }
        if e.flags & OP3 != 0 {
            x.imm = self.insn_get_signed(OT8)?;
        }
        Ok(Some(x))
    }

    /// The `decode_0F*`, `decode_group*` and `decode_sse_unary` functions.
    fn sse_subdecode(&mut self, x: &mut Dx, b: u32) -> R {
        let col = self.prefix_col();
        let p = self.d.prefix;
        let mut n = x.e;
        n.dec = Dec::None;
        x.e = match x.e.dec {
            Dec::None => return Ok(()),
            Dec::D0f10 | Dec::D0f11 | Dec::D0f12 | Dec::D0f16 => {
                let reg = self.get_modrm(x)? >> 6 == 3;
                let (rt, mt) = match x.e.dec {
                    Dec::D0f10 => (&tab::OPCODES_0F10_REG, &tab::OPCODES_0F10_MEM),
                    Dec::D0f11 => (&tab::OPCODES_0F11_REG, &tab::OPCODES_0F11_MEM),
                    Dec::D0f12 => (&tab::OPCODES_0F12_REG, &tab::OPCODES_0F12_MEM),
                    _ => (&tab::OPCODES_0F16_REG, &tab::OPCODES_0F16_MEM),
                };
                if reg {
                    rt[col]
                } else {
                    let mut m = mt[col];
                    if x.e.dec == Dec::D0f12 && p & PREFIX_REPNZ != 0 && self.d.vex_l {
                        m.s[2] = Z::Qq;
                    }
                    m
                }
            }
            Dec::D0f2a => tab::OPCODES_0F2A[col],
            Dec::D0f2b => tab::OPCODES_0F2B[col],
            Dec::D0f2c => tab::OPCODES_0F2C[col],
            Dec::D0f2d => tab::OPCODES_0F2D[col],
            Dec::D0f5a => tab::OPCODES_0F5A[col],
            Dec::D0f5b => tab::OPCODES_0F5B[col],
            Dec::D0f6f => tab::OPCODES_0F6F[col],
            Dec::D0f70 => tab::OPCODES_0F70[col],
            Dec::D0f78 => tab::OPCODES_0F78[col],
            Dec::D0f7e => tab::OPCODES_0F7E[col],
            Dec::D0f7f => tab::OPCODES_0F7F[col],
            Dec::D0fd6 => tab::OPCODES_0FD6[col],
            Dec::D0fe6 => tab::OPCODES_0FE6[col],
            Dec::D0f77 => {
                if p & PREFIX_VEX == 0 {
                    e(Gen::Emms, Dec::None, [T::None; 3], [Z::None; 3], 0, Ft::None)
                } else {
                    let g = if self.d.vex_l { Gen::Vzeroall } else { Gen::Vzeroupper };
                    e(g, Dec::None, [T::None; 3], [Z::None; 3], 8, Ft::None)
                }
            }
            Dec::D0f79 => {
                n.g = if p & PREFIX_REPNZ != 0 {
                    Gen::InsertqR
                } else if p & PREFIX_DATA != 0 {
                    Gen::ExtrqR
                } else {
                    Gen::None
                };
                n
            }
            Dec::SseUnary => {
                if p & REP == 0 {
                    n.op[1] = T::None;
                    n.s[1] = Z::None;
                }
                n.g = match b {
                    0x51 => Gen::Vsqrt,
                    0x52 => Gen::Vrsqrt,
                    _ => Gen::Vrcp,
                };
                n
            }
            Dec::Vxcomisx => {
                let z = if p & PREFIX_DATA != 0 { Z::Sd } else { Z::Ss };
                n.s[1] = z;
                n.s[2] = z;
                n.g = if b == 0x2e { Gen::Vucomi } else { Gen::Vcomi };
                n
            }
            Dec::Group12 | Dec::Group13 | Dec::Group14 => {
                let op = ((self.get_modrm(x)? >> 3) & 7) as usize;
                match x.e.dec {
                    Dec::Group12 => tab::GROUP12[op],
                    Dec::Group13 => tab::GROUP13[op],
                    _ => tab::GROUP14[op],
                }
            }
            Dec::Vinsertps => {
                if self.get_modrm(x)? >> 6 == 3 {
                    tab::VINSERTPS_REG
                } else {
                    tab::VINSERTPS_MEM
                }
            }
        };
        Ok(())
    }

    /// `decode_op_size()`.
    fn sse_size(&self, e: &E, z: Z) -> Option<u32> {
        let mmx = self.mmx_noprefix(e);
        Some(match z {
            // QEMU uses -1; nothing reads the size of these operands.
            Z::None => 0,
            Z::B => OT8,
            Z::W => OT16,
            Z::D | Z::Ss => OT32,
            Z::Q | Z::Sd => OT64,
            Z::Y => self.ot_y(),
            Z::Dq => {
                if mmx {
                    OT64
                } else if self.d.vex_l && e.s[0] != Z::Qq && e.s[1] != Z::Qq {
                    return None;
                } else {
                    4
                }
            }
            Z::Qq => {
                if !self.d.vex_l {
                    return None;
                }
                5
            }
            Z::X => {
                if mmx {
                    OT64
                } else if self.d.vex_l {
                    5
                } else {
                    4
                }
            }
            Z::Xh => {
                if self.d.vex_l {
                    4
                } else {
                    OT64
                }
            }
        })
    }

    /// `decode_op()`. `false` means #UD.
    fn sse_decode_op(&mut self, x: &mut Dx, i: usize) -> R<bool> {
        let vex = self.d.prefix & PREFIX_VEX != 0;
        let mut t = x.e.op[i];
        if t == T::H && !vex {
            // Without VEX, the H operand is the destination.
            t = if i == 0 { x.e.op[1] } else { x.e.op[0] };
        }
        let mmx = self.mmx_noprefix(&x.e);
        let vu = if mmx { Unit::Mmx } else { Unit::Sse };
        let op = &mut x.op[i];
        match t {
            T::None => return Ok(true),
            T::Op0 => {
                x.op[i] = x.op[0];
                return Ok(true);
            }
            T::B => {
                op.unit = Unit::Int;
                op.n = self.d.vex_v;
                return Ok(true);
            }
            T::H => {
                op.unit = Unit::Sse;
                op.n = self.d.vex_v;
                return Ok(true);
            }
            T::I => {
                op.unit = Unit::Imm;
                let ot = op.ot;
                x.imm = self.insn_get_signed(ot)?;
                return Ok(true);
            }
            T::G | T::P | T::V => {
                let unit = match t {
                    T::G => Unit::Int,
                    T::P => Unit::Mmx,
                    _ => vu,
                };
                let m = self.get_modrm(x)?;
                let op = &mut x.op[i];
                op.unit = unit;
                op.n = ((m >> 3) & 7) as usize;
                if unit != Unit::Mmx {
                    op.n |= self.d.rex_r;
                }
                return Ok(true);
            }
            _ => {}
        }
        // The operands that come from the modrm r/m field.
        let (unit, need) = match t {
            T::E => (Unit::Int, None),
            T::Q => (Unit::Mmx, None),
            T::W => (vu, None),
            T::N => (Unit::Mmx, Some(true)),
            T::U => (vu, Some(true)),
            T::Wm => (Unit::Sse, Some(false)),
            _ => (Unit::Skip, Some(false)),
        };
        let m = self.get_modrm(x)?;
        let reg = m >> 6 == 3;
        if need.is_some_and(|r| r != reg) {
            return Ok(false);
        }
        x.op[i].unit = unit;
        if reg {
            let mut n = (m & 7) as usize;
            if unit != Unit::Mmx {
                n |= self.d.rex_b;
            }
            x.op[i].n = n;
        } else {
            if x.mem.is_none() {
                let vsib = x.e.flags & CLASS == VEX12;
                x.mem = Some(self.lea_modrm_0v(m, vsib)?);
            }
            x.op[i].ea = true;
        }
        Ok(true)
    }

    fn sse_feature(&self, f: Ft) -> bool {
        let ft = &self.d.feat;
        match f {
            Ft::None => true,
            Ft::Aes => {
                ft.aes
                    && (self.d.prefix & PREFIX_VEX == 0 || (ft.avx && (!self.d.vex_l || ft.vaes)))
            }
            Ft::Avx => ft.avx,
            Ft::Avx2 => ft.avx2,
            Ft::Bmi2 => ft.bmi2,
            Ft::Cmpccxadd => ft.cmpccxadd,
            Ft::F16c => ft.f16c,
            Ft::Fma => ft.fma,
            Ft::Pclmulqdq => ft.pclmulqdq,
            Ft::ShaNi => ft.sha_ni,
            Ft::Sse3 => ft.sse3,
            Ft::Sse41 => ft.sse41,
            Ft::Sse42 => ft.sse42,
            Ft::Sse4a => ft.sse4a,
            Ft::Ssse3 => ft.ssse3,
        }
    }

    /// `validate_vex()`. Raises the exception and returns `false` if the instruction may
    /// not run.
    fn validate_vex(&mut self, x: &mut Dx) -> bool {
        let e = x.e;
        let p = self.d.prefix;
        let vex = p & PREFIX_VEX != 0;
        let fl = self.d.flags;
        let ud = match e.flags & VEX_SPECIAL {
            REP_SCALAR if p & REP != 0 => {
                x.class = if x.class < 4 { 3 } else { 5 };
                if x.op[2].ea {
                    x.op[2].ot = if p & PREFIX_REPZ != 0 { OT32 } else { OT64 };
                }
                self.d.vex_l
            }
            AVX2_256 => vex && self.d.vex_l && !self.d.feat.avx2,
            _ => false,
        };
        if ud {
            self.gen_illegal_opcode();
            return false;
        }
        let ud = match x.class {
            0 => {
                if vex {
                    self.gen_illegal_opcode();
                    return false;
                }
                return true;
            }
            1..=5 | 7 => {
                if vex {
                    fl & HF_AVX_EN_MASK == 0
                } else {
                    !self.mmx_noprefix(&e) && fl & HF_OSFXSR_MASK == 0
                }
            }
            6 | 11 | 12 => {
                (x.class == 12 && (x.modrm.unwrap_or(0) & 7 != 4 || self.d.aflag == OT16))
                    || (x.class == 12 && self.vsib_overlap(x))
                    || !vex
                    || fl & HF_AVX_EN_MASK == 0
            }
            8 => fl & HF_AVX_EN_MASK == 0,
            _ => {
                if !vex || self.d.vex_l {
                    self.gen_illegal_opcode();
                    return false;
                }
                return true;
            }
        };
        let has_vvvv = e.op.iter().any(|&t| t == T::H || t == T::B);
        if ud || (self.d.vex_v != 0 && !has_vvvv) {
            self.gen_illegal_opcode();
            return false;
        }
        if fl & HF_TS_MASK != 0 {
            self.gen_exception(EXCP07_PREX);
            return false;
        }
        let w = self.d.vex_w;
        if fl & HF_EM_MASK != 0 || (e.flags & CHK_W0 != 0 && w) || (e.flags & CHK_W1 != 0 && !w) {
            self.gen_illegal_opcode();
            return false;
        }
        true
    }

    /// The register overlap checks of VEX class 12: the destination, the mask and the VSIB
    /// index must be three different registers.
    fn vsib_overlap(&self, x: &Dx) -> bool {
        let index = x.mem.map_or(-1, |m| m.index);
        let n = |i: usize| x.op[i].n as i32;
        (!x.op[0].ea && (n(0) == index || n(0) == n(1)))
            || n(1) == index
            || (!x.op[2].ea && (n(2) == index || n(2) == n(1)))
    }

    /// `disas_insn()` from the CPUID check on.
    fn sse_disas(&mut self, mut x: Dx) -> R {
        let e = x.e;
        if e.g == Gen::None
            || !self.sse_feature(e.cpuid)
            || (e.flags & CHK_O64 != 0 && !self.d.code64)
        {
            self.gen_illegal_opcode();
            return Ok(());
        }
        match e.flags & SPECIAL {
            OP0_RD if !x.op[0].ea => x.op[0].ot = OT32,
            OP2_RY if !x.op[2].ea => x.op[2].ot = self.ot_y(),
            AVX_MOVX => {
                if !x.op[2].ea {
                    x.op[2].ot = if self.d.vex_l { 5 } else { 4 };
                } else if self.d.vex_l {
                    x.op[2].ot += 1;
                }
            }
            _ => {}
        }
        if !self.validate_vex(&mut x) {
            return Ok(());
        }
        if self.mmx_noprefix(&e) {
            self.env_call(&vec::ENTER_MMX, None, &[]);
        }
        if let Some(mut mem) = x.mem {
            if x.class == VEX12 {
                // The VSIB index is a vector register, added per element by the emitter.
                mem.index = -1;
            }
            let ea = self.lea_modrm_1(mem);
            let (aflag, ovr) = (self.d.aflag, self.d.override_seg);
            self.lea_v_seg(aflag, ea, mem.def_seg, ovr);
        }
        for op in &mut x.op {
            op.off = match op.unit {
                Unit::Mmx if op.ea => MMX_T0,
                Unit::Mmx => fpreg(op.n),
                Unit::Sse if op.ea => XMM_T0,
                Unit::Sse => zmm(op.n),
                _ => 0,
            };
        }
        let (t0, t1) = (self.g.t0, self.g.t1);
        self.sse_load(&x, 1, t0);
        self.sse_load(&x, 2, t1);
        if self.sse_gen(&x)? {
            self.sse_writeback(&x);
        }
        Ok(())
    }

    /// `gen_load()`.
    fn sse_load(&mut self, x: &Dx, i: usize, v: TempI64) {
        let op = x.op[i];
        let a0 = self.g.a0;
        match op.unit {
            Unit::Int if op.ea => self.ld_v(op.ot, v, a0),
            Unit::Int => self.mov_v_reg(op.ot, v, op.n),
            Unit::Imm => {
                let imm = x.imm;
                self.f().gen_movi_i64(v, imm);
            }
            Unit::Mmx | Unit::Sse if op.ea => {
                let aligned = self.sse_needs_alignment(x, op.ot);
                self.load_sse(v, op.ot, op.off, aligned);
            }
            _ => {}
        }
    }

    /// `gen_writeback()` for operand 0.
    fn sse_writeback(&mut self, x: &Dx) {
        let op = x.op[0];
        let (t0, a0) = (self.g.t0, self.g.a0);
        match op.unit {
            Unit::Int if op.ea => self.st_v(op.ot, t0, a0),
            Unit::Int => self.mov_reg_v(op.ot, op.n, t0),
            Unit::Sse if !op.ea && self.d.prefix & PREFIX_VEX != 0 && op.ot <= 4 => {
                self.zero_env(zmm(op.n) + 16, 16);
            }
            _ => {}
        }
    }

    /// `sse_needs_alignment()`.
    fn sse_needs_alignment(&self, x: &Dx, ot: u32) -> bool {
        match x.class {
            2 | 4 => {
                self.d.prefix & PREFIX_VEX == 0
                    && x.e.flags & VEX_SPECIAL != SSE_UNALIGNED
                    && ot >= 4
            }
            1 => ot >= 4,
            _ => false,
        }
    }

    /// The atomicity of 128-bit accesses: AVX makes aligned 16-byte accesses atomic.
    fn atom128(&self) -> MemOp {
        if self.d.feat.avx { MemOp::ATOM_IFALIGN } else { MemOp::ATOM_IFALIGN_PAIR }
    }

    /// `gen_load_sse()`: load `ot` from A0 into `env` at `off`, using `t` for small sizes.
    fn load_sse(&mut self, t: TempI64, ot: u32, off: usize, aligned: bool) {
        let (env, a0, idx) = (self.g.env, self.g.a0, self.d.mem_index);
        let off = off as i64;
        match ot {
            OT8 | OT16 | OT32 => {
                self.ld_v(ot, t, a0);
                let f = self.f();
                match ot {
                    OT8 => f.gen_st8_i64(t, env, off),
                    OT16 => f.gen_st16_i64(t, env, off),
                    _ => f.gen_st32_i64(t, env, off),
                }
            }
            OT64 => {
                let f = self.f();
                f.gen_qemu_ld_i64(t, a0, idx, MemOp::LEUQ);
                f.gen_st_i64(t, env, off);
            }
            4 => {
                let al = if aligned { MemOp::ALIGN_16 } else { MemOp(0) };
                let mop = MemOp::LEUO | self.atom128() | al;
                let f = self.f();
                let v = f.temp_new_i128();
                f.gen_qemu_ld_i128(v, a0, idx, mop);
                f.gen_st_i128(v, env, off);
            }
            _ => {
                let al = if aligned { MemOp::ALIGN_32 } else { MemOp(0) };
                let mop = MemOp::LEUO | MemOp::ATOM_IFALIGN_PAIR;
                let f = self.f();
                let (v0, v1) = (f.temp_new_i128(), f.temp_new_i128());
                let hi = f.temp_new_i64();
                f.gen_qemu_ld_i128(v0, a0, idx, mop | al);
                f.gen_addi_i64(hi, a0, 16);
                f.gen_qemu_ld_i128(v1, hi, idx, mop);
                f.gen_st_i128(v0, env, off);
                f.gen_st_i128(v1, env, off + 16);
            }
        }
    }

    /// `gen_store_sse()`: store `vector_len()` bytes at `src` into operand 0.
    fn store_sse(&mut self, x: &Dx, src: usize) {
        let op0 = x.op[0];
        if !op0.ea {
            let n = self.vec_len(x) as usize;
            self.copy_env(op0.off, src, n);
            return;
        }
        let aligned = self.sse_needs_alignment(x, op0.ot);
        let (env, a0, idx) = (self.g.env, self.g.a0, self.d.mem_index);
        let src = src as i64;
        match op0.ot {
            OT64 => {
                let f = self.f();
                let t = f.temp_new_i64();
                f.gen_ld_i64(t, env, src);
                f.gen_qemu_st_i64(t, a0, idx, MemOp::LEUQ);
            }
            4 => {
                let al = if aligned { MemOp::ALIGN_16 } else { MemOp(0) };
                let mop = MemOp::LEUO | self.atom128() | al;
                self.sto(src, mop);
            }
            5 => {
                let al = if aligned { MemOp::ALIGN_32 } else { MemOp(0) };
                let mop = MemOp::LEUO | MemOp::ATOM_IFALIGN_PAIR;
                let f = self.f();
                let (v0, v1) = (f.temp_new_i128(), f.temp_new_i128());
                let hi = f.temp_new_i64();
                f.gen_ld_i128(v0, env, src);
                f.gen_ld_i128(v1, env, src + 16);
                f.gen_qemu_st_i128(v0, a0, idx, mop | al);
                f.gen_addi_i64(hi, a0, 16);
                f.gen_qemu_st_i128(v1, hi, idx, mop);
            }
            _ => {}
        }
    }

    /// Store 16 bytes of `env` at `src` to A0.
    fn sto(&mut self, src: i64, mop: MemOp) {
        let (env, a0, idx) = (self.g.env, self.g.a0, self.d.mem_index);
        let f = self.f();
        let v: TempI128 = f.temp_new_i128();
        f.gen_ld_i128(v, env, src);
        f.gen_qemu_st_i128(v, a0, idx, mop);
    }

    /// Copy `n` bytes of `env` from `src` to `dst`.
    fn copy_env(&mut self, dst: usize, src: usize, n: usize) {
        if dst == src {
            return;
        }
        let env = self.g.env;
        let t = self.new64();
        for i in (0..n).step_by(8) {
            let f = self.f();
            f.gen_ld_i64(t, env, (src + i) as i64);
            f.gen_st_i64(t, env, (dst + i) as i64);
        }
    }

    /// Clear `n` bytes of `env` at `off`.
    fn zero_env(&mut self, off: usize, n: usize) {
        let env = self.g.env;
        let z = self.c64(0);
        for i in (0..n).step_by(8) {
            self.f().gen_st_i64(z, env, (off + i) as i64);
        }
    }

    fn ld_env(&mut self, ot: u32, t: TempI64, off: usize) {
        let env = self.g.env;
        let off = off as i64;
        let f = self.f();
        match ot {
            OT8 => f.gen_ld8u_i64(t, env, off),
            OT16 => f.gen_ld16u_i64(t, env, off),
            OT32 => f.gen_ld32u_i64(t, env, off),
            _ => f.gen_ld_i64(t, env, off),
        }
    }

    fn st_env(&mut self, ot: u32, t: TempI64, off: usize) {
        let env = self.g.env;
        let off = off as i64;
        let f = self.f();
        match ot {
            OT8 => f.gen_st8_i64(t, env, off),
            OT16 => f.gen_st16_i64(t, env, off),
            OT32 => f.gen_st32_i64(t, env, off),
            _ => f.gen_st_i64(t, env, off),
        }
    }

    /// Call the vector kernel: `o` holds the destination and up to three sources.
    fn kern(
        &mut self,
        k: K,
        var: u32,
        len: u32,
        imm: u32,
        o: [usize; 4],
        x: Option<TempI64>,
    ) -> TempI64 {
        let ret = self.new64();
        let op = self.c32(sse_op(k, var, len, imm));
        let offs = self.c64(sse_offs(o[0], o[1], o[2], o[3]));
        let x = match x {
            Some(t) => t,
            None => self.c64(0),
        };
        self.env_call(&vec::SSE, Some(ret.into()), &[op.into(), offs.into(), x.into()]);
        ret
    }

    /// Put flags returned by the kernel in `cc_src`, for `CC_OP_EFLAGS`.
    fn sse_flags(&mut self, ret: TempI64) {
        let cc_src = self.g.cc_src;
        self.f().gen_mov_i64(cc_src, ret);
        self.set_cc_op(CC_OP_EFLAGS);
    }

    /// The emitters. `false` skips the writeback.
    #[allow(clippy::too_many_lines)]
    fn sse_gen(&mut self, x: &Dx) -> R<bool> {
        let g = self.g;
        let [o0, o1, o2] = [x.op[0].off, x.op[1].off, x.op[2].off];
        let vl = self.vec_len(x);
        let p = self.d.prefix;
        let data = p & PREFIX_DATA != 0;
        let repz = p & PREFIX_REPZ != 0;
        let repnz = p & PREFIX_REPNZ != 0;
        let vex = p & PREFIX_VEX != 0;
        let fvar = if repz {
            2
        } else if repnz {
            3
        } else {
            u32::from(data)
        };
        let pdv = if data { 3 } else { 2 };
        let imm = (x.imm & 0xff) as u32;
        let idx = self.d.mem_index;
        let eg = x.e.g;

        if let Some((k, var, bin)) = int_op(eg) {
            if bin && x.e.flags & SPECIAL == MMX && vex && !data {
                // VEX encoding is not applicable to MMX instructions.
                self.gen_illegal_opcode();
                return Ok(false);
            }
            self.kern(k, var, vl, 0, [o0, o1, o2, 0], None);
            return Ok(true);
        }
        if let Some((k, var)) = shift_imm_op(eg) {
            self.kern(k, var, vl, imm, [o0, o1, 0, 0], None);
            return Ok(true);
        }
        if let Some((k, var)) = unary_op(eg) {
            self.kern(k, var, vl, 0, [o0, o2, o2, 0], None);
            return Ok(true);
        }
        if let Some((k, var)) = imm_op(eg) {
            self.kern(k, var, vl, imm, [o0, o1, o2, 0], None);
            return Ok(true);
        }
        if let Some((k, var)) = imm_unary_op(eg) {
            self.kern(k, var, vl, imm, [o0, o1, 0, 0], None);
            return Ok(true);
        }
        if let Some(k) = fp_op(eg) {
            self.kern(k, fvar, vl, 0, [o0, o1, o2, 0], None);
            return Ok(true);
        }
        if let Some((order, even, odd, scalar)) = fma_op(eg) {
            let (a, b, c) = match order {
                231 => (o1, o2, o0),
                213 => (o1, o0, o2),
                _ => (o0, o2, o1),
            };
            let var = even | odd << 3 | u32::from(self.d.vex_w) << 6 | u32::from(scalar) << 7;
            self.kern(K::Fma, var, vl, 0, [o0, a, b, c], None);
            return Ok(true);
        }

        match eg {
            Gen::Pblendvb | Gen::Blendvps | Gen::Blendvpd => {
                let var = match eg {
                    Gen::Pblendvb => 0,
                    Gen::Blendvps => 2,
                    _ => 3,
                };
                self.kern(K::BlendV, var, vl, 0, [o0, o1, o2, zmm(0)], None);
            }
            Gen::Vpblendvb | Gen::Vblendvps | Gen::Vblendvpd => {
                let var = match eg {
                    Gen::Vpblendvb => 0,
                    Gen::Vblendvps => 2,
                    _ => 3,
                };
                let c = zmm((imm >> 4) as usize);
                self.kern(K::BlendV, var, vl, 0, [o0, o1, o2, c], None);
            }
            Gen::Vshuf => {
                self.kern(K::ShufP, pdv, vl, imm, [o0, o1, o2, 0], None);
            }
            Gen::Vunpcklpx | Gen::Vunpckhpx => {
                let k = if eg == Gen::Vunpcklpx { K::UnpckL } else { K::UnpckH };
                self.kern(k, pdv, vl, 0, [o0, o1, o2, 0], None);
            }
            Gen::Vpsllv | Gen::Vpsrlv | Gen::Vpsrav | Gen::Vpmaskmov => {
                let k = match eg {
                    Gen::Vpsllv => K::ShlV,
                    Gen::Vpsrlv => K::ShrV,
                    Gen::Vpsrav => K::SarV,
                    _ => K::MaskLd,
                };
                let var = if self.d.vex_w { 3 } else { 2 };
                self.kern(k, var, vl, 0, [o0, o1, o2, 0], None);
            }
            Gen::Vphminposuw => {
                self.kern(K::Phminposuw, 0, 16, 0, [o0, o2, o2, 0], None);
            }
            Gen::Vptest | Gen::Vtestps | Gen::Vtestpd => {
                let var = match eg {
                    Gen::Vptest => 0,
                    Gen::Vtestps => 2,
                    _ => 3,
                };
                let r = self.kern(K::Ptest, var, vl, 0, [XMM_T0, o1, o2, 0], None);
                self.sse_flags(r);
            }
            Gen::Vcomi | Gen::Vucomi => {
                let k = if eg == Gen::Vcomi { K::Comi } else { K::Ucomi };
                let r = self.kern(k, u32::from(data), 16, 0, [XMM_T0, o1, o2, 0], None);
                self.sse_flags(r);
            }
            Gen::Movmsk | Gen::Pmovmskb => {
                let var = if eg == Gen::Pmovmskb { 0 } else { pdv };
                let r = self.kern(K::MovMsk, var, vl, 0, [XMM_T0, o2, o2, 0], None);
                self.f().gen_mov_i64(g.t0, r);
            }
            Gen::Vsqrt => {
                if p & REP != 0 {
                    self.kern(K::FSqrt, fvar, vl, 0, [o0, o1, o2, 0], None);
                } else {
                    self.kern(K::FSqrt, u32::from(data), vl, 0, [o0, o2, o2, 0], None);
                }
            }
            Gen::Vrsqrt | Gen::Vrcp => {
                let k = if eg == Gen::Vrsqrt { K::FRsqrt } else { K::FRcp };
                if data || repnz {
                    self.gen_illegal_opcode();
                    return Ok(false);
                } else if repz {
                    self.kern(k, 2, vl, 0, [o0, o1, o2, 0], None);
                } else {
                    self.kern(k, 0, vl, 0, [o0, o2, o2, 0], None);
                }
            }
            Gen::Vhadd | Gen::Vhsub | Gen::Vaddsub => {
                let k = match eg {
                    Gen::Vhadd => K::FHAdd,
                    Gen::Vhsub => K::FHSub,
                    _ => K::FAddSub,
                };
                self.kern(k, u32::from(data), vl, 0, [o0, o1, o2, 0], None);
            }
            Gen::Vcmp => {
                let i = imm & if vex { 31 } else { 7 };
                self.kern(K::FCmp, fvar, vl, i, [o0, o1, o2, 0], None);
            }
            Gen::Vroundps | Gen::Vroundpd => {
                let var = u32::from(eg == Gen::Vroundpd);
                self.kern(K::Round, var, vl, imm, [o0, o1, o1, 0], None);
            }
            Gen::Vroundss | Gen::Vroundsd => {
                let var = if eg == Gen::Vroundss { 2 } else { 3 };
                self.kern(K::Round, var, vl, imm, [o0, o1, o2, 0], None);
            }
            Gen::Vcvtps2ph => {
                self.kern(K::CvtPs2Ph, 0, vl, imm, [o0, o1, 0, 0], None);
                if x.op[0].ea {
                    self.store_sse(x, o0);
                }
            }
            Gen::Vcvtss2sd | Gen::Vcvtsd2ss => {
                let k = if eg == Gen::Vcvtss2sd { K::CvtSs2Sd } else { K::CvtSd2Ss };
                self.kern(k, 0, vl, 0, [o0, o1, o2, 0], None);
            }
            Gen::Vcvtsi2sx => {
                let var = u32::from(repnz) | u32::from(x.op[2].ot == OT64) << 1;
                self.kern(K::CvtSi2S, var, vl, 0, [o0, o1, 0, 0], Some(g.t1));
            }
            Gen::Vcvtsx2si | Gen::Vcvttsx2si => {
                let var = u32::from(repnz)
                    | u32::from(x.op[0].ot == OT64) << 1
                    | u32::from(eg == Gen::Vcvttsx2si) << 2;
                let r = self.kern(K::CvtS2Si, var, 16, 0, [XMM_T0, o2, o2, 0], None);
                self.f().gen_mov_i64(g.t0, r);
            }
            Gen::Cvtpi2px => {
                self.env_call(&vec::ENTER_MMX, None, &[]);
                let k = if data { K::CvtPi2Pd } else { K::CvtPi2Ps };
                self.kern(k, 0, 16, 0, [o0, o2, o2, 0], None);
            }
            Gen::Cvtpx2pi | Gen::Cvttpx2pi => {
                self.env_call(&vec::ENTER_MMX, None, &[]);
                let k = if data { K::CvtPd2Pi } else { K::CvtPs2Pi };
                let var = u32::from(eg == Gen::Cvttpx2pi) << 2;
                self.kern(k, var, 8, 0, [o0, o2, o2, 0], None);
            }
            Gen::Movdq => self.store_sse(x, o2),
            Gen::MovdTo => {
                self.zero_env(o0, vl as usize);
                let ot = x.op[2].ot;
                self.st_env(ot, g.t1, o0);
            }
            Gen::MovdFrom => {
                let ot = x.op[2].ot;
                self.ld_env(ot, g.t0, o2);
            }
            Gen::Movq | Gen::MovqDq => {
                if eg == Gen::MovqDq {
                    self.env_call(&vec::ENTER_MMX, None, &[]);
                }
                let t = self.new64();
                self.ld_env(OT64, t, o2);
                if x.op[0].ea {
                    self.f().gen_qemu_st_i64(t, g.a0, idx, MemOp::LEUQ);
                } else {
                    self.zero_env(o0, vl as usize);
                    self.st_env(OT64, t, o0);
                }
            }
            Gen::Pextrb | Gen::Pextrw | Gen::Pextr | Gen::Vextractps => {
                let ot = match eg {
                    Gen::Pextrb => OT8,
                    Gen::Pextrw => OT16,
                    Gen::Pextr => x.op[0].ot,
                    _ => OT32,
                };
                let i = imm & ((vl >> ot) - 1);
                self.ld_env(ot, g.t0, o1 + ((i as usize) << ot));
            }
            Gen::Pinsrb | Gen::Pinsrw | Gen::Pinsr => {
                let ot = match eg {
                    Gen::Pinsrb => OT8,
                    Gen::Pinsrw => OT16,
                    _ => x.op[2].ot,
                };
                let i = imm & ((vl >> ot) - 1);
                if o1 != o0 {
                    self.store_sse(x, o1);
                }
                self.st_env(ot, g.t1, o0 + ((i as usize) << ot));
            }
            Gen::VinsertpsR => {
                self.kern(K::InsertPs, 0, 16, imm, [o0, o1, o2, 0], None);
            }
            Gen::VinsertpsM => {
                let t = self.new64();
                self.f().gen_qemu_ld_i64(t, g.a0, idx, MemOp::LEUL);
                self.kern(K::InsertPs, 1, 16, imm, [o0, o1, 0, 0], Some(t));
            }
            Gen::Vinsertx128 => {
                self.kern(K::Insert128, 0, 32, imm, [o0, o1, o2, 0], None);
            }
            Gen::Vextractx128 => {
                let src = o1 + 16 * (imm as usize & 1);
                if x.op[0].ea {
                    let mop = MemOp::LEUO | self.atom128();
                    self.sto(src as i64, mop);
                } else {
                    self.copy_env(o0, src, 16);
                }
            }
            Gen::Vmovss => {
                let t = self.new64();
                self.ld_env(OT32, t, o2);
                self.copy_env(o0, o1, vl as usize);
                self.st_env(OT32, t, o0);
            }
            Gen::VmovssLd => {
                let t = self.new64();
                self.f().gen_qemu_ld_i64(t, g.a0, idx, MemOp::LEUL);
                self.zero_env(o0, vl as usize);
                self.st_env(OT32, t, o0);
            }
            Gen::VmovssSt => {
                let t = self.new64();
                self.ld_env(OT32, t, o2);
                self.f().gen_qemu_st_i64(t, g.a0, idx, MemOp::LEUL);
            }
            Gen::VmovsdLd => {
                let t = self.new64();
                self.f().gen_qemu_ld_i64(t, g.a0, idx, MemOp::LEUQ);
                self.zero_env(o0 + 8, 8);
                self.st_env(OT64, t, o0);
            }
            Gen::Vmovlpx => {
                let t = self.new64();
                self.ld_env(OT64, t, o2);
                self.copy_env(o0, o1, vl as usize);
                self.st_env(OT64, t, o0);
            }
            Gen::VmovlpxLd => {
                let t = self.new64();
                self.f().gen_qemu_ld_i64(t, g.a0, idx, MemOp::LEUQ);
                self.copy_env(o0, o1, vl as usize);
                self.st_env(OT64, t, o0);
            }
            Gen::VmovlpxSt => {
                let t = self.new64();
                self.ld_env(OT64, t, o2);
                self.f().gen_qemu_st_i64(t, g.a0, idx, MemOp::LEUQ);
            }
            Gen::VmovhpxLd => {
                let t = self.new64();
                self.f().gen_qemu_ld_i64(t, g.a0, idx, MemOp::LEUQ);
                self.st_env(OT64, t, o0 + 8);
                if o0 != o1 {
                    self.copy_env(o0, o1, 8);
                }
            }
            Gen::VmovhpxSt => {
                let t = self.new64();
                self.ld_env(OT64, t, o2 + 8);
                self.f().gen_qemu_st_i64(t, g.a0, idx, MemOp::LEUQ);
            }
            Gen::Vmovhpx => {
                if o0 != o2 {
                    self.copy_env(o0 + 8, o2 + 8, 8);
                }
                if o0 != o1 {
                    self.copy_env(o0, o1, 8);
                }
            }
            Gen::Vmovhlps => {
                self.copy_env(o0, o2 + 8, 8);
                if o0 != o1 {
                    self.copy_env(o0 + 8, o1 + 8, 8);
                }
            }
            Gen::Vmovlhps => {
                self.copy_env(o0 + 8, o2, 8);
                if o0 != o1 {
                    self.copy_env(o0, o1, 8);
                }
            }
            Gen::Maskmov => {
                let (aflag, ovr) = (self.d.aflag, self.d.override_seg);
                let edi = g.regs[R_EDI];
                self.lea_v_seg(aflag, edi, R_DS as i32, ovr);
                let n = if data { 16 } else { 8 };
                self.masked_store(o1, o2, 1, n);
            }
            Gen::VmaskmovpsSt | Gen::VmaskmovpdSt | Gen::VpmaskmovSt => {
                let size = match eg {
                    Gen::VmaskmovpsSt => 4,
                    Gen::VmaskmovpdSt => 8,
                    _ if self.d.vex_w => 8,
                    _ => 4,
                };
                let n = if self.d.vex_l { 32 } else { 16 } / size;
                self.masked_store(o1, o2, size, n);
            }
            Gen::Emms => self.env_call(&vec::EMMS, None, &[]),
            Gen::Vzeroupper => {
                for i in 0..CPU_NB_REGS {
                    self.zero_env(zmm(i) + 16, 16);
                }
            }
            Gen::Vzeroall => {
                for i in 0..CPU_NB_REGS {
                    self.zero_env(zmm(i), ZMM_SIZE);
                }
            }
            Gen::Vaesenc | Gen::Vaesenclast | Gen::Vaesdec | Gen::Vaesdeclast => {
                let var = match eg {
                    Gen::Vaesenc => 0,
                    Gen::Vaesenclast => 1,
                    Gen::Vaesdec => 2,
                    _ => 3,
                };
                self.kern(K::Aes, var, vl, 0, [o0, o1, o2, 0], None);
            }
            Gen::Vaesimc => {
                self.kern(K::AesImc, 0, 16, 0, [o0, o2, o2, 0], None);
            }
            Gen::Vaeskeygen => {
                self.kern(K::AesKeygen, 0, 16, imm, [o0, o1, o1, 0], None);
            }
            Gen::Sha1rnds4 => {
                // SHA1RNDS4 has the operands V, W and I: the source is operand 1.
                self.kern(K::Sha, 0, 16, imm & 3, [o0, o0, o1, 0], None);
            }
            Gen::Sha1nexte
            | Gen::Sha1msg1
            | Gen::Sha1msg2
            | Gen::Sha256rnds2
            | Gen::Sha256msg1
            | Gen::Sha256msg2 => {
                let var = match eg {
                    Gen::Sha1nexte => 1,
                    Gen::Sha1msg1 => 2,
                    Gen::Sha1msg2 => 3,
                    Gen::Sha256rnds2 => 4,
                    Gen::Sha256msg1 => 5,
                    _ => 6,
                };
                self.kern(K::Sha, var, 16, 0, [o0, o1, o2, zmm(0)], None);
            }
            Gen::Pcmpestri | Gen::Pcmpestrm | Gen::Pcmpistri | Gen::Pcmpistrm => {
                let expl = matches!(eg, Gen::Pcmpestri | Gen::Pcmpestrm);
                let mask = matches!(eg, Gen::Pcmpestrm | Gen::Pcmpistrm);
                let var = u32::from(expl) | u32::from(mask) << 1 | u32::from(self.d.rex_w) << 2;
                let rax = if expl {
                    // RDX goes through the MMX scratch register, which the instruction does
                    // not otherwise use.
                    let rdx = g.regs[R_EDX];
                    self.f().gen_st_i64(rdx, g.env, MMX_T0 as i64);
                    Some(g.regs[R_EAX])
                } else {
                    None
                };
                let dst = if mask { zmm(0) } else { XMM_T0 };
                let r = self.kern(K::Pcmpstr, var, 16, imm, [dst, o1, o2, MMX_T0], rax);
                if mask {
                    if vex {
                        self.zero_env(zmm(0) + 16, 16);
                    }
                } else {
                    self.f().gen_shri_i64(g.regs[R_ECX], r, 32);
                }
                self.f().gen_ext32u_i64(r, r);
                self.sse_flags(r);
            }
            Gen::Vpgatherd | Gen::Vpgatherq => self.gather(x, eg == Gen::Vpgatherq),
            _ => {
                // Listed as not implemented in the module documentation.
                self.gen_illegal_opcode();
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// VPGATHERDD, VPGATHERDQ, VPGATHERQD, VPGATHERQQ and the VGATHER forms, which are the
    /// same operation: `helper_vpgather*` inline. Each element whose mask sign bit is set
    /// is loaded from A0 plus the scaled, sign extended index element; the mask element is
    /// cleared as it completes, so a fault leaves the finished elements behind.
    fn gather(&mut self, x: &Dx, qidx: bool) {
        let (a0, env, idx) = (self.g.a0, self.g.env, self.d.mem_index);
        let Some(mem) = x.mem else { return };
        let iof = zmm(mem.index as usize);
        let (d, m) = (x.op[0].off, x.op[1].off);
        let vl = if self.d.vex_l { 32 } else { 16 };
        let dot = if self.d.vex_w { OT64 } else { OT32 };
        let dsz = 1usize << dot;
        let isz = if qidx { 8 } else { 4 };
        let n = vl / dsz.max(isz);
        let a32 = self.d.aflag != OT64;
        let (t, a) = (self.new64(), self.new64());
        let z = self.c64(0);
        for i in 0..n {
            let skip = self.label();
            self.ld_env(OT8, t, m + i * dsz + dsz - 1);
            let f = self.f();
            f.gen_andi_i64(t, t, 0x80);
            f.gen_brcondi_i64(Cond::Eq, t, 0, skip);
            if qidx {
                f.gen_ld_i64(a, env, (iof + i * 8) as i64);
            } else {
                f.gen_ld32s_i64(a, env, (iof + i * 4) as i64);
            }
            f.gen_shli_i64(a, a, i64::from(mem.scale));
            f.gen_add_i64(a, a, a0);
            if a32 {
                f.gen_ext32u_i64(a, a);
            }
            let mop = if dot == OT64 { MemOp::LEUQ } else { MemOp::LEUL };
            f.gen_qemu_ld_i64(t, a, idx, mop);
            self.st_env(dot, t, d + i * dsz);
            self.set_label(skip);
            self.st_env(dot, z, m + i * dsz);
        }
        if qidx && dot == OT32 {
            // Quadword indices with doubleword data fill half of the destination.
            let half = vl / 2;
            self.zero_env(d + half, half);
            self.zero_env(m + half, half);
        }
        if !self.d.vex_l {
            // There are two outputs: the writeback clears the upper half of the destination,
            // and this the upper half of the mask.
            self.zero_env(m + 16, 16);
        }
    }

    /// Store the `n` elements of `size` bytes at `data` whose sign bit is set in the
    /// corresponding element at `mask` to A0: MASKMOVQ, MASKMOVDQU and the VEX masked
    /// stores.
    fn masked_store(&mut self, data: usize, mask: usize, size: usize, n: usize) {
        let (a0, idx) = (self.g.a0, self.d.mem_index);
        let ot = size.trailing_zeros();
        let t = self.new64();
        let a = self.new64();
        for i in 0..n {
            let skip = self.label();
            self.ld_env(OT8, t, mask + i * size + size - 1);
            let f = self.f();
            f.gen_andi_i64(t, t, 0x80);
            f.gen_brcondi_i64(Cond::Eq, t, 0, skip);
            self.ld_env(ot, t, data + i * size);
            let f = self.f();
            f.gen_addi_i64(a, a0, (i * size) as i64);
            f.gen_qemu_st_i64(t, a, idx, MemOp(ot));
            self.set_label(skip);
        }
    }

    /// VEX encoded LDMXCSR and STMXCSR (VEX.0F AE /2 and /3).
    pub(super) fn vldst_mxcsr(&mut self) -> R {
        let g = self.g;
        let m = self.ldub()? as u32;
        let op = (m >> 3) & 7;
        if m >> 6 == 3 || !(op == 2 || op == 3) || self.d.prefix & SSE_PREFIX != 0 {
            self.gen_illegal_opcode();
            return Ok(());
        }
        let a = self.lea_modrm_0(m)?;
        let fl = self.d.flags;
        if fl & HF_AVX_EN_MASK == 0 || self.d.vex_v != 0 {
            self.gen_illegal_opcode();
            return Ok(());
        }
        if fl & HF_TS_MASK != 0 {
            self.gen_exception(EXCP07_PREX);
            return Ok(());
        }
        if fl & HF_EM_MASK != 0 || self.d.vex_l {
            self.gen_illegal_opcode();
            return Ok(());
        }
        let ea = self.lea_modrm_1(a);
        let (aflag, ovr) = (self.d.aflag, self.d.override_seg);
        self.lea_v_seg(aflag, ea, a.def_seg, ovr);
        if op == 2 {
            self.ld_v(OT32, g.t0, g.a0);
            self.st_env32(g.t0, MXCSR);
        } else {
            let t = self.ld_env32(MXCSR);
            self.st_v(OT32, t, g.a0);
        }
        Ok(())
    }
}
