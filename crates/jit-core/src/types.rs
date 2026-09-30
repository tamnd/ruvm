// SPDX-License-Identifier: MIT OR Apache-2.0

//! The small value types of the IR: temp types and kinds, conditions, memory operations and the
//! flag words that ops and helper calls carry.
//!
//! The numeric values match QEMU's `TCGType`, `TCGTempKind`, `TCGCond`, `MemOp`, `TCGBar` and the
//! `TCG_CALL_*` flags, so that dumps print the same numbers and a front end ported from C can pass
//! the same constants.

use std::fmt;

/// The type of a temp or of an op, `TCGType`.
///
/// There is no separate `TCG_TYPE_PTR` or `TCG_TYPE_REG`: the IR always models a 64-bit host, so
/// both are [`Type::I64`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Type {
    /// A 32-bit integer.
    I32 = 0,
    /// A 64-bit integer.
    I64 = 1,
    /// A 128-bit integer, stored as two consecutive `I64` temps.
    I128 = 2,
    /// A 64-bit vector.
    V64 = 3,
    /// A 128-bit vector.
    V128 = 4,
    /// A 256-bit vector.
    V256 = 5,
}

impl Type {
    /// The host register type, `TCG_TYPE_REG`.
    pub const REG: Type = Type::I64;
    /// The host pointer type, `TCG_TYPE_PTR`.
    pub const PTR: Type = Type::I64;

    /// The size of a value of this type in bytes, `tcg_type_size`.
    pub const fn size(self) -> u32 {
        match self {
            Type::I32 => 4,
            Type::I64 | Type::V64 => 8,
            Type::I128 | Type::V128 => 16,
            Type::V256 => 32,
        }
    }

    /// The width in bits.
    pub const fn bits(self) -> u32 {
        self.size() * 8
    }

    /// True for the three vector types.
    pub const fn is_vector(self) -> bool {
        matches!(self, Type::V64 | Type::V128 | Type::V256)
    }

    /// True for `I32` and `I64`.
    pub const fn is_int(self) -> bool {
        matches!(self, Type::I32 | Type::I64)
    }

    /// Decode the numeric value used in QEMU.
    pub const fn from_u8(v: u8) -> Option<Type> {
        Some(match v {
            0 => Type::I32,
            1 => Type::I64,
            2 => Type::I128,
            3 => Type::V64,
            4 => Type::V128,
            5 => Type::V256,
            _ => return None,
        })
    }
}

/// How long a temp lives and where it is kept, `TCGTempKind`.
///
/// The order matters: the optimizer prefers the copy with the larger kind, as QEMU does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TempKind {
    /// Dead at the end of the extended basic block where it was defined.
    Ebb = 0,
    /// Lives until the end of the translation block.
    Tb = 1,
    /// Lives across translation blocks, kept in the CPU state.
    Global = 2,
    /// A global that is fixed to a host register, such as `env`.
    Fixed = 3,
    /// A read-only constant.
    Const = 4,
}

/// A comparison condition, `TCGCond`, with the same encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Cond {
    /// Never true.
    Never = 0,
    /// Always true.
    Always = 1,
    /// Signed less than.
    Lt = 2,
    /// Signed greater or equal.
    Ge = 3,
    /// Signed greater than.
    Gt = 6,
    /// Signed less or equal.
    Le = 7,
    /// Equal.
    Eq = 8,
    /// Not equal.
    Ne = 9,
    /// Unsigned less than.
    Ltu = 10,
    /// Unsigned greater or equal.
    Geu = 11,
    /// `(a & b) == 0`.
    TstEq = 12,
    /// `(a & b) != 0`.
    TstNe = 13,
    /// Unsigned greater than.
    Gtu = 14,
    /// Unsigned less or equal.
    Leu = 15,
}

impl Cond {
    /// Decode the numeric value used in QEMU. Values 4 and 5 are not conditions.
    pub const fn from_u64(v: u64) -> Option<Cond> {
        Some(match v {
            0 => Cond::Never,
            1 => Cond::Always,
            2 => Cond::Lt,
            3 => Cond::Ge,
            6 => Cond::Gt,
            7 => Cond::Le,
            8 => Cond::Eq,
            9 => Cond::Ne,
            10 => Cond::Ltu,
            11 => Cond::Geu,
            12 => Cond::TstEq,
            13 => Cond::TstNe,
            14 => Cond::Gtu,
            15 => Cond::Leu,
            _ => return None,
        })
    }

    const fn raw(v: u8) -> Cond {
        match Cond::from_u64(v as u64) {
            Some(c) => c,
            None => panic!("not a condition"),
        }
    }

    /// `tcg_invert_cond`.
    pub const fn invert(self) -> Cond {
        Cond::raw(self as u8 ^ 1)
    }

    /// `tcg_swap_cond`: the condition with the operands exchanged.
    pub const fn swap(self) -> Cond {
        let c = self as u8;
        Cond::raw(c ^ ((c & 2) << 1))
    }

    /// `is_signed_cond`.
    pub const fn is_signed(self) -> bool {
        (self as u8 & (8 | 2)) == 2
    }

    /// `is_unsigned_cond`.
    pub const fn is_unsigned(self) -> bool {
        (self as u8 & (8 | 2)) == (8 | 2)
    }

    /// `is_tst_cond`.
    pub const fn is_tst(self) -> bool {
        (self as u8 | 1) == Cond::TstNe as u8
    }

    /// `tcg_unsigned_cond`.
    pub const fn unsigned(self) -> Cond {
        if self.is_signed() { Cond::raw(self as u8 + 8) } else { self }
    }

    /// `tcg_signed_cond`.
    pub const fn signed(self) -> Cond {
        if self.is_unsigned() { Cond::raw(self as u8 - 8) } else { self }
    }

    /// `tcg_tst_eqne_cond`: TSTEQ becomes EQ and TSTNE becomes NE.
    pub const fn tst_eqne(self) -> Cond {
        if self.is_tst() { Cond::raw(self as u8 - 4) } else { self }
    }

    /// `tcg_tst_ltge_cond`: TSTEQ becomes GE and TSTNE becomes LT.
    pub const fn tst_ltge(self) -> Cond {
        if self.is_tst() { Cond::raw(self as u8 ^ 0xf) } else { self }
    }

    /// `tcg_high_cond`.
    pub const fn high(self) -> Cond {
        match self {
            Cond::Ge | Cond::Le | Cond::Geu | Cond::Leu => Cond::raw(self as u8 ^ (4 | 1)),
            _ => self,
        }
    }

    /// The name used in dumps.
    pub const fn name(self) -> &'static str {
        match self {
            Cond::Never => "never",
            Cond::Always => "always",
            Cond::Eq => "eq",
            Cond::Ne => "ne",
            Cond::Lt => "lt",
            Cond::Ge => "ge",
            Cond::Le => "le",
            Cond::Gt => "gt",
            Cond::Ltu => "ltu",
            Cond::Geu => "geu",
            Cond::Leu => "leu",
            Cond::Gtu => "gtu",
            Cond::TstEq => "tsteq",
            Cond::TstNe => "tstne",
        }
    }

    /// Evaluate the condition on two 64-bit values.
    pub const fn eval_u64(self, x: u64, y: u64) -> bool {
        match self {
            Cond::Never => false,
            Cond::Always => true,
            Cond::Eq => x == y,
            Cond::Ne => x != y,
            Cond::Lt => (x as i64) < (y as i64),
            Cond::Ge => (x as i64) >= (y as i64),
            Cond::Le => (x as i64) <= (y as i64),
            Cond::Gt => (x as i64) > (y as i64),
            Cond::Ltu => x < y,
            Cond::Geu => x >= y,
            Cond::Leu => x <= y,
            Cond::Gtu => x > y,
            Cond::TstEq => (x & y) == 0,
            Cond::TstNe => (x & y) != 0,
        }
    }

    /// Evaluate the condition on two 32-bit values.
    pub const fn eval_u32(self, x: u32, y: u32) -> bool {
        match self {
            Cond::Lt | Cond::Ge | Cond::Le | Cond::Gt => {
                self.eval_u64(x as i32 as i64 as u64, y as i32 as i64 as u64)
            }
            _ => self.eval_u64(x as u64, y as u64),
        }
    }
}

impl fmt::Display for Cond {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A guest memory operation, `MemOp`: size, sign, byte order, alignment and atomicity.
///
/// The encoding is QEMU's as seen on a little-endian host, so [`MemOp::BSWAP`] means big endian
/// no matter what the real host is. The interpreter reads it that way too.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MemOp(pub u32);

#[allow(missing_docs)]
impl MemOp {
    pub const MO_8: MemOp = MemOp(0);
    pub const MO_16: MemOp = MemOp(1);
    pub const MO_32: MemOp = MemOp(2);
    pub const MO_64: MemOp = MemOp(3);
    pub const MO_128: MemOp = MemOp(4);
    pub const MO_256: MemOp = MemOp(5);
    pub const MO_512: MemOp = MemOp(6);
    pub const MO_1024: MemOp = MemOp(7);
    pub const SIZE: MemOp = MemOp(7);
    pub const SIGN: MemOp = MemOp(8);
    pub const SSIZE: MemOp = MemOp(0xf);
    pub const BSWAP: MemOp = MemOp(0x10);
    pub const LE: MemOp = MemOp(0);
    pub const BE: MemOp = MemOp(0x10);
    pub const ASHIFT: u32 = 5;
    pub const AMASK: MemOp = MemOp(7 << 5);
    pub const UNALN: MemOp = MemOp(0);
    pub const ALIGN_2: MemOp = MemOp(1 << 5);
    pub const ALIGN_4: MemOp = MemOp(2 << 5);
    pub const ALIGN_8: MemOp = MemOp(3 << 5);
    pub const ALIGN_16: MemOp = MemOp(4 << 5);
    pub const ALIGN_32: MemOp = MemOp(5 << 5);
    pub const ALIGN_64: MemOp = MemOp(6 << 5);
    pub const ALIGN: MemOp = MemOp(7 << 5);
    pub const ALIGN_TLB_ONLY: MemOp = MemOp(1 << 8);
    pub const ATOM_SHIFT: u32 = 9;
    pub const ATOM_IFALIGN: MemOp = MemOp(0);
    pub const ATOM_IFALIGN_PAIR: MemOp = MemOp(1 << 9);
    pub const ATOM_WITHIN16: MemOp = MemOp(2 << 9);
    pub const ATOM_WITHIN16_PAIR: MemOp = MemOp(3 << 9);
    pub const ATOM_SUBALIGN: MemOp = MemOp(4 << 9);
    pub const ATOM_NONE: MemOp = MemOp(5 << 9);
    pub const ATOM_MASK: MemOp = MemOp(7 << 9);

    pub const UB: MemOp = MemOp(0);
    pub const UW: MemOp = MemOp(1);
    pub const UL: MemOp = MemOp(2);
    pub const UQ: MemOp = MemOp(3);
    pub const UO: MemOp = MemOp(4);
    pub const SB: MemOp = MemOp(8);
    pub const SW: MemOp = MemOp(9);
    pub const SL: MemOp = MemOp(10);
    pub const SQ: MemOp = MemOp(11);
    pub const SO: MemOp = MemOp(12);
    pub const LEUW: MemOp = MemOp(1);
    pub const LEUL: MemOp = MemOp(2);
    pub const LEUQ: MemOp = MemOp(3);
    pub const LEUO: MemOp = MemOp(4);
    pub const LESW: MemOp = MemOp(9);
    pub const LESL: MemOp = MemOp(10);
    pub const LESQ: MemOp = MemOp(11);
    pub const BEUW: MemOp = MemOp(0x11);
    pub const BEUL: MemOp = MemOp(0x12);
    pub const BEUQ: MemOp = MemOp(0x13);
    pub const BEUO: MemOp = MemOp(0x14);
    pub const BESW: MemOp = MemOp(0x19);
    pub const BESL: MemOp = MemOp(0x1a);
    pub const BESQ: MemOp = MemOp(0x1b);

    /// The access size as a `MO_8` .. `MO_1024` value.
    pub const fn size(self) -> u32 {
        self.0 & 7
    }

    /// The access size in bytes, `memop_size`.
    pub const fn size_bytes(self) -> u32 {
        1 << self.size()
    }

    /// True if the access sign extends.
    pub const fn is_signed(self) -> bool {
        self.0 & 8 != 0
    }

    /// True if the access is big endian.
    pub const fn is_bswap(self) -> bool {
        self.0 & 0x10 != 0
    }

    /// `memop_alignment_bits`: log2 of the required alignment.
    pub const fn alignment_bits(self) -> u32 {
        let a = self.0 & Self::AMASK.0;
        if a == Self::ALIGN.0 { self.size() } else { a >> Self::ASHIFT }
    }

    /// The bits in both.
    pub const fn and(self, o: MemOp) -> MemOp {
        MemOp(self.0 & o.0)
    }

    /// The bits in either.
    pub const fn or(self, o: MemOp) -> MemOp {
        MemOp(self.0 | o.0)
    }

    /// The bits of self that are not in `o`.
    pub const fn without(self, o: MemOp) -> MemOp {
        MemOp(self.0 & !o.0)
    }

    /// `size_memop`: the size field for an access of `bytes` bytes.
    pub const fn from_size_bytes(bytes: u32) -> MemOp {
        MemOp(bytes.trailing_zeros())
    }
}

impl std::ops::BitOr for MemOp {
    type Output = MemOp;
    fn bitor(self, o: MemOp) -> MemOp {
        MemOp(self.0 | o.0)
    }
}

impl std::ops::BitAnd for MemOp {
    type Output = MemOp;
    fn bitand(self, o: MemOp) -> MemOp {
        MemOp(self.0 & o.0)
    }
}

impl std::ops::Not for MemOp {
    type Output = MemOp;
    fn not(self) -> MemOp {
        MemOp(!self.0)
    }
}

impl fmt::Debug for MemOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemOp({:#x})", self.0)
    }
}

/// A memory operation together with an MMU index, `MemOpIdx`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MemOpIdx(pub u32);

impl MemOpIdx {
    /// `make_memop_idx`.
    pub const fn new(op: MemOp, mmu_idx: u32) -> MemOpIdx {
        MemOpIdx((op.0 << 4) | mmu_idx)
    }

    /// `get_memop`.
    pub const fn memop(self) -> MemOp {
        MemOp(self.0 >> 4)
    }

    /// `get_mmuidx`.
    pub const fn mmu_idx(self) -> u32 {
        self.0 & 15
    }
}

/// Op definition flags, `TCG_OPF_*`.
pub mod opf {
    /// Instruction exits the translation block.
    pub const BB_EXIT: u32 = 0x01;
    /// Instruction defines the end of a basic block.
    pub const BB_END: u32 = 0x02;
    /// Instruction clobbers call registers and potentially updates globals.
    pub const CALL_CLOBBER: u32 = 0x04;
    /// Instruction has side effects: it cannot be removed if its outputs are not used.
    pub const SIDE_EFFECTS: u32 = 0x08;
    /// Instruction operands may be I32 or I64.
    pub const INT: u32 = 0x10;
    /// Instruction is optional and not implemented by the host, or is a generic marker.
    pub const NOT_PRESENT: u32 = 0x20;
    /// Instruction operands are vectors.
    pub const VECTOR: u32 = 0x40;
    /// Instruction is a conditional branch.
    pub const COND_BRANCH: u32 = 0x80;
    /// Instruction produces carry out.
    pub const CARRY_OUT: u32 = 0x100;
    /// Instruction consumes carry in.
    pub const CARRY_IN: u32 = 0x200;
}

/// Helper call flags, `TCG_CALL_*`.
pub mod call_flags {
    /// The helper does not read globals.
    pub const NO_READ_GLOBALS: u32 = 0x0001;
    /// The helper does not write globals.
    pub const NO_WRITE_GLOBALS: u32 = 0x0002;
    /// The helper can be removed if its result is not used.
    pub const NO_SIDE_EFFECTS: u32 = 0x0004;
    /// The helper never returns (it raises an exception).
    pub const NO_RETURN: u32 = 0x0008;
    /// `TCG_CALL_NO_RWG`.
    pub const NO_RWG: u32 = NO_READ_GLOBALS;
    /// `TCG_CALL_NO_WG`.
    pub const NO_WG: u32 = NO_WRITE_GLOBALS;
    /// `TCG_CALL_NO_SE`.
    pub const NO_SE: u32 = NO_SIDE_EFFECTS;
    /// `TCG_CALL_NO_RWG_SE`.
    pub const NO_RWG_SE: u32 = NO_RWG | NO_SE;
    /// `TCG_CALL_NO_WG_SE`.
    pub const NO_WG_SE: u32 = NO_WG | NO_SE;
}

/// Flags for the bswap ops, `TCG_BSWAP_*`.
pub mod bswap {
    /// The input is zero extended.
    pub const IZ: u32 = 1;
    /// Zero extend the output.
    pub const OZ: u32 = 2;
    /// Sign extend the output.
    pub const OS: u32 = 4;
}

/// Memory ordering constraints, `TCGBar` and `TCG_MO_*`.
pub mod mo {
    /// Load then load.
    pub const LD_LD: u32 = 0x01;
    /// Store then load.
    pub const ST_LD: u32 = 0x02;
    /// Load then store.
    pub const LD_ST: u32 = 0x04;
    /// Store then store.
    pub const ST_ST: u32 = 0x08;
    /// Every ordering.
    pub const ALL: u32 = 0x0f;
    /// Acquire barrier.
    pub const BAR_LDAQ: u32 = 0x10;
    /// Release barrier.
    pub const BAR_STRL: u32 = 0x20;
    /// Sequentially consistent barrier.
    pub const BAR_SC: u32 = 0x30;
}

/// The low bits of an `exit_tb` value, `TB_EXIT_*`.
pub mod tb_exit {
    /// The mask for the exit index.
    pub const MASK: u64 = 3;
    /// Exit through goto_tb slot 0.
    pub const IDX0: u64 = 0;
    /// Exit through goto_tb slot 1.
    pub const IDX1: u64 = 1;
    /// The largest goto_tb slot.
    pub const IDXMAX: u64 = 1;
    /// The exit was requested, for example by an interrupt.
    pub const REQUESTED: u64 = 3;
}

/// Where a plugin callback sits, `enum plugin_gen_from`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PluginFrom {
    /// At the start of the block.
    Tb = 0,
    /// At the start of an instruction.
    Insn = 1,
    /// After an instruction.
    AfterInsn = 2,
    /// After the block.
    AfterTb = 3,
}

/// The number of words an `insn_start` op carries, `INSN_START_WORDS`.
pub const INSN_START_WORDS: usize = 3;

/// `dup_const`: replicate the low `8 << vece` bits of `c` across 64 bits.
pub const fn dup_const(vece: u32, c: u64) -> u64 {
    match vece {
        0 => 0x0101_0101_0101_0101u64.wrapping_mul(c as u8 as u64),
        1 => 0x0001_0001_0001_0001u64.wrapping_mul(c as u16 as u64),
        2 => 0x0000_0001_0000_0001u64.wrapping_mul(c as u32 as u64),
        3 => c,
        _ => panic!("dup_const: bad vece"),
    }
}
