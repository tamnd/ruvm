// SPDX-License-Identifier: MIT OR Apache-2.0

//! The opcode table, a port of `include/tcg/tcg-opc.h`.
//!
//! Every generic op from QEMU 11.1 is here with the same name, the same argument counts and the
//! same flags. Integer ops carry their type (I32 or I64) on the op rather than in the opcode, as
//! in QEMU 11, so `add` covers both `add_i32` and `add_i64`.

use crate::types::opf;

/// The static description of an opcode, `TCGOpDef`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpDef {
    /// The name used in dumps.
    pub name: &'static str,
    /// Number of output arguments.
    pub nb_oargs: u8,
    /// Number of input arguments.
    pub nb_iargs: u8,
    /// Number of constant arguments.
    pub nb_cargs: u8,
    /// `opf` flags.
    pub flags: u32,
}

impl OpDef {
    /// Total number of arguments.
    pub const fn nb_args(&self) -> usize {
        self.nb_oargs as usize + self.nb_iargs as usize + self.nb_cargs as usize
    }
}

macro_rules! opcodes {
    ($($v:ident => $name:literal, $o:expr, $i:expr, $c:expr, $f:expr;)*) => {
        /// A generic IR opcode, `TCGOpcode`.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[allow(missing_docs)]
        pub enum Opcode {
            $($v,)*
        }

        /// Every opcode, in table order.
        pub const ALL_OPCODES: &[Opcode] = &[$(Opcode::$v,)*];

        impl Opcode {
            /// The definition of this opcode.
            pub const fn def(self) -> &'static OpDef {
                match self {
                    $(Opcode::$v => &OpDef {
                        name: $name,
                        nb_oargs: $o,
                        nb_iargs: $i,
                        nb_cargs: $c,
                        flags: $f,
                    },)*
                }
            }
        }
    };
}

impl Opcode {
    /// The name used in dumps.
    pub const fn name(self) -> &'static str {
        self.def().name
    }

    /// The `opf` flags.
    pub const fn flags(self) -> u32 {
        self.def().flags
    }

    /// Look an opcode up by its name.
    pub fn from_name(name: &str) -> Option<Opcode> {
        ALL_OPCODES.iter().copied().find(|o| o.name() == name)
    }
}

opcodes! {
    Discard => "discard", 1, 0, 0, opf::NOT_PRESENT;
    SetLabel => "set_label", 0, 0, 1, opf::BB_END | opf::NOT_PRESENT;
    Call => "call", 0, 0, 3, opf::CALL_CLOBBER | opf::NOT_PRESENT;
    Br => "br", 0, 0, 1, opf::BB_END | opf::NOT_PRESENT;
    Brcond => "brcond", 0, 2, 2, opf::BB_END | opf::COND_BRANCH | opf::INT;
    Mb => "mb", 0, 0, 1, opf::NOT_PRESENT;
    Mov => "mov", 1, 1, 0, opf::INT | opf::NOT_PRESENT;
    Add => "add", 1, 2, 0, opf::INT;
    And => "and", 1, 2, 0, opf::INT;
    Andc => "andc", 1, 2, 0, opf::INT;
    Bswap16 => "bswap16", 1, 1, 1, opf::INT;
    Bswap32 => "bswap32", 1, 1, 1, opf::INT;
    Bswap64 => "bswap64", 1, 1, 1, opf::INT;
    Clz => "clz", 1, 2, 0, opf::INT;
    Ctpop => "ctpop", 1, 1, 0, opf::INT;
    Ctz => "ctz", 1, 2, 0, opf::INT;
    Deposit => "deposit", 1, 2, 2, opf::INT;
    Divs => "divs", 1, 2, 0, opf::INT;
    Divs2 => "divs2", 2, 3, 0, opf::INT;
    Divu => "divu", 1, 2, 0, opf::INT;
    Divu2 => "divu2", 2, 3, 0, opf::INT;
    Eqv => "eqv", 1, 2, 0, opf::INT;
    Extract => "extract", 1, 1, 2, opf::INT;
    Extract2 => "extract2", 1, 2, 1, opf::INT;
    Ld8u => "ld8u", 1, 1, 1, opf::INT;
    Ld8s => "ld8s", 1, 1, 1, opf::INT;
    Ld16u => "ld16u", 1, 1, 1, opf::INT;
    Ld16s => "ld16s", 1, 1, 1, opf::INT;
    Ld32u => "ld32u", 1, 1, 1, opf::INT;
    Ld32s => "ld32s", 1, 1, 1, opf::INT;
    Ld => "ld", 1, 1, 1, opf::INT;
    Movcond => "movcond", 1, 4, 1, opf::INT;
    Mul => "mul", 1, 2, 0, opf::INT;
    Muls2 => "muls2", 2, 2, 0, opf::INT;
    Mulsh => "mulsh", 1, 2, 0, opf::INT;
    Mulu2 => "mulu2", 2, 2, 0, opf::INT;
    Muluh => "muluh", 1, 2, 0, opf::INT;
    Nand => "nand", 1, 2, 0, opf::INT;
    Neg => "neg", 1, 1, 0, opf::INT;
    Negsetcond => "negsetcond", 1, 2, 1, opf::INT;
    Nor => "nor", 1, 2, 0, opf::INT;
    Not => "not", 1, 1, 0, opf::INT;
    Or => "or", 1, 2, 0, opf::INT;
    Orc => "orc", 1, 2, 0, opf::INT;
    Rems => "rems", 1, 2, 0, opf::INT;
    Remu => "remu", 1, 2, 0, opf::INT;
    Rotl => "rotl", 1, 2, 0, opf::INT;
    Rotr => "rotr", 1, 2, 0, opf::INT;
    Sar => "sar", 1, 2, 0, opf::INT;
    Setcond => "setcond", 1, 2, 1, opf::INT;
    Sextract => "sextract", 1, 1, 2, opf::INT;
    Shl => "shl", 1, 2, 0, opf::INT;
    Shr => "shr", 1, 2, 0, opf::INT;
    St8 => "st8", 0, 2, 1, opf::INT;
    St16 => "st16", 0, 2, 1, opf::INT;
    St32 => "st32", 0, 2, 1, opf::INT;
    St => "st", 0, 2, 1, opf::INT;
    Sub => "sub", 1, 2, 0, opf::INT;
    Xor => "xor", 1, 2, 0, opf::INT;
    Addco => "addco", 1, 2, 0, opf::INT | opf::CARRY_OUT;
    Addc1o => "addc1o", 1, 2, 0, opf::INT | opf::CARRY_OUT;
    Addci => "addci", 1, 2, 0, opf::INT | opf::CARRY_IN;
    Addcio => "addcio", 1, 2, 0, opf::INT | opf::CARRY_IN | opf::CARRY_OUT;
    Subbo => "subbo", 1, 2, 0, opf::INT | opf::CARRY_OUT;
    Subb1o => "subb1o", 1, 2, 0, opf::INT | opf::CARRY_OUT;
    Subbi => "subbi", 1, 2, 0, opf::INT | opf::CARRY_IN;
    Subbio => "subbio", 1, 2, 0, opf::INT | opf::CARRY_IN | opf::CARRY_OUT;
    ExtI32I64 => "ext_i32_i64", 1, 1, 0, 0;
    ExtuI32I64 => "extu_i32_i64", 1, 1, 0, 0;
    ExtrlI64I32 => "extrl_i64_i32", 1, 1, 0, 0;
    ExtrhI64I32 => "extrh_i64_i32", 1, 1, 0, 0;
    InsnStart => "insn_start", 0, 0, 3, opf::NOT_PRESENT;
    ExitTb => "exit_tb", 0, 0, 1, opf::BB_EXIT | opf::BB_END | opf::NOT_PRESENT;
    GotoTb => "goto_tb", 0, 0, 1, opf::BB_EXIT | opf::BB_END | opf::NOT_PRESENT;
    GotoPtr => "goto_ptr", 0, 1, 0, opf::BB_EXIT | opf::BB_END;
    PluginCb => "plugin_cb", 0, 0, 1, opf::NOT_PRESENT;
    PluginMemCb => "plugin_mem_cb", 0, 1, 1, opf::NOT_PRESENT;
    QemuLd => "qemu_ld", 1, 1, 1, opf::CALL_CLOBBER | opf::SIDE_EFFECTS | opf::INT;
    QemuSt => "qemu_st", 0, 2, 1, opf::CALL_CLOBBER | opf::SIDE_EFFECTS | opf::INT;
    QemuLd2 => "qemu_ld2", 2, 1, 1, opf::CALL_CLOBBER | opf::SIDE_EFFECTS | opf::INT;
    QemuSt2 => "qemu_st2", 0, 3, 1, opf::CALL_CLOBBER | opf::SIDE_EFFECTS | opf::INT;
    MovVec => "mov_vec", 1, 1, 0, opf::VECTOR | opf::NOT_PRESENT;
    DupVec => "dup_vec", 1, 1, 0, opf::VECTOR;
    LdVec => "ld_vec", 1, 1, 1, opf::VECTOR;
    StVec => "st_vec", 0, 2, 1, opf::VECTOR;
    DupmVec => "dupm_vec", 1, 1, 1, opf::VECTOR;
    AddVec => "add_vec", 1, 2, 0, opf::VECTOR;
    SubVec => "sub_vec", 1, 2, 0, opf::VECTOR;
    MulVec => "mul_vec", 1, 2, 0, opf::VECTOR;
    NegVec => "neg_vec", 1, 1, 0, opf::VECTOR;
    AbsVec => "abs_vec", 1, 1, 0, opf::VECTOR;
    SsaddVec => "ssadd_vec", 1, 2, 0, opf::VECTOR;
    UsaddVec => "usadd_vec", 1, 2, 0, opf::VECTOR;
    SssubVec => "sssub_vec", 1, 2, 0, opf::VECTOR;
    UssubVec => "ussub_vec", 1, 2, 0, opf::VECTOR;
    SminVec => "smin_vec", 1, 2, 0, opf::VECTOR;
    UminVec => "umin_vec", 1, 2, 0, opf::VECTOR;
    SmaxVec => "smax_vec", 1, 2, 0, opf::VECTOR;
    UmaxVec => "umax_vec", 1, 2, 0, opf::VECTOR;
    AndVec => "and_vec", 1, 2, 0, opf::VECTOR;
    OrVec => "or_vec", 1, 2, 0, opf::VECTOR;
    XorVec => "xor_vec", 1, 2, 0, opf::VECTOR;
    AndcVec => "andc_vec", 1, 2, 0, opf::VECTOR;
    OrcVec => "orc_vec", 1, 2, 0, opf::VECTOR;
    NandVec => "nand_vec", 1, 2, 0, opf::VECTOR;
    NorVec => "nor_vec", 1, 2, 0, opf::VECTOR;
    EqvVec => "eqv_vec", 1, 2, 0, opf::VECTOR;
    NotVec => "not_vec", 1, 1, 0, opf::VECTOR;
    ShliVec => "shli_vec", 1, 1, 1, opf::VECTOR;
    ShriVec => "shri_vec", 1, 1, 1, opf::VECTOR;
    SariVec => "sari_vec", 1, 1, 1, opf::VECTOR;
    RotliVec => "rotli_vec", 1, 1, 1, opf::VECTOR;
    ShlsVec => "shls_vec", 1, 2, 0, opf::VECTOR;
    ShrsVec => "shrs_vec", 1, 2, 0, opf::VECTOR;
    SarsVec => "sars_vec", 1, 2, 0, opf::VECTOR;
    RotlsVec => "rotls_vec", 1, 2, 0, opf::VECTOR;
    ShlvVec => "shlv_vec", 1, 2, 0, opf::VECTOR;
    ShrvVec => "shrv_vec", 1, 2, 0, opf::VECTOR;
    SarvVec => "sarv_vec", 1, 2, 0, opf::VECTOR;
    RotlvVec => "rotlv_vec", 1, 2, 0, opf::VECTOR;
    RotrvVec => "rotrv_vec", 1, 2, 0, opf::VECTOR;
    CmpVec => "cmp_vec", 1, 2, 1, opf::VECTOR;
    BitselVec => "bitsel_vec", 1, 3, 0, opf::VECTOR;
    CmpselVec => "cmpsel_vec", 1, 4, 1, opf::VECTOR;
}
