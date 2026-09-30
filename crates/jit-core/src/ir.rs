// SPDX-License-Identifier: MIT OR Apache-2.0

//! The function under construction: temps, labels, ops and helper descriptions.
//!
//! This is the part of QEMU's `TCGContext` that describes one translation block. Ops live in an
//! arena and are linked into a doubly linked list, so that passes can insert and remove ops while
//! walking the list the way QEMU's `QTAILQ` based passes do. A removed op stays in the arena but is
//! no longer linked.
//!
//! Temps are numbered as in QEMU: globals first, then everything else in allocation order. A
//! 128-bit temp is two consecutive 64-bit temps, low half first, as on a 64-bit little-endian
//! host.

use std::collections::{BTreeSet, HashMap};

use crate::opcode::Opcode;
use crate::types::{INSN_START_WORDS, TempKind, Type};

/// An untyped reference to a temp.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Temp(pub(crate) u32);

impl Temp {
    /// The index of the temp in the function, as QEMU's `temp_idx`.
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// The temp with the given index.
    pub const fn from_index(i: usize) -> Temp {
        Temp(i as u32)
    }

    /// The op argument that names this temp.
    pub const fn arg(self) -> u64 {
        self.0 as u64
    }

    /// The temp named by an op argument.
    pub const fn from_arg(a: u64) -> Temp {
        Temp(a as u32)
    }
}

macro_rules! typed_temp {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name(pub(crate) Temp);

        impl $name {
            /// The untyped temp.
            pub const fn temp(self) -> Temp {
                self.0
            }

            /// The op argument that names this temp.
            pub const fn arg(self) -> u64 {
                self.0.arg()
            }
        }

        impl From<$name> for Temp {
            fn from(t: $name) -> Temp {
                t.0
            }
        }
    };
}

typed_temp!(
    /// A 32-bit temp, `TCGv_i32`.
    TempI32
);
typed_temp!(
    /// A 64-bit temp, `TCGv_i64`.
    TempI64
);
typed_temp!(
    /// A 128-bit temp, `TCGv_i128`. It names the low of two consecutive 64-bit temps.
    TempI128
);
typed_temp!(
    /// A host pointer temp, `TCGv_ptr`. Its type is `I64`.
    TempPtr
);
typed_temp!(
    /// A vector temp, `TCGv_vec`.
    TempVec
);

impl TempI128 {
    /// `TCGV128_LOW`.
    pub const fn low(self) -> TempI64 {
        TempI64(self.0)
    }

    /// `TCGV128_HIGH`.
    pub const fn high(self) -> TempI64 {
        TempI64(Temp(self.0.0 + 1))
    }
}

impl TempPtr {
    /// View the pointer as a 64-bit integer temp, which it is on a 64-bit host.
    pub const fn as_i64(self) -> TempI64 {
        TempI64(self.0)
    }
}

impl TempI64 {
    /// View the temp as a pointer.
    pub const fn as_ptr(self) -> TempPtr {
        TempPtr(self.0)
    }
}

/// A branch target, `TCGLabel`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Label(pub(crate) u32);

impl Label {
    /// The label number printed as `$L<id>`.
    pub const fn id(self) -> u32 {
        self.0
    }

    /// The op argument that names this label.
    pub const fn arg(self) -> u64 {
        self.0 as u64
    }

    /// The label named by an op argument.
    pub const fn from_arg(a: u64) -> Label {
        Label(a as u32)
    }
}

/// A reference to an op in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpId(pub(crate) u32);

impl OpId {
    /// The arena index.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A registered helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HelperId(pub(crate) u32);

impl HelperId {
    /// The index into [`Func::helpers`].
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// The declared type of a helper argument or return value, QEMU's `dh_typecode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HelperType {
    /// No value; only valid as a return type.
    Void,
    /// An unsigned 32-bit value.
    I32,
    /// A signed 32-bit value.
    S32,
    /// An unsigned 64-bit value.
    I64,
    /// A signed 64-bit value.
    S64,
    /// A host pointer.
    Ptr,
    /// A 128-bit value.
    I128,
}

impl HelperType {
    /// The IR type that carries a value of this helper type.
    pub const fn ir_type(self) -> Option<Type> {
        match self {
            HelperType::Void => None,
            HelperType::I32 | HelperType::S32 => Some(Type::I32),
            HelperType::I64 | HelperType::S64 | HelperType::Ptr => Some(Type::I64),
            HelperType::I128 => Some(Type::I128),
        }
    }

    /// The number of op argument slots a value of this type takes.
    pub const fn slots(self) -> usize {
        match self {
            HelperType::Void => 0,
            HelperType::I128 => 2,
            _ => 1,
        }
    }
}

/// The description of a helper function, `TCGHelperInfo`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HelperInfo {
    /// The name printed in dumps and used to find the implementation.
    pub name: String,
    /// `call_flags` bits.
    pub flags: u32,
    /// The return type.
    pub ret: HelperType,
    /// The argument types.
    pub args: Vec<HelperType>,
}

impl HelperInfo {
    /// A helper description.
    pub fn new(name: &str, flags: u32, ret: HelperType, args: &[HelperType]) -> HelperInfo {
        HelperInfo { name: name.to_string(), flags, ret, args: args.to_vec() }
    }

    /// Number of output slots, `nr_out`.
    pub fn nr_out(&self) -> usize {
        self.ret.slots()
    }

    /// Number of input slots, `nr_in`.
    pub fn nr_in(&self) -> usize {
        self.args.iter().map(|a| a.slots()).sum()
    }
}

/// Everything known about one temp, `TCGTemp`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TempData {
    /// The lifetime class.
    pub kind: TempKind,
    /// The type of this part. For the halves of an I128 this is I64.
    pub ty: Type,
    /// The type of the whole value.
    pub base_type: Type,
    /// Which half of an I128 this is.
    pub subindex: u8,
    /// The value of a constant. I32 constants are stored sign extended.
    pub val: i64,
    /// The name of a global.
    pub name: Option<String>,
    /// The pointer temp a global is stored relative to.
    pub mem_base: Option<Temp>,
    /// The offset of a global from its base.
    pub mem_offset: i64,
    /// The global lives relative to another global rather than to a fixed register.
    pub indirect_reg: bool,
    /// Some other global lives relative to this one.
    pub indirect_base: bool,
    /// The temp is allocated (not on a free list).
    pub allocated: bool,
}

impl TempData {
    pub(crate) fn new(kind: TempKind, ty: Type, base_type: Type) -> TempData {
        TempData {
            kind,
            ty,
            base_type,
            subindex: 0,
            val: 0,
            name: None,
            mem_base: None,
            mem_offset: 0,
            indirect_reg: false,
            indirect_base: false,
            allocated: true,
        }
    }

    /// True for a constant temp.
    pub fn is_const(&self) -> bool {
        self.kind == TempKind::Const
    }
}

/// The maximum number of arguments an op can carry.
pub const MAX_OP_ARGS: usize = 24;

/// One IR operation, `TCGOp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Op {
    /// The opcode.
    pub opc: Opcode,
    /// The operand type, `TCGOP_TYPE`.
    pub ty: Type,
    /// The vector element size as log2 of bytes, `TCGOP_VECE`.
    pub vece: u8,
    /// Per op flags, `TCGOP_FLAGS`.
    pub flags: u8,
    /// Number of call outputs, `TCGOP_CALLO`.
    pub callo: u8,
    /// Number of call inputs, `TCGOP_CALLI`.
    pub calli: u8,
    /// The liveness result: `SYNC_ARG` and `DEAD_ARG` bits.
    pub life: u32,
    /// Number of meaningful entries in `args`.
    pub nargs: u8,
    /// The arguments: temps, then constants.
    pub args: [u64; MAX_OP_ARGS],
}

/// Bit for an output that must be synced to memory, `SYNC_ARG`.
pub const SYNC_ARG: u32 = 1;
/// Shift of the per argument dead bits, `DEAD_ARG`.
pub const DEAD_ARG: u32 = 1 << 4;

impl Op {
    fn new(opc: Opcode, ty: Type, nargs: usize) -> Op {
        assert!(nargs <= MAX_OP_ARGS, "too many op arguments");
        Op {
            opc,
            ty,
            vece: 0,
            flags: 0,
            callo: 0,
            calli: 0,
            life: 0,
            nargs: nargs as u8,
            args: [0; MAX_OP_ARGS],
        }
    }

    /// Number of output temps, counting call outputs.
    pub fn nb_oargs(&self) -> usize {
        if self.opc == Opcode::Call {
            self.callo as usize
        } else {
            self.opc.def().nb_oargs as usize
        }
    }

    /// Number of input temps, counting call inputs.
    pub fn nb_iargs(&self) -> usize {
        if self.opc == Opcode::Call {
            self.calli as usize
        } else {
            self.opc.def().nb_iargs as usize
        }
    }

    /// The temp in argument `i`.
    pub fn arg_temp(&self, i: usize) -> Temp {
        Temp::from_arg(self.args[i])
    }

    /// The label in argument `i`.
    pub fn arg_label(&self, i: usize) -> Label {
        Label::from_arg(self.args[i])
    }

    /// `IS_DEAD_ARG`.
    pub fn is_dead_arg(&self, n: usize) -> bool {
        self.life & (DEAD_ARG << n) != 0
    }

    /// `NEED_SYNC_ARG`.
    pub fn need_sync_arg(&self, n: usize) -> bool {
        self.life & (SYNC_ARG << n) != 0
    }

    /// For a call, the helper it calls.
    pub fn call_helper(&self) -> HelperId {
        HelperId(self.args[self.callo as usize + self.calli as usize] as u32)
    }
}

#[derive(Clone, Debug)]
struct OpNode {
    op: Op,
    prev: u32,
    next: u32,
    linked: bool,
}

const NIL: u32 = u32::MAX;

/// A branch target and the branches that use it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LabelData {
    /// `gen_set_label` has been called.
    pub present: bool,
    /// The br and brcond ops that jump here, in the order they were added.
    pub branches: Vec<OpId>,
}

/// Settings that QEMU takes from the translation block and the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FuncConfig {
    /// `CF_PARALLEL`: other vCPUs run at the same time, so atomics must be atomic.
    pub parallel: bool,
    /// Build for user mode emulation. `tcg_gen_mb` only emits a barrier in user mode when
    /// `parallel` is set; system mode always emits it.
    pub user_only: bool,
    /// The memory ordering the guest requires, `TCGCPUOps::guest_default_memory_order`.
    pub guest_mo: u32,
    /// The memory ordering the host gives for free, `TCG_TARGET_DEFAULT_MO`.
    pub target_default_mo: u32,
    /// The guest address type, `TCGContext::addr_type`.
    pub addr_type: Type,
    /// `CF_NO_GOTO_PTR`.
    pub no_goto_ptr: bool,
}

impl Default for FuncConfig {
    fn default() -> FuncConfig {
        FuncConfig {
            parallel: false,
            user_only: false,
            guest_mo: 0,
            target_default_mo: 0,
            addr_type: Type::I64,
            no_goto_ptr: false,
        }
    }
}

/// A translation block under construction, the per TB half of `TCGContext`.
#[derive(Clone, Debug)]
pub struct Func {
    pub(crate) temps: Vec<TempData>,
    pub(crate) nb_globals: usize,
    pub(crate) nb_indirects: usize,
    const_table: HashMap<(Type, i64), Temp>,
    free_temps: [BTreeSet<u32>; 6],
    pub(crate) labels: Vec<LabelData>,
    nodes: Vec<OpNode>,
    first: u32,
    last: u32,
    nb_ops: usize,
    pub(crate) helpers: Vec<HelperInfo>,
    helper_names: HashMap<String, HelperId>,
    /// The settings for this block.
    pub config: FuncConfig,
    pub(crate) last_insn_start: Option<OpId>,
    /// Number of `insn_start` ops emitted, `num_insns`.
    pub num_insns: u32,
}

impl Func {
    /// A new function with only the fixed `env` temp, which is temp 0.
    pub fn new(config: FuncConfig) -> Func {
        let mut f = Func {
            temps: Vec::new(),
            nb_globals: 0,
            nb_indirects: 0,
            const_table: HashMap::new(),
            free_temps: Default::default(),
            labels: Vec::new(),
            nodes: Vec::new(),
            first: NIL,
            last: NIL,
            nb_ops: 0,
            helpers: Vec::new(),
            helper_names: HashMap::new(),
            config,
            last_insn_start: None,
            num_insns: 0,
        };
        let mut env = TempData::new(TempKind::Fixed, Type::PTR, Type::PTR);
        env.name = Some("env".to_string());
        f.temps.push(env);
        f.nb_globals = 1;
        f
    }

    /// The `env` pointer, `tcg_env`.
    pub const fn env(&self) -> TempPtr {
        TempPtr(Temp(0))
    }

    /// Forget every op, label and non-global temp, `tcg_func_start`.
    pub fn reset(&mut self) {
        self.temps.truncate(self.nb_globals);
        self.const_table.clear();
        for f in &mut self.free_temps {
            f.clear();
        }
        self.labels.clear();
        self.nodes.clear();
        self.first = NIL;
        self.last = NIL;
        self.nb_ops = 0;
        self.last_insn_start = None;
        self.num_insns = 0;
    }

    // Temps.

    /// Number of globals, including `env`.
    pub fn nb_globals(&self) -> usize {
        self.nb_globals
    }

    /// Number of temps.
    pub fn nb_temps(&self) -> usize {
        self.temps.len()
    }

    /// Number of globals that live relative to another global.
    pub fn nb_indirects(&self) -> usize {
        self.nb_indirects
    }

    /// The data for a temp.
    pub fn temp(&self, t: impl Into<Temp>) -> &TempData {
        &self.temps[t.into().index()]
    }

    pub(crate) fn temp_mut(&mut self, t: Temp) -> &mut TempData {
        &mut self.temps[t.index()]
    }

    /// All temps, in index order.
    pub fn temps(&self) -> &[TempData] {
        &self.temps
    }

    pub(crate) fn temp_alloc(&mut self, td: TempData) -> Temp {
        let t = Temp(self.temps.len() as u32);
        self.temps.push(td);
        t
    }

    fn global_mem_new_internal(&mut self, base: Temp, offset: i64, name: &str, ty: Type) -> Temp {
        assert_eq!(self.nb_globals, self.temps.len(), "globals must be created before any temps");
        let indirect_reg = match self.temps[base.index()].kind {
            TempKind::Fixed => false,
            TempKind::Global => {
                assert!(
                    !self.temps[base.index()].indirect_reg,
                    "double-indirect registers are not supported"
                );
                self.temps[base.index()].indirect_base = true;
                self.nb_indirects += 1;
                true
            }
            _ => panic!("the base of a global must be a global"),
        };
        let mut td = TempData::new(TempKind::Global, ty, ty);
        td.indirect_reg = indirect_reg;
        td.mem_base = Some(base);
        td.mem_offset = offset;
        td.name = Some(name.to_string());
        let t = self.temp_alloc(td);
        self.nb_globals += 1;
        t
    }

    /// `tcg_global_mem_new_i32`.
    pub fn global_mem_new_i32(&mut self, base: TempPtr, offset: i64, name: &str) -> TempI32 {
        TempI32(self.global_mem_new_internal(base.0, offset, name, Type::I32))
    }

    /// `tcg_global_mem_new_i64`.
    pub fn global_mem_new_i64(&mut self, base: TempPtr, offset: i64, name: &str) -> TempI64 {
        TempI64(self.global_mem_new_internal(base.0, offset, name, Type::I64))
    }

    /// `tcg_global_mem_new_ptr`.
    pub fn global_mem_new_ptr(&mut self, base: TempPtr, offset: i64, name: &str) -> TempPtr {
        TempPtr(self.global_mem_new_internal(base.0, offset, name, Type::PTR))
    }

    /// `tcg_temp_new_internal`.
    pub fn temp_new_internal(&mut self, ty: Type, kind: TempKind) -> Temp {
        if kind == TempKind::Ebb {
            let list = &mut self.free_temps[ty as usize];
            if let Some(&idx) = list.iter().next() {
                list.remove(&idx);
                let td = &mut self.temps[idx as usize];
                td.allocated = true;
                debug_assert_eq!(td.base_type, ty);
                debug_assert_eq!(td.kind, kind);
                return Temp(idx);
            }
        } else {
            assert_eq!(kind, TempKind::Tb, "only EBB and TB temps can be allocated");
        }
        if ty == Type::I128 {
            let t = self.temp_alloc(TempData::new(kind, Type::REG, ty));
            let mut hi = TempData::new(kind, Type::REG, ty);
            hi.subindex = 1;
            self.temp_alloc(hi);
            t
        } else {
            self.temp_alloc(TempData::new(kind, ty, ty))
        }
    }

    /// `tcg_temp_free_internal`. Freeing a TB temp or a constant does nothing.
    pub fn temp_free(&mut self, t: impl Into<Temp>) {
        let t = t.into();
        let td = &mut self.temps[t.index()];
        match td.kind {
            TempKind::Const | TempKind::Tb => {}
            TempKind::Ebb => {
                assert!(td.allocated, "temp freed twice");
                td.allocated = false;
                let bt = td.base_type as usize;
                self.free_temps[bt].insert(t.0);
            }
            _ => panic!("globals cannot be freed"),
        }
    }

    /// `tcg_temp_ebb_reset_freed`: freed EBB temps are not reused after this.
    pub fn temp_ebb_reset_freed(&mut self) {
        for f in &mut self.free_temps {
            f.clear();
        }
    }

    /// `tcg_constant_internal`: the constant temp for `val`, created on first use.
    pub fn constant_internal(&mut self, ty: Type, val: i64) -> Temp {
        if let Some(&t) = self.const_table.get(&(ty, val)) {
            return t;
        }
        let mut td = TempData::new(TempKind::Const, ty, ty);
        td.val = val;
        let t = self.temp_alloc(td);
        self.const_table.insert((ty, val), t);
        t
    }

    /// `tcg_temp_new_i32`.
    pub fn temp_new_i32(&mut self) -> TempI32 {
        TempI32(self.temp_new_internal(Type::I32, TempKind::Tb))
    }

    /// `tcg_temp_ebb_new_i32`.
    pub fn temp_ebb_new_i32(&mut self) -> TempI32 {
        TempI32(self.temp_new_internal(Type::I32, TempKind::Ebb))
    }

    /// `tcg_temp_new_i64`.
    pub fn temp_new_i64(&mut self) -> TempI64 {
        TempI64(self.temp_new_internal(Type::I64, TempKind::Tb))
    }

    /// `tcg_temp_ebb_new_i64`.
    pub fn temp_ebb_new_i64(&mut self) -> TempI64 {
        TempI64(self.temp_new_internal(Type::I64, TempKind::Ebb))
    }

    /// `tcg_temp_new_i128`.
    pub fn temp_new_i128(&mut self) -> TempI128 {
        TempI128(self.temp_new_internal(Type::I128, TempKind::Tb))
    }

    /// `tcg_temp_ebb_new_i128`.
    pub fn temp_ebb_new_i128(&mut self) -> TempI128 {
        TempI128(self.temp_new_internal(Type::I128, TempKind::Ebb))
    }

    /// `tcg_temp_new_ptr`.
    pub fn temp_new_ptr(&mut self) -> TempPtr {
        TempPtr(self.temp_new_internal(Type::PTR, TempKind::Tb))
    }

    /// `tcg_temp_ebb_new_ptr`.
    pub fn temp_ebb_new_ptr(&mut self) -> TempPtr {
        TempPtr(self.temp_new_internal(Type::PTR, TempKind::Ebb))
    }

    /// `tcg_temp_new_vec`.
    pub fn temp_new_vec(&mut self, ty: Type) -> TempVec {
        assert!(ty.is_vector(), "not a vector type");
        TempVec(self.temp_new_internal(ty, TempKind::Ebb))
    }

    /// `tcg_temp_new_vec_matching`.
    pub fn temp_new_vec_matching(&mut self, m: TempVec) -> TempVec {
        let ty = self.temps[m.0.index()].base_type;
        self.temp_new_vec(ty)
    }

    /// `tcg_constant_i32`.
    pub fn constant_i32(&mut self, val: i32) -> TempI32 {
        TempI32(self.constant_internal(Type::I32, val as i64))
    }

    /// `tcg_constant_i64`.
    pub fn constant_i64(&mut self, val: i64) -> TempI64 {
        TempI64(self.constant_internal(Type::I64, val))
    }

    /// `tcg_constant_ptr`.
    pub fn constant_ptr(&mut self, val: i64) -> TempPtr {
        TempPtr(self.constant_internal(Type::PTR, val))
    }

    /// `tcg_constant_vec`.
    pub fn constant_vec(&mut self, ty: Type, vece: u32, val: i64) -> TempVec {
        let v = crate::types::dup_const(vece, val as u64) as i64;
        TempVec(self.constant_internal(ty, v))
    }

    /// `tcg_constant_vec_matching`.
    pub fn constant_vec_matching(&mut self, m: TempVec, vece: u32, val: i64) -> TempVec {
        let ty = self.temps[m.0.index()].base_type;
        self.constant_vec(ty, vece, val)
    }

    /// True if the temp is a constant.
    pub fn is_const(&self, t: impl Into<Temp>) -> bool {
        self.temps[t.into().index()].kind == TempKind::Const
    }

    // Labels.

    /// `gen_new_label`.
    pub fn new_label(&mut self) -> Label {
        let l = Label(self.labels.len() as u32);
        self.labels.push(LabelData::default());
        l
    }

    /// The data for a label.
    pub fn label(&self, l: Label) -> &LabelData {
        &self.labels[l.0 as usize]
    }

    /// Number of labels.
    pub fn nb_labels(&self) -> usize {
        self.labels.len()
    }

    pub(crate) fn add_label_use(&mut self, l: Label, op: OpId) {
        self.labels[l.0 as usize].branches.push(op);
    }

    fn remove_label_use(&mut self, op: OpId, idx: usize) {
        let l = self.op(op).arg_label(idx);
        let b = &mut self.labels[l.0 as usize].branches;
        let pos = b.iter().position(|&o| o == op).expect("label use not found");
        b.remove(pos);
    }

    /// `move_label_uses`: make every branch to `from` branch to `to` instead.
    pub(crate) fn move_label_uses(&mut self, to: Label, from: Label) {
        let uses = std::mem::take(&mut self.labels[from.0 as usize].branches);
        for &u in &uses {
            let op = self.op_mut(u);
            match op.opc {
                Opcode::Br => op.args[0] = to.arg(),
                Opcode::Brcond => op.args[3] = to.arg(),
                _ => unreachable!("label use is not a branch"),
            }
        }
        self.labels[to.0 as usize].branches.extend(uses);
    }

    // Helpers.

    /// Register a helper, or return the one already registered under the same name.
    pub fn helper(&mut self, info: HelperInfo) -> HelperId {
        if let Some(&id) = self.helper_names.get(&info.name) {
            assert_eq!(self.helpers[id.index()], info, "helper {} redeclared", info.name);
            return id;
        }
        let id = HelperId(self.helpers.len() as u32);
        self.helper_names.insert(info.name.clone(), id);
        self.helpers.push(info);
        id
    }

    /// The description of a registered helper.
    pub fn helper_info(&self, id: HelperId) -> &HelperInfo {
        &self.helpers[id.index()]
    }

    /// Every registered helper.
    pub fn helpers(&self) -> &[HelperInfo] {
        &self.helpers
    }

    // Ops.

    /// The op with this id, linked or not.
    pub fn op(&self, id: OpId) -> &Op {
        &self.nodes[id.index()].op
    }

    /// The op with this id, for changing in place.
    pub fn op_mut(&mut self, id: OpId) -> &mut Op {
        &mut self.nodes[id.index()].op
    }

    /// Number of linked ops, `nb_ops`.
    pub fn nb_ops(&self) -> usize {
        self.nb_ops
    }

    /// The first linked op.
    pub fn first_op(&self) -> Option<OpId> {
        (self.first != NIL).then_some(OpId(self.first))
    }

    /// The last linked op, `tcg_last_op`.
    pub fn last_op(&self) -> Option<OpId> {
        (self.last != NIL).then_some(OpId(self.last))
    }

    /// The op after `id`.
    pub fn next_op(&self, id: OpId) -> Option<OpId> {
        let n = self.nodes[id.index()].next;
        (n != NIL).then_some(OpId(n))
    }

    /// The op before `id`.
    pub fn prev_op(&self, id: OpId) -> Option<OpId> {
        let p = self.nodes[id.index()].prev;
        (p != NIL).then_some(OpId(p))
    }

    /// True if the op is still in the list.
    pub fn is_linked(&self, id: OpId) -> bool {
        self.nodes[id.index()].linked
    }

    /// The linked ops, in order.
    pub fn op_ids(&self) -> Vec<OpId> {
        let mut v = Vec::with_capacity(self.nb_ops);
        let mut cur = self.first;
        while cur != NIL {
            v.push(OpId(cur));
            cur = self.nodes[cur as usize].next;
        }
        v
    }

    /// Iterate over the linked ops in order.
    pub fn ops(&self) -> impl Iterator<Item = (OpId, &Op)> + '_ {
        let mut cur = self.first;
        std::iter::from_fn(move || {
            if cur == NIL {
                return None;
            }
            let id = cur;
            cur = self.nodes[id as usize].next;
            Some((OpId(id), &self.nodes[id as usize].op))
        })
    }

    fn alloc_op(&mut self, opc: Opcode, ty: Type, nargs: usize) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(OpNode { op: Op::new(opc, ty, nargs), prev: NIL, next: NIL, linked: true });
        self.nb_ops += 1;
        id
    }

    /// Append an op at the end, `tcg_emit_op` followed by filling the arguments.
    pub fn emit_op(&mut self, opc: Opcode, ty: Type, args: &[u64]) -> OpId {
        let id = self.alloc_op(opc, ty, args.len());
        self.nodes[id as usize].op.args[..args.len()].copy_from_slice(args);
        self.nodes[id as usize].prev = self.last;
        if self.last == NIL {
            self.first = id;
        } else {
            self.nodes[self.last as usize].next = id;
        }
        self.last = id;
        OpId(id)
    }

    /// `tcg_op_insert_before`.
    pub fn insert_before(&mut self, old: OpId, opc: Opcode, ty: Type, nargs: usize) -> OpId {
        let id = self.alloc_op(opc, ty, nargs);
        let prev = self.nodes[old.index()].prev;
        self.nodes[id as usize].prev = prev;
        self.nodes[id as usize].next = old.0;
        self.nodes[old.index()].prev = id;
        if prev == NIL {
            self.first = id;
        } else {
            self.nodes[prev as usize].next = id;
        }
        OpId(id)
    }

    /// `tcg_op_insert_after`.
    pub fn insert_after(&mut self, old: OpId, opc: Opcode, ty: Type, nargs: usize) -> OpId {
        let id = self.alloc_op(opc, ty, nargs);
        let next = self.nodes[old.index()].next;
        self.nodes[id as usize].prev = old.0;
        self.nodes[id as usize].next = next;
        self.nodes[old.index()].next = id;
        if next == NIL {
            self.last = id;
        } else {
            self.nodes[next as usize].prev = id;
        }
        OpId(id)
    }

    /// `tcg_op_remove`: unlink an op, dropping its label use if it is a branch.
    pub fn remove_op(&mut self, id: OpId) {
        match self.op(id).opc {
            Opcode::Br => self.remove_label_use(id, 0),
            Opcode::Brcond => self.remove_label_use(id, 3),
            _ => {}
        }
        let n = &self.nodes[id.index()];
        assert!(n.linked, "op removed twice");
        let (prev, next) = (n.prev, n.next);
        if prev == NIL {
            self.first = next;
        } else {
            self.nodes[prev as usize].next = next;
        }
        if next == NIL {
            self.last = prev;
        } else {
            self.nodes[next as usize].prev = prev;
        }
        let n = &mut self.nodes[id.index()];
        n.linked = false;
        n.prev = NIL;
        n.next = NIL;
        self.nb_ops -= 1;
    }

    /// `tcg_remove_ops_after`: remove every op after `op`, or every op if `op` is `None`.
    pub fn remove_ops_after(&mut self, op: Option<OpId>) {
        while let Some(last) = self.last_op() {
            if Some(last) == op {
                return;
            }
            self.remove_op(last);
        }
    }

    /// `tcg_set_insn_start_param`.
    pub fn set_insn_start_param(&mut self, op: OpId, arg: usize, v: u64) {
        assert!(arg < INSN_START_WORDS);
        self.op_mut(op).args[arg] = v;
    }

    /// `tcg_get_insn_start_param`.
    pub fn insn_start_param(&self, op: OpId, arg: usize) -> u64 {
        self.op(op).args[arg]
    }

    /// The most recent `insn_start` op, `tcg_ctx->emit_insn_start` style bookkeeping.
    pub fn last_insn_start(&self) -> Option<OpId> {
        self.last_insn_start
    }
}
