// SPDX-License-Identifier: MIT OR Apache-2.0

//! An IR verifier.
//!
//! QEMU has no single verifier; it relies on `tcg_debug_assert` calls spread through the builder
//! and the backends. This pass collects the checks that matter for a portable IR in one place:
//!
//! - every op has the argument count its opcode declares, and every temp argument names a temp
//!   that exists;
//! - every temp argument has the type the op expects, including helper calls, which are checked
//!   against the declared signature of the helper;
//! - an EBB temp is defined in the current extended basic block before it is read, and a TB temp
//!   is defined somewhere earlier in the op list before it is read;
//! - every branch names a label that exists and is set exactly once, and no label is set twice;
//! - every carry-in op directly follows a carry-out op.

use std::fmt;

use crate::ir::{Func, HelperType, Label, Op, OpId, Temp};
use crate::opcode::Opcode;
use crate::types::{TempKind, Type, opf};

/// One problem found by [`Func::verify`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyError {
    /// The op the problem was found at, if any.
    pub op: Option<OpId>,
    /// A description of the problem.
    pub msg: String,
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.op {
            Some(op) => write!(f, "op {}: {}", op.index(), self.msg),
            None => f.write_str(&self.msg),
        }
    }
}

impl std::error::Error for VerifyError {}

/// What a temp argument must look like.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Want {
    /// An integer temp of exactly this type.
    Exact(Type),
    /// Any I32 or I64 temp.
    AnyInt,
    /// A vector temp at least as wide as this type.
    Vec(Type),
}

struct Checker<'a> {
    f: &'a Func,
    errors: Vec<VerifyError>,
}

impl Checker<'_> {
    fn err(&mut self, op: Option<OpId>, msg: String) {
        self.errors.push(VerifyError { op, msg });
    }

    fn check_type(&mut self, id: OpId, op: &Op, i: usize, want: Want) {
        let t = op.arg_temp(i);
        let ty = self.f.temp(t).ty;
        let ok = match want {
            Want::Exact(w) => ty == w,
            Want::AnyInt => ty.is_int(),
            Want::Vec(w) => ty.is_vector() && ty >= w,
        };
        if !ok {
            let name = self.f.temp_name(t);
            self.err(
                Some(id),
                format!(
                    "{}: argument {i} ({name}) has type {ty:?}, expected {want:?}",
                    op.opc.name()
                ),
            );
        }
    }

    /// The expected type of each temp argument of a non-call op.
    fn wants(&self, op: &Op) -> Vec<Want> {
        let def = op.opc.def();
        let n = def.nb_oargs as usize + def.nb_iargs as usize;
        let ty = op.ty;
        let ptr = Want::Exact(Type::PTR);
        let addr = Want::Exact(self.f.config.addr_type);
        match op.opc {
            Opcode::Mov => vec![Want::Exact(ty), Want::AnyInt],
            Opcode::Discard => vec![Want::AnyInt],
            Opcode::ExtI32I64 | Opcode::ExtuI32I64 => {
                vec![Want::Exact(Type::I64), Want::Exact(Type::I32)]
            }
            Opcode::ExtrlI64I32 | Opcode::ExtrhI64I32 => {
                vec![Want::Exact(Type::I32), Want::Exact(Type::I64)]
            }
            Opcode::Ld8u
            | Opcode::Ld8s
            | Opcode::Ld16u
            | Opcode::Ld16s
            | Opcode::Ld32u
            | Opcode::Ld32s
            | Opcode::Ld
            | Opcode::St8
            | Opcode::St16
            | Opcode::St32
            | Opcode::St => vec![Want::Exact(ty), ptr],
            Opcode::QemuLd | Opcode::QemuSt => vec![Want::Exact(ty), addr],
            Opcode::QemuLd2 | Opcode::QemuSt2 => {
                vec![Want::Exact(Type::I64), Want::Exact(Type::I64), addr]
            }
            Opcode::GotoPtr => vec![ptr],
            Opcode::PluginMemCb => vec![Want::Exact(Type::I64)],
            Opcode::DupVec => vec![Want::Vec(ty), Want::AnyInt],
            Opcode::LdVec | Opcode::StVec | Opcode::DupmVec => vec![Want::Vec(ty), ptr],
            Opcode::ShlsVec | Opcode::ShrsVec | Opcode::SarsVec | Opcode::RotlsVec => {
                vec![Want::Vec(ty), Want::Vec(ty), Want::Exact(Type::I32)]
            }
            _ if def.flags & opf::VECTOR != 0 => vec![Want::Vec(ty); n],
            _ => vec![Want::Exact(ty); n],
        }
    }

    fn check_op_types(&mut self, id: OpId, op: &Op) {
        let def = op.opc.def();
        if def.flags & opf::INT != 0
            && !matches!(op.opc, Opcode::QemuLd2 | Opcode::QemuSt2)
            && !matches!(op.ty, Type::I32 | Type::I64)
        {
            self.err(Some(id), format!("{}: op type {:?} is not I32 or I64", def.name, op.ty));
            return;
        }
        if def.flags & opf::VECTOR != 0 {
            if !op.ty.is_vector() {
                self.err(Some(id), format!("{}: op type {:?} is not a vector", def.name, op.ty));
                return;
            }
            if op.vece > 3 {
                self.err(Some(id), format!("{}: bad element size {}", def.name, op.vece));
            }
        }
        let wants = self.wants(op);
        for (i, w) in wants.into_iter().enumerate() {
            self.check_type(id, op, i, w);
        }
    }

    fn check_call(&mut self, id: OpId, op: &Op) {
        let hid = op.call_helper();
        if hid.0 as usize >= self.f.helpers().len() {
            self.err(Some(id), format!("call: unknown helper {}", hid.0));
            return;
        }
        let info = self.f.helper_info(hid);
        if op.callo as usize != info.nr_out() || op.calli as usize != info.nr_in() {
            self.err(
                Some(id),
                format!(
                    "call {}: {} outputs and {} inputs, the helper declares {} and {}",
                    info.name,
                    op.callo,
                    op.calli,
                    info.nr_out(),
                    info.nr_in()
                ),
            );
            return;
        }
        let mut slots: Vec<Want> = Vec::new();
        let mut push = |t: HelperType| match t.ir_type() {
            None => {}
            Some(Type::I128) => {
                slots.push(Want::Exact(Type::I64));
                slots.push(Want::Exact(Type::I64));
            }
            Some(ty) => slots.push(Want::Exact(ty)),
        };
        push(info.ret);
        for &a in &info.args {
            push(a);
        }
        for (i, w) in slots.into_iter().enumerate() {
            self.check_type(id, op, i, w);
        }
    }

    fn run(&mut self) {
        let f = self.f;
        let nb_temps = f.nb_temps();
        let nb_labels = f.nb_labels();
        // For EBB temps: defined in the current EBB. For TB temps: defined earlier in the list.
        let mut ebb_def = vec![false; nb_temps];
        let mut tb_def = vec![false; nb_temps];
        let mut label_set = vec![false; nb_labels];
        let mut label_used: Vec<Option<OpId>> = vec![None; nb_labels];
        let mut prev: Option<Opcode> = None;

        for (id, op) in f.ops() {
            let def = op.opc.def();
            let (nb_o, nb_i) = (op.nb_oargs(), op.nb_iargs());
            if op.opc == Opcode::Call {
                if op.nargs as usize != nb_o + nb_i + 2 {
                    self.err(Some(id), format!("call: {} arguments", op.nargs));
                    prev = Some(op.opc);
                    continue;
                }
            } else if op.nargs as usize != def.nb_args() {
                self.err(
                    Some(id),
                    format!("{}: {} arguments, expected {}", def.name, op.nargs, def.nb_args()),
                );
                prev = Some(op.opc);
                continue;
            }

            let mut bad_temp = false;
            for i in 0..nb_o + nb_i {
                let a = op.args[i];
                if a as usize >= nb_temps {
                    self.err(Some(id), format!("{}: argument {i} is not a temp", def.name));
                    bad_temp = true;
                }
            }
            if bad_temp {
                prev = Some(op.opc);
                continue;
            }

            if op.opc == Opcode::Call {
                self.check_call(id, op);
            } else {
                self.check_op_types(id, op);
            }

            if def.flags & opf::CARRY_IN != 0 {
                let ok = prev.is_some_and(|p| p.flags() & opf::CARRY_OUT != 0);
                if !ok {
                    self.err(
                        Some(id),
                        format!("{}: carry-in op does not follow a carry-out op", def.name),
                    );
                }
            }

            // Reads happen before writes.
            for i in nb_o..nb_o + nb_i {
                let t = op.arg_temp(i);
                let td = f.temp(t);
                let missing = match td.kind {
                    TempKind::Ebb => !ebb_def[t.index()],
                    TempKind::Tb => !tb_def[t.index()],
                    _ => false,
                };
                if missing {
                    self.err(
                        Some(id),
                        format!(
                            "{}: {} is read before it is written{}",
                            def.name,
                            f.temp_name(t),
                            if td.kind == TempKind::Ebb { " in this EBB" } else { "" }
                        ),
                    );
                }
            }
            for i in 0..nb_o {
                let t = op.arg_temp(i);
                let d = op.opc != Opcode::Discard;
                // A discard leaves the temp undefined for both kinds.
                ebb_def[t.index()] = d;
                tb_def[t.index()] = d;
            }

            match op.opc {
                Opcode::SetLabel | Opcode::Br | Opcode::Brcond => {
                    let k = if op.opc == Opcode::Brcond { 3 } else { 0 };
                    let l = op.args[k];
                    if l as usize >= nb_labels {
                        self.err(Some(id), format!("{}: label {l} does not exist", def.name));
                    } else if op.opc == Opcode::SetLabel {
                        if label_set[l as usize] {
                            let lab = Label::from_arg(l);
                            self.err(Some(id), format!("label $L{} is set twice", lab.id()));
                        }
                        label_set[l as usize] = true;
                        // A label starts a new extended basic block.
                        ebb_def.iter_mut().for_each(|d| *d = false);
                    } else if label_used[l as usize].is_none() {
                        label_used[l as usize] = Some(id);
                    }
                }
                _ => {}
            }
            prev = Some(op.opc);
        }

        for (l, used) in label_used.iter().enumerate() {
            if let Some(op) = used {
                if !label_set[l] {
                    self.err(Some(*op), format!("branch to label $L{l} which is never set"));
                }
            }
        }
    }
}

impl Func {
    /// Check the op list, returning every problem found.
    pub fn verify(&self) -> Result<(), Vec<VerifyError>> {
        let mut c = Checker { f: self, errors: Vec::new() };
        c.run();
        if c.errors.is_empty() { Ok(()) } else { Err(c.errors) }
    }

    /// The type of temp `t`, a convenience for callers that report verifier errors.
    pub fn temp_type(&self, t: Temp) -> Type {
        self.temp(t).ty
    }
}
