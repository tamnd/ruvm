// SPDX-License-Identifier: GPL-2.0-or-later

//! The riscv guest: a port of QEMU's `target/riscv` TCG front end for RV64.
//!
//! It covers RV64GC (I, M, A, F, D, C with Zicsr and Zifencei), the machine, supervisor and
//! user privilege levels with Sv39, Sv48 and Sv57 paging and PMP, and the Zba, Zbb, Zbc,
//! Zbs, Zfa, Zicbom, Zicboz, Zawrs and Sstc extensions of QEMU's default `rv64` CPU. The
//! vector and hypervisor extensions are not there yet.
//!
//! - [`cpu`]: `CPURISCVState` as a struct the generated code addresses by offset, and the
//!   architectural constants.
//! - [`tcg`]: the translator, the helpers, the page walk and the `CpuOps` glue.

pub mod cpu;
pub mod tcg;

#[allow(missing_docs, unreachable_pub, clippy::pedantic, clippy::nursery)]
mod decode {
    pub mod insn32 {
        include!(concat!(env!("OUT_DIR"), "/insn32.rs"));

        /// Decode and translate one 32-bit instruction.
        pub(crate) fn decode32(ctx: &mut impl DecodeInsn32, insn: u32) -> bool {
            decode_insn32(ctx, insn)
        }
    }
    pub mod insn16 {
        use super::insn32::*;
        include!(concat!(env!("OUT_DIR"), "/insn16.rs"));

        /// Decode and translate one 16-bit instruction.
        pub(crate) fn decode16(ctx: &mut impl DecodeInsn16, insn: u16) -> bool {
            decode_insn16(ctx, insn)
        }
    }
    pub mod xlrbr {
        use super::insn32::*;
        include!(concat!(env!("OUT_DIR"), "/xlrbr.rs"));

        /// Decode and translate one XLRBR instruction.
        pub(crate) fn decode(ctx: &mut impl DecodeXlrbr, insn: u32) -> bool {
            decode_xlrbr(ctx, insn)
        }
    }
}
