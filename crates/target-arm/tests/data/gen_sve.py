# SPDX-License-Identifier: GPL-2.0-or-later
#
# Regenerate a64_sve.txt for tests/a64_sve.rs from QEMU itself.
#
#     python3 gen_sve.py > a64_sve.txt
#
# This needs llvm-mc and llvm-objcopy (Homebrew's llvm) and qemu-system-aarch64 on the PATH or
# in /opt/homebrew. For each vector length it assembles a bare metal program that, for every
# case, loads the initial state described in a64_sve.rs (and init_state below), runs the one
# instruction word, and prints every register and the data buffer in hex on the PL011 UART.
# It runs that under
#
#     qemu-system-aarch64 -M virt -cpu max,sve-max-vq=N -accel tcg
#
# at EL1 with CPACR_EL1.FPEN and ZEN set and ZCR_EL1.LEN = N - 1, then writes out what each
# instruction changed. A synchronous exception records ESR_EL1 and skips the instruction.

import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

LLVM = "/opt/homebrew/opt/llvm/bin"
QEMU = "/opt/homebrew/bin/qemu-system-aarch64"
MARCH = "+sve2,+sve2-aes,+sve2-sha3,+sve2-sm4,+sve2-bitperm,+f32mm,+f64mm"
LOAD = 0x4020_0000
MID = 0x4030_0000
RAM_END = 0x4100_0000
BUF = 1024
M64 = (1 << 64) - 1
VQS = [1, 2, 3]
# The number of Z registers in the state: Z8 to Z13 hold floating point values.
NZ = 14

# Floating point values for Z8 to Z13: ones, twos, halves, pi, zeros of both signs,
# infinities, quiet and signaling NaNs, denormals, the largest finite values, a third and
# values out of the integer ranges.
FP_H = [0x3C00, 0xC000, 0x3800, 0x4248, 0x0000, 0x8000, 0x7C00, 0x7E00, 0x0001, 0x7BFF,
        0xBE00, 0x5640, 0x7D00, 0xFC00, 0x3555, 0xD140]
FP_S = [0x3F80_0000, 0xC000_0000, 0x3F00_0000, 0x4049_0FDB, 0x0000_0000, 0x8000_0000,
        0x7F80_0000, 0x7FC0_0000, 0x0000_0001, 0x7F7F_FFFF, 0xBFC0_0000, 0x42C8_0000,
        0x7FA0_0000, 0xCF00_0001, 0x3EAA_AAAB, 0x4F80_0000]
FP_D = [0x3FF0_0000_0000_0000, 0xC000_0000_0000_0000, 0x3FE0_0000_0000_0000,
        0x4009_21FB_5444_2D18, 0, 0x8000_0000_0000_0000, 0x7FF0_0000_0000_0000,
        0x7FF8_0000_0000_0000, 1, 0x7FEF_FFFF_FFFF_FFFF, 0xBFF8_0000_0000_0000,
        0x4059_0000_0000_0000, 0x7FF4_0000_0000_0000, 0xC3E0_0000_0000_0001,
        0x3FD5_5555_5555_5555, 0x43F0_0000_0000_0000]


def fp_reg(vl, table, size, mul, add):
    """Lane i of the register is table[(i * mul + add) % 16]."""
    n = vl // size
    return b"".join(table[(i * mul + add) % 16].to_bytes(size, "little") for i in range(n))


class Rng:
    def __init__(self):
        self.x = 0x9E37_79B9_7F4A_7C15

    def next(self):
        x = self.x
        x ^= (x << 13) & M64
        x ^= x >> 7
        x ^= (x << 17) & M64
        self.x = x
        return x


def init_state(vq):
    """The state each case starts from, as byte strings and integers."""
    vl = 16 * vq
    r = Rng()
    z = []
    for n in range(8):
        if n == 1:
            w = [(r.next() % 7) * 8 for _ in range(vl // 8)]
        elif n == 5:
            lanes = [((r.next() % 11) - 4) * 8 & 0xFFFF_FFFF for _ in range(vl // 4)]
            w = [lanes[2 * i] | lanes[2 * i + 1] << 32 for i in range(vl // 8)]
        elif n == 6:
            w = [MID - 64 + 24 * i for i in range(vl // 8)]
        elif n == 7:
            lanes = [(MID - 64 + 12 * i) & 0xFFFF_FFFF for i in range(vl // 4)]
            w = [lanes[2 * i] | lanes[2 * i + 1] << 32 for i in range(vl // 8)]
        else:
            w = [r.next() for _ in range(vl // 8)]
            if n == 0:
                w[0] = 8
        z.append(b"".join(v.to_bytes(8, "little") for v in w))
    for mul, add in [(3, 0), (5, 1)]:
        z += [fp_reg(vl, FP_H, 2, mul, add), fp_reg(vl, FP_S, 4, mul, add),
              fp_reg(vl, FP_D, 8, mul, add)]
    pmask = (1 << (vl // 8 * 8)) - 1
    p = []
    for n in range(8):
        v = pmask if n == 1 else r.next() & pmask
        p.append(v.to_bytes(vl // 8, "little"))
    x = [r.next() for _ in range(8)]
    x += [MID, 3, (-16) & M64, 5, 17, 0xFFFF_FFFE, RAM_END - 16, 0x8000_0000_0000_0003]
    nzcv = 0xA
    mem = bytes((j * 37 + 11) & 0xFF for j in range(BUF))
    return z, p, x, nzcv, mem


def cases():
    """The assembly of every case. Registers: Z2 to Z4 random, Z0 random but for a first
    64-bit lane of 8, Z1 small unsigned byte offsets, Z5 small signed 32-bit offsets, Z6 and
    Z7 64 and 32-bit addresses into the buffer, P1 all true, X8 the middle of the buffer, X14
    16 bytes before the end of RAM and X9 to X15 small values."""
    T = ["b", "h", "s", "d"]
    out = []

    def each(tmpl, sizes=T):
        for t in sizes:
            out.append(tmpl.format(t=t))

    for op in ["add", "sub", "subr", "and", "orr", "eor", "bic", "smax", "umax", "smin",
               "umin", "sabd", "uabd", "mul", "smulh", "umulh", "asr", "lsr", "lsl", "asrr",
               "lsrr", "lslr", "srshl", "urshl", "sqshl", "uqshl", "sqrshl", "uqrshl",
               "srshlr", "uqshlr", "shadd", "uhadd", "shsub", "uhsub", "shsubr", "srhadd",
               "urhadd", "sqadd", "uqadd", "sqsub", "uqsub", "sqsubr", "uqsubr", "suqadd",
               "usqadd"]:
        each(op + " z2.{t}, p2/m, z2.{t}, z3.{t}")
    for op in ["sdiv", "udiv", "sdivr", "udivr"]:
        each(op + " z2.{t}, p2/m, z2.{t}, z3.{t}", ["s", "d"])
    for op in ["asr", "lsr", "lsl"]:
        each(op + " z2.{t}, p2/m, z2.{t}, z3.d", ["b", "h", "s"])
        each(op + " z2.{t}, z3.{t}, z4.d", ["b", "h", "s"])
    for t, h in [("h", "b"), ("s", "h"), ("d", "s")]:
        out.append(f"sadalp z2.{t}, p2/m, z3.{h}")
        out.append(f"uadalp z2.{t}, p2/m, z3.{h}")
    for op in ["mla", "mls", "mad", "msb"]:
        each(op + " z2.{t}, p2/m, z3.{t}, z4.{t}")
    for op in ["add", "sub", "sqadd", "uqadd", "sqsub", "uqsub", "mul", "smulh", "umulh",
               "sqdmulh", "sqrdmulh", "saba", "uaba", "bext", "bdep", "bgrp", "sqrdmlah",
               "sqrdmlsh"]:
        each(op + " z2.{t}, z3.{t}, z4.{t}")
    out.append("pmul z2.b, z3.b, z4.b")
    for op in ["and", "orr", "eor", "bic"]:
        out.append(op + " z2.d, z3.d, z4.d")
    for op in ["eor3", "bcax", "bsl", "bsl1n", "bsl2n", "nbsl"]:
        out.append(op + " z2.d, z2.d, z3.d, z4.d")
    for t, n in [("b", 3), ("h", 11), ("s", 1), ("d", 37)]:
        out.append(f"xar z2.{t}, z2.{t}, z3.{t}, #{n}")

    for op in ["cls", "clz", "cnt", "cnot", "not", "abs", "neg", "rbit", "sqabs", "sqneg"]:
        each(op + " z2.{t}, p2/m, z3.{t}")
    for op in ["fabs", "fneg", "sxtb", "uxtb", "revb"]:
        each(op + " z2.{t}, p2/m, z3.{t}", ["h", "s", "d"])
    for op in ["sxth", "uxth", "revh"]:
        each(op + " z2.{t}, p2/m, z3.{t}", ["s", "d"])
    for op in ["sxtw", "uxtw", "revw"]:
        out.append(op + " z2.d, p2/m, z3.d")
    out += ["urecpe z2.s, p2/m, z3.s", "ursqrte z2.s, p2/m, z3.s"]
    for op in ["orv", "eorv", "andv", "smaxv", "umaxv", "sminv", "uminv"]:
        each(op + " {t}2, p2, z3.{t}")
    each("uaddv d2, p2, z3.{t}")
    each("saddv d2, p2, z3.{t}", ["b", "h", "s"])
    out += ["movprfx z2, z3"]
    each("movprfx z2.{t}, p2/z, z3.{t}")
    each("movprfx z2.{t}, p2/m, z3.{t}")

    for t, a, b in [("b", 1, 7), ("h", 3, 15), ("s", 9, 31), ("d", 33, 63)]:
        for op in ["asr", "lsr", "asrd", "srshr", "urshr"]:
            out.append(f"{op} z2.{t}, p2/m, z2.{t}, #{a}")
        for op in ["lsl", "sqshl", "uqshl", "sqshlu"]:
            out.append(f"{op} z2.{t}, p2/m, z2.{t}, #{b}")
        for op in ["asr", "lsr", "ssra", "usra", "srsra", "ursra", "sri"]:
            out.append(f"{op} z2.{t}, z3.{t}, #{a}")
        for op in ["lsl", "sli"]:
            out.append(f"{op} z2.{t}, z3.{t}, #{b}")
    for op in ["add", "sub", "subr", "sqadd", "uqadd", "sqsub", "uqsub"]:
        each(op + " z2.{t}, z2.{t}, #200")
        each(op + " z2.{t}, z2.{t}, #3, lsl #8", ["h", "s", "d"])
    for op, imm in [("smax", -100), ("smin", 50), ("umax", 200), ("umin", 7), ("mul", -3)]:
        each(op + " z2.{t}, z2.{t}, #" + str(imm))
    out += ["and z2.s, z2.s, #0xff00ff00", "orr z2.d, z2.d, #0x7ffe", "eor z2.h, z2.h, #0x3c",
            "dupm z2.s, #0xffff0001", "dupm z2.b, #0x81"]
    each("dup z2.{t}, #-5")
    each("dup z2.{t}, #0x7f, lsl #8", ["h", "s", "d"])
    each("fmov z2.{t}, #-1.5", ["h", "s", "d"])

    for op in ["cmpeq", "cmpne", "cmpge", "cmpgt", "cmphs", "cmphi"]:
        each(op + " p3.{t}, p2/z, z3.{t}, z4.{t}")
    for op in ["cmpeq", "cmpne", "cmpge", "cmpgt", "cmphs", "cmphi", "cmplt", "cmple", "cmplo",
               "cmpls"]:
        each(op + " p3.{t}, p2/z, z3.{t}, z4.d", ["b", "h", "s"])
    for op in ["cmpeq", "cmpne", "cmpge", "cmpgt", "cmplt", "cmple"]:
        each(op + " p3.{t}, p2/z, z3.{t}, #-3")
        each(op + " p3.{t}, p1/z, z1.{t}, #8")
    for op in ["cmphs", "cmphi", "cmplo", "cmpls"]:
        each(op + " p3.{t}, p2/z, z3.{t}, #100")
        each(op + " p3.{t}, p1/z, z1.{t}, #16")
    for op in ["match", "nmatch"]:
        each(op + " p3.{t}, p2/z, z3.{t}, z4.{t}", ["b", "h"])
        each(op + " p3.{t}, p1/z, z1.{t}, z5.{t}", ["b", "h"])
    each("histcnt z2.{t}, p2/z, z1.{t}, z5.{t}", ["s", "d"])
    each("histcnt z2.{t}, p2/z, z3.{t}, z4.{t}", ["s", "d"])
    out += ["histseg z2.b, z1.b, z5.b", "histseg z2.b, z3.b, z4.b"]

    for op in ["and", "ands", "bic", "bics", "eor", "eors", "orr", "orrs", "orn", "orns", "nor",
               "nors", "nand", "nands"]:
        out.append(op + " p3.b, p2/z, p4.b, p5.b")
    out += ["sel p3.b, p2, p4.b, p5.b", "brkpa p3.b, p2/z, p4.b, p5.b",
            "brkpb p3.b, p2/z, p4.b, p5.b", "brkpas p3.b, p2/z, p4.b, p5.b",
            "brkpbs p3.b, p2/z, p4.b, p5.b", "brka p3.b, p2/z, p4.b", "brkb p3.b, p2/z, p4.b",
            "brka p3.b, p2/m, p4.b", "brkb p3.b, p2/m, p4.b", "brkas p3.b, p2/z, p4.b",
            "brkbs p3.b, p2/z, p4.b", "brkn p3.b, p2/z, p4.b, p3.b",
            "brkns p3.b, p2/z, p4.b, p3.b", "ptest p2, p3.b", "ptest p1, p0.b", "pfalse p3.b",
            "rdffr p3.b", "rdffr p3.b, p2/z", "rdffrs p3.b, p2/z", "wrffr p3.b", "setffr",
            "pfirst p3.b, p2, p3.b", "pfirst p3.b, p4, p3.b"]
    each("pnext p3.{t}, p2, p3.{t}")
    each("pnext p3.{t}, p1, p3.{t}")
    for pat in ["pow2", "vl1", "vl3", "vl7", "vl16", "vl32", "mul4", "mul3", "all", "#14"]:
        each("ptrue p3.{t}, " + pat)
    each("ptrues p3.{t}, vl5")
    each("cntp x0, p2, p3.{t}")

    for c in ["b", "h", "w", "d"]:
        out += [f"cnt{c} x0, pow2", f"cnt{c} x0, vl3, mul #3", f"cnt{c} x0",
                f"inc{c} x11, vl7", f"dec{c} x11, all, mul #2",
                f"sqinc{c} x13, w13, all, mul #4", f"uqinc{c} w14, all, mul #16",
                f"sqdec{c} x15", f"uqdec{c} x11, all, mul #2", f"sqdec{c} x11, w11, vl64",
                f"uqdec{c} w12", f"sqinc{c} x14"]
    for c, t in [("h", "h"), ("w", "s"), ("d", "d")]:
        out += [f"inc{c} z2.{t}, vl4", f"dec{c} z2.{t}, all, mul #3",
                f"sqinc{c} z2.{t}, all, mul #16", f"uqdec{c} z2.{t}", f"uqinc{c} z2.{t}, vl256"]
    each("incp x11, p3.{t}")
    each("decp x11, p3.{t}")
    each("sqincp x13, p3.{t}, w13")
    each("sqdecp x15, p3.{t}")
    each("uqincp w14, p3.{t}")
    each("uqdecp x11, p3.{t}")
    each("incp z2.{t}, p3.{t}", ["h", "s", "d"])
    each("sqdecp z2.{t}, p3.{t}", ["h", "s", "d"])
    each("uqincp z2.{t}, p3.{t}", ["h", "s", "d"])
    out += ["addvl x0, x8, #-3", "addpl x0, x8, #7", "rdvl x0, #-5"]

    each("index z2.{t}, #-3, #5")
    each("index z2.{t}, #2, w11")
    each("index z2.{t}, w12, #-1")
    each("index z2.{t}, w11, w15")
    out.append("index z2.d, x15, x13")
    each("dup z2.{t}, w15")
    out.append("dup z2.d, x15")
    for t, i in [("b", 5), ("b", 17), ("b", 63), ("h", 6), ("h", 15), ("s", 3), ("s", 9),
                 ("d", 1), ("d", 5), ("q", 0), ("q", 1), ("q", 2), ("q", 3)]:
        out.append(f"mov z2.{t}, z3.{t}[{i}]")
    each("insr z2.{t}, w15")
    out.append("insr z2.d, x15")
    each("insr z2.{t}, {t}3")
    each("rev z2.{t}, z3.{t}")
    for op in ["zip1", "zip2", "uzp1", "uzp2", "trn1", "trn2"]:
        each(op + " z2.{t}, z3.{t}, z4.{t}")
        each(op + " p3.{t}, p4.{t}, p5.{t}")
    each("tbl z2.{t}, {{z3.{t}}}, z1.{t}")
    each("tbl z2.{t}, {{z3.{t}, z4.{t}}}, z1.{t}")
    each("tbx z2.{t}, z3.{t}, z1.{t}")
    each("tbl z2.{t}, {{z3.{t}}}, z4.{t}")
    each("rev p3.{t}, p4.{t}")
    out += ["punpklo p3.h, p4.b", "punpkhi p3.h, p4.b"]
    for op in ["sunpklo", "sunpkhi", "uunpklo", "uunpkhi"]:
        out += [f"{op} z2.h, z3.b", f"{op} z2.s, z3.h", f"{op} z2.d, z3.s"]
    out += ["ext z2.b, z2.b, z3.b, #5", "ext z2.b, z2.b, z3.b, #40",
            "ext z2.b, {z3.b, z4.b}, #17", "ext z2.b, {z3.b, z4.b}, #200"]
    each("splice z2.{t}, p2, z2.{t}, z3.{t}")
    each("splice z2.{t}, p2, {{z3.{t}, z4.{t}}}")
    each("compact z2.{t}, p2, z3.{t}", ["s", "d"])
    each("sel z2.{t}, p2, z3.{t}, z4.{t}")
    for op in ["clasta", "clastb"]:
        each(op + " z2.{t}, p2, z2.{t}, z3.{t}")
        each(op + " {t}2, p2, {t}2, z3.{t}")
        each(op + " w11, p2, w11, z3.{t}", ["b", "h", "s"])
        out.append(op + " x15, p2, x15, z3.d")
        out.append(op + " w15, p0, w15, z3.h")
    for op in ["lasta", "lastb"]:
        each(op + " {t}2, p2, z3.{t}")
        each(op + " w11, p2, z3.{t}", ["b", "h", "s"])
        out.append(op + " x11, p2, z3.d")
    each("mov z2.{t}, p2/m, w15")
    out.append("mov z2.d, p2/m, x15")
    each("mov z2.{t}, p2/m, {t}3")
    each("mov z2.{t}, p2/m, #-7")
    each("mov z2.{t}, p2/z, #5")
    each("mov z2.{t}, p2/z, #-2, lsl #8", ["h", "s", "d"])
    each("fmov z2.{t}, p2/m, #-2.0", ["h", "s", "d"])
    out += ["adr z2.d, [z6.d, z1.d, sxtw #2]", "adr z2.d, [z6.d, z5.d, uxtw #1]",
            "adr z2.s, [z7.s, z5.s, lsl #3]", "adr z2.d, [z6.d, z3.d]"]
    for op in ["whilelt", "whilele", "whilelo", "whilels", "whilegt", "whilege", "whilehi",
               "whilehs"]:
        each(op + " p3.{t}, x11, x12")
        each(op + " p3.{t}, w12, w11")
        each(op + " p3.{t}, x15, x11")
        each(op + " p3.{t}, w13, w14")
    for op in ["whilewr", "whilerw"]:
        each(op + " p3.{t}, x11, x12")
        each(op + " p3.{t}, x12, x11")
    out += ["ctermeq x11, x12", "ctermne x11, x12", "ctermeq w11, w11", "ctermne w13, w14"]

    # Loads and stores. Contiguous ones with X8 and X9 = 3, gathers and scatters with the
    # offsets in Z1 and Z5 and the addresses in Z6 and Z7.
    for m, sz, ts in [("b", 0, "bhsd"), ("h", 1, "hsd"), ("w", 2, "sd"), ("d", 3, "d")]:
        sh = f", lsl #{sz}" if sz else ""
        for t in ts:
            out.append(f"ld1{m} {{z2.{t}}}, p2/z, [x8, x9{sh}]")
            out.append(f"ld1{m} {{z2.{t}}}, p2/z, [x8, #-1, mul vl]")
            # The first fault and non fault loads are governed by P7, whose element 0 is
            # active at every length: QEMU 11.1's sve_ldnfff1_r() reads the wrong predicate
            # bits when the first active element is not at a multiple of 64 bytes.
            out.append(f"ldff1{m} {{z2.{t}}}, p7/z, [x8, x9{sh}]")
            out.append(f"ldnf1{m} {{z2.{t}}}, p7/z, [x8, #1, mul vl]")
            out.append(f"st1{m} {{z2.{t}}}, p2, [x8, x9{sh}]")
            out.append(f"st1{m} {{z2.{t}}}, p2, [x8, #-2, mul vl]")
            out.append(f"ld1r{m} {{z2.{t}}}, p2/z, [x8, #{(5 << sz)}]")
        for t in "bhsd"[sz + 1:]:
            if m != "d":
                out.append(f"ld1s{m} {{z2.{t}}}, p2/z, [x8, x9{sh}]")
                out.append(f"ld1rs{m} {{z2.{t}}}, p2/z, [x8, #{(63 << sz)}]")
                out.append(f"ldnf1s{m} {{z2.{t}}}, p7/z, [x8, #-1, mul vl]")
        t = "bhsd"[sz]
        out.append(f"ldff1{m} {{z2.{t}}}, p1/z, [x8]")
        out.append(f"ld1rq{m} {{z2.{t}}}, p2/z, [x8, #-32]")
        out.append(f"ld1rq{m} {{z2.{t}}}, p2/z, [x8, x9{sh}]")
        out.append(f"ldnt1{m} {{z2.{t}}}, p2/z, [x8, x9{sh}]")
        out.append(f"stnt1{m} {{z2.{t}}}, p2, [x8, #1, mul vl]")
        for n in [2, 3, 4]:
            regs = ", ".join(f"z{2 + k}.{t}" for k in range(n))
            out.append(f"ld{n}{m} {{{regs}}}, p2/z, [x8, x9{sh}]")
            out.append(f"ld{n}{m} {{{regs}}}, p2/z, [x8, #-{n}, mul vl]")
            out.append(f"st{n}{m} {{{regs}}}, p2, [x8, x9{sh}]")
            out.append(f"st{n}{m} {{{regs}}}, p1, [x8, #{n}, mul vl]")
    out += ["ldr z2, [x8, #-3, mul vl]", "ldr p3, [x8, #5, mul vl]",
            "str z2, [x8, #2, mul vl]", "str p3, [x8, #-7, mul vl]", "ldr z2, [x8]"]
    out += [
        "ld1b {z2.s}, p2/z, [x8, z5.s, sxtw]", "ld1sh {z2.s}, p2/z, [x8, z1.s, uxtw #1]",
        "ld1w {z2.s}, p2/z, [x8, z5.s, sxtw #2]", "ld1sb {z2.s}, p2/z, [x8, z1.s, uxtw]",
        "ld1h {z2.s}, p2/z, [x8, z5.s, sxtw]", "ld1d {z2.d}, p2/z, [x8, z1.d, lsl #3]",
        "ld1sw {z2.d}, p2/z, [x8, z1.d]", "ld1h {z2.d}, p2/z, [x8, z5.d, sxtw #1]",
        "ld1b {z2.d}, p2/z, [x8, z1.d, uxtw]", "ld1sw {z2.d}, p2/z, [x8, z1.d, uxtw #2]",
        "ld1w {z2.s}, p2/z, [z7.s, #8]", "ld1d {z2.d}, p2/z, [z6.d, #16]",
        "ld1sb {z2.d}, p2/z, [z6.d, #3]", "ld1h {z2.s}, p2/z, [z7.s, #62]",
        "ldff1d {z2.d}, p2/z, [x8, z1.d, lsl #3]", "ldff1w {z2.s}, p2/z, [z7.s]",
        "ldff1sh {z2.d}, p2/z, [x8, z5.d, sxtw #1]", "ldff1b {z2.s}, p1/z, [z7.s, #31]",
        "ldnt1w {z2.s}, p2/z, [z7.s, x10]", "ldnt1d {z2.d}, p2/z, [z6.d, x10]",
        "ldnt1sb {z2.d}, p2/z, [z6.d]", "ldnt1sh {z2.s}, p2/z, [z7.s, x8]",
        "st1w {z2.s}, p2, [x8, z5.s, sxtw #2]", "st1d {z2.d}, p2, [x8, z1.d, lsl #3]",
        "st1b {z2.d}, p2, [z6.d, #5]", "st1h {z2.s}, p2, [z7.s, #2]",
        "st1h {z2.d}, p1, [x8, z1.d, uxtw #1]", "st1b {z2.s}, p2, [x8, z1.s, uxtw]",
        "stnt1w {z2.s}, p2, [z7.s, x10]", "stnt1d {z2.d}, p2, [z6.d]",
        "stnt1b {z2.s}, p1, [z7.s, x9]",
        # Loads that run off the end of RAM, where nothing is mapped.
        "ldff1b {z2.b}, p1/z, [x14, x9]", "ldff1h {z2.h}, p7/z, [x14, x9, lsl #1]",
        "ldff1d {z2.d}, p1/z, [x14, x9, lsl #3]", "ldnf1d {z2.d}, p1/z, [x14]",
        "ldnf1w {z2.s}, p1/z, [x14, #1, mul vl]", "ldnf1b {z2.b}, p7/z, [x14]",
        "ldff1d {z2.d}, p1/z, [x8, z0.d]", "ldff1w {z2.s}, p1/z, [x14, z1.s, uxtw]",
        "ldff1b {z2.d}, p7/z, [x14, z1.d]", "ld1d {z2.d}, p1/z, [x14, x9, lsl #3]",
        "ld1b {z2.b}, p1/z, [x14]", "st1b {z2.b}, p1, [x14, x9]", "st1d {z2.d}, p1, [x8, z0.d]",
        "ld1d {z2.d}, p1/z, [x8, z0.d]", "ldr z2, [x14]", "str p3, [x14, #2, mul vl]",
        "prfb pldl1keep, p2, [x8, x9]", "prfw pldl2strm, p2, [x8, #1, mul vl]",
        "prfd pldl1keep, p2, [z6.d, #8]", "prfh pstl1keep, p2, [x8, z5.s, sxtw #1]",
    ]
    out += fp_cases()
    return out


def fp_cases():
    """The floating point and crypto cases. Z8 to Z10 (A) and Z11 to Z13 (B) hold half,
    single and double values from the FP_ tables."""
    out = []
    A = {"h": 8, "s": 9, "d": 10}
    B = {"h": 11, "s": 12, "d": 13}
    F = ["h", "s", "d"]

    def each(tmpl, sizes=F):
        for t in sizes:
            out.append(tmpl.format(t=t, a=A[t], b=B[t]))

    for op in ["fadd", "fsub", "fmul", "fmaxnm", "fminnm", "fmax", "fmin", "fabd", "fscale",
               "fmulx", "fdiv", "fdivr", "fsubr", "faddp", "fmaxnmp", "fminnmp", "fmaxp",
               "fminp"]:
        each(op + " z{a}.{t}, p2/m, z{a}.{t}, z{b}.{t}")
    for op in ["fadd", "fmul", "fdiv", "fmaxp"]:
        each(op + " z2.{t}, p2/m, z2.{t}, z3.{t}")
    for op in ["fadd", "fsub", "fmul", "ftsmul", "frecps", "frsqrts", "ftssel"]:
        each(op + " z2.{t}, z{a}.{t}, z{b}.{t}")
    each("ftssel z2.{t}, z{a}.{t}, z3.{t}")
    for op in ["fcmge", "fcmgt", "fcmeq", "fcmne", "fcmuo", "facge", "facgt", "fcmle"]:
        each(op + " p3.{t}, p2/z, z{a}.{t}, z{b}.{t}")
    for op in ["fcmge", "fcmgt", "fcmle", "fcmlt", "fcmeq", "fcmne"]:
        each(op + " p3.{t}, p2/z, z{a}.{t}, #0.0")
    for op in ["faddv", "fmaxnmv", "fminnmv", "fmaxv", "fminv"]:
        each(op + " {t}2, p2, z{a}.{t}")
        each(op + " {t}2, p1, z{b}.{t}")
    each("fadda {t}{a}, p2, {t}{a}, z{b}.{t}")
    each("fadda {t}2, p1, {t}2, z3.{t}")
    for op in ["frintn", "frintp", "frintm", "frintz", "frinta", "frintx", "frinti", "frecpx",
               "fsqrt", "flogb"]:
        each(op + " z2.{t}, p2/m, z{a}.{t}")
        each(op + " z2.{t}, p2/m, z{b}.{t}")
    each("frintx z2.{t}, p2/m, z3.{t}")
    for op in ["frecpe", "frsqrte"]:
        each(op + " z2.{t}, z{a}.{t}")
        each(op + " z2.{t}, z3.{t}")
    each("fexpa z2.{t}, z3.{t}")
    for op, imms in [("fadd", ["0.5", "1.0"]), ("fsub", ["0.5", "1.0"]),
                     ("fsubr", ["0.5", "1.0"]), ("fmul", ["0.5", "2.0"]),
                     ("fmaxnm", ["0.0", "1.0"]), ("fminnm", ["0.0", "1.0"]),
                     ("fmax", ["0.0", "1.0"]), ("fmin", ["0.0", "1.0"])]:
        for imm in imms:
            each(op + " z{a}.{t}, p2/m, z{a}.{t}, #" + imm)
    for op in ["fmla", "fmls", "fnmla", "fnmls", "fmad", "fmsb", "fnmad", "fnmsb"]:
        each(op + " z{a}.{t}, p2/m, z{b}.{t}, z{a}.{t}")
        each(op + " z2.{t}, p2/m, z3.{t}, z4.{t}")
    for rot in [90, 270]:
        each("fcadd z{a}.{t}, p2/m, z{a}.{t}, z{b}.{t}, #" + str(rot))
    for rot in [0, 90, 180, 270]:
        each("fcmla z{a}.{t}, p2/m, z{b}.{t}, z{a}.{t}, #" + str(rot))
    for rot, i in [(0, 0), (90, 3), (180, 1), (270, 2)]:
        out.append(f"fcmla z8.h, z11.h, z3.h[{i}], #{rot}")
        out.append(f"fcmla z9.s, z12.s, z9.s[{i % 2}], #{rot}")
    for op in ["fmla", "fmls"]:
        out += [f"{op} z8.h, z11.h, z3.h[5]", f"{op} z9.s, z12.s, z4.s[2]",
                f"{op} z10.d, z13.d, z13.d[1]", f"{op} z2.s, z3.s, z4.s[0]"]
    out += ["fmul z2.h, z8.h, z3.h[7]", "fmul z2.s, z9.s, z4.s[3]",
            "fmul z2.d, z10.d, z13.d[0]", "fmul z2.d, z3.d, z4.d[1]"]
    for imm in [0, 3, 5, 7]:
        each("ftmad z{a}.{t}, z{a}.{t}, z{b}.{t}, #" + str(imm))
    for d, n in [("h", "s"), ("s", "h"), ("h", "d"), ("d", "h"), ("s", "d"), ("d", "s")]:
        for src in [A[n], 3]:
            out.append(f"fcvt z2.{d}, p2/m, z{src}.{n}")
    for op in ["fcvtzs", "fcvtzu"]:
        for d, n in [("h", "h"), ("s", "h"), ("d", "h"), ("s", "s"), ("s", "d"), ("d", "s"),
                     ("d", "d")]:
            out.append(f"{op} z2.{d}, p2/m, z{A[n]}.{n}")
            out.append(f"{op} z2.{d}, p2/m, z{B[n]}.{n}")
    for op in ["scvtf", "ucvtf"]:
        for d, n in [("h", "h"), ("h", "s"), ("h", "d"), ("s", "s"), ("d", "s"), ("s", "d"),
                     ("d", "d")]:
            out.append(f"{op} z2.{d}, p2/m, z5.{n}")
            out.append(f"{op} z2.{d}, p2/m, z3.{n}")
    out += ["fcvtnt z2.h, p2/m, z9.s", "fcvtnt z2.s, p2/m, z10.d", "fcvtlt z2.s, p2/m, z8.h",
            "fcvtlt z2.d, p2/m, z9.s", "fcvtx z2.s, p2/m, z10.d", "fcvtxnt z2.s, p2/m, z10.d",
            "fcvtnt z2.h, p2/m, z3.s", "fcvtx z2.s, p2/m, z3.d", "fcvtlt z2.s, p2/m, z3.h"]
    out += ["fmmla z9.s, z12.s, z9.s", "fmmla z2.s, z3.s, z4.s", "fmmla z10.d, z13.d, z10.d",
            "fmmla z2.d, z3.d, z4.d"]
    out += ["aese z2.b, z2.b, z3.b", "aesd z2.b, z2.b, z3.b", "aesmc z2.b, z2.b",
            "aesimc z2.b, z2.b", "sm4e z2.s, z2.s, z3.s", "sm4ekey z2.s, z3.s, z4.s",
            "rax1 z2.d, z3.d, z4.d"]
    for d, n in [("q", "d"), ("h", "b"), ("d", "s")]:
        out += [f"pmullb z2.{d}, z3.{n}, z4.{n}", f"pmullt z2.{d}, z3.{n}, z4.{n}"]
    return out


def assemble(lines):
    """Assemble each line on its own (a MOVPRFX is diagnosed against the next line), and
    return (word, asm) for those that assemble."""

    def one(asm):
        r = subprocess.run([f"{LLVM}/llvm-mc", "-triple=aarch64", f"-mattr={MARCH}",
                            "-show-encoding"], input=asm + "\n", capture_output=True, text=True)
        m = re.search(r"encoding: \[(0x..),(0x..),(0x..),(0x..)\]", r.stdout)
        if r.returncode != 0 or not m:
            print(f"dropped: {asm}", file=sys.stderr)
            return None
        a, b, c, d = (int(v, 16) for v in m.groups())
        return (d << 24 | c << 16 | b << 8 | a, asm)

    with ThreadPoolExecutor(8) as ex:
        return [r for r in ex.map(one, lines) if r]


def ppad(pl):
    """The bytes the nine predicate registers take in the dump, kept 16-byte aligned as the
    MMU is off and Device memory needs aligned accesses."""
    return (9 * pl + 15) & ~15


def movx(reg, v):
    s = [f"movz {reg}, #{v & 0xffff}"]
    for k in range(1, 4):
        h = (v >> (16 * k)) & 0xFFFF
        if h:
            s.append(f"movk {reg}, #{h}, lsl #{16 * k}")
    return s


def program(vq, words):
    vl = 16 * vq
    pl = vl // 8
    z, p, x, nzcv, mem = init_state(vq)
    a = [".text", "_start:"]
    a += movx("x0", (3 << 20) | (3 << 16))
    a += ["msr cpacr_el1, x0", "isb", f"mov x0, #{vq - 1}", "msr s3_0_c1_c2_0, x0", "isb",
          "adr x0, vectors", "msr vbar_el1, x0", "adr x0, stack_top", "mov sp, x0"]
    a += movx("x27", 0x0900_0000)
    a += ["mov w0, #0x301", "str w0, [x27, #0x30]", "isb"]
    for w in words:
        a += ["bl init", f".inst 0x{w:08x}", "bl dump"]
    a += ["mov x0, #0x18", "adr x1, exit_block", "hlt #0xf000", "b ."]
    a += ["init:",
          "adr x29, pristine", "adr x28, buf", f"mov x25, #{BUF // 16}",
          "1: ldp x0, x1, [x29], #16", "stp x0, x1, [x28], #16", "subs x25, x25, #1",
          "b.ne 1b", "adr x29, zinit"]
    a += [f"ldr z{n}, [x29, #{n}, mul vl]" for n in range(NZ)]
    a += ["adr x29, pinit"] + [f"ldr p{n}, [x29, #{n}, mul vl]" for n in range(8)]
    a += ["setffr", "msr fpsr, xzr", "adr x29, esr_slot", "str xzr, [x29]", "adr x29, xinit",
          "ldr x28, [x29, #128]", "msr nzcv, x28"]
    a += [f"ldp x{2 * k}, x{2 * k + 1}, [x29, #{16 * k}]" for k in range(8)]
    a += ["ret"]
    a += ["dump:", "mov x26, x30", "adr x29, out"]
    a += [f"str z{n}, [x29, #{n}, mul vl]" for n in range(NZ)]
    a += [f"add x29, x29, #{NZ * vl}"]
    a += [f"str p{n}, [x29, #{n}, mul vl]" for n in range(8)]
    a += ["rdffr p0.b", "str p0, [x29, #8, mul vl]", f"add x29, x29, #{ppad(pl)}"]
    a += [f"stp x{2 * k}, x{2 * k + 1}, [x29], #16" for k in range(8)]
    a += ["mrs x28, nzcv", "str x28, [x29], #8", "adr x28, esr_slot", "ldr x28, [x28]",
          "str x28, [x29], #8", "mrs x28, fpsr", "str x28, [x29], #8",
          "mov w2, #82", "strb w2, [x27]", "mov w2, #32", "strb w2, [x27]",
          "adr x0, out", f"mov x1, #{NZ * vl + ppad(pl) + 128 + 24}", "bl hexdump",
          "adr x0, buf", f"mov x1, #{BUF}", "bl hexdump",
          "mov w2, #10", "strb w2, [x27]", "ret x26"]
    a += ["hexdump:",
          "1: ldrb w2, [x0], #1", "lsr w3, w2, #4", "and w2, w2, #15",
          "add w3, w3, #48", "cmp w3, #57", "b.ls 2f", "add w3, w3, #39", "2: strb w3, [x27]",
          "add w2, w2, #48", "cmp w2, #57", "b.ls 3f", "add w2, w2, #39", "3: strb w2, [x27]",
          "subs x1, x1, #1", "b.ne 1b", "ret"]
    a += [".balign 2048", "vectors:"]
    for k in range(16):
        a += [f".balign 128", "stp x0, x1, [sp, #-16]!", "adr x1, esr_slot",
              "mrs x0, esr_el1", "str x0, [x1]", "mrs x0, elr_el1", "add x0, x0, #4",
              "msr elr_el1, x0", "ldp x0, x1, [sp], #16", "eret"]
    a += [".balign 16", "exit_block: .quad 0x20026, 0", "esr_slot: .quad 0"]

    def data(label, bs):
        r = [".balign 16", f"{label}:"]
        r += [".byte " + ", ".join(str(c) for c in bs[i:i + 16]) for i in range(0, len(bs), 16)]
        return r

    a += data("zinit", b"".join(z))
    a += data("pinit", b"".join(p) + bytes(-len(b"".join(p)) % 16))
    xb = b"".join(v.to_bytes(8, "little") for v in x) + (nzcv << 28).to_bytes(8, "little")
    a += data("xinit", xb)
    a += data("pristine", mem)
    a += [".balign 16", "out:", f".skip {NZ * vl + ppad(pl) + 160}", ".balign 16",
          "stack:", ".skip 4096", "stack_top:"]
    # The buffer sits at a fixed address, MID - 512, well past the end of the program.
    text = "\n".join(a) + "\n"
    return text


def build_and_run(vq, words):
    with tempfile.TemporaryDirectory() as d:
        src = program(vq, words)
        # The buffer is not in the image: place it with an absolute symbol.
        src = src.replace("adr x28, buf", "\n".join(movx("x28", MID - BUF // 2)))
        src = src.replace("adr x0, buf", "\n".join(movx("x0", MID - BUF // 2)))
        open(os.path.join(d, "h.s"), "w").write(src)
        subprocess.run([f"{LLVM}/llvm-mc", "-triple=aarch64", f"-mattr={MARCH}", "-filetype=obj",
                        "-o", os.path.join(d, "h.o"), os.path.join(d, "h.s")], check=True)
        subprocess.run([f"{LLVM}/llvm-objcopy", "-O", "binary", "-j", ".text",
                        os.path.join(d, "h.o"), os.path.join(d, "h.bin")], check=True)
        r = subprocess.run([QEMU, "-M", "virt", "-cpu", f"max,sve-max-vq={vq}", "-accel", "tcg",
                            "-display", "none", "-monitor", "none", "-serial", "stdio",
                            "-m", "16M", "-semihosting", "-device",
                            f"loader,file={d}/h.bin,addr={LOAD:#x},cpu-num=0"],
                           capture_output=True, text=True, timeout=600)
    rows = [l[2:].strip() for l in r.stdout.splitlines() if l.startswith("R ")]
    if len(rows) != len(words):
        sys.exit(f"vq {vq}: {len(rows)} rows for {len(words)} cases\n{r.stderr}")
    return [bytes.fromhex(row) for row in rows]


def le_hex(bs):
    return bs[::-1].hex()


def main():
    cs = assemble(cases())
    words = [w for w, _ in cs]
    print("# SPDX-License-Identifier: GPL-2.0-or-later")
    print("#")
    print("# SVE and SVE2 cases for tests/a64_sve.rs, generated by gen_sve.py from QEMU 11.1")
    print("# (qemu-system-aarch64 -M virt -cpu max,sve-max-vq=N -accel tcg) at EL1.")
    print("#")
    print("# A line \"vq N\" starts the cases for a vector length of N quadwords. Each case is")
    print("# the instruction word, then what it changed from the initial state of a64_sve.rs:")
    print("# zN, pN and ffr as one hex number (element 0 lowest), xN, nzcv (the four flag")
    print("# bits), esr (ESR_EL1 if the instruction raised an exception) and mOFF, the 64-bit")
    print("# little endian word at byte offset OFF from X8, fpsr (FPSR if not zero). Then the")
    print("# assembly.")
    for vq in VQS:
        vl = 16 * vq
        pl = vl // 8
        z, p, x, nzcv, mem = init_state(vq)
        print(f"vq {vq}")
        for (w, asm), row in zip(cs, build_and_run(vq, words)):
            ch = []
            o = 0
            for n in range(NZ):
                if row[o:o + vl] != z[n]:
                    ch.append(f"z{n}={le_hex(row[o:o + vl])}")
                o += vl
            for n in range(9):
                v = row[o:o + pl]
                init = p[n] if n < 8 else bytes([0xFF] * pl)
                if v != init:
                    ch.append(f"{'ffr' if n == 8 else f'p{n}'}={le_hex(v)}")
                o += pl
            o += ppad(pl) - 9 * pl
            for n in range(16):
                v = int.from_bytes(row[o:o + 8], "little")
                if v != x[n]:
                    ch.append(f"x{n}={v:x}")
                o += 8
            v = int.from_bytes(row[o:o + 8], "little") >> 28
            if v != nzcv:
                ch.append(f"nzcv={v:x}")
            o += 8
            esr = int.from_bytes(row[o:o + 8], "little")
            if esr:
                ch.append(f"esr={esr:x}")
            o += 8
            fpsr = int.from_bytes(row[o:o + 8], "little")
            if fpsr:
                ch.append(f"fpsr={fpsr:x}")
            o += 8
            m = row[o:o + BUF]
            for j in range(0, BUF, 8):
                if m[j:j + 8] != mem[j:j + 8]:
                    ch.append(f"m{j - BUF // 2}={int.from_bytes(m[j:j + 8], 'little'):x}")
            print(f"{w:08x} => {' '.join(ch)} ; {asm}")


if __name__ == "__main__":
    main()
