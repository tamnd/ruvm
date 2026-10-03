#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Generate src/tcg/translate/sse_tab.rs, the vector opcode tables, from QEMU's
# target/i386/tcg/decode-new.c.inc.
#
# Usage: python3 gen_sse_tab.py [path/to/decode-new.c.inc] > ../../src/tcg/translate/sse_tab.rs
# then run rustfmt on the output. The path defaults to the QEMU 11.1 tree in ~/src.
#
# Only the tables of the MMX, SSE and AVX instructions are converted: the 0F rows that hold
# vector instructions, the whole 0F 38 00 to EF and 0F 3A maps and the small tables of the
# decoders that those rows use. The 0F 38 F0 to FF rows and group 15 stay hand written in ext.rs
# and insn.rs.

import os
import re
import sys

DEFAULT = os.path.expanduser("~/src/qemu-v11.1.0/target/i386/tcg/decode-new.c.inc")

# The rows of opcodes_0F that are vector instructions.
SSE_0F = (
    list(range(0x10, 0x18))
    + list(range(0x28, 0x30))
    + list(range(0x50, 0x80))
    + [0xC2, 0xC4, 0xC5, 0xC6]
    + list(range(0xD0, 0xFF))
)

# Tables to convert: C name, Rust name, size (None for a single entry).
TABLES = [
    ("opcodes_group12", "GROUP12", 8),
    ("opcodes_group13", "GROUP13", 8),
    ("opcodes_group14", "GROUP14", 8),
    ("opcodes_0F6F", "OPCODES_0F6F", 4),
    ("pshufw", "OPCODES_0F70", 4),
    ("opcodes_0F78", "OPCODES_0F78", 4),
    ("opcodes_0F7E", "OPCODES_0F7E", 4),
    ("opcodes_0F7F", "OPCODES_0F7F", 4),
    ("movq", "OPCODES_0FD6", 4),
    ("opcodes_0F38_00toEF", "OPCODES_0F38", 240),
    ("vinsertps_reg", "VINSERTPS_REG", None),
    ("vinsertps_mem", "VINSERTPS_MEM", None),
    ("opcodes_0F3A", "OPCODES_0F3A", 256),
    ("opcodes_0F10_reg", "OPCODES_0F10_REG", 4),
    ("opcodes_0F10_mem", "OPCODES_0F10_MEM", 4),
    ("opcodes_0F11_reg", "OPCODES_0F11_REG", 4),
    ("opcodes_0F11_mem", "OPCODES_0F11_MEM", 4),
    ("opcodes_0F12_mem", "OPCODES_0F12_MEM", 4),
    ("opcodes_0F12_reg", "OPCODES_0F12_REG", 4),
    ("opcodes_0F16_mem", "OPCODES_0F16_MEM", 4),
    ("opcodes_0F16_reg", "OPCODES_0F16_REG", 4),
    ("opcodes_0F2A", "OPCODES_0F2A", 4),
    ("opcodes_0F2B", "OPCODES_0F2B", 4),
    ("opcodes_0F2C", "OPCODES_0F2C", 4),
    ("opcodes_0F2D", "OPCODES_0F2D", 4),
    ("opcodes_0F5A", "OPCODES_0F5A", 4),
    ("opcodes_0F5B", "OPCODES_0F5B", 4),
    ("opcodes_0FE6", "OPCODES_0FE6", 4),
    ("opcodes_0F", "OPCODES_0F", 256),
]

# Operand count and layout of each macro: which of the fixed arguments go to op0, op1, op2.
# "2op" copies operand 0.
SHAPES = {
    "ENTRY3": (7, lambda a: [(a[1], a[2]), (a[3], a[4]), (a[5], a[6])]),
    "ENTRY4": (7, lambda a: [(a[1], a[2]), (a[3], a[4]), (a[5], a[6])]),
    "GROUP3": (7, lambda a: [(a[1], a[2]), (a[3], a[4]), (a[5], a[6])]),
    "ENTRY2": (5, lambda a: [(a[1], a[2]), ("2op", a[2]), (a[3], a[4])]),
    "GROUP2": (5, lambda a: [(a[1], a[2]), ("2op", a[2]), (a[3], a[4])]),
    "ENTRYwr": (5, lambda a: [(a[1], a[2]), (a[3], a[4]), ("None", "None")]),
    "GROUPwr": (5, lambda a: [(a[1], a[2]), (a[3], a[4]), ("None", "None")]),
    "ENTRYrr": (5, lambda a: [("None", "None"), (a[1], a[2]), (a[3], a[4])]),
    "ENTRYw": (3, lambda a: [(a[1], a[2]), ("None", "None"), ("None", "None")]),
    "GROUPw": (3, lambda a: [(a[1], a[2]), ("None", "None"), ("None", "None")]),
    "ENTRYr": (3, lambda a: [("None", "None"), (a[1], a[2]), ("None", "None")]),
    "ENTRY1": (3, lambda a: [(a[1], a[2]), ("2op", a[2]), ("None", "None")]),
    "GROUP1": (3, lambda a: [(a[1], a[2]), ("2op", a[2]), ("None", "None")]),
    "ENTRY0": (1, lambda a: [("None", "None")] * 3),
    "GROUP0": (1, lambda a: [("None", "None")] * 3),
}

TYPES = {
    "None": "None", "2op": "Op0", "B": "B", "E": "E", "G": "G", "H": "H", "I": "I",
    "M": "M", "N": "N", "P": "P", "Q": "Q", "R": "R", "U": "U", "V": "V", "W": "W",
    "WM": "Wm",
}
SIZES = {
    "None": "None", "b": "B", "w": "W", "d": "D", "q": "Q", "y": "Y", "dq": "Dq",
    "qq": "Qq", "x": "X", "xh": "Xh", "ss": "Ss", "sd": "Sd",
}
SPECIALS = {
    "mmx": "MMX", "op0_Rd": "OP0_RD", "op2_Ry": "OP2_RY", "avx_movx": "AVX_MOVX",
    "sextT0": "SEXTT0", "zextT0": "ZEXTT0", "xchg": "XCHG",
}
VEX_SPECIALS = {
    "vex1_rep3": ("VEX1", "REP_SCALAR"), "vex2_rep3": ("VEX2", "REP_SCALAR"),
    "vex4_rep5": ("VEX4", "REP_SCALAR"), "vex4_unal": ("VEX4", "SSE_UNALIGNED"),
}
CHECKS = {"o64": "CHK_O64", "VEX128": "CHK_VEX128", "W0": "CHK_W0", "W1": "CHK_W1"}
PREFIXES = {"00": "P_00", "66": "P_66", "f3": "P_F3", "f2": "P_F2"}

# The emitters that only the decoder functions pick, so they never appear in a table.
EXTRA_GENS = {
    "Emms", "Vzeroupper", "Vzeroall", "Vucomi", "Vcomi", "Vsqrt", "Vrsqrt", "Vrcp",
    "InsertqR", "ExtrqR",
}


def camel(name):
    parts = [p for p in name.split("_") if p]
    s = "".join(p[0].upper() + p[1:].lower() for p in parts)
    return "X" + s if s[0].isdigit() else s


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def balanced(text, i):
    """Index just past the parenthesis or brace group that starts at text[i]."""
    open_c = text[i]
    close_c = {"(": ")", "{": "}"}[open_c]
    depth = 0
    for j in range(i, len(text)):
        if text[j] == open_c:
            depth += 1
        elif text[j] == close_c:
            depth -= 1
            if depth == 0:
                return j + 1
    raise SystemExit("unbalanced group")


def split_args(s):
    out, depth, cur = [], 0, ""
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += c
    out.append(cur.strip())
    return out


def parse_entry(text):
    """An X86_OP_* invocation into (gen, dec, ops, sizes, flags, cpuid)."""
    m = re.match(r"X86_OP_(\w+)\s*\(", text)
    kind = m.group(1)
    if kind == "NONE":
        return None
    args = split_args(text[m.end():-1])
    nfixed, layout = SHAPES[kind]
    ops = layout(args)
    flags_text = " ".join(args[nfixed:])
    name = args[0]
    flags, cpuid = [], "None"
    vex, vex_special = None, None
    for tok in re.findall(r"\w+\([^)]*\)|\w+", flags_text):
        if tok.startswith("cpuid("):
            cpuid = camel(tok[6:-1])
        elif tok.startswith(("chk(", "chk2(", "chk3(")):
            for c in tok[tok.index("(") + 1:-1].split(","):
                flags.append(CHECKS[c.strip()])
        elif tok.startswith("svm("):
            pass
        elif tok in VEX_SPECIALS:
            vex, vex_special = VEX_SPECIALS[tok]
        elif re.fullmatch(r"vex\d+", tok):
            vex = tok.upper()
        elif tok == "avx2_256":
            vex_special = "AVX2_256"
        elif tok in SPECIALS:
            flags.append(SPECIALS[tok])
        elif tok.startswith("p_"):
            flags.extend(PREFIXES[p] for p in tok[2:].split("_"))
        else:
            raise SystemExit("unknown flag %r in %s" % (tok, name))
    if vex:
        flags.insert(0, vex)
    if vex_special:
        flags.insert(1 if vex else 0, vex_special)
    if kind == "ENTRY4":
        flags.append("OP3")
    is_decode = kind.startswith("GROUP")
    gen = "None" if is_decode else camel(name)
    dec = camel(name) if is_decode else "None"
    if is_decode and name[0].isdigit():
        dec = "D" + name.lower()
    t = [TYPES[o.strip()] for o, _ in ops]
    z = [SIZES[s.strip()] for _, s in ops]
    return gen, dec, t, z, flags, cpuid


def parse_table(text, cname, size, keep=None):
    if size is None:
        m = re.search(r"\b%s\s*=\s*" % re.escape(cname), text)
        start = m.end()
        end = balanced(text, text.index("(", start)) if text[start:].startswith("X86_OP_") else 0
        return parse_entry(text[start:end].strip())
    m = re.search(r"static const X86OpEntry %s\[[^\]]*\]\s*=\s*\{" % re.escape(cname), text)
    if not m:
        raise SystemExit("table %s not found" % cname)
    body_start = m.end() - 1
    body = text[body_start + 1:balanced(text, body_start) - 1]
    out = [None] * size
    pos, idx = 0, 0
    while True:
        while pos < len(body) and body[pos] in " \t\n,":
            pos += 1
        if pos >= len(body):
            break
        im = re.match(r"\[\s*(0x[0-9a-fA-F]+|\d+)\s*\]\s*=\s*", body[pos:])
        if im:
            idx = int(im.group(1), 0)
            pos += im.end()
        if body[pos] == "{":
            pos = balanced(body, pos)
            idx += 1
            continue
        em = re.match(r"X86_OP_\w+\s*", body[pos:])
        if not em:
            raise SystemExit("cannot parse %s at %r" % (cname, body[pos:pos + 40]))
        end = balanced(body, pos + em.end())
        if keep is None or idx in keep:
            out[idx] = parse_entry(body[pos:end])
        idx += 1
        pos = end
    return out


def fmt(entry):
    if entry is None:
        return "E0"
    gen, dec, t, z, flags, cpuid = entry
    ts = ", ".join("T::" + x for x in t)
    zs = ", ".join("Z::" + x for x in z)
    fl = " | ".join(flags) if flags else "0"
    return "e(G::%s, Dec::%s, [%s], [%s], %s, Ft::%s)" % (gen, dec, ts, zs, fl, cpuid)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    with open(path) as f:
        text = strip_comments(f.read())
    gens, decs, feats = set(EXTRA_GENS), set(), set()
    out = []
    for cname, rname, size in TABLES:
        tab = parse_table(text, cname, size, SSE_0F if cname == "opcodes_0F" else None)
        entries = [tab] if size is None else tab
        for x in entries:
            if x:
                gens.add(x[0])
                decs.add(x[1])
                feats.add(x[5])
        if size is None:
            out.append("pub(super) const %s: E = %s;" % (rname, fmt(tab)))
        else:
            out.append("pub(super) static %s: [E; %d] = [" % (rname, size))
            for i, x in enumerate(tab):
                out.append("    // 0x%02x" % i)
                out.append("    %s," % fmt(x))
            out.append("];")
        out.append("")
    head = [
        "// SPDX-License-Identifier: GPL-2.0-or-later",
        "",
        "//! The vector opcode tables of `decode-new.c.inc`, generated by",
        "//! `tests/data/gen_sse_tab.py`. Do not edit by hand.",
        "",
        "use super::sse::flags::*;",
        "use super::sse::{E, E0, T, Z, e};",
        "",
    ]
    enums = [
        ("G", "The emitters, QEMU's `gen_*` functions.", gens),
        ("Dec", "The decoders, QEMU's `decode_*` functions.", decs),
        ("Ft", "The CPUID features, `X86CPUIDFeature`.", feats),
    ]
    for name, doc, items in enums:
        head.append("/// %s" % doc)
        head.append("#[derive(Clone, Copy, Debug, PartialEq, Eq)]")
        head.append("pub(super) enum %s {" % name)
        for v in ["None"] + sorted(items - {"None"}):
            head.append("    %s," % v)
        head.append("}")
        head.append("")
    print("\n".join(head + out).rstrip("\n"))


if __name__ == "__main__":
    main()
