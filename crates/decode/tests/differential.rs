// SPDX-License-Identifier: GPL-2.0-or-later

//! Differential testing against QEMU's own generator. For each .decode file the test builds two
//! programs that feed the same instruction words to a decoder and print, for every word, each
//! translate method the decoder calls with the argument values it was given, and the result:
//!
//! - a C program around the output of QEMU 11.1's scripts/decodetree.py, built with `cc`;
//! - a Rust program around the output of this crate, built with `rustc` in debug mode, so that
//!   an arithmetic overflow in the generated code would panic rather than pass unnoticed.
//!
//! Translate methods return true or false from a hash of their name and the word, so that
//! overlapping groups fall through to later patterns as they would in QEMU, and `!function=`
//! functions return a hash of their name and argument. Both programs make the words with the
//! same generator: half are uniformly random and half are aimed at a randomly chosen pattern
//! (random bits with the pattern's fixed bits forced), and 16 bit decoders get every word. The
//! two outputs are compared line by line.
//!
//! The test needs python3, a C compiler and a QEMU 11.1 source tree, found at `$QEMU_SRC` or
//! `~/src/qemu-v11.1.0`. Without them it says so and passes. `RUVM_DECODE_DIFF_WORDS` sets the
//! number of words for the main three decoders (two million each by default).

use std::fmt::Write as _;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use ruvm_decode::{Info, Invocation, Options, generate_files, inspect, rust_ident};

/// How many words to try.
#[derive(Clone, Copy)]
enum Words {
    /// Every word of a 16 bit decoder.
    All,
    /// This many random words.
    Random(u64),
}

struct Case {
    name: &'static str,
    file: &'static str,
    args: &'static [&'static str],
    words: Words,
}

fn main_words() -> u64 {
    std::env::var("RUVM_DECODE_DIFF_WORDS").ok().and_then(|s| s.parse().ok()).unwrap_or(2_000_000)
}

fn qemu_src() -> Option<PathBuf> {
    let dir = match std::env::var_os("QEMU_SRC") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(std::env::var_os("HOME")?).join("src/qemu-v11.1.0"),
    };
    dir.join("scripts/decodetree.py").is_file().then_some(dir)
}

fn tool_works(cmd: &str, arg: &str) -> bool {
    Command::new(cmd)
        .arg(arg)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// The tools the test needs, or `None` with a note on stderr.
fn tools() -> Option<(PathBuf, String)> {
    let Some(src) = qemu_src() else {
        eprintln!("skipping: no QEMU 11.1 source tree (set QEMU_SRC)");
        return None;
    };
    if !tool_works("python3", "--version") {
        eprintln!("skipping: python3 not found");
        return None;
    }
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    if !tool_works(&cc, "--version") {
        eprintln!("skipping: no C compiler");
        return None;
    }
    Some((src, cc))
}

fn vendor_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor-qemu/decode")
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// The code both programs share, in C and in Rust: the word generator and the hashes.
const C_COMMON: &str = r#"
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef struct DisasContext { int unused; } DisasContext;

static inline uint32_t extract32(uint32_t value, int start, int length)
{
    assert(start >= 0 && length > 0 && length <= 32 - start);
    return (value >> start) & (~0U >> (32 - length));
}
static inline int32_t sextract32(uint32_t value, int start, int length)
{
    assert(start >= 0 && length > 0 && length <= 32 - start);
    return ((int32_t)(value << (32 - length - start))) >> (32 - length);
}
static inline uint64_t extract64(uint64_t value, int start, int length)
{
    assert(start >= 0 && length > 0 && length <= 64 - start);
    return (value >> start) & (~0ULL >> (64 - length));
}
static inline int64_t sextract64(uint64_t value, int start, int length)
{
    assert(start >= 0 && length > 0 && length <= 64 - start);
    return ((int64_t)(value << (64 - length - start))) >> (64 - length);
}
static inline uint32_t deposit32(uint32_t value, int start, int length, uint32_t fieldval)
{
    assert(start >= 0 && length > 0 && length <= 32 - start);
    uint32_t mask = (~0U >> (32 - length)) << start;
    return (value & ~mask) | ((fieldval << start) & mask);
}
static inline uint64_t deposit64(uint64_t value, int start, int length, uint64_t fieldval)
{
    assert(start >= 0 && length > 0 && length <= 64 - start);
    uint64_t mask = (~0ULL >> (64 - length)) << start;
    return (value & ~mask) | ((fieldval << start) & mask);
}

static uint64_t cur_insn;
static uint8_t cur_bytes[8];
static int cur_len;
static uint64_t rng;

static uint64_t next(void)
{
    uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static uint64_t mix(uint64_t z)
{
    z ^= z >> 33;
    z *= 0xff51afd7ed558ccdULL;
    z ^= z >> 33;
    z *= 0xc4ceb9fe1a85ec53ULL;
    return z ^ (z >> 33);
}
static bool ret(uint64_t name)
{
    return (mix(name ^ cur_insn) & 3) != 0;
}
"#;

const RUST_COMMON: &str = r#"
use std::io::Write as _;

struct Ctx {
    out: std::io::BufWriter<std::io::StdoutLock<'static>>,
    insn: u64,
    bytes: [u8; 8],
    len: i32,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

fn mix(mut z: u64) -> u64 {
    z ^= z >> 33;
    z = z.wrapping_mul(0xff51afd7ed558ccd);
    z ^= z >> 33;
    z = z.wrapping_mul(0xc4ceb9fe1a85ec53);
    z ^ (z >> 33)
}

impl Ctx {
    fn ret(&self, name: u64) -> bool {
        (mix(name ^ self.insn) & 3) != 0
    }
}
"#;

/// Both programs for one case, as text: (C, Rust).
fn harnesses(case: &Case, opts: &Options, info: &Info, seed: u64) -> (String, String) {
    let width = opts.insn_width;
    let insnmask: u64 = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
    let (cst, rst) = if width == 64 { ("int64_t", "i64") } else { ("int", "i32") };
    let (cins, rins) = match width {
        16 => ("uint16_t", "u16"),
        32 => ("uint32_t", "u32"),
        _ => ("uint64_t", "u64"),
    };
    let df = &opts.decode_function;
    let varwidth = opts.var_insn_width;

    let mut c = String::from(C_COMMON);
    let mut r = String::from(RUST_COMMON);

    // Argument sets that another file owns.
    for set in info.arg_sets.iter().filter(|s| s.is_extern) {
        c += "typedef struct {\n";
        for f in &set.fields {
            let _ = writeln!(c, "    {} {};", f.c_type, f.name);
        }
        let _ = writeln!(c, "}} arg_{};\n", set.name);
        r += "#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]\n";
        let _ = writeln!(r, "struct arg_{} {{", set.name);
        for f in &set.fields {
            let _ = writeln!(r, "    {}: {},", rust_ident(&f.name), f.rust_type);
        }
        r += "}\n\n";
    }

    // Functions, and the byte loader for variable width decoders.
    let mut methods = String::new();
    for f in &info.functions {
        let h = fnv(&f.name);
        if f.takes_value {
            let _ = writeln!(
                c,
                "static {cst} {}(DisasContext *ctx, {cst} x)\n{{\n    return ({cst})mix({h:#x}ULL ^ (uint64_t)(int64_t)x ^ cur_insn);\n}}",
                f.name
            );
            let _ = writeln!(
                methods,
                "    fn {}(&mut self, x: {rst}) -> {rst} {{\n        mix({h:#x} ^ (x as i64 as u64) ^ self.insn) as {rst}\n    }}",
                rust_ident(&f.name)
            );
        } else {
            let _ = writeln!(
                c,
                "static {cst} {}(DisasContext *ctx)\n{{\n    return ({cst})mix({h:#x}ULL ^ cur_insn);\n}}",
                f.name
            );
            let _ = writeln!(
                methods,
                "    fn {}(&mut self) -> {rst} {{\n        mix({h:#x} ^ self.insn) as {rst}\n    }}",
                rust_ident(&f.name)
            );
        }
    }
    if varwidth {
        let _ = writeln!(
            c,
            "static {cins} {df}_load_bytes(DisasContext *ctx, {cins} insn, int i, int n)\n{{\n    for (int k = i; k < n; k++) {{\n        insn |= ({cins})cur_bytes[k] << ({width} - 8 - 8 * k);\n    }}\n    if (n > cur_len) {{\n        cur_len = n;\n    }}\n    return insn;\n}}"
        );
        let _ = writeln!(
            methods,
            "    fn {}(&mut self, mut insn: {rins}, i: i32, n: i32) -> {rins} {{\n        for k in i..n {{\n            insn |= ({rins}::from(self.bytes[k as usize])) << ({width} - 8 - 8 * k);\n        }}\n        if n > self.len {{\n            self.len = n;\n        }}\n        insn\n    }}",
            rust_ident(&format!("{df}_load_bytes"))
        );
    }

    c += "\n#include \"gen.c.inc\"\n\n";
    r += "include!(\"gen.rs\");\n\n";

    // Translate methods, one per pattern name.
    let mut seen = std::collections::HashSet::new();
    let _ = writeln!(r, "impl {} for Ctx {{", rust_ident(&opts.effective_trait_name()));
    r += &methods;
    for p in &info.patterns {
        if !seen.insert(p.name.clone()) {
            continue;
        }
        let set = info.arg_sets.iter().find(|s| s.name == p.arg_set).unwrap();
        let tname = format!("{}_{}", opts.translate_prefix, p.name);
        let h = fnv(&p.name);
        let _ = writeln!(c, "static bool {tname}(DisasContext *ctx, arg_{} *a)\n{{", p.name);
        let _ = writeln!(c, "    printf(\"T {}\");", p.name);
        let _ = writeln!(
            r,
            "    fn {}(&mut self, a: &mut arg_{}) -> bool {{",
            rust_ident(&tname),
            p.name
        );
        let _ = writeln!(r, "        let _ = write!(self.out, \"T {}\");", p.name);
        for f in &set.fields {
            let _ = writeln!(c, "    printf(\" {0}=%lld\", (long long)a->{0});", f.name);
            let _ = writeln!(
                r,
                "        let _ = write!(self.out, \" {}={{}}\", a.{} as i64);",
                f.name,
                rust_ident(&f.name)
            );
        }
        let _ = writeln!(c, "    printf(\"\\n\");\n    return ret({h:#x}ULL);\n}}\n");
        let _ =
            writeln!(r, "        let _ = writeln!(self.out);\n        self.ret({h:#x})\n    }}");
    }
    r += "}\n\n";

    // The pattern table and the main loop.
    let n = info.patterns.len();
    c += "static const uint64_t pats[][2] = {\n";
    r += "const PATS: &[(u64, u64)] = &[\n";
    for p in &info.patterns {
        let _ = writeln!(c, "    {{ {:#x}ULL, {:#x}ULL }},", p.fixed_mask, p.fixed_bits);
        let _ = writeln!(r, "    ({:#x}, {:#x}),", p.fixed_mask, p.fixed_bits);
    }
    if n == 0 {
        c += "    { 0, 0 },\n";
    }
    c += "};\n\n";
    r += "];\n\n";

    let (count, exhaustive) = match case.words {
        Words::All => (1u64 << width, true),
        Words::Random(k) => (k, false),
    };
    let ex = if exhaustive { 1 } else { 0 };
    let load_c = if varwidth {
        format!(
            "        for (int k = 0; k < {bytes}; k++) {{\n            cur_bytes[k] = (uint8_t)(w >> ({width} - 8 - 8 * k));\n        }}\n        cur_len = 0;\n        cur_insn = w;\n        {cins} insn = {df}_load(&ctx);\n        printf(\"L %d %llx\\n\", cur_len, (unsigned long long)insn);\n",
            bytes = width / 8
        )
    } else {
        format!("        {cins} insn = ({cins})w;\n")
    };
    let load_r = if varwidth {
        format!(
            "        for k in 0..{bytes} {{\n            ctx.bytes[k] = (w >> ({width} - 8 - 8 * k)) as u8;\n        }}\n        ctx.len = 0;\n        ctx.insn = w;\n        let insn = {df}_load(&mut ctx);\n        let _ = writeln!(ctx.out, \"L {{}} {{:x}}\", ctx.len, insn);\n",
            bytes = width / 8
        )
    } else {
        format!("        let insn = w as {rins};\n")
    };
    let _ = writeln!(
        c,
        "int main(void)\n{{\n    static char buf[1 << 20];\n    setvbuf(stdout, buf, _IOFBF, sizeof(buf));\n    DisasContext ctx;\n    rng = {seed:#x}ULL;\n    for (uint64_t i = 0; i < {count}ULL; i++) {{\n        uint64_t w;\n        if ({ex}) {{\n            w = i;\n        }} else if ({n} == 0 || (next() & 1) == 0) {{\n            w = next() & {insnmask:#x}ULL;\n        }} else {{\n            uint64_t p = next() % {n}ULL;\n            w = ((next() & ~pats[p][0]) | pats[p][1]) & {insnmask:#x}ULL;\n        }}\n        printf(\"I %llx\\n\", (unsigned long long)w);\n{load_c}        cur_insn = insn;\n        bool r = {df}(&ctx, insn);\n        printf(\"R %d\\n\", r ? 1 : 0);\n    }}\n    fflush(stdout);\n    return 0;\n}}"
    );
    let _ = writeln!(
        r,
        "fn main() {{\n    let mut ctx = Ctx {{ out: std::io::BufWriter::with_capacity(1 << 20, std::io::stdout().lock()), insn: 0, bytes: [0; 8], len: 0 }};\n    let mut rng = Rng({seed:#x});\n    for i in 0..{count}u64 {{\n        let w: u64 = if {exhaustive} {{\n            i\n        }} else if {n} == 0 || (rng.next() & 1) == 0 {{\n            rng.next() & {insnmask:#x}\n        }} else {{\n            let p = (rng.next() % {n}u64) as usize;\n            ((rng.next() & !PATS[p].0) | PATS[p].1) & {insnmask:#x}\n        }};\n        let _ = writeln!(ctx.out, \"I {{:x}}\", w);\n{load_r}        ctx.insn = insn as u64;\n        let r = {df}(&mut ctx, insn);\n        let _ = writeln!(ctx.out, \"R {{}}\", r as i32);\n    }}\n    let _ = ctx.out.flush();\n}}"
    );
    (c, r)
}

fn run(cmd: &mut Command, what: &str) {
    let out = cmd.output().unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(
        out.status.success(),
        "{what} failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Build both programs for `case`, run them, and compare what they print.
fn check(case: &Case, src: &Path, cc: &str) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("decode-diff").join(case.name);
    std::fs::create_dir_all(&dir).unwrap();
    let input = vendor_dir().join(case.file);
    let text = std::fs::read_to_string(&input).unwrap();

    let mut argv: Vec<&str> = case.args.to_vec();
    argv.push(case.file);
    let opts = Invocation::parse(&argv).unwrap().options;
    let info = inspect(&[(case.file, &text)], &opts).unwrap();
    let rust = generate_files(&[(case.file, &text)], &opts).unwrap();
    std::fs::write(dir.join("gen.rs"), rust).unwrap();

    run(
        Command::new("python3")
            .arg(src.join("scripts/decodetree.py"))
            .args(case.args)
            .arg("-o")
            .arg(dir.join("gen.c.inc"))
            .arg(&input),
        "decodetree.py",
    );

    let (c, r) = harnesses(case, &opts, &info, fnv(case.name));
    std::fs::write(dir.join("main.c"), c).unwrap();
    std::fs::write(dir.join("main.rs"), r).unwrap();
    let cbin = dir.join(format!("c-decoder{}", std::env::consts::EXE_SUFFIX));
    let rbin = dir.join(format!("rust-decoder{}", std::env::consts::EXE_SUFFIX));
    run(Command::new(cc).args(["-O1", "-w", "-o"]).arg(&cbin).arg(dir.join("main.c")), "cc");
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    run(
        Command::new(rustc)
            .args(["--edition=2024", "-C", "opt-level=1", "-C", "debug-assertions=on"])
            .args(["-C", "overflow-checks=on", "-A", "warnings", "-o"])
            .arg(&rbin)
            .arg(dir.join("main.rs")),
        "rustc",
    );

    let mut cp = Command::new(&cbin).stdout(Stdio::piped()).spawn().unwrap();
    let mut rp = Command::new(&rbin).stdout(Stdio::piped()).spawn().unwrap();
    let mut cl = BufReader::new(cp.stdout.take().unwrap()).lines();
    let mut rl = BufReader::new(rp.stdout.take().unwrap()).lines();
    let mut words = 0u64;
    let mut calls = 0u64;
    let mut context: Vec<String> = Vec::new();
    loop {
        let a = cl.next().map(|l| l.unwrap());
        let b = rl.next().map(|l| l.unwrap());
        if a != b {
            let _ = cp.kill();
            let _ = rp.kill();
            panic!(
                "{}: outputs differ after {words} words\nthis word so far:\n{}\nC:    {a:?}\nRust: {b:?}",
                case.name,
                context.join("\n")
            );
        }
        let Some(line) = a else { break };
        if line.starts_with('I') {
            words += 1;
            context.clear();
        } else if line.starts_with('T') {
            calls += 1;
        }
        context.push(line);
    }
    assert!(cp.wait().unwrap().success(), "{}: the C decoder failed", case.name);
    assert!(rp.wait().unwrap().success(), "{}: the Rust decoder failed", case.name);
    let expected = match case.words {
        Words::All => 1u64 << opts.insn_width,
        Words::Random(k) => k,
    };
    assert_eq!(words, expected, "{}", case.name);
    assert!(calls > 0, "{}: no translate method was ever called", case.name);
    eprintln!("{}: {words} words, {calls} translate calls, identical", case.name);
    // Disk is scarce; a failing case keeps its files for a look.
    let _ = std::fs::remove_dir_all(&dir);
}

fn check_all(cases: &[Case]) {
    let Some((src, cc)) = tools() else { return };
    std::thread::scope(|s| {
        let handles: Vec<_> = cases.iter().map(|c| s.spawn(|| check(c, &src, &cc))).collect();
        let mut failed = Vec::new();
        for (h, c) in handles.into_iter().zip(cases) {
            if h.join().is_err() {
                failed.push(c.name);
            }
        }
        assert!(failed.is_empty(), "differences in {failed:?}");
    });
}

/// The three decoders the task names: RISC-V 32 bit, AArch64 and Thumb-2, with millions of
/// words each.
#[test]
fn riscv_a64_t32_match_decodetree_py() {
    let n = main_words();
    check_all(&[
        Case {
            name: "riscv-insn32",
            file: "riscv/insn32.decode",
            args: &["--static-decode=decode_insn32"],
            words: Words::Random(n),
        },
        Case {
            name: "arm-a64",
            file: "arm/tcg/a64.decode",
            args: &["--static-decode=disas_a64"],
            words: Words::Random(n),
        },
        Case {
            name: "arm-t32",
            file: "arm/tcg/t32.decode",
            args: &["--static-decode=disas_t32"],
            words: Words::Random(n),
        },
    ]);
}

/// Every other file under target/, with fewer words, including the 16 bit decoders in full,
/// PowerPC's 64 bit prefixed instructions and RX's variable width loader.
#[test]
fn other_targets_match_decodetree_py() {
    const K: Words = Words::Random(200_000);
    check_all(&[
        Case {
            name: "arm-t16",
            file: "arm/tcg/t16.decode",
            args: &["-w", "16", "--static-decode=disas_t16"],
            words: Words::All,
        },
        Case {
            name: "riscv-insn16",
            file: "riscv/insn16.decode",
            args: &["--static-decode=decode_insn16", "--insnwidth=16"],
            words: Words::All,
        },
        Case {
            name: "avr",
            file: "avr/insn.decode",
            args: &["--decode", "decode_insn", "--insnwidth", "16"],
            words: Words::All,
        },
        Case {
            name: "ppc-insn64",
            file: "ppc/insn64.decode",
            args: &["--static-decode=decode_insn64", "--insnwidth=64"],
            words: K,
        },
        Case {
            name: "ppc-insn32",
            file: "ppc/insn32.decode",
            args: &["--static-decode=decode_insn32"],
            words: K,
        },
        Case { name: "rx", file: "rx/insns.decode", args: &["--varinsnwidth", "32"], words: K },
        Case {
            name: "arm-sve",
            file: "arm/tcg/sve.decode",
            args: &["--decode=disas_sve"],
            words: K,
        },
        Case {
            name: "arm-sme",
            file: "arm/tcg/sme.decode",
            args: &["--decode=disas_sme"],
            words: K,
        },
        Case {
            name: "arm-sme-fa64",
            file: "arm/tcg/sme-fa64.decode",
            args: &["--static-decode=disas_sme_fa64"],
            words: K,
        },
        Case {
            name: "arm-neon-shared",
            file: "arm/tcg/neon-shared.decode",
            args: &["--decode=disas_neon_shared"],
            words: K,
        },
        Case {
            name: "arm-neon-dp",
            file: "arm/tcg/neon-dp.decode",
            args: &["--decode=disas_neon_dp"],
            words: K,
        },
        Case {
            name: "arm-neon-ls",
            file: "arm/tcg/neon-ls.decode",
            args: &["--decode=disas_neon_ls"],
            words: K,
        },
        Case {
            name: "arm-vfp",
            file: "arm/tcg/vfp.decode",
            args: &["--decode=disas_vfp"],
            words: K,
        },
        Case {
            name: "arm-vfp-uncond",
            file: "arm/tcg/vfp-uncond.decode",
            args: &["--decode=disas_vfp_uncond"],
            words: K,
        },
        Case {
            name: "arm-m-nocp",
            file: "arm/tcg/m-nocp.decode",
            args: &["--decode=disas_m_nocp"],
            words: K,
        },
        Case {
            name: "arm-mve",
            file: "arm/tcg/mve.decode",
            args: &["--decode=disas_mve"],
            words: K,
        },
        Case {
            name: "arm-a32",
            file: "arm/tcg/a32.decode",
            args: &["--static-decode=disas_a32"],
            words: K,
        },
        Case {
            name: "arm-a32-uncond",
            file: "arm/tcg/a32-uncond.decode",
            args: &["--static-decode=disas_a32_uncond"],
            words: K,
        },
        Case { name: "hppa", file: "hppa/insns.decode", args: &[], words: K },
        Case { name: "loongarch", file: "loongarch/insns.decode", args: &[], words: K },
        Case { name: "microblaze", file: "microblaze/insns.decode", args: &[], words: K },
        Case { name: "or1k", file: "or1k/insns.decode", args: &[], words: K },
        Case { name: "sparc", file: "sparc/insns.decode", args: &[], words: K },
        Case {
            name: "riscv-xthead",
            file: "riscv/xthead.decode",
            args: &["--static-decode=decode_xthead"],
            words: K,
        },
        Case {
            name: "riscv-xventana",
            file: "riscv/XVentanaCondOps.decode",
            args: &["--static-decode=decode_XVentanaCodeOps"],
            words: K,
        },
        Case {
            name: "riscv-xmips",
            file: "riscv/xmips.decode",
            args: &["--static-decode=decode_xmips"],
            words: K,
        },
        Case {
            name: "riscv-xlrbr",
            file: "riscv/xlrbr.decode",
            args: &["--static-decode=decode_xlrbr"],
            words: K,
        },
        Case {
            name: "mips-rel6",
            file: "mips/tcg/rel6.decode",
            args: &["--decode=decode_isa_rel6"],
            words: K,
        },
        Case {
            name: "mips-msa",
            file: "mips/tcg/msa.decode",
            args: &["--decode=decode_ase_msa"],
            words: K,
        },
        Case {
            name: "mips-tx79",
            file: "mips/tcg/tx79.decode",
            args: &["--static-decode=decode_tx79"],
            words: K,
        },
        Case {
            name: "mips-vr54xx",
            file: "mips/tcg/vr54xx.decode",
            args: &["--decode=decode_ext_vr54xx"],
            words: K,
        },
        Case {
            name: "mips-octeon",
            file: "mips/tcg/octeon.decode",
            args: &["--decode=decode_ext_octeon"],
            words: K,
        },
        Case {
            name: "mips-lcsr",
            file: "mips/tcg/lcsr.decode",
            args: &["--decode=decode_ase_lcsr"],
            words: K,
        },
        Case {
            name: "mips-godson2",
            file: "mips/tcg/godson2.decode",
            args: &["--static-decode=decode_godson2"],
            words: K,
        },
        Case {
            name: "mips-loong-ext",
            file: "mips/tcg/loong-ext.decode",
            args: &["--static-decode=decode_loong_ext"],
            words: K,
        },
    ]);
}
