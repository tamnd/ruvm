// SPDX-License-Identifier: GPL-2.0-or-later

//! QEMU's own test plugins, built from source with the system C compiler against QEMU 11.1's
//! `qemu-plugin.h` and loaded unmodified, instrumenting a toy guest on the JIT.
//!
//! The sources in `tests/plugins` are copies of `tests/tcg/plugins/*.c`,
//! `contrib/plugins/hotblocks.c` and `include/plugins/qemu-plugin.h` from QEMU 11.1. The
//! tests skip, passing, when there is no C compiler or no glib.
//!
//! The plugin host is global state that cannot be torn down, like QEMU's, so every test runs
//! its guest in a child process: the test binary runs itself with the test's name and an
//! environment variable that says which plugins to load.
//!
//! The toy ISA has eight 64-bit registers and 32-bit little endian instructions:
//! `op | a << 8 | b << 12 | imm16 << 16`, with `imm16` signed.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use ruvm_jit::cpu_exec::cpu_exec;
use ruvm_jit::translate::TbBuild;
use ruvm_jit::{
    Cpu, CpuLoopExit, CpuOps, DisasContextBase, DisasJumpType, ENV_TARGET_OFFSET, InterpBackend,
    Jit, JitConfig, MmuAccessType, Ra, Tb, TbCpuState, TranslatorOps, Vcpu, excp, page, plugin,
    translator_loop,
};
use ruvm_jit_core::types::Cond;
use ruvm_jit_core::{Func, HelperInfo, HelperType, MemOp, Temp};
use ruvm_jit_interp::{HelperEnv, HelperRegistry, Unwind};
use ruvm_mem::{
    AccessCtx, AccessSize, AddressSpace, Endian, MemResult, MemTxAttrs, MemorySystem, MmioOps,
};
use ruvm_plugin::{GdbReg, PluginDesc, PluginTarget, QemuInfo};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const PAGE: u64 = 0x1000;

const LI: u32 = 1;
const ADDI: u32 = 2;
const ADD: u32 = 3;
const LD: u32 = 4;
const ST: u32 = 5;
const BNEZ: u32 = 6;
const J: u32 = 7;
const STOP: u32 = 9;
const SWI: u32 = 11;
const LIH: u32 = 15;

const NAMES: [&str; 16] = [
    "?", "li", "addi", "add", "ld", "st", "bnez", "j", "?", "stop", "?", "swi", "?", "?", "?",
    "lih",
];

const fn reg_off(r: usize) -> usize {
    ENV_TARGET_OFFSET + 8 * r
}
const PC_OFF: usize = ENV_TARGET_OFFSET + 64;
const ENV_SIZE: usize = 80;

const CODE: u64 = 0x1000;
const DATA: u64 = 0x4000;
const DEV_BASE: u64 = 0x10_0000;

fn enc(op: u32, a: u32, b: u32, imm: i32) -> u32 {
    assert!((-0x8000..0x8000).contains(&imm));
    op | (a << 8) | (b << 12) | ((imm as u32 & 0xffff) << 16)
}

/// A tiny assembler. Branch offsets are relative to the next instruction.
struct Asm {
    words: Vec<u32>,
}

impl Asm {
    fn new() -> Asm {
        Asm { words: Vec::new() }
    }
    fn pc(&self) -> u64 {
        CODE + 4 * self.words.len() as u64
    }
    fn i(&mut self, op: u32, a: u32, b: u32, imm: i32) -> &mut Asm {
        self.words.push(enc(op, a, b, imm));
        self
    }
    fn li(&mut self, a: u32, imm: i32) -> &mut Asm {
        self.i(LI, a, 0, imm)
    }
    /// Load a 32-bit constant.
    fn li32(&mut self, a: u32, v: u32) -> &mut Asm {
        self.li(a, (v >> 16) as i32).i(LIH, a, 0, (v & 0xffff) as u16 as i16 as i32)
    }
    fn addi(&mut self, a: u32, b: u32, imm: i32) -> &mut Asm {
        self.i(ADDI, a, b, imm)
    }
    fn jump_to(&mut self, op: u32, a: u32, target: u64) -> &mut Asm {
        let off = target as i64 - (self.pc() as i64 + 4);
        self.i(op, a, 0, off as i32)
    }
    fn bnez(&mut self, a: u32, target: u64) -> &mut Asm {
        self.jump_to(BNEZ, a, target)
    }
    fn stop(&mut self) -> &mut Asm {
        self.i(STOP, 0, 0, 0)
    }
    fn swi(&mut self) -> &mut Asm {
        self.i(SWI, 0, 0, 0)
    }
    fn load(&self, as_: &AddressSpace) {
        let bytes: Vec<u8> = self.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert!(as_.write(CODE, U, &bytes).is_ok());
    }
}

fn rd64(env: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(env[off..off + 8].try_into().unwrap())
}

fn wr64(env: &mut [u8], off: usize, v: u64) {
    env[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn reg(v: &Vcpu, r: usize) -> u64 {
    rd64(&v.env, reg_off(r))
}

/// The toy CPU: no paging, one MMU index.
#[derive(Debug, Default)]
struct ToyOps;

impl CpuOps for ToyOps {
    fn translate_code(&self, cpu: &mut Cpu<'_>, tb: &mut TbBuild) -> Result<(), CpuLoopExit> {
        translator_loop(cpu, tb, &mut ToyDisas)
    }

    fn get_tb_cpu_state(&self, cpu: &Cpu<'_>) -> TbCpuState {
        TbCpuState { pc: rd64(cpu.env, PC_OFF), flags: 0, cflags: 0, cs_base: 0 }
    }

    fn restore_state_to_opc(&self, cpu: &mut Cpu<'_>, _tb: &Tb, data: &[u64; 3]) {
        wr64(cpu.env, PC_OFF, data[0]);
    }

    fn set_pc(&self, cpu: &mut Cpu<'_>, pc: u64) {
        wr64(cpu.env, PC_OFF, pc);
    }

    fn get_pc(&self, cpu: &Cpu<'_>) -> u64 {
        rd64(cpu.env, PC_OFF)
    }

    fn cpu_exec_interrupt(&self, _cpu: &mut Cpu<'_>, _interrupt_request: u32) -> bool {
        false
    }

    fn do_interrupt(&self, cpu: &mut Cpu<'_>) {
        panic!("unexpected exception {}", cpu.core.exception_index);
    }

    fn tlb_fill(
        &self,
        cpu: &mut Cpu<'_>,
        addr: u64,
        _size: usize,
        _access_type: MmuAccessType,
        mmu_idx: usize,
        _probe: bool,
        _ra: Ra,
    ) -> Result<bool, CpuLoopExit> {
        let vpage = addr & !(PAGE - 1);
        cpu.tlb_set_page(vpage, vpage, page::RWX, mmu_idx, PAGE);
        Ok(true)
    }

    fn do_unaligned_access(
        &self,
        cpu: &mut Cpu<'_>,
        _addr: u64,
        _access_type: MmuAccessType,
        _mmu_idx: usize,
        ra: Ra,
    ) -> CpuLoopExit {
        cpu.raise_exception(excp::DEBUG, ra)
    }

    fn mmu_index(&self, _cpu: &Cpu<'_>, _ifetch: bool) -> usize {
        0
    }
}

/// `toy_raise(env, excp)`.
fn helper_raise(_h: &mut HelperEnv<'_>, args: &[u64]) -> Result<u128, Unwind> {
    Err(Unwind::Exception(args[1] & 0xffff_ffff))
}

/// `toy_swi(env)`: a system call, number in r0 and arguments in r1 to r6, result in r0. The
/// toy "kernel" returns the number plus 100. The plugin hooks run in the order of linux-user's
/// `do_syscall()`.
fn helper_swi(h: &mut HelperEnv<'_>, _args: &[u64]) -> Result<u128, Unwind> {
    let mut cpu = Cpu::from_helper_env(h).expect("runs under the runtime");
    let num = rd64(cpu.env, reg_off(0)) as i64;
    let mut args = [0u64; 8];
    for (i, a) in args.iter_mut().enumerate().take(6) {
        *a = rd64(cpu.env, reg_off(i + 1));
    }
    let r = (|| -> Result<(), CpuLoopExit> {
        plugin::vcpu_syscall(&mut cpu, num, &args)?;
        let ret = plugin::vcpu_syscall_filter(&mut cpu, num, &args)?.unwrap_or(num + 100);
        plugin::vcpu_syscall_ret(&mut cpu, num, ret)?;
        wr64(cpu.env, reg_off(0), ret as u64);
        Ok(())
    })();
    match r {
        Ok(()) => Ok(0),
        Err(e) => Err(cpu.unwind(e)),
    }
}

fn raise_info() -> HelperInfo {
    HelperInfo::new("toy_raise", 0, HelperType::Void, &[HelperType::Ptr, HelperType::I32])
}

fn swi_info() -> HelperInfo {
    HelperInfo::new("toy_swi", 0, HelperType::Void, &[HelperType::Ptr])
}

fn backend() -> Arc<InterpBackend> {
    let mut reg = HelperRegistry::new();
    reg.register_info(&raise_info(), helper_raise);
    reg.register_info(&swi_info(), helper_swi);
    Arc::new(InterpBackend::with_helpers(reg))
}

/// The front end.
struct ToyDisas;

fn ld_reg(f: &mut Func, r: u32) -> ruvm_jit_core::ir::TempI64 {
    let t = f.temp_new_i64();
    let env = f.env();
    f.gen_ld_i64(t, env, reg_off(r as usize) as i64);
    t
}

fn st_reg(f: &mut Func, r: u32, t: ruvm_jit_core::ir::TempI64) {
    let env = f.env();
    f.gen_st_i64(t, env, reg_off(r as usize) as i64);
}

fn set_pc(f: &mut Func, pc: u64) {
    let t = f.constant_i64(pc as i64);
    let env = f.env();
    f.gen_st_i64(t, env, PC_OFF as i64);
}

fn gen_goto(db: &mut DisasContextBase<'_>, n: u64, dest: u64) {
    let use_goto_tb = db.translator_use_goto_tb(dest);
    let id = db.tb.id;
    let f = &mut db.tb.f;
    if use_goto_tb {
        f.gen_goto_tb(n);
        set_pc(f, dest);
        f.gen_exit_tb(id, n);
    } else {
        set_pc(f, dest);
        f.gen_lookup_and_goto_ptr();
    }
}

impl TranslatorOps for ToyDisas {
    fn insn_start(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        let pc = db.pc_next;
        db.tb.f.gen_insn_start(&[pc]);
    }

    fn translate_insn(
        &mut self,
        db: &mut DisasContextBase<'_>,
        cpu: &mut Cpu<'_>,
    ) -> Result<(), CpuLoopExit> {
        let pc = db.pc_next;
        let w = db.translator_ldl(cpu, pc, Endian::Little)?;
        db.pc_next = pc + 4;
        let next = pc + 4;
        let op = w & 0xff;
        let a = (w >> 8) & 7;
        let b = (w >> 12) & 7;
        let imm = i64::from((w >> 16) as u16 as i16);
        let target = next.wrapping_add(imm as u64);
        let f = &mut db.tb.f;
        match op {
            LI => {
                let t = f.constant_i64(imm);
                st_reg(f, a, t);
            }
            LIH => {
                let t = ld_reg(f, a);
                f.gen_shli_i64(t, t, 16);
                f.gen_ori_i64(t, t, imm & 0xffff);
                st_reg(f, a, t);
            }
            ADDI => {
                let t = ld_reg(f, b);
                f.gen_addi_i64(t, t, imm);
                st_reg(f, a, t);
            }
            ADD => {
                let x = ld_reg(f, a);
                let y = ld_reg(f, b);
                f.gen_add_i64(x, x, y);
                st_reg(f, a, x);
            }
            LD => {
                let addr = ld_reg(f, b);
                f.gen_addi_i64(addr, addr, imm);
                let v = f.temp_new_i64();
                f.gen_qemu_ld_i64(v, addr, 0, MemOp::UL);
                st_reg(f, a, v);
            }
            ST => {
                let addr = ld_reg(f, b);
                f.gen_addi_i64(addr, addr, imm);
                let v = ld_reg(f, a);
                f.gen_qemu_st_i64(v, addr, 0, MemOp::UL);
            }
            BNEZ => {
                let t = ld_reg(f, a);
                let l = f.new_label();
                f.gen_brcondi_i64(Cond::Ne, t, 0, l);
                gen_goto(db, 1, next);
                db.tb.f.gen_set_label(l);
                gen_goto(db, 0, target);
                db.is_jmp = DisasJumpType::NoReturn;
            }
            J => {
                gen_goto(db, 0, target);
                db.is_jmp = DisasJumpType::NoReturn;
            }
            SWI => {
                let h = f.helper(swi_info());
                let env = f.env();
                f.gen_call(h, None, &[Temp::from(env)]);
            }
            _ => {
                // STOP, and anything unknown.
                let h = f.helper(raise_info());
                let env = f.env();
                let c = f.constant_i32(excp::DEBUG);
                f.gen_call(h, None, &[Temp::from(env), Temp::from(c)]);
                db.is_jmp = DisasJumpType::NoReturn;
            }
        }
        Ok(())
    }

    fn tb_stop(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        match db.is_jmp {
            DisasJumpType::Next | DisasJumpType::TooMany => {
                let next = db.pc_next;
                gen_goto(db, 0, next);
            }
            _ => {}
        }
    }
}

/// What the plugins learn about the toy: its registers, names for its instructions.
#[derive(Debug)]
struct ToyTarget;

impl PluginTarget for ToyTarget {
    fn registers(&self, _cpu: &Cpu<'_>) -> Vec<GdbReg> {
        (0..9)
            .map(|n| GdbReg {
                num: n,
                name: Some(if n == 8 { "pc".to_string() } else { format!("r{n}") }),
                feature: "org.ruvm.toy".to_string(),
            })
            .collect()
    }

    fn read_register(&self, cpu: &mut Cpu<'_>, reg: u32, buf: &mut Vec<u8>) -> usize {
        if reg > 8 {
            return 0;
        }
        buf.extend_from_slice(&rd64(cpu.env, reg_off(reg as usize)).to_le_bytes());
        8
    }

    fn write_register(&self, cpu: &mut Cpu<'_>, reg: u32, buf: &[u8]) -> usize {
        if reg > 7 || buf.len() < 8 {
            return 0;
        }
        wr64(cpu.env, reg_off(reg as usize), u64::from_le_bytes(buf[..8].try_into().unwrap()));
        8
    }

    fn disas(&self, _cpu: &Cpu<'_>, _vaddr: u64, bytes: &[u8]) -> String {
        let w = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        let (op, a, b) = (w & 0xff, (w >> 8) & 7, (w >> 12) & 7);
        let imm = (w >> 16) as u16 as i16;
        let name = NAMES.get(op as usize).copied().unwrap_or("?");
        match op {
            LD | ST => format!("{name} r{a}, {imm}(r{b})"),
            ADDI => format!("{name} r{a}, r{b}, {imm}"),
            _ => format!("{name} r{a}, {imm}"),
        }
    }
}

/// A device that returns `0x40 + offset`.
struct Dev;

impl MmioOps for Dev {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(0x40 + offset)
    }
    fn write(&self, _cx: &AccessCtx, _o: u64, _size: AccessSize, _value: u64) -> MemResult<()> {
        Ok(())
    }
}

/// The machine, built after the plugins are loaded.
struct World {
    _sys: MemorySystem,
    as_: Arc<AddressSpace>,
    v: Vcpu,
}

fn world(a: &Asm) -> World {
    let sys = MemorySystem::new();
    let root = sys.new_container("system", 1 << 64).unwrap();
    let as_ = sys.address_space_init(root, "memory").unwrap();
    let ram = sys.new_ram("ram", 0x10_0000).unwrap();
    sys.add_subregion(root, 0, ram).unwrap();
    let io = sys.new_io("dev", 0x1000, Arc::new(Dev)).unwrap();
    sys.add_subregion(root, DEV_BASE, io).unwrap();
    a.load(&as_);
    // One thread per vCPU, so that queued work kicks the vCPU out of its loop.
    let jit = Jit::new(JitConfig { mttcg: true, ..JitConfig::default() }, backend());
    ruvm_plugin::attach(&jit, Arc::new(ToyTarget));
    let mut v = jit.create_vcpu(Arc::new(ToyOps), as_.clone(), ENV_SIZE);
    wr64(&mut v.env, PC_OFF, CODE);
    World { _sys: sys, as_, v }
}

impl World {
    /// Run until the guest stops, then run the `atexit` callbacks.
    fn run(&mut self) {
        let mut stopped = false;
        for _ in 0..100_000 {
            let mut cpu = self.v.cpu();
            cpu.process_queued_cpu_work();
            if stopped {
                break;
            }
            if cpu_exec(&mut cpu) == excp::DEBUG {
                // Run what the last block queued, then stop.
                stopped = true;
            }
        }
        assert!(stopped, "the guest did not stop");
        ruvm_plugin::qemu_plugin_atexit_cb();
    }
}

/// The program most tests run: `n` times, load a word, add one and store it back.
fn counting_loop(n: i32) -> Asm {
    let mut a = Asm::new();
    a.li(1, n).li(2, DATA as i32).li(3, 0);
    let head = a.pc();
    a.i(LD, 4, 2, 0).addi(4, 4, 1).i(ST, 4, 2, 0).addi(1, 1, -1).bnez(1, head);
    a.stop();
    a
}

// The parent side.

/// The C compiler and glib flags, or `None` when either is missing.
fn toolchain() -> Option<(String, Vec<String>)> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let ok = Command::new(&cc).arg("--version").output().is_ok_and(|o| o.status.success());
    if !ok {
        eprintln!("skipping: no C compiler ({cc})");
        return None;
    }
    let out = Command::new("pkg-config").args(["--cflags", "--libs", "glib-2.0"]).output();
    match out {
        Ok(o) if o.status.success() => {
            let flags = String::from_utf8_lossy(&o.stdout);
            Some((cc, flags.split_whitespace().map(str::to_string).collect()))
        }
        _ => {
            eprintln!("skipping: no glib-2.0 for pkg-config");
            None
        }
    }
}

fn plugin_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/plugins")
}

/// Build `src` (a file in `tests/plugins`, or C source text when `text` is given) into
/// `<tmp>/<out>.so`.
fn build(tc: &(String, Vec<String>), src: &str, text: Option<&str>, out: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("ruvm-plugin");
    std::fs::create_dir_all(&dir).unwrap();
    let src = match text {
        Some(t) => {
            let p = dir.join(format!("{out}.c"));
            std::fs::write(&p, t).unwrap();
            p
        }
        None => plugin_dir().join(src),
    };
    let so = dir.join(format!("{out}.so"));
    let mut cmd = Command::new(&tc.0);
    cmd.args(["-shared", "-fPIC", "-O1", "-Wall", "-I"]).arg(plugin_dir());
    if cfg!(target_os = "macos") {
        cmd.args(["-undefined", "dynamic_lookup"]);
    }
    cmd.arg("-o").arg(&so).arg(&src).args(&tc.1);
    let o = cmd.output().unwrap();
    assert!(
        o.status.success(),
        "building {} failed:\n{}",
        src.display(),
        String::from_utf8_lossy(&o.stderr)
    );
    so
}

const CHILD: &str = "RUVM_PLUGIN_TEST_CHILD";
const OUT: &str = "RUVM_PLUGIN_TEST_OUT";
const SPECS: &str = "RUVM_PLUGIN_TEST_SPECS";

/// The `-plugin` options to load in the child, each `path,arg,...`.
fn specs() -> Vec<String> {
    std::env::var(SPECS).unwrap().split('\n').map(str::to_string).collect()
}

/// Run test `name` in a child process. `plugins` gives, for each plugin, the source file and
/// the arguments after the path in its `-plugin` option. In the child, run `body` and return
/// `None`; in the parent, return what the plugins printed, or `None` when the test skips.
fn in_child(name: &str, plugins: &[(&str, &str)], body: impl FnOnce()) -> Option<String> {
    if std::env::var(CHILD).as_deref() == Ok(name) {
        body();
        return None;
    }
    let tc = toolchain()?;
    let mut specs = Vec::new();
    for (i, (src, args)) in plugins.iter().enumerate() {
        let so = build(&tc, src, None, &format!("{name}-{i}"));
        let mut s = so.display().to_string();
        if !args.is_empty() {
            s.push(',');
            s.push_str(args);
        }
        specs.push(s);
    }
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("ruvm-plugin/{name}.out"));
    let _ = std::fs::remove_file(&out);
    let o = Command::new(std::env::current_exe().unwrap())
        .args([name, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD, name)
        .env(OUT, &out)
        .env(SPECS, specs.join("\n"))
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "the child failed:\n{}\n{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    Some(std::fs::read_to_string(&out).unwrap_or_default())
}

// The child side.

/// Parse the `-plugin` options, load the plugins and capture what they print.
fn load(specs: &[String]) -> Arc<Mutex<String>> {
    let log = Arc::new(Mutex::new(String::new()));
    let l = log.clone();
    ruvm_plugin::set_log(Some(Arc::new(move |s: &str| l.lock().unwrap().push_str(s))));
    let mut head: Vec<PluginDesc> = Vec::new();
    for s in specs {
        if let Err(e) = ruvm_plugin::qemu_plugin_opt_parse(s, &mut head) {
            panic!("bad -plugin {s}: {e:?}");
        }
    }
    let info = QemuInfo {
        target_name: "x86_64".to_string(),
        system_emulation: true,
        smp_vcpus: 1,
        max_vcpus: 1,
    };
    if let Err(e) = ruvm_plugin::qemu_plugin_load_list(&mut head, &info) {
        panic!("{}", e.message());
    }
    log
}

/// Load the plugins, run `a` and write what the plugins printed for the parent.
fn run_guest(a: &Asm) -> World {
    let log = load(&specs());
    let mut w = world(a);
    w.run();
    std::fs::write(std::env::var(OUT).unwrap(), log.lock().unwrap().as_str()).unwrap();
    w
}

// With 100 iterations, the loop runs the first block (8 instructions) once, the loop block (5)
// 99 times and the block with STOP (1) once: 101 blocks, 504 instructions and 200 accesses.

#[test]
fn empty() {
    let Some(out) = in_child("empty", &[("empty.c", "")], || {
        let w = run_guest(&counting_loop(100));
        assert_eq!(reg(&w.v, 4), 100);
        assert_eq!(ruvm_plugin::loaded(), 1);
    }) else {
        return;
    };
    assert_eq!(out, "");
}

#[test]
fn bb() {
    let Some(out) = in_child("bb", &[("bb.c", "")], || {
        run_guest(&counting_loop(100));
    }) else {
        return;
    };
    assert_eq!(out, "CPU0: bb's: 101, insns: 504\nTotal: bb's: 101, insns: 504\n");
}

#[test]
fn bb_inline() {
    let Some(out) = in_child("bb_inline", &[("bb.c", "inline=on")], || {
        run_guest(&counting_loop(100));
    }) else {
        return;
    };
    assert_eq!(out, "CPU0: bb's: 101, insns: 504\nTotal: bb's: 101, insns: 504\n");
}

#[test]
fn insn() {
    // insn.c reads every register in its vcpu_init callback and asserts that works.
    let Some(out) = in_child("insn", &[("insn.c", "match=ld")], || {
        run_guest(&counting_loop(100));
    }) else {
        return;
    };
    // The match records are keyed by a string insn.c frees, so how many there are is up to
    // the allocator; the match count is not.
    assert!(out.starts_with("Created record for: 100c ld r4, 0(r2)\n"), "{out}");
    assert!(out.contains("cpu 0 insns: 504\ntotal insns: 504\nMatch: ld, hits 100\n"), "{out}");
}

#[test]
fn insn_inline_and_sizes() {
    let plugins = [("insn.c", "inline=true"), ("insn.c", "sizes=true")];
    let Some(out) = in_child("insn_inline_and_sizes", &plugins, || {
        run_guest(&counting_loop(100));
    }) else {
        return;
    };
    // Sizes are counted at translation: the three blocks have 14 instructions.
    // Callbacks are kept newest first, as QEMU does, so the second plugin reports first.
    assert_eq!(out, "len 4 bytes: 14 insns\ncpu 0 insns: 504\ntotal insns: 504\n");
}

#[test]
fn mem() {
    let Some(out) = in_child("mem", &[("mem.c", "inline=true")], || {
        run_guest(&counting_loop(100));
    }) else {
        return;
    };
    assert_eq!(out, "mem accesses: 200\n");
}

#[test]
fn mem_haddr_and_regions() {
    let args = "callback=true,haddr=true,region-summary=true";
    let Some(out) = in_child("mem_haddr_and_regions", &[("mem.c", args)], || {
        let mut a = counting_loop(3);
        a.words.pop();
        a.li(6, 0x10).i(LIH, 6, 0, 0).i(LD, 5, 6, 8).stop();
        let w = run_guest(&a);
        assert_eq!(reg(&w.v, 5), 0x48);
    }) else {
        return;
    };
    assert!(out.contains("mem accesses: 6\nio accesses: 1\n"), "{out}");
    assert!(out.contains("0x0000000000004000, 3, 3, true\n"), "{out}");
    // mem.c does not know what the device returns.
    assert!(out.contains("Warning: 0x0000000000100000:8 read an un-instrumented value\n"));
    assert!(out.contains("0x0000000000100000, 1, 0, false\n"), "{out}");
}

#[test]
fn mem_print_accesses() {
    let Some(out) = in_child("mem_print_accesses", &[("mem.c", "print-accesses=true")], || {
        run_guest(&counting_loop(2));
    }) else {
        return;
    };
    let want = "insn_vaddr,insn_symbol,mem_vaddr,mem_hwaddr,access_size,access_type,mem_value\n\
                0x100c,,0x4000,0x4000,32,load,0x00000000\n\
                0x1014,,0x4000,0x4000,32,store,0x00000001\n\
                0x100c,,0x4000,0x4000,32,load,0x00000001\n\
                0x1014,,0x4000,0x4000,32,store,0x00000002\n";
    assert_eq!(out, want);
}

#[test]
fn inline() {
    let Some(out) = in_child("inline", &[("inline.c", "")], || {
        run_guest(&counting_loop(100));
    }) else {
        return;
    };
    // inline.c checks that callbacks, inline operations and conditional callbacks agree.
    let want =
        "cpu 0: tb (101, 101, 1 * 100 + 1) | insn (504, 504, 5 * 100 + 4) | mem (200, 200)\n";
    assert!(out.starts_with(want), "{out}");
    assert!(out.contains("tb: 101\ntb: 101 (per vcpu)\n"), "{out}");
    assert!(out.contains("insn: 504 (cond cb)\n"), "{out}");
    assert!(out.contains("mem: 200 (per vcpu inline)\n"), "{out}");
}

#[test]
fn inline_conditions_trigger() {
    let Some(out) = in_child("inline_conditions_trigger", &[("inline.c", "")], || {
        run_guest(&counting_loop(1100));
    }) else {
        return;
    };
    // 1101 blocks and 5504 instructions: the conditional callbacks fired 11 and 55 times.
    let want = "cpu 0: tb (1101, 1101, 11 * 100 + 1) | insn (5504, 5504, 55 * 100 + 4) |";
    assert!(out.starts_with(want), "{out}");
}

#[test]
fn hotblocks() {
    let Some(out) = in_child("hotblocks", &[("hotblocks.c", "")], || {
        run_guest(&counting_loop(100));
    }) else {
        return;
    };
    assert!(out.starts_with("collected 3 entries in the hash table\npc, tcount, icount, ecount\n"));
    assert!(out.contains("0x000000000000100c, 1, 5, 99\n"), "{out}");
    assert!(out.contains("0x0000000000001000, 1, 8, 1\n"), "{out}");
    assert!(out.contains("0x0000000000001020, 1, 1, 1\n"), "{out}");
}

#[test]
fn hotblocks_inline() {
    let Some(out) = in_child("hotblocks_inline", &[("hotblocks.c", "inline=on,limit=1")], || {
        run_guest(&counting_loop(100));
    }) else {
        return;
    };
    assert_eq!(
        out,
        "collected 3 entries in the hash table\npc, tcount, icount, ecount\n\
         0x000000000000100c, 1, 5, 99\n"
    );
}

/// write(1, "hello", 5), the magic call of syscall.c and syscall 7.
fn syscalls() -> Asm {
    let mut a = Asm::new();
    a.li(0, 1).li(1, 1).li(2, 0x4100).li(3, 5).swi();
    a.li(0, 4096).li32(1, 0x66CCFF).swi().addi(6, 0, 0);
    a.li(0, 7).swi();
    a.stop();
    a
}

#[test]
fn syscall() {
    let plugins = [("syscall.c", "print=on,log_writes=on")];
    let Some(out) = in_child("syscall", &plugins, || {
        let log = load(&specs());
        let mut w = world(&syscalls());
        assert!(w.as_.write(0x4100, U, b"hello").is_ok());
        w.run();
        std::fs::write(std::env::var(OUT).unwrap(), log.lock().unwrap().as_str()).unwrap();
        assert_eq!(reg(&w.v, 6), 0xFFCC66);
        assert_eq!(reg(&w.v, 0), 107);
    }) else {
        return;
    };
    let pad = "   ".repeat(11);
    let want = format!(
        "syscall #1\n68 65 6c 6c 6f {pad} | hello\nsyscall #1 returned -> 101\n\
         syscall #4096\nmagic syscall filtered, set magic return\n\
         syscall #4096 returned -> 16764006\nsyscall #7\nsyscall #7 returned -> 107\n"
    );
    assert_eq!(out, want);
}

#[test]
fn syscall_statistics() {
    let Some(out) = in_child("syscall_statistics", &[("syscall.c", "")], || {
        run_guest(&syscalls());
    }) else {
        return;
    };
    let mut lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.remove(0), "syscall no.  calls  errors");
    lines.sort_unstable();
    assert_eq!(lines, ["1            1      0", "4096         1      0", "7            1      0"]);
}

#[test]
fn reset_and_uninstall() {
    let Some(out) = in_child("reset_and_uninstall", &[("reset.c", "")], || {
        let w = run_guest(&counting_loop(100));
        assert_eq!(reg(&w.v, 4), 100);
        assert_eq!(ruvm_plugin::loaded(), 0);
    }) else {
        return;
    };
    // The destructor of reset.c prints "plugin exit" if the host unloads it on dlclose().
    assert!(out.starts_with("reset done\nuninstall done\n"), "{out}");
}

#[test]
fn setpc() {
    let Some(out) = in_child("setpc", &[("setpc.c", "")], || {
        // r6 counts the stages that were reached. A marker the plugin should jump over sets
        // r5 to a bad value.
        let mut a = Asm::new();
        // Stage 1: arm the instruction marker at `src`, which jumps to `t1`.
        let src = CODE + 4 * 7;
        let t1 = src + 8;
        a.li(0, 4096).li(1, 1).li(2, src as i32).li(3, t1 as i32).li(4, 0).swi().li(6, 1);
        assert_eq!(a.pc(), src);
        a.li(5, 99).stop();
        assert_eq!(a.pc(), t1);
        // Stage 2: arm the memory marker on DATA + 8, which jumps to `t2`.
        let t2 = t1 + 4 * 11;
        a.li(6, 2).li(0, 4096).li(1, 1).li(2, 0).li(3, t2 as i32);
        a.li(4, (DATA + 8) as i32).swi().i(LD, 7, 4, 0).li(5, 98).stop();
        a.li(6, 99);
        assert_eq!(a.pc(), t2);
        // Stage 3: jump straight to `done`.
        let done = t2 + 4 * 6;
        a.li(6, 3).li(0, 4096).li(1, 0).li(2, done as i32).swi().stop();
        assert_eq!(a.pc(), done);
        a.li(6, 4).stop();
        let w = run_guest(&a);
        assert_eq!(reg(&w.v, 6), 4);
        assert_eq!(reg(&w.v, 5), 0);
    }) else {
        return;
    };
    assert_eq!(
        out,
        "Magic syscall detected, set target_pc / target_vaddr\n\
         Marker insn detected, jump to clean return\n\
         Magic syscall detected, set target_pc / target_vaddr\n\
         Marker mem access detected, jump to clean return\n\
         Magic syscall detected, jump to clean exit\n"
    );
}

#[test]
fn load_errors() {
    let Some(tc) = toolchain() else { return };
    let none = build(&tc, "", Some("int qemu_plugin_install(void) { return 0; }\n"), "nover");
    let newer = build(
        &tc,
        "",
        Some(
            "int qemu_plugin_version = 99;\n\
             int qemu_plugin_install(void) { return 0; }\n",
        ),
        "newer",
    );
    let syscall = build(&tc, "syscall.c", None, "load_errors-syscall");
    let specs = [
        format!("{},bogus=1", syscall.display()),
        none.display().to_string(),
        newer.display().to_string(),
        "/nonexistent/plugin.so".to_string(),
    ]
    .join("\n");
    let o = Command::new(std::env::current_exe().unwrap())
        .args(["load_errors_child", "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD, "load_errors_child")
        .env(SPECS, specs)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("unsupported argument: bogus=1\n"), "{err}");
}

#[test]
fn load_errors_child() {
    if std::env::var(CHILD).as_deref() != Ok("load_errors_child") {
        return;
    }
    let info = QemuInfo { target_name: "x86_64".to_string(), ..QemuInfo::default() };
    let specs = specs();
    let msg = |spec: &String| {
        let mut head = Vec::new();
        ruvm_plugin::qemu_plugin_opt_parse(spec, &mut head).unwrap();
        let e = ruvm_plugin::qemu_plugin_load_list(&mut head, &info).unwrap_err();
        assert_eq!(head.len(), 1);
        e.message().to_string()
    };
    let path = |spec: &String| spec.split(',').next().unwrap().to_string();
    assert_eq!(
        msg(&specs[0]),
        format!(
            "Could not load plugin {}: qemu_plugin_install returned error code -1",
            path(&specs[0])
        )
    );
    let m = msg(&specs[1]);
    let want = format!(
        "Could not load plugin {}: plugin does not declare API version 'qemu_plugin_version': ",
        specs[1]
    );
    assert!(m.starts_with(&want), "{m}");
    assert_eq!(
        msg(&specs[2]),
        format!(
            "Could not load plugin {}: plugin requires API version 99, but this QEMU supports \
             only up to version 7",
            specs[2]
        )
    );
    assert!(msg(&specs[3]).starts_with("Could not load plugin /nonexistent/plugin.so: "));
    assert_eq!(ruvm_plugin::loaded(), 0);
}

#[test]
fn version_is_seven() {
    assert_eq!(ruvm_plugin::QEMU_PLUGIN_VERSION, 7);
    let header = std::fs::read_to_string(plugin_dir().join("qemu-plugin.h")).unwrap();
    assert!(header.contains("#define QEMU_PLUGIN_VERSION 7\n"));
    let n = header.lines().filter(|l| *l == "QEMU_PLUGIN_API").count();
    assert_eq!(n, 65, "the header declares {n} API functions");
}
