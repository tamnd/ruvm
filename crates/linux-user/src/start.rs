// SPDX-License-Identifier: GPL-2.0-or-later

//! `main()` of `linux-user/main.c`, which is the same for every target: the options, the
//! guest address space, the ELF loader, the translator and the first vCPU, then `cpu_loop()`.

use std::fs::File;
use std::process::ExitCode;
use std::sync::Arc;

use ruvm_jit::{Jit, JitConfig, Vcpu, cf};
use ruvm_mem::{AddressSpace, MemorySystem};
use ruvm_user_common::GuestSpace;

use crate::elf::{self, Arch, Creds, Exec, ImageInfo};
use crate::guest::{self, Guest};
use crate::host;
use crate::opts::{self, Exit};
use crate::procfs::Image;
use crate::signal::{self, Task};
use crate::syscall::Proc;

/// What `main()` needs of a target besides its [`Guest`].
pub(crate) trait Target {
    /// The target.
    fn guest(&self) -> &'static Guest;
    /// The name `-cpu help` and the usage text give, `TARGET_NAME`.
    fn name(&self) -> &'static str;
    /// `TASK_UNMAPPED_BASE` and `ELF_ET_DYN_BASE`.
    fn layout(&self) -> (u64, u64);
    /// `cpu_create()` of the model `-cpu` names, with its features.
    fn select_cpu(&mut self, cpu: &str) -> Result<(), String>;
    /// `ELF_MACHINE`, `ELF_PLATFORM` and the hardware capabilities of the model.
    fn arch(&self) -> Arch;
    /// The translator, its configuration as `config` leaves it.
    fn new_jit(&self, config: &dyn Fn(&mut JitConfig)) -> Arc<Jit>;
    /// The first vCPU, with the registers `target_cpu_copy_regs()` gives it.
    fn create_vcpu(
        &mut self,
        jit: &Arc<Jit>,
        space: &Arc<GuestSpace>,
        as_: Arc<AddressSpace>,
        info: &ImageInfo,
    ) -> Result<Vcpu, String>;
}

/// The host side of the auxiliary vector.
fn creds() -> Creds {
    let id = |nr| host::sys(nr, &[]) as u64;
    Creds {
        ids: [
            id(libc::SYS_getuid),
            id(libc::SYS_geteuid),
            id(libc::SYS_getgid),
            id(libc::SYS_getegid),
        ],
        clktck: 100,
        secure: 0,
        random: host::random16(),
    }
}

/// `prepare_binprm()` and the format check of `loader_exec()`: whether `file` is a regular,
/// executable file starting with the ELF magic. Every failure is `ENOEXEC` to the user.
fn is_exec(file: &File) -> bool {
    use std::os::unix::fs::{FileExt, PermissionsExt};
    let Ok(m) = file.metadata() else { return false };
    if !m.is_file() || m.permissions().mode() & 0o111 == 0 {
        return false;
    }
    let mut head = [0u8; 4];
    matches!(file.read_at(&mut head, 0), Ok(4)) && elf::is_elf(&head)
}

/// `main()` of `qemu-<target>`.
pub(crate) fn main(argv0: &str, args: &[String], target: &mut dyn Target) -> ExitCode {
    let g = target.guest();
    guest::set(g);
    // Signals stay blocked until the guest runs, here and in any thread started meanwhile.
    let host_mask = host::set_mask(!0);
    let prog = argv0.rsplit('/').next().unwrap_or(argv0);
    let env: Vec<(String, String)> = std::env::vars_os()
        .map(|(k, v)| (k.to_string_lossy().into_owned(), v.to_string_lossy().into_owned()))
        .collect();
    let mut stack_size = opts::DEFAULT_STACK_SIZE;
    if let Some(cur) = host::stack_rlimit() {
        stack_size = stack_size.max(cur);
    }
    let mut o = match opts::parse(target.name(), env, args) {
        Ok(o) => o,
        Err(Exit::Usage(text, code)) => {
            print!("{text}");
            return code;
        }
        Err(Exit::Error(msg)) => {
            eprintln!("{msg}");
            return ExitCode::FAILURE;
        }
    };
    if o.stack_size == opts::DEFAULT_STACK_SIZE {
        o.stack_size = stack_size;
    }
    host::reset_sigpipe();

    let cpu_name = o.cpu.clone().unwrap_or_else(|| "max".to_string());
    if let Err(e) = target.select_cpu(&cpu_name) {
        eprintln!("{prog}: {e}");
        return ExitCode::FAILURE;
    }

    let file = match File::open(&o.exec_path) {
        Ok(f) => f,
        Err(e) => {
            println!("Error while loading {}: {}", o.exec_path, crate::strerror_of(&e));
            return ExitCode::FAILURE;
        }
    };
    // real_exec_path, what /proc/self/exe names.
    let real_exec_path = match std::fs::canonicalize(&o.exec_path) {
        Ok(p) => p.to_string_lossy().into_owned(),
        Err(_) => {
            println!("Could not resolve {}", o.exec_path);
            o.exec_path.clone()
        }
    };
    if !is_exec(&file) {
        println!("Error while loading {}: {}", o.exec_path, crate::strerror(libc::ENOEXEC));
        return ExitCode::FAILURE;
    }

    let want =
        if o.reserved_va != 0 { o.reserved_va } else { ruvm_user_common::space::DEFAULT_RESERVE };
    let (unmapped_base, et_dyn_base) = target.layout();
    let space = match GuestSpace::new(want, unmapped_base, et_dyn_base) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("{prog}: Unable to reserve guest address space: {}", crate::strerror_of(&e));
            return ExitCode::FAILURE;
        }
    };

    let argv: Vec<Vec<u8>> = o.args.iter().map(|a| a.clone().into_bytes()).collect();
    let envp: Vec<Vec<u8>> = o.env.iter().map(|a| a.clone().into_bytes()).collect();
    let exec = Exec {
        arch: target.arch(),
        filename: &o.exec_path,
        file: &file,
        argv: &argv,
        envp: &envp,
        stack_size: o.stack_size,
        ld_prefix: &o.ld_prefix,
        creds: creds(),
    };
    let info = match elf::load_elf_binary(&space, &exec) {
        Ok(i) => i,
        Err(e) => {
            // error_reportf_err() and exit(-1).
            eprintln!("{prog}: {e}");
            return ExitCode::from(255);
        }
    };
    drop(file);

    let jit = target.new_jit(&|config| {
        config.user_only = true;
        // Every guest thread has a vCPU and a host thread; the first runs serial code until
        // there is a second.
        config.mttcg = true;
        if let Some(mb) = o.tb_size {
            if mb != 0 {
                config.code_gen_buffer_size = usize::try_from(mb << 20).unwrap_or(usize::MAX);
            }
        }
        if o.one_insn_per_tb {
            config.one_insn_per_tb = true;
        }
    });

    let ms = MemorySystem::new();
    let as_ = (|| {
        let root = ms.new_container("system", 1u128 << 64)?;
        let ram = ms.new_ram_from_block(Arc::clone(space.block()))?;
        ms.add_subregion(root, 0, ram)?;
        ms.address_space_init(root, "memory")
    })();
    let as_ = match as_ {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{prog}: {e}");
            return ExitCode::FAILURE;
        }
    };
    {
        let j = Arc::downgrade(&jit);
        let block = Arc::clone(space.block());
        space.set_code_hook(Box::new(move |s, l| {
            if let Some(j) = j.upgrade() {
                j.tb_invalidate_phys_block(&block, s, l);
            }
        }));
    }

    let mut v = match target.create_vcpu(&jit, &space, as_, &info) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    v.core.tcg_cflags &= !cf::PARALLEL;
    let image = Image {
        argv,
        stack_limit: info.stack_limit,
        brk: info.brk,
        start_stack: info.start_stack,
        auxv: (info.saved_auxv, info.auxv_len),
    };
    let proc = Arc::new(Proc::new(
        Arc::clone(&space),
        real_exec_path,
        o.uname_release.clone(),
        o.ld_prefix.clone(),
        image,
    ));
    let mut cpu = v.cpu();
    signal::signal_init(prog);
    let mut task = Task::new(host_mask, cpu.shared());
    host::set_mask(task.run_mask());
    (g.cpu_loop)(&proc, &mut task, &mut cpu);
    // The main thread called exit with others left: it ends, and the process with the last.
    task.exit_thread();
    drop(v);
    loop {
        host::sys(libc::SYS_exit, &[0]);
    }
}
