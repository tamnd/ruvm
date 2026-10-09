// SPDX-License-Identifier: GPL-2.0-or-later

//! The ELF loader, `linux-user/elfload.c`: `load_elf_image()`, `load_elf_binary()`,
//! `setup_arg_pages()`, `copy_elf_strings()` and `create_elf_tables()` for a 64-bit little
//! endian guest.

use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;

use ruvm_user_common::{GuestSpace, MapKind, PAGE_SIZE, page, page_align};

/// `PT_LOAD`.
const PT_LOAD: u32 = 1;
/// `PT_INTERP`.
const PT_INTERP: u32 = 3;
/// `PT_GNU_STACK`.
const PT_GNU_STACK: u32 = 0x6474_e551;
/// `ET_EXEC`.
const ET_EXEC: u16 = 2;
/// `ET_DYN`.
const ET_DYN: u16 = 3;
/// `PF_X`.
const PF_X: u32 = 1;
/// `PF_W`.
const PF_W: u32 = 2;
/// `PF_R`.
const PF_R: u32 = 4;
/// `sizeof(struct elf64_phdr)`.
const PHENT: u64 = 56;
/// `STACK_LOWER_LIMIT`: the smallest stack, 32 pages.
const STACK_LOWER_LIMIT: u64 = 32 * PAGE_SIZE;

/// What a target tells the loader, `ELF_MACHINE` and friends.
#[derive(Clone, Copy, Debug)]
pub struct Arch {
    /// `ELF_MACHINE`.
    pub machine: u16,
    /// `ELF_PLATFORM`, if the target has one.
    pub platform: Option<&'static str>,
    /// `ELF_HWCAP`.
    pub hwcap: u64,
}

/// The parts of `struct image_info` the loader fills in.
#[derive(Clone, Debug, Default)]
pub struct ImageInfo {
    /// `load_bias`.
    pub load_bias: u64,
    /// `load_addr`.
    pub load_addr: u64,
    /// `entry`.
    pub entry: u64,
    /// `phdr_addr`, for `AT_PHDR`.
    pub phdr_addr: u64,
    /// `e_phnum`.
    pub phnum: u64,
    /// `start_code`.
    pub start_code: u64,
    /// `end_code`.
    pub end_code: u64,
    /// `start_data`.
    pub start_data: u64,
    /// `end_data`.
    pub end_data: u64,
    /// `brk`.
    pub brk: u64,
    /// `exec_stack`.
    pub exec_stack: bool,
    /// `stack_limit`.
    pub stack_limit: u64,
    /// `start_stack`, the stack pointer the program starts with.
    pub start_stack: u64,
    /// `arg_strings`.
    pub arg_start: u64,
    /// `env_strings`.
    pub env_start: u64,
    /// `file_string`, for `AT_EXECFN`.
    pub file_string: u64,
    /// `saved_auxv`.
    pub saved_auxv: u64,
    /// `auxv_len`.
    pub auxv_len: u64,
}

/// The ELF header fields the loader uses.
struct Ehdr {
    e_type: u16,
    e_entry: u64,
    e_phoff: u64,
    e_phnum: u16,
}

#[derive(Clone, Copy)]
struct Phdr {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(v)
}

const INVALID: &str = "Invalid ELF image for this architecture";

/// Whether `head` starts with the ELF magic, the check `loader_exec()` makes before anything
/// else.
pub fn is_elf(head: &[u8]) -> bool {
    head.starts_with(b"\x7fELF")
}

fn read_exact_at(f: &File, buf: &mut [u8], off: u64) -> Result<(), String> {
    f.read_exact_at(buf, off).map_err(|e| match e.kind() {
        std::io::ErrorKind::UnexpectedEof => "File too short".to_string(),
        _ => format!("Error reading file: {}", crate::strerror_of(&e)),
    })
}

/// `elf_check_ident()` and `elf_check_ehdr()`, then the header.
fn read_ehdr(f: &File, arch: &Arch) -> Result<Ehdr, String> {
    let mut b = [0u8; 64];
    read_exact_at(f, &mut b, 0)?;
    // ELFCLASS64, ELFDATA2LSB, EV_CURRENT.
    if !is_elf(&b) || b[4] != 2 || b[5] != 1 || b[6] != 1 {
        return Err(INVALID.into());
    }
    let e_type = u16_at(&b, 16);
    let machine = u16_at(&b, 18);
    let phentsize = u16_at(&b, 54);
    if machine != arch.machine
        || phentsize as u64 != PHENT
        || (e_type != ET_EXEC && e_type != ET_DYN)
    {
        return Err(INVALID.into());
    }
    Ok(Ehdr { e_type, e_entry: u64_at(&b, 24), e_phoff: u64_at(&b, 32), e_phnum: u16_at(&b, 56) })
}

fn read_phdrs(f: &File, eh: &Ehdr) -> Result<Vec<Phdr>, String> {
    let mut b = vec![0u8; eh.e_phnum as usize * PHENT as usize];
    read_exact_at(f, &mut b, eh.e_phoff)?;
    Ok(b.chunks_exact(PHENT as usize)
        .map(|p| Phdr {
            p_type: u32_at(p, 0),
            p_flags: u32_at(p, 4),
            p_offset: u64_at(p, 8),
            p_vaddr: u64_at(p, 16),
            p_filesz: u64_at(p, 32),
            p_memsz: u64_at(p, 40),
            p_align: u64_at(p, 48),
        })
        .collect())
}

fn errno_msg(e: i32) -> String {
    crate::strerror(e)
}

const PRIVATE_ANON: MapKind = MapKind { fixed: false, noreplace: false, shared: false, anon: true };

/// `zero_bss()`: zero the end of the page the file data ends in and map anonymous pages for
/// the rest of the segment.
fn zero_bss(space: &GuestSpace, start_bss: u64, end_bss: u64, prot: u32) -> Result<(), String> {
    if prot & page::WRITE == 0 {
        return Err("PT_LOAD with non-writable bss".into());
    }
    let mut align_bss = page_align(start_bss).ok_or(INVALID)?;
    let end_bss = page_align(end_bss).ok_or(INVALID)?;
    if start_bss < align_bss {
        let flags = space.page_flags(start_bss);
        if flags & page::RWX == 0 {
            align_bss -= PAGE_SIZE;
        } else {
            if flags & page::WRITE == 0 {
                return Err("PT_LOAD with bss overlapping non-writable page".into());
            }
            let zeros = vec![0u8; (align_bss - start_bss) as usize];
            space.write_raw(start_bss, &zeros);
        }
    }
    if align_bss < end_bss {
        let kind = MapKind { fixed: true, ..PRIVATE_ANON };
        space.mmap(align_bss, end_bss - align_bss, prot, kind, None, 0).map_err(errno_msg)?;
    }
    Ok(())
}

/// `load_elf_image()`. `interp` is `Some` for the main program, to receive `PT_INTERP`.
fn load_elf_image(
    space: &GuestSpace,
    arch: &Arch,
    f: &File,
    info: &mut ImageInfo,
    interp: Option<&mut Option<String>>,
) -> Result<(), String> {
    let eh = read_ehdr(f, arch)?;
    let phdrs = read_phdrs(f, &eh)?;
    let mut lo = u64::MAX;
    let mut hi = 0u64;
    let mut align = 0u64;
    info.exec_stack = true;
    let is_main = interp.is_some();
    let mut interp_name = None;
    for p in &phdrs {
        match p.p_type {
            PT_LOAD => {
                lo = lo.min(p.p_vaddr & !(PAGE_SIZE - 1));
                hi = hi.max(p.p_vaddr.wrapping_add(p.p_memsz).wrapping_sub(1));
                align |= p.p_align;
            }
            PT_INTERP if is_main => {
                if interp_name.is_some() {
                    return Err("Multiple PT_INTERP entries".into());
                }
                let mut b = vec![0u8; p.p_filesz as usize];
                read_exact_at(f, &mut b, p.p_offset)?;
                if b.last() != Some(&0) {
                    return Err("Invalid PT_INTERP entry".into());
                }
                b.pop();
                interp_name = Some(String::from_utf8_lossy(&b).into_owned());
            }
            PT_GNU_STACK => info.exec_stack = p.p_flags & PF_X != 0,
            _ => {}
        }
    }
    if lo > hi {
        return Err(INVALID.into());
    }
    let mut load_addr = lo;
    let align = align.checked_next_power_of_two().unwrap_or(0);
    if is_main && eh.e_type == ET_DYN {
        load_addr = load_addr.wrapping_add(space.et_dyn_base());
        if align != 0 {
            load_addr &= align.wrapping_neg();
        }
    }
    let reserve_size = hi - lo + 1;
    let mut align_size = reserve_size;
    if eh.e_type != ET_EXEC && align > PAGE_SIZE {
        align_size += align - 1;
    }
    let kind = MapKind { noreplace: eh.e_type == ET_EXEC, ..PRIVATE_ANON };
    let mut load_addr = space.mmap(load_addr, align_size, 0, kind, None, 0).map_err(errno_msg)?;
    if align_size != reserve_size {
        let align_addr = load_addr.next_multiple_of(align);
        let align_end = page_align(align_addr + reserve_size).ok_or(INVALID)?;
        let load_end = page_align(load_addr + align_size).ok_or(INVALID)?;
        if align_addr != load_addr {
            let _ = space.munmap(load_addr, align_addr - load_addr);
        }
        if align_end != load_end {
            let _ = space.munmap(align_end, load_end - align_end);
        }
        load_addr = align_addr;
    }
    let load_bias = load_addr.wrapping_sub(lo);
    info.load_bias = load_bias;
    info.load_addr = load_addr;
    info.entry = eh.e_entry.wrapping_add(load_bias);
    info.phdr_addr = load_addr.wrapping_add(eh.e_phoff);
    info.phnum = eh.e_phnum.into();
    info.start_code = u64::MAX;
    info.end_code = 0;
    info.start_data = u64::MAX;
    info.end_data = 0;
    info.brk = page_align(hi.wrapping_add(load_bias)).ok_or(INVALID)?;
    for p in phdrs.iter().filter(|p| p.p_type == PT_LOAD) {
        let mut prot = 0;
        if p.p_flags & PF_R != 0 {
            prot |= page::READ;
        }
        if p.p_flags & PF_W != 0 {
            prot |= page::WRITE;
        }
        if p.p_flags & PF_X != 0 {
            prot |= page::EXEC;
        }
        let vaddr = load_bias.wrapping_add(p.p_vaddr);
        let vaddr_po = vaddr & (PAGE_SIZE - 1);
        let vaddr_ps = vaddr & !(PAGE_SIZE - 1);
        let vaddr_ef = vaddr.wrapping_add(p.p_filesz);
        let vaddr_em = vaddr.wrapping_add(p.p_memsz);
        if p.p_offset <= eh.e_phoff && eh.e_phoff < p.p_offset + p.p_filesz {
            info.phdr_addr = vaddr + (eh.e_phoff - p.p_offset);
        }
        if p.p_filesz != 0 {
            let kind = MapKind { fixed: true, ..MapKind::default() };
            let off = p.p_offset.checked_sub(vaddr_po).ok_or(INVALID)?;
            space
                .mmap(vaddr_ps, p.p_filesz + vaddr_po, prot, kind, Some(f.as_raw_fd()), off)
                .map_err(errno_msg)?;
        }
        if vaddr_ef < vaddr_em {
            zero_bss(space, vaddr_ef, vaddr_em, prot)?;
        }
        if prot & page::EXEC != 0 {
            info.start_code = info.start_code.min(vaddr);
            info.end_code = info.end_code.max(vaddr_ef);
        }
        if prot & page::WRITE != 0 {
            info.start_data = info.start_data.min(vaddr);
            info.end_data = info.end_data.max(vaddr_ef);
        }
    }
    if info.end_data == 0 {
        info.start_data = info.end_code;
        info.end_data = info.end_code;
    }
    if let Some(out) = interp {
        *out = interp_name;
    }
    Ok(())
}

/// `setup_arg_pages()`: maps the stack and returns the first free word at its top.
fn setup_arg_pages(
    space: &GuestSpace,
    stack_size: u64,
    info: &mut ImageInfo,
) -> Result<u64, String> {
    let size = stack_size.max(STACK_LOWER_LIMIT);
    let guard = PAGE_SIZE;
    let mut prot = page::READ | page::WRITE;
    if info.exec_stack {
        prot |= page::EXEC;
    }
    let base = space
        .mmap(0, size + guard, prot, PRIVATE_ANON, None, 0)
        .map_err(|e| format!("mmap stack: {}", crate::strerror(e)))?;
    let _ = space.mprotect(base, guard, 0);
    info.stack_limit = base + guard;
    Ok(info.stack_limit + size - 8)
}

/// `copy_elf_strings()`: puts `strings` just below `p`, the first lowest, and returns the new
/// bottom, or `None` when they do not fit above `limit`.
fn copy_elf_strings(space: &GuestSpace, strings: &[Vec<u8>], p: u64, limit: u64) -> Option<u64> {
    let mut buf = Vec::new();
    for s in strings {
        buf.extend_from_slice(s);
        buf.push(0);
    }
    let len = buf.len() as u64;
    if len > p - limit {
        return None;
    }
    let at = p - len;
    space.write_raw(at, &buf);
    Some(at)
}

fn put_u64(space: &GuestSpace, addr: u64, v: u64) {
    space.write_raw(addr, &v.to_le_bytes());
}

/// The program's credentials and random bytes, what `create_elf_tables()` asks the host for.
#[derive(Clone, Copy, Debug)]
pub struct Creds {
    /// `getuid()`, `geteuid()`, `getgid()`, `getegid()`.
    pub ids: [u64; 4],
    /// `sysconf(_SC_CLK_TCK)`.
    pub clktck: u64,
    /// `getauxval(AT_SECURE)`.
    pub secure: u64,
    /// `AT_RANDOM`.
    pub random: [u8; 16],
}

/// `create_elf_tables()`: argc, argv, envp and the auxiliary vector below the strings.
#[allow(clippy::too_many_arguments)]
fn create_elf_tables(
    space: &GuestSpace,
    arch: &Arch,
    p: u64,
    argv: &[Vec<u8>],
    envp: &[Vec<u8>],
    info: &mut ImageInfo,
    interp: Option<&ImageInfo>,
    creds: &Creds,
) -> u64 {
    const N: u64 = 8;
    let argc = argv.len() as u64;
    let envc = envp.len() as u64;
    let mut sp = p;
    let mut u_platform = 0;
    if let Some(plat) = arch.platform {
        let len = plat.len() as u64 + 1;
        sp -= (len + N - 1) & !(N - 1);
        u_platform = sp;
        let mut b = plat.as_bytes().to_vec();
        b.push(0);
        space.write_raw(sp, &b);
    }
    sp &= !15;
    sp -= 16;
    let u_rand_bytes = sp;
    space.write_raw(sp, &creds.random);

    // DLINFO_ITEMS is 16, plus AT_NULL.
    let mut size = (16 + 1) * 2;
    if arch.platform.is_some() {
        size += 2;
    }
    info.auxv_len = size * N;
    size += envc + argc + 2;
    size += 1;
    size *= N;
    let u_argc = (sp - size) & !15;
    let u_argv = u_argc + N;
    let u_envp = u_argv + (argc + 1) * N;
    let mut u_auxv = u_envp + (envc + 1) * N;
    info.saved_auxv = u_auxv;
    let mut aux = |id: u64, v: u64| {
        put_u64(space, u_auxv, id);
        put_u64(space, u_auxv + N, v);
        u_auxv += 2 * N;
    };
    aux(3, info.phdr_addr); // AT_PHDR
    aux(4, PHENT); // AT_PHENT
    aux(5, info.phnum); // AT_PHNUM
    aux(6, PAGE_SIZE); // AT_PAGESZ
    aux(7, interp.map_or(0, |i| i.load_addr)); // AT_BASE
    aux(8, 0); // AT_FLAGS
    aux(9, info.entry); // AT_ENTRY
    aux(11, creds.ids[0]); // AT_UID
    aux(12, creds.ids[1]); // AT_EUID
    aux(13, creds.ids[2]); // AT_GID
    aux(14, creds.ids[3]); // AT_EGID
    aux(16, arch.hwcap); // AT_HWCAP
    aux(17, creds.clktck); // AT_CLKTCK
    aux(25, u_rand_bytes); // AT_RANDOM
    aux(23, creds.secure); // AT_SECURE
    aux(31, info.file_string); // AT_EXECFN
    if u_platform != 0 {
        aux(15, u_platform); // AT_PLATFORM
    }
    aux(0, 0); // AT_NULL

    put_u64(space, u_argc, argc);
    let strings = |mut at: u64, mut s: u64, list: &[Vec<u8>]| {
        for item in list {
            put_u64(space, at, s);
            at += N;
            s += item.len() as u64 + 1;
        }
        put_u64(space, at, 0);
    };
    strings(u_argv, info.arg_start, argv);
    strings(u_envp, info.env_start, envp);
    u_argc
}

/// Where `path()` finds `name` under the `-L` prefix: there if it exists, else `name`.
pub fn path(prefix: &str, name: &str) -> String {
    if name.starts_with('/') && !prefix.is_empty() {
        let p = format!("{}{}", prefix.trim_end_matches('/'), name);
        if std::fs::symlink_metadata(&p).is_ok() {
            return p;
        }
    }
    name.to_string()
}

/// What `load_elf_binary()` needs besides the guest space.
#[derive(Debug)]
pub struct Exec<'a> {
    /// The target.
    pub arch: Arch,
    /// `bprm->filename`.
    pub filename: &'a str,
    /// The opened program.
    pub file: &'a File,
    /// `argv`.
    pub argv: &'a [Vec<u8>],
    /// `envp`.
    pub envp: &'a [Vec<u8>],
    /// `guest_stack_size`.
    pub stack_size: u64,
    /// `interp_prefix`.
    pub ld_prefix: &'a str,
    /// The host side of the auxiliary vector.
    pub creds: Creds,
}

/// `load_elf_binary()`. Errors are complete messages.
pub fn load_elf_binary(space: &GuestSpace, e: &Exec<'_>) -> Result<ImageInfo, String> {
    let mut info = ImageInfo::default();
    let mut interp_name = None;
    load_elf_image(space, &e.arch, e.file, &mut info, Some(&mut interp_name))
        .map_err(|m| format!("{}: {m}", e.filename))?;
    let mut p = setup_arg_pages(space, e.stack_size, &mut info)?;
    let filename = vec![e.filename.as_bytes().to_vec()];
    let limit = info.stack_limit;
    let e2big = || format!("{}: {}", e.filename, crate::strerror(libc::E2BIG));
    p = copy_elf_strings(space, &filename, p, limit).ok_or_else(e2big)?;
    info.file_string = p;
    p = copy_elf_strings(space, e.envp, p, limit).ok_or_else(e2big)?;
    info.env_start = p;
    p = copy_elf_strings(space, e.argv, p, limit).ok_or_else(e2big)?;
    info.arg_start = p;

    let mut interp_info = None;
    if let Some(name) = &interp_name {
        let real = path(e.ld_prefix, name);
        let f = File::open(&real)
            .map_err(|err| format!("Could not open '{name}': {}", crate::strerror_of(&err)))?;
        let mut ii = ImageInfo::default();
        load_elf_image(space, &e.arch, &f, &mut ii, None).map_err(|m| format!("{name}: {m}"))?;
        if ii.brk > info.brk && ii.load_bias.wrapping_sub(info.brk) < 16 << 20 {
            info.brk = ii.brk;
        }
        interp_info = Some(ii);
    }

    let sp = create_elf_tables(
        space,
        &e.arch,
        p,
        e.argv,
        e.envp,
        &mut info,
        interp_info.as_ref(),
        &e.creds,
    );
    info.start_stack = sp;
    if let Some(ii) = interp_info {
        info.load_bias = ii.load_bias;
        info.entry = ii.entry;
    }
    Ok(info)
}
