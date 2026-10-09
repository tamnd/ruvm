// SPDX-License-Identifier: GPL-2.0-or-later

//! The files of `/proc/self` that describe the guest rather than the emulator, from
//! `maybe_do_fake_open()`: `maps`, `smaps`, `stat`, `auxv` and `cmdline` are written to a
//! memfd when the guest opens them, as QEMU writes them.

use std::fmt::Write;

use ruvm_user_common::{PAGE_SIZE, page};

use crate::host::sys;
use crate::signal::Task;
use crate::syscall::{Proc, THREADS};

/// What the files say about the program, from its `image_info` and `linux_binprm`.
#[derive(Debug, Default)]
pub(crate) struct Image {
    /// `bprm->argv`.
    pub(crate) argv: Vec<Vec<u8>>,
    /// `info->stack_limit`, where `[stack]` starts.
    pub(crate) stack_limit: u64,
    /// `info->brk`, where `[heap]` starts.
    pub(crate) brk: u64,
    /// `info->start_stack`.
    pub(crate) start_stack: u64,
    /// `info->saved_auxv` and `info->auxv_len`.
    pub(crate) auxv: (u64, u64),
}

/// `is_proc_myself()`: whether `name` is `/proc/self/<entry>` or `/proc/<pid>/<entry>` of
/// this process.
pub(crate) fn is_proc_myself(name: &[u8], entry: &str) -> bool {
    let Some(rest) = name.strip_prefix(b"/proc/") else { return false };
    let rest = if let Some(r) = rest.strip_prefix(b"self/") {
        r
    } else if rest.first().is_some_and(|c| (b'1'..=b'9').contains(c)) {
        let me = format!("{}/", sys(libc::SYS_getpid, &[]));
        match rest.strip_prefix(me.as_bytes()) {
            Some(r) => r,
            None => return false,
        }
    } else {
        return false;
    };
    rest == entry.as_bytes()
}

/// The fake files, `fakes[]`.
const FAKES: [&str; 5] = ["maps", "smaps", "stat", "auxv", "cmdline"];

/// `maybe_do_fake_open()` past `/proc/self/exe`: a memfd holding the file when `name` is one
/// of [`FAKES`], or `None` to open `name` as it is.
pub(crate) fn fake_open(p: &Proc, t: &Task, name: &[u8]) -> Option<i64> {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    // If this is a file from the /proc/ filesystem, expand the full name.
    let real = std::fs::canonicalize(std::ffi::OsStr::from_bytes(name))
        .ok()
        .map(|r| r.into_os_string().into_vec())
        .filter(|r| r.starts_with(b"/proc/"));
    let path = real.as_deref().unwrap_or(name);
    let cpuinfo = crate::guest::guest().cpuinfo.filter(|_| path == b"/proc/cpuinfo");
    let data = if let Some(cpuinfo) = cpuinfo {
        cpuinfo().into_bytes()
    } else {
        let entry = FAKES.iter().find(|e| is_proc_myself(path, e))?;
        match *entry {
            "maps" => maps(p, false),
            "smaps" => maps(p, true),
            "stat" => stat(p, t),
            "auxv" => auxv(p),
            _ => cmdline(p),
        }
    };
    let fd = sys(libc::SYS_memfd_create, &[c"qemu-open".as_ptr() as u64, 0]);
    if fd < 0 {
        return Some(fd);
    }
    let mut done = 0;
    while done < data.len() {
        let r = sys(
            libc::SYS_write,
            &[fd as u64, data[done..].as_ptr() as u64, (data.len() - done) as u64],
        );
        if r <= 0 {
            sys(libc::SYS_close, &[fd as u64]);
            return Some(if r < 0 { r } else { -i64::from(libc::EIO) });
        }
        done += r as usize;
    }
    sys(libc::SYS_lseek, &[fd as u64, 0, libc::SEEK_SET as u64]);
    Some(fd)
}

/// One line of the host's `/proc/self/maps`, `MapInfo`.
struct MapInfo {
    start: u64,
    end: u64,
    offset: u64,
    dev: (u32, u32),
    inode: u64,
    is_priv: bool,
    path: Option<String>,
}

/// `read_self_maps()`.
fn read_self_maps() -> Option<Vec<MapInfo>> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    let mut v = Vec::new();
    for line in maps.lines() {
        let f: Vec<&str> = line.splitn(6, ' ').collect();
        if f.len() < 5 {
            continue;
        }
        let parse = || -> Option<MapInfo> {
            let (s, e) = f[0].split_once('-')?;
            let (maj, min) = f[3].split_once(':')?;
            let path = f.get(5).map(|p| p.trim_start_matches(' ').to_string());
            Some(MapInfo {
                start: u64::from_str_radix(s, 16).ok()?,
                end: u64::from_str_radix(e, 16).ok()?,
                offset: u64::from_str_radix(f[2], 16).ok()?,
                dev: (u32::from_str_radix(maj, 16).ok()?, u32::from_str_radix(min, 16).ok()?),
                inode: f[4].parse().ok()?,
                is_priv: f[1].as_bytes().get(3) == Some(&b'p'),
                path,
            })
        };
        if let Some(m) = parse() {
            v.push(m);
        }
    }
    Some(v)
}

/// `open_self_maps_1()`: a line of `maps`, or a region of `smaps`, for each guest region and
/// each host mapping it spans.
fn maps(p: &Proc, smaps: bool) -> Vec<u8> {
    let mut out = String::new();
    let _mm = p.mm();
    let host = read_self_maps();
    let space = p.space();
    for (start, end, flags) in space.ranges() {
        let anon = MapInfo {
            start: 0,
            end: 0,
            offset: 0,
            dev: (0, 0),
            inode: 0,
            is_priv: true,
            path: None,
        };
        let Some(host) = &host else {
            maps_line(&mut out, p, &anon, start, end, flags, smaps);
            continue;
        };
        // open_self_maps_2(): split the region where the host mappings behind it change.
        let mut g = start;
        while g < end {
            let h = space.g2h(g) as u64;
            let Some(mi) = host.iter().find(|m| m.start <= h && h < m.end) else {
                maps_line(&mut out, p, &anon, g, end, flags, smaps);
                break;
            };
            let this_end = end.min(g + (mi.end - h));
            maps_line(&mut out, p, mi, g, this_end, flags, smaps);
            g = this_end;
        }
    }
    out.into_bytes()
}

/// `open_self_maps_4()`.
fn maps_line(
    out: &mut String,
    p: &Proc,
    mi: &MapInfo,
    start: u64,
    end: u64,
    flags: u32,
    smaps: bool,
) {
    let img = p.image();
    let path = if start == img.stack_limit {
        Some("[stack]")
    } else if start == img.brk {
        Some("[heap]")
    } else {
        mi.path.as_deref()
    };
    // Except the null device (MAP_ANON), adjust the offset for this fragment.
    let mut offset = mi.offset;
    if mi.dev != (0, 0) {
        offset += p.space().g2h(start) as u64 - mi.start;
    }
    let is_priv = mi.is_priv;
    let bit = |b: u32, c: char| if flags & b != 0 { c } else { '-' };
    let line = format!(
        "{start:x}-{end:x} {}{}{}{} {offset:08x} {:02x}:{:02x} {}",
        bit(page::READ, 'r'),
        bit(page::WRITE, 'w'),
        bit(page::EXEC, 'x'),
        if is_priv { 'p' } else { 's' },
        mi.dev.0,
        mi.dev.1,
        mi.inode
    );
    out.push_str(&line);
    if let Some(path) = path {
        let _ = writeln!(out, "{:w$}{path}", "", w = 73usize.saturating_sub(line.len()));
    } else {
        out.push('\n');
    }
    if smaps {
        let size_kb = (end - start) >> 10;
        let page_kb = PAGE_SIZE >> 10;
        let on = |b: u32, s: &'static str| if flags & b != 0 { s } else { "" };
        let _ = write!(
            out,
            "Size:                  {size_kb} kB\n\
             KernelPageSize:        {page_kb} kB\n\
             MMUPageSize:           {page_kb} kB\n\
             Rss:                   0 kB\n\
             Pss:                   0 kB\n\
             Pss_Dirty:             0 kB\n\
             Shared_Clean:          0 kB\n\
             Shared_Dirty:          0 kB\n\
             Private_Clean:         0 kB\n\
             Private_Dirty:         0 kB\n\
             Referenced:            0 kB\n\
             Anonymous:             {} kB\n\
             LazyFree:              0 kB\n\
             AnonHugePages:         0 kB\n\
             ShmemPmdMapped:        0 kB\n\
             FilePmdMapped:         0 kB\n\
             Shared_Hugetlb:        0 kB\n\
             Private_Hugetlb:       0 kB\n\
             Swap:                  0 kB\n\
             SwapPss:               0 kB\n\
             Locked:                0 kB\n\
             THPeligible:    0\n\
             VmFlags:{}{}{}{}{}{}{}{}\n",
            if flags & page::ANON != 0 { size_kb } else { 0 },
            on(page::READ, " rd"),
            on(page::WRITE, " wr"),
            on(page::EXEC, " ex"),
            if is_priv { "" } else { " sh" },
            on(page::READ, " mr"),
            on(page::WRITE, " mw"),
            on(page::EXEC, " me"),
            if is_priv { "" } else { " ms" },
        );
    }
}

/// `open_self_stat()`: the pid, the name, the parent, the group, the threads, the start time
/// and the stack, and 0 for the rest of the 44 fields.
fn stat(p: &Proc, t: &Task) -> Vec<u8> {
    let img = p.image();
    let mut out = String::new();
    for i in 0..44 {
        match i {
            0 => {
                let _ = write!(out, "{} ", sys(libc::SYS_getpid, &[]));
            }
            1 => {
                let argv0 = img.argv.first().map_or(&[][..], Vec::as_slice);
                let bin = argv0.rsplit(|&c| c == b'/').next().unwrap_or(argv0);
                let bin = &bin[..bin.len().min(15)];
                let _ = write!(out, "({}) ", String::from_utf8_lossy(bin));
            }
            2 => out.push_str("R "),
            3 => {
                let _ = write!(out, "{} ", sys(libc::SYS_getppid, &[]));
            }
            4 => {
                let _ = write!(out, "{} ", sys(libc::SYS_getpgrp, &[]));
            }
            19 => {
                let _ = write!(out, "{} ", THREADS.load(std::sync::atomic::Ordering::Acquire));
            }
            21 => {
                let _ = write!(out, "{} ", t.start_boottime);
            }
            27 => {
                let _ = write!(out, "{} ", img.start_stack as i64);
            }
            43 => out.push_str("0\n"),
            _ => out.push_str("0 "),
        }
    }
    out.into_bytes()
}

/// `open_self_auxv()`: the auxiliary vector as it is on the guest's stack.
fn auxv(p: &Proc) -> Vec<u8> {
    let (at, len) = p.image().auxv;
    let mut b = vec![0u8; len as usize];
    if p.space().check(at, len, page::READ) && p.space().read(at, &mut b) { b } else { Vec::new() }
}

/// `open_self_cmdline()`.
fn cmdline(p: &Proc) -> Vec<u8> {
    let mut out = Vec::new();
    for a in &p.image().argv {
        out.extend_from_slice(a);
        out.push(0);
    }
    out
}
