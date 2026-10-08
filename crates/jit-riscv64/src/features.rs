// SPDX-License-Identifier: GPL-2.0-or-later

//! The instruction set extensions the backend may use, QEMU's `cpuinfo` bits that
//! `cpuinfo_init` in `util/cpuinfo-riscv.c` sets: from the `riscv_hwprobe` system call, and
//! for what that does not report, by running one instruction of the extension and catching
//! `SIGILL`.

/// Optional RISC-V extensions. RV64GC is the baseline and always used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HostFeatures {
    /// Zba: `add.uw`, `CPUINFO_ZBA`.
    pub zba: bool,
    /// Zbb: `andn`, `orn`, `xnor`, `clz`, `ctz`, `cpop`, `rol`, `ror`, `rev8`, `sext.b`,
    /// `sext.h` and `zext.h`, `CPUINFO_ZBB`.
    pub zbb: bool,
    /// Zbs: `bexti`, `CPUINFO_ZBS`.
    pub zbs: bool,
    /// Zicond: `czero.eqz` and `czero.nez`, `CPUINFO_ZICOND`.
    pub zicond: bool,
}

impl HostFeatures {
    /// Plain RV64GC.
    pub const BASELINE: HostFeatures =
        HostFeatures { zba: false, zbb: false, zbs: false, zicond: false };

    /// Every extension the backend knows about.
    pub const ALL: HostFeatures = HostFeatures { zba: true, zbb: true, zbs: true, zicond: true };

    /// The extensions of the CPU this runs on; [`HostFeatures::BASELINE`] on other hosts.
    pub fn detect() -> HostFeatures {
        #[cfg(all(target_os = "linux", target_arch = "riscv64"))]
        {
            probe::detect()
        }
        #[cfg(not(all(target_os = "linux", target_arch = "riscv64")))]
        {
            HostFeatures::BASELINE
        }
    }

    /// The same set: no extension here depends on another. Kept so that every backend's
    /// feature set has the same shape.
    pub fn normalized(self) -> HostFeatures {
        self
    }

    /// The extensions in both sets.
    pub fn intersect(self, other: HostFeatures) -> HostFeatures {
        HostFeatures {
            zba: self.zba && other.zba,
            zbb: self.zbb && other.zbb,
            zbs: self.zbs && other.zbs,
            zicond: self.zicond && other.zicond,
        }
        .normalized()
    }
}

#[cfg(all(target_os = "linux", target_arch = "riscv64"))]
mod probe {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::HostFeatures;

    /// `__NR_riscv_hwprobe`.
    const SYS_RISCV_HWPROBE: libc::c_long = 258;
    /// `RISCV_HWPROBE_KEY_IMA_EXT_0`.
    const KEY_IMA_EXT_0: i64 = 4;
    /// `RISCV_HWPROBE_EXT_ZBA`, `RISCV_HWPROBE_EXT_ZBB`, `RISCV_HWPROBE_EXT_ZBS` and
    /// `RISCV_HWPROBE_EXT_ZICOND`.
    const EXT_ZBA: u64 = 1 << 3;
    const EXT_ZBB: u64 = 1 << 4;
    const EXT_ZBS: u64 = 1 << 5;
    const EXT_ZICOND: u64 = 1 << 35;

    /// `struct riscv_hwprobe`.
    #[repr(C)]
    struct Pair {
        key: i64,
        value: u64,
    }

    /// Set by the `SIGILL` handler, `got_sigill`.
    static GOT_SIGILL: AtomicBool = AtomicBool::new(false);

    pub(super) fn detect() -> HostFeatures {
        let mut f = HostFeatures::BASELINE;
        let mut pair = Pair { key: KEY_IMA_EXT_0, value: 0 };
        // SAFETY: the system call writes one `struct riscv_hwprobe` at the pointer, which is a
        // live local of that layout; a kernel without it returns an error and writes nothing.
        let r = unsafe {
            libc::syscall(SYS_RISCV_HWPROBE, &mut pair as *mut Pair, 1usize, 0usize, 0usize, 0u32)
        };
        let mut left_zb = true;
        if r == 0 && pair.key >= 0 {
            f.zba = pair.value & EXT_ZBA != 0;
            f.zbb = pair.value & EXT_ZBB != 0;
            f.zbs = pair.value & EXT_ZBS != 0;
            f.zicond = pair.value & EXT_ZICOND != 0;
            left_zb = false;
        }
        // A kernel older than the Zicond bit reports it clear, as QEMU's build against old
        // headers does not know it; probe for it then, and for everything without hwprobe.
        if left_zb || !f.zicond {
            sigill_probe(&mut f, left_zb);
        }
        f.normalized()
    }

    extern "C" fn on_sigill(_sig: libc::c_int, _info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
        let uc = uc as *mut libc::ucontext_t;
        // SAFETY: the kernel passes the interrupted context of this thread; skipping the
        // faulting instruction, which the probes below make 4 bytes long, is all this does.
        unsafe { (*uc).uc_mcontext.__gregs[libc::REG_PC] += 4 };
        GOT_SIGILL.store(true, Ordering::SeqCst);
    }

    /// Run one instruction of each extension still unknown and see whether it traps. Only the
    /// thread that runs this sees the handler do anything, as in QEMU, which runs it from a
    /// constructor; another thread's `SIGILL` in the meantime would be skipped too, which is
    /// why the handler is only installed for the few instructions of the probe.
    fn sigill_probe(f: &mut HostFeatures, zb: bool) {
        // SAFETY: `sigaction` with a zeroed struct plus a handler and flags is the documented
        // way to install a handler, and the old one is put back below.
        let mut old: libc::sigaction = unsafe { std::mem::zeroed() };
        // SAFETY: as above.
        let mut new: libc::sigaction = unsafe { std::mem::zeroed() };
        new.sa_sigaction = on_sigill as *const () as usize;
        new.sa_flags = libc::SA_SIGINFO;
        // SAFETY: both structs are live locals.
        unsafe { libc::sigaction(libc::SIGILL, &new, &mut old) };
        let run = |probe: fn()| {
            GOT_SIGILL.store(false, Ordering::SeqCst);
            probe();
            !GOT_SIGILL.load(Ordering::SeqCst)
        };
        if zb {
            f.zba = run(probe_zba);
            f.zbb = run(probe_zbb);
            f.zbs = run(probe_zbs);
        }
        f.zicond = run(probe_zicond);
        // SAFETY: puts back the handler saved above.
        unsafe { libc::sigaction(libc::SIGILL, &old, std::ptr::null_mut()) };
    }

    /// `add.uw zero, zero, zero`.
    fn probe_zba() {
        // SAFETY: the instruction writes `zero` only; without Zba it traps and the handler
        // skips it.
        unsafe { core::arch::asm!(".insn r 0x3b, 0, 0x04, zero, zero, zero") };
    }

    /// `andn zero, zero, zero`.
    fn probe_zbb() {
        // SAFETY: as `probe_zba`.
        unsafe { core::arch::asm!(".insn r 0x33, 7, 0x20, zero, zero, zero") };
    }

    /// `bext zero, zero, zero`.
    fn probe_zbs() {
        // SAFETY: as `probe_zba`.
        unsafe { core::arch::asm!(".insn r 0x33, 5, 0x24, zero, zero, zero") };
    }

    /// `czero.eqz zero, zero, zero`.
    fn probe_zicond() {
        // SAFETY: as `probe_zba`.
        unsafe { core::arch::asm!(".insn r 0x33, 5, 0x07, zero, zero, zero") };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_keeps_common_extensions() {
        let a = HostFeatures { zba: true, zbb: true, zbs: false, zicond: true };
        let b = HostFeatures { zba: true, zbb: false, zbs: true, zicond: true };
        assert_eq!(
            a.intersect(b),
            HostFeatures { zba: true, zbb: false, zbs: false, zicond: true }
        );
        assert_eq!(HostFeatures::ALL.intersect(HostFeatures::BASELINE), HostFeatures::BASELINE);
    }

    #[test]
    fn detect_is_stable() {
        assert_eq!(HostFeatures::detect(), HostFeatures::detect());
    }
}
