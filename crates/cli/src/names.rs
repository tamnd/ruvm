// SPDX-License-Identifier: GPL-2.0-or-later

//! Every binary name QEMU 11.1 installs, and what ruvm does when it is started under one.
//!
//! The target lists are the `-softmmu`, `-linux-user` and `-bsd-user` entries in QEMU's
//! `configs/targets/`. User mode binaries are called `qemu-<target>` whether they come from
//! linux-user or bsd-user, because a host only ever builds one of the two.

/// The target names QEMU builds a `qemu-system-<target>` binary for.
pub(crate) const SYSTEM_TARGETS: &[&str] = &[
    "aarch64",
    "alpha",
    "arm",
    "avr",
    "hexagon",
    "hppa",
    "i386",
    "loongarch64",
    "m68k",
    "microblaze",
    "mips",
    "mips64",
    "mips64el",
    "mipsel",
    "or1k",
    "ppc",
    "ppc64",
    "riscv32",
    "riscv64",
    "rx",
    "s390x",
    "sh4",
    "sh4eb",
    "sparc",
    "sparc64",
    "tricore",
    "x86_64",
    "xtensa",
    "xtensaeb",
];

/// The target names QEMU builds a `qemu-<target>` user mode binary for on Linux.
pub(crate) const LINUX_USER_TARGETS: &[&str] = &[
    "aarch64",
    "aarch64_be",
    "alpha",
    "arm",
    "armeb",
    "hexagon",
    "hppa",
    "i386",
    "loongarch64",
    "m68k",
    "microblaze",
    "microblazeel",
    "mips",
    "mips64",
    "mips64el",
    "mipsel",
    "mipsn32",
    "mipsn32el",
    "or1k",
    "ppc",
    "ppc64",
    "ppc64le",
    "riscv32",
    "riscv64",
    "s390x",
    "sh4",
    "sh4eb",
    "sparc",
    "sparc32plus",
    "sparc64",
    "x86_64",
    "xtensa",
    "xtensaeb",
];

/// The target names QEMU builds a `qemu-<target>` user mode binary for on FreeBSD.
pub(crate) const BSD_USER_TARGETS: &[&str] = &["aarch64", "arm", "i386", "riscv64", "x86_64"];

/// The tools, which each have a personality of their own.
pub(crate) const TOOLS: &[Tool] = &[
    Tool::Img,
    Tool::Io,
    Tool::Nbd,
    Tool::StorageDaemon,
    Tool::PrHelper,
    Tool::BridgeHelper,
    Tool::VmsrHelper,
    Tool::Vnc,
    Tool::Edid,
    Tool::Keymap,
    Tool::Elf2dmp,
];

/// A QEMU tool binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tool {
    Img,
    Io,
    Nbd,
    StorageDaemon,
    PrHelper,
    BridgeHelper,
    VmsrHelper,
    Vnc,
    Edid,
    Keymap,
    Elf2dmp,
}

impl Tool {
    /// The name QEMU installs the tool under.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Tool::Img => "qemu-img",
            Tool::Io => "qemu-io",
            Tool::Nbd => "qemu-nbd",
            Tool::StorageDaemon => "qemu-storage-daemon",
            Tool::PrHelper => "qemu-pr-helper",
            Tool::BridgeHelper => "qemu-bridge-helper",
            Tool::VmsrHelper => "qemu-vmsr-helper",
            Tool::Vnc => "qemu-vnc",
            Tool::Edid => "qemu-edid",
            Tool::Keymap => "qemu-keymap",
            Tool::Elf2dmp => "elf2dmp",
        }
    }
}

/// What the binary is being asked to be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Personality {
    /// `ruvm` itself.
    Ruvm,
    /// `qemu-system-<target>`.
    System(&'static str),
    /// `qemu-<target>`, user mode.
    User(&'static str),
    /// One of the tools.
    Tool(Tool),
}

impl Personality {
    /// The name the personality answers to, which is also what an installer symlinks.
    pub(crate) fn name(&self) -> String {
        match self {
            Personality::Ruvm => "ruvm".to_string(),
            Personality::System(t) => format!("qemu-system-{t}"),
            Personality::User(t) => format!("qemu-{t}"),
            Personality::Tool(tool) => tool.name().to_string(),
        }
    }

    /// Every personality, in the order `ruvm --list` prints them.
    pub(crate) fn all() -> Vec<Personality> {
        let mut all = vec![Personality::Ruvm];
        all.extend(SYSTEM_TARGETS.iter().map(|t| Personality::System(t)));
        let mut user: Vec<&'static str> =
            LINUX_USER_TARGETS.iter().chain(BSD_USER_TARGETS).copied().collect();
        user.sort_unstable();
        user.dedup();
        all.extend(user.into_iter().map(Personality::User));
        all.extend(TOOLS.iter().map(|t| Personality::Tool(*t)));
        all
    }

    /// The personality for a program name, as it appears in `argv[0]`.
    ///
    /// Only the file name counts, and on Windows a trailing `.exe` is ignored, so
    /// `C:\qemu\qemu-img.exe` and `/usr/bin/qemu-img` are the same program. Case matters
    /// everywhere, because it does to QEMU and to libvirt's search of `PATH`.
    pub(crate) fn from_argv0(argv0: &str) -> Option<Personality> {
        let base = argv0.rsplit(['/', '\\']).next().unwrap_or(argv0);
        let base = base.strip_suffix(".exe").unwrap_or(base);
        Self::from_name(base)
    }

    /// The personality with exactly this name.
    pub(crate) fn from_name(name: &str) -> Option<Personality> {
        if name == "ruvm" {
            return Some(Personality::Ruvm);
        }
        // ruvm-system-<target> is the same program under ruvm's own name.
        let system =
            name.strip_prefix("qemu-system-").or_else(|| name.strip_prefix("ruvm-system-"));
        if let Some(target) = system {
            return SYSTEM_TARGETS.iter().find(|t| **t == target).map(|t| Personality::System(t));
        }
        if let Some(tool) = TOOLS.iter().find(|t| t.name() == name) {
            return Some(Personality::Tool(*tool));
        }
        let target = name.strip_prefix("qemu-")?;
        LINUX_USER_TARGETS
            .iter()
            .chain(BSD_USER_TARGETS)
            .find(|t| **t == target)
            .map(|t| Personality::User(t))
    }
}

#[cfg(test)]
mod tests {
    use super::{Personality, Tool};

    #[test]
    fn paths_and_exe_suffixes_are_ignored() {
        let want = Some(Personality::System("x86_64"));
        assert_eq!(Personality::from_argv0("qemu-system-x86_64"), want);
        assert_eq!(Personality::from_argv0("/usr/bin/ruvm-system-x86_64"), want);
        assert_eq!(Personality::from_argv0("/usr/bin/qemu-system-x86_64"), want);
        assert_eq!(Personality::from_argv0(r"C:\qemu\qemu-system-x86_64.exe"), want);
    }

    #[test]
    fn tools_win_over_user_mode_targets() {
        assert_eq!(Personality::from_name("qemu-img"), Some(Personality::Tool(Tool::Img)));
        assert_eq!(Personality::from_name("qemu-io"), Some(Personality::Tool(Tool::Io)));
        assert_eq!(Personality::from_name("elf2dmp"), Some(Personality::Tool(Tool::Elf2dmp)));
    }

    #[test]
    fn user_mode_names_resolve() {
        assert_eq!(Personality::from_name("qemu-aarch64"), Some(Personality::User("aarch64")));
        assert_eq!(Personality::from_name("qemu-ppc64le"), Some(Personality::User("ppc64le")));
    }

    #[test]
    fn unknown_names_do_not() {
        assert_eq!(Personality::from_name("qemu-system-z80"), None);
        assert_eq!(Personality::from_name("qemu-kvm"), None);
        assert_eq!(Personality::from_name("QEMU-IMG"), None);
    }

    #[test]
    fn every_name_round_trips() {
        let all = Personality::all();
        for p in &all {
            assert_eq!(Personality::from_name(&p.name()).as_ref(), Some(p), "{}", p.name());
        }
        let mut names: Vec<String> = all.iter().map(Personality::name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), all.len(), "two personalities share a name");
    }
}
