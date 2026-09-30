// SPDX-License-Identifier: GPL-2.0-or-later

//! The option table of the system emulator, generated from qemu-options.hx, and
//! `lookup_opt()` and `help()` from system/vl.c.

use ruvm_base::report::{Location, LocationGuard, push_location};
use ruvm_base::{Error, Result};

/// The `QEMU_ARCH_*` masks. A target belongs to one of them, and [`Arch::ALL`] matches every
/// target.
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arch {
    ALL,
    ALPHA,
    ARM,
    I386,
    M68K,
    MICROBLAZE,
    MIPS,
    PPC,
    S390X,
    SH4,
    SPARC,
    XTENSA,
    OR1K,
    TRICORE,
    HPPA,
    RISCV,
    RX,
    AVR,
    HEXAGON,
    LOONGARCH,
}

impl Arch {
    /// The mask a target is in, from its name as in `qemu-system-<target>`.
    pub fn of_target(target: &str) -> Option<Arch> {
        Some(match target {
            "alpha" => Arch::ALPHA,
            "arm" | "aarch64" => Arch::ARM,
            "i386" | "x86_64" => Arch::I386,
            "m68k" => Arch::M68K,
            "microblaze" | "microblazeel" => Arch::MICROBLAZE,
            "mips" | "mipsel" | "mips64" | "mips64el" => Arch::MIPS,
            "ppc" | "ppc64" => Arch::PPC,
            "s390x" => Arch::S390X,
            "sh4" | "sh4eb" => Arch::SH4,
            "sparc" | "sparc64" => Arch::SPARC,
            "xtensa" | "xtensaeb" => Arch::XTENSA,
            "or1k" => Arch::OR1K,
            "tricore" => Arch::TRICORE,
            "hppa" => Arch::HPPA,
            "riscv32" | "riscv64" => Arch::RISCV,
            "rx" => Arch::RX,
            "avr" => Arch::AVR,
            "hexagon" => Arch::HEXAGON,
            "loongarch64" => Arch::LOONGARCH,
            _ => return None,
        })
    }
}

/// `qemu_arch_available()`: whether a mask includes the target.
pub fn arch_available(mask: &[Arch], target: &str) -> bool {
    mask.contains(&Arch::ALL) || Arch::of_target(target).is_some_and(|a| mask.contains(&a))
}

/// One entry of the table, `QEMUOption`.
#[derive(Debug)]
pub struct QemuOption {
    /// The name without the leading dash.
    pub name: &'static str,
    /// `HAS_ARG`.
    pub has_arg: bool,
    pub index: Opt,
    pub arch: &'static [Arch],
}

/// A line of the help text.
#[derive(Debug)]
pub struct HelpItem {
    pub text: &'static str,
    /// Headings are printed with a newline after them, like `puts()`.
    pub heading: bool,
    pub arch: &'static [Arch],
}

include!(concat!(env!("OUT_DIR"), "/options.rs"));

/// An option found on the command line.
#[derive(Debug)]
pub struct Found<'a> {
    pub option: &'static QemuOption,
    pub arg: Option<&'a str>,
    /// The location the option's argument is processed under, `-machine none: `. Errors
    /// reported while it lives carry it.
    pub location: LocationGuard,
}

/// `lookup_opt()`: the option at `args[*optind]`, and its argument. `*optind` moves past both.
/// `--foo` is the same as `-foo`. The errors are QEMU's, reported under the option's location
/// the way `loc_set_cmdline()` sets it.
pub fn lookup_opt<'a>(args: &'a [String], optind: &mut usize) -> Result<Found<'a>> {
    let r = &args[*optind];
    let guard = push_location(Location::CmdLine { option: r.clone(), arg: None });
    *optind += 1;
    let name = r.strip_prefix("--").or_else(|| r.strip_prefix('-')).unwrap_or(r);
    let Some(option) = QEMU_OPTIONS.iter().find(|o| o.name == name) else {
        return Err(located("invalid option", guard));
    };
    let arg = if option.has_arg {
        let Some(arg) = args.get(*optind) else {
            return Err(located("requires an argument", guard));
        };
        *optind += 1;
        Some(arg.as_str())
    } else {
        None
    };
    drop(guard);
    let location =
        push_location(Location::CmdLine { option: r.clone(), arg: arg.map(str::to_string) });
    Ok(Found { option, arg, location })
}

/// An error whose text already has the location in front, since the guard is gone by the time
/// the caller prints it.
fn located(msg: &str, guard: LocationGuard) -> Error {
    let prefix = ruvm_base::report::location_prefix();
    drop(guard);
    Error::generic(format!("{prefix}{msg}"))
}

/// The text `help()` prints, for a target and the program name QEMU gets from
/// `g_get_prgname()`. `version` is what `-version` prints.
pub fn help_text(target: &str, prgname: &str, version: &str) -> String {
    let mut out = String::from(version);
    out.push_str(&format!(
        "usage: {prgname} [options] [disk_image]\n\n'disk_image' is a raw hard disk image for IDE hard disk 0\n\n"
    ));
    for item in HELP.iter().filter(|i| arch_available(i.arch, target)) {
        out.push_str(item.text);
        if item.heading {
            out.push('\n');
        }
    }
    out.push_str(concat!(
        "\nDuring emulation, the following keys are useful:\n",
        "ctrl-alt-f      toggle full screen\n",
        "ctrl-alt-n      switch to virtual console 'n'\n",
        "ctrl-alt-g      toggle mouse and keyboard grab\n",
        "\n",
        "When using -nographic, press 'ctrl-a h' to get some help.\n",
        "\n",
        "See <https://qemu.org/contribute/report-a-bug> for how to report bugs.\n",
        "More information on the QEMU project at <https://qemu.org>.\n",
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn table() {
        // 115 DEF() entries at v11.1.0, fewer the ones this build leaves out, plus `-h`.
        assert!(QEMU_OPTIONS.len() > 100 && QEMU_OPTIONS.len() <= 116);
        assert_eq!(QEMU_OPTIONS[0].name, "h");
        let find = |n: &str| QEMU_OPTIONS.iter().find(|o| o.name == n).unwrap();
        assert_eq!(find("help").index, Opt::H);
        assert!(find("machine").has_arg);
        assert!(!find("S").has_arg);
        assert_eq!(find("add-fd").index, Opt::AddFd);
        assert_eq!(find("smbios").arch, &[Arch::I386, Arch::ARM, Arch::LOONGARCH, Arch::RISCV]);
    }

    #[test]
    fn lookup() {
        let a = args(&["-machine", "none", "--S", "-nodefaults", "-foo", "-qmp"]);
        let mut i = 0;
        let f = lookup_opt(&a, &mut i).unwrap();
        assert_eq!((f.option.index, f.arg), (Opt::Machine, Some("none")));
        assert_eq!(ruvm_base::report::location_prefix(), "-machine none: ");
        drop(f);
        assert_eq!(i, 2);
        assert_eq!(lookup_opt(&a, &mut i).unwrap().option.index, Opt::S);
        assert_eq!(lookup_opt(&a, &mut i).unwrap().option.index, Opt::Nodefaults);
        assert_eq!(lookup_opt(&a, &mut i).unwrap_err().to_string(), "-foo: invalid option");
        assert_eq!(lookup_opt(&a, &mut i).unwrap_err().to_string(), "-qmp: requires an argument");
        assert_eq!(ruvm_base::report::location_prefix(), "");
    }

    #[test]
    fn help() {
        let x86 = help_text("x86_64", "qemu-system-x86_64", "V\n");
        assert!(x86.starts_with("V\nusage: qemu-system-x86_64 [options] [disk_image]\n"));
        assert!(x86.contains("\nStandard options:\n-h or -help     display this help and exit\n"));
        assert!(x86.contains("i386 target only:\n"));
        assert!(!help_text("s390x", "q", "").contains("i386 target only:"));
    }
}
