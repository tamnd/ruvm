// SPDX-License-Identifier: GPL-2.0-or-later

//! What each personality prints for its version option, byte for byte what QEMU 11.1 prints.
//!
//! spec/02-compat-contract.md settles the format. The numeric version is the QEMU release ruvm
//! tracks, because libvirt refuses anything older than its minimum and turns features on by
//! version. ruvm's own version goes in the package suffix, which is where QEMU puts the
//! distribution's version when it is configured with `--with-pkgversion`, so every parser that
//! copes with a Debian or Fedora build copes with ruvm.

use crate::names::{Personality, Tool};

/// The QEMU release whose behavior ruvm reproduces.
pub(crate) const QEMU_VERSION: &str = "11.1.0";

/// ruvm's own version.
pub(crate) const RUVM_VERSION: &str = env!("CARGO_PKG_VERSION");

/// QEMU's `QEMU_COPYRIGHT` from `include/qemu/help-texts.h`.
pub(crate) const QEMU_COPYRIGHT: &str =
    "Copyright (c) 2003-2026 Fabrice Bellard and the QEMU Project developers";

const FREE_SOFTWARE: &str = "This is free software; see the source for copying conditions.  There is NO\nwarranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n";

/// `QEMU_FULL_VERSION`: the release, then the package version in parentheses.
pub(crate) fn full_version() -> String {
    format!("{QEMU_VERSION} (ruvm {RUVM_VERSION})")
}

/// The version text for a personality, or `None` for the tools that have no version option.
///
/// `argv0` is used as given by the tools that print `argv[0]` rather than a fixed name, which are
/// qemu-nbd, qemu-pr-helper and qemu-vmsr-helper.
pub(crate) fn text(personality: &Personality, argv0: &str) -> Option<String> {
    let full = full_version();
    let text = match personality {
        Personality::Ruvm => {
            format!("ruvm {RUVM_VERSION}\nCompatible with QEMU {QEMU_VERSION}\n")
        }
        Personality::System(_) => format!("QEMU emulator version {full}\n{QEMU_COPYRIGHT}\n"),
        Personality::User(target) => format!("qemu-{target} version {full}\n{QEMU_COPYRIGHT}\n"),
        Personality::Tool(tool) => match tool {
            Tool::Img | Tool::Io | Tool::StorageDaemon => {
                format!("{} version {full}\n{QEMU_COPYRIGHT}\n", tool.name())
            }
            Tool::Nbd => written_by(argv0, &full, "Anthony Liguori"),
            Tool::PrHelper => written_by(argv0, &full, "Paolo Bonzini"),
            Tool::VmsrHelper => written_by(argv0, &full, "Anthony Harivel"),
            Tool::Vnc => format!("qemu-vnc {full}\n"),
            Tool::BridgeHelper | Tool::Edid | Tool::Keymap | Tool::Elf2dmp => return None,
        },
    };
    Some(text)
}

fn written_by(argv0: &str, full: &str, author: &str) -> String {
    format!("{argv0} {full}\nWritten by {author}.\n\n{QEMU_COPYRIGHT}\n{FREE_SOFTWARE}")
}

#[cfg(test)]
mod tests {
    use super::{Personality, QEMU_VERSION, text};
    use crate::names::Tool;

    /// libvirt's reading of a `-version` line: the text after "QEMU emulator version ", then
    /// major, minor and an optional micro separated by dots, then an optional package in
    /// parentheses. This is the shape of the parser in libvirt's `qemu_capabilities.c` from the
    /// years it read `-version` rather than asking QMP, and the shape every script copied from it.
    fn libvirt_parse(output: &str) -> Option<(u32, u32, u32, Option<String>)> {
        let rest = output.split("QEMU emulator version ").nth(1)?;
        let line = rest.lines().next()?;
        let (number, package) = match line.split_once(' ') {
            Some((n, p)) => (n, Some(p)),
            None => (line, None),
        };
        let mut parts = number.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let micro = parts.next().map_or(Some(0), |m| m.parse().ok())?;
        let package = match package {
            Some(p) => Some(p.strip_prefix('(')?.strip_suffix(')')?.to_string()),
            None => None,
        };
        Some((major, minor, micro, package))
    }

    #[test]
    fn libvirt_reads_the_system_version() {
        let out = text(&Personality::System("x86_64"), "qemu-system-x86_64").unwrap();
        let (major, minor, micro, package) = libvirt_parse(&out).unwrap();
        assert_eq!(format!("{major}.{minor}.{micro}"), QEMU_VERSION);
        assert!(package.unwrap().starts_with("ruvm "));
        // libvirt's floor is 7.2.0, see spec/02-compat-contract.md.
        assert!((major, minor) >= (7, 2));
    }

    #[test]
    fn the_formats_match_qemu() {
        let full = super::full_version();
        let copyright = super::QEMU_COPYRIGHT;
        assert_eq!(
            text(&Personality::Tool(Tool::Img), "qemu-img").unwrap(),
            format!("qemu-img version {full}\n{copyright}\n")
        );
        assert_eq!(
            text(&Personality::User("aarch64"), "qemu-aarch64").unwrap(),
            format!("qemu-aarch64 version {full}\n{copyright}\n")
        );
        assert!(
            text(&Personality::Tool(Tool::Nbd), "/usr/bin/qemu-nbd")
                .unwrap()
                .starts_with(&format!("/usr/bin/qemu-nbd {full}\nWritten by Anthony Liguori.\n\n"))
        );
        assert_eq!(text(&Personality::Tool(Tool::Edid), "qemu-edid"), None);
    }
}
