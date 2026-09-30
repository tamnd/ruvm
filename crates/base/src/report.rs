// SPDX-License-Identifier: MIT OR Apache-2.0

//! `error_report()` and friends from util/error-report.c.
//!
//! QEMU prefixes every message with the program name and, when it is processing a command line
//! option or a config file line, with that location. Scripts and test suites match these lines, so
//! the format is copied exactly.

use std::cell::RefCell;
use std::io::Write;
use std::sync::OnceLock;

static PROGRAM: OnceLock<String> = OnceLock::new();

/// Sets the name printed in front of every message. The dispatcher calls this with the name the
/// binary was run as, so messages say `qemu-system-x86_64:` rather than `ruvm:`.
pub fn set_program_name(name: &str) {
    let _ = PROGRAM.set(name.to_string());
}

pub fn program_name() -> &'static str {
    PROGRAM.get().map(String::as_str).unwrap_or("ruvm")
}

/// Where the program is in its input, for messages. This is QEMU's `Location`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Location {
    /// A command line option and its argument, printed as `-device virtio-blk-pci,drive=x: `.
    CmdLine { option: String, arg: Option<String> },
    /// A line in a file, printed as `vm.cfg:12: `.
    File { name: String, line: u32 },
}

thread_local! {
    static LOCATIONS: RefCell<Vec<Location>> = const { RefCell::new(Vec::new()) };
}

/// Pushes a location for as long as the guard lives, like `loc_push_restore()` and
/// `loc_pop()`.
pub fn push_location(loc: Location) -> LocationGuard {
    LOCATIONS.with(|l| l.borrow_mut().push(loc));
    LocationGuard { _private: () }
}

#[derive(Debug)]
#[must_use = "the location is popped when the guard is dropped"]
pub struct LocationGuard {
    _private: (),
}

impl Drop for LocationGuard {
    fn drop(&mut self) {
        LOCATIONS.with(|l| l.borrow_mut().pop());
    }
}

/// The current location, for code that reports on it later as QEMU does with the `loc` a
/// `QemuOpts` keeps.
pub fn current_location() -> Option<Location> {
    LOCATIONS.with(|l| l.borrow().last().cloned())
}

/// The current location, formatted the way `error_print_loc()` does, or an empty string.
pub fn location_prefix() -> String {
    LOCATIONS.with(|l| match l.borrow().last() {
        None => String::new(),
        Some(Location::CmdLine { option, arg: Some(arg) }) => format!("{option} {arg}: "),
        Some(Location::CmdLine { option, arg: None }) => format!("{option}: "),
        Some(Location::File { name, line }) => format!("{name}:{line}: "),
    })
}

/// Formats a message line the way `error_vprintf()` with `error_print_loc()` would.
pub fn format_message(kind: &str, msg: &str) -> String {
    format!("{}: {}{kind}{msg}\n", program_name(), location_prefix())
}

pub fn error_report(msg: &str) {
    emit(&format_message("", msg));
}

pub fn warn_report(msg: &str) {
    emit(&format_message("warning: ", msg));
}

pub fn info_report(msg: &str) {
    emit(&format_message("info: ", msg));
}

/// `error_report_err()`: the message, then the hint if there is one.
pub fn report_error(e: &crate::Error) {
    let mut out = format_message("", e.message());
    if let Some(h) = e.hint_text() {
        out.push_str(h);
    }
    emit(&out);
}

fn emit(s: &str) {
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(s.as_bytes());
    let _ = err.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locations_nest_and_pop() {
        assert_eq!(location_prefix(), "");
        let outer = push_location(Location::File { name: "vm.cfg".into(), line: 12 });
        assert_eq!(location_prefix(), "vm.cfg:12: ");
        {
            let _inner = push_location(Location::CmdLine {
                option: "-device".into(),
                arg: Some("virtio-blk-pci,drive=nope".into()),
            });
            assert_eq!(location_prefix(), "-device virtio-blk-pci,drive=nope: ");
        }
        assert_eq!(location_prefix(), "vm.cfg:12: ");
        drop(outer);
        assert_eq!(location_prefix(), "");
    }

    #[test]
    fn warnings_carry_the_prefix() {
        let line = format_message("warning: ", "x");
        assert!(line.ends_with(": warning: x\n"));
    }
}
