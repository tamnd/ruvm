// SPDX-License-Identifier: MIT OR Apache-2.0

//! The error type. It has the same shape as QEMU's `Error` because its class and message go out on
//! the wire in QMP replies and management tools match on the message text.

use std::fmt;
use std::panic::Location as SrcLocation;

/// The classes from `QapiErrorClass` in qapi/error.json, in schema order. Almost every error is a
/// `GenericError`, the others exist for old clients that match on them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorClass {
    GenericError,
    CommandNotFound,
    DeviceNotActive,
    DeviceNotFound,
    KvmMissingCap,
}

impl ErrorClass {
    /// Every class, in the order qapi/error.json declares them.
    pub const ALL: [ErrorClass; 5] = [
        ErrorClass::GenericError,
        ErrorClass::CommandNotFound,
        ErrorClass::DeviceNotActive,
        ErrorClass::DeviceNotFound,
        ErrorClass::KvmMissingCap,
    ];

    /// The name as it appears in the `class` member of a QMP error reply.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorClass::GenericError => "GenericError",
            ErrorClass::CommandNotFound => "CommandNotFound",
            ErrorClass::DeviceNotActive => "DeviceNotActive",
            ErrorClass::DeviceNotFound => "DeviceNotFound",
            ErrorClass::KvmMissingCap => "KVMMissingCap",
        }
    }
}

impl fmt::Display for ErrorClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An error with a class, a message, an optional hint and the place in the source that made it.
///
/// The hint is what `error_append_hint()` adds. It is printed on the command line and left out of
/// QMP replies, as in QEMU.
pub struct Error {
    class: ErrorClass,
    msg: Box<str>,
    hint: Option<Box<str>>,
    src: &'static SrcLocation<'static>,
    cause: Option<Box<dyn std::error::Error + Send + Sync>>,
}

pub type Result<T, E = Error> = core::result::Result<T, E>;

impl Error {
    #[track_caller]
    pub fn new(class: ErrorClass, msg: impl Into<String>) -> Self {
        Error {
            class,
            msg: msg.into().into_boxed_str(),
            hint: None,
            src: SrcLocation::caller(),
            cause: None,
        }
    }

    #[track_caller]
    pub fn generic(msg: impl Into<String>) -> Self {
        Error::new(ErrorClass::GenericError, msg)
    }

    /// Wraps a host error, keeping it as the cause. The message is the one given here, because
    /// QEMU's messages put the strerror text in a fixed place that callers have to spell out.
    #[track_caller]
    pub fn with_cause(
        msg: impl Into<String>,
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        let mut e = Error::generic(msg);
        e.cause = Some(Box::new(cause));
        e
    }

    /// The equivalent of `error_setg_errno()`: the message followed by `: ` and the strerror text.
    #[track_caller]
    pub fn from_io(msg: impl fmt::Display, io: std::io::Error) -> Self {
        let text = strerror(&io);
        Error::with_cause(format!("{msg}: {text}"), io)
    }

    pub fn class(&self) -> ErrorClass {
        self.class
    }

    pub fn message(&self) -> &str {
        &self.msg
    }

    pub fn hint_text(&self) -> Option<&str> {
        self.hint.as_deref()
    }

    pub fn source_location(&self) -> &'static SrcLocation<'static> {
        self.src
    }

    /// `error_prepend()`: puts text in front of the message.
    pub fn prepend(mut self, prefix: impl fmt::Display) -> Self {
        self.msg = format!("{prefix}{}", self.msg).into_boxed_str();
        self
    }

    /// `error_append_hint()`: adds to the hint. QEMU hints usually end in a newline and so do these.
    pub fn hint(mut self, text: impl fmt::Display) -> Self {
        let mut h = self.hint.map(String::from).unwrap_or_default();
        h.push_str(&text.to_string());
        self.hint = Some(h.into_boxed_str());
        self
    }
}

/// The text of a host error without Rust's " (os error N)" suffix, which is what `strerror()` gives
/// and what QEMU's messages contain.
pub fn strerror(e: &std::io::Error) -> String {
    let s = e.to_string();
    match s.rfind(" (os error ") {
        Some(i) => s[..i].to_string(),
        None => s,
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Error")
            .field("class", &self.class)
            .field("msg", &self.msg)
            .field("hint", &self.hint)
            .field("src", &format_args!("{}:{}", self.src.file(), self.src.line()))
            .finish()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.msg)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause.as_deref().map(|c| c as &(dyn std::error::Error + 'static))
    }
}

/// What `&error_fatal`, `&error_abort` and `&error_warn` do in QEMU, as methods.
pub trait ResultExt<T> {
    /// Reports the error with the current location and exits with status 1.
    fn or_fatal(self) -> T;
    /// Panics with the error. For errors that mean ruvm itself is wrong.
    fn or_abort(self) -> T;
    /// Reports the error as a warning and carries on without a value.
    fn or_warn(self) -> Option<T>;
    /// `error_prepend()` on the error side of a result.
    fn prepend(self, prefix: impl fmt::Display) -> Self;
}

impl<T> ResultExt<T> for Result<T> {
    fn or_fatal(self) -> T {
        match self {
            Ok(v) => v,
            Err(e) => {
                crate::report::report_error(&e);
                std::process::exit(1)
            }
        }
    }

    #[track_caller]
    fn or_abort(self) -> T {
        match self {
            Ok(v) => v,
            Err(e) => panic!("unexpected error: {e} (made at {}:{})", e.src.file(), e.src.line()),
        }
    }

    fn or_warn(self) -> Option<T> {
        match self {
            Ok(v) => Some(v),
            Err(e) => {
                crate::report::warn_report(e.message());
                None
            }
        }
    }

    fn prepend(self, prefix: impl fmt::Display) -> Self {
        self.map_err(|e| e.prepend(prefix))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepend_and_hint_build_up_like_qemu() {
        let e = Error::generic("Property 'drive' can't find value 'nope'")
            .prepend("-device virtio-blk-pci: ")
            .hint("Try 'info block'\n")
            .hint("or 'query-block'\n");
        assert_eq!(e.message(), "-device virtio-blk-pci: Property 'drive' can't find value 'nope'");
        assert_eq!(e.hint_text(), Some("Try 'info block'\nor 'query-block'\n"));
        assert_eq!(e.class(), ErrorClass::GenericError);
    }

    #[test]
    fn the_source_location_is_the_caller() {
        let e = crate::err!("x = {}", 1);
        assert!(e.source_location().file().ends_with("error.rs"));
        assert_eq!(e.message(), "x = 1");
    }

    #[test]
    fn io_errors_read_like_strerror() {
        let io = std::io::Error::from_raw_os_error(2);
        let e = Error::from_io("Could not open 'x'", io);
        assert_eq!(e.message(), "Could not open 'x': No such file or directory");
        assert!(std::error::Error::source(&e).is_some());
    }

    #[test]
    fn class_names_match_the_schema() {
        let names: Vec<_> = ErrorClass::ALL.iter().map(|c| c.as_str()).collect();
        assert_eq!(
            names,
            [
                "GenericError",
                "CommandNotFound",
                "DeviceNotActive",
                "DeviceNotFound",
                "KVMMissingCap"
            ]
        );
    }
}
