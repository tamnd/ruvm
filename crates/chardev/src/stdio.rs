// SPDX-License-Identifier: GPL-2.0-or-later

//! The `stdio` chardev, chardev/char-stdio.c.
//!
//! On Unix the terminal on standard input goes into raw mode while the chardev exists, so each
//! key reaches the guest as it is typed, and it is put back as it was when the chardev goes
//! away. QEMU also does that from `atexit()`, which a Rust program does not run when it calls
//! `std::process::exit()`, so whoever exits the process calls [`term_exit`] first.
//!
//! Only one chardev can have standard input and output at a time.

use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(unix)]
use std::sync::{Mutex, MutexGuard};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::ChardevStdio;

use crate::local::{Local, LocalKind};

#[cfg(unix)]
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `stdio_in_use`.
static IN_USE: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
static TERMINAL: Mutex<Option<Terminal>> = Mutex::new(None);

/// A terminal in the mode `stdio` wants, `oldtty` and `stdio_chr_set_echo()`. Dropping it puts
/// the terminal back as it was.
#[cfg(unix)]
#[derive(Debug)]
pub struct Terminal {
    fd: std::os::fd::OwnedFd,
    /// What the terminal was, or `None` when `fd` is not a terminal.
    old: Option<rustix::termios::Termios>,
    allow_signal: bool,
    echo: AtomicBool,
}

#[cfg(unix)]
impl Terminal {
    /// Remembers how `fd` is set up and turns echo and line editing off. With `allow_signal`
    /// off, Ctrl-C and friends reach the guest instead of stopping ruvm. Nothing happens when
    /// `fd` is not a terminal.
    pub fn new(fd: std::os::fd::OwnedFd, allow_signal: bool) -> Terminal {
        let old = rustix::termios::tcgetattr(&fd).ok();
        let t = Terminal { fd, old, allow_signal, echo: AtomicBool::new(false) };
        t.set_echo(false);
        t
    }

    /// `stdio_chr_set_echo()`: back to the old mode with echo on, raw without it.
    pub fn set_echo(&self, echo: bool) {
        use rustix::termios::{
            ControlModes, InputModes, LocalModes, OptionalActions, OutputModes, SpecialCodeIndex,
            tcsetattr,
        };

        self.echo.store(echo, Ordering::Relaxed);
        let Some(old) = &self.old else { return };
        let mut tty = old.clone();
        if !echo {
            tty.input_modes &= !(InputModes::IGNBRK
                | InputModes::BRKINT
                | InputModes::PARMRK
                | InputModes::ISTRIP
                | InputModes::INLCR
                | InputModes::IGNCR
                | InputModes::ICRNL
                | InputModes::IXON);
            tty.output_modes |= OutputModes::OPOST;
            tty.local_modes &=
                !(LocalModes::ECHO | LocalModes::ECHONL | LocalModes::ICANON | LocalModes::IEXTEN);
            tty.control_modes &= !(ControlModes::CSIZE | ControlModes::PARENB);
            tty.control_modes |= ControlModes::CS8;
            tty.special_codes[SpecialCodeIndex::VMIN] = 1;
            tty.special_codes[SpecialCodeIndex::VTIME] = 0;
        }
        if !self.allow_signal {
            tty.local_modes &= !LocalModes::ISIG;
        }
        // QEMU does not look at the result either.
        let _ = tcsetattr(&self.fd, OptionalActions::Now, &tty);
    }

    /// Whether echo is on, `stdio_echo_state`.
    pub fn echo(&self) -> bool {
        self.echo.load(Ordering::Relaxed)
    }

    /// Puts the terminal back as it was.
    pub fn restore(&self) {
        if let Some(old) = &self.old {
            let _ =
                rustix::termios::tcsetattr(&self.fd, rustix::termios::OptionalActions::Now, old);
        }
    }
}

#[cfg(unix)]
impl Drop for Terminal {
    fn drop(&mut self) {
        self.restore();
    }
}

/// `term_exit()`: puts the terminal back and lets another chardev have stdio.
pub fn term_exit() {
    #[cfg(unix)]
    drop(lock(&TERMINAL).take());
    IN_USE.store(false, Ordering::Release);
}

/// `qemu_chr_fe_set_echo()` for the stdio chardev.
pub(crate) fn set_echo(echo: bool) {
    #[cfg(unix)]
    if let Some(t) = lock(&TERMINAL).as_ref() {
        t.set_echo(echo);
    }
    #[cfg(not(unix))]
    let _ = echo;
}

/// Held by the stdio chardev. Dropping it is `char_stdio_finalize()`.
#[derive(Debug)]
pub(crate) struct Claim(());

impl Drop for Claim {
    fn drop(&mut self) {
        term_exit();
    }
}

/// `stdio_chr_open()`. The claim goes with the chardev, the [`Local`] is its I/O.
pub(crate) fn open(opts: &ChardevStdio) -> Result<(Local, Claim)> {
    if IN_USE.swap(true, Ordering::AcqRel) {
        return Err(Error::generic("cannot use stdio by multiple character devices"));
    }
    let claim = Claim(());
    let allow_signal = opts.signal.unwrap_or(true);
    let local = open_io(allow_signal)?;
    Ok((local, claim))
}

#[cfg(unix)]
fn open_io(allow_signal: bool) -> Result<Local> {
    use std::os::fd::AsFd;

    use crate::local::FdSource;

    let dup = |e| Error::from_io("Failed to duplicate standard input", e);
    let input = std::io::stdin().as_fd().try_clone_to_owned().map_err(dup)?;
    let tty = input.try_clone().map_err(dup)?;
    *lock(&TERMINAL) = Some(Terminal::new(tty, allow_signal));
    Ok(Local::new(
        LocalKind::Stdio,
        Some(Box::new(FdSource(std::fs::File::from(input)))),
        Box::new(std::io::stdout()),
    ))
}

/// The Windows console as it is, without raw mode: QEMU's win-stdio reads the console on a
/// thread of its own too.
#[cfg(not(unix))]
fn open_io(allow_signal: bool) -> Result<Local> {
    let _ = allow_signal;
    let src = crate::local::ThreadSource::spawn("chardev-stdio".to_string(), std::io::stdin())
        .map_err(|e| Error::from_io("Failed to start the stdio thread", e))?;
    Ok(Local::new(LocalKind::Stdio, Some(Box::new(src)), Box::new(std::io::stdout())))
}
