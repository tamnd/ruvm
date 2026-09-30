// SPDX-License-Identifier: GPL-2.0-or-later

//! util/qemu-progress.c: the `-p` progress line.
//!
//! Without `-p` QEMU prints the progress to stderr when it gets `SIGUSR1` or `SIGINFO`. There
//! is no signal handler here, so without `-p` nothing is printed.

use std::io::Write;
use std::sync::Mutex;

struct State {
    enabled: bool,
    current: f32,
    last_print: f32,
    min_skip: f32,
}

static STATE: Mutex<State> =
    Mutex::new(State { enabled: false, current: 0.0, last_print: 0.0, min_skip: 0.0 });

/// `qemu_progress_init()`.
pub(crate) fn init(enabled: bool, min_skip: f32) {
    let mut s = STATE.lock().unwrap();
    *s = State { enabled, current: 0.0, last_print: 0.0, min_skip };
}

/// `qemu_progress_end()`.
pub(crate) fn end() {
    if STATE.lock().unwrap().enabled {
        println!();
    }
}

/// `qemu_progress_print()`: `delta` is the new percentage with `max` 0, otherwise it adds
/// `delta` percent of `max`.
pub(crate) fn print(delta: f32, max: i32) {
    let mut s = STATE.lock().unwrap();
    let mut current = if max == 0 { delta } else { s.current + delta / 100.0 * max as f32 };
    if current > 100.0 {
        current = 100.0;
    }
    s.current = current;
    if current > s.last_print + s.min_skip
        || current < s.last_print - s.min_skip
        || current == 100.0
        || current == 0.0
    {
        s.last_print = current;
        if s.enabled {
            print!("    ({current:3.2}/100%)\r");
            let _ = std::io::stdout().flush();
        }
    }
}
