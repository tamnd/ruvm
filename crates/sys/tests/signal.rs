// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(unix)]

use std::process::Command;
use std::sync::mpsc;
use std::time::Duration;

use ruvm_sys::signal::{Killed, on_termination};

#[test]
fn sigterm_from_another_process() {
    let (tx, rx) = mpsc::channel();
    on_termination(move |k| {
        let _ = tx.send(k);
    })
    .unwrap();
    let me = std::process::id();
    let mut kill = Command::new("kill").arg("-TERM").arg(me.to_string()).spawn().unwrap();
    let kill_pid = kill.id() as i32;
    let got = rx.recv_timeout(Duration::from_secs(10)).unwrap();
    kill.wait().unwrap();
    assert_eq!(got, Killed { signo: libc::SIGTERM, pid: kill_pid });
}
