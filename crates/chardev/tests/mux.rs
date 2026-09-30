// SPDX-License-Identifier: GPL-2.0-or-later

//! The mux, much of it ported from `char_mux_test` in QEMU's tests/unit/test-char.c.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{Rec, settle, wait_until};
use ruvm_chardev::mux::{MAX_MUX, help_text};
use ruvm_chardev::opts::chardev_opts;
use ruvm_chardev::{Chardev, Chardevs, ChrEvent};

fn mux_on_ringbuf(chardevs: &Chardevs, id: &str) -> Arc<Chardev> {
    mux_on_ringbuf_sized(chardevs, id, 128)
}

fn mux_on_ringbuf_sized(chardevs: &Chardevs, id: &str, size: usize) -> Arc<Chardev> {
    let mut list = chardev_opts();
    let opts = list.parse(&format!("ringbuf,id={id},size={size},mux=on"), true).unwrap();
    chardevs.new_from_opts(opts).unwrap().unwrap()
}

/// Waits until the mux wrote `n` bytes to its ringbuf backend and takes them.
fn collect(chardevs: &Chardevs, id: &str, n: usize) -> String {
    let got = std::sync::Mutex::new(String::new());
    wait_until(|| {
        let mut g = got.lock().unwrap();
        g.push_str(&output(chardevs, id));
        g.len() >= n
    });
    got.into_inner().unwrap()
}

/// What the mux wrote to its ringbuf backend so far.
fn output(chardevs: &Chardevs, id: &str) -> String {
    chardevs.ringbuf_read(&format!("{id}-base"), 1024, None).unwrap()
}

#[test]
fn char_mux_test() {
    let chardevs = Chardevs::new();
    let chr = mux_on_ringbuf(&chardevs, "mux-label");
    assert!(chr.is_mux());
    assert_eq!(chr.typename(), "chardev-mux");
    chardevs.remove("mux-label").unwrap();
    chardevs.remove("mux-label-base").unwrap();

    let chr = mux_on_ringbuf(&chardevs, "mux-label");
    let base = chardevs.find("mux-label-base").unwrap();
    assert!(Arc::ptr_eq(chr.mux_base().unwrap(), &base));
    // The mux is the frontend of its backend.
    assert_eq!(
        chardevs.remove("mux-label-base").unwrap_err().message(),
        "Chardev 'mux-label-base' is busy"
    );

    let (h1, h2) = (Rec::new(), Rec::new());
    let fe1 = chr.attach(h1.clone()).unwrap();
    let fe2 = chr.attach(h2.clone()).unwrap();
    fe2.take_focus();

    base.be_write(b"hello\0");
    assert_eq!(h2.wait_bytes(6), b"hello\0");
    assert_eq!(h1.len(), 0);
    assert_eq!(h1.last_event(), Some(ChrEvent::MuxOut));
    assert_eq!(h2.last_event(), Some(ChrEvent::MuxIn));

    // An event on the backend goes to every frontend, one on the mux to the focused one.
    base.be_event(ChrEvent::Opened);
    assert_eq!(h1.last_event(), Some(ChrEvent::Opened));
    assert_eq!(h2.last_event(), Some(ChrEvent::Opened));
    chr.be_event(ChrEvent::Closed);
    assert_eq!(h1.last_event(), Some(ChrEvent::Opened));
    assert_eq!(h2.last_event(), Some(ChrEvent::Closed));

    base.be_write(b"\x01b");
    wait_until(|| h2.last_event() == Some(ChrEvent::Break));
    assert_eq!(h1.last_event(), Some(ChrEvent::Opened));

    // Switch the focus.
    base.be_write(b"\x01c");
    wait_until(|| h1.last_event() == Some(ChrEvent::MuxIn));
    assert_eq!(h2.last_event(), Some(ChrEvent::MuxOut));
    chr.be_event(ChrEvent::Closed);
    assert_eq!(h1.last_event(), Some(ChrEvent::Closed));
    assert_eq!(h2.last_event(), Some(ChrEvent::MuxOut));

    base.be_write(b"hello\0");
    assert_eq!(h1.wait_bytes(6), b"hello\0");
    assert_eq!(h2.len(), 0);

    base.be_write(b"\x01b");
    wait_until(|| h1.last_event() == Some(ChrEvent::Break));
    assert_eq!(h2.last_event(), Some(ChrEvent::MuxOut));

    // Both frontends have a connection while the backend is open.
    assert!(chr.frontend_open());
    assert_eq!(h1.opens.load(Ordering::SeqCst), 1);
    assert_eq!(h2.opens.load(Ordering::SeqCst), 1);

    // With the focused frontend gone, input goes nowhere until the focus moves on.
    drop(fe1);
    assert!(!chr.frontend_open());
    base.be_write(b"hello\0");
    settle();
    assert_eq!((h1.len(), h2.len()), (0, 0));
    base.be_write(b"\x01c");
    base.be_write(b"hello\0");
    assert_eq!(h2.wait_bytes(6), b"hello\0");
    assert_eq!(h1.len(), 0);

    // Help goes out through the backend.
    base.be_write(b"\x01?");
    assert!(!collect(&chardevs, "mux-label", 1).is_empty());

    assert_eq!(chardevs.remove("mux-label").unwrap_err().message(), "Chardev 'mux-label' is busy");
    drop(fe2);
    chardevs.remove("mux-label").unwrap();
    chardevs.remove("mux-label-base").unwrap();
}

#[test]
fn help() {
    let want = "\n\r\
                C-a h    print this help\n\r\
                C-a x    exit emulator\n\r\
                C-a s    save disk data back to file (if -snapshot)\n\r\
                C-a t    toggle console timestamps\n\r\
                C-a b    send break (magic sysrq)\n\r\
                C-a c    switch between console and monitor\n\r\
                C-a C-a  sends C-a\n\r";
    assert_eq!(help_text(1), want);
    assert!(help_text(0x1d).starts_with(
        "\n\rEscape-Char set to Ascii: 0x1d\n\r\n\rEscape-Char h    print this help\n\r"
    ));
    assert!(help_text(5).ends_with("C-e C-e  sends C-e\n\r"));

    let chardevs = Chardevs::new();
    let chr = mux_on_ringbuf_sized(&chardevs, "m", 1024);
    let base = chardevs.find("m-base").unwrap();
    let _fe = chr.attach(Rec::new()).unwrap();
    base.be_write(b"\x01h");
    assert_eq!(collect(&chardevs, "m", want.len()), want);

    // Another escape character.
    chardevs.set_escape_char(0x05);
    assert_eq!(chardevs.escape_char(), 5);
    base.be_write(b"\x05h");
    assert_eq!(collect(&chardevs, "m", want.len()), help_text(5));
}

#[test]
fn escapes() {
    let chardevs = Chardevs::new();
    let quits = Arc::new(AtomicUsize::new(0));
    let commits = Arc::new(AtomicUsize::new(0));
    let q = quits.clone();
    chardevs.set_mux_quit_handler(Arc::new(move || {
        q.fetch_add(1, Ordering::SeqCst);
    }));
    let c = commits.clone();
    chardevs.set_mux_commit_handler(Arc::new(move || {
        c.fetch_add(1, Ordering::SeqCst);
    }));
    let chr = mux_on_ringbuf(&chardevs, "m");
    let base = chardevs.find("m-base").unwrap();
    let rec = Rec::new();
    let fe = chr.attach(rec.clone()).unwrap();

    // C-a C-a sends one C-a, other letters do nothing.
    base.be_write(b"a\x01\x01b\x01zc");
    assert_eq!(rec.wait_bytes(4), b"a\x01bc");

    base.be_write(b"\x01s");
    wait_until(|| commits.load(Ordering::SeqCst) == 1);

    base.be_write(b"\x01x");
    wait_until(|| quits.load(Ordering::SeqCst) == 1);
    assert_eq!(output(&chardevs, "m"), "QEMU: Terminated\n\r");

    // Timestamps start with the next line.
    base.be_write(b"\x01t!");
    assert_eq!(rec.wait_bytes(1), b"!");
    fe.write_all(b"a\nb\n").unwrap();
    let out = output(&chardevs, "m");
    assert!(out.starts_with("a\n[00:00:00."), "{out:?}");
    assert!(out.ends_with("] b\n"), "{out:?}");
    assert_eq!(out.len(), "a\n[00:00:00.000] b\n".len());
    base.be_write(b"\x01t!");
    assert_eq!(rec.wait_bytes(1), b"!");
    fe.write_all(b"c\n").unwrap();
    assert_eq!(output(&chardevs, "m"), "c\n");
}

#[test]
fn focus_and_limits() {
    let chardevs = Chardevs::new();
    let chr = mux_on_ringbuf(&chardevs, "m");
    let base = chardevs.find("m-base").unwrap();
    let recs: Vec<Arc<Rec>> = (0..MAX_MUX).map(|_| Rec::new()).collect();
    let fes: Vec<_> = recs.iter().map(|r| chr.attach(r.clone()).unwrap()).collect();
    let e = chr.attach(Rec::new()).unwrap_err();
    assert_eq!(e.message(), "too many uses of multiplexed chardev 'm' (maximum is 4)");

    // The last one to attach has the focus, and C-a c goes round.
    for (k, i) in [3, 0, 1, 2, 3].into_iter().enumerate() {
        if k > 0 {
            base.be_write(b"\x01c");
        }
        base.be_write(&[b'0' + i as u8]);
        assert_eq!(recs[i].wait_bytes(1), [b'0' + i as u8]);
    }
    fes[2].take_focus();
    base.be_write(b"x");
    assert_eq!(recs[2].wait_bytes(1), b"x");

    // What each frontend writes goes out through the backend.
    fes[1].write_all(b"one ").unwrap();
    fes[3].write_all(b"three").unwrap();
    assert_eq!(output(&chardevs, "m"), "one three");
    for r in &recs {
        assert_eq!(r.len(), 0);
    }
}

#[test]
fn compat_names() {
    let chardevs = Chardevs::new();
    let mut list = chardev_opts();

    // `mon:` makes a mux for the caller to put a monitor on.
    let (chr, mux) = chardevs.new_from_name(&mut list, "serial0", "mon:null", true).unwrap();
    assert!(mux && chr.is_mux());
    assert_eq!(chr.label(), "serial0");
    assert!(chardevs.find("serial0-base").is_some());
    assert!(list.find(Some("serial0")).is_none());

    let (again, mux) = chardevs.new_from_name(&mut list, "x", "chardev:serial0", true).unwrap();
    assert!(!mux && Arc::ptr_eq(&again, &chr));
    assert!(chardevs.new_from_name(&mut list, "x", "chardev:nope", true).unwrap_err().is_none());

    let e = chardevs.new_from_name(&mut list, "p", "mon:null", false).unwrap_err().unwrap();
    assert_eq!(e.message(), "mon: isn't supported in this context");
    let e = chardevs.new_from_name(&mut list, "v", "vc", false).unwrap_err().unwrap();
    assert_eq!(e.message(), "chardev backend 'vc' is not supported by ruvm yet");
    let e = chardevs.new_from_name(&mut list, "b", "bogus", false).unwrap_err().unwrap();
    assert_eq!(e.message(), "'bogus' is not a valid char driver");
    assert!(chardevs.new_from_name(&mut list, "t", "tcp:nope", false).unwrap_err().is_none());

    // A mux on a missing backend.
    let mut list = chardev_opts();
    let opts = list.parse("mux,id=mx,chardev=gone", true).unwrap();
    let e = chardevs.new_from_opts(opts).unwrap_err();
    assert_eq!(e.message(), "mux: base chardev gone not found");
}
