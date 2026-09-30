// SPDX-License-Identifier: GPL-2.0-or-later

//! Golden tests: the same command lines through QEMU's qemu-io and through `ruvm` called as
//! qemu-io must give the same standard output, standard error, exit status and image contents.
//! Only the timing parts of the statistics lines and the paths of the images are normalized.
//! The tests skip themselves when QEMU's tools or the `ruvm` binary are missing.

#![cfg(unix)]

use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

const QEMU_IO: &str = "/opt/homebrew/bin/qemu-io";
const QEMU_IMG: &str = "/opt/homebrew/bin/qemu-img";

fn ruvm() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/ruvm")
}

/// Whether everything the tests run is there; prints why not otherwise.
fn available() -> bool {
    for p in [PathBuf::from(QEMU_IO), PathBuf::from(QEMU_IMG), ruvm()] {
        if !p.exists() {
            eprintln!("skipping: {} is missing", p.display());
            return false;
        }
    }
    true
}

/// A scratch directory that is removed at the end of the test.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ruvm-io-golden-{}-{name}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn qemu_img(args: &[&str]) {
    let st = Command::new(QEMU_IMG).args(args).stdout(Stdio::null()).status().unwrap();
    assert!(st.success(), "qemu-img {args:?} failed");
}

/// The part of a statistics line that depends on how long the request took.
fn normalize_line(line: &str) -> String {
    // "512 bytes, 1 ops; 00.00 sec (1.234 MiB/sec and 2000.0000 ops/sec)"
    if let Some(i) = line.find(" ops; ") {
        if line.ends_with(" ops/sec)") {
            return format!("{} ops; TIME", &line[..i]);
        }
    }
    // "-C": "512,1,0:00:00.00,12345.678,24.113"
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() == 5
        && fields[..2].iter().all(|f| !f.is_empty() && f.bytes().all(|b| b.is_ascii_digit()))
        && fields[2].contains(':')
    {
        return format!("{},{},TIME", fields[0], fields[1]);
    }
    line.to_string()
}

fn normalize(out: &[u8], dir: &Path) -> String {
    let s = String::from_utf8_lossy(out).replace(&*dir.to_string_lossy(), "$DIR");
    s.split_inclusive('\n')
        .map(|l| {
            let (body, nl) = l.strip_suffix('\n').map_or((l, ""), |b| (b, "\n"));
            normalize_line(body) + nl
        })
        .collect()
}

/// What one run left behind.
#[derive(Debug, PartialEq)]
struct Outcome {
    status: Option<i32>,
    stdout: String,
    stderr: String,
    image: Vec<u8>,
}

/// Runs `bin` as `qemu-io` with `args` in a fresh copy of `image` called `x.img`, where
/// `$IMG` in `args` stands for its path.
fn run_one(bin: &Path, image: &Path, args: &[&str], stdin: &str, scratch: &Scratch) -> Outcome {
    let dir = scratch.path(if bin == Path::new(QEMU_IO) { "qemu" } else { "ruvm" });
    std::fs::create_dir_all(&dir).unwrap();
    let img = dir.join("x.img");
    std::fs::copy(image, &img).unwrap();
    let img_s = img.to_string_lossy().into_owned();
    let args: Vec<String> = args.iter().map(|a| a.replace("$IMG", &img_s)).collect();
    let mut child = Command::new(bin)
        .arg0("qemu-io")
        .args(&args)
        .current_dir(&dir)
        .env_remove("POSIXLY_CORRECT")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        use std::io::Write;
        let mut input = child.stdin.take().unwrap();
        let _ = input.write_all(stdin.as_bytes());
    }
    let out = child.wait_with_output().unwrap();
    Outcome {
        status: out.status.code(),
        stdout: normalize(&out.stdout, &dir),
        stderr: normalize(&out.stderr, &dir),
        image: std::fs::read(&img).unwrap(),
    }
}

/// Runs every case through both programs and fails with all the differences.
fn compare(name: &str, image: &Path, cases: &[(&[&str], &str)]) {
    let scratch = Scratch::new(name);
    let mut failures = Vec::new();
    for (args, stdin) in cases {
        let want = run_one(Path::new(QEMU_IO), image, args, stdin, &scratch);
        let got = run_one(&ruvm(), image, args, stdin, &scratch);
        if want.status != got.status || want.stdout != got.stdout || want.stderr != got.stderr {
            failures.push(format!(
                "{args:?}:\n  qemu: {:?}\n{}{}  ruvm: {:?}\n{}{}",
                want.status, want.stdout, want.stderr, got.status, got.stdout, got.stderr
            ));
        } else if want.image != got.image {
            failures.push(format!("{args:?}: the images differ"));
        }
    }
    assert!(failures.is_empty(), "{} case(s) differ:\n{}", failures.len(), failures.join("\n"));
}

/// `-c` for each of `cmds`, then the image options.
fn with_cmds<'a>(cmds: &[&'a str], image: &[&'a str]) -> Vec<&'a str> {
    let mut v = Vec::new();
    for c in cmds {
        v.push("-c");
        v.push(*c);
    }
    v.extend_from_slice(image);
    v
}

/// Commands that are checked one by one on every format.
const COMMANDS: &[&str] = &[
    "info",
    "length",
    "map",
    "alloc 0",
    "alloc 0 1M",
    "alloc 0 3M",
    "read 0 512",
    "read -v 500 40",
    "read -C 0 512",
    "read -P 0 0 64k",
    "read -P 1 0 512",
    "read -P 0 -s 10 -l 5 0 512",
    "read -l 1 0 1",
    "read -q 0 1",
    "read 0 1 -q",
    "read -- 0 1",
    "read 0 3G",
    "read 2M 1",
    "read -b 0 512",
    "read -b 1 512",
    "write -P 0x12 512 4k",
    "write -z 0 64k",
    "write -z -u 0 64k",
    "write -f 0 1k",
    "write -c 0 512",
    "write -b 0 512",
    "write -b -z 0 1",
    "write -n 0 1",
    "write -z -P 1 0 1",
    "write -s /nonexistent 0 1",
    "write -s /dev/null 0 1",
    "write 0 3G",
    "writev -P 3 1k 1 2 3",
    "readv -P 0 0 512 512",
    "readv 0 2G 2G",
    "aio_read -P 0 0 512",
    "aio_read -i 0 1",
    "aio_write -P 5 0 1k",
    "aio_write -z 0 1k",
    "aio_write -z 0 1 2",
    "aio_write -i 0 1",
    "aio_flush",
    "flush",
    "discard 0 64k",
    "aio_discard -C 0 1k",
    "truncate 1M",
    "truncate -m bad 1M",
    "reopen -r",
    "reopen -r -w",
    "reopen -c none",
    "reopen -c bad",
    "reopen x",
    "reopen -o bogus=1",
    "break a b",
    "remove_break a",
    "zone_report 0 1",
    "zo 0 512",
    "zap 0 512 512",
    "zf 1 2",
    "sleep x",
    "sigraise 100",
    "open x",
    "close",
    "nope",
    "read",
    "alloc 1 2 3",
];

#[test]
fn raw() {
    if !available() {
        return;
    }
    let scratch = Scratch::new("raw-base");
    let base = scratch.path("base.raw");
    qemu_img(&["create", "-f", "raw", &base.to_string_lossy(), "2M"]);
    std::fs::write(scratch.path("pattern"), b"abc").unwrap();
    let pattern = scratch.path("pattern").to_string_lossy().into_owned();
    let write_s = format!("write -s {pattern} 0 1k");

    let mut owned: Vec<Vec<&str>> = COMMANDS.iter().map(|c| with_cmds(&[c], &["$IMG"])).collect();
    owned.push(with_cmds(&["write -P 7 0 1k", "read -v 0 64", "map"], &["-f", "raw", "$IMG"]));
    owned.push(with_cmds(&[&write_s, "read -v 0 32"], &["-f", "raw", "$IMG"]));
    owned.push(with_cmds(&["help"], &[]));
    for c in [
        "help read",
        "help readv",
        "help write",
        "help writev",
        "help aio_read",
        "help aio_write",
        "help discard",
        "help aio_discard",
        "help open",
        "help reopen",
        "help sigraise",
        "help nope",
        "help ?",
    ] {
        owned.push(with_cmds(&[c], &[]));
    }
    for args in [
        &["-h"][..],
        &["-x"],
        &["a", "b"],
        &["-f", "raw", "--image-opts", "$IMG"],
        &["-t", "bad", "$IMG"],
        &["-d", "bad", "$IMG"],
        &["-i", "bad", "$IMG"],
        &["nonexistent"],
        &["-r", "-f", "raw", "$IMG", "-c", "write 0 1"],
        &["-r", "-f", "raw", "$IMG", "-c", "reopen -w", "-c", "write 0 1"],
        &["-U", "-f", "raw", "$IMG", "-c", "info"],
        &["-n", "-f", "raw", "$IMG", "-c", "write -f 0 1"],
        &["-t", "writethrough", "-d", "unmap", "-f", "raw", "$IMG", "-c", "discard 0 1k"],
        &["-C", "-k", "-m", "-f", "raw", "$IMG", "-c", "read 0 1"],
        &["-c", "open -r -f raw $IMG", "-c", "write 0 1"],
        &["-c", "open -o driver=raw $IMG", "-c", "info"],
        &["-c", "open -o driver=raw -o read-only=on $IMG"],
        &[
            "-f",
            "raw",
            "$IMG",
            "-c",
            "reopen -o read-only=off -r",
            "-c",
            "reopen -o cache.direct=on -c none",
            "-c",
            "reopen -r",
            "-c",
            "write 0 1",
            "-c",
            "reopen -w",
            "-c",
            "write 0 1",
            "-c",
            "reopen -c writeback -o bad=x",
        ],
        &["-c", "open -t bad", "-c", "open -x"],
        &["-c", "open $IMG", "-c", "open $IMG"],
        &["--image-opts", "-c", "open driver=raw,file.filename=$IMG", "-c", "info"],
        &["--image-opts", "-c", "open -o driver=raw $IMG"],
        &["-U", "--image-opts", "driver=raw,file.filename=$IMG,force-share=off"],
        &["-f", "raw", "$IMG", "-c", "q", "-c", "read 0 1"],
        &["--object", "bogus,id=x"],
    ] {
        owned.push(args.to_vec());
    }
    let mut cases: Vec<(&[&str], &str)> = owned.iter().map(|a| (a.as_slice(), "")).collect();
    let interactive = ["-f", "raw", "$IMG"];
    cases.push((&interactive, "read 0 1\nhelp nope\n\n   \nwrite -P 1 0 1\nquit\nread 0 1\n"));
    cases.push((&interactive, "read -P 0 0 1"));
    compare("raw", &base, &cases);
}

#[test]
fn luks() {
    if !available() {
        return;
    }
    let scratch = Scratch::new("luks-base");
    let base = scratch.path("base.luks");
    qemu_img(&[
        "create",
        "-f",
        "luks",
        "--object",
        "secret,id=sec0,data=hunter2",
        "-o",
        "key-secret=sec0,iter-time=10",
        &base.to_string_lossy(),
        "1M",
    ]);
    let open = [
        "--object",
        "secret,id=sec0,data=hunter2",
        "--image-opts",
        "driver=luks,key-secret=sec0,file.filename=$IMG",
    ];
    let mut owned: Vec<Vec<&str>> = COMMANDS.iter().map(|c| with_cmds(&[c], &open)).collect();
    owned
        .push(with_cmds(&["write -P 7 1000 5000", "read -P 7 1000 5000", "read -v 990 32"], &open));
    owned.push(with_cmds(&["reopen -o key-secret=sec0", "truncate 2M", "length"], &open));
    owned.push(vec![
        "--object",
        "secret,id=sec0,data=wrong",
        "--image-opts",
        "driver=luks,key-secret=sec0,file.filename=$IMG",
    ]);
    owned.push(vec!["--image-opts", "driver=luks,file.filename=$IMG"]);
    owned.push(vec!["$IMG", "-c", "info"]);
    let cases: Vec<(&[&str], &str)> = owned.iter().map(|a| (a.as_slice(), "")).collect();
    compare("luks", &base, &cases);
}

#[test]
fn qcow2() {
    if !ruvm_block::tools::format_exists("qcow2") {
        eprintln!("skipping: there is no qcow2 driver");
        return;
    }
    if !available() {
        return;
    }
    let scratch = Scratch::new("qcow2-base");
    let base = scratch.path("base.qcow2");
    qemu_img(&["create", "-f", "qcow2", &base.to_string_lossy(), "2M"]);
    let open = ["-f", "qcow2", "$IMG"];
    let mut owned: Vec<Vec<&str>> = COMMANDS.iter().map(|c| with_cmds(&[c], &open)).collect();
    owned.push(with_cmds(&["write -P 1 64k 64k", "write -z 192k 64k", "map", "alloc 0 1M"], &open));
    owned.push(with_cmds(&["write -c 0 64k", "read -P 0xcd 0 64k"], &open));
    owned.push(with_cmds(&["write -b 0 512", "read -b -v 0 32"], &open));
    owned.push(with_cmds(&["reopen -o lazy-refcounts=on", "info"], &open));
    let cases: Vec<(&[&str], &str)> = owned.iter().map(|a| (a.as_slice(), "")).collect();
    compare("qcow2", &base, &cases);
}

#[test]
fn statistics_lines_are_normalized() {
    assert_eq!(
        normalize_line("512 bytes, 1 ops; 00.00 sec (1.234 MiB/sec and 2000.0000 ops/sec)"),
        "512 bytes, 1 ops; TIME"
    );
    assert_eq!(normalize_line("512,1,0:00:00.00,12345.678,24.113"), "512,1,TIME");
    assert_eq!(normalize_line("read 512/512 bytes at offset 0"), "read 512/512 bytes at offset 0");
    assert_eq!(normalize(b"x /d/x.img\n", Path::new("/d")), "x $DIR/x.img\n");
}
