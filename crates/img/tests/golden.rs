// SPDX-License-Identifier: GPL-2.0-or-later

//! Golden tests: the same qemu-img command lines through QEMU's qemu-img and through ours
//! must give the same standard output, standard error, exit status and images.
//!
//! Every case runs in two fresh copies of a set of images made by QEMU's tools, one for each
//! program, with the images named relative to the working directory. Only the parts that
//! cannot match are normalized: timings, the scratch directory and generated node names.
//! Images that are not byte for byte the same must at least hold the same data and map the
//! same way according to QEMU's qemu-img. The tests skip themselves when QEMU's tools are
//! not installed.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

const QEMU_IMG: &str = "/opt/homebrew/bin/qemu-img";
const QEMU_IO: &str = "/opt/homebrew/bin/qemu-io";
const OURS: &str = env!("CARGO_BIN_EXE_ruvm-qemu-img");

fn available() -> bool {
    for p in [QEMU_IMG, QEMU_IO] {
        if !Path::new(p).exists() {
            eprintln!("skipping: {p} is missing");
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
            "ruvm-img-golden-{}-{name}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run_ok(dir: &Path, bin: &str, args: &[&str]) {
    let out = Command::new(bin).args(args).current_dir(dir).output().unwrap();
    assert!(out.status.success(), "{bin} {args:?}: {}", String::from_utf8_lossy(&out.stderr));
}

fn io(dir: &Path, args: &[&str]) {
    run_ok(dir, QEMU_IO, args);
}

const SECRET: &str = "secret,id=s0,data=abc";

/// The images every case starts from, all made by QEMU.
fn make_images(dir: &Path) {
    let img = |args: &[&str]| run_ok(dir, QEMU_IMG, args);
    img(&["create", "-q", "-f", "raw", "a.raw", "4M"]);
    io(dir, &["-f", "raw", "-c", "write -P 0x11 0 64k", "-c", "write -P 0x22 1M 128k", "a.raw"]);
    img(&["create", "-q", "-f", "raw", "z.raw", "1M"]);
    img(&["create", "-q", "-f", "raw", "c.raw", "4M"]);
    io(dir, &["-f", "raw", "-c", "write -P 0x11 0 64k", "-c", "write -P 0x23 1M 4k", "c.raw"]);
    img(&["create", "-q", "-f", "qed", "-b", "a.raw", "-F", "raw", "b.qed"]);
    io(dir, &["-f", "qed", "-c", "write -P 0x33 2M 64k", "b.qed"]);
    img(&["create", "-q", "-f", "qed", "-b", "b.qed", "-F", "qed", "t.qed"]);
    io(dir, &["-f", "qed", "-c", "write -P 0x44 1M 4k", "-c", "write -z 3M 64k", "t.qed"]);
    img(&["create", "-q", "-f", "qed", "e.qed", "4M"]);
    img(&[
        "create",
        "-q",
        "--object",
        SECRET,
        "-f",
        "luks",
        "-o",
        "key-secret=s0,iter-time=10",
        "l.luks",
        "4M",
    ]);
    io(
        dir,
        &[
            "--object",
            SECRET,
            "--image-opts",
            "-c",
            "write -P 0x55 0 64k",
            "driver=luks,key-secret=s0,file.filename=l.luks",
        ],
    );
    if qcow2() {
        img(&["create", "-q", "-f", "qcow2", "-b", "a.raw", "-F", "raw", "q.qcow2"]);
        io(dir, &["-f", "qcow2", "-c", "write -P 0x66 512k 64k", "q.qcow2"]);
    }
}

fn qcow2() -> bool {
    ruvm_block::tools::format_exists("qcow2")
}

fn normalize(s: &[u8], dir: &Path) -> String {
    let s = String::from_utf8_lossy(s).replace(&*dir.to_string_lossy(), "$DIR");
    let mut out = String::new();
    let mut in_formats = false;
    for line in s.split_inclusive('\n') {
        // Which formats there are depends on the drivers the block layer has so far.
        if line.starts_with("Supported image formats:") {
            in_formats = true;
            out.push_str(line);
            continue;
        }
        if in_formats {
            if line.starts_with("See <") {
                in_formats = false;
                out.push_str("  FORMATS\n\n");
            } else {
                continue;
            }
        }
        let line = if line.starts_with("qemu-img version ") {
            "qemu-img version VERSION\n".to_string()
        } else if line.trim_start().starts_with("cid: ")
            || line.trim_start().starts_with("\"cid\": ")
        {
            // VMDK content ids are random.
            let (head, _) = line.split_once("cid").unwrap();
            let comma = if line.trim_end().ends_with(',') { "," } else { "" };
            format!("{head}cid: CID{comma}\n")
        } else if line.starts_with("Run completed in ") {
            "Run completed in TIME seconds.\n".to_string()
        } else {
            node_names(line)
        };
        out.push_str(&line);
    }
    out
}

/// The lines of `what` that differ, QEMU's first.
fn diff(what: &str, q: &str, r: &str) -> String {
    if q == r {
        return String::new();
    }
    let mut out = format!("  {what}:\n");
    let (ql, rl): (Vec<_>, Vec<_>) = (q.lines().collect(), r.lines().collect());
    for i in 0..ql.len().max(rl.len()) {
        let (a, b) = (ql.get(i), rl.get(i));
        if a != b {
            out += &format!("    - {}\n    + {}\n", a.unwrap_or(&"<none>"), b.unwrap_or(&"<none>"));
        }
    }
    out
}

/// Generated node names, `#block123`, depend on how many nodes were made before.
fn node_names(line: &str) -> String {
    let mut out = String::new();
    let mut rest = line;
    while let Some(i) = rest.find("#block") {
        out.push_str(&rest[..i + 6]);
        rest = rest[i + 6..].trim_start_matches(|c: char| c.is_ascii_digit());
        out.push('N');
    }
    out.push_str(rest);
    out
}

#[derive(Debug, PartialEq)]
struct Step {
    args: String,
    status: Option<i32>,
    stdout: String,
    stderr: String,
}

fn run_case(bin: &str, dir: &Path, case: &[&str]) -> Vec<Step> {
    case.iter()
        .map(|line| {
            // `''` stands for an empty argument.
            let args: Vec<&str> =
                line.split_whitespace().map(|a| if a == "''" { "" } else { a }).collect();
            let out = Command::new(bin)
                .args(&args)
                .current_dir(dir)
                .stdin(Stdio::null())
                .output()
                .unwrap();
            Step {
                args: line.to_string(),
                status: out.status.code(),
                stdout: normalize(&out.stdout, dir).replace(QEMU_IMG, "qemu-img"),
                // QEMU names itself by the path it was run with.
                stderr: normalize(&out.stderr, dir).replace(QEMU_IMG, "qemu-img"),
            }
        })
        .collect()
}

/// Whether two images hold the same thing, as QEMU sees it.
fn same_image(a: &Path, b: &Path) -> Result<(), String> {
    if std::fs::read(a).unwrap() == std::fs::read(b).unwrap() {
        return Ok(());
    }
    let name = a.file_name().unwrap().to_string_lossy().into_owned();
    // Encrypted images differ in their keys and salts; only their contents can be compared.
    let (fa, fb) = if name.ends_with(".luks") {
        let o = |p: &Path| format!("driver=luks,key-secret=s0,file.filename={}", p.display());
        (o(a), o(b))
    } else {
        (a.display().to_string(), b.display().to_string())
    };
    let mut args = vec!["compare", "-U"];
    if name.ends_with(".luks") {
        args.extend(["--object", SECRET, "--image-opts"]);
    }
    let out = Command::new(QEMU_IMG).args(&args).args([&fa, &fb]).output().unwrap();
    if !out.status.success() {
        return Err(format!("{name}: {}", String::from_utf8_lossy(&out.stdout)));
    }
    if name.ends_with(".luks") {
        return Ok(());
    }
    let map = |p: &Path| {
        let out = Command::new(QEMU_IMG)
            .args(["map", "-U", "--output=json"])
            .arg(p)
            .current_dir(p.parent().unwrap())
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let (ma, mb) = (map(a), map(b));
    if ma != mb {
        return Err(format!("{name}: maps differ:\n{ma}\n{mb}"));
    }
    Ok(())
}

/// Runs each case through both programs and fails with every difference.
fn check(name: &str, cases: &[&[&str]]) {
    if !available() {
        return;
    }
    let scratch = Scratch::new(name);
    let template = scratch.0.join("template");
    std::fs::create_dir_all(&template).unwrap();
    make_images(&template);

    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut results = Vec::new();
        let mut dirs = Vec::new();
        for (who, bin) in [("qemu", QEMU_IMG), ("ruvm", OURS)] {
            // Both copies live at paths of the same length, so the texts line up.
            let dir = scratch.0.join(format!("{i:03}-{who}"));
            std::fs::create_dir_all(&dir).unwrap();
            for e in std::fs::read_dir(&template).unwrap() {
                let e = e.unwrap();
                std::fs::copy(e.path(), dir.join(e.file_name())).unwrap();
            }
            results.push(run_case(bin, &dir, case));
            dirs.push(dir);
        }
        if results[0] != results[1] {
            for (q, r) in results[0].iter().zip(&results[1]) {
                if q != r {
                    failures.push(format!(
                        "case {case:?}, step {:?}\n  status {:?} vs {:?}\n{}{}",
                        q.args,
                        q.status,
                        r.status,
                        diff("stdout", &q.stdout, &r.stdout),
                        diff("stderr", &q.stderr, &r.stderr)
                    ));
                    break;
                }
            }
            continue;
        }
        let mut names: Vec<_> =
            std::fs::read_dir(&dirs[0]).unwrap().map(|e| e.unwrap().file_name()).collect();
        names.sort();
        for n in names {
            let (q, r) = (dirs[0].join(&n), dirs[1].join(&n));
            if !r.exists() {
                failures.push(format!("case {case:?}: {} was not made", n.to_string_lossy()));
            } else if let Err(e) = same_image(&q, &r) {
                failures.push(format!("case {case:?}: {e}"));
            }
        }
    }
    assert!(failures.is_empty(), "{} differences:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn help_and_usage() {
    check(
        "help",
        &[
            &["-h"],
            &["--help"],
            &["-V"],
            &[""],
            &["nosuchcmd"],
            &["create -h"],
            &["convert --help"],
            &["info -h"],
            &["check -h"],
            &["commit -h"],
            &["compare -h"],
            &["map -h"],
            &["measure -h"],
            &["rebase -h"],
            &["resize -h"],
            &["snapshot -h"],
            &["amend -h"],
            &["bench -h"],
            &["bitmap -h"],
            &["dd -h"],
            &["info"],
            &["info -x a.raw"],
            &["info --bogus a.raw"],
            &["info a.raw z.raw"],
        ],
    );
}

#[test]
fn create_and_info() {
    check(
        "create",
        &[
            &["create -f raw n.raw 1M", "info n.raw", "info --output=json n.raw"],
            &["create -f qed n.qed 8M", "info --output=json n.qed"],
            &["create -f qed -b a.raw -F raw n.qed", "info --backing-chain n.qed"],
            &["create -f qed -o cluster_size=65536,table_size=4 n.qed 16M", "info n.qed"],
            &["create -f qed -o help"],
            &["create -f raw -o help"],
            &["create -f raw -o bogus=1 n.raw 1M"],
            &["create -f nosuch n.raw 1M"],
            &["create -f raw n.raw"],
            &["create -f raw n.raw 1Q"],
            &["create -f raw n.raw -1"],
            &["create -q -f qed -b nosuch.raw -F raw n.qed"],
            &["create -f qed -u -b nosuch.raw -F raw n.qed 1M", "info n.qed"],
            // What a new LUKS image reads before it is written depends on its key, so the
            // image is filled before it is compared.
            &[
                "create --object secret,id=s0,data=abc -f luks -o key-secret=s0,iter-time=10 n.luks \
                 4M",
                "convert -n --object secret,id=s0,data=abc --target-image-opts a.raw \
                 driver=luks,key-secret=s0,file.filename=n.luks",
            ],
            &["create -f luks n.luks 1M"],
            &["info --output=json --backing-chain t.qed"],
            &["info --output=json l.luks"],
            &["info nosuch.raw"],
            &["info -f qed a.raw"],
            &["info --output=xml a.raw"],
            &["info --image-opts driver=raw,file.filename=a.raw"],
            &["info -U -f raw a.raw"],
        ],
    );
}

#[test]
fn convert() {
    check(
        "convert",
        &[
            &["convert -O raw t.qed o.raw"],
            &["convert -O qed t.qed o.qed", "info o.qed"],
            &["convert -S 0 -O raw a.raw o.raw"],
            &["convert -S 4k -O raw a.raw o.raw"],
            &["convert -m 4 -W -O raw a.raw o.raw"],
            &["convert -m 0 -O raw a.raw o.raw"],
            &["convert -m 17 -O raw a.raw o.raw"],
            &["convert -B a.raw -F raw -O qed t.qed o.qed", "info o.qed"],
            &["convert -c -O raw a.raw o.raw"],
            &["convert -O qed a.raw z.raw o.qed", "info o.qed"],
            &["create -q -f raw o.raw 4M", "convert -n -O raw t.qed o.raw"],
            &["convert -n -O raw t.qed nosuch.raw"],
            &["convert -r 1M -O raw a.raw o.raw"],
            &["convert -r 0 -O raw a.raw o.raw"],
            &["convert --salvage -O raw t.qed o.raw"],
            &["convert -O raw nosuch.raw o.raw"],
            &["convert -O nosuch a.raw o.raw"],
            &["convert -O raw a.raw"],
            &["convert --bitmaps -O raw a.raw o.raw"],
            &["convert -p -O raw a.raw o.raw"],
            &[
                "create -q -f raw o.raw 4M",
                "convert -n --target-image-opts a.raw driver=raw,file.filename=o.raw",
            ],
            &["convert --target-image-opts a.raw driver=raw,file.filename=o.raw"],
            &["convert --object secret,id=s0,data=abc --image-opts \
           driver=luks,key-secret=s0,file.filename=l.luks -O raw o.raw"],
            &[
                "convert --object secret,id=s0,data=abc -O luks -o key-secret=s0,iter-time=10 a.raw \
           o.luks",
            ],
            &["convert -l snap -O raw a.raw o.raw"],
        ],
    );
}

#[test]
fn check_compare_map_measure() {
    check(
        "inspect",
        &[
            &["check t.qed"],
            &["check --output=json t.qed"],
            &["check a.raw"],
            &["check -r all e.qed"],
            &["check -r bogus e.qed"],
            &["compare a.raw a.raw"],
            &["compare a.raw c.raw"],
            &["compare -s a.raw z.raw"],
            &["compare a.raw z.raw"],
            &["compare -q a.raw c.raw"],
            &["compare a.raw nosuch.raw"],
            &["compare -f raw -F qed a.raw t.qed"],
            &["map t.qed"],
            &["map --output=json t.qed"],
            &["map --output=json -s 1M -l 2M t.qed"],
            &["map --output=json --start-offset=5M t.qed"],
            &["map --output=json a.raw"],
            &["measure -O raw a.raw"],
            &["measure -O qed --output=json t.qed"],
            &["measure -O raw --size 1G"],
            &["measure -O qed"],
            &["measure --output=json -O raw -o size=4M"],
        ],
    );
}

#[test]
fn resize_amend_snapshot() {
    check(
        "resize",
        &[
            &["resize a.raw 8M", "info a.raw"],
            &["resize a.raw +1M", "info a.raw"],
            &["resize a.raw -- -1M"],
            &["resize --shrink a.raw -- -1M", "info a.raw"],
            &["resize --preallocation=full z.raw 2M", "info z.raw"],
            &["resize -f qed e.qed 8M", "info e.qed"],
            &["resize a.raw"],
            &["resize a.raw 1Q"],
            &["amend -o size=8M e.qed", "info e.qed"],
            &["amend -o help -f qed"],
            &["amend -o bogus=1 e.qed"],
            &["amend a.raw"],
            &["snapshot -l a.raw"],
            &["snapshot -c s1 a.raw"],
            &["snapshot -l e.qed"],
        ],
    );
}

#[test]
fn rebase_and_commit() {
    check(
        "chain",
        &[
            &["rebase -u -b c.raw -F raw b.qed", "info b.qed"],
            &["rebase -b c.raw -F raw b.qed", "info b.qed"],
            &["rebase -b a.raw -F raw t.qed", "info t.qed"],
            &["rebase -b '' t.qed", "info t.qed"],
            &["rebase -b nosuch.raw -F raw t.qed"],
            &["rebase -u t.qed"],
            &["rebase -c -b a.raw -F raw t.qed"],
            &["commit t.qed", "info t.qed"],
            &["commit -d t.qed"],
            &["commit -b a.raw t.qed"],
            &["commit -b nosuch.raw t.qed"],
            &["commit -q b.qed"],
            &["commit e.qed"],
            &["commit a.raw"],
            &["commit -t bogus t.qed"],
        ],
    );
}

#[test]
fn dd_bench_bitmap() {
    check(
        "dd",
        &[
            &["dd if=a.raw of=o.raw"],
            &["dd -f raw if=a.raw of=o.raw bs=4k count=20"],
            &["dd if=a.raw of=o.raw bs=4k skip=10"],
            &["dd if=a.raw of=o.raw bs=4k skip=1000"],
            &["dd if=a.raw of=o.raw bs=3000 count=100 skip=3"],
            &["dd -O qed if=t.qed of=o.qed bs=64k", "info o.qed"],
            &["dd if=a.raw"],
            &["dd of=o.raw"],
            &["dd if=a.raw of=o.raw foo=1"],
            &["dd if=a.raw of=o.raw foo"],
            &["dd if=a.raw of=o.raw bs=0"],
            &["dd if=a.raw of=o.raw count=x"],
            &["dd if=a.raw of=o.raw skip=-1"],
            &["dd -O foo if=a.raw of=o.raw"],
            &["dd if=nosuch.raw of=o.raw"],
            &["dd --image-opts if=driver=raw,file.filename=a.raw of=o.raw"],
            &["bench -c 100 -d 4 a.raw"],
            &["bench -w -c 50 -d 2 -s 8k -S 16k --pattern 7 --flush-interval 4 z.raw"],
            &["bench -w -c 50 -d 4 --flush-interval 5 --no-drain z.raw"],
            &["bench -c 10 --flush-interval 4 a.raw"],
            &["bench -w -c 10 -d 8 --flush-interval 4 a.raw"],
            &["bench -i bogus a.raw"],
            &["bench -c 0 a.raw"],
            &["bench -t bogus a.raw"],
            &["bitmap a.raw b0"],
            &["bitmap --add a.raw"],
            &["bitmap --remove-all a.raw b0"],
            &["bitmap -g 64k --remove a.raw b0"],
            &["bitmap -F raw --add a.raw b0"],
            &["bitmap -b a.raw --add a.raw b0"],
            &["bitmap --add e.qed b0"],
        ],
    );
}

#[test]
fn qcow2_images() {
    if !qcow2() {
        eprintln!("skipping: the qcow2 driver is not registered");
        return;
    }
    check(
        "qcow2",
        &[
            &["info --output=json q.qcow2"],
            &["check q.qcow2"],
            &["map --output=json q.qcow2"],
            &["convert -O qcow2 q.qcow2 o.qcow2", "info o.qcow2"],
            &["convert -c -O qcow2 a.raw o.qcow2", "check o.qcow2"],
            &["create -f qcow2 n.qcow2 1G", "info --output=json n.qcow2"],
            &["measure -O qcow2 --output=json a.raw"],
            &["snapshot -c s1 q.qcow2"],
            &["bitmap --add q.qcow2 b0", "info q.qcow2"],
            &["commit q.qcow2", "info q.qcow2"],
            &["amend -o compat=0.10 q.qcow2", "info q.qcow2"],
        ],
    );
}

#[test]
fn other_formats() {
    for fmt in ["qcow", "vdi", "vpc", "vhdx", "vmdk", "parallels"] {
        if !ruvm_block::tools::format_exists(fmt) {
            eprintln!("skipping {fmt}: the driver is not registered");
            continue;
        }
        let create = format!("create -f {fmt} n.{fmt} 4M");
        let info = format!("info n.{fmt}");
        let convert = format!("convert -O {fmt} t.qed o.{fmt}");
        let info_json = format!("info --output=json o.{fmt}");
        let check_o = format!("check o.{fmt}");
        let compare = format!("compare t.qed o.{fmt}");
        let map = format!("map --output=json o.{fmt}");
        let measure = format!("measure -O {fmt} a.raw");
        check(
            fmt,
            &[
                &[create.as_str(), info.as_str()],
                &[
                    convert.as_str(),
                    info_json.as_str(),
                    check_o.as_str(),
                    compare.as_str(),
                    map.as_str(),
                ],
                &[measure.as_str()],
            ],
        );
    }
}
