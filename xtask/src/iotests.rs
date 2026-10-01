// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask iotests`: run QEMU's own qemu-iotests against ruvm's storage tools and hold them
//! to the M3 bar of spec/14-block-layer-and-tools.md: every test that passes with QEMU's tools
//! passes with ours, or is listed in `xtask/ruvm-iotests-expected.toml` with a reason.
//!
//! The suite comes from a QEMU 11.1 build tree, because `check` wants the Python virtual
//! environment, the generated `common.env` and a real `qemu-system-x86_64` that configure leaves
//! there. A minimal one is enough:
//!
//! ```text
//! ../configure --target-list=x86_64-softmmu --disable-docs --enable-tools
//! make qemu-system-x86_64 qemu-img qemu-io qemu-nbd qemu-storage-daemon
//! ```
//!
//! Each format runs twice through `check -tap`, first with QEMU's tools as the baseline and then
//! with `QEMU_IMG_PROG`, `QEMU_IO_PROG`, `QEMU_NBD_PROG` and `QSD_PROG` pointing at symlinks to
//! ours, named like QEMU's so that messages carry the same program names. `QEMU_PROG` stays the
//! real system emulator, so the tests that start a VM measure QEMU there and our tools around it.
//! The test tree itself is never modified.
//!
//! Usage:
//!
//! ```text
//! cargo xtask iotests --qemu-build DIR [--formats qcow2,raw,nbd] [--groups quick,auto]
//!                     [-j N] [--release] [--reuse-baseline] [--timeout SECS] [TEST...]
//! ```
//!
//! `RUVM_IOTESTS_QEMU_BUILD` stands in for `--qemu-build`. `--reuse-baseline` takes QEMU's
//! results from the previous run instead of running them again, which halves the time of a
//! fix and retest loop. `--timeout` bounds each run of one of our tools but the storage daemon,
//! 600 seconds by default, so that a hang fails its test instead of the whole run. Naming tests
//! runs only those. Results, the TAP logs of both runs, go to `target/iotests/`; the diffs of
//! failed tests stay where `check` leaves them, under the build tree's
//! `tests/qemu-iotests/scratch/`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The tools ruvm stands in for: the binary cargo builds, the name `check` knows it by and the
/// variable that points `check` at it.
const TOOLS: [(&str, &str, &str, &str); 4] = [
    ("ruvm-img", "ruvm-qemu-img", "qemu-img", "QEMU_IMG_PROG"),
    ("ruvm-io", "ruvm-qemu-io", "qemu-io", "QEMU_IO_PROG"),
    ("ruvm-nbd", "ruvm-qemu-nbd", "qemu-nbd", "QEMU_NBD_PROG"),
    ("ruvm-storage-daemon", "ruvm-qemu-storage-daemon", "qemu-storage-daemon", "QSD_PROG"),
];

const EXPECTED: &str = "xtask/ruvm-iotests-expected.toml";

struct Options {
    qemu_build: PathBuf,
    formats: Vec<String>,
    groups: String,
    jobs: usize,
    release: bool,
    reuse_baseline: bool,
    timeout: u32,
    tests: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Outcome {
    Pass,
    Fail,
    Skip,
}

pub(crate) fn run(root: &Path, args: &[String]) -> Result<(), String> {
    let opts = parse(args)?;
    let expected = load_expected(root)?;
    let check = opts.qemu_build.join("tests/qemu-iotests/check");
    let runner = opts.qemu_build.join("run");
    for need in [&check, &runner, &opts.qemu_build.join("qemu-img")] {
        if !need.exists() {
            return Err(format!(
                "{} is missing; --qemu-build must be a configured and built QEMU 11.1 tree",
                need.display()
            ));
        }
    }
    let out = root.join("target/iotests");
    let bin = build_tools(root, &out, opts.release, opts.timeout)?;

    let mut regressions = 0;
    let mut summary = Vec::new();
    for fmt in &opts.formats {
        let base_log = out.join(format!("qemu-{fmt}.tap"));
        let base = if opts.reuse_baseline && base_log.exists() {
            println!("== {fmt}: QEMU baseline from {}", base_log.display());
            parse_tap(&std::fs::read_to_string(&base_log).map_err(|e| e.to_string())?)
        } else {
            println!("== {fmt}: QEMU");
            run_check(&opts, &runner, &check, fmt, None, &base_log)?
        };
        println!("== {fmt}: ruvm");
        let ours = run_check(
            &opts,
            &runner,
            &check,
            fmt,
            Some(&bin),
            &out.join(format!("ruvm-{fmt}.tap")),
        )?;

        let excluded = expected.get(fmt.as_str());
        let mut report = Report::default();
        for (test, want) in &base {
            if !opts.tests.is_empty() && !opts.tests.contains(test) {
                continue;
            }
            let got = ours.get(test).copied().unwrap_or(Outcome::Fail);
            let reason = excluded.and_then(|e| e.get(test.as_str()));
            if *want == Outcome::Pass {
                report.qemu_pass += 1;
            }
            match (want, got, reason) {
                (Outcome::Pass, Outcome::Pass, None) => report.ruvm_pass += 1,
                (Outcome::Pass, Outcome::Pass, Some(_)) => {
                    report.ruvm_pass += 1;
                    report.stale.push(test.clone());
                }
                (Outcome::Pass, _, Some(_)) => report.excluded.push(test.clone()),
                (Outcome::Pass, _, None) => report.regressions.push(test.clone()),
                (_, Outcome::Pass, _) => report.ruvm_pass += 1,
                _ => {}
            }
        }
        regressions += report.regressions.len();
        summary.push(report.render(fmt));
    }

    println!();
    for line in &summary {
        println!("{line}");
    }
    if regressions > 0 {
        Err(format!(
            "{regressions} tests pass with QEMU's tools and fail with ours; fix them or list them \
             in {EXPECTED} with a reason"
        ))
    } else {
        Ok(())
    }
}

#[derive(Default)]
struct Report {
    qemu_pass: usize,
    ruvm_pass: usize,
    excluded: Vec<String>,
    regressions: Vec<String>,
    stale: Vec<String>,
}

impl Report {
    fn render(&self, fmt: &str) -> String {
        let mut s = format!(
            "{fmt}: {} pass with QEMU, {} with ruvm, {} excluded, {} regressions",
            self.qemu_pass,
            self.ruvm_pass,
            self.excluded.len(),
            self.regressions.len()
        );
        if !self.regressions.is_empty() {
            s += &format!("\n  failing: {}", self.regressions.join(" "));
        }
        if !self.excluded.is_empty() {
            s += &format!("\n  excluded: {}", self.excluded.join(" "));
        }
        if !self.stale.is_empty() {
            s += &format!(
                "\n  excluded but passing, drop them from {EXPECTED}: {}",
                self.stale.join(" ")
            );
        }
        s
    }
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut opts = Options {
        qemu_build: std::env::var_os("RUVM_IOTESTS_QEMU_BUILD")
            .map(PathBuf::from)
            .unwrap_or_default(),
        formats: vec!["qcow2".into(), "raw".into(), "nbd".into()],
        groups: "quick,auto".into(),
        jobs: std::thread::available_parallelism().map_or(1, |n| n.get()),
        release: false,
        reuse_baseline: false,
        timeout: 600,
        tests: Vec::new(),
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().cloned().ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "--qemu-build" => opts.qemu_build = PathBuf::from(value(a)?),
            "--formats" => opts.formats = value(a)?.split(',').map(str::to_string).collect(),
            "--groups" => opts.groups = value(a)?,
            "-j" => opts.jobs = value(a)?.parse().map_err(|e| format!("-j: {e}"))?,
            "--release" => opts.release = true,
            "--reuse-baseline" => opts.reuse_baseline = true,
            "--timeout" => {
                opts.timeout = value(a)?.parse().map_err(|e| format!("--timeout: {e}"))?
            }
            t if !t.starts_with('-') => opts.tests.push(t.to_string()),
            other => return Err(format!("unknown option {other}")),
        }
    }
    if opts.qemu_build.as_os_str().is_empty() {
        return Err("give the QEMU build tree with --qemu-build or RUVM_IOTESTS_QEMU_BUILD".into());
    }
    opts.qemu_build = std::path::absolute(&opts.qemu_build).map_err(|e| e.to_string())?;
    for f in &opts.formats {
        if !matches!(f.as_str(), "qcow2" | "raw" | "nbd") {
            return Err(format!("{f} is not a format this harness runs"));
        }
    }
    Ok(opts)
}

/// Builds the four tools and puts them under `out/bin` by QEMU's names.
///
/// `check` has no time limit per test, so a tool of ours that hangs would stall the run for
/// good. Each name is therefore a small script that runs the real binary, a link in
/// `out/bin/real`, under `timeout`, with `argv[0]` still the script's path: that is the path
/// `check` knows the tool by and filters out of the output, as it does for QEMU's own tools.
fn build_tools(root: &Path, out: &Path, release: bool, timeout: u32) -> Result<PathBuf, String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.current_dir(root).args(["build", "--bins"]);
    for (krate, ..) in TOOLS {
        cmd.args(["-p", krate]);
    }
    if release {
        cmd.arg("--release");
    }
    let status = cmd.status().map_err(|e| format!("cargo build: {e}"))?;
    if !status.success() {
        return Err("building the tools failed".into());
    }
    let target =
        std::env::var_os("CARGO_TARGET_DIR").map_or_else(|| root.join("target"), PathBuf::from);
    let built = std::path::absolute(target.join(if release { "release" } else { "debug" }))
        .map_err(|e| e.to_string())?;
    let bin = std::path::absolute(out.join("bin")).map_err(|e| e.to_string())?;
    let real = bin.join("real");
    std::fs::create_dir_all(&real).map_err(|e| e.to_string())?;
    for (_, file, name, _) in TOOLS {
        let link = real.join(name);
        let _ = std::fs::remove_file(&link);
        symlink(&built.join(file), &link)?;
        // `timeout` forks, so the tool runs as its grandchild. iotests.py's QemuStorageDaemon
        // checks that the pid in `--pidfile` is the one it started, and the daemon is killed
        // by the test anyway, so it runs without one.
        let script = if name == "qemu-storage-daemon" {
            format!(
                "#!/bin/bash\n# SPDX-License-Identifier: MIT OR Apache-2.0\n\
                 exec -a \"$0\" {} \"$@\"\n",
                link.display()
            )
        } else {
            format!(
                "#!/bin/bash\n# SPDX-License-Identifier: MIT OR Apache-2.0\n\
                 exec timeout --foreground -k 10 {timeout} bash -c 'exec -a \"$0\" {} \"$@\"' \
                 \"$0\" \"$@\"\n",
                link.display()
            )
        };
        write_script(&bin.join(name), &script)?;
    }
    Ok(bin)
}

#[cfg(unix)]
fn symlink(from: &Path, to: &Path) -> Result<(), String> {
    std::os::unix::fs::symlink(from, to).map_err(|e| format!("{}: {e}", to.display()))
}

#[cfg(unix)]
fn write_script(path: &Path, text: &str) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
        .map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(not(unix))]
fn symlink(_: &Path, _: &Path) -> Result<(), String> {
    Err("qemu-iotests need a Unix host".into())
}

#[cfg(not(unix))]
fn write_script(_: &Path, _: &str) -> Result<(), String> {
    Err("qemu-iotests need a Unix host".into())
}

/// One `check -tap` run, echoed as it goes and kept in `log`.
fn run_check(
    opts: &Options,
    runner: &Path,
    check: &Path,
    fmt: &str,
    ours: Option<&Path>,
    log: &Path,
) -> Result<BTreeMap<String, Outcome>, String> {
    let mut cmd = Command::new(runner);
    cmd.current_dir(check.parent().unwrap_or(Path::new(".")))
        .arg(check)
        .arg("-tap")
        .arg(format!("-{fmt}"))
        .args(["-j", &opts.jobs.to_string()]);
    if opts.tests.is_empty() {
        cmd.args(["-g", &opts.groups]);
    } else {
        cmd.args(&opts.tests);
    }
    if let Some(bin) = ours {
        for (_, _, name, var) in TOOLS {
            cmd.env(var, bin.join(name));
        }
    } else {
        for (_, _, name, var) in TOOLS {
            let path = if name == "qemu-storage-daemon" {
                opts.qemu_build.join("storage-daemon").join(name)
            } else {
                opts.qemu_build.join(name)
            };
            cmd.env(var, path);
        }
    }
    let output =
        cmd.stderr(Stdio::inherit()).output().map_err(|e| format!("{}: {e}", runner.display()))?;
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    if let Some(dir) = log.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(log, &text).map_err(|e| format!("{}: {e}", log.display()))?;
    let results = parse_tap(&text);
    let count = |o| results.values().filter(|&&v| v == o).count();
    println!(
        "   {} pass, {} fail, {} not run, log in {}",
        count(Outcome::Pass),
        count(Outcome::Fail),
        count(Outcome::Skip),
        log.display()
    );
    if results.is_empty() {
        return Err(format!("check printed no results, see {}", log.display()));
    }
    Ok(results)
}

/// `ok FMT TEST`, `not ok FMT TEST` and `ok FMT TEST # SKIP why`, as testrunner.py prints them.
fn parse_tap(text: &str) -> BTreeMap<String, Outcome> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let (outcome, rest) = if let Some(r) = line.strip_prefix("not ok ") {
            (Outcome::Fail, r)
        } else if let Some(r) = line.strip_prefix("ok ") {
            (if line.contains(" # SKIP") { Outcome::Skip } else { Outcome::Pass }, r)
        } else {
            continue;
        };
        if let Some(test) = rest.split_whitespace().nth(1) {
            out.insert(test.to_string(), outcome);
        }
    }
    out
}

/// The exclusion file: one table per format, test name to reason.
fn load_expected(root: &Path) -> Result<BTreeMap<String, BTreeMap<String, String>>, String> {
    let path = root.join(EXPECTED);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let table: toml::Table =
        text.parse().map_err(|e| format!("{EXPECTED} is not valid TOML: {e}"))?;
    let mut out = BTreeMap::new();
    let known: BTreeSet<&str> = ["qcow2", "raw", "nbd"].into();
    for (fmt, tests) in table {
        if !known.contains(fmt.as_str()) {
            return Err(format!("{EXPECTED}: [{fmt}] is not a format this harness runs"));
        }
        let tests = tests.as_table().ok_or(format!("{EXPECTED}: [{fmt}] is not a table"))?;
        let mut m = BTreeMap::new();
        for (test, reason) in tests {
            let reason = reason
                .as_str()
                .filter(|r| !r.trim().is_empty())
                .ok_or(format!("{EXPECTED}: {fmt} {test} needs a reason"))?;
            m.insert(test.clone(), reason.to_string());
        }
        out.insert(fmt, m);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tap_lines() {
        let r = parse_tap(
            "TAP version 13\n# QEMU_IMG -- x\n1..3\nok qcow2 001\nnot ok qcow2 002\n\
             ok qcow2 tests/foo # SKIP not suitable\n",
        );
        assert_eq!(r["001"], Outcome::Pass);
        assert_eq!(r["002"], Outcome::Fail);
        assert_eq!(r["tests/foo"], Outcome::Skip);
    }

    #[test]
    fn expected_file_parses() {
        let root = super::super::root();
        load_expected(&root).unwrap();
    }
}
