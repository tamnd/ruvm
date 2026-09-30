// SPDX-License-Identifier: MIT OR Apache-2.0

//! `vendor-qemu/`, the QEMU source files ruvm reads as data.
//!
//! spec/24-workspace-layout.md explains why they are copied rather than submoduled: a build never
//! needs the network, and the exact upstream commit is one file a reviewer can see. This module
//! does two things with them.
//!
//! `upstream-sync <tag>` copies them out of a QEMU checkout at that tag, rewrites `UPSTREAM` and
//! `MANIFEST`, and prints what changed: files, and inside the files the things ruvm has to
//! implement, meaning QMP commands and events, command line options, HMP commands, decode
//! patterns, trace points and targets. That report is the work list for a sync, per
//! spec/02-compat-contract.md. Run against the tag already vendored it prints an empty delta,
//! which is how the M0 exit criterion is checked.
//!
//! `vendor-check` rehashes every file against `MANIFEST`, so a hand edit to a vendored file fails
//! CI instead of quietly becoming a fork of QEMU.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

const QEMU_GIT: &str = "https://gitlab.com/qemu-project/qemu.git";

/// One kind of vendored input: where it lives in QEMU, and where it goes under `vendor-qemu/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Qapi,
    Decode,
    TraceEvents,
    Hx,
    AcpiExpected,
    Targets,
}

impl Kind {
    const ALL: [Kind; 6] =
        [Kind::Qapi, Kind::Decode, Kind::TraceEvents, Kind::Hx, Kind::AcpiExpected, Kind::Targets];

    fn dir(self) -> &'static str {
        match self {
            Kind::Qapi => "qapi",
            Kind::Decode => "decode",
            Kind::TraceEvents => "trace-events",
            Kind::Hx => "hx",
            Kind::AcpiExpected => "acpi-expected",
            Kind::Targets => "targets",
        }
    }

    /// Where a QEMU source path goes under `vendor-qemu/`, if it is an input of this kind.
    fn destination(self, path: &str) -> Option<String> {
        let dir = self.dir();
        match self {
            Kind::Qapi => {
                if let Some(name) = path.strip_prefix("qapi/") {
                    return (name.ends_with(".json") && !name.contains('/'))
                        .then(|| format!("{dir}/{name}"));
                }
                match path {
                    "qga/qapi-schema.json" => Some(format!("{dir}/qga/qapi-schema.json")),
                    "storage-daemon/qapi/qapi-schema.json" => {
                        Some(format!("{dir}/storage-daemon/qapi-schema.json"))
                    }
                    _ => None,
                }
            }
            Kind::Decode => {
                let rest = path.strip_prefix("target/")?;
                rest.ends_with(".decode").then(|| format!("{dir}/{rest}"))
            }
            Kind::TraceEvents => {
                if path.starts_with("tests/") || path.starts_with("roms/") {
                    return None;
                }
                (path == "trace-events" || path.ends_with("/trace-events"))
                    .then(|| format!("{dir}/{path}"))
            }
            Kind::Hx => {
                matches!(path, "qemu-options.hx" | "hmp-commands.hx" | "hmp-commands-info.hx")
                    .then(|| format!("{dir}/{path}"))
            }
            Kind::AcpiExpected => {
                path.strip_prefix("tests/data/acpi/").map(|rest| format!("{dir}/{rest}"))
            }
            Kind::Targets => {
                let name = path.strip_prefix("configs/targets/")?;
                (name.ends_with(".mak") && !name.contains('/')).then(|| format!("{dir}/{name}"))
            }
        }
    }

    /// The names inside a file of this kind that ruvm has to implement one by one.
    fn names(self, path: &str, text: &str) -> Vec<(&'static str, String)> {
        match self {
            Kind::Qapi if path.starts_with("qapi/qga/") => qapi_names(text)
                .into_iter()
                .map(|(category, name)| {
                    (if category == "QMP command" { "guest agent command" } else { category }, name)
                })
                .collect(),
            Kind::Qapi => qapi_names(text),
            Kind::Decode => decode_names(text)
                .into_iter()
                .map(|n| ("decode pattern", format!("{}:{n}", strip_dir(path))))
                .collect(),
            Kind::TraceEvents => {
                trace_names(text).into_iter().map(|n| ("trace point", n)).collect()
            }
            Kind::Hx => hx_names(path, text),
            Kind::Targets => {
                vec![("target", strip_dir(path).trim_end_matches(".mak").to_string())]
            }
            Kind::AcpiExpected => Vec::new(),
        }
    }
}

fn strip_dir(path: &str) -> &str {
    path.split_once('/').map_or(path, |(_, rest)| rest)
}

/// `cargo xtask upstream-sync <tag> [--from <qemu checkout>]`.
pub(crate) fn sync(root: &Path, args: &[String]) -> Result<(), String> {
    let mut tag = None;
    let mut from = std::env::var_os("QEMU_SRC").map(PathBuf::from);
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--from" => from = Some(iter.next().ok_or("--from needs a directory")?.into()),
            other if tag.is_none() && !other.starts_with('-') => tag = Some(other.to_string()),
            other => return Err(format!("upstream-sync does not take {other}")),
        }
    }
    let tag = tag.ok_or("usage: cargo xtask upstream-sync <tag> [--from <qemu checkout>]")?;
    let checkout = match from {
        Some(dir) => dir,
        None => clone(root, &tag)?,
    };
    let commit = commit_at(&checkout, &tag)?;

    let vendor = root.join("vendor-qemu");
    let before = read_tree(&vendor)?;
    let after = collect(&checkout)?;

    let report = delta(&before, &after);
    write_tree(&vendor, &before, &after)?;
    std::fs::write(vendor.join("UPSTREAM"), upstream_file(&tag, &commit))
        .map_err(|e| format!("could not write UPSTREAM: {e}"))?;
    std::fs::write(vendor.join("MANIFEST"), manifest(&after))
        .map_err(|e| format!("could not write MANIFEST: {e}"))?;

    println!("vendor-qemu is at {tag}, commit {commit}, {} files", after.len());
    for (category, names) in inventory(&after) {
        println!("  {:>6} {category}s", names.len());
    }
    if report.is_empty() {
        println!("the delta is empty");
    } else {
        print!("{report}");
    }
    Ok(())
}

/// `cargo xtask vendor-check`.
pub(crate) fn check(root: &Path) -> Result<(), String> {
    let vendor = root.join("vendor-qemu");
    let text = std::fs::read_to_string(vendor.join("MANIFEST"))
        .map_err(|e| format!("could not read vendor-qemu/MANIFEST: {e}"))?;
    let mut listed = BTreeMap::new();
    for line in text.lines() {
        let (hash, path) =
            line.split_once("  ").ok_or_else(|| format!("MANIFEST has a bad line: {line}"))?;
        listed.insert(path.to_string(), hash.to_string());
    }
    let tree = read_tree(&vendor)?;
    let mut problems = Vec::new();
    for (path, hash) in &listed {
        match tree.get(path) {
            None => problems.push(format!("{path} is in MANIFEST but not on disk")),
            Some(bytes) if &sha256(bytes) != hash => {
                problems.push(format!("{path} is not the file upstream-sync wrote"))
            }
            Some(_) => {}
        }
    }
    for path in tree.keys() {
        if !listed.contains_key(path) {
            problems.push(format!("{path} is on disk but not in MANIFEST"));
        }
    }
    if problems.is_empty() {
        let upstream = std::fs::read_to_string(vendor.join("UPSTREAM")).unwrap_or_default();
        let tag = upstream.lines().find_map(|l| l.strip_prefix("tag ")).unwrap_or("?");
        println!("vendor-qemu matches its MANIFEST: {} files from QEMU {tag}", listed.len());
        Ok(())
    } else {
        for problem in &problems {
            eprintln!("  {problem}");
        }
        Err(format!("{} files in vendor-qemu differ from upstream", problems.len()))
    }
}

fn clone(root: &Path, tag: &str) -> Result<PathBuf, String> {
    let dir = root.join("target").join(format!("qemu-{tag}"));
    if dir.join(".git").is_dir() {
        return Ok(dir);
    }
    println!("cloning QEMU {tag} into {}", dir.display());
    let dir_arg = dir.to_string_lossy().to_string();
    git(None, &["clone", "--quiet", "--depth", "1", "--branch", tag, QEMU_GIT, &dir_arg])?;
    Ok(dir)
}

/// The commit the checkout is at, which has to be the commit the tag names.
fn commit_at(checkout: &Path, tag: &str) -> Result<String, String> {
    let head = git(Some(checkout), &["rev-parse", "HEAD"])?;
    let tagged = git(Some(checkout), &["rev-parse", &format!("{tag}^{{commit}}")])?;
    if head != tagged {
        return Err(format!(
            "{} is at {head}, but {tag} is {tagged}, so check out the tag first",
            checkout.display()
        ));
    }
    Ok(head)
}

fn git(dir: Option<&Path>, args: &[&str]) -> Result<String, String> {
    let mut command = Command::new("git");
    if let Some(dir) = dir {
        command.arg("-C").arg(dir);
    }
    let out = command.args(args).output().map_err(|e| format!("could not run git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Every input file in a QEMU checkout, keyed by its path under `vendor-qemu/`.
fn collect(checkout: &Path) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let mut files = Vec::new();
    walk(checkout, checkout, &mut files)?;
    let mut out = BTreeMap::new();
    for path in files {
        for kind in Kind::ALL {
            if let Some(dest) = kind.destination(&path) {
                let bytes = std::fs::read(checkout.join(&path))
                    .map_err(|e| format!("could not read {path}: {e}"))?;
                out.insert(dest, bytes);
                break;
            }
        }
    }
    if out.is_empty() {
        return Err(format!("{} does not look like a QEMU checkout", checkout.display()));
    }
    Ok(out)
}

/// The vendored files as they are now, without `UPSTREAM`, `MANIFEST` and `README.md`, which are
/// ours.
fn read_tree(vendor: &Path) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let mut out = BTreeMap::new();
    if !vendor.is_dir() {
        return Ok(out);
    }
    let mut files = Vec::new();
    walk(vendor, vendor, &mut files)?;
    for path in files {
        if matches!(path.as_str(), "UPSTREAM" | "MANIFEST" | "README.md") {
            continue;
        }
        let bytes = std::fs::read(vendor.join(&path))
            .map_err(|e| format!("could not read vendor-qemu/{path}: {e}"))?;
        out.insert(path, bytes);
    }
    Ok(out)
}

fn walk(base: &Path, dir: &Path, out: &mut Vec<String>) -> Result<(), String> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("could not read {}: {e}", dir.display()))?;
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let name = entry.file_name();
        if name.to_string_lossy().starts_with(".git") || name == "build" {
            continue;
        }
        let file_type = entry.file_type().map_err(|e| e.to_string())?;
        if file_type.is_dir() {
            walk(base, &path, out)?;
        } else if file_type.is_file() {
            let relative = path.strip_prefix(base).map_err(|e| e.to_string())?;
            let parts: Vec<_> = relative.iter().map(|p| p.to_string_lossy()).collect();
            out.push(parts.join("/"));
        }
    }
    Ok(())
}

fn write_tree(
    vendor: &Path,
    before: &BTreeMap<String, Vec<u8>>,
    after: &BTreeMap<String, Vec<u8>>,
) -> Result<(), String> {
    for path in before.keys().filter(|p| !after.contains_key(*p)) {
        std::fs::remove_file(vendor.join(path))
            .map_err(|e| format!("could not remove vendor-qemu/{path}: {e}"))?;
    }
    for (path, bytes) in after {
        if before.get(path) == Some(bytes) {
            continue;
        }
        let dest = vendor.join(path);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&dest, bytes)
            .map_err(|e| format!("could not write vendor-qemu/{path}: {e}"))?;
    }
    Ok(())
}

fn upstream_file(tag: &str, commit: &str) -> String {
    format!(
        "# The QEMU release the files in this directory come from. Written by cargo xtask upstream-sync.\nrepository {QEMU_GIT}\ntag {tag}\ncommit {commit}\n"
    )
}

fn manifest(files: &BTreeMap<String, Vec<u8>>) -> String {
    files.iter().map(|(path, bytes)| format!("{}  {path}\n", sha256(bytes))).collect()
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().fold(String::with_capacity(64), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn kind_of(path: &str) -> Option<Kind> {
    let dir = path.split('/').next()?;
    Kind::ALL.into_iter().find(|k| k.dir() == dir)
}

/// Everything ruvm has to implement, by category, for a set of vendored files.
fn inventory(files: &BTreeMap<String, Vec<u8>>) -> BTreeMap<&'static str, BTreeSet<String>> {
    let mut out: BTreeMap<&'static str, BTreeSet<String>> = BTreeMap::new();
    for (path, bytes) in files {
        let Some(kind) = kind_of(path) else { continue };
        let text = String::from_utf8_lossy(bytes);
        for (category, name) in kind.names(path, &text) {
            out.entry(category).or_default().insert(name);
        }
    }
    out
}

/// The report for a sync, empty when nothing changed.
fn delta(before: &BTreeMap<String, Vec<u8>>, after: &BTreeMap<String, Vec<u8>>) -> String {
    let mut out = String::new();
    let added: Vec<_> = after.keys().filter(|p| !before.contains_key(*p)).collect();
    let removed: Vec<_> = before.keys().filter(|p| !after.contains_key(*p)).collect();
    let changed: Vec<_> =
        after.iter().filter(|(p, b)| before.get(*p).is_some_and(|old| old != *b)).collect();
    if added.is_empty() && removed.is_empty() && changed.is_empty() {
        return out;
    }
    let _ = writeln!(
        out,
        "files: {} added, {} removed, {} changed",
        added.len(),
        removed.len(),
        changed.len()
    );
    for kind in Kind::ALL {
        let count = |list: &[&String]| list.iter().filter(|p| kind_of(p) == Some(kind)).count();
        let changed_here = changed.iter().filter(|(p, _)| kind_of(p) == Some(kind)).count();
        let (a, r) = (count(&added), count(&removed));
        if a + r + changed_here > 0 {
            let _ =
                writeln!(out, "  {}: {a} added, {r} removed, {changed_here} changed", kind.dir());
        }
    }
    let old = inventory(before);
    let new = inventory(after);
    let empty = BTreeSet::new();
    let categories: BTreeSet<_> = old.keys().chain(new.keys()).copied().collect();
    for category in categories {
        let was = old.get(category).unwrap_or(&empty);
        let is = new.get(category).unwrap_or(&empty);
        let gained: Vec<_> = is.difference(was).collect();
        let lost: Vec<_> = was.difference(is).collect();
        if gained.is_empty() && lost.is_empty() {
            continue;
        }
        let _ = writeln!(out, "{category}: {} new, {} gone", gained.len(), lost.len());
        for name in gained {
            let _ = writeln!(out, "  + {name}");
        }
        for name in lost {
            let _ = writeln!(out, "  - {name}");
        }
    }
    out
}

/// QMP commands and events from a QAPI schema file. The schema is JSON with single quotes and
/// comments, so this reads the two keys it needs rather than parsing it.
fn qapi_names(text: &str) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim_start();
        for (key, category) in [("{ 'command': '", "QMP command"), ("{ 'event': '", "QMP event")] {
            if let Some(rest) = line.strip_prefix(key) {
                if let Some(name) = rest.split('\'').next() {
                    out.push((category, name.to_string()));
                }
            }
        }
    }
    out
}

/// Command line options from qemu-options.hx and HMP commands from the two hmp-commands files.
fn hx_names(path: &str, text: &str) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    let info = path.ends_with("hmp-commands-info.hx");
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("DEF(\"") {
            if let Some(name) = rest.split('"').next() {
                out.push(("command line option", format!("-{name}")));
            }
        }
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix(".name") {
            let rest = rest.trim_start().trim_start_matches('=').trim_start();
            if let Some(name) = rest.strip_prefix('"').and_then(|r| r.split('"').next()) {
                let name = if info { format!("info {name}") } else { name.to_string() };
                out.push(("HMP command", name));
            }
        }
    }
    out
}

/// Pattern names from a decodetree file: the first word of each line that is not a field,
/// argument set, format, group bracket, comment or continuation.
fn decode_names(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut continued = false;
    for line in text.lines() {
        let was_continued = continued;
        continued = line.trim_end().ends_with('\\');
        if was_continued {
            continue;
        }
        let trimmed = line.trim();
        let Some(first) = trimmed.split_whitespace().next() else { continue };
        if first.starts_with(['#', '%', '&', '@', '{', '}', '[', ']']) {
            continue;
        }
        if first.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && first.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        {
            out.push(first.to_string());
        }
    }
    out
}

/// Trace point names from a trace-events file, with any properties such as `disable` skipped.
fn trace_names(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some(open) = trimmed.find('(') else { continue };
        if let Some(name) = trimmed[..open].split_whitespace().last() {
            out.push(name.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destinations_follow_the_spec_layout() {
        assert_eq!(Kind::Qapi.destination("qapi/block-core.json").unwrap(), "qapi/block-core.json");
        assert_eq!(Kind::Qapi.destination("qapi/trace-events"), None);
        assert_eq!(
            Kind::Qapi.destination("storage-daemon/qapi/qapi-schema.json").unwrap(),
            "qapi/storage-daemon/qapi-schema.json"
        );
        assert_eq!(
            Kind::Decode.destination("target/arm/tcg/a64.decode").unwrap(),
            "decode/arm/tcg/a64.decode"
        );
        assert_eq!(Kind::Decode.destination("tests/decode/err_field1.decode"), None);
        assert_eq!(
            Kind::TraceEvents.destination("hw/pci/trace-events").unwrap(),
            "trace-events/hw/pci/trace-events"
        );
        assert_eq!(Kind::TraceEvents.destination("tests/qtest/trace-events"), None);
        assert_eq!(
            Kind::AcpiExpected.destination("tests/data/acpi/x86/q35/DSDT").unwrap(),
            "acpi-expected/x86/q35/DSDT"
        );
    }

    #[test]
    fn names_are_read_out_of_each_format() {
        let qapi =
            "{ 'command': 'query-status',\n  'returns': 'StatusInfo' }\n{ 'event': 'STOP' }\n";
        assert_eq!(
            qapi_names(qapi),
            vec![("QMP command", "query-status".into()), ("QMP event", "STOP".into())]
        );
        let hx = "DEF(\"machine\", HAS_ARG, QEMU_OPTION_machine, \\\n    {\n        .name       = \"help|?\",\n";
        assert_eq!(
            hx_names("hmp-commands.hx", hx),
            vec![("command line option", "-machine".into()), ("HMP command", "help|?".into())]
        );
        let decode = "%imm 0:8\n&rr rd rs\n@rr .... .... rs:4 rd:4 &rr\n# a comment\nADD 0000 0001 .... .... @rr\nSUB 0000 0010 \\\n  .... .... @rr\n{\n  NOP 1111 1111 0000 0000\n}\n";
        assert_eq!(decode_names(decode), vec!["ADD", "SUB", "NOP"]);
        let trace =
            "# pci.c\npci_route_irq(int dev_irq) \"IRQ %d\"\ndisable vcpu_thing(int x) \"x %d\"\n";
        assert_eq!(trace_names(trace), vec!["pci_route_irq", "vcpu_thing"]);
    }

    #[test]
    fn an_unchanged_tree_has_an_empty_delta() {
        let mut tree = BTreeMap::new();
        tree.insert("qapi/x.json".to_string(), b"{ 'command': 'a' }\n".to_vec());
        assert!(delta(&tree, &tree).is_empty());
        let mut next = tree.clone();
        next.insert(
            "qapi/x.json".to_string(),
            b"{ 'command': 'a' }\n{ 'command': 'b' }\n".to_vec(),
        );
        let report = delta(&tree, &next);
        assert!(report.contains("QMP command: 1 new, 0 gone"), "{report}");
        assert!(report.contains("  + b"), "{report}");
    }
}
