// SPDX-License-Identifier: MIT OR Apache-2.0

//! The license provenance rule.
//!
//! ruvm is two kinds of crate. The ones that port QEMU code are GPL-2.0-or-later because the
//! code they port is. The ones that do not are MIT OR Apache-2.0, so that rust-vmm and the other
//! Rust VMMs can use them. The split only holds if a permissive crate never depends on a GPL one,
//! since the build that links them is then GPL whatever its manifest says, and if every source
//! file says which side it is on, so that a file copied between crates carries its license along.

use std::path::Path;

use crate::workspace::{self, Package};

pub(crate) const GPL: &str = "GPL-2.0-or-later";
pub(crate) const PERMISSIVE: &str = "MIT OR Apache-2.0";

pub(crate) fn check(root: &Path) -> Result<(), String> {
    let packages = workspace::packages(root)?;
    let mut problems = Vec::new();
    let mut files = 0usize;

    for package in &packages {
        if package.license != GPL && package.license != PERMISSIVE {
            problems.push(format!(
                "{} is licensed {:?}, which is neither {GPL} nor {PERMISSIVE}",
                package.name, package.license
            ));
            continue;
        }
        problems.extend(edges(package, &packages));

        let mut sources = Vec::new();
        workspace::rust_files(&package.dir, &mut sources)?;
        for file in sources {
            let text = std::fs::read_to_string(&file)
                .map_err(|e| format!("could not read {}: {e}", file.display()))?;
            let shown = file.strip_prefix(root).unwrap_or(&file).display().to_string();
            if let Some(problem) = header(&shown, &text, &package.license) {
                problems.push(problem);
            }
            files += 1;
        }
    }

    if problems.is_empty() {
        let permissive = packages.iter().filter(|p| p.license == PERMISSIVE).count();
        println!(
            "provenance holds: {permissive} permissive crates depend on no GPL crate, and {files} files carry the right header"
        );
        Ok(())
    } else {
        for problem in &problems {
            eprintln!("  {problem}");
        }
        Err(format!("{} provenance violations", problems.len()))
    }
}

/// A permissive crate that depends on a GPL crate, for normal and build dependencies. A dev
/// dependency is allowed, because tests are not distributed and a permissive crate is allowed to
/// be tested against the GPL code that uses it.
fn edges(package: &Package, all: &[Package]) -> Vec<String> {
    if package.license != PERMISSIVE {
        return Vec::new();
    }
    package
        .deps
        .iter()
        .filter(|d| d.kind.as_deref() != Some("dev"))
        .filter_map(|d| all.iter().find(|p| p.name == d.name))
        .filter(|dep| dep.license == GPL)
        .map(|dep| {
            format!("{} is {PERMISSIVE} and depends on {}, which is {GPL}", package.name, dep.name)
        })
        .collect()
}

/// The first line of every source file names the crate's license.
fn header(name: &str, text: &str, license: &str) -> Option<String> {
    let want = format!("// SPDX-License-Identifier: {license}");
    let first = text.lines().next().unwrap_or_default().trim_end();
    if first == want {
        None
    } else if first.starts_with("// SPDX-License-Identifier:") {
        Some(format!("{name}:1: says {first:?} but the crate is {license}"))
    } else {
        Some(format!("{name}:1: has no SPDX header, it should start with {want:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::{GPL, PERMISSIVE, edges, header};
    use crate::workspace::{Dep, Package};

    fn package(name: &str, license: &str, deps: &[(&str, Option<&str>)]) -> Package {
        Package {
            name: name.into(),
            license: license.into(),
            dir: Default::default(),
            deps: deps
                .iter()
                .map(|(n, k)| Dep { name: (*n).into(), kind: k.map(str::to_string) })
                .collect(),
            unsafe_budget: Some(0),
        }
    }

    #[test]
    fn permissive_on_gpl_is_caught_but_not_as_a_dev_dependency() {
        let all = [
            package("p", PERMISSIVE, &[("g", None), ("g", Some("dev"))]),
            package("g", GPL, &[("p", None)]),
        ];
        assert_eq!(edges(&all[0], &all).len(), 1);
        assert!(edges(&all[1], &all).is_empty());
    }

    #[test]
    fn headers_must_match_the_crate() {
        let gpl = format!("// SPDX-License-Identifier: {GPL}\n");
        assert!(header("a.rs", &gpl, GPL).is_none());
        assert!(header("a.rs", &gpl, PERMISSIVE).is_some());
        assert!(header("a.rs", "fn main() {}\n", GPL).is_some());
    }
}
