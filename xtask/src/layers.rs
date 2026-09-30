// SPDX-License-Identifier: MIT OR Apache-2.0

//! The layer rule from spec/24-workspace-layout.md.
//!
//! `xtask/layers.toml` puts every crate in one of six layers, L0 foundation up to L5 binaries. A
//! crate may depend on crates in its own layer or below. Dev dependencies are held to the same
//! rule, because a test that reaches upward is how an upward dependency starts.

use std::collections::BTreeMap;
use std::path::Path;

use crate::workspace::{self, Package};

pub(crate) fn check(root: &Path) -> Result<(), String> {
    let ranks = read_ranks(root)?;
    let packages = workspace::packages(root)?;
    let problems = violations(&ranks, &packages);
    if problems.is_empty() {
        println!("the layer rule holds across {} crates", packages.len() - 1);
        Ok(())
    } else {
        for problem in &problems {
            eprintln!("  {problem}");
        }
        Err(format!("{} layer violations", problems.len()))
    }
}

fn violations(ranks: &BTreeMap<String, u32>, packages: &[Package]) -> Vec<String> {
    let mut problems = Vec::new();
    for package in packages {
        if package.name == "xtask" {
            continue;
        }
        let Some(&rank) = ranks.get(&package.name) else {
            problems.push(format!("{} has no layer in xtask/layers.toml", package.name));
            continue;
        };
        for dep in &package.deps {
            let Some(&dep_rank) = ranks.get(&dep.name) else {
                problems
                    .push(format!("{} depends on {}, which has no layer", package.name, dep.name));
                continue;
            };
            if dep_rank > rank {
                problems.push(format!(
                    "{} in L{rank} depends on {} in L{dep_rank}, which is above it",
                    package.name, dep.name
                ));
            }
        }
    }
    for name in ranks.keys() {
        if !packages.iter().any(|p| &p.name == name) {
            problems.push(format!("xtask/layers.toml lists {name}, which is not a crate"));
        }
    }
    problems
}

pub(crate) fn read_ranks(root: &Path) -> Result<BTreeMap<String, u32>, String> {
    let path = root.join("xtask/layers.toml");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let value: toml::Table =
        text.parse().map_err(|e| format!("xtask/layers.toml is not valid TOML: {e}"))?;
    let table = value
        .get("ranks")
        .and_then(toml::Value::as_table)
        .ok_or("xtask/layers.toml has no [ranks] table")?;
    let mut ranks = BTreeMap::new();
    for (name, rank) in table {
        let rank = rank
            .as_integer()
            .and_then(|r| u32::try_from(r).ok())
            .filter(|r| *r <= 5)
            .ok_or_else(|| format!("{name} has a layer that is not 0 to 5"))?;
        ranks.insert(name.clone(), rank);
    }
    Ok(ranks)
}

#[cfg(test)]
mod tests {
    use super::violations;
    use crate::workspace::{Dep, Package};
    use std::collections::BTreeMap;

    fn package(name: &str, deps: &[&str]) -> Package {
        Package {
            name: name.into(),
            license: String::new(),
            dir: Default::default(),
            deps: deps.iter().map(|d| Dep { name: (*d).into(), kind: None }).collect(),
            unsafe_budget: Some(0),
        }
    }

    fn ranks() -> BTreeMap<String, u32> {
        [("low", 0), ("mid", 2), ("high", 5)].into_iter().map(|(n, r)| (n.into(), r)).collect()
    }

    #[test]
    fn downward_and_sideways_edges_are_fine() {
        let packages =
            [package("low", &[]), package("mid", &["low"]), package("high", &["mid", "high"])];
        assert!(violations(&ranks(), &packages).is_empty());
    }

    #[test]
    fn an_upward_edge_is_caught() {
        let packages = [package("low", &["high"]), package("mid", &[]), package("high", &[])];
        assert_eq!(violations(&ranks(), &packages).len(), 1);
    }

    #[test]
    fn a_crate_without_a_layer_is_caught() {
        let packages =
            [package("low", &[]), package("mid", &[]), package("high", &[]), package("new", &[])];
        assert_eq!(violations(&ranks(), &packages).len(), 1);
    }
}
