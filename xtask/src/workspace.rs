// SPDX-License-Identifier: MIT OR Apache-2.0

//! The workspace as `cargo metadata` describes it.
//!
//! Every check here that asks about crates asks cargo rather than walking directories, so that a
//! crate cargo builds is a crate the checks see, whatever its directory is called.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// One package in the workspace, with only the fields the checks read.
#[derive(Debug)]
pub(crate) struct Package {
    pub(crate) name: String,
    pub(crate) license: String,
    pub(crate) dir: PathBuf,
    pub(crate) deps: Vec<Dep>,
    pub(crate) unsafe_budget: Option<u64>,
}

/// A dependency edge to another workspace crate.
#[derive(Debug)]
pub(crate) struct Dep {
    pub(crate) name: String,
    /// `None` for a normal dependency, `Some("dev")` or `Some("build")` otherwise.
    pub(crate) kind: Option<String>,
}

pub(crate) fn packages(root: &Path) -> Result<Vec<Package>, String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let output = Command::new(cargo)
        .current_dir(root)
        .args(["metadata", "--format-version", "1", "--no-deps", "--offline"])
        .output()
        .map_err(|e| format!("could not run cargo metadata: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let json: Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("cargo metadata printed something that is not JSON: {e}"))?;
    let list = json["packages"].as_array().ok_or("cargo metadata has no packages")?;

    let mut out = Vec::new();
    for p in list {
        let name = p["name"].as_str().unwrap_or_default().to_string();
        let manifest = PathBuf::from(p["manifest_path"].as_str().unwrap_or_default());
        let dir = manifest.parent().map(Path::to_path_buf).unwrap_or_default();
        let license = p["license"].as_str().unwrap_or_default().to_string();
        let deps = p["dependencies"]
            .as_array()
            .map(|deps| {
                deps.iter()
                    .filter(|d| d["path"].is_string())
                    .map(|d| Dep {
                        name: d["name"].as_str().unwrap_or_default().to_string(),
                        kind: d["kind"].as_str().map(str::to_string),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let unsafe_budget = p["metadata"]["ruvm"]["unsafe-budget"].as_u64();
        out.push(Package { name, license, dir, deps, unsafe_budget });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Every `.rs` file under a directory, skipping build output.
pub(crate) fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("could not read {}: {e}", dir.display()))?;
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        if name == "target" || name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            rust_files(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    out.sort();
    Ok(())
}
