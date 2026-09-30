// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask version <x.y.z>`, the one place the workspace version changes.
//!
//! Every crate inherits the version from the workspace and every internal dependency is pinned to
//! it exactly, so a bump is one line in `[workspace.package]` and one per crate in
//! `[workspace.dependencies]`. Doing that by hand is how a release ends up with one pin behind.

use std::path::Path;

pub(crate) fn set(root: &Path, version: Option<&str>) -> Result<(), String> {
    let version = version.ok_or("usage: cargo xtask version <x.y.z>")?;
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() != 3 || parts.iter().any(|p| p.parse::<u32>().is_err()) {
        return Err(format!("{version} is not a version of the form x.y.z"));
    }
    let path = root.join("Cargo.toml");
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("could not read Cargo.toml: {e}"))?;
    let (next, pins) = rewrite(&text, version)?;
    std::fs::write(&path, next).map_err(|e| format!("could not write Cargo.toml: {e}"))?;
    crate::cargo(&["update", "--workspace", "--offline"])?;
    println!("the workspace is at {version}, with {pins} internal pins moved");
    Ok(())
}

fn rewrite(text: &str, version: &str) -> Result<(String, usize), String> {
    let mut out = String::with_capacity(text.len());
    let mut section = String::new();
    let mut set_package = false;
    let mut pins = 0usize;
    for line in text.lines() {
        if line.starts_with('[') {
            section = line.trim().to_string();
        }
        if section == "[workspace.package]" && line.starts_with("version = ") {
            out.push_str(&format!("version = \"{version}\"\n"));
            set_package = true;
            continue;
        }
        if section == "[workspace.dependencies]" && line.starts_with("ruvm-") {
            if let Some(start) = line.find("version = \"=") {
                let from = start + "version = \"=".len();
                let end = line[from..].find('"').map(|e| from + e).ok_or("an unclosed pin")?;
                out.push_str(&line[..from]);
                out.push_str(version);
                out.push_str(&line[end..]);
                out.push('\n');
                pins += 1;
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    if !set_package {
        return Err("Cargo.toml has no version in [workspace.package]".into());
    }
    Ok((out, pins))
}

#[cfg(test)]
mod tests {
    use super::rewrite;

    #[test]
    fn the_package_version_and_every_pin_move_together() {
        let text = "[workspace.package]\nversion = \"0.0.0\"\n\n[workspace.dependencies]\nruvm-base = { version = \"=0.0.0\", path = \"crates/base\" }\nserde_json = \"1.0.145\"\n";
        let (next, pins) = rewrite(text, "0.1.0").unwrap();
        assert_eq!(pins, 1);
        assert!(next.contains("version = \"0.1.0\"\n"));
        assert!(next.contains("ruvm-base = { version = \"=0.1.0\", path = \"crates/base\" }"));
        assert!(next.contains("serde_json = \"1.0.145\""));
    }
}
