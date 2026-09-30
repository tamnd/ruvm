// SPDX-License-Identifier: GPL-2.0-or-later

//! What `qemu-nbd --list` prints: the body of `qemu_nbd_client_list()`.

use std::fmt::Write;

use ruvm_block::nbd::{NbdExportInfo, NbdMode};

/// The flag names by bit number, from `NBD_FLAG_READ_ONLY_BIT` up.
const FLAG_NAMES: [(u32, &str); 12] = [
    (1, "readonly"),
    (2, "flush"),
    (3, "fua"),
    (4, "rotational"),
    (5, "trim"),
    (6, "zeroes"),
    (7, "df"),
    (8, "multi"),
    (9, "resize"),
    (10, "cache"),
    (11, "fast-zero"),
    (12, "block-status-payload"),
];

/// Formats the export list the way `qemu_nbd_client_list()` prints it.
pub fn format_export_list(list: &[NbdExportInfo]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "exports available: {}", list.len());
    for e in list {
        let _ = writeln!(s, " export: '{}'", e.name);
        if let Some(d) = e.description.as_deref().filter(|d| !d.is_empty()) {
            let _ = writeln!(s, "  description: {d}");
        }
        if e.flags & ruvm_block::nbd::NBD_FLAG_HAS_FLAGS != 0 {
            let _ = writeln!(s, "  size:  {}", e.size);
            let _ = write!(s, "  flags: 0x{:x} (", e.flags);
            for (bit, name) in FLAG_NAMES {
                if u32::from(e.flags) & (1 << bit) != 0 {
                    let _ = write!(s, " {name}");
                }
            }
            s.push_str(" )\n");
        }
        if e.min_block != 0 {
            let _ = writeln!(s, "  min block: {}", e.min_block);
            let _ = writeln!(s, "  opt block: {}", e.opt_block);
            let _ = writeln!(s, "  max block: {}", e.max_block);
        }
        let size = if e.mode >= NbdMode::Extended { "64-bit" } else { "32-bit" };
        let _ = writeln!(s, "  transaction size: {size}");
        if !e.contexts.is_empty() {
            let _ = writeln!(s, "  available meta contexts: {}", e.contexts.len());
            for c in &e.contexts {
                let _ = writeln!(s, "   {c}");
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_like_qemu() {
        let list = vec![
            NbdExportInfo {
                name: "a".into(),
                description: Some("disk".into()),
                size: 1048576,
                flags: 0x58f,
                min_block: 1,
                opt_block: 4096,
                max_block: 33554432,
                mode: NbdMode::Extended,
                contexts: vec!["base:allocation".into()],
                ..Default::default()
            },
            NbdExportInfo {
                name: "b".into(),
                description: Some(String::new()),
                mode: NbdMode::Structured,
                ..Default::default()
            },
        ];
        assert_eq!(
            format_export_list(&list),
            "exports available: 2\n export: 'a'\n  description: disk\n  size:  1048576\n  \
             flags: 0x58f ( readonly flush fua df multi cache )\n  min block: 1\n  opt block: \
             4096\n  max block: 33554432\n  transaction size: 64-bit\n  available meta \
             contexts: 1\n   base:allocation\n export: 'b'\n  transaction size: 32-bit\n"
        );
    }
}
