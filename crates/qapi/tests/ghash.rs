// SPDX-License-Identifier: GPL-2.0-or-later

//! Replays operations recorded from GLib 2.90 (tests/data/ghash-order.c) and checks that every
//! iteration order matches.

use ruvm_qapi::ghash::{GHashTable, g_str_hash};

const WORDS: [&str; 40] = [
    "type",
    "realized",
    "parent_bus",
    "hotplugged",
    "hotpluggable",
    "id",
    "addr",
    "bus",
    "irq",
    "memory",
    "chardev",
    "device",
    "audiodevs",
    "chardevs",
    "objects",
    "backend",
    "machine",
    "unattached",
    "peripheral",
    "peripheral-anon",
    "sysbus",
    "ioport",
    "legacy-iommu",
    "x-migrate",
    "romfile",
    "multifunction",
    "rombar",
    "failover_pair_id",
    "acpi-index",
    "x-pcie-lnksta-dllla",
    "x-pcie-extcap-init",
    "busnr",
    "x-max-bounce-buffer-size",
    "cpu",
    "kvm-type",
    "dump-guest-core",
    "mem-merge",
    "usb",
    "dt-compatible",
    "firmware",
];

#[test]
fn iteration_order_matches_glib() {
    let data = include_str!("data/ghash-order.txt");
    let mut table: GHashTable<()> = GHashTable::new();
    let mut checks = 0;
    for line in data.lines() {
        let (op, rest) = line.split_once(' ').unwrap_or((line, ""));
        match op {
            "h" => {
                let want: Vec<u32> = rest.split(' ').map(|h| h.parse().unwrap()).collect();
                let got: Vec<u32> = WORDS.iter().map(|w| g_str_hash(w)).collect();
                assert_eq!(got, want);
            }
            "new" => table = GHashTable::new(),
            "i" => {
                table.insert(rest, ());
            }
            "r" => {
                table.remove(rest);
            }
            "R" => {
                table.remove_no_resize(rest);
            }
            "=" => {
                let got: Vec<&str> = table.keys().collect();
                let want: Vec<&str> =
                    if rest.is_empty() { Vec::new() } else { rest.split(' ').collect() };
                assert_eq!(got, want);
                checks += 1;
            }
            _ => panic!("bad line {line:?}"),
        }
    }
    assert!(checks > 100);
}
