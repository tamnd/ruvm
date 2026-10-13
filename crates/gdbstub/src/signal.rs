// SPDX-License-Identifier: GPL-2.0-or-later

//! `gdb_signal_table` of `gdbstub/user-target.c`: GDB's signal numbers, which are its
//! `enum gdb_signal`, and the target's.
//!
//! Every target here numbers its signals as the generic Linux ABI does, so one table serves
//! them all.

/// `GDB_SIGNAL_TRAP`.
pub const GDB_SIGNAL_TRAP: u32 = 5;
/// `GDB_SIGNAL_UNKNOWN`.
pub const GDB_SIGNAL_UNKNOWN: u32 = 143;

/// `__SIGRTMIN` of the host's C library, which the table uses for the target's.
const RTMIN: i32 = 32;

/// The target signal for each GDB signal, -1 for none.
const TABLE: [i32; 151] = table();

const fn table() -> [i32; 151] {
    let mut t = [-1i32; 151];
    // SIGHUP to SIGPWR, by GDB's order: 7 is SIGEMT and 29 SIGLOST.
    let first: [i32; 33] = [
        0, 1, 2, 3, 4, 5, 6, -1, 8, 9, 7, 11, 31, 13, 14, 15, 23, 19, 20, 18, 17, 21, 22, 29, 24,
        25, 26, 27, 28, -1, 10, 12, 30,
    ];
    let mut i = 0;
    while i < 33 {
        t[i] = first[i];
        i += 1;
    }
    // GDB_SIGNAL_REALTIME_33 to _63.
    let mut k = 1;
    while k <= 31 {
        t[44 + k as usize] = RTMIN + k;
        k += 1;
    }
    // GDB_SIGNAL_REALTIME_32, then _64 to _127.
    t[77] = RTMIN;
    let mut k = 32;
    while k <= 95 {
        t[46 + k as usize] = RTMIN + k;
        k += 1;
    }
    t
}

/// `gdb_signal_to_target()`: -1 for a GDB signal the target does not have.
pub fn to_target(sig: u64) -> i32 {
    TABLE.get(sig as u32 as usize).copied().unwrap_or(-1)
}

/// `gdb_target_signal_to_gdb()`.
pub fn to_gdb(sig: i32) -> u32 {
    TABLE.iter().position(|&s| s == sig).map_or(GDB_SIGNAL_UNKNOWN, |i| i as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers() {
        assert_eq!(to_gdb(5), 5);
        assert_eq!(to_gdb(11), 11);
        assert_eq!(to_gdb(7), 10);
        assert_eq!(to_gdb(10), 30);
        assert_eq!(to_gdb(31), 12);
        assert_eq!(to_gdb(32), 77);
        assert_eq!(to_gdb(33), 45);
        assert_eq!(to_gdb(63), 75);
        assert_eq!(to_gdb(64), 78);
        assert_eq!(to_gdb(0), 0);
        assert_eq!(to_gdb(200), GDB_SIGNAL_UNKNOWN);
        assert_eq!(to_target(10), 7);
        assert_eq!(to_target(7), -1);
        assert_eq!(to_target(150), -1);
        assert_eq!(to_target(151), -1);
        assert_eq!(to_target(u64::MAX), -1);
    }
}
