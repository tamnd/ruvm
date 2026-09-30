// SPDX-License-Identifier: GPL-2.0-or-later

//! The permission bits from include/block/block-common.h and `bdrv_perm_names()`.

/// The user needs the data to be consistent while reading.
pub const BLK_PERM_CONSISTENT_READ: u64 = 0x01;
/// The user may change the data.
pub const BLK_PERM_WRITE: u64 = 0x02;
/// The user may write, but only data that is already there, as copy-on-read does.
pub const BLK_PERM_WRITE_UNCHANGED: u64 = 0x04;
/// The user may change the size of the node.
pub const BLK_PERM_RESIZE: u64 = 0x08;
/// Every permission there is.
pub const BLK_PERM_ALL: u64 = 0x0f;

/// `DEFAULT_PERM_PASSTHROUGH`: what a filter forwards from its parents to its child.
pub(crate) const DEFAULT_PERM_PASSTHROUGH: u64 =
    BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE | BLK_PERM_WRITE_UNCHANGED | BLK_PERM_RESIZE;
/// `DEFAULT_PERM_UNCHANGED`: what a filter always shares.
pub(crate) const DEFAULT_PERM_UNCHANGED: u64 = BLK_PERM_ALL & !DEFAULT_PERM_PASSTHROUGH;

/// `bdrv_perm_names()`: the names of the bits in `perm`, joined with `, `.
pub fn perm_names(perm: u64) -> String {
    const NAMES: [(u64, &str); 4] = [
        (BLK_PERM_CONSISTENT_READ, "consistent read"),
        (BLK_PERM_WRITE, "write"),
        (BLK_PERM_WRITE_UNCHANGED, "write unchanged"),
        (BLK_PERM_RESIZE, "resize"),
    ];
    NAMES.iter().filter(|(bit, _)| perm & bit != 0).map(|(_, n)| *n).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(perm_names(BLK_PERM_WRITE), "write");
        assert_eq!(perm_names(BLK_PERM_ALL), "consistent read, write, write unchanged, resize");
        assert_eq!(perm_names(0), "");
    }
}
