// SPDX-License-Identifier: GPL-2.0-or-later

//! The consistency check of a QED image, block/qed-check.c: every table and data cluster the
//! tables point to must be inside the image file and used once, and clusters nothing points
//! to are leaks. Repairing drops the table entries that point outside the file; leaks are
//! only counted.

use std::io;

use ruvm_qapi::types::BlkdebugEvent;

use crate::node::{CheckResult, Node};

use super::table::{is_unalloc_cluster, is_zero_cluster, write_table};
use super::{F_NEED_CHECK, Header, State, check_cluster_offset};

/// `QEDCheck`.
struct QedCheck {
    fix: bool,
    result: CheckResult,
    /// The referenced cluster bitmap, by `qed_bytes_to_clusters()` of the offset.
    used_clusters: Vec<bool>,
}

impl QedCheck {
    /// `qed_set_used_clusters()`: marks `n` clusters from `offset` as used. Clusters used
    /// already are corruptions; returns whether there were none.
    fn set_used_clusters(&mut self, s: &State, offset: u64, n: u32) -> bool {
        let first = s.bytes_to_clusters(offset);
        let mut corruptions = 0;
        for cluster in first..first + u64::from(n) {
            // A valid offset is inside the file, so this is inside the bitmap.
            if let Some(used) = self.used_clusters.get_mut(cluster as usize) {
                if *used {
                    corruptions += 1;
                }
                *used = true;
            }
        }
        self.result.corruptions += corruptions;
        corruptions == 0
    }

    /// `qed_check_l2_table()`: counts and marks the data clusters of `table`, and returns how
    /// many entries are invalid. With `fix` they are dropped from `table`.
    fn check_l2_table(&mut self, s: &State, header: &Header, table: &mut [u64]) -> u32 {
        let mut num_invalid = 0;
        let mut last_offset = 0u64;
        for e in table.iter_mut() {
            let offset = *e;
            if is_unalloc_cluster(offset) || is_zero_cluster(offset) {
                continue;
            }
            self.result.bfi.allocated_clusters += 1;
            if last_offset != 0
                && last_offset.wrapping_add(u64::from(header.cluster_size)) != offset
            {
                self.result.bfi.fragmented_clusters += 1;
            }
            last_offset = offset;

            if !check_cluster_offset(header, s.file_size, offset) {
                if self.fix {
                    *e = 0;
                    self.result.corruptions_fixed += 1;
                } else {
                    self.result.corruptions += 1;
                }
                num_invalid += 1;
                continue;
            }
            self.set_used_clusters(s, offset, 1);
        }
        num_invalid
    }

    /// `qed_check_l1_table()`. Returns the last I/O error.
    fn check_l1_table(&mut self, s: &mut State, file: &Node) -> io::Result<()> {
        let mut num_invalid_l1 = 0;
        let mut last_error = Ok(());

        let (l1_offset, table_size) = (s.header.l1_table_offset, s.header.table_size);
        self.set_used_clusters(s, l1_offset, table_size);

        for i in 0..s.table_nelems as usize {
            let offset = s.l1_table[i];
            if is_unalloc_cluster(offset) {
                continue;
            }
            if !s.check_table_offset(offset) {
                if self.fix {
                    s.l1_table[i] = 0;
                    self.result.corruptions_fixed += 1;
                } else {
                    self.result.corruptions += 1;
                }
                num_invalid_l1 += 1;
                continue;
            }
            if !self.set_used_clusters(s, offset, table_size) {
                continue; // skip an invalid table
            }

            let mut table = match s.read_l2_table(file, offset) {
                Ok(t) => std::mem::take(t),
                Err(e) => {
                    self.result.check_errors += 1;
                    last_error = Err(e);
                    continue;
                }
            };
            let header = s.header;
            let num_invalid_l2 = self.check_l2_table(s, &header, &mut table);
            let r = if num_invalid_l2 > 0 && self.fix {
                file.debug_event(BlkdebugEvent::L2Update);
                write_table(file, offset, &table, 0, table.len(), false)
            } else {
                Ok(())
            };
            // The cache entry was taken out above for the check to look at the state.
            if let Some(t) = s.l2_cache.find(offset) {
                *t = table;
            }
            if let Err(e) = r {
                self.result.check_errors += 1;
                last_error = Err(e);
                continue;
            }
        }

        if num_invalid_l1 > 0 && self.fix {
            if let Err(e) = s.write_l1_table(file, 0, s.table_nelems as usize) {
                self.result.check_errors += 1;
                last_error = Err(e);
            }
        }
        last_error
    }

    /// `qed_check_for_leaks()`.
    fn check_for_leaks(&mut self, s: &State) {
        let start = s.header.header_size as usize;
        self.result.leaks +=
            self.used_clusters.iter().skip(start).filter(|used| !**used).count() as i64;
    }
}

/// `qed_check_mark_clean()`: clears the need-check flag of an image found consistent.
fn mark_clean(s: &mut State, file: &Node, result: &CheckResult) {
    if result.corruptions > 0 || result.check_errors > 0 {
        return;
    }
    if s.header.features & F_NEED_CHECK == 0 {
        return;
    }
    // QEMU flushes the QED node, which flushes the file under it.
    let _ = file.flush();
    s.header.features &= !F_NEED_CHECK;
    let _ = s.write_header_sync(file);
}

/// `qed_check()`: the result, and the last I/O error the check ran into.
pub(super) fn qed_check(s: &mut State, file: &Node, fix: bool) -> (CheckResult, io::Result<()>) {
    let nclusters = s.bytes_to_clusters(s.file_size);
    let mut check = QedCheck {
        fix,
        result: CheckResult::default(),
        used_clusters: vec![false; nclusters as usize],
    };
    check.result.bfi.total_clusters =
        s.header.image_size.div_ceil(u64::from(s.header.cluster_size));
    let ret = check.check_l1_table(s, file);
    if ret.is_ok() {
        check.check_for_leaks(s);
        if fix {
            mark_clean(s, file, &check.result);
        }
    }
    (check.result, ret)
}
