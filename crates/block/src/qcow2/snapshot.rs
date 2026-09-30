// SPDX-License-Identifier: GPL-2.0-or-later

//! Internal snapshots, from block/qcow2-snapshot.c: the snapshot table, and creating, applying,
//! deleting and listing snapshots.

use std::io;

use ruvm_base::report::report_error;
use ruvm_base::{Error, Result};

use super::cache::set_be64;
use super::check::{CheckResult, FIX_ERRORS};
use super::header::{NB_SNAPSHOTS_OFFSET, be16, be32, be64};
use super::state::*;
use crate::node::errno;

/// Size of `QCowSnapshotHeader`.
pub(crate) const SNAPSHOT_HEADER_SIZE: u64 = 40;
/// Size of `QCowSnapshotExtraData`.
pub(crate) const SNAPSHOT_EXTRA_SIZE: u64 = 24;

/// `QCowSnapshot`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Snapshot {
    pub l1_table_offset: u64,
    pub l1_size: u32,
    pub id_str: String,
    pub name: String,
    pub disk_size: u64,
    pub vm_state_size: u64,
    pub date_sec: u32,
    pub date_nsec: u32,
    pub vm_clock_nsec: u64,
    /// `-1ULL` in QEMU when unknown.
    pub icount: u64,
    pub extra_data_size: u32,
    pub unknown_extra_data: Vec<u8>,
}

/// `QEMUSnapshotInfo`, what callers pass in and get back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SnapshotInfo {
    pub id_str: String,
    pub name: String,
    pub vm_state_size: u64,
    pub date_sec: u32,
    pub date_nsec: u32,
    pub vm_clock_nsec: u64,
    /// `None` is `-1`, unknown.
    pub icount: Option<u64>,
}

impl State {
    /// `bs->total_sectors * BDRV_SECTOR_SIZE`.
    pub(crate) fn disk_size_sectors(&self) -> u64 {
        self.total_size.next_multiple_of(512)
    }

    /// `qcow2_do_read_snapshots()`. With `repair`, a table that is too long is cut short and
    /// extra data that is too long is dropped; the counts go to the two out parameters.
    pub(crate) fn do_read_snapshots(
        &mut self,
        nb_snapshots: u32,
        repair: bool,
        nb_clusters_reduced: &mut u64,
        extra_data_dropped: &mut u64,
    ) -> Result<()> {
        self.snapshots.clear();
        if nb_snapshots == 0 {
            self.snapshots_size = 0;
            return Ok(());
        }
        let fail = |e: io::Error| Error::from_io("Failed to read snapshot table", e);
        let mut offset = self.snapshots_offset;
        let mut table_length = 0u64;
        for i in 0..nb_snapshots {
            let mut truncate_unknown_extra_data = false;
            let pre_sn_offset = offset;
            table_length = table_length.next_multiple_of(8);
            offset = offset.next_multiple_of(8);
            let mut h = [0u8; SNAPSHOT_HEADER_SIZE as usize];
            self.file.pread(offset, &mut h).map_err(fail)?;
            offset += SNAPSHOT_HEADER_SIZE;

            let mut sn = Snapshot {
                l1_table_offset: be64(&h, 0),
                l1_size: be32(&h, 8),
                vm_state_size: be32(&h, 32) as u64,
                date_sec: be32(&h, 16),
                date_nsec: be32(&h, 20),
                vm_clock_nsec: be64(&h, 24),
                extra_data_size: be32(&h, 36),
                ..Default::default()
            };
            let id_str_size = be16(&h, 12) as u64;
            let name_size = be16(&h, 14) as u64;

            if sn.extra_data_size > QCOW_MAX_SNAPSHOT_EXTRA_DATA {
                if !repair {
                    return Err(Error::generic(format!(
                        "Too much extra metadata in snapshot table entry {i}"
                    ))
                    .hint(
                        "You can force-remove this extra metadata with qemu-img check -r all\n",
                    ));
                }
                eprintln!(
                    "Discarding too much extra metadata in snapshot table entry {i} ({} > \
                     {QCOW_MAX_SNAPSHOT_EXTRA_DATA})",
                    sn.extra_data_size
                );
                *extra_data_dropped += 1;
                truncate_unknown_extra_data = true;
            }

            let known = (sn.extra_data_size as u64).min(SNAPSHOT_EXTRA_SIZE);
            let mut extra = [0u8; SNAPSHOT_EXTRA_SIZE as usize];
            self.file.pread(offset, &mut extra[..known as usize]).map_err(fail)?;
            offset += known;
            if sn.extra_data_size >= 8 {
                sn.vm_state_size = be64(&extra, 0);
            }
            sn.disk_size =
                if sn.extra_data_size >= 16 { be64(&extra, 8) } else { self.disk_size_sectors() };
            sn.icount = if sn.extra_data_size >= 24 { be64(&extra, 16) } else { u64::MAX };

            if sn.extra_data_size as u64 > SNAPSHOT_EXTRA_SIZE {
                let extra_data_end = offset + sn.extra_data_size as u64 - SNAPSHOT_EXTRA_SIZE;
                if truncate_unknown_extra_data {
                    sn.extra_data_size = QCOW_MAX_SNAPSHOT_EXTRA_DATA;
                }
                let n = sn.extra_data_size as u64 - SNAPSHOT_EXTRA_SIZE;
                let mut v = vec![0u8; n as usize];
                self.file.pread(offset, &mut v).map_err(fail)?;
                sn.unknown_extra_data = v;
                offset = extra_data_end;
            }

            let mut id = vec![0u8; id_str_size as usize];
            self.file.pread(offset, &mut id).map_err(fail)?;
            offset += id_str_size;
            sn.id_str = String::from_utf8_lossy(&id).into_owned();

            let mut name = vec![0u8; name_size as usize];
            self.file.pread(offset, &mut name).map_err(fail)?;
            offset += name_size;
            sn.name = String::from_utf8_lossy(&name).into_owned();

            // The extra data may have been cut short.
            table_length +=
                SNAPSHOT_HEADER_SIZE + sn.extra_data_size as u64 + id_str_size + name_size;
            if !repair {
                assert_eq!(table_length, offset - self.snapshots_offset);
            }
            if table_length > QCOW_MAX_SNAPSHOTS_SIZE
                || offset - self.snapshots_offset > i32::MAX as u64
            {
                if !repair {
                    let left = nb_snapshots - i;
                    return Err(Error::generic("Snapshot table is too big").hint(format!(
                        "You can force-remove all {left} overhanging snapshots with qemu-img \
                         check -r all\n"
                    )));
                }
                eprintln!(
                    "Discarding {} overhanging snapshots (snapshot table is too big)",
                    nb_snapshots - i
                );
                *nb_clusters_reduced += (nb_snapshots - i) as u64;
                // This leaks the rest of the table and the clusters of the snapshots, which
                // the refcount check then takes care of.
                offset = pre_sn_offset;
                break;
            }
            self.snapshots.push(sn);
        }
        self.snapshots_size = offset - self.snapshots_offset;
        Ok(())
    }

    /// `qcow2_read_snapshots()`.
    pub(crate) fn read_snapshots(&mut self, nb_snapshots: u32) -> Result<()> {
        let (mut a, mut b) = (0, 0);
        let r = self.do_read_snapshots(nb_snapshots, false, &mut a, &mut b);
        if r.is_err() {
            self.snapshots.clear();
        }
        r
    }

    /// `qcow2_write_snapshots()`: writes the list to newly allocated clusters and points the
    /// header at it.
    pub(crate) fn write_snapshots(&mut self) -> io::Result<()> {
        let mut size = 0u64;
        for sn in &self.snapshots {
            size = size.next_multiple_of(8);
            size += SNAPSHOT_HEADER_SIZE;
            size += SNAPSHOT_EXTRA_SIZE.max(sn.extra_data_size as u64);
            size += sn.id_str.len() as u64 + sn.name.len() as u64;
            if size > QCOW_MAX_SNAPSHOTS_SIZE {
                return Err(errno(libc::EFBIG));
            }
        }
        let snapshots_size = size;

        let snapshots_offset = self.alloc_clusters(snapshots_size)?;
        let r = self.write_snapshots_at(snapshots_offset, snapshots_size);
        if let Err(e) = r {
            if snapshots_offset > 0 {
                self.free_clusters(snapshots_offset, snapshots_size, DiscardType::Always);
            }
            return Err(e);
        }
        let old = (self.snapshots_offset, self.snapshots_size);
        self.free_clusters(old.0, old.1, DiscardType::Snapshot);
        self.snapshots_offset = snapshots_offset;
        self.snapshots_size = snapshots_size;
        Ok(())
    }

    fn write_snapshots_at(&mut self, snapshots_offset: u64, snapshots_size: u64) -> io::Result<()> {
        self.flush_all()?;
        // The table position is not in the header yet, so these clusters must be free.
        self.pre_write_overlap_check(0, snapshots_offset, snapshots_size, false)?;

        let mut buf = Vec::with_capacity(snapshots_size as usize);
        for sn in &self.snapshots {
            buf.resize(buf.len().next_multiple_of(8), 0);
            let mut h = [0u8; SNAPSHOT_HEADER_SIZE as usize];
            h[0..8].copy_from_slice(&sn.l1_table_offset.to_be_bytes());
            h[8..12].copy_from_slice(&sn.l1_size.to_be_bytes());
            h[12..14].copy_from_slice(&(sn.id_str.len() as u16).to_be_bytes());
            h[14..16].copy_from_slice(&(sn.name.len() as u16).to_be_bytes());
            h[16..20].copy_from_slice(&sn.date_sec.to_be_bytes());
            h[20..24].copy_from_slice(&sn.date_nsec.to_be_bytes());
            h[24..32].copy_from_slice(&sn.vm_clock_nsec.to_be_bytes());
            // Older implementations should see a VM state that does not fit 32 bits as a disk
            // only snapshot rather than a truncated one.
            if sn.vm_state_size <= 0xffff_ffff {
                h[32..36].copy_from_slice(&(sn.vm_state_size as u32).to_be_bytes());
            }
            let eds = (SNAPSHOT_EXTRA_SIZE as u32).max(sn.extra_data_size);
            h[36..40].copy_from_slice(&eds.to_be_bytes());
            buf.extend_from_slice(&h);
            buf.extend_from_slice(&sn.vm_state_size.to_be_bytes());
            buf.extend_from_slice(&sn.disk_size.to_be_bytes());
            buf.extend_from_slice(&sn.icount.to_be_bytes());
            if sn.extra_data_size as u64 > SNAPSHOT_EXTRA_SIZE {
                buf.extend_from_slice(&sn.unknown_extra_data);
            }
            buf.extend_from_slice(sn.id_str.as_bytes());
            buf.extend_from_slice(sn.name.as_bytes());
        }
        self.file.pwrite(snapshots_offset, &buf)?;

        // The new table and its refcounts must be stable before the header points to it.
        self.flush_all()?;
        let mut d = [0u8; 12];
        d[..4].copy_from_slice(&(self.snapshots.len() as u32).to_be_bytes());
        d[4..].copy_from_slice(&snapshots_offset.to_be_bytes());
        self.file.pwrite(NB_SNAPSHOTS_OFFSET, &d)?;
        self.file.flush()
    }

    /// `qcow2_check_read_snapshot_table()`.
    pub(crate) fn check_read_snapshot_table(
        &mut self,
        result: &mut CheckResult,
        fix: u32,
    ) -> io::Result<()> {
        let mut d = [0u8; 12];
        if let Err(e) = self.file.pread(NB_SNAPSHOTS_OFFSET, &mut d) {
            result.check_errors += 1;
            eprintln!(
                "ERROR failed to read the snapshot table pointer from the image header: {}",
                ruvm_base::error::strerror(&e)
            );
            return Err(e);
        }
        self.snapshots_offset = be64(&d, 4);
        let mut nb = be32(&d, 0);
        let mut nb_clusters_reduced = 0u64;
        let mut extra_data_dropped = 0u64;

        if nb as u64 > QCOW_MAX_SNAPSHOTS && fix & FIX_ERRORS != 0 {
            eprintln!("Discarding {} overhanging snapshots", nb as u64 - QCOW_MAX_SNAPSHOTS);
            nb_clusters_reduced += nb as u64 - QCOW_MAX_SNAPSHOTS;
            nb = QCOW_MAX_SNAPSHOTS as u32;
        }

        if let Err((e, n)) = self.validate_table(
            self.snapshots_offset,
            nb as u64,
            SNAPSHOT_HEADER_SIZE,
            SNAPSHOT_HEADER_SIZE * QCOW_MAX_SNAPSHOTS,
            "snapshot table",
        ) {
            result.check_errors += 1;
            report_error(&e.prepend("ERROR "));
            if nb as u64 > QCOW_MAX_SNAPSHOTS {
                eprintln!(
                    "You can force-remove all {} overhanging snapshots with qemu-img check -r all",
                    nb as u64 - QCOW_MAX_SNAPSHOTS
                );
            }
            self.snapshots_offset = 0;
            self.snapshots.clear();
            return Err(errno(n));
        }

        if let Err(e) = self.do_read_snapshots(
            nb,
            fix & FIX_ERRORS != 0,
            &mut nb_clusters_reduced,
            &mut extra_data_dropped,
        ) {
            result.check_errors += 1;
            report_error(&e.prepend("ERROR failed to read the snapshot table: "));
            self.snapshots_offset = 0;
            self.snapshots.clear();
            return Err(errno(libc::EINVAL));
        }
        result.corruptions += (nb_clusters_reduced + extra_data_dropped) as i64;

        if nb_clusters_reduced > 0 {
            // The header must agree with the in-memory count before the refcount check, which
            // then fixes the leaks.
            assert!(fix & FIX_ERRORS != 0);
            let n = (self.snapshots.len() as u32).to_be_bytes();
            if let Err(e) =
                self.file.pwrite(NB_SNAPSHOTS_OFFSET, &n).and_then(|()| self.file.flush())
            {
                result.check_errors += 1;
                eprintln!(
                    "ERROR failed to update the snapshot count in the image header: {}",
                    ruvm_base::error::strerror(&e)
                );
                return Err(e);
            }
            result.corruptions_fixed += nb_clusters_reduced as i64;
            result.corruptions -= nb_clusters_reduced as i64;
        }

        // Every snapshot table entry of a v3 image needs at least 16 bytes of extra data.
        if self.qcow_version >= 3 {
            for (i, sn) in self.snapshots.iter().enumerate() {
                if sn.extra_data_size < 16 {
                    result.corruptions += 1;
                    eprintln!(
                        "{} snapshot table entry {i} is incomplete",
                        if fix & FIX_ERRORS != 0 { "Repairing" } else { "ERROR" }
                    );
                }
            }
        }
        Ok(())
    }

    /// `qcow2_check_fix_snapshot_table()`.
    pub(crate) fn check_fix_snapshot_table(
        &mut self,
        result: &mut CheckResult,
        fix: u32,
    ) -> io::Result<()> {
        if result.corruptions != 0 && fix & FIX_ERRORS != 0 {
            if let Err(e) = self.write_snapshots() {
                result.check_errors += 1;
                eprintln!(
                    "ERROR failed to update snapshot table: {}",
                    ruvm_base::error::strerror(&e)
                );
                return Err(e);
            }
            result.corruptions_fixed += result.corruptions;
            result.corruptions = 0;
        }
        Ok(())
    }

    fn find_new_snapshot_id(&self) -> String {
        let max = self.snapshots.iter().map(|sn| strtoul(&sn.id_str)).max().unwrap_or(0);
        (max + 1).to_string()
    }

    /// `find_snapshot_by_id_and_name()`.
    pub(crate) fn find_snapshot_by_id_and_name(
        &self,
        id: Option<&str>,
        name: Option<&str>,
    ) -> Option<usize> {
        match (id, name) {
            (Some(id), Some(name)) => {
                self.snapshots.iter().position(|s| s.id_str == id && s.name == name)
            }
            (Some(id), None) => self.snapshots.iter().position(|s| s.id_str == id),
            (None, Some(name)) => self.snapshots.iter().position(|s| s.name == name),
            (None, None) => None,
        }
    }

    /// `find_snapshot_by_id_or_name()`.
    pub(crate) fn find_snapshot_by_id_or_name(&self, id_or_name: &str) -> Option<usize> {
        self.find_snapshot_by_id_and_name(Some(id_or_name), None)
            .or_else(|| self.find_snapshot_by_id_and_name(None, Some(id_or_name)))
    }

    /// `qcow2_snapshot_create()`. The new ID is written back into `info`.
    pub(crate) fn snapshot_create(&mut self, info: &mut SnapshotInfo) -> io::Result<()> {
        if self.snapshots.len() as u64 >= QCOW_MAX_SNAPSHOTS {
            return Err(errno(libc::EFBIG));
        }
        if self.has_data_file() {
            return Err(errno(libc::ENOTSUP));
        }
        info.id_str = self.find_new_snapshot_id();
        let mut sn = Snapshot {
            id_str: info.id_str.clone(),
            name: info.name.clone(),
            disk_size: self.disk_size_sectors(),
            vm_state_size: info.vm_state_size,
            date_sec: info.date_sec,
            date_nsec: info.date_nsec,
            vm_clock_nsec: info.vm_clock_nsec,
            icount: info.icount.unwrap_or(u64::MAX),
            extra_data_size: SNAPSHOT_EXTRA_SIZE as u32,
            ..Default::default()
        };

        let l1_bytes = self.l1_size as u64 * L1E_SIZE;
        sn.l1_table_offset = self.alloc_clusters(l1_bytes)?;
        sn.l1_size = self.l1_size;
        let mut buf = vec![0u8; l1_bytes as usize];
        for (i, v) in self.l1_table.iter().enumerate() {
            set_be64(&mut buf, i, *v);
        }
        self.pre_write_overlap_check(0, sn.l1_table_offset, l1_bytes, false)?;
        self.file.pwrite(sn.l1_table_offset, &buf)?;

        // Take the references and make them stable before the table points to the new L1.
        self.update_snapshot_refcount(self.l1_table_offset, self.l1_size, 1)?;

        let vm_state_size = sn.vm_state_size;
        self.snapshots.push(sn);
        if let Err(e) = self.write_snapshots() {
            self.snapshots.pop();
            return Err(e);
        }

        // The VM state is not needed in the active L1 table any more, and would only cause
        // expensive COW for the next snapshot.
        let _ = self.cluster_discard(
            self.vm_state_offset(),
            vm_state_size.next_multiple_of(self.cluster_size),
            DiscardType::Never,
            false,
        );
        Ok(())
    }

    /// `qcow2_snapshot_goto()`.
    pub(crate) fn snapshot_goto(&mut self, snapshot_id: &str) -> io::Result<()> {
        if self.has_data_file() {
            return Err(errno(libc::ENOTSUP));
        }
        let Some(idx) = self.find_snapshot_by_id_or_name(snapshot_id) else {
            return Err(errno(libc::ENOENT));
        };
        let sn = self.snapshots[idx].clone();
        if let Err((e, n)) = self.validate_table(
            sn.l1_table_offset,
            sn.l1_size as u64,
            L1E_SIZE,
            QCOW_MAX_L1_SIZE,
            "Snapshot L1 table",
        ) {
            report_error(&e);
            return Err(errno(n));
        }
        if sn.disk_size != self.disk_size_sectors() {
            if let Err(e) = self.truncate(sn.disk_size, true, super::resize::Prealloc::Off) {
                report_error(&e);
                return Err(errno(libc::EINVAL));
            }
        }

        // The current L1 table must hold the whole snapshot table; a smaller one is padded
        // with zeroes.
        self.grow_l1_table(sn.l1_size as u64, true)?;
        let cur_l1_bytes = self.l1_size as u64 * L1E_SIZE;
        let sn_l1_bytes = sn.l1_size as u64 * L1E_SIZE;
        let mut sn_l1 = vec![0u8; cur_l1_bytes as usize];
        self.file.pread(sn.l1_table_offset, &mut sn_l1[..sn_l1_bytes as usize])?;

        // Take the references of the new table before the old one is overwritten on disk, and
        // drop those of the old one only after.
        self.update_snapshot_refcount(sn.l1_table_offset, sn.l1_size, 1)?;
        self.pre_write_overlap_check(OL_ACTIVE_L1, self.l1_table_offset, cur_l1_bytes, false)?;
        self.file.pwrite(self.l1_table_offset, &sn_l1)?;
        self.file.flush()?;

        // The in-memory table still is the old one, which update_snapshot_refcount() uses for
        // the active table instead of reading it from disk.
        let ret = self.update_snapshot_refcount(self.l1_table_offset, self.l1_size, -1);
        for i in 0..self.l1_size as usize {
            self.l1_table[i] = super::cache::get_be64(&sn_l1, i);
        }
        ret?;
        // The COPIED flags in the active table may have changed with the old references gone.
        self.update_snapshot_refcount(self.l1_table_offset, self.l1_size, 0)
    }

    /// `qcow2_snapshot_delete()`.
    pub(crate) fn snapshot_delete(&mut self, id: Option<&str>, name: Option<&str>) -> Result<()> {
        if self.has_data_file() {
            return Err(Error::from_io("", errno(libc::ENOTSUP)));
        }
        let Some(idx) = self.find_snapshot_by_id_and_name(id, name) else {
            return Err(Error::generic("Can't find the snapshot"));
        };
        let sn = self.snapshots[idx].clone();
        self.validate_table(
            sn.l1_table_offset,
            sn.l1_size as u64,
            L1E_SIZE,
            QCOW_MAX_L1_SIZE,
            "Snapshot L1 table",
        )
        .map_err(|(e, _)| e)?;

        self.snapshots.remove(idx);
        if let Err(e) = self.write_snapshots() {
            self.snapshots.insert(idx, sn);
            return Err(Error::from_io("Failed to remove snapshot from snapshot list", e));
        }

        // The snapshot is gone from the table. Failing from here on only leaks clusters.
        self.update_snapshot_refcount(sn.l1_table_offset, sn.l1_size, -1)
            .map_err(|e| Error::from_io("Failed to free the cluster and L1 table", e))?;
        self.free_clusters(sn.l1_table_offset, sn.l1_size as u64 * L1E_SIZE, DiscardType::Snapshot);

        // The COPIED flags of the active table may have changed.
        self.update_snapshot_refcount(self.l1_table_offset, self.l1_size, 0)
            .map_err(|e| Error::from_io("Failed to update snapshot status in disk", e))
    }

    /// `qcow2_snapshot_list()`.
    pub(crate) fn snapshot_list(&self) -> io::Result<Vec<SnapshotInfo>> {
        if self.has_data_file() {
            return Err(errno(libc::ENOTSUP));
        }
        Ok(self
            .snapshots
            .iter()
            .map(|sn| SnapshotInfo {
                id_str: sn.id_str.clone(),
                name: sn.name.clone(),
                vm_state_size: sn.vm_state_size,
                date_sec: sn.date_sec,
                date_nsec: sn.date_nsec,
                vm_clock_nsec: sn.vm_clock_nsec,
                icount: if sn.icount == u64::MAX { None } else { Some(sn.icount) },
            })
            .collect())
    }

    /// `qcow2_snapshot_load_tmp()`: switches a read-only image to a snapshot's L1 table.
    pub(crate) fn snapshot_load_tmp(&mut self, id: Option<&str>, name: Option<&str>) -> Result<()> {
        assert!(!self.writable());
        let Some(idx) = self.find_snapshot_by_id_and_name(id, name) else {
            return Err(Error::generic("Can't find snapshot"));
        };
        let sn = self.snapshots[idx].clone();
        self.validate_table(
            sn.l1_table_offset,
            sn.l1_size as u64,
            L1E_SIZE,
            QCOW_MAX_L1_SIZE,
            "Snapshot L1 table",
        )
        .map_err(|(e, _)| e)?;
        let mut buf = vec![0u8; sn.l1_size as usize * 8];
        self.file
            .pread(sn.l1_table_offset, &mut buf)
            .map_err(|_| Error::generic("Failed to read l1 table for snapshot"))?;
        self.l1_size = sn.l1_size;
        self.l1_table_offset = sn.l1_table_offset;
        self.l1_table =
            buf.chunks_exact(8).map(|c| u64::from_be_bytes(c.try_into().unwrap())).collect();
        Ok(())
    }
}

/// `strtoul(s, NULL, 10)`: leading digits, 0 without any, saturating like `ULONG_MAX`.
fn strtoul(s: &str) -> u64 {
    let s = s.trim_start();
    let s = s.strip_prefix('+').unwrap_or(s);
    let mut v: u64 = 0;
    for c in s.bytes() {
        if !c.is_ascii_digit() {
            break;
        }
        v = v.saturating_mul(10).saturating_add((c - b'0') as u64);
    }
    v
}
