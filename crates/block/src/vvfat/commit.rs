// SPDX-License-Identifier: GPL-2.0-or-later

//! The write support of block/vvfat.c. The guest's writes go to the write target; after each
//! one the file system on the disk is checked, and when it is consistent the changes are
//! committed to the host directory:
//!
//! 1. check that all data is consistent, recording renames, changed files and new files and
//!    directories (in `commits`);
//! 2. stop if it is not;
//! 3. handle renames, and create new files and directories (without their contents yet);
//! 4. walk the directories, fixing the mappings and directory entries and marking the
//!    handled mappings as not deleted;
//! 5. write the contents of the files;
//! 6. remove deleted files and directories.

use std::io::{self, Seek, SeekFrom, Write};

use ruvm_base::report;

use super::{
    BDRV_SECTOR_SIZE, Commit, DIR_KANJI, DIR_KANJI_FAKE, Direntry, MODE_DELETED, MODE_DIRECTORY,
    MODE_NORMAL, Mapping, Node, State, begin_of_direntry, errno, fat_chksum, fat_entry,
    filesize_of_direntry, is_directory, is_dot, is_file, is_free, is_long_name, is_short_name,
    is_volume_label, to_valid_short_char, valid_filename,
};

/// Bits of `used_clusters`: part of a directory, of a file, and written by the guest.
const USED_DIRECTORY: u8 = 1;
const USED_FILE: u8 = 2;
const USED_ANY: u8 = 3;
const USED_ALLOCATED: u8 = 4;

/// The host's `PATH_MAX`.
#[cfg(target_os = "linux")]
const PATH_MAX: usize = 4096;
#[cfg(windows)]
const PATH_MAX: usize = 260;
#[cfg(not(any(target_os = "linux", windows)))]
const PATH_MAX: usize = 1024;

/// `long_file_name`: the name being put together from long name entries.
#[derive(Debug)]
struct LongFileName {
    name: Vec<u8>,
    name2: Vec<u16>,
    checksum: u32,
    len: usize,
    sequence_number: i32,
}

impl LongFileName {
    /// `lfn_init()`.
    fn new() -> Self {
        LongFileName {
            name: Vec::new(),
            name2: vec![0; 0x3f * 13 + 1],
            checksum: 0x100,
            len: 0,
            sequence_number: 0,
        }
    }

    /// `parse_long_name()`: 0 when the entry was taken, more when it is no long name entry,
    /// less on an error.
    fn parse_long_name(&mut self, d: &Direntry) -> i32 {
        if !is_long_name(d) {
            return 1;
        }
        if d[0] & 0x40 != 0 {
            // The first entry.
            self.sequence_number = i32::from(d[0] & 0x3f);
            self.checksum = u32::from(d[13]);
            self.name.clear();
        } else {
            self.sequence_number -= 1;
            if i32::from(d[0] & 0x3f) != self.sequence_number {
                // Not the expected sequence number.
                return -1;
            } else if u32::from(d[13]) != self.checksum {
                // Not the expected checksum.
                return -2;
            } else if d[12] != 0 || d[26] != 0 || d[27] != 0 {
                // Fields that must be zero.
                return -3;
            }
        }
        if self.sequence_number < 1 {
            return -1;
        }

        let offset = 13 * (self.sequence_number as usize - 1);
        let mut i = 0;
        let mut j = 1;
        while i < 13 {
            if j == 11 {
                j = 14;
            } else if j == 26 {
                j = 28;
            }
            if d[j] == 0 && d[j + 1] == 0 {
                // The end of the long file name.
                break;
            }
            self.name2[offset + i] = u16::from_le_bytes([d[j], d[j + 1]]);
            i += 1;
            j += 2;
        }

        if d[0] & 0x40 != 0 {
            self.len = offset + i;
        }
        if d[0] & 0x3f == 0x01 {
            // The last entry.
            let Ok(utf8) = String::from_utf16(&self.name2[..self.len]) else { return -4 };
            self.name = utf8.into_bytes();
            self.len = self.name.len();
        }
        0
    }

    /// `parse_short_name()`: 0 when the entry was taken, more when it is no short name entry,
    /// less on an error.
    fn parse_short_name(&mut self, downcase: bool, d: &Direntry) -> i32 {
        if !is_short_name(d) {
            return 1;
        }
        let lower = |c: u8| if downcase { c.to_ascii_lowercase() } else { c };
        let mut name = Vec::with_capacity(12);
        let base = d[..8].iter().rposition(|&c| c != b' ').map_or(0, |j| j + 1);
        for &c in &d[..base] {
            if c != to_valid_short_char(char::from(c)) {
                return -1;
            }
            name.push(lower(c));
        }
        let ext = d[8..11].iter().rposition(|&c| c != b' ').map_or(0, |j| j + 1);
        if ext > 0 {
            name.push(b'.');
            for &c in &d[8..8 + ext] {
                if c != to_valid_short_char(char::from(c)) {
                    return -2;
                }
                name.push(lower(c));
            }
        }
        if name.first() == Some(&DIR_KANJI_FAKE) {
            name[0] = DIR_KANJI;
        }
        // strlen().
        if let Some(nul) = name.iter().position(|&c| c == 0) {
            name.truncate(nul);
        }
        self.len = name.len();
        self.name = name;
        0
    }
}

/// `get_basename()`.
fn get_basename(path: &str) -> &str {
    path.rsplit_once('/').map_or(path, |(_, b)| b)
}

/// `strerror()`.
fn strerror(e: &io::Error) -> String {
    let s = e.to_string();
    match s.find(" (os error") {
        Some(i) => s[..i].to_string(),
        None => s,
    }
}

impl State {
    /// `modified_fat_get()`: the FAT as the guest wrote it.
    fn modified_fat_get(&self, cluster: u32) -> u32 {
        if cluster < self.last_cluster_of_root_directory {
            if cluster + 1 == self.last_cluster_of_root_directory {
                return self.max_fat_value;
            }
            return cluster + 1;
        }
        fat_entry(self.fat2.as_deref().unwrap_or(&[]), self.fat_type, cluster)
    }

    /// `cluster_was_modified()`: whether the guest wrote to the cluster. Not knowing counts
    /// as written.
    fn cluster_was_modified(&self, q: Option<&Node>, cluster_num: u32) -> bool {
        let Some(q) = q.filter(|_| self.qcow) else { return false };
        (0..u64::from(self.sectors_per_cluster)).any(|i| {
            q.is_allocated(
                (self.cluster2sector(cluster_num) + i) * BDRV_SECTOR_SIZE,
                BDRV_SECTOR_SIZE,
            )
            .map_or(true, |(a, _)| a)
        })
    }

    /// The `used_clusters` entry of a cluster, `None` past the end.
    fn used(&mut self, cluster: u32) -> Option<&mut u8> {
        self.used_clusters.get_mut(cluster as usize)
    }

    /// `get_cluster_count_for_direntry()`: how many clusters the file of `d` takes, noting
    /// whether it was renamed or changed. A file is renamed only if there was a file with
    /// the same first cluster and a different name. The files handled here are not deleted.
    fn get_cluster_count_for_direntry(
        &mut self,
        q: Option<&Node>,
        d: &Direntry,
        path: &str,
    ) -> i32 {
        // If the guest inserted a cluster into a chain (15 -> 16 became 15 -> 32 -> 16),
        // writing the new cluster at its offset would overwrite data that should move to a
        // later position. That is detected, and the clusters to be overwritten are copied
        // into the write target.
        let mut copy_it = false;
        let mut was_modified = false;
        let mut ret = 0;

        let mut cluster_num = begin_of_direntry(d);
        let mut offset = 0u32;
        let mut mapping: Option<usize> = None;
        let basename2 = get_basename(path).to_string();

        self.close_current_file();

        // The root directory.
        if cluster_num == 0 {
            return 0;
        }

        if self.qcow {
            mapping = self.find_mapping_for_cluster(cluster_num);
            if let Some(m) = mapping {
                let mm = &mut self.mapping[m];
                if mm.mode & MODE_DELETED == 0 || mm.mode & MODE_NORMAL == 0 {
                    // Two entries share the cluster, or a file took a directory's.
                    return -1;
                }
                mm.mode &= !MODE_DELETED;
                if get_basename(&mm.path) != basename2 {
                    self.commits
                        .push(Commit::Rename { cluster: cluster_num, path: path.to_string() });
                }
            } else if is_file(d) {
                self.commits
                    .push(Commit::NewFile { first_cluster: cluster_num, path: path.to_string() });
            } else {
                return 0;
            }
        }

        loop {
            if self.qcow {
                if !copy_it && self.cluster_was_modified(q, cluster_num) {
                    let inside = mapping.is_some_and(|m| {
                        self.mapping[m].begin <= cluster_num && self.mapping[m].end > cluster_num
                    });
                    if !inside {
                        mapping = self.find_mapping_for_cluster(cluster_num);
                    }
                    if let Some(m) = mapping.filter(|&m| self.mapping[m].mode & MODE_DIRECTORY == 0)
                    {
                        let mm = &self.mapping[m];
                        // Written in the write target.
                        let expected = self
                            .cluster_size
                            .wrapping_mul((cluster_num - mm.begin).wrapping_add(mm.offset()));
                        if offset != expected {
                            // The offset of this cluster in the chain has changed.
                            copy_it = true;
                        } else if offset == 0 && get_basename(&mm.path) != basename2 {
                            copy_it = true;
                        }
                        // Does it need writing out?
                        if !was_modified && is_file(d) {
                            was_modified = true;
                            let dir_index = mm.dir_index as usize;
                            self.commits
                                .push(Commit::Writeout { dir_index, modified_offset: offset });
                        }
                    }
                }

                if copy_it {
                    // Horribly inefficient, but rarely done if at all.
                    let offs = self.cluster2sector(cluster_num);
                    self.close_current_file();
                    let Some(q) = q else { return -1 };
                    let mut sector = [0u8; 0x200];
                    for i in 0..u64::from(self.sectors_per_cluster) {
                        let at = (offs + i) * BDRV_SECTOR_SIZE;
                        match q.is_allocated(at, BDRV_SECTOR_SIZE) {
                            Err(_) => return -1,
                            Ok((true, _)) => {}
                            Ok((false, _)) => {
                                if self.read(Some(q), offs + i, &mut sector, 1).is_err() {
                                    return -1;
                                }
                                if q.pwrite(at, &sector).is_err() {
                                    return -2;
                                }
                            }
                        }
                    }
                }
            }

            ret += 1;
            let Some(used) = self.used(cluster_num) else { return -1 };
            if *used & USED_ANY != 0 {
                return 0;
            }
            *used = USED_FILE;

            cluster_num = self.modified_fat_get(cluster_num);

            if self.fat_eof(cluster_num) {
                return ret;
            } else if cluster_num < 2 || cluster_num > self.max_fat_value - 16 {
                return -1;
            }

            offset = offset.wrapping_add(self.cluster_size);
        }
    }

    /// `check_directory_consistency()`: looks at the directory as the guest changed it and
    /// returns the number of clusters it, its subdirectories and their files use, or 0 when
    /// it is inconsistent.
    fn check_directory_consistency(
        &mut self,
        q: Option<&Node>,
        mut cluster_num: u32,
        path: &str,
    ) -> i32 {
        let mut ret = 0;
        let mut cluster = vec![0u8; self.cluster_size as usize];
        let mapping = self.find_mapping_for_cluster(cluster_num);

        if let Some(m) = mapping {
            let mm = &mut self.mapping[m];
            if mm.mode & MODE_DIRECTORY == 0 || mm.mode & MODE_DELETED == 0 {
                return 0;
            }
            mm.mode &= !MODE_DELETED;
            if get_basename(&mm.path) != get_basename(path) {
                self.commits.push(Commit::Rename { cluster: cluster_num, path: path.to_string() });
            }
        } else {
            // A new directory.
            self.commits.push(Commit::Mkdir { cluster: cluster_num, path: path.to_string() });
        }

        let mut lfn = LongFileName::new();
        loop {
            ret += 1;

            match self.used(cluster_num) {
                Some(used) if *used & USED_ANY == 0 => *used = USED_DIRECTORY,
                _ => {
                    eprintln!("cluster {} used more than once", cluster_num as i32);
                    return 0;
                }
            }

            let sector = self.cluster2sector(cluster_num);
            let spc = self.sectors_per_cluster as usize;
            if self.read(q, sector, &mut cluster, spc).is_err() {
                eprintln!("Error fetching direntries");
                return 0;
            }

            for i in 0..0x10 * spc {
                let d: Direntry = cluster[i * 32..i * 32 + 32].try_into().unwrap();
                if is_volume_label(&d) || is_dot(&d) || is_free(&d) {
                    continue;
                }

                let subret = lfn.parse_long_name(&d);
                if subret < 0 {
                    eprintln!("Error in long name");
                    return 0;
                }
                if subret == 0 {
                    continue;
                }

                if u32::from(fat_chksum(&d)) != lfn.checksum {
                    let subret = lfn.parse_short_name(self.downcase_short_names, &d);
                    if subret < 0 {
                        eprintln!("Error in short name ({subret})");
                        return 0;
                    }
                    if subret > 0 || lfn.name == b"." || lfn.name == b".." {
                        continue;
                    }
                }
                // A long name cannot be used twice.
                lfn.checksum = 0x100;

                if !valid_filename(&lfn.name) {
                    eprintln!("Invalid file name");
                    return 0;
                }
                let name = String::from_utf8_lossy(&lfn.name).into_owned();
                if path.len() + 1 + lfn.len >= PATH_MAX {
                    eprintln!("Name too long: {path}/{name}");
                    return 0;
                }
                let path2 = format!("{path}/{name}");

                let cluster_count = if is_directory(&d) {
                    if begin_of_direntry(&d) == 0 {
                        return 0;
                    }
                    let n = self.check_directory_consistency(q, begin_of_direntry(&d), &path2);
                    if n == 0 {
                        return 0;
                    }
                    n
                } else if is_file(&d) {
                    // Check the file size against the FAT.
                    let n = self.get_cluster_count_for_direntry(q, &d, &path2);
                    if i64::from(n)
                        != i64::from(filesize_of_direntry(&d).div_ceil(self.cluster_size))
                    {
                        return 0;
                    }
                    n
                } else {
                    return 0;
                };
                ret += cluster_count;
            }

            cluster_num = self.modified_fat_get(cluster_num);
            if self.fat_eof(cluster_num) {
                break;
            }
        }
        ret
    }

    /// `is_consistent()`: the number of clusters in use when the file system the guest left
    /// is consistent, 0 otherwise.
    fn is_consistent(&mut self, q: Option<&Node>) -> i32 {
        // Get the FAT as the guest wrote it, then walk the directories from the root through
        // this driver's reads so they see the changes, checking the FAT against the sizes
        // and counting the clusters, and check that count against the FAT.
        let size = 0x200 * self.sectors_per_fat as usize;
        let mut fat2 = self.fat2.take().unwrap_or_else(|| self.fat[..size].to_vec());
        let r =
            self.read(q, u64::from(self.offset_to_fat), &mut fat2, self.sectors_per_fat as usize);
        self.fat2 = Some(fat2);
        if r.is_err() {
            eprintln!("Could not copy fat");
            return 0;
        }
        let n = self.sector2cluster(i64::from(self.sector_count)).max(0) as usize;
        for u in self.used_clusters.iter_mut().take(n) {
            *u &= !USED_ANY;
        }

        self.commits.clear();

        // Mark every mapped file and directory as deleted; check_directory_consistency()
        // unmarks those still there.
        if self.qcow {
            for m in &mut self.mapping {
                if m.first_mapping_index < 0 {
                    m.mode |= MODE_DELETED;
                }
            }
        }

        let root = self.path.clone();
        let used_clusters_count = self.check_directory_consistency(q, 0, &root);
        if used_clusters_count <= 0 {
            return 0;
        }

        let mut check = self.last_cluster_of_root_directory as i32;
        for i in self.last_cluster_of_root_directory..n as u32 {
            let used = self.used_clusters.get(i as usize).copied().unwrap_or(0);
            if self.modified_fat_get(i) != 0 {
                if used == 0 {
                    // The FAT was changed, but the cluster is not used.
                    return 0;
                }
                check += 1;
            }
            if used == USED_ALLOCATED {
                // Allocated, but not used.
                return 0;
            }
        }

        if check != used_clusters_count {
            return 0;
        }
        used_clusters_count
    }

    /// `adjust_mapping_indices()`.
    fn adjust_mapping_indices(&mut self, offset: i32, adjust: i32) {
        for m in &mut self.mapping {
            if m.first_mapping_index >= offset {
                m.first_mapping_index += adjust;
            }
            if m.mode & MODE_DIRECTORY != 0 && m.parent_mapping_index() >= offset {
                let p = m.parent_mapping_index() + adjust;
                m.set_parent_mapping_index(p);
            }
        }
    }

    /// `insert_mapping()`: the mapping starting at `begin`, made if there is none, whose end
    /// becomes `end`.
    fn insert_mapping(&mut self, begin: u32, end: u32) -> usize {
        self.close_current_file();
        let mut index = self.find_mapping_for_cluster_aux(i64::from(begin), 0, self.mapping.len());
        if index < self.mapping.len() && self.mapping[index].begin < begin {
            self.mapping[index].end = begin;
            index += 1;
        }
        if index >= self.mapping.len() || self.mapping[index].begin > begin {
            // array_insert() leaves a copy of the mapping that was there.
            let mut m = self
                .mapping
                .get(index)
                .cloned()
                .unwrap_or(Mapping { first_mapping_index: -1, ..Mapping::default() });
            m.path = String::new();
            self.mapping.insert(index, m);
            self.adjust_mapping_indices(index as i32, 1);
        }
        self.mapping[index].begin = begin;
        self.mapping[index].end = end;
        index
    }

    /// `remove_mapping()`.
    fn remove_mapping(&mut self, mapping_index: usize) {
        self.close_current_file();
        self.mapping.remove(mapping_index);
        self.adjust_mapping_indices(mapping_index as i32, -1);
    }

    /// `adjust_dirindices()`.
    fn adjust_dirindices(&mut self, offset: usize, adjust: isize) {
        for m in &mut self.mapping {
            if m.dir_index as usize >= offset {
                m.dir_index = (m.dir_index as isize + adjust) as u32;
            }
            if m.mode & MODE_DIRECTORY != 0 && m.first_dir_index >= offset as i32 {
                m.first_dir_index += adjust as i32;
            }
        }
    }

    /// `insert_direntries()`.
    fn insert_direntries(&mut self, dir_index: usize, count: usize) {
        let at = dir_index.min(self.directory.len());
        self.directory.splice(at..at, std::iter::repeat_n([0u8; 32], count));
        self.adjust_dirindices(dir_index, count as isize);
    }

    /// `remove_direntries()`.
    fn remove_direntries(&mut self, dir_index: usize, count: usize) {
        let end = (dir_index + count).min(self.directory.len());
        self.directory.drain(dir_index.min(end)..end);
        self.adjust_dirindices(dir_index, -(count as isize));
    }

    /// `commit_mappings()`: follows the chain from `first_cluster` in the changed FAT and
    /// adjusts the mappings to it.
    fn commit_mappings(&mut self, first_cluster: u32, dir_index: i32) -> i32 {
        let Some(mut m) = self.find_mapping_for_cluster(first_cluster) else { return -1 };
        let mut cluster = first_cluster;

        self.close_current_file();

        if self.mapping[m].begin != first_cluster {
            return -1;
        }
        let is_dir =
            dir_index <= 0 || self.directory.get(dir_index as usize).is_some_and(is_directory);
        let mm = &mut self.mapping[m];
        mm.first_mapping_index = -1;
        mm.dir_index = dir_index as u32;
        mm.mode = if is_dir { MODE_DIRECTORY } else { MODE_NORMAL };

        let mut steps = 0;
        while !self.fat_eof(cluster) {
            steps += 1;
            if steps > self.cluster_count + 2 {
                return -1;
            }
            let mut c = cluster;
            let mut c1 = self.modified_fat_get(c);
            while c.wrapping_add(1) == c1 {
                c = c1;
                c1 = self.modified_fat_get(c1);
            }
            c += 1;

            if c > self.mapping[m].end {
                let max_i = self.mapping.len() - m;
                let mut i = 1;
                while i < max_i && self.mapping[m + i].begin < c {
                    i += 1;
                }
                while i > 1 {
                    i -= 1;
                    self.remove_mapping(m + 1);
                }
            }
            self.mapping[m].end = c;

            if !self.fat_eof(c1) {
                let i = self.find_mapping_for_cluster_aux(i64::from(c1), 0, self.mapping.len());
                let next = match self.mapping.get(i) {
                    Some(n) if n.begin <= c1 => i,
                    _ => {
                        let mut i1 = m;
                        let n = self.insert_mapping(c1, c1 + 1);
                        if c1 < c {
                            i1 += 1;
                        }
                        m = i1;
                        n
                    }
                };

                let cur = self.mapping[m].clone();
                let spc = self.sectors_per_cluster;
                let nm = &mut self.mapping[next];
                nm.dir_index = cur.dir_index;
                nm.first_mapping_index =
                    if cur.first_mapping_index < 0 { m as i32 } else { cur.first_mapping_index };
                nm.path = cur.path.clone();
                nm.mode = cur.mode;
                nm.read_only = cur.read_only;
                if cur.mode & MODE_DIRECTORY != 0 {
                    nm.set_parent_mapping_index(cur.parent_mapping_index());
                    nm.first_dir_index =
                        cur.first_dir_index + (0x10 * spc * (cur.end - cur.begin)) as i32;
                } else {
                    nm.info0 = cur.offset() + (cur.end - cur.begin);
                }
                m = next;
            }

            cluster = c1;
        }
        0
    }

    /// The length of the chain from `first` in a FAT, or `None` for a loop.
    fn chain_len(&self, first: u32, next: impl Fn(&Self, u32) -> u32) -> Option<usize> {
        let mut n = 0;
        let mut c = first;
        while !self.fat_eof(c) {
            n += 1;
            if n > self.cluster_count as usize + 2 {
                return None;
            }
            c = next(self, c);
        }
        Some(n)
    }

    /// `commit_direntries()`: takes the guest's version of the directory whose entry is
    /// `dir_index` (0 for the root), and of the directories below it.
    fn commit_direntries(
        &mut self,
        q: Option<&Node>,
        dir_index: usize,
        parent_mapping_index: i32,
    ) -> i32 {
        let Some(&direntry) = self.directory.get(dir_index) else { return -1 };
        let first_cluster = if dir_index == 0 { 0 } else { begin_of_direntry(&direntry) };
        let Some(m) = self.find_mapping_for_cluster(first_cluster) else { return -1 };
        let factor = 0x10 * self.sectors_per_cluster as usize;

        {
            let mm = &self.mapping[m];
            if mm.begin != first_cluster || mm.mode & MODE_DIRECTORY == 0 {
                return -1;
            }
        }
        if dir_index != 0 && !is_directory(&direntry) {
            return -1;
        }

        let mut current_dir_index = self.mapping[m].first_dir_index as usize;
        let first_dir_index = current_dir_index;
        self.mapping[m].set_parent_mapping_index(parent_mapping_index);

        let (old_cluster_count, new_cluster_count) = if first_cluster == 0 {
            let n = self.last_cluster_of_root_directory as usize;
            (n, n)
        } else {
            let old = self.chain_len(first_cluster, State::fat_get);
            let new = self.chain_len(first_cluster, State::modified_fat_get);
            match (old, new) {
                (Some(o), Some(n)) => (o, n),
                _ => return -1,
            }
        };

        if new_cluster_count > old_cluster_count {
            self.insert_direntries(
                current_dir_index + factor * old_cluster_count,
                factor * (new_cluster_count - old_cluster_count),
            );
        } else if new_cluster_count < old_cluster_count {
            self.remove_direntries(
                current_dir_index + factor * new_cluster_count,
                factor * (old_cluster_count - new_cluster_count),
            );
        }

        let mut buf = vec![0u8; self.cluster_size as usize];
        let mut c = first_cluster;
        let mut steps = 0;
        while !self.fat_eof(c) {
            steps += 1;
            if steps > new_cluster_count {
                return -1;
            }
            let spc = self.sectors_per_cluster as usize;
            if self.read(q, self.cluster2sector(c), &mut buf, spc).is_err() {
                return -1;
            }
            if current_dir_index + factor > self.directory.len() {
                return -1;
            }
            for (k, chunk) in buf.chunks(32).enumerate() {
                self.directory[current_dir_index + k].copy_from_slice(chunk);
            }
            current_dir_index += factor;
            c = self.modified_fat_get(c);
        }

        let ret = self.commit_mappings(first_cluster, dir_index as i32);
        if ret != 0 {
            return ret;
        }

        // Recurse.
        for i in 0..factor * new_cluster_count {
            let Some(d) = self.directory.get(first_dir_index + i) else { return -1 };
            if is_directory(d) && !is_dot(d) {
                // The parent's mapping, found by its first cluster as in QEMU.
                let Some(m) = self.find_mapping_for_cluster(first_cluster) else { return -1 };
                if self.mapping[m].mode & MODE_DIRECTORY == 0 {
                    return -1;
                }
                let ret = self.commit_direntries(q, first_dir_index + i, m as i32);
                if ret != 0 {
                    return ret;
                }
            }
        }
        0
    }

    /// `commit_one_file()`: writes the file of entry `dir_index` from `offset` on to the host
    /// and adjusts its mappings.
    fn commit_one_file(&mut self, q: Option<&Node>, dir_index: usize, mut offset: u32) -> i32 {
        let Some(&direntry) = self.directory.get(dir_index) else { return -1 };
        let mut c = begin_of_direntry(&direntry);
        let first_cluster = c;
        let Some(m) = self.find_mapping_for_cluster(c) else { return -1 };
        let size = filesize_of_direntry(&direntry);

        if offset >= size || offset % self.cluster_size != 0 {
            return -1;
        }

        let mut i = 0;
        while i < offset {
            c = self.modified_fat_get(c);
            i += self.cluster_size;
        }

        let path = self.mapping[m].path.clone();
        let mut oo = std::fs::OpenOptions::new();
        oo.read(true).write(true).create(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut oo, 0o666);
        let mut f = match oo.open(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!(
                    "Could not open {path}... ({}, {})",
                    strerror(&e),
                    e.raw_os_error().unwrap_or(0)
                );
                return -1;
            }
        };
        if offset > 0 && f.seek(SeekFrom::Start(u64::from(offset))).ok() != Some(u64::from(offset))
        {
            return -3;
        }

        let mut cluster = vec![0u8; self.cluster_size as usize];
        while offset < size {
            let rest_size = (size - offset).min(self.cluster_size) as usize;
            let c1 = self.modified_fat_get(c);
            if c < 2 || self.fat_eof(c) {
                return -1;
            }
            let n = rest_size.div_ceil(0x200);
            if self.read(q, self.cluster2sector(c), &mut cluster[..n * 0x200], n).is_err() {
                return -1;
            }
            if f.write_all(&cluster[..rest_size]).is_err() {
                return -2;
            }
            offset += rest_size as u32;
            c = c1;
        }

        if let Err(e) = f.set_len(u64::from(size)) {
            eprintln!("ftruncate(): {}", strerror(&e));
            return -4;
        }
        drop(f);

        self.commit_mappings(first_cluster, dir_index as i32)
    }

    /// `handle_renames_and_mkdirs()`.
    fn handle_renames_and_mkdirs(&mut self) -> i32 {
        let mut i = 0;
        while i < self.commits.len() {
            match self.commits[i].clone() {
                Commit::Rename { cluster, path } => {
                    let Some(m) = self.find_mapping_for_cluster(cluster) else { return -1 };
                    let old_path = std::mem::replace(&mut self.mapping[m].path, path.clone());
                    if std::fs::rename(&old_path, &path).is_err() {
                        return -2;
                    }
                    // The later parts of the file have the same path.
                    for (k, mm) in self.mapping.iter_mut().enumerate() {
                        if mm.first_mapping_index == m as i32 && k != m {
                            mm.path = path.clone();
                        }
                    }

                    if self.mapping[m].mode & MODE_DIRECTORY != 0 {
                        let l2 = old_path.len();
                        let base = self.mapping[m].first_dir_index as usize;
                        let mut c = self.mapping[m].begin;
                        let per_cluster = 0x10 * self.sectors_per_cluster as usize;
                        let mut j = 0;
                        let mut steps = 0;
                        // Recurse.
                        while !self.fat_eof(c) {
                            steps += 1;
                            if steps > self.cluster_count + 2 {
                                return -1;
                            }
                            loop {
                                let Some(&d) = self.directory.get(base + j) else { return -1 };
                                if is_file(&d) || (is_directory(&d) && !is_dot(&d)) {
                                    let Some(m2) =
                                        self.find_mapping_for_cluster(begin_of_direntry(&d))
                                    else {
                                        return -1;
                                    };
                                    let p2 = &self.mapping[m2].path;
                                    let Some(rest) = p2.get(l2..) else { return -1 };
                                    let new_path = format!("{path}{rest}");
                                    let begin = self.mapping[m2].begin;
                                    self.commits
                                        .push(Commit::Rename { cluster: begin, path: new_path });
                                }
                                j += 1;
                                if j % per_cluster == 0 {
                                    break;
                                }
                            }
                            c = self.fat_get(c);
                        }
                    }

                    self.commits.remove(i);
                }
                Commit::Mkdir { cluster, path } => {
                    #[cfg(unix)]
                    let b = {
                        let mut b = std::fs::DirBuilder::new();
                        std::os::unix::fs::DirBuilderExt::mode(&mut b, 0o755);
                        b
                    };
                    #[cfg(not(unix))]
                    let b = std::fs::DirBuilder::new();
                    if b.create(&path).is_err() {
                        return -5;
                    }

                    let m = self.insert_mapping(cluster, cluster + 1);
                    {
                        let mm = &mut self.mapping[m];
                        mm.mode = MODE_DIRECTORY;
                        mm.read_only = false;
                        mm.path = path.clone();
                    }
                    let j = self.directory.len();
                    self.insert_direntries(j, 0x10 * self.sectors_per_cluster as usize);
                    self.mapping[m].first_dir_index = j as i32;

                    let parent_path_len = path.len() - get_basename(&path).len() - 1;
                    let parent = self.mapping.iter().enumerate().position(|(k, mm)| {
                        mm.first_mapping_index < 0
                            && k != m
                            && mm.path.len() == parent_path_len
                            && path.as_bytes().starts_with(mm.path.as_bytes())
                    });
                    let Some(parent) = parent else { return -6 };
                    self.mapping[m].set_parent_mapping_index(parent as i32);

                    self.commits.remove(i);
                }
                _ => i += 1,
            }
        }
        0
    }

    /// `handle_commits()`: writes the changed and new files.
    fn handle_commits(&mut self, q: Option<&Node>) -> i32 {
        let mut fail = 0;
        self.close_current_file();

        let mut i = 0;
        while fail == 0 && i < self.commits.len() {
            match self.commits[i].clone() {
                Commit::Rename { .. } | Commit::Mkdir { .. } => fail = -2,
                Commit::Writeout { dir_index, modified_offset } => {
                    let Some(entry) = self.directory.get(dir_index) else { return -3 };
                    let begin = begin_of_direntry(entry);
                    match self.find_mapping_for_cluster(begin) {
                        Some(m) if self.mapping[m].begin == begin => {}
                        _ => return -3,
                    }
                    if self.commit_one_file(q, dir_index, modified_offset) != 0 {
                        fail = -3;
                    }
                }
                Commit::NewFile { first_cluster: begin, path } => {
                    let mut mapping = self.find_mapping_for_cluster(begin);

                    // Find the directory entry.
                    let j = self
                        .directory
                        .iter()
                        .position(|e| is_file(e) && begin_of_direntry(e) == begin);
                    let Some(j) = j else {
                        fail = -6;
                        i += 1;
                        continue;
                    };

                    // Make sure there is a first mapping.
                    if let Some(m) = mapping.filter(|&m| self.mapping[m].begin != begin) {
                        self.mapping[m].end = begin;
                        mapping = None;
                    }
                    let m = match mapping {
                        Some(m) => m,
                        None => self.insert_mapping(begin, begin + 1),
                    };
                    // commit_mappings() fixes most of the fields.
                    let mm = &mut self.mapping[m];
                    mm.path = path;
                    mm.read_only = false;
                    mm.mode = MODE_NORMAL;
                    mm.info0 = 0;

                    if self.commit_one_file(q, j, 0) != 0 {
                        fail = -7;
                    }
                }
            }
            i += 1;
        }
        if i > 0 {
            self.commits.drain(..i.min(self.commits.len()));
        }
        fail
    }

    /// `handle_deletes()`: removes the files and directories whose mappings are still marked
    /// as deleted.
    fn handle_deletes(&mut self) -> i32 {
        let mut deferred = 1;
        let mut deleted = 1;

        while deferred != 0 && deleted != 0 {
            deferred = 0;
            deleted = 0;

            let mut i = 1;
            while i < self.mapping.len() {
                let m = &self.mapping[i];
                if m.mode & MODE_DELETED != 0 {
                    let entry_free = self.directory.get(m.dir_index as usize).is_none_or(is_free);
                    if entry_free {
                        if m.mode & MODE_DIRECTORY != 0 {
                            let first_dir_index = m.first_dir_index;
                            if let Err(e) = std::fs::remove_dir(&m.path) {
                                if e.raw_os_error() == Some(libc::ENOTEMPTY) {
                                    deferred += 1;
                                    i += 1;
                                    continue;
                                }
                                return -5;
                            }
                            let mut next_dir_index = self.directory.len() as i32;
                            for mm in &self.mapping[1..] {
                                if mm.mode & MODE_DIRECTORY != 0
                                    && mm.first_dir_index > first_dir_index
                                    && mm.first_dir_index < next_dir_index
                                {
                                    next_dir_index = mm.first_dir_index;
                                }
                            }
                            self.remove_direntries(
                                first_dir_index as usize,
                                (next_dir_index - first_dir_index) as usize,
                            );
                            deleted += 1;
                        }
                    } else {
                        if std::fs::remove_file(&m.path).is_err() {
                            return -4;
                        }
                        deleted += 1;
                    }
                    self.remove_mapping(i);
                }
                i += 1;
            }
        }
        0
    }

    /// `do_commit()`: brings the host directory in line with the checked file system.
    fn do_commit(&mut self, q: Option<&Node>) -> i32 {
        // The real work is in the commits. Nothing to do? Move along!
        if self.commits.is_empty() {
            return 0;
        }

        self.close_current_file();

        let ret = self.handle_renames_and_mkdirs();
        if ret != 0 {
            eprintln!("Error handling renames ({ret})");
            return ret;
        }

        // Copy the FAT.
        let size = 0x200 * self.sectors_per_fat as usize;
        if let Some(fat2) = &self.fat2 {
            self.fat[..size].copy_from_slice(&fat2[..size]);
        }

        // Recurse the directories from the root.
        let ret = self.commit_direntries(q, 0, -1);
        if ret != 0 {
            eprintln!("Fatal: error while committing ({ret})");
            return ret;
        }

        let ret = self.handle_commits(q);
        if ret != 0 {
            eprintln!("Error handling commits ({ret})");
            return ret;
        }

        let ret = self.handle_deletes();
        if ret != 0 {
            eprintln!("Error deleting");
            return ret;
        }

        if let Some(q) = q {
            let _ = q.make_empty();
        }

        let n = self.sector2cluster(i64::from(self.sector_count)).max(0) as usize;
        for u in self.used_clusters.iter_mut().take(n) {
            *u = 0;
        }
        0
    }

    /// `try_commit()`.
    fn try_commit(&mut self, q: Option<&Node>) -> i32 {
        self.close_current_file();
        if self.is_consistent(q) == 0 {
            return -1;
        }
        self.do_commit(q)
    }

    /// `vvfat_write()`.
    pub(super) fn write(
        &mut self,
        q: Option<&Node>,
        sector_num: u64,
        buf: &[u8],
        nb_sectors: usize,
    ) -> io::Result<()> {
        // Read-only mode?
        let Some(q) = q.filter(|_| self.qcow) else { return Err(errno(libc::EACCES)) };

        self.close_current_file();

        let obs = self.offset_to_bootsector as usize;
        if sector_num == obs as u64 && nb_sectors == 1 {
            // A write to the boot sector may only change the reserved1 field, which marks
            // the volume dirty. LATER TODO in QEMU: this is wrong for FAT32, which gets a
            // FAT16 boot sector too.
            const RESERVED1_OFFSET: usize = 37;
            let bootsector = &mut self.first_sectors[obs * 0x200..][..0x200];
            for i in 0..0x200 {
                if i != RESERVED1_OFFSET && bootsector[i] != buf[i] {
                    eprintln!("Tried to write to protected bootsector");
                    return Err(errno(libc::EPERM));
                }
            }
            // Take the only byte that may change.
            bootsector[RESERVED1_OFFSET] = buf[RESERVED1_OFFSET];
            return Ok(());
        }

        // No writes to the boot sector.
        if sector_num < u64::from(self.offset_to_fat) {
            return Err(errno(libc::EPERM));
        }

        // Negative for writes to the FAT, which is before the root directory.
        let first_cluster = self.sector2cluster(sector_num as i64);
        let last_cluster = self.sector2cluster((sector_num + nb_sectors as u64 - 1) as i64);

        let mut i = first_cluster;
        while i <= last_cluster {
            let mapping = if i >= 0 { self.find_mapping_for_cluster(i as u32) } else { None };
            let Some(m) = mapping else {
                i += 1;
                continue;
            };
            let mm = &self.mapping[m];
            if mm.read_only {
                eprintln!("Tried to write to write-protected file {}", mm.path);
                return Err(errno(libc::EPERM));
            }

            if mm.mode & MODE_DIRECTORY != 0 {
                let spc = i64::from(self.sectors_per_cluster);
                let sn = sector_num as i64;
                let begin = (self.cluster2sector(i as u32) as i64).max(sn);
                let end = (self.cluster2sector(i as u32) as i64 + spc).min(sn + nb_sectors as i64);
                // As in QEMU, from the mapping's directory entry and without the offset of
                // the root directory.
                let dir_index =
                    i64::from(mm.dir_index) + 0x10 * (begin - i64::from(mm.begin) * spc);
                let from = ((begin - sn) * 0x200) as usize;
                for k in 0..((end - begin) * 0x10) as usize {
                    let d: Direntry = buf[from + k * 32..from + k * 32 + 32].try_into().unwrap();
                    // No access to the entry of a read-only file.
                    if is_short_name(&d) && d[11] & 1 != 0 {
                        let old = usize::try_from(dir_index + k as i64)
                            .ok()
                            .and_then(|x| self.directory.get(x));
                        if old != Some(&d) {
                            report::warn_report("tried to write to write-protected file");
                            return Err(errno(libc::EPERM));
                        }
                    }
                }
            }
            i = mm.end as i32;
        }

        // Write to the write target and commit later.
        if let Err(e) = q.pwrite(sector_num * BDRV_SECTOR_SIZE, &buf[..nb_sectors * 0x200]) {
            eprintln!("Error writing to qcow backend");
            return Err(e);
        }

        for i in first_cluster.max(0)..=last_cluster {
            if let Some(u) = self.used(i as u32) {
                *u |= USED_ALLOCATED;
            }
        }

        // TODO in QEMU: add a timeout.
        self.try_commit(Some(q));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn short(name: &[u8; 11]) -> Direntry {
        let mut d = [0u8; 32];
        d[..11].copy_from_slice(name);
        d[11] = 0x20;
        d
    }

    #[test]
    fn short_names() {
        let mut lfn = LongFileName::new();
        assert_eq!(lfn.parse_short_name(true, &short(b"README  TXT")), 0);
        assert_eq!(lfn.name, b"readme.txt");
        assert_eq!(lfn.parse_short_name(false, &short(b"A~1        ")), 0);
        assert_eq!(lfn.name, b"A~1");
        assert_eq!(lfn.parse_short_name(true, &short(b"a       TXT")), -1);
        assert_eq!(lfn.parse_short_name(true, &short(b"A       T+T")), -2);
    }

    #[test]
    fn long_names() {
        // "hello world.txt" in two entries, the second one first on disk.
        let name: Vec<u16> = "hello world.txt".encode_utf16().chain([0]).collect();
        let mut entries = [[0u8; 32]; 2];
        for i in 0..26 * 2 {
            let e = &mut entries[1 - i / 26];
            let o = i % 26;
            let o = if o < 10 {
                1 + o
            } else if o < 22 {
                14 + o - 10
            } else {
                28 + o - 22
            };
            e[o] = match name.get(i / 2) {
                None => 0xff,
                Some(c) => (if i % 2 == 0 { *c } else { *c >> 8 }) as u8,
            };
        }
        entries[0][0] = 0x42;
        entries[1][0] = 0x01;
        for e in &mut entries {
            e[11] = 0xf;
            e[13] = 0x33;
        }
        let mut lfn = LongFileName::new();
        assert_eq!(lfn.parse_long_name(&entries[0]), 0);
        assert_eq!(lfn.parse_long_name(&entries[1]), 0);
        assert_eq!(lfn.name, b"hello world.txt");
        assert_eq!(lfn.checksum, 0x33);
        let mut lfn = LongFileName::new();
        assert_eq!(lfn.parse_long_name(&entries[0]), 0);
        entries[1][13] = 0x34;
        assert_eq!(lfn.parse_long_name(&entries[1]), -2);
    }

    #[test]
    fn basename() {
        assert_eq!(get_basename("/a/b"), "b");
        assert_eq!(get_basename("b"), "b");
    }
}
