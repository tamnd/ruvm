// SPDX-License-Identifier: GPL-2.0-or-later

//! The `vvfat` protocol driver from block/vvfat.c: a host directory shown to the guest as a
//! disk with a FAT12, FAT16 or FAT32 file system, `fat:[12:|16:|32:][floppy:][rw:]<dir>`.
//!
//! The disk is built the way QEMU builds it, so a guest sees the same bytes: the geometry
//! (a 1.44 or 2.88 MB floppy, a 32 MB FAT12 disk or a 504 MB disk), the MBR with its one
//! partition, the boot sector, both FATs, the volume label, and the directories with a long
//! file name entry for every name, the short names with their `~N` tails, the attributes, the
//! host time stamps in local time and the clusters handed out in the same order. File data is
//! read from the host files when the guest reads it.
//!
//! With `rw` the guest's writes go to a temporary qcow image (the `write-target` child), and
//! after every write the driver checks whether the file system on the disk is consistent. When
//! it is, the changes are written back to the host directory: renamed, new and removed files
//! and directories and changed file contents. This needs the `qcow` driver.
//!
//! Differences from QEMU:
//! - The directory is read with `std::fs::read_dir()`, which leaves out `.` and `..`. QEMU
//!   reads them from `readdir()` and takes the time stamps of a subdirectory's `.` and `..`
//!   entries from them; here the same `stat()` calls are made for them before the other
//!   entries, which is where Linux and macOS return them, so the only thing that can differ
//!   is when the "Too many entries in root directory" check fires on a file system that lists
//!   them elsewhere.
//! - Host names that are not UTF-8 are left out, after the "vvfat: invalid UTF-8 name"
//!   message QEMU prints for them. QEMU goes on to make a short name from the bytes.
//! - `localtime_r()` is done by reading the time zone files (see the `localtime` module),
//!   since this crate may not call the C library. On Windows the time stamps are in UTC.
//! - Where QEMU aborts the process or asserts while committing guest changes, the commit
//!   stops with the message printed, as for the errors QEMU does return; the write itself
//!   succeeds, since QEMU ignores what `try_commit()` returns. A chain whose clusters moved
//!   inside a file (the case QEMU marks with `abort()` before copying the clusters) is copied.
//! - When copying such clusters, QEMU reads and writes sector `offs` for every sector of the
//!   cluster instead of `offs + i`; here each sector is copied.
//! - Where QEMU would read or write past its arrays because the guest wrote nonsense into
//!   the FAT (cluster numbers past the end of the disk), the check fails instead.
//! - After a rename, the later mappings of a fragmented file get the new path too; QEMU leaves
//!   them pointing at the freed old path until the next commit rebuilds them.
//! - There is no migration blocker, as there is no migration in the block layer.
//! - The temporary file is `vl.XXXXXX` in `$TMPDIR` (`/var/tmp` for `/tmp`) as in QEMU, with
//!   a name from the process id, the time and a counter rather than `mkstemp()`.
//!
//! Kept from QEMU: a file whose directory entry the guest freed is not removed from the host
//! (`handle_deletes()` only unlinks files whose entry is still in use), the tail of the last
//! cluster of a file shows whatever the cluster buffer held before, and the FAT32 variant has
//! QEMU's FAT16 style boot sector, which QEMU warns about.

mod commit;
mod localtime;

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex};

use ruvm_base::{Error, Result, report};
use ruvm_qapi::types::{BlockdevOptions, BlockdevOptionsU, BlockdevRef};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_qapi::{QDict, QValue};

use crate::drivers::{DriverDef, OpenArgs};
use crate::node::{
    BDRV_BLOCK_DATA, BDRV_CHILD_DATA, BDRV_CHILD_METADATA, BDRV_SECTOR_SIZE, BlockLimits,
    BlockStatus, Driver, Node, errno,
};
use crate::perm::{BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, BLK_PERM_WRITE_UNCHANGED};

use localtime::Zone;

/// `bdrv_vvfat`.
pub(crate) static VVFAT: DriverDef = DriverDef::protocol("vvfat", "fat", vvfat_open)
    .with_parse_filename(vvfat_parse_filename)
    .with_strong_opts(&["dir", "fat-type", "floppy", "label", "rw"]);

/// `BOOTSECTOR_OEM_NAME`.
const BOOTSECTOR_OEM_NAME: &[u8; 8] = b"MSWIN4.1";

const DIR_DELETED: u8 = 0xe5;
const DIR_KANJI: u8 = DIR_DELETED;
const DIR_KANJI_FAKE: u8 = 0x05;
const DIR_FREE: u8 = 0x00;

const MODE_UNDEFINED: u32 = 0;
const MODE_NORMAL: u32 = 1;
const MODE_DIRECTORY: u32 = 4;
const MODE_DELETED: u32 = 8;

/// The name of the child the guest's writes go to.
const WRITE_TARGET: &str = "write-target";

/// A `direntry_t`, as the 32 bytes the guest sees.
type Direntry = [u8; 32];

fn le16(d: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([d[at], d[at + 1]])
}

fn put16(d: &mut [u8], at: usize, v: u16) {
    d[at..at + 2].copy_from_slice(&v.to_le_bytes());
}

fn put32(d: &mut [u8], at: usize, v: u32) {
    d[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn is_free(d: &Direntry) -> bool {
    d[0] == DIR_DELETED || d[0] == DIR_FREE
}

fn is_volume_label(d: &Direntry) -> bool {
    d[11] == 0x28
}

fn is_long_name(d: &Direntry) -> bool {
    d[11] == 0xf
}

fn is_short_name(d: &Direntry) -> bool {
    !is_volume_label(d) && !is_long_name(d) && !is_free(d)
}

fn is_directory(d: &Direntry) -> bool {
    d[11] & 0x10 != 0 && d[0] != DIR_DELETED
}

fn is_dot(d: &Direntry) -> bool {
    is_short_name(d) && d[0] == b'.'
}

fn is_file(d: &Direntry) -> bool {
    is_short_name(d) && !is_directory(d)
}

fn begin_of_direntry(d: &Direntry) -> u32 {
    u32::from(le16(d, 26)) | (u32::from(le16(d, 20)) << 16)
}

fn filesize_of_direntry(d: &Direntry) -> u32 {
    u32::from_le_bytes([d[28], d[29], d[30], d[31]])
}

fn set_begin_of_direntry(d: &mut Direntry, begin: u32) {
    put16(d, 26, begin as u16);
    put16(d, 20, (begin >> 16) as u16);
}

/// `valid_filename()`.
fn valid_filename(name: &[u8]) -> bool {
    if name == b"." || name == b".." {
        return false;
    }
    name.iter()
        .all(|&c| c.is_ascii_alphanumeric() || c > 127 || b" $%'-_@~`!(){}^#&.+,;=[]".contains(&c))
}

/// `to_valid_short_char()`, with `strchr()` looking at the low byte of the character as it
/// does in QEMU.
fn to_valid_short_char(c: char) -> u8 {
    let mut up = c.to_uppercase();
    let c = match (up.next(), up.next()) {
        (Some(u), None) => u,
        _ => c,
    };
    let low = c as u32 as u8;
    if c.is_ascii_digit()
        || c.is_ascii_uppercase()
        || (low != 0 && b"$%'-_@~`!(){}^#&".contains(&low))
    {
        low
    } else {
        0
    }
}

/// `fat_chksum()`.
fn fat_chksum(d: &Direntry) -> u8 {
    d[..11].iter().fold(0u8, |sum, &c| (sum >> 1 | (sum & 1) << 7).wrapping_add(c))
}

/// `fat_datetime()`: the FAT date, or the time when `return_time`.
fn fat_datetime(zone: &Zone, time: i64, return_time: bool) -> u16 {
    let t = zone.localtime(time);
    if return_time {
        ((t.sec / 2) | (t.min << 5) | (t.hour << 11)) as u16
    } else {
        (t.mday | ((t.mon + 1) << 5) | ((t.year - 80) << 9)) as u16
    }
}

/// `mapping_t`: which clusters belong to which host file or directory.
#[derive(Debug, Clone, Default)]
struct Mapping {
    /// The first cluster; `end` is the last plus one.
    begin: u32,
    end: u32,
    /// The index of the short name entry in the directory.
    dir_index: u32,
    /// For the later parts of a fragmented file, the index of the first part.
    first_mapping_index: i32,
    /// `info.file.offset` (in clusters) or `info.dir.parent_mapping_index`, which share
    /// their storage in QEMU.
    info0: u32,
    /// `info.dir.first_dir_index`.
    first_dir_index: i32,
    /// The full host path.
    path: String,
    mode: u32,
    read_only: bool,
}

impl Mapping {
    fn offset(&self) -> u32 {
        self.info0
    }

    fn parent_mapping_index(&self) -> i32 {
        self.info0 as i32
    }

    fn set_parent_mapping_index(&mut self, i: i32) {
        self.info0 = i as u32;
    }
}

/// Where `s->cluster` points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClusterSrc {
    /// Into the directory, from this entry on.
    Dir(usize),
    /// The cluster buffer.
    Buffer,
}

/// `commit_t`.
#[derive(Debug, Clone)]
enum Commit {
    Rename { cluster: u32, path: String },
    Writeout { dir_index: usize, modified_offset: u32 },
    NewFile { first_cluster: u32, path: String },
    Mkdir { cluster: u32, path: String },
}

/// `BDRVVVFATState`.
#[derive(Debug)]
struct State {
    first_sectors: Vec<u8>,
    fat_type: u32,
    fat: Vec<u8>,
    directory: Vec<Direntry>,
    mapping: Vec<Mapping>,
    volume_label: [u8; 11],
    offset_to_bootsector: u32,
    cluster_size: u32,
    sectors_per_cluster: u32,
    sectors_per_fat: u32,
    last_cluster_of_root_directory: u32,
    root_entries: u32,
    sector_count: u32,
    cluster_count: u32,
    max_fat_value: u32,
    offset_to_fat: u32,
    offset_to_root_dir: u32,
    /// `bs->total_sectors`.
    total_sectors: u64,

    current_file: Option<File>,
    current_mapping: Option<usize>,
    cluster: ClusterSrc,
    cluster_buffer: Vec<u8>,
    current_cluster: u32,

    /// Whether there is a write target.
    qcow: bool,
    fat2: Option<Vec<u8>>,
    used_clusters: Vec<u8>,
    commits: Vec<Commit>,
    /// The host directory, without a trailing `/`.
    path: String,
    downcase_short_names: bool,
}

/// `vvfat_parse_filename()`.
fn vvfat_parse_filename(filename: &str, options: &mut QDict) -> Result<()> {
    if !filename.starts_with("fat:") {
        return Err(Error::generic("File name string must start with 'fat:'"));
    }
    let fat_type: i64 = if filename.contains(":32:") {
        32
    } else if filename.contains(":16:") {
        16
    } else if filename.contains(":12:") {
        12
    } else {
        0
    };
    let floppy = filename.contains(":floppy:");
    let rw = filename.contains(":rw:");

    // The directory name without the options.
    let b = filename.as_bytes();
    let i = filename.rfind(':').expect("the prefix has a colon");
    let dir = if b[i - 2] == b':' && b[i - 1].is_ascii_alphabetic() {
        // A DOS drive name.
        &filename[i - 1..]
    } else {
        &filename[i + 1..]
    };

    options.put("dir", dir);
    options.put("fat-type", fat_type);
    options.put("floppy", floppy);
    options.put("rw", rw);
    Ok(())
}

/// `vvfat_open()`.
fn vvfat_open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Vvfat(o) = opts else { unreachable!("vvfat driver with other options") };
    // qemu_opt_get_number() goes into an int.
    let mut fat_type = o.fat_type.unwrap_or(0) as i32;
    let floppy = o.floppy.unwrap_or(false);

    let mut volume_label = [b' '; 11];
    match &o.label {
        Some(label) => {
            if label.len() > 11 {
                return Err(Error::generic("vvfat label cannot be longer than 11 bytes"));
            }
            volume_label[..label.len()].copy_from_slice(label.as_bytes());
        }
        None => volume_label[..10].copy_from_slice(b"QEMU VVFAT"),
    }

    let (cyls, heads, secs, offset_to_bootsector): (u32, u32, u32, u32);
    if floppy {
        // 1.44MB or 2.88MB floppy. 2.88MB can be FAT12 (default) or FAT16.
        if fat_type == 0 {
            fat_type = 12;
            secs = 36;
        } else {
            secs = if fat_type == 12 { 18 } else { 36 };
        }
        cyls = 80;
        heads = 2;
        offset_to_bootsector = 0;
    } else {
        // 32MB or 504MB disk.
        if fat_type == 0 {
            fat_type = 16;
        }
        offset_to_bootsector = 0x3f;
        cyls = if fat_type == 12 { 64 } else { 1024 };
        heads = 16;
        secs = 63;
    }

    match fat_type {
        32 => report::warn_report("FAT32 has not been tested. You are welcome to do so!"),
        16 | 12 => {}
        _ => return Err(Error::generic("Valid FAT types are only 12, 16 and 32")),
    }

    let total = cyls * heads * secs;
    let mut s = State {
        first_sectors: vec![0; 0x40 * 0x200],
        fat_type: fat_type as u32,
        fat: Vec::new(),
        directory: Vec::new(),
        mapping: Vec::new(),
        volume_label,
        offset_to_bootsector,
        cluster_size: 0,
        // LATER TODO in QEMU: if FAT32, adjust.
        sectors_per_cluster: 0x10,
        sectors_per_fat: 0,
        last_cluster_of_root_directory: 0,
        root_entries: 0,
        sector_count: total - offset_to_bootsector,
        cluster_count: 0,
        max_fat_value: 0,
        offset_to_fat: 0,
        offset_to_root_dir: 0,
        total_sectors: u64::from(total),
        current_file: None,
        current_mapping: None,
        cluster: ClusterSrc::Buffer,
        cluster_buffer: Vec::new(),
        current_cluster: u32::MAX,
        qcow: false,
        fat2: None,
        used_clusters: Vec::new(),
        commits: Vec::new(),
        path: String::new(),
        downcase_short_names: true,
    };

    if o.rw.unwrap_or(false) {
        if !args.flags.read_only {
            s.enable_write_target(args)?;
        } else {
            return Err(Error::generic("Unable to set VVFAT to 'rw' when drive is read-only"));
        }
    } else {
        args.apply_auto_read_only(None)?;
    }

    s.init_directories(&Zone::local(), &o.dir, heads, secs)?;

    s.sector_count = s.offset_to_root_dir + s.sectors_per_cluster * s.cluster_count;

    if s.offset_to_bootsector > 0 {
        s.init_mbr(cyls, heads, secs);
    }

    Ok(Box::new(Vvfat { s: Mutex::new(s) }))
}

/// `sector2CHS()`: fills `chs` and returns true when the position is past the geometry.
fn sector2chs(chs: &mut [u8], spos: u32, cyls: u32, heads: u32, secs: u32) -> bool {
    let sector = spos % secs;
    let spos = spos / secs;
    let head = spos % heads;
    let spos = spos / heads;
    if spos >= cyls {
        // Windows and DOS take 1023/255/63 as not representable.
        chs.copy_from_slice(&[0xff, 0xff, 0xff]);
        return true;
    }
    chs[0] = head as u8;
    chs[1] = ((sector + 1) | ((spos >> 8) << 6)) as u8;
    chs[2] = spos as u8;
    false
}

/// A unique name for the temporary write target, `create_tmp_file()`.
fn create_tmp_file() -> Result<String> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let mut tmpdir = std::env::temp_dir().to_string_lossy().into_owned();
    while tmpdir.len() > 1 && tmpdir.ends_with('/') {
        tmpdir.pop();
    }
    if cfg!(not(windows)) && tmpdir == "/tmp" {
        // Temporary images can get large; /tmp is often a tmpfs.
        tmpdir = "/var/tmp".to_string();
    }
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut last = None;
    for _ in 0..100 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        let mut v = nanos
            ^ (u64::from(std::process::id()) << 32)
            ^ u64::from(COUNTER.fetch_add(1, Ordering::Relaxed))
                .wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut suffix = String::new();
        for _ in 0..6 {
            suffix.push(CHARS[(v % CHARS.len() as u64) as usize] as char);
            v /= CHARS.len() as u64;
        }
        let name = format!("{tmpdir}/vl.{suffix}");
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&name) {
            Ok(_) => return Ok(name),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => last = Some((name, e)),
            Err(e) => {
                return Err(Error::from_io(format!("Could not open temporary file '{name}'"), e));
            }
        }
    }
    let (name, e) = last.expect("the loop ran");
    Err(Error::from_io(format!("Could not open temporary file '{name}'"), e))
}

impl State {
    /// `enable_write_target()`: creates the temporary qcow image the writes go to and opens
    /// it as the `write-target` child.
    fn enable_write_target(&mut self, args: &mut OpenArgs<'_>) -> Result<()> {
        // offset_to_root_dir is still 0 here, as in QEMU.
        let size = self.sector2cluster(i64::from(self.sector_count));
        self.used_clusters = vec![0; size.max(0) as usize];

        let qcow_filename = create_tmp_file()?;
        let r = Self::open_write_target(args, &qcow_filename, self.total_sectors);
        #[cfg(not(windows))]
        let _ = std::fs::remove_file(&qcow_filename);
        r?;
        self.qcow = true;
        Ok(())
    }

    fn open_write_target(
        args: &mut OpenArgs<'_>,
        filename: &str,
        total_sectors: u64,
    ) -> Result<()> {
        let Some(create_opts) = crate::qcow::QCOW.create_opts else {
            return Err(Error::generic("Failed to locate qcow driver"));
        };
        let mut opts = QDict::new();
        opts.put("size", total_sectors * BDRV_SECTOR_SIZE);
        opts.put("backing_file", "fat:");
        create_opts(filename, &mut opts)?;

        // vvfat_qcow_options(): the child is writable and never flushed. The image names
        // "fat:" as its backing file, which is not opened: reads the image does not have go
        // to this driver's own data.
        let json = format!(
            r#"{{"driver": "qcow", "read-only": false, "auto-read-only": false,
                "cache": {{"no-flush": true}}, "backing": null,
                "file": {{"driver": "file", "filename": {}}}}}"#,
            QValue::Str(filename.to_string()).to_json()
        );
        let value = ruvm_qapi::json::from_str(&json)?;
        let mut v = QObjectInputVisitor::new(value);
        let mut options = BlockdevOptions::default();
        BlockdevOptions::visit(&mut v, None, &mut options)?;
        args.open_child(
            BlockdevRef::Definition(Box::new(options)),
            WRITE_TARGET,
            BDRV_CHILD_DATA | BDRV_CHILD_METADATA,
        )?;
        Ok(())
    }

    fn sector2cluster(&self, sector_num: i64) -> i32 {
        ((sector_num - i64::from(self.offset_to_root_dir)) / i64::from(self.sectors_per_cluster))
            as i32
    }

    fn cluster2sector(&self, cluster_num: u32) -> u64 {
        u64::from(self.offset_to_root_dir)
            + u64::from(self.sectors_per_cluster) * u64::from(cluster_num)
    }

    /// `init_mbr()`.
    fn init_mbr(&mut self, cyls: u32, heads: u32, secs: u32) {
        let total = self.total_sectors as u32;
        let obs = self.offset_to_bootsector;
        let mbr = &mut self.first_sectors[..512];
        mbr.fill(0);
        // The Windows NT disk signature.
        put32(mbr, 0x1b8, 0xbe1a_fdfa);
        let p = 0x1be;
        mbr[p] = 0x80; // bootable
        // LBA is used when the partition is outside the CHS geometry.
        let mut lba = sector2chs(&mut mbr[p + 1..p + 4], obs, cyls, heads, secs);
        lba |= sector2chs(&mut mbr[p + 5..p + 8], total - 1, cyls, heads, secs);
        put32(mbr, p + 8, obs);
        put32(mbr, p + 12, total - obs);
        // DOS uses different types when the partition is LBA.
        mbr[p + 4] = match self.fat_type {
            12 => 0x1,
            16 => {
                if lba {
                    0xe
                } else {
                    0x06
                }
            }
            _ => {
                if lba {
                    0xc
                } else {
                    0x0b
                }
            }
        };
        mbr[510] = 0x55;
        mbr[511] = 0xaa;
    }

    /// `init_fat()`.
    fn init_fat(&mut self) {
        let spf = self.sectors_per_fat as usize;
        // The size array_ensure_allocated() gives the array in QEMU.
        self.fat = if self.fat_type == 12 {
            vec![0; spf * 0x200 * 3 / 2 - 1 + 32]
        } else {
            let item = if self.fat_type == 32 { 4 } else { 2 };
            vec![0; (spf * 0x200 / item - 1 + 32) * item]
        };
        self.max_fat_value = match self.fat_type {
            12 => 0xfff,
            16 => 0xffff,
            32 => 0x0fff_ffff,
            _ => 0,
        };
    }

    /// `fat_set()`.
    fn fat_set(&mut self, cluster: u32, value: u32) {
        let c = cluster as usize;
        let fat = &mut self.fat;
        match self.fat_type {
            32 => {
                if let Some(e) = fat.get_mut(c * 4..c * 4 + 4) {
                    e.copy_from_slice(&value.to_le_bytes());
                }
            }
            16 => {
                if let Some(e) = fat.get_mut(c * 2..c * 2 + 2) {
                    e.copy_from_slice(&(value as u16).to_le_bytes());
                }
            }
            _ => {
                let o = c * 3 / 2;
                if o + 1 >= fat.len() {
                    return;
                }
                if c & 1 == 0 {
                    fat[o] = value as u8;
                    fat[o + 1] = (fat[o + 1] & 0xf0) | ((value >> 8) & 0xf) as u8;
                } else {
                    fat[o] = (fat[o] & 0xf) | ((value & 0xf) << 4) as u8;
                    fat[o + 1] = (value >> 4) as u8;
                }
            }
        }
    }

    /// `fat_get()`. Entries past the array read as free.
    fn fat_get(&self, cluster: u32) -> u32 {
        fat_entry(&self.fat, self.fat_type, cluster)
    }

    /// `fat_eof()`.
    fn fat_eof(&self, fat_entry: u32) -> bool {
        fat_entry > self.max_fat_value.wrapping_sub(8)
    }

    /// `create_long_filename()`: the long name entries for `filename`, returning the index of
    /// the first.
    fn create_long_filename(&mut self, filename: &str) -> usize {
        let mut longname: Vec<u16> = filename.encode_utf16().collect();
        let length = longname.len();
        longname.push(0);
        let number_of_entries = (length * 2).div_ceil(26);
        let first = self.directory.len();
        for i in 0..number_of_entries {
            let mut e = [0u8; 32];
            e[11] = 0xf;
            e[0] = (number_of_entries - i) as u8 | if i == 0 { 0x40 } else { 0 };
            self.directory.push(e);
        }
        let next = self.directory.len();
        for i in 0..26 * number_of_entries {
            let entry = &mut self.directory[next - 1 - i / 26];
            let offset = i % 26;
            let offset = if offset < 10 {
                1 + offset
            } else if offset < 22 {
                14 + offset - 10
            } else {
                28 + offset - 22
            };
            entry[offset] = if i >= 2 * length + 2 {
                0xff
            } else if i % 2 == 0 {
                longname[i / 2] as u8
            } else {
                (longname[i / 2] >> 8) as u8
            };
        }
        first
    }

    /// `create_short_filename()`: the short name entry for `filename`, unique among the
    /// entries from `directory_start` on. Returns its index.
    fn create_short_filename(&mut self, filename: &str, directory_start: usize) -> Option<usize> {
        let mut name = [b' '; 11];
        let mut j = 0;
        let mut last_dot: Option<usize> = None;
        let mut lossy_conversion = false;

        // Copy the file name and find the last dot.
        for (p, c) in filename.char_indices() {
            if c == '.' {
                if j == 0 {
                    // A '.' at the start of the name.
                    lossy_conversion = true;
                } else {
                    if last_dot.is_some() {
                        lossy_conversion = true;
                    }
                    last_dot = Some(p);
                }
            } else if last_dot.is_none() {
                // The first part of the name.
                let v = to_valid_short_char(c);
                if j < 8 && v != 0 {
                    name[j] = v;
                    j += 1;
                } else {
                    lossy_conversion = true;
                }
            }
        }

        // Copy the extension, if any.
        if let Some(dot) = last_dot {
            j = 0;
            for c in filename[dot + 1..].chars() {
                let v = to_valid_short_char(c);
                if j < 3 && v != 0 {
                    name[8 + j] = v;
                    j += 1;
                } else {
                    lossy_conversion = true;
                }
            }
        }

        if name[0] == DIR_KANJI {
            name[0] = DIR_KANJI_FAKE;
        }

        // Numeric tail generation.
        let j = name[..8].iter().position(|&c| c == b' ').unwrap_or(8);
        let entry = self.directory.len();
        for i in u32::from(lossy_conversion)..999_999 {
            if i > 0 {
                let tail = format!("~{i}");
                let at = j.min(8 - tail.len());
                name[at..at + tail.len()].copy_from_slice(tail.as_bytes());
            }
            let dupe = self.directory[directory_start..entry]
                .iter()
                .any(|e| !is_long_name(e) && e[..11] == name);
            if !dupe {
                let mut e = [0u8; 32];
                e[..11].copy_from_slice(&name);
                self.directory.push(e);
                return Some(entry);
            }
        }
        None
    }

    /// `create_short_and_long_name()`: returns the index of the short name entry.
    fn create_short_and_long_name(
        &mut self,
        directory_start: usize,
        filename: &str,
        is_dot: bool,
    ) -> Option<usize> {
        if is_dot {
            let mut e = [0u8; 32];
            e[..11].fill(b' ');
            e[..filename.len()].copy_from_slice(filename.as_bytes());
            self.directory.push(e);
            return Some(self.directory.len() - 1);
        }
        let long_index = self.create_long_filename(filename);
        let entry = self.create_short_filename(filename, directory_start)?;
        // Propagate the checksum to the long name.
        let chksum = fat_chksum(&self.directory[entry]);
        for e in &mut self.directory[long_index..entry] {
            if !is_long_name(e) {
                break;
            }
            e[13] = chksum;
        }
        Some(entry)
    }

    /// `read_directory()`: the entries of the directory of mapping `mapping_index`, with
    /// mappings for its subdirectories and non-empty files.
    fn read_directory(&mut self, zone: &Zone, mapping_index: usize) -> i32 {
        let dirname = self.mapping[mapping_index].path.clone();
        let first_cluster = self.mapping[mapping_index].begin;
        let parent_index = self.mapping[mapping_index].parent_mapping_index();
        let first_cluster_of_parent =
            if parent_index >= 0 { self.mapping[parent_index as usize].begin } else { u32::MAX };

        let dir = match std::fs::read_dir(&dirname) {
            Ok(d) => d,
            Err(_) => {
                let m = &mut self.mapping[mapping_index];
                m.end = m.begin;
                return -1;
            }
        };

        let i = if first_cluster == 0 { 0 } else { self.directory.len() };
        self.mapping[mapping_index].first_dir_index = i as i32;

        if first_cluster != 0 {
            // The top entries of a subdirectory.
            self.create_short_and_long_name(i, ".", true);
            self.create_short_and_long_name(i, "..", true);
        }

        let names = [".".to_string(), "..".to_string()].into_iter().map(Some).chain(
            dir.filter_map(|e| e.ok()).map(|e| {
                let name = e.file_name();
                match name.to_str() {
                    Some(n) => Some(n.to_string()),
                    None => {
                        eprintln!("vvfat: invalid UTF-8 name: {}", name.to_string_lossy());
                        None
                    }
                }
            }),
        );

        for name in names {
            if first_cluster == 0 && self.directory.len() >= self.root_entries as usize - 1 {
                eprintln!("Too many entries in root directory");
                return -2;
            }
            let Some(name) = name else { continue };
            let is_dot = name == ".";
            let is_dotdot = name == "..";
            if first_cluster == 0 && (is_dotdot || is_dot) {
                continue;
            }

            let buffer = format!("{dirname}/{name}");
            let Ok(st) = std::fs::metadata(&buffer) else { continue };
            let is_dir = st.is_dir();
            let st_size = st.len();

            // The directory entry for this file.
            let d = if !is_dot && !is_dotdot {
                match self.create_short_and_long_name(i, &name, false) {
                    Some(d) => d,
                    None => return -1,
                }
            } else if is_dot {
                i
            } else {
                i + 1
            };
            let (ctime, atime, mtime) = host_times(&st);
            let e = &mut self.directory[d];
            e[11] = if is_dir { 0x10 } else { 0x20 };
            e[12] = 0;
            e[13] = 0;
            put16(e, 14, fat_datetime(zone, ctime, true));
            put16(e, 16, fat_datetime(zone, ctime, false));
            put16(e, 18, fat_datetime(zone, atime, false));
            put16(e, 20, 0);
            put16(e, 22, fat_datetime(zone, mtime, true));
            put16(e, 24, fat_datetime(zone, mtime, false));
            if is_dotdot {
                set_begin_of_direntry(e, first_cluster_of_parent);
            } else if is_dot {
                set_begin_of_direntry(e, first_cluster);
            } else {
                // Done later.
                put16(e, 26, 0);
            }
            if st_size > 0x7fff_ffff {
                eprintln!("File {buffer} is larger than 2GB");
                return -2;
            }
            put32(e, 28, if is_dir { 0 } else { st_size as u32 });

            // The mapping for this file.
            if !is_dot && !is_dotdot && (is_dir || st_size > 0) {
                let mut m = Mapping {
                    begin: 0,
                    end: st_size as u32,
                    // The most recent entry is the short name with everything that matters.
                    dir_index: self.directory.len() as u32 - 1,
                    first_mapping_index: -1,
                    path: buffer,
                    read_only: host_read_only(&st),
                    ..Mapping::default()
                };
                if is_dir {
                    m.mode = MODE_DIRECTORY;
                    m.set_parent_mapping_index(mapping_index as i32);
                } else {
                    m.mode = MODE_UNDEFINED;
                }
                self.mapping.push(m);
            }
        }

        // Fill with zeroes up to the end of the cluster.
        let per_cluster = 0x10 * self.sectors_per_cluster as usize;
        while self.directory.len() % per_cluster != 0 {
            self.directory.push([0; 32]);
        }

        if self.fat_type != 32
            && mapping_index == 0
            && self.directory.len() < self.root_entries as usize
        {
            // The root directory.
            self.directory.resize(self.root_entries as usize, [0; 32]);
        }

        let m = &mut self.mapping[mapping_index];
        let first_dir_index = m.first_dir_index as usize;
        m.end = first_cluster
            + ((self.directory.len() - first_dir_index) * 0x20 / self.cluster_size as usize) as u32;
        let (dir_index, begin) = (m.dir_index as usize, m.begin);
        set_begin_of_direntry(&mut self.directory[dir_index], begin);
        0
    }

    /// `init_directories()`.
    fn init_directories(
        &mut self,
        zone: &Zone,
        dirname: &str,
        heads: u32,
        secs: u32,
    ) -> Result<()> {
        self.first_sectors.fill(0);
        self.cluster_size = self.sectors_per_cluster * 0x200;
        self.cluster_buffer = vec![0; self.cluster_size as usize];

        // sc = spf+1+spf*spc*(512*8/fat_type): sc is the sector count, spf the sectors
        // per FAT and spc the sectors per cluster.
        let i = 1 + self.sectors_per_cluster * 0x200 * 8 / self.fat_type;
        self.sectors_per_fat = (self.sector_count + i) / i;

        self.offset_to_fat = self.offset_to_bootsector + 1;
        self.offset_to_root_dir = self.offset_to_fat + self.sectors_per_fat * 2;

        self.mapping.clear();
        self.directory.clear();

        // The volume label.
        let mut label = [0u8; 32];
        label[11] = 0x28; // archive | volume label
        label[..11].copy_from_slice(&self.volume_label);
        self.directory.push(label);

        // Now build the FAT, and write back information into the directory.
        self.init_fat();

        self.root_entries = 0x02 * 0x10 * self.sectors_per_cluster;
        self.cluster_count = self.sector2cluster(i64::from(self.sector_count)) as u32;

        let mut path = dirname.to_string();
        if path.ends_with('/') {
            path.pop();
        }
        self.path = path.clone();
        self.mapping.push(Mapping {
            begin: 0,
            dir_index: 0,
            first_mapping_index: -1,
            info0: u32::MAX,
            path,
            mode: MODE_DIRECTORY,
            read_only: false,
            ..Mapping::default()
        });

        let mut cluster = 0u32;
        let mut i = 0;
        while i < self.mapping.len() {
            // MS-DOS expects the FAT to be 0 for the root directory (except for the media
            // byte).
            let mut fix_fat = i != 0;
            if self.mapping[i].mode & MODE_DIRECTORY != 0 {
                self.mapping[i].begin = cluster;
                if self.read_directory(zone, i) != 0 {
                    return Err(Error::generic(format!(
                        "Could not read directory {}",
                        self.mapping[i].path
                    )));
                }
            } else {
                let cluster_size = self.cluster_size;
                let m = &mut self.mapping[i];
                m.mode = MODE_NORMAL;
                m.begin = cluster;
                if m.end > 0 {
                    m.end = cluster + 1 + (m.end - 1) / cluster_size;
                    let (d, begin) = (m.dir_index as usize, m.begin);
                    set_begin_of_direntry(&mut self.directory[d], begin);
                } else {
                    m.end = cluster + 1;
                    fix_fat = false;
                }
            }

            let (begin, end) = (self.mapping[i].begin, self.mapping[i].end);
            // The next free cluster.
            cluster = end;

            if cluster > self.cluster_count {
                return Err(Error::generic(format!(
                    "Directory does not fit in FAT{} (capacity {:.2} MB)",
                    self.fat_type,
                    f64::from(self.sector_count) / 2000.0
                )));
            }

            if fix_fat {
                for j in begin..end.saturating_sub(1) {
                    self.fat_set(j, j + 1);
                }
                self.fat_set(end - 1, self.max_fat_value);
            }
            i += 1;
        }

        self.last_cluster_of_root_directory = self.mapping[0].end;

        // The FAT signature.
        self.fat_set(0, self.max_fat_value);
        self.fat_set(1, self.max_fat_value);

        self.current_mapping = None;

        let obs = self.offset_to_bootsector;
        let media_type = if obs > 0 { 0xf8 } else { 0xf0 };
        self.fat[0] = media_type;
        let sector_count = self.sector_count;
        let bs = &mut self.first_sectors[obs as usize * 0x200..][..0x200];
        bs[..3].copy_from_slice(&[0xeb, 0x3e, 0x90]);
        bs[3..11].copy_from_slice(BOOTSECTOR_OEM_NAME);
        put16(bs, 11, 0x200);
        bs[13] = self.sectors_per_cluster as u8;
        put16(bs, 14, 1); // reserved sectors
        bs[16] = 2; // number of FATs
        put16(bs, 17, self.root_entries as u16);
        put16(bs, 19, if sector_count > 0xffff { 0 } else { sector_count as u16 });
        // The media descriptor: hard disk 0xf8, floppy 0xf0.
        bs[21] = media_type;
        put16(bs, 22, self.sectors_per_fat as u16);
        put16(bs, 24, secs as u16);
        put16(bs, 26, heads as u16);
        put32(bs, 28, obs);
        put32(bs, 32, if sector_count > 0xffff { sector_count } else { 0 });
        // LATER TODO in QEMU: this is wrong for FAT32. The drive number: fda=0, hda=0x80.
        bs[36] = if obs == 0 { 0 } else { 0x80 };
        bs[38] = 0x29;
        put32(bs, 39, 0xfabe_1afd);
        bs[43..54].copy_from_slice(&self.volume_label);
        bs[54..62].copy_from_slice(if self.fat_type == 12 { b"FAT12   " } else { b"FAT16   " });
        bs[510] = 0x55;
        bs[511] = 0xaa;
        Ok(())
    }

    /// `vvfat_close_current_file()`.
    fn close_current_file(&mut self) {
        if self.current_mapping.is_some() {
            self.current_mapping = None;
            self.current_file = None;
        }
        self.current_cluster = u32::MAX;
    }

    /// `find_mapping_for_cluster_aux()`: with the mappings from `index1` to `index2 - 1`
    /// ordered, the index of the last mapping whose end is past `cluster_num`.
    fn find_mapping_for_cluster_aux(
        &self,
        cluster_num: i64,
        mut index1: usize,
        mut index2: usize,
    ) -> usize {
        loop {
            let index3 = (index1 + index2) / 2;
            let Some(m) = self.mapping.get(index3) else { return index1 };
            if i64::from(m.begin) >= cluster_num {
                if index2 == index3 {
                    return index1;
                }
                index2 = index3;
            } else {
                if index1 == index3 {
                    return if i64::from(m.end) <= cluster_num { index2 } else { index1 };
                }
                index1 = index3;
            }
        }
    }

    /// `find_mapping_for_cluster()`.
    fn find_mapping_for_cluster(&self, cluster_num: u32) -> Option<usize> {
        let index =
            self.find_mapping_for_cluster_aux(i64::from(cluster_num), 0, self.mapping.len());
        let m = self.mapping.get(index)?;
        if m.begin > cluster_num || m.end <= cluster_num {
            return None;
        }
        Some(index)
    }

    /// `open_file()`.
    fn open_file(&mut self, mapping: Option<usize>) -> i32 {
        let Some(m) = mapping else { return -1 };
        let same =
            self.current_mapping.is_some_and(|c| self.mapping[c].path == self.mapping[m].path);
        if !same || self.current_file.is_none() {
            let Ok(f) = File::open(&self.mapping[m].path) else { return -1 };
            self.close_current_file();
            self.current_file = Some(f);
        }
        self.current_mapping = Some(m);
        0
    }

    /// `read_cluster()`: makes `self.cluster` point at cluster `cluster_num`.
    fn read_cluster(&mut self, cluster_num: u32) -> i32 {
        if self.current_cluster == cluster_num {
            return 0;
        }
        let in_current = self.current_mapping.is_some_and(|c| {
            let m = &self.mapping[c];
            m.begin <= cluster_num && m.end > cluster_num
        });
        let is_dir;
        if !in_current {
            // A binary search of the mappings.
            let mapping = self.find_mapping_for_cluster(cluster_num);
            if let Some(m) = mapping.filter(|&m| self.mapping[m].mode & MODE_DIRECTORY != 0) {
                self.close_current_file();
                self.current_mapping = Some(m);
                is_dir = true;
            } else {
                if self.open_file(mapping) != 0 {
                    return -2;
                }
                is_dir = false;
            }
        } else {
            is_dir = self.mapping[self.current_mapping.unwrap()].mode & MODE_DIRECTORY != 0;
        }
        let m = &self.mapping[self.current_mapping.unwrap()];
        if is_dir {
            let offset = self.cluster_size as usize * (cluster_num - m.begin) as usize;
            self.cluster = ClusterSrc::Dir(offset / 0x20 + m.first_dir_index as usize);
            self.current_cluster = cluster_num;
            return 0;
        }

        let offset = u64::from(self.cluster_size)
            * u64::from((cluster_num - m.begin).wrapping_add(m.offset()));
        let Some(f) = self.current_file.as_mut() else { return -3 };
        if f.seek(SeekFrom::Start(offset)).ok() != Some(offset) {
            return -3;
        }
        self.cluster = ClusterSrc::Buffer;
        // One read() call in QEMU; a short read leaves the rest of the buffer as it was.
        let mut got = 0;
        while got < self.cluster_buffer.len() {
            match f.read(&mut self.cluster_buffer[got..]) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => {
                    self.current_cluster = u32::MAX;
                    return -1;
                }
            }
        }
        self.current_cluster = cluster_num;
        0
    }

    /// Copies sector `sector` of the current cluster to `out`.
    fn copy_cluster_sector(&self, sector: usize, out: &mut [u8]) {
        match self.cluster {
            ClusterSrc::Buffer => {
                out.copy_from_slice(&self.cluster_buffer[sector * 0x200..][..0x200])
            }
            ClusterSrc::Dir(first) => {
                for (k, chunk) in out.chunks_mut(0x20).enumerate() {
                    match self.directory.get(first + sector * 0x10 + k) {
                        Some(e) => chunk.copy_from_slice(e),
                        None => chunk.fill(0),
                    }
                }
            }
        }
    }

    /// `vvfat_read()`: `nb_sectors` sectors from `sector_num` on, from the write target
    /// `q` where it has them.
    fn read(
        &mut self,
        q: Option<&Node>,
        mut sector_num: u64,
        buf: &mut [u8],
        nb_sectors: usize,
    ) -> io::Result<()> {
        let mut i = 0;
        while i < nb_sectors {
            if sector_num >= self.total_sectors {
                return Err(errno(libc::EPERM));
            }
            if let Some(q) = q.filter(|_| self.qcow) {
                let (allocated, n) = q.is_allocated(
                    sector_num * BDRV_SECTOR_SIZE,
                    (nb_sectors - i) as u64 * BDRV_SECTOR_SIZE,
                )?;
                if allocated {
                    let k = ((n / BDRV_SECTOR_SIZE) as usize).max(1);
                    q.pread(sector_num * BDRV_SECTOR_SIZE, &mut buf[i * 0x200..(i + k) * 0x200])
                        .map_err(|_| errno(libc::EPERM))?;
                    i += k;
                    sector_num += k as u64;
                    continue;
                }
            }
            let out = &mut buf[i * 0x200..(i + 1) * 0x200];
            let sn = sector_num as usize;
            if sector_num < u64::from(self.offset_to_root_dir) {
                let fat_start = self.offset_to_fat as usize;
                let spf = self.sectors_per_fat as usize;
                if sn < fat_start {
                    out.copy_from_slice(&self.first_sectors[sn * 0x200..][..0x200]);
                } else if sn < fat_start + spf {
                    out.copy_from_slice(&self.fat[(sn - fat_start) * 0x200..][..0x200]);
                } else {
                    out.copy_from_slice(&self.fat[(sn - fat_start - spf) * 0x200..][..0x200]);
                }
            } else {
                let sector = (sector_num - u64::from(self.offset_to_root_dir)) as u32;
                let sector_offset_in_cluster = sector % self.sectors_per_cluster;
                let cluster_num = sector / self.sectors_per_cluster;
                if cluster_num > self.cluster_count || self.read_cluster(cluster_num) != 0 {
                    // LATER TODO in QEMU: strict: return -1.
                    out.fill(0);
                } else {
                    self.copy_cluster_sector(sector_offset_in_cluster as usize, out);
                }
            }
            i += 1;
            sector_num += 1;
        }
        Ok(())
    }
}

/// An entry of a FAT held in `fat`; entries past its end read as free.
fn fat_entry(fat: &[u8], fat_type: u32, cluster: u32) -> u32 {
    let c = cluster as usize;
    match fat_type {
        32 => fat.get(c * 4..c * 4 + 4).map_or(0, |e| u32::from_le_bytes(e.try_into().unwrap())),
        16 => fat.get(c * 2..c * 2 + 2).map_or(0, |e| u32::from(le16(e, 0))),
        _ => {
            let o = c * 3 / 2;
            match fat.get(o..o + 2) {
                Some(x) => (u32::from(le16(x, 0)) >> if c & 1 != 0 { 4 } else { 0 }) & 0x0fff,
                None => 0,
            }
        }
    }
}

/// The `st_ctime`, `st_atime` and `st_mtime` of a host file.
#[cfg(unix)]
fn host_times(st: &std::fs::Metadata) -> (i64, i64, i64) {
    use std::os::unix::fs::MetadataExt;
    (st.ctime(), st.atime(), st.mtime())
}

/// The `st_ctime`, `st_atime` and `st_mtime` of a host file; on Windows `st_ctime` is the
/// creation time.
#[cfg(not(unix))]
fn host_times(st: &std::fs::Metadata) -> (i64, i64, i64) {
    let secs = |t: io::Result<std::time::SystemTime>| {
        t.ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs() as i64)
    };
    (secs(st.created()), secs(st.accessed()), secs(st.modified()))
}

/// Whether nobody may write the host file, `(st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) == 0`.
#[cfg(unix)]
fn host_read_only(st: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    st.mode() & 0o222 == 0
}

/// Whether the host file is read-only; Windows has only the owner's write bit.
#[cfg(not(unix))]
fn host_read_only(st: &std::fs::Metadata) -> bool {
    st.permissions().readonly()
}

/// The driver, the state behind `s->lock`.
#[derive(Debug)]
struct Vvfat {
    s: Mutex<State>,
}

impl Vvfat {
    fn write_target(bs: &Node) -> Option<Arc<Node>> {
        bs.child(WRITE_TARGET).map(|c| c.node)
    }
}

fn check_aligned(offset: u64, len: usize) -> io::Result<()> {
    if offset % BDRV_SECTOR_SIZE != 0 || len as u64 % BDRV_SECTOR_SIZE != 0 {
        return Err(errno(libc::EINVAL));
    }
    Ok(())
}

impl Driver for Vvfat {
    /// `vvfat_co_preadv()`.
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        check_aligned(offset, buf.len())?;
        let q = Self::write_target(bs);
        let mut s = self.s.lock().unwrap();
        let n = buf.len() / 0x200;
        s.read(q.as_deref(), offset / BDRV_SECTOR_SIZE, buf, n)
    }

    /// `vvfat_co_pwritev()`.
    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        check_aligned(offset, buf.len())?;
        let q = Self::write_target(bs);
        let mut s = self.s.lock().unwrap();
        let n = buf.len() / 0x200;
        s.write(q.as_deref(), offset / BDRV_SECTOR_SIZE, buf, n)
    }

    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        Ok(self.s.lock().unwrap().total_sectors * BDRV_SECTOR_SIZE)
    }

    /// `vvfat_co_block_status()`: everything is data.
    fn block_status(
        &self,
        _bs: &Node,
        _want: u32,
        _offset: u64,
        bytes: u64,
    ) -> Option<io::Result<BlockStatus>> {
        Some(Ok(BlockStatus { ret: BDRV_BLOCK_DATA, pnum: bytes, map: 0, file: None }))
    }

    /// `vvfat_refresh_limits()`: no sub-sector I/O.
    fn refresh_limits(&self, _bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        bl.request_alignment = BDRV_SECTOR_SIZE as u32;
        Ok(())
    }

    /// `vvfat_child_perm()`: the write target is private to this node.
    fn child_perm(&self, _index: usize, _perm: u64, _shared: u64) -> (u64, u64) {
        (BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE, BLK_PERM_WRITE_UNCHANGED)
    }

    /// `vvfat_close()`.
    fn close(&self, _bs: &Node) {
        self.s.lock().unwrap().close_current_file();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(f: &str) -> QDict {
        let mut o = QDict::new();
        vvfat_parse_filename(f, &mut o).unwrap();
        o
    }

    #[test]
    fn parse_filename() {
        let o = parse("fat:floppy:rw:/some/dir");
        assert_eq!(o.get_str("dir"), Some("/some/dir"));
        assert_eq!(o.get("floppy"), Some(&QValue::Bool(true)));
        assert_eq!(o.get("rw"), Some(&QValue::Bool(true)));
        let o = parse("fat:32:/d");
        assert_eq!(o.get("fat-type"), Some(&QValue::Int(32)));
        assert_eq!(parse("fat:16:C:/x").get_str("dir"), Some("C:/x"));
        assert_eq!(parse("fat:dir").get_str("dir"), Some("dir"));
        let mut o = QDict::new();
        let e = vvfat_parse_filename("vvfat:/x", &mut o).unwrap_err();
        assert_eq!(e.message(), "File name string must start with 'fat:'");
    }

    #[test]
    fn short_chars() {
        assert_eq!(to_valid_short_char('a'), b'A');
        assert_eq!(to_valid_short_char('~'), b'~');
        assert_eq!(to_valid_short_char('+'), 0);
        assert_eq!(to_valid_short_char('é'), 0);
        // U+0140 uppercases to U+013F, whose low byte is '?'.
        assert_eq!(to_valid_short_char('\u{0140}'), 0);
        // strchr() sees the low byte 0x21, '!'.
        assert_eq!(to_valid_short_char('\u{0221}'), b'!');
    }

    #[test]
    fn chs() {
        let mut c = [0u8; 3];
        assert!(!sector2chs(&mut c, 0x3f, 1024, 16, 63));
        assert_eq!(c, [1, 1, 0]);
        assert!(sector2chs(&mut c, 1024 * 16 * 63, 1024, 16, 63));
        assert_eq!(c, [0xff; 3]);
    }
}
