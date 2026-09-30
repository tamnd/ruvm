// SPDX-License-Identifier: GPL-2.0-or-later

//! The CFI01 parallel flash (Intel command set) from hw/block/pflash_cfi01.c, and the part of
//! hw/i386/pc_sysfw.c that maps the two PC system flashes (OVMF CODE and VARS) below 4 GiB.
//!
//! [`Pflash::new`] does what `pflash_cfi01_realize()` does: it creates a ROM device region, the
//! guest reads the RAM behind it directly while the flash is in read array mode, and every write
//! goes to the command state machine. The first write of a command sequence takes the region out
//! of romd mode (`memory_region_rom_device_set_romd(false)`), so reads then go to the device and
//! return status, CFI or ID data, and going back to read array mode switches romd on again.
//!
//! The contents come from a [`PflashBacking`]. For a file, the whole image is read into the RAM
//! block when the device is created and every program or erase is written back at once, widened
//! to 512 byte sectors like `pflash_update()` does. A read only backing sets `ro`: programs and
//! erases then only set the error bits in the status register and change nothing.
//!
//! [`pc_system_flash_map`] is the check and placement logic of `pc_system_firmware_init()` and
//! `pc_system_flash_map()`: pflash0 ends at 4 GiB, pflash1 sits right below it, and the last
//! 128 KiB of pflash0 are aliased read only at 0xe0000 as "isa-bios" (`x86_isa_bios_init()`,
//! used since the 9.1 machine types). The board creates the devices with
//! [`PflashProps::pc_system_flash`], which has the properties `pc_pflash_create()` sets.
//!
//! Not ported: VMState, trace points, the `secure` property, the unimplemented command log
//! (`LOG_UNIMP`), and the isa-bios copy that machine types before 9.1 use (`pc_isa_bios_init()`).

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use ruvm_base::error::strerror;
use ruvm_base::error_report;
use ruvm_mem::{AccessCtx, AccessSize, MemResult, MemorySystem, MmioOps, RamBlock, RegionId};

/// `TYPE_PFLASH_CFI01`.
pub const TYPE_PFLASH_CFI01: &str = "cfi.pflash01";

/// `FLASH_SECTOR_SIZE` in hw/i386/pc_sysfw.c: the erase block size of the PC system flashes and
/// the unit their sizes must be a multiple of.
pub const FLASH_SECTOR_SIZE: u64 = 4096;

/// The names `pc_system_flash_create()` gives the two PC flashes. They are the region names too.
pub const PC_FLASH_NAMES: [&str; 2] = ["system.flash0", "system.flash1"];

/// The machine properties that alias the flashes' `drive` properties.
pub const PC_FLASH_DRIVE_PROPS: [&str; 2] = ["pflash0", "pflash1"];

/// The error `pc_system_firmware_init()` gives when KVM cannot make read only memory slots.
pub const PFLASH_KVM_READONLY_ERROR: &str = "pflash with kvm requires KVM readonly memory support";

/// `BDRV_SECTOR_SIZE`: write backs are widened to this, and raw images are this granular.
const BDRV_SECTOR_SIZE: u64 = 512;

/// The size of `cfi_table[]`.
const CFI_TABLE_LEN: usize = 0x52;

/// The largest isa-bios window, 128 KiB.
const ISA_BIOS_MAX: u64 = 128 * 1024;

/// Status register: the write state machine is ready.
const STATUS_READY: u8 = 0x80;
/// Status register: block erase error.
const STATUS_ERASE_ERROR: u8 = 0x20;
/// Status register: programming error.
const STATUS_PROGRAM_ERROR: u8 = 0x10;

/// The properties of a `cfi.pflash01` device, with QEMU's defaults (all zero) from
/// [`Default`]. `name` and `drive` are the other two arguments of [`Pflash::new`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PflashProps {
    /// `num-blocks`: the size of the device divided by `sector-length`.
    pub num_blocks: u32,
    /// `sector-length`: the erase block size in bytes.
    pub sector_len: u64,
    /// `width`: the bank width in bytes.
    pub width: u8,
    /// `device-width`: the width of each chip in the bank, 0 for the old unspecified behaviour.
    pub device_width: u8,
    /// `max-device-width`: 0 means the same as `device-width`.
    pub max_device_width: u8,
    /// `big-endian`.
    pub big_endian: bool,
    /// `id0`: the manufacturer ID.
    pub id0: u16,
    /// `id1`: the device ID.
    pub id1: u16,
    /// `id2`.
    pub id2: u16,
    /// `id3`.
    pub id3: u16,
}

impl PflashProps {
    /// What `pc_pflash_create()` and `pc_system_flash_map()` set on a PC system flash of `size`
    /// bytes: `sector-length` 4096, `width` 1 and `num-blocks` `size / 4096`. Everything else
    /// keeps its default, so `device-width` is 0, the IDs are 0 and the flash is little endian.
    pub fn pc_system_flash(size: u64) -> PflashProps {
        PflashProps {
            num_blocks: u32::try_from(size / FLASH_SECTOR_SIZE).unwrap_or(u32::MAX),
            sector_len: FLASH_SECTOR_SIZE,
            width: 1,
            ..PflashProps::default()
        }
    }

    /// The size of the device, `sector-length * num-blocks`.
    pub fn total_len(&self) -> u64 {
        self.sector_len.saturating_mul(u64::from(self.num_blocks))
    }
}

/// Where the contents of a flash come from, the `drive` property.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PflashBacking {
    /// No drive: the flash starts zeroed and writes stay in memory.
    None,
    /// An image in memory, for tests. It must be exactly the size of the device and writes stay
    /// in memory.
    Bytes(Vec<u8>),
    /// A raw image file. Unless `read_only` is set it is opened for writing, and programs and
    /// erases are written back to it.
    File {
        /// The image.
        path: PathBuf,
        /// `readonly=on`: the flash ignores programs and erases and the file is never written.
        read_only: bool,
    },
}

/// The command state machine, the part of `PFlashCFI01` that changes at run time.
#[derive(Debug)]
struct State {
    /// 0 when the flash reads normally, else the cycle of the command sequence.
    wcycle: u8,
    cmd: u8,
    status: u8,
    counter: u64,
    /// The write to buffer update buffer, `blk_bytes`.
    blk_bytes: Vec<u8>,
    /// Where `blk_bytes` goes, `blk_offset` with -1 as `None`.
    blk_offset: Option<u64>,
    /// The image file when it is writable.
    file: Option<File>,
}

/// What a write does after the state machine ran, the labels at the end of `pflash_write()`.
enum Flow {
    Done,
    ReadArray,
    Error,
}

/// A CFI01 flash device, `PFlashCFI01`. It is the region's callbacks too.
pub struct Pflash {
    name: String,
    props: PflashProps,
    /// `max_device_width` after realize, when 0 means `device_width`.
    max_device_width: u8,
    total_len: u64,
    ro: bool,
    cfi_table: [u8; CFI_TABLE_LEN],
    writeblock_size: u32,
    mem: Weak<MemorySystem>,
    region: OnceLock<RegionId>,
    ram: OnceLock<Arc<RamBlock>>,
    state: Mutex<State>,
}

impl fmt::Debug for Pflash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pflash")
            .field("name", &self.name)
            .field("props", &self.props)
            .field("ro", &self.ro)
            .field("region", &self.region.get())
            .finish_non_exhaustive()
    }
}

impl Pflash {
    /// `pflash_cfi01_realize()`: checks the properties, creates the ROM device region `name` in
    /// romd mode, fills it from `backing` and resets the state machine. The region is not mapped
    /// anywhere; the caller adds [`Pflash::region`] to its container.
    ///
    /// The device keeps only a weak reference to `mem`, which owns the device through the region.
    pub fn new(
        mem: &Arc<MemorySystem>,
        name: &str,
        props: PflashProps,
        backing: PflashBacking,
    ) -> Result<Arc<Pflash>, String> {
        if props.sector_len == 0 {
            return Err("attribute \"sector-length\" not specified or zero.".to_string());
        }
        if props.num_blocks == 0 {
            return Err("attribute \"num-blocks\" not specified or zero.".to_string());
        }
        if !matches!(props.width, 1 | 2 | 4) {
            // QEMU leaves a zero or odd width to crash later; refuse it up front.
            return Err(format!("{TYPE_PFLASH_CFI01}: unsupported width {}", props.width));
        }
        let total_len = props
            .sector_len
            .checked_mul(u64::from(props.num_blocks))
            .ok_or_else(|| format!("{TYPE_PFLASH_CFI01} device '{name}' is too large"))?;

        let (contents, ro, file) = load_backing(name, total_len, backing)?;

        let (cfi_table, writeblock_size) = fill_cfi_table(&props);
        let max_device_width =
            if props.max_device_width == 0 { props.device_width } else { props.max_device_width };
        let dev = Arc::new(Pflash {
            name: name.to_string(),
            props,
            max_device_width,
            total_len,
            ro,
            cfi_table,
            writeblock_size,
            mem: Arc::downgrade(mem),
            region: OnceLock::new(),
            ram: OnceLock::new(),
            state: Mutex::new(State {
                wcycle: 0,
                cmd: 0x00,
                status: STATUS_READY,
                counter: 0,
                blk_bytes: vec![0; writeblock_size as usize],
                blk_offset: None,
                file,
            }),
        });

        let ops: Arc<dyn MmioOps> = Arc::clone(&dev) as Arc<dyn MmioOps>;
        let region = mem.new_rom_device(name, total_len, ops).map_err(|e| e.to_string())?;
        let ram = mem
            .ram_block(region)
            .ok_or_else(|| format!("{TYPE_PFLASH_CFI01} device '{name}' has no RAM block"))?;
        if let Some(c) = contents {
            ram.write(0, &c).map_err(|e| e.to_string())?;
        }
        let _ = dev.region.set(region);
        let _ = dev.ram.set(ram);
        Ok(dev)
    }

    /// The ROM device region, `pflash_cfi01_get_memory()`.
    pub fn region(&self) -> RegionId {
        *self.region.get().expect("Pflash::new sets the region")
    }

    fn ram(&self) -> &RamBlock {
        self.ram.get().expect("Pflash::new sets the RAM block")
    }

    /// The name, which is also the region name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The properties the device was created with.
    pub fn props(&self) -> PflashProps {
        self.props
    }

    /// The size in bytes, `sector-length * num-blocks`.
    pub fn size(&self) -> u64 {
        self.total_len
    }

    /// Whether the backing is read only (`pfl->ro`).
    pub fn read_only(&self) -> bool {
        self.ro
    }

    /// The CFI query table, `cfi_table[]`.
    pub fn cfi_table(&self) -> [u8; CFI_TABLE_LEN] {
        self.cfi_table
    }

    /// The write to buffer size in bytes, `writeblock_size`.
    pub fn writeblock_size(&self) -> u32 {
        self.writeblock_size
    }

    /// The status register.
    pub fn status(&self) -> u8 {
        self.lock().status
    }

    /// The current flash contents, the RAM block of the region.
    pub fn contents(&self) -> Vec<u8> {
        let mut buf = vec![0; self.total_len as usize];
        let _ = self.ram().read(0, &mut buf);
        buf
    }

    /// `pflash_cfi01_system_reset()`: back to read array mode with romd on, status ready and no
    /// write to buffer in progress.
    pub fn reset(&self) {
        let mut st = self.lock();
        st.cmd = 0x00;
        st.wcycle = 0;
        self.set_romd(true);
        st.status = STATUS_READY;
        st.blk_offset = None;
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `memory_region_rom_device_set_romd()`.
    fn set_romd(&self, romd: bool) {
        let (Some(mem), Some(&region)) = (self.mem.upgrade(), self.region.get()) else { return };
        let romd_now = mem.region(region).map(|r| r.romd_mode);
        if romd_now != Some(romd) {
            let _ = mem.set_romd(region, romd);
        }
    }

    fn be(&self) -> bool {
        self.props.big_endian
    }

    /// `pflash_data_read()`.
    fn data_read(&self, offset: u64, width: u32) -> u32 {
        let mut b = [0u8; 4];
        let n = width.min(4) as usize;
        let _ = self.ram().read(offset, &mut b[..n]);
        load(&b[..n], self.be())
    }

    /// `pflash_data_write()`.
    fn data_write(&self, st: &mut State, offset: u64, value: u32, width: u32) {
        let n = width.min(4) as usize;
        let bytes = store(value, n, self.be());
        match st.blk_offset {
            Some(base) => {
                // Block write: the data goes to the update buffer.
                let wb = u64::from(self.writeblock_size);
                if offset < base || offset + n as u64 > base + wb {
                    st.status |= STATUS_PROGRAM_ERROR;
                    return;
                }
                let at = (offset - base) as usize;
                st.blk_bytes[at..at + n].copy_from_slice(&bytes[..n]);
            }
            None => {
                let _ = self.ram().write(offset, &bytes[..n]);
            }
        }
    }

    /// `pflash_update()`: writes the range back to the image, widened to 512 byte sectors.
    fn update(&self, st: &mut State, offset: u64, size: u64) {
        let Some(file) = st.file.as_mut() else { return };
        let start = offset & !(BDRV_SECTOR_SIZE - 1);
        let end = (offset + size).next_multiple_of(BDRV_SECTOR_SIZE).min(self.total_len);
        let mut buf = vec![0u8; (end - start) as usize];
        let _ = self.ram().read(start, &mut buf);
        let r = file.seek(SeekFrom::Start(start)).and_then(|_| file.write_all(&buf));
        if let Err(e) = r {
            // QEMU has a TODO to set an error bit in the status here.
            error_report(&format!("Could not update PFLASH: {}", strerror(&e)));
        }
    }

    /// The shift that turns a query address into a table index, used by
    /// `pflash_cfi_query()` and `pflash_devid_query()`.
    fn query_shift(&self) -> u32 {
        let ctz = |v: u8| u32::from(v).trailing_zeros();
        (ctz(self.props.width) + ctz(self.max_device_width))
            .saturating_sub(ctz(self.props.device_width))
    }

    /// Replicates a response for each device in the bank.
    fn replicate(&self, mut resp: u32) -> u32 {
        let dw = u32::from(self.props.device_width);
        let bank = u32::from(self.props.width);
        if dw < bank {
            let mut i = dw;
            while i < bank {
                resp = deposit32(resp, 8 * i, 8 * dw, resp);
                i += dw;
            }
        }
        resp
    }

    /// `pflash_cfi_query()`.
    fn cfi_query(&self, offset: u64) -> u32 {
        let boff = offset >> self.query_shift();
        if boff >= CFI_TABLE_LEN as u64 {
            return 0;
        }
        let byte = u32::from(self.cfi_table[boff as usize]);
        let mut resp = byte;
        if self.props.device_width != self.max_device_width {
            // The only case supported is x8 mode for a wider part.
            if self.props.device_width != 1 || self.props.width > 4 {
                return 0;
            }
            // CFI query data is repeated, rather than zero padded, for wide devices in x8 mode.
            for i in 1..u32::from(self.max_device_width) {
                resp = deposit32(resp, 8 * i, 8, byte);
            }
        }
        self.replicate(resp)
    }

    /// `pflash_devid_query()`.
    fn devid_query(&self, offset: u64) -> u32 {
        let boff = offset >> self.query_shift();
        // Offsets 2 and 3 are the block lock status, which is not emulated.
        let resp = match boff & 0xff {
            0 => u32::from(self.props.id0),
            1 => u32::from(self.props.id1),
            _ => return 0,
        };
        self.replicate(resp)
    }

    /// The table index of the old behaviour without `device-width`.
    fn legacy_boff(&self, offset: u64) -> u64 {
        let boff = offset & 0xff;
        match self.props.width {
            2 => boff >> 1,
            4 => boff >> 2,
            _ => boff,
        }
    }

    /// `pflash_read()`.
    fn pflash_read(&self, st: &mut State, offset: u64, width: u32) -> u32 {
        let dw = u32::from(self.props.device_width);
        let bank = u32::from(self.props.width);
        match st.cmd {
            0x00 => self.data_read(offset, width),
            0x10 | 0x20 | 0x28 | 0x40 | 0x50 | 0x60 | 0x70 | 0xe8 => {
                // Status register read, the status of each device in the bank.
                let status = u32::from(st.status);
                let mut ret = status;
                if dw != 0 && width > dw {
                    let mut shift = dw * 8;
                    while shift + dw * 8 <= width * 8 {
                        ret |= status.checked_shl(shift).unwrap_or(0);
                        shift += dw * 8;
                    }
                } else if dw == 0 && width > 2 {
                    // 32 bit flash without a device width, the behaviour from before it existed.
                    ret |= status << 16;
                }
                ret
            }
            0x90 => {
                if dw == 0 {
                    let (a, b) = match self.legacy_boff(offset) {
                        0 => (self.props.id0, self.props.id1),
                        1 => (self.props.id2, self.props.id3),
                        _ => return 0,
                    };
                    (u32::from(a) << 8) | u32::from(b)
                } else {
                    // A read wider than the bank combines several queries.
                    let mut ret = u32::MAX;
                    let mut i = 0;
                    while i < width {
                        let q = self.devid_query(offset + u64::from(i * bank));
                        ret = deposit32(ret, i * 8, bank * 8, q);
                        i += bank;
                    }
                    ret
                }
            }
            0x98 => {
                if dw == 0 {
                    let boff = self.legacy_boff(offset) as usize;
                    self.cfi_table.get(boff).copied().map_or(0, u32::from)
                } else {
                    let mut ret = u32::MAX;
                    let mut i = 0;
                    while i < width {
                        let q = self.cfi_query(offset + u64::from(i * bank));
                        ret = deposit32(ret, i * 8, bank * 8, q);
                        i += bank;
                    }
                    ret
                }
            }
            _ => {
                // Should never happen: reset the state and treat it as a read.
                st.wcycle = 0;
                st.cmd = 0x00;
                self.data_read(offset, width)
            }
        }
    }

    /// `pflash_write()`.
    fn pflash_write(&self, st: &mut State, offset: u64, value: u32, width: u32) {
        let cmd = value as u8;
        if st.wcycle == 0 {
            // Put the device in I/O access mode.
            self.set_romd(false);
        }
        let flow = match st.wcycle {
            0 => self.write_cycle0(st, offset, cmd),
            1 => self.write_cycle1(st, offset, value, width, cmd),
            2 => self.write_cycle2(st, offset, value, width),
            3 => {
                // Confirm mode.
                if st.cmd == 0xe8 {
                    if cmd == 0xd0 && st.status & STATUS_PROGRAM_ERROR == 0 {
                        self.blk_write_flush(st);
                        st.wcycle = 0;
                        st.status |= STATUS_READY;
                        Flow::Done
                    } else {
                        st.blk_offset = None;
                        Flow::ReadArray
                    }
                } else {
                    st.blk_offset = None;
                    Flow::Error
                }
            }
            // Should never happen.
            _ => Flow::ReadArray,
        };
        match flow {
            Flow::Done => {}
            // The unimplemented command sequence is not logged.
            Flow::ReadArray | Flow::Error => {
                self.set_romd(true);
                st.wcycle = 0;
                st.cmd = 0x00;
            }
        }
    }

    fn write_cycle0(&self, st: &mut State, offset: u64, cmd: u8) -> Flow {
        match cmd {
            // 0x00 is this model's read array value, 0xf0 the AMD probe and 0xff read array.
            0x00 | 0xf0 | 0xff => return Flow::ReadArray,
            // Single byte program, block (un)lock and CFI query just move to the next cycle.
            0x10 | 0x40 | 0x60 | 0x98 => {}
            0x20 => {
                // Block erase happens right away; the confirm cycle only acknowledges it.
                let offset = offset & !(self.props.sector_len - 1);
                if self.ro {
                    st.status |= STATUS_ERASE_ERROR;
                } else {
                    let _ = self.ram().fill(offset, self.props.sector_len, 0xff);
                    self.update(st, offset, self.props.sector_len);
                }
                st.status |= STATUS_READY;
            }
            0x50 => {
                // Clear status bits.
                st.status = 0;
                return Flow::ReadArray;
            }
            0x70 | 0x90 => {
                // Read status register or device ID. wcycle stays 0.
                st.cmd = cmd;
                return Flow::Done;
            }
            0xe8 => {
                // Write to buffer.
                st.status |= STATUS_READY;
            }
            _ => return Flow::Error,
        }
        st.wcycle += 1;
        st.cmd = cmd;
        Flow::Done
    }

    fn write_cycle1(&self, st: &mut State, offset: u64, value: u32, width: u32, cmd: u8) -> Flow {
        match st.cmd {
            0x10 | 0x40 => {
                // Single byte program.
                if self.ro {
                    st.status |= STATUS_PROGRAM_ERROR;
                } else {
                    self.data_write(st, offset, value, width);
                    self.update(st, offset, u64::from(width));
                }
                st.status |= STATUS_READY;
                st.wcycle = 0;
                Flow::Done
            }
            0x20 | 0x28 => match cmd {
                0xd0 => {
                    st.wcycle = 0;
                    st.status |= STATUS_READY;
                    Flow::Done
                }
                0xff => Flow::ReadArray,
                _ => Flow::Error,
            },
            0xe8 => {
                // The word count, masked to the device width, or the bank width without one.
                let bytes = if self.props.device_width != 0 {
                    self.props.device_width
                } else {
                    self.props.width
                };
                st.counter = u64::from(extract32(value, 0, u32::from(bytes) * 8));
                st.wcycle += 1;
                Flow::Done
            }
            0x60 => match cmd {
                // Lock (0x01) and unlock (0xd0) are accepted and do nothing else.
                0xd0 | 0x01 => {
                    st.wcycle = 0;
                    st.status |= STATUS_READY;
                    Flow::Done
                }
                // Read array, or an unknown (un)locking command.
                _ => Flow::ReadArray,
            },
            0x98 => {
                // Anything but read array leaves the flash in query mode.
                if cmd == 0xff { Flow::ReadArray } else { Flow::Done }
            }
            _ => Flow::Error,
        }
    }

    fn write_cycle2(&self, st: &mut State, offset: u64, value: u32, width: u32) -> Flow {
        if st.cmd != 0xe8 {
            return Flow::Error;
        }
        // Block write. QEMU does not check the offset or the width either.
        if st.blk_offset.is_none() && st.counter != 0 {
            // `pflash_blk_write_start()`: copy the current contents to the update buffer.
            let base = offset & !(u64::from(self.writeblock_size) - 1);
            st.blk_offset = Some(base);
            let _ = self.ram().read(base, &mut st.blk_bytes);
        }
        if !self.ro && st.blk_offset.is_some() {
            self.data_write(st, offset, value, width);
        } else {
            st.status |= STATUS_PROGRAM_ERROR;
        }
        st.status |= STATUS_READY;
        if st.counter == 0 {
            // Block write finished.
            st.wcycle += 1;
        } else {
            st.counter -= 1;
        }
        Flow::Done
    }

    /// `pflash_blk_write_flush()`: commits the update buffer.
    fn blk_write_flush(&self, st: &mut State) {
        let Some(base) = st.blk_offset else { return };
        let _ = self.ram().write(base, &st.blk_bytes);
        self.update(st, base, u64::from(self.writeblock_size));
        st.blk_offset = None;
    }
}

impl MmioOps for Pflash {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        let mut st = self.lock();
        let v = self.pflash_read(&mut st, offset, size.bytes());
        Ok(u64::from(v) & size.mask())
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        let mut st = self.lock();
        self.pflash_write(&mut st, offset, value as u32, size.bytes());
        Ok(())
    }
}

/// Reads the backing into a buffer of `total_len` bytes and opens the file for write back.
type Loaded = (Option<Vec<u8>>, bool, Option<File>);

fn load_backing(name: &str, total_len: u64, backing: PflashBacking) -> Result<Loaded, String> {
    let size_error = |len: u64| {
        format!(
            "{TYPE_PFLASH_CFI01} device '{name}' requires {total_len} bytes, \
             block backend provides {len} bytes"
        )
    };
    match backing {
        PflashBacking::None => Ok((None, false, None)),
        PflashBacking::Bytes(b) => {
            if b.len() as u64 != total_len {
                return Err(size_error(b.len() as u64));
            }
            Ok((Some(b), false, None))
        }
        PflashBacking::File { path, read_only } => {
            let open_err = |e: std::io::Error| {
                format!("Could not open '{}': {}", path.display(), strerror(&e))
            };
            let mut file = if read_only {
                File::open(&path).map_err(open_err)?
            } else {
                OpenOptions::new().read(true).write(true).open(&path).map_err(open_err)?
            };
            let mut data = Vec::new();
            file.read_to_end(&mut data).map_err(|e| {
                format!(
                    "can't read block backend for {TYPE_PFLASH_CFI01} device '{name}': {}",
                    strerror(&e)
                )
            })?;
            // A raw image is as long as its file rounded up to whole 512 byte sectors, and the
            // tail reads as zeros.
            let len = raw_block_length(data.len() as u64);
            if len != total_len {
                return Err(size_error(len));
            }
            data.resize(total_len as usize, 0);
            Ok((Some(data), read_only, if read_only { None } else { Some(file) }))
        }
    }
}

/// `pflash_cfi01_fill_cfi_table()`: the CFI query table and the write to buffer size.
fn fill_cfi_table(p: &PflashProps) -> ([u8; CFI_TABLE_LEN], u32) {
    let num_devices = u64::from(p.width.checked_div(p.device_width).unwrap_or(1));
    let blocks_per_device = u64::from(p.num_blocks);
    let sector_len_per_device = p.sector_len / num_devices.max(1);
    let device_len = sector_len_per_device * blocks_per_device;

    let mut t = [0u8; CFI_TABLE_LEN];
    // Standard "QRY" string.
    t[0x10] = b'Q';
    t[0x11] = b'R';
    t[0x12] = b'Y';
    // Command set (Intel).
    t[0x13] = 0x01;
    t[0x14] = 0x00;
    // Primary extended table address.
    t[0x15] = 0x31;
    t[0x16] = 0x00;
    // Alternate command set and extended table (none).
    t[0x17] = 0x00;
    t[0x18] = 0x00;
    t[0x19] = 0x00;
    t[0x1a] = 0x00;
    // Vcc min and max.
    t[0x1b] = 0x45;
    t[0x1c] = 0x55;
    // Vpp min and max (no Vpp pin).
    t[0x1d] = 0x00;
    t[0x1e] = 0x00;
    // Reserved.
    t[0x1f] = 0x07;
    // Timeout for min size buffer write.
    t[0x20] = 0x07;
    // Typical timeout for block erase.
    t[0x21] = 0x0a;
    // Typical timeout for full chip erase (4096 ms).
    t[0x22] = 0x00;
    // Reserved.
    t[0x23] = 0x04;
    // Max timeout for buffer write.
    t[0x24] = 0x04;
    // Max timeout for block erase.
    t[0x25] = 0x04;
    // Max timeout for chip erase.
    t[0x26] = 0x00;
    // Device size, ctz32() of the length as QEMU has it.
    t[0x27] = (device_len as u32).trailing_zeros() as u8;
    // Flash device interface (8 and 16 bits).
    t[0x28] = 0x02;
    t[0x29] = 0x00;
    // Max number of bytes in multi-bytes write.
    t[0x2a] = if p.width == 1 { 0x08 } else { 0x0b };
    let mut writeblock_size = 1u32 << t[0x2a];
    if num_devices > 1 {
        writeblock_size *= num_devices as u32;
    }
    t[0x2b] = 0x00;
    // Number of erase block regions (uniform).
    t[0x2c] = 0x01;
    // Erase block region 1.
    t[0x2d] = (blocks_per_device.wrapping_sub(1)) as u8;
    t[0x2e] = (blocks_per_device.wrapping_sub(1) >> 8) as u8;
    t[0x2f] = (sector_len_per_device >> 8) as u8;
    t[0x30] = (sector_len_per_device >> 16) as u8;
    // Extended query table.
    t[0x31] = b'P';
    t[0x32] = b'R';
    t[0x33] = b'I';
    t[0x34] = b'1';
    t[0x35] = b'0';
    // Number of protection fields.
    t[0x3f] = 0x01;
    (t, writeblock_size)
}

/// `deposit32()`.
fn deposit32(value: u32, start: u32, length: u32, field: u32) -> u32 {
    if start >= 32 || length == 0 {
        return value;
    }
    let length = length.min(32 - start);
    let mask = (u32::MAX >> (32 - length)) << start;
    (value & !mask) | ((field << start) & mask)
}

/// `extract32()`.
fn extract32(value: u32, start: u32, length: u32) -> u32 {
    if length >= 32 { value >> start } else { (value >> start) & ((1 << length) - 1) }
}

/// `ldn_le_p()` or `ldn_be_p()`.
fn load(b: &[u8], be: bool) -> u32 {
    let mut v = 0u32;
    if be {
        for &x in b {
            v = (v << 8) | u32::from(x);
        }
    } else {
        for &x in b.iter().rev() {
            v = (v << 8) | u32::from(x);
        }
    }
    v
}

/// `stn_le_p()` or `stn_be_p()` of `n` bytes.
fn store(value: u32, n: usize, be: bool) -> [u8; 4] {
    let le = value.to_le_bytes();
    let mut out = [0u8; 4];
    for (i, o) in out.iter_mut().take(n).enumerate() {
        *o = if be { le[n - 1 - i] } else { le[i] };
    }
    out
}

/// The length the raw block driver reports for a file of `file_len` bytes: rounded up to whole
/// 512 byte sectors. This is the size `pc_system_flash_map()` checks, so pass it to
/// [`pc_system_flash_map`] rather than the bare file length.
pub fn raw_block_length(file_len: u64) -> u64 {
    file_len.next_multiple_of(BDRV_SECTOR_SIZE)
}

/// A drive attached to one of the PC flashes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlashDrive {
    /// The block backend name QEMU prints, `blk_name()`: "pflash0" for the first
    /// `-drive if=pflash` and so on.
    pub name: String,
    /// The image size as the block layer reports it, see [`raw_block_length`].
    pub size: u64,
}

/// Where one flash goes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FlashMapping {
    /// 0 for pflash0 (OVMF CODE), 1 for pflash1 (OVMF VARS).
    pub index: usize,
    /// The device and region name, from [`PC_FLASH_NAMES`].
    pub name: &'static str,
    /// The guest physical address of the first byte.
    pub base: u64,
    /// The size in bytes.
    pub size: u64,
    /// The device properties, [`PflashProps::pc_system_flash`] of `size`.
    pub props: PflashProps,
}

/// The "isa-bios" alias `x86_isa_bios_init()` makes of the top of pflash0: `size` bytes of the
/// flash region from `flash_offset`, mapped at `addr` in the ROM memory container (the PCI memory
/// region, or system memory without PCI) with priority 1, read only.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct IsaBiosAlias {
    /// The region name, "isa-bios".
    pub name: &'static str,
    /// The offset in the pflash0 region.
    pub flash_offset: u64,
    /// The size, at most 128 KiB.
    pub size: u64,
    /// Where it goes, `1 MiB - size`.
    pub addr: u64,
    /// The subregion priority.
    pub priority: i32,
    /// `memory_region_set_readonly()`. This only keeps direct RAM writes out: in command mode
    /// and for writes the alias still reaches the flash callbacks, as in QEMU.
    pub readonly: bool,
}

/// The result of [`pc_system_flash_map`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SystemFlashMap {
    /// The flashes to create and map, pflash0 first. Empty when pflash0 has no drive, in which
    /// case the board loads bios.bin as ROM instead.
    pub flashes: Vec<FlashMapping>,
    /// The isa-bios alias of pflash0, when there is a pflash0.
    pub isa_bios: Option<IsaBiosAlias>,
}

/// Why the flashes cannot be mapped. [`fmt::Display`] gives QEMU's `error_report()` text and
/// [`FlashMapError::info`] the `info_report()` line that follows it, if any.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlashMapError {
    /// pflash`index` has a drive but the one before it has none.
    Gap {
        /// The flash with the drive.
        index: usize,
    },
    /// A drive is empty or not a multiple of 4 KiB.
    InvalidSize {
        /// `blk_name()`.
        name: String,
        /// The size found.
        size: u64,
    },
    /// The flashes together are larger than `max-fw-size`.
    TooLarge {
        /// `max-fw-size`.
        max_fw_size: u64,
    },
}

impl FlashMapError {
    /// The `info_report()` line QEMU prints after the error, without the "info: " prefix.
    pub fn info(&self) -> Option<String> {
        match self {
            FlashMapError::InvalidSize { .. } => {
                Some(format!("its size must be a non-zero multiple of 0x{FLASH_SECTOR_SIZE:x}"))
            }
            _ => None,
        }
    }
}

impl fmt::Display for FlashMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlashMapError::Gap { index } => {
                write!(f, "pflash{index} requires pflash{}", index - 1)
            }
            FlashMapError::InvalidSize { name, size } => {
                write!(f, "system firmware block device {name} has invalid size {size}")
            }
            FlashMapError::TooLarge { max_fw_size } => {
                write!(f, "combined size of system firmware exceeds {max_fw_size} bytes")
            }
        }
    }
}

impl std::error::Error for FlashMapError {}

/// The pflash part of `pc_system_firmware_init()` and `pc_system_flash_map()` for a machine
/// with PCI and the isa-bios alias (every q35 type from 9.1 on).
///
/// `drives[i]` is the drive of pflash`i`. A drive for pflash1 without one for pflash0 is an
/// error. The flashes are stacked downward from 4 GiB without gaps: pflash0 at
/// `4 GiB - size0`, pflash1 at `4 GiB - size0 - size1`. Each size must be a non-zero multiple of
/// 4 KiB and the total must not exceed `max_fw_size` (the `max-fw-size` machine property,
/// 8 MiB by default).
///
/// The board must also refuse pflash under KVM without read only memory slots, with
/// [`PFLASH_KVM_READONLY_ERROR`], before mapping anything.
pub fn pc_system_flash_map(
    drives: &[Option<FlashDrive>; 2],
    max_fw_size: u64,
) -> Result<SystemFlashMap, FlashMapError> {
    // Reject gaps.
    for i in 1..drives.len() {
        if drives[i].is_some() && drives[i - 1].is_none() {
            return Err(FlashMapError::Gap { index: i });
        }
    }

    let mut map = SystemFlashMap::default();
    let mut total: u64 = 0;
    for (index, drive) in drives.iter().enumerate() {
        let Some(drive) = drive else { break };
        let size = drive.size;
        if size == 0 || size % FLASH_SECTOR_SIZE != 0 {
            return Err(FlashMapError::InvalidSize { name: drive.name.clone(), size });
        }
        match total.checked_add(size) {
            Some(t) if t <= max_fw_size => total = t,
            _ => return Err(FlashMapError::TooLarge { max_fw_size }),
        }
        let base = (1u64 << 32).saturating_sub(total);
        map.flashes.push(FlashMapping {
            index,
            name: PC_FLASH_NAMES[index],
            base,
            size,
            props: PflashProps::pc_system_flash(size),
        });
        if index == 0 {
            let isa = size.min(ISA_BIOS_MAX);
            map.isa_bios = Some(IsaBiosAlias {
                name: "isa-bios",
                flash_offset: size - isa,
                size: isa,
                addr: 0x10_0000 - isa,
                priority: 1,
                readonly: true,
            });
        }
    }
    Ok(map)
}
