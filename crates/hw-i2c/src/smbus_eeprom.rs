// SPDX-License-Identifier: GPL-2.0-or-later

//! The 256 byte SMBus EEPROM and the SPD data generator, QEMU's `hw/i2c/smbus_eeprom.c`.
//!
//! PCs hang one of these at 0x50 and up for each memory module, holding the module's Serial
//! Presence Detect data. The EEPROM keeps a single address pointer: a write sets it from the
//! command byte and stores the rest, a read returns the byte under it. Both move it on by one.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::Error;

use crate::i2c::{I2cBus, I2cSlave};
use crate::smbus::{SmbusDevice, SmbusSlave};

/// `SMBUS_EEPROM_SIZE`.
pub const SMBUS_EEPROM_SIZE: usize = 256;

/// The first EEPROM address used by [`smbus_eeprom_init`].
pub const SMBUS_EEPROM_BASE_ADDR: u8 = 0x50;

struct EepromState {
    data: [u8; SMBUS_EEPROM_SIZE],
    offset: u8,
    accessed: bool,
}

/// An EEPROM plugged into an I2C bus.
pub type SmbusEepromSlave = SmbusSlave<SmbusEeprom>;

/// `smbus-eeprom`, `SMBusEEPROMDevice`.
pub struct SmbusEeprom {
    /// `init_data`: what the contents go back to on reset.
    init_data: [u8; SMBUS_EEPROM_SIZE],
    state: Mutex<EepromState>,
}

impl fmt::Debug for SmbusEeprom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.lock();
        f.debug_struct("SmbusEeprom")
            .field("offset", &s.offset)
            .field("accessed", &s.accessed)
            .finish_non_exhaustive()
    }
}

impl SmbusEeprom {
    /// An EEPROM that starts out holding `init_data`.
    pub fn new(init_data: [u8; SMBUS_EEPROM_SIZE]) -> Arc<SmbusEeprom> {
        Arc::new(SmbusEeprom {
            init_data,
            state: Mutex::new(EepromState { data: init_data, offset: 0, accessed: false }),
        })
    }

    fn lock(&self) -> MutexGuard<'_, EepromState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The current contents.
    pub fn data(&self) -> [u8; SMBUS_EEPROM_SIZE] {
        self.lock().data
    }

    /// The address pointer.
    pub fn offset(&self) -> u8 {
        self.lock().offset
    }

    /// Whether the guest has touched the EEPROM, which decides if QEMU migrates it.
    pub fn accessed(&self) -> bool {
        self.lock().accessed
    }
}

impl SmbusDevice for SmbusEeprom {
    /// `eeprom_receive_byte()`.
    fn receive_byte(&self) -> u8 {
        let mut s = self.lock();
        let val = s.data[usize::from(s.offset)];
        s.offset = s.offset.wrapping_add(1);
        s.accessed = true;
        val
    }

    /// `eeprom_write_data()`.
    fn write_data(&self, buf: &[u8]) {
        let mut s = self.lock();
        s.accessed = true;
        let Some((&offset, rest)) = buf.split_first() else {
            return;
        };
        s.offset = offset;
        for &b in rest {
            let o = usize::from(s.offset);
            s.data[o] = b;
            s.offset = s.offset.wrapping_add(1);
        }
    }

    /// `smbus_eeprom_reset()`: back to the initial contents, as if QEMU had been restarted.
    fn reset(&self) {
        let mut s = self.lock();
        s.data = self.init_data;
        s.offset = 0;
    }
}

/// `smbus_eeprom_init_one()`: an EEPROM holding `eeprom_buf` at `address` on `bus`.
pub fn smbus_eeprom_init_one(
    bus: &I2cBus,
    address: u8,
    eeprom_buf: [u8; SMBUS_EEPROM_SIZE],
) -> Arc<SmbusEepromSlave> {
    let slave = SmbusSlave::new(SmbusEeprom::new(eeprom_buf));
    bus.attach(address, Arc::clone(&slave) as Arc<dyn I2cSlave>);
    slave
}

/// `smbus_eeprom_init()`: `nb_eeprom` EEPROMs at 0x50 and up.
///
/// `eeprom_spd` is laid across them in 256 byte pieces, so the first EEPROM gets the first 256
/// bytes and so on. The rest is zero. Q35 calls this with 8 EEPROMs and no SPD data.
pub fn smbus_eeprom_init(
    bus: &I2cBus,
    nb_eeprom: usize,
    eeprom_spd: &[u8],
) -> Result<Vec<Arc<SmbusEepromSlave>>, Error> {
    const MAX: usize = 8;
    if nb_eeprom > MAX {
        return Err(Error::generic(format!("smbus-eeprom: {nb_eeprom} EEPROMs, at most 8")));
    }
    if eeprom_spd.len() > MAX * SMBUS_EEPROM_SIZE {
        return Err(Error::generic("smbus-eeprom: SPD data larger than 8 EEPROMs"));
    }
    let mut buf = vec![0u8; MAX * SMBUS_EEPROM_SIZE];
    buf[..eeprom_spd.len()].copy_from_slice(eeprom_spd);
    let mut out = Vec::with_capacity(nb_eeprom);
    for (i, chunk) in buf.chunks_exact(SMBUS_EEPROM_SIZE).take(nb_eeprom).enumerate() {
        let mut init = [0u8; SMBUS_EEPROM_SIZE];
        init.copy_from_slice(chunk);
        out.push(smbus_eeprom_init_one(bus, SMBUS_EEPROM_BASE_ADDR + i as u8, init));
    }
    Ok(out)
}

/// The memory type in byte 2 of SPD data, `enum sdram_type`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SdramType {
    Sdr = 0x4,
    Ddr = 0x7,
    Ddr2 = 0x8,
}

/// `spd_data_generate()`: SPD data describing one module of `ram_size` bytes.
///
/// The size has to be a power of two number of MiB and at least the smallest module of the
/// type (4 MiB for SDR, 32 MiB for DDR, 128 MiB for DDR2). QEMU asserts on anything else, this
/// returns an error. Bytes 63 and below are filled in and byte 63 is the checksum.
pub fn spd_data_generate(sdram: SdramType, ram_size: u64) -> Result<[u8; 256], Error> {
    let (min_log2, max_log2) = match sdram {
        SdramType::Sdr => (2, 9),
        SdramType::Ddr => (5, 12),
        SdramType::Ddr2 => (7, 14),
    };
    // Work in MiB. QEMU keeps the size in a 32 bit variable.
    let size = (ram_size >> 20) as u32;
    if size == 0 || u64::from(size) << 20 != ram_size || !size.is_power_of_two() {
        return Err(Error::generic(format!(
            "SPD: RAM size {ram_size:#x} is not a power of two number of MiB"
        )));
    }
    let mut sz_log2 = size.trailing_zeros();
    if sz_log2 < min_log2 {
        return Err(Error::generic(format!("SPD: RAM size {ram_size:#x} is too small")));
    }

    let mut nbanks: u32 = 1;
    while sz_log2 > max_log2 && nbanks < 8 {
        sz_log2 -= 1;
        nbanks *= 2;
    }
    if u64::from(size) != (1u64 << sz_log2) * u64::from(nbanks) {
        return Err(Error::generic(format!("SPD: RAM size {ram_size:#x} is too large")));
    }

    // Split to 2 banks if possible, to avoid a bug in the MIPS Malta firmware.
    if nbanks == 1 && sz_log2 > min_log2 {
        sz_log2 -= 1;
        nbanks += 1;
    }

    let density = 1u64 << (sz_log2 - 2);
    let density = match sdram {
        SdramType::Ddr2 => (density & 0xe0) | (density >> 8 & 0x1f),
        SdramType::Ddr => (density & 0xf8) | (density >> 8 & 0x07),
        SdramType::Sdr => density & 0xff,
    } as u8;

    let ddr2 = sdram == SdramType::Ddr2;
    let mut spd = [0u8; 256];
    spd[0] = 128; // data bytes in EEPROM
    spd[1] = 8; // log2 size of EEPROM
    spd[2] = sdram as u8;
    spd[3] = 13; // row address bits
    spd[4] = 10; // column address bits
    spd[5] = if ddr2 { nbanks - 1 } else { nbanks } as u8;
    spd[6] = 64; // module data width
    spd[8] = 4; // interface voltage level
    spd[9] = 0x25; // highest CAS latency
    spd[10] = 1; // access time
    spd[12] = 0x82; // refresh requirements
    spd[13] = 8; // primary SDRAM width
    spd[15] = if ddr2 { 0 } else { 1 }; // reserved / delay for random col rd
    spd[16] = 12; // burst lengths supported
    spd[17] = 4; // banks per SDRAM device
    spd[18] = 12; // ~CAS latencies supported
    spd[19] = if ddr2 { 0 } else { 1 }; // reserved / ~CS latencies supported
    spd[20] = 2; // DIMM type / ~WE latencies
    spd[21] = if ddr2 { 0 } else { 0x20 }; // module features
    spd[23] = 0x12; // clock cycle time @ medium CAS latency
    spd[27] = 20; // min. row precharge time
    spd[28] = 15; // min. row active row delay
    spd[29] = 20; // min. ~RAS to ~CAS delay
    spd[30] = 45; // min. active to precharge time
    spd[31] = density;
    spd[32] = 20; // addr/cmd setup time
    spd[33] = 8; // addr/cmd hold time
    spd[34] = 20; // data input setup time
    spd[35] = 8; // data input hold time
    spd[36] = if ddr2 { 13 << 2 } else { 0 }; // min. write recovery time

    spd[63] = spd[..63].iter().fold(0u8, |sum, &b| sum.wrapping_add(b));
    Ok(spd)
}
