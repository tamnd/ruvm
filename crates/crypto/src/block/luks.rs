// SPDX-License-Identifier: GPL-2.0-or-later

//! The LUKS1 on disk format, ported from QEMU's `crypto/block-luks.c`.
//!
//! The header is 592 bytes, big endian, followed by the key material of eight keyslots. Each
//! keyslot holds the master key, split into 4000 stripes by the anti-forensic splitter and
//! encrypted with a key derived from a password by PBKDF2. The master key is verified against a
//! PBKDF2 digest stored in the header.
//!
//! Differences from QEMU:
//!
//! * An `iter-time` of zero, which makes QEMU divide by zero, scales the calibrated iteration
//!   count to zero, so the minimum iteration count is used. A negative `iter-time` is treated
//!   as a huge unsigned value, as in C, and fails the scaling check.
//! * `ERANGE` errors on non-unix hosts use the glibc text "Numerical result out of range".

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{
    QCryptoBlockAmendOptionsLUKS, QCryptoBlockCreateOptionsLUKS, QCryptoBlockInfoLUKS,
    QCryptoBlockInfoLUKSSlot, QCryptoBlockLUKSKeyslotState, QCryptoBlockOptionsLUKS,
    QCryptoCipherAlgo, QCryptoCipherMode, QCryptoHashAlgo, QCryptoIVGenAlgo,
};

use super::{
    QCRYPTO_BLOCK_OPEN_DETACHED, QCRYPTO_BLOCK_OPEN_NO_IO, QCryptoBlock, QCryptoBlockIo,
    cipher_encdec,
};
use crate::afsplit::{afsplit_decode, afsplit_encode};
use crate::cipher::{Cipher, cipher_get_iv_len, cipher_get_key_len};
use crate::hash::hash_digest_len;
use crate::ivgen::IvGen;
use crate::pbkdf::{pbkdf2, pbkdf2_count_iters};
use crate::random::random_bytes;
use crate::secret::secret_lookup_as_utf8;

/// `QCRYPTO_BLOCK_LUKS_SECTOR_SIZE`.
pub const SECTOR_SIZE: u64 = 512;
/// `QCRYPTO_BLOCK_LUKS_MAGIC`.
pub const MAGIC: [u8; 6] = *b"LUKS\xBA\xBE";
/// `QCRYPTO_BLOCK_LUKS_VERSION`.
pub const VERSION: u16 = 1;
/// `QCRYPTO_BLOCK_LUKS_NUM_KEY_SLOTS`.
pub const NUM_KEY_SLOTS: usize = 8;
/// `QCRYPTO_BLOCK_LUKS_STRIPES`.
pub const STRIPES: u32 = 4000;
/// `QCRYPTO_BLOCK_LUKS_KEY_SLOT_DISABLED`.
pub const KEY_SLOT_DISABLED: u32 = 0x0000_DEAD;
/// `QCRYPTO_BLOCK_LUKS_KEY_SLOT_ENABLED`.
pub const KEY_SLOT_ENABLED: u32 = 0x00AC_71F3;
/// `QCRYPTO_BLOCK_LUKS_KEY_SLOT_OFFSET`: where the first keyslot starts, in bytes.
pub const KEY_SLOT_OFFSET: u64 = 4096;
/// `QCRYPTO_BLOCK_LUKS_DEFAULT_ITER_TIME_MS`.
pub const DEFAULT_ITER_TIME_MS: u64 = 2000;
/// `QCRYPTO_BLOCK_LUKS_MIN_SLOT_KEY_ITERS`.
pub const MIN_SLOT_KEY_ITERS: u64 = 1000;
/// `QCRYPTO_BLOCK_LUKS_MIN_MASTER_KEY_ITERS`.
pub const MIN_MASTER_KEY_ITERS: u64 = 1000;
/// `QCRYPTO_BLOCK_LUKS_ERASE_ITERATIONS`.
pub const ERASE_ITERATIONS: usize = 40;
/// The size of the on disk header, `sizeof(QCryptoBlockLUKSHeader)`.
pub const HEADER_LEN: usize = 592;

const CIPHER_NAME_LEN: usize = 32;
const CIPHER_MODE_LEN: usize = 32;
const HASH_SPEC_LEN: usize = 32;
const DIGEST_LEN: usize = 20;
const SALT_LEN: usize = 32;
const UUID_LEN: usize = 40;
const CIPHER_NAME_OFFSET: usize = 8;

/// `QCryptoBlockLUKSKeySlot`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeySlot {
    pub active: u32,
    pub iterations: u32,
    pub salt: [u8; SALT_LEN],
    pub key_offset_sector: u32,
    pub stripes: u32,
}

/// `QCryptoBlockLUKSHeader`, in host byte order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub magic: [u8; 6],
    pub version: u16,
    pub cipher_name: [u8; CIPHER_NAME_LEN],
    pub cipher_mode: [u8; CIPHER_MODE_LEN],
    pub hash_spec: [u8; HASH_SPEC_LEN],
    pub payload_offset_sector: u32,
    pub master_key_len: u32,
    pub master_key_digest: [u8; DIGEST_LEN],
    pub master_key_salt: [u8; SALT_LEN],
    pub master_key_iterations: u32,
    pub uuid: [u8; UUID_LEN],
    pub key_slots: [KeySlot; NUM_KEY_SLOTS],
}

impl Default for Header {
    fn default() -> Header {
        Header {
            magic: [0; 6],
            version: 0,
            cipher_name: [0; CIPHER_NAME_LEN],
            cipher_mode: [0; CIPHER_MODE_LEN],
            hash_spec: [0; HASH_SPEC_LEN],
            payload_offset_sector: 0,
            master_key_len: 0,
            master_key_digest: [0; DIGEST_LEN],
            master_key_salt: [0; SALT_LEN],
            master_key_iterations: 0,
            uuid: [0; UUID_LEN],
            key_slots: [KeySlot::default(); NUM_KEY_SLOTS],
        }
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let mut a = [0u8; N];
        a.copy_from_slice(&self.buf[self.pos..self.pos + N]);
        self.pos += N;
        a
    }
    fn be16(&mut self) -> u16 {
        u16::from_be_bytes(self.take())
    }
    fn be32(&mut self) -> u32 {
        u32::from_be_bytes(self.take())
    }
}

impl Header {
    /// Decodes the on disk (big endian) header. `buf` must hold at least [`HEADER_LEN`] bytes.
    pub fn from_bytes(buf: &[u8]) -> Header {
        let mut r = Reader { buf: &buf[..HEADER_LEN], pos: 0 };
        let mut h = Header {
            magic: r.take(),
            version: r.be16(),
            cipher_name: r.take(),
            cipher_mode: r.take(),
            hash_spec: r.take(),
            payload_offset_sector: r.be32(),
            master_key_len: r.be32(),
            master_key_digest: r.take(),
            master_key_salt: r.take(),
            master_key_iterations: r.be32(),
            uuid: r.take(),
            key_slots: [KeySlot::default(); NUM_KEY_SLOTS],
        };
        for s in &mut h.key_slots {
            s.active = r.be32();
            s.iterations = r.be32();
            s.salt = r.take();
            s.key_offset_sector = r.be32();
            s.stripes = r.be32();
        }
        h
    }

    /// Encodes the header in its on disk (big endian) form.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(HEADER_LEN);
        v.extend_from_slice(&self.magic);
        v.extend_from_slice(&self.version.to_be_bytes());
        v.extend_from_slice(&self.cipher_name);
        v.extend_from_slice(&self.cipher_mode);
        v.extend_from_slice(&self.hash_spec);
        v.extend_from_slice(&self.payload_offset_sector.to_be_bytes());
        v.extend_from_slice(&self.master_key_len.to_be_bytes());
        v.extend_from_slice(&self.master_key_digest);
        v.extend_from_slice(&self.master_key_salt);
        v.extend_from_slice(&self.master_key_iterations.to_be_bytes());
        v.extend_from_slice(&self.uuid);
        for s in &self.key_slots {
            v.extend_from_slice(&s.active.to_be_bytes());
            v.extend_from_slice(&s.iterations.to_be_bytes());
            v.extend_from_slice(&s.salt);
            v.extend_from_slice(&s.key_offset_sector.to_be_bytes());
            v.extend_from_slice(&s.stripes.to_be_bytes());
        }
        debug_assert_eq!(v.len(), HEADER_LEN);
        v
    }
}

/// The text of a NUL terminated header field.
fn cstr(field: &[u8]) -> String {
    let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
    String::from_utf8_lossy(&field[..end]).into_owned()
}

fn set_cstr(field: &mut [u8], s: &str) {
    field.fill(0);
    field[..s.len()].copy_from_slice(s.as_bytes());
}

/// `QCryptoBlockLUKS`: the parsed header and the algorithms it names.
#[derive(Debug)]
pub struct Luks {
    pub header: Header,
    pub cipher_alg: QCryptoCipherAlgo,
    pub cipher_mode: QCryptoCipherMode,
    pub ivgen_alg: QCryptoIVGenAlgo,
    pub ivgen_hash_alg: QCryptoHashAlgo,
    pub ivgen_cipher_alg: QCryptoCipherAlgo,
    pub hash_alg: QCryptoHashAlgo,
    /// The id of the secret that was used to open or create the image.
    pub secret: Option<String>,
}

impl Luks {
    fn new(header: Header) -> Luks {
        Luks {
            header,
            cipher_alg: QCryptoCipherAlgo::Aes128,
            cipher_mode: QCryptoCipherMode::Ecb,
            ivgen_alg: QCryptoIVGenAlgo::Plain,
            ivgen_hash_alg: QCryptoHashAlgo::Md5,
            ivgen_cipher_alg: QCryptoCipherAlgo::Aes128,
            hash_alg: QCryptoHashAlgo::Md5,
            secret: None,
        }
    }

    fn mklen(&self) -> usize {
        self.header.master_key_len as usize
    }

    fn slot_active(&self, i: usize) -> bool {
        self.header.key_slots[i].active == KEY_SLOT_ENABLED
    }

    fn count_active_slots(&self) -> usize {
        (0..NUM_KEY_SLOTS).filter(|&i| self.slot_active(i)).count()
    }

    fn find_free_keyslot(&self) -> Option<usize> {
        (0..NUM_KEY_SLOTS).find(|&i| !self.slot_active(i))
    }

    /// `qcrypto_block_luks_splitkeylen_sectors()`: sectors of key material for one keyslot,
    /// rounded up to a multiple of the header size as cryptsetup does.
    fn splitkeylen_sectors(&self, header_sectors: u32, stripes: u32) -> u32 {
        let splitkeylen = u64::from(self.header.master_key_len) * u64::from(stripes);
        let sectors = splitkeylen.div_ceil(SECTOR_SIZE);
        let hs = u64::from(header_sectors);
        (sectors.div_ceil(hs) * hs) as u32
    }
}

const CIPHER_NAME_MAP: &[(&str, &[(u32, QCryptoCipherAlgo)])] = &[
    (
        "aes",
        &[
            (16, QCryptoCipherAlgo::Aes128),
            (24, QCryptoCipherAlgo::Aes192),
            (32, QCryptoCipherAlgo::Aes256),
        ],
    ),
    ("cast5", &[(16, QCryptoCipherAlgo::Cast5_128)]),
    (
        "serpent",
        &[
            (16, QCryptoCipherAlgo::Serpent128),
            (24, QCryptoCipherAlgo::Serpent192),
            (32, QCryptoCipherAlgo::Serpent256),
        ],
    ),
    (
        "twofish",
        &[
            (16, QCryptoCipherAlgo::Twofish128),
            (24, QCryptoCipherAlgo::Twofish192),
            (32, QCryptoCipherAlgo::Twofish256),
        ],
    ),
    ("sm4", &[(16, QCryptoCipherAlgo::Sm4)]),
];

/// `qcrypto_block_luks_cipher_name_lookup()`.
fn cipher_name_lookup(
    name: &str,
    mode: QCryptoCipherMode,
    key_bytes: u32,
) -> Result<QCryptoCipherAlgo> {
    let key_bytes = if mode == QCryptoCipherMode::Xts { key_bytes / 2 } else { key_bytes };
    for (n, sizes) in CIPHER_NAME_MAP {
        if *n != name {
            continue;
        }
        if let Some((_, id)) = sizes.iter().find(|(k, _)| *k == key_bytes) {
            return Ok(*id);
        }
    }
    Err(Error::generic(format!(
        "Algorithm '{name}' with key size {} bytes not supported",
        key_bytes as i32
    )))
}

/// `qcrypto_block_luks_cipher_alg_lookup()`.
fn cipher_alg_lookup(alg: QCryptoCipherAlgo) -> Result<&'static str> {
    for (n, sizes) in CIPHER_NAME_MAP {
        if sizes.iter().any(|(_, id)| *id == alg) {
            return Ok(n);
        }
    }
    Err(Error::generic(format!("Algorithm '{}' not supported", alg.as_str())))
}

fn name_lookup<T>(name: &str, parse: fn(&str) -> Option<T>, ty: &str) -> Result<T> {
    parse(name).ok_or_else(|| Error::generic(format!("{ty} '{name}' not supported")))
}

/// `qcrypto_block_luks_has_format()`.
pub fn has_format(buf: &[u8]) -> bool {
    buf.len() >= CIPHER_NAME_OFFSET
        && buf[..6] == MAGIC
        && u16::from_be_bytes([buf[6], buf[7]]) == VERSION
}

/// `qcrypto_block_luks_essiv_cipher()`: dm-crypt sizes the ESSIV cipher key to the digest, so
/// AES-128 with SHA-256 uses AES-256 for the IVs.
pub fn essiv_cipher(cipher: QCryptoCipherAlgo, hash: QCryptoHashAlgo) -> Result<QCryptoCipherAlgo> {
    use QCryptoCipherAlgo as C;
    let digestlen = hash_digest_len(hash);
    if digestlen == cipher_get_key_len(cipher) {
        return Ok(cipher);
    }
    let (family, name) = match cipher {
        C::Aes128 | C::Aes192 | C::Aes256 => ([C::Aes128, C::Aes192, C::Aes256], "AES"),
        C::Serpent128 | C::Serpent192 | C::Serpent256 => {
            ([C::Serpent128, C::Serpent192, C::Serpent256], "Serpent")
        }
        C::Twofish128 | C::Twofish192 | C::Twofish256 => {
            ([C::Twofish128, C::Twofish192, C::Twofish256], "Twofish")
        }
        _ => {
            return Err(Error::generic(format!(
                "Cipher {} not supported with essiv",
                cipher.as_str()
            )));
        }
    };
    family.into_iter().find(|&c| cipher_get_key_len(c) == digestlen).ok_or_else(|| {
        Error::generic(format!("No {name} cipher with key size {digestlen} available"))
    })
}

fn erange(msg: String) -> Error {
    #[cfg(unix)]
    {
        Error::from_io(msg, std::io::Error::from_raw_os_error(34))
    }
    #[cfg(not(unix))]
    {
        Error::generic(format!("{msg}: Numerical result out of range"))
    }
}

/// `qcrypto_block_luks_check_header()`.
fn check_header(luks: &Luks, flags: u32) -> Result<()> {
    let h = &luks.header;
    let header_sectors = (KEY_SLOT_OFFSET / SECTOR_SIZE) as u32;
    let detached = flags & QCRYPTO_BLOCK_OPEN_DETACHED != 0;
    let err = |m: String| Err(Error::generic(m));

    if h.magic != MAGIC {
        return err("Volume is not in LUKS format".into());
    }
    if h.version != VERSION {
        return err(format!("LUKS version {} is not supported", h.version));
    }
    if !h.cipher_name.contains(&0) {
        return err("LUKS header cipher name is not NUL terminated".into());
    }
    if !h.cipher_mode.contains(&0) {
        return err("LUKS header cipher mode is not NUL terminated".into());
    }
    if !h.hash_spec.contains(&0) {
        return err("LUKS header hash spec is not NUL terminated".into());
    }
    if !detached && h.payload_offset_sector < header_sectors {
        return err("LUKS payload is overlapping with the header".into());
    }
    if h.master_key_iterations == 0 {
        return err("LUKS key iteration count is zero".into());
    }
    for i in 0..NUM_KEY_SLOTS {
        let s1 = &h.key_slots[i];
        let start1 = s1.key_offset_sector;
        let len1 = luks.splitkeylen_sectors(header_sectors, s1.stripes);
        if s1.stripes != STRIPES {
            return err(format!(
                "Keyslot {i} is corrupted (stripes {} != {})",
                s1.stripes as i32, STRIPES
            ));
        }
        if s1.active != KEY_SLOT_DISABLED && s1.active != KEY_SLOT_ENABLED {
            return err(format!("Keyslot {i} state (active/disable) is corrupted"));
        }
        if s1.active == KEY_SLOT_ENABLED && s1.iterations == 0 {
            return err(format!("Keyslot {i} iteration count is zero"));
        }
        if start1 < header_sectors {
            return err(format!("Keyslot {i} is overlapping with the LUKS header"));
        }
        if !detached && start1.wrapping_add(len1) > h.payload_offset_sector {
            return err(format!("Keyslot {i} is overlapping with the encrypted payload"));
        }
        for j in i + 1..NUM_KEY_SLOTS {
            let s2 = &h.key_slots[j];
            let start2 = s2.key_offset_sector;
            let len2 = luks.splitkeylen_sectors(header_sectors, s2.stripes);
            if ranges_overlap(start1, len1, start2, len2) {
                return err(format!("Keyslots {i} and {j} are overlapping in the header"));
            }
        }
    }
    Ok(())
}

/// `ranges_overlap()` from `qemu/range.h`.
fn ranges_overlap(first1: u32, len1: u32, first2: u32, len2: u32) -> bool {
    let last1 = u64::from(first1) + u64::from(len1) - 1;
    let last2 = u64::from(first2) + u64::from(len2) - 1;
    !(last2 < u64::from(first1) || last1 < u64::from(first2))
}

/// `qcrypto_block_luks_parse_header()`: turns the header strings into algorithms.
fn parse_header(luks: &mut Luks) -> Result<()> {
    let full_mode = cstr(&luks.header.cipher_mode);
    let Some((mode_name, rest)) = full_mode.split_once('-') else {
        return Err(Error::generic(format!("Unexpected cipher mode string format '{full_mode}'")));
    };
    let (ivgen_name, ivhash_name) = match rest.split_once(':') {
        Some((a, b)) => (a, Some(b)),
        None => (rest, None),
    };
    luks.ivgen_hash_alg = match ivhash_name {
        None => QCryptoHashAlgo::Md5,
        Some(n) => name_lookup(n, QCryptoHashAlgo::from_name, "Hash algorithm")?,
    };
    luks.cipher_mode = name_lookup(mode_name, QCryptoCipherMode::from_name, "Cipher mode")?;
    luks.cipher_alg = cipher_name_lookup(
        &cstr(&luks.header.cipher_name),
        luks.cipher_mode,
        luks.header.master_key_len,
    )?;
    luks.hash_alg =
        name_lookup(&cstr(&luks.header.hash_spec), QCryptoHashAlgo::from_name, "Hash algorithm")?;
    luks.ivgen_alg = name_lookup(ivgen_name, QCryptoIVGenAlgo::from_name, "IV generator")?;
    if luks.ivgen_alg == QCryptoIVGenAlgo::Essiv {
        if ivhash_name.is_none() {
            return Err(Error::generic("Missing IV generator hash specification"));
        }
        luks.ivgen_cipher_alg = essiv_cipher(luks.cipher_alg, luks.ivgen_hash_alg)?;
    } else {
        // A hash given with plain or plain64 is parsed but ignored, as dm-crypt does.
        luks.ivgen_cipher_alg = luks.cipher_alg;
    }
    Ok(())
}

fn store_header(luks: &Luks, io: &mut dyn QCryptoBlockIo) -> Result<()> {
    io.write(0, &luks.header.to_bytes())
}

fn load_header(io: &mut dyn QCryptoBlockIo) -> Result<Header> {
    let mut buf = vec![0u8; HEADER_LEN];
    io.read(0, &mut buf)?;
    Ok(Header::from_bytes(&buf))
}

/// `qcrypto_block_luks_store_key()`: puts `masterkey` into keyslot `slot_idx`, protected by
/// `password`, and writes out the key material and the header.
fn store_key(
    luks: &mut Luks,
    niv: usize,
    slot_idx: usize,
    password: &str,
    masterkey: &[u8],
    iter_time: u64,
    io: &mut dyn QCryptoBlockIo,
) -> Result<()> {
    let mklen = luks.mklen();
    let mut salt = [0u8; SALT_LEN];
    random_bytes(&mut salt)?;
    luks.header.key_slots[slot_idx].salt = salt;
    let stripes = luks.header.key_slots[slot_idx].stripes;
    let splitkeylen = mklen * stripes as usize;

    // How many iterations take one second of CPU time with this password.
    let iters = pbkdf2_count_iters(luks.hash_alg, password.as_bytes(), &salt, mklen)?;
    let iters = scale_iters(iters, iter_time, 1)?;
    let iterations = iters.max(MIN_SLOT_KEY_ITERS) as u32;
    luks.header.key_slots[slot_idx].iterations = iterations;

    // The key that encrypts the master key, derived from the password.
    let mut slotkey = vec![0u8; mklen];
    let mut splitkey = vec![0u8; splitkeylen];
    let r = (|| {
        pbkdf2(luks.hash_alg, password.as_bytes(), &salt, u64::from(iterations), &mut slotkey)?;
        let mut cipher = Cipher::new(luks.cipher_alg, luks.cipher_mode, &slotkey)?;
        let ivgen =
            IvGen::new(luks.ivgen_alg, luks.ivgen_cipher_alg, luks.ivgen_hash_alg, &slotkey)?;
        // Spread the master key over many stripes to defeat forensic recovery.
        afsplit_encode(luks.hash_alg, mklen, stripes, masterkey, &mut splitkey)?;
        cipher_encdec(
            &mut cipher,
            niv,
            Some(&ivgen),
            SECTOR_SIZE as usize,
            0,
            &mut splitkey,
            true,
        )?;
        let off = u64::from(luks.header.key_slots[slot_idx].key_offset_sector) * SECTOR_SIZE;
        io.write(off, &splitkey)
    })();
    slotkey.fill(0);
    splitkey.fill(0);
    r?;
    luks.header.key_slots[slot_idx].active = KEY_SLOT_ENABLED;
    store_header(luks, io)
}

/// Scales an iterations-per-second rate to `iter_time` milliseconds, then divides by `div`,
/// with QEMU's overflow errors.
fn scale_iters(iters: u64, iter_time: u64, div: u64) -> Result<u64> {
    if iter_time != 0 && iters > u64::MAX / iter_time {
        return Err(erange(format!("PBKDF iterations {iters} too large to scale")));
    }
    // iter_time is in milliseconds but the rate is per second.
    let scaled = iters * iter_time / 1000 / div;
    if scaled > u64::from(u32::MAX) {
        return Err(erange(format!("PBKDF iterations {scaled} larger than {}", u32::MAX)));
    }
    Ok(scaled)
}

/// `qcrypto_block_luks_load_key()`: tries `password` on keyslot `slot_idx`. Returns whether it
/// unlocked the slot, in which case `masterkey` holds the master key.
fn load_key(
    luks: &Luks,
    slot_idx: usize,
    password: &str,
    masterkey: &mut [u8],
    io: &mut dyn QCryptoBlockIo,
) -> Result<bool> {
    let slot = &luks.header.key_slots[slot_idx];
    if slot.active != KEY_SLOT_ENABLED {
        return Ok(false);
    }
    let mklen = luks.mklen();
    let splitkeylen = mklen * slot.stripes as usize;
    let mut splitkey = vec![0u8; splitkeylen];
    let mut possiblekey = vec![0u8; mklen];
    let r = (|| {
        // Derive a candidate key from the password; the digest check below tells whether it
        // was right.
        pbkdf2(
            luks.hash_alg,
            password.as_bytes(),
            &slot.salt,
            u64::from(slot.iterations),
            &mut possiblekey,
        )?;
        io.read(u64::from(slot.key_offset_sector) * SECTOR_SIZE, &mut splitkey)?;
        let mut cipher = Cipher::new(luks.cipher_alg, luks.cipher_mode, &possiblekey)?;
        let niv = cipher_get_iv_len(luks.cipher_alg, luks.cipher_mode);
        let ivgen =
            IvGen::new(luks.ivgen_alg, luks.ivgen_cipher_alg, luks.ivgen_hash_alg, &possiblekey)?;
        cipher_encdec(
            &mut cipher,
            niv,
            Some(&ivgen),
            SECTOR_SIZE as usize,
            0,
            &mut splitkey,
            false,
        )?;
        afsplit_decode(luks.hash_alg, mklen, slot.stripes, &splitkey, masterkey)?;
        let mut keydigest = [0u8; DIGEST_LEN];
        pbkdf2(
            luks.hash_alg,
            masterkey,
            &luks.header.master_key_salt,
            u64::from(luks.header.master_key_iterations),
            &mut keydigest,
        )?;
        Ok(keydigest == luks.header.master_key_digest)
    })();
    splitkey.fill(0);
    possiblekey.fill(0);
    r
}

/// `qcrypto_block_luks_find_key()`: tries `password` on every keyslot.
fn find_key(
    luks: &Luks,
    password: &str,
    masterkey: &mut [u8],
    io: &mut dyn QCryptoBlockIo,
) -> Result<()> {
    for i in 0..NUM_KEY_SLOTS {
        if load_key(luks, i, password, masterkey, io)? {
            return Ok(());
        }
    }
    Err(Error::generic("Invalid password, cannot unlock any keyslot"))
}

/// `qcrypto_block_luks_erase_key()`: disables keyslot `slot_idx` and overwrites its key
/// material with random data several times.
fn erase_key(luks: &mut Luks, slot_idx: usize, io: &mut dyn QCryptoBlockIo) -> Result<()> {
    let slot = &mut luks.header.key_slots[slot_idx];
    let splitkeylen = luks.header.master_key_len as usize * slot.stripes as usize;
    assert!(splitkeylen > 0);
    let mut garbage = vec![0u8; splitkeylen];
    slot.salt = [0; SALT_LEN];
    slot.iterations = 0;
    slot.active = KEY_SLOT_DISABLED;
    let off = u64::from(slot.key_offset_sector) * SECTOR_SIZE;
    let mut first_err = store_header(luks, io).err();

    // Erase the key material even if the header update failed.
    for i in 0..ERASE_ITERATIONS {
        if let Err(e) = random_bytes(&mut garbage) {
            // Without random data, still write zeros over the key material once.
            let e = first_err.take().unwrap_or(e);
            if i > 0 {
                return Err(e);
            }
            first_err = Some(e);
        }
        if let Err(e) = io.write(off, &garbage) {
            return Err(first_err.unwrap_or(e));
        }
    }
    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// `qcrypto_block_luks_open()`.
pub(super) fn open(
    block: &mut QCryptoBlock,
    options: &QCryptoBlockOptionsLUKS,
    optprefix: &str,
    io: &mut dyn QCryptoBlockIo,
    flags: u32,
) -> Result<Luks> {
    let mut password = None;
    if flags & QCRYPTO_BLOCK_OPEN_NO_IO == 0 {
        let Some(secret) = &options.key_secret else {
            return Err(Error::generic(format!(
                "Parameter '{optprefix}key-secret' is required for cipher"
            )));
        };
        password = Some(secret_lookup_as_utf8(secret)?);
    }
    let mut luks = Luks::new(load_header(io)?);
    luks.secret = options.key_secret.clone();
    check_header(&luks, flags)?;
    parse_header(&mut luks)?;

    if let Some(password) = password {
        let mut masterkey = vec![0u8; luks.mklen()];
        let r = (|| {
            find_key(&luks, &password, &mut masterkey, io)?;
            block.kdfhash = luks.hash_alg;
            block.niv = cipher_get_iv_len(luks.cipher_alg, luks.cipher_mode);
            block.ivgen = Some(IvGen::new(
                luks.ivgen_alg,
                luks.ivgen_cipher_alg,
                luks.ivgen_hash_alg,
                &masterkey,
            )?);
            block.init_cipher(luks.cipher_alg, luks.cipher_mode, &masterkey)
        })();
        masterkey.fill(0);
        r?;
    }
    block.sector_size = SECTOR_SIZE;
    block.payload_offset = u64::from(luks.header.payload_offset_sector) * SECTOR_SIZE;
    block.detached_header = block.payload_offset == 0;
    Ok(luks)
}

/// `qemu_uuid_generate()` and `qemu_uuid_unparse()`: a random version 4 UUID in text form.
fn uuid_gen() -> Result<String> {
    let mut u = [0u8; 16];
    random_bytes(&mut u)?;
    u[6] = (u[6] & 0x0F) | 0x40;
    u[8] = (u[8] & 0x3F) | 0x80;
    let mut s = String::with_capacity(36);
    for (i, b) in u.iter().enumerate() {
        if matches!(i, 4 | 6 | 8 | 10) {
            s.push('-');
        }
        s.push_str(&format!("{b:02x}"));
    }
    Ok(s)
}

/// `qcrypto_block_luks_create()`.
pub(super) fn create(
    block: &mut QCryptoBlock,
    opts: &QCryptoBlockCreateOptionsLUKS,
    optprefix: &str,
    io: &mut dyn QCryptoBlockIo,
) -> Result<Luks> {
    let iter_time = opts.iter_time.map_or(DEFAULT_ITER_TIME_MS, |t| t as u64);
    let cipher_alg = opts.cipher_alg.unwrap_or(QCryptoCipherAlgo::Aes256);
    let cipher_mode = opts.cipher_mode.unwrap_or(QCryptoCipherMode::Xts);
    let ivgen_alg = opts.ivgen_alg.unwrap_or(QCryptoIVGenAlgo::Plain64);
    let hash_alg = opts.hash_alg.unwrap_or(QCryptoHashAlgo::Sha256);
    let mut ivgen_hash = opts.ivgen_hash_alg;
    if ivgen_alg == QCryptoIVGenAlgo::Essiv && ivgen_hash.is_none() {
        ivgen_hash = Some(QCryptoHashAlgo::Sha256);
    }

    let mut luks = Luks::new(Header::default());
    luks.cipher_alg = cipher_alg;
    luks.cipher_mode = cipher_mode;
    luks.ivgen_alg = ivgen_alg;
    luks.ivgen_hash_alg = ivgen_hash.unwrap_or(QCryptoHashAlgo::Md5);
    luks.hash_alg = hash_alg;

    let Some(secret) = &opts.key_secret else {
        return Err(Error::generic(format!(
            "Parameter '{optprefix}key-secret' is required for cipher"
        )));
    };
    luks.secret = Some(secret.clone());
    let password = secret_lookup_as_utf8(secret)?;

    luks.header.magic = MAGIC;
    luks.header.version = VERSION;
    set_cstr(&mut luks.header.uuid, &uuid_gen()?);

    let cipher_name = cipher_alg_lookup(cipher_alg)?;
    let cipher_mode_spec = match ivgen_hash {
        Some(h) => format!("{}-{}:{}", cipher_mode.as_str(), ivgen_alg.as_str(), h.as_str()),
        None => format!("{}-{}", cipher_mode.as_str(), ivgen_alg.as_str()),
    };
    let hash_name = hash_alg.as_str();
    if cipher_name.len() >= CIPHER_NAME_LEN {
        return Err(Error::generic(format!(
            "Cipher name '{cipher_name}' is too long for LUKS header"
        )));
    }
    if cipher_mode_spec.len() >= CIPHER_MODE_LEN {
        return Err(Error::generic(format!(
            "Cipher mode '{cipher_mode_spec}' is too long for LUKS header"
        )));
    }
    if hash_name.len() >= HASH_SPEC_LEN {
        return Err(Error::generic(format!("Hash name '{hash_name}' is too long for LUKS header")));
    }
    luks.ivgen_cipher_alg = if ivgen_alg == QCryptoIVGenAlgo::Essiv {
        essiv_cipher(cipher_alg, luks.ivgen_hash_alg)?
    } else {
        cipher_alg
    };
    set_cstr(&mut luks.header.cipher_name, cipher_name);
    set_cstr(&mut luks.header.cipher_mode, &cipher_mode_spec);
    set_cstr(&mut luks.header.hash_spec, hash_name);

    let mut mklen = cipher_get_key_len(cipher_alg);
    if cipher_mode == QCryptoCipherMode::Xts {
        mklen *= 2;
    }
    luks.header.master_key_len = mklen as u32;
    random_bytes(&mut luks.header.master_key_salt)?;
    let mut masterkey = vec![0u8; mklen];
    let r = (|| {
        random_bytes(&mut masterkey)?;
        block.init_cipher(cipher_alg, cipher_mode, &masterkey)?;
        block.kdfhash = hash_alg;
        block.niv = cipher_get_iv_len(cipher_alg, cipher_mode);
        block.ivgen =
            Some(IvGen::new(ivgen_alg, luks.ivgen_cipher_alg, luks.ivgen_hash_alg, &masterkey)?);

        // The master key digest gets an eighth of the time budget.
        let iters =
            pbkdf2_count_iters(hash_alg, &masterkey, &luks.header.master_key_salt, DIGEST_LEN)?;
        let iters = scale_iters(iters, iter_time, 8)?;
        luks.header.master_key_iterations = iters.max(MIN_MASTER_KEY_ITERS) as u32;
        pbkdf2(
            hash_alg,
            &masterkey,
            &luks.header.master_key_salt,
            u64::from(luks.header.master_key_iterations),
            &mut luks.header.master_key_digest,
        )?;

        let header_sectors = (KEY_SLOT_OFFSET / SECTOR_SIZE) as u32;
        let split_key_sectors = luks.splitkeylen_sectors(header_sectors, STRIPES);
        for (i, slot) in luks.header.key_slots.iter_mut().enumerate() {
            slot.active = KEY_SLOT_DISABLED;
            slot.key_offset_sector = header_sectors + i as u32 * split_key_sectors;
            slot.stripes = STRIPES;
        }
        let header_end = header_sectors + NUM_KEY_SLOTS as u32 * split_key_sectors;
        luks.header.payload_offset_sector = if block.detached_header { 0 } else { header_end };
        block.sector_size = SECTOR_SIZE;
        block.payload_offset = u64::from(luks.header.payload_offset_sector) * SECTOR_SIZE;
        io.init(u64::from(header_end) * SECTOR_SIZE)?;
        store_key(&mut luks, block.niv, 0, &password, &masterkey, iter_time, io)
    })();
    masterkey.fill(0);
    if r.is_err() {
        block.params = None;
        block.free_ciphers.lock().unwrap_or_else(|e| e.into_inner()).clear();
        block.ivgen = None;
    }
    r?;
    Ok(luks)
}

fn luks_mut(block: &mut QCryptoBlock) -> &mut Luks {
    match &mut block.format {
        super::Format::Luks(l) => l,
        super::Format::Qcow => unreachable!("amend on a LUKS block"),
    }
}

/// `qcrypto_block_luks_amend_options()`.
pub(super) fn amend(
    block: &mut QCryptoBlock,
    io: &mut dyn QCryptoBlockIo,
    opts: &QCryptoBlockAmendOptionsLUKS,
    force: bool,
) -> Result<()> {
    match opts.state {
        QCryptoBlockLUKSKeyslotState::Active => amend_add_keyslot(block, io, opts, force),
        QCryptoBlockLUKSKeyslotState::Inactive => amend_erase_keyslots(block, io, opts, force),
    }
}

fn amend_add_keyslot(
    block: &mut QCryptoBlock,
    io: &mut dyn QCryptoBlockIo,
    opts: &QCryptoBlockAmendOptionsLUKS,
    force: bool,
) -> Result<()> {
    let niv = block.niv;
    let luks = luks_mut(block);
    let iter_time = opts.iter_time.map_or(DEFAULT_ITER_TIME_MS, |t| t as u64);
    let secret = opts.secret.clone().or_else(|| luks.secret.clone());

    let Some(new_secret) = &opts.new_secret else {
        return Err(Error::generic("'new-secret' is required to activate a keyslot"));
    };
    if opts.old_secret.is_some() {
        return Err(Error::generic("'old-secret' must not be given when activating keyslots"));
    }
    let keyslot = match opts.keyslot {
        Some(k) => {
            let k = k as i32;
            if k < 0 || k >= NUM_KEY_SLOTS as i32 {
                return Err(Error::generic(format!(
                    "Invalid keyslot {} specified, must be between 0 and {}",
                    k as u32,
                    NUM_KEY_SLOTS - 1
                )));
            }
            k as usize
        }
        None => luks
            .find_free_keyslot()
            .ok_or_else(|| Error::generic("Can't add a keyslot - all keyslots are in use"))?,
    };
    if !force && luks.slot_active(keyslot) {
        return Err(Error::generic(format!(
            "Refusing to overwrite active keyslot {keyslot} - please erase it first"
        )));
    }
    // QEMU passes a NULL id here when the image was opened without a secret, which fails the
    // lookup with the same message as an empty id.
    let old_password = secret_lookup_as_utf8(secret.as_deref().unwrap_or(""))?;
    let mut master_key = vec![0u8; luks.mklen()];
    let r = (|| {
        find_key(luks, &old_password, &mut master_key, io)
            .map_err(|e| e.hint("Failed to retrieve the master key"))?;
        let new_password = secret_lookup_as_utf8(new_secret)?;
        store_key(luks, niv, keyslot, &new_password, &master_key, iter_time, io)
            .map_err(|e| e.hint(format!("Failed to write to keyslot {keyslot}")))
    })();
    master_key.fill(0);
    r
}

fn amend_erase_keyslots(
    block: &mut QCryptoBlock,
    io: &mut dyn QCryptoBlockIo,
    opts: &QCryptoBlockAmendOptionsLUKS,
    force: bool,
) -> Result<()> {
    let luks = luks_mut(block);
    if opts.new_secret.is_some() {
        return Err(Error::generic("'new-secret' must not be given when erasing keyslots"));
    }
    if opts.iter_time.is_some() {
        return Err(Error::generic("'iter-time' must not be given when erasing keyslots"));
    }
    if opts.secret.is_some() {
        return Err(Error::generic("'secret' must not be given when erasing keyslots"));
    }
    let old_password = match &opts.old_secret {
        Some(s) => Some(secret_lookup_as_utf8(s)?),
        None => None,
    };
    let mut tmpkey = vec![0u8; luks.mklen()];

    if let Some(k) = opts.keyslot {
        let keyslot = k as i32;
        if keyslot < 0 || keyslot >= NUM_KEY_SLOTS as i32 {
            return Err(Error::generic(format!(
                "Invalid keyslot {keyslot} specified, must be between 0 and {}",
                NUM_KEY_SLOTS - 1
            )));
        }
        let ks = keyslot as usize;
        if let Some(pw) = &old_password {
            let found = load_key(luks, ks, pw, &mut tmpkey, io);
            tmpkey.fill(0);
            if !found? {
                return Err(Error::generic(format!(
                    "Given keyslot {keyslot} doesn't contain the given old password for erase operation"
                )));
            }
        }
        if !force && !luks.slot_active(ks) {
            return Err(Error::generic(format!(
                "Given keyslot {keyslot} is already erased (inactive) "
            )));
        }
        if !force && luks.count_active_slots() == 1 {
            return Err(Error::generic(format!(
                "Attempt to erase the only active keyslot {keyslot} which will erase all the data in the image irreversibly - refusing operation"
            )));
        }
        erase_key(luks, ks, io).map_err(|e| e.hint(format!("Failed to erase keyslot {keyslot}")))
    } else if let Some(pw) = &old_password {
        let mut to_erase = Vec::new();
        for i in 0..NUM_KEY_SLOTS {
            let found = load_key(luks, i, pw, &mut tmpkey, io);
            tmpkey.fill(0);
            if found? {
                to_erase.push(i);
            }
        }
        if to_erase.is_empty() {
            return Err(Error::generic(
                "No keyslots match given (old) password for erase operation",
            ));
        }
        if !force && to_erase.len() == luks.count_active_slots() {
            return Err(Error::generic(
                "All the active keyslots match the (old) password that was given and erasing them will erase all the data in the image irreversibly - refusing operation",
            ));
        }
        for i in to_erase {
            erase_key(luks, i, io).map_err(|e| e.hint(format!("Failed to erase keyslot {i}")))?;
        }
        Ok(())
    } else {
        Err(Error::generic(
            "To erase keyslot(s), either explicit keyslot index or the password currently contained in them must be given",
        ))
    }
}

/// `qcrypto_block_luks_get_info()`.
pub(super) fn get_info(block: &QCryptoBlock, luks: &Luks) -> QCryptoBlockInfoLUKS {
    let slots = luks
        .header
        .key_slots
        .iter()
        .map(|s| {
            let active = s.active == KEY_SLOT_ENABLED;
            QCryptoBlockInfoLUKSSlot {
                active,
                iters: active.then_some(i64::from(s.iterations)),
                stripes: active.then_some(i64::from(s.stripes)),
                key_offset: i64::from(s.key_offset_sector) * SECTOR_SIZE as i64,
            }
        })
        .collect();
    QCryptoBlockInfoLUKS {
        cipher_alg: luks.cipher_alg,
        cipher_mode: luks.cipher_mode,
        ivgen_alg: luks.ivgen_alg,
        ivgen_hash_alg: (luks.ivgen_alg == QCryptoIVGenAlgo::Essiv).then_some(luks.ivgen_hash_alg),
        hash_alg: luks.hash_alg,
        detached_header: block.detached_header,
        payload_offset: block.payload_offset as i64,
        master_key_iters: i64::from(luks.header.master_key_iterations),
        uuid: cstr(&luks.header.uuid),
        slots,
    }
}

impl QCryptoBlock {
    /// The parsed LUKS header, if this is a LUKS block.
    pub fn luks(&self) -> Option<&Luks> {
        match &self.format {
            super::Format::Luks(l) => Some(l),
            super::Format::Qcow => None,
        }
    }
}
