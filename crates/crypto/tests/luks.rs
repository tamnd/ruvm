// SPDX-License-Identifier: GPL-2.0-or-later

//! The LUKS1 and legacy qcow `QCryptoBlock` formats over an in-memory header, modelled on QEMU's
//! tests/unit/test-crypto-block.c, plus the amend rules and header corruption checks.

use std::sync::Once;

use ruvm_base::Result;
use ruvm_crypto::block::luks::{self, HEADER_LEN, Header, KEY_SLOT_DISABLED};
use ruvm_crypto::block::{
    QCRYPTO_BLOCK_CREATE_DETACHED, QCRYPTO_BLOCK_OPEN_DETACHED, QCRYPTO_BLOCK_OPEN_NO_IO,
    QCryptoBlock, QCryptoBlockIo, has_format,
};
use ruvm_crypto::secret::secret_object_add;
use ruvm_qapi::types::{
    QCryptoBlockAmendOptions, QCryptoBlockAmendOptionsLUKS, QCryptoBlockAmendOptionsU,
    QCryptoBlockCreateOptions, QCryptoBlockCreateOptionsLUKS, QCryptoBlockCreateOptionsU,
    QCryptoBlockFormat, QCryptoBlockInfoU, QCryptoBlockLUKSKeyslotState, QCryptoBlockOpenOptions,
    QCryptoBlockOpenOptionsU, QCryptoBlockOptionsLUKS, QCryptoBlockOptionsQCow,
    QCryptoCipherAlgo as C, QCryptoCipherMode as M, QCryptoHashAlgo as H, QCryptoIVGenAlgo as I,
};
use ruvm_qom::Registry;

#[derive(Default)]
struct Mem {
    buf: Vec<u8>,
    headerlen: u64,
}

impl QCryptoBlockIo for Mem {
    fn read(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let o = offset as usize;
        assert!(o + buf.len() <= self.buf.len(), "read past the end of the header");
        buf.copy_from_slice(&self.buf[o..o + buf.len()]);
        Ok(())
    }
    fn write(&mut self, offset: u64, buf: &[u8]) -> Result<()> {
        let o = offset as usize;
        if self.buf.len() < o + buf.len() {
            self.buf.resize(o + buf.len(), 0);
        }
        self.buf[o..o + buf.len()].copy_from_slice(buf);
        Ok(())
    }
    fn init(&mut self, headerlen: u64) -> Result<()> {
        self.headerlen = headerlen;
        self.buf.resize(headerlen as usize, 0);
        Ok(())
    }
}

fn setup() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Keep keyslots at the 1000 iteration floor so the tests stay fast.
        ruvm_crypto::pbkdf::set_iters_per_second_override(Some(1000));
        let r = Registry::global();
        secret_object_add(r, "secret,id=sec0,data=123456").unwrap();
        secret_object_add(r, "secret,id=sec1,data=654321").unwrap();
        secret_object_add(r, "secret,id=sec2,data=abcdef").unwrap();
    });
}

fn msg<T>(r: Result<T>) -> String {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => e.message().to_string(),
    }
}

fn luks_create_opts(secret: &str) -> QCryptoBlockCreateOptionsLUKS {
    QCryptoBlockCreateOptionsLUKS {
        key_secret: Some(secret.into()),
        iter_time: Some(10),
        ..Default::default()
    }
}

fn create(o: QCryptoBlockCreateOptionsLUKS, io: &mut Mem, flags: u32) -> Result<QCryptoBlock> {
    QCryptoBlock::create(
        &QCryptoBlockCreateOptions { u: QCryptoBlockCreateOptionsU::Luks(o) },
        None,
        io,
        flags,
    )
}

fn open(secret: Option<&str>, io: &mut Mem, flags: u32) -> Result<QCryptoBlock> {
    let o = QCryptoBlockOptionsLUKS { key_secret: secret.map(String::from) };
    QCryptoBlock::open(
        &QCryptoBlockOpenOptions { u: QCryptoBlockOpenOptionsU::Luks(o) },
        None,
        io,
        flags,
    )
}

fn amend(
    b: &mut QCryptoBlock,
    io: &mut Mem,
    o: QCryptoBlockAmendOptionsLUKS,
    force: bool,
) -> Result<()> {
    b.amend_options(io, &QCryptoBlockAmendOptions { u: QCryptoBlockAmendOptionsU::Luks(o) }, force)
}

fn add(new: &str, slot: Option<i64>) -> QCryptoBlockAmendOptionsLUKS {
    QCryptoBlockAmendOptionsLUKS {
        state: QCryptoBlockLUKSKeyslotState::Active,
        new_secret: Some(new.into()),
        keyslot: slot,
        iter_time: Some(10),
        ..Default::default()
    }
}

fn erase(old: Option<&str>, slot: Option<i64>) -> QCryptoBlockAmendOptionsLUKS {
    QCryptoBlockAmendOptionsLUKS {
        state: QCryptoBlockLUKSKeyslotState::Inactive,
        old_secret: old.map(String::from),
        keyslot: slot,
        ..Default::default()
    }
}

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 7 + i / 512) as u8).collect()
}

/// Encrypting with one handle and decrypting with a freshly opened one gives back the data, and
/// outside ECB mode the ciphertext depends on the sector (so the IV generator is in use).
fn roundtrip(o: QCryptoBlockCreateOptionsLUKS) {
    let ecb = o.cipher_mode == Some(M::Ecb);
    let mut io = Mem::default();
    let b = create(o, &mut io, 0).unwrap();
    assert!(has_format(QCryptoBlockFormat::Luks, &io.buf));
    assert_eq!(b.payload_offset(), io.headerlen);
    assert_eq!(b.sector_size(), 512);
    let plain = pattern(4096);
    let mut data = plain.clone();
    b.encrypt(1 << 20, &mut data).unwrap();
    assert_ne!(data, plain);
    let mut again = plain.clone();
    b.encrypt(2 << 20, &mut again).unwrap();
    assert_eq!(data == again, ecb);

    let b2 = open(Some("sec0"), &mut io, 0).unwrap();
    b2.decrypt(1 << 20, &mut data).unwrap();
    assert_eq!(data, plain);
    assert_eq!(msg(open(Some("sec1"), &mut io, 0)), "Invalid password, cannot unlock any keyslot");
}

#[test]
fn luks_default() {
    setup();
    let mut io = Mem::default();
    let b = create(luks_create_opts("sec0"), &mut io, 0).unwrap();
    // aes-256-xts-plain64, sha256: 8 keyslots of 4000 stripes x 64 bytes, each rounded up to
    // 4 KiB, after 4 KiB of header. qemu-img info reports the same 2068480.
    assert_eq!(b.payload_offset(), 2068480);
    let h = Header::from_bytes(&io.buf[..HEADER_LEN]);
    assert_eq!(h.version, 1);
    assert_eq!(&h.cipher_name[..4], b"aes\0");
    assert_eq!(&h.cipher_mode[..12], b"xts-plain64\0");
    assert_eq!(&h.hash_spec[..7], b"sha256\0");
    assert_eq!(h.master_key_len, 64);
    assert_eq!(h.payload_offset_sector, 4040);
    assert_eq!(h.key_slots[0].iterations, 1000);
    assert_eq!(h.master_key_iterations, 1000);
    for s in &h.key_slots[1..] {
        assert_eq!(s.active, KEY_SLOT_DISABLED);
    }
    roundtrip(luks_create_opts("sec0"));
}

#[test]
fn luks_combinations() {
    setup();
    let cases: &[(C, M, I, Option<H>, H)] = &[
        (C::Aes128, M::Cbc, I::Plain, None, H::Sha1),
        (C::Aes256, M::Cbc, I::Plain64, None, H::Sha256),
        (C::Aes256, M::Cbc, I::Essiv, Some(H::Sha256), H::Sha1),
        (C::Aes128, M::Cbc, I::Essiv, Some(H::Sha256), H::Sha1),
        (C::Aes192, M::Xts, I::Plain64, None, H::Sha512),
        (C::Serpent256, M::Xts, I::Plain64, None, H::Sha256),
        (C::Twofish128, M::Cbc, I::Essiv, Some(H::Sha256), H::Ripemd160),
        (C::Cast5_128, M::Cbc, I::Plain64, None, H::Sha1),
        (C::Aes256, M::Ctr, I::Plain64, None, H::Sha384),
        (C::Aes256, M::Ecb, I::Plain64, None, H::Sha224),
        (C::Sm4, M::Ecb, I::Plain64, None, H::Sm3),
    ];
    for &(alg, mode, ivgen, ivhash, hash) in cases {
        let o = QCryptoBlockCreateOptionsLUKS {
            cipher_alg: Some(alg),
            cipher_mode: Some(mode),
            ivgen_alg: Some(ivgen),
            ivgen_hash_alg: ivhash,
            hash_alg: Some(hash),
            ..luks_create_opts("sec0")
        };
        roundtrip(o);
    }
}

#[test]
fn luks_create_errors() {
    setup();
    let mut io = Mem::default();
    let o = QCryptoBlockCreateOptionsLUKS { key_secret: None, ..luks_create_opts("x") };
    assert_eq!(msg(create(o, &mut io, 0)), "Parameter 'key-secret' is required for cipher");
    let o = QCryptoBlockCreateOptionsLUKS { key_secret: None, ..luks_create_opts("x") };
    let r = QCryptoBlock::create(
        &QCryptoBlockCreateOptions { u: QCryptoBlockCreateOptionsU::Luks(o) },
        Some("encrypt."),
        &mut io,
        0,
    );
    assert_eq!(msg(r), "Parameter 'encrypt.key-secret' is required for cipher");
    assert_eq!(msg(create(luks_create_opts("nosuch"), &mut io, 0)), "No secret with id 'nosuch'");
    let o = QCryptoBlockCreateOptionsLUKS {
        cipher_alg: Some(C::Cast5_128),
        ivgen_alg: Some(I::Essiv),
        ..luks_create_opts("sec0")
    };
    assert_eq!(msg(create(o, &mut io, 0)), "Cipher cast5-128 not supported with essiv");
    let o = QCryptoBlockCreateOptionsLUKS {
        cipher_alg: Some(C::Twofish128),
        ivgen_alg: Some(I::Essiv),
        ivgen_hash_alg: Some(H::Sha1),
        ..luks_create_opts("sec0")
    };
    assert_eq!(msg(create(o, &mut io, 0)), "No Twofish cipher with key size 20 available");
}

#[test]
fn luks_open_errors() {
    setup();
    let mut io = Mem::default();
    create(luks_create_opts("sec0"), &mut io, 0).unwrap();
    assert_eq!(msg(open(None, &mut io, 0)), "Parameter 'key-secret' is required for cipher");

    // NO_IO needs no secret and only parses the header.
    let b = open(None, &mut io, QCRYPTO_BLOCK_OPEN_NO_IO).unwrap();
    assert_eq!(b.payload_offset(), 2068480);

    let pristine = io.buf.clone();
    let corrupt = |f: &dyn Fn(&mut Header)| {
        let mut h = Header::from_bytes(&pristine[..HEADER_LEN]);
        f(&mut h);
        let mut io = Mem { buf: pristine.clone(), headerlen: 0 };
        io.buf[..HEADER_LEN].copy_from_slice(&h.to_bytes());
        msg(open(None, &mut io, QCRYPTO_BLOCK_OPEN_NO_IO))
    };
    assert_eq!(corrupt(&|h| h.magic[0] = b'X'), "Volume is not in LUKS format");
    assert_eq!(corrupt(&|h| h.version = 2), "LUKS version 2 is not supported");
    assert_eq!(
        corrupt(&|h| h.cipher_name = [b'a'; 32]),
        "LUKS header cipher name is not NUL terminated"
    );
    assert_eq!(
        corrupt(&|h| h.payload_offset_sector = 1),
        "LUKS payload is overlapping with the header"
    );
    assert_eq!(corrupt(&|h| h.master_key_iterations = 0), "LUKS key iteration count is zero");
    assert_eq!(
        corrupt(&|h| h.key_slots[3].stripes = 5),
        "Keyslot 3 is corrupted (stripes 5 != 4000)"
    );
    assert_eq!(
        corrupt(&|h| h.key_slots[2].active = 1),
        "Keyslot 2 state (active/disable) is corrupted"
    );
    assert_eq!(corrupt(&|h| h.key_slots[0].iterations = 0), "Keyslot 0 iteration count is zero");
    assert_eq!(
        corrupt(&|h| h.key_slots[0].key_offset_sector = 1),
        "Keyslot 0 is overlapping with the LUKS header"
    );
    assert_eq!(
        corrupt(&|h| h.key_slots[7].key_offset_sector = 4000),
        "Keyslot 7 is overlapping with the encrypted payload"
    );
    assert_eq!(
        corrupt(&|h| h.key_slots[5].key_offset_sector = h.key_slots[4].key_offset_sector),
        "Keyslots 4 and 5 are overlapping in the header"
    );
    assert_eq!(
        corrupt(&|h| h.cipher_name[..4].copy_from_slice(b"foo\0")),
        "Algorithm 'foo' with key size 32 bytes not supported"
    );
}

#[test]
fn luks_detached() {
    setup();
    let mut io = Mem::default();
    let b = create(luks_create_opts("sec0"), &mut io, QCRYPTO_BLOCK_CREATE_DETACHED).unwrap();
    assert!(b.detached_header());
    assert_eq!(b.payload_offset(), 0);
    assert_eq!(io.headerlen, 2068480);
    let b2 = open(Some("sec0"), &mut io, QCRYPTO_BLOCK_OPEN_DETACHED).unwrap();
    assert!(b2.detached_header());
    let QCryptoBlockInfoU::Luks(info) = b2.get_info().unwrap().u else { panic!() };
    assert!(info.detached_header);
    assert_eq!(info.payload_offset, 0);
}

#[test]
fn luks_payload_offset() {
    setup();
    let o =
        QCryptoBlockCreateOptions { u: QCryptoBlockCreateOptionsU::Luks(luks_create_opts("sec0")) };
    assert_eq!(QCryptoBlock::calculate_payload_offset(&o, None).unwrap(), 2068480);
    let o = QCryptoBlockCreateOptions {
        u: QCryptoBlockCreateOptionsU::Luks(QCryptoBlockCreateOptionsLUKS {
            cipher_alg: Some(C::Aes128),
            cipher_mode: Some(M::Cbc),
            ..luks_create_opts("sec0")
        }),
    };
    // 16 byte keys: 8 slots of 4000 x 16 bytes, each rounded to 4 KiB, after 4 KiB of header.
    assert_eq!(QCryptoBlock::calculate_payload_offset(&o, None).unwrap(), 4096 + 8 * 65536);
}

#[test]
fn luks_info() {
    setup();
    let mut io = Mem::default();
    let o = QCryptoBlockCreateOptionsLUKS {
        cipher_mode: Some(M::Cbc),
        ivgen_alg: Some(I::Essiv),
        ..luks_create_opts("sec0")
    };
    create(o, &mut io, 0).unwrap();
    let b = open(Some("sec0"), &mut io, 0).unwrap();
    let QCryptoBlockInfoU::Luks(info) = b.get_info().unwrap().u else { panic!() };
    assert_eq!(info.cipher_alg, C::Aes256);
    assert_eq!(info.cipher_mode, M::Cbc);
    assert_eq!(info.ivgen_alg, I::Essiv);
    assert_eq!(info.ivgen_hash_alg, Some(H::Sha256));
    assert_eq!(info.hash_alg, H::Sha256);
    assert!(!info.detached_header);
    assert_eq!(info.payload_offset, 4096 + 8 * 128 * 1024);
    assert_eq!(info.uuid.len(), 36);
    assert_eq!(info.slots.len(), 8);
    assert!(info.slots[0].active);
    assert_eq!(info.slots[0].iters, Some(1000));
    assert_eq!(info.slots[0].stripes, Some(4000));
    assert_eq!(info.slots[0].key_offset, 4096);
    assert!(!info.slots[1].active);
    assert_eq!(info.slots[1].iters, None);
    assert_eq!(info.slots[1].key_offset, 4096 + 128 * 1024);
    assert!(b.luks().is_some());
    assert_eq!(b.kdf_hash(), H::Sha256);
}

#[test]
fn luks_amend() {
    setup();
    let mut io = Mem::default();
    create(luks_create_opts("sec0"), &mut io, 0).unwrap();
    let mut b = open(Some("sec0"), &mut io, 0).unwrap();

    // Adding keyslots.
    amend(&mut b, &mut io, add("sec1", None), false).unwrap();
    assert!(open(Some("sec1"), &mut io, 0).is_ok());
    let o = QCryptoBlockAmendOptionsLUKS { new_secret: None, ..add("sec1", None) };
    assert_eq!(
        msg(amend(&mut b, &mut io, o, false)),
        "'new-secret' is required to activate a keyslot"
    );
    let o = QCryptoBlockAmendOptionsLUKS { old_secret: Some("sec0".into()), ..add("sec1", None) };
    assert_eq!(
        msg(amend(&mut b, &mut io, o, false)),
        "'old-secret' must not be given when activating keyslots"
    );
    assert_eq!(
        msg(amend(&mut b, &mut io, add("sec2", Some(8)), false)),
        "Invalid keyslot 8 specified, must be between 0 and 7"
    );
    assert_eq!(
        msg(amend(&mut b, &mut io, add("sec2", Some(-1)), false)),
        "Invalid keyslot 4294967295 specified, must be between 0 and 7"
    );
    assert_eq!(
        msg(amend(&mut b, &mut io, add("sec2", Some(1)), false)),
        "Refusing to overwrite active keyslot 1 - please erase it first"
    );
    let o = QCryptoBlockAmendOptionsLUKS { secret: Some("sec2".into()), ..add("sec2", Some(5)) };
    let e = amend(&mut b, &mut io, o, false).unwrap_err();
    assert_eq!(e.message(), "Invalid password, cannot unlock any keyslot");
    assert_eq!(e.hint_text(), Some("Failed to retrieve the master key"));
    amend(&mut b, &mut io, add("sec2", Some(5)), false).unwrap();
    // Forcing overwrites an active slot.
    amend(&mut b, &mut io, add("sec2", Some(1)), true).unwrap();
    assert_eq!(msg(open(Some("sec1"), &mut io, 0)), "Invalid password, cannot unlock any keyslot");
    // Slots 0 (sec0), 1 and 5 (sec2) are active now.

    // Erasing keyslots.
    let o =
        QCryptoBlockAmendOptionsLUKS { new_secret: Some("sec1".into()), ..erase(None, Some(1)) };
    assert_eq!(
        msg(amend(&mut b, &mut io, o, false)),
        "'new-secret' must not be given when erasing keyslots"
    );
    let o = QCryptoBlockAmendOptionsLUKS { iter_time: Some(1), ..erase(None, Some(1)) };
    assert_eq!(
        msg(amend(&mut b, &mut io, o, false)),
        "'iter-time' must not be given when erasing keyslots"
    );
    let o = QCryptoBlockAmendOptionsLUKS { secret: Some("sec0".into()), ..erase(None, Some(1)) };
    assert_eq!(
        msg(amend(&mut b, &mut io, o, false)),
        "'secret' must not be given when erasing keyslots"
    );
    assert_eq!(
        msg(amend(&mut b, &mut io, erase(None, None), false)),
        "To erase keyslot(s), either explicit keyslot index or the password currently contained in them must be given"
    );
    assert_eq!(
        msg(amend(&mut b, &mut io, erase(None, Some(9)), false)),
        "Invalid keyslot 9 specified, must be between 0 and 7"
    );
    assert_eq!(
        msg(amend(&mut b, &mut io, erase(None, Some(-2)), false)),
        "Invalid keyslot -2 specified, must be between 0 and 7"
    );
    assert_eq!(
        msg(amend(&mut b, &mut io, erase(None, Some(3)), false)),
        "Given keyslot 3 is already erased (inactive) "
    );
    assert_eq!(
        msg(amend(&mut b, &mut io, erase(Some("sec0"), Some(1)), false)),
        "Given keyslot 1 doesn't contain the given old password for erase operation"
    );
    assert_eq!(
        msg(amend(&mut b, &mut io, erase(Some("sec1"), None), false)),
        "No keyslots match given (old) password for erase operation"
    );
    // Erase both sec2 slots by password.
    amend(&mut b, &mut io, erase(Some("sec2"), None), false).unwrap();
    assert_eq!(msg(open(Some("sec2"), &mut io, 0)), "Invalid password, cannot unlock any keyslot");
    let QCryptoBlockInfoU::Luks(info) = b.get_info().unwrap().u else { panic!() };
    let active: Vec<bool> = info.slots.iter().map(|s| s.active).collect();
    assert_eq!(active, [true, false, false, false, false, false, false, false]);
    // An erased slot is filled with random data, not left as it was.
    assert_eq!(
        msg(amend(&mut b, &mut io, erase(None, Some(0)), false)),
        "Attempt to erase the only active keyslot 0 which will erase all the data in the image irreversibly - refusing operation"
    );
    assert_eq!(
        msg(amend(&mut b, &mut io, erase(Some("sec0"), None), false)),
        "All the active keyslots match the (old) password that was given and erasing them will erase all the data in the image irreversibly - refusing operation"
    );
    amend(&mut b, &mut io, add("sec1", Some(7)), false).unwrap();
    amend(&mut b, &mut io, erase(None, Some(0)), false).unwrap();
    assert!(open(Some("sec1"), &mut io, 0).is_ok());
    assert_eq!(msg(open(Some("sec0"), &mut io, 0)), "Invalid password, cannot unlock any keyslot");
    // Forced erase of the last slot is allowed.
    amend(&mut b, &mut io, erase(None, Some(7)), true).unwrap();
    assert_eq!(msg(open(Some("sec1"), &mut io, 0)), "Invalid password, cannot unlock any keyslot");
}

#[test]
fn qcow_legacy() {
    setup();
    let mut io = Mem::default();
    let opts = |s: Option<&str>| QCryptoBlockOpenOptions {
        u: QCryptoBlockOpenOptionsU::Qcow(QCryptoBlockOptionsQCow {
            key_secret: s.map(String::from),
        }),
    };
    let r = QCryptoBlock::open(&opts(None), Some("encrypt."), &mut io, 0);
    assert_eq!(msg(r), "Parameter 'encrypt.key-secret' is required for cipher");
    let b = QCryptoBlock::open(&opts(None), None, &mut io, QCRYPTO_BLOCK_OPEN_NO_IO).unwrap();
    assert_eq!(b.sector_size(), 512);
    assert_eq!(b.payload_offset(), 0);
    assert!(matches!(b.get_info().unwrap().u, QCryptoBlockInfoU::Qcow));
    assert!(!has_format(QCryptoBlockFormat::Qcow, &[0; 512]));

    let b = QCryptoBlock::open(&opts(Some("sec0")), None, &mut io, 0).unwrap();
    // AES-128-CBC with a plain64 IV and the password "123456" zero padded to 16 bytes as the key.
    let mut data = [0u8; 512];
    b.encrypt(512 * 3, &mut data).unwrap();
    let key: [u8; 16] = *b"123456\0\0\0\0\0\0\0\0\0\0";
    let mut iv = [0u8; 16];
    iv[0] = 3;
    let mut c = ruvm_crypto::cipher::Cipher::new(C::Aes128, M::Cbc, &key).unwrap();
    let mut expect = [0u8; 512];
    c.setiv(&iv).unwrap();
    c.encrypt(&mut expect).unwrap();
    assert_eq!(data, expect);
    b.decrypt(512 * 3, &mut data).unwrap();
    assert_eq!(data, [0u8; 512]);

    let r = b_amend_qcow(
        &mut QCryptoBlock::open(&opts(Some("sec0")), None, &mut io, 0).unwrap(),
        &mut io,
    );
    assert_eq!(msg(r), "Crypto format qcow doesn't support format options amendment");
}

fn b_amend_qcow(b: &mut QCryptoBlock, io: &mut Mem) -> Result<()> {
    b.amend_options(io, &QCryptoBlockAmendOptions { u: QCryptoBlockAmendOptionsU::Qcow }, false)
}

#[test]
fn luks_has_format() {
    let mut buf = [0u8; 8];
    buf[..6].copy_from_slice(&luks::MAGIC);
    buf[7] = 1;
    assert!(has_format(QCryptoBlockFormat::Luks, &buf));
    assert!(!has_format(QCryptoBlockFormat::Luks, &buf[..7]));
    buf[7] = 2;
    assert!(!has_format(QCryptoBlockFormat::Luks, &buf));
}
