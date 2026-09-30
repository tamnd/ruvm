// SPDX-License-Identifier: GPL-2.0-or-later

//! The `luks` format driver from block/crypto.c: a LUKS1 volume whose payload is encrypted
//! sector by sector with the master key the header protects.
//!
//! The header, key slots and ciphers are ruvm-crypto's [`QCryptoBlock`]; this driver reads the
//! header through the `file` child (or the `header` child for a detached header), then maps
//! every guest request to the payload that follows the header, bouncing it through a buffer of
//! at most 1 MiB that is decrypted after reading and encrypted before writing. Requests are
//! aligned to the 512 byte encryption sector by the generic layer, which pads unaligned requests
//! with a read-modify-write, just as `request_alignment` does in QEMU.
//!
//! Differences from QEMU:
//!
//! - QEMU runs `block_crypto_amend_prepare()` and `block_crypto_amend_cleanup()` as the
//!   `.bdrv_amend_pre_run` and `.bdrv_amend_clean` hooks of the amend job. Here
//!   [`Driver::amend`] does both itself around the key slot update: it sets `updating_keys`,
//!   which makes [`Driver::child_perm_for`] ask for exclusive write access to `file`, refreshes
//!   the permissions of `file`, updates the slots, then clears the flag and refreshes again.
//! - `.bdrv_amend_options` (`qemu-img amend -o`) and `.bdrv_measure` are not wired up, the
//!   block layer has no hook for them yet.
//! - A failed `qcrypto_block_encrypt()` or `qcrypto_block_decrypt()` is `EIO` as in QEMU; the
//!   message of the crypto error is dropped the same way QEMU passes a NULL `errp`.
//! - `bdrv_co_delete_file_noerr()` after a failed `qemu-img create` removes the file only for
//!   the `file` protocol, the only one here that can delete.
//! - Like QEMU, creating a volume whose key material is not a whole number of sectors (AES-192
//!   and other 24 byte keys: 24 * 4000 bytes) trips the sector alignment assertion of the
//!   cipher helper and panics, where QEMU aborts.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard};

use ruvm_base::{Error, Result, report};
use ruvm_crypto::block::{
    QCRYPTO_BLOCK_CREATE_DETACHED, QCRYPTO_BLOCK_OPEN_DETACHED, QCRYPTO_BLOCK_OPEN_NO_IO,
    QCryptoBlock, QCryptoBlockIo, has_format,
};
use ruvm_qapi::types::{
    BlockdevAmendOptions, BlockdevAmendOptionsLUKS, BlockdevAmendOptionsU,
    BlockdevCreateOptionsLUKS, BlockdevCreateOptionsU, BlockdevOptionsU, ImageInfoSpecific,
    ImageInfoSpecificLUKSWrapper, ImageInfoSpecificU, PreallocMode, QCryptoBlockAmendOptions,
    QCryptoBlockAmendOptionsLUKS, QCryptoBlockAmendOptionsU, QCryptoBlockCreateOptions,
    QCryptoBlockCreateOptionsLUKS, QCryptoBlockCreateOptionsU, QCryptoBlockFormat,
    QCryptoBlockInfoU, QCryptoBlockOpenOptions, QCryptoBlockOpenOptionsU, QCryptoBlockOptionsLUKS,
};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit, parse_option_size, qapi_bool_parse};
use ruvm_qapi::{QDict, QValue};

use crate::drivers::{DriverDef, OpenArgs};
use crate::graph::BlockGraph;
use crate::node::{
    BDRV_CHILD_IMAGE, BDRV_CHILD_METADATA, BDRV_CHILD_PRIMARY, BDRV_REQ_FUA, BlockDriverInfo,
    BlockLimits, Driver, Node, ReopenState, errno,
};
use crate::perm::{
    BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE, PermCtx, default_perms,
};

/// `bdrv_crypto_luks`.
pub(crate) static LUKS: DriverDef = DriverDef::format("luks", luks_open_node)
    .with_probe(probe)
    .with_create_opts(create_opts)
    .with_create_opts_list(&crate::tools::LUKS_CREATE_OPTS)
    .with_amend_opts_list(&crate::tools::LUKS_AMEND_OPTS)
    .with_measure(crate::tools::luks_measure)
    .with_strong_opts(&["key-secret"])
    .with_create(create);

/// `BLOCK_CRYPTO_MAX_IO_SIZE`: the bounce buffer size.
const MAX_IO_SIZE: u64 = 1024 * 1024;

/// The `-o` options of `qemu-img create -f luks` that go to the crypto layer,
/// `block_crypto_create_opts_luks` without `size`, `preallocation` and `detached-header`.
const CRYPTO_CREATE_OPTS: &[&str] = &[
    "key-secret",
    "cipher-alg",
    "cipher-mode",
    "ivgen-alg",
    "ivgen-hash-alg",
    "hash-alg",
    "iter-time",
];

/// `block_crypto_probe_luks()`.
pub(crate) fn probe(buf: &[u8], _filename: Option<&str>) -> i32 {
    if has_format(QCryptoBlockFormat::Luks, buf) { 100 } else { 0 }
}

/// The read and write callbacks QEMU hands to `qcrypto_block_open()` and
/// `qcrypto_block_amend_options()`: `block_crypto_read_func()` and `block_crypto_write_func()`
/// on the node that holds the header.
struct NodeHeaderIo<'a> {
    node: &'a Node,
}

impl QCryptoBlockIo for NodeHeaderIo<'_> {
    fn read(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.node
            .pread(offset, buf)
            .map_err(|e| Error::from_io("Could not read encryption header", e))
    }

    fn write(&mut self, offset: u64, buf: &[u8]) -> Result<()> {
        self.node
            .pwrite(offset, buf)
            .map_err(|e| Error::from_io("Could not write encryption header", e))
    }
}

/// The callbacks of `block_crypto_co_create_generic()`: `block_crypto_create_init_func()`
/// grows the file to the header plus the payload, and `block_crypto_create_write_func()`
/// writes the header.
struct CreateIo<'a> {
    node: &'a Node,
    size: u64,
    prealloc: PreallocMode,
}

impl QCryptoBlockIo for CreateIo<'_> {
    fn read(&mut self, _offset: u64, _buf: &mut [u8]) -> Result<()> {
        // Creating never reads, QEMU passes no read callback.
        Err(Error::generic("Could not read encryption header"))
    }

    fn write(&mut self, offset: u64, buf: &[u8]) -> Result<()> {
        self.node
            .pwrite(offset, buf)
            .map_err(|e| Error::from_io("Could not write encryption header", e))
    }

    fn init(&mut self, headerlen: u64) -> Result<()> {
        if self.size > i64::MAX as u64 || headerlen > i64::MAX as u64 - self.size {
            return Err(Error::generic("The requested file size is too large"));
        }
        self.node.truncate_full((self.size + headerlen) as i64, false, self.prealloc, 0)
    }
}

/// `BlockCrypto`.
pub(crate) struct LuksDriver {
    /// Requests take the lock shared, amend takes it exclusively to rewrite the key slots.
    block: RwLock<QCryptoBlock>,
    /// `qcrypto_block_get_payload_offset()`, fixed once open.
    payload_offset: u64,
    /// `qcrypto_block_get_sector_size()`.
    sector_size: u64,
    /// `crypto->header`: the header is in its own `header` child.
    has_header: bool,
    /// `crypto->updating_keys`: amend holds exclusive write access to the file.
    updating_keys: AtomicBool,
    /// `bs->supported_write_flags`: FUA if the file has it.
    supported_write_flags: u32,
}

impl std::fmt::Debug for LuksDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LuksDriver")
            .field("payload_offset", &self.payload_offset)
            .field("has_header", &self.has_header)
            .finish_non_exhaustive()
    }
}

/// `block_crypto_open_generic()` once the children are open: `file` is the payload,
/// `header` the detached header if there is one.
pub(crate) fn luks_open(
    key_secret: Option<String>,
    file: &Node,
    header: Option<&Node>,
    no_io: bool,
) -> Result<LuksDriver> {
    let mut flags = 0;
    if no_io {
        flags |= QCRYPTO_BLOCK_OPEN_NO_IO;
    }
    if header.is_some() {
        flags |= QCRYPTO_BLOCK_OPEN_DETACHED;
    }
    let opts = QCryptoBlockOpenOptions {
        u: QCryptoBlockOpenOptionsU::Luks(QCryptoBlockOptionsLUKS { key_secret }),
    };
    let mut io = NodeHeaderIo { node: header.unwrap_or(file) };
    let block = QCryptoBlock::open(&opts, None, &mut io, flags)?;
    Ok(LuksDriver {
        payload_offset: block.payload_offset(),
        sector_size: block.sector_size(),
        block: RwLock::new(block),
        has_header: header.is_some(),
        updating_keys: AtomicBool::new(false),
        supported_write_flags: BDRV_REQ_FUA & file.driver.supported_write_flags(),
    })
}

/// The `QCryptoBlockCreateOptionsLUKS` part of `BlockdevCreateOptionsLUKS`,
/// `qapi_BlockdevCreateOptionsLUKS_base()`.
pub(crate) fn create_base(o: &BlockdevCreateOptionsLUKS) -> QCryptoBlockCreateOptions {
    QCryptoBlockCreateOptions {
        u: QCryptoBlockCreateOptionsU::Luks(QCryptoBlockCreateOptionsLUKS {
            key_secret: o.key_secret.clone(),
            cipher_alg: o.cipher_alg,
            cipher_mode: o.cipher_mode,
            ivgen_alg: o.ivgen_alg,
            ivgen_hash_alg: o.ivgen_hash_alg,
            hash_alg: o.hash_alg,
            iter_time: o.iter_time,
        }),
    }
}

/// The checks at the top of `block_crypto_co_create_luks()`.
pub(crate) fn check_create(o: &BlockdevCreateOptionsLUKS) -> Result<()> {
    if o.header.is_none() && o.file.is_none() {
        return Err(Error::generic("Either the parameter 'header' or 'file' must be specified"));
    }
    if o.preallocation.unwrap_or_default() != PreallocMode::Off && o.file.is_none() {
        return Err(Error::generic(
            "Parameter 'preallocation' requires 'file' to be specified for formatting LUKS disk",
        ));
    }
    Ok(())
}

/// `block_crypto_co_create_generic()`: writes a new LUKS header to `node`, growing it to the
/// header plus `size` bytes of payload. With `detached` the header is all there is and `size`
/// is ignored. The caller holds `node` in a backend with write and resize permissions, as
/// QEMU's `blk_co_new_with_bs()` there does.
pub(crate) fn create_generic(
    node: &Node,
    size: u64,
    opts: &QCryptoBlockCreateOptions,
    prealloc: PreallocMode,
    detached: bool,
) -> Result<()> {
    let prealloc = if prealloc == PreallocMode::Metadata { PreallocMode::Off } else { prealloc };
    let mut io = CreateIo { node, size: if detached { 0 } else { size }, prealloc };
    let flags = if detached { QCRYPTO_BLOCK_CREATE_DETACHED } else { 0 };
    QCryptoBlock::create(opts, None, &mut io, flags)?;
    Ok(())
}

/// `block_crypto_co_format_luks_payload()`: sizes the payload file of a volume whose header is
/// detached.
fn format_payload(graph: &BlockGraph, o: &BlockdevCreateOptionsLUKS) -> Result<()> {
    let Some(file) = o.file.clone() else { return Ok(()) };
    let blk = graph.open_create_blk(file)?;
    let node = blk.root().expect("a new backend has its node");
    node.truncate_full(o.size as i64, true, o.preallocation.unwrap_or_default(), 0)
}

/// `block_crypto_co_create_luks()`: `blockdev-create` with `driver: luks`.
fn create(graph: &BlockGraph, options: BlockdevCreateOptionsU) -> Result<()> {
    let BlockdevCreateOptionsU::Luks(o) = options else {
        unreachable!("luks driver with other create options")
    };
    check_create(&o)?;
    let opts = create_base(&o);
    if let Some(header) = o.header.clone() {
        // A LUKS volume with a detached header: format the header node, then size the
        // payload node.
        let blk = graph.open_create_blk(header)?;
        let node = blk.root().expect("a new backend has its node");
        create_generic(&node, 0, &opts, PreallocMode::Off, true)?;
        drop(blk);
        format_payload(graph, &o)
    } else if let Some(file) = o.file.clone() {
        let blk = graph.open_create_blk(file)?;
        let node = blk.root().expect("a new backend has its node");
        create_generic(&node, o.size, &opts, o.preallocation.unwrap_or_default(), false)
    } else {
        unreachable!("check_create() wants a header or a file")
    }
}

/// Takes `key` out of the `-o` options as a string.
fn take(options: &mut QDict, key: &str) -> Option<String> {
    options.remove(key).and_then(|v| v.as_str().map(str::to_owned))
}

/// `block_crypto_create_opts_init()` for the `-o` options of `qemu-img create`: the crypto
/// options in `options` (taken out) as `QCryptoBlockCreateOptions` of `format`.
pub(crate) fn create_opts_init(
    options: &mut QDict,
    format: &str,
) -> Result<QCryptoBlockCreateOptions> {
    let mut crypto = QDict::new();
    for key in CRYPTO_CREATE_OPTS {
        if let Some(v) = options.remove(key) {
            crypto.put(*key, v);
        }
    }
    crypto.put("format", format);
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(crypto));
    let mut opts = QCryptoBlockCreateOptions::default();
    QCryptoBlockCreateOptions::visit(&mut v, None, &mut opts)?;
    Ok(opts)
}

/// `block_crypto_co_create_opts_luks()`: `qemu-img create -f luks`.
fn create_opts(filename: &str, options: &mut QDict) -> Result<()> {
    let size = match take(options, "size") {
        Some(v) => parse_option_size("size", &v)?,
        None => 0,
    };
    let prealloc = match take(options, "preallocation") {
        Some(v) => PreallocMode::from_name(&v)
            .ok_or_else(|| Error::generic(format!("Invalid parameter '{v}'")))?,
        None => PreallocMode::Off,
    };
    let detached = match take(options, "detached-header") {
        Some(v) => qapi_bool_parse("detached-header", &v)?,
        None => false,
    };
    let opts = create_opts_init(options, "luks")?;

    let graph = BlockGraph::new();
    graph.create_file(filename, options)?;
    let blk = graph.open_protocol_blk(filename)?;
    let node = blk.root().expect("a new backend has its node");
    let r = create_generic(&node, size, &opts, prealloc, detached);
    if r.is_err() {
        // The file was truncated and is of no use, even if it existed before.
        delete_file_noerr(&node);
    }
    r
}

/// `bdrv_co_delete_file_noerr()`: removes the file behind a protocol node, for the `file`
/// driver; other protocols cannot delete, which is not reported.
fn delete_file_noerr(node: &Node) {
    if node.driver_name != "file" {
        return;
    }
    let Some(name) = node.filename() else { return };
    let is_file = std::fs::metadata(&name).is_ok_and(|m| m.is_file());
    let r = if is_file {
        std::fs::remove_file(&name)
            .map_err(|e| Error::from_io(format!("Failed to delete file '{name}'"), e))
    } else {
        Err(Error::from_io(format!("{name} is not a regular file"), errno(libc::ENOENT)))
    };
    if let Err(e) = r {
        report::error_report(e.message());
    }
}

/// `block_crypto_open_luks()`.
fn luks_open_node(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Luks(o) = opts else { unreachable!("luks driver with other options") };
    let file = args.open_child(*o.file, "file", BDRV_CHILD_IMAGE | BDRV_CHILD_PRIMARY)?;
    let header = args.open_child_opt(o.header.map(|h| *h), "header", BDRV_CHILD_METADATA, true)?;
    let d = luks_open(o.key_secret, &file, header.as_deref(), args.flags.no_io)?;
    args.meta.encrypted = true;
    Ok(Box::new(d))
}

/// The `QCryptoBlockAmendOptionsLUKS` part of `BlockdevAmendOptionsLUKS`.
///
/// The amend code is reached through [`Driver::amend`], which nothing calls until the graph
/// has `x-blockdev-amend`.
#[allow(dead_code)]
pub(crate) fn amend_base(o: &BlockdevAmendOptionsLUKS) -> QCryptoBlockAmendOptions {
    QCryptoBlockAmendOptions {
        u: QCryptoBlockAmendOptionsU::Luks(QCryptoBlockAmendOptionsLUKS {
            state: o.state,
            new_secret: o.new_secret.clone(),
            old_secret: o.old_secret.clone(),
            keyslot: o.keyslot,
            iter_time: o.iter_time,
            secret: o.secret.clone(),
        }),
    }
}

#[allow(dead_code)] // See amend_base().
impl LuksDriver {
    /// The encryption context, for callers that need the header details.
    pub(crate) fn block(&self) -> RwLockReadGuard<'_, QCryptoBlock> {
        self.block.read().unwrap_or_else(|e| e.into_inner())
    }

    /// The node that holds the header: the `header` child if there is one, else `file`.
    fn header_node(&self, bs: &Node) -> Arc<Node> {
        if self.has_header {
            if let Some(c) = bs.child("header") {
                return c.node;
            }
        }
        bs.file()
    }

    /// `block_crypto_amend_prepare()` and `block_crypto_amend_cleanup()`: set or clear the
    /// flag that makes [`Driver::child_perm_for`] ask for exclusive write access.
    pub(crate) fn set_updating_keys(&self, on: bool) {
        self.updating_keys.store(on, Ordering::SeqCst);
    }

    /// `block_crypto_amend_options_generic_luks()`: adds or erases key slots. The caller holds
    /// the exclusive permissions, see [`LuksDriver::set_updating_keys`].
    ///
    /// Requests wait for the update to finish, which QEMU gets from draining the node.
    pub(crate) fn amend(
        &self,
        bs: &Node,
        opts: &QCryptoBlockAmendOptions,
        force: bool,
    ) -> Result<()> {
        let node = self.header_node(bs);
        let mut io = NodeHeaderIo { node: &node };
        let mut block = self.block.write().unwrap_or_else(|e| e.into_inner());
        block.amend_options(&mut io, opts, force)
    }

    /// `block_crypto_amend_cleanup()`: gives back the exclusive access to the file.
    fn release_perms(&self, bs: &Node, file: Option<&crate::node::Child>) {
        if let Some(c) = file {
            if let Err(e) = bs.refresh_child_perms(c) {
                report::error_report(e.message());
            }
        }
    }

    fn payload_offset(&self) -> u64 {
        self.payload_offset
    }
}

impl Driver for LuksDriver {
    /// `block_crypto_co_preadv()`.
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let sector_size = self.sector_size;
        assert!(offset % sector_size == 0 && buf.len() as u64 % sector_size == 0);
        let file = bs.file();
        let payload = self.payload_offset();
        let block = self.block();
        let mut done = 0usize;
        // The chunks of `buf` are the bounce buffer here: the ciphertext only ever sits in the
        // caller's buffer, which is ours until we return, never in guest memory.
        while done < buf.len() {
            let cur = (buf.len() - done).min(MAX_IO_SIZE as usize);
            let chunk = &mut buf[done..done + cur];
            file.pread(payload + offset + done as u64, chunk)?;
            block.decrypt(offset + done as u64, chunk).map_err(|_| errno(libc::EIO))?;
            done += cur;
        }
        Ok(())
    }

    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.pwrite_flags(bs, offset, buf, 0)
    }

    /// `block_crypto_co_pwritev()`.
    fn pwrite_flags(&self, bs: &Node, offset: u64, buf: &[u8], flags: u32) -> io::Result<()> {
        let sector_size = self.sector_size;
        assert!(offset % sector_size == 0 && buf.len() as u64 % sector_size == 0);
        let file = bs.file();
        let payload = self.payload_offset();
        let block = self.block();
        let mut bounce = vec![0u8; buf.len().min(MAX_IO_SIZE as usize)];
        let mut done = 0usize;
        while done < buf.len() {
            let cur = (buf.len() - done).min(MAX_IO_SIZE as usize);
            let data = &mut bounce[..cur];
            data.copy_from_slice(&buf[done..done + cur]);
            block.encrypt(offset + done as u64, data).map_err(|_| errno(libc::EIO))?;
            file.pwrite_flags(payload + offset + done as u64, data, flags)?;
            done += cur;
        }
        Ok(())
    }

    fn supported_write_flags(&self) -> u32 {
        self.supported_write_flags
    }

    /// `block_crypto_co_getlength()`.
    fn getlength(&self, bs: &Node) -> io::Result<u64> {
        let len = bs.file().getlength()?;
        len.checked_sub(self.payload_offset()).ok_or_else(|| errno(libc::EIO))
    }

    fn truncate(&self, bs: &Node, len: u64) -> Result<()> {
        self.truncate_full(bs, len, false, PreallocMode::Off, 0)
    }

    /// `block_crypto_co_truncate()`.
    fn truncate_full(
        &self,
        bs: &Node,
        offset: u64,
        exact: bool,
        prealloc: PreallocMode,
        _flags: u32,
    ) -> Result<()> {
        let payload = self.payload_offset();
        if offset > i64::MAX as u64 || offset > i64::MAX as u64 - payload {
            return Err(Error::generic("The requested file size is too large"));
        }
        bs.file().truncate_full((offset + payload) as i64, exact, prealloc, 0)
    }

    /// `block_crypto_refresh_limits()`: no sub-sector I/O.
    fn refresh_limits(&self, _bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        bl.request_alignment = self.sector_size as u32;
        Ok(())
    }

    /// `block_crypto_child_perms()`.
    fn child_perm_for(&self, ctx: &PermCtx<'_>, perm: u64, shared: u64) -> (u64, u64) {
        let (mut nperm, mut nshared) = default_perms(ctx, perm, shared);
        // For backward compatibility, share the write and resize permissions.
        nshared |= shared & (BLK_PERM_WRITE | BLK_PERM_RESIZE);
        // Not quite a format driver: only ask for write and resize when the parents do.
        nperm &= !(BLK_PERM_WRITE | BLK_PERM_RESIZE);
        nperm |= perm & (BLK_PERM_WRITE | BLK_PERM_RESIZE);
        if self.updating_keys.load(Ordering::SeqCst) {
            // Updating key slots needs exclusive write access to the header.
            nperm |= BLK_PERM_WRITE;
            nshared &= !(BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE);
        }
        (nperm, nshared)
    }

    /// `block_crypto_reopen_prepare()`: nothing needs checking.
    fn reopen_prepare(&self, _bs: &Node, _state: &mut ReopenState) -> Option<Result<()>> {
        Some(Ok(()))
    }

    /// `block_crypto_co_get_info_luks()`: the cluster size of the file.
    fn get_info(&self, bs: &Node) -> Option<io::Result<BlockDriverInfo>> {
        Some(
            bs.file().get_info().map(|sub| BlockDriverInfo {
                cluster_size: sub.cluster_size,
                ..Default::default()
            }),
        )
    }

    /// `block_crypto_get_specific_info_luks()`.
    fn get_specific_info(&self, _bs: &Node) -> Result<Option<ImageInfoSpecific>> {
        match self.block().get_info()?.u {
            QCryptoBlockInfoU::Luks(data) => Ok(Some(ImageInfoSpecific {
                u: ImageInfoSpecificU::Luks(ImageInfoSpecificLUKSWrapper { data }),
            })),
            QCryptoBlockInfoU::Qcow => unreachable!("a luks node holds a LUKS block"),
        }
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    /// `block_crypto_co_amend_luks()` between `block_crypto_amend_prepare()` and
    /// `block_crypto_amend_cleanup()`.
    fn amend(&self, bs: &Node, opts: &BlockdevAmendOptions, force: bool) -> Option<Result<()>> {
        let BlockdevAmendOptionsU::Luks(o) = &opts.u else { return None };
        let amend_opts = amend_base(o);
        let file = bs.child("file");
        // Exclusive read and write access to the file while the key slots change.
        self.set_updating_keys(true);
        if let Some(c) = &file {
            if let Err(e) = bs.refresh_child_perms(c) {
                self.set_updating_keys(false);
                self.release_perms(bs, file.as_ref());
                return Some(Err(e));
            }
        }
        let r = LuksDriver::amend(self, bs, &amend_opts, force);
        self.set_updating_keys(false);
        self.release_perms(bs, file.as_ref());
        Some(r)
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use ruvm_qapi::types::{
        QCryptoBlockLUKSKeyslotState, QCryptoCipherAlgo, QCryptoCipherMode, QCryptoHashAlgo,
        QCryptoIVGenAlgo,
    };

    fn setup() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            ruvm_crypto::pbkdf::set_iters_per_second_override(Some(1000));
            ruvm_crypto::secret::secret_object_add_global("secret,id=ut0,data=first").unwrap();
            ruvm_crypto::secret::secret_object_add_global("secret,id=ut1,data=second").unwrap();
        });
    }

    fn scratch(test: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ruvm-luks-{}-{test}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn qdict(pairs: &[(&str, &str)]) -> QDict {
        let mut d = QDict::new();
        for (k, v) in pairs {
            d.put(*k, *v);
        }
        d
    }

    fn open(g: &BlockGraph, path: &str, secret: &str) -> Result<Arc<Node>> {
        let o = qdict(&[
            ("driver", "luks"),
            ("key-secret", secret),
            ("file.driver", "file"),
            ("file.filename", path),
        ]);
        let name = g.open_image(None, o)?;
        g.lookup_bs(&name)
    }

    fn info(node: &Node) -> ruvm_qapi::types::QCryptoBlockInfoLUKS {
        match node.driver.get_specific_info(node).unwrap().unwrap().u {
            ImageInfoSpecificU::Luks(w) => w.data,
            _ => panic!("not luks info"),
        }
    }

    #[test]
    fn probe_luks() {
        assert_eq!(probe(b"LUKS\xba\xbe\x00\x01", None), 100);
        assert_eq!(probe(b"LUKS\xba\xbe\x00\x02", None), 0);
        assert_eq!(probe(b"QFI\xfb", Some("x")), 0);
    }

    #[test]
    fn create_opts_all_options() {
        setup();
        let dir = scratch("create-opts");
        let path = dir.join("a.luks");
        let path = path.to_str().unwrap();
        let mut o = qdict(&[
            ("size", "1M"),
            ("key-secret", "ut0"),
            ("cipher-alg", "aes-128"),
            ("cipher-mode", "cbc"),
            ("ivgen-alg", "essiv"),
            ("ivgen-hash-alg", "sha256"),
            ("hash-alg", "sha1"),
            ("iter-time", "10"),
            ("preallocation", "full"),
        ]);
        create_opts(path, &mut o).unwrap();
        assert!(o.is_empty());
        // Header, eight 64 KiB key slots of AES-128 key material, then the payload.
        let payload = 4096 + 8 * 64 * 1024;
        assert_eq!(std::fs::metadata(path).unwrap().len(), payload + (1 << 20));

        let g = BlockGraph::new();
        let node = open(&g, path, "ut0").unwrap();
        assert!(node.meta.lock().unwrap().encrypted);
        assert_eq!(node.getlength().unwrap(), 1 << 20);
        let i = info(&node);
        assert_eq!(i.cipher_alg, QCryptoCipherAlgo::Aes128);
        assert_eq!(i.cipher_mode, QCryptoCipherMode::Cbc);
        assert_eq!(i.ivgen_alg, QCryptoIVGenAlgo::Essiv);
        assert_eq!(i.ivgen_hash_alg, Some(QCryptoHashAlgo::Sha256));
        assert_eq!(i.hash_alg, QCryptoHashAlgo::Sha1);
        assert_eq!(i.payload_offset, payload as i64);
        assert!(!i.detached_header);

        // Data goes through encrypted.
        node.pwrite(0, &[0x5a; 4096]).unwrap();
        let mut raw = vec![0u8; 4096];
        node.file().pread(payload, &mut raw).unwrap();
        assert_ne!(raw, vec![0x5a; 4096]);
        let mut back = vec![0u8; 4096];
        node.pread(0, &mut back).unwrap();
        assert_eq!(back, vec![0x5a; 4096]);

        // Resizing keeps the header.
        node.truncate(2 << 20).unwrap();
        assert_eq!(node.getlength().unwrap(), 2 << 20);
        assert_eq!(std::fs::metadata(path).unwrap().len(), payload + (2 << 20));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn create_opts_detached_and_errors() {
        setup();
        let dir = scratch("create-opts-errors");
        let path = dir.join("h.luks");
        let path = path.to_str().unwrap();
        let mut o = qdict(&[
            ("size", "1M"),
            ("key-secret", "ut0"),
            ("iter-time", "10"),
            ("detached-header", "on"),
        ]);
        create_opts(path, &mut o).unwrap();
        // A detached header has no payload: header plus key material only.
        assert_eq!(std::fs::metadata(path).unwrap().len(), 2068480);

        let mut o = qdict(&[("preallocation", "bogus")]);
        assert_eq!(create_opts(path, &mut o).unwrap_err().message(), "Invalid parameter 'bogus'");
        let mut o = qdict(&[("detached-header", "maybe")]);
        assert_eq!(
            create_opts(path, &mut o).unwrap_err().message(),
            "Parameter 'detached-header' expects 'on' or 'off'"
        );
        let mut o = qdict(&[("cipher-alg", "rot13")]);
        assert_eq!(
            create_opts(path, &mut o).unwrap_err().message(),
            "Parameter 'cipher-alg' does not accept value 'rot13'"
        );

        // Without a secret the header cannot be written, and the new file goes away.
        let other = dir.join("nosecret.luks");
        let other = other.to_str().unwrap();
        let mut o = qdict(&[("size", "1M")]);
        assert_eq!(
            create_opts(other, &mut o).unwrap_err().message(),
            "Parameter 'key-secret' is required for cipher"
        );
        assert!(!std::path::Path::new(other).exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn amend_key_slots() {
        setup();
        let dir = scratch("amend");
        let path = dir.join("a.luks");
        let path = path.to_str().unwrap();
        let mut o = qdict(&[("size", "64K"), ("key-secret", "ut0"), ("iter-time", "10")]);
        create_opts(path, &mut o).unwrap();

        let amend = |node: &Node, o: BlockdevAmendOptionsLUKS, force: bool| {
            let opts = BlockdevAmendOptions { u: BlockdevAmendOptionsU::Luks(o) };
            node.driver.amend(node, &opts, force).unwrap()
        };
        let base = BlockdevAmendOptionsLUKS {
            state: QCryptoBlockLUKSKeyslotState::Active,
            new_secret: None,
            old_secret: None,
            keyslot: None,
            iter_time: Some(10),
            secret: None,
        };
        {
            let g = BlockGraph::new();
            let node = open(&g, path, "ut0").unwrap();
            node.pwrite(0, &[7; 512]).unwrap();
            let add = BlockdevAmendOptionsLUKS { new_secret: Some("ut1".into()), ..base.clone() };
            amend(&node, add, false).unwrap();
            let slots = info(&node).slots;
            assert!(slots[0].active && slots[1].active && !slots[2].active);
            // The exclusive access is given back.
            assert!(
                !node
                    .driver
                    .as_any()
                    .unwrap()
                    .downcast_ref::<LuksDriver>()
                    .unwrap()
                    .updating_keys
                    .load(Ordering::SeqCst)
            );

            // Erasing the only slot of a password needs force when it is the last one; here
            // slot 0 goes and slot 1 stays.
            let erase = BlockdevAmendOptionsLUKS {
                state: QCryptoBlockLUKSKeyslotState::Inactive,
                old_secret: Some("ut0".into()),
                iter_time: None,
                ..base.clone()
            };
            amend(&node, erase, false).unwrap();
            let slots = info(&node).slots;
            assert!(!slots[0].active && slots[1].active);

            let erase_last = BlockdevAmendOptionsLUKS {
                state: QCryptoBlockLUKSKeyslotState::Inactive,
                keyslot: Some(1),
                iter_time: None,
                ..base.clone()
            };
            assert_eq!(
                amend(&node, erase_last, false).unwrap_err().message(),
                "Attempt to erase the only active keyslot 1 which will erase all the data in the image irreversibly - refusing operation"
            );

            // Amending a node of another driver's options is not ours.
            let q = BlockdevAmendOptions { u: BlockdevAmendOptionsU::Qcow2(Default::default()) };
            assert!(node.driver.amend(&node, &q, false).is_none());
        }
        // The old password is gone, the new one opens the image and the data is intact.
        let g = BlockGraph::new();
        assert_eq!(
            open(&g, path, "ut0").unwrap_err().message(),
            "Invalid password, cannot unlock any keyslot"
        );
        let node = open(&g, path, "ut1").unwrap();
        let mut buf = [0u8; 512];
        node.pread(0, &mut buf).unwrap();
        assert_eq!(buf, [7; 512]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
