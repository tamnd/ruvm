// SPDX-License-Identifier: GPL-2.0-or-later

//! Creating VMDK images: `vmdk_co_create_opts()` for `qemu-img create` and
//! `vmdk_co_create()` for `blockdev-create`.

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{
    BlockdevCreateOptionsU, BlockdevCreateOptionsVmdk, BlockdevRef, BlockdevVmdkAdapterType,
    BlockdevVmdkSubformat, PreallocMode,
};
use ruvm_qapi::{QDict, QValue};

use super::open::read_cid;
use super::{
    VMDK4_COMPRESSION_DEFLATE, VMDK4_FLAG_COMPRESS, VMDK4_FLAG_MARKER, VMDK4_FLAG_NL_DETECT,
    VMDK4_FLAG_RGD, VMDK4_FLAG_ZERO_GRAIN, VMDK4_MAGIC, VmdkDriver,
};
use crate::backend::BlockBackend;
use crate::graph::{BlockGraph, OpenCtx};
use crate::imgopts::{take_bool, take_size, take_str};
use crate::node::Node;

const BDRV_SECTOR_SIZE: u64 = 512;

/// Makes or opens the file of extent `idx` with `size` bytes: 0 is the descriptor file,
/// which is the image itself for monolithic sparse and stream-optimized images. A `size` of
/// `None` only asks whether the extent exists.
type ExtentFn<'a> = dyn FnMut(Option<u64>, usize, &Kind) -> Result<Option<BlockBackend>> + 'a;

/// What kind of extents an image has.
struct Kind {
    flat: bool,
    split: bool,
    compress: bool,
    zeroed_grain: bool,
}

/// The node of a backend made for creating an image.
fn root(blk: &BlockBackend) -> std::sync::Arc<Node> {
    blk.root().expect("a new backend has its node")
}

/// `vmdk_init_extent()`: writes an empty sparse extent, or sizes a flat one.
pub(super) fn init_extent(
    node: &Node,
    filesize: u64,
    flat: bool,
    compress: bool,
    zeroed_grain: bool,
) -> Result<()> {
    if flat {
        return node.truncate_full(filesize as i64, false, PreallocMode::Off, 0);
    }
    let header = sparse_header(filesize, compress, zeroed_grain);
    let rgd_offset = super::le64(&header, super::open::header4::RGD_OFFSET);
    let gd_offset = super::le64(&header, super::open::header4::GD_OFFSET);
    let grain_offset = super::le64(&header, super::open::header4::GRAIN_OFFSET);
    let (gd_sectors, gt_size, gt_count) = grain_directory_geometry(filesize);

    node.pwrite(0, &VMDK4_MAGIC.to_be_bytes())
        .map_err(|e| Error::from_io("failed to write VMDK magic", e))?;
    node.pwrite(4, &header).map_err(|e| Error::from_io("failed to write VMDK header", e))?;
    node.truncate_full((grain_offset << 9) as i64, false, PreallocMode::Off, 0)?;

    // The grain directory, then its backup.
    let mut gd_buf = vec![0u8; (gd_sectors * BDRV_SECTOR_SIZE) as usize];
    let fill = |start: u64, buf: &mut [u8]| {
        let mut tmp = start as u32;
        for i in 0..gt_count as usize {
            buf[i * 4..i * 4 + 4].copy_from_slice(&tmp.to_le_bytes());
            tmp = tmp.wrapping_add(gt_size as u32);
        }
    };
    fill(rgd_offset + gd_sectors, &mut gd_buf);
    node.pwrite(rgd_offset * BDRV_SECTOR_SIZE, &gd_buf)
        .map_err(|e| Error::from_io("failed to write VMDK grain directory", e))?;
    fill(gd_offset + gd_sectors, &mut gd_buf);
    node.pwrite(gd_offset * BDRV_SECTOR_SIZE, &gd_buf)
        .map_err(|e| Error::from_io("failed to write VMDK backup grain directory", e))?;
    Ok(())
}

/// The grain directory of a new sparse extent: its size in sectors, the size of a grain
/// table in sectors and the number of grain tables.
fn grain_directory_geometry(filesize: u64) -> (u64, u64, u64) {
    let granularity = 128;
    let num_gtes_per_gt = 512;
    let grains = (filesize / BDRV_SECTOR_SIZE).div_ceil(granularity);
    let gt_size = (num_gtes_per_gt * 4u64).div_ceil(BDRV_SECTOR_SIZE);
    let gt_count = grains.div_ceil(num_gtes_per_gt);
    let gd_sectors = (gt_count * 4).div_ceil(BDRV_SECTOR_SIZE);
    // uint32_t in QEMU.
    (gd_sectors & 0xffff_ffff, gt_size, gt_count & 0xffff_ffff)
}

/// The `VMDK4Header` of a new sparse extent of `filesize` bytes.
pub(super) fn sparse_header(filesize: u64, compress: bool, zeroed_grain: bool) -> Vec<u8> {
    use super::open::header4 as h;
    let mut header = vec![0u8; h::SIZE];
    let version: u32 = if compress {
        3
    } else if zeroed_grain {
        2
    } else {
        1
    };
    let flags = VMDK4_FLAG_RGD
        | VMDK4_FLAG_NL_DETECT
        | if compress { VMDK4_FLAG_COMPRESS | VMDK4_FLAG_MARKER } else { 0 }
        | if zeroed_grain { VMDK4_FLAG_ZERO_GRAIN } else { 0 };
    let granularity: u64 = 128;
    let (gd_sectors, gt_size, gt_count) = grain_directory_geometry(filesize);
    let desc_offset: u64 = 1;
    let desc_size: u64 = 20;
    let rgd_offset = desc_offset + desc_size;
    let gd_offset = rgd_offset + gd_sectors + gt_size * gt_count;
    let grain_offset =
        super::round_up_mask(gd_offset + gd_sectors + gt_size * gt_count, granularity);

    header[h::VERSION..h::VERSION + 4].copy_from_slice(&version.to_le_bytes());
    header[h::FLAGS..h::FLAGS + 4].copy_from_slice(&flags.to_le_bytes());
    header[h::CAPACITY..h::CAPACITY + 8]
        .copy_from_slice(&(filesize / BDRV_SECTOR_SIZE).to_le_bytes());
    header[h::GRANULARITY..h::GRANULARITY + 8].copy_from_slice(&granularity.to_le_bytes());
    header[h::DESC_OFFSET..h::DESC_OFFSET + 8].copy_from_slice(&desc_offset.to_le_bytes());
    header[h::DESC_SIZE..h::DESC_SIZE + 8].copy_from_slice(&desc_size.to_le_bytes());
    header[h::NUM_GTES_PER_GT..h::NUM_GTES_PER_GT + 4].copy_from_slice(&512u32.to_le_bytes());
    header[h::RGD_OFFSET..h::RGD_OFFSET + 8].copy_from_slice(&rgd_offset.to_le_bytes());
    header[h::GD_OFFSET..h::GD_OFFSET + 8].copy_from_slice(&gd_offset.to_le_bytes());
    header[h::GRAIN_OFFSET..h::GRAIN_OFFSET + 8].copy_from_slice(&grain_offset.to_le_bytes());
    header[h::CHECK_BYTES..h::CHECK_BYTES + 4].copy_from_slice(&[0x0a, 0x20, 0x0d, 0x0a]);
    let algo: u16 = if compress { VMDK4_COMPRESSION_DEFLATE } else { 0 };
    header[h::COMPRESS_ALGORITHM..h::COMPRESS_ALGORITHM + 2].copy_from_slice(&algo.to_le_bytes());
    header
}

/// `filename_decompose()`: the directory with its separator, the base name without its
/// extension, and the extension with its dot.
pub(super) fn filename_decompose(filename: &str) -> Result<(String, String, String)> {
    if filename.is_empty() {
        return Err(Error::generic("No filename provided"));
    }
    let sep = filename.rfind('/').or_else(|| filename.rfind('\\')).or_else(|| filename.rfind(':'));
    let (path, p) = match sep {
        Some(i) => (&filename[..=i], &filename[i + 1..]),
        None => ("", filename),
    };
    let (prefix, postfix) = match p.rfind('.') {
        Some(q) => (&p[..q], &p[q..]),
        None => (p, ""),
    };
    Ok((path.to_string(), prefix.to_string(), postfix.to_string()))
}

/// `g_path_get_basename()`.
fn basename(path: &str) -> String {
    if path.is_empty() {
        return ".".to_string();
    }
    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() {
        return "/".to_string();
    }
    match trimmed.rfind('/') {
        Some(i) => trimmed[i + 1..].to_string(),
        None => trimmed.to_string(),
    }
}

/// `bs->filename` of a node.
fn node_filename(node: &Node) -> String {
    node.refresh_filename();
    node.filename().unwrap_or_default()
}

/// The options of `vmdk_co_do_create()`.
struct CreateArgs<'a> {
    size: u64,
    subformat: BlockdevVmdkSubformat,
    adapter_type: BlockdevVmdkAdapterType,
    backing_file: Option<&'a str>,
    hw_version: Option<&'a str>,
    toolsversion: Option<&'a str>,
    compat6: bool,
    zeroed_grain: bool,
}

/// The descriptor of a new image.
#[allow(clippy::too_many_arguments, reason = "the fields of the descriptor template")]
pub(super) fn descriptor(
    cid: u32,
    parent_cid: u32,
    create_type: &str,
    parent_desc_line: &str,
    extent_lines: &str,
    hw_version: &str,
    cylinders: i64,
    heads: u32,
    adapter_type: &str,
    toolsversion: &str,
) -> String {
    format!(
        "# Disk DescriptorFile\n\
         version=1\n\
         CID={cid:x}\n\
         parentCID={parent_cid:x}\n\
         createType=\"{create_type}\"\n\
         {parent_desc_line}\
         \n\
         # Extent description\n\
         {extent_lines}\
         \n\
         # The Disk Data Base\n\
         #DDB\n\
         \n\
         ddb.virtualHWVersion = \"{hw_version}\"\n\
         ddb.geometry.cylinders = \"{cylinders}\"\n\
         ddb.geometry.heads = \"{heads}\"\n\
         ddb.geometry.sectors = \"63\"\n\
         ddb.adapterType = \"{adapter_type}\"\n\
         ddb.toolsVersion = \"{toolsversion}\"\n"
    )
}

/// `vmdk_co_do_create()`.
fn do_create(a: CreateArgs<'_>, extent_fn: &mut ExtentFn<'_>) -> Result<()> {
    const SPLIT_SIZE: u64 = 0x8000_0000;
    let mut hw_version = a.hw_version;
    if a.compat6 {
        if hw_version.is_some() {
            return Err(Error::generic("compat6 cannot be enabled with hwversion set"));
        }
        hw_version = Some("6");
    }
    let hw_version = hw_version.unwrap_or("4");
    let toolsversion = a.toolsversion.unwrap_or("2147483647");

    // VMware uses this many heads for images with other adapters than IDE.
    let number_heads: u32 = if a.adapter_type == BlockdevVmdkAdapterType::Ide { 16 } else { 255 };
    let kind = Kind {
        split: matches!(
            a.subformat,
            BlockdevVmdkSubformat::TwoGbMaxExtentFlat | BlockdevVmdkSubformat::TwoGbMaxExtentSparse
        ),
        flat: matches!(
            a.subformat,
            BlockdevVmdkSubformat::MonolithicFlat | BlockdevVmdkSubformat::TwoGbMaxExtentFlat
        ),
        compress: a.subformat == BlockdevVmdkSubformat::StreamOptimized,
        zeroed_grain: a.zeroed_grain,
    };
    let extent_line = |size: u64, filename: &str| {
        let sectors = size.div_ceil(BDRV_SECTOR_SIZE);
        if kind.flat {
            format!("RW {sectors} FLAT \"{}\" 0\n", basename(filename))
        } else {
            format!("RW {sectors} SPARSE \"{}\"\n", basename(filename))
        }
    };
    if kind.flat && a.backing_file.is_some() {
        return Err(Error::generic("Flat image can't have backing file"));
    }
    if kind.flat && a.zeroed_grain {
        return Err(Error::generic("Flat image can't enable zeroed grain"));
    }

    let extent_size = if kind.split { SPLIT_SIZE } else { a.size };
    let mut created_size = if !kind.split && !kind.flat { extent_size } else { 0 };
    let mut ext_desc_lines = String::new();

    // The descriptor file.
    let blk = extent_fn(Some(created_size), 0, &kind)?
        .ok_or_else(|| Error::generic("Could not create the descriptor file"))?;
    let desc_node = root(&blk);
    let desc_filename = node_filename(&desc_node);
    if !kind.split && !kind.flat {
        ext_desc_lines.push_str(&extent_line(created_size, &desc_filename));
    }

    let mut parent_cid: u32 = 0xffff_ffff;
    let mut parent_desc_line = String::new();
    if let Some(backing_file) = a.backing_file {
        let full = crate::tools::full_backing_filename_from_filename(&desc_filename, backing_file)?;
        let mut options = QDict::new();
        options.put("backing", QValue::Null);
        options.put("read-only", "on");
        let graph = BlockGraph::new();
        let (backing, _, _) = graph.open_nodes_qdict(Some(&full), options, OpenCtx::default())?;
        if backing.driver_name != "vmdk" {
            return Err(Error::generic(format!(
                "Invalid backing file format: {}. Must be vmdk",
                backing.driver_name
            )));
        }
        let cid = backing
            .driver
            .as_any()
            .and_then(|d| d.downcast_ref::<VmdkDriver>())
            .zip(backing.child("file"))
            .and_then(|(d, f)| read_cid(&f.node, d.desc_offset, false).ok());
        parent_cid = cid.ok_or_else(|| Error::generic("Failed to read parent CID"))?;
        parent_desc_line = format!("parentFileNameHint=\"{backing_file}\"");
    }

    let mut extent_idx = 1;
    while created_size < a.size {
        let cur_size = (a.size - created_size).min(extent_size);
        let extent_blk = extent_fn(Some(cur_size), extent_idx, &kind)?
            .ok_or_else(|| Error::generic("Could not create the extent"))?;
        ext_desc_lines.push_str(&extent_line(cur_size, &node_filename(&root(&extent_blk))));
        created_size += cur_size;
        extent_idx += 1;
    }

    // Check whether there are extents left over.
    if matches!(extent_fn(None, extent_idx, &kind), Ok(Some(_))) {
        return Err(Error::generic("List of extents contains unused extents"));
    }

    let mut cid = [0u8; 4];
    ruvm_crypto::random::random_bytes(&mut cid).ok();
    let desc = descriptor(
        u32::from_ne_bytes(cid),
        parent_cid,
        a.subformat.as_str(),
        &parent_desc_line,
        &ext_desc_lines,
        hw_version,
        (a.size as i64) / (63 * i64::from(number_heads) * 512),
        number_heads,
        a.adapter_type.as_str(),
        toolsversion,
    );
    let desc_offset = if !kind.split && !kind.flat { 0x200 } else { 0 };
    desc_node
        .pwrite(desc_offset, desc.as_bytes())
        .map_err(|e| Error::from_io("Could not write description", e))?;
    // The descriptor file ends with the text, without padding to a sector.
    if desc_offset == 0 {
        desc_node.truncate_full(desc.len() as i64, false, PreallocMode::Off, 0)?;
    }
    Ok(())
}

/// `vmdk_co_create_opts()`: `qemu-img create -f vmdk`.
pub(super) fn vmdk_co_create_opts(filename: &str, options: &mut QDict) -> Result<()> {
    if let Some(fmt) = take_str(options, "backing_fmt") {
        if fmt != "vmdk" {
            return Err(Error::generic("backing_file must be a vmdk image"));
        }
    }
    let (path, prefix, postfix) = filename_decompose(filename)?;

    let total_size = super::round_up_mask(take_size(options, "size")?.unwrap_or(0), 512);
    let adapter_type = take_str(options, "adapter_type");
    let backing_file = take_str(options, "backing_file");
    let hw_version = take_str(options, "hwversion").filter(|v| v != "undefined");
    let toolsversion = take_str(options, "toolsversion");
    let compat6 = take_bool(options, "compat6")?.unwrap_or(false);
    let fmt = take_str(options, "subformat");
    let zeroed_grain = take_bool(options, "zeroed_grain")?.unwrap_or(false);

    let adapter_type = match adapter_type {
        Some(s) => {
            let i = BlockdevVmdkAdapterType::LOOKUP.parse(&s)?;
            BlockdevVmdkAdapterType::ALL[i]
        }
        None => BlockdevVmdkAdapterType::Ide,
    };
    let subformat = match fmt {
        Some(s) => {
            let i = BlockdevVmdkSubformat::LOOKUP.parse(&s)?;
            BlockdevVmdkSubformat::ALL[i]
        }
        // monolithicSparse by default.
        None => BlockdevVmdkSubformat::MonolithicSparse,
    };

    let graph = BlockGraph::new();
    // Every extent file is made with the same protocol options.
    let saved = options.clone();
    let mut first = Some(options);
    let mut extent_fn = |size: Option<u64>, idx: usize, k: &Kind| -> Result<Option<BlockBackend>> {
        // Done, no more extents.
        let Some(size) = size else { return Ok(None) };
        let rel_filename = if idx == 0 {
            format!("{prefix}{postfix}")
        } else if k.split {
            format!("{prefix}-{}{idx:03}{postfix}", if k.flat { 'f' } else { 's' })
        } else {
            format!("{prefix}-flat{postfix}")
        };
        let ext_filename = format!("{path}{rel_filename}");
        match first.take() {
            Some(o) => graph.create_file(&ext_filename, o)?,
            None => graph.create_file(&ext_filename, &mut saved.clone())?,
        }
        let blk = graph.open_protocol_blk(&ext_filename)?;
        init_extent(&root(&blk), size, k.flat, k.compress, k.zeroed_grain)?;
        Ok(Some(blk))
    };
    do_create(
        CreateArgs {
            size: total_size,
            subformat,
            adapter_type,
            backing_file: backing_file.as_deref(),
            hw_version: hw_version.as_deref(),
            toolsversion: toolsversion.as_deref(),
            compat6,
            zeroed_grain,
        },
        &mut extent_fn,
    )
}

/// The extent `idx` of `blockdev-create`: `file` for 0, then `extents[idx - 1]`.
fn blockdev_extent(o: &BlockdevCreateOptionsVmdk, idx: usize) -> Result<BlockdevRef> {
    if idx == 0 {
        return Ok(o.file.clone());
    }
    let list = o.extents.as_deref().unwrap_or_default();
    let mut pos = 0;
    for i in 1..idx {
        if pos >= list.len() || pos + 1 >= list.len() {
            return Err(Error::generic(format!("Extent [{i}] not specified")));
        }
        pos += 1;
    }
    match list.get(pos) {
        Some(r) => Ok(r.clone()),
        None => Err(Error::generic(format!("Extent [{}] not specified", idx - 1))),
    }
}

/// `vmdk_co_create()`: `blockdev-create` with `driver: vmdk`.
pub(super) fn vmdk_co_create(graph: &BlockGraph, options: BlockdevCreateOptionsU) -> Result<()> {
    let BlockdevCreateOptionsU::Vmdk(o) = options else {
        unreachable!("vmdk driver with other create options")
    };
    if o.size % BDRV_SECTOR_SIZE != 0 {
        return Err(Error::generic("Image size must be a multiple of 512 bytes"));
    }
    let mut extent_fn = |size: Option<u64>, idx: usize, k: &Kind| -> Result<Option<BlockBackend>> {
        let r = blockdev_extent(&o, idx)?;
        let blk = graph.open_create_blk(r)?;
        if let Some(size) = size {
            init_extent(&root(&blk), size, k.flat, k.compress, k.zeroed_grain)?;
        }
        Ok(Some(blk))
    };
    do_create(
        CreateArgs {
            size: o.size,
            subformat: o.subformat.unwrap_or_default(),
            adapter_type: o.adapter_type.unwrap_or_default(),
            backing_file: o.backing_file.as_deref(),
            hw_version: o.hwversion.as_deref(),
            toolsversion: o.toolsversion.as_deref(),
            compat6: false,
            zeroed_grain: o.zeroed_grain.unwrap_or(false),
        },
        &mut extent_fn,
    )
}
