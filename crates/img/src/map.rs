// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img map`.

use std::io;

use ruvm_base::error::strerror;
use ruvm_base::report::error_report;
use ruvm_block::tools::{
    BDRV_BLOCK_ALLOCATED, BDRV_BLOCK_COMPRESSED, BDRV_BLOCK_DATA, BDRV_BLOCK_OFFSET_VALID,
    BDRV_BLOCK_ZERO,
};

use crate::common::{
    OPTION_IMAGE_OPTS, OPTION_OBJECT, OPTION_OUTPUT, Opts, OutputFormat, cvtnum, error_exit, graph,
    img_open, lo, object_add, parse_output_format, root, tryhelp,
};
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("format", HasArg::Required, 'f'),
    lo("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("start-offset", HasArg::Required, 's'),
    lo("max-length", HasArg::Required, 'l'),
    lo("force-share", HasArg::No, 'U'),
    lo("output", HasArg::Required, OPTION_OUTPUT),
    lo("object", HasArg::Required, OPTION_OBJECT),
];

/// `MapEntry`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MapEntry {
    pub start: u64,
    pub length: u64,
    pub data: bool,
    pub zero: bool,
    pub compressed: bool,
    pub offset: u64,
    pub has_offset: bool,
    pub depth: u64,
    pub present: bool,
    pub filename: Option<String>,
}

/// `printf("%#-16" PRIx64)`: `0x` before anything but zero, padded to 16 columns.
fn hex16(v: u64) -> String {
    let s = if v == 0 { "0".to_string() } else { format!("{v:#x}") };
    format!("{s:<16}")
}

/// `dump_map_entry()`. `Err` when the human format cannot show the entry.
fn dump_map_entry(
    out: &mut String,
    format: OutputFormat,
    e: &MapEntry,
    next: Option<&mut MapEntry>,
) -> Result<(), ()> {
    match format {
        OutputFormat::Human => {
            if e.data && !e.has_offset {
                error_report("File contains external, encrypted or compressed clusters.");
                return Err(());
            }
            if e.data && !e.zero {
                out.push_str(&format!(
                    "{}{}{}{}\n",
                    hex16(e.start),
                    hex16(e.length),
                    hex16(if e.has_offset { e.offset } else { 0 }),
                    e.filename.as_deref().unwrap_or("")
                ));
            }
            // This format ignores the difference between 0, ZERO and ZERO|DATA, so the
            // flags of the next entry are changed to let more of them merge.
            if let Some(next) = next {
                if !next.data || next.zero {
                    next.data = false;
                    next.zero = true;
                }
            }
        }
        OutputFormat::Json => {
            out.push_str(&format!(
                "{{ \"start\": {}, \"length\": {}, \"depth\": {}, \"present\": {}, \"zero\": {}, \
                 \"data\": {}, \"compressed\": {}",
                e.start, e.length, e.depth, e.present, e.zero, e.data, e.compressed
            ));
            if e.has_offset {
                out.push_str(&format!(", \"offset\": {}", e.offset));
            }
            out.push('}');
            if next.is_some() {
                out.push_str(",\n");
            }
        }
    }
    Ok(())
}

/// `get_block_status()`: the status of `node` at `offset`, going down the backing chain
/// until some layer has the data or knows it reads as zeroes.
pub(crate) fn get_block_status(node: &str, offset: u64, bytes: u64) -> io::Result<MapEntry> {
    let g = graph();
    let mut bs = node.to_string();
    let mut bytes = bytes;
    let mut depth = 0;
    let mut st;
    loop {
        bs = g.skip_filters(&bs).unwrap_or(bs);
        st = g.block_status(&bs, offset, bytes)?;
        bytes = st.pnum;
        if st.ret & (BDRV_BLOCK_ZERO | BDRV_BLOCK_DATA) != 0 {
            break;
        }
        match g.cow_bs(&bs).ok().flatten() {
            Some(b) => bs = b,
            None => {
                st.ret = 0;
                break;
            }
        }
        depth += 1;
    }
    let has_offset = st.ret & BDRV_BLOCK_OFFSET_VALID != 0;
    let filename = match (&st.file, has_offset) {
        (Some(f), true) => g.node_details(f).ok().map(|d| d.filename),
        _ => None,
    };
    Ok(MapEntry {
        start: offset,
        length: bytes,
        data: st.ret & BDRV_BLOCK_DATA != 0,
        zero: st.ret & BDRV_BLOCK_ZERO != 0,
        compressed: st.ret & BDRV_BLOCK_COMPRESSED != 0,
        offset: st.map,
        has_offset,
        depth,
        present: st.ret & BDRV_BLOCK_ALLOCATED != 0,
        filename,
    })
}

/// `entry_mergeable()`.
fn entry_mergeable(curr: &MapEntry, next: &MapEntry) -> bool {
    if curr.length == 0 {
        return false;
    }
    if curr.zero != next.zero
        || curr.data != next.data
        || curr.compressed != next.compressed
        || curr.depth != next.depth
        || curr.present != next.present
        || curr.filename.is_some() != next.filename.is_some()
        || curr.has_offset != next.has_offset
    {
        return false;
    }
    if curr.filename.is_some() && curr.filename != next.filename {
        return false;
    }
    if curr.has_offset && curr.offset + curr.length != next.offset {
        return false;
    }
    true
}

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut output_format = OutputFormat::Human;
    let mut fmt = None;
    let mut image_opts = false;
    let mut force_share = false;
    let mut start_offset = 0u64;
    let mut max_length = None;
    let mut o = Opts::new(args, "hf:s:l:U", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => {
                return Err(cmd.help(
                    concat!(
                        "[-f FMT | --image-opts]\n",
                        "        [--start-offset OFFSET] [--max-length LENGTH]\n",
                        "        [--output human|json] [-U] [--object OBJDEF] FILE\n",
                    ),
                    concat!(
                        "  -f, --format FMT\n",
                        "     specify FILE image format explicitly (default: probing is used)\n",
                        "  --image-opts\n",
                        "     treat FILE as an option string (key=value,..), not a file name\n",
                        "     (incompatible with -f|--format)\n",
                        "  -s, --start-offset OFFSET\n",
                        "     start at the given OFFSET in the image, not at the beginning\n",
                        "  -l, --max-length LENGTH\n",
                        "     process at most LENGTH bytes instead of up to the end of the image\n",
                        "  --output human|json\n",
                        "     specify output format name (default: human)\n",
                        "  -U, --force-share\n",
                        "     open image in shared mode for concurrent access\n",
                        "  --object OBJDEF\n",
                        "     defines QEMU user-creatable object\n",
                        "  FILE\n",
                        "     the image file name, or option string (key=value,..)\n",
                        "     with --image-opts, to operate on\n",
                    ),
                ));
            }
            'f' => fmt = Some(arg),
            OPTION_IMAGE_OPTS => image_opts = true,
            's' => match cvtnum("start offset", &arg, true) {
                Some(v) => start_offset = v as u64,
                None => return Ok(1),
            },
            'l' => match cvtnum("max length", &arg, true) {
                Some(v) => max_length = Some(v as u64),
                None => return Ok(1),
            },
            OPTION_OUTPUT => output_format = parse_output_format(&o.argv0, &arg)?,
            'U' => force_share = true,
            OPTION_OBJECT => object_add(&arg)?,
            _ => return Err(tryhelp(&o.argv0)),
        }
    }
    let rest = o.rest();
    if rest.len() != 1 {
        return Err(error_exit(&o.argv0, "Expecting one image file name"));
    }
    let filename = &rest[0];
    let Some(blk) =
        img_open(image_opts, filename, fmt.as_deref(), Default::default(), false, force_share)
    else {
        return Ok(1);
    };
    let bs = root(&blk);

    let mut out = String::new();
    match output_format {
        OutputFormat::Human => out
            .push_str(&format!("{:<16}{:<16}{:<16}{}\n", "Offset", "Length", "Mapped to", "File")),
        OutputFormat::Json => out.push('['),
    }
    let mut length = match blk.getlength() {
        Ok(l) => l,
        Err(_) => {
            print!("{out}");
            error_report(&format!("Failed to get size for '{filename}'"));
            return Ok(1);
        }
    };
    if let Some(max) = max_length {
        length = length.min(start_offset.saturating_add(max));
    }

    let mut curr = MapEntry { start: start_offset, ..Default::default() };
    let mut failed = false;
    while curr.start + curr.length < length {
        let offset = curr.start + curr.length;
        let mut next = match get_block_status(&bs, offset, length - offset) {
            Ok(e) => e,
            Err(e) => {
                print!("{out}");
                out.clear();
                error_report(&format!("Could not read file metadata: {}", strerror(&e)));
                failed = true;
                break;
            }
        };
        if entry_mergeable(&curr, &next) {
            curr.length += next.length;
            continue;
        }
        if curr.length > 0
            && dump_map_entry(&mut out, output_format, &curr, Some(&mut next)).is_err()
        {
            failed = true;
            break;
        }
        curr = next;
        // Keep stdout and stderr in order for long maps.
        if out.len() > 1 << 16 {
            print!("{out}");
            out.clear();
        }
    }
    if !failed {
        failed = dump_map_entry(&mut out, output_format, &curr, None).is_err();
        if !failed && output_format == OutputFormat::Json {
            out.push_str("]\n");
        }
    }
    print!("{out}");
    Ok(i32::from(failed))
}
