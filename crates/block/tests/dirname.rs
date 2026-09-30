// SPDX-License-Identifier: GPL-2.0-or-later

//! `bdrv_dirname()`: a backing file name relative to the overlay is found next to the
//! overlay's file, also when the overlay is opened from options and its own name is a
//! `json:` pseudo-filename.

#![cfg(unix)]

use std::fs;
use std::path::PathBuf;

use ruvm_block::BlockGraph;
use ruvm_block::tools::OpenFlags;
use ruvm_qapi::{QDict, QValue};

fn scratch(test: &str) -> PathBuf {
    let base = option_env!("CARGO_TARGET_TMPDIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join("ruvm-block-dirname").join(test);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("sub")).unwrap();
    dir
}

/// `base.raw` holding 0x5a bytes, and `sub/top.qed` naming it as `../base.raw`.
fn images(dir: &std::path::Path) -> String {
    fs::write(dir.join("base.raw"), vec![0x5au8; 65536]).unwrap();
    let top = dir.join("sub/top.qed").to_str().unwrap().to_string();
    let mut o = QDict::new();
    o.put("size", "64k");
    o.put("backing_file", "../base.raw");
    o.put("backing_fmt", "raw");
    BlockGraph::new().create_image("qed", &top, &mut o).unwrap();
    top
}

fn reads_backing(g: &BlockGraph, filename: Option<&str>, options: QDict) {
    let blk = g.blk_new_open(filename, options, OpenFlags::default()).unwrap();
    let mut buf = vec![0u8; 512];
    blk.pread(0, &mut buf).unwrap();
    assert!(buf.iter().all(|&b| b == 0x5a));
    let node = blk.node_name().unwrap();
    let info = g.query_block_graph_info(&node, false).unwrap();
    let full = info.full_backing_filename.unwrap();
    assert!(full.ends_with("/sub/../base.raw"), "{full}");
}

#[test]
fn relative_backing_by_file_name() {
    let dir = scratch("file-name");
    let top = images(&dir);
    reads_backing(&BlockGraph::new(), Some(&top), QDict::new());
}

#[test]
fn relative_backing_by_options() {
    let dir = scratch("options");
    let top = images(&dir);
    let mut file = QDict::new();
    file.put("driver", "file");
    file.put("filename", top.as_str());
    let mut o = QDict::new();
    o.put("driver", "qed");
    o.put("file", QValue::Dict(file));
    reads_backing(&BlockGraph::new(), None, o);
}
