// SPDX-License-Identifier: GPL-2.0-or-later

//! Just enough of libfdt's read-write API to build the microvm device tree byte for byte.
//!
//! `create_device_tree()` makes an empty 1 MiB blob and the board then adds nodes and
//! properties in place. libfdt puts a new subnode right after its parent's properties, so in
//! front of older siblings, and a new property right after the node name, so in front of older
//! properties. Property names go into the strings block, reusing any earlier string that ends
//! with the same bytes. The tree here keeps each list in blob order so serializing it gives the
//! same bytes libfdt would leave in the buffer.

const FDT_MAGIC: u32 = 0xd00d_feed;
const FDT_BEGIN_NODE: u32 = 0x1;
const FDT_END_NODE: u32 = 0x2;
const FDT_PROP: u32 = 0x3;
const FDT_END: u32 = 0x9;
/// `FDT_MAX_SIZE` in system/device_tree.c.
pub(super) const FDT_MAX_SIZE: usize = 0x10_0000;
/// `sizeof(struct fdt_header)` rounded up to a reserve entry, `fdt_create()`.
const OFF_MEM_RSVMAP: usize = 0x30;
/// The reserve map holds only its terminating entry.
const OFF_DT_STRUCT: usize = OFF_MEM_RSVMAP + 16;

#[derive(Debug, Default)]
struct Node {
    name: String,
    /// `(name, value)` in blob order.
    props: Vec<(String, Vec<u8>)>,
    /// In blob order.
    children: Vec<Node>,
}

/// A flattened device tree under construction.
#[derive(Debug, Default)]
pub(super) struct Fdt {
    root: Node,
    strings: Vec<u8>,
    next_phandle: u32,
}

impl Fdt {
    /// `create_device_tree()`.
    pub(super) fn new() -> Self {
        Fdt { root: Node::default(), strings: Vec::new(), next_phandle: 0x8000 }
    }

    fn node_mut(&mut self, path: &str) -> &mut Node {
        let mut node = &mut self.root;
        for part in path.split('/').filter(|p| !p.is_empty()) {
            let i = node
                .children
                .iter()
                .position(|c| c.name == part)
                .unwrap_or_else(|| panic!("no FDT node {path}"));
            node = &mut node.children[i];
        }
        node
    }

    /// `qemu_fdt_add_subnode()`.
    pub(super) fn add_subnode(&mut self, path: &str) {
        let (parent, name) = path.rsplit_once('/').expect("absolute FDT path");
        let parent = self.node_mut(parent);
        parent.children.insert(0, Node { name: name.to_string(), ..Node::default() });
    }

    /// `fdt_find_add_string_()`.
    fn add_string(&mut self, s: &str) {
        let mut needle = s.as_bytes().to_vec();
        needle.push(0);
        if !self.strings.windows(needle.len()).any(|w| w == needle.as_slice()) {
            self.strings.extend_from_slice(&needle);
        }
    }

    fn string_offset(&self, s: &str) -> u32 {
        let mut needle = s.as_bytes().to_vec();
        needle.push(0);
        let pos = self.strings.windows(needle.len()).position(|w| w == needle.as_slice());
        pos.expect("property name in the strings block") as u32
    }

    /// `qemu_fdt_setprop()`.
    pub(super) fn setprop(&mut self, path: &str, name: &str, value: &[u8]) {
        let exists = {
            let node = self.node_mut(path);
            if let Some(p) = node.props.iter_mut().find(|p| p.0 == name) {
                p.1 = value.to_vec();
                true
            } else {
                false
            }
        };
        if !exists {
            self.add_string(name);
            let node = self.node_mut(path);
            node.props.insert(0, (name.to_string(), value.to_vec()));
        }
    }

    /// `qemu_fdt_setprop_string()`.
    pub(super) fn setprop_string(&mut self, path: &str, name: &str, value: &str) {
        let mut v = value.as_bytes().to_vec();
        v.push(0);
        self.setprop(path, name, &v);
    }

    /// `qemu_fdt_setprop_cell()`.
    pub(super) fn setprop_cell(&mut self, path: &str, name: &str, value: u32) {
        self.setprop(path, name, &value.to_be_bytes());
    }

    /// `qemu_fdt_setprop_cells()`.
    pub(super) fn setprop_cells(&mut self, path: &str, name: &str, values: &[u32]) {
        let v: Vec<u8> = values.iter().flat_map(|c| c.to_be_bytes()).collect();
        self.setprop(path, name, &v);
    }

    /// `qemu_fdt_setprop_sized_cells(fdt, path, name, 2, base, 2, size)`.
    pub(super) fn setprop_reg64(&mut self, path: &str, name: &str, base: u64, size: u64) {
        let mut v = base.to_be_bytes().to_vec();
        v.extend_from_slice(&size.to_be_bytes());
        self.setprop(path, name, &v);
    }

    /// `qemu_fdt_alloc_phandle()` with no `phandle-start`.
    pub(super) fn alloc_phandle(&mut self) -> u32 {
        let p = self.next_phandle;
        self.next_phandle += 1;
        p
    }

    fn put_node(&self, node: &Node, out: &mut Vec<u8>) {
        out.extend_from_slice(&FDT_BEGIN_NODE.to_be_bytes());
        out.extend_from_slice(node.name.as_bytes());
        out.push(0);
        pad4(out);
        for (name, value) in &node.props {
            out.extend_from_slice(&FDT_PROP.to_be_bytes());
            out.extend_from_slice(&(value.len() as u32).to_be_bytes());
            out.extend_from_slice(&self.string_offset(name).to_be_bytes());
            out.extend_from_slice(value);
            pad4(out);
        }
        for child in &node.children {
            self.put_node(child, out);
        }
        out.extend_from_slice(&FDT_END_NODE.to_be_bytes());
    }

    /// The whole [`FDT_MAX_SIZE`] buffer, as `fw_cfg_add_file(..., "etc/fdt", fdt, size)` sees
    /// it.
    pub(super) fn to_blob(&self) -> Vec<u8> {
        let mut dt_struct = Vec::new();
        self.put_node(&self.root, &mut dt_struct);
        dt_struct.extend_from_slice(&FDT_END.to_be_bytes());

        let off_strings = OFF_DT_STRUCT + dt_struct.len();
        let header = [
            FDT_MAGIC,
            FDT_MAX_SIZE as u32,
            OFF_DT_STRUCT as u32,
            off_strings as u32,
            OFF_MEM_RSVMAP as u32,
            17, // version
            16, // last_comp_version
            0,  // boot_cpuid_phys
            self.strings.len() as u32,
            dt_struct.len() as u32,
        ];
        let mut out: Vec<u8> = header.iter().flat_map(|w| w.to_be_bytes()).collect();
        out.resize(OFF_DT_STRUCT, 0);
        out.extend_from_slice(&dt_struct);
        out.extend_from_slice(&self.strings);
        assert!(out.len() <= FDT_MAX_SIZE, "device tree too large");
        out.resize(FDT_MAX_SIZE, 0);
        out
    }
}

fn pad4(out: &mut Vec<u8>) {
    while out.len() % 4 != 0 {
        out.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tree_matches_create_device_tree() {
        let blob = Fdt::new().to_blob();
        assert_eq!(blob.len(), FDT_MAX_SIZE);
        assert_eq!(&blob[0..4], &FDT_MAGIC.to_be_bytes());
        // BEGIN_NODE "" END_NODE END
        assert_eq!(&blob[0x40..0x50], &[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 9]);
        let word = |o: usize| u32::from_be_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]]);
        assert_eq!(word(12), 0x50); // off_dt_strings
        assert_eq!(word(36), 16); // size_dt_struct
    }

    #[test]
    fn newest_first_and_suffix_strings() {
        let mut f = Fdt::new();
        f.add_subnode("/a");
        f.add_subnode("/b");
        f.setprop_cell("/a", "linux,phandle", 1);
        f.setprop_cell("/a", "phandle", 1);
        assert_eq!(f.root.children[0].name, "b");
        assert_eq!(f.root.children[1].props[0].0, "phandle");
        assert_eq!(f.strings, b"linux,phandle\0");
        assert_eq!(f.string_offset("phandle"), 6);
    }
}
