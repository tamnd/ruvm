// SPDX-License-Identifier: GPL-2.0-or-later

//! Just enough of libfdt's read-write API to build the microvm device tree byte for byte.
//!
//! `create_device_tree()` makes an empty 1 MiB blob and the board then adds nodes and
//! properties in place. This keeps the same flat buffer and edits it the way libfdt does:
//! a new subnode goes right after its parent's properties (so in front of older siblings), a
//! new property right after the node name (so in front of older properties), and property
//! names go into the strings block, reusing any earlier string that ends with the same bytes.
//! Working on the buffer rather than on a tree matters for one detail: libfdt opens a gap with
//! `memmove()` and only copies the value into it, so the alignment padding after a property
//! keeps whatever bytes were there before. QEMU's `etc/fdt` has those bytes, and so does ours.

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

/// A flattened device tree under construction.
pub(super) struct Fdt {
    /// The whole [`FDT_MAX_SIZE`] buffer; the header fields below are written out by
    /// [`Fdt::to_blob`].
    buf: Vec<u8>,
    off_dt_strings: usize,
    size_dt_struct: usize,
    size_dt_strings: usize,
    next_phandle: u32,
}

impl std::fmt::Debug for Fdt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fdt")
            .field("size_dt_struct", &self.size_dt_struct)
            .field("size_dt_strings", &self.size_dt_strings)
            .finish_non_exhaustive()
    }
}

fn tag_align(n: usize) -> usize {
    (n + 3) & !3
}

impl Fdt {
    /// `create_device_tree()`: a root node and nothing else.
    pub(super) fn new() -> Self {
        let mut buf = vec![0; FDT_MAX_SIZE];
        let root = [FDT_BEGIN_NODE, 0, FDT_END_NODE, FDT_END];
        for (i, w) in root.iter().enumerate() {
            buf[OFF_DT_STRUCT + i * 4..OFF_DT_STRUCT + i * 4 + 4].copy_from_slice(&w.to_be_bytes());
        }
        Fdt {
            buf,
            off_dt_strings: OFF_DT_STRUCT + 16,
            size_dt_struct: 16,
            size_dt_strings: 0,
            next_phandle: 0x8000,
        }
    }

    fn word(&self, off: usize) -> u32 {
        let b = &self.buf[off..off + 4];
        u32::from_be_bytes([b[0], b[1], b[2], b[3]])
    }

    fn put_word(&mut self, off: usize, v: u32) {
        self.buf[off..off + 4].copy_from_slice(&v.to_be_bytes());
    }

    /// `fdt_next_tag()`: the tag at `off` (an offset into the buffer) and where the next one
    /// starts.
    fn next_tag(&self, off: usize) -> (u32, usize) {
        let tag = self.word(off);
        let next = match tag {
            FDT_BEGIN_NODE => {
                let name = &self.buf[off + 4..];
                let len = name.iter().position(|&b| b == 0).expect("node name") + 1;
                off + 4 + tag_align(len)
            }
            FDT_PROP => off + 12 + tag_align(self.word(off + 4) as usize),
            _ => off + 4,
        };
        (tag, next)
    }

    fn node_name(&self, off: usize) -> &[u8] {
        let name = &self.buf[off + 4..];
        &name[..name.iter().position(|&b| b == 0).expect("node name")]
    }

    /// Where the node's properties end, which is where its first subnode (or its end tag)
    /// starts.
    fn after_props(&self, node: usize) -> usize {
        let (_, mut off) = self.next_tag(node);
        loop {
            let (tag, next) = self.next_tag(off);
            if tag != FDT_PROP {
                return off;
            }
            off = next;
        }
    }

    /// `fdt_subnode_offset()`.
    fn subnode(&self, parent: usize, name: &str) -> Option<usize> {
        let mut off = self.after_props(parent);
        loop {
            let (tag, next) = self.next_tag(off);
            if tag != FDT_BEGIN_NODE {
                return None;
            }
            if self.node_name(off) == name.as_bytes() {
                return Some(off);
            }
            // Skip the whole subtree.
            let mut depth = 1;
            off = next;
            while depth > 0 {
                let (tag, next) = self.next_tag(off);
                match tag {
                    FDT_BEGIN_NODE => depth += 1,
                    FDT_END_NODE => depth -= 1,
                    _ => {}
                }
                off = next;
            }
        }
    }

    /// `fdt_path_offset()`.
    fn node(&self, path: &str) -> usize {
        let mut node = OFF_DT_STRUCT;
        for part in path.split('/').filter(|p| !p.is_empty()) {
            node = self.subnode(node, part).unwrap_or_else(|| panic!("no FDT node {path}"));
        }
        node
    }

    /// `fdt_splice_()`: moves everything from `p + oldlen` to the end of the strings block
    /// so that `oldlen` bytes at `p` become `newlen`. The bytes of a new gap are left as they
    /// were.
    fn splice(&mut self, p: usize, oldlen: usize, newlen: usize) {
        let end = self.off_dt_strings + self.size_dt_strings;
        assert!(end - oldlen + newlen <= FDT_MAX_SIZE, "device tree too large");
        self.buf.copy_within(p + oldlen..end, p + newlen);
    }

    /// `fdt_splice_struct_()`.
    fn splice_struct(&mut self, p: usize, oldlen: usize, newlen: usize) {
        self.splice(p, oldlen, newlen);
        self.size_dt_struct = self.size_dt_struct + newlen - oldlen;
        self.off_dt_strings = self.off_dt_strings + newlen - oldlen;
    }

    /// `fdt_find_add_string_()`.
    fn find_add_string(&mut self, s: &str) -> u32 {
        let mut needle = s.as_bytes().to_vec();
        needle.push(0);
        let strings = &self.buf[self.off_dt_strings..self.off_dt_strings + self.size_dt_strings];
        if let Some(pos) = strings.windows(needle.len()).position(|w| w == needle.as_slice()) {
            return pos as u32;
        }
        let off = self.size_dt_strings;
        let p = self.off_dt_strings + off;
        self.splice(p, 0, needle.len());
        self.buf[p..p + needle.len()].copy_from_slice(&needle);
        self.size_dt_strings += needle.len();
        off as u32
    }

    fn string_at(&self, off: u32) -> &[u8] {
        let s = &self.buf[self.off_dt_strings + off as usize..];
        &s[..s.iter().position(|&b| b == 0).expect("string")]
    }

    /// `qemu_fdt_add_subnode()`.
    pub(super) fn add_subnode(&mut self, path: &str) {
        let (parent, name) = path.rsplit_once('/').expect("absolute FDT path");
        let parent = self.node(parent);
        assert!(self.subnode(parent, name).is_none(), "FDT node {path} exists");
        let off = self.after_props(parent);
        let namelen = tag_align(name.len() + 1);
        self.splice_struct(off, 0, 4 + namelen + 4);
        self.put_word(off, FDT_BEGIN_NODE);
        self.buf[off + 4..off + 4 + namelen].fill(0);
        self.buf[off + 4..off + 4 + name.len()].copy_from_slice(name.as_bytes());
        self.put_word(off + 4 + namelen, FDT_END_NODE);
    }

    /// `qemu_fdt_setprop()`, which is `fdt_setprop()`: resizes the property if the node has
    /// it, adds it in front of the others if not, then copies the value in.
    pub(super) fn setprop(&mut self, path: &str, name: &str, value: &[u8]) {
        let node = self.node(path);
        let (_, mut off) = self.next_tag(node);
        let mut found = None;
        loop {
            let (tag, next) = self.next_tag(off);
            if tag != FDT_PROP {
                break;
            }
            if self.string_at(self.word(off + 8)) == name.as_bytes() {
                found = Some(off);
                break;
            }
            off = next;
        }
        let prop = match found {
            // fdt_resize_property_()
            Some(prop) => {
                let oldlen = self.word(prop + 4) as usize;
                self.splice_struct(prop + 12, tag_align(oldlen), tag_align(value.len()));
                prop
            }
            // fdt_add_property_()
            None => {
                let nameoff = self.find_add_string(name);
                let (_, prop) = self.next_tag(self.node(path));
                self.splice_struct(prop, 0, 12 + tag_align(value.len()));
                self.put_word(prop, FDT_PROP);
                self.put_word(prop + 8, nameoff);
                prop
            }
        };
        self.put_word(prop + 4, value.len() as u32);
        self.buf[prop + 12..prop + 12 + value.len()].copy_from_slice(value);
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

    /// The whole [`FDT_MAX_SIZE`] buffer, as `fw_cfg_add_file(..., "etc/fdt", fdt, size)` sees
    /// it.
    pub(super) fn to_blob(&self) -> Vec<u8> {
        let header = [
            FDT_MAGIC,
            FDT_MAX_SIZE as u32,
            OFF_DT_STRUCT as u32,
            self.off_dt_strings as u32,
            OFF_MEM_RSVMAP as u32,
            17, // version
            16, // last_comp_version
            0,  // boot_cpuid_phys
            self.size_dt_strings as u32,
            self.size_dt_struct as u32,
        ];
        let mut out = self.buf.clone();
        for (i, w) in header.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
        }
        out
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
        let blob = f.to_blob();
        let root = OFF_DT_STRUCT;
        // "b" comes first, and "phandle" sits in front of "linux,phandle".
        let b = f.after_props(root);
        assert_eq!(f.node_name(b), b"b");
        let a = f.subnode(root, "a").unwrap();
        let (_, first) = f.next_tag(a);
        assert_eq!(f.string_at(f.word(first + 8)), b"phandle");
        assert_eq!(f.word(first + 8), 6);
        assert_eq!(&blob[f.off_dt_strings..f.off_dt_strings + 14], b"linux,phandle\0");
    }

    #[test]
    fn padding_keeps_the_old_bytes() {
        // The gap for "q" opens where "p" was, and only one byte of it is written, so the
        // padding keeps the last three bytes of the old value of "p".
        let mut f = Fdt::new();
        f.add_subnode("/a");
        f.setprop("/a", "p", &[1, 2, 3, 4]);
        f.setprop("/a", "q", &[0xaa]);
        let a = f.subnode(OFF_DT_STRUCT, "a").unwrap();
        let (_, prop) = f.next_tag(a);
        assert_eq!(f.string_at(f.word(prop + 8)), b"q");
        assert_eq!(&f.buf[prop + 12..prop + 16], &[0xaa, 2, 3, 4]);
    }
}
