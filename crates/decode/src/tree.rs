// SPDX-License-Identifier: GPL-2.0-or-later

//! Building the decision tree, following `prop_masks`, `build_tree`, `prop_format`,
//! `prop_width`, `build_size_tree` and `prop_size` in scripts/decodetree.py. The order in which
//! patterns are binned and the masks chosen at each level are the script's, so the generated
//! checks run in the same order as QEMU's.

use crate::Error;
use crate::model::{Model, Node, Tree, TreeSub};
use crate::parse::Ctx;

/// `MultiPattern.prop_masks`, recursively for every group below `g`.
pub(crate) fn prop_masks(m: &mut Model, ctx: &Ctx, g: usize) {
    let mut fixedmask = ctx.insnmask;
    let mut undefmask = ctx.insnmask;
    let pats = m.groups[g].pats.clone();
    for &p in &pats {
        if let Node::Group(sub) = p {
            prop_masks(m, ctx, sub);
        }
        let (_, fm) = m.node_fixed(p);
        let um = match p {
            Node::Pattern(i) => m.patterns[i].undefmask,
            Node::Group(i) => m.groups[i].undefmask,
        };
        fixedmask &= fm;
        undefmask &= um;
    }

    // Widen fixedmask until all fixedbits match. As in the script, when the mask runs out the
    // bits keep whatever the last pass computed.
    let mut repeat = true;
    let mut fixedbits: Option<u64> = Some(0);
    while repeat && fixedmask != 0 {
        fixedbits = None;
        let mut broke = false;
        for &p in &pats {
            let thisbits = m.node_fixed(p).0 & fixedmask;
            match fixedbits {
                None => fixedbits = Some(thisbits),
                Some(fb) if fb != thisbits => {
                    fixedmask &= !(fb ^ thisbits);
                    broke = true;
                    break;
                }
                Some(_) => {}
            }
        }
        if !broke {
            repeat = false;
        }
    }

    let grp = &mut m.groups[g];
    // An empty group leaves the script's fixedbits at None. Nothing reads it before the empty
    // group is reported, except the parent's binning, where zero is what None would have to be.
    grp.fixedbits = fixedbits.unwrap_or(0);
    grp.fixedmask = fixedmask;
    grp.undefmask = undefmask;
}

/// `build_tree` for group `g` and every group below it.
pub(crate) fn build_tree(m: &mut Model, ctx: &Ctx, g: usize) -> Result<(), Error> {
    if m.groups[g].overlapping && m.groups[g].pats.is_empty() {
        let grp = &m.groups[g];
        return Err(Error {
            file: grp.file.clone(),
            line: grp.lineno,
            message: "empty pattern group".into(),
        });
    }
    let pats = m.groups[g].pats.clone();
    for &p in &pats {
        if let Node::Group(sub) = p {
            build_tree(m, ctx, sub)?;
        }
    }
    if !m.groups[g].overlapping {
        let mask = m.groups[g].fixedmask;
        let tree = exc_build_tree(m, ctx, &pats, mask)?;
        m.groups[g].tree = Some(tree);
    }
    Ok(())
}

fn node_str(m: &Model, ctx: &Ctx, n: Node) -> String {
    match n {
        Node::Pattern(p) => {
            let p = &m.patterns[p];
            format!("{} {}", p.name, ctx.str_match_bits(p.fixedbits, p.fixedmask))
        }
        Node::Group(g) => {
            let g = &m.groups[g];
            format!("group {}", ctx.str_match_bits(g.fixedbits, g.fixedmask))
        }
    }
}

/// `ExcMultiPattern.__build_tree`. The script also passes the outer fixed bits down, but never
/// reads them.
fn exc_build_tree(m: &Model, ctx: &Ctx, pats: &[Node], outermask: u64) -> Result<Tree, Error> {
    let mut innermask = !outermask & ctx.insnmask;
    for &p in pats {
        innermask &= m.node_fixed(p).1;
    }

    if innermask == 0 {
        if pats.len() == 1 {
            return Ok(Tree {
                thismask: innermask,
                subs: vec![(0, TreeSub::Node(pats[0]))],
                base: None,
            });
        }
        let mut text = String::from("overlapping patterns:");
        for &p in pats {
            let (file, line) = m.node_file_line(p);
            text += &format!("\n{file}:{line}: {}", node_str(m, ctx, p));
        }
        let (file, line) = m.node_file_line(pats[0]);
        return Err(Error { file: file.to_string(), line, message: text });
    }

    let fullmask = outermask | innermask;

    // Bins in insertion order, as a Python dict iterates.
    let mut bins: Vec<(u64, Vec<Node>)> = Vec::new();
    for &p in pats {
        let fb = m.node_fixed(p).0 & innermask;
        match bins.iter_mut().find(|(b, _)| *b == fb) {
            Some((_, l)) => l.push(p),
            None => bins.push((fb, vec![p])),
        }
    }

    let mut t = Tree { thismask: innermask, subs: Vec::new(), base: None };
    for (b, l) in bins {
        let s = l[0];
        let sub = if l.len() > 1 || m.node_fixed(s).1 & !fullmask != 0 {
            TreeSub::Tree(Box::new(exc_build_tree(m, ctx, &l, fullmask)?))
        } else {
            TreeSub::Node(s)
        };
        t.subs.push((b, sub));
    }
    Ok(t)
}

/// `prop_format` for group `g` and every group below it.
pub(crate) fn prop_format(m: &mut Model, g: usize) {
    let pats = m.groups[g].pats.clone();
    for p in pats {
        if let Node::Group(sub) = p {
            prop_format(m, sub);
        }
    }
    if let Some(mut tree) = m.groups[g].tree.take() {
        prop_format_tree(m, &mut tree);
        m.groups[g].tree = Some(tree);
    }
}

fn prop_format_tree(m: &Model, tree: &mut Tree) {
    for (_, s) in &mut tree.subs {
        if let TreeSub::Tree(t) = s {
            prop_format_tree(m, t);
        }
    }
    let mut f: Option<usize> = None;
    for (_, s) in &tree.subs {
        let base = match s {
            TreeSub::Tree(t) => t.base,
            TreeSub::Node(Node::Pattern(p)) => Some(m.patterns[*p].base),
            TreeSub::Node(Node::Group(_)) => None,
        };
        match f {
            None => {
                f = base;
                if f.is_none() {
                    return;
                }
            }
            Some(have) => {
                if base != Some(have) {
                    return;
                }
            }
        }
    }
    tree.base = f;
}

fn node_width(m: &Model, n: Node) -> Option<u32> {
    match n {
        Node::Pattern(p) => Some(m.patterns[p].width),
        Node::Group(g) => m.groups[g].width,
    }
}

/// `MultiPattern.prop_width` for group `g`.
pub(crate) fn prop_width(m: &mut Model, g: usize) -> Result<(), Error> {
    let pats = m.groups[g].pats.clone();
    // The script starts from None and takes the first child's width while it is still None,
    // so a leading child without a width does not count, but a later one is a mismatch.
    let mut width: Option<u32> = None;
    for p in pats {
        if let Node::Group(sub) = p {
            prop_width(m, sub)?;
        }
        let w = node_width(m, p);
        match width {
            None => width = w,
            Some(have) if Some(have) != w => {
                let grp = &m.groups[g];
                return Err(Error {
                    file: grp.file.clone(),
                    line: grp.lineno,
                    message: "width mismatch in patterns within braces".into(),
                });
            }
            Some(_) => {}
        }
    }
    m.groups[g].width = width;
    Ok(())
}

/// The tree `decode_load` walks to find how many bytes an instruction has.
#[derive(Debug)]
pub(crate) enum SizeTree {
    Node { mask: u64, width: u32, subs: Vec<(u64, SizeTree)> },
    Leaf { width: u32 },
}

impl SizeTree {
    fn width(&self) -> u32 {
        match self {
            SizeTree::Node { width, .. } | SizeTree::Leaf { width } => *width,
        }
    }
}

/// `build_size_tree`.
pub(crate) fn build_size_tree(
    m: &Model,
    ctx: &Ctx,
    pats: &[Node],
    width: u32,
    outerbits: u64,
    outermask: u64,
) -> Result<SizeTree, Error> {
    let mut innermask: u64 = if width <= ctx.insnwidth {
        (0xffu64 << (ctx.insnwidth - width)) & !outermask
    } else {
        (0xffu64 >> (width - ctx.insnwidth)) & !outermask
    };
    let mut minwidth: Option<u32> = None;
    let mut onewidth = true;
    for &p in pats {
        innermask &= m.node_fixed(p).1;
        // A group whose width is None compares as unequal to any number, and Python refuses
        // to order None against an int; neither happens for a parsed file, since every group
        // got a width from prop_width.
        let w = node_width(m, p).unwrap_or(0);
        match minwidth {
            None => minwidth = Some(w),
            Some(mw) if mw != w => {
                onewidth = false;
                if mw < w {
                    minwidth = Some(w);
                }
            }
            Some(_) => {}
        }
    }
    let minwidth = minwidth.unwrap_or(0);

    if onewidth {
        return Ok(SizeTree::Leaf { width: minwidth });
    }

    if innermask == 0 {
        if width < minwidth {
            return build_size_tree(m, ctx, pats, width + 8, outerbits, outermask);
        }
        let mut pnames = Vec::new();
        for &p in pats {
            let (name, file, line) = match p {
                Node::Pattern(i) => {
                    let pt = &m.patterns[i];
                    (pt.name.clone(), pt.file.clone(), pt.lineno)
                }
                // The script reads `.name` here, which a group does not have.
                Node::Group(i) => {
                    let gr = &m.groups[i];
                    ("group".to_string(), gr.file.clone(), gr.lineno)
                }
            };
            pnames.push(format!("'{name}:{file}:{line}'"));
        }
        let (file, line) = m.node_file_line(pats[0]);
        return Err(Error {
            file: file.to_string(),
            line,
            message: format!("overlapping patterns size {width}: [{}]", pnames.join(", ")),
        });
    }

    let mut bins: Vec<(u64, Vec<Node>)> = Vec::new();
    for &p in pats {
        let fb = m.node_fixed(p).0 & innermask;
        match bins.iter_mut().find(|(b, _)| *b == fb) {
            Some((_, l)) => l.push(p),
            None => bins.push((fb, vec![p])),
        }
    }

    let fullmask = outermask | innermask;
    if bins.len() == 1 {
        let (b, l) = &bins[0];
        return build_size_tree(m, ctx, l, width + 8, b | outerbits, fullmask);
    }

    let mut subs = Vec::new();
    for (b, l) in &bins {
        let s = build_size_tree(m, ctx, l, width, b | outerbits, fullmask)?;
        subs.push((*b, s));
    }
    Ok(SizeTree::Node { mask: innermask, width, subs })
}

/// `prop_size`: a node needs no more bytes than its smallest child.
pub(crate) fn prop_size(tree: &mut SizeTree) -> u32 {
    match tree {
        SizeTree::Node { width, subs, .. } => {
            let mut min: Option<u32> = None;
            for (_, s) in subs.iter_mut() {
                let w = prop_size(s);
                if min.is_none_or(|m| m > w) {
                    min = Some(w);
                }
            }
            let min = min.unwrap_or(*width);
            *width = min;
            min
        }
        SizeTree::Leaf { width } => *width,
    }
}

impl SizeTree {
    pub(crate) fn node_width(&self) -> u32 {
        self.width()
    }
}
