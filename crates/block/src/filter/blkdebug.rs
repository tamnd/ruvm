// SPDX-License-Identifier: GPL-2.0-or-later

//! `blkdebug` from block/blkdebug.c: a filter that injects errors and changes state when the
//! driver above it reports debug events, and that can impose request limits on the node.
//!
//! Rules come from the config file (`config`) and from the `inject-error` and `set-state`
//! options, in that order. They are kept per event, newest first, as QEMU's lists are; an
//! event activates its inject-error rules for the current state, and the first active rule
//! that matches a request fails it.
//!
//! Differences from QEMU:
//!
//! - Requests are synchronous, so a suspended request (a `break` breakpoint) blocks its
//!   thread until another thread resumes it. `immediately` makes no difference: there is no
//!   coroutine to reschedule before failing.
//! - `delay-ns` sleeps the calling thread.
//! - Duplicate section ids in the config file are not diagnosed.
//! - The image in a `blkdebug:config:image` file name becomes `image` options for the
//!   protocol driver the name picks, instead of QEMU's internal `x-image` option. QEMU probes
//!   the format of that image when blkdebug is not itself opened as a protocol; here it is
//!   always opened without probing.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use ruvm_base::{Error, Result};
use ruvm_qapi::cutils::{Errno, bool_parse, strtou64};
use ruvm_qapi::types::{
    BlkdebugEvent, BlkdebugIOType, BlkdebugInjectErrorOptions, BlkdebugSetStateOptions,
    BlockPermission, BlockdevOptionsBlkdebug, BlockdevOptionsU,
};
use ruvm_qapi::{QDict, QValue};

use crate::drivers::{self, DriverDef, OpenArgs};
use crate::node::{
    BDRV_BLOCK_OFFSET_VALID, BDRV_BLOCK_RAW, BDRV_CHILD_FILTERED, BDRV_CHILD_PRIMARY, BDRV_REQ_FUA,
    BDRV_REQ_MAY_UNMAP, BDRV_REQ_NO_FALLBACK, BDRV_REQ_WRITE_UNCHANGED, BlockLimits, BlockStatus,
    Driver, Node, ReopenState, errno,
};
use crate::open::join;
use crate::perm::{
    BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE, BLK_PERM_WRITE_UNCHANGED, PermCtx,
    default_perms,
};

/// `bdrv_blkdebug`.
pub(crate) static BLKDEBUG: DriverDef = DriverDef::filter("blkdebug", blkdebug_open)
    .with_protocol("blkdebug")
    .with_parse_filename(blkdebug_parse_filename)
    .with_strong_opts(&[
        "config",
        "inject-error.",
        "set-state.",
        "align",
        "max-transfer",
        "opt-write-zero",
        "max-write-zero",
        "opt-discard",
        "max-discard",
    ])
    .with_size_opts(&[
        "align",
        "max-transfer",
        "opt-write-zero",
        "max-write-zero",
        "opt-discard",
        "max-discard",
    ]);

const INT_MAX: u64 = i32::MAX as u64;

#[derive(Clone, Debug)]
enum Action {
    InjectError {
        iotype_mask: u64,
        error: i32,
        once: bool,
        /// Byte offset, or -1 for any.
        offset: i64,
        delay_ns: i64,
    },
    SetState {
        new_state: i64,
    },
    Suspend {
        tag: String,
    },
}

/// `BlkdebugRule`.
#[derive(Clone, Debug)]
struct Rule {
    id: u64,
    state: i64,
    action: Action,
}

/// The mutable part of `BDRVBlkdebugState`.
#[derive(Debug, Default)]
struct State {
    state: i64,
    /// `rules[event]`, newest first.
    rules: Vec<Vec<Rule>>,
    /// Ids of the active inject-error rules, in match order.
    active: Vec<u64>,
    /// Suspended requests: (tag, ticket), newest first.
    suspended: Vec<(String, u64)>,
    next_id: u64,
}

impl State {
    fn add(&mut self, event: BlkdebugEvent, state: i64, action: Action) {
        self.next_id += 1;
        let r = Rule { id: self.next_id, state, action };
        self.rules[event as usize].insert(0, r);
    }

    fn remove(&mut self, id: u64) {
        for l in &mut self.rules {
            l.retain(|r| r.id != id);
        }
        self.active.retain(|&a| a != id);
    }

    fn find(&self, id: u64) -> Option<&Rule> {
        self.rules.iter().flatten().find(|r| r.id == id)
    }
}

/// `BDRVBlkdebugState`.
#[derive(Debug)]
pub(crate) struct BlkdebugDriver {
    align: u64,
    max_transfer: u64,
    opt_write_zero: u64,
    max_write_zero: u64,
    opt_discard: u64,
    max_discard: u64,
    config_file: Option<String>,
    take_child_perms: u64,
    unshare_child_perms: u64,
    /// Whether the node was opened with nothing but a config file and an image, the options
    /// `blkdebug_refresh_filename()` lets through.
    plain: bool,
    st: Mutex<State>,
    resumed: Condvar,
}

/// `blkdebug_parse_filename()`: `blkdebug:[config]:image`, where the image part becomes the
/// file name of the `image` child.
fn blkdebug_parse_filename(filename: &str, options: &mut QDict) -> Result<()> {
    let Some(rest) = filename.strip_prefix("blkdebug:") else {
        return put_filename_child(options, "image", filename);
    };
    let Some(c) = rest.find(':') else {
        return Err(Error::generic("blkdebug requires both config file and image path"));
    };
    if c != 0 {
        options.put("config", &rest[..c]);
    }
    put_filename_child(options, "image", &rest[c + 1..])
}

/// What `bdrv_open_child()` makes of a file name for the child `key`, which QEMU passes as
/// `x-image` and similar: the options of the protocol node the name opens, put under `key`.
/// Options given for the child explicitly win over the ones from the name.
pub(crate) fn put_filename_child(options: &mut QDict, key: &str, filename: &str) -> Result<()> {
    let drv = drivers::find_protocol(filename, true)?;
    let mut child = QDict::new();
    child.put("driver", drv.format_name);
    child.put("filename", filename);
    if let Some(pf) = drv.parse_filename {
        pf(filename, &mut child)?;
        if !drv.needs_filename {
            child.remove("filename");
        }
    }
    // A format driver given for the child opens its own `file` from the name, the way
    // `bdrv_open_inherit()` does with a file name and a format driver.
    let format = match options.get(key) {
        Some(QValue::Dict(d)) => d
            .get_str("driver")
            .and_then(drivers::find_format)
            .is_some_and(|d| d.protocol_name.is_none()),
        _ => false,
    };
    if format {
        let mut f = QDict::new();
        f.put("file", child);
        child = f;
    }
    let mut src = QDict::new();
    src.put(key, child);
    join(options, src);
    Ok(())
}

/// `bdrv_qapi_perm_to_blk_perm()`.
fn qapi_perm(p: BlockPermission) -> u64 {
    match p {
        BlockPermission::ConsistentRead => BLK_PERM_CONSISTENT_READ,
        BlockPermission::Write => BLK_PERM_WRITE,
        BlockPermission::WriteUnchanged => BLK_PERM_WRITE_UNCHANGED,
        BlockPermission::Resize => BLK_PERM_RESIZE,
    }
}

/// One `[group]` of the config file: its name and its `key = "value"` pairs in order.
type Section = (String, Vec<(String, String)>);

/// `qemu_config_parse()` for the two blkdebug groups.
fn parse_config_file(path: &str) -> Result<Vec<Section>> {
    // fopen() blocks on a fifo until a writer shows up, and the QMP out-of-band test relies
    // on that. File::open() behaves the same way.
    let f = File::open(path).map_err(|e| Error::from_io(format!("Could not open '{path}'"), e))?;
    parse_config(BufReader::new(f))
}

/// `qemu_config_foreach()` with `qemu_config_do_parse()`.
fn parse_config(r: impl BufRead) -> Result<Vec<Section>> {
    let mut out: Vec<Section> = Vec::new();
    let mut cur: Option<Section> = None;
    for line in r.lines() {
        let line = line.map_err(|e| Error::from_io("Cannot read config file", e))?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(inner) = line.strip_prefix('[') {
            if let Some(g) = parse_group(inner) {
                if let Some(prev) = cur.take() {
                    check_group(&prev.0)?;
                    out.push(prev);
                }
                let mut s = (g.0, Vec::new());
                if let Some(id) = g.1 {
                    s.1.push(("id".to_string(), id));
                }
                cur = Some(s);
                continue;
            }
        }
        match parse_assignment(&line) {
            Some((k, v)) => match &mut cur {
                Some(s) => s.1.push((k, v)),
                None => return Err(Error::generic("no group defined")),
            },
            None => return Err(Error::generic("parse error")),
        }
    }
    if let Some(s) = cur {
        check_group(&s.0)?;
        out.push(s);
    }
    Ok(out)
}

fn check_group(g: &str) -> Result<()> {
    if g == "inject-error" || g == "set-state" {
        Ok(())
    } else {
        Err(Error::generic(format!("There is no option group '{g}'")))
    }
}

/// `[group "id"]` or `[group]`, after the `[`.
fn parse_group(s: &str) -> Option<(String, Option<String>)> {
    let s = s.trim_end_matches(['\n', '\r']);
    // sscanf("[%63s \"%63[^\"]\"]"): a word, blanks, then a quoted id.
    let word_end = s.find(char::is_whitespace);
    if let Some(we) = word_end {
        let rest = s[we..].trim_start();
        if let Some(q) = rest.strip_prefix('"') {
            if let Some(end) = q.find('"') {
                if end > 0 {
                    return Some((s[..we].to_string(), Some(q[..end].to_string())));
                }
            }
        }
    }
    // sscanf("[%63[^]]]"): anything up to the first `]`, which must not be empty.
    let end = s.find(']').unwrap_or(s.len());
    if end == 0 {
        return None;
    }
    Some((s[..end].to_string(), None))
}

/// ` name = "value"` or ` name = ""`.
fn parse_assignment(line: &str) -> Option<(String, String)> {
    let s = line.trim_start();
    let end = s.find(char::is_whitespace).unwrap_or(s.len());
    let name = &s[..end];
    if name.is_empty() {
        return None;
    }
    let rest = s[end..].trim_start().strip_prefix('=')?.trim_start().strip_prefix('"')?;
    let value = &rest[..rest.find('"')?];
    Some((name.to_string(), value.to_string()))
}

fn opt_number(name: &str, v: &str) -> Result<u64> {
    match strtou64(v, 0, true) {
        Ok((n, _)) => Ok(n),
        Err((Errno::Range, _)) => {
            Err(Error::generic(format!("Value '{v}' is too large for parameter '{name}'")))
        }
        Err(_) => Err(Error::generic(format!("Parameter '{name}' expects a number"))),
    }
}

fn opt_bool(name: &str, v: &str) -> Result<bool> {
    bool_parse(v).ok_or_else(|| Error::generic(format!("Parameter '{name}' expects 'on' or 'off'")))
}

fn opt_event(v: Option<&str>) -> Result<BlkdebugEvent> {
    let Some(v) = v else {
        return Err(Error::generic("Missing event name for rule"));
    };
    BlkdebugEvent::from_name(v)
        .ok_or_else(|| Error::generic(format!("invalid parameter value: {v}")))
}

const DEFAULT_IOTYPES: u64 = (1 << BlkdebugIOType::Read as u64)
    | (1 << BlkdebugIOType::Write as u64)
    | (1 << BlkdebugIOType::WriteZeroes as u64)
    | (1 << BlkdebugIOType::Discard as u64)
    | (1 << BlkdebugIOType::Flush as u64);

/// `add_rule()` for a config file section.
fn add_section(st: &mut State, group: &str, kv: &[(String, String)]) -> Result<()> {
    let inject = group == "inject-error";
    let allowed: &[&str] = if inject {
        &["event", "state", "iotype", "errno", "sector", "once", "immediately", "delay-ns"]
    } else {
        &["event", "state", "new_state"]
    };
    let mut event = None;
    let (mut state, mut iotype, mut err, mut sector) = (0u64, None, libc::EIO as u64, u64::MAX);
    let (mut once, mut delay, mut new_state) = (false, 0u64, 0u64);
    // qemu_opts_from_qdict() validates every option before add_rule() looks at them.
    for (k, v) in kv {
        if k == "id" {
            continue;
        }
        if !allowed.contains(&k.as_str()) {
            return Err(Error::generic(format!("Invalid parameter '{k}'")));
        }
        match k.as_str() {
            "event" => event = Some(v.as_str()),
            "state" => state = opt_number(k, v)?,
            "iotype" => iotype = Some(v.as_str()),
            "errno" => err = opt_number(k, v)?,
            "sector" => sector = opt_number(k, v)?,
            "once" => once = opt_bool(k, v)?,
            "immediately" => {
                opt_bool(k, v)?;
            }
            "delay-ns" => delay = opt_number(k, v)?,
            "new_state" => new_state = opt_number(k, v)?,
            _ => unreachable!(),
        }
    }
    let event = opt_event(event)?;
    let state = state as i64;
    if inject {
        let iotype_mask = match iotype {
            None => DEFAULT_IOTYPES,
            Some(t) => match BlkdebugIOType::from_name(t) {
                Some(t) => 1 << t as u64,
                None => return Err(Error::generic(format!("invalid parameter value: {t}"))),
            },
        };
        let sector = sector as i64;
        let action = Action::InjectError {
            iotype_mask,
            error: err as i32,
            once,
            offset: if sector == -1 { -1 } else { sector.wrapping_mul(512) },
            delay_ns: delay as i64,
        };
        st.add(event, state, action);
    } else {
        st.add(event, state, Action::SetState { new_state: new_state as i64 });
    }
    Ok(())
}

fn add_inject(st: &mut State, o: &BlkdebugInjectErrorOptions) {
    let iotype_mask = match o.iotype {
        Some(t) => 1 << t as u64,
        None => DEFAULT_IOTYPES,
    };
    let sector = o.sector.unwrap_or(-1);
    let action = Action::InjectError {
        iotype_mask,
        error: o.errno.unwrap_or(i64::from(libc::EIO)) as i32,
        once: o.once.unwrap_or(false),
        offset: if sector == -1 { -1 } else { sector.wrapping_mul(512) },
        delay_ns: o.delay_ns.unwrap_or(0),
    };
    st.add(o.event, o.state.unwrap_or(0), action);
}

fn add_set_state(st: &mut State, o: &BlkdebugSetStateOptions) {
    st.add(o.event, o.state.unwrap_or(0), Action::SetState { new_state: o.new_state });
}

fn blkdebug_open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Blkdebug(o) = opts else {
        unreachable!("the blkdebug driver gets blkdebug options");
    };
    let o: BlockdevOptionsBlkdebug = *o;

    // read_config(): the config file first, then the rules from the options.
    let mut st = State { rules: vec![Vec::new(); BlkdebugEvent::ALL.len()], ..State::default() };
    if let Some(path) = &o.config {
        for (g, kv) in parse_config_file(path)? {
            add_section(&mut st, &g, &kv)?;
        }
    }
    for r in o.inject_error.iter().flatten() {
        add_inject(&mut st, r);
    }
    for r in o.set_state.iter().flatten() {
        add_set_state(&mut st, r);
    }
    st.state = 1;

    let take_child_perms = o.take_child_perms.iter().flatten().fold(0, |a, &p| a | qapi_perm(p));
    let unshare_child_perms =
        o.unshare_child_perms.iter().flatten().fold(0, |a, &p| a | qapi_perm(p));

    let child = args.open_child(*o.image, "image", BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY)?;

    let size = |v: Option<i32>| v.map_or(0, |v| v as i64 as u64);
    let align_opt = o.align.map_or(0, |v| v as u64);
    if align_opt != 0 && (align_opt >= INT_MAX || !align_opt.is_power_of_two()) {
        return Err(Error::generic(format!("Cannot meet constraints with align {align_opt}")));
    }
    let align = align_opt.max(child.request_alignment());
    let check = |name: &str, v: u64, a: u64| -> Result<()> {
        if v != 0 && (v >= INT_MAX || v % a != 0) {
            return Err(Error::generic(format!("Cannot meet constraints with {name} {v}")));
        }
        Ok(())
    };
    let max_transfer = size(o.max_transfer);
    check("max-transfer", max_transfer, align)?;
    let opt_write_zero = size(o.opt_write_zero);
    check("opt-write-zero", opt_write_zero, align)?;
    let max_write_zero = size(o.max_write_zero);
    check("max-write-zero", max_write_zero, opt_write_zero.max(align))?;
    let opt_discard = size(o.opt_discard);
    check("opt-discard", opt_discard, align)?;
    let max_discard = size(o.max_discard);
    check("max-discard", max_discard, opt_discard.max(align))?;

    let plain = o.inject_error.as_ref().is_none_or(|v| v.is_empty())
        && o.set_state.as_ref().is_none_or(|v| v.is_empty())
        && o.align.is_none()
        && o.max_transfer.is_none()
        && o.opt_write_zero.is_none()
        && o.max_write_zero.is_none()
        && o.opt_discard.is_none()
        && o.max_discard.is_none()
        && o.take_child_perms.is_none()
        && o.unshare_child_perms.is_none();

    let d = BlkdebugDriver {
        align: align_opt,
        max_transfer,
        opt_write_zero,
        max_write_zero,
        opt_discard,
        max_discard,
        config_file: o.config,
        take_child_perms,
        unshare_child_perms,
        plain,
        st: Mutex::new(st),
        resumed: Condvar::new(),
    };
    // bdrv_debug_event(bs, BLKDBG_NONE) runs before the node exists; the driver sees it here.
    d.process_event(None, BlkdebugEvent::None);
    Ok(Box::new(d))
}

impl BlkdebugDriver {
    /// `rule_check()`: the error of the first active rule that matches, if any.
    fn rule_check(&self, offset: u64, bytes: u64, iotype: BlkdebugIOType) -> io::Result<()> {
        let (error, delay_ns) = {
            let mut st = self.st.lock().unwrap();
            let found = st.active.iter().copied().find(|&id| {
                let Some(Rule {
                    action: Action::InjectError { iotype_mask, offset: o, .. }, ..
                }) = st.find(id)
                else {
                    return false;
                };
                let o = *o;
                (o == -1 || (bytes != 0 && o as u64 >= offset && (o as u64) < offset + bytes))
                    && iotype_mask & (1 << iotype as u64) != 0
            });
            let Some(id) = found else {
                return Ok(());
            };
            let Some(Rule { action: Action::InjectError { error, once, delay_ns, .. }, .. }) =
                st.find(id).cloned()
            else {
                unreachable!("active rules inject errors");
            };
            if once {
                st.remove(id);
            }
            (error, delay_ns)
        };
        if delay_ns > 0 {
            std::thread::sleep(Duration::from_nanos(delay_ns as u64));
        }
        if error == 0 {
            return Ok(());
        }
        Err(errno(error))
    }

    /// `blkdebug_co_debug_event()`: runs the rules for `event` and then waits for the
    /// requests it suspended to be resumed.
    fn process_event(&self, bs: Option<&Node>, event: BlkdebugEvent) {
        let _ = bs;
        let mut tickets = Vec::new();
        let mut st = self.st.lock().unwrap();
        let mut new_state = st.state;
        let mut inject_count = 0;
        let rules = st.rules[event as usize].clone();
        for rule in rules {
            if rule.state != 0 && rule.state != st.state {
                continue;
            }
            match &rule.action {
                Action::InjectError { .. } => {
                    inject_count += 1;
                    if inject_count == 1 {
                        st.active.clear();
                    }
                    st.active.insert(0, rule.id);
                }
                Action::SetState { new_state: n } => new_state = *n,
                Action::Suspend { tag } => {
                    // suspend_request().
                    st.remove(rule.id);
                    st.next_id += 1;
                    let ticket = st.next_id;
                    st.suspended.insert(0, (tag.clone(), ticket));
                    tickets.push(ticket);
                    println!("blkdebug: Suspended request '{tag}'");
                }
            }
        }
        st.state = new_state;
        for t in tickets {
            while st.suspended.iter().any(|s| s.1 == t) {
                st = self.resumed.wait(st).unwrap();
            }
        }
    }

    /// `resume_req_by_tag()`.
    fn resume_by_tag(st: &mut State, tag: &str, all: bool) -> bool {
        let mut found = false;
        while let Some(i) = st.suspended.iter().position(|s| s.0 == tag) {
            println!("blkdebug: Resuming request '{tag}'");
            st.suspended.remove(i);
            found = true;
            if !all {
                break;
            }
        }
        found
    }

    /// The current state, for tests.
    #[cfg(test)]
    pub(crate) fn state(&self) -> i64 {
        self.st.lock().unwrap().state
    }

    fn check_aligned(bs: &Node, offset: u64, bytes: u64) {
        let a = bs.request_alignment();
        debug_assert!(offset % a == 0 && bytes % a == 0, "blkdebug got an unaligned request");
        let mt = u64::from(bs.limits().max_transfer);
        debug_assert!(mt == 0 || bytes <= mt, "blkdebug got a request over max-transfer");
    }
}

impl Driver for BlkdebugDriver {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.preadv_flags(bs, offset, buf, 0)
    }

    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.pwrite_flags(bs, offset, buf, 0)
    }

    fn pwrite_flags(&self, bs: &Node, offset: u64, buf: &[u8], flags: u32) -> io::Result<()> {
        Self::check_aligned(bs, offset, buf.len() as u64);
        self.rule_check(offset, buf.len() as u64, BlkdebugIOType::Write)?;
        bs.file().pwrite_flags(offset, buf, flags)
    }

    fn supported_write_flags(&self) -> u32 {
        BDRV_REQ_WRITE_UNCHANGED | BDRV_REQ_FUA
    }

    fn pwrite_zeroes(&self, bs: &Node, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        self.pwrite_zeroes_flags(bs, offset, bytes, if may_unmap { BDRV_REQ_MAY_UNMAP } else { 0 })
    }

    fn pwrite_zeroes_flags(
        &self,
        bs: &Node,
        offset: u64,
        bytes: u64,
        flags: u32,
    ) -> io::Result<()> {
        let bl = bs.limits();
        let align = u64::from(bl.request_alignment.max(bl.pwrite_zeroes_alignment).max(1));
        // Only requests of at least the preferred alignment go through, so that the fallback
        // to writes on unaligned parts gets tested.
        if bytes < align {
            return Err(errno(libc::ENOTSUP));
        }
        debug_assert!(offset % align == 0 && bytes % align == 0);
        self.rule_check(offset, bytes, BlkdebugIOType::WriteZeroes)?;
        bs.file().pwrite_zeroes_flags(offset, bytes, flags)
    }

    fn supported_zero_flags(&self) -> u32 {
        BDRV_REQ_WRITE_UNCHANGED | BDRV_REQ_FUA | BDRV_REQ_MAY_UNMAP | BDRV_REQ_NO_FALLBACK
    }

    fn pdiscard(&self, bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        if bytes < bs.request_alignment() {
            return Err(errno(libc::ENOTSUP));
        }
        self.rule_check(offset, bytes, BlkdebugIOType::Discard)?;
        bs.file().pdiscard(offset, bytes)
    }

    fn flush_to_disk(&self, bs: &Node) -> io::Result<()> {
        self.rule_check(0, 0, BlkdebugIOType::Flush)?;
        bs.file().flush()
    }

    fn getlength(&self, bs: &Node) -> io::Result<u64> {
        bs.file().getlength()
    }

    fn has_truncate(&self) -> bool {
        false
    }

    fn block_status(
        &self,
        bs: &Node,
        _want: u32,
        offset: u64,
        bytes: u64,
    ) -> Option<io::Result<BlockStatus>> {
        if let Err(e) = self.rule_check(offset, bytes, BlkdebugIOType::BlockStatus) {
            return Some(Err(e));
        }
        Some(Ok(BlockStatus {
            ret: BDRV_BLOCK_RAW | BDRV_BLOCK_OFFSET_VALID,
            pnum: bytes,
            map: offset,
            file: Some(bs.file()),
        }))
    }

    fn refresh_limits(&self, _bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        if self.align != 0 {
            bl.request_alignment = self.align as u32;
        }
        if self.max_transfer != 0 {
            bl.max_transfer = self.max_transfer as u32;
        }
        if self.opt_write_zero != 0 {
            bl.pwrite_zeroes_alignment = self.opt_write_zero as u32;
        }
        if self.max_write_zero != 0 {
            bl.max_pwrite_zeroes = self.max_write_zero;
        }
        if self.opt_discard != 0 {
            bl.pdiscard_alignment = self.opt_discard as u32;
        }
        if self.max_discard != 0 {
            bl.max_pdiscard = self.max_discard;
        }
        Ok(())
    }

    fn child_perm_for(&self, ctx: &PermCtx<'_>, perm: u64, shared: u64) -> (u64, u64) {
        let (p, s) = default_perms(ctx, perm, shared);
        (p | self.take_child_perms, s & !self.unshare_child_perms)
    }

    fn reopen_prepare(&self, _bs: &Node, _state: &mut ReopenState) -> Option<Result<()>> {
        Some(Ok(()))
    }

    fn debug_event(&self, bs: &Node, event: BlkdebugEvent) {
        self.process_event(Some(bs), event);
    }

    fn debug_breakpoint(&self, _bs: &Node, event: &str, tag: &str) -> Option<io::Result<()>> {
        let Some(ev) = BlkdebugEvent::from_name(event) else {
            return Some(Err(errno(libc::ENOENT)));
        };
        self.st.lock().unwrap().add(ev, 0, Action::Suspend { tag: tag.to_string() });
        Some(Ok(()))
    }

    fn debug_remove_breakpoint(&self, _bs: &Node, tag: &str) -> Option<io::Result<()>> {
        let mut st = self.st.lock().unwrap();
        let ids: Vec<u64> = st
            .rules
            .iter()
            .flatten()
            .filter(|r| matches!(&r.action, Action::Suspend { tag: t } if t == tag))
            .map(|r| r.id)
            .collect();
        let mut found = !ids.is_empty();
        for id in ids {
            st.remove(id);
        }
        found |= Self::resume_by_tag(&mut st, tag, true);
        drop(st);
        self.resumed.notify_all();
        Some(if found { Ok(()) } else { Err(errno(libc::ENOENT)) })
    }

    fn debug_resume(&self, _bs: &Node, tag: &str) -> Option<io::Result<()>> {
        let found = Self::resume_by_tag(&mut self.st.lock().unwrap(), tag, false);
        self.resumed.notify_all();
        Some(if found { Ok(()) } else { Err(errno(libc::ENOENT)) })
    }

    fn debug_is_suspended(&self, _bs: &Node, tag: &str) -> Option<bool> {
        Some(self.st.lock().unwrap().suspended.iter().any(|s| s.0 == tag))
    }

    fn exact_filename(&self, bs: &Node) -> Option<String> {
        if !self.plain {
            return None;
        }
        let child = bs.file().meta.lock().unwrap().exact_filename.clone();
        if child.is_empty() {
            return None;
        }
        let f = format!("blkdebug:{}:{child}", self.config_file.as_deref().unwrap_or(""));
        // PATH_MAX: a longer name would be cut and unusable, so there is none.
        (f.len() < 4096).then_some(f)
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

impl BlkdebugDriver {
    fn preadv_flags(&self, bs: &Node, offset: u64, buf: &mut [u8], flags: u32) -> io::Result<()> {
        Self::check_aligned(bs, offset, buf.len() as u64);
        self.rule_check(offset, buf.len() as u64, BlkdebugIOType::Read)?;
        bs.file().preadv_flags(offset, buf, flags)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ruvm_qapi::QValue;

    use super::*;

    #[test]
    fn config_parsing() {
        let text = "# comment\n\n[inject-error]\nevent = \"read_aio\"\nerrno = \"5\"\n\
                    once = \"on\"\n[set-state]\nevent = \"l2_load\"\nstate = \"1\"\n\
                    new_state = \"2\"\n";
        let s = parse_config(text.as_bytes()).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].0, "inject-error");
        assert_eq!(s[0].1[0], ("event".to_string(), "read_aio".to_string()));
        assert_eq!(s[1].1.len(), 3);

        let e = parse_config("event = \"x\"\n".as_bytes()).unwrap_err();
        assert_eq!(e.message(), "no group defined");
        let e = parse_config("[inject-error]\nevent\n".as_bytes()).unwrap_err();
        assert_eq!(e.message(), "parse error");
        let e = parse_config("[foo]\nevent = \"x\"\n".as_bytes()).unwrap_err();
        assert_eq!(e.message(), "There is no option group 'foo'");
        let s = parse_config("[inject-error \"id1\"]\nevent = \"\"\n".as_bytes()).unwrap();
        assert_eq!(s[0].1[0], ("id".to_string(), "id1".to_string()));
        assert_eq!(s[0].1[1], ("event".to_string(), String::new()));
    }

    #[test]
    fn section_errors() {
        let mut st =
            State { rules: vec![Vec::new(); BlkdebugEvent::ALL.len()], ..State::default() };
        let kv = |v: &[(&str, &str)]| {
            v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect::<Vec<_>>()
        };
        let e = add_section(&mut st, "inject-error", &kv(&[("errno", "5")])).unwrap_err();
        assert_eq!(e.message(), "Missing event name for rule");
        let e = add_section(&mut st, "inject-error", &kv(&[("event", "nope")])).unwrap_err();
        assert_eq!(e.message(), "invalid parameter value: nope");
        let e = add_section(&mut st, "set-state", &kv(&[("event", "l2_load"), ("errno", "1")]))
            .unwrap_err();
        assert_eq!(e.message(), "Invalid parameter 'errno'");
        let e = add_section(&mut st, "inject-error", &kv(&[("event", "l2_load"), ("once", "x")]))
            .unwrap_err();
        assert_eq!(e.message(), "Parameter 'once' expects 'on' or 'off'");
        let e = add_section(&mut st, "inject-error", &kv(&[("event", "l2_load"), ("errno", "x")]))
            .unwrap_err();
        assert_eq!(e.message(), "Parameter 'errno' expects a number");
        let e =
            add_section(&mut st, "inject-error", &kv(&[("event", "l2_load"), ("iotype", "bogus")]))
                .unwrap_err();
        assert_eq!(e.message(), "invalid parameter value: bogus");
        add_section(&mut st, "inject-error", &kv(&[("event", "l2_load"), ("sector", "3")]))
            .unwrap();
        let r = &st.rules[BlkdebugEvent::L2Load as usize][0];
        assert!(matches!(r.action, Action::InjectError { offset: 1536, error: 5, .. }));
    }

    #[test]
    fn parse_filename() {
        let mut d = QDict::new();
        blkdebug_parse_filename("blkdebug:cfg:null-co://", &mut d).unwrap();
        assert_eq!(d.get_str("config"), Some("cfg"));
        let image = d.get("image").and_then(QValue::as_dict).unwrap();
        assert_eq!(image.get_str("driver"), Some("null-co"));
        assert!(!image.contains_key("filename"));
        let mut d = QDict::new();
        d.put("image", QDict::new().with("driver", "null-aio"));
        blkdebug_parse_filename("blkdebug::null-co://", &mut d).unwrap();
        assert!(!d.contains_key("config"));
        let image = d.get("image").and_then(QValue::as_dict).unwrap();
        assert_eq!(image.get_str("driver"), Some("null-aio"));
        let e = blkdebug_parse_filename("blkdebug::null-co://x", &mut QDict::new()).unwrap_err();
        assert_eq!(e.message(), "The only allowed filename for this driver is 'null-co://'");
        let e = blkdebug_parse_filename("blkdebug:x", &mut QDict::new()).unwrap_err();
        assert_eq!(e.message(), "blkdebug requires both config file and image path");
    }

    fn open(g: &crate::graph::BlockGraph, name: &str, extra: &[(&str, &str)]) -> Result<Arc<Node>> {
        let mut o = QDict::new();
        for (k, v) in extra {
            o.put(*k, *v);
        }
        open_dict(g, name, o)
    }

    fn open_dict(g: &crate::graph::BlockGraph, name: &str, mut o: QDict) -> Result<Arc<Node>> {
        o.put("driver", "blkdebug");
        o.put("node-name", name);
        o.put("image.driver", "null-co");
        o.put("image.read-zeroes", "on");
        let n = g.open_image(None, o)?;
        Ok(g.find_node(&n).unwrap())
    }

    /// A list option. The flat `key.0.member` form needs list support in `crumple()`.
    fn list(items: &[&[(&str, &str)]]) -> QValue {
        QValue::List(
            items
                .iter()
                .map(|kv| {
                    let mut d = QDict::new();
                    for (k, v) in *kv {
                        d.put(*k, *v);
                    }
                    QValue::Dict(d)
                })
                .collect(),
        )
    }

    fn state_of(bs: &Node) -> i64 {
        bs.driver.as_any().unwrap().downcast_ref::<BlkdebugDriver>().unwrap().state()
    }

    #[test]
    fn config_file_rules() {
        let path = std::env::temp_dir().join(format!("ruvm-blkdebug-{}.conf", std::process::id()));
        std::fs::write(
            &path,
            "[inject-error]\nevent = \"read_aio\"\nerrno = \"28\"\nonce = \"on\"\n\
             immediately = \"off\"\n\n[set-state]\nevent = \"read_aio\"\nstate = \"1\"\n\
             new_state = \"2\"\n",
        )
        .unwrap();
        let g = crate::graph::BlockGraph::new();
        let bs = open(&g, "d0", &[("config", path.to_str().unwrap())]).unwrap();
        std::fs::remove_file(&path).unwrap();
        let mut buf = [0u8; 512];
        // No event yet, so no active rule.
        bs.pread(0, &mut buf).unwrap();
        assert_eq!(state_of(&bs), 1);

        bs.debug_event(BlkdebugEvent::ReadAio);
        assert_eq!(state_of(&bs), 2);
        let e = bs.pread(0, &mut buf).unwrap_err();
        assert_eq!(e.raw_os_error(), Some(28));
        // `once` removed the rule.
        bs.pread(0, &mut buf).unwrap();

        let e = open(&g, "d1", &[("config", "/nonexistent/blkdebug.conf")]).unwrap_err();
        assert_eq!(
            e.message(),
            "Could not open '/nonexistent/blkdebug.conf': No such file or directory"
        );
    }

    #[test]
    fn inline_rules() {
        let g = crate::graph::BlockGraph::new();
        let mut o = QDict::new();
        o.put(
            "inject-error",
            list(&[&[
                ("event", "write_aio"),
                ("iotype", "write"),
                ("sector", "1"),
                ("state", "2"),
            ]]),
        );
        o.put("set-state", list(&[&[("event", "flush_to_os"), ("new_state", "2")]]));
        let bs = open_dict(&g, "d0", o).unwrap();
        let data = [0u8; 512];
        // The rule is for state 2 only.
        bs.debug_event(BlkdebugEvent::WriteAio);
        bs.pwrite(512, &data).unwrap();

        bs.debug_event(BlkdebugEvent::FlushToOs);
        assert_eq!(state_of(&bs), 2);
        bs.debug_event(BlkdebugEvent::WriteAio);
        // Only writes that cover sector 1 fail, and reads never do.
        bs.pwrite(0, &data).unwrap();
        let e = bs.pwrite(512, &data).unwrap_err();
        assert_eq!(e.raw_os_error(), Some(libc::EIO));
        let e = bs.pwrite(0, &[0u8; 1024]).unwrap_err();
        assert_eq!(e.raw_os_error(), Some(libc::EIO));
        bs.pread(512, &mut [0u8; 512]).unwrap();

        let mut o = QDict::new();
        o.put("inject-error", list(&[&[("event", "bogus")]]));
        let e = open_dict(&g, "d1", o).unwrap_err();
        assert_eq!(e.message(), "Parameter 'event' does not accept value 'bogus'");
    }

    #[test]
    fn constraint_errors() {
        let g = crate::graph::BlockGraph::new();
        let cases: &[(&str, &str, &str)] = &[
            ("align", "1000", "Cannot meet constraints with align 1000"),
            ("max-transfer", "1001", "Cannot meet constraints with max-transfer 1001"),
            ("opt-write-zero", "102", "Cannot meet constraints with opt-write-zero 102"),
            ("max-discard", "3", "Cannot meet constraints with max-discard 3"),
        ];
        for (i, (k, v, msg)) in cases.iter().enumerate() {
            let e = open(&g, &format!("c{i}"), &[("align", "4"), (k, v)]).unwrap_err();
            assert_eq!(e.message(), *msg);
        }
        let bs = open(&g, "ok", &[("align", "4096"), ("max-transfer", "65536")]).unwrap();
        assert_eq!(bs.request_alignment(), 4096);
        assert_eq!(bs.limits().max_transfer, 65536);
        let e =
            open(&g, "mwz", &[("opt-write-zero", "8192"), ("max-write-zero", "4096")]).unwrap_err();
        assert_eq!(e.message(), "Cannot meet constraints with max-write-zero 4096");
    }

    #[test]
    fn child_perms() {
        let g = crate::graph::BlockGraph::new();
        let mut o = QDict::new();
        o.put("take-child-perms", QValue::List(vec![QValue::str("resize")]));
        o.put("unshare-child-perms", QValue::List(vec![QValue::str("write")]));
        let bs = open_dict(&g, "p0", o).unwrap();
        let (perm, shared) = bs.child("image").unwrap().perm();
        assert_ne!(perm & BLK_PERM_RESIZE, 0);
        assert_eq!(shared & BLK_PERM_WRITE, 0);
    }

    #[test]
    fn suspend_and_resume() {
        let g = crate::graph::BlockGraph::new();
        let bs = open(&g, "s0", &[]).unwrap();
        let e = bs.debug_breakpoint("no_such_event", "t").unwrap_err();
        assert_eq!(e.raw_os_error(), Some(libc::ENOENT));
        bs.debug_breakpoint("read_aio", "t").unwrap();
        let b2 = bs.clone();
        let h = std::thread::spawn(move || b2.debug_event(BlkdebugEvent::ReadAio));
        while !bs.debug_is_suspended("t") {
            std::thread::sleep(Duration::from_millis(1));
        }
        bs.debug_resume("t").unwrap();
        h.join().unwrap();
        assert!(!bs.debug_is_suspended("t"));
        // The breakpoint went with the request it suspended.
        assert_eq!(bs.debug_resume("t").unwrap_err().raw_os_error(), Some(libc::ENOENT));
    }
}
