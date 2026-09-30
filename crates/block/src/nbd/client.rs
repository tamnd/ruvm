// SPDX-License-Identifier: GPL-2.0-or-later

//! The client side of the handshake from nbd/client.c: oldstyle and fixed newstyle negotiation,
//! the option requests, the export list, and meta context negotiation.
//!
//! Everything here is generic over the channel so the byte traces can be checked against an
//! in-memory stream.

use std::io::{Read, Write};

use ruvm_base::{Error, Result};

use super::proto::*;

/// `NBDExportInfo`: what the client asks for and what the server told it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NbdExportInfo {
    /// Ask for `NBD_INFO_BLOCK_SIZE` (set by the caller).
    pub request_sizes: bool,
    /// The meta context to negotiate instead of `base:allocation` (set by the caller).
    pub x_dirty_bitmap: Option<String>,
    /// The export name. Set by the caller, or by the server in an export list.
    pub name: String,
    /// In: the highest mode to try. Out: the mode negotiated.
    pub mode: NbdMode,
    /// In: whether to negotiate a block status context. Out: whether it was.
    pub base_allocation: bool,
    pub size: u64,
    pub flags: u16,
    pub min_block: u32,
    pub opt_block: u32,
    pub max_block: u32,
    /// The id of the negotiated meta context.
    pub context_id: u32,
    /// Only set by [`nbd_receive_export_list`].
    pub description: Option<String>,
    /// Only set by [`nbd_receive_export_list`].
    pub contexts: Vec<String>,
}

/// `NBDOptionReply`.
#[derive(Clone, Copy, Debug)]
struct OptReply {
    option: u32,
    typ: u32,
    length: u32,
}

/// `nbd_send_option_request()`.
fn send_option_request<C: Write + ?Sized>(c: &mut C, opt: u32, data: &[u8]) -> Result<()> {
    let mut hdr = Vec::with_capacity(16);
    hdr.extend_from_slice(&NBD_OPTS_MAGIC.to_be_bytes());
    hdr.extend_from_slice(&opt.to_be_bytes());
    hdr.extend_from_slice(&(data.len() as u32).to_be_bytes());
    nbd_write(c, &hdr).map_err(|e| e.prepend("Failed to send option request header: "))?;
    if !data.is_empty() {
        nbd_write(c, data).map_err(|e| e.prepend("Failed to send option request data: "))?;
    }
    Ok(())
}

/// `nbd_send_opt_abort()`: a courtesy, whose failure does not matter.
fn send_opt_abort<C: Write + ?Sized>(c: &mut C) {
    let _ = send_option_request(c, NBD_OPT_ABORT, &[]);
}

/// Sends `NBD_OPT_ABORT` and passes the error on.
fn abort<C: Write + ?Sized, T>(c: &mut C, e: Error) -> Result<T> {
    send_opt_abort(c);
    Err(e)
}

/// `nbd_receive_option_reply()`.
fn receive_option_reply<C: Read + Write + ?Sized>(c: &mut C, opt: u32) -> Result<OptReply> {
    let mut b = [0u8; 20];
    if let Err(e) = nbd_read(c, &mut b, Some("option reply")) {
        return abort(c, e);
    }
    let magic = be64(&b[0..]);
    let reply = OptReply { option: be32(&b[8..]), typ: be32(&b[12..]), length: be32(&b[16..]) };
    if magic != NBD_REP_MAGIC {
        return abort(c, Error::generic("Unexpected option reply magic"));
    }
    if reply.option != opt {
        return abort(
            c,
            Error::generic(format!(
                "Unexpected option type {} ({}), expected {} ({})",
                reply.option,
                nbd_opt_lookup(reply.option),
                opt,
                nbd_opt_lookup(opt)
            )),
        );
    }
    Ok(reply)
}

/// `nbd_handle_reply_err()`: `Ok(true)` for a reply that is not an error, `Ok(false)` for an
/// error that is not fatal (always so for `NBD_REP_ERR_UNSUP` or when not `strict`).
fn handle_reply_err<C: Read + Write + ?Sized>(
    c: &mut C,
    r: &OptReply,
    strict: bool,
) -> Result<bool> {
    if r.typ & (1 << 31) == 0 {
        return Ok(true);
    }
    let mut msg = None;
    if r.length > 0 {
        if r.length > NBD_MAX_BUFFER_SIZE {
            return abort(
                c,
                Error::generic(format!(
                    "server error {} ({}) message is too long",
                    r.typ,
                    nbd_rep_lookup(r.typ)
                )),
            );
        }
        let mut b = vec![0u8; r.length as usize];
        if let Err(e) = nbd_read(c, &mut b, None) {
            let e = e.prepend(format!(
                "Failed to read option error {} ({}) message: ",
                r.typ,
                nbd_rep_lookup(r.typ)
            ));
            return abort(c, e);
        }
        msg = Some(cstr(&b));
    }
    if r.typ == NBD_REP_ERR_UNSUP || !strict {
        return Ok(false);
    }
    let o = r.option;
    let ol = nbd_opt_lookup(o);
    let mut e = match r.typ {
        NBD_REP_ERR_POLICY => Error::generic(format!("Denied by server for option {o} ({ol})")),
        NBD_REP_ERR_INVALID => Error::generic(format!("Invalid parameters for option {o} ({ol})")),
        NBD_REP_ERR_PLATFORM => {
            Error::generic(format!("Server lacks support for option {o} ({ol})"))
        }
        NBD_REP_ERR_TLS_REQD => {
            Error::generic(format!("TLS negotiation required before option {o} ({ol})"))
                .hint("Did you forget a valid tls-creds?\n")
        }
        NBD_REP_ERR_UNKNOWN => Error::generic("Requested export not available"),
        NBD_REP_ERR_SHUTDOWN => {
            Error::generic(format!("Server shutting down before option {o} ({ol})"))
        }
        NBD_REP_ERR_BLOCK_SIZE_REQD => {
            Error::generic(format!("Server requires INFO_BLOCK_SIZE for option {o} ({ol})"))
        }
        _ => Error::generic(format!("Unknown error code when asking for option {o} ({ol})")),
    };
    if let Some(m) = msg {
        e = e.hint(format!("server reported: {m}\n"));
    }
    abort(c, e)
}

/// The bytes up to the first NUL as text, the way C code sees a received string.
fn cstr(b: &[u8]) -> String {
    let end = b.iter().position(|&x| x == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

/// `nbd_receive_list()`: `Ok(None)` at the end of the list, `Ok(Some((name, description)))` for
/// an entry. An error reply the client can live with also ends the list.
fn receive_list<C: Read + Write + ?Sized>(c: &mut C) -> Result<Option<(String, Option<String>)>> {
    let reply = receive_option_reply(c, NBD_OPT_LIST)?;
    if !handle_reply_err(c, &reply, true)? {
        return Ok(None);
    }
    let mut len = reply.length;
    if reply.typ == NBD_REP_ACK {
        if len != 0 {
            return abort(c, Error::generic("length too long for option end"));
        }
        return Ok(None);
    } else if reply.typ != NBD_REP_SERVER {
        return abort(
            c,
            Error::generic(format!(
                "Unexpected reply type {} ({}), expected {} ({})",
                reply.typ,
                nbd_rep_lookup(reply.typ),
                NBD_REP_SERVER,
                nbd_rep_lookup(NBD_REP_SERVER)
            )),
        );
    }
    if !(4..=NBD_MAX_BUFFER_SIZE).contains(&len) {
        return abort(c, Error::generic(format!("incorrect option length {len}")));
    }
    let namelen = match nbd_read32(c, "option name length") {
        Ok(n) => n,
        Err(e) => return abort(c, e),
    };
    len -= 4;
    if len < namelen || namelen as usize > NBD_MAX_STRING_SIZE {
        return abort(c, Error::generic("incorrect name length in server's list response"));
    }
    let mut name = vec![0u8; namelen as usize];
    if let Err(e) = nbd_read(c, &mut name, Some("export name")) {
        return abort(c, e);
    }
    len -= namelen;
    let mut desc = None;
    if len > 0 {
        if len as usize > NBD_MAX_STRING_SIZE {
            return abort(
                c,
                Error::generic("incorrect description length in server's list response"),
            );
        }
        let mut d = vec![0u8; len as usize];
        if let Err(e) = nbd_read(c, &mut d, Some("export description")) {
            return abort(c, e);
        }
        desc = Some(cstr(&d));
    }
    Ok(Some((cstr(&name), desc)))
}

/// `nbd_opt_info_or_go()`: `Ok(true)` when the server answered, `Ok(false)` when it does not
/// support the option.
fn opt_info_or_go<C: Read + Write + ?Sized>(
    c: &mut C,
    opt: u32,
    info: &mut NbdExportInfo,
) -> Result<bool> {
    info.flags = 0;
    let name = info.name.as_bytes();
    let mut buf = Vec::new();
    buf.extend_from_slice(&(name.len() as u32).to_be_bytes());
    buf.extend_from_slice(name);
    buf.extend_from_slice(&u16::from(info.request_sizes).to_be_bytes());
    if info.request_sizes {
        buf.extend_from_slice(&NBD_INFO_BLOCK_SIZE.to_be_bytes());
    }
    send_option_request(c, opt, &buf)?;

    loop {
        let reply = receive_option_reply(c, opt)?;
        if !handle_reply_err(c, &reply, true)? {
            return Ok(false);
        }
        let mut len = reply.length;
        if reply.typ == NBD_REP_ACK {
            if len != 0 {
                return Err(Error::generic("server sent invalid NBD_REP_ACK"));
            }
            if info.flags == 0 {
                return Err(Error::generic("broken server omitted NBD_INFO_EXPORT"));
            }
            return Ok(true);
        }
        if reply.typ != NBD_REP_INFO {
            return abort(
                c,
                Error::generic(format!(
                    "unexpected reply type {} ({}), expected {} ({})",
                    reply.typ,
                    nbd_rep_lookup(reply.typ),
                    NBD_REP_INFO,
                    nbd_rep_lookup(NBD_REP_INFO)
                )),
            );
        }
        if len < 2 {
            return abort(c, Error::generic(format!("NBD_REP_INFO length {len} is too short")));
        }
        let typ = match nbd_read16(c, "info type") {
            Ok(t) => t,
            Err(e) => return abort(c, e),
        };
        len -= 2;
        let unexpected = |len: u32| {
            Error::generic(format!("remaining export info len {len} is unexpected size"))
        };
        match typ {
            NBD_INFO_EXPORT => {
                if len != 8 + 2 {
                    return abort(c, unexpected(len));
                }
                info.size = match nbd_read64(c, "info size") {
                    Ok(v) => v,
                    Err(e) => return abort(c, e),
                };
                info.flags = match nbd_read16(c, "info flags") {
                    Ok(v) => v,
                    Err(e) => return abort(c, e),
                };
                if info.min_block != 0 && info.size % u64::from(info.min_block) != 0 {
                    return abort(
                        c,
                        Error::generic(format!(
                            "export size {} is not multiple of minimum block size {}",
                            info.size, info.min_block
                        )),
                    );
                }
            }
            NBD_INFO_BLOCK_SIZE => {
                if len != 4 * 3 {
                    return abort(c, unexpected(len));
                }
                info.min_block = match nbd_read32(c, "info minimum block size") {
                    Ok(v) => v,
                    Err(e) => return abort(c, e),
                };
                if !info.min_block.is_power_of_two() {
                    return abort(
                        c,
                        Error::generic(format!(
                            "server minimum block size {} is not a power of two",
                            info.min_block
                        )),
                    );
                }
                info.opt_block = match nbd_read32(c, "info preferred block size") {
                    Ok(v) => v,
                    Err(e) => return abort(c, e),
                };
                if !info.opt_block.is_power_of_two() || info.opt_block < info.min_block {
                    return abort(
                        c,
                        Error::generic(format!(
                            "server preferred block size {} is not valid",
                            info.opt_block
                        )),
                    );
                }
                info.max_block = match nbd_read32(c, "info maximum block size") {
                    Ok(v) => v,
                    Err(e) => return abort(c, e),
                };
                if info.max_block < info.min_block {
                    return abort(
                        c,
                        Error::generic(format!(
                            "server maximum block size {} is not valid",
                            info.max_block
                        )),
                    );
                }
            }
            _ => {
                if let Err(e) = nbd_drop(c, u64::from(len)) {
                    return abort(c, e.prepend("Failed to read info payload: "));
                }
            }
        }
    }
}

/// `nbd_receive_query_exports()`: fails when the server lists exports but not `wantname`.
fn receive_query_exports<C: Read + Write + ?Sized>(c: &mut C, wantname: &str) -> Result<()> {
    send_option_request(c, NBD_OPT_LIST, &[])?;
    let mut list_empty = true;
    let mut found = false;
    while let Some((name, _)) = receive_list(c)? {
        list_empty = false;
        if name == wantname {
            found = true;
        }
    }
    if !list_empty && !found {
        return abort(c, Error::generic(format!("No export with name '{wantname}' available")));
    }
    Ok(())
}

/// `nbd_request_simple_option()`: `Ok(true)` when the server acked the option.
fn request_simple_option<C: Read + Write + ?Sized>(
    c: &mut C,
    opt: u32,
    strict: bool,
) -> Result<bool> {
    send_option_request(c, opt, &[])?;
    let reply = receive_option_reply(c, opt)?;
    if !handle_reply_err(c, &reply, strict)? {
        return Ok(false);
    }
    if reply.typ != NBD_REP_ACK {
        return abort(
            c,
            Error::generic(format!(
                "Server answered option {} ({}) with unexpected reply {} ({})",
                opt,
                nbd_opt_lookup(opt),
                reply.typ,
                nbd_rep_lookup(reply.typ)
            )),
        );
    }
    if reply.length != 0 {
        return abort(
            c,
            Error::generic(format!(
                "Option {} ('{}') response length is {} (it should be zero)",
                opt,
                nbd_opt_lookup(opt),
                reply.length
            )),
        );
    }
    Ok(true)
}

/// `nbd_send_meta_query()`.
fn send_meta_query<C: Write + ?Sized>(
    c: &mut C,
    opt: u32,
    export: &str,
    query: Option<&str>,
) -> Result<()> {
    let mut d = Vec::new();
    d.extend_from_slice(&(export.len() as u32).to_be_bytes());
    d.extend_from_slice(export.as_bytes());
    d.extend_from_slice(&u32::from(query.is_some()).to_be_bytes());
    if let Some(q) = query {
        d.extend_from_slice(&(q.len() as u32).to_be_bytes());
        d.extend_from_slice(q.as_bytes());
    }
    send_option_request(c, opt, &d)
}

/// `nbd_receive_one_meta_context()`: `Ok(None)` at the end of the replies.
fn receive_one_meta_context<C: Read + Write + ?Sized>(
    c: &mut C,
    opt: u32,
) -> Result<Option<(String, u32)>> {
    let reply = receive_option_reply(c, opt)?;
    if !handle_reply_err(c, &reply, false)? {
        return Ok(None);
    }
    if reply.typ == NBD_REP_ACK {
        if reply.length != 0 {
            return abort(c, Error::generic("Unexpected length to ACK response"));
        }
        return Ok(None);
    } else if reply.typ != NBD_REP_META_CONTEXT {
        return abort(
            c,
            Error::generic(format!(
                "Unexpected reply type {} ({}), expected {} ({})",
                reply.typ,
                nbd_rep_lookup(reply.typ),
                NBD_REP_META_CONTEXT,
                nbd_rep_lookup(NBD_REP_META_CONTEXT)
            )),
        );
    }
    if reply.length <= 4 || reply.length > NBD_MAX_BUFFER_SIZE {
        return abort(
            c,
            Error::generic(format!(
                "Failed to negotiate meta context, server answered with unexpected length {}",
                reply.length
            )),
        );
    }
    let id = nbd_read32(c, "context id")?;
    let mut name = vec![0u8; reply.length as usize - 4];
    nbd_read(c, &mut name, Some("context name"))?;
    Ok(Some((cstr(&name), id)))
}

/// `nbd_negotiate_simple_meta_context()`: `Ok(true)` when the server agreed to the context.
fn negotiate_simple_meta_context<C: Read + Write + ?Sized>(
    c: &mut C,
    info: &mut NbdExportInfo,
) -> Result<bool> {
    let context = info.x_dirty_bitmap.clone().unwrap_or_else(|| "base:allocation".into());
    send_meta_query(c, NBD_OPT_SET_META_CONTEXT, &info.name, Some(&context))?;
    let mut received = false;
    let mut ret = receive_one_meta_context(c, NBD_OPT_SET_META_CONTEXT)?;
    if let Some((name, id)) = ret {
        info.context_id = id;
        if name != context {
            return abort(
                c,
                Error::generic(format!(
                    "Failed to negotiate meta context '{context}', server answered with \
                     different context '{name}'"
                )),
            );
        }
        received = true;
        ret = receive_one_meta_context(c, NBD_OPT_SET_META_CONTEXT)?;
    }
    if ret.is_some() {
        return abort(c, Error::generic("Server answered with more than one context"));
    }
    Ok(received)
}

/// `nbd_list_meta_contexts()`.
fn list_meta_contexts<C: Read + Write + ?Sized>(c: &mut C, info: &mut NbdExportInfo) -> Result<()> {
    let mut seen_any = false;
    let mut seen_qemu = false;
    send_meta_query(c, NBD_OPT_LIST_META_CONTEXT, &info.name, None)?;
    loop {
        let r = receive_one_meta_context(c, NBD_OPT_LIST_META_CONTEXT)?;
        match r {
            None if seen_any && !seen_qemu => {
                // qemu 3.0 forgot the "qemu:" replies to an empty query, so ask for them.
                seen_qemu = true;
                send_meta_query(c, NBD_OPT_LIST_META_CONTEXT, &info.name, Some("qemu:"))?;
            }
            None => return Ok(()),
            Some((name, _)) => {
                seen_any = true;
                seen_qemu |= name.starts_with("qemu:");
                info.contexts.push(name);
            }
        }
    }
}

/// `nbd_start_negotiate()`: reads the greeting and settles the mode. Returns the mode and
/// whether the handshake ends with 124 zero bytes.
fn start_negotiate<C: Read + Write + ?Sized>(
    c: &mut C,
    max_mode: NbdMode,
) -> Result<(NbdMode, bool)> {
    let mut zeroes = true;
    let magic = nbd_read64(c, "initial magic")?;
    if magic != NBD_INIT_MAGIC {
        return Err(Error::generic(format!("Bad initial magic received: 0x{magic:x}")));
    }
    let magic = nbd_read64(c, "server magic")?;
    if magic == NBD_OPTS_MAGIC {
        let mut clientflags = 0u32;
        let globalflags = nbd_read16(c, "server flags")?;
        let fixed = globalflags & NBD_FLAG_FIXED_NEWSTYLE != 0;
        if fixed {
            clientflags |= NBD_FLAG_C_FIXED_NEWSTYLE;
        }
        if globalflags & NBD_FLAG_NO_ZEROES != 0 {
            zeroes = false;
            clientflags |= NBD_FLAG_C_NO_ZEROES;
        }
        nbd_write(c, &clientflags.to_be_bytes())
            .map_err(|e| e.prepend("Failed to send clientflags field: "))?;
        if !fixed {
            return Ok((NbdMode::ExportName, zeroes));
        }
        if max_mode >= NbdMode::Extended
            && request_simple_option(c, NBD_OPT_EXTENDED_HEADERS, false)?
        {
            return Ok((NbdMode::Extended, zeroes));
        }
        if max_mode >= NbdMode::Structured
            && request_simple_option(c, NBD_OPT_STRUCTURED_REPLY, false)?
        {
            return Ok((NbdMode::Structured, zeroes));
        }
        Ok((NbdMode::Simple, zeroes))
    } else if magic == NBD_CLIENT_MAGIC {
        Ok((NbdMode::Oldstyle, zeroes))
    } else {
        Err(Error::generic(format!("Bad server magic received: 0x{magic:x}")))
    }
}

/// `nbd_negotiate_finish_oldstyle()`.
fn finish_oldstyle<C: Read + ?Sized>(c: &mut C, info: &mut NbdExportInfo) -> Result<()> {
    info.size = nbd_read64(c, "export length")?;
    let oldflags = nbd_read32(c, "export flags")?;
    if oldflags & !0xffff != 0 {
        // QEMU's format string is "%0x" PRIx32, which prints a stray 'x'.
        return Err(Error::generic(format!("Unexpected export flags {oldflags:x}x")));
    }
    info.flags = oldflags as u16;
    Ok(())
}

/// `nbd_receive_negotiate()`: the whole client handshake, up to the transmission phase.
///
/// TLS is not available, so there are no credentials to pass.
pub fn nbd_receive_negotiate<C: Read + Write + ?Sized>(
    c: &mut C,
    info: &mut NbdExportInfo,
) -> Result<()> {
    assert!(info.name.len() <= NBD_MAX_STRING_SIZE);
    let base_allocation = info.base_allocation;
    let (mode, zeroes) = start_negotiate(c, info.mode)?;
    info.mode = mode;
    info.base_allocation = false;

    let mut export_name = mode == NbdMode::ExportName;
    if mode >= NbdMode::Structured && base_allocation {
        info.base_allocation = negotiate_simple_meta_context(c, info)?;
    }
    if mode >= NbdMode::Simple {
        if opt_info_or_go(c, NBD_OPT_GO, info)? {
            return Ok(());
        }
        receive_query_exports(c, &info.name.clone())?;
        export_name = true;
    }
    if export_name {
        send_option_request(c, NBD_OPT_EXPORT_NAME, info.name.as_bytes())?;
        info.size = nbd_read64(c, "export length")?;
        info.flags = nbd_read16(c, "export flags")?;
    } else {
        if !info.name.is_empty() {
            return Err(Error::generic("Server does not support non-empty export names"));
        }
        finish_oldstyle(c, info)?;
    }
    if zeroes {
        nbd_drop(c, 124).map_err(|e| e.prepend("Failed to read reserved block: "))?;
    }
    Ok(())
}

/// `nbd_receive_export_list()`: what `qemu-nbd --list` prints.
pub fn nbd_receive_export_list<C: Read + Write + ?Sized>(c: &mut C) -> Result<Vec<NbdExportInfo>> {
    let (mode, _) = start_negotiate(c, NbdMode::Extended)?;
    let mut array = Vec::new();
    match mode {
        NbdMode::Simple | NbdMode::Structured | NbdMode::Extended => {
            send_option_request(c, NBD_OPT_LIST, &[])?;
            while let Some((name, description)) = receive_list(c)? {
                array.push(NbdExportInfo { name, description, mode, ..Default::default() });
            }
            for e in &mut array {
                e.request_sizes = true;
                if !opt_info_or_go(c, NBD_OPT_INFO, e)? {
                    break;
                }
                if mode >= NbdMode::Structured {
                    list_meta_contexts(c, e)?;
                }
            }
            send_opt_abort(c);
        }
        NbdMode::ExportName => {
            return Err(Error::generic("Server does not support export lists"));
        }
        NbdMode::Oldstyle => {
            let mut e = NbdExportInfo {
                name: String::new(),
                mode: NbdMode::Oldstyle,
                ..Default::default()
            };
            finish_oldstyle(c, &mut e)?;
            if nbd_drop(c, 124).is_ok() {
                let req = NbdRequest { typ: NBD_CMD_DISC, mode, ..Default::default() };
                let _ = c.write_all(&req.encode());
            }
            array.push(e);
        }
    }
    Ok(array)
}
