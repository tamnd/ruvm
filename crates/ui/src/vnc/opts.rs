// SPDX-License-Identifier: GPL-2.0-or-later

//! The `-vnc` option, from `vnc_parse()`, `vnc_init_func()`, `vnc_display_open()` and
//! `vnc_display_get_addresses()` in QEMU's ui/vnc.c.
//!
//! The options are read in the order QEMU reads them, so the first bad one is the one
//! reported. Websockets, TLS, SASL, reverse connections and audio are refused with a message
//! saying ruvm does not have them yet.

use std::sync::{Arc, LazyLock, Mutex};

use ruvm_base::report::{Location, current_location, push_location, report_error};
use ruvm_base::{Error, Result};
use ruvm_qapi::opts::{
    OptsHandle, QemuOptDesc, QemuOptType, QemuOpts, QemuOptsList, is_help_option,
};

use crate::console::DisplayState;
use crate::input::InputState;
use crate::keymaps::{self, KbdLayout};

use super::net::{self, ListenAddr, Listener};
use super::{Auth, Config, Hooks, SharePolicy, VncDisplay, lock, register_display};

const VNC_OPT_DESCS: &[QemuOptDesc] = &[
    QemuOptDesc::new("vnc", QemuOptType::String),
    QemuOptDesc::new("websocket", QemuOptType::String),
    QemuOptDesc::new("tls-creds", QemuOptType::String),
    QemuOptDesc::new("share", QemuOptType::String),
    QemuOptDesc::new("display", QemuOptType::String),
    QemuOptDesc::new("head", QemuOptType::Number),
    QemuOptDesc::new("connections", QemuOptType::Number),
    QemuOptDesc::new("to", QemuOptType::Number),
    QemuOptDesc::new("ipv4", QemuOptType::Bool),
    QemuOptDesc::new("ipv6", QemuOptType::Bool),
    QemuOptDesc::new("password", QemuOptType::Bool),
    QemuOptDesc::new("password-secret", QemuOptType::String),
    QemuOptDesc::new("reverse", QemuOptType::Bool),
    QemuOptDesc::new("lock-key-sync", QemuOptType::Bool),
    QemuOptDesc::new("key-delay-ms", QemuOptType::Number),
    QemuOptDesc::new("sasl", QemuOptType::Bool),
    QemuOptDesc::new("tls-authz", QemuOptType::String),
    QemuOptDesc::new("sasl-authz", QemuOptType::String),
    QemuOptDesc::new("lossy", QemuOptType::Bool),
    QemuOptDesc::new("non-adaptive", QemuOptType::Bool),
    QemuOptDesc::new("audiodev", QemuOptType::String),
    QemuOptDesc::new("power-control", QemuOptType::Bool),
];

/// `qemu_vnc_opts`.
pub fn opts_list() -> QemuOptsList {
    QemuOptsList::new("vnc", VNC_OPT_DESCS).with_implied_opt_name("vnc")
}

/// The `-vnc` options seen so far, with where each came from for the messages of
/// `vnc_init_func()`.
struct Parsed {
    list: QemuOptsList,
    locs: Vec<(OptsHandle, Option<Location>)>,
}

static OPTS: LazyLock<Mutex<Parsed>> =
    LazyLock::new(|| Mutex::new(Parsed { list: opts_list(), locs: Vec::new() }));

/// `vnc_parse()`: `-vnc` and the argument of `-display vnc=`. The error is the exit status.
pub fn parse(arg: &str) -> std::result::Result<(), u8> {
    let mut p = lock(&OPTS);
    let loc = current_location();
    let Some((handle, has_id)) =
        p.list.parse_noisily(arg, !is_help_option(arg)).map(|o| (o.handle(), o.id().is_some()))
    else {
        return Err(1);
    };
    if !has_id {
        let id = auto_assign_id(&p.list);
        if let Some(o) = p.list.get_mut(handle) {
            o.set_id(Some(id));
        }
    }
    p.locs.push((handle, loc));
    Ok(())
}

/// Whether there is a `-vnc` option, which is what `display_remote` counts in QEMU's
/// system/vl.c when it picks the default display.
pub fn configured() -> bool {
    !lock(&OPTS).locs.is_empty()
}

/// `vnc_init_func()` over every `-vnc`: opens the displays in order and reports the first
/// failure where its option came from. `name` is `-name`, for the desktop name.
pub fn init(name: Option<&str>, hooks: Arc<dyn Hooks>) -> std::result::Result<(), u8> {
    let p = lock(&OPTS);
    for (handle, loc) in &p.locs {
        let _guard = loc.clone().map(push_location);
        let Some(opts) = p.list.get(*handle) else { continue };
        let id = opts.id().unwrap_or_default().to_string();
        let ds = DisplayState::global();
        if let Err(e) = open(opts, &id, name, ds, InputState::global(), Arc::clone(&hooks)) {
            report_error(&e);
            return Err(1);
        }
    }
    Ok(())
}

/// `vnc_auto_assign_id()`.
fn auto_assign_id(list: &QemuOptsList) -> String {
    let mut id = "default".to_string();
    let mut i = 2;
    while list.find(Some(&id)).is_some() {
        id = format!("vnc{i}");
        i += 1;
    }
    id
}

fn not_yet(what: &str) -> Error {
    Error::generic(format!("VNC {what} not supported by ruvm yet"))
}

/// `vnc_display_get_address()` for one `vnc` value.
pub(crate) fn get_address(
    addrstr: &str,
    to: i32,
    ipv4: Option<bool>,
    ipv6: Option<bool>,
) -> Result<ListenAddr> {
    if let Some(path) = addrstr.strip_prefix("unix:") {
        if to != 0 {
            return Err(Error::generic("Port range not support with UNIX socket"));
        }
        return Ok(ListenAddr::Unix(path.to_string()));
    }
    let Some(colon) = addrstr.rfind(':') else {
        return Err(Error::generic("no vnc port specified"));
    };
    let port = &addrstr[colon + 1..];
    if port.is_empty() {
        return Err(Error::generic("vnc port cannot be empty"));
    }
    let mut host = &addrstr[..colon];
    if host.len() >= 2 && host.starts_with('[') && host.ends_with(']') {
        host = &host[1..host.len() - 1];
    }
    let Ok((baseport, _)) = ruvm_qapi::cutils::parse_uint(port, 10, true) else {
        return Err(Error::generic(format!("can't convert to a number: {port}")));
    };
    let baseport: u64 = baseport;
    if baseport > 65535 || baseport + 5900 > 65535 {
        return Err(Error::generic(format!("port {port} out of range")));
    }
    let to = if to != 0 { Some(to.wrapping_add(5900).max(0) as u32) } else { None };
    Ok(ListenAddr::Inet { host: host.to_string(), port: baseport as u32 + 5900, to, ipv4, ipv6 })
}

/// `vnc_display_get_addresses()`, without the websocket half.
pub(crate) fn get_addresses(opts: &QemuOpts) -> Result<Vec<ListenAddr>> {
    let to = opts.get_number("to", 0) as i32;
    let ipv4 = opts.get("ipv4").map(|_| opts.get_bool("ipv4", false));
    let ipv6 = opts.get("ipv6").map(|_| opts.get_bool("ipv6", false));
    match opts.get("vnc") {
        None | Some("none") => return Ok(Vec::new()),
        Some(_) => {}
    }
    let mut addrs = Vec::new();
    for v in opts.iter_values(Some("vnc")) {
        addrs.push(get_address(v, to, ipv4, ipv6)?);
    }
    if opts.get("websocket").is_some() {
        return Err(not_yet("websocket is"));
    }
    Ok(addrs)
}

/// `vnc_display_new()` and `vnc_display_open()`.
pub(crate) fn open(
    opts: &QemuOpts,
    id: &str,
    name: Option<&str>,
    ds: Arc<DisplayState>,
    input: Arc<InputState>,
    hooks: Arc<dyn Hooks>,
) -> Result<Arc<VncDisplay>> {
    // vnc_display_new() loads the layout before anything else.
    let layout = KbdLayout::new(keymaps::keyboard_layout().as_deref().unwrap_or("en-us"))?;
    let reverse = opts.get_bool("reverse", false);
    let addrs = get_addresses(opts)?;

    let (password, secret) = match opts.get("password-secret") {
        Some(secret_id) => {
            if opts.get("password").is_some() {
                return Err(Error::generic("'password' flag is redundant with 'password-secret'"));
            }
            (true, Some(ruvm_crypto::secret::secret_lookup_as_utf8(secret_id)?))
        }
        None => (opts.get_bool("password", false), None),
    };
    let lock_key_sync = opts.get_bool("lock-key-sync", true);
    let key_delay_ms = opts.get_number("key-delay-ms", 10) as u32;
    if opts.get_bool("sasl", false) {
        return Err(not_yet("SASL auth is"));
    }
    if opts.get("tls-creds").is_some() {
        return Err(not_yet("TLS is"));
    }
    if opts.get("tls-authz").is_some() {
        return Err(Error::generic("'tls-authz' provided but TLS is not enabled"));
    }
    if opts.get("sasl-authz").is_some() {
        return Err(Error::generic("'sasl-authz' provided but SASL auth is not enabled"));
    }
    let share_policy = match opts.get("share") {
        None | Some("allow-exclusive") => SharePolicy::AllowExclusive,
        Some("ignore") => SharePolicy::Ignore,
        Some("force-shared") => SharePolicy::ForceShared,
        Some(_) => return Err(Error::generic("unknown vnc share= option")),
    };
    let connections_limit = opts.get_number("connections", 32);
    // QEMU only reads lossy with JPEG support and then always runs non-adaptive, so a build
    // without JPEG, like this one, is lossless and non-adaptive whatever the options say.
    let power_control = opts.get_bool("power-control", false);
    let auth = if password { Auth::Vnc } else { Auth::None };
    if opts.get("audiodev").is_some() {
        return Err(not_yet("audio is"));
    }
    let con = match opts.get("display") {
        Some(device_id) => {
            let head = opts.get_number("head", 0) as u32;
            Some(ds.lookup_by_device_name(device_id, head)?)
        }
        None => ds.lookup_default(),
    };

    let listeners: Vec<Listener> = if addrs.is_empty() {
        Vec::new()
    } else if reverse {
        return Err(not_yet("reverse connections are"));
    } else {
        net::listen(&addrs)?
    };

    let cfg = Config {
        auth,
        password: secret,
        share_policy,
        connections_limit,
        power_control,
        lossy: false,
        lock_key_sync,
        key_delay_ms,
    };
    let vd = VncDisplay::new(id, cfg, layout, name, ds, con, input, hooks);
    if !listeners.is_empty() {
        vd.listen(listeners);
        if opts.get("to").is_some() {
            // vnc_display_print_local_addr()
            if let Some(info) = vd.server_info() {
                if info.family != ruvm_qapi::types::NetworkAddressFamily::Unix {
                    eprintln!("VNC server running on {}:{}", info.host, info.service);
                }
            }
        }
    }
    register_display(Arc::clone(&vd));
    Ok(vd)
}
