// SPDX-License-Identifier: GPL-2.0-or-later

//! A hand-written binding to libslirp 4.x, loaded with `dlopen` when `-netdev user` first needs
//! it.
//!
//! Only the parts of libslirp.h that net/slirp.c uses are here. The structs follow the header
//! field for field, since libslirp reads them by layout. The library is looked up once and never
//! unloaded, so the function pointers stay valid for the life of the process.

#![allow(unsafe_code)]

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::OnceLock;

/// `Slirp *`: one instance of the stack. Only ever handled as a pointer.
pub(crate) type SlirpPtr = *mut c_void;

pub(crate) const POLL_IN: c_int = 1 << 0;
pub(crate) const POLL_OUT: c_int = 1 << 1;
pub(crate) const POLL_PRI: c_int = 1 << 2;
pub(crate) const POLL_ERR: c_int = 1 << 3;
pub(crate) const POLL_HUP: c_int = 1 << 4;

/// `SLIRP_HOSTFWD_UDP`.
pub(crate) const HOSTFWD_UDP: c_int = 1;

/// `struct in_addr`, in network byte order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct InAddr {
    pub(crate) s_addr: u32,
}

impl From<std::net::Ipv4Addr> for InAddr {
    fn from(a: std::net::Ipv4Addr) -> Self {
        InAddr { s_addr: u32::from_ne_bytes(a.octets()) }
    }
}

/// `struct in6_addr`. The C type is a union with a 32-bit member, hence the alignment.
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct In6Addr {
    pub(crate) s6_addr: [u8; 16],
}

impl From<std::net::Ipv6Addr> for In6Addr {
    fn from(a: std::net::Ipv6Addr) -> Self {
        In6Addr { s6_addr: a.octets() }
    }
}

/// `SlirpWriteCb`.
pub(crate) type WriteCb =
    unsafe extern "C" fn(buf: *const c_void, len: usize, opaque: *mut c_void) -> isize;
/// `SlirpTimerCb`.
pub(crate) type TimerCb = unsafe extern "C" fn(opaque: *mut c_void);
/// `SlirpAddPollCb` and `SlirpAddPollSocketCb`, which are the same on Unix.
pub(crate) type AddPollCb =
    unsafe extern "C" fn(fd: c_int, events: c_int, opaque: *mut c_void) -> c_int;
/// `SlirpGetREventsCb`.
pub(crate) type GetReventsCb = unsafe extern "C" fn(idx: c_int, opaque: *mut c_void) -> c_int;

/// `SlirpCb`, up to the version 6 fields.
#[repr(C)]
pub(crate) struct SlirpCb {
    pub(crate) send_packet: Option<WriteCb>,
    pub(crate) guest_error: Option<unsafe extern "C" fn(msg: *const c_char, opaque: *mut c_void)>,
    pub(crate) clock_get_ns: Option<unsafe extern "C" fn(opaque: *mut c_void) -> i64>,
    pub(crate) timer_new: Option<
        unsafe extern "C" fn(
            cb: TimerCb,
            cb_opaque: *mut c_void,
            opaque: *mut c_void,
        ) -> *mut c_void,
    >,
    pub(crate) timer_free: Option<unsafe extern "C" fn(timer: *mut c_void, opaque: *mut c_void)>,
    pub(crate) timer_mod:
        Option<unsafe extern "C" fn(timer: *mut c_void, expire_ms: i64, opaque: *mut c_void)>,
    pub(crate) register_poll_fd: Option<unsafe extern "C" fn(fd: c_int, opaque: *mut c_void)>,
    pub(crate) unregister_poll_fd: Option<unsafe extern "C" fn(fd: c_int, opaque: *mut c_void)>,
    pub(crate) notify: Option<unsafe extern "C" fn(opaque: *mut c_void)>,
    pub(crate) init_completed: Option<unsafe extern "C" fn(slirp: SlirpPtr, opaque: *mut c_void)>,
    pub(crate) timer_new_opaque: Option<
        unsafe extern "C" fn(id: c_int, cb_opaque: *mut c_void, opaque: *mut c_void) -> *mut c_void,
    >,
    pub(crate) register_poll_socket: Option<unsafe extern "C" fn(fd: c_int, opaque: *mut c_void)>,
    pub(crate) unregister_poll_socket: Option<unsafe extern "C" fn(fd: c_int, opaque: *mut c_void)>,
}

/// `SlirpConfig`, up to the version 6 fields.
#[repr(C)]
pub(crate) struct SlirpConfig {
    pub(crate) version: u32,
    pub(crate) restricted: c_int,
    pub(crate) in_enabled: bool,
    pub(crate) vnetwork: InAddr,
    pub(crate) vnetmask: InAddr,
    pub(crate) vhost: InAddr,
    pub(crate) in6_enabled: bool,
    pub(crate) vprefix_addr6: In6Addr,
    pub(crate) vprefix_len: u8,
    pub(crate) vhost6: In6Addr,
    pub(crate) vhostname: *const c_char,
    pub(crate) tftp_server_name: *const c_char,
    pub(crate) tftp_path: *const c_char,
    pub(crate) bootfile: *const c_char,
    pub(crate) vdhcp_start: InAddr,
    pub(crate) vnameserver: InAddr,
    pub(crate) vnameserver6: In6Addr,
    pub(crate) vdnssearch: *const *const c_char,
    pub(crate) vdomainname: *const c_char,
    pub(crate) if_mtu: usize,
    pub(crate) if_mru: usize,
    pub(crate) disable_host_loopback: bool,
    pub(crate) enable_emu: bool,
    pub(crate) outbound_addr: *const c_void,
    pub(crate) outbound_addr6: *const c_void,
    pub(crate) disable_dns: bool,
    pub(crate) disable_dhcp: bool,
    pub(crate) mfr_id: u32,
    pub(crate) oob_eth_addr: [u8; 6],
}

/// The entry points of a loaded libslirp.
#[derive(Debug)]
pub(crate) struct Lib {
    /// `slirp_version_string()`, for example "4.9.1".
    pub(crate) version: String,
    /// `major * 1000 + minor`, what `SLIRP_CHECK_VERSION` compares.
    pub(crate) version_code: u32,
    pub(crate) new: unsafe extern "C" fn(
        cfg: *const SlirpConfig,
        cb: *const SlirpCb,
        opaque: *mut c_void,
    ) -> SlirpPtr,
    pub(crate) cleanup: unsafe extern "C" fn(slirp: SlirpPtr),
    /// `slirp_pollfds_fill_socket()`, or `slirp_pollfds_fill()` before 4.9.
    pub(crate) pollfds_fill: unsafe extern "C" fn(
        slirp: SlirpPtr,
        timeout: *mut u32,
        add: AddPollCb,
        opaque: *mut c_void,
    ),
    pub(crate) pollfds_poll: unsafe extern "C" fn(
        slirp: SlirpPtr,
        select_error: c_int,
        get_revents: GetReventsCb,
        opaque: *mut c_void,
    ),
    pub(crate) input: unsafe extern "C" fn(slirp: SlirpPtr, pkt: *const u8, len: c_int),
    /// `slirp_handle_timer()`, 4.7 and later.
    pub(crate) handle_timer:
        Option<unsafe extern "C" fn(slirp: SlirpPtr, id: c_int, cb_opaque: *mut c_void)>,
    pub(crate) add_hostxfwd: unsafe extern "C" fn(
        slirp: SlirpPtr,
        haddr: *const c_void,
        haddrlen: u32,
        gaddr: *const c_void,
        gaddrlen: u32,
        flags: c_int,
    ) -> c_int,
    pub(crate) remove_hostxfwd: unsafe extern "C" fn(
        slirp: SlirpPtr,
        haddr: *const c_void,
        haddrlen: u32,
        flags: c_int,
    ) -> c_int,
    pub(crate) add_exec: unsafe extern "C" fn(
        slirp: SlirpPtr,
        cmdline: *const c_char,
        guest_addr: *mut InAddr,
        guest_port: c_int,
    ) -> c_int,
    pub(crate) add_guestfwd: unsafe extern "C" fn(
        slirp: SlirpPtr,
        write_cb: WriteCb,
        opaque: *mut c_void,
        guest_addr: *mut InAddr,
        guest_port: c_int,
    ) -> c_int,
    pub(crate) socket_can_recv:
        unsafe extern "C" fn(slirp: SlirpPtr, guest_addr: InAddr, guest_port: c_int) -> usize,
    pub(crate) socket_recv: unsafe extern "C" fn(
        slirp: SlirpPtr,
        guest_addr: InAddr,
        guest_port: c_int,
        buf: *const u8,
        size: c_int,
    ),
    pub(crate) connection_info: unsafe extern "C" fn(slirp: SlirpPtr) -> *mut c_char,
}

impl Lib {
    /// `SLIRP_CHECK_VERSION(major, minor, 0)`.
    pub(crate) fn at_least(&self, major: u32, minor: u32) -> bool {
        self.version_code >= major * 1000 + minor
    }
}

/// The environment variable that names the library to load instead of the usual names.
pub(crate) const LIB_ENV: &str = "RUVM_LIBSLIRP";

fn candidates() -> Vec<String> {
    let mut names = Vec::new();
    if let Ok(v) = std::env::var(LIB_ENV) {
        if !v.is_empty() {
            names.push(v);
        }
    }
    if cfg!(target_os = "macos") {
        names.extend(
            [
                "libslirp.0.dylib",
                "/opt/homebrew/lib/libslirp.0.dylib",
                "/usr/local/lib/libslirp.0.dylib",
                "/opt/local/lib/libslirp.0.dylib",
            ]
            .map(String::from),
        );
    } else {
        names.extend(["libslirp.so.0", "libslirp.so"].map(String::from));
    }
    names
}

/// The loaded library, or why it could not be loaded.
pub(crate) fn lib() -> Result<&'static Lib, String> {
    static LIB: OnceLock<Result<Lib, String>> = OnceLock::new();
    LIB.get_or_init(load).as_ref().map_err(Clone::clone)
}

fn load() -> Result<Lib, String> {
    let mut tried = Vec::new();
    for name in candidates() {
        let Ok(c) = CString::new(name.clone()) else {
            continue;
        };
        match open(&c) {
            Ok(handle) => return resolve(handle).map_err(|e| format!("{name}: {e}")),
            Err(e) => tried.push(e),
        }
    }
    Err(tried.join("; "))
}

/// `dlopen()`, with `dlerror()` for the reason on failure.
fn open(name: &CStr) -> Result<*mut c_void, String> {
    // SAFETY: the name is a valid C string. Loading runs the library's constructors, which for
    // libslirp and GLib only set up their own state. dlerror() returns a thread-local string that
    // is copied before anything else can call into the dl functions on this thread.
    unsafe {
        let handle = libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if handle.is_null() {
            let e = libc::dlerror();
            if e.is_null() {
                return Err(name.to_string_lossy().into_owned());
            }
            return Err(CStr::from_ptr(e).to_string_lossy().into_owned());
        }
        Ok(handle)
    }
}

/// Looks up `name` in `handle` as a value of type `T`.
///
/// # Safety
///
/// `T` must be a function pointer type (or an `Option` of one) matching the C prototype of the
/// symbol, as declared in libslirp.h.
unsafe fn sym<T: Copy>(handle: *mut c_void, name: &CStr) -> Option<T> {
    assert_eq!(size_of::<T>(), size_of::<*mut c_void>());
    // SAFETY: handle came from dlopen and was never closed, and name is a valid C string. The
    // result is a plain address, reinterpreted as the pointer type the caller vouches for.
    unsafe {
        let p = libc::dlsym(handle, name.as_ptr());
        if p.is_null() { None } else { Some(std::mem::transmute_copy::<*mut c_void, T>(&p)) }
    }
}

fn parse_version(v: &str) -> u32 {
    let mut parts = v.split('.').map(|p| {
        p.bytes().take_while(u8::is_ascii_digit).fold(0u32, |a, b| a * 10 + u32::from(b - b'0'))
    });
    let major = parts.next().unwrap_or(0);
    let minor = parts.next().unwrap_or(0);
    major * 1000 + minor
}

/// [`sym`] for a symbol that has to be there.
///
/// # Safety
///
/// As for [`sym`].
unsafe fn need<T: Copy>(handle: *mut c_void, name: &CStr) -> Result<T, String> {
    // SAFETY: the caller vouches for T, as sym() requires.
    let f = unsafe { sym(handle, name) };
    f.ok_or_else(|| format!("missing symbol {}", name.to_string_lossy()))
}

fn resolve(handle: *mut c_void) -> Result<Lib, String> {
    // SAFETY: every symbol is looked up as the type of the Lib field it goes into, and each field
    // is declared after the prototype of the symbol in libslirp.h. slirp_pollfds_fill and
    // slirp_pollfds_fill_socket share one prototype on Unix, where slirp_os_socket is int.
    // slirp_version_string() returns a static string or null.
    unsafe {
        let version_string: unsafe extern "C" fn() -> *const c_char =
            need(handle, c"slirp_version_string")?;
        let p = version_string();
        let version = if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        };
        let version_code = parse_version(&version);
        let fill =
            if version_code >= 4009 { c"slirp_pollfds_fill_socket" } else { c"slirp_pollfds_fill" };
        Ok(Lib {
            version,
            version_code,
            new: need(handle, c"slirp_new")?,
            cleanup: need(handle, c"slirp_cleanup")?,
            pollfds_fill: need(handle, fill)?,
            pollfds_poll: need(handle, c"slirp_pollfds_poll")?,
            input: need(handle, c"slirp_input")?,
            handle_timer: sym(handle, c"slirp_handle_timer"),
            add_hostxfwd: need(handle, c"slirp_add_hostxfwd")?,
            remove_hostxfwd: need(handle, c"slirp_remove_hostxfwd")?,
            add_exec: need(handle, c"slirp_add_exec")?,
            add_guestfwd: need(handle, c"slirp_add_guestfwd")?,
            socket_can_recv: need(handle, c"slirp_socket_can_recv")?,
            socket_recv: need(handle, c"slirp_socket_recv")?,
            connection_info: need(handle, c"slirp_connection_info")?,
        })
    }
}

/// Copies a string libslirp allocated with `g_malloc` and frees it. GLib's allocator has been
/// the system one since 2.46, so `free()` is `g_free()`.
///
/// # Safety
///
/// `p` must be null or a NUL-terminated string allocated with `g_malloc`, owned by the caller.
pub(crate) unsafe fn take_gstring(p: *mut c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: per the contract p is a valid C string the caller owns, freed exactly once here.
    unsafe {
        let s = CStr::from_ptr(p).to_string_lossy().into_owned();
        libc::free(p.cast());
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions() {
        assert_eq!(parse_version("4.9.1"), 4009);
        assert_eq!(parse_version("4.7.0-dirty"), 4007);
        assert_eq!(parse_version(""), 0);
    }

    #[test]
    fn layout() {
        // Offsets from libslirp.h on an LP64 host.
        if size_of::<usize>() == 8 {
            assert_eq!(std::mem::offset_of!(SlirpConfig, vprefix_addr6), 28);
            assert_eq!(std::mem::offset_of!(SlirpConfig, vhost6), 48);
            assert_eq!(std::mem::offset_of!(SlirpConfig, vhostname), 64);
            assert_eq!(std::mem::offset_of!(SlirpConfig, vnameserver6), 104);
            assert_eq!(std::mem::offset_of!(SlirpConfig, vdnssearch), 120);
            assert_eq!(std::mem::offset_of!(SlirpConfig, if_mtu), 136);
            assert_eq!(std::mem::offset_of!(SlirpConfig, outbound_addr), 160);
            assert_eq!(std::mem::offset_of!(SlirpConfig, mfr_id), 180);
            assert_eq!(size_of::<SlirpConfig>(), 192);
            assert_eq!(size_of::<SlirpCb>(), 13 * 8);
        }
    }
}
