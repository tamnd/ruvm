// SPDX-License-Identifier: GPL-2.0-or-later

//! The unsafe edge of the plugin host: `dlopen()` and friends, the raw pointers plugins hand to
//! the API, the glib functions the API has to call, and the current vCPU pointer.
//!
//! The pointer helpers trust what the plugin API promises about their argument (a C string, a
//! `GByteArray`, a buffer of the given size); they are only called with pointers a plugin passed
//! to an API function for that purpose. Everything else in the crate is safe code.

use std::cell::Cell;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::marker::PhantomData;
use std::sync::Mutex;

use ruvm_jit::Cpu;

/// A `dlopen()` handle, kept as an integer so that it can live in shared state.
pub(crate) type Handle = usize;

fn last_dlerror() -> String {
    // SAFETY: dlerror() takes no arguments and returns NULL or a C string that stays valid
    // until the next dl call on this thread, which is after we copied it.
    let p = unsafe { libc::dlerror() };
    cstr(p).unwrap_or_default()
}

/// `g_module_open(path, G_MODULE_BIND_LOCAL)`: the handle, or `g_module_error()`.
pub(crate) fn dlopen(path: &str) -> Result<Handle, String> {
    let Ok(c) = CString::new(path) else {
        return Err(format!("{path}: file name contains a NUL byte"));
    };
    // SAFETY: c is a valid C string; dlopen runs the library's constructors, which is what
    // loading a plugin means.
    let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if h.is_null() { Err(last_dlerror()) } else { Ok(h as Handle) }
}

/// `g_module_symbol()`: the address of `name` in `handle`, which can be 0 for a symbol whose
/// value is NULL, or the glib error message.
pub(crate) fn dlsym(handle: Handle, name: &str) -> Result<usize, String> {
    let c = CString::new(name).map_err(|e| e.to_string())?;
    last_dlerror();
    // SAFETY: handle came from dlopen() and was not closed, or is RTLD_DEFAULT; c is a valid
    // C string.
    let p = unsafe { libc::dlsym(handle as *mut c_void, c.as_ptr()) };
    let err = last_dlerror();
    if p.is_null() && !err.is_empty() { Err(format!("'{name}': {err}")) } else { Ok(p as usize) }
}

/// `g_module_close()`.
pub(crate) fn dlclose(handle: Handle) -> Result<(), String> {
    // SAFETY: handle came from dlopen() and is closed once, after the plugin's callbacks were
    // removed so nothing calls into it any more.
    let r = unsafe { libc::dlclose(handle as *mut c_void) };
    if r == 0 { Ok(()) } else { Err(last_dlerror()) }
}

/// `RTLD_DEFAULT` as a handle.
pub(crate) fn rtld_default() -> Handle {
    libc::RTLD_DEFAULT as Handle
}

/// The `int` at `addr`, the value of `qemu_plugin_version`.
pub(crate) fn read_int(addr: usize) -> i32 {
    // SAFETY: addr is the non-NULL address of the plugin's `int qemu_plugin_version`, which
    // the header declares with that type.
    unsafe { *(addr as *const c_int) }
}

/// `qemu_plugin_install_func_t`.
pub(crate) type InstallFn =
    extern "C-unwind" fn(u64, *const QemuInfoC, c_int, *const *const c_char) -> c_int;

/// The address of `qemu_plugin_install` as the function it is.
pub(crate) fn install_fn(addr: usize) -> InstallFn {
    assert_ne!(addr, 0);
    // SAFETY: addr is the non-NULL address of the plugin's qemu_plugin_install, which
    // qemu-plugin.h declares with exactly this prototype; function pointers and usize have the
    // same size.
    unsafe { std::mem::transmute::<usize, InstallFn>(addr) }
}

/// `qemu_info_t`.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct QemuInfoC {
    pub(crate) target_name: *const c_char,
    pub(crate) version_min: c_int,
    pub(crate) version_cur: c_int,
    pub(crate) system_emulation: bool,
    pub(crate) smp_vcpus: c_int,
    pub(crate) max_vcpus: c_int,
}

/// The head of glib's `GArray` and `GByteArray`.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct GArray {
    pub(crate) data: *mut u8,
    pub(crate) len: c_uint,
}

type GArrayNew = extern "C" fn(c_int, c_int, c_uint) -> *mut GArray;
type GArrayAppendVals = extern "C" fn(*mut GArray, *const c_void, c_uint) -> *mut GArray;
type GByteArraySetSize = extern "C" fn(*mut GArray, c_uint) -> *mut GArray;
type GByteArrayAppend = extern "C" fn(*mut GArray, *const u8, c_uint) -> *mut GArray;
type GStrdup = extern "C" fn(*const c_char) -> *mut c_char;

/// The glib functions the API calls, found in the process or in a loaded plugin.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Glib {
    array_new: GArrayNew,
    array_append_vals: GArrayAppendVals,
    byte_array_set_size: GByteArraySetSize,
    byte_array_append: GByteArrayAppend,
    strdup: GStrdup,
}

static GLIB: Mutex<Option<Glib>> = Mutex::new(None);

fn find(handles: &[Handle], name: &str) -> Option<usize> {
    std::iter::once(rtld_default())
        .chain(handles.iter().copied())
        .find_map(|h| dlsym(h, name).ok().filter(|&a| a != 0))
}

/// Keep the library that holds `addr` loaded for the life of the process. glib may have come
/// in with a plugin, and must stay when that plugin is unloaded because its functions are kept.
fn pin(addr: usize) {
    // SAFETY: dladdr() only reads addr, and fills info, which is a plain C struct for which all
    // zeroes is a valid value. dli_fname is NULL or the C string of a loaded library, which is
    // valid until that library is unloaded, and it is not while dlopen() takes a reference.
    // The handle is never closed.
    unsafe {
        let mut info = std::mem::zeroed::<libc::Dl_info>();
        if libc::dladdr(addr as *const c_void, &mut info) != 0 && !info.dli_fname.is_null() {
            libc::dlopen(info.dli_fname, libc::RTLD_NOW | libc::RTLD_LOCAL | libc::RTLD_NODELETE);
        }
    }
}

impl Glib {
    /// Find glib, looking in the process first and then in `handles`, the loaded plugins,
    /// which all link it. The result is kept once found.
    pub(crate) fn get(handles: &[Handle]) -> Option<Glib> {
        let mut g = GLIB.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(glib) = *g {
            return Some(glib);
        }
        let a = find(handles, "g_array_new")?;
        let b = find(handles, "g_array_append_vals")?;
        let c = find(handles, "g_byte_array_set_size")?;
        let d = find(handles, "g_byte_array_append")?;
        let e = find(handles, "g_strdup")?;
        pin(a);
        // SAFETY: each address is the glib function of that name, and the types are glib's
        // prototypes for them (gboolean and guint are int and unsigned int).
        let glib = unsafe {
            Glib {
                array_new: std::mem::transmute::<usize, GArrayNew>(a),
                array_append_vals: std::mem::transmute::<usize, GArrayAppendVals>(b),
                byte_array_set_size: std::mem::transmute::<usize, GByteArraySetSize>(c),
                byte_array_append: std::mem::transmute::<usize, GByteArrayAppend>(d),
                strdup: std::mem::transmute::<usize, GStrdup>(e),
            }
        };
        *g = Some(glib);
        Some(glib)
    }

    /// `g_array_new(true, true, sizeof(T))` filled with `items`.
    pub(crate) fn array_of<T: Copy>(&self, items: &[T]) -> *mut GArray {
        let size = c_uint::try_from(size_of::<T>()).expect("element too large");
        let arr = (self.array_new)(1, 1, size);
        let n = c_uint::try_from(items.len()).expect("too many elements");
        if n > 0 {
            (self.array_append_vals)(arr, items.as_ptr().cast(), n);
        }
        arr
    }

    /// `g_strdup(s)`.
    pub(crate) fn strdup(&self, s: &str) -> *mut c_char {
        let c = CString::new(s.replace('\0', "")).unwrap_or_default();
        (self.strdup)(c.as_ptr())
    }

    /// `g_byte_array_append()` of `bytes` to the plugin's `ba`.
    pub(crate) fn byte_array_append(&self, ba: *mut GArray, bytes: &[u8]) {
        let n = c_uint::try_from(bytes.len()).expect("register too large");
        (self.byte_array_append)(ba, bytes.as_ptr(), n);
    }

    /// `g_byte_array_set_size(ba, bytes.len())` and copy `bytes` in.
    pub(crate) fn byte_array_fill(&self, ba: *mut GArray, bytes: &[u8]) {
        let n = c_uint::try_from(bytes.len()).expect("buffer too large");
        (self.byte_array_set_size)(ba, n);
        // SAFETY: ba is a GByteArray the plugin passed in, which g_byte_array_set_size() just
        // made exactly bytes.len() long.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), (*ba).data, bytes.len()) }
    }
}

/// A copy of the bytes of the plugin's `GByteArray`.
pub(crate) fn byte_array_bytes(ba: *const GArray) -> Vec<u8> {
    if ba.is_null() {
        return Vec::new();
    }
    // SAFETY: ba is a GByteArray the plugin passed in: data points to len bytes.
    unsafe {
        let a = &*ba;
        if a.data.is_null() || a.len == 0 {
            return Vec::new();
        }
        std::slice::from_raw_parts(a.data, a.len as usize).to_vec()
    }
}

/// A copy of the C string at `p`, or `None` for NULL.
pub(crate) fn cstr(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    // SAFETY: p is a NUL terminated string the plugin (or libc) passed in.
    Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

/// `*p = v` for an out parameter of the plugin.
pub(crate) fn write_out<T: Copy>(p: *mut T, v: T) {
    if !p.is_null() {
        // SAFETY: p is an out parameter the plugin passed in, valid for writing a T.
        unsafe { p.write(v) }
    }
}

/// Copy `src` to the plugin's buffer `dest`, which holds at least `src.len()` bytes.
pub(crate) fn copy_out(dest: *mut c_void, src: &[u8]) {
    if !dest.is_null() && !src.is_empty() {
        // SAFETY: the plugin passed dest with room for at least the length it asked for, and
        // src is no longer than that.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dest.cast::<u8>(), src.len()) }
    }
}

thread_local! {
    static CUR: Cell<*mut ()> = const { Cell::new(std::ptr::null_mut()) };
    static BUSY: Cell<bool> = const { Cell::new(false) };
}

/// Restores the previous current vCPU when dropped.
struct Restore(*mut ());

impl Drop for Restore {
    fn drop(&mut self) {
        CUR.with(|c| c.set(self.0));
    }
}

/// Run `f` with `cpu` as `current_cpu`, for the API functions a callback calls.
pub(crate) fn enter_cpu<R>(cpu: &mut Cpu<'_>, f: impl FnOnce() -> R) -> R {
    let p = std::ptr::from_mut(cpu).cast::<()>();
    let _restore = Restore(CUR.with(|c| c.replace(p)));
    f()
}

/// Whether there is a current vCPU.
pub(crate) fn has_cpu() -> bool {
    CUR.with(|c| !c.get().is_null())
}

/// Restores the busy flag when dropped.
struct Unbusy(PhantomData<*mut ()>);

impl Drop for Unbusy {
    fn drop(&mut self) {
        BUSY.with(|b| b.set(false));
    }
}

/// Run `f` on `current_cpu`.
///
/// # Panics
///
/// When there is no current vCPU, like QEMU's `g_assert(current_cpu)`, or when called from
/// inside `f`.
pub(crate) fn with_cpu<R>(f: impl FnOnce(&mut Cpu<'_>) -> R) -> R {
    let p = CUR.with(Cell::get);
    assert!(!p.is_null(), "assertion failed: (current_cpu)");
    assert!(!BUSY.with(|b| b.replace(true)), "current_cpu used recursively");
    let _unbusy = Unbusy(PhantomData);
    // SAFETY: p was set by enter_cpu() from a `&mut Cpu` that enter_cpu() keeps borrowed until
    // it returns and resets CUR, and this is the same thread. While f runs, BUSY stops a second
    // reference from being made, and the reborrow cannot outlive this call. The lifetime
    // parameter is not known here; the reborrow gives f a shorter one.
    let cpu = unsafe { &mut *p.cast::<Cpu<'_>>() };
    f(&mut cpu.rb())
}
