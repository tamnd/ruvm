// SPDX-License-Identifier: GPL-2.0-or-later

//! `whp_dispatch`: the WinHvPlatform.dll and WinHvEmulation.dll entry points, looked up at run
//! time like `init_whp_dispatch()` and `load_whp_dispatch_fns()` do.
//!
//! windows-sys declares these functions too, but as imports the loader resolves at start up,
//! which would stop ruvm from starting at all on a Windows without the Hypervisor Platform
//! feature. So only its types come from there and the prototypes are written out here.

use std::ffi::{CStr, c_void};
use std::sync::OnceLock;

use windows_sys::Win32::Foundation::HMODULE;
use windows_sys::Win32::System::Hypervisor::{
    WHV_EMULATOR_CALLBACKS, WHV_EMULATOR_STATUS, WHV_INTERRUPT_CONTROL, WHV_MEMORY_ACCESS_CONTEXT,
    WHV_PARTITION_HANDLE, WHV_TRANSLATE_GVA_RESULT, WHV_VP_EXIT_CONTEXT,
    WHV_X64_IO_PORT_ACCESS_CONTEXT,
};
use windows_sys::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExA,
};

use crate::WhpxError;

/// An HRESULT.
pub(crate) type Hr = i32;
type Handle = WHV_PARTITION_HANDLE;

/// One `WHV_REGISTER_VALUE`, as two halves. Every register fits in 128 bits and the union is
/// 16 byte aligned.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reg {
    pub(crate) lo: u64,
    pub(crate) hi: u64,
}

impl Reg {
    pub(crate) fn u64(v: u64) -> Reg {
        Reg { lo: v, hi: 0 }
    }

    pub(crate) fn u128(v: u128) -> Reg {
        Reg { lo: v as u64, hi: (v >> 64) as u64 }
    }
}

/// The functions QEMU's `whp_dispatch` holds. The optional ones are missing on older Windows.
#[derive(Debug)]
pub(crate) struct Dispatch {
    pub(crate) get_capability: unsafe extern "system" fn(i32, *mut c_void, u32, *mut u32) -> Hr,
    pub(crate) create_partition: unsafe extern "system" fn(*mut Handle) -> Hr,
    pub(crate) setup_partition: unsafe extern "system" fn(Handle) -> Hr,
    pub(crate) delete_partition: unsafe extern "system" fn(Handle) -> Hr,
    pub(crate) get_partition_property:
        unsafe extern "system" fn(Handle, i32, *mut c_void, u32, *mut u32) -> Hr,
    pub(crate) set_partition_property:
        unsafe extern "system" fn(Handle, i32, *const c_void, u32) -> Hr,
    pub(crate) map_gpa_range: unsafe extern "system" fn(Handle, *const c_void, u64, u64, i32) -> Hr,
    pub(crate) unmap_gpa_range: unsafe extern "system" fn(Handle, u64, u64) -> Hr,
    pub(crate) translate_gva: unsafe extern "system" fn(
        Handle,
        u32,
        u64,
        i32,
        *mut WHV_TRANSLATE_GVA_RESULT,
        *mut u64,
    ) -> Hr,
    pub(crate) create_vp: unsafe extern "system" fn(Handle, u32, u32) -> Hr,
    pub(crate) delete_vp: unsafe extern "system" fn(Handle, u32) -> Hr,
    pub(crate) run_vp: unsafe extern "system" fn(Handle, u32, *mut c_void, u32) -> Hr,
    pub(crate) cancel_run_vp: unsafe extern "system" fn(Handle, u32, u32) -> Hr,
    pub(crate) get_vp_registers:
        unsafe extern "system" fn(Handle, u32, *const i32, u32, *mut Reg) -> Hr,
    pub(crate) set_vp_registers:
        unsafe extern "system" fn(Handle, u32, *const i32, u32, *const Reg) -> Hr,
    pub(crate) request_interrupt:
        Option<unsafe extern "system" fn(Handle, *const WHV_INTERRUPT_CONTROL, u32) -> Hr>,
    pub(crate) get_lapic_state2:
        Option<unsafe extern "system" fn(Handle, u32, *mut c_void, u32, *mut u32) -> Hr>,
    pub(crate) set_lapic_state2:
        Option<unsafe extern "system" fn(Handle, u32, *const c_void, u32) -> Hr>,
    pub(crate) emulator_create:
        unsafe extern "system" fn(*const WHV_EMULATOR_CALLBACKS, *mut *mut c_void) -> Hr,
    pub(crate) emulator_destroy: unsafe extern "system" fn(*const c_void) -> Hr,
    pub(crate) emulator_try_io: unsafe extern "system" fn(
        *const c_void,
        *const c_void,
        *const WHV_VP_EXIT_CONTEXT,
        *const WHV_X64_IO_PORT_ACCESS_CONTEXT,
        *mut WHV_EMULATOR_STATUS,
    ) -> Hr,
    pub(crate) emulator_try_mmio: unsafe extern "system" fn(
        *const c_void,
        *const c_void,
        *const WHV_VP_EXIT_CONTEXT,
        *const WHV_MEMORY_ACCESS_CONTEXT,
        *mut WHV_EMULATOR_STATUS,
    ) -> Hr,
}

fn load(name: &'static CStr) -> Result<HMODULE, WhpxError> {
    // SAFETY: the name is NUL terminated and the flags only restrict where Windows looks, so a
    // DLL planted next to the executable is not picked up.
    let lib = unsafe {
        LoadLibraryExA(name.as_ptr().cast(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32)
    };
    if lib.is_null() {
        return Err(WhpxError::Library(name.to_str().unwrap_or("?")));
    }
    Ok(lib)
}

/// Looks up `name` in `lib`.
///
/// # Safety
///
/// `T` must be the function pointer type of the export.
unsafe fn sym<T: Copy>(lib: HMODULE, name: &'static CStr) -> Option<T> {
    // SAFETY: `lib` is a loaded module that is never unloaded and the name is NUL terminated.
    let f = unsafe { GetProcAddress(lib, name.as_ptr().cast()) }?;
    assert_eq!(size_of::<T>(), size_of_val(&f));
    // SAFETY: both are plain function pointers of the same size, and the caller vouches for
    // the prototype.
    Some(unsafe { std::mem::transmute_copy::<unsafe extern "system" fn() -> isize, T>(&f) })
}

/// Like [`sym`], but a missing export is an error.
///
/// # Safety
///
/// As for [`sym`].
unsafe fn need<T: Copy>(lib: HMODULE, name: &'static CStr) -> Result<T, WhpxError> {
    // SAFETY: passed on from the caller.
    unsafe { sym(lib, name) }.ok_or(WhpxError::Function(name.to_str().unwrap_or("?")))
}

fn init() -> Result<Dispatch, WhpxError> {
    let p = load(c"WinHvPlatform.dll")?;
    let e = load(c"WinHvEmulation.dll")?;
    // SAFETY: every field's type is the prototype from WinHvPlatform.h or WinHvEmulation.h,
    // with `Reg` standing in for `WHV_REGISTER_VALUE`, which has the same size and alignment.
    unsafe {
        Ok(Dispatch {
            get_capability: need(p, c"WHvGetCapability")?,
            create_partition: need(p, c"WHvCreatePartition")?,
            setup_partition: need(p, c"WHvSetupPartition")?,
            delete_partition: need(p, c"WHvDeletePartition")?,
            get_partition_property: need(p, c"WHvGetPartitionProperty")?,
            set_partition_property: need(p, c"WHvSetPartitionProperty")?,
            map_gpa_range: need(p, c"WHvMapGpaRange")?,
            unmap_gpa_range: need(p, c"WHvUnmapGpaRange")?,
            translate_gva: need(p, c"WHvTranslateGva")?,
            create_vp: need(p, c"WHvCreateVirtualProcessor")?,
            delete_vp: need(p, c"WHvDeleteVirtualProcessor")?,
            run_vp: need(p, c"WHvRunVirtualProcessor")?,
            cancel_run_vp: need(p, c"WHvCancelRunVirtualProcessor")?,
            get_vp_registers: need(p, c"WHvGetVirtualProcessorRegisters")?,
            set_vp_registers: need(p, c"WHvSetVirtualProcessorRegisters")?,
            request_interrupt: sym(p, c"WHvRequestInterrupt"),
            get_lapic_state2: sym(p, c"WHvGetVirtualProcessorInterruptControllerState2"),
            set_lapic_state2: sym(p, c"WHvSetVirtualProcessorInterruptControllerState2"),
            emulator_create: need(e, c"WHvEmulatorCreateEmulator")?,
            emulator_destroy: need(e, c"WHvEmulatorDestroyEmulator")?,
            emulator_try_io: need(e, c"WHvEmulatorTryIoEmulation")?,
            emulator_try_mmio: need(e, c"WHvEmulatorTryMmioEmulation")?,
        })
    }
}

/// The dispatch table, loaded on first use. The libraries stay loaded for the life of the
/// process.
pub(crate) fn dispatch() -> Result<&'static Dispatch, WhpxError> {
    static D: OnceLock<Result<Dispatch, WhpxError>> = OnceLock::new();
    D.get_or_init(init).as_ref().map_err(Clone::clone)
}
