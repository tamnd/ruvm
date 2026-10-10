// SPDX-License-Identifier: GPL-2.0-or-later

//! The Hypervisor.framework calls QEMU makes, from the macOS SDK headers `hv.h`, `hv_vm.h`,
//! `hv_vcpu.h`, `hv_vm_config.h`, `hv_vcpu_config.h` and `hv_gic.h`.
//!
//! The calls from macOS 15 and later (nested virtualization and the GIC) are not linked
//! directly, since a binary that names them does not load on older macOS. They are looked up
//! with `dlsym()` instead, the way QEMU guards them with `__builtin_available()`.

use std::ffi::{c_char, c_void};
use std::sync::OnceLock;

/// `hv_return_t`.
pub(crate) type HvReturn = i32;
/// `hv_vcpu_t`.
pub(crate) type HvVcpu = u64;
/// `hv_ipa_t`.
pub(crate) type HvIpa = u64;
/// `hv_memory_flags_t`.
pub(crate) type HvMemoryFlags = u64;
/// `hv_vm_config_t`, `hv_vcpu_config_t`, `hv_gic_config_t` and `hv_gic_state_t`: OS objects,
/// released with `os_release()`.
pub(crate) type OsObject = *mut c_void;

pub(crate) const HV_MEMORY_READ: HvMemoryFlags = 1;
pub(crate) const HV_MEMORY_WRITE: HvMemoryFlags = 2;
pub(crate) const HV_MEMORY_EXEC: HvMemoryFlags = 4;

pub(crate) const HV_EXIT_REASON_CANCELED: u32 = 0;
pub(crate) const HV_EXIT_REASON_EXCEPTION: u32 = 1;
pub(crate) const HV_EXIT_REASON_VTIMER_ACTIVATED: u32 = 2;

/// `HV_REG_PC`. X0 to X30 are 0 to 30.
pub(crate) const HV_REG_PC: u32 = 31;
pub(crate) const HV_REG_FPCR: u32 = 32;
pub(crate) const HV_REG_FPSR: u32 = 33;
pub(crate) const HV_REG_CPSR: u32 = 34;

pub(crate) const HV_INTERRUPT_TYPE_IRQ: u32 = 0;
pub(crate) const HV_INTERRUPT_TYPE_FIQ: u32 = 1;

/// `hv_vcpu_exit_exception_t`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct HvExitException {
    pub(crate) syndrome: u64,
    pub(crate) virtual_address: u64,
    pub(crate) physical_address: u64,
}

/// `hv_vcpu_exit_t`, owned by the framework and filled by each `hv_vcpu_run()`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct HvVcpuExit {
    pub(crate) reason: u32,
    pub(crate) exception: HvExitException,
}

#[link(name = "Hypervisor", kind = "framework")]
unsafe extern "C" {
    pub(crate) fn hv_vm_create(config: OsObject) -> HvReturn;
    pub(crate) fn hv_vm_destroy() -> HvReturn;
    pub(crate) fn hv_vm_map(
        addr: *mut c_void,
        ipa: HvIpa,
        size: usize,
        flags: HvMemoryFlags,
    ) -> HvReturn;
    pub(crate) fn hv_vm_unmap(ipa: HvIpa, size: usize) -> HvReturn;
    pub(crate) fn hv_vm_protect(ipa: HvIpa, size: usize, flags: HvMemoryFlags) -> HvReturn;

    pub(crate) fn hv_vm_config_create() -> OsObject;
    pub(crate) fn hv_vm_config_get_max_ipa_size(ipa_bit_length: *mut u32) -> HvReturn;
    pub(crate) fn hv_vm_config_get_default_ipa_size(ipa_bit_length: *mut u32) -> HvReturn;
    pub(crate) fn hv_vm_config_set_ipa_size(config: OsObject, ipa_bit_length: u32) -> HvReturn;

    pub(crate) fn hv_vcpu_config_create() -> OsObject;
    pub(crate) fn hv_vcpu_config_get_feature_reg(
        config: OsObject,
        feature_reg: u32,
        value: *mut u64,
    ) -> HvReturn;

    pub(crate) fn hv_vcpu_create(
        vcpu: *mut HvVcpu,
        exit: *mut *const HvVcpuExit,
        config: OsObject,
    ) -> HvReturn;
    pub(crate) fn hv_vcpu_destroy(vcpu: HvVcpu) -> HvReturn;
    pub(crate) fn hv_vcpu_run(vcpu: HvVcpu) -> HvReturn;
    pub(crate) fn hv_vcpus_exit(vcpus: *const HvVcpu, vcpu_count: u32) -> HvReturn;
    pub(crate) fn hv_vcpu_get_reg(vcpu: HvVcpu, reg: u32, value: *mut u64) -> HvReturn;
    pub(crate) fn hv_vcpu_set_reg(vcpu: HvVcpu, reg: u32, value: u64) -> HvReturn;
    /// `hv_simd_fp_uchar16_t` is a 16 byte vector, which a 16 byte aligned u128 matches in
    /// memory.
    pub(crate) fn hv_vcpu_get_simd_fp_reg(vcpu: HvVcpu, reg: u32, value: *mut u128) -> HvReturn;
    pub(crate) fn hv_vcpu_get_sys_reg(vcpu: HvVcpu, reg: u16, value: *mut u64) -> HvReturn;
    pub(crate) fn hv_vcpu_set_sys_reg(vcpu: HvVcpu, reg: u16, value: u64) -> HvReturn;
    pub(crate) fn hv_vcpu_set_pending_interrupt(vcpu: HvVcpu, kind: u32, pending: bool)
    -> HvReturn;
    pub(crate) fn hv_vcpu_set_vtimer_mask(vcpu: HvVcpu, vtimer_is_masked: bool) -> HvReturn;
    pub(crate) fn hv_vcpu_set_vtimer_offset(vcpu: HvVcpu, vtimer_offset: u64) -> HvReturn;
}

// `hv_vcpu_set_simd_fp_reg()` takes the vector by value, in V0, and stable Rust cannot pass
// a SIMD type through FFI. This shim takes a pointer in X2 instead, loads the vector into Q0
// and tail calls the framework with X0 and W1 untouched.
std::arch::global_asm!(
    ".globl _ruvm_hv_vcpu_set_simd_fp_reg",
    ".p2align 2",
    "_ruvm_hv_vcpu_set_simd_fp_reg:",
    "ldr q0, [x2]",
    "b _hv_vcpu_set_simd_fp_reg",
);

unsafe extern "C" {
    /// `hv_vcpu_set_simd_fp_reg()` through the shim above.
    pub(crate) fn ruvm_hv_vcpu_set_simd_fp_reg(
        vcpu: HvVcpu,
        reg: u32,
        value: *const u128,
    ) -> HvReturn;
    /// `os_release()` from libSystem.
    pub(crate) fn os_release(object: *mut c_void);
    /// `mach_absolute_time()`: the host counter, which is what CNTVCT reads without an offset.
    pub(crate) fn mach_absolute_time() -> u64;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

/// `RTLD_DEFAULT` on macOS.
const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;

/// The calls that appeared in macOS 15.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Macos15 {
    pub(crate) vm_config_get_el2_supported: unsafe extern "C" fn(*mut bool) -> HvReturn,
    pub(crate) vm_config_set_el2_enabled: unsafe extern "C" fn(OsObject, bool) -> HvReturn,
    pub(crate) gic_config_create: unsafe extern "C" fn() -> OsObject,
    pub(crate) gic_config_set_distributor_base: unsafe extern "C" fn(OsObject, HvIpa) -> HvReturn,
    pub(crate) gic_config_set_redistributor_base: unsafe extern "C" fn(OsObject, HvIpa) -> HvReturn,
    pub(crate) gic_create: unsafe extern "C" fn(OsObject) -> HvReturn,
    pub(crate) gic_set_spi: unsafe extern "C" fn(u32, bool) -> HvReturn,
    pub(crate) gic_send_msi: unsafe extern "C" fn(HvIpa, u32) -> HvReturn,
    pub(crate) gic_reset: unsafe extern "C" fn() -> HvReturn,
    pub(crate) gic_get_distributor_size: unsafe extern "C" fn(*mut usize) -> HvReturn,
    pub(crate) gic_get_redistributor_region_size: unsafe extern "C" fn(*mut usize) -> HvReturn,
    pub(crate) gic_get_redistributor_size: unsafe extern "C" fn(*mut usize) -> HvReturn,
    pub(crate) gic_get_spi_interrupt_range: unsafe extern "C" fn(*mut u32, *mut u32) -> HvReturn,
    pub(crate) gic_state_create: unsafe extern "C" fn() -> OsObject,
    pub(crate) gic_state_get_size: unsafe extern "C" fn(OsObject, *mut usize) -> HvReturn,
    pub(crate) gic_state_get_data: unsafe extern "C" fn(OsObject, *mut c_void) -> HvReturn,
    pub(crate) gic_set_state: unsafe extern "C" fn(*const c_void, usize) -> HvReturn,
}

/// The macOS 15 calls, or `None` on an older macOS.
pub(crate) fn macos15() -> Option<&'static Macos15> {
    static CELL: OnceLock<Option<Macos15>> = OnceLock::new();
    CELL.get_or_init(load_macos15).as_ref()
}

/// Reads a symbol address as the function pointer type `F`.
///
/// # Safety
///
/// `p` must be the address of a function whose prototype is `F`.
unsafe fn fn_ptr<F: Copy>(p: *mut c_void) -> F {
    assert_eq!(size_of::<F>(), size_of::<*mut c_void>());
    // SAFETY: the sizes match, and the caller promises `p` is a function of type `F`.
    unsafe { std::mem::transmute_copy(&p) }
}

fn load_macos15() -> Option<Macos15> {
    let sym = |name: &std::ffi::CStr| {
        // SAFETY: dlsym() only reads the NUL terminated name, and RTLD_DEFAULT searches the
        // images already loaded, which include Hypervisor.framework since this crate links it.
        let p = unsafe { dlsym(RTLD_DEFAULT, name.as_ptr()) };
        if p.is_null() { None } else { Some(p) }
    };
    macro_rules! load {
        ($($field:ident = $name:literal),* $(,)?) => {
            Macos15 {
                $(
                    // SAFETY: the symbol is the framework function of that name, and the field
                    // type is its prototype from the SDK header, so the pointer has that type.
                    $field: unsafe { fn_ptr(sym($name)?) },
                )*
            }
        };
    }
    Some(load! {
        vm_config_get_el2_supported = c"hv_vm_config_get_el2_supported",
        vm_config_set_el2_enabled = c"hv_vm_config_set_el2_enabled",
        gic_config_create = c"hv_gic_config_create",
        gic_config_set_distributor_base = c"hv_gic_config_set_distributor_base",
        gic_config_set_redistributor_base = c"hv_gic_config_set_redistributor_base",
        gic_create = c"hv_gic_create",
        gic_set_spi = c"hv_gic_set_spi",
        gic_send_msi = c"hv_gic_send_msi",
        gic_reset = c"hv_gic_reset",
        gic_get_distributor_size = c"hv_gic_get_distributor_size",
        gic_get_redistributor_region_size = c"hv_gic_get_redistributor_region_size",
        gic_get_redistributor_size = c"hv_gic_get_redistributor_size",
        gic_get_spi_interrupt_range = c"hv_gic_get_spi_interrupt_range",
        gic_state_create = c"hv_gic_state_create",
        gic_state_get_size = c"hv_gic_state_get_size",
        gic_state_get_data = c"hv_gic_state_get_data",
        gic_set_state = c"hv_gic_set_state",
    })
}
