// SPDX-License-Identifier: GPL-2.0-or-later

//! The display side of the command line and the monitor: `-vga` (`select_vgahw()` of QEMU's
//! system/vl.c), `pc_vga_init()` of hw/i386/pc.c, `-device VGA`, `bochs-display`, `ramfb`,
//! `virtio-gpu-pci` and `virtio-gpu-device` on q35, microvm and Arm virt, and the QMP
//! `screendump` command of ui/ui-qmp-cmds.c. The virtio keyboard, mouse, tablet and multitouch
//! devices go through here too, in their `-pci` and `-device` forms, because they take the
//! same buses and are realized in the same order as the display devices.
//!
//! Where this differs from QEMU:
//! - q35 gets a VGA card only with `-vga std` or `-device VGA`. QEMU plugs the std VGA by
//!   default.
//! - `-vga` knows `std` and `none`. The other types fail with "... not available", which is
//!   what a QEMU built without them says. `retrace=` is checked and has no effect.
//! - On Arm virt the display functions get no option ROM, so `romfile` has no effect there.
//! - The HMP `screendump` command is not here.
//! - `virtio-gpu-device` on q35 fails with QEMU's "No 'virtio-bus' bus found" error when it is
//!   realized, after the properties are checked, instead of before. On x86 it takes no `bus`.
//! - virtio-gpu takes the properties listed in [`plan`]. `hostmem`, `outputs` and the generic
//!   virtio and PCI ones such as `ats` or `rombar` are not there. The same goes for the virtio
//!   input devices, which take `serial`, `display` and `head`.
//! - The virtio input devices are not there on RISC-V virt yet.
//!
//! It also connects the input layer of ui/input.c: the i8042 of q35, the clock `send-key` paces
//! its keys on, and the `query-mice`, `send-key` and `input-send-event` commands.
//!
//! The local displays are here too: `-display`, `-nographic` and `-full-screen` as `dpy` of
//! QEMU's system/vl.c, the default display, `query-display-options` and opening the display.
//! The local displays are GTK, with the `ui-gtk` feature, and SDL, with `ui-sdl`. Without either
//! the default is `none`, where QEMU would also open VNC on localhost:0. With `ui-dbus` there is
//! also `-display dbus`, which gets `-name`, `-uuid` and the PCI address of each display function
//! from here. With `ui-cocoa` on macOS there is `-display cocoa`, the default when neither GTK
//! nor SDL is built in, which leaves [`take_ui_main`] behind for vl.rs to hand the main thread
//! to.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use ruvm_base::report::{Location, error_report, warn_report};
use ruvm_base::{Error, Result};
use ruvm_hw_core::Clock;
use ruvm_hw_core::fw_cfg::{DmaMemory, FwCfgState};
use ruvm_hw_display::bochs_display::{BOCHS_DISPLAY_ROMFILE, BochsDisplay, BochsDisplayProps};
use ruvm_hw_display::edid::EdidInfo;
use ruvm_hw_display::ramfb::Ramfb;
use ruvm_hw_display::vga_pci::{VGA_ROMFILE, VgaPci, VgaPciProps};
use ruvm_hw_display::virtio_gpu::{TYPE_VIRTIO_GPU, TYPE_VIRTIO_GPU_PCI, VirtioGpu, VirtioGpuConf};
use ruvm_hw_input::virtio_input::{VirtioInput, VirtioInputConf, VirtioInputKind};
use ruvm_hw_pci::regs::PCI_ROM_SLOT;
use ruvm_hw_pci::{PciBus, PciDevice};
use ruvm_hw_virtio::mmio::VIRTIO_MMIO_FORCE_LEGACY_DEFAULT;
use ruvm_hw_virtio::{
    AddressSpaceMemory, SharedGuestMemory, VirtioBackend, VirtioDeviceClass, VirtioMmio, VirtioPci,
    VirtioPciProps,
};
use ruvm_machine_arm::virt::VirtMachine;
use ruvm_machine_x86::{FirmwareSearch, VirtioHandle, X86Board};
use ruvm_mem::AddressSpace;
use ruvm_monitor::{Commands, MonitorQmp};
use ruvm_qapi::commands::{
    register_input_send_event, register_query_display_options, register_query_mice,
    register_screendump, register_send_key,
};
use ruvm_qapi::opts::QemuOpts;
use ruvm_qapi::types::{DisplayGLMode, DisplayOptions, DisplayOptionsU, ImageFormat};
use ruvm_qapi::visit::{QObjectInputVisitor, Visitor, VisitorExt};
use ruvm_qapi::{QDict, QValue};
use ruvm_ui::console::{ConsoleDevice, DisplayState};
use ruvm_ui::input::InputState;
use ruvm_ui::{keymaps, screendump};

use crate::vl::Vm;
use crate::x86::Located;

/// `vga_interfaces[]`: the `-vga` name, the description, and whether ruvm has the device.
const VGA_INTERFACES: &[(&str, &str, bool)] = &[
    ("none", "no graphic card", true),
    ("std", "standard VGA", true),
    ("cirrus", "Cirrus VGA", false),
    ("vmware", "VMWare SVGA", false),
    ("virtio", "Virtio VGA", false),
    ("qxl", "QXL VGA", false),
    ("tcx", "TCX framebuffer", false),
    ("cg3", "CG3 framebuffer", false),
];

/// `vga_interface_created`: a board plugged the card `-vga` asked for.
static VGA_INTERFACE_CREATED: AtomicBool = AtomicBool::new(false);

/// What `-vga` selected, `vga_interface_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VgaInterface {
    None,
    Std,
}

/// `select_vgahw()` for the last `-vga`. `Ok(None)` is `help`, after the list is printed. The
/// default is `std` on every machine, as QEMU picks it when there is no Cirrus VGA.
pub(crate) fn select_vgahw(p: &str) -> std::result::Result<Option<VgaInterface>, String> {
    if p == "help" {
        for (name, desc, available) in VGA_INTERFACES {
            if *available {
                let default = if *name == "std" { " (default)" } else { "" };
                println!("{name:<20} {desc}{default}");
            }
        }
        return Ok(None);
    }
    let unknown = || format!("unknown vga type: {p}");
    let Some((i, mut opts)) = VGA_INTERFACES
        .iter()
        .enumerate()
        .find_map(|(i, (name, ..))| p.strip_prefix(name).map(|rest| (i, rest)))
    else {
        return Err(unknown());
    };
    let (_, desc, available) = VGA_INTERFACES[i];
    if !available {
        return Err(format!("{desc} not available"));
    }
    while !opts.is_empty() {
        let rest = opts.strip_prefix(",retrace=").ok_or_else(unknown)?;
        opts = rest
            .strip_prefix("dumb")
            .or_else(|| rest.strip_prefix("precise"))
            .ok_or_else(unknown)?;
    }
    Ok(Some(if i == 0 { VgaInterface::None } else { VgaInterface::Std }))
}

/// The end of `qemu_create_cli_devices()`: warns when `-vga` asked for a card the machine
/// does not plug.
pub(crate) fn check_vga_created(vga: Option<VgaInterface>) {
    if vga == Some(VgaInterface::Std) && !VGA_INTERFACE_CREATED.load(Ordering::Acquire) {
        warn_report(
            "A -vga option was passed but this machine type does not use that option; No VGA device has been created",
        );
    }
}

/// `dpy` of QEMU's system/vl.c: the last `-display`, with `-nographic` and `-full-screen`.
static DPY: Mutex<Option<DisplayOptions>> = Mutex::new(None);

fn dpy() -> MutexGuard<'static, Option<DisplayOptions>> {
    DPY.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `qemu_display_help()`.
pub(crate) fn display_help() -> String {
    let mut types = String::from("none\n");
    if cfg!(feature = "ui-gtk") {
        types.push_str("gtk\n");
    }
    if cfg!(feature = "ui-sdl") {
        types.push_str("sdl\n");
    }
    if cfg!(all(feature = "ui-cocoa", target_os = "macos")) {
        types.push_str("cocoa\n");
    }
    if cfg!(all(feature = "ui-dbus", unix)) {
        types.push_str("dbus\n");
    }
    format!(
        "Available display backend types:\n{types}\nSome display backends support suboptions, which can be set with\n   -display backend,option=value,option=value...\nFor a short list of the suboptions for each display, see the top-level -help output; more detail is in the documentation.\n"
    )
}

/// `parse_display_qapi()`: a `-display` replaces the options before it.
pub(crate) fn set_display(opts: DisplayOptions) {
    *dpy() = Some(opts);
}

/// `-nographic`, which keeps the other options of an earlier `-display`.
pub(crate) fn set_display_none() {
    dpy().get_or_insert_default().u = DisplayOptionsU::None;
}

/// `-full-screen`.
pub(crate) fn set_full_screen() {
    dpy().get_or_insert_default().full_screen = Some(true);
}

/// `qemu_setup_display()` with the display checks of `qemu_create_early_backends()`, then
/// `qemu_display_init()`. `version` is what `-version` prints, for the About panel of Cocoa.
/// The error is the exit status.
pub(crate) fn init_displays(vm: &Vm, version: &str) -> std::result::Result<(), u8> {
    let opts = {
        let mut guard = dpy();
        let d = guard.get_or_insert_default();
        if d.u == DisplayOptionsU::Default && !ruvm_ui::vnc::configured() {
            d.u = default_display();
        }
        if d.u == DisplayOptionsU::Default {
            d.u = DisplayOptionsU::None;
        }
        d.clone()
    };
    if opts.window_close.is_some() && !is_gtk(&opts) && !is_sdl(&opts) {
        error_report("window-close is only valid for GTK and SDL, ignoring option");
    }
    if opts.gl.is_some_and(|gl| gl != DisplayGLMode::Off) {
        // early_dbus_init() only warns, and the check below fails.
        if is_dbus(&opts) {
            error_report("dbus: GL rendering is not supported");
        }
        error_report("OpenGL support was not enabled in this build of QEMU");
        return Err(1);
    }
    open_display(vm, &opts, version)
}

/// `qemu_display_find_default()`: the first of GTK, SDL and Cocoa that is built in.
fn default_display() -> DisplayOptionsU {
    #[cfg(feature = "ui-gtk")]
    return DisplayOptionsU::Gtk(Default::default());
    #[cfg(all(feature = "ui-sdl", not(feature = "ui-gtk")))]
    return DisplayOptionsU::Sdl(Default::default());
    #[cfg(all(
        feature = "ui-cocoa",
        target_os = "macos",
        not(any(feature = "ui-gtk", feature = "ui-sdl"))
    ))]
    return DisplayOptionsU::Cocoa(Default::default());
    #[cfg(not(any(
        feature = "ui-gtk",
        feature = "ui-sdl",
        all(feature = "ui-cocoa", target_os = "macos")
    )))]
    DisplayOptionsU::Default
}

#[cfg(feature = "ui-gtk")]
fn is_gtk(opts: &DisplayOptions) -> bool {
    matches!(opts.u, DisplayOptionsU::Gtk(_))
}

#[cfg(not(feature = "ui-gtk"))]
fn is_gtk(_opts: &DisplayOptions) -> bool {
    false
}

#[cfg(feature = "ui-sdl")]
fn is_sdl(opts: &DisplayOptions) -> bool {
    matches!(opts.u, DisplayOptionsU::Sdl(_))
}

#[cfg(not(feature = "ui-sdl"))]
fn is_sdl(_opts: &DisplayOptions) -> bool {
    false
}

#[cfg(all(feature = "ui-cocoa", target_os = "macos"))]
fn is_cocoa(opts: &DisplayOptions) -> bool {
    matches!(opts.u, DisplayOptionsU::Cocoa(_))
}

#[cfg(all(feature = "ui-dbus", unix))]
fn is_dbus(opts: &DisplayOptions) -> bool {
    matches!(opts.u, DisplayOptionsU::Dbus(_))
}

#[cfg(not(all(feature = "ui-dbus", unix)))]
fn is_dbus(_opts: &DisplayOptions) -> bool {
    false
}

/// `qemu_uuid`, which `-display dbus` shows.
static QEMU_UUID: Mutex<Option<[u8; 16]>> = Mutex::new(None);

/// Sets the `qemu_uuid` the displays see, from `-uuid` or `-smbios type=1,uuid=`.
pub(crate) fn set_qemu_uuid(uuid: Option<[u8; 16]>) {
    *QEMU_UUID.lock().unwrap_or_else(PoisonError::into_inner) = uuid;
}

/// `qemu_uuid` as `qemu_uuid_unparse()` writes it, all zeros when none was given.
pub(crate) fn qemu_uuid_string() -> String {
    let u = QEMU_UUID.lock().unwrap_or_else(PoisonError::into_inner).unwrap_or_default();
    let hex =
        |r: std::ops::Range<usize>| u[r].iter().map(|b| format!("{b:02x}")).collect::<String>();
    format!("{}-{}-{}-{}-{}", hex(0..4), hex(4..6), hex(6..8), hex(8..10), hex(10..16))
}

/// What the main thread runs instead of the machine's main loop, which then moves to another
/// thread: `qemu_main` of QEMU, which Cocoa sets.
static UI_MAIN: Mutex<Option<fn() -> !>> = Mutex::new(None);

/// The function the main thread hands itself to once the machine is up, if a display wants it.
pub(crate) fn take_ui_main() -> Option<fn() -> !> {
    UI_MAIN.lock().unwrap_or_else(PoisonError::into_inner).take()
}

#[cfg_attr(
    not(any(
        feature = "ui-gtk",
        feature = "ui-sdl",
        all(feature = "ui-cocoa", target_os = "macos"),
        all(feature = "ui-dbus", unix)
    )),
    allow(unused_variables)
)]
fn open_display(vm: &Vm, opts: &DisplayOptions, version: &str) -> std::result::Result<(), u8> {
    #[cfg(feature = "ui-gtk")]
    if is_gtk(opts) {
        let hooks = Arc::new(GtkHooks(Arc::downgrade(&vm.runstate)));
        return ruvm_ui::gtk::init(
            DisplayState::global(),
            InputState::global(),
            opts,
            vm.name.as_deref(),
            hooks,
        );
    }
    #[cfg(feature = "ui-sdl")]
    if is_sdl(opts) {
        let hooks = Arc::new(SdlHooks(Arc::downgrade(&vm.runstate)));
        return ruvm_ui::sdl::init(
            DisplayState::global(),
            InputState::global(),
            opts,
            vm.name.as_deref(),
            hooks,
        );
    }
    #[cfg(all(feature = "ui-cocoa", target_os = "macos"))]
    if is_cocoa(opts) {
        let hooks = Arc::new(CocoaHooks(Arc::downgrade(&vm.runstate)));
        ruvm_ui::cocoa::init(
            DisplayState::global(),
            InputState::global(),
            opts,
            vm.name.as_deref(),
            version,
            hooks,
        )?;
        *UI_MAIN.lock().unwrap_or_else(PoisonError::into_inner) = Some(ruvm_ui::cocoa::run);
        return Ok(());
    }
    #[cfg(all(feature = "ui-dbus", unix))]
    if is_dbus(opts) {
        let (a, b, c) = ruvm_monitor::control::QEMU_VERSION;
        let name = vm.name.clone().unwrap_or_else(|| format!("QEMU {a}.{b}.{c}"));
        let uuid = QEMU_UUID.lock().unwrap_or_else(PoisonError::into_inner).unwrap_or_default();
        return ruvm_ui::dbus::init(DisplayState::global(), InputState::global(), opts, name, uuid);
    }
    Ok(())
}

/// A system reset request, `qemu_system_reset_request()`.
type ResetRequest = Arc<dyn Fn() + Send + Sync>;

/// The reset the `Machine` menu of the GTK and Cocoa windows asks for, which the machine sets.
static RESET: Mutex<Option<ResetRequest>> = Mutex::new(None);

/// Sets what the `Reset` item of the GTK and Cocoa windows does.
pub(crate) fn set_reset_request(f: ResetRequest) {
    *RESET.lock().unwrap_or_else(PoisonError::into_inner) = Some(f);
}

#[cfg_attr(
    not(any(feature = "ui-gtk", all(feature = "ui-cocoa", target_os = "macos"))),
    allow(dead_code)
)]
fn reset_request() -> Option<ResetRequest> {
    RESET.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

/// What the GTK window needs from the machine.
#[cfg(feature = "ui-gtk")]
struct GtkHooks(Weak<crate::runstate::Runstate>);

#[cfg(feature = "ui-gtk")]
impl ruvm_ui::gtk::Hooks for GtkHooks {
    fn is_running(&self) -> bool {
        self.0.upgrade().is_some_and(|r| r.is_running())
    }

    fn stop(&self) {
        if let Some(r) = self.0.upgrade() {
            let _ = r.qmp_stop();
        }
    }

    fn cont(&self) {
        if let Some(r) = self.0.upgrade() {
            let _ = r.qmp_cont();
        }
    }

    fn can_reset(&self) -> bool {
        reset_request().is_some()
    }

    fn reset(&self) {
        if let Some(f) = reset_request() {
            f();
        }
    }

    fn can_powerdown(&self) -> bool {
        false
    }

    fn powerdown(&self) {}

    fn quit(&self) {
        if let Some(r) = self.0.upgrade() {
            r.shutdown_request(ruvm_qapi::types::ShutdownCause::HostQmpQuit);
        }
    }
}

/// What the SDL window needs from the machine.
#[cfg(feature = "ui-sdl")]
struct SdlHooks(Weak<crate::runstate::Runstate>);

#[cfg(feature = "ui-sdl")]
impl ruvm_ui::sdl::Hooks for SdlHooks {
    fn is_running(&self) -> bool {
        self.0.upgrade().is_some_and(|r| r.is_running())
    }

    fn close(&self) {
        if let Some(r) = self.0.upgrade() {
            r.shutdown_request(ruvm_qapi::types::ShutdownCause::HostUi);
        }
    }
}

/// What the Cocoa window needs from the machine.
#[cfg(all(feature = "ui-cocoa", target_os = "macos"))]
struct CocoaHooks(Weak<crate::runstate::Runstate>);

#[cfg(all(feature = "ui-cocoa", target_os = "macos"))]
impl ruvm_ui::cocoa::Hooks for CocoaHooks {
    fn stop(&self) {
        if let Some(r) = self.0.upgrade() {
            let _ = r.qmp_stop();
        }
    }

    fn cont(&self) {
        if let Some(r) = self.0.upgrade() {
            let _ = r.qmp_cont();
        }
    }

    fn can_reset(&self) -> bool {
        reset_request().is_some()
    }

    fn reset(&self) {
        if let Some(f) = reset_request() {
            f();
        }
    }

    fn can_powerdown(&self) -> bool {
        false
    }

    fn powerdown(&self) {}

    /// There is no `-no-shutdown` or `-action shutdown=pause`, so the shutdown action is
    /// `poweroff` already.
    fn quit(&self) {
        if let Some(r) = self.0.upgrade() {
            r.shutdown_request(ruvm_qapi::types::ShutdownCause::HostUi);
        }
    }
}

/// The display types `-device` knows, and the virtio input devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DisplayModel {
    /// "VGA", the std VGA PCI card.
    Vga,
    /// "bochs-display".
    BochsDisplay,
    /// "ramfb".
    Ramfb,
    /// "virtio-gpu-pci", which "virtio-gpu" is an alias of on x86 and Arm.
    VirtioGpuPci,
    /// "virtio-gpu-device", on a virtio-mmio transport.
    VirtioGpuDevice,
    /// "virtio-keyboard-pci" and the other virtio input functions. "virtio-keyboard",
    /// "virtio-mouse" and "virtio-tablet" are aliases on x86 and Arm.
    VirtioInputPci(VirtioInputKind),
    /// "virtio-keyboard-device" and the others on a virtio-mmio transport.
    VirtioInputDevice(VirtioInputKind),
}

/// The `-device` names of the display types, with the aliases of `qdev_alias_table[]`.
const DISPLAY_TYPES: &[(&str, DisplayModel)] = &[
    ("VGA", DisplayModel::Vga),
    ("bochs-display", DisplayModel::BochsDisplay),
    ("ramfb", DisplayModel::Ramfb),
    ("virtio-gpu-pci", DisplayModel::VirtioGpuPci),
    ("virtio-gpu", DisplayModel::VirtioGpuPci),
    ("virtio-gpu-device", DisplayModel::VirtioGpuDevice),
    ("virtio-keyboard-pci", DisplayModel::VirtioInputPci(VirtioInputKind::Keyboard)),
    ("virtio-keyboard", DisplayModel::VirtioInputPci(VirtioInputKind::Keyboard)),
    ("virtio-keyboard-device", DisplayModel::VirtioInputDevice(VirtioInputKind::Keyboard)),
    ("virtio-mouse-pci", DisplayModel::VirtioInputPci(VirtioInputKind::Mouse)),
    ("virtio-mouse", DisplayModel::VirtioInputPci(VirtioInputKind::Mouse)),
    ("virtio-mouse-device", DisplayModel::VirtioInputDevice(VirtioInputKind::Mouse)),
    ("virtio-tablet-pci", DisplayModel::VirtioInputPci(VirtioInputKind::Tablet)),
    ("virtio-tablet", DisplayModel::VirtioInputPci(VirtioInputKind::Tablet)),
    ("virtio-tablet-device", DisplayModel::VirtioInputDevice(VirtioInputKind::Tablet)),
    ("virtio-multitouch-pci", DisplayModel::VirtioInputPci(VirtioInputKind::MultiTouch)),
    ("virtio-multitouch-device", DisplayModel::VirtioInputDevice(VirtioInputKind::MultiTouch)),
];

/// The `vectors` virtio-gpu-pci starts with.
const VIRTIO_GPU_PCI_VECTORS: u32 = 3;
/// The `vectors` of `virtio_input_pci_properties`.
const VIRTIO_INPUT_PCI_VECTORS: u32 = 2;

/// A display or virtio input device to plug, with its properties.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DisplayPlug {
    pub model: DisplayModel,
    pub id: Option<String>,
    /// `addr`.
    devfn: Option<u8>,
    /// `romfile`, `None` for an empty one.
    romfile: Option<String>,
    /// VGA `vgamem_mb`.
    vgamem_mb: u32,
    /// bochs-display `vgamem`, in bytes.
    vgamem: u64,
    mmio: bool,
    qemu_extended_regs: bool,
    edid: bool,
    xres: u32,
    yres: u32,
    xmax: u32,
    ymax: u32,
    refresh_rate: u32,
    big_endian: bool,
    /// ramfb `use-legacy-x86-rom`, which the PC machines' compat properties turn on.
    legacy_rom: Option<bool>,
    /// virtio-gpu `max_outputs`, `max_hostmem` and `blob`.
    max_outputs: u32,
    max_hostmem: u64,
    blob: bool,
    /// virtio-gpu-pci and virtio input `vectors`.
    vectors: u32,
    /// virtio-gpu-device and virtio input `bus=virtio-mmio-bus.<n>`.
    mmio_bus: Option<usize>,
    /// virtio input `serial`, `display` and `head`.
    serial: Option<String>,
    input_display: Option<String>,
    head: u32,
    pub loc: Option<Location>,
}

impl DisplayModel {
    fn is_virtio_gpu(self) -> bool {
        matches!(self, DisplayModel::VirtioGpuPci | DisplayModel::VirtioGpuDevice)
    }

    fn is_pci(self) -> bool {
        !matches!(
            self,
            DisplayModel::Ramfb
                | DisplayModel::VirtioGpuDevice
                | DisplayModel::VirtioInputDevice(_)
        )
    }

    fn input_kind(self) -> Option<VirtioInputKind> {
        match self {
            DisplayModel::VirtioInputPci(k) | DisplayModel::VirtioInputDevice(k) => Some(k),
            _ => None,
        }
    }

    /// A virtio PCI function.
    fn is_virtio_pci(self) -> bool {
        matches!(self, DisplayModel::VirtioGpuPci | DisplayModel::VirtioInputPci(_))
    }

    /// A virtio device on a virtio-mmio transport.
    fn is_virtio_mmio(self) -> bool {
        matches!(self, DisplayModel::VirtioGpuDevice | DisplayModel::VirtioInputDevice(_))
    }
}

impl DisplayPlug {
    fn new(model: DisplayModel, loc: Option<Location>) -> DisplayPlug {
        let romfile = match model {
            DisplayModel::Vga => Some(VGA_ROMFILE.to_string()),
            DisplayModel::BochsDisplay => Some(BOCHS_DISPLAY_ROMFILE.to_string()),
            _ => None,
        };
        let gpu = VirtioGpuConf::default();
        let (xres, yres) = if model.is_virtio_gpu() { (gpu.xres, gpu.yres) } else { (0, 0) };
        DisplayPlug {
            model,
            id: None,
            devfn: None,
            romfile,
            vgamem_mb: 16,
            vgamem: 16 << 20,
            mmio: true,
            qemu_extended_regs: true,
            edid: true,
            xres,
            yres,
            xmax: 0,
            ymax: 0,
            refresh_rate: 0,
            big_endian: false,
            legacy_rom: None,
            max_outputs: gpu.max_outputs,
            max_hostmem: gpu.max_hostmem,
            blob: gpu.blob,
            vectors: if model.input_kind().is_some() {
                VIRTIO_INPUT_PCI_VECTORS
            } else {
                VIRTIO_GPU_PCI_VECTORS
            },
            mmio_bus: None,
            serial: None,
            input_display: None,
            head: 0,
            loc,
        }
    }

    pub(crate) fn typename(&self) -> &'static str {
        match self.model {
            DisplayModel::Vga => "VGA",
            DisplayModel::BochsDisplay => "bochs-display",
            DisplayModel::Ramfb => "ramfb",
            DisplayModel::VirtioGpuPci => TYPE_VIRTIO_GPU_PCI,
            DisplayModel::VirtioGpuDevice => TYPE_VIRTIO_GPU,
            DisplayModel::VirtioInputPci(k) => k.pci_typename(),
            DisplayModel::VirtioInputDevice(k) => k.typename(),
        }
    }

    fn at(&self, e: impl Into<String>) -> Located {
        Located::new(&self.loc, e)
    }

    /// The virtio-gpu device model, with its consoles under this device.
    fn virtio_gpu(&self) -> VirtioGpu {
        let conf = VirtioGpuConf {
            max_outputs: self.max_outputs,
            edid: self.edid,
            xres: self.xres,
            yres: self.yres,
            max_hostmem: self.max_hostmem,
            blob: self.blob,
        };
        // virtio_gpu_pci_base_realize() moves the consoles over to the PCI function.
        let dev = ConsoleDevice { id: self.id.clone(), typename: self.typename().to_string() };
        VirtioGpu::new(conf, DisplayState::global(), dev)
    }

    /// The virtio device model, virtio-gpu or one of the input devices.
    fn virtio_class(&self) -> Box<dyn VirtioDeviceClass> {
        let Some(kind) = self.model.input_kind() else {
            return Box::new(self.virtio_gpu());
        };
        let conf = VirtioInputConf {
            serial: self.serial.clone(),
            display: self.input_display.clone(),
            head: self.head,
        };
        Box::new(VirtioInput::new(kind, conf, InputState::global(), DisplayState::global()))
    }

    /// The virtio PCI properties: `virtio_pci_force_virtio_1()` turns legacy off and modern on.
    /// The input functions take their class from the type.
    fn virtio_pci_props(&self) -> VirtioPciProps {
        VirtioPciProps {
            disable_legacy: Some(true),
            disable_modern: false,
            vectors: Some(self.vectors),
            class_code: self.model.input_kind().map_or(0, VirtioInputKind::pci_class),
            id: self.id.clone(),
            ..VirtioPciProps::default()
        }
    }

    /// Lets the device reach the guest from outside a queue kick, through the PCI function.
    fn connect_pci(&self, dev: &VirtioPci) {
        match self.model.input_kind() {
            Some(_) => connect_input_pci(dev),
            None => connect_gpu_pci(dev),
        }
    }

    /// [`DisplayPlug::connect_pci`] for a virtio-mmio transport.
    fn connect_mmio(&self, t: &Arc<VirtioMmio>) {
        match self.model.input_kind() {
            Some(_) => connect_input_mmio(t),
            None => connect_gpu_mmio(t),
        }
    }

    fn edid_info(&self) -> EdidInfo {
        EdidInfo {
            prefx: self.xres,
            prefy: self.yres,
            maxx: self.xmax,
            maxy: self.ymax,
            refresh_rate: self.refresh_rate,
            ..EdidInfo::default()
        }
    }
}

/// A qdev property from its command line string, read the way the keyval input visitor reads
/// it, so the errors are QEMU's.
fn prop<T: Default>(
    name: &str,
    value: &str,
    visit: impl FnOnce(&mut QObjectInputVisitor, &str, &mut T) -> Result<()>,
) -> Result<T> {
    let mut d = QDict::new();
    d.put(name, QValue::Str(value.to_string()));
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(d));
    v.start_struct(None)?;
    let mut out = T::default();
    visit(&mut v, name, &mut out)?;
    v.end_struct();
    Ok(out)
}

fn prop_u32(name: &str, value: &str) -> Result<u32> {
    prop(name, value, |v, n, out| v.type_uint32(Some(n), out))
}

fn prop_bool(name: &str, value: &str) -> Result<bool> {
    prop(name, value, |v, n, out| v.type_bool(Some(n), out))
}

fn prop_size(name: &str, value: &str) -> Result<u64> {
    prop(name, value, |v, n, out| v.type_size(Some(n), out))
}

/// The `addr` property, `SLOT[.FN]` in hex.
pub(crate) fn parse_devfn(v: &str) -> Option<u8> {
    let (slot, func) = v.split_once('.').unwrap_or((v, "0"));
    let hex = |s: &str| {
        let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
        if s.is_empty() { None } else { u32::from_str_radix(s, 16).ok() }
    };
    let (slot, func) = (hex(slot)?, hex(func)?);
    (slot <= 31 && func <= 7).then_some((slot << 3 | func) as u8)
}

/// `qdev_device_add()` up to realize for a display type: the bus, then the properties.
/// `None` when `driver` is not a display type or a virtio input device. `pci` says whether the
/// machine has the root bus `pcie.0`, `sysbus` whether it takes a `ramfb`. Whether there is a
/// virtio-mmio transport for `virtio-gpu-device` or an input `-device` is found out when it is
/// realized.
pub(crate) fn plan_device(
    driver: &str,
    opts: &QemuOpts,
    loc: &Option<Location>,
    pci: bool,
    sysbus: bool,
) -> Option<std::result::Result<DisplayPlug, Located>> {
    // The abstract bases of the virtio input devices.
    if matches!(
        driver,
        "virtio-input-device"
            | "virtio-input-hid-device"
            | "virtio-input-pci"
            | "virtio-input-hid-pci"
    ) {
        let msg = "Parameter 'driver' expects a non-abstract device type".to_string();
        return Some(Err(Located::new(loc, msg)));
    }
    let &(_, model) = DISPLAY_TYPES.iter().find(|(t, _)| *t == driver)?;
    let typename = DisplayPlug::new(model, None).typename();
    Some(plan(typename, model, opts, loc, pci, sysbus))
}

fn plan(
    typename: &str,
    model: DisplayModel,
    opts: &QemuOpts,
    loc: &Option<Location>,
    pci: bool,
    sysbus: bool,
) -> std::result::Result<DisplayPlug, Located> {
    let is_pci = model.is_pci();
    let gpu = model.is_virtio_gpu();
    let mut mmio_bus = None;
    if let Some(b) = opts.get("bus") {
        // The bus is looked up by name first, then its type is checked.
        mmio_bus = b.strip_prefix("virtio-mmio-bus.").and_then(|n| n.parse::<usize>().ok());
        let bus_type = match b {
            "pcie.0" if pci => "PCIE",
            "main-system-bus" => "System",
            _ if mmio_bus.is_some() && sysbus => "virtio-mmio-bus",
            _ => return Err(Located::new(loc, format!("Bus '{b}' not found"))),
        };
        let fits = match model {
            _ if model.is_virtio_mmio() => bus_type == "virtio-mmio-bus",
            _ => (bus_type == "PCIE") == is_pci,
        };
        if !fits {
            let msg = format!("Device '{typename}' can't go on {bus_type} bus");
            return Err(Located::new(loc, msg));
        }
    }
    if is_pci && !pci {
        return Err(Located::new(loc, format!("No 'PCI' bus found for device '{typename}'")));
    }
    if model.is_virtio_mmio() && !sysbus {
        let msg = format!("No 'virtio-bus' bus found for device '{typename}'");
        return Err(Located::new(loc, msg));
    }
    let mut plug = DisplayPlug::new(model, loc.clone());
    plug.id = opts.id().map(str::to_string);
    plug.mmio_bus = mmio_bus;
    let at = |e: Error| Located(loc.clone(), e);
    let vga = model == DisplayModel::Vga;
    let bochs = model == DisplayModel::BochsDisplay;
    // The EDID and framebuffer properties VGA and bochs-display share.
    let vga_like = vga || bochs;
    let ramfb = model == DisplayModel::Ramfb;
    let input = model.input_kind().is_some();
    let virtio_pci = model.is_virtio_pci();
    for (k, v) in opts.iter() {
        match k {
            "driver" | "bus" => {}
            "id" => plug.id = Some(v.to_string()),
            "addr" if is_pci => {
                plug.devfn = Some(parse_devfn(v).ok_or_else(|| {
                    Located::new(
                        loc,
                        format!("Property '{typename}.addr' doesn't take value '{v}'"),
                    )
                })?);
            }
            "romfile" if is_pci => plug.romfile = (!v.is_empty()).then(|| v.to_string()),
            "vgamem_mb" if vga => plug.vgamem_mb = prop_u32(k, v).map_err(at)?,
            "mmio" if vga => plug.mmio = prop_bool(k, v).map_err(at)?,
            "qemu-extended-regs" if vga => plug.qemu_extended_regs = prop_bool(k, v).map_err(at)?,
            "vgamem" if bochs => plug.vgamem = prop_size(k, v).map_err(at)?,
            "edid" if vga_like || gpu => plug.edid = prop_bool(k, v).map_err(at)?,
            "xres" if vga_like || gpu => plug.xres = prop_u32(k, v).map_err(at)?,
            "yres" if vga_like || gpu => plug.yres = prop_u32(k, v).map_err(at)?,
            "xmax" if vga_like => plug.xmax = prop_u32(k, v).map_err(at)?,
            "ymax" if vga_like => plug.ymax = prop_u32(k, v).map_err(at)?,
            "refresh_rate" if vga_like => plug.refresh_rate = prop_u32(k, v).map_err(at)?,
            "big-endian-framebuffer" if vga_like => {
                plug.big_endian = prop_bool(k, v).map_err(at)?;
            }
            "max_outputs" if gpu => plug.max_outputs = prop_u32(k, v).map_err(at)?,
            "max_hostmem" if gpu => plug.max_hostmem = prop_size(k, v).map_err(at)?,
            "blob" if gpu => plug.blob = prop_bool(k, v).map_err(at)?,
            "serial" if input => plug.serial = Some(v.to_string()),
            "display" if input => plug.input_display = Some(v.to_string()),
            "head" if input => plug.head = prop_u32(k, v).map_err(at)?,
            "vectors" if virtio_pci => plug.vectors = prop_u32(k, v).map_err(at)?,
            // virtio_pci_force_virtio_1() overrides disable-modern. ioeventfd is a host detail,
            // and only virtio-gpu-pci has it.
            "disable-modern" if virtio_pci => {
                prop_bool(k, v).map_err(at)?;
            }
            "ioeventfd" if gpu && virtio_pci => {
                prop_bool(k, v).map_err(at)?;
            }
            // An OnOffAuto.
            "disable-legacy" if virtio_pci => {
                if !matches!(v, "on" | "off" | "auto") {
                    let msg = format!("Parameter '{k}' does not accept value '{v}'");
                    return Err(Located::new(loc, msg));
                }
            }
            "use-legacy-x86-rom" if ramfb => {
                plug.legacy_rom = Some(prop_bool(k, v).map_err(at)?);
            }
            // Only what migrates depends on it.
            "x-migrate" if ramfb => {
                prop_bool(k, v).map_err(at)?;
            }
            _ => return Err(Located::new(loc, format!("Property '{typename}.{k}' not found"))),
        }
    }
    if !is_pci && !sysbus {
        return Err(Located::new(
            loc,
            format!("Option '-device {typename}' cannot be handled by this machine"),
        ));
    }
    Ok(plug)
}

/// Guest memory for ramfb that does not keep the machine alive.
struct WeakMemory(Weak<AddressSpace>);

impl DmaMemory for WeakMemory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.0.upgrade().is_some_and(|a| DmaMemory::read(&*a, addr, buf))
    }

    fn write(&self, addr: u64, buf: &[u8]) -> bool {
        self.0.upgrade().is_some_and(|a| DmaMemory::write(&*a, addr, buf))
    }
}

/// `ramfb_realizefn()`, and with `legacy_rom` the "vgaroms/vgabios-ramfb.bin" file that
/// `rom_add_vga()` puts in fw_cfg for SeaBIOS to run.
fn realize_ramfb(
    plug: &DisplayPlug,
    fw_cfg: &FwCfgState,
    memory: &Arc<AddressSpace>,
    legacy_rom: Option<&FirmwareSearch>,
) -> std::result::Result<(), Located> {
    let mem = Arc::new(WeakMemory(Arc::downgrade(memory)));
    Ramfb::realize(plug.id.clone(), Some(fw_cfg), mem, &DisplayState::global())
        .map_err(|e| Located(plug.loc.clone(), e))?;
    if let Some(firmware) = legacy_rom {
        const ROM: &str = "vgabios-ramfb.bin";
        match firmware.load(ROM) {
            Some(data) => {
                fw_cfg
                    .add_file(&format!("vgaroms/{ROM}"), data)
                    .map_err(|e| plug.at(e.to_string()))?;
            }
            // rom_add_file() reports it and the device works without it.
            None => eprintln!(
                "rom: file {ROM:<20}: error Failed to open file \u{201c}{ROM}\u{201d}: No such file or directory"
            ),
        }
    }
    Ok(())
}

/// Realizes a VGA or bochs-display function on `bus`. Gives the function and the name its
/// option ROM takes, `<vmsd name>.rom`.
fn realize_pci(
    bus: &PciBus,
    plug: &DisplayPlug,
    x86: bool,
) -> std::result::Result<(Arc<PciDevice>, &'static str), Located> {
    let ds = DisplayState::global();
    let at = |e: Error| Located(plug.loc.clone(), e);
    let taken: Vec<bool> = ds.consoles().iter().map(|c| c.device().is_some()).collect();
    let (dev, rom_name) = match plug.model {
        DisplayModel::Vga => {
            let props = VgaPciProps {
                id: plug.id.clone(),
                vgamem_mb: plug.vgamem_mb,
                mmio: plug.mmio,
                qemu_extended_regs: plug.qemu_extended_regs,
                edid: plug.edid,
                edid_info: plug.edid_info(),
                x86,
                big_endian: plug.big_endian,
            };
            let vga = VgaPci::realize(bus, plug.devfn, &props, &ds).map_err(at)?;
            (Arc::clone(vga.pci_device()), "vga")
        }
        DisplayModel::BochsDisplay => {
            let props = BochsDisplayProps {
                id: plug.id.clone(),
                vgamem: plug.vgamem,
                edid: plug.edid,
                edid_info: plug.edid_info(),
                big_endian: plug.big_endian,
            };
            // pci_bus_is_express(): both q35's pcie.0 and the GPEX root bus are.
            let dev = BochsDisplay::realize(bus, plug.devfn, true, &props, &ds).map_err(at)?;
            (Arc::clone(dev.pci_device()), "bochs-display")
        }
        _ => unreachable!("{} is not realized here", plug.typename()),
    };
    // qemu_console_fill_device_address() for the consoles the function took. The display
    // functions sit on a root bus, so there is no bridge in front.
    let devfn = dev.devfn();
    for (i, con) in ds.consoles().iter().enumerate() {
        if con.device().is_some() && !taken.get(i).copied().unwrap_or(false) {
            con.set_device_address(format!("pci/0000/{:02x}.{:x}", devfn >> 3, devfn & 7));
        }
    }
    Ok((dev, rom_name))
}

/// Raises the config interrupt for a UI size change through the PCI function, which the
/// device's consoles must not keep alive.
fn connect_gpu_pci(dev: &VirtioPci) {
    let weak = dev.downgrade();
    dev.with_device::<VirtioGpu, _>(|_, gpu| {
        gpu.set_config_notifier(Some(Box::new(move || {
            if let Some(dev) = weak.upgrade() {
                dev.with_device::<VirtioGpu, _>(|vdev, gpu| gpu.config_notify(vdev));
            }
        })));
    });
}

/// [`connect_gpu_pci`] for a virtio-mmio transport.
fn connect_gpu_mmio(t: &Arc<VirtioMmio>) {
    let weak = Arc::downgrade(t);
    t.with_device::<VirtioGpu, _>(|_, gpu| {
        gpu.set_config_notifier(Some(Box::new(move || {
            if let Some(t) = weak.upgrade() {
                t.with_device::<VirtioGpu, _>(|vdev, gpu| gpu.config_notify(vdev));
            }
        })));
    });
}

/// Sends the input batches the device finished through the PCI function, which the input
/// handler must not keep alive.
fn connect_input_pci(dev: &VirtioPci) {
    let weak = dev.downgrade();
    dev.with_device::<VirtioInput, _>(|_, input| {
        input.set_kick(Some(Box::new(move || {
            if let Some(dev) = weak.upgrade() {
                dev.with_device::<VirtioInput, _>(|vdev, input| input.flush(vdev));
            }
        })));
    });
}

/// [`connect_input_pci`] for a virtio-mmio transport.
fn connect_input_mmio(t: &Arc<VirtioMmio>) {
    let weak = Arc::downgrade(t);
    t.with_device::<VirtioInput, _>(|_, input| {
        input.set_kick(Some(Box::new(move || {
            if let Some(t) = weak.upgrade() {
                t.with_device::<VirtioInput, _>(|vdev, input| input.flush(vdev));
            }
        })));
    });
}

/// Realizes virtio-gpu-pci or a virtio input function on q35, or virtio-gpu-device or an input
/// `-device` on microvm.
fn realize_x86_virtio(
    board: &mut X86Board,
    plug: &DisplayPlug,
    firmware: &FirmwareSearch,
) -> std::result::Result<(), Located> {
    let at = |e: Error| Located(plug.loc.clone(), e);
    let typename = plug.typename();
    let handle = match &mut *board {
        X86Board::Q35(m, devs) if plug.model.is_virtio_pci() => {
            let mem: SharedGuestMemory =
                Arc::new(AddressSpaceMemory::new(Arc::clone(m.memory_as())));
            let backend = VirtioBackend::new(plug.virtio_class(), mem).map_err(at)?;
            let props = plug.virtio_pci_props();
            let dev = VirtioPci::new(m.pci_bus(), plug.devfn, backend, &props).map_err(at)?;
            plug.connect_pci(&dev);
            devs.push(dev.clone());
            VirtioHandle::Pci(dev)
        }
        X86Board::Microvm(_) if plug.model.is_virtio_mmio() => {
            if let Some(n) = plug.mmio_bus {
                return Err(plug.at(format!("Bus 'virtio-mmio-bus.{n}' not found")));
            }
            let handle = board
                .attach_virtio(plug.virtio_class())
                .map_err(|e| plug.at(format!("{e} '{typename}'")))?;
            if let VirtioHandle::Mmio(t) = &handle {
                plug.connect_mmio(t);
            }
            handle
        }
        _ if plug.model.is_virtio_mmio() => {
            return Err(plug.at(format!("No 'virtio-bus' bus found for device '{typename}'")));
        }
        _ => return Err(plug.at(format!("No 'PCI' bus found for device '{typename}'"))),
    };
    // pci_add_option_rom()
    if let Some(name) = &plug.romfile {
        let Some(data) = firmware.load(name) else {
            return Err(plug.at(format!("failed to find romfile \"{name}\"")));
        };
        if data.is_empty() {
            return Err(plug.at(format!("romfile \"{name}\" is empty")));
        }
        board.add_option_rom(&handle, typename, &data).map_err(|e| plug.at(e))?;
    }
    Ok(())
}

/// `pc_vga_init()`: q35 plugs the `-vga std` card before the `-device` functions, so it takes
/// the first free slot, 00:01.0.
pub(crate) fn realize_x86_vga(
    board: &mut X86Board,
    vga: Option<VgaInterface>,
    firmware: &FirmwareSearch,
) -> std::result::Result<(), Located> {
    if vga != Some(VgaInterface::Std) || !matches!(board, X86Board::Q35(..)) {
        return Ok(());
    }
    VGA_INTERFACE_CREATED.store(true, Ordering::Release);
    realize_x86(board, &DisplayPlug::new(DisplayModel::Vga, None), firmware)
}

/// Connects an x86 board to the input layer: the i8042 handlers `ps2_kbd_realize()` and
/// `ps2_mouse_realize()` register, the virtual clock the key queue runs on, and the data
/// directories the keyboard layouts are looked for in.
pub(crate) fn connect_x86_input(board: &X86Board, clock: &Arc<Clock>, firmware: &FirmwareSearch) {
    let input = InputState::global();
    input.set_clock(clock);
    keymaps::set_data_dirs(firmware.dirs().to_vec());
    if let X86Board::Q35(m, _) = board {
        if let Some(i8042) = m.i8042() {
            i8042.register_input(&input);
        }
    }
}

/// Realizes a planned display device on an x86 board.
pub(crate) fn realize_x86(
    board: &mut X86Board,
    plug: &DisplayPlug,
    firmware: &FirmwareSearch,
) -> std::result::Result<(), Located> {
    if plug.model == DisplayModel::Ramfb {
        let fw_cfg = match board {
            X86Board::Q35(m, _) => Arc::clone(m.fw_cfg()),
            X86Board::Microvm(m) => Arc::clone(m.fw_cfg()),
        };
        // The PC machines' compat properties set use-legacy-x86-rom.
        let legacy = plug.legacy_rom.unwrap_or(true).then_some(firmware);
        return realize_ramfb(plug, &fw_cfg, board.memory_as(), legacy);
    }
    if plug.model.is_virtio_pci() || plug.model.is_virtio_mmio() {
        return realize_x86_virtio(board, plug, firmware);
    }
    let X86Board::Q35(m, _) = board else {
        return Err(plug.at(format!("No 'PCI' bus found for device '{}'", plug.typename())));
    };
    let (dev, rom_name) = realize_pci(m.pci_bus(), plug, true)?;
    // pci_add_option_rom()
    if let Some(name) = &plug.romfile {
        let Some(data) = firmware.load(name) else {
            return Err(plug.at(format!("failed to find romfile \"{name}\"")));
        };
        if data.is_empty() {
            return Err(plug.at(format!("romfile \"{name}\" is empty")));
        }
        let devfn = dev.devfn();
        let block = format!("0000:00:{:02x}.{:x}/{rom_name}.rom", devfn >> 3, devfn & 7);
        let rom = m.add_device_rom(&block, &data).map_err(|e| plug.at(e))?;
        dev.register_bar(PCI_ROM_SLOT, 0, rom);
    }
    Ok(())
}

/// Realizes a planned display device on Arm virt.
pub(crate) fn realize_virt(
    board: &VirtMachine,
    plug: &DisplayPlug,
) -> std::result::Result<(), Located> {
    if plug.model == DisplayModel::Ramfb {
        return realize_ramfb(plug, board.fw_cfg().state(), board.memory_as(), None);
    }
    if plug.model.input_kind().is_some() {
        // The clock send-key paces its keys on, which connect_x86_input() sets on x86.
        InputState::global().set_clock(board.clock());
    }
    let typename = plug.typename();
    if plug.model.is_virtio_pci() {
        let dev = board
            .attach_virtio_pci(plug.virtio_class(), plug.devfn, &plug.virtio_pci_props())
            .map_err(|e| plug.at(e))?;
        plug.connect_pci(&dev);
        return Ok(());
    }
    if plug.model.is_virtio_mmio() {
        let class = plug.virtio_class();
        let index = match plug.mmio_bus {
            Some(n) => board
                .attach_virtio_at(n, class, VIRTIO_MMIO_FORCE_LEGACY_DEFAULT)
                .map(|()| n)
                .map_err(|e| plug.at(e))?,
            None => board.attach_virtio(class).map_err(|e| plug.at(format!("{e} '{typename}'")))?,
        };
        if let Some(t) = board.virtio_transport(index) {
            plug.connect_mmio(&t);
        }
        return Ok(());
    }
    realize_pci(board.gpex().bus(), plug, false).map(drop)
}

/// The QMP `screendump` command.
pub(crate) fn register(cmds: &mut Commands) {
    register_screendump(cmds, |_: &MonitorQmp, arg| {
        let format = arg.format.map(|f| match f {
            ImageFormat::Ppm => screendump::ImageFormat::Ppm,
            ImageFormat::Png => screendump::ImageFormat::Png,
        });
        let ds = DisplayState::global();
        screendump::screendump(&ds, &arg.filename, arg.device.as_deref(), arg.head, format)
    });
    register_query_display_options(cmds, |_: &MonitorQmp| Ok(dpy().clone().unwrap_or_default()));
    register_query_mice(cmds, |_: &MonitorQmp| Ok(InputState::global().query_mice()));
    register_send_key(cmds, |_: &MonitorQmp, arg| InputState::global().qmp_send_key(arg));
    register_input_send_event(cmds, |_: &MonitorQmp, arg| {
        InputState::global().qmp_input_send_event(&DisplayState::global(), arg)
    });
    crate::vnc::register(cmds);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_qapi::opts::QemuOptsList;

    fn plan_one(arg: &str, pci: bool, sysbus: bool) -> std::result::Result<DisplayPlug, String> {
        let mut list = QemuOptsList::new("device", &[]).with_implied_opt_name("driver");
        let opts = list.parse(arg, true).unwrap();
        let driver = opts.get("driver").unwrap().to_string();
        plan_device(&driver, opts, &None, pci, sysbus)
            .expect("a display type")
            .map_err(|e| e.1.message().to_string())
    }

    #[test]
    fn vga_types() {
        assert_eq!(select_vgahw("std"), Ok(Some(VgaInterface::Std)));
        assert_eq!(select_vgahw("none"), Ok(Some(VgaInterface::None)));
        assert_eq!(select_vgahw("std,retrace=dumb,retrace=precise"), Ok(Some(VgaInterface::Std)));
        assert_eq!(select_vgahw("stdx"), Err("unknown vga type: stdx".to_string()));
        assert_eq!(select_vgahw("std,retrace=x"), Err("unknown vga type: std,retrace=x".into()));
        assert_eq!(select_vgahw("foo"), Err("unknown vga type: foo".to_string()));
        assert_eq!(select_vgahw("cirrus"), Err("Cirrus VGA not available".to_string()));
    }

    #[test]
    fn device_properties() {
        let p = plan_one("VGA,id=v,addr=3,vgamem_mb=32,edid=off,xres=800", true, true).unwrap();
        assert_eq!(p.model, DisplayModel::Vga);
        assert_eq!((p.id.as_deref(), p.devfn, p.vgamem_mb), (Some("v"), Some(0x18), 32));
        assert!(!p.edid);
        assert_eq!(p.xres, 800);
        assert_eq!(p.romfile.as_deref(), Some(VGA_ROMFILE));
        let p = plan_one("bochs-display,vgamem=64M,romfile=", true, true).unwrap();
        assert_eq!((p.vgamem, p.romfile), (64 << 20, None));
        let p = plan_one("ramfb,use-legacy-x86-rom=off", false, true).unwrap();
        assert_eq!(p.legacy_rom, Some(false));

        assert_eq!(plan_one("VGA,foo=1", true, true).unwrap_err(), "Property 'VGA.foo' not found");
        assert_eq!(
            plan_one("bochs-display,vgamem_mb=1", true, true).unwrap_err(),
            "Property 'bochs-display.vgamem_mb' not found"
        );
        assert_eq!(
            plan_one("VGA,addr=20", true, true).unwrap_err(),
            "Property 'VGA.addr' doesn't take value '20'"
        );
        assert_eq!(plan_one("VGA,bus=pci.0", true, true).unwrap_err(), "Bus 'pci.0' not found");
        assert_eq!(
            plan_one("ramfb,bus=pcie.0", true, true).unwrap_err(),
            "Device 'ramfb' can't go on PCIE bus"
        );
        assert_eq!(
            plan_one("VGA,bus=main-system-bus", true, true).unwrap_err(),
            "Device 'VGA' can't go on System bus"
        );
        assert_eq!(
            plan_one("VGA", false, true).unwrap_err(),
            "No 'PCI' bus found for device 'VGA'"
        );
        assert_eq!(
            plan_one("ramfb", false, false).unwrap_err(),
            "Option '-device ramfb' cannot be handled by this machine"
        );
        let mut list = QemuOptsList::new("device", &[]).with_implied_opt_name("driver");
        let opts = list.parse("virtio-rng-pci", true).unwrap();
        assert!(plan_device("virtio-rng-pci", opts, &None, true, true).is_none());
    }

    #[test]
    fn virtio_gpu_properties() {
        let p = plan_one("virtio-gpu", true, true).unwrap();
        assert_eq!((p.model, p.typename()), (DisplayModel::VirtioGpuPci, "virtio-gpu-pci"));
        assert_eq!((p.xres, p.yres, p.max_outputs, p.vectors), (1280, 800, 1, 3));
        assert_eq!((p.max_hostmem, p.blob, p.edid, p.romfile), (256 << 20, false, true, None));
        let p = plan_one(
            "virtio-gpu-pci,id=g,addr=4,max_outputs=2,xres=1024,yres=768,max_hostmem=64M,edid=off,\
             vectors=4,disable-legacy=off,disable-modern=on,ioeventfd=on,bus=pcie.0",
            true,
            true,
        )
        .unwrap();
        assert_eq!((p.id.as_deref(), p.devfn, p.max_outputs), (Some("g"), Some(0x20), 2));
        assert_eq!(
            (p.xres, p.yres, p.max_hostmem, p.edid, p.vectors),
            (1024, 768, 64 << 20, false, 4)
        );
        let props = p.virtio_pci_props();
        assert_eq!((props.disable_legacy, props.disable_modern), (Some(true), false));
        assert_eq!((props.vectors, props.id.as_deref()), (Some(4), Some("g")));
        let p = plan_one("virtio-gpu-device,bus=virtio-mmio-bus.3,blob=on", true, true).unwrap();
        assert_eq!((p.model, p.mmio_bus, p.blob), (DisplayModel::VirtioGpuDevice, Some(3), true));

        let err = |arg, pci, sysbus| plan_one(arg, pci, sysbus).unwrap_err();
        assert_eq!(
            err("virtio-gpu,xmax=3", true, true),
            "Property 'virtio-gpu-pci.xmax' not found"
        );
        assert_eq!(
            err("virtio-gpu-device,addr=3", true, true),
            "Property 'virtio-gpu-device.addr' not found"
        );
        assert_eq!(
            err("virtio-gpu-device,vectors=3", true, true),
            "Property 'virtio-gpu-device.vectors' not found"
        );
        assert_eq!(
            err("virtio-gpu-pci,disable-legacy=yes", true, true),
            "Parameter 'disable-legacy' does not accept value 'yes'"
        );
        assert_eq!(
            err("virtio-gpu-pci,max_hostmem=1x", true, true),
            "Parameter 'max_hostmem' expects size"
        );
        assert_eq!(
            err("virtio-gpu-device,bus=pcie.0", true, true),
            "Device 'virtio-gpu-device' can't go on PCIE bus"
        );
        assert_eq!(
            err("virtio-gpu-pci,bus=virtio-mmio-bus.1", true, true),
            "Device 'virtio-gpu-pci' can't go on virtio-mmio-bus bus"
        );
        assert_eq!(
            err("virtio-gpu-pci", false, true),
            "No 'PCI' bus found for device 'virtio-gpu-pci'"
        );
        assert_eq!(
            err("virtio-gpu-device,foo=1", false, false),
            "No 'virtio-bus' bus found for device 'virtio-gpu-device'"
        );
        assert_eq!(
            err("virtio-gpu-device,bus=virtio-mmio-bus.0", false, false),
            "Bus 'virtio-mmio-bus.0' not found"
        );
        assert_eq!(
            err("virtio-gpu-device,x-migrate=on", true, true),
            "Property 'virtio-gpu-device.x-migrate' not found"
        );
    }

    #[test]
    fn virtio_input_properties() {
        let p = plan_one("virtio-tablet,id=t,addr=4,serial=abc,display=vga0,head=1", true, true)
            .unwrap();
        assert_eq!(p.model, DisplayModel::VirtioInputPci(VirtioInputKind::Tablet));
        assert_eq!(p.typename(), "virtio-tablet-pci");
        assert_eq!((p.id.as_deref(), p.devfn, p.vectors), (Some("t"), Some(0x20), 2));
        assert_eq!(
            (p.serial.as_deref(), p.input_display.as_deref(), p.head),
            (Some("abc"), Some("vga0"), 1)
        );
        let props = p.virtio_pci_props();
        assert_eq!((props.disable_legacy, props.class_code), (Some(true), 0x0980));
        let p = plan_one("virtio-keyboard-pci,vectors=0", true, true).unwrap();
        assert_eq!((p.vectors, p.virtio_pci_props().class_code), (0, 0x0900));
        let p = plan_one("virtio-mouse", true, true).unwrap();
        assert_eq!(p.virtio_pci_props().class_code, 0x0902);
        let p = plan_one("virtio-multitouch-device,bus=virtio-mmio-bus.2", true, true).unwrap();
        assert_eq!(p.model, DisplayModel::VirtioInputDevice(VirtioInputKind::MultiTouch));
        assert_eq!((p.typename(), p.mmio_bus), ("virtio-multitouch-device", Some(2)));

        let err = |arg, pci, sysbus| plan_one(arg, pci, sysbus).unwrap_err();
        assert_eq!(
            err("virtio-keyboard-device,vectors=3", true, true),
            "Property 'virtio-keyboard-device.vectors' not found"
        );
        assert_eq!(
            err("virtio-mouse-pci,xres=3", true, true),
            "Property 'virtio-mouse-pci.xres' not found"
        );
        assert_eq!(
            err("virtio-input-hid-pci", true, true),
            "Parameter 'driver' expects a non-abstract device type"
        );
        assert_eq!(
            err("virtio-keyboard-pci,ioeventfd=off", true, true),
            "Property 'virtio-keyboard-pci.ioeventfd' not found"
        );
        assert_eq!(
            err("virtio-tablet-device,bus=pcie.0", true, true),
            "Device 'virtio-tablet-device' can't go on PCIE bus"
        );
        assert_eq!(
            err("virtio-keyboard-device", false, false),
            "No 'virtio-bus' bus found for device 'virtio-keyboard-device'"
        );
        assert_eq!(
            err("virtio-keyboard", false, true),
            "No 'PCI' bus found for device 'virtio-keyboard-pci'"
        );
        // There is no alias for the multitouch device.
        assert!(!DISPLAY_TYPES.iter().any(|(t, _)| *t == "virtio-multitouch"));
    }
}
