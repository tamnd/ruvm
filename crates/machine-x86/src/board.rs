// SPDX-License-Identifier: GPL-2.0-or-later

//! One type for the x86 boards an accelerator can run: [`Microvm`] and [`Q35`].
//!
//! The accelerator loop in [`crate::kvm_run`] and the command line code in the system crate do
//! not care which board they drive, as long as they can reach its address spaces, its
//! interrupt controllers and its reset. [`X86Board`] gives them that in one place.

use std::collections::BTreeMap;
use std::fmt;
use std::io::Read;
use std::sync::{Arc, OnceLock};

use ruvm_firmware::smbios::{SmbiosOptions, SmbiosTopology};
use ruvm_firmware::x86_linux::{LINUXBOOT_DMA_ROM, PVH_ROM};
use ruvm_hw_acpi::SystemRequestHandler;
use ruvm_hw_char::serial::{Serial, SerialBackend};
use ruvm_hw_core::{Clock, IrqPin};
use ruvm_hw_intc::i8259::I8259Pair;
use ruvm_hw_intc::ioapic::{IoApic, IoApicMsiHandler, IoApics};
use ruvm_hw_iommu::virtio_iommu::{
    ReservedRegion, TYPE_VIRTIO_IOMMU_MEMORY_REGION, VIRTIO_IOMMU_RESV_MEM_T_MSI,
};
use ruvm_hw_iommu::{Viommu, VirtioIommu, VirtioIommuConf};
use ruvm_hw_virtio::virtio::VIRTIO_F_IOMMU_PLATFORM;
use ruvm_hw_virtio::{
    AddressSpaceMemory, SharedGuestMemory, VirtIODevice, VirtioBackend, VirtioDeviceClass,
    VirtioMmio, VirtioPci, VirtioPciProps,
};
use ruvm_mem::{AddressSpace, MemorySystem, RamBlock, RegionId};

use crate::firmware::FirmwareSearch;
use crate::microvm::{
    KernelConfig, MICROVM_DESC, Microvm, MicrovmConfig, MicrovmProps, OnOffAuto,
    default_firmware_name,
};
use crate::pc::{GsiHook, err};
use crate::q35::{
    CpuIdent, KVMVAPIC_ROM, PflashDrive, Q35, Q35_BIOS_FILENAME, Q35_DESC, Q35_MACHINE_ALIAS,
    Q35_MACHINE_NAME, Q35_OLDER_MACHINE_NAMES, Q35MachineConfig, Q35Props,
};

/// The x86 boards the system emulator can build, as `-machine help` lists them: name, alias
/// and description.
pub const X86_BOARDS: &[(&str, Option<&str>, &str)] = &[
    ("microvm", None, MICROVM_DESC),
    (Q35_MACHINE_NAME, Some(Q35_MACHINE_ALIAS), Q35_DESC),
    (Q35_OLDER_MACHINE_NAMES[0], None, Q35_DESC),
    (Q35_OLDER_MACHINE_NAMES[1], None, Q35_DESC),
];

/// The machine type a `-machine type=` value names, with an alias resolved (`q35` is
/// `pc-q35-11.1`), if it is one of [`X86_BOARDS`].
pub fn canonical_machine_name(name: &str) -> Option<&'static str> {
    X86_BOARDS.iter().find(|(n, a, _)| *n == name || *a == Some(name)).map(|(n, _, _)| *n)
}

/// A plugged virtio device, reached through its transport.
#[derive(Clone, Debug)]
pub enum VirtioHandle {
    /// A virtio PCI function on q35.
    Pci(VirtioPci),
    /// A virtio-mmio transport of microvm.
    Mmio(Arc<VirtioMmio>),
}

impl VirtioHandle {
    /// Runs `f` on the device model as its concrete type `D`, together with the core state.
    /// `None` if it is not a `D` or the device is busy on this thread.
    pub fn with_device<D: VirtioDeviceClass, R>(
        &self,
        f: impl FnOnce(&mut VirtIODevice, &mut D) -> R,
    ) -> Option<R> {
        match self {
            VirtioHandle::Pci(p) => p.with_device(f),
            VirtioHandle::Mmio(m) => m.with_device(f),
        }
    }
}

/// A microvm or q35 board.
pub enum X86Board {
    /// `-M microvm`.
    Microvm(Box<Microvm>),
    /// `-M q35`, `-M pc-q35-11.1`.
    Q35(Box<Q35>, Vec<VirtioPci>),
}

impl fmt::Debug for X86Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            X86Board::Microvm(m) => m.fmt(f),
            X86Board::Q35(m, _) => m.fmt(f),
        }
    }
}

impl From<Microvm> for X86Board {
    fn from(m: Microvm) -> Self {
        X86Board::Microvm(Box::new(m))
    }
}

impl From<Q35> for X86Board {
    fn from(m: Q35) -> Self {
        X86Board::Q35(Box::new(m), Vec::new())
    }
}

impl X86Board {
    /// The machine type name.
    pub fn name(&self) -> &'static str {
        match self {
            X86Board::Microvm(_) => "microvm",
            X86Board::Q35(m, _) => m.machine_name(),
        }
    }

    /// `default_kernel_irqchip_split` of the machine class: microvm asks for the split
    /// irqchip, the PC boards for the full in-kernel one.
    pub fn default_kernel_irqchip_split(&self) -> bool {
        matches!(self, X86Board::Microvm(_))
    }

    /// Whether the board wants a PIT at all, the `pit` machine property not being off.
    pub fn pit_wanted(&self) -> bool {
        match self {
            X86Board::Microvm(m) => m.props().pit != OnOffAuto::Off,
            X86Board::Q35(m, _) => m.props().pit != OnOffAuto::Off,
        }
    }

    /// Whether the kernel irqchip on mode sets up the PC GSI routing table,
    /// `kvm_pc_setup_irq_routing()`. The PC boards do; microvm keeps KVM's default.
    pub fn sets_up_irq_routing(&self) -> bool {
        matches!(self, X86Board::Q35(..))
    }

    /// The guest RAM size in bytes.
    pub fn ram_size(&self) -> u64 {
        match self {
            X86Board::Microvm(m) => m.ram_size(),
            X86Board::Q35(m, _) => m.ram_size(),
        }
    }

    /// Plugs a virtio device: into a virtio-mmio transport on microvm, into a new virtio PCI
    /// function on q35. Gives the device's handle.
    pub fn attach_virtio(
        &mut self,
        class: Box<dyn VirtioDeviceClass>,
    ) -> Result<VirtioHandle, String> {
        self.attach_virtio_with(class, &VirtioPciProps::default(), false)
    }

    /// [`attach_virtio`](Self::attach_virtio) with the virtio PCI properties and
    /// `iommu_platform`, which puts the device's DMA behind the vIOMMU. microvm has neither.
    pub fn attach_virtio_with(
        &mut self,
        class: Box<dyn VirtioDeviceClass>,
        props: &VirtioPciProps,
        iommu_platform: bool,
    ) -> Result<VirtioHandle, String> {
        match self {
            X86Board::Microvm(m) => {
                let i = m.attach_virtio(class)?;
                let t = m.virtio_transport(i).ok_or("virtio-mmio transport missing")?;
                Ok(VirtioHandle::Mmio(t))
            }
            X86Board::Q35(m, devs) => {
                // virtio_bus_device_plugged(): with iommu_platform the device's DMA goes to
                // its PCI address space, otherwise to system memory.
                let bdf = Arc::new(OnceLock::new());
                let mem: SharedGuestMemory = if iommu_platform {
                    Arc::new(m.pci_dma().bus_master(Arc::clone(&bdf)))
                } else {
                    Arc::new(AddressSpaceMemory::new(Arc::clone(m.memory_as())))
                };
                let mut backend = VirtioBackend::new(class, mem).map_err(err)?;
                if iommu_platform {
                    backend.vdev_mut().set_host_feature(VIRTIO_F_IOMMU_PLATFORM, true);
                }
                let dev = VirtioPci::new(m.pci_bus(), None, backend, props).map_err(err)?;
                let _ = bdf.set(dev.pci_dev().requester_id());
                devs.push(dev.clone());
                Ok(VirtioHandle::Pci(dev))
            }
        }
    }

    /// Plugs a `virtio-iommu-pci` with the properties `conf`: `pc_machine_device_pre_plug_cb()`
    /// and the realize of the PCI proxy and of the device. The function translates the DMA of
    /// every function on the root bus and gets a VIOT.
    pub fn attach_virtio_iommu(
        &mut self,
        conf: &VirtioIommuConf,
        id: Option<String>,
    ) -> Result<VirtioHandle, String> {
        let X86Board::Q35(m, devs) = self else {
            return Err("No 'PCI' bus found for device 'virtio-iommu-pci'".to_string());
        };
        // Declare the APIC range as the reserved MSI region.
        let conf = VirtioIommuConf {
            reserved_regions: vec![ReservedRegion {
                low: 0xfee0_0000,
                high: 0xfeef_ffff,
                kind: VIRTIO_IOMMU_RESV_MEM_T_MSI,
            }],
            ..conf.clone()
        };
        if m.pci_dma().has_iommu() {
            return Err("QEMU does not support multiple vIOMMUs for x86 yet.".to_string());
        }
        conf.check_pci().map_err(err)?;
        let viommu = Viommu::new(m.memory_as(), &conf);
        let class = Box::new(VirtioIommu::new(Arc::clone(&viommu), conf));
        let mem: SharedGuestMemory = Arc::new(AddressSpaceMemory::new(Arc::clone(m.memory_as())));
        let backend = VirtioBackend::new(class, mem).map_err(err)?;
        // virtio_pci_force_virtio_1()
        let props = VirtioPciProps {
            disable_legacy: Some(true),
            disable_modern: false,
            id,
            ..VirtioPciProps::default()
        };
        let dev = VirtioPci::new(m.pci_bus(), None, backend, &props).map_err(err)?;

        let bus = Arc::downgrade(m.pci_bus());
        viommu.set_endpoint_check(move |sid| {
            bus.upgrade().is_some_and(|b| b.find_device((sid >> 8) as u8, sid as u8).is_some())
        });
        let weak = dev.downgrade();
        viommu.set_fault_sink(move || {
            if let Some(d) = weak.upgrade() {
                // Busy on this thread: the device sends the fault when it is done.
                let _ = d.with_device::<VirtioIommu, _>(|vdev, dev| dev.report_faults(vdev));
            }
        });
        let ops = Arc::clone(&viommu);
        let factory = Box::new(move |bdf: u16| ops.device_ops(u32::from(bdf)));
        m.pci_dma().set_iommu(TYPE_VIRTIO_IOMMU_MEMORY_REGION, factory);
        let reset = Arc::clone(&viommu);
        m.add_reset_hook(Box::new(move || reset.system_reset()));
        m.set_virtio_iommu(dev.pci_dev().requester_id());
        devs.push(dev.clone());
        Ok(VirtioHandle::Pci(dev))
    }

    /// `pci_add_option_rom()`: gives the PCI function behind `dev` a ROM BAR holding `data`,
    /// for a `romfile`. The ROM is `<typename>.rom` in the migration stream, as QEMU names
    /// it, and is the next power of two up from the file. A virtio-mmio device has no ROM.
    pub fn add_option_rom(
        &mut self,
        dev: &VirtioHandle,
        typename: &str,
        data: &[u8],
    ) -> Result<(), String> {
        let (X86Board::Q35(m, _), VirtioHandle::Pci(dev)) = (self, dev) else {
            return Err(format!("Property '{typename}.romfile' not found"));
        };
        if data.is_empty() {
            return Err(format!("romfile for '{typename}' is empty"));
        }
        let pdev = dev.pci_dev();
        let devfn = pdev.devfn();
        let name = format!("0000:00:{:02x}.{:x}/{typename}.rom", devfn >> 3, devfn & 7);
        let rom = m.add_device_rom(&name, data)?;
        pdev.register_bar(ruvm_hw_pci::regs::PCI_ROM_SLOT, 0, rom);
        Ok(())
    }

    /// Plugs a `-drive if=ide,index=N` disk. Only q35 has an IDE (AHCI) controller.
    pub fn attach_ide_drive(
        &self,
        index: usize,
        blk: Arc<dyn ruvm_hw_storage::BlockBackend>,
    ) -> Result<(), String> {
        match self {
            X86Board::Microvm(_) => {
                Err(format!("machine type does not support if=ide,bus={index},unit=0"))
            }
            X86Board::Q35(m, _) => {
                m.attach_drive(index, ruvm_hw_storage::DriveConfig::hd(), Some(blk))
            }
        }
    }

    /// The machine-done notifiers and the first reset.
    pub fn machine_done(&mut self) -> Result<(), String> {
        match self {
            X86Board::Microvm(m) => m.machine_done(),
            X86Board::Q35(m, _) => m.machine_done(),
        }
    }

    /// The device part of a system reset.
    pub fn system_reset(&mut self) -> Result<(), String> {
        match self {
            X86Board::Microvm(m) => m.system_reset(),
            X86Board::Q35(m, _) => m.system_reset(),
        }
    }

    /// The memory system all regions live in.
    pub fn memory_system(&self) -> &Arc<MemorySystem> {
        match self {
            X86Board::Microvm(m) => m.memory_system(),
            X86Board::Q35(m, _) => m.memory_system(),
        }
    }

    /// `get_system_memory()`, the root region of `address_space_memory`.
    pub fn system_memory(&self) -> RegionId {
        match self {
            X86Board::Microvm(m) => m.system_memory(),
            X86Board::Q35(m, _) => m.system_memory(),
        }
    }

    /// `address_space_memory`.
    pub fn memory_as(&self) -> &Arc<AddressSpace> {
        match self {
            X86Board::Microvm(m) => m.memory_as(),
            X86Board::Q35(m, _) => m.memory_as(),
        }
    }

    /// `address_space_io`.
    pub fn io_as(&self) -> &Arc<AddressSpace> {
        match self {
            X86Board::Microvm(m) => m.io_as(),
            X86Board::Q35(m, _) => m.io_as(),
        }
    }

    /// The APIC IDs of the possible CPUs that are present at startup.
    pub fn apic_ids(&self) -> Vec<u32> {
        match self {
            X86Board::Microvm(m) => m.apic_ids(),
            X86Board::Q35(m, _) => m.apic_ids(),
        }
    }

    /// The 8259 pair, if the board has one.
    pub fn pic(&self) -> Option<&I8259Pair> {
        match self {
            X86Board::Microvm(m) => m.pic(),
            X86Board::Q35(m, _) => m.pic(),
        }
    }

    /// The INTR output of the master 8259, what goes to the BSP's LINT0.
    pub fn pic_output(&self) -> &Arc<IrqPin> {
        match self {
            X86Board::Microvm(m) => m.pic_output(),
            X86Board::Q35(m, _) => m.pic_output(),
        }
    }

    /// Every IOAPIC of the board.
    pub fn ioapics(&self) -> &IoApics {
        match self {
            X86Board::Microvm(m) => m.ioapics(),
            X86Board::Q35(m, _) => m.ioapics(),
        }
    }

    /// The IOAPICs in GSI order: the first one and microvm's second one if it has it.
    pub fn ioapic_list(&self) -> Vec<Arc<IoApic>> {
        match self {
            X86Board::Microvm(m) => {
                let mut v = vec![Arc::clone(m.ioapic())];
                if let Some(s) = m.ioapic2() {
                    v.push(Arc::clone(s));
                }
                v
            }
            X86Board::Q35(m, _) => vec![Arc::clone(m.ioapic())],
        }
    }

    /// Replaces the MSI delivery of the IOAPICs and, on q35, of the PCI devices and the HPET.
    pub fn set_msi_handler(&self, handler: Option<IoApicMsiHandler>) {
        match self {
            X86Board::Microvm(m) => m.set_msi_handler(handler),
            X86Board::Q35(m, _) => m.set_msi_handler(handler),
        }
    }

    /// Puts a hook in front of the GSI handler.
    pub fn set_gsi_hook(&self, hook: Option<GsiHook>) {
        match self {
            X86Board::Microvm(m) => m.set_gsi_hook(hook),
            X86Board::Q35(m, _) => m.set_gsi_hook(hook),
        }
    }

    /// Where guest reset and power off requests go: the ICH9 PM block, the i8042 and port
    /// 0x92 on q35, the generic event device on microvm. A microvm without ACPI has nothing
    /// to report and ignores the handler.
    pub fn set_request_handler(&self, handler: SystemRequestHandler) {
        match self {
            X86Board::Microvm(m) => {
                if let Some(g) = m.ged() {
                    g.set_request_handler(handler);
                }
            }
            X86Board::Q35(m, _) => m.set_request_handler(Some(handler)),
        }
    }

    /// The ISA serial port `index`, 0 being COM1, if there is one. microvm has at most COM1,
    /// q35 up to COM4.
    pub fn serial(&self, index: usize) -> Option<&Arc<Serial>> {
        match self {
            X86Board::Microvm(m) => m.serial().filter(|_| index == 0),
            X86Board::Q35(m, _) => m.serial(index),
        }
    }

    /// Connects the chardev of the ISA serial port `index`. Returns false if there is no such
    /// port.
    pub fn set_serial_backend(
        &self,
        index: usize,
        backend: Option<Arc<dyn SerialBackend>>,
    ) -> bool {
        match self {
            X86Board::Microvm(m) => index == 0 && m.set_serial_backend(backend),
            X86Board::Q35(m, _) => m.set_serial_backend(index, backend),
        }
    }

    /// The warnings QEMU would have printed while the board was built.
    pub fn warnings(&self) -> &[String] {
        match self {
            X86Board::Microvm(m) => m.warnings(),
            X86Board::Q35(m, _) => m.warnings(),
        }
    }
}

/// Which board to build.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BoardKind {
    /// `microvm`.
    Microvm,
    /// `pc-q35-11.1`, alias `q35`, and the older versions in [`Q35_OLDER_MACHINE_NAMES`].
    Q35,
}

impl BoardKind {
    /// The board a `-machine type=` value names, if it is one of these.
    pub fn from_name(name: &str) -> Option<BoardKind> {
        match canonical_machine_name(name)? {
            "microvm" => Some(BoardKind::Microvm),
            _ => Some(BoardKind::Q35),
        }
    }

    /// See [`X86Board::default_kernel_irqchip_split`].
    pub fn default_kernel_irqchip_split(self) -> bool {
        self == BoardKind::Microvm
    }
}

/// A `-kernel` boot, as file names.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelFiles {
    /// `-kernel`.
    pub kernel: String,
    /// `-initrd`.
    pub initrd: Option<String>,
    /// `-append`.
    pub append: String,
}

/// What [`build_board`] needs: the command line after parsing, with nothing read yet.
pub struct BoardSpec {
    pub kind: BoardKind,
    /// The machine type name, from [`canonical_machine_name`]: `pc-q35-11.0` builds a q35 board
    /// that calls itself that.
    pub machine_type: &'static str,
    /// The `-machine` properties other than `type`, `accel` and the accelerator ones, in
    /// command line order.
    pub props: Vec<(String, String)>,
    /// `-m`, or `None` for the board's default.
    pub ram_size: Option<u64>,
    /// The RAM of `-machine memory-backend=`, which the board takes instead of making its
    /// own. Its size is the RAM size.
    pub memdev: Option<Arc<RamBlock>>,
    /// `-machine aux-ram-share=`: the RAM and ROM the board makes are shared memory that CPR
    /// can hand to the next process.
    pub aux_ram_share: bool,
    /// `-smp cpus=`.
    pub cpus: u32,
    /// `-smp maxcpus=`, 0 for the same as `cpus`.
    pub max_cpus: u32,
    /// Whether the accelerator is KVM.
    pub kvm: bool,
    /// `kvm_pit_in_kernel()`.
    pub pit_in_kernel: bool,
    /// Whether the accelerator can run SMM.
    pub smm_available: bool,
    /// `phys-bits` of the CPU model.
    pub phys_bits: u32,
    /// The vendor and CPUID signature of the CPU model (q35 only).
    pub cpu: CpuIdent,
    /// `-bios`.
    pub bios: Option<String>,
    /// The drives of pflash0 and pflash1 (q35 only).
    pub pflash: [Option<PflashDrive>; 2],
    /// `-uuid`, or the UUID of `-smbios type=1,uuid=`.
    pub uuid: Option<[u8; 16]>,
    /// The `-smbios` options (q35 only).
    pub smbios: SmbiosOptions,
    /// The `-smp` topology, for the SMBIOS processor tables. `None` for one socket.
    pub topology: Option<SmbiosTopology>,
    /// `-kernel`, `-initrd` and `-append`.
    pub kernel: Option<KernelFiles>,
    /// Where firmware and option ROMs are looked up.
    pub firmware: FirmwareSearch,
    /// Whether `serial_hd(i)` exists, for each `-serial` in order (`false` for
    /// `-serial none`). microvm looks at the first only, q35 at the first four.
    pub serial_hds: Vec<bool>,
    /// `QEMU_CLOCK_VIRTUAL`.
    pub clock: Arc<Clock>,
    /// `rtc_clock`.
    pub rtc_clock: Arc<Clock>,
}

impl fmt::Debug for BoardSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoardSpec")
            .field("kind", &self.kind)
            .field("machine_type", &self.machine_type)
            .field("props", &self.props)
            .field("ram_size", &self.ram_size)
            .field("memdev", &self.memdev.as_ref().map(|b| b.name()))
            .field("aux_ram_share", &self.aux_ram_share)
            .field("cpus", &self.cpus)
            .field("bios", &self.bios)
            .field("kernel", &self.kernel)
            .finish_non_exhaustive()
    }
}

/// The C library's text for an I/O error, without Rust's " (os error N)" suffix.
fn strerror(e: &std::io::Error) -> String {
    let s = e.to_string();
    match s.find(" (os error") {
        Some(i) => s[..i].to_string(),
        None => s,
    }
}

/// Reads the `-kernel` and `-initrd` files. The errors are those of `x86_load_linux()`, which
/// opens the kernel with `fopen()` and maps the initrd with `g_mapped_file_new()`.
pub fn load_kernel(files: &KernelFiles) -> Result<KernelConfig, String> {
    let k = &files.kernel;
    let mut file = std::fs::File::open(k)
        .map_err(|e| format!("qemu: could not open kernel file '{k}': {}", strerror(&e)))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| format!("qemu: could not load kernel '{k}': {}", strerror(&e)))?;
    let initrd = match &files.initrd {
        Some(f) => {
            Some(read_initrd(f).map_err(|e| format!("qemu: error reading initrd {f}: {e}"))?)
        }
        None => None,
    };
    Ok(KernelConfig {
        filename: files.kernel.clone(),
        data,
        cmdline: files.append.clone(),
        initrd,
        ..KernelConfig::default()
    })
}

/// Reads the `-initrd` file, failing with the messages of `g_mapped_file_new()`.
fn read_initrd(f: &str) -> Result<Vec<u8>, String> {
    let mut file = std::fs::File::open(f).map_err(|e| {
        format!("Failed to open file \u{201c}{f}\u{201d}: open() failed: {}", strerror(&e))
    })?;
    let mut data = Vec::new();
    match file.read_to_end(&mut data) {
        Ok(_) => Ok(data),
        // GLib maps the file, and mmap() of a directory fails with ENODEV.
        Err(e) if e.kind() == std::io::ErrorKind::IsADirectory => {
            Err(format!("Failed to map {f}' {f}': mmap() failed: No such device"))
        }
        Err(e) => Err(format!("Failed to map {f}' {f}': mmap() failed: {}", strerror(&e))),
    }
}

/// Builds the board `spec` describes, with its firmware, option ROMs and kernel loaded. The
/// devices from `-device` and `-drive` are plugged afterwards, before `machine_done()`.
/// Also gives the warnings setting the properties produced; the board's own are in
/// [`X86Board::warnings`].
pub fn build_board(mut spec: BoardSpec) -> Result<(X86Board, Vec<String>), String> {
    // qemu_resolve_machine_memdev() takes the backend's size when there is no -m, and
    // machine_run_board_init() wants the two to agree otherwise.
    if let Some(block) = &spec.memdev {
        match spec.ram_size {
            None => spec.ram_size = Some(block.len()),
            Some(size) if size != block.len() => {
                return Err(
                    "Machine memory size does not match the size of the memory backend".into()
                );
            }
            Some(_) => {}
        }
    }
    let kernel = spec.kernel.as_ref().map(load_kernel).transpose()?;
    let mut rom_files = BTreeMap::new();
    // The APIC's kvmvapic device asks for its option ROM on q35 (microvm creates its CPUs
    // after the option ROMs are loaded, so it never gets one).
    if spec.kind == BoardKind::Q35 {
        if let Some(data) = spec.firmware.load(KVMVAPIC_ROM) {
            rom_files.insert(KVMVAPIC_ROM.to_string(), data);
        }
    }
    if kernel.is_some() {
        for name in [LINUXBOOT_DMA_ROM, PVH_ROM] {
            if let Some(data) = spec.firmware.load(name) {
                rom_files.insert(name.to_string(), data);
            }
        }
    }
    match spec.kind {
        BoardKind::Microvm => {
            let mut props = MicrovmProps::default();
            for (k, v) in &spec.props {
                props.set(k, v)?;
            }
            let name =
                spec.bios.clone().unwrap_or_else(|| default_firmware_name(&props).to_string());
            let mut cfg = MicrovmConfig {
                cpus: spec.cpus,
                max_cpus: spec.max_cpus,
                kvm: spec.kvm,
                firmware: spec.firmware.load(&name),
                firmware_name: spec.bios.clone(),
                props,
                kernel,
                rom_files,
                // serial_hds_isa_init(isa_bus, 0, 1)
                serial_hd: spec.serial_hds.first().copied().unwrap_or(false),
                clock: spec.clock,
                rtc_clock: spec.rtc_clock,
                pit_in_kernel: spec.pit_in_kernel,
                memdev: spec.memdev,
                aux_ram_share: spec.aux_ram_share,
                ..MicrovmConfig::default()
            };
            if let Some(size) = spec.ram_size {
                cfg.ram_size = size;
            }
            if let Some(uuid) = spec.uuid {
                cfg.fw_cfg.uuid = uuid;
            }
            Ok((Microvm::new(cfg)?.into(), Vec::new()))
        }
        BoardKind::Q35 => {
            let mut props = Q35Props::default();
            let mut warnings = Vec::new();
            for (k, v) in &spec.props {
                props.set(k, v, &mut warnings)?;
            }
            let name = spec.bios.clone().unwrap_or_else(|| Q35_BIOS_FILENAME.to_string());
            let mut cfg = Q35MachineConfig {
                machine_name: spec.machine_type,
                cpus: spec.cpus,
                max_cpus: spec.max_cpus,
                kvm: spec.kvm,
                smm_available: spec.smm_available,
                phys_bits: spec.phys_bits,
                cpu: spec.cpu,
                // With a pflash0 drive the BIOS is never read.
                firmware: spec.pflash[0].is_none().then(|| spec.firmware.load(&name)).flatten(),
                firmware_name: spec.bios.clone(),
                pflash: spec.pflash,
                smbios: spec.smbios,
                topology: spec.topology,
                props,
                kernel,
                rom_files,
                serial_hds: spec.serial_hds,
                clock: spec.clock,
                rtc_clock: spec.rtc_clock,
                pit_in_kernel: spec.pit_in_kernel,
                memdev: spec.memdev,
                aux_ram_share: spec.aux_ram_share,
                ..Q35MachineConfig::default()
            };
            if let Some(size) = spec.ram_size {
                cfg.ram_size = size;
            }
            if let Some(uuid) = spec.uuid {
                cfg.fw_cfg.uuid = uuid;
            }
            Ok((Q35::new(cfg)?.into(), warnings))
        }
    }
}
