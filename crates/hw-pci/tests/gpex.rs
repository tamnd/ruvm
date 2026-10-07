// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests of the generic PCIe host: the root function through ECAM, the windows that read as
//! all ones where nothing is mapped, BARs in the MMIO window and the INTx swizzle, with the
//! host mapped the way hw/riscv/virt.c maps it.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

use ruvm_hw_core::IrqLine;
use ruvm_hw_pci::regs::*;
use ruvm_hw_pci::*;
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemorySystem};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const ECAM: u64 = 0x3000_0000;
const PIO: u64 = 0x0300_0000;
const MMIO: u64 = 0x4000_0000;

struct Virt {
    mem: Arc<MemorySystem>,
    mem_as: Arc<AddressSpace>,
    gpex: GpexHost,
    levels: Arc<[AtomicI32; 4]>,
}

impl Virt {
    fn new() -> Virt {
        let mem = Arc::new(MemorySystem::new());
        let sysmem = mem.new_container("system", 1 << 64).unwrap();
        let gpex = GpexHost::new(Arc::clone(&mem), sysmem, GpexConfig::default()).unwrap();
        let ecam = mem.new_alias("pcie-ecam", gpex.ecam(), 0, 0x1000_0000).unwrap();
        mem.add_subregion(sysmem, ECAM, ecam).unwrap();
        let mmio = mem.new_alias("pcie-mmio", gpex.mmio_window(), MMIO, 0x4000_0000).unwrap();
        mem.add_subregion(sysmem, MMIO, mmio).unwrap();
        mem.add_subregion(sysmem, PIO, gpex.ioport_window()).unwrap();
        let levels: Arc<[AtomicI32; 4]> = Arc::new(Default::default());
        for i in 0..4 {
            let l = Arc::clone(&levels);
            gpex.irq(i)
                .unwrap()
                .connect(IrqLine::from_fn(move |v| l[i].store(v, Ordering::SeqCst)));
            gpex.set_irq_num(i, 0x20 + i as i32).unwrap();
        }
        let mem_as = mem.address_space_init(sysmem, "memory").unwrap();
        gpex.reset();
        Virt { mem, mem_as, gpex, levels }
    }

    fn read(&self, addr: u64) -> u32 {
        self.mem_as.load(addr, 4, Endian::Little, ATTRS).0 as u32
    }

    fn write(&self, addr: u64, v: u32) {
        assert!(self.mem_as.store(addr, 4, v.into(), Endian::Little, ATTRS).is_ok());
    }

    fn ecam(bus: u8, devfn: u8, off: u32) -> u64 {
        ECAM | (u64::from(bus) << 20) | (u64::from(devfn) << 12) | u64::from(off)
    }

    fn levels(&self) -> [i32; 4] {
        std::array::from_fn(|i| self.levels[i].load(Ordering::SeqCst))
    }
}

#[test]
fn root_function_through_ecam() {
    let v = Virt::new();
    assert_eq!(v.read(Virt::ecam(0, 0, 0)), 0x0008_1b36);
    assert_eq!(v.read(Virt::ecam(0, 0, PCI_CLASS_REVISION as u32)), 0x0600_0000);
    // A conventional PCI function, so its config space ends at 0x100 and extended config
    // reads as all ones, as pci_host_config_read_common() does past the limit.
    assert_eq!(v.read(Virt::ecam(0, 0, 0x100)), 0xffff_ffff);
    assert_eq!(v.read(Virt::ecam(0, pci_devfn(1, 0), 0)), 0xffff_ffff);
    assert_eq!(v.read(Virt::ecam(1, 0, 0)), 0xffff_ffff);
    assert_eq!(v.gpex.bus().root_bus_path(), "0000:00");
    assert_eq!(v.gpex.route_intx_pin_to_irq(2), Some(0x22));
    assert_eq!(v.gpex.route_intx_pin_to_irq(4), None);
}

#[test]
fn unmapped_windows_read_as_ones() {
    let v = Virt::new();
    assert_eq!(v.read(MMIO), 0xffff_ffff);
    assert_eq!(v.read(MMIO + 0x3fff_fffc), 0xffff_ffff);
    v.write(MMIO, 0);
    assert_eq!(v.read(PIO + 0x1000), 0xffff_ffff);
    v.write(PIO + 0x1000, 0);
}

#[test]
fn bar_and_intx_swizzle() {
    let v = Virt::new();
    let info = PciDeviceInfo {
        name: "test".to_string(),
        vendor_id: 0x1af4,
        device_id: 0x1005,
        class_id: 0x00ff,
        ..Default::default()
    };
    let dev = v.gpex.bus().register_device(&info, Some(pci_devfn(2, 0))).unwrap();
    let bar = v.mem.new_ram("bar", 0x1000).unwrap();
    dev.register_bar(0, PCI_BASE_ADDRESS_SPACE_MEMORY, bar);
    assert_eq!(v.read(Virt::ecam(0, pci_devfn(2, 0), 0)), 0x1005_1af4);
    let cfg = |off: usize| Virt::ecam(0, pci_devfn(2, 0), off as u32);
    v.write(cfg(PCI_BASE_ADDRESS_0), 0x4000_1000);
    v.write(cfg(PCI_COMMAND), u32::from(PCI_COMMAND_MEMORY));
    v.write(MMIO + 0x1000, 0x1234_5678);
    assert_eq!(v.read(MMIO + 0x1000), 0x1234_5678);
    assert_eq!(v.read(MMIO + 0x2000), 0xffff_ffff);

    // Slot 2, INTB swizzles to line (2 + 1) % 4.
    dev.irq_handler(1, 1);
    assert_eq!(v.levels(), [0, 0, 0, 1]);
    dev.irq_handler(1, 0);
    // Slot 2, INTC wraps around to line 0.
    dev.irq_handler(2, 1);
    assert_eq!(v.levels(), [1, 0, 0, 0]);
}
