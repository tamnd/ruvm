// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests of the Q35 host bridge, driven through 0xcf8/0xcfc and ECAM the way firmware and
//! QEMU's tests/qtest/q35-test.c drive it.

use std::sync::Arc;

use ruvm_hw_pci::q35::*;
use ruvm_hw_pci::regs::*;
use ruvm_hw_pci::*;
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemorySystem, RegionId};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const MIB: u64 = 1 << 20;
/// `TSEG_SIZE_TEST_GUEST_RAM_MBYTES`.
const RAM_MBYTES: u64 = 128;

/// What the "BIOS" in the PCI address space holds at every byte.
const BIOS_BYTE: u8 = 0xb1;
/// What the "VGA" window in the PCI address space holds at every byte.
const VGA_BYTE: u8 = 0x5a;

struct Machine {
    mem: Arc<MemorySystem>,
    ram: RegionId,
    mem_as: Arc<AddressSpace>,
    io_as: Arc<AddressSpace>,
    smm_as: Option<Arc<AddressSpace>>,
    q35: Q35PciHost,
}

impl Machine {
    fn new() -> Machine {
        Machine::with(|_| {})
    }

    /// A q35 machine with 128 MiB of RAM, a 128 KiB BIOS in the PCI space at 0xe0000 and a VGA
    /// window at 0xa0000, like `-M q35 -m 128M`.
    fn with(tweak: impl FnOnce(&mut Q35Config)) -> Machine {
        let mem = Arc::new(MemorySystem::new());
        let sysmem = mem.new_container("system", 1 << 64).unwrap();
        let io = mem.new_container("io", 1 << 16).unwrap();
        let pci = mem.new_container("pci", 1 << 64).unwrap();
        let ram = mem.new_ram("pc.ram", RAM_MBYTES * MIB).unwrap();
        let ram_below_4g =
            mem.new_alias("ram-below-4g", ram, 0, u128::from(RAM_MBYTES * MIB)).unwrap();
        mem.add_subregion(sysmem, 0, ram_below_4g).unwrap();

        let bios = mem.new_rom("isa-bios", 0x20000).unwrap();
        mem.ram_block(bios).unwrap().fill(0, 0x20000, BIOS_BYTE).unwrap();
        mem.add_subregion_overlap(pci, 0xe0000, bios, 1).unwrap();
        let vga = mem.new_ram("vga-lowmem", 0x20000).unwrap();
        mem.ram_block(vga).unwrap().fill(0, 0x20000, VGA_BYTE).unwrap();
        mem.add_subregion_overlap(pci, 0xa0000, vga, 1).unwrap();

        let mem_as = mem.address_space_init(sysmem, "memory").unwrap();
        let io_as = mem.address_space_init(io, "I/O").unwrap();

        let mut cfg = Q35Config::new(ram, pci, sysmem, io);
        cfg.below_4g_mem_size = RAM_MBYTES * MIB;
        tweak(&mut cfg);
        let q35 = Q35PciHost::new(Arc::clone(&mem), cfg).unwrap();

        // The SMM address space of the CPUs: SMRAM over normal memory.
        let smm_as = q35.smram().map(|smram| {
            let smm = mem.new_container("memory-smm", 1 << 64).unwrap();
            let sys = mem.new_alias("smm-memory", sysmem, 0, 1 << 64).unwrap();
            mem.add_subregion_overlap(smm, 0, sys, 0).unwrap();
            mem.add_subregion_overlap(smm, 0, smram, 10).unwrap();
            mem.address_space_init(smm, "smm").unwrap()
        });

        q35.reset();
        Machine { mem, ram, mem_as, io_as, smm_as, q35 }
    }

    fn cfg_addr(devfn: u8, off: u32) -> u32 {
        0x8000_0000 | (u32::from(devfn) << 8) | (off & 0xfc)
    }

    fn cfg_read(&self, devfn: u8, off: u32, len: u32) -> u32 {
        let st =
            self.io_as.store(0xcf8, 4, Self::cfg_addr(devfn, off).into(), Endian::Little, ATTRS);
        assert!(st.is_ok());
        let (v, r) = self.io_as.load(0xcfc + u64::from(off & 3), len, Endian::Little, ATTRS);
        assert!(r.is_ok());
        v as u32
    }

    fn cfg_write(&self, devfn: u8, off: u32, len: u32, v: u32) {
        let st =
            self.io_as.store(0xcf8, 4, Self::cfg_addr(devfn, off).into(), Endian::Little, ATTRS);
        assert!(st.is_ok());
        let r = self.io_as.store(0xcfc + u64::from(off & 3), len, v.into(), Endian::Little, ATTRS);
        assert!(r.is_ok());
    }

    /// `qpci_config_readb()` on the MCH.
    fn readb(&self, off: usize) -> u8 {
        self.cfg_read(0, off as u32, 1) as u8
    }

    /// `qpci_config_writeb()` on the MCH.
    fn writeb(&self, off: usize, v: u8) {
        self.cfg_write(0, off as u32, 1, v.into());
    }

    fn memb(&self, addr: u64) -> u8 {
        self.mem_as.load(addr, 1, Endian::Little, ATTRS).0 as u8
    }

    fn set_memb(&self, addr: u64, v: u8) {
        let _ = self.mem_as.store(addr, 1, v.into(), Endian::Little, ATTRS);
    }

    fn meml(&self, addr: u64) -> u32 {
        self.mem_as.load(addr, 4, Endian::Little, ATTRS).0 as u32
    }

    fn smm_memb(&self, addr: u64) -> u8 {
        self.smm_as.as_ref().unwrap().load(addr, 1, Endian::Little, ATTRS).0 as u8
    }

    fn ram_fill(&self, off: u64, len: u64, b: u8) {
        self.mem.ram_block(self.ram).unwrap().fill(off, len, b).unwrap();
    }

    fn smram_set_bit(&self, mask: u8, enabled: bool) {
        let mut smram = self.readb(MCH_HOST_BRIDGE_SMRAM);
        if enabled {
            smram |= mask;
        } else {
            smram &= !mask;
        }
        self.writeb(MCH_HOST_BRIDGE_SMRAM, smram);
    }

    fn smram_test_bit(&self, mask: u8) -> bool {
        self.readb(MCH_HOST_BRIDGE_SMRAM) & mask != 0
    }
}

#[test]
fn mch_ids_and_defaults() {
    let m = Machine::new();
    assert_eq!(m.cfg_read(0, 0, 4), 0x29c0_8086);
    assert_eq!(m.cfg_read(0, PCI_CLASS_DEVICE as u32, 2), 0x0600);
    assert_eq!(m.cfg_read(0, PCI_REVISION_ID as u32, 1), 0);
    assert_eq!(m.cfg_read(0, MCH_HOST_BRIDGE_PCIEXBAR as u32, 4), 0xb000_0000);
    assert_eq!(m.cfg_read(0, MCH_HOST_BRIDGE_PCIEXBAR as u32 + 4, 4), 0);
    assert_eq!(m.readb(MCH_HOST_BRIDGE_SMRAM), MCH_HOST_BRIDGE_SMRAM_DEFAULT);
    assert_eq!(m.readb(MCH_HOST_BRIDGE_ESMRAMC), MCH_HOST_BRIDGE_ESMRAMC_DEFAULT);
    // mch_reset() writes the query value and mch_update() answers it right away.
    assert_eq!(m.cfg_read(0, MCH_HOST_BRIDGE_EXT_TSEG_MBYTES as u32, 2), 64);
    assert_eq!(m.readb(MCH_HOST_BRIDGE_F_SMBASE), 0);
    // Nothing else on the bus.
    assert_eq!(m.cfg_read(8, 0, 4), 0xffff_ffff);

    // The MMCONFIG window is off at reset.
    assert_eq!(m.q35.mcfg_base(), PCIE_BASE_ADDR_UNMAPPED);
    assert_eq!(m.q35.pci_hole_start(), (RAM_MBYTES * MIB) as u32);
    assert_eq!(m.q35.pci_hole_end(), 0xfec0_0000);
    assert!(m.q35.has_smm_ranges());
    assert_eq!(m.q35.below_4g_mem_size(), RAM_MBYTES * MIB);
    assert_eq!(m.q35.pci_hole64_size(), 1 << 35);
}

#[test]
fn pci_hole64_defaults_without_bars() {
    let m = Machine::with(|c| c.pc_pci_hole64_start = 0x1_0000_0000);
    assert_eq!(m.q35.pci_hole64_start(), 0x1_0000_0000);
    // Start plus 32 GiB, rounded up to 1 GiB.
    assert_eq!(m.q35.pci_hole64_end(), 0x1_0000_0000 + (1 << 35));

    let m = Machine::with(|c| {
        c.pc_pci_hole64_start = 0x1_0000_0000;
        c.pci_hole64_fix = false;
    });
    assert_eq!(m.q35.pci_hole64_start(), 0);
    assert_eq!(m.q35.pci_hole64_end(), 0);
}

#[test]
fn pam_controls_bios_area() {
    let m = Machine::new();
    m.ram_fill(0xc0000, 0x40000, 0x22);

    // At reset every segment decodes to PCI: the BIOS shows and writes go nowhere.
    assert_eq!(m.memb(0xf0000), BIOS_BYTE);
    assert_eq!(m.memb(0xe0000), BIOS_BYTE);
    m.set_memb(0xf0000, 0x33);
    assert_eq!(m.memb(0xf0000), BIOS_BYTE);
    for i in 0..PAM_REGIONS_COUNT {
        assert_eq!(m.q35.mch().pam_region(i).current(), 0);
    }

    // PAM0 high nibble = 3: DRAM read/write for 0xf0000..0xfffff.
    m.writeb(MCH_HOST_BRIDGE_PAM0, 0x30);
    assert_eq!(m.memb(0xf0000), 0x22);
    assert_eq!(m.memb(0xfffff), 0x22);
    m.set_memb(0xf0000, 0x33);
    assert_eq!(m.memb(0xf0000), 0x33);
    // Other segments untouched.
    assert_eq!(m.memb(0xe0000), BIOS_BYTE);

    // Read only: reads come from DRAM and the range is read only, so CPU stores are dropped.
    // Like QEMU's address_space_write(), a debug style store through the address space still
    // reaches the RAM behind the read-only alias, so only the flat view flag is checked here.
    m.writeb(MCH_HOST_BRIDGE_PAM0, 0x10);
    assert_eq!(m.memb(0xf0000), 0x33);
    let view = m.mem_as.flatview();
    let fr = view.lookup(0xf0000).unwrap();
    assert!(fr.readonly());

    // Write only: QEMU maps DRAM for both directions.
    m.writeb(MCH_HOST_BRIDGE_PAM0, 0x20);
    assert!(!m.mem_as.flatview().lookup(0xf0000).unwrap().readonly());
    m.set_memb(0xf0000, 0x55);
    assert_eq!(m.memb(0xf0000), 0x55);

    // Back to PCI.
    m.writeb(MCH_HOST_BRIDGE_PAM0, 0x00);
    assert_eq!(m.memb(0xf0000), BIOS_BYTE);

    // PAM5: low nibble is 0xe0000..0xe3fff, high nibble 0xe4000..0xe7fff.
    m.writeb(MCH_HOST_BRIDGE_PAM5, 0x03);
    assert_eq!(m.memb(0xe0000), 0x22);
    assert_eq!(m.memb(0xe3fff), 0x22);
    assert_eq!(m.memb(0xe4000), BIOS_BYTE);
    m.writeb(MCH_HOST_BRIDGE_PAM5, 0x30);
    assert_eq!(m.memb(0xe0000), BIOS_BYTE);
    assert_eq!(m.memb(0xe4000), 0x22);

    // PAM1 low nibble is 0xc0000, where nothing lives on PCI.
    m.writeb(MCH_HOST_BRIDGE_PAM1, 0x33);
    assert_eq!(m.memb(0xc0000), 0x22);
    assert_eq!(m.memb(0xc4000), 0x22);
    assert_eq!(m.q35.mch().pam_region(1).current(), 3);
    assert_eq!(m.q35.mch().pam_region(2).current(), 3);

    // A dword write covering PAM0..PAM3 updates them all.
    m.cfg_write(0, MCH_HOST_BRIDGE_PAM0 as u32, 4, 0x0000_0030);
    assert_eq!(m.memb(0xf0000), 0x55);
    assert_eq!(m.q35.mch().pam_region(1).current(), 0);
    assert_eq!(m.q35.mch().pam_region(2).current(), 0);
    // The PCI space is a container with nothing at 0xc0000, so as in QEMU the RAM mapped
    // below it shows through.
    assert_eq!(m.memb(0xc0000), 0x22);

    // Neither mch_reset() nor pci_do_device_reset() touches the PAM registers, so the mapping
    // survives a reset.
    m.q35.reset();
    assert_eq!(m.readb(MCH_HOST_BRIDGE_PAM0), 0x30);
    assert_eq!(m.memb(0xf0000), 0x55);
}

#[test]
fn pciexbar_and_ecam() {
    let m = Machine::new();
    let base = 0xb000_0000u64;
    // Before enabling, the window is not there.
    assert_eq!(m.meml(base), 0);

    // 256 MiB at 0xb0000000.
    m.cfg_write(0, MCH_HOST_BRIDGE_PCIEXBAR as u32 + 4, 4, 0);
    m.cfg_write(0, MCH_HOST_BRIDGE_PCIEXBAR as u32, 4, (base | MCH_HOST_BRIDGE_PCIEXBAREN) as u32);
    assert_eq!(m.q35.mcfg_base(), base);
    assert_eq!(m.q35.mcfg_size(), 256 * MIB);
    assert_eq!(m.meml(base), 0x29c0_8086);
    assert_eq!(m.meml(base + PCI_CLASS_REVISION as u64), 0x0600_0000);
    // Only one byte, word at an offset.
    assert_eq!(m.mem_as.load(base + 2, 2, Endian::Little, ATTRS).0, 0x29c0);
    // Nothing at 00:01.0 or on bus 1.
    assert_eq!(m.meml(base + (1 << 15)), 0xffff_ffff);
    assert_eq!(m.meml(base + (1 << 20)), 0xffff_ffff);
    // The MCH is conventional PCI, so its config space ends at 256 bytes.
    assert_eq!(m.meml(base + 0x100), 0xffff_ffff);

    // ECAM writes reach the MCH like 0xcfc writes do.
    m.ram_fill(0xf0000, 0x10000, 0x77);
    assert!(
        m.mem_as.store(base + MCH_HOST_BRIDGE_PAM0 as u64, 1, 0x30, Endian::Little, ATTRS).is_ok()
    );
    assert_eq!(m.readb(MCH_HOST_BRIDGE_PAM0), 0x30);
    assert_eq!(m.memb(0xf0000), 0x77);

    // 128 MiB: bit 27 of the address counts now, and the window is smaller.
    let base128 = 0xc800_0000u64;
    m.cfg_write(
        0,
        MCH_HOST_BRIDGE_PCIEXBAR as u32,
        4,
        (base128 | MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_128M | MCH_HOST_BRIDGE_PCIEXBAREN) as u32,
    );
    assert_eq!(m.q35.mcfg_base(), base128);
    assert_eq!(m.q35.mcfg_size(), 128 * MIB);
    assert_eq!(m.meml(base), 0);
    assert_eq!(m.meml(base128), 0x29c0_8086);

    // 64 MiB with bit 26.
    let base64 = 0xe400_0000u64;
    m.cfg_write(
        0,
        MCH_HOST_BRIDGE_PCIEXBAR as u32,
        4,
        (base64 | MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_64M | MCH_HOST_BRIDGE_PCIEXBAREN) as u32,
    );
    assert_eq!(m.q35.mcfg_base(), base64);
    assert_eq!(m.q35.mcfg_size(), 64 * MIB);
    assert_eq!(m.meml(base64), 0x29c0_8086);

    // In 256 MiB mode bits 27 and 26 are masked off.
    m.cfg_write(
        0,
        MCH_HOST_BRIDGE_PCIEXBAR as u32,
        4,
        (base64 | MCH_HOST_BRIDGE_PCIEXBAREN) as u32,
    );
    assert_eq!(m.q35.mcfg_base(), 0xe000_0000);
    assert_eq!(m.meml(0xe000_0000), 0x29c0_8086);

    // A reserved length leaves the window where it was.
    m.cfg_write(
        0,
        MCH_HOST_BRIDGE_PCIEXBAR as u32,
        4,
        (base | MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_RVD | MCH_HOST_BRIDGE_PCIEXBAREN) as u32,
    );
    assert_eq!(m.q35.mcfg_base(), 0xe000_0000);

    // Disable.
    m.cfg_write(0, MCH_HOST_BRIDGE_PCIEXBAR as u32, 4, base as u32);
    assert_eq!(m.q35.mcfg_base(), PCIE_BASE_ADDR_UNMAPPED);
    assert_eq!(m.meml(0xe000_0000), 0);

    // Reset puts the default back, disabled.
    m.cfg_write(0, MCH_HOST_BRIDGE_PCIEXBAR as u32, 4, (base | MCH_HOST_BRIDGE_PCIEXBAREN) as u32);
    m.q35.reset();
    assert_eq!(m.q35.mcfg_base(), PCIE_BASE_ADDR_UNMAPPED);
    assert_eq!(m.cfg_read(0, MCH_HOST_BRIDGE_PCIEXBAR as u32, 4), 0xb000_0000);
}

#[test]
fn ecam_decoding() {
    assert_eq!(pcie_mmcfg_bus(0x0ab0_0000), 0xab);
    assert_eq!(pcie_mmcfg_devfn(0x000f_f000), 0xff);
    assert_eq!(pcie_mmcfg_confoffset(0x0000_0fff), 0xfff);
    assert_eq!(pcie_mmcfg_devfn(0x0000_8000), pci_devfn(1, 0));
}

/// `test_smram_lock()`.
#[test]
fn smram_lock() {
    let m = Machine::new();

    // Open is settable.
    m.smram_set_bit(MCH_HOST_BRIDGE_SMRAM_D_OPEN, false);
    assert!(!m.smram_test_bit(MCH_HOST_BRIDGE_SMRAM_D_OPEN));
    m.smram_set_bit(MCH_HOST_BRIDGE_SMRAM_D_OPEN, true);
    assert!(m.smram_test_bit(MCH_HOST_BRIDGE_SMRAM_D_OPEN));

    // Lock: open is cleared and cannot be set.
    m.smram_set_bit(MCH_HOST_BRIDGE_SMRAM_D_LCK, true);
    assert!(!m.smram_test_bit(MCH_HOST_BRIDGE_SMRAM_D_OPEN));
    m.smram_set_bit(MCH_HOST_BRIDGE_SMRAM_D_OPEN, true);
    assert!(!m.smram_test_bit(MCH_HOST_BRIDGE_SMRAM_D_OPEN));
    // ESMRAMC is frozen too.
    let esmramc = m.readb(MCH_HOST_BRIDGE_ESMRAMC);
    m.writeb(MCH_HOST_BRIDGE_ESMRAMC, esmramc ^ MCH_HOST_BRIDGE_ESMRAMC_T_EN);
    assert_eq!(m.readb(MCH_HOST_BRIDGE_ESMRAMC), esmramc);

    m.q35.reset();

    // Settable again.
    m.smram_set_bit(MCH_HOST_BRIDGE_SMRAM_D_OPEN, false);
    assert!(!m.smram_test_bit(MCH_HOST_BRIDGE_SMRAM_D_OPEN));
    m.smram_set_bit(MCH_HOST_BRIDGE_SMRAM_D_OPEN, true);
    assert!(m.smram_test_bit(MCH_HOST_BRIDGE_SMRAM_D_OPEN));
}

#[test]
fn smram_open_close_regions() {
    let m = Machine::new();
    const SMRAM_BYTE: u8 = 0x11;
    m.ram_fill(0xa0000, 0x20000, SMRAM_BYTE);

    // Closed at reset: normal accesses see VGA, SMM ones too since G_SMRAME is off.
    assert_eq!(m.memb(0xa0000), VGA_BYTE);
    assert_eq!(m.smm_memb(0xa0000), VGA_BYTE);
    assert_eq!(m.memb(0xfeda0000), 0);

    // G_SMRAME: SMM sees low SMRAM, normal accesses still see VGA.
    m.writeb(MCH_HOST_BRIDGE_SMRAM, MCH_HOST_BRIDGE_SMRAM_G_SMRAME | 2);
    assert_eq!(m.memb(0xa0000), VGA_BYTE);
    assert_eq!(m.smm_memb(0xa0000), SMRAM_BYTE);
    assert_eq!(m.smm_memb(0xbffff), SMRAM_BYTE);

    // D_OPEN: normal accesses see SMRAM too.
    m.writeb(MCH_HOST_BRIDGE_SMRAM, MCH_HOST_BRIDGE_SMRAM_G_SMRAME | MCH_HOST_BRIDGE_SMRAM_D_OPEN);
    assert_eq!(m.memb(0xa0000), SMRAM_BYTE);
    // C_BASE_SEG is read only.
    assert_eq!(m.readb(MCH_HOST_BRIDGE_SMRAM) & MCH_HOST_BRIDGE_SMRAM_C_BASE_SEG_MASK, 2);

    // H_SMRAME moves SMRAM up to 0xfeda0000 and gives VGA back.
    m.writeb(MCH_HOST_BRIDGE_ESMRAMC, MCH_HOST_BRIDGE_ESMRAMC_H_SMRAME);
    assert_eq!(m.memb(0xa0000), VGA_BYTE);
    assert_eq!(m.memb(0xfeda0000), SMRAM_BYTE);
    assert_eq!(m.smm_memb(0xa0000), VGA_BYTE);
    assert_eq!(m.smm_memb(0xfeda0000), SMRAM_BYTE);

    // Close: high SMRAM only for SMM.
    m.writeb(MCH_HOST_BRIDGE_SMRAM, MCH_HOST_BRIDGE_SMRAM_G_SMRAME);
    assert_eq!(m.memb(0xfeda0000), 0);
    assert_eq!(m.smm_memb(0xfeda0000), SMRAM_BYTE);

    // Lock while open: open is dropped, so SMRAM hides from normal accesses.
    m.writeb(MCH_HOST_BRIDGE_ESMRAMC, 0);
    m.writeb(MCH_HOST_BRIDGE_SMRAM, MCH_HOST_BRIDGE_SMRAM_G_SMRAME | MCH_HOST_BRIDGE_SMRAM_D_OPEN);
    assert_eq!(m.memb(0xa0000), SMRAM_BYTE);
    m.writeb(
        MCH_HOST_BRIDGE_SMRAM,
        MCH_HOST_BRIDGE_SMRAM_G_SMRAME | MCH_HOST_BRIDGE_SMRAM_D_OPEN | MCH_HOST_BRIDGE_SMRAM_D_LCK,
    );
    assert_eq!(m.memb(0xa0000), VGA_BYTE);
    assert_eq!(m.smm_memb(0xa0000), SMRAM_BYTE);
    // Only D_CLS stays writable.
    m.writeb(MCH_HOST_BRIDGE_SMRAM, MCH_HOST_BRIDGE_SMRAM_D_CLS);
    assert_eq!(
        m.readb(MCH_HOST_BRIDGE_SMRAM),
        MCH_HOST_BRIDGE_SMRAM_G_SMRAME
            | MCH_HOST_BRIDGE_SMRAM_D_LCK
            | MCH_HOST_BRIDGE_SMRAM_D_CLS
            | MCH_HOST_BRIDGE_SMRAM_C_BASE_SEG
    );
    assert_eq!(m.smm_memb(0xa0000), SMRAM_BYTE);

    m.q35.reset();
    assert_eq!(m.memb(0xa0000), VGA_BYTE);
    assert_eq!(m.smm_memb(0xa0000), VGA_BYTE);
}

/// `test_tseg_size()`.
fn tseg_size(esmramc_tseg_sz: u8, extended_tseg_mbytes: u16, expected_tseg_mbytes: u64) {
    let m = if esmramc_tseg_sz == MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_MASK {
        Machine::with(|c| c.ext_tseg_mbytes = extended_tseg_mbytes)
    } else {
        Machine::new()
    };

    // Set the TSEG size and restrict TSEG to SMM with T_EN.
    let mut esmramc = m.readb(MCH_HOST_BRIDGE_ESMRAMC);
    esmramc &= !MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_MASK;
    esmramc |= esmramc_tseg_sz;
    esmramc |= MCH_HOST_BRIDGE_ESMRAMC_T_EN;
    m.writeb(MCH_HOST_BRIDGE_ESMRAMC, esmramc);

    // Enable TSEG with G_SMRAME and close it with D_CLS.
    let mut smram = m.readb(MCH_HOST_BRIDGE_SMRAM);
    smram &= !(MCH_HOST_BRIDGE_SMRAM_D_OPEN | MCH_HOST_BRIDGE_SMRAM_D_LCK);
    smram |= MCH_HOST_BRIDGE_SMRAM_D_CLS | MCH_HOST_BRIDGE_SMRAM_G_SMRAME;
    m.writeb(MCH_HOST_BRIDGE_SMRAM, smram);

    // Lock TSEG.
    smram |= MCH_HOST_BRIDGE_SMRAM_D_LCK;
    m.writeb(MCH_HOST_BRIDGE_SMRAM, smram);

    // The byte right before TSEG is read/write, the first TSEG byte reads as 0xff.
    let mut ram_offs = (RAM_MBYTES - expected_tseg_mbytes) * MIB - 1;
    assert_eq!(m.memb(ram_offs), 0);
    m.set_memb(ram_offs, 1);
    assert_eq!(m.memb(ram_offs), 1);

    ram_offs += 1;
    assert_eq!(m.memb(ram_offs), 0xff);
    m.set_memb(ram_offs, 1);
    assert_eq!(m.memb(ram_offs), 0xff);
    // SMM sees the RAM behind it, which the write above did not reach.
    assert_eq!(m.smm_memb(ram_offs), 0);
    assert_eq!(m.smm_memb(RAM_MBYTES * MIB - 1), 0);
}

#[test]
fn tseg_size_1mb() {
    tseg_size(MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_1MB, 0, 1);
}

#[test]
fn tseg_size_2mb() {
    tseg_size(MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_2MB, 0, 2);
}

#[test]
fn tseg_size_8mb() {
    tseg_size(MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_8MB, 0, 8);
}

#[test]
fn tseg_size_ext_16mb() {
    tseg_size(MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_MASK, 16, 16);
}

#[test]
fn tseg_off_without_t_en() {
    let m = Machine::new();
    m.writeb(MCH_HOST_BRIDGE_ESMRAMC, MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_1MB);
    m.writeb(MCH_HOST_BRIDGE_SMRAM, MCH_HOST_BRIDGE_SMRAM_G_SMRAME);
    let top = RAM_MBYTES * MIB - 1;
    m.set_memb(top, 9);
    assert_eq!(m.memb(top), 9);
}

#[test]
fn ext_tseg_mbytes_query() {
    let m = Machine::with(|c| c.ext_tseg_mbytes = 48);
    assert_eq!(m.cfg_read(0, MCH_HOST_BRIDGE_EXT_TSEG_MBYTES as u32, 2), 48);
    // Firmware asks by writing 0xffff and reads back the size.
    m.cfg_write(0, MCH_HOST_BRIDGE_EXT_TSEG_MBYTES as u32, 2, 0);
    assert_eq!(m.cfg_read(0, MCH_HOST_BRIDGE_EXT_TSEG_MBYTES as u32, 2), 0);
    m.cfg_write(0, MCH_HOST_BRIDGE_EXT_TSEG_MBYTES as u32, 2, 0xffff);
    assert_eq!(m.cfg_read(0, MCH_HOST_BRIDGE_EXT_TSEG_MBYTES as u32, 2), 48);

    // With the feature off, the register stays whatever the guest writes.
    let m = Machine::with(|c| c.ext_tseg_mbytes = 0);
    assert_eq!(m.cfg_read(0, MCH_HOST_BRIDGE_EXT_TSEG_MBYTES as u32, 2), 0);
    m.cfg_write(0, MCH_HOST_BRIDGE_EXT_TSEG_MBYTES as u32, 2, 0xffff);
    assert_eq!(m.cfg_read(0, MCH_HOST_BRIDGE_EXT_TSEG_MBYTES as u32, 2), 0xffff);
}

#[test]
fn ext_tseg_mbytes_too_large() {
    let mem = Arc::new(MemorySystem::new());
    let sys = mem.new_container("system", 1 << 64).unwrap();
    let io = mem.new_container("io", 1 << 16).unwrap();
    let pci = mem.new_container("pci", 1 << 64).unwrap();
    let ram = mem.new_ram("ram", MIB).unwrap();
    let mut cfg = Q35Config::new(ram, pci, sys, io);
    cfg.ext_tseg_mbytes = 0x1000;
    assert!(Q35PciHost::new(mem, cfg).is_err());
}

/// `test_smram_smbase_lock()`.
#[test]
fn smram_smbase_lock() {
    const SMBASE: u64 = 0x30000;
    const SMRAM_TEST_PATTERN: u8 = 0x32;
    const SMRAM_TEST_RESET_PATTERN: u8 = 0x23;
    let m = Machine::new();

    // SMRAM at SMBASE is off by default.
    assert_eq!(m.readb(MCH_HOST_BRIDGE_F_SMBASE), 0);
    m.set_memb(SMBASE, SMRAM_TEST_PATTERN);
    assert_eq!(m.memb(SMBASE), SMRAM_TEST_PATTERN);

    // Enable it.
    m.writeb(MCH_HOST_BRIDGE_F_SMBASE, 0xff);
    assert_eq!(m.readb(MCH_HOST_BRIDGE_F_SMBASE), 0x01);
    // Lock it.
    m.writeb(MCH_HOST_BRIDGE_F_SMBASE, 0x02);
    assert_eq!(m.readb(MCH_HOST_BRIDGE_F_SMBASE), 0x02);

    // Locked for good.
    assert_eq!(m.memb(SMBASE), 0xff);
    for i in 0..=0xffu8 {
        m.writeb(MCH_HOST_BRIDGE_F_SMBASE, i);
        assert_eq!(m.readb(MCH_HOST_BRIDGE_F_SMBASE), 0x02);
        m.set_memb(SMBASE, SMRAM_TEST_PATTERN);
        assert_eq!(m.memb(SMBASE), 0xff);
    }
    // SMM still sees the RAM.
    assert_eq!(m.smm_memb(SMBASE), SMRAM_TEST_PATTERN);

    m.q35.reset();

    // RAM at SMBASE is back.
    assert_eq!(m.memb(SMBASE), SMRAM_TEST_PATTERN);
    assert_eq!(m.readb(MCH_HOST_BRIDGE_F_SMBASE), 0);
    m.set_memb(SMBASE, SMRAM_TEST_RESET_PATTERN);
    assert_eq!(m.memb(SMBASE), SMRAM_TEST_RESET_PATTERN);
}

#[test]
fn without_smm_ranges() {
    let m = Machine::with(|c| c.has_smm_ranges = false);
    assert!(m.smm_as.is_none());
    assert!(!m.q35.has_smm_ranges());
    m.ram_fill(0xa0000, 0x20000, 0x11);
    // SMRAM register writes change nothing, VGA always shows.
    m.writeb(MCH_HOST_BRIDGE_SMRAM, MCH_HOST_BRIDGE_SMRAM_G_SMRAME | MCH_HOST_BRIDGE_SMRAM_D_OPEN);
    assert_eq!(m.memb(0xa0000), VGA_BYTE);
    // The register is plain storage, as the generic wmask allows.
    assert_eq!(m.readb(MCH_HOST_BRIDGE_SMRAM), 0x48);
    // PAM still works.
    m.writeb(MCH_HOST_BRIDGE_PAM0, 0x30);
    assert_eq!(m.memb(0xf0000), 0);
}

#[test]
fn mch_host_and_bus_vmstate_round_trip() {
    let src = Machine::new();
    src.ram_fill(0xa0000, 0x60000, 0x22);
    src.writeb(MCH_HOST_BRIDGE_PAM0, 0x30);
    src.writeb(
        MCH_HOST_BRIDGE_SMRAM,
        MCH_HOST_BRIDGE_SMRAM_G_SMRAME | MCH_HOST_BRIDGE_SMRAM_D_OPEN,
    );
    // Leaves CONFIG_ADDRESS pointing at the SMRAM register.
    assert!(src.smram_test_bit(MCH_HOST_BRIDGE_SMRAM_D_OPEN));
    let mch = src.q35.mch_vmstate_save();
    let host = src.q35.host_state().vmstate_save();
    let bus = src.q35.bus().vmstate_save();
    assert_eq!(mch.parent_obj.config.len(), 256);
    assert_eq!(host.config_reg, Machine::cfg_addr(0, MCH_HOST_BRIDGE_SMRAM as u32));
    assert_eq!(bus.nirq as usize, bus.irq_count.len());

    let dst = Machine::new();
    dst.ram_fill(0xa0000, 0x60000, 0x22);
    assert_eq!(dst.memb(0xf0000), BIOS_BYTE);
    assert_eq!(dst.memb(0xa0000), VGA_BYTE);
    dst.q35.mch_vmstate_load(&mch).unwrap();
    dst.q35.host_state().vmstate_load(&host);
    dst.q35.bus().vmstate_load(&bus).unwrap();
    assert_eq!(dst.q35.mch_vmstate_save(), mch);
    assert_eq!(dst.q35.host_state().vmstate_save(), host);
    assert_eq!(dst.q35.bus().vmstate_save(), bus);

    // mch_post_load() remapped PAM and SMRAM from the loaded config space.
    assert_eq!(dst.memb(0xf0000), 0x22);
    assert_eq!(dst.q35.mch().pam_region(0).current(), 3);
    assert_eq!(dst.memb(0xa0000), 0x22);

    let mut bad = bus.clone();
    bad.nirq += 1;
    bad.irq_count.push(0);
    assert!(dst.q35.bus().vmstate_load(&bad).is_err());
    let mut bad = bus;
    bad.nirq += 1;
    assert!(dst.q35.bus().vmstate_load(&bad).is_err());
}
