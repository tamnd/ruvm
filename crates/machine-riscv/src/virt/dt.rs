// SPDX-License-Identifier: GPL-2.0-or-later

//! The virt device tree: `create_fdt()` and `finalize_fdt()` of hw/riscv/virt.c, the helpers of
//! hw/riscv/fdt-common.c they call, `riscv_isa_write_fdt()` of target/riscv/cpu.c,
//! `riscv_pmu_generate_fdt_node()` and `platform_bus_add_all_fdt_nodes()`. Each function makes
//! the same libfdt calls in the same order as its QEMU namesake, so with the same ISA strings
//! and `rng-seed` the packed blob is byte for byte what `-M virt,dumpdtb=` writes (see the test
//! against `tests/data/virt.dtb`).

use ruvm_hw_misc::sifive_test::{FINISHER_PASS, FINISHER_RESET};
use ruvm_machine_arm::fdt::{Fdt, sized_cells};
use ruvm_target_riscv::cpu::{IRQ_M_EXT, IRQ_M_SOFT, IRQ_M_TIMER, IRQ_S_EXT};

use super::{
    PCIE_IRQ, RTC_IRQ, UART0_IRQ, VIRT_CLINT, VIRT_CLINT_SIZE, VIRT_DRAM, VIRT_FLASH,
    VIRT_FLASH_SIZE, VIRT_FW_CFG, VIRT_FW_CFG_SIZE, VIRT_IRQCHIP_NUM_SOURCES, VIRT_PCIE_ECAM,
    VIRT_PCIE_ECAM_SIZE, VIRT_PCIE_MMIO, VIRT_PCIE_MMIO_SIZE, VIRT_PCIE_PIO, VIRT_PCIE_PIO_SIZE,
    VIRT_PLATFORM_BUS, VIRT_PLATFORM_BUS_IRQ, VIRT_PLATFORM_BUS_SIZE, VIRT_PLIC, VIRT_PLIC_SIZE,
    VIRT_RTC, VIRT_RTC_SIZE, VIRT_TEST, VIRT_TEST_SIZE, VIRT_UART0, VIRT_UART0_SIZE, VIRT_VIRTIO,
    VIRT_VIRTIO_SIZE, VIRTIO_COUNT, VIRTIO_IRQ, high_pcie_base,
};

/// `RISCV_ACLINT_DEFAULT_TIMEBASE_FREQ`, the `timebase-frequency` of `/cpus`.
const TIMEBASE_FREQ: u32 = 10_000_000;
/// `VIRT64_HIGH_PCIE_MMIO_SIZE`.
pub(crate) const VIRT64_HIGH_PCIE_MMIO_SIZE: u64 = 16 << 30;
/// `PCIE_MMCFG_SIZE_MIN`.
const PCIE_MMCFG_SIZE_MIN: u64 = 1 << 20;
/// `FDT_PCI_RANGE_IOPORT`, `FDT_PCI_RANGE_MMIO` and `FDT_PCI_RANGE_MMIO_64BIT`.
const FDT_PCI_RANGE_IOPORT: u32 = 0x0100_0000;
const FDT_PCI_RANGE_MMIO: u32 = 0x0200_0000;
const FDT_PCI_RANGE_MMIO_64BIT: u32 = 0x0300_0000;
/// `PCI_NUM_PINS`.
const PCI_NUM_PINS: u32 = 4;
/// The `pmu-mask` of the default CPU, `MAKE_64BIT_MASK(3, 16)`: mhpmcounter3 to 18.
const PMU_MASK: u32 = 0x0007_fff8;
/// The block size of the cache block management instructions, `cbom_blocksize`,
/// `cboz_blocksize` and `cbop_blocksize`.
const CBO_BLOCK_SIZE: u32 = 64;

/// The `riscv,isa` string QEMU 11.1 writes for its default `rv64` CPU.
pub const QEMU_RV64_ISA: &str = "rv64imafdch_zic64b_zicbom_zicbop_zicboz_ziccamoa_ziccif_\
                                 zicclsm_ziccrse_zicntr_zicsr_zifencei_zihintntl_zihintpause_\
                                 zihpm_zmmul_za64rs_zaamo_zalrsc_zawrs_zfa_zca_zcd_zba_zbb_zbc_\
                                 zbs_sdtrig_shcounterenw_shgatpa_shtvala_shvsatpa_shvstvala_\
                                 shvstvecd_ssccptr_sscounterenw_ssstrict_sstc_sstvala_sstvecd_\
                                 ssu64xl_svadu_svvptc";

/// The `riscv,isa` string of the default CPU here, the same as QEMU's: H is on by default.
pub const RUVM_RV64_ISA: &str = QEMU_RV64_ISA;

/// `riscv,isa-extensions` for a `riscv,isa` string, as `riscv_isa_write_fdt()` builds both
/// from the same list: each single letter extension after `rv64`, then each multi-letter one.
pub fn isa_extensions(isa: &str) -> Vec<String> {
    let rest = isa.strip_prefix("rv64").or_else(|| isa.strip_prefix("rv32")).unwrap_or(isa);
    let mut parts = rest.split('_');
    let mut out: Vec<String> = parts.next().unwrap_or("").chars().map(|c| c.to_string()).collect();
    out.extend(parts.filter(|p| !p.is_empty()).map(str::to_string));
    out
}

/// A string list property, `qemu_fdt_setprop_string_array()`.
fn string_array<S: AsRef<str>>(items: &[S]) -> Vec<u8> {
    let mut v = Vec::new();
    for s in items {
        v.extend_from_slice(s.as_ref().as_bytes());
        v.push(0);
    }
    v
}

fn cells(values: &[(u32, u64)], what: &str) -> Result<Vec<u8>, String> {
    sized_cells(values).ok_or_else(|| format!("qemu_fdt_setprop_sized_cells: {what}"))
}

fn be_cells(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|c| c.to_be_bytes()).collect()
}

/// `create_board_device_tree()` and the rest of `create_fdt()`: the root, `/soc`, the empty
/// PCIe node, `/chosen` with `rng-seed`, `/aliases`, the flash, fw_cfg and PMU nodes.
pub(crate) fn create_fdt(fdt: &mut Fdt, rng_seed: &[u8; 32]) -> Result<(), String> {
    fdt.setprop_string("/", "model", "riscv-virtio,qemu")?;
    fdt.setprop_string("/", "compatible", "riscv-virtio")?;
    fdt.setprop_cell("/", "#size-cells", 0x2)?;
    fdt.setprop_cell("/", "#address-cells", 0x2)?;

    fdt.add_subnode("/soc")?;
    fdt.setprop("/soc", "ranges", &[])?;
    fdt.setprop_string("/soc", "compatible", "simple-bus")?;
    fdt.setprop_cell("/soc", "#size-cells", 0x2)?;
    fdt.setprop_cell("/soc", "#address-cells", 0x2)?;

    fdt.add_subnode(&format!("/soc/pci@{VIRT_PCIE_ECAM:x}"))?;

    fdt.add_subnode("/chosen")?;
    // Pass seed to RNG.
    fdt.setprop("/chosen", "rng-seed", rng_seed)?;

    fdt.add_subnode("/aliases")?;

    create_fdt_flash(fdt)?;
    create_fdt_fw_cfg(fdt)?;
    create_fdt_pmu(fdt)
}

/// `create_fdt_flash()`.
fn create_fdt_flash(fdt: &mut Fdt) -> Result<(), String> {
    let flashsize = VIRT_FLASH_SIZE / 2;
    let name = format!("/flash@{VIRT_FLASH:x}");
    fdt.add_subnode(&name)?;
    fdt.setprop_string(&name, "compatible", "cfi-flash")?;
    let reg = cells(
        &[(2, VIRT_FLASH), (2, flashsize), (2, VIRT_FLASH + flashsize), (2, flashsize)],
        "flash reg",
    )?;
    fdt.setprop(&name, "reg", &reg)?;
    fdt.setprop_cell(&name, "bank-width", 4)
}

/// `create_fdt_fw_cfg()`.
fn create_fdt_fw_cfg(fdt: &mut Fdt) -> Result<(), String> {
    let name = format!("/fw-cfg@{VIRT_FW_CFG:x}");
    fdt.add_subnode(&name)?;
    fdt.setprop_string(&name, "compatible", "qemu,fw-cfg-mmio")?;
    fdt.setprop(&name, "reg", &cells(&[(2, VIRT_FW_CFG), (2, VIRT_FW_CFG_SIZE)], "fw-cfg reg")?)?;
    fdt.setprop(&name, "dma-coherent", &[])
}

/// `create_fdt_pmu()` and `riscv_pmu_generate_fdt_node()`: the SBI PMU event map, with the
/// cycle and instret events on their fixed counters and the TLB miss events on the
/// programmable ones.
fn create_fdt_pmu(fdt: &mut Fdt) -> Result<(), String> {
    let name = "/pmu";
    fdt.add_subnode(name)?;
    fdt.setprop_string(name, "compatible", "riscv,pmu")?;
    let cmask = PMU_MASK;
    let map: [u32; 15] = [
        // SBI_PMU_HW_CPU_CYCLES: 0x01 : 0x01 : 0x00001
        0x1,
        0x1,
        cmask | 1,
        // SBI_PMU_HW_INSTRUCTIONS: 0x02 : 0x02 : 0x00004
        0x2,
        0x2,
        cmask | 1 << 2,
        // SBI_PMU_HW_CACHE_DTLB : READ : MISS : 0x00019
        0x0001_0019,
        0x0001_0019,
        cmask,
        // SBI_PMU_HW_CACHE_DTLB : WRITE : MISS : 0x0001b
        0x0001_001b,
        0x0001_001b,
        cmask,
        // SBI_PMU_HW_CACHE_ITLB : READ : MISS : 0x00021
        0x0001_0021,
        0x0001_0021,
        cmask,
    ];
    fdt.setprop(name, "riscv,event-to-mhpmcounters", &be_cells(&map))
}

/// What `finalize_fdt()` needs to know about the board.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FinalizeArgs<'a> {
    /// The number of harts, all in socket 0.
    pub(crate) smp: usize,
    /// The RAM size.
    pub(crate) ram_size: u64,
    /// The `riscv,isa` string of the harts.
    pub(crate) isa: &'a str,
}

/// `finalize_fdt()`: the CPU, memory, CLINT, PLIC, platform bus, virtio, PCIe, reset, UART
/// and RTC nodes, with the phandles counted from 1.
pub(crate) fn finalize_fdt(fdt: &mut Fdt, args: FinalizeArgs<'_>) -> Result<(), String> {
    let mut phandle = 1u32;
    let irq_phandle = create_fdt_sockets(fdt, args, &mut phandle)?;
    create_fdt_virtio(fdt, irq_phandle)?;
    create_fdt_pcie(fdt, irq_phandle, args.ram_size)?;
    create_fdt_reset(fdt, &mut phandle)?;
    create_fdt_uart(fdt, irq_phandle)?;
    create_fdt_rtc(fdt, irq_phandle)
}

/// `create_fdt_sockets()` for one socket and the SiFive CLINT and PLIC. Returns the PLIC's
/// phandle, the interrupt parent of the devices.
fn create_fdt_sockets(
    fdt: &mut Fdt,
    args: FinalizeArgs<'_>,
    phandle: &mut u32,
) -> Result<u32, String> {
    // fdt_create_cpu_socket_subnode().
    fdt.add_subnode("/cpus")?;
    fdt.setprop_cell("/cpus", "timebase-frequency", TIMEBASE_FREQ)?;
    fdt.setprop_cell("/cpus", "#size-cells", 0x0)?;
    fdt.setprop_cell("/cpus", "#address-cells", 0x1)?;
    fdt.add_subnode("/cpus/cpu-map")?;

    let socket = 0;
    let clust_name = format!("/cpus/cpu-map/cluster{socket}");
    fdt.add_subnode(&clust_name)?;
    let intc_phandles = create_fdt_socket_cpus(fdt, args, &clust_name, phandle)?;
    create_fdt_socket_memory(fdt, args.ram_size)?;
    create_fdt_socket_clint(fdt, &intc_phandles)?;

    // create_fdt_socket_plic().
    let mut plic_cells = Vec::with_capacity(intc_phandles.len() * 4);
    for &intc in &intc_phandles {
        plic_cells.extend_from_slice(&[intc, IRQ_M_EXT, intc, IRQ_S_EXT]);
    }
    let plic_phandle = *phandle;
    *phandle += 1;
    let plic_name = format!("/soc/interrupt-controller@{VIRT_PLIC:x}");
    create_fdt_plic(fdt, &plic_name, &plic_cells, plic_phandle)?;
    platform_bus_add_all_fdt_nodes(
        fdt,
        &plic_name,
        VIRT_PLATFORM_BUS,
        VIRT_PLATFORM_BUS_SIZE,
        VIRT_PLATFORM_BUS_IRQ,
    )?;
    Ok(plic_phandle)
}

/// `create_fdt_socket_cpus()` and `create_fdt_socket_cpu_internal()`, from the last hart
/// down. Returns the phandles of the harts' interrupt controllers, by hart.
fn create_fdt_socket_cpus(
    fdt: &mut Fdt,
    args: FinalizeArgs<'_>,
    clust_name: &str,
    phandle: &mut u32,
) -> Result<Vec<u32>, String> {
    let mut intc_phandles = vec![0u32; args.smp];
    let extensions = isa_extensions(args.isa);
    for cpu in (0..args.smp).rev() {
        let cpu_phandle = *phandle;
        *phandle += 1;
        let cpu_name = format!("/cpus/cpu@{cpu}");
        fdt.add_subnode(&cpu_name)?;
        fdt.setprop_string(&cpu_name, "mmu-type", "riscv,sv57")?;
        // riscv_isa_write_fdt().
        fdt.setprop_string(&cpu_name, "riscv,isa", args.isa)?;
        fdt.setprop_string(&cpu_name, "riscv,isa-base", "rv64i")?;
        fdt.setprop(&cpu_name, "riscv,isa-extensions", &string_array(&extensions))?;
        fdt.setprop_cell(&cpu_name, "riscv,cbom-block-size", CBO_BLOCK_SIZE)?;
        fdt.setprop_cell(&cpu_name, "riscv,cboz-block-size", CBO_BLOCK_SIZE)?;
        fdt.setprop_cell(&cpu_name, "riscv,cbop-block-size", CBO_BLOCK_SIZE)?;
        fdt.setprop_string(&cpu_name, "compatible", "riscv")?;
        fdt.setprop_string(&cpu_name, "status", "okay")?;
        fdt.setprop_cell(&cpu_name, "reg", cpu as u32)?;
        fdt.setprop_string(&cpu_name, "device_type", "cpu")?;
        fdt.setprop_cell(&cpu_name, "phandle", cpu_phandle)?;

        let intc = *phandle;
        *phandle += 1;
        intc_phandles[cpu] = intc;
        let intc_name = format!("{cpu_name}/interrupt-controller");
        fdt.add_subnode(&intc_name)?;
        fdt.setprop_cell(&intc_name, "phandle", intc)?;
        fdt.setprop_string(&intc_name, "compatible", "riscv,cpu-intc")?;
        fdt.setprop(&intc_name, "interrupt-controller", &[])?;
        fdt.setprop_cell(&intc_name, "#interrupt-cells", 1)?;

        let core_name = format!("{clust_name}/core{cpu}");
        fdt.add_subnode(&core_name)?;
        fdt.setprop_cell(&core_name, "cpu", cpu_phandle)?;
    }
    Ok(intc_phandles)
}

/// `create_fdt_socket_memory()`.
fn create_fdt_socket_memory(fdt: &mut Fdt, ram_size: u64) -> Result<(), String> {
    let name = format!("/memory@{VIRT_DRAM:x}");
    fdt.add_subnode(&name)?;
    fdt.setprop(&name, "reg", &cells(&[(2, VIRT_DRAM), (2, ram_size)], "memory reg")?)?;
    fdt.setprop_string(&name, "device_type", "memory")
}

/// `create_fdt_socket_clint()`: the software and timer interrupts of each hart.
fn create_fdt_socket_clint(fdt: &mut Fdt, intc_phandles: &[u32]) -> Result<(), String> {
    let mut clint_cells = Vec::with_capacity(intc_phandles.len() * 4);
    for &intc in intc_phandles {
        clint_cells.extend_from_slice(&[intc, IRQ_M_SOFT, intc, IRQ_M_TIMER]);
    }
    let name = format!("/soc/clint@{VIRT_CLINT:x}");
    fdt.add_subnode(&name)?;
    fdt.setprop(&name, "compatible", &string_array(&["sifive,clint0", "riscv,clint0"]))?;
    fdt.setprop(&name, "reg", &cells(&[(2, VIRT_CLINT), (2, VIRT_CLINT_SIZE)], "clint reg")?)?;
    fdt.setprop(&name, "interrupts-extended", &be_cells(&clint_cells))
}

/// `create_fdt_plic()`.
fn create_fdt_plic(
    fdt: &mut Fdt,
    name: &str,
    plic_cells: &[u32],
    plic_phandle: u32,
) -> Result<(), String> {
    fdt.add_subnode(name)?;
    fdt.setprop_cell(name, "#interrupt-cells", 1)?;
    fdt.setprop_cell(name, "#address-cells", 0)?;
    fdt.setprop(name, "compatible", &string_array(&["sifive,plic-1.0.0", "riscv,plic0"]))?;
    fdt.setprop(name, "interrupt-controller", &[])?;
    fdt.setprop(name, "interrupts-extended", &be_cells(plic_cells))?;
    fdt.setprop(name, "reg", &cells(&[(2, VIRT_PLIC), (2, VIRT_PLIC_SIZE)], "plic reg")?)?;
    fdt.setprop_cell(name, "riscv,ndev", VIRT_IRQCHIP_NUM_SOURCES - 1)?;
    fdt.setprop_cell(name, "phandle", plic_phandle)
}

/// `platform_bus_add_all_fdt_nodes()` with no dynamic sysbus devices: the bus node only.
fn platform_bus_add_all_fdt_nodes(
    fdt: &mut Fdt,
    intc: &str,
    addr: u64,
    bus_size: u64,
    _irq_start: u32,
) -> Result<(), String> {
    let node = format!("/platform-bus@{addr:x}");
    fdt.add_subnode(&node)?;
    fdt.setprop(&node, "compatible", b"qemu,platform\0simple-bus\0")?;
    // Our platform bus region is less than 32 bits, so 1 cell is enough for address and size.
    fdt.setprop_cell(&node, "#size-cells", 1)?;
    fdt.setprop_cell(&node, "#address-cells", 1)?;
    fdt.setprop_cells(&node, "ranges", &[0, (addr >> 32) as u32, addr as u32, bus_size as u32])?;
    let ph = fdt.get_phandle(intc)?;
    fdt.setprop_cell(&node, "interrupt-parent", ph)
}

/// `create_fdt_virtio()`.
fn create_fdt_virtio(fdt: &mut Fdt, irq_virtio_phandle: u32) -> Result<(), String> {
    for i in 0..VIRTIO_COUNT {
        let addr = VIRT_VIRTIO + i as u64 * VIRT_VIRTIO_SIZE;
        let name = format!("/soc/virtio_mmio@{addr:x}");
        fdt.add_subnode(&name)?;
        fdt.setprop_string(&name, "compatible", "virtio,mmio")?;
        fdt.setprop(&name, "reg", &cells(&[(2, addr), (2, VIRT_VIRTIO_SIZE)], "virtio reg")?)?;
        fdt.setprop_cell(&name, "interrupt-parent", irq_virtio_phandle)?;
        fdt.setprop_cell(&name, "interrupts", VIRTIO_IRQ + i as u32)?;
    }
    Ok(())
}

/// `create_fdt_pcie()` and `create_pcie_irq_map()`.
fn create_fdt_pcie(fdt: &mut Fdt, irq_pcie_phandle: u32, ram_size: u64) -> Result<(), String> {
    let name = format!("/soc/pci@{VIRT_PCIE_ECAM:x}");
    fdt.setprop_cell(&name, "#address-cells", 3)?;
    fdt.setprop_cell(&name, "#interrupt-cells", 1)?;
    fdt.setprop_cell(&name, "#size-cells", 2)?;
    fdt.setprop_string(&name, "compatible", "pci-host-ecam-generic")?;
    fdt.setprop_string(&name, "device_type", "pci")?;
    fdt.setprop_cell(&name, "linux,pci-domain", 0)?;
    fdt.setprop_cells(
        &name,
        "bus-range",
        &[0, (VIRT_PCIE_ECAM_SIZE / PCIE_MMCFG_SIZE_MIN - 1) as u32],
    )?;
    fdt.setprop(&name, "dma-coherent", &[])?;
    fdt.setprop(&name, "reg", &cells(&[(2, VIRT_PCIE_ECAM), (2, VIRT_PCIE_ECAM_SIZE)], "reg")?)?;
    let high = high_pcie_base(ram_size);
    let ranges = cells(
        &[
            (1, u64::from(FDT_PCI_RANGE_IOPORT)),
            (2, 0),
            (2, VIRT_PCIE_PIO),
            (2, VIRT_PCIE_PIO_SIZE),
            (1, u64::from(FDT_PCI_RANGE_MMIO)),
            (2, VIRT_PCIE_MMIO),
            (2, VIRT_PCIE_MMIO),
            (2, VIRT_PCIE_MMIO_SIZE),
            (1, u64::from(FDT_PCI_RANGE_MMIO_64BIT)),
            (2, high),
            (2, high),
            (2, VIRT64_HIGH_PCIE_MMIO_SIZE),
        ],
        "pci ranges",
    )?;
    fdt.setprop(&name, "ranges", &ranges)?;

    // A standard swizzle of interrupts such that each device's first interrupt is based on
    // its PCI_SLOT number.
    let mut map = Vec::with_capacity((PCI_NUM_PINS * PCI_NUM_PINS * 6) as usize);
    for dev in 0..PCI_NUM_PINS {
        let devfn = dev * 0x8;
        for pin in 0..PCI_NUM_PINS {
            let irq_nr = PCIE_IRQ + ((pin + (devfn >> 3)) % PCI_NUM_PINS);
            map.extend_from_slice(&[devfn << 8, 0, 0, pin + 1, irq_pcie_phandle, irq_nr]);
        }
    }
    fdt.setprop(&name, "interrupt-map", &be_cells(&map))?;
    fdt.setprop_cells(&name, "interrupt-map-mask", &[0x1800, 0, 0, 0x7])
}

/// `create_fdt_reset()`: the test finisher and the syscon reboot and poweroff nodes on it.
fn create_fdt_reset(fdt: &mut Fdt, phandle: &mut u32) -> Result<(), String> {
    let test_phandle = *phandle;
    *phandle += 1;
    let name = format!("/soc/test@{VIRT_TEST:x}");
    fdt.add_subnode(&name)?;
    fdt.setprop(&name, "compatible", &string_array(&["sifive,test1", "sifive,test0", "syscon"]))?;
    fdt.setprop(&name, "reg", &cells(&[(2, VIRT_TEST), (2, VIRT_TEST_SIZE)], "test reg")?)?;
    fdt.setprop_cell(&name, "phandle", test_phandle)?;
    let test_phandle = fdt.get_phandle(&name)?;

    for (node, compat, value) in [
        ("/reboot", "syscon-reboot", FINISHER_RESET),
        ("/poweroff", "syscon-poweroff", FINISHER_PASS),
    ] {
        fdt.add_subnode(node)?;
        fdt.setprop_string(node, "compatible", compat)?;
        fdt.setprop_cell(node, "regmap", test_phandle)?;
        fdt.setprop_cell(node, "offset", 0x0)?;
        fdt.setprop_cell(node, "value", u32::from(value))?;
    }
    Ok(())
}

/// `create_fdt_uart()`.
fn create_fdt_uart(fdt: &mut Fdt, irq_mmio_phandle: u32) -> Result<(), String> {
    let name = format!("/soc/serial@{VIRT_UART0:x}");
    fdt.add_subnode(&name)?;
    fdt.setprop_string(&name, "compatible", "ns16550a")?;
    fdt.setprop(&name, "reg", &cells(&[(2, VIRT_UART0), (2, VIRT_UART0_SIZE)], "uart reg")?)?;
    fdt.setprop_cell(&name, "clock-frequency", 3_686_400)?;
    fdt.setprop_cell(&name, "interrupt-parent", irq_mmio_phandle)?;
    fdt.setprop_cell(&name, "interrupts", UART0_IRQ)?;
    fdt.setprop_string("/chosen", "stdout-path", &name)?;
    fdt.setprop_string("/aliases", "serial0", &name)
}

/// `create_fdt_rtc()`.
fn create_fdt_rtc(fdt: &mut Fdt, irq_mmio_phandle: u32) -> Result<(), String> {
    let name = format!("/soc/rtc@{VIRT_RTC:x}");
    fdt.add_subnode(&name)?;
    fdt.setprop_string(&name, "compatible", "google,goldfish-rtc")?;
    fdt.setprop(&name, "reg", &cells(&[(2, VIRT_RTC), (2, VIRT_RTC_SIZE)], "rtc reg")?)?;
    fdt.setprop_cell(&name, "interrupt-parent", irq_mmio_phandle)?;
    fdt.setprop_cell(&name, "interrupts", RTC_IRQ)
}

fn be32(b: &[u8], off: usize) -> usize {
    b.get(off..off + 4).map_or(0, |w| u32::from_be_bytes([w[0], w[1], w[2], w[3]]) as usize)
}

/// `fdt_pack()`: the blocks moved down to right after the header, in libfdt's order, and the
/// free space at the end dropped.
pub(crate) fn fdt_pack(fdt: &Fdt) -> Result<Fdt, String> {
    const HEADER_SIZE: usize = 40;
    let mut b = fdt.as_bytes().to_vec();
    let off_struct = be32(&b, 8);
    let off_strings = be32(&b, 12);
    let off_rsvmap = be32(&b, 16);
    let size_strings = be32(&b, 32);
    let size_struct = be32(&b, 36);
    // fdt_num_mem_rsv(): the entries up to the one with size 0.
    let mut n = 0;
    loop {
        let e = off_rsvmap + n * 16;
        if e + 16 > b.len() {
            return Err("invalid device-tree".to_string());
        }
        if b[e + 8..e + 16].iter().all(|&x| x == 0) {
            break;
        }
        n += 1;
    }
    let mem_rsv_size = (n + 1) * 16;
    let rsv_to = HEADER_SIZE;
    let struct_to = rsv_to + mem_rsv_size;
    let strings_to = struct_to + size_struct;
    for (from, len) in [(off_rsvmap, mem_rsv_size), (off_struct, size_struct)] {
        if from + len > b.len() {
            return Err("invalid device-tree".to_string());
        }
    }
    if off_strings + size_strings > b.len() {
        return Err("invalid device-tree".to_string());
    }
    b.copy_within(off_rsvmap..off_rsvmap + mem_rsv_size, rsv_to);
    b.copy_within(off_struct..off_struct + size_struct, struct_to);
    b.copy_within(off_strings..off_strings + size_strings, strings_to);
    let totalsize = strings_to + size_strings;
    for (field, v) in [(4, totalsize), (8, struct_to), (12, strings_to), (16, rsv_to)] {
        b[field..field + 4].copy_from_slice(&(v as u32).to_be_bytes());
    }
    b.truncate(totalsize);
    Fdt::open_into(&b, totalsize).map_err(|e| format!("invalid device-tree: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tree of `qemu-system-riscv64 -M virt,dumpdtb=virt.dtb` from QEMU 11.1.2, one hart
    /// and 128 MiB.
    const QEMU_VIRT_DTB: &[u8] = include_bytes!("../../tests/data/virt.dtb");

    /// The `rng-seed` in that tree.
    const SEED: [u32; 8] = [
        0xc1f3_9876,
        0x8e43_775e,
        0x088b_7f80,
        0xd62c_abba,
        0x7b3f_304b,
        0xf5c9_2389,
        0xd544_5b45,
        0x1adb_640c,
    ];

    fn seed() -> [u8; 32] {
        let mut s = [0u8; 32];
        for (i, w) in SEED.iter().enumerate() {
            s[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
        }
        s
    }

    fn build(isa: &str, smp: usize, ram_size: u64) -> Fdt {
        let mut fdt = Fdt::new();
        create_fdt(&mut fdt, &seed()).unwrap();
        finalize_fdt(&mut fdt, FinalizeArgs { smp, ram_size, isa }).unwrap();
        fdt_pack(&fdt).unwrap()
    }

    #[test]
    fn matches_qemu_byte_for_byte() {
        let fdt = build(QEMU_RV64_ISA, 1, 128 << 20);
        assert_eq!(fdt.as_bytes().len(), QEMU_VIRT_DTB.len());
        assert!(fdt.as_bytes() == QEMU_VIRT_DTB, "the tree differs from QEMU's");
    }

    #[test]
    fn isa_extension_list() {
        let ext = isa_extensions(QEMU_RV64_ISA);
        assert_eq!(&ext[..8], ["i", "m", "a", "f", "d", "c", "h", "zic64b"]);
        assert_eq!(ext.last().unwrap(), "svvptc");
        assert_eq!(ext.len(), 49);
        assert_eq!(isa_extensions(RUVM_RV64_ISA), ext);
    }

    #[test]
    fn two_harts_phandles() {
        let fdt = build(RUVM_RV64_ISA, 2, 256 << 20);
        // The harts are made from the last down: cpu@1 then its controller, then cpu@0.
        assert_eq!(fdt.get_phandle("/cpus/cpu@1").unwrap(), 1);
        assert_eq!(fdt.get_phandle("/cpus/cpu@1/interrupt-controller").unwrap(), 2);
        assert_eq!(fdt.get_phandle("/cpus/cpu@0").unwrap(), 3);
        assert_eq!(fdt.get_phandle("/cpus/cpu@0/interrupt-controller").unwrap(), 4);
        assert_eq!(fdt.get_phandle("/soc/interrupt-controller@c000000").unwrap(), 5);
        assert_eq!(fdt.get_phandle("/soc/test@100000").unwrap(), 6);
        let clint = fdt.getprop("/soc/clint@2000000", "interrupts-extended").unwrap();
        assert_eq!(clint, be_cells(&[4, 3, 4, 7, 2, 3, 2, 7]));
        let plic = fdt.getprop("/soc/interrupt-controller@c000000", "interrupts-extended");
        assert_eq!(plic.unwrap(), be_cells(&[4, 11, 4, 9, 2, 11, 2, 9]));
        assert_eq!(fdt.getprop_cell("/cpus/cpu-map/cluster0/core1", "cpu").unwrap(), 1);
        let mem = fdt.getprop("/memory@80000000", "reg").unwrap();
        assert_eq!(mem, be_cells(&[0, 0x8000_0000, 0, 0x1000_0000]));
    }
}
