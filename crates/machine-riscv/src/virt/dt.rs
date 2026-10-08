// SPDX-License-Identifier: GPL-2.0-or-later

//! The virt device tree: `create_fdt()` and `finalize_fdt()` of hw/riscv/virt.c, the helpers of
//! hw/riscv/fdt-common.c they call, `riscv_isa_write_fdt()` of target/riscv/cpu.c,
//! `riscv_pmu_generate_fdt_node()` and `platform_bus_add_all_fdt_nodes()`. Each function makes
//! the same libfdt calls in the same order as its QEMU namesake, so with the same CPU
//! configuration and `rng-seed` the packed blob is byte for byte what `-M virt,dumpdtb=`
//! writes (see the tests against the trees in `tests/data`).

use ruvm_hw_misc::sifive_test::{FINISHER_PASS, FINISHER_RESET};
use ruvm_machine_arm::fdt::{Fdt, sized_cells};
use ruvm_target_riscv::cfg::RiscvCfg;
use ruvm_target_riscv::cpu::{IRQ_M_EXT, IRQ_M_SOFT, IRQ_M_TIMER, IRQ_S_EXT};

use ruvm_hw_intc::riscv_imsic::imsic_hart_size;

use super::aia::imsic_num_bits;
use super::{
    PCIE_IRQ, RTC_IRQ, UART0_IRQ, VIRT_APLIC_M, VIRT_APLIC_S, VIRT_APLIC_SIZE, VIRT_CLINT,
    VIRT_CLINT_SIZE, VIRT_DRAM, VIRT_FLASH, VIRT_FLASH_SIZE, VIRT_FW_CFG, VIRT_FW_CFG_SIZE,
    VIRT_IMSIC_M, VIRT_IMSIC_S, VIRT_IRQCHIP_NUM_MSIS, VIRT_IRQCHIP_NUM_SOURCES, VIRT_PCIE_ECAM,
    VIRT_PCIE_ECAM_SIZE, VIRT_PCIE_MMIO, VIRT_PCIE_MMIO_SIZE, VIRT_PCIE_PIO, VIRT_PCIE_PIO_SIZE,
    VIRT_PLATFORM_BUS, VIRT_PLATFORM_BUS_IRQ, VIRT_PLATFORM_BUS_SIZE, VIRT_PLIC, VIRT_PLIC_SIZE,
    VIRT_RTC, VIRT_RTC_SIZE, VIRT_TEST, VIRT_TEST_SIZE, VIRT_UART0, VIRT_UART0_SIZE, VIRT_VIRTIO,
    VIRT_VIRTIO_SIZE, VIRTIO_COUNT, VIRTIO_IRQ, VirtAia, high_pcie_base,
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
/// `FDT_APLIC_ADDR_CELLS`, `FDT_APLIC_INT_CELLS` and `FDT_IMSIC_INT_CELLS`.
const FDT_APLIC_ADDR_CELLS: u32 = 0;
const FDT_APLIC_INT_CELLS: u32 = 2;
const FDT_IMSIC_INT_CELLS: u32 = 0;
/// The trigger type the devices give with the AIA, `IRQ_TYPE_LEVEL_HIGH`.
const IRQ_TYPE_LEVEL_HIGH: u32 = 0x4;

/// The `riscv,isa` string QEMU 11.1 writes for its default `rv64` CPU.
pub const QEMU_RV64_ISA: &str = "rv64imafdch_zic64b_zicbom_zicbop_zicboz_ziccamoa_ziccif_\
                                 zicclsm_ziccrse_zicntr_zicsr_zifencei_zihintntl_zihintpause_\
                                 zihpm_zmmul_za64rs_zaamo_zalrsc_zawrs_zfa_zca_zcd_zba_zbb_zbc_\
                                 zbs_sdtrig_shcounterenw_shgatpa_shtvala_shvsatpa_shvstvala_\
                                 shvstvecd_ssccptr_sscounterenw_ssstrict_sstc_sstvala_sstvecd_\
                                 ssu64xl_svadu_svvptc";

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
/// `pmu_avail_ctrs` is the `pmu_avail_ctrs` of hart 0.
pub(crate) fn create_fdt(
    fdt: &mut Fdt,
    rng_seed: &[u8; 32],
    pmu_avail_ctrs: u32,
) -> Result<(), String> {
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
    create_fdt_pmu(fdt, pmu_avail_ctrs)
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
fn create_fdt_pmu(fdt: &mut Fdt, cmask: u32) -> Result<(), String> {
    let name = "/pmu";
    fdt.add_subnode(name)?;
    fdt.setprop_string(name, "compatible", "riscv,pmu")?;
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
    /// The configuration of the harts.
    pub(crate) cpu: &'a RiscvCfg,
    /// The interrupt controller, `aia`.
    pub(crate) aia: VirtAia,
    /// The guest interrupt files of each S level IMSIC, `aia-guests`.
    pub(crate) aia_guests: u32,
}

/// `finalize_fdt()`: the CPU, memory, CLINT, interrupt controller, platform bus, virtio,
/// PCIe, reset, UART and RTC nodes, with the phandles counted from 1.
pub(crate) fn finalize_fdt(fdt: &mut Fdt, args: FinalizeArgs<'_>) -> Result<(), String> {
    let mut phandle = 1u32;
    let (irq_phandle, msi_pcie_phandle) = create_fdt_sockets(fdt, args, &mut phandle)?;
    let aia = args.aia;
    create_fdt_virtio(fdt, aia, irq_phandle)?;
    create_fdt_pcie(fdt, aia, irq_phandle, msi_pcie_phandle, args.ram_size)?;
    create_fdt_reset(fdt, &mut phandle)?;
    create_fdt_uart(fdt, aia, irq_phandle)?;
    create_fdt_rtc(fdt, aia, irq_phandle)
}

/// The `interrupts` of a device on source `irq`: the source alone with the PLIC, the source
/// and its trigger type with an APLIC.
fn set_interrupts(fdt: &mut Fdt, name: &str, aia: VirtAia, irq: u32) -> Result<(), String> {
    if aia == VirtAia::None {
        fdt.setprop_cell(name, "interrupts", irq)
    } else {
        fdt.setprop_cells(name, "interrupts", &[irq, IRQ_TYPE_LEVEL_HIGH])
    }
}

/// `create_fdt_sockets()` for one socket, with the SiFive CLINT and the PLIC or the AIA
/// devices. Returns the phandle of the interrupt parent of the devices and that of the MSI
/// parent of PCIe (0 without IMSICs).
fn create_fdt_sockets(
    fdt: &mut Fdt,
    args: FinalizeArgs<'_>,
    phandle: &mut u32,
) -> Result<(u32, u32), String> {
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

    let (mut msi_m_phandle, mut msi_s_phandle) = (0, 0);
    if args.aia == VirtAia::AplicImsic {
        // create_fdt_imsic().
        msi_m_phandle = *phandle;
        *phandle += 1;
        msi_s_phandle = *phandle;
        *phandle += 1;
        create_fdt_one_imsic(fdt, VIRT_IMSIC_M, &intc_phandles, msi_m_phandle, true, 0)?;
        let bits = imsic_num_bits(args.aia_guests + 1);
        create_fdt_one_imsic(fdt, VIRT_IMSIC_S, &intc_phandles, msi_s_phandle, false, bits)?;
    }
    if args.aia != VirtAia::None {
        let aplic_s_phandle = create_fdt_socket_aplic(
            fdt,
            args.aia,
            msi_m_phandle,
            msi_s_phandle,
            phandle,
            &intc_phandles,
        )?;
        return Ok((aplic_s_phandle, msi_s_phandle));
    }

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
    Ok((plic_phandle, 0))
}

/// `create_fdt_one_imsic()` for one socket: the IMSICs of level M or S of all the harts.
fn create_fdt_one_imsic(
    fdt: &mut Fdt,
    base_addr: u64,
    intc_phandles: &[u32],
    msi_phandle: u32,
    m_mode: bool,
    imsic_guest_bits: u32,
) -> Result<(), String> {
    let irq = if m_mode { IRQ_M_EXT } else { IRQ_S_EXT };
    let imsic_cells: Vec<u32> = intc_phandles.iter().flat_map(|&intc| [intc, irq]).collect();
    // The cells are 32 bits wide, as in QEMU.
    let imsic_size = imsic_hart_size(imsic_guest_bits) * intc_phandles.len() as u64;
    let imsic_regs = [0, base_addr as u32, 0, imsic_size as u32];

    let name = format!("/soc/interrupt-controller@{base_addr:x}");
    fdt.add_subnode(&name)?;
    fdt.setprop(&name, "compatible", &string_array(&["qemu,imsics", "riscv,imsics"]))?;
    fdt.setprop_cell(&name, "#interrupt-cells", FDT_IMSIC_INT_CELLS)?;
    fdt.setprop(&name, "interrupt-controller", &[])?;
    fdt.setprop(&name, "msi-controller", &[])?;
    fdt.setprop(&name, "interrupts-extended", &be_cells(&imsic_cells))?;
    fdt.setprop(&name, "reg", &be_cells(&imsic_regs))?;
    fdt.setprop_cell(&name, "riscv,num-ids", VIRT_IRQCHIP_NUM_MSIS)?;
    if imsic_guest_bits != 0 {
        fdt.setprop_cell(&name, "riscv,guest-index-bits", imsic_guest_bits)?;
    }
    fdt.setprop_cell(&name, "phandle", msi_phandle)
}

/// `create_fdt_one_aplic()`.
#[allow(clippy::too_many_arguments)]
fn create_fdt_one_aplic(
    fdt: &mut Fdt,
    aia: VirtAia,
    aplic_addr: u64,
    msi_phandle: u32,
    intc_phandles: &[u32],
    aplic_phandle: u32,
    aplic_child_phandle: u32,
    m_mode: bool,
) -> Result<(), String> {
    let irq = if m_mode { IRQ_M_EXT } else { IRQ_S_EXT };
    let aplic_cells: Vec<u32> = intc_phandles.iter().flat_map(|&intc| [intc, irq]).collect();

    let name = format!("/soc/interrupt-controller@{aplic_addr:x}");
    fdt.add_subnode(&name)?;
    fdt.setprop(&name, "compatible", &string_array(&["qemu,aplic", "riscv,aplic"]))?;
    fdt.setprop_cell(&name, "#address-cells", FDT_APLIC_ADDR_CELLS)?;
    fdt.setprop_cell(&name, "#interrupt-cells", FDT_APLIC_INT_CELLS)?;
    fdt.setprop(&name, "interrupt-controller", &[])?;
    if aia == VirtAia::Aplic {
        fdt.setprop(&name, "interrupts-extended", &be_cells(&aplic_cells))?;
    } else {
        fdt.setprop_cell(&name, "msi-parent", msi_phandle)?;
    }
    let reg = cells(&[(2, aplic_addr), (2, VIRT_APLIC_SIZE)], "aplic reg")?;
    fdt.setprop(&name, "reg", &reg)?;
    fdt.setprop_cell(&name, "riscv,num-sources", VIRT_IRQCHIP_NUM_SOURCES)?;
    if aplic_child_phandle != 0 {
        fdt.setprop_cell(&name, "riscv,children", aplic_child_phandle)?;
        fdt.setprop_cells(
            &name,
            "riscv,delegation",
            &[aplic_child_phandle, 0x1, VIRT_IRQCHIP_NUM_SOURCES],
        )?;
    }
    fdt.setprop_cell(&name, "phandle", aplic_phandle)
}

/// `create_fdt_socket_aplic()` for socket 0: the M level domain, then the S level one, its
/// child, and the platform bus on the S level one. Returns the phandle of the S level domain.
fn create_fdt_socket_aplic(
    fdt: &mut Fdt,
    aia: VirtAia,
    msi_m_phandle: u32,
    msi_s_phandle: u32,
    phandle: &mut u32,
    intc_phandles: &[u32],
) -> Result<u32, String> {
    let aplic_m_phandle = *phandle;
    *phandle += 1;
    let aplic_s_phandle = *phandle;
    *phandle += 1;

    // M-level APLIC node
    create_fdt_one_aplic(
        fdt,
        aia,
        VIRT_APLIC_M,
        msi_m_phandle,
        intc_phandles,
        aplic_m_phandle,
        aplic_s_phandle,
        true,
    )?;

    // S-level APLIC node
    create_fdt_one_aplic(
        fdt,
        aia,
        VIRT_APLIC_S,
        msi_s_phandle,
        intc_phandles,
        aplic_s_phandle,
        0,
        false,
    )?;

    let aplic_name = format!("/soc/interrupt-controller@{VIRT_APLIC_S:x}");
    platform_bus_add_all_fdt_nodes(
        fdt,
        &aplic_name,
        VIRT_PLATFORM_BUS,
        VIRT_PLATFORM_BUS_SIZE,
        VIRT_PLATFORM_BUS_IRQ,
    )?;
    Ok(aplic_s_phandle)
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
    let cfg = args.cpu;
    let isa = cfg.isa_string();
    let extensions = cfg.isa_extensions();
    let mmu_type = cfg.mmu_type();
    for cpu in (0..args.smp).rev() {
        let cpu_phandle = *phandle;
        *phandle += 1;
        let cpu_name = format!("/cpus/cpu@{cpu}");
        fdt.add_subnode(&cpu_name)?;
        if let Some(mmu_type) = &mmu_type {
            fdt.setprop_string(&cpu_name, "mmu-type", mmu_type)?;
        }
        // riscv_isa_write_fdt().
        fdt.setprop_string(&cpu_name, "riscv,isa", &isa)?;
        fdt.setprop_string(&cpu_name, "riscv,isa-base", "rv64i")?;
        fdt.setprop(&cpu_name, "riscv,isa-extensions", &string_array(&extensions))?;
        if cfg.ext_zicbom {
            let size = u32::from(cfg.cbom_blocksize);
            fdt.setprop_cell(&cpu_name, "riscv,cbom-block-size", size)?;
        }
        if cfg.ext_zicboz {
            let size = u32::from(cfg.cboz_blocksize);
            fdt.setprop_cell(&cpu_name, "riscv,cboz-block-size", size)?;
        }
        if cfg.ext_zicbop {
            let size = u32::from(cfg.cbop_blocksize);
            fdt.setprop_cell(&cpu_name, "riscv,cbop-block-size", size)?;
        }
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
fn create_fdt_virtio(fdt: &mut Fdt, aia: VirtAia, irq_virtio_phandle: u32) -> Result<(), String> {
    for i in 0..VIRTIO_COUNT {
        let addr = VIRT_VIRTIO + i as u64 * VIRT_VIRTIO_SIZE;
        let name = format!("/soc/virtio_mmio@{addr:x}");
        fdt.add_subnode(&name)?;
        fdt.setprop_string(&name, "compatible", "virtio,mmio")?;
        fdt.setprop(&name, "reg", &cells(&[(2, addr), (2, VIRT_VIRTIO_SIZE)], "virtio reg")?)?;
        fdt.setprop_cell(&name, "interrupt-parent", irq_virtio_phandle)?;
        set_interrupts(fdt, &name, aia, VIRTIO_IRQ + i as u32)?;
    }
    Ok(())
}

/// `create_fdt_pcie()` and `create_pcie_irq_map()`.
fn create_fdt_pcie(
    fdt: &mut Fdt,
    aia: VirtAia,
    irq_pcie_phandle: u32,
    msi_pcie_phandle: u32,
    ram_size: u64,
) -> Result<(), String> {
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
    if aia == VirtAia::AplicImsic {
        fdt.setprop_cell(&name, "msi-parent", msi_pcie_phandle)?;
    }
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
    let mut map = Vec::with_capacity((PCI_NUM_PINS * PCI_NUM_PINS * 7) as usize);
    for dev in 0..PCI_NUM_PINS {
        let devfn = dev * 0x8;
        for pin in 0..PCI_NUM_PINS {
            let irq_nr = PCIE_IRQ + ((pin + (devfn >> 3)) % PCI_NUM_PINS);
            map.extend_from_slice(&[devfn << 8, 0, 0, pin + 1, irq_pcie_phandle, irq_nr]);
            if aia != VirtAia::None {
                map.push(IRQ_TYPE_LEVEL_HIGH);
            }
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
fn create_fdt_uart(fdt: &mut Fdt, aia: VirtAia, irq_mmio_phandle: u32) -> Result<(), String> {
    let name = format!("/soc/serial@{VIRT_UART0:x}");
    fdt.add_subnode(&name)?;
    fdt.setprop_string(&name, "compatible", "ns16550a")?;
    fdt.setprop(&name, "reg", &cells(&[(2, VIRT_UART0), (2, VIRT_UART0_SIZE)], "uart reg")?)?;
    fdt.setprop_cell(&name, "clock-frequency", 3_686_400)?;
    fdt.setprop_cell(&name, "interrupt-parent", irq_mmio_phandle)?;
    set_interrupts(fdt, &name, aia, UART0_IRQ)?;
    fdt.setprop_string("/chosen", "stdout-path", &name)?;
    fdt.setprop_string("/aliases", "serial0", &name)
}

/// `create_fdt_rtc()`.
fn create_fdt_rtc(fdt: &mut Fdt, aia: VirtAia, irq_mmio_phandle: u32) -> Result<(), String> {
    let name = format!("/soc/rtc@{VIRT_RTC:x}");
    fdt.add_subnode(&name)?;
    fdt.setprop_string(&name, "compatible", "google,goldfish-rtc")?;
    fdt.setprop(&name, "reg", &cells(&[(2, VIRT_RTC), (2, VIRT_RTC_SIZE)], "rtc reg")?)?;
    fdt.setprop_cell(&name, "interrupt-parent", irq_mmio_phandle)?;
    set_interrupts(fdt, &name, aia, RTC_IRQ)
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

    fn build(cpu: &RiscvCfg, smp: usize, ram_size: u64) -> Fdt {
        build_aia(cpu, smp, ram_size, VirtAia::None, 0, &seed())
    }

    fn build_aia(
        cpu: &RiscvCfg,
        smp: usize,
        ram_size: u64,
        aia: VirtAia,
        aia_guests: u32,
        seed: &[u8; 32],
    ) -> Fdt {
        let mut fdt = Fdt::new();
        create_fdt(&mut fdt, seed, cpu.pmu_mask).unwrap();
        finalize_fdt(&mut fdt, FinalizeArgs { smp, ram_size, cpu, aia, aia_guests }).unwrap();
        fdt_pack(&fdt).unwrap()
    }

    /// Build the tree of `golden`, a `dumpdtb` of QEMU 11.1.2 with 128 MiB, `-bios none` and
    /// the given `-smp` and `-M virt,aia=...,aia-guests=...`, with its `rng-seed`, and check
    /// that the two are the same bytes.
    fn check_golden(golden: &[u8], smp: usize, aia: VirtAia, aia_guests: u32) -> Fdt {
        let qemu = Fdt::open_into(golden, golden.len()).unwrap();
        let seed: [u8; 32] = qemu.getprop("/chosen", "rng-seed").unwrap().try_into().unwrap();
        let imsic = aia == VirtAia::AplicImsic;
        let cpu = RiscvCfg { ext_smaia: imsic, ext_ssaia: imsic, ..RiscvCfg::default() };
        let fdt = build_aia(&cpu, smp, 128 << 20, aia, aia_guests, &seed);
        assert_eq!(fdt.as_bytes().len(), golden.len());
        assert!(fdt.as_bytes() == golden, "the tree differs from QEMU's");
        fdt
    }

    #[test]
    fn aplic_matches_qemu() {
        let golden = include_bytes!("../../tests/data/virt-aplic-smp2.dtb");
        let fdt = check_golden(golden, 2, VirtAia::Aplic, 0);
        // cpu@1 and its controller, cpu@0 and its controller, then the M and S domains.
        assert_eq!(fdt.get_phandle("/soc/interrupt-controller@c000000").unwrap(), 5);
        assert_eq!(fdt.get_phandle("/soc/interrupt-controller@d000000").unwrap(), 6);
        let m = fdt.getprop("/soc/interrupt-controller@c000000", "interrupts-extended");
        assert_eq!(m.unwrap(), be_cells(&[4, 11, 2, 11]));
        let s = fdt.getprop("/soc/interrupt-controller@d000000", "interrupts-extended");
        assert_eq!(s.unwrap(), be_cells(&[4, 9, 2, 9]));
        assert_eq!(fdt.getprop("/soc/serial@10000000", "interrupts").unwrap(), be_cells(&[10, 4]));
    }

    #[test]
    fn aplic_imsic_matches_qemu() {
        let golden = include_bytes!("../../tests/data/virt-aplic-imsic.dtb");
        let fdt = check_golden(golden, 1, VirtAia::AplicImsic, 0);
        let isa = fdt.getprop("/cpus/cpu@0", "riscv,isa").unwrap();
        assert!(isa.windows(13).any(|w| w == b"_smaia_ssaia_"));
        let pci = fdt.getprop_cell("/soc/pci@30000000", "msi-parent").unwrap();
        assert_eq!(pci, fdt.get_phandle("/soc/interrupt-controller@28000000").unwrap());
    }

    #[test]
    fn aplic_imsic_guests_match_qemu() {
        let golden = include_bytes!("../../tests/data/virt-aplic-imsic-smp2-g3.dtb");
        let fdt = check_golden(golden, 2, VirtAia::AplicImsic, 3);
        let s = "/soc/interrupt-controller@28000000";
        assert_eq!(fdt.getprop_cell(s, "riscv,guest-index-bits").unwrap(), 2);
        // Two harts of four pages each.
        assert_eq!(fdt.getprop(s, "reg").unwrap(), be_cells(&[0, 0x2800_0000, 0, 0x8000]));
    }

    #[test]
    fn matches_qemu_byte_for_byte() {
        let fdt = build(&RiscvCfg::default(), 1, 128 << 20);
        assert_eq!(fdt.as_bytes().len(), QEMU_VIRT_DTB.len());
        assert!(fdt.as_bytes() == QEMU_VIRT_DTB, "the tree differs from QEMU's");
    }

    #[test]
    fn isa_extension_list() {
        let cfg = RiscvCfg::default();
        assert_eq!(cfg.isa_string(), QEMU_RV64_ISA);
        let ext = cfg.isa_extensions();
        assert_eq!(&ext[..8], ["i", "m", "a", "f", "d", "c", "h", "zic64b"]);
        assert_eq!(ext.last().unwrap(), "svvptc");
        assert_eq!(ext.len(), 49);
    }

    #[test]
    fn cpu_node_follows_the_model() {
        let cpu = RiscvCfg::model("sifive-u54");
        let fdt = build(&cpu, 1, 128 << 20);
        let get = |p: &str| fdt.getprop("/cpus/cpu@0", p);
        assert_eq!(get("mmu-type").unwrap(), b"riscv,sv39\0");
        assert_eq!(get("riscv,isa").unwrap(), b"rv64imafdc_zicntr_zicsr_zifencei_zihpm_sdtrig\0");
        assert!(get("riscv,cbom-block-size").is_err());
        let pmu = fdt.getprop("/pmu", "riscv,event-to-mhpmcounters").unwrap();
        assert_eq!(&pmu[8..12], &(0x0007_fff8u32 | 1).to_be_bytes());
    }

    #[test]
    fn two_harts_phandles() {
        let fdt = build(&RiscvCfg::default(), 2, 256 << 20);
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
