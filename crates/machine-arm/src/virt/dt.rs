// SPDX-License-Identifier: GPL-2.0-or-later

//! The board part of the virt device tree, hw/arm/virt.c: `create_fdt()`, the timer, CPU and
//! GIC nodes, the devices, and the platform bus node `virt_machine_done()` adds. Each function
//! makes the same libfdt calls in the same order as its QEMU namesake, so the blob comes out
//! byte for byte the same.

use ruvm_target_arm::cpu::ArmCpuModel;

use super::{
    VIRT_FLASH, VIRT_FLASH_SIZE, VIRT_FW_CFG, VIRT_FW_CFG_SIZE, VIRT_GIC_DIST, VIRT_GIC_REDIST,
    VIRT_GIC_REDIST_SIZE, VIRT_MMIO, VIRT_MMIO_IRQ, VIRT_MMIO_SIZE, VIRT_PLATFORM_BUS,
    VIRT_PLATFORM_BUS_SIZE, VIRT_RTC, VIRT_RTC_IRQ, VIRT_RTC_SIZE, VIRT_SECURE_MEM,
    VIRT_SECURE_MEM_SIZE, VIRT_UART, VIRT_UART_IRQ, VIRT_UART_SIZE, VIRT_UART1, VIRT_UART1_IRQ,
    VIRTIO_TRANSPORTS,
};
use crate::fdt::{Fdt, sized_cells};
use ruvm_hw_intc::gicv3::GICV3_DIST_SIZE;

/// `GIC_FDT_IRQ_TYPE_SPI`.
const GIC_FDT_IRQ_TYPE_SPI: u32 = 0;
/// `GIC_FDT_IRQ_TYPE_PPI`.
const GIC_FDT_IRQ_TYPE_PPI: u32 = 1;
/// `GIC_FDT_IRQ_FLAGS_EDGE_LO_HI`.
const GIC_FDT_IRQ_FLAGS_EDGE_LO_HI: u32 = 1;
/// `GIC_FDT_IRQ_FLAGS_LEVEL_HI`.
const GIC_FDT_IRQ_FLAGS_LEVEL_HI: u32 = 4;
/// `ARCH_TIMER_S_EL1_IRQ`, `ARCH_TIMER_NS_EL1_IRQ`, `ARCH_TIMER_VIRT_IRQ` and
/// `ARCH_TIMER_NS_EL2_IRQ`, as PPI numbers.
const ARCH_TIMER_S_EL1_IRQ: u32 = 13;
const ARCH_TIMER_NS_EL1_IRQ: u32 = 14;
const ARCH_TIMER_VIRT_IRQ: u32 = 11;
const ARCH_TIMER_NS_EL2_IRQ: u32 = 10;
/// `ARCH_TIMER_NS_EL2_VIRT_IRQ` as a PPI number.
const ARCH_TIMER_NS_EL2_VIRT_IRQ: u32 = 12;
/// `ARCH_GIC_MAINT_IRQ` as a PPI number.
const ARCH_GIC_MAINT_IRQ: u32 = 9;
/// `ARM_AFF3_MASK`.
const ARM_AFF3_MASK: u64 = 0xff << 32;
/// `CLIDR_CTYPE_MAX_CACHE_LEVEL`, also the mask of one Ctype field.
const CLIDR_CTYPE_MAX_CACHE_LEVEL: u32 = 7;

fn err_sized(path: &str, name: &str) -> String {
    format!("qemu_fdt_setprop_sized_cells: Couldn't set {path}/{name}: FDT_ERR_BADVALUE")
}

/// `qemu_fdt_setprop_sized_cells()`.
pub(crate) fn setprop_sized_cells(
    fdt: &mut Fdt,
    path: &str,
    name: &str,
    values: &[(u32, u64)],
) -> Result<(), String> {
    let v = sized_cells(values).ok_or_else(|| err_sized(path, name))?;
    fdt.setprop(path, name, &v)
}

/// `create_fdt()`. Returns the clock phandle. `dtb-randomness` is off: no `kaslr-seed` and
/// no `rng-seed`. `secure` is `secure=on`, which adds `/secure-chosen`.
pub(crate) fn create_fdt(fdt: &mut Fdt, secure: bool) -> Result<u32, String> {
    fdt.setprop_string("/", "compatible", "linux,dummy-virt")?;
    fdt.setprop_cell("/", "#address-cells", 0x2)?;
    fdt.setprop_cell("/", "#size-cells", 0x2)?;
    fdt.setprop_string("/", "model", "linux,dummy-virt")?;
    // The virt board's devices are all cache coherent.
    fdt.setprop("/", "dma-coherent", &[])?;

    // /chosen must exist for load_dtb to fill in necessary properties later.
    fdt.add_subnode("/chosen")?;
    if secure {
        fdt.add_subnode("/secure-chosen")?;
    }
    // /aliases is filled in by the devices.
    fdt.add_subnode("/aliases")?;

    // Clock node, for the benefit of the UART. The kernel device tree binding documentation
    // claims the PL011 node clock properties are optional but in practice if you omit them
    // the kernel refuses to probe for the device.
    let clock = fdt.alloc_phandle();
    fdt.add_subnode("/apb-pclk")?;
    fdt.setprop_string("/apb-pclk", "compatible", "fixed-clock")?;
    fdt.setprop_cell("/apb-pclk", "#clock-cells", 0x0)?;
    fdt.setprop_cell("/apb-pclk", "clock-frequency", 24_000_000)?;
    fdt.setprop_string("/apb-pclk", "clock-output-names", "clk24mhz")?;
    fdt.setprop_cell("/apb-pclk", "phandle", clock)?;
    Ok(clock)
}

/// `fdt_add_timer_nodes()` for a GICv3. `ns_el2_virt_timer_irq` adds the EL2 virtual timer,
/// which a CPU with EL2 and FEAT_VHE has.
pub(crate) fn add_timer_nodes(fdt: &mut Fdt, ns_el2_virt_timer_irq: bool) -> Result<(), String> {
    let irqflags = GIC_FDT_IRQ_FLAGS_LEVEL_HI;
    fdt.add_subnode("/timer")?;
    fdt.setprop("/timer", "compatible", b"arm,armv8-timer\0arm,armv7-timer\0")?;
    fdt.setprop("/timer", "always-on", &[])?;
    let mut ppis = vec![
        ARCH_TIMER_S_EL1_IRQ,
        ARCH_TIMER_NS_EL1_IRQ,
        ARCH_TIMER_VIRT_IRQ,
        ARCH_TIMER_NS_EL2_IRQ,
    ];
    if ns_el2_virt_timer_irq {
        ppis.push(ARCH_TIMER_NS_EL2_VIRT_IRQ);
    }
    let cells: Vec<u32> =
        ppis.iter().flat_map(|&ppi| [GIC_FDT_IRQ_TYPE_PPI, ppi, irqflags]).collect();
    fdt.setprop_cells("/timer", "interrupts", &cells)
}

/// `CPUCoreCaches`.
#[derive(Clone, Copy, Debug)]
struct Cache {
    data: bool,
    level: u32,
    linesize: u32,
    sets: u32,
    size: u32,
}

/// `set_cpu_cache()` with the legacy CCSIDR layout.
fn cpu_cache(model: &ArmCpuModel, data: bool, level: u32, is_i_cache0: bool) -> Cache {
    let bank = (((level - 1) * 2) | u32::from(is_i_cache0)) as usize;
    let ccsidr = model.ccsidr[bank];
    let linesize = 1u32 << ((ccsidr & 7) + 4);
    let assoc = ((ccsidr >> 3) & 0x3ff) as u32 + 1;
    let sets = ((ccsidr >> 13) & 0x7fff) as u32 + 1;
    Cache { data, level, linesize, sets, size: assoc.wrapping_mul(sets).wrapping_mul(linesize) }
}

/// `virt_get_caches()`: the caches CLIDR describes, D before I at each level. `None` marks a
/// unified cache.
fn virt_get_caches(model: &ArmCpuModel) -> Result<Vec<Option<Cache>>, String> {
    let clidr = model.clidr;
    let mut caches = Vec::new();
    for level in 1..=CLIDR_CTYPE_MAX_CACHE_LEVEL {
        let ctype = (clidr >> (3 * (level - 1))) as u32 & CLIDR_CTYPE_MAX_CACHE_LEVEL;
        match ctype {
            0 => break,
            3 => {
                caches.push(Some(cpu_cache(model, true, level, false)));
                caches.push(Some(cpu_cache(model, false, level, true)));
            }
            4 => {
                let mut c = cpu_cache(model, true, level, false);
                c.data = false;
                if level == 1 {
                    caches.push(None);
                } else {
                    caches.push(Some(c));
                }
            }
            2 => caches.push(Some(cpu_cache(model, true, level, false))),
            1 => caches.push(Some(cpu_cache(model, false, level, true))),
            _ => return Err("Unrecognized cache type".to_string()),
        }
    }
    Ok(caches)
}

/// `fdt_add_cpu_nodes()` for the default topology (one socket, one cluster, a core per CPU,
/// no `smp-cache`), which describes only the L1 caches.
pub(crate) fn add_cpu_nodes(
    fdt: &mut Fdt,
    model: &ArmCpuModel,
    mpidrs: &[u64],
    psci: bool,
) -> Result<(), String> {
    let smp_cpus = mpidrs.len();
    let caches = virt_get_caches(model)?;
    let addr_cells = if mpidrs.iter().any(|m| m & ARM_AFF3_MASK != 0) { 2 } else { 1 };

    fdt.add_subnode("/cpus")?;
    fdt.setprop_cell("/cpus", "#address-cells", addr_cells)?;
    fdt.setprop_cell("/cpus", "#size-cells", 0x0)?;

    let mut phandles = vec![0u32; smp_cpus];
    for cpu in (0..smp_cpus).rev() {
        let nodename = format!("/cpus/cpu@{cpu}");
        fdt.add_subnode(&nodename)?;
        fdt.setprop_string(&nodename, "device_type", "cpu")?;
        fdt.setprop_string(&nodename, "compatible", model.dtb_compatible)?;
        if psci && smp_cpus > 1 {
            fdt.setprop_string(&nodename, "enable-method", "psci")?;
        }
        if addr_cells == 2 {
            fdt.setprop_u64(&nodename, "reg", mpidrs[cpu])?;
        } else {
            fdt.setprop_cell(&nodename, "reg", mpidrs[cpu] as u32)?;
        }
        let phandle = fdt.alloc_phandle();
        fdt.setprop_cell(&nodename, "phandle", phandle)?;
        phandles[cpu] = phandle;
        for c in &caches {
            match c {
                // Only level 1 in the CPU entry.
                Some(c) if c.level > 1 => {}
                Some(c) => {
                    let prefix = if c.data { "d-cache" } else { "i-cache" };
                    fdt.setprop_cell(&nodename, &format!("{prefix}-block-size"), c.linesize)?;
                    fdt.setprop_cell(&nodename, &format!("{prefix}-size"), c.size)?;
                    fdt.setprop_cell(&nodename, &format!("{prefix}-sets"), c.sets)?;
                }
                None => return Err("Unified type is not implemented at level 1".to_string()),
            }
        }
        fdt.setprop_cell(&nodename, "next-level-cache", 0)?;
    }

    fdt.add_subnode("/cpus/cpu-map")?;
    for cpu in (0..smp_cpus).rev() {
        let map_path = format!("/cpus/cpu-map/socket0/cluster0/core{cpu}");
        fdt.add_path(&map_path)?;
        fdt.setprop_cell(&map_path, "cpu", phandles[cpu])?;
    }
    Ok(())
}

/// `virt_flash_fdt()`. Without a separate secure address space (`secure=off`) both flashes
/// are one node; with `secure=on` the first is marked as for the secure world only.
pub(crate) fn virt_flash_fdt(fdt: &mut Fdt, secure: bool) -> Result<(), String> {
    let flashsize = VIRT_FLASH_SIZE / 2;
    let flashbase = VIRT_FLASH;
    if !secure {
        // Report both flash devices as a single node in the DT.
        let nodename = format!("/flash@{flashbase:x}");
        fdt.add_subnode(&nodename)?;
        fdt.setprop_string(&nodename, "compatible", "cfi-flash")?;
        setprop_sized_cells(
            fdt,
            &nodename,
            "reg",
            &[(2, flashbase), (2, flashsize), (2, flashbase + flashsize), (2, flashsize)],
        )?;
        fdt.setprop_cell(&nodename, "bank-width", 4)
    } else {
        // Report the devices as separate nodes so we can mark one as only visible to the
        // secure world.
        let nodename = format!("/secflash@{flashbase:x}");
        fdt.add_subnode(&nodename)?;
        fdt.setprop_string(&nodename, "compatible", "cfi-flash")?;
        setprop_sized_cells(fdt, &nodename, "reg", &[(2, flashbase), (2, flashsize)])?;
        fdt.setprop_cell(&nodename, "bank-width", 4)?;
        fdt.setprop_string(&nodename, "status", "disabled")?;
        fdt.setprop_string(&nodename, "secure-status", "okay")?;

        let nodename = format!("/flash@{:x}", flashbase + flashsize);
        fdt.add_subnode(&nodename)?;
        fdt.setprop_string(&nodename, "compatible", "cfi-flash")?;
        setprop_sized_cells(fdt, &nodename, "reg", &[(2, flashbase + flashsize), (2, flashsize)])?;
        fdt.setprop_cell(&nodename, "bank-width", 4)
    }
}

/// `fdt_add_gic_node()` for a GICv3 with one redistributor region and no ITS. `virt` is
/// `virtualization=on`, which describes the maintenance interrupt. Returns the GIC phandle.
pub(crate) fn add_gic_node(fdt: &mut Fdt, virt: bool) -> Result<u32, String> {
    let gic = fdt.alloc_phandle();
    fdt.setprop_cell("/", "interrupt-parent", gic)?;

    let nodename = format!("/intc@{VIRT_GIC_DIST:x}");
    fdt.add_subnode(&nodename)?;
    fdt.setprop_cell(&nodename, "#interrupt-cells", 3)?;
    fdt.setprop(&nodename, "interrupt-controller", &[])?;
    fdt.setprop_cell(&nodename, "#address-cells", 0x2)?;
    fdt.setprop_cell(&nodename, "#size-cells", 0x2)?;
    fdt.setprop(&nodename, "ranges", &[])?;
    fdt.setprop_string(&nodename, "compatible", "arm,gic-v3")?;
    fdt.setprop_cell(&nodename, "#redistributor-regions", 1)?;
    setprop_sized_cells(
        fdt,
        &nodename,
        "reg",
        &[
            (2, VIRT_GIC_DIST),
            (2, GICV3_DIST_SIZE),
            (2, VIRT_GIC_REDIST),
            (2, VIRT_GIC_REDIST_SIZE),
        ],
    )?;
    if virt {
        fdt.setprop_cells(
            &nodename,
            "interrupts",
            &[GIC_FDT_IRQ_TYPE_PPI, ARCH_GIC_MAINT_IRQ, GIC_FDT_IRQ_FLAGS_LEVEL_HI],
        )?;
    }
    fdt.setprop_cell(&nodename, "phandle", gic)?;
    Ok(gic)
}

/// Which UART [`create_uart`] describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Uart {
    /// `VIRT_UART0`, the console.
    Uart0,
    /// `VIRT_UART1`, for the normal world (`-serial` given twice without `secure=on`).
    Uart1,
    /// `VIRT_UART1` for the secure world only, with `secure=on`.
    SecureUart1,
}

/// The FDT part of `create_uart()`.
pub(crate) fn create_uart(fdt: &mut Fdt, clock: u32, uart: Uart) -> Result<(), String> {
    let (base, size, irq) = match uart {
        Uart::Uart0 => (VIRT_UART, VIRT_UART_SIZE, VIRT_UART_IRQ),
        Uart::Uart1 | Uart::SecureUart1 => (VIRT_UART1, VIRT_UART_SIZE, VIRT_UART1_IRQ),
    };
    let nodename = format!("/pl011@{base:x}");
    fdt.add_subnode(&nodename)?;
    // Note that we can't use setprop_string because of the embedded NUL.
    fdt.setprop(&nodename, "compatible", b"arm,pl011\0arm,primecell\0")?;
    setprop_sized_cells(fdt, &nodename, "reg", &[(2, base), (2, size)])?;
    fdt.setprop_cells(
        &nodename,
        "interrupts",
        &[GIC_FDT_IRQ_TYPE_SPI, irq, GIC_FDT_IRQ_FLAGS_LEVEL_HI],
    )?;
    fdt.setprop_cells(&nodename, "clocks", &[clock, clock])?;
    fdt.setprop(&nodename, "clock-names", b"uartclk\0apb_pclk\0")?;
    if uart == Uart::Uart0 {
        fdt.setprop_string("/chosen", "stdout-path", &nodename)?;
        fdt.setprop_string("/aliases", "serial0", &nodename)?;
    } else {
        fdt.setprop_string("/aliases", "serial1", &nodename)?;
    }
    if uart == Uart::SecureUart1 {
        // Mark as not usable by the normal world.
        fdt.setprop_string(&nodename, "status", "disabled")?;
        fdt.setprop_string(&nodename, "secure-status", "okay")?;
        fdt.setprop_string("/secure-chosen", "stdout-path", &nodename)?;
    }
    Ok(())
}

/// The FDT part of `create_secure_ram()`.
pub(crate) fn create_secure_ram(fdt: &mut Fdt) -> Result<(), String> {
    let nodename = format!("/secram@{VIRT_SECURE_MEM:x}");
    fdt.add_subnode(&nodename)?;
    fdt.setprop_string(&nodename, "device_type", "memory")?;
    setprop_sized_cells(fdt, &nodename, "reg", &[(2, VIRT_SECURE_MEM), (2, VIRT_SECURE_MEM_SIZE)])?;
    fdt.setprop_string(&nodename, "status", "disabled")?;
    fdt.setprop_string(&nodename, "secure-status", "okay")
}

/// The FDT part of `create_rtc()`.
pub(crate) fn create_rtc(fdt: &mut Fdt, clock: u32) -> Result<(), String> {
    let nodename = format!("/pl031@{VIRT_RTC:x}");
    fdt.add_subnode(&nodename)?;
    fdt.setprop(&nodename, "compatible", b"arm,pl031\0arm,primecell\0")?;
    setprop_sized_cells(fdt, &nodename, "reg", &[(2, VIRT_RTC), (2, VIRT_RTC_SIZE)])?;
    fdt.setprop_cells(
        &nodename,
        "interrupts",
        &[GIC_FDT_IRQ_TYPE_SPI, VIRT_RTC_IRQ, GIC_FDT_IRQ_FLAGS_LEVEL_HI],
    )?;
    fdt.setprop_cell(&nodename, "clocks", clock)?;
    fdt.setprop_string(&nodename, "clock-names", "apb_pclk")
}

/// The FDT part of `create_virtio_devices()`. The nodes go in from the highest address down
/// so that the guest kernel, which walks them in reverse, probes the lowest transport first.
pub(crate) fn add_virtio_nodes(fdt: &mut Fdt) -> Result<(), String> {
    for i in (0..VIRTIO_TRANSPORTS).rev() {
        let irq = VIRT_MMIO_IRQ + i as u32;
        let base = VIRT_MMIO + i as u64 * VIRT_MMIO_SIZE;
        let nodename = format!("/virtio_mmio@{base:x}");
        fdt.add_subnode(&nodename)?;
        fdt.setprop_string(&nodename, "compatible", "virtio,mmio")?;
        setprop_sized_cells(fdt, &nodename, "reg", &[(2, base), (2, VIRT_MMIO_SIZE)])?;
        fdt.setprop_cells(
            &nodename,
            "interrupts",
            &[GIC_FDT_IRQ_TYPE_SPI, irq, GIC_FDT_IRQ_FLAGS_EDGE_LO_HI],
        )?;
        fdt.setprop(&nodename, "dma-coherent", &[])?;
    }
    Ok(())
}

/// The FDT part of `create_fw_cfg()`.
pub(crate) fn add_fw_cfg_node(fdt: &mut Fdt) -> Result<(), String> {
    let nodename = format!("/fw-cfg@{VIRT_FW_CFG:x}");
    fdt.add_subnode(&nodename)?;
    fdt.setprop_string(&nodename, "compatible", "qemu,fw-cfg-mmio")?;
    setprop_sized_cells(fdt, &nodename, "reg", &[(2, VIRT_FW_CFG), (2, VIRT_FW_CFG_SIZE)])?;
    fdt.setprop(&nodename, "dma-coherent", &[])
}

/// `platform_bus_add_all_fdt_nodes()` with no dynamic sysbus devices: just the bus node.
pub(crate) fn add_platform_bus_node(fdt: &mut Fdt) -> Result<(), String> {
    let addr = VIRT_PLATFORM_BUS;
    let size = VIRT_PLATFORM_BUS_SIZE;
    let node = format!("/platform-bus@{addr:x}");
    fdt.add_subnode(&node)?;
    fdt.setprop(&node, "compatible", b"qemu,platform\0simple-bus\0")?;
    // Platform bus devices use a 32 bit address space and the root node uses 2 cells.
    fdt.setprop_cell(&node, "#size-cells", 1)?;
    fdt.setprop_cell(&node, "#address-cells", 1)?;
    fdt.setprop_cells(&node, "ranges", &[0, (addr >> 32) as u32, addr as u32, size as u32])?;
    let intc = fdt.get_phandle("/intc")?;
    fdt.setprop_cell(&node, "interrupt-parent", intc)
}
