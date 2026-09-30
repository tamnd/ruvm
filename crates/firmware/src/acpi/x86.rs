// SPDX-License-Identifier: GPL-2.0-or-later

//! Tables shared by the x86 machines, from hw/i386/acpi-common.c.

use super::aml::append_int_noprefix;
use super::linker::BiosLinker;
use super::table::AcpiTable;

/// `APIC_DEFAULT_ADDRESS`, where every local APIC sits.
pub const APIC_DEFAULT_ADDRESS: u32 = 0xfee0_0000;
/// `IO_APIC_DEFAULT_ADDRESS`.
pub const IO_APIC_DEFAULT_ADDRESS: u32 = 0xfec0_0000;
/// `IO_APIC_SECONDARY_ADDRESS`.
pub const IO_APIC_SECONDARY_ADDRESS: u32 = IO_APIC_DEFAULT_ADDRESS + 0x10000;
/// `IO_APIC_SECONDARY_IRQBASE`. The first IOAPIC has GSIs 0 to 23 and the second 24 to 47.
pub const IO_APIC_SECONDARY_IRQBASE: u32 = 24;
/// `ACPI_BUILD_IOAPIC_ID`.
pub const ACPI_BUILD_IOAPIC_ID: u8 = 0x0;

/// One possible CPU, from `possible_cpu_arch_ids()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PossibleCpu {
    /// The APIC ID.
    pub arch_id: u32,
    /// Whether the CPU exists now. Hotpluggable CPUs that are not plugged are listed disabled.
    pub present: bool,
}

/// The parts of an `X86MachineState` the MADT depends on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MadtConfig {
    pub cpus: Vec<PossibleCpu>,
    /// `x86ms->pic != ON_OFF_AUTO_OFF`, reported as PCAT_COMPAT.
    pub pic: bool,
    pub ioapic2: bool,
    /// `X86MachineClass::apic_xrupt_override`, set for pc and q35 but not microvm.
    pub apic_xrupt_override: bool,
    /// `x86ms->pci_irq_mask`, ISA IRQs routed to PCI that get a level triggered override.
    pub pci_irq_mask: u16,
}

/// `pc_madt_cpu_entry()`.
pub fn madt_cpu_entry(uid: u32, cpu: &PossibleCpu, entry: &mut Vec<u8>, force_enabled: bool) {
    let flags = u64::from(cpu.present || force_enabled);
    if cpu.arch_id < 255 {
        entry.extend_from_slice(&[0, 8, uid as u8, cpu.arch_id as u8]); // Processor Local APIC
        append_int_noprefix(entry, flags, 4);
    } else {
        entry.extend_from_slice(&[9, 16]); // Processor Local x2APIC
        append_int_noprefix(entry, 0, 2);
        append_int_noprefix(entry, cpu.arch_id.into(), 4);
        append_int_noprefix(entry, flags, 4);
        append_int_noprefix(entry, uid.into(), 4);
    }
}

fn build_ioapic(entry: &mut Vec<u8>, id: u8, addr: u32, irq: u32) {
    entry.extend_from_slice(&[1, 12, id, 0]);
    append_int_noprefix(entry, addr.into(), 4);
    append_int_noprefix(entry, irq.into(), 4);
}

fn build_xrupt_override(entry: &mut Vec<u8>, src: u8, gsi: u32, flags: u16) {
    entry.extend_from_slice(&[2, 10, 0, src]);
    append_int_noprefix(entry, gsi.into(), 4);
    append_int_noprefix(entry, flags.into(), 2);
}

/// `acpi_build_madt()`.
pub fn build_madt(
    table_data: &mut Vec<u8>,
    linker: &mut BiosLinker,
    m: &MadtConfig,
    oem_id: &str,
    oem_table_id: &str,
) {
    let table = AcpiTable::begin("APIC", 3, oem_id, oem_table_id, table_data);
    append_int_noprefix(table_data, APIC_DEFAULT_ADDRESS.into(), 4);
    append_int_noprefix(table_data, u64::from(m.pic), 4);
    let mut x2apic_mode = false;
    for (i, cpu) in m.cpus.iter().enumerate() {
        madt_cpu_entry(i as u32, cpu, table_data, false);
        x2apic_mode |= cpu.arch_id > 254;
    }
    build_ioapic(table_data, ACPI_BUILD_IOAPIC_ID, IO_APIC_DEFAULT_ADDRESS, 0);
    if m.ioapic2 {
        build_ioapic(
            table_data,
            ACPI_BUILD_IOAPIC_ID + 1,
            IO_APIC_SECONDARY_ADDRESS,
            IO_APIC_SECONDARY_IRQBASE,
        );
    }
    if m.apic_xrupt_override {
        // Conforms to the specifications of the bus.
        build_xrupt_override(table_data, 0, 2, 0);
    }
    for i in 1..16u8 {
        if m.pci_irq_mask & (1 << i) != 0 {
            // Active high, level triggered.
            build_xrupt_override(table_data, i, i.into(), 0xd);
        }
    }
    if x2apic_mode {
        // Local x2APIC NMI for all processors on LINT1.
        table_data.extend_from_slice(&[0xA, 12, 0, 0]);
        append_int_noprefix(table_data, 0xFFFF_FFFF, 4);
        table_data.extend_from_slice(&[1, 0, 0, 0]);
    } else {
        // Local APIC NMI for all processors on LINT1.
        table_data.extend_from_slice(&[4, 6, 0xFF, 0, 0, 1]);
    }
    table.end(Some(linker), table_data);
}
