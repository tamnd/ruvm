// SPDX-License-Identifier: GPL-2.0-or-later

//! The TPM parts of the tables: the TPM2 and TCPA tables, the TPM device in the DSDT and the
//! Physical Presence Interface `_DSM`, from hw/acpi/tpm.c, hw/acpi/aml-build.c and the TPM
//! devices' `build_dev_aml` hooks.

use super::aml::{
    self, AccessType, Aml, LockRule, ReadWrite, RegionSpace, Serialize, UpdateRule,
    append_int_noprefix,
};
use super::linker::BiosLinker;
use super::table::{AcpiTable, TABLE_FILE, TPMLOG_FILE};

/// `TPM_TIS_ADDR_BASE`.
pub const TIS_ADDR_BASE: u64 = 0xfed4_0000;
/// `TPM_TIS_ADDR_SIZE`.
pub const TIS_ADDR_SIZE: u64 = 0x5000;
/// `TPM_CRB_ADDR_BASE`.
pub const CRB_ADDR_BASE: u64 = 0xfed4_0000;
/// `TPM_CRB_ADDR_SIZE`.
pub const CRB_ADDR_SIZE: u64 = 0x1000;
/// `TPM_CRB_ADDR_CTRL`, the CRB_CTRL_REQ register.
pub const CRB_ADDR_CTRL: u64 = CRB_ADDR_BASE + 0x40;
/// `TPM_PPI_ADDR_BASE`.
pub const PPI_ADDR_BASE: u64 = 0xfed4_5000;
/// `TPM_PPI_ADDR_SIZE`.
pub const PPI_ADDR_SIZE: u64 = 0x400;
/// `TPM_LOG_AREA_MINIMUM_SIZE`.
pub const LOG_AREA_MINIMUM_SIZE: usize = 64 << 10;

const TCPA_ACPI_CLASS_CLIENT: u64 = 0;
const TPM2_ACPI_CLASS_CLIENT: u64 = 0;
const TPM2_START_METHOD_MMIO: u64 = 6;
const TPM2_START_METHOD_CRB: u64 = 7;

const PPI_FUNC_NOT_IMPLEMENTED: u64 = 0;
const PPI_FUNC_BLOCKED: u64 = 2;
const PPI_FUNC_MASK: u64 = 7;

/// `TPMVersion`, without the unspecified value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TpmVersion {
    V1_2,
    V2_0,
}

/// The TPM front ends the tables tell apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TpmInterface {
    /// tpm-tis on the ISA bus.
    TisIsa,
    /// tpm-tis-device on the platform bus.
    TisSysbus,
    /// tpm-crb.
    Crb,
}

/// What the tables need to know about the machine's TPM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpmInfo {
    pub version: TpmVersion,
    pub interface: TpmInterface,
    /// The device's `ppi` property.
    pub ppi: bool,
}

/// Allocates the TPM event log and points the table's Log Area Start Address at it.
fn log_area(tbl: &mut Vec<u8>, linker: &mut BiosLinker, tcpalog: &mut Vec<u8>) {
    tcpalog.resize(tcpalog.len() + LOG_AREA_MINIMUM_SIZE, 0);
    linker.alloc(TPMLOG_FILE, 1, false);
    let log_addr_offset = tbl.len() as u32;
    append_int_noprefix(tbl, 0, 8);
    linker.add_pointer(TABLE_FILE, tbl, log_addr_offset, 8, TPMLOG_FILE, 0);
}

/// `build_tpm_tcpa()`, the table for a TPM 1.2.
pub fn build_tcpa(
    tbl: &mut Vec<u8>,
    linker: &mut BiosLinker,
    tcpalog: &mut Vec<u8>,
    oem_id: &str,
    oem_table_id: &str,
) {
    let table = AcpiTable::begin("TCPA", 2, oem_id, oem_table_id, tbl);
    append_int_noprefix(tbl, TCPA_ACPI_CLASS_CLIENT, 2); // Platform Class
    append_int_noprefix(tbl, LOG_AREA_MINIMUM_SIZE as u64, 4); // Log Area Minimum Length
    log_area(tbl, linker, tcpalog);
    table.end(Some(linker), tbl);
}

/// `build_tpm2()`, the table for a TPM 2.0.
pub fn build_tpm2(
    tbl: &mut Vec<u8>,
    linker: &mut BiosLinker,
    tcpalog: &mut Vec<u8>,
    interface: TpmInterface,
    oem_id: &str,
    oem_table_id: &str,
) {
    let table = AcpiTable::begin("TPM2", 4, oem_id, oem_table_id, tbl);
    append_int_noprefix(tbl, TPM2_ACPI_CLASS_CLIENT, 2); // Platform Class
    append_int_noprefix(tbl, 0, 2); // Reserved
    let (control_area, start_method) = match interface {
        TpmInterface::TisIsa | TpmInterface::TisSysbus => (0, TPM2_START_METHOD_MMIO),
        TpmInterface::Crb => (CRB_ADDR_CTRL, TPM2_START_METHOD_CRB),
    };
    append_int_noprefix(tbl, control_area, 8); // Address of Control Area
    append_int_noprefix(tbl, start_method, 4); // Start Method
    tbl.extend_from_slice(&[0; 12]); // Platform Specific Parameters
    append_int_noprefix(tbl, LOG_AREA_MINIMUM_SIZE as u64, 4); // Log Area Minimum Length
    log_area(tbl, linker, tcpalog);
    table.end(Some(linker), tbl);
}

/// `build_tpm_tis_isa_aml()`, the tpm-tis device in the LPC scope.
pub fn tis_isa_aml(scope: &mut Aml, tpm: &TpmInfo) {
    let mut dev = aml::device("TPM");
    if tpm.version == TpmVersion::V2_0 {
        dev.append(&aml::name_decl("_HID", &aml::string("MSFT0101")));
        dev.append(&aml::name_decl("_STR", &aml::string("TPM 2.0 Device")));
    } else {
        dev.append(&aml::name_decl("_HID", &aml::eisaid("PNP0C31")));
    }
    dev.append(&aml::name_decl("_UID", &aml::int(1)));
    dev.append(&aml::name_decl("_STA", &aml::int(0xf)));
    let mut crs = aml::resource_template();
    crs.append(&aml::memory32_fixed(
        TIS_ADDR_BASE as u32,
        TIS_ADDR_SIZE as u32,
        ReadWrite::ReadWrite,
    ));
    // QEMU leaves the IRQ out because IRQ 5 clashes with the PCI link devices.
    dev.append(&aml::name_decl("_CRS", &crs));
    if tpm.ppi {
        build_ppi_acpi(&mut dev, PPI_ADDR_BASE);
    }
    scope.append(&dev);
}

/// The tpm-crb device that `build_dsdt()` puts in `\_SB`.
pub fn crb_aml(scope: &mut Aml, tpm: &TpmInfo) {
    let mut dev = aml::device("TPM");
    dev.append(&aml::name_decl("_HID", &aml::string("MSFT0101")));
    dev.append(&aml::name_decl("_STR", &aml::string("TPM 2.0 Device")));
    let mut crs = aml::resource_template();
    crs.append(&aml::memory32_fixed(
        CRB_ADDR_BASE as u32,
        CRB_ADDR_SIZE as u32,
        ReadWrite::ReadWrite,
    ));
    dev.append(&aml::name_decl("_CRS", &crs));
    dev.append(&aml::name_decl("_STA", &aml::int(0xf)));
    dev.append(&aml::name_decl("_UID", &aml::int(1)));
    if tpm.ppi {
        build_ppi_acpi(&mut dev, PPI_ADDR_BASE);
    }
    scope.append(&dev);
}

/// `acpi_dsdt_add_tpm()` of Arm virt: the tpm-tis-device on the platform bus, its registers
/// at `base` and its PPI memory, when it has the `ppi` property on, at `ppi_base`.
pub fn tis_sysbus_aml(scope: &mut Aml, base: u64, ppi_base: Option<u64>) {
    let mut dev = aml::device("TPM0");
    dev.append(&aml::name_decl("_HID", &aml::string("MSFT0101")));
    dev.append(&aml::name_decl("_STR", &aml::string("TPM 2.0 Device")));
    dev.append(&aml::name_decl("_UID", &aml::int(1)));
    dev.append(&aml::name_decl("_STA", &aml::int(0xf)));
    let mut crs = aml::resource_template();
    crs.append(&aml::memory32_fixed(base as u32, TIS_ADDR_SIZE as u32, ReadWrite::ReadWrite));
    dev.append(&aml::name_decl("_CRS", &crs));
    if let Some(ppi_base) = ppi_base {
        build_ppi_acpi(&mut dev, ppi_base);
    }
    scope.append(&dev);
}

/// `tpm_build_ppi_acpi()`: the PPI registers in the PPI memory and the `_DSM` for the
/// Physical Presence Interface 1.3 and the TCG Platform Reset Attack Mitigation.
pub fn build_ppi_acpi(dev: &mut Aml, ppi_base: u64) {
    let zero = aml::int(0);
    let one = aml::int(1);
    let func_mask = aml::int(PPI_FUNC_MASK);
    let not_implemented = aml::int(PPI_FUNC_NOT_IMPLEMENTED);

    // TPP2 holds the registers ACPI uses to pass the PPI code and parameter (PPRQ, PPRM) to
    // the firmware.
    dev.append(&aml::operation_region(
        "TPP2",
        RegionSpace::SystemMemory,
        &aml::int(ppi_base + 0x100),
        0x5a,
    ));
    let mut field = aml::field("TPP2", AccessType::Any, LockRule::NoLock, UpdateRule::Preserve);
    field.append(&aml::named_field("PPIN", 8));
    for name in ["PPIP", "PPRP", "PPRQ", "PPRM", "LPPR"] {
        field.append(&aml::named_field(name, 32));
    }
    dev.append(&field);
    let pprq = aml::name("PPRQ");
    let pprm = aml::name("PPRM");

    // MOVV, see docs/specs/tpm.rst.
    dev.append(&aml::operation_region(
        "TPP3",
        RegionSpace::SystemMemory,
        &aml::int(ppi_base + 0x15a),
        0x1,
    ));
    let mut field = aml::field("TPP3", AccessType::Byte, LockRule::NoLock, UpdateRule::Preserve);
    field.append(&aml::named_field("MOVV", 8));
    dev.append(&field);

    // DerefOf in Windows is broken with SystemMemory, so FUNC[op] comes from a dynamic
    // operation region inside a method.
    let mut method = aml::method("TPFN", 1, Serialize::Serialized);
    {
        let op = aml::arg(0);
        let mut ifctx = aml::if_(&aml::lgreater_equal(&op, &aml::int(0x100)));
        ifctx.append(&aml::return_(&zero));
        method.append(&ifctx);
        method.append(&aml::operation_region(
            "TPP1",
            RegionSpace::SystemMemory,
            &aml::add(&aml::int(ppi_base), &op, None),
            0x1,
        ));
        let mut field =
            aml::field("TPP1", AccessType::Byte, LockRule::NoLock, UpdateRule::Preserve);
        field.append(&aml::named_field("TPPF", 8));
        method.append(&field);
        method.append(&aml::return_(&aml::name("TPPF")));
    }
    dev.append(&method);

    // Global TPM2 and TPM3 packages work around a Windows ACPI bug with returning packages.
    let mut pak = aml::package(2);
    pak.append(&zero);
    pak.append(&zero);
    dev.append(&aml::name_decl("TPM2", &pak));
    let tpm2 = aml::name("TPM2");
    let mut pak = aml::package(3);
    pak.append(&zero);
    pak.append(&zero);
    pak.append(&zero);
    dev.append(&aml::name_decl("TPM3", &pak));
    let tpm3 = aml::name("TPM3");

    let mut method = aml::method("_DSM", 4, Serialize::Serialized);
    {
        let uuid = aml::arg(0);
        let rev = aml::arg(1);
        let function = aml::arg(2);
        let arguments = aml::arg(3);
        let op = aml::local(0);
        let op_flags = aml::local(1);
        let is = |n: u64| aml::equal(&function, &aml::int(n));
        // op = arguments[0], op_flags = TPFN(op)
        let get_op = |ctx: &mut Aml| {
            ctx.append(&aml::store(&aml::derefof(&aml::index(&arguments, &zero)), &op));
            ctx.append(&aml::store(&aml::call("TPFN", &[&op]), &op_flags));
        };
        let flags_are = |v: &Aml| aml::equal(&aml::and(&op_flags, &func_mask, None), v);

        // Physical Presence Interface
        let mut ifctx =
            aml::if_(&aml::equal(&uuid, &aml::touuid("3DDDFAA6-361B-4EB4-A424-8D10089D1653")));
        {
            // The standard query function: functions 1 to 8.
            let mut ifctx2 = aml::if_(&is(0));
            ifctx2.append(&aml::return_(&aml::buffer(2, Some(&[0xff, 0x01][..]))));
            ifctx.append(&ifctx2);

            // PPI 1.0: 2.1.1 Get Physical Presence Interface Version
            let mut ifctx2 = aml::if_(&is(1));
            ifctx2.append(&aml::return_(&aml::string("1.3")));
            ifctx.append(&ifctx2);

            // PPI 1.0: 2.1.3 Submit TPM Operation Request to Pre-OS Environment
            let mut ifctx2 = aml::if_(&is(2));
            {
                get_op(&mut ifctx2);
                let mut ifctx3 = aml::if_(&flags_are(&not_implemented));
                ifctx3.append(&aml::return_(&one)); // Not supported
                ifctx2.append(&ifctx3);
                ifctx2.append(&aml::store(&op, &pprq));
                ifctx2.append(&aml::store(&zero, &pprm));
                ifctx2.append(&aml::return_(&zero)); // Success
            }
            ifctx.append(&ifctx2);

            // PPI 1.0: 2.1.4 Get Pending TPM Operation Requested By the OS
            let mut ifctx2 = aml::if_(&is(3));
            {
                // Revision 1 has no integer after the second.
                let mut ifctx3 = aml::if_(&aml::equal(&rev, &one));
                ifctx3.append(&aml::store(&pprq, &aml::index(&tpm2, &one)));
                ifctx3.append(&aml::return_(&tpm2));
                ifctx2.append(&ifctx3);

                // {0, 23, 1} means operation 23 with argument 1 is pending.
                let mut ifctx3 = aml::if_(&aml::equal(&rev, &aml::int(2)));
                ifctx3.append(&aml::store(&pprq, &aml::index(&tpm3, &one)));
                ifctx3.append(&aml::store(&pprm, &aml::index(&tpm3, &aml::int(2))));
                ifctx3.append(&aml::return_(&tpm3));
                ifctx2.append(&ifctx3);
            }
            ifctx.append(&ifctx2);

            // PPI 1.0: 2.1.5 Get Platform-Specific Action to Transition to Pre-OS Environment
            let mut ifctx2 = aml::if_(&is(4));
            ifctx2.append(&aml::return_(&aml::int(2))); // Reboot
            ifctx.append(&ifctx2);

            // PPI 1.0: 2.1.6 Return TPM Operation Response to OS Environment
            let mut ifctx2 = aml::if_(&is(5));
            ifctx2.append(&aml::store(&aml::name("LPPR"), &aml::index(&tpm3, &one)));
            ifctx2.append(&aml::store(&aml::name("PPRP"), &aml::index(&tpm3, &aml::int(2))));
            ifctx2.append(&aml::return_(&tpm3));
            ifctx.append(&ifctx2);

            // PPI 1.0: 2.1.7 Submit preferred user language
            let mut ifctx2 = aml::if_(&is(6));
            ifctx2.append(&aml::return_(&aml::int(3))); // Not implemented
            ifctx.append(&ifctx2);

            // PPI 1.1: 2.1.7 Submit TPM Operation Request to Pre-OS Environment 2
            let mut ifctx2 = aml::if_(&is(7));
            {
                get_op(&mut ifctx2);
                let mut ifctx3 = aml::if_(&flags_are(&not_implemented));
                ifctx3.append(&aml::return_(&one)); // Not implemented
                ifctx2.append(&ifctx3);
                let mut ifctx3 = aml::if_(&flags_are(&aml::int(PPI_FUNC_BLOCKED)));
                ifctx3.append(&aml::return_(&aml::int(3))); // Blocked by firmware
                ifctx2.append(&ifctx3);

                let mut ifctx3 = aml::if_(&aml::equal(&rev, &one));
                ifctx3.append(&aml::store(&op, &pprq));
                ifctx3.append(&aml::store(&zero, &pprm));
                ifctx2.append(&ifctx3);

                let mut ifctx3 = aml::if_(&aml::equal(&rev, &aml::int(2)));
                let op_arg = aml::derefof(&aml::index(&arguments, &one));
                ifctx3.append(&aml::store(&op, &pprq));
                ifctx3.append(&aml::store(&op_arg, &pprm));
                ifctx2.append(&ifctx3);
                ifctx2.append(&aml::return_(&zero)); // Success
            }
            ifctx.append(&ifctx2);

            // PPI 1.1: 2.1.8 Get User Confirmation Status for Operation
            let mut ifctx2 = aml::if_(&is(8));
            get_op(&mut ifctx2);
            ifctx2.append(&aml::return_(&aml::and(&op_flags, &func_mask, None)));
            ifctx.append(&ifctx2);

            ifctx.append(&aml::return_(&aml::buffer(1, Some(&[0][..]))));
        }
        method.append(&ifctx);

        // TCG Platform Reset Attack Mitigation Specification 1.00, chapter 6.
        let mut ifctx =
            aml::if_(&aml::equal(&uuid, &aml::touuid("376054ED-CC13-4675-901C-4756D7F2D45D")));
        {
            // The standard query function: functions 1 and 2.
            let mut ifctx2 = aml::if_(&is(0));
            ifctx2.append(&aml::return_(&aml::buffer(1, Some(&[0x03][..]))));
            ifctx.append(&ifctx2);

            let mut ifctx2 = aml::if_(&is(1));
            ifctx2.append(&aml::store(&aml::derefof(&aml::index(&arguments, &zero)), &op));
            ifctx2.append(&aml::store(&op, &aml::name("MOVV")));
            ifctx2.append(&aml::return_(&zero)); // Success
            ifctx.append(&ifctx2);
        }
        method.append(&ifctx);
    }
    dev.append(&method);
}
