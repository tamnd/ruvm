// SPDX-License-Identifier: GPL-2.0-or-later

//! ACPI PCI hotplug AML from hw/acpi/pcihp.c: the register block, the eject and acpi-index
//! helpers and the per slot device objects.

use super::aml::{self, AccessType, Aml, LockRule, RegionSpace, Serialize, UpdateRule};
use super::pci;

/// `ACPI_PCIHP_SEJ_BASE`.
pub const SEJ_BASE: u64 = 0x8;
/// `ACPI_PCIHP_BNMR_BASE`.
pub const BNMR_BASE: u64 = 0x10;
/// `ACPI_PCIHP_SIZE`.
pub const SIZE: u16 = 0x18;
/// `ACPI_PCIHP_ADDR_ICH9`.
pub const ADDR_ICH9: u16 = 0x0cc0;

fn dword_field(name: &str) -> Aml {
    aml::field(name, AccessType::Dword, LockRule::NoLock, UpdateRule::WriteAsZeros)
}

/// `aml_pci_pdsm()`, the `_DSM` handler every hotplug slot forwards to.
fn pdsm() -> Aml {
    let ret = aml::local(0);
    let caps = aml::local(1);
    let acpi_index = aml::local(2);
    let zero = aml::int(0);
    let one = aml::int(1);
    let not_supp = aml::int(0xFFFF_FFFF);
    let func = aml::arg(2);
    let params = aml::arg(4);
    let bnum = aml::derefof(&aml::index(&params, &aml::int(0)));
    let sunum = aml::derefof(&aml::index(&params, &aml::int(1)));
    let mut method = aml::method("PDSM", 5, Serialize::Serialized);

    // Supported functions.
    let mut if_ctx = aml::if_(&aml::equal(&func, &zero));
    pci::dsm_func0_common(&mut if_ctx, &ret);
    if_ctx.append(&aml::store(&zero, &caps));
    if_ctx.append(&aml::store(&aml::call("AIDX", &[&bnum, &sunum]), &acpi_index));
    // Function 7 only when the device has an acpi-index. 0 means none and 0xFFFFFFFF means an
    // old QEMU without the PIDX register.
    let mut if_ctx1 = aml::if_(&aml::lnot(&aml::or(
        &aml::equal(&acpi_index, &zero),
        &aml::equal(&acpi_index, &not_supp),
        None,
    )));
    if_ctx1.append(&aml::or(&caps, &one, Some(&caps)));
    if_ctx1.append(&aml::or(&caps, &aml::shiftleft(&one, &aml::int(7)), Some(&caps)));
    if_ctx.append(&if_ctx1);
    if_ctx.append(&aml::store(&caps, &aml::index(&ret, &zero)));
    if_ctx.append(&aml::return_(&ret));
    method.append(&if_ctx);

    // Function 7. Windows calls it without checking function 0, so it always gets a two element
    // package, left uninitialized when there is no acpi-index so Windows ignores it.
    let mut if_ctx = aml::if_(&aml::equal(&func, &aml::int(7)));
    if_ctx.append(&aml::store(&aml::call("AIDX", &[&bnum, &sunum]), &acpi_index));
    if_ctx.append(&aml::store(&aml::package(2), &ret));
    let mut if_ctx1 = aml::if_(&aml::lnot(&aml::lor(
        &aml::equal(&acpi_index, &zero),
        &aml::equal(&acpi_index, &not_supp),
    )));
    if_ctx1.append(&aml::store(&acpi_index, &aml::index(&ret, &zero)));
    if_ctx1.append(&aml::store(&aml::string(""), &aml::index(&ret, &one)));
    if_ctx.append(&if_ctx1);
    if_ctx.append(&aml::return_(&ret));
    method.append(&if_ctx);
    method
}

/// `build_acpi_pci_hotplug()`.
pub fn build_hotplug(table: &mut Aml, rs: RegionSpace, addr: u64) {
    let mut scope = aml::scope("_SB.PCI0");

    scope.append(&aml::operation_region("PCST", rs, &aml::int(addr), 0x08));
    let mut field = dword_field("PCST");
    field.append(&aml::named_field("PCIU", 32));
    field.append(&aml::named_field("PCID", 32));
    scope.append(&field);

    scope.append(&aml::operation_region("SEJ", rs, &aml::int(addr + SEJ_BASE), 0x04));
    let mut field = dword_field("SEJ");
    field.append(&aml::named_field("B0EJ", 32));
    scope.append(&field);

    scope.append(&aml::operation_region("BNMR", rs, &aml::int(addr + BNMR_BASE), 0x08));
    let mut field = dword_field("BNMR");
    field.append(&aml::named_field("BNUM", 32));
    field.append(&aml::named_field("PIDX", 32));
    scope.append(&field);

    scope.append(&aml::mutex("BLCK", 0));

    let blck = aml::name("BLCK");
    let mut method = aml::method("PCEJ", 2, Serialize::NotSerialized);
    method.append(&aml::acquire(&blck, 0xFFFF));
    method.append(&aml::store(&aml::arg(0), &aml::name("BNUM")));
    method.append(&aml::store(&aml::shiftleft(&aml::int(1), &aml::arg(1)), &aml::name("B0EJ")));
    method.append(&aml::release(&blck));
    method.append(&aml::return_(&aml::int(0)));
    scope.append(&method);

    let mut method = aml::method("AIDX", 2, Serialize::NotSerialized);
    method.append(&aml::acquire(&blck, 0xFFFF));
    method.append(&aml::store(&aml::arg(0), &aml::name("BNUM")));
    method.append(&aml::store(&aml::shiftleft(&aml::int(1), &aml::arg(1)), &aml::name("PIDX")));
    method.append(&aml::store(&aml::name("PIDX"), &aml::local(0)));
    method.append(&aml::release(&blck));
    method.append(&aml::return_(&aml::local(0)));
    scope.append(&method);

    scope.append(&pdsm());
    table.append(&scope);
}

/// `build_append_pcihp_resources()`, reserving the register block under `\_SB.PCI0`.
pub fn build_resources(scope: &mut Aml, io_addr: u16, io_len: u16) {
    let mut dev = aml::device("PHPR");
    dev.append(&aml::name_decl("_HID", &aml::string("PNP0A06")));
    dev.append(&aml::name_decl("_UID", &aml::string("PCI Hotplug resources")));
    // Present, functioning, decoding, not shown in UI.
    dev.append(&aml::name_decl("_STA", &aml::int(0xB)));
    let mut crs = aml::resource_template();
    crs.append(&aml::io(aml::IoDecode::Decode16, io_addr, io_addr, 1, io_len as u8));
    dev.append(&aml::name_decl("_CRS", &crs));
    scope.append(&dev);
}

/// `aml_pci_static_endpoint_dsm()`, for a cold plugged device with an acpi-index.
pub fn static_endpoint_dsm(acpi_index: u32) -> Aml {
    let params = aml::local(0);
    let mut method = aml::method("_DSM", 4, Serialize::Serialized);
    let mut pkg = aml::package(1);
    pkg.append(&aml::int(acpi_index.into()));
    method.append(&aml::store(&pkg, &params));
    method.append(&aml::return_(&aml::call(
        "EDSM",
        &[&aml::arg(0), &aml::arg(1), &aml::arg(2), &aml::arg(3), &params],
    )));
    method
}
