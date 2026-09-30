// SPDX-License-Identifier: GPL-2.0-or-later

//! The CPU hotplug AML from hw/acpi/cpu.c: the register block device, `\_SB.CPUS` with its
//! scan, status, eject and OST methods, and one processor object per possible CPU.

use super::aml::{self, AccessType, Aml, LockRule, RegionSpace, Serialize, UpdateRule};
use super::x86::{PossibleCpu, madt_cpu_entry};

/// `ACPI_CPU_HOTPLUG_REG_LEN`.
pub const REG_LEN: u8 = 12;
/// `ICH9_CPU_HOTPLUG_IO_BASE`.
pub const ICH9_IO_BASE: u16 = 0x0cd8;

const FLAGS_OFFSET_RW: usize = 4;
const OVMF_CPUHP_SMI_CMD: u64 = 4;
const CMD_GET_NEXT_CPU_WITH_EVENT: u64 = 0;
const CMD_OST_EVENT: u64 = 1;
const CMD_OST_STATUS: u64 = 2;

const RES_DEVICE: &str = "PRES";
const LOCK: &str = "CPLK";
const STS_METHOD: &str = "CSTA";
const SCAN_METHOD: &str = "CSCN";
const NOTIFY_METHOD: &str = "CTFY";
const EJECT_METHOD: &str = "CEJ0";
const OST_METHOD: &str = "COST";
const ADDED_LIST: &str = "CNEW";
const EJ_LIST: &str = "CEJL";
const ENABLED: &str = "CPEN";
const SELECTOR: &str = "CSEL";
const COMMAND: &str = "CCMD";
const DATA: &str = "CDAT";
const INSERT_EVENT: &str = "CINS";
const REMOVE_EVENT: &str = "CRMV";
const EJECT_EVENT: &str = "CEJ0";
const FW_EJECT_EVENT: &str = "CEJF";

fn cpu_name(i: usize) -> String {
    format!("C{i:03X}")
}

/// `CPUHotplugFeatures`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Features {
    /// Use Processor objects where the APIC ID allows, for ACPI 1.0 guests.
    pub acpi_1_compatible: bool,
    /// The SMI command field to poke when the firmware negotiated CPU hotplug over SMI.
    pub smi_path: Option<String>,
    /// The firmware does the unplug, so eject goes through `CEJF` and an SMI.
    pub fw_unplugs_cpu: bool,
}

/// `build_cpus_aml()` with `pc_madt_cpu_entry` for `_MAT`. `cpus` are the possible CPUs and the
/// first one is the boot CPU, which cannot be ejected.
#[allow(clippy::too_many_arguments)]
pub fn build_cpus(
    table: &mut Aml,
    cpus: &[PossibleCpu],
    opts: &Features,
    base_addr: u64,
    res_root: &str,
    event_handler_method: &str,
    rs: RegionSpace,
) {
    let zero = aml::int(0);
    let one = aml::int(1);
    let mut sb_scope = aml::scope("_SB");
    let res_path = format!("{res_root}.{RES_DEVICE}");

    let mut ctrl_dev = aml::device(&res_path);
    ctrl_dev.append(&aml::name_decl("_HID", &aml::eisaid("PNP0A06")));
    ctrl_dev.append(&aml::name_decl("_UID", &aml::string("CPU Hotplug resources")));
    ctrl_dev.append(&aml::mutex(LOCK, 0));
    let mut crs = aml::resource_template();
    match rs {
        RegionSpace::SystemIo => crs.append(&aml::io(
            aml::IoDecode::Decode16,
            base_addr as u16,
            base_addr as u16,
            1,
            REG_LEN,
        )),
        RegionSpace::SystemMemory => crs.append(&aml::memory32_fixed(
            base_addr as u32,
            REG_LEN.into(),
            aml::ReadWrite::ReadWrite,
        )),
        RegionSpace::PciConfig => panic!("CPU hotplug registers in PCI config space"),
    }
    ctrl_dev.append(&aml::name_decl("_CRS", &crs));
    ctrl_dev.append(&aml::operation_region("PRST", rs, &aml::int(base_addr), REG_LEN.into()));

    let mut field =
        aml::field("PRST", AccessType::Byte, LockRule::NoLock, UpdateRule::WriteAsZeros);
    field.append(&aml::reserved_field(FLAGS_OFFSET_RW * 8));
    field.append(&aml::named_field(ENABLED, 1));
    field.append(&aml::named_field(INSERT_EVENT, 1));
    field.append(&aml::named_field(REMOVE_EVENT, 1));
    field.append(&aml::named_field(EJECT_EVENT, 1));
    field.append(&aml::named_field(FW_EJECT_EVENT, 1));
    field.append(&aml::reserved_field(3));
    field.append(&aml::named_field(COMMAND, 8));
    ctrl_dev.append(&field);

    let mut field = aml::field("PRST", AccessType::Dword, LockRule::NoLock, UpdateRule::Preserve);
    field.append(&aml::named_field(SELECTOR, 32));
    field.append(&aml::reserved_field(4 * 8)); // flags, command and padding
    field.append(&aml::named_field(DATA, 32));
    ctrl_dev.append(&field);
    sb_scope.append(&ctrl_dev);

    let reg = |r: &str| aml::name(&format!("{res_path}.{r}"));
    let ctrl_lock = reg(LOCK);
    let cpu_selector = reg(SELECTOR);
    let is_enabled = reg(ENABLED);
    let cpu_cmd = reg(COMMAND);
    let cpu_data = reg(DATA);
    let ins_evt = reg(INSERT_EVENT);
    let rm_evt = reg(REMOVE_EVENT);
    let ej_evt = reg(EJECT_EVENT);
    let fw_ej_evt = reg(FW_EJECT_EVENT);

    let mut cpus_dev = aml::device("\\_SB.CPUS");
    cpus_dev.append(&aml::name_decl("_HID", &aml::string("ACPI0010")));
    cpus_dev.append(&aml::name_decl("_CID", &aml::eisaid("PNP0A05")));

    let mut method = aml::method(NOTIFY_METHOD, 2, Serialize::NotSerialized);
    for i in 0..cpus.len() {
        let mut if_ctx = aml::if_(&aml::equal(&aml::arg(0), &aml::int(i as u64)));
        if_ctx.append(&aml::notify(&aml::name(&cpu_name(i)), &aml::arg(1)));
        method.append(&if_ctx);
    }
    cpus_dev.append(&method);

    let mut method = aml::method(STS_METHOD, 1, Serialize::Serialized);
    let sta = aml::local(0);
    method.append(&aml::acquire(&ctrl_lock, 0xFFFF));
    method.append(&aml::store(&aml::arg(0), &cpu_selector));
    method.append(&aml::store(&zero, &sta));
    let mut if_ctx = aml::if_(&aml::equal(&is_enabled, &one));
    if_ctx.append(&aml::store(&aml::int(0xF), &sta));
    method.append(&if_ctx);
    method.append(&aml::release(&ctrl_lock));
    method.append(&aml::return_(&sta));
    cpus_dev.append(&method);

    let mut method = aml::method(EJECT_METHOD, 1, Serialize::Serialized);
    method.append(&aml::acquire(&ctrl_lock, 0xFFFF));
    method.append(&aml::store(&aml::arg(0), &cpu_selector));
    if opts.fw_unplugs_cpu {
        let smi_path = opts.smi_path.as_deref().expect("fw_unplugs_cpu needs an SMI path");
        method.append(&aml::store(&one, &fw_ej_evt));
        method.append(&aml::store(&aml::int(OVMF_CPUHP_SMI_CMD), &aml::name(smi_path)));
    } else {
        method.append(&aml::store(&one, &ej_evt));
    }
    method.append(&aml::release(&ctrl_lock));
    cpus_dev.append(&method);

    cpus_dev.append(&scan_method(
        cpus.len(),
        opts,
        &ctrl_lock,
        &cpu_selector,
        &cpu_cmd,
        &cpu_data,
        &ins_evt,
        &rm_evt,
    ));

    let mut method = aml::method(OST_METHOD, 4, Serialize::Serialized);
    method.append(&aml::acquire(&ctrl_lock, 0xFFFF));
    method.append(&aml::store(&aml::arg(0), &cpu_selector));
    method.append(&aml::store(&aml::int(CMD_OST_EVENT), &cpu_cmd));
    method.append(&aml::store(&aml::arg(1), &cpu_data));
    method.append(&aml::store(&aml::int(CMD_OST_STATUS), &cpu_cmd));
    method.append(&aml::store(&aml::arg(2), &cpu_data));
    method.append(&aml::release(&ctrl_lock));
    cpus_dev.append(&method);

    for (i, cpu) in cpus.iter().enumerate() {
        let uid = aml::int(i as u64);
        let mut dev = if opts.acpi_1_compatible && cpu.arch_id < 255 {
            aml::processor(i as u8, 0, 0, &cpu_name(i))
        } else {
            let mut dev = aml::device(&cpu_name(i));
            dev.append(&aml::name_decl("_HID", &aml::string("ACPI0007")));
            dev.append(&aml::name_decl("_UID", &uid));
            dev
        };

        let mut method = aml::method("_STA", 0, Serialize::Serialized);
        method.append(&aml::return_(&aml::call(STS_METHOD, &[&uid])));
        dev.append(&method);

        let mut mat = Vec::new();
        madt_cpu_entry(i as u32, cpu, &mut mat, true);
        dev.append(&aml::name_decl("_MAT", &aml::buffer(mat.len(), Some(&mat))));

        if i != 0 {
            let mut method = aml::method("_EJ0", 1, Serialize::NotSerialized);
            method.append(&aml::call(EJECT_METHOD, &[&uid]));
            dev.append(&method);
        }

        let mut method = aml::method("_OST", 3, Serialize::Serialized);
        method.append(&aml::call(OST_METHOD, &[&uid, &aml::arg(0), &aml::arg(1), &aml::arg(2)]));
        dev.append(&method);
        cpus_dev.append(&dev);
    }
    sb_scope.append(&cpus_dev);
    table.append(&sb_scope);

    let mut method = aml::method(event_handler_method, 0, Serialize::NotSerialized);
    method.append(&aml::call(&format!("\\_SB.CPUS.{SCAN_METHOD}"), &[]));
    table.append(&method);
}

/// The `CSCN` method. It collects CPUs with insert or remove events in batches of up to 255,
/// the most an ACPI 1.0 package holds, then notifies the OS about each one.
#[allow(clippy::too_many_arguments)]
fn scan_method(
    ncpus: usize,
    opts: &Features,
    ctrl_lock: &Aml,
    cpu_selector: &Aml,
    cpu_cmd: &Aml,
    cpu_data: &Aml,
    ins_evt: &Aml,
    rm_evt: &Aml,
) -> Aml {
    const MAX_CPUS_PER_PASS: u8 = 255;
    let zero = aml::int(0);
    let one = aml::int(1);
    let has_event = aml::local(0);
    let dev_chk = aml::int(1);
    let eject_req = aml::int(3);
    let next_cpu_cmd = aml::int(CMD_GET_NEXT_CPU_WITH_EVENT);
    let num_added_cpus = aml::local(1);
    let cpu_idx = aml::local(2);
    let uid = aml::local(3);
    let has_job = aml::local(4);
    let new_cpus = aml::name(ADDED_LIST);
    let ej_cpus = aml::name(EJ_LIST);
    let num_ej_cpus = aml::local(5);
    let max = aml::int(MAX_CPUS_PER_PASS.into());

    let mut method = aml::method(SCAN_METHOD, 0, Serialize::Serialized);
    method.append(&aml::acquire(ctrl_lock, 0xFFFF));
    // Named packages because old Windows does not take a package in a local.
    method.append(&aml::name_decl(ADDED_LIST, &aml::package(MAX_CPUS_PER_PASS)));
    method.append(&aml::name_decl(EJ_LIST, &aml::package(MAX_CPUS_PER_PASS)));
    method.append(&aml::store(&zero, &uid));
    method.append(&aml::store(&one, &has_job));

    let mut while_ctx2 = aml::while_(&aml::equal(&has_job, &one));
    while_ctx2.append(&aml::store(&zero, &has_job));
    while_ctx2.append(&aml::store(&one, &has_event));
    while_ctx2.append(&aml::store(&zero, &num_added_cpus));
    while_ctx2.append(&aml::store(&zero, &num_ej_cpus));

    let mut while_ctx = aml::while_(&aml::land(
        &aml::equal(&has_event, &one),
        &aml::lless(&uid, &aml::int(ncpus as u64)),
    ));
    while_ctx.append(&aml::store(&zero, &has_event));
    while_ctx.append(&aml::store(&uid, cpu_selector));
    while_ctx.append(&aml::store(&next_cpu_cmd, cpu_cmd));
    // Wrapped around, the scan is complete.
    let mut if_ctx = aml::if_(&aml::lless(cpu_data, &uid));
    if_ctx.append(&aml::break_());
    while_ctx.append(&if_ctx);
    // A list is full, handle this batch first.
    let mut if_ctx =
        aml::if_(&aml::lor(&aml::equal(&num_added_cpus, &max), &aml::equal(&num_ej_cpus, &max)));
    if_ctx.append(&aml::store(&one, &has_job));
    if_ctx.append(&aml::break_());
    while_ctx.append(&if_ctx);
    while_ctx.append(&aml::store(cpu_data, &uid));
    let mut if_ctx = aml::if_(&aml::equal(ins_evt, &one));
    if_ctx.append(&aml::store(&uid, &aml::index(&new_cpus, &num_added_cpus)));
    if_ctx.append(&aml::increment(&num_added_cpus));
    if_ctx.append(&aml::store(&one, &has_event));
    while_ctx.append(&if_ctx);
    let mut if_ctx = aml::if_(&aml::equal(rm_evt, &one));
    if_ctx.append(&aml::store(&uid, &aml::index(&ej_cpus, &num_ej_cpus)));
    if_ctx.append(&aml::increment(&num_ej_cpus));
    if_ctx.append(&aml::store(&one, &has_event));
    while_ctx.append(&if_ctx);
    while_ctx.append(&aml::increment(&uid));
    while_ctx2.append(&while_ctx);

    // Let the firmware pull in new CPUs before the OS wakes them.
    if let Some(smi_path) = &opts.smi_path {
        let mut if_ctx = aml::if_(&aml::lgreater(&num_added_cpus, &zero));
        if_ctx.append(&aml::store(&aml::int(OVMF_CPUHP_SMI_CMD), &aml::name(smi_path)));
        while_ctx2.append(&if_ctx);
    }

    while_ctx2.append(&aml::store(&zero, &cpu_idx));
    let mut while_ctx = aml::while_(&aml::lless(&cpu_idx, &num_added_cpus));
    while_ctx.append(&aml::store(&aml::derefof(&aml::index(&new_cpus, &cpu_idx)), &uid));
    while_ctx.append(&aml::call(NOTIFY_METHOD, &[&uid, &dev_chk]));
    while_ctx.append(&aml::store(&uid, &aml::debug()));
    while_ctx.append(&aml::store(&uid, cpu_selector));
    while_ctx.append(&aml::store(&one, ins_evt));
    while_ctx.append(&aml::increment(&cpu_idx));
    while_ctx2.append(&while_ctx);

    while_ctx2.append(&aml::store(&zero, &cpu_idx));
    let mut while_ctx = aml::while_(&aml::lless(&cpu_idx, &num_ej_cpus));
    while_ctx.append(&aml::store(&aml::derefof(&aml::index(&ej_cpus, &cpu_idx)), &uid));
    while_ctx.append(&aml::call(NOTIFY_METHOD, &[&uid, &eject_req]));
    while_ctx.append(&aml::store(&uid, cpu_selector));
    while_ctx.append(&aml::store(&one, rm_evt));
    while_ctx.append(&aml::increment(&cpu_idx));
    while_ctx2.append(&while_ctx);

    method.append(&while_ctx2);
    method.append(&aml::release(ctrl_lock));
    method
}
