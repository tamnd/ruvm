// SPDX-License-Identifier: GPL-2.0-or-later

//! Table framing and the generic tables from hw/acpi/aml-build.c: the System Description Table
//! header, RSDP, RSDT, XSDT and FADT.

use super::aml::{AddressSpace, append_int_noprefix};
use super::linker::BiosLinker;

/// `ACPI_BUILD_APPNAME6`, the default OEM ID.
pub const APPNAME6: &str = "BOCHS ";
/// `ACPI_BUILD_APPNAME8`, the default OEM table ID. Its first four bytes are the creator ID.
pub const APPNAME8: &str = "BXPC    ";

/// `ACPI_BUILD_TABLE_FILE`.
pub const TABLE_FILE: &str = "etc/acpi/tables";
/// `ACPI_BUILD_RSDP_FILE`.
pub const RSDP_FILE: &str = "etc/acpi/rsdp";
/// `ACPI_BUILD_TPMLOG_FILE`.
pub const TPMLOG_FILE: &str = "etc/tpm/log";
/// `ACPI_BUILD_LOADER_FILE`.
pub const LOADER_FILE: &str = "etc/table-loader";

/// Offset of the checksum byte in a System Description Table header.
const CHECKSUM_OFFSET: usize = 9;

/// `acpi_checksum()`: the byte that makes `data` sum to zero.
pub fn checksum(data: &[u8]) -> u8 {
    0u8.wrapping_sub(data.iter().fold(0u8, |s, &b| s.wrapping_add(b)))
}

/// `build_append_padded_str()`.
pub fn append_padded_str(buf: &mut Vec<u8>, s: &str, maxlen: usize, pad: u8) {
    assert!(s.len() <= maxlen, "{s:?} is longer than {maxlen}");
    buf.extend_from_slice(s.as_bytes());
    buf.resize(buf.len() + maxlen - s.len(), pad);
}

/// `struct AcpiGenericAddress`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Gas {
    pub space_id: u8,
    pub bit_width: u8,
    pub bit_offset: u8,
    pub access_width: u8,
    pub address: u64,
}

impl Gas {
    /// A register in the given address space.
    pub fn new(space: AddressSpace, bit_width: u8, address: u64) -> Self {
        Self { space_id: space as u8, bit_width, bit_offset: 0, access_width: 0, address }
    }
}

/// `build_append_gas()`.
pub fn append_gas(
    buf: &mut Vec<u8>,
    space: AddressSpace,
    bit_width: u8,
    bit_offset: u8,
    access_width: u8,
    address: u64,
) {
    append_gas_from(
        buf,
        &Gas { space_id: space as u8, bit_width, bit_offset, access_width, address },
    );
}

/// `build_append_gas_from_struct()`.
pub fn append_gas_from(buf: &mut Vec<u8>, g: &Gas) {
    buf.extend_from_slice(&[g.space_id, g.bit_width, g.bit_offset, g.access_width]);
    append_int_noprefix(buf, g.address, 8);
}

/// `AcpiTable`: a table being written into a blob, opened by [`AcpiTable::begin`] and closed by
/// [`AcpiTable::end`].
#[derive(Debug)]
pub struct AcpiTable {
    offset: usize,
}

impl AcpiTable {
    /// `acpi_table_begin()`: writes the header with a zero length and checksum.
    pub fn begin(
        sig: &str,
        rev: u8,
        oem_id: &str,
        oem_table_id: &str,
        array: &mut Vec<u8>,
    ) -> Self {
        assert_eq!(sig.len(), 4, "table signature {sig:?}");
        let offset = array.len();
        array.extend_from_slice(sig.as_bytes());
        append_int_noprefix(array, 0, 4); // Length, patched by end()
        array.push(rev);
        array.push(0); // Checksum
        append_padded_str(array, oem_id, 6, 0);
        append_padded_str(array, oem_table_id, 8, 0);
        append_int_noprefix(array, 1, 4); // OEM Revision
        array.extend_from_slice(&APPNAME8.as_bytes()[..4]); // Creator ID
        append_int_noprefix(array, 1, 4); // Creator Revision
        Self { offset }
    }

    /// Where the table starts in its blob.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// `acpi_table_end()`: patches the length, then either asks the firmware to fill in the
    /// checksum or fills it in right away when there is no linker.
    pub fn end(self, linker: Option<&mut BiosLinker>, array: &mut [u8]) {
        let len = array.len() - self.offset;
        let len32 = u32::try_from(len).expect("table fits in 32 bits");
        array[self.offset + 4..self.offset + 8].copy_from_slice(&len32.to_le_bytes());
        match linker {
            Some(l) => l.add_checksum(
                TABLE_FILE,
                self.offset as u32,
                len32,
                (self.offset + CHECKSUM_OFFSET) as u32,
            ),
            None => {
                let table = &mut array[self.offset..];
                table[CHECKSUM_OFFSET] = checksum(table);
            }
        }
    }
}

/// `acpi_add_table()`: remembers where the next table starts so the RSDT or XSDT can point at it.
pub fn add_table(table_offsets: &mut Vec<u32>, table_data: &[u8]) {
    table_offsets.push(table_data.len() as u32);
}

/// `AcpiRsdpData`.
#[derive(Clone, Debug)]
pub struct RsdpData<'a> {
    pub revision: u8,
    pub oem_id: &'a str,
    pub xsdt_tbl_offset: Option<u32>,
    pub rsdt_tbl_offset: Option<u32>,
}

/// `build_rsdp()`. `tbl` is the blob of [`RSDP_FILE`], and it is allocated here.
pub fn build_rsdp(tbl: &mut Vec<u8>, linker: &mut BiosLinker, rsdp: &RsdpData<'_>) {
    let tbl_off = tbl.len() as u32;
    match rsdp.revision {
        // ACPI 1.0 needs an RSDT pointer, 2.0 and later an XSDT pointer.
        0 => assert!(rsdp.rsdt_tbl_offset.is_some()),
        2 => assert!(rsdp.xsdt_tbl_offset.is_some()),
        r => panic!("RSDP revision {r}"),
    }
    linker.alloc(RSDP_FILE, 16, true);
    tbl.extend_from_slice(b"RSD PTR ");
    tbl.push(0); // Checksum
    append_padded_str(tbl, rsdp.oem_id, 6, 0);
    tbl.push(rsdp.revision);
    append_int_noprefix(tbl, 0, 4); // RsdtAddress
    if let Some(off) = rsdp.rsdt_tbl_offset {
        linker.add_pointer(RSDP_FILE, tbl, tbl_off + 16, 4, TABLE_FILE, off);
    }
    // The ACPI 1.0 part is 20 bytes.
    linker.add_checksum(RSDP_FILE, tbl_off, 20, 8);
    if rsdp.revision == 0 {
        return;
    }
    append_int_noprefix(tbl, 36, 4); // Length
    append_int_noprefix(tbl, 0, 8); // XsdtAddress
    let xsdt = rsdp.xsdt_tbl_offset.expect("checked above");
    linker.add_pointer(RSDP_FILE, tbl, tbl_off + 24, 8, TABLE_FILE, xsdt);
    append_int_noprefix(tbl, 0, 1); // Extended Checksum
    append_int_noprefix(tbl, 0, 3); // Reserved
    linker.add_checksum(RSDP_FILE, tbl_off, 36, 32);
}

fn build_sdt(
    sig: &str,
    entry_size: u8,
    table_data: &mut Vec<u8>,
    linker: &mut BiosLinker,
    table_offsets: &[u32],
    oem_id: &str,
    oem_table_id: &str,
) {
    let table = AcpiTable::begin(sig, 1, oem_id, oem_table_id, table_data);
    for &off in table_offsets {
        let entry = table_data.len() as u32;
        append_int_noprefix(table_data, 0, entry_size.into());
        linker.add_pointer(TABLE_FILE, table_data, entry, entry_size, TABLE_FILE, off);
    }
    table.end(Some(linker), table_data);
}

/// `build_rsdt()`, ACPI 1.0.
pub fn build_rsdt(
    table_data: &mut Vec<u8>,
    linker: &mut BiosLinker,
    table_offsets: &[u32],
    oem_id: &str,
    oem_table_id: &str,
) {
    build_sdt("RSDT", 4, table_data, linker, table_offsets, oem_id, oem_table_id);
}

/// `build_xsdt()`, ACPI 2.0.
pub fn build_xsdt(
    table_data: &mut Vec<u8>,
    linker: &mut BiosLinker,
    table_offsets: &[u32],
    oem_id: &str,
    oem_table_id: &str,
) {
    build_sdt("XSDT", 8, table_data, linker, table_offsets, oem_id, oem_table_id);
}

/// `AcpiFadtData`. The three table offsets are `None` when that table does not exist, in which
/// case the field is left zero.
#[derive(Clone, Debug, Default)]
pub struct FadtData {
    pub pm1a_cnt: Gas,
    pub pm1a_evt: Gas,
    pub pm_tmr: Gas,
    pub gpe0_blk: Gas,
    pub reset_reg: Gas,
    pub sleep_ctl: Gas,
    pub sleep_sts: Gas,
    pub reset_val: u8,
    pub rev: u8,
    pub flags: u32,
    pub smi_cmd: u32,
    pub sci_int: u16,
    pub int_model: u8,
    pub acpi_enable_cmd: u8,
    pub acpi_disable_cmd: u8,
    pub rtc_century: u8,
    pub plvl2_lat: u16,
    pub plvl3_lat: u16,
    pub arm_boot_arch: u16,
    pub iapc_boot_arch: u16,
    pub minor_ver: u8,
    pub facs_tbl_offset: Option<u32>,
    pub dsdt_tbl_offset: Option<u32>,
    pub xdsdt_tbl_offset: Option<u32>,
}

/// FADT flag bits used by the x86 machines.
pub mod fadt_flags {
    pub const WBINVD: u32 = 0;
    pub const PROC_C1: u32 = 2;
    pub const SLP_BUTTON: u32 = 5;
    pub const RTC_S4: u32 = 7;
    pub const USE_PLATFORM_CLOCK: u32 = 15;
    pub const RESET_REG_SUP: u32 = 10;
    pub const HW_REDUCED_ACPI: u32 = 20;
    pub const FORCE_APIC_CLUSTER_MODEL: u32 = 18;
    pub const FORCE_APIC_PHYSICAL_DESTINATION_MODE: u32 = 19;
    pub const LOW_POWER_S0_IDLE_CAPABLE: u32 = 21;
}

/// `build_fadt()`.
pub fn build_fadt(
    tbl: &mut Vec<u8>,
    linker: &mut BiosLinker,
    f: &FadtData,
    oem_id: &str,
    oem_table_id: &str,
) {
    let table = AcpiTable::begin("FACP", f.rev, oem_id, oem_table_id, tbl);
    let pointer = |tbl: &mut Vec<u8>, linker: &mut BiosLinker, size: u8, target: Option<u32>| {
        let off = tbl.len() as u32;
        append_int_noprefix(tbl, 0, size.into());
        if let Some(t) = target {
            linker.add_pointer(TABLE_FILE, tbl, off, size, TABLE_FILE, t);
        }
    };
    pointer(tbl, linker, 4, f.facs_tbl_offset); // FIRMWARE_CTRL
    pointer(tbl, linker, 4, f.dsdt_tbl_offset); // DSDT
    // ACPI 1.0: INT_MODEL, 2.0 and later: reserved.
    tbl.push(f.int_model);
    tbl.push(0); // Preferred_PM_Profile, unspecified
    append_int_noprefix(tbl, f.sci_int.into(), 2);
    append_int_noprefix(tbl, f.smi_cmd.into(), 4);
    tbl.push(f.acpi_enable_cmd);
    tbl.push(f.acpi_disable_cmd);
    tbl.push(0); // S4BIOS_REQ
    tbl.push(0); // PSTATE_CNT
    append_int_noprefix(tbl, f.pm1a_evt.address, 4); // PM1a_EVT_BLK
    append_int_noprefix(tbl, 0, 4); // PM1b_EVT_BLK
    append_int_noprefix(tbl, f.pm1a_cnt.address, 4); // PM1a_CNT_BLK
    append_int_noprefix(tbl, 0, 4); // PM1b_CNT_BLK
    append_int_noprefix(tbl, 0, 4); // PM2_CNT_BLK
    append_int_noprefix(tbl, f.pm_tmr.address, 4); // PM_TMR_BLK
    append_int_noprefix(tbl, f.gpe0_blk.address, 4); // GPE0_BLK
    append_int_noprefix(tbl, 0, 4); // GPE1_BLK
    tbl.push(f.pm1a_evt.bit_width / 8); // PM1_EVT_LEN
    tbl.push(f.pm1a_cnt.bit_width / 8); // PM1_CNT_LEN
    tbl.push(0); // PM2_CNT_LEN
    tbl.push(f.pm_tmr.bit_width / 8); // PM_TMR_LEN
    tbl.push(f.gpe0_blk.bit_width / 8); // GPE0_BLK_LEN
    tbl.push(0); // GPE1_BLK_LEN
    tbl.push(0); // GPE1_BASE
    tbl.push(0); // CST_CNT
    append_int_noprefix(tbl, f.plvl2_lat.into(), 2);
    append_int_noprefix(tbl, f.plvl3_lat.into(), 2);
    append_int_noprefix(tbl, 0, 2); // FLUSH_SIZE
    append_int_noprefix(tbl, 0, 2); // FLUSH_STRIDE
    tbl.push(0); // DUTY_OFFSET
    tbl.push(0); // DUTY_WIDTH
    tbl.push(0); // DAY_ALRM
    tbl.push(0); // MON_ALRM
    tbl.push(f.rtc_century);
    // IAPC_BOOT_ARCH exists since ACPI 2.0.
    let boot_arch = if f.rev == 1 { 0 } else { f.iapc_boot_arch };
    append_int_noprefix(tbl, boot_arch.into(), 2);
    tbl.push(0); // Reserved
    append_int_noprefix(tbl, f.flags.into(), 4);
    if f.rev != 1 {
        append_gas_from(tbl, &f.reset_reg);
        tbl.push(f.reset_val);
        if f.rev >= 6 || (f.rev == 5 && f.minor_ver > 0) {
            // Since ACPI 5.1.
            append_int_noprefix(tbl, f.arm_boot_arch.into(), 2);
            tbl.push(f.minor_ver);
        } else {
            append_int_noprefix(tbl, 0, 3); // Reserved up to ACPI 5.0
        }
        append_int_noprefix(tbl, 0, 8); // X_FIRMWARE_CTRL
        pointer(tbl, linker, 8, f.xdsdt_tbl_offset); // X_DSDT
        let zero = Gas::default();
        append_gas_from(tbl, &f.pm1a_evt); // X_PM1a_EVT_BLK
        append_gas_from(tbl, &zero); // X_PM1b_EVT_BLK
        append_gas_from(tbl, &f.pm1a_cnt); // X_PM1a_CNT_BLK
        append_gas_from(tbl, &zero); // X_PM1b_CNT_BLK
        append_gas_from(tbl, &zero); // X_PM2_CNT_BLK
        append_gas_from(tbl, &f.pm_tmr); // X_PM_TMR_BLK
        append_gas_from(tbl, &f.gpe0_blk); // X_GPE0_BLK
        append_gas_from(tbl, &zero); // X_GPE1_BLK
        if f.rev > 4 {
            append_gas_from(tbl, &f.sleep_ctl); // SLEEP_CONTROL_REG
            append_gas_from(tbl, &f.sleep_sts); // SLEEP_STATUS_REG
            if f.rev != 5 {
                // Hypervisor Vendor Identity. Revisions past 6 add fields QEMU does not write yet.
                assert_eq!(f.rev, 6, "FADT revision {}", f.rev);
                append_padded_str(tbl, "QEMU", 8, 0);
            }
        }
    }
    table.end(Some(linker), tbl);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_without_linker_is_checksummed() {
        let mut blob = vec![0xAA; 3];
        let t = AcpiTable::begin("TEST", 2, APPNAME6, APPNAME8, &mut blob);
        blob.extend_from_slice(&[1, 2, 3]);
        t.end(None, &mut blob);
        let table = &blob[3..];
        assert_eq!(table.len(), 39);
        assert_eq!(&table[..4], b"TEST");
        assert_eq!(table[4], 39);
        assert_eq!(&table[10..16], b"BOCHS ");
        assert_eq!(&table[16..24], b"BXPC    ");
        assert_eq!(&table[28..32], b"BXPC");
        assert_eq!(checksum(table), 0);
    }

    #[test]
    fn rsdp_revision_2_layout() {
        let mut linker = BiosLinker::new();
        linker.alloc(TABLE_FILE, 64, false);
        let mut rsdp = Vec::new();
        let data = RsdpData {
            revision: 2,
            oem_id: APPNAME6,
            xsdt_tbl_offset: Some(0x80),
            rsdt_tbl_offset: None,
        };
        build_rsdp(&mut rsdp, &mut linker, &data);
        assert_eq!(rsdp.len(), 36);
        assert_eq!(&rsdp[..8], b"RSD PTR ");
        assert_eq!(rsdp[15], 2);
        assert_eq!(&rsdp[20..24], [36, 0, 0, 0]);
        assert_eq!(rsdp[24], 0x80);
    }
}
