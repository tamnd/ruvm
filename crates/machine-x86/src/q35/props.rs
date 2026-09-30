// SPDX-License-Identifier: GPL-2.0-or-later

//! The `-machine q35,...` properties: the ones of the PC machine, the x86 machine ones it
//! inherits and the generic machine switches, with QEMU's defaults and error messages.

use crate::microvm::props::{ACPI_BUILD_APPNAME6, ACPI_BUILD_APPNAME8, OnOffAuto, parse_bool};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// The QOM type name the property errors mention.
pub const Q35_MACHINE_TYPE: &str = "pc-q35-11.1-machine";

/// The default `max-fw-size`, 8 MiB.
pub const MAX_FW_SIZE_DEFAULT: u64 = 8 * MIB;

/// `SmbiosEntryPointType`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum SmbiosEntryPointType {
    /// `32`: the SMBIOS 2.1 entry point.
    Ep32,
    /// `64`: the SMBIOS 3.0 entry point.
    Ep64,
    /// `auto`: 32 unless the tables need 64, the default of current machine types.
    #[default]
    Auto,
}

/// The properties of the `pc-q35-11.1` machine type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Q35Props {
    /// `max-ram-below-4g`: 0 means 4 GiB.
    pub max_ram_below_4g: u64,
    /// `max-fw-size`: only used by pflash, which is not modelled. Kept for its checks.
    pub max_fw_size: u64,
    /// `smm`, from the x86 machine.
    pub smm: OnOffAuto,
    /// `acpi`, from the x86 machine.
    pub acpi: OnOffAuto,
    /// `pit`, from the x86 machine.
    pub pit: OnOffAuto,
    /// `pic`, from the x86 machine.
    pub pic: OnOffAuto,
    /// `hpet`.
    pub hpet: bool,
    /// `sata`: the ICH9 AHCI controller at 00:1f.2.
    pub sata: bool,
    /// `smbus`: accepted, but the ICH9 SMBus controller is not modelled.
    pub smbus: bool,
    /// `i8042`: the PS/2 controller, and with it port 0x92.
    pub i8042: bool,
    /// `vmport`: `auto` means on unless the i8042 is off. The VMware port itself is not
    /// modelled.
    pub vmport: OnOffAuto,
    /// `fd-bootchk`: bit 0 of CMOS byte 0x38.
    pub fd_bootchk: bool,
    /// `default-bus-bypass-iommu`. There is no IOMMU, so it has no effect.
    pub default_bus_bypass_iommu: bool,
    /// `graphics`, from the generic machine, which fw_cfg reports. There is no VGA device
    /// either way.
    pub graphics: bool,
    /// `usb`, from the generic machine.
    pub usb: bool,
    /// `smbios-entry-point-type`.
    pub smbios_entry_point_type: SmbiosEntryPointType,
    /// `oem-id`, at most 6 bytes.
    pub oem_id: String,
    /// `oem-table-id`, at most 8 bytes.
    pub oem_table_id: String,
    /// `x-option-roms`: a microvm property in QEMU, accepted here too. Off skips the option
    /// ROMs, including the one that boots `-kernel`.
    pub option_roms: bool,
    /// `wdat`: the watchdog action table. Only off is supported.
    pub wdat: bool,
}

impl Default for Q35Props {
    /// `pc_machine_initfn()`, `x86_machine_initfn()` and `machine_initfn()`.
    fn default() -> Self {
        Q35Props {
            max_ram_below_4g: 0,
            max_fw_size: MAX_FW_SIZE_DEFAULT,
            smm: OnOffAuto::Auto,
            acpi: OnOffAuto::Auto,
            pit: OnOffAuto::Auto,
            pic: OnOffAuto::Auto,
            hpet: true,
            sata: true,
            smbus: true,
            i8042: true,
            vmport: OnOffAuto::Auto,
            fd_bootchk: true,
            default_bus_bypass_iommu: false,
            graphics: true,
            usb: false,
            smbios_entry_point_type: SmbiosEntryPointType::Auto,
            oem_id: ACPI_BUILD_APPNAME6.to_string(),
            oem_table_id: ACPI_BUILD_APPNAME8.to_string(),
            option_roms: true,
            wdat: false,
        }
    }
}

/// `qemu_strtosz()` as the keyval visitor uses it for size properties: a number with an
/// optional fraction or a hex number, and an optional binary unit suffix. Errors carry the
/// keyval visitor's message.
pub fn parse_size(name: &str, value: &str) -> Result<u64, String> {
    let bad = || format!("Parameter '{name}' expects size");
    let (body, unit) = match value.chars().last() {
        Some(c) if c.is_ascii_alphabetic() && !value.starts_with("0x") => {
            let shift = match c.to_ascii_uppercase() {
                'B' => 0,
                'K' => 10,
                'M' => 20,
                'G' => 30,
                'T' => 40,
                'P' => 50,
                'E' => 60,
                _ => return Err(bad()),
            };
            (&value[..value.len() - 1], 1u64 << shift)
        }
        _ => (value, 1),
    };
    if body.is_empty() {
        return Err(bad());
    }
    if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        let v = u64::from_str_radix(hex, 16).map_err(|_| bad())?;
        return v.checked_mul(unit).ok_or_else(bad);
    }
    let (int, frac) = match body.split_once('.') {
        Some((i, f)) => (i, f),
        None => (body, ""),
    };
    if int.is_empty() || !int.bytes().all(|b| b.is_ascii_digit()) {
        return Err(bad());
    }
    if !frac.bytes().all(|b| b.is_ascii_digit()) {
        return Err(bad());
    }
    let whole = int.parse::<u64>().map_err(|_| bad())?.checked_mul(unit).ok_or_else(bad)?;
    if frac.is_empty() || unit == 1 {
        if !frac.is_empty() && frac.bytes().any(|b| b != b'0') {
            // A fraction of a byte.
            return Err(bad());
        }
        return Ok(whole);
    }
    let f: f64 = format!("0.{frac}").parse().map_err(|_| bad())?;
    let extra = (f * unit as f64) as u64;
    whole.checked_add(extra).ok_or_else(bad)
}

impl Q35Props {
    /// Sets one property from its command line text, `object_set_properties_from_keyval()`.
    /// Errors carry QEMU's message; warnings QEMU prints on stderr go to `warnings`.
    pub fn set(
        &mut self,
        name: &str,
        value: &str,
        warnings: &mut Vec<String>,
    ) -> Result<(), String> {
        match name {
            "max-ram-below-4g" => {
                let v = parse_size(name, value)?;
                // pc_machine_set_max_ram_below_4g()
                if v > 4 * GIB {
                    return Err(format!(
                        "Machine option 'max-ram-below-4g={v}' expects size less than or equal \
                         to 4G"
                    ));
                }
                if v < MIB {
                    warnings.push(format!(
                        "Only {v} bytes of RAM below the 4GiB boundary,BIOS may not work with \
                         less than 1MiB"
                    ));
                }
                self.max_ram_below_4g = v;
            }
            "max-fw-size" => {
                let v = parse_size(name, value)?;
                // pc_machine_set_max_fw_size()
                if v > 16 * MIB {
                    return Err(format!(
                        "User specified max allowed firmware size {v} is greater than 16MiB. If \
                         combined firmware size exceeds 16MiB the system may not boot, or \
                         experience intermittentstability issues."
                    ));
                }
                self.max_fw_size = v;
            }
            "smm" => self.smm = OnOffAuto::parse(name, value)?,
            "acpi" => self.acpi = OnOffAuto::parse(name, value)?,
            "pit" => self.pit = OnOffAuto::parse(name, value)?,
            "pic" => self.pic = OnOffAuto::parse(name, value)?,
            "hpet" => self.hpet = parse_bool(name, value)?,
            "sata" => self.sata = parse_bool(name, value)?,
            "smbus" => self.smbus = parse_bool(name, value)?,
            "i8042" => self.i8042 = parse_bool(name, value)?,
            "vmport" => self.vmport = OnOffAuto::parse(name, value)?,
            "fd-bootchk" => self.fd_bootchk = parse_bool(name, value)?,
            "default-bus-bypass-iommu" => self.default_bus_bypass_iommu = parse_bool(name, value)?,
            "graphics" => self.graphics = parse_bool(name, value)?,
            "usb" => self.usb = parse_bool(name, value)?,
            "x-option-roms" => self.option_roms = parse_bool(name, value)?,
            "wdat" => self.wdat = parse_bool(name, value)?,
            "smbios-entry-point-type" => {
                self.smbios_entry_point_type = match value {
                    "32" => SmbiosEntryPointType::Ep32,
                    "64" => SmbiosEntryPointType::Ep64,
                    "auto" => SmbiosEntryPointType::Auto,
                    _ => {
                        return Err(format!("Parameter '{name}' does not accept value '{value}'"));
                    }
                }
            }
            "oem-id" => {
                if value.len() > 6 {
                    return Err(
                        "User specified oem-id value is bigger than 6 bytes in size".to_string()
                    );
                }
                self.oem_id = value.to_string();
            }
            "oem-table-id" => {
                if value.len() > 8 {
                    return Err("User specified oem-table-id value is bigger than 8 bytes in size"
                        .to_string());
                }
                self.oem_table_id = value.to_string();
            }
            _ => return Err(format!("Property '{Q35_MACHINE_TYPE}.{name}' not found")),
        }
        Ok(())
    }

    /// Applies a `key=value,key=value` list, stopping at the first error. Commas inside
    /// values are not supported.
    pub fn set_all(&mut self, opts: &str, warnings: &mut Vec<String>) -> Result<(), String> {
        for item in opts.split(',').filter(|s| !s.is_empty()) {
            let Some((k, v)) = item.split_once('=') else {
                return Err(format!("Expected '=' after parameter '{item}'"));
            };
            self.set(k, v, warnings)?;
        }
        Ok(())
    }

    /// `x86_machine_is_acpi_enabled()`.
    pub fn acpi_enabled(&self) -> bool {
        self.acpi != OnOffAuto::Off
    }

    /// `x86_machine_is_smm_enabled()`: `smm_available` is whether the accelerator can do SMM
    /// (always with TCG). Errors when `smm=on` cannot be honoured.
    pub fn smm_enabled(&self, smm_available: bool) -> Result<bool, String> {
        if self.smm == OnOffAuto::Off {
            return Ok(false);
        }
        if smm_available {
            return Ok(true);
        }
        if self.smm == OnOffAuto::On {
            return Err("System Management Mode not supported by this hypervisor.".to_string());
        }
        Ok(false)
    }
}
