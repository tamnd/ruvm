// SPDX-License-Identifier: GPL-2.0-or-later

//! The `-machine microvm,...` properties: the ones microvm.c adds, the x86 machine ones it
//! inherits and the generic `usb` switch, with QEMU's defaults and error messages.

use std::fmt;

/// `OnOffAuto`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum OnOffAuto {
    /// `auto`: the board decides.
    #[default]
    Auto,
    /// `on`.
    On,
    /// `off`.
    Off,
}

impl OnOffAuto {
    /// Parses `value` for property `name` the way the QAPI enum visitor does.
    pub fn parse(name: &str, value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(OnOffAuto::Auto),
            "on" => Ok(OnOffAuto::On),
            "off" => Ok(OnOffAuto::Off),
            _ => Err(format!("Parameter '{name}' does not accept value '{value}'")),
        }
    }

    /// The QAPI name of the value.
    pub fn as_str(self) -> &'static str {
        match self {
            OnOffAuto::Auto => "auto",
            OnOffAuto::On => "on",
            OnOffAuto::Off => "off",
        }
    }
}

impl fmt::Display for OnOffAuto {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// `qapi_bool_parse()` as the keyval input visitor uses it for `-machine` options.
pub fn parse_bool(name: &str, value: &str) -> Result<bool, String> {
    match value {
        "on" | "yes" | "true" | "y" => Ok(true),
        "off" | "no" | "false" | "n" => Ok(false),
        _ => Err(format!("Parameter '{name}' expects 'on' or 'off'")),
    }
}

/// The properties of the `microvm` machine type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MicrovmProps {
    /// `rtc`: the MC146818 RTC. `auto` means on without KVM and off with it.
    pub rtc: OnOffAuto,
    /// `pcie`: the GPEX host bridge, only with ACPI.
    pub pcie: OnOffAuto,
    /// `ioapic2`: the second IOAPIC, only with ACPI.
    pub ioapic2: OnOffAuto,
    /// `isa-serial`.
    pub isa_serial: bool,
    /// `x-option-roms`: load option ROMs, including the one that boots `-kernel`.
    pub option_roms: bool,
    /// `auto-kernel-cmdline`: add the virtio-mmio devices to the kernel command line when
    /// ACPI is off.
    pub auto_kernel_cmdline: bool,
    /// `acpi`, from the x86 machine.
    pub acpi: OnOffAuto,
    /// `pit`, from the x86 machine.
    pub pit: OnOffAuto,
    /// `pic`, from the x86 machine.
    pub pic: OnOffAuto,
    /// `smm`, from the x86 machine. Kept for the command line, nothing here uses it.
    pub smm: OnOffAuto,
    /// `oem-id`, at most 6 bytes.
    pub oem_id: String,
    /// `oem-table-id`, at most 8 bytes.
    pub oem_table_id: String,
    /// `usb`, from the generic machine. microvm only builds an XHCI controller with ACPI.
    pub usb: bool,
}

impl Default for MicrovmProps {
    /// `microvm_machine_initfn()` and `x86_machine_initfn()`.
    fn default() -> Self {
        MicrovmProps {
            rtc: OnOffAuto::Auto,
            pcie: OnOffAuto::Auto,
            ioapic2: OnOffAuto::Auto,
            isa_serial: true,
            option_roms: true,
            auto_kernel_cmdline: true,
            acpi: OnOffAuto::Auto,
            pit: OnOffAuto::Auto,
            pic: OnOffAuto::Auto,
            smm: OnOffAuto::Auto,
            oem_id: ACPI_BUILD_APPNAME6.to_string(),
            oem_table_id: ACPI_BUILD_APPNAME8.to_string(),
            usb: false,
        }
    }
}

/// `ACPI_BUILD_APPNAME6`.
pub const ACPI_BUILD_APPNAME6: &str = "BOCHS ";
/// `ACPI_BUILD_APPNAME8`.
pub const ACPI_BUILD_APPNAME8: &str = "BXPC    ";

impl MicrovmProps {
    /// Sets one property from its command line text, `object_set_properties_from_keyval()`.
    /// Errors carry QEMU's message.
    pub fn set(&mut self, name: &str, value: &str) -> Result<(), String> {
        match name {
            "rtc" => self.rtc = OnOffAuto::parse(name, value)?,
            "pcie" => self.pcie = OnOffAuto::parse(name, value)?,
            "ioapic2" => self.ioapic2 = OnOffAuto::parse(name, value)?,
            "isa-serial" => self.isa_serial = parse_bool(name, value)?,
            "x-option-roms" => self.option_roms = parse_bool(name, value)?,
            "auto-kernel-cmdline" => self.auto_kernel_cmdline = parse_bool(name, value)?,
            "acpi" => self.acpi = OnOffAuto::parse(name, value)?,
            "pit" => self.pit = OnOffAuto::parse(name, value)?,
            "pic" => self.pic = OnOffAuto::parse(name, value)?,
            "smm" => self.smm = OnOffAuto::parse(name, value)?,
            "usb" => self.usb = parse_bool(name, value)?,
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
            _ => return Err(format!("Property 'microvm-machine.{name}' not found")),
        }
        Ok(())
    }

    /// Applies a `key=value,key=value` list, stopping at the first error. Commas inside
    /// values are not supported.
    pub fn set_all(&mut self, opts: &str) -> Result<(), String> {
        for item in opts.split(',').filter(|s| !s.is_empty()) {
            let Some((k, v)) = item.split_once('=') else {
                return Err(format!("Expected '=' after parameter '{item}'"));
            };
            self.set(k, v)?;
        }
        Ok(())
    }

    /// `x86_machine_is_acpi_enabled()`.
    pub fn acpi_enabled(&self) -> bool {
        self.acpi != OnOffAuto::Off
    }
}
