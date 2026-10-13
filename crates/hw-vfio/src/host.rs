// SPDX-License-Identifier: GPL-2.0-or-later

//! The `host` property of `vfio-pci`, `set_pci_host_devaddr()` of
//! hw/core/qdev-properties-system.c.

use std::fmt;

/// `PCIHostDeviceAddress`: where a host PCI function is, `[domain:]bus:slot.function`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PciHostDeviceAddress {
    pub domain: u32,
    pub bus: u32,
    pub slot: u32,
    pub function: u32,
}

impl PciHostDeviceAddress {
    /// `set_pci_host_devaddr()`: the bus, slot and domain in hex, the function in decimal, the
    /// way `strtoul()` reads them. `None` is QEMU's "Property 'vfio-pci.host' doesn't take
    /// value" error.
    pub fn parse(s: &str) -> Option<Self> {
        let b = s.as_bytes();
        let (val, mut e) = strtoul(b, 0, 16)?;
        if b.get(e) != Some(&b':') {
            return None;
        }
        let (mut dom, mut bus) = (0, val);
        let (mut val, e2) = strtoul(b, e + 1, 16)?;
        e = e2;
        if b.get(e) == Some(&b':') {
            dom = bus;
            bus = val;
            (val, e) = strtoul(b, e + 1, 16)?;
        }
        // slot and func are unsigned int in QEMU.
        let slot = val as u32;
        if b.get(e) != Some(&b'.') {
            return None;
        }
        let (val, e) = strtoul(b, e + 1, 10)?;
        let function = val as u32;
        if dom > 0xffff || bus > 0xff || slot > 0x1f || function > 7 || e != b.len() {
            return None;
        }
        Some(PciHostDeviceAddress { domain: dom as u32, bus: bus as u32, slot, function })
    }

    /// The sysfs directory of the function, which `vfio_pci_realize()` builds.
    pub fn sysfsdev(&self) -> String {
        format!("/sys/bus/pci/devices/{self}")
    }
}

impl fmt::Display for PciHostDeviceAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04x}:{:02x}:{:02x}.{:01x}", self.domain, self.bus, self.slot, self.function)
    }
}

/// `strtoul()` on `b` from `start`: the value and where the number ends, or `None` when there
/// is no number. Leading blanks and a sign are taken, `0x` too in base 16, a negative number
/// wraps and an overflow gives `ULONG_MAX`.
fn strtoul(b: &[u8], start: usize, base: u32) -> Option<(u64, usize)> {
    let mut i = start;
    while b.get(i).is_some_and(|c| c.is_ascii_whitespace()) {
        i += 1;
    }
    let neg = match b.get(i) {
        Some(b'-') => {
            i += 1;
            true
        }
        Some(b'+') => {
            i += 1;
            false
        }
        _ => false,
    };
    let digit = |c: Option<&u8>| c.and_then(|c| char::from(*c).to_digit(base));
    if base == 16
        && b.get(i) == Some(&b'0')
        && matches!(b.get(i + 1), Some(b'x' | b'X'))
        && digit(b.get(i + 2)).is_some()
    {
        i += 2;
    }
    let first = i;
    let mut val: u64 = 0;
    let mut overflow = false;
    while let Some(d) = digit(b.get(i)) {
        match val.checked_mul(u64::from(base)).and_then(|v| v.checked_add(u64::from(d))) {
            Some(v) => val = v,
            None => overflow = true,
        }
        i += 1;
    }
    if i == first {
        return None;
    }
    let val = if overflow {
        u64::MAX
    } else if neg {
        val.wrapping_neg()
    } else {
        val
    };
    Some((val, i))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(domain: u32, bus: u32, slot: u32, function: u32) -> Option<PciHostDeviceAddress> {
        Some(PciHostDeviceAddress { domain, bus, slot, function })
    }

    #[test]
    fn parses_like_qemu() {
        assert_eq!(PciHostDeviceAddress::parse("00:04.0"), addr(0, 0, 4, 0));
        assert_eq!(PciHostDeviceAddress::parse("0000:00:04.0"), addr(0, 0, 4, 0));
        assert_eq!(PciHostDeviceAddress::parse("1:ff:1f.7"), addr(1, 0xff, 0x1f, 7));
        assert_eq!(PciHostDeviceAddress::parse("0x1:0x2.3"), addr(0, 1, 2, 3));
        assert_eq!(PciHostDeviceAddress::parse(" 1: 2. 3"), addr(0, 1, 2, 3));
        for bad in ["zz", "", "00:04", "00:04.8", "00:20.0", "100:00.0", "10000:0:0.0", "0:0.0x"] {
            assert_eq!(PciHostDeviceAddress::parse(bad), None, "{bad}");
        }
        assert_eq!(addr(0, 0, 4, 0).unwrap().sysfsdev(), "/sys/bus/pci/devices/0000:00:04.0");
    }
}
