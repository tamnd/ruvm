// SPDX-License-Identifier: GPL-2.0-or-later

//! The `PCIDevice` description every PCI function embeds, from hw/pci/pci.c, and the q35 host
//! side sections: `mch` (hw/pci-host/q35.c), `PCIHost` (hw/pci/pci_host.c) and `PCIBUS`.

use std::sync::{Arc, LazyLock};

use ruvm_base::{Result, bail, err};
use ruvm_hw_pci::{
    MchVmState, PciBus, PciBusVmState, PciDeviceVmState, PciHostState, PciHostVmState, Q35PciHost,
};
use ruvm_migration::SaveVm;
use ruvm_vmstate::info::{Int32Equal, Int32Le};
use ruvm_vmstate::{StreamReader, StreamWriter, VmStateDescription, VmStateField, VmStateInfo};

/// `PCI_CONFIG_SPACE_SIZE`.
const PCI_CONFIG_SPACE_SIZE: usize = 256;
/// `PCIE_CONFIG_SPACE_SIZE`.
const PCIE_CONFIG_SPACE_SIZE: usize = 4096;

/// `vmstate_info_pci_config`: the whole config space. QEMU checks the incoming bytes against
/// `cmask` and `wmask` in `get_pci_config_device()`; that is in
/// [`PciDevice::vmstate_load`](ruvm_hw_pci::PciDevice::vmstate_load).
#[derive(Debug)]
struct PciConfig(usize);

static PCI_CONFIG: PciConfig = PciConfig(PCI_CONFIG_SPACE_SIZE);
static PCIE_CONFIG: PciConfig = PciConfig(PCIE_CONFIG_SPACE_SIZE);

impl VmStateInfo<Vec<u8>> for PciConfig {
    fn name(&self) -> &'static str {
        "pci config"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut Vec<u8>, _size: usize) -> Result<()> {
        v.resize(self.0, 0);
        f.get_buffer(v);
        Ok(())
    }

    fn save(&self, f: &mut StreamWriter, v: &Vec<u8>, _size: usize) -> Result<()> {
        if v.len() != self.0 {
            bail!("config space of {} bytes, expected {}", v.len(), self.0);
        }
        f.put_buffer(v);
        Ok(())
    }
}

/// `vmstate_info_pci_irq_state`: the four INTx levels as big endian words, each 0 or 1.
#[derive(Debug)]
struct PciIrqState;

impl VmStateInfo<[i32; 4]> for PciIrqState {
    fn name(&self) -> &'static str {
        "pci irq state"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut [i32; 4], _size: usize) -> Result<()> {
        let mut irq_state = [0; 4];
        for s in &mut irq_state {
            let x = f.get_be32();
            if x > 1 {
                bail!("irq state {} must be 0 or 1", x as i32);
            }
            *s = x as i32;
        }
        *v = irq_state;
        Ok(())
    }

    fn save(&self, f: &mut StreamWriter, v: &[i32; 4], _size: usize) -> Result<()> {
        for &s in v {
            f.put_be32(s as u32);
        }
        Ok(())
    }
}

/// `vmstate_pci_device`, what `VMSTATE_PCI_DEVICE` and `VMSTATE_PCIE_DEVICE` nest. The config
/// field that matches the length of `config` is the one sent, as `pci_is_express()` picks it in
/// QEMU. `pci_post_load()` is in [`PciDevice::vmstate_load`](ruvm_hw_pci::PciDevice::vmstate_load).
pub(crate) static VMSTATE_PCI_DEVICE: LazyLock<VmStateDescription<PciDeviceVmState>> =
    LazyLock::new(|| {
        type S = PciDeviceVmState;
        VmStateDescription::new("PCIDevice").version_id(2).minimum_version_id(1).fields([
            VmStateField::single("version_id", &Int32Le, |s: &mut S| &mut s.version_id),
            VmStateField::single("config", &PCI_CONFIG, |s: &mut S| &mut s.config)
                .test(|s: &S, _| s.config.len() <= PCI_CONFIG_SPACE_SIZE),
            VmStateField::single("config", &PCIE_CONFIG, |s: &mut S| &mut s.config)
                .test(|s: &S, _| s.config.len() > PCI_CONFIG_SPACE_SIZE),
            VmStateField::single("irq_state", &PciIrqState, |s: &mut S| &mut s.irq_state)
                .version(2),
        ])
    });

/// `vmstate_mch`. `mch_post_load()` is in [`Q35PciHost::mch_vmstate_load`].
pub(crate) static VMSTATE_MCH: LazyLock<VmStateDescription<MchVmState>> = LazyLock::new(|| {
    type S = MchVmState;
    VmStateDescription::new("mch").version_id(1).minimum_version_id(1).fields([
        VmStateField::structure("parent_obj", &VMSTATE_PCI_DEVICE, |s: &mut S| &mut s.parent_obj),
        // Used to be smm_enabled.
        VmStateField::unused(1),
    ])
});

/// `vmstate_pcihost`. `pci_host_needed()` is `mig_enabled`, the
/// `x-config-reg-migration-enabled` property, which is on for q35.
pub(crate) static VMSTATE_PCIHOST: LazyLock<VmStateDescription<PciHostVmState>> =
    LazyLock::new(|| {
        type S = PciHostVmState;
        VmStateDescription::new("PCIHost")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|_: &S| true)
            .field(VmStateField::scalar("config_reg", |s: &mut S| &mut s.config_reg))
    });

/// `vmstate_pcibus`.
pub(crate) static VMSTATE_PCIBUS: LazyLock<VmStateDescription<PciBusVmState>> =
    LazyLock::new(|| {
        type S = PciBusVmState;
        VmStateDescription::new("PCIBUS").version_id(1).minimum_version_id(1).fields([
            VmStateField::single("nirq", &Int32Equal, |s: &mut S| &mut s.nirq),
            VmStateField::varray(
                "irq_count",
                |s: &S| usize::try_from(s.nirq).unwrap_or(0),
                |s: &mut S| &mut s.irq_count,
            ),
        ])
    });

/// Registers the MCH's `0000:00:00.0/mch` section, instance 0.
pub(crate) fn register_mch(savevm: &mut SaveVm, q35: &Arc<Q35PciHost>) {
    let (get, put) = (Arc::clone(q35), Arc::clone(q35));
    savevm.register_vmsd(
        "0000:00:00.0/",
        Some(0),
        &VMSTATE_MCH,
        move || Ok(get.mch_vmstate_save()),
        move |s| put.mch_vmstate_load(&s).map_err(|e| err!("{e}")),
    );
}

/// Registers the `PCIHost` section of the host bridge, instance 0 with no path prefix: QEMU
/// registers it with a NULL device.
pub(crate) fn register_pci_host(savevm: &mut SaveVm, host: &Arc<PciHostState>) {
    let (get, put) = (Arc::clone(host), Arc::clone(host));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_PCIHOST,
        move || Ok(get.vmstate_save()),
        move |s| {
            put.vmstate_load(&s);
            Ok(())
        },
    );
}

/// Registers the `PCIBUS` section of the root bus, instance 0 with no path prefix as
/// `pci_root_bus_internal_init()` does.
pub(crate) fn register_pci_bus(savevm: &mut SaveVm, bus: &Arc<PciBus>) {
    let (get, put) = (Arc::clone(bus), Arc::clone(bus));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_PCIBUS,
        move || Ok(get.vmstate_save()),
        move |s| put.vmstate_load(&s).map_err(|e| err!("{e}")),
    );
}

#[cfg(test)]
mod tests {
    use ruvm_vmstate::{vmstate_load_state, vmstate_save_state};

    use super::*;

    fn pci_device(len: usize) -> PciDeviceVmState {
        let mut config = vec![0; len];
        config[0] = 0x86;
        config[len - 1] = 0x5a;
        PciDeviceVmState { version_id: 2, config, irq_state: [0, 1, 0, 1] }
    }

    #[test]
    fn pci_device_layout_matches_qemu() {
        let mut s = pci_device(PCI_CONFIG_SPACE_SIZE);
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_PCI_DEVICE, &mut s).unwrap();
        let b = f.into_inner();
        assert_eq!(b.len(), 276);
        assert_eq!(&b[..4], &2i32.to_be_bytes());
        assert_eq!((b[4], b[259]), (0x86, 0x5a));
        assert_eq!(&b[260..], &[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]);

        let mut back = pci_device(PCI_CONFIG_SPACE_SIZE);
        back.config.fill(0);
        back.irq_state = [0; 4];
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_PCI_DEVICE, &mut back, 2).unwrap();
        assert_eq!(back, s);

        // A version 1 stream has no irq_state.
        let mut back = pci_device(PCI_CONFIG_SPACE_SIZE);
        back.irq_state = [0; 4];
        vmstate_load_state(&mut StreamReader::new(&b[..260]), &VMSTATE_PCI_DEVICE, &mut back, 1)
            .unwrap();
        assert_eq!(back.irq_state, [0; 4]);

        // INT32_POSITIVE_LE: a newer version_id than ours is refused.
        let mut bad = b.clone();
        bad[3] = 3;
        let mut back = pci_device(PCI_CONFIG_SPACE_SIZE);
        let r = vmstate_load_state(&mut StreamReader::new(&bad), &VMSTATE_PCI_DEVICE, &mut back, 2);
        assert!(r.is_err());

        // An INTx level other than 0 or 1 is refused.
        let mut bad = b;
        bad[275] = 2;
        let mut back = pci_device(PCI_CONFIG_SPACE_SIZE);
        let r = vmstate_load_state(&mut StreamReader::new(&bad), &VMSTATE_PCI_DEVICE, &mut back, 2);
        assert!(r.is_err());
    }

    #[test]
    fn pcie_device_sends_the_extended_space() {
        let mut s = pci_device(PCIE_CONFIG_SPACE_SIZE);
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_PCI_DEVICE, &mut s).unwrap();
        let b = f.into_inner();
        assert_eq!(b.len(), 4 + 4096 + 16);
        assert_eq!(b[4 + 4095], 0x5a);

        let mut back = pci_device(PCIE_CONFIG_SPACE_SIZE);
        back.config.fill(0);
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_PCI_DEVICE, &mut back, 2).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn mch_layout_matches_qemu() {
        let mut s = MchVmState { parent_obj: pci_device(PCI_CONFIG_SPACE_SIZE) };
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_MCH, &mut s).unwrap();
        let b = f.into_inner();
        assert_eq!(b.len(), 277);
        assert_eq!(b[276], 0);

        let mut back = MchVmState { parent_obj: pci_device(PCI_CONFIG_SPACE_SIZE) };
        back.parent_obj.config.fill(0);
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_MCH, &mut back, 1).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn pci_host_and_bus_layout_match_qemu() {
        let mut s = PciHostVmState { config_reg: 0x8000_f8d8 };
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_PCIHOST, &mut s).unwrap();
        let b = f.into_inner();
        assert_eq!(b, 0x8000_f8d8u32.to_be_bytes());
        let mut back = PciHostVmState::default();
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_PCIHOST, &mut back, 1).unwrap();
        assert_eq!(back, s);

        let mut s = PciBusVmState { nirq: 8, irq_count: vec![0, 1, 0, 0, 2, 0, 0, 0] };
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_PCIBUS, &mut s).unwrap();
        let b = f.into_inner();
        assert_eq!(b.len(), 4 + 8 * 4);
        assert_eq!(&b[..4], &8i32.to_be_bytes());
        assert_eq!(&b[20..24], &2i32.to_be_bytes());
        let mut back = PciBusVmState { nirq: 8, irq_count: vec![0; 8] };
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_PCIBUS, &mut back, 1).unwrap();
        assert_eq!(back, s);

        // INT32_EQUAL: a bus with a different number of lines is refused.
        let mut back = PciBusVmState { nirq: 4, irq_count: vec![0; 4] };
        let r = vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_PCIBUS, &mut back, 1);
        assert!(r.is_err());
    }
}
