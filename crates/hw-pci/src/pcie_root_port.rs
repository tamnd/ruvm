// SPDX-License-Identifier: GPL-2.0-or-later

//! The generic PCI Express root port, `pcie-root-port` (1b36:000c), from
//! hw/pci-bridge/pcie_root_port.c, gen_pcie_root_port.c, hw/pci/pcie_port.c and the hotplug
//! parts of hw/pci/pcie.c.
//!
//! The port is a PCI-to-PCI bridge with this config space layout, the same as QEMU's:
//!
//! - 0x40: subsystem vendor ID capability (1b36:0000).
//! - 0x48: MSI-X with one vector in an exclusive BAR 0, used for hotplug events.
//! - 0x54: PCI Express capability, version 2, root port type, with a hotplug slot.
//! - 0x90: the Red Hat resource reserve capability, only when some reserve is set.
//! - 0x100: AER. The registers are there but errors are never injected.
//! - 0x148: ACS.
//!
//! Hotplug is driven by the board: it registers a function on [`PcieRootPort::sec_bus`] and
//! calls [`PcieRootPort::plug`], or asks for removal with [`PcieRootPort::unplug_request`]. The
//! guest then talks to the slot registers the usual way (attention button, power controller,
//! power indicator) and the port removes the functions when the guest powers the slot off.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use ruvm_base::Error;

use crate::bridge::{BridgeInner, PciBridge, disable_base_limit};
use crate::bus::PciBus;
use crate::device::{PciDevice, PciDeviceInfo, PciDeviceOps};
use crate::pcie::*;
use crate::regs::*;

/// `PCI_DEVICE_ID_REDHAT_PCIE_RP`.
pub const PCI_DEVICE_ID_REDHAT_PCIE_RP: u16 = 0x000c;
/// `GEN_PCIE_ROOT_PORT_AER_OFFSET`.
pub const GEN_PCIE_ROOT_PORT_AER_OFFSET: u16 = 0x100;
/// `GEN_PCIE_ROOT_PORT_ACS_OFFSET`, right after AER.
pub const GEN_PCIE_ROOT_PORT_ACS_OFFSET: u16 = GEN_PCIE_ROOT_PORT_AER_OFFSET + PCI_ERR_SIZEOF;
/// `GEN_PCIE_ROOT_PORT_MSIX_NR_VECTOR`.
pub const GEN_PCIE_ROOT_PORT_MSIX_NR_VECTOR: u32 = 1;
/// `GEN_PCIE_ROOT_DEFAULT_IO_RANGE`, the I/O reserve forced by the 6.1 compat knob.
pub const GEN_PCIE_ROOT_DEFAULT_IO_RANGE: u64 = 4096;

/// `PCI_SSVID_SIZEOF`.
pub const PCI_SSVID_SIZEOF: u8 = 8;
/// `PCI_SSVID_SVID`.
pub const PCI_SSVID_SVID: usize = 4;
/// `PCI_SSVID_SSID`.
pub const PCI_SSVID_SSID: usize = 6;

/// `REDHAT_PCI_CAP_TYPE_OFFSET`.
pub const REDHAT_PCI_CAP_TYPE_OFFSET: usize = 3;
/// `REDHAT_PCI_CAP_RESOURCE_RESERVE`.
pub const REDHAT_PCI_CAP_RESOURCE_RESERVE: u8 = 1;
/// `REDHAT_PCI_CAP_RES_RESERVE_BUS_RES`.
pub const REDHAT_PCI_CAP_RES_RESERVE_BUS_RES: usize = 4;
/// `REDHAT_PCI_CAP_RES_RESERVE_IO`.
pub const REDHAT_PCI_CAP_RES_RESERVE_IO: usize = 8;
/// `REDHAT_PCI_CAP_RES_RESERVE_MEM`.
pub const REDHAT_PCI_CAP_RES_RESERVE_MEM: usize = 16;
/// `REDHAT_PCI_CAP_RES_RESERVE_PREF_MEM_32`.
pub const REDHAT_PCI_CAP_RES_RESERVE_PREF_MEM_32: usize = 20;
/// `REDHAT_PCI_CAP_RES_RESERVE_PREF_MEM_64`.
pub const REDHAT_PCI_CAP_RES_RESERVE_PREF_MEM_64: usize = 24;
/// `sizeof(PCIBridgeQemuCap)`.
pub const REDHAT_PCI_CAP_RES_RESERVE_SIZEOF: u8 = 32;

const GIB: u64 = 1 << 30;

/// Extra resources firmware should give the bridge, `PCIResReserve`. `None` is QEMU's -1, not
/// set.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PciResReserve {
    /// `bus-reserve`: buses to reserve below the bridge.
    pub bus: Option<u32>,
    /// `io-reserve`.
    pub io: Option<u64>,
    /// `mem-reserve`: non-prefetchable memory.
    pub mem_non_pref: Option<u64>,
    /// `pref32-reserve`.
    pub mem_pref_32: Option<u64>,
    /// `pref64-reserve`.
    pub mem_pref_64: Option<u64>,
}

impl PciResReserve {
    fn is_empty(&self) -> bool {
        self.bus.is_none()
            && self.io.is_none()
            && self.mem_non_pref.is_none()
            && self.mem_pref_32.is_none()
            && self.mem_pref_64.is_none()
    }
}

/// `pci_bridge_qemu_reserve_cap_init()`: adds the Red Hat vendor capability that tells
/// firmware how much to reserve for hotplug. Nothing is added when no reserve is set. Returns
/// the offset of the capability, if one was added.
pub fn pci_bridge_qemu_reserve_cap_init(
    dev: &PciDevice,
    cap_offset: u8,
    res: &PciResReserve,
) -> Result<Option<u8>, Error> {
    if res.mem_pref_32.is_some() && res.mem_pref_64.is_some() {
        return Err(Error::generic("PCI resource reserve cap: PREF32 and PREF64 conflict"));
    }
    if res.mem_non_pref.is_some_and(|v| v >= 4 * GIB) {
        return Err(Error::generic("PCI resource reserve cap: mem-reserve must be less than 4G"));
    }
    if res.mem_pref_32.is_some_and(|v| v >= 4 * GIB) {
        // The double space is in QEMU's message too.
        return Err(Error::generic(
            "PCI resource reserve cap: pref32-reserve  must be less than 4G",
        ));
    }
    if res.is_empty() {
        return Ok(None);
    }
    let off = dev.add_capability(PCI_CAP_ID_VNDR, cap_offset, REDHAT_PCI_CAP_RES_RESERVE_SIZEOF)?;
    let o = usize::from(off);
    dev.with_config(|c| {
        let cfg = c.config;
        cfg[o + 2] = REDHAT_PCI_CAP_RES_RESERVE_SIZEOF;
        cfg[o + REDHAT_PCI_CAP_TYPE_OFFSET] = REDHAT_PCI_CAP_RESOURCE_RESERVE;
        // The C code stores the 64 bit properties into narrower fields, so -1 becomes all
        // ones of whatever width the field has.
        pci_set_long(cfg, o + REDHAT_PCI_CAP_RES_RESERVE_BUS_RES, res.bus.unwrap_or(u32::MAX));
        pci_set_quad(cfg, o + REDHAT_PCI_CAP_RES_RESERVE_IO, res.io.unwrap_or(u64::MAX));
        pci_set_long(
            cfg,
            o + REDHAT_PCI_CAP_RES_RESERVE_MEM,
            res.mem_non_pref.map_or(u32::MAX, |v| v as u32),
        );
        pci_set_long(
            cfg,
            o + REDHAT_PCI_CAP_RES_RESERVE_PREF_MEM_32,
            res.mem_pref_32.map_or(u32::MAX, |v| v as u32),
        );
        pci_set_quad(
            cfg,
            o + REDHAT_PCI_CAP_RES_RESERVE_PREF_MEM_64,
            res.mem_pref_64.unwrap_or(u64::MAX),
        );
    });
    Ok(Some(off))
}

/// `pci_bridge_ssvid_init()`: the subsystem vendor ID capability of a bridge.
pub fn pci_bridge_ssvid_init(
    dev: &PciDevice,
    offset: u8,
    svid: u16,
    ssid: u16,
) -> Result<u8, Error> {
    let pos = dev.add_capability(PCI_CAP_ID_SSVID, offset, PCI_SSVID_SIZEOF)?;
    let p = usize::from(pos);
    dev.with_config(|c| {
        pci_set_word(c.config, p + PCI_SSVID_SVID, svid);
        pci_set_word(c.config, p + PCI_SSVID_SSID, ssid);
    });
    Ok(pos)
}

/// `pcie_port_init_reg()`: status bits and bridge control bits that do not apply to PCI
/// Express.
pub fn pcie_port_init_reg(dev: &PciDevice) {
    dev.with_config(|c| {
        pci_set_word(c.config, PCI_STATUS, 0);
        pci_set_word(c.config, PCI_SEC_STATUS, 0);
        let m = PCI_BRIDGE_CTL_MASTER_ABORT
            | PCI_BRIDGE_CTL_FAST_BACK
            | PCI_BRIDGE_CTL_DISCARD
            | PCI_BRIDGE_CTL_SEC_DISCARD
            | PCI_BRIDGE_CTL_DISCARD_STATUS
            | PCI_BRIDGE_CTL_DISCARD_SERR;
        let v = pci_get_word(c.wmask, PCI_BRIDGE_CONTROL) & !m;
        pci_set_word(c.wmask, PCI_BRIDGE_CONTROL, v);
    });
}

/// The chassis and slot numbers in use, QEMU's `chassis` list in hw/pci/pcie_port.c. Two ports
/// sharing one registry cannot have the same chassis and slot. QEMU has one per process; here
/// a board keeps one per machine.
#[derive(Debug, Default)]
pub struct PcieChassisRegistry {
    slots: Mutex<Vec<(u8, u16)>>,
}

impl PcieChassisRegistry {
    /// An empty registry.
    pub fn new() -> PcieChassisRegistry {
        PcieChassisRegistry::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(u8, u16)>> {
        self.slots.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `pcie_chassis_add_slot()`. Fails with -EBUSY when the slot is taken.
    pub fn add_slot(&self, chassis: u8, slot: u16) -> Result<(), i32> {
        let mut g = self.lock();
        if g.contains(&(chassis, slot)) {
            return Err(-16);
        }
        g.push((chassis, slot));
        Ok(())
    }

    /// `pcie_chassis_del_slot()`.
    pub fn del_slot(&self, chassis: u8, slot: u16) {
        self.lock().retain(|&e| e != (chassis, slot));
    }

    /// Whether `slot` of `chassis` is taken.
    pub fn contains(&self, chassis: u8, slot: u16) -> bool {
        self.lock().contains(&(chassis, slot))
    }
}

/// The properties of a `pcie-root-port`, with QEMU's defaults.
#[derive(Clone)]
pub struct PcieRootPortConfig {
    /// The `id` of the device.
    pub id: Option<String>,
    /// The name of the secondary bus, which is the id in QEMU.
    pub sec_bus_name: String,
    /// `chassis`.
    pub chassis: u8,
    /// `slot`, the physical slot number.
    pub slot: u16,
    /// `port`, the port number in the link capabilities.
    pub port: u8,
    /// `hotplug`.
    pub hotplug: bool,
    /// `x-do-not-expose-native-hotplug-cap`.
    pub hide_native_hotplug_cap: bool,
    /// `power_controller_present`.
    pub power_controller_present: bool,
    /// `x-speed`.
    pub speed: PcieLinkSpeed,
    /// `x-width`.
    pub width: PcieLinkWidth,
    /// `bus-reserve`, `io-reserve`, `mem-reserve`, `pref32-reserve` and `pref64-reserve`.
    pub res_reserve: PciResReserve,
    /// `aer_log_max`.
    pub aer_log_max: u16,
    /// `multifunction`.
    pub multifunction: bool,
    /// `x-pcie-lnksta-dllla`, whether the data link layer active bit follows presence.
    pub lnksta_dllla: bool,
    /// Whether the port itself is hot-plugged.
    pub hotplugged: bool,
    /// The chassis and slot registry. Without one the uniqueness check is skipped.
    pub chassis_registry: Option<Arc<PcieChassisRegistry>>,
}

impl Default for PcieRootPortConfig {
    fn default() -> PcieRootPortConfig {
        PcieRootPortConfig {
            id: None,
            sec_bus_name: "pcie.1".to_string(),
            chassis: 0,
            slot: 0,
            port: 0,
            hotplug: true,
            hide_native_hotplug_cap: false,
            power_controller_present: true,
            speed: PcieLinkSpeed::Gt16,
            width: PcieLinkWidth::X32,
            res_reserve: PciResReserve::default(),
            aer_log_max: PCIE_AER_LOG_MAX_DEFAULT,
            multifunction: false,
            lnksta_dllla: true,
            hotplugged: false,
            chassis_registry: None,
        }
    }
}

impl fmt::Debug for PcieRootPortConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PcieRootPortConfig")
            .field("id", &self.id)
            .field("sec_bus_name", &self.sec_bus_name)
            .field("chassis", &self.chassis)
            .field("slot", &self.slot)
            .field("port", &self.port)
            .field("hotplug", &self.hotplug)
            .field("speed", &self.speed)
            .field("width", &self.width)
            .field("res_reserve", &self.res_reserve)
            .finish_non_exhaustive()
    }
}

/// Called with each function the port removes from its secondary bus, after it is gone, so
/// the board can drop the model behind it.
pub type PcieUnplugFn = Arc<dyn Fn(&Arc<PciDevice>) + Send + Sync>;

/// The state the config hooks need, `PCIExpressDevice` plus the bits of `PCIESlot`.
struct PortState {
    bridge: Arc<BridgeInner>,
    exp_cap: u8,
    aer_cap: u16,
    acs_cap: u16,
    lnksta_dllla: bool,
    /// `PCIExpressDevice::hpev_notified`.
    hpev_notified: AtomicBool,
    unplug_notifier: RwLock<Option<PcieUnplugFn>>,
}

impl PortState {
    fn sec_bus(&self) -> &Arc<PciBus> {
        self.bridge.sec_bus()
    }

    fn exp_word(&self, dev: &PciDevice, reg: usize) -> u16 {
        pci_get_word(&dev.config_bytes(), usize::from(self.exp_cap) + reg)
    }

    fn exp_long(&self, dev: &PciDevice, reg: usize) -> u32 {
        pci_get_long(&dev.config_bytes(), usize::from(self.exp_cap) + reg)
    }

    /// Sets `mask` in the PCIe capability word `reg`, returning the bits that were already set.
    fn exp_set_word(&self, dev: &PciDevice, reg: usize, mask: u16) -> u16 {
        let off = usize::from(self.exp_cap) + reg;
        dev.with_config(|c| {
            let v = pci_get_word(c.config, off);
            pci_set_word(c.config, off, v | mask);
            v & mask
        })
    }

    /// Clears `mask` in the PCIe capability word `reg`, returning the bits that were set.
    fn exp_clear_word(&self, dev: &PciDevice, reg: usize, mask: u16) -> u16 {
        let off = usize::from(self.exp_cap) + reg;
        dev.with_config(|c| {
            let v = pci_get_word(c.config, off);
            pci_set_word(c.config, off, v & !mask);
            v & mask
        })
    }

    fn dllla_follows_presence(&self, dev: &PciDevice) -> bool {
        self.lnksta_dllla || self.exp_long(dev, PCI_EXP_LNKCAP) & PCI_EXP_LNKCAP_DLLLARC != 0
    }

    /// `pcie_cap_update_power()`: powers the functions below the slot on or off.
    fn update_power(&self, dev: &PciDevice) {
        let sltcap = self.exp_long(dev, PCI_EXP_SLTCAP);
        let sltctl = self.exp_word(dev, PCI_EXP_SLTCTL);
        let power = sltcap & PCI_EXP_SLTCAP_PCP == 0
            || sltctl & PCI_EXP_SLTCTL_PCC == PCI_EXP_SLTCTL_PWR_ON;
        for d in self.sec_bus().devices() {
            d.set_enabled(power);
        }
    }

    /// `hotplug_event_update_event_status()`.
    fn update_event_status(&self, dev: &PciDevice) {
        let sltctl = self.exp_word(dev, PCI_EXP_SLTCTL);
        let sltsta = self.exp_word(dev, PCI_EXP_SLTSTA);
        let notified =
            sltctl & PCI_EXP_SLTCTL_HPIE != 0 && sltsta & sltctl & PCI_EXP_HP_EV_SUPPORTED != 0;
        self.hpev_notified.store(notified, Ordering::SeqCst);
    }

    /// `hotplug_event_notify()`: interrupts the guest when the notification state changes.
    fn hotplug_event_notify(&self, dev: &PciDevice) {
        let prev = self.hpev_notified.load(Ordering::SeqCst);
        self.update_event_status(dev);
        let now = self.hpev_notified.load(Ordering::SeqCst);
        if prev == now {
            return;
        }
        // Like QEMU this ignores whether interrupts are masked; a masked MSI-X vector is left
        // pending and fires when unmasked, which section 6.7.3.4 allows.
        if dev.msix_enabled() {
            dev.msix_notify(pcie_cap_flags_get_vector(dev, self.exp_cap));
        } else if dev.msi_enabled() {
            dev.msi_notify(pcie_cap_flags_get_vector(dev, self.exp_cap));
        } else if dev.intx() != -1 {
            dev.set_irq(i32::from(now));
        }
    }

    /// `hotplug_event_clear()`.
    fn hotplug_event_clear(&self, dev: &PciDevice) {
        self.update_event_status(dev);
        if !dev.msix_enabled()
            && !dev.msi_enabled()
            && dev.intx() != -1
            && !self.hpev_notified.load(Ordering::SeqCst)
        {
            dev.set_irq(0);
        }
    }

    /// `pcie_cap_slot_event()`.
    fn slot_event(&self, dev: &PciDevice, event: u16) {
        if self.exp_set_word(dev, PCI_EXP_SLTSTA, event) == event {
            // Nothing changed, so there is nothing to tell the guest.
            return;
        }
        self.hotplug_event_notify(dev);
    }

    /// `pcie_cap_slot_reset()`.
    fn slot_reset(&self, dev: &PciDevice, power_controller_present: bool) {
        self.exp_clear_word(
            dev,
            PCI_EXP_SLTCTL,
            PCI_EXP_SLTCTL_EIC
                | PCI_EXP_SLTCTL_PIC
                | PCI_EXP_SLTCTL_AIC
                | PCI_EXP_SLTCTL_HPIE
                | PCI_EXP_SLTCTL_CCIE
                | PCI_EXP_SLTCTL_PDCE
                | PCI_EXP_SLTCTL_ABPE,
        );
        self.exp_set_word(
            dev,
            PCI_EXP_SLTCTL,
            PCI_EXP_SLTCTL_PWR_IND_OFF | PCI_EXP_SLTCTL_ATTN_IND_OFF,
        );
        if power_controller_present {
            // Downstream ports only have device 0.
            let populated = self.sec_bus().device(0).is_some();
            if populated {
                self.exp_clear_word(dev, PCI_EXP_SLTCTL, PCI_EXP_SLTCTL_PCC);
            } else {
                self.exp_set_word(dev, PCI_EXP_SLTCTL, PCI_EXP_SLTCTL_PCC);
            }
            // QEMU ORs the "on" pattern into a field that already reads "off", so the power
            // indicator stays off either way. Kept as is.
            let pic =
                if populated { PCI_EXP_SLTCTL_PWR_IND_ON } else { PCI_EXP_SLTCTL_PWR_IND_OFF };
            self.exp_set_word(dev, PCI_EXP_SLTCTL, pic);
        }
        // On reset the electromechanical lock is released.
        self.exp_clear_word(
            dev,
            PCI_EXP_SLTSTA,
            PCI_EXP_SLTSTA_EIS | PCI_EXP_SLTSTA_CC | PCI_EXP_SLTSTA_PDC | PCI_EXP_SLTSTA_ABP,
        );
        self.update_power(dev);
        self.update_event_status(dev);
    }

    /// `pcie_unplug_device()` plus the unrealize done by `pcie_cap_slot_unplug_cb()`.
    fn unplug_device(&self, child: &Arc<PciDevice>) {
        self.sec_bus().unregister_device(child);
        let notifier = self.unplug_notifier.read().unwrap_or_else(|p| p.into_inner()).clone();
        if let Some(f) = notifier {
            f(child);
        }
    }

    /// `pcie_cap_slot_do_unplug()`: removes everything below the slot.
    fn do_unplug(&self, dev: &PciDevice) {
        for child in self.sec_bus().devices() {
            self.unplug_device(&child);
        }
        self.exp_clear_word(dev, PCI_EXP_SLTSTA, PCI_EXP_SLTSTA_PDS);
        if self.dllla_follows_presence(dev) {
            self.exp_clear_word(dev, PCI_EXP_LNKSTA, PCI_EXP_LNKSTA_DLLLA);
        }
        self.exp_set_word(dev, PCI_EXP_SLTSTA, PCI_EXP_SLTSTA_PDC);
    }

    /// `pcie_cap_slot_write_config()`.
    fn slot_write_config(
        &self,
        dev: &PciDevice,
        old_slt_ctl: u16,
        old_slt_sta: u16,
        addr: u32,
        val: u32,
        len: u32,
    ) {
        let pos = u64::from(self.exp_cap);
        let (a, l) = (u64::from(addr), u64::from(len));
        let mut sltsta = self.exp_word(dev, PCI_EXP_SLTSTA);

        if ranges_overlap(a, l, pos + PCI_EXP_SLTSTA as u64, 2) {
            // Guests tend to clear every bit during init. Clearing bits that were not set is
            // racy and would lose events, so put the event bits back the way they were.
            const EVENTS: u16 = PCI_EXP_SLTSTA_ABP
                | PCI_EXP_SLTSTA_PFD
                | PCI_EXP_SLTSTA_MRLSC
                | PCI_EXP_SLTSTA_PDC
                | PCI_EXP_SLTSTA_CC;
            if val as u16 & !old_slt_sta & EVENTS != 0 {
                sltsta = (sltsta & !EVENTS) | (old_slt_sta & EVENTS);
                let off = usize::from(self.exp_cap) + PCI_EXP_SLTSTA;
                dev.with_config(|c| pci_set_word(c.config, off, sltsta));
            }
            self.hotplug_event_clear(dev);
        }

        if !ranges_overlap(a, l, pos + PCI_EXP_SLTCTL as u64, 2) {
            return;
        }

        if self.exp_clear_word(dev, PCI_EXP_SLTCTL, PCI_EXP_SLTCTL_EIC) != 0 {
            // Writing 1 to the interlock control toggles the interlock.
            sltsta ^= PCI_EXP_SLTSTA_EIS;
            let off = usize::from(self.exp_cap) + PCI_EXP_SLTSTA;
            dev.with_config(|c| pci_set_word(c.config, off, sltsta));
        }

        // With the slot populated and both the power controller and the power indicator off,
        // the functions can go. Only on the transition: some guests rewrite the control of
        // slots that are already off before turning them on.
        if sltsta & PCI_EXP_SLTSTA_PDS != 0
            && pcie_sltctl_powered_off(val as u16)
            && !pcie_sltctl_powered_off(old_slt_ctl)
        {
            self.do_unplug(dev);
        }
        self.update_power(dev);
        self.hotplug_event_notify(dev);

        // Commands complete at once, so report the completion right away (6.7.3.2).
        self.slot_event(dev, PCI_EXP_HP_EV_CCI);
    }

    fn aer_root_cmd(&self, dev: &PciDevice) -> u32 {
        pci_get_long(&dev.config_bytes(), usize::from(self.aer_cap) + PCI_ERR_ROOT_COMMAND)
    }
}

/// The config and reset hooks of the root port, `rp_write_config()` and `rp_reset_hold()`.
struct RootPortOps {
    state: Arc<PortState>,
    power_controller_present: bool,
}

impl PciDeviceOps for RootPortOps {
    fn config_read(&self, dev: &PciDevice, addr: u32, len: u32) -> u32 {
        // pci_default_read_config() refreshes the link status of downstream ports first.
        let lnksta = u64::from(self.state.exp_cap) + PCI_EXP_LNKSTA as u64;
        if ranges_overlap(u64::from(addr), u64::from(len), lnksta, 2) {
            let target = self.state.sec_bus().device(0);
            pcie_sync_bridge_lnk(dev, self.state.exp_cap, target.as_deref());
        }
        dev.default_read_config(addr, len)
    }

    fn config_write(&self, dev: &PciDevice, addr: u32, val: u32, len: u32) {
        let st = &self.state;
        let root_cmd = st.aer_root_cmd(dev);
        let (slt_ctl, slt_sta) = pcie_cap_slot_get(dev, st.exp_cap);

        st.bridge.write_config(dev, addr, val, len);
        // rp_aer_vector_update(): the generic root port uses vector 0.
        pcie_aer_root_set_vector(dev, st.aer_cap, 0);
        st.slot_write_config(dev, slt_ctl, slt_sta, addr, val, len);
        pcie_aer_write_config(dev, st.aer_cap);
        pcie_aer_root_write_config(dev, st.aer_cap, root_cmd);
    }

    fn reset(&self, dev: &PciDevice) {
        let st = &self.state;
        pcie_aer_root_set_vector(dev, st.aer_cap, 0);
        pcie_cap_root_reset(dev, st.exp_cap);
        pcie_cap_deverr_reset(dev, st.exp_cap);
        st.slot_reset(dev, self.power_controller_present);
        pcie_cap_arifwd_reset(dev, st.exp_cap);
        pcie_acs_reset(dev, st.acs_cap);
        pcie_aer_root_reset(dev, st.aer_cap);
        st.bridge.reset(dev);
        disable_base_limit(dev);
    }
}

/// A `pcie-root-port` and its secondary bus.
pub struct PcieRootPort {
    bridge: PciBridge,
    state: Arc<PortState>,
    chassis: u8,
    slot: u16,
    registry: Option<Arc<PcieChassisRegistry>>,
    reserve_cap: Option<u8>,
}

impl fmt::Debug for PcieRootPort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PcieRootPort")
            .field("bridge", &self.bridge)
            .field("chassis", &self.chassis)
            .field("slot", &self.slot)
            .finish_non_exhaustive()
    }
}

impl PcieRootPort {
    /// Creates the port on `parent`, `rp_realize()` and `gen_rp_realize()`.
    ///
    /// The parent bus must accept MSI (`PciBus::set_msi_nonbroken`), since the port cannot
    /// work without its MSI-X vector, as in QEMU.
    pub fn new(
        parent: &Arc<PciBus>,
        cfg: &PcieRootPortConfig,
        devfn: Option<u8>,
    ) -> Result<PcieRootPort, Error> {
        let info = PciDeviceInfo {
            name: "pcie-root-port".to_string(),
            id: cfg.id.clone(),
            vendor_id: PCI_VENDOR_ID_REDHAT,
            device_id: PCI_DEVICE_ID_REDHAT_PCIE_RP,
            revision: 0,
            class_id: PCI_CLASS_BRIDGE_PCI,
            multifunction: cfg.multifunction,
            express: true,
            ..PciDeviceInfo::default()
        };
        let bridge = PciBridge::new(parent, &info, devfn, &cfg.sec_bus_name)?;
        let dev = Arc::clone(bridge.device());
        let undo = |e: Error| {
            parent.unregister_device(&dev);
            Err(e)
        };

        // The interrupt pin is set before pci_bridge_initfn(), which does not touch it.
        dev.with_config(|c| c.config[PCI_INTERRUPT_PIN] = 1);
        pcie_port_init_reg(&dev);

        if let Err(e) = pci_bridge_ssvid_init(&dev, 0, PCI_VENDOR_ID_REDHAT, 0) {
            return undo(e);
        }
        // gen_rp_interrupts_init().
        if let Err(e) = dev.msix_init_exclusive_bar(GEN_PCIE_ROOT_PORT_MSIX_NR_VECTOR, 0) {
            return undo(e);
        }
        dev.msix_vector_use(0);

        let exp_cap = match pcie_cap_init(&dev, 0, PCI_EXP_TYPE_ROOT_PORT, cfg.port) {
            Ok(p) => p,
            Err(e) => return undo(e),
        };
        pcie_cap_fill_slot_lnk(&dev, exp_cap, cfg.width, cfg.speed);
        pcie_cap_arifwd_init(&dev, exp_cap);
        pcie_cap_deverr_init(&dev, exp_cap);
        let slot = PcieSlotParams {
            slot: cfg.slot,
            hotplug: cfg.hotplug,
            hide_native_hotplug_cap: cfg.hide_native_hotplug_cap,
            hotplugged: cfg.hotplugged,
            power_controller_present: cfg.power_controller_present,
        };
        pcie_cap_slot_init(&dev, exp_cap, &slot);
        pcie_cap_root_init(&dev, exp_cap);

        if let Some(reg) = &cfg.chassis_registry {
            if let Err(rc) = reg.add_slot(cfg.chassis, cfg.slot) {
                return undo(Error::generic(format!("Can't add chassis slot, error {rc}")));
            }
        }
        let del_slot = || {
            if let Some(reg) = &cfg.chassis_registry {
                reg.del_slot(cfg.chassis, cfg.slot);
            }
        };

        let aer_cap = GEN_PCIE_ROOT_PORT_AER_OFFSET;
        if let Err(e) = pcie_aer_init(&dev, aer_cap, cfg.aer_log_max) {
            del_slot();
            return undo(e);
        }
        pcie_aer_root_init(&dev, aer_cap);
        pcie_aer_root_set_vector(&dev, aer_cap, 0);
        let acs_cap = GEN_PCIE_ROOT_PORT_ACS_OFFSET;
        pcie_acs_init(&dev, acs_cap);

        // gen_rp_realize(). Reserving I/O broke things in 6.1; the compat knob keeps that.
        let mut res = cfg.res_reserve;
        if cfg.hide_native_hotplug_cap && res.io.is_none() && cfg.hotplug {
            res.io = Some(GEN_PCIE_ROOT_DEFAULT_IO_RANGE);
        }
        let reserve_cap = match pci_bridge_qemu_reserve_cap_init(&dev, 0, &res) {
            Ok(off) => off,
            Err(e) => {
                del_slot();
                return undo(e);
            }
        };
        if res.io == Some(0) {
            dev.with_config(|c| {
                let v = pci_get_word(c.wmask, PCI_COMMAND) & !PCI_COMMAND_IO;
                pci_set_word(c.wmask, PCI_COMMAND, v);
                c.wmask[PCI_IO_BASE] = 0;
                c.wmask[PCI_IO_LIMIT] = 0;
            });
        }

        let state = Arc::new(PortState {
            bridge: Arc::clone(bridge.inner()),
            exp_cap,
            aer_cap,
            acs_cap,
            lnksta_dllla: cfg.lnksta_dllla,
            hpev_notified: AtomicBool::new(false),
            unplug_notifier: RwLock::new(None),
        });
        dev.set_ops(Arc::new(RootPortOps {
            state: Arc::clone(&state),
            power_controller_present: cfg.power_controller_present,
        }));
        Ok(PcieRootPort {
            bridge,
            state,
            chassis: cfg.chassis,
            slot: cfg.slot,
            registry: cfg.chassis_registry.clone(),
            reserve_cap,
        })
    }

    /// The port function on the parent bus.
    pub fn device(&self) -> &Arc<PciDevice> {
        self.bridge.device()
    }

    /// The bridge side of the port: windows and the secondary bus.
    pub fn bridge(&self) -> &PciBridge {
        &self.bridge
    }

    /// The secondary bus, where the device in the slot lives.
    pub fn sec_bus(&self) -> &Arc<PciBus> {
        self.bridge.sec_bus()
    }

    /// The offset of the PCI Express capability.
    pub fn exp_cap(&self) -> u8 {
        self.state.exp_cap
    }

    /// The offset of the AER capability.
    pub fn aer_cap(&self) -> u16 {
        self.state.aer_cap
    }

    /// The offset of the ACS capability.
    pub fn acs_cap(&self) -> u16 {
        self.state.acs_cap
    }

    /// The offset of the resource reserve capability, if one was added.
    pub fn reserve_cap(&self) -> Option<u8> {
        self.reserve_cap
    }

    /// `chassis`.
    pub fn chassis(&self) -> u8 {
        self.chassis
    }

    /// `slot`.
    pub fn slot(&self) -> u16 {
        self.slot
    }

    /// Whether a hotplug interrupt is being signalled, `hpev_notified`.
    pub fn hotplug_event_notified(&self) -> bool {
        self.state.hpev_notified.load(Ordering::SeqCst)
    }

    /// Sets the callback run for each function the port removes.
    pub fn set_unplug_notifier(&self, f: Option<PcieUnplugFn>) {
        *self.state.unplug_notifier.write().unwrap_or_else(|p| p.into_inner()) = f;
    }

    /// `pcie_cap_slot_pre_plug_cb()`: whether a function may be plugged now. Call it before
    /// registering the function on the secondary bus.
    pub fn pre_plug(&self, hotplugged: bool) -> Result<(), Error> {
        let dev = self.device();
        let sltcap = self.state.exp_long(dev, PCI_EXP_SLTCAP);
        if hotplugged && sltcap & PCI_EXP_SLTCAP_HPC == 0 {
            return Err(Error::generic(format!(
                "Hot-plug failed: unsupported by the port device '{}'",
                dev.id().unwrap_or("(null)")
            )));
        }
        self.check_interlock()
    }

    /// `pcie_cap_slot_plug_common()`.
    fn check_interlock(&self) -> Result<(), Error> {
        let sltsta = self.state.exp_word(self.device(), PCI_EXP_SLTSTA);
        if sltsta & PCI_EXP_SLTSTA_EIS != 0 {
            return Err(Error::generic("slot is electromechanically locked"));
        }
        Ok(())
    }

    /// `pcie_cap_slot_plug_cb()`: `child` was registered on the secondary bus. A cold plugged
    /// function just shows up as present. A hot plugged one also raises presence detect
    /// changed and the attention button once function 0 is there, so add function 0 last.
    pub fn plug(&self, child: &Arc<PciDevice>, hotplugged: bool) {
        let dev = self.device();
        let st = &self.state;
        if !hotplugged {
            st.exp_set_word(dev, PCI_EXP_SLTSTA, PCI_EXP_SLTSTA_PDS);
            if st.dllla_follows_presence(dev) {
                st.exp_set_word(dev, PCI_EXP_LNKSTA, PCI_EXP_LNKSTA_DLLLA);
            }
            st.update_power(dev);
            return;
        }
        let _ = child;
        if self.sec_bus().device(0).is_some() {
            st.exp_set_word(dev, PCI_EXP_SLTSTA, PCI_EXP_SLTSTA_PDS);
            if st.dllla_follows_presence(dev) {
                st.exp_set_word(dev, PCI_EXP_LNKSTA, PCI_EXP_LNKSTA_DLLLA);
            }
            st.slot_event(dev, PCI_EXP_HP_EV_PDC | PCI_EXP_HP_EV_ABP);
            st.update_power(dev);
        }
    }

    /// `pcie_cap_slot_unplug_request_cb()`: asks for `child` to go. Usually this pushes the
    /// attention button and the guest powers the slot off, which removes the functions. A
    /// slot that is already off is emptied at once, and so is a lone function whose function
    /// 0 never arrived.
    pub fn unplug_request(&self, child: &Arc<PciDevice>) -> Result<(), Error> {
        let dev = self.device();
        let st = &self.state;
        let sltcap = st.exp_long(dev, PCI_EXP_SLTCAP);
        let sltctl = st.exp_word(dev, PCI_EXP_SLTCTL);
        if sltcap & PCI_EXP_SLTCAP_HPC == 0 {
            return Err(Error::generic(format!(
                "Hot-unplug failed: unsupported by the port device '{}'",
                dev.id().unwrap_or("(null)")
            )));
        }
        self.check_interlock()?;
        if sltctl & PCI_EXP_SLTCTL_PIC == PCI_EXP_SLTCTL_PWR_IND_BLINK {
            return Err(Error::generic(
                "Hot-unplug failed: guest is busy (power indicator blinking)",
            ));
        }

        // A cancelled multi-function hot add: remove the function the guest never saw.
        if child.devfn() != 0 && self.sec_bus().device(0).is_none() {
            st.unplug_device(child);
            return Ok(());
        }

        if pcie_sltctl_powered_off(sltctl) {
            // Already off, so no round trip through the guest.
            st.do_unplug(dev);
            st.hotplug_event_notify(dev);
            st.exp_clear_word(dev, PCI_EXP_SLTSTA, PCI_EXP_SLTSTA_ABP);
            return Ok(());
        }

        self.push_attention_button();
        Ok(())
    }

    /// `pcie_cap_slot_push_attention_button()`.
    pub fn push_attention_button(&self) {
        self.state.slot_event(self.device(), PCI_EXP_HP_EV_ABP);
    }

    /// `pcie_cap_slot_enable_power()`: turns the power controller on without telling the
    /// guest.
    pub fn enable_power(&self) {
        let dev = self.device();
        if self.state.exp_long(dev, PCI_EXP_SLTCAP) & PCI_EXP_SLTCAP_PCP != 0 {
            self.state.exp_clear_word(dev, PCI_EXP_SLTCTL, PCI_EXP_SLTCTL_PCC);
        }
    }

    /// Removes the port from its bus and frees its chassis slot, `rp_exit()`. Whatever is on
    /// the secondary bus stays there.
    pub fn unrealize(self, parent: &Arc<PciBus>) {
        if let Some(reg) = &self.registry {
            reg.del_slot(self.chassis, self.slot);
        }
        let dev = self.device();
        dev.msix_uninit();
        parent.unregister_device(dev);
    }
}
