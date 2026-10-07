// SPDX-License-Identifier: GPL-2.0-or-later

//! The MSI-X capability, from hw/pci/msix.c.
//!
//! The capability in config space only holds the enable and function mask bits and says where
//! the vector table and the pending bit array (PBA) live. Both are MMIO regions placed inside
//! one of the device's memory BARs. Each table entry is 16 bytes: a 64 bit address, 32 bits of
//! data and a vector control word whose bit 0 masks the vector.
//!
//! A model calls [`PciDevice::msix_vector_use`] for the vectors it drives and then
//! [`PciDevice::msix_notify`]. A masked vector (or any vector while the function mask is set or
//! MSI-X is disabled) gets its PBA bit set instead, and fires once unmasked.
//!
//! The vector use, release and poll notifiers (used by vhost and VFIO) and the Xen paths are
//! not ported.

use std::fmt;
use std::sync::{Arc, Weak};

use ruvm_base::Error;
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, Endian, MemResult, MmioOps, RegionId};

use crate::device::{DevState, Effects, MsiMessage, PciDevice};
use crate::regs::*;

const MSIX_CONTROL_OFFSET: usize = PCI_MSIX_FLAGS + 1;
const MSIX_ENABLE_MASK: u8 = (PCI_MSIX_FLAGS_ENABLE >> 8) as u8;
const MSIX_MASKALL_MASK: u8 = (PCI_MSIX_FLAGS_MASKALL >> 8) as u8;

pub(crate) struct Msix {
    cap: u8,
    entries: u32,
    table: Vec<u8>,
    pba: Vec<u8>,
    used: Vec<u32>,
    function_masked: bool,
    table_bar: RegionId,
    table_mmio: RegionId,
    pba_bar: RegionId,
    pba_mmio: RegionId,
    /// The BAR created by `msix_init_exclusive_bar()`.
    exclusive_bar: Option<(usize, RegionId)>,
}

impl Msix {
    fn pending_mask(vector: u32) -> u8 {
        1 << (vector % 8)
    }

    fn is_pending(&self, vector: u32) -> bool {
        self.pba[(vector / 8) as usize] & Self::pending_mask(vector) != 0
    }

    fn set_pending(&mut self, vector: u32) {
        self.pba[(vector / 8) as usize] |= Self::pending_mask(vector);
    }

    fn clr_pending(&mut self, vector: u32) {
        self.pba[(vector / 8) as usize] &= !Self::pending_mask(vector);
    }

    fn ctrl_off(vector: u32) -> usize {
        vector as usize * PCI_MSIX_ENTRY_SIZE + PCI_MSIX_ENTRY_VECTOR_CTRL
    }

    /// `msix_vector_masked()`.
    fn vector_masked(&self, vector: u32, fmask: bool) -> bool {
        fmask || self.table[Self::ctrl_off(vector)] & PCI_MSIX_ENTRY_CTRL_MASKBIT != 0
    }

    /// `msix_is_masked()`.
    fn is_masked(&self, vector: u32) -> bool {
        self.vector_masked(vector, self.function_masked)
    }

    /// `msix_prepare_message()`.
    fn message(&self, vector: u32) -> MsiMessage {
        let e = vector as usize * PCI_MSIX_ENTRY_SIZE;
        MsiMessage {
            address: pci_get_quad(&self.table, e + PCI_MSIX_ENTRY_LOWER_ADDR),
            data: pci_get_long(&self.table, e + PCI_MSIX_ENTRY_DATA),
        }
    }
}

/// `QEMU_ALIGN_UP(nentries, 64) / 8`.
fn pba_size(nentries: u32) -> u32 {
    nentries.next_multiple_of(64) / 8
}

impl DevState {
    fn msix_enabled(&self) -> bool {
        self.msix.as_ref().is_some_and(|m| {
            self.config[usize::from(m.cap) + MSIX_CONTROL_OFFSET] & MSIX_ENABLE_MASK != 0
        })
    }

    fn msix_masked(&self) -> bool {
        self.msix.as_ref().is_some_and(|m| {
            self.config[usize::from(m.cap) + MSIX_CONTROL_OFFSET] & MSIX_MASKALL_MASK != 0
        })
    }

    /// `msix_update_function_masked()`.
    fn msix_update_function_masked(&mut self) {
        let masked = !self.msix_enabled() || self.msix_masked();
        if let Some(m) = self.msix.as_mut() {
            m.function_masked = masked;
        }
    }

    fn msix_mut(&mut self) -> &mut Msix {
        self.msix.as_mut().expect("MSI-X is not initialized")
    }

    fn msix_ref(&self) -> &Msix {
        self.msix.as_ref().expect("MSI-X is not initialized")
    }

    /// `msix_notify()`.
    pub(crate) fn msix_notify(&mut self, fx: &mut Effects, vector: u32) {
        let m = self.msix_mut();
        assert!(vector < m.entries, "MSI-X vector {vector} out of range");
        if m.used[vector as usize] == 0 {
            return;
        }
        if m.is_masked(vector) {
            m.set_pending(vector);
            return;
        }
        let msg = m.message(vector);
        self.send_message(fx, msg);
    }

    /// `msix_handle_mask_update()`.
    fn msix_handle_mask_update(&mut self, fx: &mut Effects, vector: u32, was_masked: bool) {
        let m = self.msix_mut();
        let is_masked = m.is_masked(vector);
        if is_masked == was_masked {
            return;
        }
        if !is_masked && m.is_pending(vector) {
            m.clr_pending(vector);
            self.msix_notify(fx, vector);
        }
    }

    /// `msix_mask_all()`.
    fn msix_mask_all(&mut self, fx: &mut Effects) {
        let entries = self.msix_ref().entries;
        for vector in 0..entries {
            let m = self.msix_mut();
            let was_masked = m.is_masked(vector);
            m.table[Msix::ctrl_off(vector)] |= PCI_MSIX_ENTRY_CTRL_MASKBIT;
            self.msix_handle_mask_update(fx, vector, was_masked);
        }
    }

    /// `msix_write_config()`.
    pub(crate) fn msix_write_config(&mut self, fx: &mut Effects, addr: u32, len: u32) {
        let Some(m) = self.msix.as_ref() else { return };
        let enable_pos = (usize::from(m.cap) + MSIX_CONTROL_OFFSET) as u64;
        if !range_covers_byte(addr.into(), len.into(), enable_pos) {
            return;
        }
        let was_masked = m.function_masked;
        self.msix_update_function_masked();
        if !self.msix_enabled() {
            return;
        }
        self.deassert_intx(fx);
        let m = self.msix_ref();
        if m.function_masked == was_masked {
            return;
        }
        for vector in 0..m.entries {
            let old = self.msix_ref().vector_masked(vector, was_masked);
            self.msix_handle_mask_update(fx, vector, old);
        }
    }

    /// `msix_reset()`.
    pub(crate) fn msix_reset(&mut self, fx: &mut Effects) {
        let Some(m) = self.msix.as_mut() else { return };
        let ctrl = usize::from(m.cap) + MSIX_CONTROL_OFFSET;
        m.pba.fill(0);
        m.table.fill(0);
        self.config[ctrl] &= !self.wmask[ctrl];
        self.msix_mask_all(fx);
    }
}

/// The MMIO view of the vector table, `msix_table_mmio_ops`.
struct TableOps(Weak<PciDevice>);

/// The MMIO view of the pending bit array, `msix_pba_mmio_ops`.
struct PbaOps(Weak<PciDevice>);

impl fmt::Debug for TableOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MsixTableOps")
    }
}

impl fmt::Debug for PbaOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MsixPbaOps")
    }
}

fn read_long(bytes: &[u8], off: usize) -> u64 {
    // An access at the very end of a PBA smaller than 4 bytes cannot happen: the PBA is a
    // multiple of 8 bytes. Guard anyway rather than panic on a guest access.
    if off + 4 > bytes.len() {
        return 0;
    }
    u64::from(pci_get_long(bytes, off))
}

impl MmioOps for TableOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        let Some(dev) = self.0.upgrade() else { return Ok(0) };
        let s = dev.lock();
        Ok(s.msix.as_ref().map_or(0, |m| read_long(&m.table, offset as usize)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        let Some(dev) = self.0.upgrade() else { return Ok(()) };
        let mut fx = Effects::new();
        {
            let mut s = dev.lock();
            let Some(m) = s.msix.as_mut() else { return Ok(()) };
            let off = offset as usize;
            if off + 4 > m.table.len() {
                return Ok(());
            }
            let vector = (off / PCI_MSIX_ENTRY_SIZE) as u32;
            let was_masked = m.is_masked(vector);
            pci_set_long(&mut m.table, off, value as u32);
            s.msix_handle_mask_update(&mut fx, vector, was_masked);
        }
        dev.run(fx);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 8)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 4)
    }

    fn endianness(&self) -> Endian {
        Endian::Little
    }
}

impl MmioOps for PbaOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        let Some(dev) = self.0.upgrade() else { return Ok(0) };
        let s = dev.lock();
        Ok(s.msix.as_ref().map_or(0, |m| read_long(&m.pba, offset as usize)))
    }

    fn write(
        &self,
        _cx: &AccessCtx,
        _offset: u64,
        _size: AccessSize,
        _value: u64,
    ) -> MemResult<()> {
        // The PBA is read-only. QEMU logs this as a guest error, which is off by default.
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 8)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 4)
    }

    fn endianness(&self) -> Endian {
        Endian::Little
    }
}

/// Where the MSI-X table and PBA go, the BAR arguments of `msix_init()`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MsixLayout {
    /// The region backing the table's BAR. The table is mapped inside it.
    pub table_bar: RegionId,
    pub table_bar_nr: u8,
    pub table_offset: u32,
    /// The region backing the PBA's BAR. It may be the same as `table_bar`.
    pub pba_bar: RegionId,
    pub pba_bar_nr: u8,
    pub pba_offset: u32,
}

impl PciDevice {
    /// `msix_init()`: adds an MSI-X capability with `nentries` vectors at `cap_pos` (0 for
    /// anywhere) and maps the table and PBA into the given BAR regions. Every vector starts
    /// masked. Returns the capability offset.
    pub fn msix_init(&self, nentries: u32, layout: MsixLayout, cap_pos: u8) -> Result<u8, Error> {
        if !self.bus().is_some_and(|b| b.msi_nonbroken()) {
            return Err(Error::generic("MSI-X is not supported by interrupt controller"));
        }
        if nentries < 1 || nentries > u32::from(PCI_MSIX_FLAGS_QSIZE) + 1 {
            return Err(Error::generic("The number of MSI-X vectors is invalid"));
        }
        let mem = self.memory();
        let table_size = nentries * PCI_MSIX_ENTRY_SIZE as u32;
        let pba_size = pba_size(nentries);
        let bar_size = |r: RegionId| mem.region(r).map_or(0, |i| i.size);
        let (to, po) = (u64::from(layout.table_offset), u64::from(layout.pba_offset));
        if (layout.table_bar_nr == layout.pba_bar_nr
            && ranges_overlap(to, table_size.into(), po, pba_size.into()))
            || u128::from(to + u64::from(table_size)) > bar_size(layout.table_bar)
            || u128::from(po + u64::from(pba_size)) > bar_size(layout.pba_bar)
            || (layout.table_offset | layout.pba_offset) & PCI_MSIX_FLAGS_BIRMASK != 0
        {
            return Err(Error::generic(
                "table & pba overlap, or they don't fit in BARs, or don't align",
            ));
        }

        let mut fx = Effects::new();
        let cap = {
            let mut s = self.lock();
            let cap =
                self.add_capability_locked(&mut s, PCI_CAP_ID_MSIX, cap_pos, MSIX_CAP_LENGTH)?;
            let c = usize::from(cap);
            s.set_word(c + PCI_MSIX_FLAGS, (nentries - 1) as u16);
            s.set_long(c + PCI_MSIX_TABLE, layout.table_offset | u32::from(layout.table_bar_nr));
            s.set_long(c + PCI_MSIX_PBA, layout.pba_offset | u32::from(layout.pba_bar_nr));
            s.wmask[c + MSIX_CONTROL_OFFSET] |= MSIX_ENABLE_MASK | MSIX_MASKALL_MASK;

            let table_mmio = mem
                .new_io("msix-table", table_size.into(), Arc::new(TableOps(self.weak())))
                .map_err(|e| Error::generic(e.to_string()))?;
            let pba_mmio = mem
                .new_io("msix-pba", pba_size.into(), Arc::new(PbaOps(self.weak())))
                .map_err(|e| Error::generic(e.to_string()))?;
            s.msix = Some(Msix {
                cap,
                entries: nentries,
                table: vec![0; table_size as usize],
                pba: vec![0; pba_size as usize],
                used: vec![0; nentries as usize],
                function_masked: true,
                table_bar: layout.table_bar,
                table_mmio,
                pba_bar: layout.pba_bar,
                pba_mmio,
                exclusive_bar: None,
            });
            s.msix_mask_all(&mut fx);
            let _t = mem.transaction();
            mem.add_subregion(layout.table_bar, to, table_mmio)
                .and_then(|()| mem.add_subregion(layout.pba_bar, po, pba_mmio))
                .map_err(|e| Error::generic(e.to_string()))?;
            cap
        };
        self.run(fx);
        Ok(cap)
    }

    /// `msix_init_exclusive_bar()`: creates a memory BAR `bar_nr` holding just the table (in
    /// the lower half of at least 4 KiB) and the PBA, and registers it.
    pub fn msix_init_exclusive_bar(&self, nentries: u32, bar_nr: u8) -> Result<u8, Error> {
        if nentries < 1 || nentries > u32::from(PCI_MSIX_FLAGS_QSIZE) + 1 {
            return Err(Error::generic("The number of MSI-X vectors is invalid"));
        }
        let mut bar_size: u32 = 4096;
        let mut bar_pba_offset = bar_size / 2;
        let bar_pba_size = pba_size(nentries);
        // Migration compatibility wants a 4 KiB BAR with the table in the lower half and the
        // PBA in the upper half for up to 128 vectors.
        if nentries * PCI_MSIX_ENTRY_SIZE as u32 > bar_pba_offset {
            bar_pba_offset = nentries * PCI_MSIX_ENTRY_SIZE as u32;
        }
        if bar_pba_offset + bar_pba_size > 4096 {
            bar_size = bar_pba_offset + bar_pba_size;
        }
        let bar_size = bar_size.next_power_of_two();

        let mem = self.memory();
        let bar = mem
            .new_container(&format!("{}-msix", self.name()), bar_size.into())
            .map_err(|e| Error::generic(e.to_string()))?;
        let layout = MsixLayout {
            table_bar: bar,
            table_bar_nr: bar_nr,
            table_offset: 0,
            pba_bar: bar,
            pba_bar_nr: bar_nr,
            pba_offset: bar_pba_offset,
        };
        let cap = match self.msix_init(nentries, layout, 0) {
            Ok(cap) => cap,
            Err(e) => {
                let _ = mem.destroy_region(bar);
                return Err(e);
            }
        };
        if let Some(m) = self.lock().msix.as_mut() {
            m.exclusive_bar = Some((usize::from(bar_nr), bar));
        }
        self.register_bar(usize::from(bar_nr), PCI_BASE_ADDRESS_SPACE_MEMORY, bar);
        Ok(cap)
    }

    /// `msix_uninit()`, also covering `msix_uninit_exclusive_bar()`.
    pub fn msix_uninit(&self) {
        let mut s = self.lock();
        let Some(m) = s.msix.take() else { return };
        crate::device::del_capability_locked(&mut s, PCI_CAP_ID_MSIX, MSIX_CAP_LENGTH);
        drop(s);
        let mem = self.memory();
        {
            let _t = mem.transaction();
            let _ = mem.del_subregion(m.pba_bar, m.pba_mmio);
            let _ = mem.del_subregion(m.table_bar, m.table_mmio);
        }
        let _ = mem.destroy_region(m.pba_mmio);
        let _ = mem.destroy_region(m.table_mmio);
    }

    /// `msix_present()`.
    pub fn msix_present(&self) -> bool {
        self.lock().msix.is_some()
    }

    /// `msix_enabled()`.
    pub fn msix_enabled(&self) -> bool {
        self.lock().msix_enabled()
    }

    /// The offset of the MSI-X capability, `PCIDevice::msix_cap`, 0 if there is none.
    pub fn msix_cap(&self) -> u8 {
        self.lock().msix.as_ref().map_or(0, |m| m.cap)
    }

    /// `msix_nr_vectors_allocated()`.
    pub fn msix_nr_vectors_allocated(&self) -> u32 {
        self.lock().msix.as_ref().map_or(0, |m| m.entries)
    }

    /// `msix_get_message()`.
    pub fn msix_get_message(&self, vector: u32) -> MsiMessage {
        self.lock().msix_ref().message(vector)
    }

    /// `msix_set_message()`: fills in a table entry and unmasks it.
    pub fn msix_set_message(&self, vector: u32, msg: MsiMessage) {
        let mut s = self.lock();
        let m = s.msix_mut();
        let e = vector as usize * PCI_MSIX_ENTRY_SIZE;
        pci_set_quad(&mut m.table, e + PCI_MSIX_ENTRY_LOWER_ADDR, msg.address);
        pci_set_long(&mut m.table, e + PCI_MSIX_ENTRY_DATA, msg.data);
        m.table[e + PCI_MSIX_ENTRY_VECTOR_CTRL] &= !PCI_MSIX_ENTRY_CTRL_MASKBIT;
    }

    /// `msix_is_masked()`.
    pub fn msix_is_masked(&self, vector: u32) -> bool {
        self.lock().msix_ref().is_masked(vector)
    }

    /// `msix_is_pending()`.
    pub fn msix_is_pending(&self, vector: u32) -> bool {
        self.lock().msix_ref().is_pending(vector)
    }

    /// `msix_set_pending()`.
    pub fn msix_set_pending(&self, vector: u32) {
        self.lock().msix_mut().set_pending(vector);
    }

    /// `msix_clr_pending()`.
    pub fn msix_clr_pending(&self, vector: u32) {
        self.lock().msix_mut().clr_pending(vector);
    }

    /// `msix_set_mask()`: sets or clears a vector's mask bit. Unmasking a pending vector sends
    /// it.
    pub fn msix_set_mask(&self, vector: u32, mask: bool) {
        let mut fx = Effects::new();
        {
            let mut s = self.lock();
            let m = s.msix_mut();
            assert!(vector < m.entries, "MSI-X vector {vector} out of range");
            let was_masked = m.is_masked(vector);
            let off = Msix::ctrl_off(vector);
            if mask {
                m.table[off] |= PCI_MSIX_ENTRY_CTRL_MASKBIT;
            } else {
                m.table[off] &= !PCI_MSIX_ENTRY_CTRL_MASKBIT;
            }
            s.msix_handle_mask_update(&mut fx, vector, was_masked);
        }
        self.run(fx);
    }

    /// `msix_notify()`: sends `vector` if it is in use, or marks it pending if masked.
    pub fn msix_notify(&self, vector: u32) {
        let mut fx = Effects::new();
        self.lock().msix_notify(&mut fx, vector);
        self.run(fx);
    }

    /// `msix_vector_use()`: marks a vector as driven by the model. Unused vectors never fire.
    pub fn msix_vector_use(&self, vector: u32) {
        let mut s = self.lock();
        let m = s.msix_mut();
        assert!(vector < m.entries, "MSI-X vector {vector} out of range");
        m.used[vector as usize] += 1;
    }

    /// `msix_vector_unuse()`.
    pub fn msix_vector_unuse(&self, vector: u32) {
        let mut s = self.lock();
        let m = s.msix_mut();
        assert!(vector < m.entries, "MSI-X vector {vector} out of range");
        let used = &mut m.used[vector as usize];
        if *used == 0 {
            return;
        }
        *used -= 1;
        if *used == 0 {
            m.clr_pending(vector);
        }
    }

    /// `msix_unuse_all_vectors()`.
    pub fn msix_unuse_all_vectors(&self) {
        let mut s = self.lock();
        let Some(m) = s.msix.as_mut() else { return };
        for vector in 0..m.entries {
            m.used[vector as usize] = 0;
            m.clr_pending(vector);
        }
    }

    /// `msix_save()`: the vector table and the pending bits, `nentries * 16` and
    /// `DIV_ROUND_UP(nentries, 8)` bytes. Both are empty without MSI-X.
    pub fn msix_vmstate_save(&self) -> (Vec<u8>, Vec<u8>) {
        let s = self.lock();
        let Some(m) = s.msix.as_ref() else { return (Vec::new(), Vec::new()) };
        let n = m.entries as usize;
        (m.table[..n * PCI_MSIX_ENTRY_SIZE].to_vec(), m.pba[..n.div_ceil(8)].to_vec())
    }

    /// `msix_load()`: puts back what [`msix_vmstate_save`](Self::msix_vmstate_save) returned
    /// and sends the vectors that are pending and no longer masked. The MSI-X control bits
    /// live in config space, which must be loaded first.
    pub fn msix_vmstate_load(&self, table: &[u8], pba: &[u8]) -> Result<(), String> {
        let mut fx = Effects::new();
        {
            let mut s = self.lock();
            let Some(m) = s.msix.as_mut() else {
                if table.is_empty() && pba.is_empty() {
                    return Ok(());
                }
                return Err("MSI-X state for a device without MSI-X".to_string());
            };
            let entries = m.entries;
            let n = entries as usize;
            if table.len() != n * PCI_MSIX_ENTRY_SIZE || pba.len() != n.div_ceil(8) {
                return Err(format!(
                    "MSI-X state of {} table and {} pending bytes for {n} vectors",
                    table.len(),
                    pba.len()
                ));
            }
            for vector in 0..entries {
                m.clr_pending(vector);
            }
            m.table[..table.len()].copy_from_slice(table);
            m.pba[..pba.len()].copy_from_slice(pba);
            s.msix_update_function_masked();
            for vector in 0..entries {
                s.msix_handle_mask_update(&mut fx, vector, true);
            }
        }
        self.run(fx);
        Ok(())
    }

    /// The BAR index created by [`PciDevice::msix_init_exclusive_bar`], if any.
    pub fn msix_exclusive_bar(&self) -> Option<usize> {
        self.lock().msix.as_ref().and_then(|m| m.exclusive_bar.map(|(n, _)| n))
    }
}
