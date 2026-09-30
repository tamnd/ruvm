// SPDX-License-Identifier: GPL-2.0-or-later

//! The MSI capability, from hw/pci/msi.c.
//!
//! The capability lives entirely in config space: flags, a 32 or 64 bit address, 16 bits of
//! data and, with per vector masking, a mask and a pending register. [`PciDevice::msi_notify`]
//! composes the message for a vector, or sets its pending bit when the vector is masked. A
//! pending vector fires when it is unmasked, either through [`PciDevice::msi_set_mask`] or by
//! a guest config write.
//!
//! Messages go to the device's MSI trigger (see [`PciDevice::set_msi_trigger`]) or the root
//! bus's MSI handler, and only while the function is a bus master. The Xen and KVM routing
//! hooks are not ported.

use ruvm_base::Error;

use crate::device::{DevState, Effects, MsiMessage, PciDevice};
use crate::regs::*;

const PCI_MSI_ADDRESS_LO_MASK: u32 = !0x3;
const PCI_MSI_32_SIZEOF: u8 = 0x0a;
const PCI_MSI_64_SIZEOF: u8 = 0x0e;
const PCI_MSI_32M_SIZEOF: u8 = 0x14;
const PCI_MSI_64M_SIZEOF: u8 = 0x18;

/// `PCI_MSI_VECTORS_MAX`.
pub const PCI_MSI_VECTORS_MAX: u32 = 32;

const QSIZE_SHIFT: u32 = PCI_MSI_FLAGS_QSIZE.trailing_zeros();
const QMASK_SHIFT: u32 = PCI_MSI_FLAGS_QMASK.trailing_zeros();

/// `msi_cap_sizeof()`.
fn cap_sizeof(flags: u16) -> u8 {
    match (flags & PCI_MSI_FLAGS_MASKBIT != 0, flags & PCI_MSI_FLAGS_64BIT != 0) {
        (true, true) => PCI_MSI_64M_SIZEOF,
        (false, true) => PCI_MSI_64_SIZEOF,
        (true, false) => PCI_MSI_32M_SIZEOF,
        (false, false) => PCI_MSI_32_SIZEOF,
    }
}

/// `msi_nr_vectors()`.
fn nr_vectors(flags: u16) -> u32 {
    1 << ((flags & PCI_MSI_FLAGS_QSIZE) >> QSIZE_SHIFT)
}

impl DevState {
    fn msi_present(&self) -> bool {
        self.msi_cap != 0
    }

    fn msi_off(&self, reg: usize) -> usize {
        usize::from(self.msi_cap) + reg
    }

    fn msi_flags(&self) -> u16 {
        self.word(self.msi_off(PCI_MSI_FLAGS))
    }

    fn msi_data_off(&self, msi64bit: bool) -> usize {
        self.msi_off(if msi64bit { PCI_MSI_DATA_64 } else { PCI_MSI_DATA_32 })
    }

    fn msi_mask_off(&self, msi64bit: bool) -> usize {
        self.msi_off(if msi64bit { PCI_MSI_MASK_64 } else { PCI_MSI_MASK_32 })
    }

    fn msi_pending_off(&self, msi64bit: bool) -> usize {
        self.msi_off(if msi64bit { PCI_MSI_PENDING_64 } else { PCI_MSI_PENDING_32 })
    }

    pub(crate) fn msi_enabled(&self) -> bool {
        self.msi_present() && self.msi_flags() & PCI_MSI_FLAGS_ENABLE != 0
    }

    /// `msi_prepare_message()`.
    fn msi_get_message(&self, vector: u32) -> MsiMessage {
        let flags = self.msi_flags();
        let msi64bit = flags & PCI_MSI_FLAGS_64BIT != 0;
        let nr = nr_vectors(flags);
        assert!(vector < nr, "MSI vector {vector} out of range");
        let lo = self.msi_off(PCI_MSI_ADDRESS_LO);
        let address =
            if msi64bit { pci_get_quad(&self.config, lo) } else { u64::from(self.long(lo)) };
        // Bits 31:16 are zero.
        let mut data = u32::from(self.word(self.msi_data_off(msi64bit)));
        if nr > 1 {
            data &= !(nr - 1);
            data |= vector;
        }
        MsiMessage { address, data }
    }

    /// `msi_is_masked()`.
    fn msi_is_masked(&self, vector: u32) -> bool {
        let flags = self.msi_flags();
        assert!(vector < PCI_MSI_VECTORS_MAX);
        if flags & PCI_MSI_FLAGS_MASKBIT == 0 {
            return false;
        }
        let mask = self.long(self.msi_mask_off(flags & PCI_MSI_FLAGS_64BIT != 0));
        mask & (1 << vector) != 0
    }

    /// `msi_notify()`.
    pub(crate) fn msi_notify(&mut self, fx: &mut Effects, vector: u32) {
        let flags = self.msi_flags();
        let msi64bit = flags & PCI_MSI_FLAGS_64BIT != 0;
        assert!(vector < nr_vectors(flags), "MSI vector {vector} out of range");
        if self.msi_is_masked(vector) {
            let off = self.msi_pending_off(msi64bit);
            let v = self.long(off) | (1 << vector);
            self.set_long(off, v);
            return;
        }
        let msg = self.msi_get_message(vector);
        self.send_message(fx, msg);
    }

    /// `msi_reset()`.
    pub(crate) fn msi_reset(&mut self) {
        if !self.msi_present() {
            return;
        }
        let flags = self.msi_flags() & !(PCI_MSI_FLAGS_QSIZE | PCI_MSI_FLAGS_ENABLE);
        let msi64bit = flags & PCI_MSI_FLAGS_64BIT != 0;
        self.set_word(self.msi_off(PCI_MSI_FLAGS), flags);
        self.set_long(self.msi_off(PCI_MSI_ADDRESS_LO), 0);
        if msi64bit {
            self.set_long(self.msi_off(PCI_MSI_ADDRESS_HI), 0);
        }
        self.set_word(self.msi_data_off(msi64bit), 0);
        if flags & PCI_MSI_FLAGS_MASKBIT != 0 {
            self.set_long(self.msi_mask_off(msi64bit), 0);
            self.set_long(self.msi_pending_off(msi64bit), 0);
        }
    }

    /// `msi_write_config()`.
    pub(crate) fn msi_write_config(&mut self, fx: &mut Effects, addr: u32, _val: u32, len: u32) {
        if !self.msi_present() {
            return;
        }
        let mut flags = self.msi_flags();
        let msi64bit = flags & PCI_MSI_FLAGS_64BIT != 0;
        let per_vector_mask = flags & PCI_MSI_FLAGS_MASKBIT != 0;
        if !ranges_overlap(
            addr.into(),
            len.into(),
            u64::from(self.msi_cap),
            u64::from(cap_sizeof(flags)),
        ) {
            return;
        }
        if flags & PCI_MSI_FLAGS_ENABLE == 0 {
            return;
        }

        // MSI is now enabled, so drop INTx. A driver may not use the enable bit to mask a
        // request, but a guest could, so the interrupts are just discarded.
        self.deassert_intx(fx);

        // The guest may ask for more vectors than we offer, which the spec forbids. Clamp it.
        let log_num_vecs = (flags & PCI_MSI_FLAGS_QSIZE) >> QSIZE_SHIFT;
        let log_max_vecs = (flags & PCI_MSI_FLAGS_QMASK) >> QMASK_SHIFT;
        if log_num_vecs > log_max_vecs {
            flags &= !PCI_MSI_FLAGS_QSIZE;
            flags |= log_max_vecs << QSIZE_SHIFT;
            self.set_word(self.msi_off(PCI_MSI_FLAGS), flags);
        }

        // Without per vector masking nothing can be pending.
        if !per_vector_mask {
            return;
        }

        let nr = nr_vectors(flags);
        // This discards pending interrupts beyond the enabled vectors.
        let poff = self.msi_pending_off(msi64bit);
        let pending = self.long(poff) & (u32::MAX >> (PCI_MSI_VECTORS_MAX - nr));
        self.set_long(poff, pending);

        // Deliver pending interrupts that are unmasked.
        for vector in 0..nr {
            if self.msi_is_masked(vector) || pending & (1 << vector) == 0 {
                continue;
            }
            let v = self.long(poff) & !(1 << vector);
            self.set_long(poff, v);
            self.msi_notify(fx, vector);
        }
    }
}

impl PciDevice {
    /// `msi_init()`: adds an MSI capability with `nr_vectors` vectors (a power of two up to
    /// 32) at `offset`, or anywhere free if `offset` is 0. Returns the capability offset.
    pub fn msi_init(
        &self,
        offset: u8,
        nr_vectors: u32,
        msi64bit: bool,
        per_vector_mask: bool,
    ) -> Result<u8, Error> {
        if !self.bus().is_some_and(|b| b.msi_nonbroken()) {
            return Err(Error::generic("MSI is not supported by interrupt controller"));
        }
        assert!(nr_vectors.is_power_of_two(), "MSI vector count must be a power of 2");
        assert!(nr_vectors <= PCI_MSI_VECTORS_MAX);
        let vectors_order = nr_vectors.trailing_zeros() as u16;

        let mut flags = vectors_order << QMASK_SHIFT;
        if msi64bit {
            flags |= PCI_MSI_FLAGS_64BIT;
        }
        if per_vector_mask {
            flags |= PCI_MSI_FLAGS_MASKBIT;
        }

        let mut s = self.lock();
        let cap = self.add_capability_locked(&mut s, PCI_CAP_ID_MSI, offset, cap_sizeof(flags))?;
        s.msi_cap = cap;

        let flags_off = s.msi_off(PCI_MSI_FLAGS);
        s.set_word(flags_off, flags);
        pci_set_word(&mut s.wmask, flags_off, PCI_MSI_FLAGS_QSIZE | PCI_MSI_FLAGS_ENABLE);
        let lo = s.msi_off(PCI_MSI_ADDRESS_LO);
        pci_set_long(&mut s.wmask, lo, PCI_MSI_ADDRESS_LO_MASK);
        if msi64bit {
            let hi = s.msi_off(PCI_MSI_ADDRESS_HI);
            pci_set_long(&mut s.wmask, hi, 0xffff_ffff);
        }
        let data = s.msi_data_off(msi64bit);
        pci_set_word(&mut s.wmask, data, 0xffff);
        if per_vector_mask {
            // Mask bits 0 to nr_vectors - 1 are writable.
            let mask = s.msi_mask_off(msi64bit);
            pci_set_long(&mut s.wmask, mask, u32::MAX >> (PCI_MSI_VECTORS_MAX - nr_vectors));
        }
        Ok(cap)
    }

    /// `msi_uninit()`.
    pub fn msi_uninit(&self) {
        let mut s = self.lock();
        if !s.msi_present() {
            return;
        }
        let size = cap_sizeof(s.msi_flags());
        crate::device::del_capability_locked(&mut s, PCI_CAP_ID_MSI, size);
        s.msi_cap = 0;
    }

    /// `msi_present()`.
    pub fn msi_present(&self) -> bool {
        self.lock().msi_present()
    }

    /// The offset of the MSI capability, `PCIDevice::msi_cap`, 0 if there is none.
    pub fn msi_cap(&self) -> u8 {
        self.lock().msi_cap
    }

    /// `msi_enabled()`.
    pub fn msi_enabled(&self) -> bool {
        self.lock().msi_enabled()
    }

    /// `msi_nr_vectors_allocated()`: the vectors the guest enabled.
    pub fn msi_nr_vectors_allocated(&self) -> u32 {
        nr_vectors(self.lock().msi_flags())
    }

    /// `msi_get_message()`.
    pub fn msi_get_message(&self, vector: u32) -> MsiMessage {
        self.lock().msi_get_message(vector)
    }

    /// `msi_set_message()`: writes the address and data registers.
    pub fn msi_set_message(&self, msg: MsiMessage) {
        let mut s = self.lock();
        let msi64bit = s.msi_flags() & PCI_MSI_FLAGS_64BIT != 0;
        let lo = s.msi_off(PCI_MSI_ADDRESS_LO);
        if msi64bit {
            pci_set_quad(&mut s.config, lo, msg.address);
        } else {
            s.set_long(lo, msg.address as u32);
        }
        let data = s.msi_data_off(msi64bit);
        s.set_word(data, msg.data as u16);
    }

    /// `msi_is_masked()`.
    pub fn msi_is_masked(&self, vector: u32) -> bool {
        self.lock().msi_is_masked(vector)
    }

    /// `msi_set_mask()`: masks or unmasks a vector. Unmasking a pending vector sends it.
    pub fn msi_set_mask(&self, vector: u32, mask: bool) -> Result<(), Error> {
        if vector >= PCI_MSI_VECTORS_MAX {
            return Err(Error::generic(format!(
                "msi: vector {} not allocated. max vector is {}",
                vector,
                PCI_MSI_VECTORS_MAX - 1
            )));
        }
        let mut fx = Effects::new();
        {
            let mut s = self.lock();
            let msi64bit = s.msi_flags() & PCI_MSI_FLAGS_64BIT != 0;
            let vector_mask = 1u32 << vector;
            let moff = s.msi_mask_off(msi64bit);
            let mut irq_state = s.long(moff);
            if mask {
                irq_state |= vector_mask;
            } else {
                irq_state &= !vector_mask;
            }
            s.set_long(moff, irq_state);

            let poff = s.msi_pending_off(msi64bit);
            let pending = s.long(poff);
            if !mask && pending & vector_mask != 0 {
                s.set_long(poff, pending & !vector_mask);
                s.msi_notify(&mut fx, vector);
            }
        }
        self.run(fx);
        Ok(())
    }

    /// `msi_notify()`: sends `vector`, or marks it pending if it is masked. The caller checks
    /// [`PciDevice::msi_enabled`] first, as in QEMU.
    pub fn msi_notify(&self, vector: u32) {
        let mut fx = Effects::new();
        self.lock().msi_notify(&mut fx, vector);
        self.run(fx);
    }

    /// `msi_send_message()`: delivers a message as this function, if it is a bus master.
    pub fn msi_send_message(&self, msg: MsiMessage) {
        let mut fx = Effects::new();
        self.lock().send_message(&mut fx, msg);
        self.run(fx);
    }
}
