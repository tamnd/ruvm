// SPDX-License-Identifier: GPL-2.0-or-later

//! The user space IOAPIC, from hw/intc/ioapic.c, hw/intc/ioapic_common.c,
//! hw/intc/ioapic_internal.h and include/hw/intc/ioapic.h.
//!
//! Each of the 24 input pins has a redirection entry. When a pin fires, `ioapic_service` turns
//! the entry into an MSI style message and writes it to the APIC window through the handler given
//! to [`IoApic::realize`], standing in for `address_space_stl_le()` on the machine's
//! `ioapic_as`. Level triggered entries keep remote IRR set until an EOI for their vector comes
//! back through [`IoApics::eoi_broadcast`], either from the LAPIC or from the version 0x20 EOI
//! register.
//!
//! Messages are delivered after the device lock is released, so the handler may call back into
//! the IOAPIC.
//!
//! VMState, trace points, QOM registration, the KVM in-kernel ioapic and the split irqchip KVM
//! route updates are not ported.

use std::fmt;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use ruvm_base::Error;
use ruvm_hw_core::irq::IrqLine;
use ruvm_hw_core::timer::{Clock, NANOSECONDS_PER_SECOND, Timer};
use ruvm_mem::{AccessCtx, AccessSize, MemResult, MmioOps};

pub const IOAPIC_NUM_PINS: usize = 24;
pub const IO_APIC_DEFAULT_ADDRESS: u64 = 0xfec0_0000;
pub const IO_APIC_SECONDARY_ADDRESS: u64 = IO_APIC_DEFAULT_ADDRESS + 0x10000;
/// Primary 0 to 23, secondary 24 to 47.
pub const IO_APIC_SECONDARY_IRQBASE: u32 = 24;

pub const TYPE_KVM_IOAPIC: &str = "kvm-ioapic";
pub const TYPE_IOAPIC: &str = "ioapic";
pub const TYPE_IOAPIC_COMMON: &str = "ioapic-common";

pub const MAX_IOAPICS: usize = 2;

pub const IOAPIC_LVT_DEST_SHIFT: u32 = 56;
pub const IOAPIC_LVT_DEST_IDX_SHIFT: u32 = 48;
pub const IOAPIC_LVT_MASKED_SHIFT: u32 = 16;
pub const IOAPIC_LVT_TRIGGER_MODE_SHIFT: u32 = 15;
pub const IOAPIC_LVT_REMOTE_IRR_SHIFT: u32 = 14;
pub const IOAPIC_LVT_POLARITY_SHIFT: u32 = 13;
pub const IOAPIC_LVT_DELIV_STATUS_SHIFT: u32 = 12;
pub const IOAPIC_LVT_DEST_MODE_SHIFT: u32 = 11;
pub const IOAPIC_LVT_DELIV_MODE_SHIFT: u32 = 8;

pub const IOAPIC_LVT_MASKED: u64 = 1 << IOAPIC_LVT_MASKED_SHIFT;
pub const IOAPIC_LVT_TRIGGER_MODE: u64 = 1 << IOAPIC_LVT_TRIGGER_MODE_SHIFT;
pub const IOAPIC_LVT_REMOTE_IRR: u64 = 1 << IOAPIC_LVT_REMOTE_IRR_SHIFT;
pub const IOAPIC_LVT_POLARITY: u64 = 1 << IOAPIC_LVT_POLARITY_SHIFT;
pub const IOAPIC_LVT_DELIV_STATUS: u64 = 1 << IOAPIC_LVT_DELIV_STATUS_SHIFT;
pub const IOAPIC_LVT_DEST_MODE: u64 = 1 << IOAPIC_LVT_DEST_MODE_SHIFT;
pub const IOAPIC_LVT_DELIV_MODE: u64 = 7 << IOAPIC_LVT_DELIV_MODE_SHIFT;

/// Bits that are read-only for IOAPIC entry.
pub const IOAPIC_RO_BITS: u64 = IOAPIC_LVT_REMOTE_IRR | IOAPIC_LVT_DELIV_STATUS;
pub const IOAPIC_RW_BITS: u64 = !IOAPIC_RO_BITS;

pub const IOAPIC_TRIGGER_EDGE: u8 = 0;
pub const IOAPIC_TRIGGER_LEVEL: u8 = 1;

// io{apic,sapic} delivery mode
pub const IOAPIC_DM_FIXED: u8 = 0x0;
pub const IOAPIC_DM_LOWEST_PRIORITY: u8 = 0x1;
pub const IOAPIC_DM_PMI: u8 = 0x2;
pub const IOAPIC_DM_NMI: u8 = 0x4;
pub const IOAPIC_DM_INIT: u8 = 0x5;
pub const IOAPIC_DM_SIPI: u8 = 0x6;
pub const IOAPIC_DM_EXTINT: u8 = 0x7;
pub const IOAPIC_DM_MASK: u8 = 0x7;

pub const IOAPIC_VECTOR_MASK: u64 = 0xff;

pub const IOAPIC_IOREGSEL: u64 = 0x00;
pub const IOAPIC_IOWIN: u64 = 0x10;
pub const IOAPIC_EOI: u64 = 0x40;

pub const IOAPIC_REG_ID: u8 = 0x00;
pub const IOAPIC_REG_VER: u8 = 0x01;
pub const IOAPIC_REG_ARB: u8 = 0x02;
pub const IOAPIC_REG_REDTBL_BASE: u8 = 0x10;
pub const IOAPIC_ID: u8 = 0x00;

pub const IOAPIC_ID_SHIFT: u32 = 24;
pub const IOAPIC_ID_MASK: u32 = 0xf;

pub const IOAPIC_VER_ENTRIES_SHIFT: u32 = 16;

/// The default of the `version` property.
pub const IOAPIC_VER_DEF: u8 = 0x20;

/// `SUCCESSIVE_IRQ_MAX_COUNT`: EOIs in a row that find the line still asserted before the
/// redelivery is delayed.
pub const SUCCESSIVE_IRQ_MAX_COUNT: i32 = 10000;

/// `APIC_DEFAULT_ADDRESS` from target/i386/cpu.h.
pub const APIC_DEFAULT_ADDRESS: u32 = 0xfee0_0000;

// From hw/i386/apic-msidef.h.
pub const MSI_DATA_VECTOR_SHIFT: u32 = 0;
pub const MSI_DATA_DELIVERY_MODE_SHIFT: u32 = 8;
pub const MSI_DATA_TRIGGER_SHIFT: u32 = 15;
pub const MSI_ADDR_DEST_MODE_SHIFT: u32 = 2;
pub const MSI_ADDR_DEST_IDX_SHIFT: u32 = 4;

/// Where interrupt messages go: a 32 bit little endian store of `data` at `addr` in the
/// IOAPIC's address space, normally the LAPIC window or an interrupt remapping IOMMU.
pub type IoApicMsiHandler = Arc<dyn Fn(u64, u32) + Send + Sync>;

/// `pic_read_irq(isa_pic)`, asked for the vector of an ExtINT entry. It must not call back into
/// the IOAPIC.
pub type PicReadIrq = Arc<dyn Fn() -> u8 + Send + Sync>;

/// `struct ioapic_entry_info`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct IoApicEntryInfo {
    // fields parsed from IOAPIC entries
    pub masked: u8,
    pub trig_mode: u8,
    pub dest_idx: u16,
    pub dest_mode: u8,
    pub delivery_mode: u8,
    pub vector: u8,

    // MSI message generated from above parsed fields
    pub addr: u32,
    pub data: u32,
}

/// `ioapic_entry_parse()`. `pic_read_irq` supplies the vector for ExtINT; without it the entry's
/// own vector field is used.
pub fn ioapic_entry_parse(entry: u64, pic_read_irq: Option<&PicReadIrq>) -> IoApicEntryInfo {
    let mut info = IoApicEntryInfo {
        masked: ((entry >> IOAPIC_LVT_MASKED_SHIFT) & 1) as u8,
        trig_mode: ((entry >> IOAPIC_LVT_TRIGGER_MODE_SHIFT) & 1) as u8,
        // By default, this would be dest_id[8] + reserved[8]. When IR is enabled, this would be
        // interrupt_index[15] + interrupt_format[1]. This field never means anything, but only
        // used to generate corresponding MSI.
        dest_idx: ((entry >> IOAPIC_LVT_DEST_IDX_SHIFT) & 0xffff) as u16,
        dest_mode: ((entry >> IOAPIC_LVT_DEST_MODE_SHIFT) & 1) as u8,
        delivery_mode: ((entry >> IOAPIC_LVT_DELIV_MODE_SHIFT) as u8) & IOAPIC_DM_MASK,
        ..IoApicEntryInfo::default()
    };
    info.vector = match pic_read_irq {
        Some(read) if info.delivery_mode == IOAPIC_DM_EXTINT => read(),
        _ => (entry & IOAPIC_VECTOR_MASK) as u8,
    };

    info.addr = APIC_DEFAULT_ADDRESS
        | (u32::from(info.dest_idx) << MSI_ADDR_DEST_IDX_SHIFT)
        | (u32::from(info.dest_mode) << MSI_ADDR_DEST_MODE_SHIFT);
    info.data = (u32::from(info.vector) << MSI_DATA_VECTOR_SHIFT)
        | (u32::from(info.trig_mode) << MSI_DATA_TRIGGER_SHIFT)
        | (u32::from(info.delivery_mode) << MSI_DATA_DELIVERY_MODE_SHIFT);
    info
}

/// The machine's list of IOAPICs, the `ioapics[]` array and `ioapic_no` counter. EOIs are
/// broadcast to every member.
#[derive(Clone, Debug, Default)]
pub struct IoApics(Arc<Mutex<Vec<Weak<IoApic>>>>);

impl IoApics {
    pub fn new() -> Self {
        Self::default()
    }

    fn list(&self) -> Vec<Arc<IoApic>> {
        let l = self.0.lock().unwrap_or_else(|p| p.into_inner());
        l.iter().filter_map(Weak::upgrade).collect()
    }

    /// `ioapic_eoi_broadcast()`: the LAPIC finished a level triggered interrupt with `vector`.
    pub fn eoi_broadcast(&self, vector: i32) {
        for s in self.list() {
            s.eoi(vector);
        }
    }
}

/// `IOAPICCommonState` minus what never changes after realize.
#[derive(Debug)]
struct IoApicState {
    id: u8,
    ioregsel: u8,
    irr: u32,
    ioredtbl: [u64; IOAPIC_NUM_PINS],
    irq_count: [u64; IOAPIC_NUM_PINS],
    irq_level: [i32; IOAPIC_NUM_PINS],
    irq_eoi: [i32; IOAPIC_NUM_PINS],
}

/// Messages collected by `ioapic_service()` and sent once the lock is dropped.
type Pending = Vec<(u64, u32)>;

/// The `ioapic` device.
pub struct IoApic {
    clock: Arc<Clock>,
    state: Mutex<IoApicState>,
    version: u8,
    msi: IoApicMsiHandler,
    pic_read_irq: Mutex<Option<PicReadIrq>>,
    delayed_ioapic_service_timer: Timer,
    ioapics: IoApics,
}

impl fmt::Debug for IoApic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IoApic")
            .field("version", &self.version)
            .field("state", &*self.lock())
            .finish_non_exhaustive()
    }
}

impl IoApic {
    /// `ioapic_common_realize()` and `ioapic_realize()`, then `ioapic_reset_common()`. The new
    /// IOAPIC joins `ioapics`. `msi` receives every interrupt message.
    pub fn realize(
        clock: &Arc<Clock>,
        version: u8,
        ioapics: &IoApics,
        msi: IoApicMsiHandler,
    ) -> Result<Arc<IoApic>, Error> {
        let mut list = ioapics.0.lock().unwrap_or_else(|p| p.into_inner());
        if list.len() >= MAX_IOAPICS {
            return Err(Error::generic(format!("Only {MAX_IOAPICS} ioapics allowed")));
        }
        if version != 0x11 && version != 0x20 {
            return Err(Error::generic(format!(
                "IOAPIC only supports version 0x11 or 0x20 (default: 0x{IOAPIC_VER_DEF:x})."
            )));
        }
        let s = Arc::new_cyclic(|weak: &Weak<IoApic>| {
            let w = weak.clone();
            let delayed_ioapic_service_timer = clock.new_timer(move || {
                if let Some(s) = w.upgrade() {
                    s.delayed_ioapic_service_cb();
                }
            });
            IoApic {
                clock: clock.clone(),
                state: Mutex::new(IoApicState {
                    id: 0,
                    ioregsel: 0,
                    irr: 0,
                    ioredtbl: [0; IOAPIC_NUM_PINS],
                    irq_count: [0; IOAPIC_NUM_PINS],
                    irq_level: [0; IOAPIC_NUM_PINS],
                    irq_eoi: [0; IOAPIC_NUM_PINS],
                }),
                version,
                msi,
                pic_read_irq: Mutex::new(None),
                delayed_ioapic_service_timer,
                ioapics: ioapics.clone(),
            }
        });
        list.push(Arc::downgrade(&s));
        drop(list);
        s.reset();
        Ok(s)
    }

    fn lock(&self) -> MutexGuard<'_, IoApicState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Connects the 8259 that answers ExtINT entries, `isa_pic`.
    pub fn set_pic_read_irq(&self, f: Option<PicReadIrq>) {
        *self.pic_read_irq.lock().unwrap_or_else(|p| p.into_inner()) = f;
    }

    /// The `version` property.
    pub fn version(&self) -> u8 {
        self.version
    }

    /// Input pin `n`, from `qdev_init_gpio_in(dev, ioapic_set_irq, IOAPIC_NUM_PINS)`.
    pub fn input(self: &Arc<Self>, n: u32) -> IrqLine {
        let w = Arc::downgrade(self);
        IrqLine::new(
            Arc::new(move |n, level| {
                if let Some(s) = w.upgrade() {
                    s.set_irq(n as i32, level);
                }
            }),
            n,
        )
    }

    /// All 24 input pins.
    pub fn inputs(self: &Arc<Self>) -> Vec<IrqLine> {
        (0..IOAPIC_NUM_PINS as u32).map(|n| self.input(n)).collect()
    }

    /// The redirection entry of `pin`.
    pub fn redirection_entry(&self, pin: usize) -> u64 {
        self.lock().ioredtbl[pin]
    }

    /// The IRR bitmap.
    pub fn irr(&self) -> u32 {
        self.lock().irr
    }

    /// `ioapic_get_statistics()`: how often each pin went high.
    pub fn irq_count(&self) -> [u64; IOAPIC_NUM_PINS] {
        self.lock().irq_count
    }

    fn deliver(&self, pending: Pending) {
        for (addr, data) in pending {
            (self.msi)(addr, data);
        }
    }

    /// `ioapic_service()`.
    fn service(&self, s: &mut IoApicState, out: &mut Pending) {
        let pic = self.pic_read_irq.lock().unwrap_or_else(|p| p.into_inner()).clone();
        for i in 0..IOAPIC_NUM_PINS {
            let mask = 1u32 << i;
            if s.irr & mask == 0 {
                continue;
            }
            let mut coalesce = false;
            let entry = s.ioredtbl[i];
            let info = ioapic_entry_parse(entry, pic.as_ref());
            if info.masked != 0 {
                continue;
            }
            if info.trig_mode == IOAPIC_TRIGGER_EDGE {
                s.irr &= !mask;
            } else {
                coalesce = s.ioredtbl[i] & IOAPIC_LVT_REMOTE_IRR != 0;
                s.ioredtbl[i] |= IOAPIC_LVT_REMOTE_IRR;
            }

            if coalesce {
                // We are level triggered interrupts, and the guest should be still working on
                // previous one, so skip it.
                continue;
            }

            // No matter whether IR is enabled, we translate the IOAPIC message into a MSI one,
            // and its address space will decide whether we need a translation.
            out.push((u64::from(info.addr), info.data));
        }
    }

    /// `delayed_ioapic_service_cb()`.
    fn delayed_ioapic_service_cb(&self) {
        let mut out = Pending::new();
        self.service(&mut self.lock(), &mut out);
        self.deliver(out);
    }

    /// `ioapic_stat_update_irq()`.
    fn stat_update_irq(s: &mut IoApicState, irq: usize, level: i32) {
        if level != s.irq_level[irq] {
            s.irq_level[irq] = level;
            if level == 1 {
                s.irq_count[irq] += 1;
            }
        }
    }

    /// `ioapic_set_irq()`.
    pub fn set_irq(&self, vector: i32, level: i32) {
        let mut out = Pending::new();
        {
            let mut guard = self.lock();
            let s = &mut *guard;
            // ISA IRQs map to GSI 1-1 except for IRQ0 which maps to GSI 2. GSI maps to ioapic
            // 1-1. This is not the cleanest way of doing it but it should work.
            if vector >= 0 && (vector as usize) < IOAPIC_NUM_PINS {
                Self::stat_update_irq(s, vector as usize, level);
            }
            let vector = if vector == 0 { 2 } else { vector };
            if vector >= 0 && (vector as usize) < IOAPIC_NUM_PINS {
                let n = vector as usize;
                let mask = 1u32 << n;
                let entry = s.ioredtbl[n];

                if ((entry >> IOAPIC_LVT_TRIGGER_MODE_SHIFT) & 1) as u8 == IOAPIC_TRIGGER_LEVEL {
                    // level triggered
                    if level != 0 {
                        s.irr |= mask;
                        if entry & IOAPIC_LVT_REMOTE_IRR == 0 {
                            self.service(s, &mut out);
                        }
                    } else {
                        s.irr &= !mask;
                    }
                } else if level != 0 && entry & IOAPIC_LVT_MASKED == 0 {
                    // According to the 82093AA manual, we must ignore edge requests if the input
                    // pin is masked.
                    s.irr |= mask;
                    self.service(s, &mut out);
                }
            }
        }
        self.deliver(out);
    }

    /// The part of `ioapic_eoi_broadcast()` that looks at one IOAPIC.
    fn eoi(&self, vector: i32) {
        let mut out = Pending::new();
        {
            let mut guard = self.lock();
            let s = &mut *guard;
            for n in 0..IOAPIC_NUM_PINS {
                let entry = s.ioredtbl[n];

                if (entry & IOAPIC_VECTOR_MASK) as i32 != vector
                    || ((entry >> IOAPIC_LVT_TRIGGER_MODE_SHIFT) & 1) as u8 != IOAPIC_TRIGGER_LEVEL
                {
                    continue;
                }

                if entry & IOAPIC_LVT_REMOTE_IRR == 0 {
                    continue;
                }

                s.ioredtbl[n] = entry & !IOAPIC_LVT_REMOTE_IRR;

                if entry & IOAPIC_LVT_MASKED == 0 && s.irr & (1u32 << n) != 0 {
                    s.irq_eoi[n] += 1;
                    if s.irq_eoi[n] >= SUCCESSIVE_IRQ_MAX_COUNT {
                        // Real hardware does not deliver the interrupt immediately during eoi
                        // broadcast, and this lets a buggy guest make slow progress even if it
                        // does not correctly handle a level-triggered interrupt. Emulate this
                        // behavior if we detect an interrupt storm.
                        s.irq_eoi[n] = 0;
                        self.delayed_ioapic_service_timer
                            .modify_anticipate(self.clock.get_ns() + NANOSECONDS_PER_SECOND / 100);
                    } else {
                        self.service(s, &mut out);
                    }
                } else {
                    s.irq_eoi[n] = 0;
                }
            }
        }
        self.deliver(out);
    }

    /// `ioapic_mem_read()`.
    pub fn mmio_read(&self, addr: u64, size: u32) -> u64 {
        let s = self.lock();
        let mut val: u32 = 0;

        match addr & 0xff {
            IOAPIC_IOREGSEL => val = u32::from(s.ioregsel),
            IOAPIC_IOWIN => {
                if size != 4 {
                    return 0;
                }
                match s.ioregsel {
                    IOAPIC_REG_ID | IOAPIC_REG_ARB => val = u32::from(s.id) << IOAPIC_ID_SHIFT,
                    IOAPIC_REG_VER => {
                        val = u32::from(self.version)
                            | (((IOAPIC_NUM_PINS - 1) as u32) << IOAPIC_VER_ENTRIES_SHIFT);
                    }
                    sel => {
                        let index = (i32::from(sel) - i32::from(IOAPIC_REG_REDTBL_BASE)) >> 1;
                        if index >= 0 && (index as usize) < IOAPIC_NUM_PINS {
                            let e = s.ioredtbl[index as usize];
                            val = if sel & 1 != 0 { (e >> 32) as u32 } else { e as u32 };
                        }
                    }
                }
            }
            _ => {}
        }
        u64::from(val)
    }

    /// `ioapic_mem_write()`.
    pub fn mmio_write(&self, addr: u64, size: u32, val: u64) {
        let mut out = Pending::new();
        match addr & 0xff {
            IOAPIC_IOREGSEL => self.lock().ioregsel = val as u8,
            IOAPIC_IOWIN => {
                if size != 4 {
                    return;
                }
                let mut guard = self.lock();
                let s = &mut *guard;
                match s.ioregsel {
                    IOAPIC_REG_ID => {
                        s.id = ((val as u32 >> IOAPIC_ID_SHIFT) & IOAPIC_ID_MASK) as u8
                    }
                    IOAPIC_REG_VER | IOAPIC_REG_ARB => {}
                    sel => {
                        let index = (i32::from(sel) - i32::from(IOAPIC_REG_REDTBL_BASE)) >> 1;
                        if index >= 0 && (index as usize) < IOAPIC_NUM_PINS {
                            let index = index as usize;
                            let ro_bits = s.ioredtbl[index] & IOAPIC_RO_BITS;
                            if sel & 1 != 0 {
                                s.ioredtbl[index] &= 0xffff_ffff;
                                s.ioredtbl[index] |= u64::from(val as u32) << 32;
                            } else {
                                s.ioredtbl[index] &= !0xffff_ffffu64;
                                s.ioredtbl[index] |= u64::from(val as u32);
                            }
                            // restore RO bits
                            s.ioredtbl[index] &= IOAPIC_RW_BITS;
                            s.ioredtbl[index] |= ro_bits;
                            s.irq_eoi[index] = 0;
                            ioapic_fix_edge_remote_irr(&mut s.ioredtbl[index]);
                            self.service(s, &mut out);
                        }
                    }
                }
            }
            IOAPIC_EOI => {
                // Explicit EOI is only supported for IOAPIC version 0x20
                if size != 4 || self.version != 0x20 {
                    return;
                }
                self.ioapics.eoi_broadcast(val as i32);
            }
            _ => {}
        }
        self.deliver(out);
    }

    /// `ioapic_reset_common()`.
    pub fn reset(&self) {
        let mut s = self.lock();
        s.id = 0;
        s.ioregsel = 0;
        s.irr = 0;
        for e in s.ioredtbl.iter_mut() {
            *e = 1 << IOAPIC_LVT_MASKED_SHIFT;
        }
    }

    /// `ioapic_print_redtbl()`, the `info pic` text.
    pub fn print_info(&self) -> String {
        const DELM_STR: [&str; 8] =
            ["fixed", "lowest", "SMI", "...", "NMI", "INIT", "...", "extINT"];
        let s = self.lock();
        let mut buf = String::new();
        let mut remote_irr = 0u32;

        let _ = write!(
            buf,
            "ioapic0: ver=0x{:x} id=0x{:02x} sel=0x{:02x}",
            self.version, s.id, s.ioregsel
        );
        if s.ioregsel != 0 {
            let _ = writeln!(
                buf,
                " (redir[{}])",
                (u32::from(s.ioregsel).wrapping_sub(u32::from(IOAPIC_REG_REDTBL_BASE))) >> 1
            );
        } else {
            buf.push('\n');
        }
        for (i, &entry) in s.ioredtbl.iter().enumerate() {
            let delm = ((entry & IOAPIC_LVT_DELIV_MODE) >> IOAPIC_LVT_DELIV_MODE_SHIFT) as usize;
            let dest_mask = if entry & IOAPIC_LVT_DEST_MODE != 0 { 0xff } else { 0xf };
            let _ = writeln!(
                buf,
                "  pin {:<2} 0x{:016x} dest={:x} vec={:<3} {} {:<5} {:<6} {:<6} {}",
                i,
                entry,
                (entry >> IOAPIC_LVT_DEST_SHIFT) & dest_mask,
                entry & IOAPIC_VECTOR_MASK,
                if entry & IOAPIC_LVT_POLARITY != 0 { "active-lo" } else { "active-hi" },
                if entry & IOAPIC_LVT_TRIGGER_MODE != 0 { "level" } else { "edge" },
                if entry & IOAPIC_LVT_MASKED != 0 { "masked" } else { "" },
                DELM_STR[delm],
                if entry & IOAPIC_LVT_DEST_MODE != 0 { "logical" } else { "physical" },
            );
            if entry & IOAPIC_LVT_TRIGGER_MODE != 0 && entry & IOAPIC_LVT_REMOTE_IRR != 0 {
                remote_irr |= 1 << i;
            }
        }
        ioapic_irr_dump(&mut buf, "  IRR", s.irr);
        ioapic_irr_dump(&mut buf, "  Remote IRR", remote_irr);
        buf
    }
}

/// `ioapic_fix_edge_remote_irr()`.
fn ioapic_fix_edge_remote_irr(entry: &mut u64) {
    if *entry & IOAPIC_LVT_TRIGGER_MODE == 0 {
        // Edge-triggered interrupts, make sure remote IRR is zero
        *entry &= !IOAPIC_LVT_REMOTE_IRR;
    }
}

/// `ioapic_irr_dump()`.
fn ioapic_irr_dump(buf: &mut String, name: &str, bitmap: u32) {
    let _ = write!(buf, "{name:<10} ");
    if bitmap == 0 {
        buf.push_str("(none)\n");
        return;
    }
    for i in 0..IOAPIC_NUM_PINS {
        if bitmap & (1 << i) != 0 {
            let _ = write!(buf, "{i:<2} ");
        }
    }
    buf.push('\n');
}

/// `ioapic_io_ops`: no size limits of its own, so the default 1 to 4 byte implementation.
impl MmioOps for IoApic {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(self.mmio_read(offset, size.bytes()))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        self.mmio_write(offset, size.bytes(), value);
        Ok(())
    }
}
