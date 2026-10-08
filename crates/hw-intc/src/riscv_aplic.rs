// SPDX-License-Identifier: GPL-2.0-or-later

//! The RISC-V advanced platform level interrupt controller, hw/intc/riscv_aplic.c and
//! include/hw/intc/riscv_aplic.h.
//!
//! [`RiscvAplic`] is `RISCVAPLICState`: one interrupt domain. Every source has a configuration
//! (`sourcecfg`: inactive, detached, edge or level triggered, or delegated to a child domain), a
//! pending bit, an enabled bit, the level of its input and a target. The root domain (the M
//! level one on the virt board) has the inputs; a source it delegates with `sourcecfg.D` goes to
//! the child domain named by the child index.
//!
//! In direct mode each hart has an interrupt delivery control (IDC) structure and an output:
//! the output is high while the domain and the IDC deliver interrupts and either `iforce` is
//! set or some source targeting the hart is pending, enabled and under the threshold. `claimi`
//! returns the best one and clears its pending bit. In MSI mode a pending and enabled source is
//! sent at once as a message: the identity of the target is stored, as a little endian word,
//! at the address the M level domain's `mmsicfgaddr` or `smsicfgaddr` registers give for the
//! target hart and guest. [`RiscvAplic::set_msi_sink`] gives the function that does the store.
//!
//! The register block takes aligned 4 byte accesses, little endian.
//!
//! Differences from QEMU:
//!
//! - Not ported: VMState, QOM properties and registration, and the KVM and split irqchip side.
//! - The `qemu_log_mask()` guest error messages and the trace points are not there.
//! - Realize claims `MIP_MEIP` or `MIP_SEIP` on the harts of a direct mode domain and sets
//!   `msi_nonbroken`. That is CPU and machine state the board owns, so the board must do it.
//! - One mutex covers each domain's state. Direct mode outputs are driven with it held, so the
//!   lines connected to them must not call back into the APLIC. MSIs are sent after the lock
//!   is dropped, with the M level domain's configuration read just before; a source delegated
//!   to a child is passed on after the parent's lock is dropped too. QEMU runs all of this
//!   under the BQL, so the order of events is the same.
//! - A failed MSI store is not reported (QEMU logs "MSI write failed").
//! - `riscv_aplic_create()` asserts on bad parameters. [`RiscvAplic::new`] panics on the same
//!   ones.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};

use ruvm_hw_core::irq::{IrqLine, IrqPin};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, Endian, MemResult, MmioOps};

/// `TYPE_RISCV_APLIC`.
pub const TYPE_RISCV_APLIC: &str = "riscv.aplic";

/// `APLIC_MIN_SIZE`.
pub const APLIC_MIN_SIZE: u64 = 0x4000;

/// `APLIC_SIZE()`: the size of the register block of a domain with `num_harts` IDCs.
pub const fn aplic_size(num_harts: u32) -> u64 {
    let idcs = 32 * num_harts as u64;
    APLIC_MIN_SIZE + ((idcs + APLIC_MIN_SIZE - 1) & !(APLIC_MIN_SIZE - 1))
}

const APLIC_MAX_IDC: u32 = 1 << 14;
const APLIC_MAX_SOURCE: u32 = 1024;
const APLIC_MIN_IPRIO_BITS: u32 = 1;
const APLIC_MAX_IPRIO_BITS: u32 = 8;
/// `QEMU_APLIC_MAX_CHILDREN`.
const QEMU_APLIC_MAX_CHILDREN: usize = 16;

const APLIC_DOMAINCFG: u64 = 0x0000;
const APLIC_DOMAINCFG_RDONLY: u32 = 0x8000_0000;
const APLIC_DOMAINCFG_IE: u32 = 1 << 8;
const APLIC_DOMAINCFG_DM: u32 = 1 << 2;

const APLIC_SOURCECFG_BASE: u64 = 0x0004;
const APLIC_SOURCECFG_D: u32 = 1 << 10;
const APLIC_SOURCECFG_CHILDIDX_MASK: u32 = 0x0000_03ff;
const APLIC_SOURCECFG_SM_MASK: u32 = 0x0000_0007;
const APLIC_SOURCECFG_SM_INACTIVE: u32 = 0x0;
const APLIC_SOURCECFG_SM_EDGE_RISE: u32 = 0x4;
const APLIC_SOURCECFG_SM_EDGE_FALL: u32 = 0x5;
const APLIC_SOURCECFG_SM_LEVEL_HIGH: u32 = 0x6;
const APLIC_SOURCECFG_SM_LEVEL_LOW: u32 = 0x7;

const APLIC_MMSICFGADDR: u64 = 0x1bc0;
const APLIC_MMSICFGADDRH: u64 = 0x1bc4;
const APLIC_SMSICFGADDR: u64 = 0x1bc8;
const APLIC_SMSICFGADDRH: u64 = 0x1bcc;
const APLIC_XMSICFGADDRH_L: u32 = 1 << 31;
const APLIC_XMSICFGADDRH_HHXS_MASK: u32 = 0x1f;
const APLIC_XMSICFGADDRH_HHXS_SHIFT: u32 = 24;
const APLIC_XMSICFGADDRH_LHXS_MASK: u32 = 0x7;
const APLIC_XMSICFGADDRH_LHXS_SHIFT: u32 = 20;
const APLIC_XMSICFGADDRH_HHXW_MASK: u32 = 0x7;
const APLIC_XMSICFGADDRH_HHXW_SHIFT: u32 = 16;
const APLIC_XMSICFGADDRH_LHXW_MASK: u32 = 0xf;
const APLIC_XMSICFGADDRH_LHXW_SHIFT: u32 = 12;
const APLIC_XMSICFGADDRH_BAPPN_MASK: u32 = 0xfff;
const APLIC_XMSICFGADDR_PPN_SHIFT: u32 = 12;
const APLIC_MMSICFGADDRH_VALID_MASK: u32 = APLIC_XMSICFGADDRH_L
    | (APLIC_XMSICFGADDRH_HHXS_MASK << APLIC_XMSICFGADDRH_HHXS_SHIFT)
    | (APLIC_XMSICFGADDRH_LHXS_MASK << APLIC_XMSICFGADDRH_LHXS_SHIFT)
    | (APLIC_XMSICFGADDRH_HHXW_MASK << APLIC_XMSICFGADDRH_HHXW_SHIFT)
    | (APLIC_XMSICFGADDRH_LHXW_MASK << APLIC_XMSICFGADDRH_LHXW_SHIFT)
    | APLIC_XMSICFGADDRH_BAPPN_MASK;
const APLIC_SMSICFGADDRH_VALID_MASK: u32 =
    (APLIC_XMSICFGADDRH_LHXS_MASK << APLIC_XMSICFGADDRH_LHXS_SHIFT) | APLIC_XMSICFGADDRH_BAPPN_MASK;

const APLIC_SETIP_BASE: u64 = 0x1c00;
const APLIC_SETIPNUM: u64 = 0x1cdc;
const APLIC_CLRIP_BASE: u64 = 0x1d00;
const APLIC_CLRIPNUM: u64 = 0x1ddc;
const APLIC_SETIE_BASE: u64 = 0x1e00;
const APLIC_SETIENUM: u64 = 0x1edc;
const APLIC_CLRIE_BASE: u64 = 0x1f00;
const APLIC_CLRIENUM: u64 = 0x1fdc;
const APLIC_SETIPNUM_LE: u64 = 0x2000;
const APLIC_SETIPNUM_BE: u64 = 0x2004;

const APLIC_ISTATE_PENDING: u32 = 1 << 0;
const APLIC_ISTATE_ENABLED: u32 = 1 << 1;
const APLIC_ISTATE_ENPEND: u32 = APLIC_ISTATE_ENABLED | APLIC_ISTATE_PENDING;
const APLIC_ISTATE_INPUT: u32 = 1 << 8;

const APLIC_GENMSI: u64 = 0x3000;

const APLIC_TARGET_BASE: u64 = 0x3004;
const APLIC_TARGET_HART_IDX_SHIFT: u32 = 18;
const APLIC_TARGET_HART_IDX_MASK: u32 = 0x3fff;
const APLIC_TARGET_GUEST_IDX_SHIFT: u32 = 12;
const APLIC_TARGET_GUEST_IDX_MASK: u32 = 0x3f;
const APLIC_TARGET_IPRIO_MASK: u32 = 0xff;
const APLIC_TARGET_EIID_MASK: u32 = 0x7ff;

const APLIC_IDC_BASE: u64 = 0x4000;
const APLIC_IDC_SIZE: u64 = 32;
const APLIC_IDC_IDELIVERY: u64 = 0x00;
const APLIC_IDC_IFORCE: u64 = 0x04;
const APLIC_IDC_ITHRESHOLD: u64 = 0x08;
const APLIC_IDC_TOPI: u64 = 0x18;
const APLIC_IDC_TOPI_ID_SHIFT: u32 = 16;
const APLIC_IDC_TOPI_ID_MASK: u32 = 0x3ff;
const APLIC_IDC_CLAIMI: u64 = 0x1c;

/// The parameters of `riscv_aplic_create()`, without the address and the parent.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RiscvAplicConfig {
    /// `aperture-size`, the size of the register block.
    pub aperture_size: u64,
    /// `hartid-base`: the hart of IDC 0.
    pub hartid_base: u32,
    /// `num-harts`: the number of IDCs (0 in MSI mode on the virt board).
    pub num_harts: u32,
    /// The number of sources, not counting the nonexistent source 0.
    pub num_sources: u32,
    /// The number of priority bits, 1 to 8.
    pub iprio_bits: u32,
    /// `msimode`: MSI delivery instead of direct delivery.
    pub msimode: bool,
    /// `mmode`: an M level domain.
    pub mmode: bool,
}

/// What stores a message: the address and the 32 bit identity, `address_space_stl_le()` on
/// `address_space_memory`.
pub type MsiSink = Box<dyn Fn(u64, u32) + Send + Sync>;

/// The register state of `RISCVAPLICState`.
#[derive(Debug, Default)]
struct AplicState {
    domaincfg: u32,
    mmsicfgaddr: u32,
    mmsicfgaddr_h: u32,
    smsicfgaddr: u32,
    smsicfgaddr_h: u32,
    genmsi: u32,
    sourcecfg: Vec<u32>,
    state: Vec<u32>,
    target: Vec<u32>,
    idelivery: Vec<u32>,
    iforce: Vec<u32>,
    ithreshold: Vec<u32>,
}

/// One message to send once the lock is dropped: hart index, guest index, identity.
type Msi = (u32, u32, u32);

/// `RISCVAPLICState`, the `riscv.aplic` device.
pub struct RiscvAplic {
    config: RiscvAplicConfig,
    iprio_mask: u32,
    num_irqs: u32,
    bitfield_words: u32,
    state: Mutex<AplicState>,
    external_irqs: Vec<IrqPin>,
    parent: Option<Weak<RiscvAplic>>,
    children: Mutex<Vec<Arc<RiscvAplic>>>,
    msi_sink: OnceLock<MsiSink>,
    weak: Weak<RiscvAplic>,
}

impl fmt::Debug for RiscvAplic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RiscvAplic")
            .field("config", &self.config)
            .field("state", &*self.lock())
            .finish_non_exhaustive()
    }
}

fn set_bit(word: &mut u32, mask: u32, on: bool) {
    if on {
        *word |= mask;
    } else {
        *word &= !mask;
    }
}

impl RiscvAplic {
    /// `riscv_aplic_create()` without the CPU side (see the module documentation), with
    /// `riscv_aplic_add_child()` when there is a `parent`. The device starts in its reset
    /// state and its outputs are disconnected.
    ///
    /// # Panics
    ///
    /// On the parameters `riscv_aplic_create()` and `riscv_aplic_add_child()` assert against.
    pub fn new(config: RiscvAplicConfig, parent: Option<&Arc<RiscvAplic>>) -> Arc<RiscvAplic> {
        let c = config;
        assert!(c.num_harts < APLIC_MAX_IDC);
        assert!(APLIC_IDC_BASE + u64::from(c.num_harts) * APLIC_IDC_SIZE <= c.aperture_size);
        assert!(c.num_sources < APLIC_MAX_SOURCE);
        assert!((APLIC_MIN_IPRIO_BITS..=APLIC_MAX_IPRIO_BITS).contains(&c.iprio_bits));
        let num_irqs = c.num_sources + 1;
        let n = num_irqs as usize;
        let h = c.num_harts as usize;
        let state = AplicState {
            sourcecfg: vec![0; n],
            state: vec![0; n],
            target: vec![0; n],
            idelivery: vec![0; h],
            iforce: vec![0; h],
            ithreshold: vec![0; h],
            ..AplicState::default()
        };
        let aplic = Arc::new_cyclic(|weak| RiscvAplic {
            config,
            iprio_mask: (1 << c.iprio_bits) - 1,
            num_irqs,
            bitfield_words: (num_irqs + 31) >> 5,
            state: Mutex::new(state),
            external_irqs: if c.msimode {
                Vec::new()
            } else {
                (0..h).map(|_| IrqPin::new()).collect()
            },
            parent: parent.map(Arc::downgrade),
            children: Mutex::new(Vec::new()),
            msi_sink: OnceLock::new(),
            weak: weak.clone(),
        });
        if let Some(p) = parent {
            assert_eq!(p.num_irqs, num_irqs);
            let mut ch = p.children.lock().unwrap_or_else(PoisonError::into_inner);
            assert!(ch.len() < QEMU_APLIC_MAX_CHILDREN);
            ch.push(aplic.clone());
        }
        aplic.reset();
        aplic
    }

    fn lock(&self) -> MutexGuard<'_, AplicState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn num_children(&self) -> usize {
        self.children.lock().unwrap_or_else(PoisonError::into_inner).len()
    }

    fn child(&self, idx: usize) -> Option<Arc<RiscvAplic>> {
        self.children.lock().unwrap_or_else(PoisonError::into_inner).get(idx).cloned()
    }

    /// The configuration the device was built with.
    pub fn config(&self) -> &RiscvAplicConfig {
        &self.config
    }

    /// The size of the MMIO region.
    pub fn mmio_size(&self) -> u64 {
        self.config.aperture_size
    }

    /// Sets the function that stores the messages of an MSI mode domain. Without one, messages
    /// are dropped.
    pub fn set_msi_sink(&self, sink: MsiSink) {
        let _ = self.msi_sink.set(sink);
    }

    /// The output of IDC `n` (direct mode only), gpio out `n`. QEMU connects it to the
    /// `IRQ_M_EXT` (M level) or `IRQ_S_EXT` input of hart `hartid_base + n`.
    pub fn external_irq(&self, n: usize) -> &IrqPin {
        &self.external_irqs[n]
    }

    /// Input `n`, from `qdev_init_gpio_in(dev, riscv_aplic_request, num_irqs)`. Only the root
    /// domain has inputs in QEMU; the inputs of a child should not be used.
    pub fn input(&self, n: u32) -> IrqLine {
        let w = self.weak.clone();
        IrqLine::new(
            Arc::new(move |n, level| {
                if let Some(s) = w.upgrade() {
                    s.set_irq(n, level);
                }
            }),
            n,
        )
    }

    /// `riscv_aplic_source_active()`: whether source `irq` exists, is not delegated and is not
    /// inactive. Its `target` reads zero otherwise.
    fn source_active(&self, s: &AplicState, irq: u32) -> bool {
        if irq == 0 || self.num_irqs <= irq {
            return false;
        }
        let sc = s.sourcecfg[irq as usize];
        sc & APLIC_SOURCECFG_D == 0 && sc & APLIC_SOURCECFG_SM_MASK != APLIC_SOURCECFG_SM_INACTIVE
    }

    /// `riscv_aplic_irq_rectified_val()`.
    fn rectified(&self, s: &AplicState, irq: u32) -> bool {
        if !self.source_active(s, irq) {
            return false;
        }
        let sm = s.sourcecfg[irq as usize] & APLIC_SOURCECFG_SM_MASK;
        let raw = s.state[irq as usize] & APLIC_ISTATE_INPUT != 0;
        let inverted = sm == APLIC_SOURCECFG_SM_LEVEL_LOW || sm == APLIC_SOURCECFG_SM_EDGE_FALL;
        raw ^ inverted
    }

    /// The sources of bitfield word `word`, skipping source 0 and those past the last.
    fn word_irqs(&self, word: u32) -> impl Iterator<Item = (u32, u32)> {
        let num_irqs = self.num_irqs;
        (0..32).map(move |i| (i, word * 32 + i)).filter(move |&(_, irq)| irq != 0 && irq < num_irqs)
    }

    /// `riscv_aplic_read_input_word()`.
    fn read_input_word(&self, s: &AplicState, word: u32) -> u32 {
        (0..32).fold(0, |acc, i| acc | (u32::from(self.rectified(s, word * 32 + i)) << i))
    }

    /// `riscv_aplic_read_pending_word()` and `riscv_aplic_read_enabled_word()`.
    fn read_state_word(&self, s: &AplicState, word: u32, bit: u32) -> u32 {
        self.word_irqs(word)
            .filter(|&(_, irq)| s.state[irq as usize] & bit != 0)
            .fold(0, |acc, (i, _)| acc | 1 << i)
    }

    /// `riscv_aplic_set_pending()`: in direct mode the pending bit of a level triggered source
    /// follows its input alone; in MSI mode it can be cleared, and set while the input is
    /// asserted.
    fn set_pending(&self, s: &mut AplicState, irq: u32, pending: bool) {
        if !self.source_active(s, irq) {
            return;
        }
        let sm = s.sourcecfg[irq as usize] & APLIC_SOURCECFG_SM_MASK;
        if (sm == APLIC_SOURCECFG_SM_LEVEL_HIGH || sm == APLIC_SOURCECFG_SM_LEVEL_LOW)
            && (!self.config.msimode || pending)
        {
            if !self.config.msimode {
                return;
            }
            let input = s.state[irq as usize] & APLIC_ISTATE_INPUT != 0;
            if input && sm == APLIC_SOURCECFG_SM_LEVEL_LOW {
                return;
            }
            if !input && sm == APLIC_SOURCECFG_SM_LEVEL_HIGH {
                return;
            }
        }
        set_bit(&mut s.state[irq as usize], APLIC_ISTATE_PENDING, pending);
    }

    /// `riscv_aplic_set_pending_word()`.
    fn set_pending_word(&self, s: &mut AplicState, word: u32, value: u32, pending: bool) {
        for (i, irq) in self.word_irqs(word) {
            if value & (1 << i) != 0 {
                self.set_pending(s, irq, pending);
            }
        }
    }

    /// `riscv_aplic_set_enabled()`.
    fn set_enabled(&self, s: &mut AplicState, irq: u32, enabled: bool) {
        if self.source_active(s, irq) {
            set_bit(&mut s.state[irq as usize], APLIC_ISTATE_ENABLED, enabled);
        }
    }

    /// `riscv_aplic_set_enabled_word()`.
    fn set_enabled_word(&self, s: &mut AplicState, word: u32, value: u32, enabled: bool) {
        for (i, irq) in self.word_irqs(word) {
            if value & (1 << i) != 0 {
                self.set_enabled(s, irq, enabled);
            }
        }
    }

    /// The M level domain whose MSI address configuration this domain uses: itself or the
    /// nearest M level ancestor.
    fn msi_domain(&self) -> Option<Arc<RiscvAplic>> {
        let mut a = self.weak.upgrade()?;
        while !a.config.mmode {
            a = a.parent.as_ref()?.upgrade()?;
        }
        Some(a)
    }

    /// The address part of `riscv_aplic_msi_send()`.
    fn msi_addr(&self, hart_idx: u32, guest_idx: u32) -> Option<u64> {
        let m = self.msi_domain()?;
        let (mut cfg, mut cfg_h) = {
            let ms = m.lock();
            (ms.mmsicfgaddr, ms.mmsicfgaddr_h)
        };
        let field = |h: u32, shift, mask| (h >> shift) & mask;
        let mut lhxs = field(cfg_h, APLIC_XMSICFGADDRH_LHXS_SHIFT, APLIC_XMSICFGADDRH_LHXS_MASK);
        let lhxw = field(cfg_h, APLIC_XMSICFGADDRH_LHXW_SHIFT, APLIC_XMSICFGADDRH_LHXW_MASK);
        let hhxs = field(cfg_h, APLIC_XMSICFGADDRH_HHXS_SHIFT, APLIC_XMSICFGADDRH_HHXS_MASK);
        let hhxw = field(cfg_h, APLIC_XMSICFGADDRH_HHXW_SHIFT, APLIC_XMSICFGADDRH_HHXW_MASK);
        if !self.config.mmode {
            let ms = m.lock();
            cfg_h = ms.smsicfgaddr_h;
            cfg = ms.smsicfgaddr;
            lhxs = field(cfg_h, APLIC_XMSICFGADDRH_LHXS_SHIFT, APLIC_XMSICFGADDRH_LHXS_MASK);
        }
        let group_idx = hart_idx >> lhxw;
        let mut addr = u64::from(cfg);
        addr |= u64::from(cfg_h & APLIC_XMSICFGADDRH_BAPPN_MASK) << 32;
        addr |= u64::from(group_idx & ((1 << hhxw) - 1)) << (hhxs + APLIC_XMSICFGADDR_PPN_SHIFT);
        addr |= u64::from(hart_idx & ((1 << lhxw) - 1)) << lhxs;
        addr |= u64::from(guest_idx & ((1 << lhxs) - 1));
        Some(addr << APLIC_XMSICFGADDR_PPN_SHIFT)
    }

    /// `riscv_aplic_msi_send()`, for the messages gathered under the lock, which must be
    /// dropped by now.
    fn send_msis(&self, msis: Vec<Msi>) {
        for (hart_idx, guest_idx, eiid) in msis {
            // QEMU logs "m-level APLIC not found" when there is no M level domain.
            if let (Some(addr), Some(sink)) = (self.msi_addr(hart_idx, guest_idx), self.msi_sink())
            {
                sink(addr, eiid);
            }
        }
    }

    /// The sink [`RiscvAplic::set_msi_sink`] gave.
    fn msi_sink(&self) -> Option<&MsiSink> {
        self.msi_sink.get()
    }

    /// `riscv_aplic_msi_irq_update()`: a pending and enabled source of an MSI mode domain
    /// stops being pending and becomes a message in `msis`.
    fn msi_irq_update(&self, s: &mut AplicState, irq: u32, msis: &mut Vec<Msi>) {
        if !self.config.msimode || self.num_irqs <= irq || s.domaincfg & APLIC_DOMAINCFG_IE == 0 {
            return;
        }
        let i = irq as usize;
        if s.state[i] & APLIC_ISTATE_ENPEND != APLIC_ISTATE_ENPEND {
            return;
        }
        s.state[i] &= !APLIC_ISTATE_PENDING;
        let t = s.target[i];
        let hart_idx = (t >> APLIC_TARGET_HART_IDX_SHIFT) & APLIC_TARGET_HART_IDX_MASK;
        // An M level domain ignores the guest index.
        let guest_idx = if self.config.mmode {
            0
        } else {
            (t >> APLIC_TARGET_GUEST_IDX_SHIFT) & APLIC_TARGET_GUEST_IDX_MASK
        };
        msis.push((hart_idx, guest_idx, t & APLIC_TARGET_EIID_MASK));
    }

    /// `riscv_aplic_idc_topi()`.
    fn idc_topi(&self, s: &AplicState, idc: u32) -> u32 {
        if self.config.num_harts <= idc {
            return 0;
        }
        let ithres = s.ithreshold[idc as usize];
        let (mut best_irq, mut best_iprio) = (u32::MAX, u32::MAX);
        for irq in 1..self.num_irqs {
            let i = irq as usize;
            if s.state[i] & APLIC_ISTATE_ENPEND != APLIC_ISTATE_ENPEND {
                continue;
            }
            let hart = (s.target[i] >> APLIC_TARGET_HART_IDX_SHIFT) & APLIC_TARGET_HART_IDX_MASK;
            if hart != idc {
                continue;
            }
            let iprio = s.target[i] & self.iprio_mask;
            if ithres != 0 && iprio >= ithres {
                continue;
            }
            if iprio < best_iprio {
                best_irq = irq;
                best_iprio = iprio;
            }
        }
        if best_irq < self.num_irqs && best_iprio <= self.iprio_mask {
            (best_irq << APLIC_IDC_TOPI_ID_SHIFT) | best_iprio
        } else {
            0
        }
    }

    /// `riscv_aplic_idc_update()`.
    fn idc_update(&self, s: &AplicState, idc: u32) {
        if self.config.msimode || self.config.num_harts <= idc {
            return;
        }
        let i = idc as usize;
        let level = s.domaincfg & APLIC_DOMAINCFG_IE != 0
            && s.idelivery[i] != 0
            && (s.iforce[i] != 0 || self.idc_topi(s, idc) != 0);
        self.external_irqs[i].set_bool(level);
    }

    /// `riscv_aplic_idc_claimi()`.
    fn idc_claimi(&self, s: &mut AplicState, idc: u32) -> u32 {
        let topi = self.idc_topi(s, idc);
        if topi == 0 {
            s.iforce[idc as usize] = 0;
            self.idc_update(s, idc);
            return 0;
        }
        let irq = ((topi >> APLIC_IDC_TOPI_ID_SHIFT) & APLIC_IDC_TOPI_ID_MASK) as usize;
        let sm = s.sourcecfg[irq] & APLIC_SOURCECFG_SM_MASK;
        let input = s.state[irq] & APLIC_ISTATE_INPUT != 0;
        let still = (sm == APLIC_SOURCECFG_SM_LEVEL_HIGH && input)
            || (sm == APLIC_SOURCECFG_SM_LEVEL_LOW && !input);
        set_bit(&mut s.state[irq], APLIC_ISTATE_PENDING, still);
        self.idc_update(s, idc);
        topi
    }

    /// `riscv_aplic_request()`: input `irq` at `level`. A delegated source goes to the child
    /// domain.
    ///
    /// # Panics
    ///
    /// If `irq` is 0 or past the last source, as QEMU asserts.
    pub fn set_irq(&self, irq: u32, level: i32) {
        assert!(0 < irq && irq < self.num_irqs);
        let i = irq as usize;
        let mut msis = Vec::new();
        {
            let mut s = self.lock();
            let sourcecfg = s.sourcecfg[i];
            if sourcecfg & APLIC_SOURCECFG_D != 0 {
                drop(s);
                let childidx = (sourcecfg & APLIC_SOURCECFG_CHILDIDX_MASK) as usize;
                if let Some(child) = self.child(childidx) {
                    child.set_irq(irq, level);
                }
                return;
            }
            let state = s.state[i];
            let input = state & APLIC_ISTATE_INPUT != 0;
            let pending = state & APLIC_ISTATE_PENDING != 0;
            let new_pending = match sourcecfg & APLIC_SOURCECFG_SM_MASK {
                APLIC_SOURCECFG_SM_EDGE_RISE if level > 0 && !input && !pending => Some(true),
                APLIC_SOURCECFG_SM_EDGE_FALL if level <= 0 && input && !pending => Some(true),
                APLIC_SOURCECFG_SM_LEVEL_HIGH if (level > 0) != pending => Some(level > 0),
                APLIC_SOURCECFG_SM_LEVEL_LOW if (level <= 0) != pending => Some(level <= 0),
                _ => None,
            };
            if let Some(p) = new_pending {
                set_bit(&mut s.state[i], APLIC_ISTATE_PENDING, p);
            }
            set_bit(&mut s.state[i], APLIC_ISTATE_INPUT, level > 0);
            if new_pending.is_some() {
                if self.config.msimode {
                    self.msi_irq_update(&mut s, irq, &mut msis);
                } else {
                    let idc =
                        (s.target[i] >> APLIC_TARGET_HART_IDX_SHIFT) & APLIC_TARGET_HART_IDX_MASK;
                    self.idc_update(&s, idc);
                }
            }
        }
        self.send_msis(msis);
    }

    /// `riscv_aplic_read()`.
    pub fn reg_read(&self, addr: u64) -> u64 {
        if addr & 3 != 0 {
            // QEMU logs "riscv_aplic_read: Invalid register read 0x%x".
            return 0;
        }
        let c = &self.config;
        let num_irqs = u64::from(self.num_irqs);
        let words = u64::from(self.bitfield_words);
        let msi_m = c.mmode && c.msimode;
        let has_children = self.num_children() != 0;
        let in_words = |base: u64| (base..base + words * 4).contains(&addr);
        let mut s = self.lock();
        let val = if addr == APLIC_DOMAINCFG {
            APLIC_DOMAINCFG_RDONLY | s.domaincfg | if c.msimode { APLIC_DOMAINCFG_DM } else { 0 }
        } else if (APLIC_SOURCECFG_BASE..APLIC_SOURCECFG_BASE + (num_irqs - 1) * 4).contains(&addr)
        {
            s.sourcecfg[((addr - APLIC_SOURCECFG_BASE) >> 2) as usize + 1]
        } else if msi_m && addr == APLIC_MMSICFGADDR {
            s.mmsicfgaddr
        } else if msi_m && addr == APLIC_MMSICFGADDRH {
            s.mmsicfgaddr_h
        } else if msi_m && addr == APLIC_SMSICFGADDR {
            // Only there for an M level domain with an S level child.
            if has_children { s.smsicfgaddr } else { 0 }
        } else if msi_m && addr == APLIC_SMSICFGADDRH {
            if has_children { s.smsicfgaddr_h } else { 0 }
        } else if in_words(APLIC_SETIP_BASE) {
            self.read_state_word(&s, ((addr - APLIC_SETIP_BASE) >> 2) as u32, APLIC_ISTATE_PENDING)
        } else if in_words(APLIC_CLRIP_BASE) {
            self.read_input_word(&s, ((addr - APLIC_CLRIP_BASE) >> 2) as u32)
        } else if in_words(APLIC_SETIE_BASE) {
            self.read_state_word(&s, ((addr - APLIC_SETIE_BASE) >> 2) as u32, APLIC_ISTATE_ENABLED)
        } else if addr == APLIC_GENMSI {
            if c.msimode { s.genmsi } else { 0 }
        } else if (APLIC_TARGET_BASE..APLIC_TARGET_BASE + (num_irqs - 1) * 4).contains(&addr) {
            let irq = ((addr - APLIC_TARGET_BASE) >> 2) as u32 + 1;
            if self.source_active(&s, irq) { s.target[irq as usize] } else { 0 }
        } else if !c.msimode
            && (APLIC_IDC_BASE..APLIC_IDC_BASE + u64::from(c.num_harts) * APLIC_IDC_SIZE)
                .contains(&addr)
        {
            let idc = ((addr - APLIC_IDC_BASE) / APLIC_IDC_SIZE) as u32;
            match (addr - APLIC_IDC_BASE) % APLIC_IDC_SIZE {
                APLIC_IDC_IDELIVERY => s.idelivery[idc as usize],
                APLIC_IDC_IFORCE => s.iforce[idc as usize],
                APLIC_IDC_ITHRESHOLD => s.ithreshold[idc as usize],
                APLIC_IDC_TOPI => self.idc_topi(&s, idc),
                APLIC_IDC_CLAIMI => self.idc_claimi(&mut s, idc),
                // QEMU logs "riscv_aplic_read: Invalid register read 0x%x".
                _ => 0,
            }
        } else {
            // SETIPNUM, CLRIPNUM, SETIENUM, the CLRIE words, CLRIENUM and SETIPNUM_LE and _BE
            // read zero, and so does everything else.
            0
        };
        u64::from(val)
    }

    /// `riscv_aplic_write()`.
    pub fn reg_write(&self, addr: u64, value: u64) {
        if addr & 3 != 0 {
            // QEMU logs "riscv_aplic_write: Invalid register write 0x%x".
            return;
        }
        let c = self.config;
        let num_irqs = u64::from(self.num_irqs);
        let words = u64::from(self.bitfield_words);
        let msi_m = c.mmode && c.msimode;
        let has_children = self.num_children() != 0;
        let in_words = |base: u64| (base..base + words * 4).contains(&addr);
        let word = |base: u64| ((addr - base) >> 2) as u32;
        let v32 = value as u32;
        let mut idc = None;
        let mut msis = Vec::new();
        {
            let mut s = self.lock();
            if addr == APLIC_DOMAINCFG {
                // Only IE is writable.
                s.domaincfg = v32 & APLIC_DOMAINCFG_IE;
            } else if (APLIC_SOURCECFG_BASE..APLIC_SOURCECFG_BASE + (num_irqs - 1) * 4)
                .contains(&addr)
            {
                let irq = word(APLIC_SOURCECFG_BASE) + 1;
                let mut v = v32;
                if !has_children && v & APLIC_SOURCECFG_D != 0 {
                    v = 0;
                }
                if v & APLIC_SOURCECFG_D != 0 {
                    v &= APLIC_SOURCECFG_D | APLIC_SOURCECFG_CHILDIDX_MASK;
                } else {
                    v &= APLIC_SOURCECFG_D | APLIC_SOURCECFG_SM_MASK;
                }
                let i = irq as usize;
                s.sourcecfg[i] = v;
                if v & APLIC_SOURCECFG_D != 0 || v == 0 {
                    s.state[i] &= !(APLIC_ISTATE_PENDING | APLIC_ISTATE_ENABLED);
                } else if self.rectified(&s, irq) {
                    s.state[i] |= APLIC_ISTATE_PENDING;
                }
            } else if msi_m && addr == APLIC_MMSICFGADDR {
                if s.mmsicfgaddr_h & APLIC_XMSICFGADDRH_L == 0 {
                    s.mmsicfgaddr = v32;
                }
            } else if msi_m && addr == APLIC_MMSICFGADDRH {
                if s.mmsicfgaddr_h & APLIC_XMSICFGADDRH_L == 0 {
                    s.mmsicfgaddr_h = v32 & APLIC_MMSICFGADDRH_VALID_MASK;
                }
            } else if msi_m && addr == APLIC_SMSICFGADDR {
                if has_children && s.mmsicfgaddr_h & APLIC_XMSICFGADDRH_L == 0 {
                    s.smsicfgaddr = v32;
                }
            } else if msi_m && addr == APLIC_SMSICFGADDRH {
                if has_children && s.mmsicfgaddr_h & APLIC_XMSICFGADDRH_L == 0 {
                    s.smsicfgaddr_h = v32 & APLIC_SMSICFGADDRH_VALID_MASK;
                }
            } else if in_words(APLIC_SETIP_BASE) {
                self.set_pending_word(&mut s, word(APLIC_SETIP_BASE), v32, true);
            } else if addr == APLIC_SETIPNUM || addr == APLIC_SETIPNUM_LE {
                self.set_pending(&mut s, v32, true);
            } else if in_words(APLIC_CLRIP_BASE) {
                self.set_pending_word(&mut s, word(APLIC_CLRIP_BASE), v32, false);
            } else if addr == APLIC_CLRIPNUM {
                self.set_pending(&mut s, v32, false);
            } else if in_words(APLIC_SETIE_BASE) {
                self.set_enabled_word(&mut s, word(APLIC_SETIE_BASE), v32, true);
            } else if addr == APLIC_SETIENUM {
                self.set_enabled(&mut s, v32, true);
            } else if in_words(APLIC_CLRIE_BASE) {
                self.set_enabled_word(&mut s, word(APLIC_CLRIE_BASE), v32, false);
            } else if addr == APLIC_CLRIENUM {
                self.set_enabled(&mut s, v32, false);
            } else if addr == APLIC_SETIPNUM_BE {
                self.set_pending(&mut s, v32.swap_bytes(), true);
            } else if addr == APLIC_GENMSI {
                if c.msimode {
                    s.genmsi = v32 & !(APLIC_TARGET_GUEST_IDX_MASK << APLIC_TARGET_GUEST_IDX_SHIFT);
                    msis.push((
                        v32 >> APLIC_TARGET_HART_IDX_SHIFT,
                        0,
                        v32 & APLIC_TARGET_EIID_MASK,
                    ));
                }
            } else if (APLIC_TARGET_BASE..APLIC_TARGET_BASE + (num_irqs - 1) * 4).contains(&addr) {
                let irq = word(APLIC_TARGET_BASE) + 1;
                if !self.source_active(&s, irq) {
                    return;
                }
                s.target[irq as usize] = if c.msimode {
                    v32
                } else {
                    let p = v32 & self.iprio_mask;
                    (v32 & !APLIC_TARGET_IPRIO_MASK) | if p != 0 { p } else { 1 }
                };
            } else if !c.msimode
                && (APLIC_IDC_BASE..APLIC_IDC_BASE + u64::from(c.num_harts) * APLIC_IDC_SIZE)
                    .contains(&addr)
            {
                let n = ((addr - APLIC_IDC_BASE) / APLIC_IDC_SIZE) as u32;
                let i = n as usize;
                match (addr - APLIC_IDC_BASE) % APLIC_IDC_SIZE {
                    APLIC_IDC_IDELIVERY => s.idelivery[i] = v32 & 1,
                    APLIC_IDC_IFORCE => s.iforce[i] = v32 & 1,
                    APLIC_IDC_ITHRESHOLD => s.ithreshold[i] = v32 & self.iprio_mask,
                    // QEMU logs "riscv_aplic_write: Invalid register write 0x%x".
                    _ => return,
                }
                idc = Some(n);
            } else {
                // QEMU logs "riscv_aplic_write: Invalid register write 0x%x".
                return;
            }

            if c.msimode {
                for irq in 1..self.num_irqs {
                    self.msi_irq_update(&mut s, irq, &mut msis);
                }
            } else if let Some(n) = idc {
                self.idc_update(&s, n);
            } else {
                for n in 0..c.num_harts {
                    self.idc_update(&s, n);
                }
            }
        }
        self.send_msis(msis);
    }

    /// `riscv_aplic_reset_enter()`: clears the configuration, the enabled bits and the IDCs, and
    /// lowers the outputs. The pending bits and input levels stay.
    pub fn reset(&self) {
        let mut s = self.lock();
        s.domaincfg = 0;
        s.sourcecfg.fill(0);
        s.target.fill(if self.config.msimode { 0 } else { 1 });
        for st in &mut s.state {
            *st &= !APLIC_ISTATE_ENABLED;
        }
        // This also unlocks mmsicfgaddrh.L.
        s.mmsicfgaddr = 0;
        s.mmsicfgaddr_h = 0;
        s.smsicfgaddr = 0;
        s.smsicfgaddr_h = 0;
        if !self.config.msimode {
            s.idelivery.fill(0);
            s.iforce.fill(0);
            s.ithreshold.fill(0);
            for pin in &self.external_irqs {
                pin.lower();
            }
        }
    }

    /// Whether source `irq` is pending.
    pub fn is_pending(&self, irq: u32) -> bool {
        self.lock().state[irq as usize] & APLIC_ISTATE_PENDING != 0
    }
}

/// `riscv_aplic_ops`.
impl MmioOps for RiscvAplic {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(self.reg_read(offset))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.reg_write(offset, value);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::exact(4)
    }

    fn endianness(&self) -> Endian {
        Endian::Little
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI32, Ordering};

    const TARGET: u64 = APLIC_TARGET_BASE;
    const IDC: u64 = APLIC_IDC_BASE;

    fn cfg(msimode: bool, mmode: bool, num_harts: u32) -> RiscvAplicConfig {
        RiscvAplicConfig {
            aperture_size: aplic_size(num_harts.max(1)),
            hartid_base: 0,
            num_harts,
            num_sources: 96,
            iprio_bits: 3,
            msimode,
            mmode,
        }
    }

    fn watch(pin: &IrqPin) -> Arc<AtomicI32> {
        let level = Arc::new(AtomicI32::new(-1));
        let l = level.clone();
        pin.connect(IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)));
        level
    }

    fn sourcecfg(irq: u64) -> u64 {
        APLIC_SOURCECFG_BASE + (irq - 1) * 4
    }

    fn target(irq: u64) -> u64 {
        TARGET + (irq - 1) * 4
    }

    #[test]
    fn sizes() {
        assert_eq!(aplic_size(512), 0x8000);
        assert_eq!(aplic_size(1), 0x8000);
        assert_eq!(aplic_size(0), 0x4000);
    }

    #[test]
    fn direct_mode_level_and_edge() {
        let a = RiscvAplic::new(cfg(false, false, 2), None);
        let out1 = watch(a.external_irq(1));
        assert_eq!(a.reg_read(APLIC_DOMAINCFG), 0x8000_0000);
        a.reg_write(APLIC_DOMAINCFG, 0xffff_ffff);
        assert_eq!(a.reg_read(APLIC_DOMAINCFG), 0x8000_0100);

        // Inactive sources have a target that reads zero; a direct mode target keeps a
        // nonzero priority.
        assert_eq!(a.reg_read(target(5)), 0);
        a.reg_write(sourcecfg(5), APLIC_SOURCECFG_SM_LEVEL_HIGH as u64);
        assert_eq!(a.reg_read(target(5)), 1);
        a.reg_write(target(5), 1 << 18 | 0x18);
        assert_eq!(a.reg_read(target(5)), 1 << 18 | 1);
        a.reg_write(target(5), 1 << 18 | 3);
        // Delegation needs a child.
        a.reg_write(sourcecfg(6), u64::from(APLIC_SOURCECFG_D) | 1);
        assert_eq!(a.reg_read(sourcecfg(6)), 0);

        a.reg_write(APLIC_SETIENUM, 5);
        assert_eq!(a.reg_read(APLIC_SETIE_BASE), 1 << 5);
        a.reg_write(IDC + 32 + APLIC_IDC_IDELIVERY, 1);
        assert_eq!(out1.load(Ordering::SeqCst), 0);
        a.input(5).raise();
        assert_eq!(out1.load(Ordering::SeqCst), 1);
        assert_eq!(a.reg_read(APLIC_CLRIP_BASE), 1 << 5);
        assert_eq!(a.reg_read(IDC + 32 + APLIC_IDC_TOPI), 5 << 16 | 3);
        // A threshold at the priority masks it.
        a.reg_write(IDC + 32 + APLIC_IDC_ITHRESHOLD, 3);
        assert_eq!(out1.load(Ordering::SeqCst), 0);
        a.reg_write(IDC + 32 + APLIC_IDC_ITHRESHOLD, 0);
        // Claiming a level source whose input is still high leaves it pending.
        assert_eq!(a.reg_read(IDC + 32 + APLIC_IDC_CLAIMI), 5 << 16 | 3);
        assert!(a.is_pending(5));
        // Software cannot set or clear a direct mode level source.
        a.reg_write(APLIC_CLRIPNUM, 5);
        assert!(a.is_pending(5));
        a.input(5).lower();
        assert!(!a.is_pending(5));
        assert_eq!(out1.load(Ordering::SeqCst), 0);

        // A rising edge source latches until claimed.
        a.reg_write(sourcecfg(7), APLIC_SOURCECFG_SM_EDGE_RISE as u64);
        a.reg_write(target(7), 1 << 18 | 2);
        a.reg_write(APLIC_SETIENUM, 7);
        a.input(7).pulse();
        assert_eq!(out1.load(Ordering::SeqCst), 1);
        assert_eq!(a.reg_read(IDC + 32 + APLIC_IDC_CLAIMI), 7 << 16 | 2);
        assert_eq!(out1.load(Ordering::SeqCst), 0);
        // iforce raises the output until a claim finds nothing.
        a.reg_write(IDC + 32 + APLIC_IDC_IFORCE, 1);
        assert_eq!(out1.load(Ordering::SeqCst), 1);
        assert_eq!(a.reg_read(IDC + 32 + APLIC_IDC_CLAIMI), 0);
        assert_eq!(out1.load(Ordering::SeqCst), 0);
        assert_eq!(a.reg_read(IDC + 32 + APLIC_IDC_IFORCE), 0);

        a.reset();
        assert_eq!(a.reg_read(APLIC_DOMAINCFG), 0x8000_0000);
        assert_eq!(a.reg_read(APLIC_SETIE_BASE), 0);
    }

    #[test]
    fn msi_mode_delegation_and_addresses() {
        let m = RiscvAplic::new(cfg(true, true, 0), None);
        let s = RiscvAplic::new(cfg(true, false, 0), Some(&m));
        let sent = Arc::new(Mutex::new(Vec::new()));
        for a in [&m, &s] {
            let sent = sent.clone();
            a.set_msi_sink(Box::new(move |addr, data| sent.lock().unwrap().push((addr, data))));
        }
        assert_eq!(m.reg_read(APLIC_DOMAINCFG), 0x8000_0004);

        // The virt board's configuration: M files at 0x24000000, S files at 0x28000000 with 4
        // pages per hart (LHXS 2), and 2 bits of hart index (LHXW 2).
        m.reg_write(APLIC_MMSICFGADDR, 0x24000);
        m.reg_write(APLIC_MMSICFGADDRH, 2 << 12);
        m.reg_write(APLIC_SMSICFGADDR, 0x28000);
        m.reg_write(APLIC_SMSICFGADDRH, 2 << 20 | 0xff00_0000);
        assert_eq!(m.reg_read(APLIC_SMSICFGADDRH), 2 << 20);
        assert_eq!(s.reg_read(APLIC_SMSICFGADDR), 0);

        // Source 10 is delegated to the S domain.
        m.reg_write(sourcecfg(10), u64::from(APLIC_SOURCECFG_D));
        assert_eq!(m.reg_read(target(10)), 0);
        s.reg_write(sourcecfg(10), APLIC_SOURCECFG_SM_EDGE_RISE as u64);
        s.reg_write(target(10), 1 << 18 | 2 << 12 | 33);
        s.reg_write(APLIC_SETIENUM, 10);
        s.reg_write(APLIC_DOMAINCFG, u64::from(APLIC_DOMAINCFG_IE));
        m.input(10).pulse();
        assert_eq!(*sent.lock().unwrap(), [(0x2800_6000, 33)]);
        assert!(!s.is_pending(10));
        sent.lock().unwrap().clear();

        // An M source ignores the guest index; IE gates delivery and the pending bit waits.
        m.reg_write(sourcecfg(3), APLIC_SOURCECFG_SM_LEVEL_HIGH as u64);
        m.reg_write(target(3), 3 << 18 | 1 << 12 | 7);
        m.reg_write(APLIC_SETIENUM, 3);
        m.input(3).raise();
        assert!(m.is_pending(3));
        assert!(sent.lock().unwrap().is_empty());
        m.reg_write(APLIC_DOMAINCFG, u64::from(APLIC_DOMAINCFG_IE));
        assert_eq!(*sent.lock().unwrap(), [(0x2400_3000, 7)]);
        // In MSI mode a level source can be set pending again while its input is high.
        m.reg_write(APLIC_SETIPNUM_BE, 3u32.swap_bytes() as u64);
        assert_eq!(sent.lock().unwrap().len(), 2);

        // genmsi, and the lock bit.
        m.reg_write(APLIC_GENMSI, 2 << 18 | 5 << 12 | 9);
        assert_eq!(sent.lock().unwrap()[2], (0x2400_2000, 9));
        assert_eq!(m.reg_read(APLIC_GENMSI), 2 << 18 | 9);
        m.reg_write(APLIC_MMSICFGADDRH, u64::from(APLIC_XMSICFGADDRH_L));
        m.reg_write(APLIC_MMSICFGADDR, 0x1);
        assert_eq!(m.reg_read(APLIC_MMSICFGADDR), 0x24000);
        m.reset();
        assert_eq!(m.reg_read(APLIC_MMSICFGADDRH), 0);
    }
}
