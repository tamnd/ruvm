// SPDX-License-Identifier: GPL-2.0-or-later

//! The state of the GICv3 that QEMU's `arm_gicv3` section carries, from `vmstate_gicv3` in
//! hw/intc/arm_gicv3_common.c and `arm_gicv3_post_load()` in hw/intc/arm_gicv3.c.
//!
//! The fields of [`GicV3VmState`] and [`GicV3CpuVmState`] are named after QEMU's. ruvm keeps no
//! GICD_STATUSR or GICR_STATUSR and has no NMI support, so the status registers go out as 0 and
//! are dropped on load, and the NMI subsections are never sent. ICC_SRE_EL1 is a constant on
//! TCG and QEMU's TCG GIC never sets the `icc_sre_el1` field, so it goes out as 0, which sends
//! the `sre_el1` subsection as QEMU does, and is dropped on load.

use super::{BMP_WORDS, GIC_INTERNAL, GICV3_MAXIRQ, GicState, GicV3};

/// `GICV3_MAXIRQ` as a length.
const MAXIRQ: usize = GICV3_MAXIRQ as usize;

/// The `icc_sre_el1` that `vmstate_gicv3_cpu_pre_load()` assumes when there is no `sre_el1`
/// subsection: SRE, DFB and DIB set.
pub const ICC_SRE_EL1: u64 = 0x7;

/// One CPU's part of the state, `vmstate_gicv3_cpu`.
#[derive(Clone, Debug, Default)]
pub struct GicV3CpuVmState {
    pub level: u32,
    pub gicr_ctlr: u32,
    pub gicr_statusr: [u32; 2],
    pub gicr_waker: u32,
    pub gicr_propbaser: u64,
    pub gicr_pendbaser: u64,
    pub gicr_igroupr0: u32,
    pub gicr_ienabler0: u32,
    pub gicr_ipendr0: u32,
    pub gicr_iactiver0: u32,
    pub edge_trigger: u32,
    pub gicr_igrpmodr0: u32,
    pub gicr_nsacr: u32,
    pub gicr_ipriorityr: [u8; GIC_INTERNAL as usize],
    pub icc_ctlr_el1: [u64; 2],
    pub icc_pmr_el1: u64,
    pub icc_bpr: [u64; 3],
    /// `icc_apr[3][4]`, row by row.
    pub icc_apr: [u64; 12],
    pub icc_igrpen: [u64; 3],
    pub icc_ctlr_el3: u64,
    /// Whether the CPU has EL2, which is when QEMU gives it list registers and sends the
    /// `arm_gicv3_cpu/virt` subsection.
    pub virt: bool,
    /// `ich_apr[3][4]`, row by row.
    pub ich_apr: [u64; 12],
    pub ich_hcr_el2: u64,
    pub ich_lr_el2: [u64; 16],
    pub ich_vmcr_el2: u64,
    pub icc_sre_el1: u64,
}

/// The whole state, `vmstate_gicv3`.
#[derive(Clone, Debug)]
pub struct GicV3VmState {
    pub gicd_ctlr: u32,
    pub gicd_statusr: [u32; 2],
    pub group: [u32; BMP_WORDS],
    pub grpmod: [u32; BMP_WORDS],
    pub enabled: [u32; BMP_WORDS],
    pub pending: [u32; BMP_WORDS],
    pub active: [u32; BMP_WORDS],
    pub level: [u32; BMP_WORDS],
    pub edge_trigger: [u32; BMP_WORDS],
    pub gicd_ipriority: Box<[u8; MAXIRQ]>,
    pub gicd_irouter: Box<[u64; MAXIRQ]>,
    pub gicd_nsacr: [u32; MAXIRQ.div_ceil(16)],
    pub cpu: Vec<GicV3CpuVmState>,
    pub gicd_no_migration_shift_bug: bool,
}

impl Default for GicV3VmState {
    fn default() -> Self {
        GicV3VmState {
            gicd_ctlr: 0,
            gicd_statusr: [0; 2],
            group: [0; BMP_WORDS],
            grpmod: [0; BMP_WORDS],
            enabled: [0; BMP_WORDS],
            pending: [0; BMP_WORDS],
            active: [0; BMP_WORDS],
            level: [0; BMP_WORDS],
            edge_trigger: [0; BMP_WORDS],
            gicd_ipriority: Box::new([0; MAXIRQ]),
            gicd_irouter: Box::new([0; MAXIRQ]),
            gicd_nsacr: [0; MAXIRQ.div_ceil(16)],
            cpu: Vec::new(),
            gicd_no_migration_shift_bug: true,
        }
    }
}

impl GicV3CpuVmState {
    /// `vmstate_gicv3_cpu_pre_load()`: with no `sre_el1` subsection, ICC_SRE_EL1 is 0x7.
    pub fn pre_load(&mut self) {
        self.icc_sre_el1 = ICC_SRE_EL1;
    }
}

impl GicV3VmState {
    /// `gicv3_pre_load()`: under TCG the flag stays set unless the stream says otherwise.
    pub fn pre_load(&mut self) {
        self.gicd_no_migration_shift_bug = true;
    }

    /// `gicv3_gicd_no_migration_shift_bug_post_load()`: a stream from a QEMU with the bitmap
    /// shift bug has the SPIs 32 bits too high in these bitmaps.
    fn fix_shift_bug(&mut self) {
        if self.gicd_no_migration_shift_bug {
            return;
        }
        for bmp in [
            &mut self.group,
            &mut self.grpmod,
            &mut self.enabled,
            &mut self.pending,
            &mut self.active,
            &mut self.edge_trigger,
        ] {
            bmp.copy_within(1.., 0);
        }
        self.gicd_no_migration_shift_bug = true;
    }
}

fn flat(a: &[[u64; 4]; 3]) -> [u64; 12] {
    let mut f = [0; 12];
    for (i, v) in a.iter().flatten().enumerate() {
        f[i] = *v;
    }
    f
}

fn unflat(f: &[u64; 12]) -> [[u64; 4]; 3] {
    let mut a = [[0; 4]; 3];
    for (i, v) in a.iter_mut().flatten().enumerate() {
        *v = f[i];
    }
    a
}

impl GicV3 {
    /// The state for the `arm_gicv3` section. `el2` says whether the CPUs have EL2.
    pub fn vmstate_save(&self, el2: bool) -> GicV3VmState {
        let s = self.lock();
        let mut v = GicV3VmState {
            gicd_ctlr: s.gicd_ctlr,
            group: s.group,
            grpmod: s.grpmod,
            enabled: s.enabled,
            pending: s.pending,
            active: s.active,
            level: s.level,
            edge_trigger: s.edge_trigger,
            ..GicV3VmState::default()
        };
        let n = s.gicd_ipriority.len().min(MAXIRQ);
        v.gicd_ipriority[..n].copy_from_slice(&s.gicd_ipriority[..n]);
        v.gicd_irouter[..n].copy_from_slice(&s.gicd_irouter[..n]);
        let n = s.gicd_nsacr.len().min(v.gicd_nsacr.len());
        v.gicd_nsacr[..n].copy_from_slice(&s.gicd_nsacr[..n]);
        v.cpu = s
            .cpu
            .iter()
            .map(|cs| GicV3CpuVmState {
                level: cs.level,
                gicr_ctlr: cs.gicr_ctlr,
                gicr_statusr: [0; 2],
                gicr_waker: cs.gicr_waker,
                gicr_propbaser: cs.gicr_propbaser,
                gicr_pendbaser: cs.gicr_pendbaser,
                gicr_igroupr0: cs.gicr_igroupr0,
                gicr_ienabler0: cs.gicr_ienabler0,
                gicr_ipendr0: cs.gicr_ipendr0,
                gicr_iactiver0: cs.gicr_iactiver0,
                edge_trigger: cs.edge_trigger,
                gicr_igrpmodr0: cs.gicr_igrpmodr0,
                gicr_nsacr: cs.gicr_nsacr,
                gicr_ipriorityr: cs.gicr_ipriorityr,
                icc_ctlr_el1: cs.icc_ctlr_el1,
                icc_pmr_el1: cs.icc_pmr_el1,
                icc_bpr: cs.icc_bpr,
                icc_apr: flat(&cs.icc_apr),
                icc_igrpen: cs.icc_igrpen,
                icc_ctlr_el3: cs.icc_ctlr_el3,
                virt: el2,
                ich_apr: flat(&cs.ich_apr),
                ich_hcr_el2: cs.ich_hcr_el2,
                ich_lr_el2: cs.ich_lr_el2,
                ich_vmcr_el2: cs.ich_vmcr_el2,
                icc_sre_el1: 0,
            })
            .collect();
        v
    }

    /// Put in the state of an `arm_gicv3` section, then `gicv3_post_load()`: work out again
    /// the best pending interrupt of each CPU and the CPU each SPI routes to. The output lines
    /// are left alone, as the CPUs bring their own line levels.
    pub fn vmstate_load(&self, v: &GicV3VmState) -> Result<(), String> {
        let mut v = v.clone();
        if v.cpu.len() != self.num_cpu {
            return Err(format!("{} CPUs in the stream, {} here", v.cpu.len(), self.num_cpu));
        }
        v.fix_shift_bug();
        let mut s = self.lock();
        load_into(&mut s, &v);
        Ok(())
    }
}

fn load_into(s: &mut GicState, v: &GicV3VmState) {
    s.gicd_ctlr = v.gicd_ctlr;
    s.group = v.group;
    s.grpmod = v.grpmod;
    s.enabled = v.enabled;
    s.pending = v.pending;
    s.active = v.active;
    s.level = v.level;
    s.edge_trigger = v.edge_trigger;
    let n = s.gicd_ipriority.len().min(MAXIRQ);
    s.gicd_ipriority[..n].copy_from_slice(&v.gicd_ipriority[..n]);
    s.gicd_irouter[..n].copy_from_slice(&v.gicd_irouter[..n]);
    let n = s.gicd_nsacr.len().min(v.gicd_nsacr.len());
    s.gicd_nsacr[..n].copy_from_slice(&v.gicd_nsacr[..n]);
    for (cs, c) in s.cpu.iter_mut().zip(&v.cpu) {
        cs.level = c.level;
        cs.gicr_ctlr = c.gicr_ctlr;
        cs.gicr_waker = c.gicr_waker;
        cs.gicr_propbaser = c.gicr_propbaser;
        cs.gicr_pendbaser = c.gicr_pendbaser;
        cs.gicr_igroupr0 = c.gicr_igroupr0;
        cs.gicr_ienabler0 = c.gicr_ienabler0;
        cs.gicr_ipendr0 = c.gicr_ipendr0;
        cs.gicr_iactiver0 = c.gicr_iactiver0;
        cs.edge_trigger = c.edge_trigger;
        cs.gicr_igrpmodr0 = c.gicr_igrpmodr0;
        cs.gicr_nsacr = c.gicr_nsacr;
        cs.gicr_ipriorityr = c.gicr_ipriorityr;
        cs.icc_ctlr_el1 = c.icc_ctlr_el1;
        cs.icc_pmr_el1 = c.icc_pmr_el1;
        cs.icc_bpr = c.icc_bpr;
        cs.icc_apr = unflat(&c.icc_apr);
        cs.icc_igrpen = c.icc_igrpen;
        cs.icc_ctlr_el3 = c.icc_ctlr_el3;
        if c.virt {
            cs.ich_apr = unflat(&c.ich_apr);
            cs.ich_hcr_el2 = c.ich_hcr_el2;
            cs.ich_lr_el2 = c.ich_lr_el2;
            cs.ich_vmcr_el2 = c.ich_vmcr_el2;
        }
    }
    // arm_gicv3_post_load().
    for cpu in 0..s.cpu.len() {
        s.update_lpi_only(cpu);
    }
    s.full_update_noirqset();
    s.cache_all_target_cpustates();
}
