// SPDX-License-Identifier: GPL-2.0-or-later

//! The `arm_gicv3` and `arm_gicv3_its` sections, from hw/intc/arm_gicv3_common.c and
//! hw/intc/arm_gicv3_its_common.c. The state itself is gathered and put back by
//! [`ruvm_hw_intc::gicv3::migrate`] and [`GicV3Its`].

use std::sync::{Arc, LazyLock};

use ruvm_base::err;
use ruvm_hw_intc::gicv3::migrate::{GicV3CpuVmState, GicV3VmState};
use ruvm_hw_intc::gicv3::{GicV3, GicV3Its, ItsVmState};
use ruvm_migration::SaveVm;
use ruvm_vmstate::{MigPriority, VmStateDescription, VmStateField};

/// `vmstate_gicv3_cpu_virt`.
static VMSTATE_GICV3_CPU_VIRT: LazyLock<VmStateDescription<GicV3CpuVmState>> =
    LazyLock::new(|| {
        type S = GicV3CpuVmState;
        VmStateDescription::new("arm_gicv3_cpu/virt")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|s: &S| s.virt)
            .fields([
                VmStateField::array("ich_apr", |s: &mut S| &mut s.ich_apr),
                VmStateField::scalar("ich_hcr_el2", |s: &mut S| &mut s.ich_hcr_el2),
                VmStateField::array("ich_lr_el2", |s: &mut S| &mut s.ich_lr_el2),
                VmStateField::scalar("ich_vmcr_el2", |s: &mut S| &mut s.ich_vmcr_el2),
            ])
    });

/// `vmstate_gicv3_cpu_sre_el1`.
static VMSTATE_GICV3_CPU_SRE_EL1: LazyLock<VmStateDescription<GicV3CpuVmState>> =
    LazyLock::new(|| {
        type S = GicV3CpuVmState;
        VmStateDescription::new("arm_gicv3_cpu/sre_el1")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|s: &S| s.icc_sre_el1 != 7)
            .field(VmStateField::scalar("icc_sre_el1", |s: &mut S| &mut s.icc_sre_el1))
    });

/// `vmstate_gicv3_cpu`. ruvm has no GICv4 and no NMIs, so the `gicv4` and `nmi` subsections
/// are never needed.
static VMSTATE_GICV3_CPU: LazyLock<VmStateDescription<GicV3CpuVmState>> = LazyLock::new(|| {
    type S = GicV3CpuVmState;
    VmStateDescription::new("arm_gicv3_cpu")
        .version_id(1)
        .minimum_version_id(1)
        .pre_load(|s: &mut S| {
            s.pre_load();
            0
        })
        .fields([
            VmStateField::scalar("level", |s: &mut S| &mut s.level),
            VmStateField::scalar("gicr_ctlr", |s: &mut S| &mut s.gicr_ctlr),
            VmStateField::array("gicr_statusr", |s: &mut S| &mut s.gicr_statusr),
            VmStateField::scalar("gicr_waker", |s: &mut S| &mut s.gicr_waker),
            VmStateField::scalar("gicr_propbaser", |s: &mut S| &mut s.gicr_propbaser),
            VmStateField::scalar("gicr_pendbaser", |s: &mut S| &mut s.gicr_pendbaser),
            VmStateField::scalar("gicr_igroupr0", |s: &mut S| &mut s.gicr_igroupr0),
            VmStateField::scalar("gicr_ienabler0", |s: &mut S| &mut s.gicr_ienabler0),
            VmStateField::scalar("gicr_ipendr0", |s: &mut S| &mut s.gicr_ipendr0),
            VmStateField::scalar("gicr_iactiver0", |s: &mut S| &mut s.gicr_iactiver0),
            VmStateField::scalar("edge_trigger", |s: &mut S| &mut s.edge_trigger),
            VmStateField::scalar("gicr_igrpmodr0", |s: &mut S| &mut s.gicr_igrpmodr0),
            VmStateField::scalar("gicr_nsacr", |s: &mut S| &mut s.gicr_nsacr),
            VmStateField::array("gicr_ipriorityr", |s: &mut S| &mut s.gicr_ipriorityr),
            VmStateField::array("icc_ctlr_el1", |s: &mut S| &mut s.icc_ctlr_el1),
            VmStateField::scalar("icc_pmr_el1", |s: &mut S| &mut s.icc_pmr_el1),
            VmStateField::array("icc_bpr", |s: &mut S| &mut s.icc_bpr),
            VmStateField::array("icc_apr", |s: &mut S| &mut s.icc_apr),
            VmStateField::array("icc_igrpen", |s: &mut S| &mut s.icc_igrpen),
            VmStateField::scalar("icc_ctlr_el3", |s: &mut S| &mut s.icc_ctlr_el3),
        ])
        .subsection(&VMSTATE_GICV3_CPU_VIRT)
        .subsection(&VMSTATE_GICV3_CPU_SRE_EL1)
});

/// `vmstate_gicv3_gicd_no_migration_shift_bug`.
static VMSTATE_GICV3_SHIFT_BUG: LazyLock<VmStateDescription<GicV3VmState>> = LazyLock::new(|| {
    type S = GicV3VmState;
    VmStateDescription::new("arm_gicv3/gicd_no_migration_shift_bug")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|_: &S| true)
        .field(VmStateField::scalar("gicd_no_migration_shift_bug", |s: &mut S| {
            &mut s.gicd_no_migration_shift_bug
        }))
});

/// `vmstate_gicv3`. `gicv3_post_load()` is [`GicV3::vmstate_load`].
static VMSTATE_GICV3: LazyLock<VmStateDescription<GicV3VmState>> = LazyLock::new(|| {
    type S = GicV3VmState;
    VmStateDescription::new("arm_gicv3")
        .version_id(1)
        .minimum_version_id(1)
        .priority(MigPriority::Gicv3)
        .pre_load(|s: &mut S| {
            s.pre_load();
            0
        })
        .fields([
            VmStateField::scalar("gicd_ctlr", |s: &mut S| &mut s.gicd_ctlr),
            VmStateField::array("gicd_statusr", |s: &mut S| &mut s.gicd_statusr),
            VmStateField::array("group", |s: &mut S| &mut s.group),
            VmStateField::array("grpmod", |s: &mut S| &mut s.grpmod),
            VmStateField::array("enabled", |s: &mut S| &mut s.enabled),
            VmStateField::array("pending", |s: &mut S| &mut s.pending),
            VmStateField::array("active", |s: &mut S| &mut s.active),
            VmStateField::array("level", |s: &mut S| &mut s.level),
            VmStateField::array("edge_trigger", |s: &mut S| &mut s.edge_trigger),
            VmStateField::array("gicd_ipriority", |s: &mut S| &mut *s.gicd_ipriority),
            VmStateField::array("gicd_irouter", |s: &mut S| &mut *s.gicd_irouter),
            VmStateField::array("gicd_nsacr", |s: &mut S| &mut s.gicd_nsacr),
            VmStateField::struct_varray(
                "cpu",
                |s: &S| s.cpu.len(),
                &VMSTATE_GICV3_CPU,
                |s: &mut S| &mut s.cpu,
            ),
        ])
        .subsection(&VMSTATE_GICV3_SHIFT_BUG)
});

/// `vmstate_its`. `gicv3_its_post_load()` is [`GicV3Its::vmstate_load`].
static VMSTATE_ITS: LazyLock<VmStateDescription<ItsVmState>> = LazyLock::new(|| {
    type S = ItsVmState;
    VmStateDescription::new("arm_gicv3_its").priority(MigPriority::Gicv3Its).fields([
        VmStateField::scalar("ctlr", |s: &mut S| &mut s.ctlr),
        VmStateField::scalar("iidr", |s: &mut S| &mut s.iidr),
        VmStateField::scalar("cbaser", |s: &mut S| &mut s.cbaser),
        VmStateField::scalar("cwriter", |s: &mut S| &mut s.cwriter),
        VmStateField::scalar("creadr", |s: &mut S| &mut s.creadr),
        VmStateField::array("baser", |s: &mut S| &mut s.baser),
    ])
});

/// Registers the `arm_gicv3` section of `gic`. `el2` says whether the CPUs have EL2.
pub(super) fn register_gicv3(savevm: &mut SaveVm, gic: &Arc<GicV3>, el2: bool) {
    let (get, put) = (Arc::clone(gic), Arc::clone(gic));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_GICV3,
        move || Ok(get.vmstate_save(el2)),
        move |s| put.vmstate_load(&s).map_err(|e| err!("arm_gicv3: {e}")),
    );
}

/// Registers the `arm_gicv3_its` section of `its`.
pub(super) fn register_its(savevm: &mut SaveVm, its: &Arc<GicV3Its>) {
    let (get, put) = (Arc::clone(its), Arc::clone(its));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_ITS,
        move || Ok(get.vmstate_save()),
        move |s| {
            put.vmstate_load(&s);
            Ok(())
        },
    );
}

#[cfg(test)]
mod tests {
    use ruvm_hw_intc::gicv3::GicV3Props;
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_load_state, vmstate_save_state};

    use super::*;

    fn save<T>(vmsd: &VmStateDescription<T>, mut s: T) -> Vec<u8> {
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, vmsd, &mut s).unwrap();
        f.into_inner()
    }

    /// Loads `bytes` into `s`, which is what the destination saves before loading.
    fn load<T>(vmsd: &VmStateDescription<T>, bytes: &[u8], mut s: T) -> T {
        let mut b = bytes.to_vec();
        // The byte after the section, which is not a subsection.
        b.push(0);
        vmstate_load_state(&mut StreamReader::new(&b), vmsd, &mut s, vmsd.version_id).unwrap();
        s
    }

    fn be32(b: &[u8], at: usize) -> u32 {
        u32::from_be_bytes(b[at..at + 4].try_into().unwrap())
    }

    fn be64(b: &[u8], at: usize) -> u64 {
        u64::from_be_bytes(b[at..at + 8].try_into().unwrap())
    }

    /// The GIC of a one-CPU `virt` out of reset.
    fn new_gic() -> Arc<GicV3> {
        let gic = GicV3::new(GicV3Props {
            num_irq: 288,
            redist_region_count: vec![1],
            mp_affinity: vec![0],
            ..GicV3Props::default()
        })
        .unwrap();
        gic.reset();
        gic.cpuif_reset(0);
        gic
    }

    /// The distributor fields come to 10344 bytes and each CPU's to 272. Without EL2, QEMU's TCG
    /// GIC sends the `sre_el1` subsection with 0 and no `virt` subsection.
    #[test]
    fn gicv3_layout_matches_qemu() {
        let gic = new_gic();
        let bytes = save(&VMSTATE_GICV3, gic.vmstate_save(false));
        assert_eq!(be32(&bytes, 0), 0x50);
        let cpu = 10344;
        assert_eq!(be32(&bytes, cpu + 16), 6, "gicr_waker");
        assert_eq!(be32(&bytes, cpu + 52), 0xffff, "edge_trigger");
        assert_eq!(be64(&bytes, cpu + 96), 0x8c00, "icc_ctlr_el1");
        let bpr: Vec<u64> = (0..3).map(|i| be64(&bytes, cpu + 120 + 8 * i)).collect();
        assert_eq!(bpr, [2, 2, 3]);
        assert_eq!(be64(&bytes, cpu + 264), 0x28c00, "icc_ctlr_el3");
        let mut tail = vec![0x05, 21];
        tail.extend_from_slice(b"arm_gicv3_cpu/sre_el1");
        tail.extend_from_slice(&1u32.to_be_bytes());
        tail.extend_from_slice(&0u64.to_be_bytes());
        tail.extend_from_slice(&[0x05, 37]);
        tail.extend_from_slice(b"arm_gicv3/gicd_no_migration_shift_bug");
        tail.extend_from_slice(&1u32.to_be_bytes());
        tail.push(1);
        assert_eq!(bytes[cpu + 272..], tail);

        // With EL2 the list registers go too.
        let el2 = save(&VMSTATE_GICV3, gic.vmstate_save(true));
        assert_eq!(el2.len(), bytes.len() + 2 + 18 + 4 + 240);
        assert_eq!(&el2[cpu + 274..cpu + 292], b"arm_gicv3_cpu/virt");
    }

    #[test]
    fn gicv3_round_trip() {
        let gic = new_gic();
        gic.spi(3).set(1);
        let bytes = save(&VMSTATE_GICV3, gic.vmstate_save(true));
        let other = new_gic();
        let v = load(&VMSTATE_GICV3, &bytes, other.vmstate_save(true));
        assert_eq!(v.cpu[0].icc_sre_el1, 0);
        other.vmstate_load(&v).unwrap();
        assert_eq!(save(&VMSTATE_GICV3, other.vmstate_save(true)), bytes);
        let two = GicV3VmState { cpu: vec![v.cpu[0].clone(); 2], ..v };
        assert!(other.vmstate_load(&two).is_err());
    }

    /// A stream from a QEMU with the shift bug has the SPI bitmaps one word too high.
    #[test]
    fn gicv3_shift_bug() {
        let gic = new_gic();
        let mut v = gic.vmstate_save(false);
        v.enabled[2] = 0x10;
        v.gicd_no_migration_shift_bug = false;
        gic.vmstate_load(&v).unwrap();
        let back = gic.vmstate_save(false);
        assert_eq!((back.enabled[1], back.enabled[2]), (0x10, 0));
    }

    /// The ITS of QEMU's `virt` out of reset.
    #[test]
    fn its_layout_matches_qemu() {
        let mut v = ItsVmState { ctlr: 0x8000_0000, ..ItsVmState::default() };
        v.baser[0] = 0x0107_0000_0000_0200;
        v.baser[1] = 0x0407_0000_0000_0200;
        let bytes = save(&VMSTATE_ITS, v.clone());
        assert_eq!(bytes.len(), 96);
        assert_eq!(be32(&bytes, 0), 0x8000_0000);
        assert_eq!(be64(&bytes, 32), 0x0107_0000_0000_0200);
        assert_eq!(be64(&bytes, 40), 0x0407_0000_0000_0200);
        let back = load(&VMSTATE_ITS, &bytes, ItsVmState::default());
        assert_eq!(back.baser, v.baser);
    }
}
