// SPDX-License-Identifier: GPL-2.0-or-later

//! `query-cpus-fast`, `qmp_query_cpus_fast()`: the CPUs of the board in `cpu_index` order, with
//! the QOM path and type QEMU gives them, the props of their slot in `possible_cpus` and the
//! host thread they run on.
//!
//! ruvm makes no QOM objects for the CPUs yet, so the paths are the ones the CPUs have in QEMU
//! and `qom-get` does not find them.

use std::fmt;
use std::sync::Arc;

use ruvm_accel::tcg::TcgVcpus;
use ruvm_accel_qtest::DummyVcpus;
use ruvm_qapi::types::{CpuInfoFast, CpuInfoFastU, CpuInstanceProperties};

/// Where the host thread ids of the vCPUs come from.
pub(crate) enum Threads {
    /// The vCPU threads of TCG.
    Tcg(Arc<TcgVcpus>),
    /// The threads of the qtest accelerator, which run no guest code.
    Dummy(DummyVcpus),
}

/// A CPU as `query-cpus-fast` lists it, apart from its thread.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CpuSlot {
    pub(crate) qom_path: String,
    pub(crate) qom_type: String,
    pub(crate) props: CpuInstanceProperties,
}

/// The CPUs of the board.
pub(crate) struct Cpus {
    slots: Vec<CpuSlot>,
    target: CpuInfoFastU,
    threads: Threads,
}

impl fmt::Debug for Cpus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cpus").field("slots", &self.slots).finish_non_exhaustive()
    }
}

impl Cpus {
    pub(crate) fn new(slots: Vec<CpuSlot>, target: CpuInfoFastU, threads: Threads) -> Cpus {
        Cpus { slots, target, threads }
    }

    /// `CPUState.thread_id` of each vCPU.
    pub(crate) fn thread_ids(&self) -> Vec<i64> {
        match &self.threads {
            Threads::Tcg(v) => v.cpus().iter().map(|c| c.thread_id()).collect(),
            Threads::Dummy(d) => d.thread_ids().to_vec(),
        }
    }

    /// `qmp_query_cpus_fast()`.
    pub(crate) fn query(&self) -> Vec<CpuInfoFast> {
        let slots = self.slots.iter().zip(self.thread_ids());
        (0..)
            .zip(slots)
            .map(|(cpu_index, (s, thread_id))| CpuInfoFast {
                cpu_index,
                qom_type: s.qom_type.clone(),
                qom_path: s.qom_path.clone(),
                thread_id,
                props: Some(s.props.clone()),
                u: self.target.clone(),
            })
            .collect()
    }
}

/// The props `virt_possible_cpu_arch_ids()` on Arm and `riscv_numa_possible_cpu_arch_ids()`
/// give possible CPU `n` in a topology of `clusters`, `cores` and `threads`.
pub(crate) fn topology_props(
    n: u32,
    clusters: u32,
    cores: u32,
    threads: u32,
) -> CpuInstanceProperties {
    CpuInstanceProperties {
        socket_id: Some(i64::from(n / (clusters * cores * threads))),
        cluster_id: Some(i64::from((n / (cores * threads)) % clusters)),
        core_id: Some(i64::from((n / threads) % cores)),
        thread_id: Some(i64::from(n % threads)),
        ..CpuInstanceProperties::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn props_follow_the_topology() {
        // -smp 4,maxcpus=8,sockets=2,clusters=2,cores=2 on Arm virt, as QEMU lists it.
        let ids = |p: CpuInstanceProperties| {
            (p.socket_id.unwrap(), p.cluster_id.unwrap(), p.core_id.unwrap(), p.thread_id.unwrap())
        };
        let got: Vec<_> = (0..8).map(|n| ids(topology_props(n, 2, 2, 1))).collect();
        assert_eq!(
            got,
            [(0, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 1, 1, 0)]
                .into_iter()
                .chain([(1, 0, 0, 0), (1, 0, 1, 0), (1, 1, 0, 0), (1, 1, 1, 0)])
                .collect::<Vec<_>>()
        );
        assert_eq!(ids(topology_props(5, 1, 2, 2)), (1, 0, 0, 1));
    }

    #[test]
    fn query_lists_the_slots_with_their_threads() {
        let slot = |i: u32| CpuSlot {
            qom_path: format!("/machine/soc0/harts[{i}]"),
            qom_type: "rv64-riscv-cpu".into(),
            props: topology_props(i, 1, 2, 1),
        };
        let cpus = Cpus::new(
            vec![slot(0), slot(1)],
            CpuInfoFastU::Riscv64,
            Threads::Dummy(DummyVcpus::start(2).unwrap()),
        );
        let out = cpus.query();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].cpu_index, 1);
        assert_eq!(out[1].qom_path, "/machine/soc0/harts[1]");
        assert_eq!(out[1].props.as_ref().unwrap().core_id, Some(1));
        assert!(out.iter().all(|c| c.thread_id > 0));
    }
}
