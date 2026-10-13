// SPDX-License-Identifier: GPL-2.0-or-later

//! `-numa`: the parsing of hw/core/numa.c and the CPU mapping of hw/core/machine.c.
//!
//! The board gives the props of its possible CPUs. The options assign nodes to them, then
//! [`NumaState::complete`] checks the nodes against the RAM and gives the CPUs no option named
//! their default node.

use ruvm_base::report::{Location, warn_report};
use ruvm_base::{Error, Result};
use ruvm_machine_arm::virt::{CpuTopology, VIRT_DEFAULT_RAM_SIZE};
use ruvm_net::OptsVisitor;
use ruvm_qapi::opts::QemuOpts;
use ruvm_qapi::types::{CpuInstanceProperties, NumaNodeOptions, NumaOptions, NumaOptionsU};
use ruvm_qapi::visit::Visit;

use crate::arm::BoardOptions;
use crate::cpus::{Cpus, topology_props};
use crate::vl::Vm;
use crate::x86::{Located, consume_memdev};

/// `MAX_NODES`.
pub(crate) const MAX_NODES: usize = 128;
/// `NUMA_DISTANCE_MIN`, the distance of a node to itself.
pub(crate) const NUMA_DISTANCE_MIN: u8 = 10;

/// `NodeInfo`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NodeInfo {
    /// `node_mem`.
    pub(crate) mem: u64,
    /// The id of `node_memdev`.
    pub(crate) memdev: Option<String>,
    present: bool,
    /// `distance[]`, 0 where none was given.
    pub(crate) distance: Vec<u8>,
}

impl Default for NodeInfo {
    fn default() -> NodeInfo {
        NodeInfo { mem: 0, memdev: None, present: false, distance: vec![0; MAX_NODES] }
    }
}

/// `NumaState`, with the props of the possible CPUs (`ms->possible_cpus`) it maps.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NumaState {
    /// `nodes[]`. Once [`complete`](NumaState::complete) is done, the nodes there are, each
    /// with `num_nodes` distances.
    pub(crate) nodes: Vec<NodeInfo>,
    pub(crate) num_nodes: usize,
    pub(crate) have_numa_distance: bool,
    /// The props of each possible CPU, with `node_id` once a node has it.
    pub(crate) cpus: Vec<CpuInstanceProperties>,
    have_memdevs: bool,
    have_mem: bool,
    /// `max_numa_nodeid`: the highest node id given, plus one.
    max_numa_nodeid: usize,
}

const HMAT_DISABLED: &str = "ACPI Heterogeneous Memory Attribute Table (HMAT) is disabled, \
                             enable it with -machine hmat=on before using any of hmat specific \
                             options";

impl NumaState {
    /// The state before any `-numa`, for a board with these possible CPUs.
    pub(crate) fn new(cpus: Vec<CpuInstanceProperties>) -> NumaState {
        NumaState {
            nodes: vec![NodeInfo::default(); MAX_NODES],
            num_nodes: 0,
            have_numa_distance: false,
            cpus,
            have_memdevs: false,
            have_mem: false,
            max_numa_nodeid: 0,
        }
    }

    /// `set_numa_options()`. `memdev_size` gives the size of a memory backend, or `None` if
    /// there is no backend with that id. The board takes no `-numa node,mem`.
    pub(crate) fn set_numa_options(
        &mut self,
        object: &NumaOptions,
        memdev_size: impl Fn(&str) -> Option<u64>,
    ) -> Result<()> {
        match &object.u {
            NumaOptionsU::Node(node) => self.parse_numa_node(node, memdev_size),
            NumaOptionsU::Dist(dist) => {
                self.parse_numa_distance(dist.src.into(), dist.dst.into(), dist.val)
            }
            NumaOptionsU::Cpu(cpu) => {
                let Some(node_id) = cpu.node_id else {
                    return Err(Error::generic("Missing mandatory node-id property"));
                };
                if !usize::try_from(node_id)
                    .is_ok_and(|n| self.nodes.get(n).is_some_and(|n| n.present))
                {
                    return Err(Error::generic(format!(
                        "Invalid node-id={node_id}, NUMA node must be defined with -numa \
                         node,nodeid=ID before it's used with -numa cpu,node-id=ID"
                    )));
                }
                let props = CpuInstanceProperties {
                    node_id: cpu.node_id,
                    drawer_id: cpu.drawer_id,
                    book_id: cpu.book_id,
                    socket_id: cpu.socket_id,
                    die_id: cpu.die_id,
                    cluster_id: cpu.cluster_id,
                    module_id: cpu.module_id,
                    core_id: cpu.core_id,
                    thread_id: cpu.thread_id,
                };
                self.set_cpu_numa_node(&props)
            }
            NumaOptionsU::HmatLb(_) | NumaOptionsU::HmatCache(_) => {
                Err(Error::generic(HMAT_DISABLED))
            }
        }
    }

    /// `parse_numa_node()`.
    fn parse_numa_node(
        &mut self,
        node: &NumaNodeOptions,
        memdev_size: impl Fn(&str) -> Option<u64>,
    ) -> Result<()> {
        let nodenr = node.nodeid.map_or(self.num_nodes, usize::from);
        if nodenr >= MAX_NODES {
            return Err(Error::generic(format!("Max number of NUMA nodes reached: {nodenr}")));
        }
        if self.nodes[nodenr].present {
            return Err(Error::generic(format!("Duplicate NUMA nodeid: {nodenr}")));
        }
        if node.initiator.is_some() {
            return Err(Error::generic(HMAT_DISABLED));
        }
        for &cpu in node.cpus.iter().flatten() {
            if usize::from(cpu) >= self.cpus.len() {
                return Err(Error::generic(format!(
                    "CPU index ({cpu}) should be smaller than maxcpus ({})",
                    self.cpus.len()
                )));
            }
            // cpu_index_to_instance_props()
            let mut props = self.cpus[usize::from(cpu)].clone();
            props.node_id = Some(nodenr as i64);
            self.set_cpu_numa_node(&props)?;
        }
        self.have_memdevs |= node.memdev.is_some();
        self.have_mem |= node.mem.is_some();
        if (node.mem.is_some() && self.have_memdevs) || (node.memdev.is_some() && self.have_mem) {
            return Err(Error::generic(
                "numa configuration should use either mem= or memdev=,mixing both is not allowed",
            ));
        }
        if node.mem.is_some() {
            return Err(Error::generic(
                "Parameter -numa node,mem is not supported by this machine type",
            )
            .hint("Use -numa node,memdev instead\n"));
        }
        if let Some(id) = &node.memdev {
            let Some(size) = memdev_size(id) else {
                return Err(Error::generic(format!("memdev={id} is ambiguous")));
            };
            self.nodes[nodenr].mem = size;
            self.nodes[nodenr].memdev = Some(id.clone());
        }
        self.nodes[nodenr].present = true;
        self.max_numa_nodeid = self.max_numa_nodeid.max(nodenr + 1);
        self.num_nodes += 1;
        Ok(())
    }

    /// `parse_numa_distance()`.
    fn parse_numa_distance(&mut self, src: usize, dst: usize, val: u8) -> Result<()> {
        if src >= MAX_NODES || dst >= MAX_NODES {
            let which = if src >= MAX_NODES { "src" } else { "dst" };
            return Err(Error::generic(format!(
                "Parameter '{which}' expects an integer between 0 and {}",
                MAX_NODES - 1
            )));
        }
        if !self.nodes[src].present || !self.nodes[dst].present {
            return Err(Error::generic(
                "Source/Destination NUMA node is missing. Please use '-numa node' option to \
                 declare it first.",
            ));
        }
        if val < NUMA_DISTANCE_MIN {
            return Err(Error::generic(format!(
                "NUMA distance ({val}) is invalid, it shouldn't be less than \
                 {NUMA_DISTANCE_MIN}."
            )));
        }
        if src == dst && val != NUMA_DISTANCE_MIN {
            return Err(Error::generic(format!(
                "Local distance of node {src} should be {NUMA_DISTANCE_MIN}."
            )));
        }
        self.nodes[src].distance[dst] = val;
        self.have_numa_distance = true;
        Ok(())
    }

    /// `machine_set_cpu_numa_node()`: gives the node of `props` to every possible CPU that
    /// matches the ids it has.
    pub(crate) fn set_cpu_numa_node(&mut self, props: &CpuInstanceProperties) -> Result<()> {
        let mut matched = false;
        for slot in &mut self.cpus {
            for (want, have, name) in [
                (props.thread_id, slot.thread_id, "thread-id"),
                (props.core_id, slot.core_id, "core-id"),
                (props.module_id, slot.module_id, "module-id"),
                (props.cluster_id, slot.cluster_id, "cluster-id"),
                (props.socket_id, slot.socket_id, "socket-id"),
                (props.die_id, slot.die_id, "die-id"),
            ] {
                if want.is_some() && have.is_none() {
                    return Err(Error::generic(format!("{name} is not supported")));
                }
            }
            let ids = [
                (props.thread_id, slot.thread_id),
                (props.core_id, slot.core_id),
                (props.module_id, slot.module_id),
                (props.cluster_id, slot.cluster_id),
                (props.die_id, slot.die_id),
                (props.socket_id, slot.socket_id),
            ];
            if ids.iter().any(|(want, have)| want.is_some() && want != have) {
                continue;
            }
            if let Some(node) = slot.node_id.filter(|&n| Some(n) != props.node_id) {
                return Err(Error::generic(format!("CPU is already assigned to node-id: {node}")));
            }
            matched = true;
            slot.node_id = props.node_id;
        }
        if !matched {
            return Err(Error::generic("no match found"));
        }
        Ok(())
    }

    /// `numa_complete_configuration()` without the nodes memory hotplug adds, then
    /// `machine_numa_finish_cpu_init()` and `validate_cpu_cluster_to_numa_boundary()`.
    /// `machine_memdev` is whether `-machine memory-backend` was given, `default_node` is
    /// `get_default_cpu_node_id()` and `qtest` whether the qtest accelerator runs, which
    /// keeps the warnings quiet.
    pub(crate) fn complete(
        &mut self,
        ram_size: u64,
        machine_memdev: bool,
        default_node: impl Fn(&NumaState, usize) -> i64,
        qtest: bool,
    ) -> Result<()> {
        // No support for sparse NUMA node IDs yet. Large ids are reported first.
        if let Some(i) = (0..self.max_numa_nodeid).rev().find(|&i| !self.nodes[i].present) {
            return Err(Error::generic(format!("numa: Node ID missing: {i}")));
        }
        let n = self.num_nodes;
        self.nodes.truncate(n);
        if n == 0 {
            return Ok(());
        }
        let total: u64 = self.nodes.iter().map(|node| node.mem).sum();
        if total != ram_size {
            return Err(Error::generic(format!(
                "total memory for NUMA nodes (0x{total:x}) should equal RAM size (0x{ram_size:x})"
            )));
        }
        if self.have_memdevs && machine_memdev {
            return Err(Error::generic(
                "'-machine memory-backend' and '-numa memdev' properties are mutually exclusive",
            ));
        }
        if self.have_numa_distance {
            self.validate_numa_distance()?;
            self.complete_init_numa_distance();
        }
        for node in &mut self.nodes {
            node.distance.truncate(n);
        }
        if !self.have_numa_distance {
            self.nodes.iter_mut().for_each(|node| node.distance.clear());
        }
        self.finish_cpu_init(default_node, qtest);
        if !qtest {
            self.validate_cpu_cluster_to_numa_boundary();
        }
        Ok(())
    }

    /// `validate_numa_distance()`.
    fn validate_numa_distance(&self) -> Result<()> {
        let n = self.num_nodes;
        let d = |src: usize, dst: usize| self.nodes[src].distance[dst];
        let mut asymmetric = false;
        for src in 0..n {
            for dst in src..n {
                if d(src, dst) == 0 && d(dst, src) == 0 && src != dst {
                    return Err(Error::generic(format!(
                        "The distance between node {src} and {dst} is missing, at least one \
                         distance value between each nodes should be provided."
                    )));
                }
                if d(src, dst) != 0 && d(dst, src) != 0 && d(src, dst) != d(dst, src) {
                    asymmetric = true;
                }
            }
        }
        if asymmetric && (0..n).any(|src| (0..n).any(|dst| src != dst && d(src, dst) == 0)) {
            return Err(Error::generic(
                "At least one asymmetrical pair of distances is given, please provide distances \
                 for both directions of all node pairs.",
            ));
        }
        Ok(())
    }

    /// `complete_init_numa_distance()`.
    fn complete_init_numa_distance(&mut self) {
        let n = self.num_nodes;
        for src in 0..n {
            for dst in 0..n {
                if self.nodes[src].distance[dst] == 0 {
                    self.nodes[src].distance[dst] =
                        if src == dst { NUMA_DISTANCE_MIN } else { self.nodes[dst].distance[src] };
                }
            }
        }
    }

    /// `machine_numa_finish_cpu_init()`.
    fn finish_cpu_init(&mut self, default_node: impl Fn(&NumaState, usize) -> i64, qtest: bool) {
        let default_mapping = self.cpus.iter().all(|c| c.node_id.is_none());
        let mut unmapped = Vec::new();
        for i in 0..self.cpus.len() {
            if self.cpus[i].node_id.is_some() {
                continue;
            }
            let node = if default_mapping {
                default_node(self, i)
            } else {
                unmapped.push(format!("CPU {i} [{}]", cpu_slot_to_string(&self.cpus[i])));
                // CPUs no option named used to go to node 0.
                0
            };
            self.cpus[i].node_id = Some(node);
        }
        if !unmapped.is_empty() && !qtest {
            warn_report(&format!("CPU(s) not present in any NUMA nodes: {}", unmapped.join(", ")));
            warn_report(
                "All CPU(s) up to maxcpus should be described in NUMA config, ability to start \
                 up with partial NUMA mappings is obsoleted and will be removed in future",
            );
        }
    }

    /// `validate_cpu_cluster_to_numa_boundary()`.
    fn validate_cpu_cluster_to_numa_boundary(&self) {
        if self.num_nodes <= 1 {
            return;
        }
        for (i, a) in self.cpus.iter().enumerate() {
            for (j, b) in self.cpus.iter().enumerate().skip(i + 1) {
                let (Some(sa), Some(ca), Some(na)) = (a.socket_id, a.cluster_id, a.node_id) else {
                    continue;
                };
                if b.socket_id == Some(sa) && b.cluster_id == Some(ca) && b.node_id != Some(na) {
                    warn_report(&format!(
                        "CPU-{i} and CPU-{j} in socket-{sa}-cluster-{ca} have been associated \
                         with node-{na} and node-{} respectively. It can cause OSes like Linux \
                         to misbehave",
                        b.node_id.unwrap_or_default()
                    ));
                }
            }
        }
    }

    /// The node of possible CPU `n`.
    #[cfg(test)]
    fn cpu_node(&self, n: usize) -> Option<i64> {
        self.cpus.get(n).and_then(|c| c.node_id)
    }
}

/// Visits one `-numa` as the `NumaOptions` it gives, as `parse_numa()` does.
pub(crate) fn visit(opts: &QemuOpts) -> Result<NumaOptions> {
    let mut o = NumaOptions::default();
    NumaOptions::visit(&mut OptsVisitor::new(opts), None, &mut o)?;
    Ok(o)
}

/// `parse_numa_opts()` for Arm virt, then the `numa_complete_configuration()` and
/// `machine_numa_finish_cpu_init()` of `machine_run_board_init()`. `list` holds each `-numa`
/// with where it was given. The memory of the nodes goes to `board`, and what `info numa`
/// shows to `vm`.
pub(crate) fn virt_numa(
    vm: &Vm,
    list: &[(QemuOpts, Option<Location>)],
    board: &mut BoardOptions,
    machine_memdev: bool,
    qtest: bool,
) -> std::result::Result<(), Located> {
    // virt_possible_cpu_arch_ids()
    let topo = board.topology.unwrap_or_else(|| CpuTopology::flat(board.max_cpus));
    let cpus = (0..board.max_cpus)
        .map(|n| topology_props(n, topo.clusters, topo.cores, topo.threads))
        .collect();
    let mut state = NumaState::new(cpus);
    let memdev_size = |id: &str| {
        let (backend, _) = vm.registry.resolve_path_type(id, "memory-backend");
        backend.map(|b| b.property_get_uint("size").unwrap_or_default())
    };
    for (opts, loc) in list {
        let located = |e| Located(loc.clone(), e);
        let object = visit(opts).map_err(located)?;
        state.set_numa_options(&object, memdev_size).map_err(located)?;
    }
    // virt_get_default_cpu_node_id()
    let by_socket =
        |s: &NumaState, i: usize| s.cpus[i].socket_id.unwrap_or_default() % s.num_nodes as i64;
    let ram_size = board.ram_size.unwrap_or(VIRT_DEFAULT_RAM_SIZE);
    let local = |e| Located(None, e);
    state.complete(ram_size, machine_memdev, by_socket, qtest).map_err(local)?;
    // numa_init_memdev_container()
    for node in &state.nodes {
        let ram = node.memdev.as_deref().map(|id| consume_memdev(vm, id)).transpose();
        board.numa_ram.push((node.mem, ram.map_err(local)?));
    }
    if state.num_nodes > 0 {
        let _ = vm.numa.set(state);
    }
    Ok(())
}

/// `cpu_slot_to_string()`.
fn cpu_slot_to_string(props: &CpuInstanceProperties) -> String {
    let ids = [
        ("socket-id", props.socket_id),
        ("die-id", props.die_id),
        ("cluster-id", props.cluster_id),
        ("module-id", props.module_id),
        ("core-id", props.core_id),
        ("thread-id", props.thread_id),
    ];
    let parts: Vec<String> =
        ids.iter().filter_map(|(name, v)| v.map(|v| format!("{name}: {v}"))).collect();
    parts.join(", ")
}

/// `qmp_x_query_numa()`, what `info numa` prints.
pub(crate) fn x_query_numa(vm: &Vm) -> String {
    let cpus = vm.cpus.get().map(Cpus::query).unwrap_or_default();
    let cpus: Vec<_> =
        cpus.iter().map(|c| (c.cpu_index, c.props.as_ref().and_then(|p| p.node_id))).collect();
    format_numa(vm.numa.get(), &cpus)
}

/// Each node with the present CPUs on it, which `cpus` lists as (cpu-index, node). No memory
/// is plugged into the nodes.
fn format_numa(numa: Option<&NumaState>, cpus: &[(i64, Option<i64>)]) -> String {
    let nodes = numa.map_or(&[][..], |s| &s.nodes[..]);
    let mut out = format!("{} nodes\n", nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        out.push_str(&format!("node {i} cpus:"));
        for (index, _) in cpus.iter().filter(|(_, n)| *n == Some(i as i64)) {
            out.push_str(&format!(" {index}"));
        }
        out.push('\n');
        out.push_str(&format!("node {i} size: {} MB\n", node.mem >> 20));
        out.push_str(&format!("node {i} plugged: 0 MB\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use ruvm_qapi::opts::QemuOptsList;

    use super::*;
    use crate::cpus::topology_props;

    const MIB: u64 = 1 << 20;

    fn state(cpus: u32, sockets: u32) -> NumaState {
        let cores = cpus / sockets;
        NumaState::new((0..cpus).map(|n| topology_props(n, 1, cores, 1)).collect())
    }

    fn set(s: &mut NumaState, arg: &str) -> Result<()> {
        let mut list = QemuOptsList::new("numa", &[]).with_implied_opt_name("type");
        let o = visit(list.parse(arg, true)?)?;
        s.set_numa_options(&o, |id| (id == "ram").then_some(128 * MIB))
    }

    // virt_get_default_cpu_node_id()
    fn by_socket(s: &NumaState, i: usize) -> i64 {
        s.cpus[i].socket_id.unwrap_or_default() % s.num_nodes as i64
    }

    fn nodes_of(s: &NumaState) -> Vec<i64> {
        (0..s.cpus.len()).map(|i| s.cpu_node(i).unwrap()).collect()
    }

    #[test]
    fn default_split_follows_the_sockets() {
        let mut s = state(8, 8);
        set(&mut s, "node,memdev=ram").unwrap();
        set(&mut s, "node").unwrap();
        s.complete(128 * MIB, false, by_socket, true).unwrap();
        assert_eq!(nodes_of(&s), [0, 1, 0, 1, 0, 1, 0, 1]);
        let cpus: Vec<_> = (0..8).map(|i| (i, s.cpu_node(i as usize))).collect();
        assert_eq!(
            format_numa(Some(&s), &cpus),
            "2 nodes\nnode 0 cpus: 0 2 4 6\nnode 0 size: 128 MB\nnode 0 plugged: 0 MB\n\
             node 1 cpus: 1 3 5 7\nnode 1 size: 0 MB\nnode 1 plugged: 0 MB\n"
        );
    }

    #[test]
    fn partial_mappings_go_to_node_0() {
        let mut s = state(8, 1);
        set(&mut s, "node,nodeid=0,memdev=ram,cpus=0-1").unwrap();
        set(&mut s, "node,nodeid=1,cpus=4-5").unwrap();
        s.complete(128 * MIB, false, by_socket, true).unwrap();
        assert_eq!(nodes_of(&s), [0, 0, 0, 0, 1, 1, 0, 0]);
    }

    #[test]
    fn cpu_options_match_the_ids() {
        let mut s = state(2, 2);
        set(&mut s, "node,nodeid=0,memdev=ram").unwrap();
        set(&mut s, "node,nodeid=1").unwrap();
        set(&mut s, "cpu,node-id=0,socket-id=1,cluster-id=0,core-id=0,thread-id=0").unwrap();
        set(&mut s, "cpu,node-id=1,socket-id=0,cluster-id=0,core-id=0,thread-id=0").unwrap();
        s.complete(128 * MIB, false, by_socket, true).unwrap();
        assert_eq!(nodes_of(&s), [1, 0]);
        let e = set(&mut s, "cpu,node-id=0,socket-id=0").unwrap_err();
        assert_eq!(e.message(), "CPU is already assigned to node-id: 1");
        let e = set(&mut s, "cpu,node-id=0,die-id=0").unwrap_err();
        assert_eq!(e.message(), "die-id is not supported");
        let e = set(&mut s, "cpu,node-id=0,socket-id=5").unwrap_err();
        assert_eq!(e.message(), "no match found");
        let e = set(&mut s, "cpu,node-id=3").unwrap_err();
        assert!(e.message().starts_with("Invalid node-id=3, NUMA node must be defined"));
    }

    #[test]
    fn distances_complete_symmetrically() {
        let mut s = state(2, 1);
        set(&mut s, "node,memdev=ram").unwrap();
        set(&mut s, "node").unwrap();
        set(&mut s, "dist,src=0,dst=1,val=21").unwrap();
        s.complete(128 * MIB, false, by_socket, true).unwrap();
        assert_eq!(s.nodes[0].distance, [10, 21]);
        assert_eq!(s.nodes[1].distance, [21, 10]);
        let e = set(&mut s, "dist,src=0,dst=0,val=11").unwrap_err();
        assert_eq!(e.message(), "Local distance of node 0 should be 10.");
    }

    #[test]
    fn errors() {
        let mut s = state(2, 1);
        set(&mut s, "node,nodeid=1,memdev=ram").unwrap();
        let e = s.clone().complete(128 * MIB, false, by_socket, true).unwrap_err();
        assert_eq!(e.message(), "numa: Node ID missing: 0");
        let e = set(&mut s, "node,nodeid=1").unwrap_err();
        assert_eq!(e.message(), "Duplicate NUMA nodeid: 1");
        set(&mut s, "node,nodeid=0").unwrap();
        let e = s.clone().complete(256 * MIB, false, by_socket, true).unwrap_err();
        assert_eq!(
            e.message(),
            "total memory for NUMA nodes (0x8000000) should equal RAM size (0x10000000)"
        );
        let e = s.clone().complete(128 * MIB, true, by_socket, true).unwrap_err();
        assert_eq!(
            e.message(),
            "'-machine memory-backend' and '-numa memdev' properties are mutually exclusive"
        );
        let e = set(&mut s, "node,nodeid=2,memdev=nope").unwrap_err();
        assert_eq!(e.message(), "memdev=nope is ambiguous");
        let e = set(&mut s, "node,nodeid=3,cpus=2").unwrap_err();
        assert_eq!(e.message(), "CPU index (2) should be smaller than maxcpus (2)");
        let e =
            set(&mut s, "hmat-lb,initiator=0,target=0,hierarchy=memory,data-type=access-latency")
                .unwrap_err();
        assert!(e.message().starts_with("ACPI Heterogeneous Memory Attribute Table"));
        assert_eq!(format_numa(None, &[]), "0 nodes\n");
    }
}
