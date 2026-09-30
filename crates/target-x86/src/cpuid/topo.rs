// SPDX-License-Identifier: GPL-2.0-or-later

//! CPU topology and APIC ID layout, from `include/hw/i386/topology.h`.

/// A topology level, as `CpuTopologyLevel`, in the order QEMU numbers them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TopoLevel {
    /// `CPU_TOPOLOGY_LEVEL_THREAD`.
    Thread,
    /// `CPU_TOPOLOGY_LEVEL_CORE`.
    Core,
    /// `CPU_TOPOLOGY_LEVEL_MODULE`.
    Module,
    /// `CPU_TOPOLOGY_LEVEL_DIE`.
    Die,
    /// `CPU_TOPOLOGY_LEVEL_SOCKET`.
    Socket,
}

/// Counts per level inside one package, as `X86CPUTopoInfo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct X86CpuTopoInfo {
    /// Dies per package.
    pub dies_per_pkg: u32,
    /// Modules per die.
    pub modules_per_die: u32,
    /// Cores per module.
    pub cores_per_module: u32,
    /// Threads per core.
    pub threads_per_core: u32,
}

impl Default for X86CpuTopoInfo {
    fn default() -> Self {
        Self { dies_per_pkg: 1, modules_per_die: 1, cores_per_module: 1, threads_per_core: 1 }
    }
}

/// Topology IDs of one CPU, as `X86CPUTopoIDs`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct X86CpuTopoIds {
    /// Package.
    pub pkg_id: u32,
    /// Die in the package.
    pub die_id: u32,
    /// Module in the die.
    pub module_id: u32,
    /// Core in the module.
    pub core_id: u32,
    /// Thread in the core.
    pub smt_id: u32,
}

/// `apicid_bitwidth_for_count()`: bits needed for IDs `0..count`.
pub fn apicid_bitwidth_for_count(count: u32) -> u32 {
    let c = count.max(1) - 1;
    if c == 0 { 0 } else { 32 - c.leading_zeros() }
}

fn low_mask(width: u32) -> u32 {
    if width >= 32 { u32::MAX } else { (1u32 << width) - 1 }
}

impl X86CpuTopoInfo {
    /// `apicid_smt_width()`.
    pub fn smt_width(&self) -> u32 {
        apicid_bitwidth_for_count(self.threads_per_core)
    }

    /// `apicid_core_width()`.
    pub fn core_width(&self) -> u32 {
        apicid_bitwidth_for_count(self.cores_per_module)
    }

    /// `apicid_module_width()`.
    pub fn module_width(&self) -> u32 {
        apicid_bitwidth_for_count(self.modules_per_die)
    }

    /// `apicid_die_width()`.
    pub fn die_width(&self) -> u32 {
        apicid_bitwidth_for_count(self.dies_per_pkg)
    }

    /// `apicid_core_offset()`.
    pub fn core_offset(&self) -> u32 {
        self.smt_width()
    }

    /// `apicid_module_offset()`.
    pub fn module_offset(&self) -> u32 {
        self.core_offset() + self.core_width()
    }

    /// `apicid_die_offset()`.
    pub fn die_offset(&self) -> u32 {
        self.module_offset() + self.module_width()
    }

    /// `apicid_pkg_offset()`.
    pub fn pkg_offset(&self) -> u32 {
        self.die_offset() + self.die_width()
    }

    /// `x86_threads_per_module()`.
    pub fn threads_per_module(&self) -> u32 {
        self.threads_per_core * self.cores_per_module
    }

    /// `x86_threads_per_die()`.
    pub fn threads_per_die(&self) -> u32 {
        self.threads_per_module() * self.modules_per_die
    }

    /// `x86_threads_per_pkg()`.
    pub fn threads_per_pkg(&self) -> u32 {
        self.threads_per_die() * self.dies_per_pkg
    }

    /// `apicid_offset_by_topo_level()`.
    pub fn offset_of(&self, level: TopoLevel) -> u32 {
        match level {
            TopoLevel::Thread => 0,
            TopoLevel::Core => self.core_offset(),
            TopoLevel::Module => self.module_offset(),
            TopoLevel::Die => self.die_offset(),
            TopoLevel::Socket => self.pkg_offset(),
        }
    }

    /// `num_threads_by_topo_level()`.
    pub fn threads_at(&self, level: TopoLevel) -> u32 {
        match level {
            TopoLevel::Thread => 1,
            TopoLevel::Core => self.threads_per_core,
            TopoLevel::Module => self.threads_per_module(),
            TopoLevel::Die => self.threads_per_die(),
            TopoLevel::Socket => self.threads_per_pkg(),
        }
    }

    /// `max_thread_ids_for_cache()`.
    pub fn max_thread_ids_for_cache(&self, share: TopoLevel) -> u32 {
        (1u32 << self.offset_of(share)) - 1
    }

    /// `max_core_ids_in_package()`.
    pub fn max_core_ids_in_package(&self) -> u32 {
        (1u32 << (self.pkg_offset() - self.core_offset())) - 1
    }

    /// Levels present in `env->avail_cpu_topo`: thread, core and socket
    /// always, module and die when there is more than one of them.
    pub fn available_levels(&self) -> Vec<TopoLevel> {
        let mut v = vec![TopoLevel::Thread, TopoLevel::Core];
        if self.modules_per_die > 1 {
            v.push(TopoLevel::Module);
        }
        if self.dies_per_pkg > 1 {
            v.push(TopoLevel::Die);
        }
        v.push(TopoLevel::Socket);
        v
    }

    /// `x86_has_extended_topo()`: module or die level present.
    pub fn has_extended_topo(&self) -> bool {
        self.modules_per_die > 1 || self.dies_per_pkg > 1
    }

    /// `x86_topo_ids_from_apicid()`.
    pub fn ids_from_apicid(&self, apicid: u32) -> X86CpuTopoIds {
        X86CpuTopoIds {
            smt_id: apicid & low_mask(self.smt_width()),
            core_id: (apicid >> self.core_offset()) & low_mask(self.core_width()),
            module_id: (apicid >> self.module_offset()) & low_mask(self.module_width()),
            die_id: (apicid >> self.die_offset()) & low_mask(self.die_width()),
            pkg_id: apicid.checked_shr(self.pkg_offset()).unwrap_or(0),
        }
    }

    /// `x86_topo_ids_from_idx()`.
    pub fn ids_from_index(&self, cpu_index: u32) -> X86CpuTopoIds {
        let t = self.threads_per_core;
        let c = self.cores_per_module;
        let m = self.modules_per_die;
        let d = self.dies_per_pkg;
        X86CpuTopoIds {
            pkg_id: cpu_index / (d * m * c * t),
            die_id: cpu_index / (m * c * t) % d,
            module_id: cpu_index / (c * t) % m,
            core_id: cpu_index / t % c,
            smt_id: cpu_index % t,
        }
    }

    /// `x86_apicid_from_topo_ids()`.
    pub fn apicid_from_ids(&self, ids: &X86CpuTopoIds) -> u32 {
        (ids.pkg_id << self.pkg_offset())
            | (ids.die_id << self.die_offset())
            | (ids.module_id << self.module_offset())
            | (ids.core_id << self.core_offset())
            | ids.smt_id
    }

    /// `x86_apicid_from_cpu_idx()`.
    pub fn apicid_from_index(&self, cpu_index: u32) -> u32 {
        self.apicid_from_ids(&self.ids_from_index(cpu_index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_for_2_dies_3_cores_2_threads() {
        let t = X86CpuTopoInfo {
            dies_per_pkg: 2,
            modules_per_die: 1,
            cores_per_module: 3,
            threads_per_core: 2,
        };
        assert_eq!(t.core_offset(), 1);
        assert_eq!(t.module_offset(), 3);
        assert_eq!(t.die_offset(), 3);
        assert_eq!(t.pkg_offset(), 4);
        assert_eq!(t.threads_per_pkg(), 12);
        // cpu 7: die 1, core 0, thread 1 -> 1 << 3 | 1.
        assert_eq!(t.apicid_from_index(7), 0b1001);
        assert_eq!(t.ids_from_apicid(0b1001).die_id, 1);
    }
}
