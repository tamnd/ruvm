// SPDX-License-Identifier: GPL-2.0-or-later

//! The runtime context: what QEMU keeps in `tb_ctx`, `tcg_ctx`, the region allocator, the page
//! descriptors of `tb-maint.c` and the CPU list.
//!
//! Differences from QEMU:
//!
//! - The code buffer is a byte budget, [`JitConfig::code_gen_buffer_size`], instead of mapped
//!   memory split into regions. Each block is charged a fixed header cost plus the size its
//!   backend reports. When the budget runs out `tb_gen_code()` flushes, as QEMU does when
//!   `tcg_region_alloc()` fails.
//! - The hash table is a map keyed by the same fields QEMU hashes, behind a reader writer lock,
//!   instead of a QHT.
//! - The per page block lists live in one map behind one mutex instead of a radix tree of page
//!   descriptors with a lock each. The "page holds code" state, which QEMU keeps as the clear
//!   `DIRTY_MEMORY_CODE` bit, is a separate set so the TLB can test it without the page lock.
//! - RAM blocks get a `ram_addr` the first time the runtime sees them; see the crate docs.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock, Weak};
use std::time::Instant;

use ruvm_mem::{AddressSpace, RamBlock};

use crate::backend::{Backend, InterpBackend};
use crate::cpu::{CpuCore, CpuOps, CpuShared, Vcpu};
use crate::cputlb::CpuTlb;
use crate::tb::{Tb, TbKey, lock};
use crate::{ENV_TARGET_OFFSET, TARGET_PAGE_BITS_MIN, cf};

/// How the runtime is set up, the TCG accelerator properties plus what the target fixes at
/// build time in QEMU.
#[derive(Clone, Debug)]
pub struct JitConfig {
    /// `TARGET_PAGE_BITS`.
    pub page_bits: u32,
    /// `NB_MMU_MODES`, at most 16.
    pub nb_mmu_modes: usize,
    /// `tb-size`: the code buffer budget in bytes.
    pub code_gen_buffer_size: usize,
    /// `thread=multi`: one host thread per vCPU, and `CF_PARALLEL` blocks.
    pub mttcg: bool,
    /// `one-insn-per-tb`.
    pub one_insn_per_tb: bool,
    /// The `nochain` debug option: never chain with `goto_tb`.
    pub nochain: bool,
    /// Run the IR optimizer.
    pub optimize: bool,
    /// `TARGET_LONG_BITS`.
    pub target_long_bits: u32,
}

impl Default for JitConfig {
    fn default() -> JitConfig {
        JitConfig {
            page_bits: 12,
            nb_mmu_modes: 2,
            code_gen_buffer_size: 32 << 20,
            mttcg: false,
            one_insn_per_tb: false,
            nochain: false,
            optimize: true,
            target_long_bits: 64,
        }
    }
}

/// What every block costs in the code buffer besides its code, standing for the
/// `TranslationBlock` header QEMU puts in front of the code.
pub(crate) const TB_HEADER_COST: usize = 128;

#[derive(Default)]
pub(crate) struct Region {
    pub(crate) used: usize,
    pub(crate) full: bool,
    pub(crate) tbs: HashMap<u64, Arc<Tb>>,
}

/// One RAM block known to the runtime.
#[derive(Default)]
pub(crate) struct RamRegistry {
    pub(crate) by_base: BTreeMap<u64, Arc<RamBlock>>,
    pub(crate) next_base: u64,
}

/// The runtime shared by all vCPUs.
pub struct Jit {
    /// The configuration.
    pub config: JitConfig,
    pub(crate) backend: Arc<dyn Backend>,
    pub(crate) htable: RwLock<HashMap<TbKey, Arc<Tb>>>,
    pub(crate) region: Mutex<Region>,
    pub(crate) next_tb_id: AtomicU64,
    pub(crate) tb_flush_count: AtomicU32,
    pub(crate) tb_phys_invalidate_count: AtomicU64,
    pub(crate) pages: Mutex<HashMap<u64, Vec<Arc<Tb>>>>,
    pub(crate) code_pages: RwLock<HashSet<u64>>,
    pub(crate) ram: RwLock<RamRegistry>,
    pub(crate) cpus: RwLock<Vec<Arc<CpuShared>>>,
    pub(crate) list_lock: Mutex<()>,
    pub(crate) pending_cpus: AtomicU32,
    pub(crate) exclusive_cond: Condvar,
    pub(crate) exclusive_resume: Condvar,
    pub(crate) work_lock: Mutex<()>,
    pub(crate) work_cond: Condvar,
    atomic: AtomicBool,
    pub(crate) rr_current_cpu: Mutex<Option<Weak<CpuShared>>>,
    pub(crate) rr_halt: Arc<(Mutex<()>, Condvar)>,
    pub(crate) start: Instant,
    pub(crate) self_ref: Weak<Jit>,
    pub(crate) plugin: crate::plugin::JitPlugin,
}

impl fmt::Debug for Jit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Jit")
            .field("config", &self.config)
            .field("tb_flush_count", &self.tb_flush_count())
            .finish_non_exhaustive()
    }
}

impl Jit {
    /// A runtime using `backend`.
    ///
    /// # Panics
    ///
    /// If the page size is below [`TARGET_PAGE_BITS_MIN`] or there are more than 16 MMU modes.
    pub fn new(config: JitConfig, backend: Arc<dyn Backend>) -> Arc<Jit> {
        assert!(config.page_bits >= TARGET_PAGE_BITS_MIN, "page_bits below TARGET_PAGE_BITS_MIN");
        assert!(config.page_bits < 32);
        assert!((1..=16).contains(&config.nb_mmu_modes), "NB_MMU_MODES must be 1 to 16");
        assert!(matches!(config.target_long_bits, 32 | 64));
        Arc::new_cyclic(|w| Jit {
            config,
            backend,
            htable: RwLock::new(HashMap::new()),
            region: Mutex::new(Region::default()),
            next_tb_id: AtomicU64::new(4),
            tb_flush_count: AtomicU32::new(0),
            tb_phys_invalidate_count: AtomicU64::new(0),
            pages: Mutex::new(HashMap::new()),
            code_pages: RwLock::new(HashSet::new()),
            ram: RwLock::new(RamRegistry::default()),
            cpus: RwLock::new(Vec::new()),
            list_lock: Mutex::new(()),
            pending_cpus: AtomicU32::new(0),
            exclusive_cond: Condvar::new(),
            exclusive_resume: Condvar::new(),
            work_lock: Mutex::new(()),
            work_cond: Condvar::new(),
            atomic: AtomicBool::new(false),
            rr_current_cpu: Mutex::new(None),
            rr_halt: Arc::new((Mutex::new(()), Condvar::new())),
            start: Instant::now(),
            self_ref: w.clone(),
            plugin: crate::plugin::JitPlugin::default(),
        })
    }

    /// A runtime using the interpreter backend.
    pub fn with_interp(config: JitConfig) -> Arc<Jit> {
        Jit::new(config, Arc::new(InterpBackend::new()))
    }

    /// The backend.
    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    /// Create a vCPU, `cpu_exec_realizefn()` plus `tcg_exec_realizefn()`. The `env` buffer is
    /// `env_size` bytes of target state after the runtime's own [`ENV_TARGET_OFFSET`] bytes.
    pub fn create_vcpu(
        &self,
        ops: Arc<dyn CpuOps>,
        as_: Arc<AddressSpace>,
        env_size: usize,
    ) -> Vcpu {
        let jit = self.self_ref.upgrade().expect("Jit is alive");
        let mut cpus = self.cpus.write().unwrap_or_else(|e| e.into_inner());
        let cpu_index = cpus.len();
        let halt = if self.config.mttcg {
            Arc::new((Mutex::new(()), Condvar::new()))
        } else {
            self.rr_halt.clone()
        };
        let tlb = CpuTlb::new(self.config.nb_mmu_modes, self.now_ns());
        let shared = Arc::new(CpuShared::new(cpu_index, self.self_ref.clone(), halt, tlb));
        cpus.push(shared.clone());
        drop(cpus);
        self.plugin_vcpu_created(&shared);
        let tcg_cflags = if self.config.mttcg { cf::PARALLEL } else { 0 };
        let mut env = vec![0u8; ENV_TARGET_OFFSET + env_size];
        env[crate::ENV_CAN_DO_IO_OFFSET as usize] = 1;
        Vcpu { env, core: CpuCore::new(jit, shared, ops, as_, tcg_cflags) }
    }

    /// The vCPUs, `CPU_FOREACH`.
    pub fn cpu_list(&self) -> Vec<Arc<CpuShared>> {
        self.cpus.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// `TARGET_PAGE_SIZE`.
    pub fn page_size(&self) -> u64 {
        1 << self.config.page_bits
    }

    /// `TARGET_PAGE_MASK`.
    pub fn page_mask(&self) -> u64 {
        !(self.page_size() - 1)
    }

    /// How many times the code buffer was flushed.
    pub fn tb_flush_count(&self) -> u32 {
        self.tb_flush_count.load(Ordering::Acquire)
    }

    /// How many blocks were invalidated.
    pub fn tb_phys_invalidate_count(&self) -> u64 {
        self.tb_phys_invalidate_count.load(Ordering::Acquire)
    }

    /// Number of blocks in the hash table.
    pub fn tb_count(&self) -> usize {
        self.htable.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Bytes of the code buffer in use.
    pub fn code_gen_used(&self) -> usize {
        lock(&self.region).used
    }

    pub(crate) fn now_ns(&self) -> i64 {
        i64::try_from(self.start.elapsed().as_nanos()).unwrap_or(i64::MAX)
    }

    pub(crate) fn atomic_lock(&self) {
        while self
            .atomic
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::thread::yield_now();
        }
    }

    pub(crate) fn atomic_unlock(&self) {
        self.atomic.store(false, Ordering::Release);
    }

    /// The `ram_addr` of the first byte of `block`, giving it one if it has none yet.
    pub fn ram_addr_base(&self, block: &Arc<RamBlock>) -> u64 {
        {
            let r = self.ram.read().unwrap_or_else(|e| e.into_inner());
            if let Some((&base, _)) = r.by_base.iter().find(|(_, b)| Arc::ptr_eq(b, block)) {
                return base;
            }
        }
        let mut r = self.ram.write().unwrap_or_else(|e| e.into_inner());
        if let Some((&base, _)) = r.by_base.iter().find(|(_, b)| Arc::ptr_eq(b, block)) {
            return base;
        }
        let base = r.next_base;
        let size = (block.len() + self.page_size() - 1) & self.page_mask();
        // Leave a page between blocks so that no block ends where the next begins.
        r.next_base = base + size + self.page_size();
        r.by_base.insert(base, block.clone());
        base
    }

    /// The RAM block holding `ram_addr` and the offset in it.
    pub fn ram_block_from_addr(&self, ram_addr: u64) -> Option<(Arc<RamBlock>, u64)> {
        let r = self.ram.read().unwrap_or_else(|e| e.into_inner());
        let (&base, block) = r.by_base.range(..=ram_addr).next_back()?;
        let off = ram_addr - base;
        if off < block.len() { Some((block.clone(), off)) } else { None }
    }

    /// `tb_invalidate_phys_range()` for a write to `len` bytes at `offset` in `block` made
    /// outside the softmmu, such as DMA.
    pub fn tb_invalidate_phys_block(&self, block: &Arc<RamBlock>, offset: u64, len: u64) {
        if len == 0 {
            return;
        }
        let base = self.ram_addr_base(block);
        self.tb_invalidate_phys_range(base + offset, base + offset + len - 1);
    }

    /// The vCPU running in round robin mode, `rr_current_cpu`.
    pub(crate) fn rr_current(&self) -> Option<Arc<CpuShared>> {
        lock(&self.rr_current_cpu).as_ref().and_then(Weak::upgrade)
    }

    /// `rr_kick_next_cpu()`: make the running vCPU leave the execution loop.
    pub(crate) fn rr_kick_next_cpu(&self) {
        loop {
            let cpu = self.rr_current();
            if let Some(c) = &cpu {
                c.cpu_exit();
            }
            let now = self.rr_current();
            let same = match (&cpu, &now) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            };
            if same {
                break;
            }
        }
    }
}
