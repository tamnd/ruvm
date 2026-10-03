// SPDX-License-Identifier: GPL-2.0-or-later

//! The `tcg` accelerator, `accel/tcg/tcg-all.c` and the run control parts of
//! `system/cpus.c`.
//!
//! [`TcgOptions`] is the `tcg-accel` object: `-accel tcg,thread=single|multi,tb-size=N,
//! split-wx=on|off,one-insn-per-tb=on|off`, parsed with QEMU's errors.
//! [`TcgOptions::jit_config`] does what `tcg_init_machine()` does with them: it picks MTTCG
//! when the guest supports it and `thread` was not given, and sizes the code buffer.
//! [`TcgOptions::backend`] picks the host backend with [`ruvm_jit::host_backend`]: the native
//! code generator for the host when there is one, the interpreter otherwise or when
//! `RUVM_JIT_BACKEND=interp` asks for it.
//!
//! [`TcgVcpus`] owns the vCPU threads ([`ruvm_jit::accel::start_vcpus`]). They start
//! stopped, as `qemu_init_vcpu()` leaves them, and run once [`VcpuControl::resume_all`] is
//! called.
//!
//! Deliberate differences from QEMU:
//!
//! - `tb-size=0`, the default, means the runtime's default code buffer of 32 MiB rather than
//!   QEMU's 1 GiB. Any other value is taken in MiB as in QEMU, with QEMU's 1 MiB minimum.
//! - `split-wx=on` fails with "jit split-wx not supported", as it does on QEMU hosts without
//!   split mappings: the code regions here are never mapped twice.
//! - There is no big QEMU lock. Pausing sets each vCPU's `stop` flag, kicks it and polls
//!   until every vCPU reports `stopped`, which is what `pause_all_vcpus()` waits for on
//!   `qemu_pause_cond`.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use ruvm_jit::accel::{VcpuThreads, start_vcpus};
use ruvm_jit::{Backend, Cpu, CpuShared, Jit, JitConfig, Vcpu, plugin};
use ruvm_jit_interp::HelperRegistry;
use ruvm_qapi::visit::{StringInputVisitor, VisitorExt, qapi_bool_parse};

use crate::VcpuControl;

/// The QOM type of the accelerator object, `ACCEL_CLASS_NAME("tcg")`.
pub const TYPE_TCG_ACCEL: &str = "tcg-accel";

/// `MIN_CODE_GEN_BUFFER_SIZE`.
const MIN_CODE_GEN_BUFFER_SIZE: usize = 1 << 20;

/// `thread=`: one host thread per vCPU, or one for all of them.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ThreadMode {
    /// `thread=single`: round robin on one thread.
    Single,
    /// `thread=multi`: MTTCG.
    Multi,
}

/// The properties of the `tcg-accel` object, `TCGState`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TcgOptions {
    /// `thread`, or `None` for QEMU's `ON_OFF_AUTO_AUTO`.
    pub thread: Option<ThreadMode>,
    /// `tb-size` in MiB, 0 for the default.
    pub tb_size: u32,
    /// `split-wx`.
    pub split_wx: bool,
    /// `one-insn-per-tb`.
    pub one_insn_per_tb: bool,
}

impl TcgOptions {
    /// Sets the property `name` from the command line string `value`, as
    /// `object_parse_property_set()` does. The errors are QEMU's.
    pub fn set(&mut self, name: &str, value: &str) -> Result<(), String> {
        match name {
            "thread" => {
                // tcg_set_thread(); icount does not exist here, so "multi" always works.
                self.thread = Some(match value {
                    "multi" => ThreadMode::Multi,
                    "single" => ThreadMode::Single,
                    _ => return Err(format!("Invalid 'thread' setting {value}")),
                });
            }
            "tb-size" => {
                let mut v = 0u32;
                StringInputVisitor::new(value)
                    .type_uint32(Some(name), &mut v)
                    .map_err(|e| e.to_string())?;
                self.tb_size = v;
            }
            "split-wx" => {
                self.split_wx = qapi_bool_parse(name, value).map_err(|e| e.to_string())?
            }
            "one-insn-per-tb" => {
                self.one_insn_per_tb = qapi_bool_parse(name, value).map_err(|e| e.to_string())?;
            }
            _ => return Err(format!("Property '{TYPE_TCG_ACCEL}.{name}' not found")),
        }
        Ok(())
    }

    /// Parses every `-accel tcg` property in order, stopping at the first error.
    pub fn from_props<'a>(
        props: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<TcgOptions, String> {
        let mut o = TcgOptions::default();
        for (k, v) in props {
            o.set(k, v)?;
        }
        Ok(o)
    }

    /// The `thread` property as `tcg_get_thread()` reports it once the machine is up.
    pub fn thread_name(mttcg: bool) -> &'static str {
        if mttcg { "multi" } else { "single" }
    }

    /// `tcg_init_machine()`'s choice: whether MTTCG is on for a guest whose CPU class has
    /// `mttcg_supported` set or not, and the warning QEMU prints for `thread=multi` on a
    /// guest that does not support it.
    pub fn mttcg(&self, mttcg_supported: bool) -> (bool, Option<&'static str>) {
        match self.thread {
            None => (mttcg_supported, None),
            Some(ThreadMode::Single) => (false, None),
            Some(ThreadMode::Multi) => (
                true,
                (!mttcg_supported)
                    .then_some("Guest not yet converted to MTTCG - you may get unexpected results"),
            ),
        }
    }

    /// The code buffer size `tcg_init()` uses, `size_code_gen_buffer()`.
    pub fn code_gen_buffer_size(&self, default: usize) -> usize {
        if self.tb_size == 0 {
            default
        } else {
            (self.tb_size as usize).saturating_mul(1 << 20).max(MIN_CODE_GEN_BUFFER_SIZE)
        }
    }

    /// The runtime configuration for a guest whose front end asks for `base`, with these
    /// options applied, and the warnings to print. Fails as QEMU does for `split-wx=on`.
    pub fn jit_config(
        &self,
        base: JitConfig,
        mttcg_supported: bool,
    ) -> Result<(JitConfig, Vec<String>), String> {
        if self.split_wx {
            return Err("jit split-wx not supported".to_string());
        }
        let (mttcg, warning) = self.mttcg(mttcg_supported);
        let config = JitConfig {
            mttcg,
            one_insn_per_tb: self.one_insn_per_tb || base.one_insn_per_tb,
            code_gen_buffer_size: self.code_gen_buffer_size(base.code_gen_buffer_size),
            ..base
        };
        Ok((config, warning.map(str::to_string).into_iter().collect()))
    }

    /// The host backend for `helpers`, sized for `config`: native code when the host has a
    /// code generator, the interpreter otherwise (see [`ruvm_jit::native`]).
    pub fn backend(config: &JitConfig, helpers: HelperRegistry) -> Arc<dyn Backend> {
        ruvm_jit::host_backend(helpers, config.code_gen_buffer_size)
    }
}

/// The vCPU threads of a machine on TCG.
#[derive(Debug)]
pub struct TcgVcpus {
    jit: Arc<Jit>,
    cpus: Vec<Arc<CpuShared>>,
    threads: Mutex<Option<VcpuThreads>>,
}

/// How often [`TcgVcpus::pause_all`] looks whether the vCPUs have stopped.
const PAUSE_POLL: Duration = Duration::from_millis(1);

impl TcgVcpus {
    /// `qemu_init_vcpu()` for each of `vcpus` and `tcg_start_vcpu_thread()`: starts the
    /// threads with every vCPU stopped.
    pub fn start(jit: &Arc<Jit>, vcpus: Vec<Vcpu>) -> TcgVcpus {
        for v in &vcpus {
            v.shared().stopped.store(true, Ordering::Release);
        }
        let threads = start_vcpus(jit, vcpus);
        let cpus = threads.cpus().to_vec();
        TcgVcpus { jit: Arc::clone(jit), cpus, threads: Mutex::new(Some(threads)) }
    }

    /// The runtime the vCPUs run on.
    pub fn jit(&self) -> &Arc<Jit> {
        &self.jit
    }

    /// The shared halves of the vCPUs, in `cpu_index` order.
    pub fn cpus(&self) -> &[Arc<CpuShared>] {
        &self.cpus
    }

    /// Whether the calling thread runs one of these vCPUs, `current_cpu != NULL`.
    pub fn on_vcpu_thread(&self) -> bool {
        self.cpus.iter().any(|c| c.is_self())
    }

    /// `run_on_cpu()` on every vCPU in turn: runs `f` on each vCPU's thread and waits. The
    /// vCPUs should be paused, as they are for a system reset. From a vCPU thread, the
    /// calls are queued instead (`async_run_on_cpu()`), since waiting there would hang.
    pub fn run_on_each(&self, f: impl Fn(&mut Cpu<'_>) + Clone + Send + 'static) {
        let on_vcpu = self.on_vcpu_thread();
        for c in &self.cpus {
            let g = f.clone();
            if on_vcpu {
                c.async_run_on_cpu(move |cpu| g(cpu));
            } else {
                c.run_on_cpu(move |cpu| g(cpu));
            }
        }
    }

    /// Ends the threads, `cpu_remove_sync()` on each vCPU, runs the plugins' `vcpu_exit`
    /// callbacks (`qemu_plugin_vcpu_exit_hook()`, from `cpu_common_unrealizefn()`) and hands
    /// the vCPUs back. Must not be called from a vCPU thread. Calling it again returns
    /// nothing.
    pub fn quit(&self) -> Vec<Vcpu> {
        let t = self.threads.lock().unwrap_or_else(PoisonError::into_inner).take();
        let mut vcpus = match t {
            Some(t) => t.stop_and_join(),
            None => Vec::new(),
        };
        for v in &mut vcpus {
            plugin::vcpu_exit(&mut v.cpu());
        }
        vcpus
    }
}

impl VcpuControl for TcgVcpus {
    fn accel_name(&self) -> &'static str {
        "tcg"
    }

    fn vcpu_count(&self) -> usize {
        self.cpus.len()
    }

    /// `resume_all_vcpus()` and `cpu_resume()`.
    fn resume_all(&self) {
        for c in &self.cpus {
            c.stop.store(false, Ordering::Release);
            c.stopped.store(false, Ordering::Release);
            c.kick();
        }
    }

    /// `pause_all_vcpus()`.
    fn pause_all(&self) {
        for c in &self.cpus {
            c.stop.store(true, Ordering::Release);
            c.kick();
        }
        if self.on_vcpu_thread() {
            return;
        }
        let running = self.threads.lock().unwrap_or_else(PoisonError::into_inner).is_some();
        while running && !self.all_paused() {
            std::thread::sleep(PAUSE_POLL);
        }
    }

    fn all_paused(&self) -> bool {
        self.cpus.iter().all(|c| c.stopped.load(Ordering::Acquire))
    }

    fn kick(&self, index: usize) {
        if let Some(c) = self.cpus.get(index) {
            c.kick();
        }
    }
}

impl Drop for TcgVcpus {
    fn drop(&mut self) {
        if !self.on_vcpu_thread() {
            drop(self.quit());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn properties_parse_like_qemu() {
        let mut o = TcgOptions::default();
        o.set("thread", "multi").unwrap();
        assert_eq!(o.thread, Some(ThreadMode::Multi));
        o.set("thread", "single").unwrap();
        assert_eq!(o.thread, Some(ThreadMode::Single));
        assert_eq!(o.set("thread", "foo").unwrap_err(), "Invalid 'thread' setting foo");
        o.set("tb-size", "64").unwrap();
        assert_eq!(o.tb_size, 64);
        assert_eq!(o.set("tb-size", "x").unwrap_err(), "Parameter 'tb-size' expects uint64");
        assert_eq!(
            o.set("tb-size", "5000000000").unwrap_err(),
            "Parameter 'tb-size' expects uint32_t"
        );
        o.set("one-insn-per-tb", "yes").unwrap();
        assert!(o.one_insn_per_tb);
        o.set("one-insn-per-tb", "off").unwrap();
        assert!(!o.one_insn_per_tb);
        assert_eq!(
            o.set("split-wx", "maybe").unwrap_err(),
            "Parameter 'split-wx' expects 'on' or 'off'"
        );
        assert_eq!(o.set("foo", "1").unwrap_err(), "Property 'tcg-accel.foo' not found");
    }

    #[test]
    fn mttcg_follows_tcg_init_machine() {
        let auto = TcgOptions::default();
        assert_eq!(auto.mttcg(true), (true, None));
        assert_eq!(auto.mttcg(false), (false, None));
        let single = TcgOptions { thread: Some(ThreadMode::Single), ..TcgOptions::default() };
        assert_eq!(single.mttcg(true), (false, None));
        let multi = TcgOptions { thread: Some(ThreadMode::Multi), ..TcgOptions::default() };
        assert_eq!(multi.mttcg(true), (true, None));
        let (on, warn) = multi.mttcg(false);
        assert!(on);
        assert_eq!(warn, Some("Guest not yet converted to MTTCG - you may get unexpected results"));
        assert_eq!(TcgOptions::thread_name(true), "multi");
        assert_eq!(TcgOptions::thread_name(false), "single");
    }

    #[test]
    fn jit_config_applies_the_options() {
        let base = JitConfig { page_bits: 12, nb_mmu_modes: 8, ..JitConfig::default() };
        let o = TcgOptions { tb_size: 16, one_insn_per_tb: true, ..TcgOptions::default() };
        let (c, warnings) = o.jit_config(base.clone(), true).unwrap();
        assert!(warnings.is_empty());
        assert!(c.mttcg);
        assert!(c.one_insn_per_tb);
        assert_eq!(c.code_gen_buffer_size, 16 << 20);
        assert_eq!(c.page_bits, 12);
        assert_eq!(c.nb_mmu_modes, 8);
        // tb-size=0 keeps the default.
        let (c, _) = TcgOptions::default().jit_config(base.clone(), false).unwrap();
        assert!(!c.mttcg);
        assert_eq!(c.code_gen_buffer_size, base.code_gen_buffer_size);
        let wx = TcgOptions { split_wx: true, ..TcgOptions::default() };
        assert_eq!(wx.jit_config(base, true).unwrap_err(), "jit split-wx not supported");
    }
}
