// SPDX-License-Identifier: GPL-2.0-or-later

//! The `globalstate` section, migration/global_state.c: the run state of the source when it
//! stopped for the switchover, which tells the destination whether to start the guest.

use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};

use ruvm_base::error_report;
use ruvm_qapi::types::RunState;
use ruvm_vmstate::{EINVAL, VmStateDescription, VmStateField};

use crate::savevm::SaveVm;

/// `GlobalState`, the part that goes on the wire.
#[derive(Debug, Clone)]
struct Wire {
    size: u32,
    runstate: [u8; 32],
    has_vm_was_suspended: u8,
    vm_was_suspended: u8,
    unused: [u8; 66],
}

impl Default for Wire {
    fn default() -> Self {
        Wire {
            size: 0,
            runstate: [0; 32],
            has_vm_was_suspended: 0,
            vm_was_suspended: 0,
            unused: [0; 66],
        }
    }
}

fn runstate_str(w: &Wire) -> String {
    let end = w.runstate.iter().position(|&b| b == 0).unwrap_or(w.runstate.len());
    String::from_utf8_lossy(&w.runstate[..end]).into_owned()
}

static VMSTATE_GLOBALSTATE: LazyLock<VmStateDescription<Wire>> = LazyLock::new(|| {
    VmStateDescription::new("globalstate")
        .version_id(1)
        .minimum_version_id(1)
        // global_state_pre_save()
        .pre_save(|s: &mut Wire| {
            s.size = runstate_str(s).len() as u32 + 1;
            0
        })
        .post_load(|s: &mut Wire, _| {
            s.runstate[31] = 0;
            let name = runstate_str(s);
            if RunState::from_name(&name).is_none() {
                error_report(&format!("invalid parameter value: {name}"));
                return -EINVAL;
            }
            0
        })
        .fields([
            VmStateField::scalar("size", |s: &mut Wire| &mut s.size),
            VmStateField::buffer("runstate", |s: &mut Wire| &mut s.runstate),
            VmStateField::scalar("has_vm_was_suspended", |s: &mut Wire| {
                &mut s.has_vm_was_suspended
            }),
            VmStateField::scalar("vm_was_suspended", |s: &mut Wire| &mut s.vm_was_suspended),
            VmStateField::buffer("unused", |s: &mut Wire| &mut s.unused),
        ])
});

#[derive(Debug, Default)]
struct Inner {
    wire: Wire,
    received: Option<RunState>,
}

/// The global state of a machine, registered as its `globalstate` section.
#[derive(Debug, Default)]
pub struct GlobalState {
    inner: Mutex<Inner>,
}

impl GlobalState {
    /// `register_global_state()`.
    pub fn register(savevm: &mut SaveVm) -> Arc<GlobalState> {
        let gs = Arc::new(GlobalState::default());
        let get = Arc::clone(&gs);
        let put = Arc::clone(&gs);
        savevm.register_vmsd(
            "",
            Some(0),
            &VMSTATE_GLOBALSTATE,
            move || Ok(get.lock().wire.clone()),
            move |w: Wire| {
                let mut g = put.lock();
                g.received = RunState::from_name(&runstate_str(&w));
                g.wire = w;
                Ok(())
            },
        );
        gs
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `global_state_store()`: records `state` to send, and whether the guest was suspended.
    pub fn store(&self, state: RunState, vm_was_suspended: bool) {
        let mut g = self.lock();
        let name = state.as_str().as_bytes();
        g.wire.runstate = [0; 32];
        g.wire.runstate[..name.len()].copy_from_slice(name);
        g.wire.has_vm_was_suspended = 1;
        g.wire.vm_was_suspended = u8::from(vm_was_suspended);
        g.wire.unused = [0; 66];
    }

    /// `global_state_received()` and `global_state_get_runstate()`: the run state the source
    /// sent, if it sent one.
    pub fn received(&self) -> Option<RunState> {
        self.lock().received
    }

    /// What `vm_set_suspended()` gets in `global_state_post_load()`.
    pub fn vm_was_suspended(&self) -> bool {
        let g = self.lock();
        g.wire.vm_was_suspended != 0 || g.received == Some(RunState::Suspended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::savevm::MachineConfig;
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_save_state};

    #[test]
    fn globalstate_round_trips() {
        let cfg = MachineConfig {
            name: "pc-q35-11.1".into(),
            page_bits: 12,
            legacy_page_bits: 12,
            uuid: None,
        };
        let mut a = SaveVm::new(cfg);
        let ga = GlobalState::register(&mut a);
        ga.store(RunState::Paused, true);
        assert_eq!(ga.lock().wire.runstate[..7], *b"paused\0");
        assert!(ga.received().is_none());
        let mut w = Wire::default();
        w.runstate[..7].copy_from_slice(b"running");
        w.has_vm_was_suspended = 1;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_GLOBALSTATE, &mut w).unwrap();
        let bytes = f.into_inner();
        assert_eq!(bytes.len(), 4 + 32 + 2 + 66);
        assert_eq!(&bytes[..4], &[0, 0, 0, 8]);
        let mut r = StreamReader::new(&bytes[..]);
        let mut back = Wire::default();
        ruvm_vmstate::vmstate_load_state(&mut r, &VMSTATE_GLOBALSTATE, &mut back, 1).unwrap();
        assert_eq!(runstate_str(&back), "running");
        back.runstate[..3].copy_from_slice(b"bad");
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_GLOBALSTATE, &mut back).unwrap();
        let bytes = f.into_inner();
        let mut r = StreamReader::new(&bytes[..]);
        assert!(ruvm_vmstate::vmstate_load_state(&mut r, &VMSTATE_GLOBALSTATE, &mut w, 1).is_err());
    }
}
