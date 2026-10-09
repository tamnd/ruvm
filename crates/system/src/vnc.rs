// SPDX-License-Identifier: GPL-2.0-or-later

//! The VNC glue of QEMU's system/vl.c and ui/ui-qmp-cmds.c: `-vnc`, opening the displays once
//! the machine is up, the VNC events on the QMP monitor and the VNC commands.
//!
//! The server itself is `ruvm_ui::vnc`.

use std::sync::Arc;

use ruvm_monitor::{Commands, MonitorQmp, Qmp};
use ruvm_qapi::commands::{
    register_change_vnc_password, register_expire_password, register_query_vnc,
    register_query_vnc_servers, register_set_password,
};
use ruvm_qapi::events::{event_vnc_connected, event_vnc_disconnected, event_vnc_initialized};
use ruvm_ui::input::InputState;
use ruvm_ui::vnc::{self, Hooks, VncEvent};

use crate::vl::Vm;

/// `vnc_parse()` for `-vnc` and `-display vnc=`. The error is the exit status.
pub(crate) fn parse(arg: &str) -> Result<(), u8> {
    vnc::parse(arg)
}

/// Sends the VNC events to the QMP monitor.
struct QmpHooks(Arc<Qmp>);

impl Hooks for QmpHooks {
    fn event(&self, event: VncEvent) {
        let policy = self.0.policy();
        let dict = match event {
            VncEvent::Connected(arg) => event_vnc_connected(&policy, arg),
            VncEvent::Initialized(arg) => event_vnc_initialized(&policy, arg),
            VncEvent::Disconnected(arg) => event_vnc_disconnected(&policy, arg),
        };
        if let Some(dict) = dict {
            self.0.emit_event(dict);
        }
    }
}

/// `qemu_opts_foreach(qemu_find_opts("vnc"), vnc_init_func, ...)` in `qemu_init_displays()`.
///
/// Before that it gives the input layer the run state check of `qmp_input_send_event()`.
pub(crate) fn init(vm: &Vm) -> Result<(), u8> {
    let runstate = Arc::downgrade(&vm.runstate);
    InputState::global().set_runstate_check(move || {
        runstate.upgrade().is_some_and(|r| crate::runstate::is_live(r.get()))
    });
    vnc::init(vm.name.as_deref(), Arc::new(QmpHooks(Arc::clone(&vm.qmp))))
}

/// The VNC commands of ui/ui-qmp-cmds.c and ui/vnc.c.
pub(crate) fn register(cmds: &mut Commands) {
    register_set_password(cmds, |_: &MonitorQmp, opts| vnc::qmp_set_password(opts));
    register_expire_password(cmds, |_: &MonitorQmp, opts| vnc::qmp_expire_password(opts));
    register_query_vnc(cmds, |_: &MonitorQmp| vnc::query_vnc());
    register_query_vnc_servers(cmds, |_: &MonitorQmp| vnc::query_vnc_servers());
    register_change_vnc_password(cmds, |_: &MonitorQmp, arg| vnc::qmp_change_vnc_password(arg));
}
