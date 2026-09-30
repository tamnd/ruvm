// SPDX-License-Identifier: GPL-2.0-or-later

//! Events the block layer raises for the monitor.
//!
//! QEMU calls `qapi_event_send_*()` straight from the block layer. Here the QAPI events are
//! built and sent by the monitor crates, which sit above this one, so the block layer hands
//! [`BlockEvent`]s to a hook the system installs with [`set_event_hook`]. Without a hook the
//! events are dropped, as they are in QEMU tools that have no monitor.

use std::sync::{Arc, RwLock};

/// An event for the monitor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockEvent {
    /// `BLOCK_WRITE_THRESHOLD`: a write went past the threshold, which is now disabled.
    WriteThreshold { node_name: String, amount_exceeded: u64, write_threshold: u64 },
    /// `DEVICE_TRAY_MOVED`: the tray of a removable medium opened or closed.
    DeviceTrayMoved {
        /// The name of the block backend, empty when it has none.
        device: String,
        /// The id of the guest device.
        id: String,
        tray_open: bool,
    },
}

/// The function [`set_event_hook`] installs.
pub type BlockEventHook = Arc<dyn Fn(&BlockEvent) + Send + Sync>;

static HOOK: RwLock<Option<BlockEventHook>> = RwLock::new(None);

/// Installs the function that receives block layer events, replacing any earlier one.
pub fn set_event_hook(hook: Option<BlockEventHook>) {
    *HOOK.write().unwrap() = hook;
}

/// Sends `ev` to the hook, if there is one.
pub(crate) fn emit(ev: BlockEvent) {
    let hook = HOOK.read().unwrap().clone();
    if let Some(h) = hook {
        h(&ev);
    }
}
