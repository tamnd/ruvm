// SPDX-License-Identifier: GPL-2.0-or-later

//! `TYPE_BUS` from hw/core/bus.c and the main system bus from hw/core/sysbus.c.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ruvm_qom::{
    BoolGetter, BoolSetter, LinkFlags, LinkSlot, Object, Registry, TYPE_OBJECT, TypeInfo,
};

/// `TYPE_BUS`.
pub const TYPE_BUS: &str = "bus";
/// `TYPE_SYSTEM_BUS`. The QOM type name really is `System`.
pub const TYPE_SYSTEM_BUS: &str = "System";
/// `TYPE_HOTPLUG_HANDLER`, the interface a bus's `hotplug-handler` link points at.
pub const TYPE_HOTPLUG_HANDLER: &str = "hotplug-handler";

/// `BusState`.
#[derive(Debug, Default)]
struct BusState {
    realized: AtomicBool,
    hotplug_handler: Arc<LinkSlot>,
}

fn state(obj: &Object) -> Arc<BusState> {
    obj.state::<BusState>().expect("a bus object")
}

pub(crate) fn register_types(registry: &Registry) {
    let bus = TypeInfo::new(TYPE_BUS)
        .parent(TYPE_OBJECT)
        .abstract_()
        .instance_state(BusState::default)
        // qbus_initfn(): the link is added per instance.
        .instance_init(|obj| {
            let slot = state(obj).hotplug_handler.clone();
            obj.property_add_link(
                "hotplug-handler",
                TYPE_HOTPLUG_HANDLER,
                slot,
                None,
                LinkFlags::STRONG,
            );
        })
        .class_init(|k| {
            let get: BoolGetter =
                Arc::new(|o: &Object| Ok(state(o).realized.load(Ordering::Acquire)));
            // bus_set_realized(). There are no devices on a bus yet, so there is nothing to
            // realize or unrealize along with it.
            let set: BoolSetter = Arc::new(|o: &Object, v: bool| {
                state(o).realized.store(v, Ordering::Release);
                Ok(())
            });
            k.property_add_bool("realized", Some(get), Some(set));
        });
    let sysbus = TypeInfo::new(TYPE_SYSTEM_BUS).parent(TYPE_BUS);
    registry.register_all([bus, sysbus]);
}
