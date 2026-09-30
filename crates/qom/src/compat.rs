// SPDX-License-Identifier: GPL-2.0-or-later

//! Global and compat properties, qom/compat-properties.c and `GlobalProperty` from qdev.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_base::{Result, ResultExt};

use crate::lock;
use crate::object::Object;

/// `GlobalProperty`: `driver.property=value`, applied to every new object that is a `driver`.
pub struct GlobalProperty {
    pub driver: String,
    pub property: String,
    pub value: String,
    /// Skip objects that do not have the property instead of failing.
    pub optional: bool,
    pub(crate) used: AtomicBool,
}

impl fmt::Debug for GlobalProperty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}={}", self.driver, self.property, self.value)
    }
}

impl GlobalProperty {
    pub fn new(driver: &str, property: &str, value: &str) -> Arc<Self> {
        Arc::new(GlobalProperty {
            driver: driver.into(),
            property: property.into(),
            value: value.into(),
            optional: false,
            used: AtomicBool::new(false),
        })
    }

    pub fn new_optional(driver: &str, property: &str, value: &str) -> Arc<Self> {
        Arc::new(GlobalProperty {
            driver: driver.into(),
            property: property.into(),
            value: value.into(),
            optional: true,
            used: AtomicBool::new(false),
        })
    }

    /// Whether any object has had this property applied, which `-global` checks to warn about
    /// settings that did nothing.
    pub fn used(&self) -> bool {
        self.used.load(Ordering::SeqCst)
    }
}

/// `object_compat_props`: the accelerator's compat properties, the machine's, and the sugar
/// properties that options like `-no-hpet` turn into.
#[derive(Debug, Default)]
pub struct CompatProps {
    accel: Mutex<Option<Vec<Arc<GlobalProperty>>>>,
    machine: Mutex<Option<Vec<Arc<GlobalProperty>>>>,
    sugar: Mutex<Vec<Arc<GlobalProperty>>>,
}

impl CompatProps {
    /// `object_register_sugar_prop()`.
    pub fn register_sugar_prop(&self, driver: &str, prop: &str, value: &str, optional: bool) {
        let g = if optional {
            GlobalProperty::new_optional(driver, prop, value)
        } else {
            GlobalProperty::new(driver, prop, value)
        };
        lock(&self.sugar).push(g);
    }

    /// `object_set_machine_compat_props()`. It can be set once.
    pub fn set_machine_compat_props(&self, props: Vec<Arc<GlobalProperty>>) {
        let mut m = lock(&self.machine);
        assert!(m.is_none(), "machine compat props are already set");
        *m = Some(props);
    }

    /// `object_set_accelerator_compat_props()`. It can be set once.
    pub fn set_accelerator_compat_props(&self, props: Vec<Arc<GlobalProperty>>) {
        let mut a = lock(&self.accel);
        assert!(a.is_none(), "accelerator compat props are already set");
        *a = Some(props);
    }

    /// `object_apply_compat_props()`. A bad accelerator or machine compat property is a bug and
    /// panics; a bad sugar property is the user's and comes back as the error.
    pub fn apply(&self, obj: &Object) -> Result<()> {
        let accel = lock(&self.accel).clone();
        if let Some(p) = accel {
            obj.apply_global_props(&p, true).or_abort();
        }
        let machine = lock(&self.machine).clone();
        if let Some(p) = machine {
            obj.apply_global_props(&p, true).or_abort();
        }
        let sugar = lock(&self.sugar).clone();
        obj.apply_global_props(&sugar, true)
    }
}
