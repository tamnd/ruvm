// SPDX-License-Identifier: GPL-2.0-or-later

//! The monitor QOM types against the object-add and object-del cases in tests/qtest/qmp-test.c.

use std::sync::Arc;

use ruvm_chardev::Chardevs;
use ruvm_chardev::opts::chardev_opts;
use ruvm_monitor::Qmp;
use ruvm_monitor::object::{monitor_new, register_types};
use ruvm_qapi::json;
use ruvm_qom::Registry;

struct Env {
    registry: Registry,
    qmp: Arc<Qmp>,
    chardevs: Arc<Chardevs>,
}

fn env() -> Env {
    let registry = Registry::new();
    let qmp = Qmp::new();
    let chardevs = Arc::new(Chardevs::new());
    register_types(&registry, &qmp, &chardevs);
    Env { registry, qmp, chardevs }
}

impl Env {
    fn null(&self, id: &str) {
        let mut list = chardev_opts();
        let opts = list.parse(&format!("null,id={id}"), true).unwrap();
        self.chardevs.new_from_opts(opts).unwrap().unwrap();
    }

    fn add(&self, args: &str) -> Result<(), String> {
        let args = json::from_str(args).unwrap();
        let args = args.as_dict().unwrap();
        self.registry
            .user_creatable_add(args, false)
            .map(|_| ())
            .map_err(|e| e.message().to_string())
    }

    fn del(&self, id: &str) -> Result<(), String> {
        self.registry.user_creatable_del(id).map_err(|e| e.message().to_string())
    }
}

#[test]
fn add_and_remove_qmp() {
    let e = env();
    e.null("mon0");
    let add = r#"{"qom-type": "monitor-qmp", "id": "qmp-mon", "chardev": "mon0"}"#;
    e.add(add).unwrap();
    assert_eq!(e.qmp.monitors().len(), 1);
    assert_eq!(e.qmp.monitors()[0].id(), "qmp-mon");
    assert!(e.chardevs.find("mon0").unwrap().is_busy());
    assert_eq!(e.chardevs.remove("mon0").unwrap_err().to_string(), "Chardev 'mon0' is busy");

    e.del("qmp-mon").unwrap();
    assert!(e.qmp.monitors().is_empty());
    assert!(!e.chardevs.find("mon0").unwrap().is_busy());

    // The same id works again once the old monitor is gone.
    e.add(add).unwrap();
    e.del("qmp-mon").unwrap();
    e.chardevs.remove("mon0").unwrap();
}

#[test]
fn bad_monitors() {
    let e = env();
    let err = e.add(r#"{"qom-type": "monitor-qmp", "id": "m", "chardev": "nope"}"#).unwrap_err();
    assert_eq!(err, "chardev \"nope\" not found");
    assert!(e.qmp.monitors().is_empty());

    e.null("c");
    e.add(r#"{"qom-type": "monitor-qmp", "id": "m1", "chardev": "c"}"#).unwrap();
    let err = e.add(r#"{"qom-type": "monitor-qmp", "id": "m2", "chardev": "c"}"#).unwrap_err();
    assert_eq!(err, "chardev 'c' is already in use");
    assert_eq!(e.qmp.monitors().len(), 1);
    assert!(e.registry.objects_root().resolve_path_component("m2").is_none());

    let err = e.add(r#"{"qom-type": "monitor", "id": "m3", "chardev": "c"}"#).unwrap_err();
    assert!(err.contains("abstract"), "{err}");
}

#[test]
fn hmp_cannot_be_deleted() {
    let e = env();
    e.null("hmp-chr");
    e.add(r#"{"qom-type": "monitor-hmp", "id": "hmp-mon", "chardev": "hmp-chr"}"#).unwrap();
    assert_eq!(e.del("hmp-mon").unwrap_err(), "Deleting HMP monitors is not supported");
    assert!(e.chardevs.find("hmp-chr").unwrap().is_busy());
}

#[test]
fn properties_and_monitor_new() {
    let e = env();
    e.null("c");
    let obj = monitor_new(&e.registry, "compat_monitor0", Some("c"), true, true).unwrap();
    assert_eq!(obj.property_get_str("chardev").unwrap(), "c");
    assert!(obj.property_get_bool("pretty").unwrap());
    assert_eq!(obj.property_get_str("close-action").unwrap(), "none");
    assert_eq!(e.qmp.monitors()[0].id(), "compat_monitor0");

    let e = env();
    e.null("h");
    let obj = monitor_new(&e.registry, "compat_monitor0", Some("h"), false, false).unwrap();
    assert!(obj.property_get_bool("readline").unwrap());
    assert!(e.qmp.monitors().is_empty());
}
