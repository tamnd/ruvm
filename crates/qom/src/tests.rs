// SPDX-License-Identifier: GPL-2.0-or-later

//! Ports of tests/unit/check-qom-proplist.c and check-qom-interface.c, plus checks of the
//! things management tools see: paths, `qom-list` order and error texts.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_base::ErrorClass;
use ruvm_qapi::visit::QEnumLookup;
use ruvm_qapi::{QDict, QValue};

use crate::*;

const TYPE_DUMMY: &str = "qemu-dummy";
const ANIMALS: QEnumLookup = QEnumLookup::new(&["frog", "alligator", "platypus"]);

#[derive(Default)]
struct Dummy {
    bv: Mutex<bool>,
    av: Mutex<usize>,
    sv: Mutex<String>,
}

fn dummy(o: &Object) -> Arc<Dummy> {
    o.state::<Dummy>().unwrap()
}

fn dummy_type() -> TypeInfo {
    TypeInfo::new(TYPE_DUMMY)
        .parent(TYPE_OBJECT)
        .interface(TYPE_USER_CREATABLE)
        .instance_state(Dummy::default)
        .instance_init(|o| {
            let get: BoolGetter = Arc::new(|o| Ok(*lock(&dummy(o).bv)));
            let set: BoolSetter = Arc::new(|o, v| {
                *lock(&dummy(o).bv) = v;
                Ok(())
            });
            o.property_add_bool("bv", Some(get), Some(set));
        })
        .class_init(|k| {
            let get: StrGetter = Arc::new(|o| Ok(lock(&dummy(o).sv).clone()));
            let set: StrSetter = Arc::new(|o, v| {
                *lock(&dummy(o).sv) = v.to_string();
                Ok(())
            });
            k.property_add_str("sv", Some(get), Some(set));
            let get: EnumGetter = Arc::new(|o| Ok(*lock(&dummy(o).av)));
            let set: EnumSetter = Arc::new(|o, v| {
                *lock(&dummy(o).av) = v;
                Ok(())
            });
            k.property_add_enum("av", "DummyAnimal", ANIMALS, Some(get), Some(set));
        })
}

fn registry() -> Registry {
    let r = Registry::new();
    r.register(dummy_type());
    r
}

fn check_dummy(o: &Object) {
    let d = dummy(o);
    assert!(*lock(&d.bv));
    assert_eq!(*lock(&d.av), 2);
    assert_eq!(*lock(&d.sv), "Hiss hiss hiss");
}

#[test]
fn dummy_create_complex() {
    let r = registry();
    let parent = r.objects_root();
    let o = r
        .object_new_with_props(
            TYPE_DUMMY,
            Some((&parent, "dummy0")),
            &[("bv", "yes"), ("sv", "Hiss hiss hiss"), ("av", "platypus")],
        )
        .unwrap();
    check_dummy(&o);
    assert_eq!(o.canonical_path().as_deref(), Some("/objects/dummy0"));
    // Our handle plus the child property.
    assert_eq!(o.ref_count(), 2);
    o.unparent();
    assert_eq!(o.ref_count(), 1);
    assert!(o.parent().is_none());
}

#[test]
fn dummy_create_parentless() {
    let r = registry();
    let o = r
        .object_new_with_props(
            TYPE_DUMMY,
            None,
            &[("bv", "yes"), ("sv", "Hiss hiss hiss"), ("av", "platypus")],
        )
        .unwrap();
    check_dummy(&o);
    assert_eq!(o.ref_count(), 1);
    assert!(o.canonical_path().is_none());
}

#[test]
fn dummy_create_cmdline() {
    let r = registry();
    let args = QDict::new()
        .with("qom-type", TYPE_DUMMY)
        .with("id", "dev0")
        .with("bv", "yes")
        .with("sv", "Hiss hiss hiss")
        .with("av", "platypus");
    let o = r.user_creatable_add(&args, true).unwrap();
    check_dummy(&o);
    assert_eq!(o.ref_count(), 2);
    drop(o);

    let e = r.user_creatable_add(&args, true).unwrap_err();
    assert_eq!(
        e.message(),
        "attempt to add duplicate property 'dev0' to object (type 'container')"
    );

    r.user_creatable_del("dev0").unwrap();
    assert!(r.objects_root().resolve_path_component("dev0").is_none());
    let o = r.user_creatable_add(&args, true).unwrap();
    check_dummy(&o);
    r.user_creatable_del("dev0").unwrap();
    assert_eq!(r.user_creatable_del("dev0").unwrap_err().message(), "object 'dev0' not found");
}

#[test]
fn dummy_bad_enum() {
    let r = registry();
    let parent = r.objects_root();
    let e = r
        .object_new_with_props(
            TYPE_DUMMY,
            Some((&parent, "dummy0")),
            &[("bv", "yes"), ("sv", "Hiss hiss hiss"), ("av", "yeti")],
        )
        .unwrap_err();
    assert_eq!(e.message(), "Parameter 'av' does not accept value 'yeti'");
    assert!(parent.resolve_path_component("dummy0").is_none());
}

#[test]
fn dummy_get_enum() {
    let r = registry();
    let o = r.object_new_with_props(TYPE_DUMMY, None, &[("av", "platypus")]).unwrap();
    assert_eq!(o.property_get_enum("av", "DummyAnimal").unwrap(), 2);
    let e = o.property_get_enum("av", "BadAnimal").unwrap_err();
    assert_eq!(e.message(), "Property av on qemu-dummy is not 'BadAnimal' enum type");
    let e = o.property_get_enum("iv", "DummyAnimal").unwrap_err();
    assert_eq!(e.message(), "Property 'qemu-dummy.iv' not found");
}

fn names(props: &[Arc<Property>]) -> Vec<&str> {
    let mut v: Vec<&str> = props.iter().map(|p| p.name()).collect();
    v.sort_unstable();
    v
}

#[test]
fn dummy_iterator() {
    let r = registry();
    let o = r.object_new(TYPE_DUMMY).unwrap();
    assert_eq!(names(&o.properties()), ["av", "bv", "sv", "type"]);
    assert_eq!(names(&o.class().properties()), ["av", "sv", "type"]);
}

#[test]
fn dummy_print_and_parse() {
    let r = registry();
    let o = r.object_new(TYPE_DUMMY).unwrap();
    o.property_parse("av", "alligator").unwrap();
    assert_eq!(o.property_print("av", false).unwrap(), "alligator");
    o.property_parse("bv", "on").unwrap();
    assert_eq!(o.property_print("bv", false).unwrap(), "true");
    assert_eq!(o.property_get_type("av").unwrap(), "DummyAnimal");
    assert_eq!(o.property_get_str("type").unwrap(), TYPE_DUMMY);
    let e = o.property_get_bool("sv").unwrap_err();
    assert_eq!(e.message(), "Invalid parameter type for 'sv', expected: boolean");
}

// The delchild test: a device with a bus child and a link to a backend.

fn delchild_types(r: &Registry, log: Arc<Mutex<Vec<String>>>) {
    #[derive(Default)]
    struct Dev {
        backend: Arc<LinkSlot>,
    }
    let l = log.clone();
    r.register(
        TypeInfo::new("qemu-dummy-dev")
            .parent(TYPE_OBJECT)
            .instance_state(Dev::default)
            .instance_init(|o| {
                let slot = o.state::<Dev>().unwrap().backend.clone();
                o.property_add_link(
                    "backend",
                    "qemu-dummy-backend",
                    slot,
                    Some(Arc::new(allow_set_link)),
                    LinkFlags::STRONG,
                );
            })
            .class_init(move |k| {
                let l = l.clone();
                k.set_unparent(move |o| {
                    lock(&l).push(format!("unparent dev {}", o.typename()));
                    if let Some(bus) = o.resolve_path_component("bus") {
                        bus.unparent();
                    }
                });
            }),
    );
    let l = log.clone();
    r.register(TypeInfo::new("qemu-dummy-bus").parent(TYPE_OBJECT).class_init(move |k| {
        let l = l.clone();
        k.set_unparent(move |_| lock(&l).push("unparent bus".into()));
    }));
    let l = log;
    r.register(
        TypeInfo::new("qemu-dummy-backend")
            .parent(TYPE_OBJECT)
            .instance_finalize(move |_| lock(&l).push("finalize backend".into())),
    );
}

#[test]
fn dummy_delchild() {
    let r = Registry::new();
    let log = Arc::new(Mutex::new(Vec::new()));
    delchild_types(&r, log.clone());
    let parent = r.objects_root();
    let dev = r.object_new_with_props("qemu-dummy-dev", Some((&parent, "dev0")), &[]).unwrap();
    let bus = r.object_new_with_props("qemu-dummy-bus", Some((&dev, "bus")), &[]).unwrap();
    let backend =
        r.object_new_with_props("qemu-dummy-backend", Some((&parent, "backend0")), &[]).unwrap();
    dev.property_set_link("backend", Some(&backend)).unwrap();
    assert_eq!(dev.property_get_str("backend").unwrap(), "/objects/backend0");
    assert!(dev.property_get_link("backend").unwrap().unwrap().ptr_eq(&backend));
    assert_eq!(dev.property_get_type("backend").unwrap(), "link<qemu-dummy-backend>");
    assert_eq!(dev.property_get_type("bus").unwrap(), "child<qemu-dummy-bus>");
    assert_eq!(bus.canonical_path().as_deref(), Some("/objects/dev0/bus"));
    drop(bus);

    backend.unparent();
    drop(backend);
    assert!(lock(&log).is_empty(), "the strong link keeps the backend alive");

    dev.unparent();
    assert_eq!(*lock(&log), ["unparent dev qemu-dummy-dev", "unparent bus"]);
    drop(dev);
    assert_eq!(lock(&log).last().map(String::as_str), Some("finalize backend"));
}

#[test]
fn partial_path() {
    let r = registry();
    let root = r.objects_root();
    let cont1 = root.add_new_container("cont1");
    let obj1 = r.object_new_with_props(TYPE_DUMMY, Some((&cont1, "obj1")), &[]).unwrap();
    r.object_new_with_props(TYPE_DUMMY, Some((&cont1, "obj2")), &[]).unwrap();
    r.object_new_with_props(TYPE_DUMMY, Some((&root, "obj2")), &[]).unwrap();

    let (found, ambiguous) = r.resolve_path_type("", TYPE_DUMMY);
    assert!(found.is_none());
    assert!(ambiguous);
    let (found, ambiguous) = r.resolve_path_type("obj2", TYPE_DUMMY);
    assert!(found.is_none());
    assert!(ambiguous);
    let (found, ambiguous) = r.resolve_path_type("obj1", TYPE_DUMMY);
    assert!(found.unwrap().ptr_eq(&obj1));
    assert!(!ambiguous);
    let (found, _) = r.resolve_path("/objects/cont1/obj1");
    assert!(found.unwrap().ptr_eq(&obj1));
    let (found, _) = r.resolve_path("/objects/cont1/nope");
    assert!(found.is_none());
}

// check-qom-interface.c

const TYPE_TEST_IF: &str = "test-interface";

#[derive(Clone, Default)]
struct TestIfClass {
    test: u32,
}

const TEST_MAGIC: u32 = 0xFAFB_FCFD;

fn interface_registry() -> Registry {
    let r = Registry::new();
    r.register(TypeInfo::new(TYPE_TEST_IF).parent(TYPE_INTERFACE));
    r.register(
        TypeInfo::new("direct-impl").parent(TYPE_OBJECT).interface(TYPE_TEST_IF).class_init(|k| {
            k.interface(TYPE_TEST_IF).unwrap().set_ext(TestIfClass { test: TEST_MAGIC });
        }),
    );
    r.register(TypeInfo::new("intermediate-impl").parent("direct-impl"));
    r
}

fn check_interface(r: &Registry, typename: &str) {
    let o = r.object_new(typename).unwrap();
    assert!(o.dynamic_cast(TYPE_TEST_IF).is_some());
    let ic = o.class().dynamic_cast(TYPE_TEST_IF).unwrap();
    assert_eq!(ic.ext::<TestIfClass>().unwrap().test, TEST_MAGIC);
    assert_eq!(ic.interface_type(), Some(TYPE_TEST_IF));
    assert_eq!(ic.name(), format!("{typename}::{TYPE_TEST_IF}"));
}

#[test]
fn interface_direct_impl() {
    check_interface(&interface_registry(), "direct-impl");
}

#[test]
fn interface_intermediate_impl() {
    check_interface(&interface_registry(), "intermediate-impl");
}

#[test]
fn interface_is_abstract() {
    let r = interface_registry();
    assert!(r.class_by_name(TYPE_TEST_IF).unwrap().is_abstract());
    let e = r.object_new_with_props(TYPE_TEST_IF, None, &[]).unwrap_err();
    assert_eq!(e.message(), "object type 'test-interface' is abstract");
}

// Beyond the C tests.

#[test]
fn class_hooks_run_in_order() {
    let r = Registry::new();
    let log = Arc::new(Mutex::new(Vec::new()));
    let (l1, l2, l3) = (log.clone(), log.clone(), log.clone());
    r.register(
        TypeInfo::new("aaa")
            .parent(TYPE_OBJECT)
            .class_base_init(move |k| lock(&l1).push(format!("a base {}", k.name())))
            .class_init(move |k| lock(&l2).push(format!("a init {}", k.name()))),
    );
    r.register(
        TypeInfo::new("bbb")
            .parent("aaa")
            .class_init(move |k| lock(&l3).push(format!("b init {}", k.name()))),
    );
    r.class_by_name("bbb").unwrap();
    assert_eq!(*lock(&log), ["a init aaa", "a base bbb", "b init bbb"]);
}

#[test]
fn instance_hooks_and_finalize() {
    let r = Registry::new();
    let log = Arc::new(Mutex::new(Vec::new()));
    let (l1, l2, l3, l4, l5) = (log.clone(), log.clone(), log.clone(), log.clone(), log.clone());
    r.register(
        TypeInfo::new("aaa")
            .parent(TYPE_OBJECT)
            .instance_init(move |_| lock(&l1).push("a init"))
            .instance_post_init(move |_| lock(&l2).push("a post"))
            .instance_finalize(move |_| lock(&l3).push("a fini")),
    );
    r.register(
        TypeInfo::new("bbb")
            .parent("aaa")
            .instance_init(move |_| lock(&l4).push("b init"))
            .instance_finalize(move |_| lock(&l5).push("b fini")),
    );
    let o = r.object_new("bbb").unwrap();
    let o2 = o.clone();
    assert_eq!(o.ref_count(), 2);
    drop(o);
    assert_eq!(*lock(&log), ["a init", "b init", "a post"]);
    drop(o2);
    assert_eq!(*lock(&log), ["a init", "b init", "a post", "b fini", "a fini"]);
}

#[test]
fn weak_handles_do_not_keep_objects() {
    let r = registry();
    let o = r.object_new(TYPE_DUMMY).unwrap();
    let w = o.downgrade();
    assert!(w.upgrade().is_some());
    drop(o);
    assert!(w.upgrade().is_none());
}

#[test]
fn child_names_with_star() {
    let r = registry();
    let root = r.objects_root();
    for _ in 0..3 {
        let o = r.object_new(TYPE_DUMMY).unwrap();
        root.property_add_child("dummy[*]", &o);
    }
    let mut kids: Vec<String> =
        root.children().iter().filter_map(Object::canonical_path_component).collect();
    kids.sort();
    assert_eq!(kids, ["dummy[0]", "dummy[1]", "dummy[2]"]);
}

#[test]
fn duplicate_property_is_an_error() {
    let r = registry();
    let o = r.object_new(TYPE_DUMMY).unwrap();
    let e = o.property_try_add(Property::new_bool("bv", None, None)).unwrap_err();
    assert_eq!(e.message(), "attempt to add duplicate property 'bv' to object (type 'qemu-dummy')");
}

#[test]
fn read_only_and_write_only() {
    let r = registry();
    let o = r.object_new(TYPE_DUMMY).unwrap();
    let e = o.property_set_str("type", "x").unwrap_err();
    assert_eq!(e.message(), "Property 'qemu-dummy.type' is not writable");
    let set: StrSetter = Arc::new(|_, _| Ok(()));
    o.property_add_str("wo", None, Some(set));
    let e = o.property_get_str("wo").unwrap_err();
    assert_eq!(e.message(), "Property 'qemu-dummy.wo' is not readable");
}

#[test]
fn uint_pointer_properties() {
    let r = registry();
    let o = r.object_new(TYPE_DUMMY).unwrap();
    let cell = Arc::new(AtomicU32::new(7));
    o.property_add_uint32_ptr("n", cell.clone(), PropFlags::READWRITE);
    assert_eq!(o.property_get_uint("n").unwrap(), 7);
    o.property_set_uint("n", 42).unwrap();
    assert_eq!(cell.load(Ordering::Relaxed), 42);
    assert_eq!(o.property_get_type("n").unwrap(), "uint32");
}

#[test]
fn alias_forwards_to_target() {
    let r = registry();
    let target = r.object_new(TYPE_DUMMY).unwrap();
    let o = r.object_new(TYPE_DUMMY).unwrap();
    o.property_add_alias("animal", &target, "av");
    o.property_set_str("animal", "frog").unwrap();
    assert_eq!(*lock(&dummy(&target).av), 0);
    assert_eq!(o.property_get_str("animal").unwrap(), "frog");
    assert_eq!(o.property_get_type("animal").unwrap(), "DummyAnimal");
}

#[test]
fn qom_list_is_reversed() {
    let r = registry();
    let o = r.object_new(TYPE_DUMMY).unwrap();
    r.objects_root().property_add_child("d", &o);
    let list = qmp::qom_list(&r, "/objects/d").unwrap();
    let got: Vec<String> = list
        .as_list()
        .unwrap()
        .iter()
        .map(|v| v.as_dict().unwrap().get_str("name").unwrap().to_string())
        .collect();
    let mut want: Vec<String> = o.properties().iter().map(|p| p.name().to_string()).collect();
    want.reverse();
    assert_eq!(got, want);
}

#[test]
fn qmp_errors() {
    let r = registry();
    let e = qmp::qom_list(&r, "/nope").unwrap_err();
    assert_eq!(e.class(), ErrorClass::DeviceNotFound);
    assert_eq!(e.message(), "Device '/nope' not found");
    let e = qmp::device_list_properties(&r, "nope").unwrap_err();
    assert_eq!(e.message(), "Device 'nope' not found");
    let e = qmp::device_list_properties(&r, TYPE_DUMMY).unwrap_err();
    assert_eq!(e.message(), "Parameter 'typename' expects a non-abstract device type");
    let e = qmp::qom_list_properties(&r, "nope").unwrap_err();
    assert_eq!(e.message(), "Class 'nope' not found");
    let e = qmp::qom_list_properties(&r, TYPE_USER_CREATABLE).unwrap_err();
    assert_eq!(e.message(), "Parameter 'typename' expects a QOM type");
    let e = qmp::object_del(&r, "x").unwrap_err();
    assert_eq!(e.message(), "object 'x' not found");
}

#[test]
fn qmp_object_add_and_get() {
    let r = registry();
    let args = QDict::new()
        .with("qom-type", TYPE_DUMMY)
        .with("id", "d0")
        .with("bv", true)
        .with("av", "alligator");
    qmp::object_add(&r, &args).unwrap();
    let v = qmp::qom_get(&r, "/objects/d0", "av").unwrap();
    assert_eq!(v.as_str(), Some("alligator"));
    qmp::qom_set(&r, "/objects/d0", "sv", QValue::str("hi")).unwrap();
    assert_eq!(qmp::qom_get(&r, "/objects/d0", "sv").unwrap().as_str(), Some("hi"));
    qmp::object_del(&r, "d0").unwrap();
}

#[test]
fn bad_id_is_refused() {
    let r = registry();
    let parent = r.objects_root();
    let e = r.object_new_with_props(TYPE_DUMMY, Some((&parent, "0bad")), &[]).unwrap_err();
    assert_eq!(e.message(), "Parameter 'id' expects an identifier");
    let e = r.object_new_with_props("nope", None, &[]).unwrap_err();
    assert_eq!(e.message(), "invalid object type: nope");
}

#[test]
fn list_types_filters_and_reports_parents() {
    let r = registry();
    let list = qmp::qom_list_types(&r, Some(TYPE_USER_CREATABLE), false);
    let names: Vec<&str> = list
        .as_list()
        .unwrap()
        .iter()
        .map(|v| v.as_dict().unwrap().get_str("name").unwrap())
        .collect();
    assert_eq!(names, [TYPE_DUMMY]);
    let d = list.as_list().unwrap()[0].as_dict().unwrap();
    assert_eq!(d.get_str("parent"), Some(TYPE_OBJECT));
    assert_eq!(d.get("abstract").and_then(QValue::as_bool), Some(false));
}

#[test]
fn global_properties_apply() {
    let r = registry();
    let good = GlobalProperty::new(TYPE_DUMMY, "sv", "global");
    let unused = GlobalProperty::new("other", "sv", "x");
    r.compat_props().set_machine_compat_props(vec![good.clone(), unused.clone()]);
    let o = r.object_new(TYPE_DUMMY).unwrap();
    r.compat_props().apply(&o).unwrap();
    assert_eq!(o.property_get_str("sv").unwrap(), "global");
    assert!(good.used());
    assert!(!unused.used());
}

#[test]
fn property_help_text() {
    let r = registry();
    let text = type_print_class_properties(&r, TYPE_DUMMY).unwrap();
    assert!(text.contains("av=<DummyAnimal>"), "{text}");
    assert!(type_print_class_properties(&r, "nope").is_none());
}
