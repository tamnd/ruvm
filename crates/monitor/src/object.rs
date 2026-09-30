// SPDX-License-Identifier: GPL-2.0-or-later

//! The `monitor`, `monitor-qmp` and `monitor-hmp` QOM types from monitor/monitor.c,
//! monitor/qmp.c and monitor/hmp.c: what `-object monitor-qmp,...`, `object-add` and the
//! `-mon` and `-qmp` options create.

use std::io;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::{Error, Result};
use ruvm_chardev::{Attachment, Chardev, Chardevs, Connection, Frontend};
use ruvm_qapi::types::MonitorQMPCloseAction;
use ruvm_qom::{
    BoolGetter, BoolSetter, EnumGetter, EnumSetter, Object, Registry, StrGetter, StrSetter,
    TYPE_OBJECT, TYPE_USER_CREATABLE, TypeInfo, UserCreatableClass,
};

use crate::qmp::{MonitorQmp, Qmp};

/// `TYPE_MONITOR`, the abstract parent of both kinds.
pub const TYPE_MONITOR: &str = "monitor";
/// `TYPE_MONITOR_QMP`.
pub const TYPE_MONITOR_QMP: &str = "monitor-qmp";
/// `TYPE_MONITOR_HMP`.
pub const TYPE_MONITOR_HMP: &str = "monitor-hmp";

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What `Monitor` holds for both kinds.
#[derive(Debug, Default)]
struct MonitorState {
    chardev_id: Mutex<Option<String>>,
    /// `mon->chr`, the frontend side of the chardev.
    attachment: Mutex<Option<Attachment>>,
}

#[derive(Debug, Default)]
struct QmpState {
    pretty: AtomicBool,
    close_action: AtomicUsize,
    mon: Mutex<Option<Arc<MonitorQmp>>>,
}

#[derive(Debug, Default)]
struct HmpState {
    readline: AtomicBool,
}

fn base(obj: &Object) -> Arc<MonitorState> {
    obj.state::<MonitorState>().expect("a monitor object")
}

fn qmp_state(obj: &Object) -> Arc<QmpState> {
    obj.state::<QmpState>().expect("a monitor-qmp object")
}

/// The chardev a monitor names, as `monitor_complete()` looks it up.
fn find_chardev(chardevs: &Chardevs, obj: &Object) -> Result<Option<Arc<Chardev>>> {
    let Some(id) = lock(&base(obj).chardev_id).clone() else { return Ok(None) };
    match chardevs.find(&id) {
        Some(chr) => Ok(Some(chr)),
        None => Err(Error::generic(format!("chardev \"{id}\" not found"))),
    }
}

/// Drops the monitor's side of its chardev, `qemu_chr_fe_deinit()`.
fn detach(obj: &Object) {
    lock(&base(obj).attachment).take();
}

/// `monitor_qmp_complete()`, with `monitor_complete()` first as the parent class would run it.
fn qmp_complete(qmp: &Arc<Qmp>, chardevs: &Chardevs, obj: &Object) -> Result<()> {
    let chr = find_chardev(chardevs, obj)?;
    if let Some(chr) = &chr {
        if chr.is_busy() {
            return Err(Error::generic(format!("chardev '{}' is already in use", chr.label())));
        }
    }
    let id = obj.canonical_path_component().unwrap_or_default();
    let st = qmp_state(obj);
    let requires_iothread = chr.as_ref().is_some_and(|c| c.socket().is_some());
    let mon = qmp.add_monitor(&id, st.pretty.load(Ordering::Acquire), requires_iothread);
    if let Some(chr) = &chr {
        match chr.attach(mon.clone()) {
            Ok(a) => *lock(&base(obj).attachment) = Some(a),
            Err(e) => {
                qmp.remove_monitor(&mon);
                return Err(e);
            }
        }
    }
    *lock(&st.mon) = Some(mon);
    Ok(())
}

/// Takes the monitor out of the dispatcher's list and off its chardev.
fn qmp_teardown(qmp: &Qmp, obj: &Object) {
    if let Some(mon) = lock(&qmp_state(obj).mon).take() {
        qmp.remove_monitor(&mon);
    }
    detach(obj);
}

/// `monitor_qmp_prepare_delete()`.
fn qmp_prepare_delete(qmp: &Qmp, obj: &Object) -> Result<()> {
    let mon = lock(&qmp_state(obj).mon).clone();
    if mon.is_some_and(|m| qmp.is_servicing(&m)) {
        return Err(Error::generic("Cannot delete the current QMP monitor"));
    }
    qmp_teardown(qmp, obj);
    Ok(())
}

/// The frontend of an HMP monitor. ruvm has no HMP yet, so it only ever sits on chardevs that
/// carry no data.
struct HmpFrontend;

impl Frontend for HmpFrontend {
    fn serve(&self, _conn: &mut Connection) -> io::Result<()> {
        Ok(())
    }
}

/// `monitor_hmp_complete()`, with `monitor_complete()` first.
fn hmp_complete(chardevs: &Chardevs, obj: &Object) -> Result<()> {
    let Some(chr) = find_chardev(chardevs, obj)? else { return Ok(()) };
    if chr.socket().is_some() {
        return Err(Error::generic("HMP monitors are not supported by ruvm yet"));
    }
    let a = chr.attach(Arc::new(HmpFrontend))?;
    *lock(&base(obj).attachment) = Some(a);
    Ok(())
}

fn str_prop<T: std::any::Any + Send + Sync>(
    get: impl Fn(&T) -> String + Send + Sync + 'static,
    set: impl Fn(&T, &str) + Send + Sync + 'static,
) -> (StrGetter, StrSetter) {
    let get = Arc::new(move |o: &Object| Ok(get(&o.state::<T>().expect("own state"))));
    let set = Arc::new(move |o: &Object, v: &str| {
        set(&o.state::<T>().expect("own state"), v);
        Ok(())
    });
    (get, set)
}

fn bool_prop<T: std::any::Any + Send + Sync>(
    cell: impl Fn(&T) -> &AtomicBool + Send + Sync + Copy + 'static,
) -> (BoolGetter, BoolSetter) {
    let get = Arc::new(move |o: &Object| {
        Ok(cell(&o.state::<T>().expect("own state")).load(Ordering::Acquire))
    });
    let set = Arc::new(move |o: &Object, v: bool| {
        cell(&o.state::<T>().expect("own state")).store(v, Ordering::Release);
        Ok(())
    });
    (get, set)
}

/// Registers the three types. Monitors created from them join `qmp` and look their chardev
/// up in `chardevs`.
pub fn register_types(registry: &Registry, qmp: &Arc<Qmp>, chardevs: &Arc<Chardevs>) {
    let monitor = TypeInfo::new(TYPE_MONITOR)
        .parent(TYPE_OBJECT)
        .abstract_()
        .interface(TYPE_USER_CREATABLE)
        .instance_state(MonitorState::default)
        .class_init(|k| {
            let (get, set) = str_prop::<MonitorState>(
                |s| lock(&s.chardev_id).clone().unwrap_or_default(),
                |s, v| *lock(&s.chardev_id) = Some(v.to_string()),
            );
            k.property_add_str("chardev", Some(get), Some(set));
        })
        .instance_finalize(detach);

    let (q, c) = (qmp.clone(), chardevs.clone());
    let q2 = qmp.clone();
    let monitor_qmp = TypeInfo::new(TYPE_MONITOR_QMP)
        .parent(TYPE_MONITOR)
        .instance_state(QmpState::default)
        .class_init(move |k| {
            let (get, set) = bool_prop::<QmpState>(|s| &s.pretty);
            k.property_add_bool("pretty", Some(get), Some(set));
            let get: EnumGetter =
                Arc::new(|o: &Object| Ok(qmp_state(o).close_action.load(Ordering::Acquire)));
            let set: EnumSetter = Arc::new(|o: &Object, v: usize| {
                qmp_state(o).close_action.store(v, Ordering::Release);
                Ok(())
            });
            k.property_add_enum(
                "close-action",
                "MonitorQMPCloseAction",
                MonitorQMPCloseAction::LOOKUP,
                Some(get),
                Some(set),
            );
            let (q, c, q2) = (q.clone(), c.clone(), q.clone());
            let uc = k.interface(TYPE_USER_CREATABLE).expect("monitors are user creatable");
            uc.set_ext(UserCreatableClass {
                complete: Some(Arc::new(move |o| qmp_complete(&q, &c, o))),
                prepare_delete: Some(Arc::new(move |o| qmp_prepare_delete(&q2, o))),
            });
        })
        .instance_finalize(move |o| qmp_teardown(&q2, o));

    let c = chardevs.clone();
    let monitor_hmp = TypeInfo::new(TYPE_MONITOR_HMP)
        .parent(TYPE_MONITOR)
        .instance_state(HmpState::default)
        .class_init(move |k| {
            let (get, set) = bool_prop::<HmpState>(|s| &s.readline);
            k.property_add_bool("readline", Some(get), Some(set));
            let c = c.clone();
            let uc = k.interface(TYPE_USER_CREATABLE).expect("monitors are user creatable");
            uc.set_ext(UserCreatableClass {
                complete: Some(Arc::new(move |o| hmp_complete(&c, o))),
                prepare_delete: Some(Arc::new(|_| {
                    Err(Error::generic("Deleting HMP monitors is not supported"))
                })),
            });
        });

    registry.register_all([monitor, monitor_qmp, monitor_hmp]);
}

/// `monitor_compat_id()`: `compat_monitor0`, `compat_monitor1` and so on.
pub fn monitor_compat_id() -> String {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    format!("compat_monitor{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

/// `monitor_new_qmp()` and `monitor_new_hmp()` for `-mon`, `-qmp` and `-monitor`: a monitor
/// object under `/objects` named `id`, or a compat id when there is none.
pub fn monitor_new(
    registry: &Registry,
    id: Option<&str>,
    chardev: Option<&str>,
    qmp_mode: bool,
    pretty: bool,
) -> Result<Object> {
    let id = id.map_or_else(monitor_compat_id, str::to_string);
    let parent = registry.objects_root();
    let mut props = Vec::new();
    if let Some(chardev) = chardev {
        props.push(("chardev", chardev));
    }
    if qmp_mode {
        props.push(("pretty", if pretty { "yes" } else { "no" }));
        registry.object_new_with_props(TYPE_MONITOR_QMP, Some((&parent, &id)), &props)
    } else {
        props.push(("readline", "on"));
        registry.object_new_with_props(TYPE_MONITOR_HMP, Some((&parent, &id)), &props)
    }
}
