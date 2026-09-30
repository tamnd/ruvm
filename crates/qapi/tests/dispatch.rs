// SPDX-License-Identifier: GPL-2.0-or-later

//! Checks `qmp_dispatch()` and the generated marshallers against the replies QEMU gives,
//! following tests/unit/test-qmp-cmds.c where the schema allows it.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ruvm_base::{Error, ErrorClass};
use ruvm_qapi::commands::*;
use ruvm_qapi::dispatch::{
    DispatchEnv, QmpCommandList, QmpCommandOptions, qmp_dispatch, qmp_is_oob,
};
use ruvm_qapi::events::{QapiEvent, event_device_deleted, event_stop};
use ruvm_qapi::types::*;
use ruvm_qapi::visit::{CompatPolicy, CompatPolicyInput};
use ruvm_qapi::{QDict, QValue, json};

#[derive(Default)]
struct Ctx {
    stops: AtomicUsize,
}

fn table() -> QmpCommandList<Ctx> {
    let mut cmds = QmpCommandList::new();
    register_stop(&mut cmds, |c: &Ctx| {
        c.stops.fetch_add(1, Ordering::Relaxed);
        Ok(())
    });
    register_qom_list(&mut cmds, |_: &Ctx, arg: QomListArg| {
        if arg.path != "/machine" {
            return Err(Error::new(
                ErrorClass::DeviceNotFound,
                format!("Device '{}' not found", arg.path),
            ));
        }
        Ok(vec![ObjectPropertyInfo {
            name: "type".into(),
            type_: "string".into(),
            ..Default::default()
        }])
    });
    register_query_kvm(&mut cmds, |_: &Ctx| Ok(KvmInfo { enabled: false, present: true }));
    register_migrate_recover(&mut cmds, |_: &Ctx, _: MigrateRecoverArg| Ok(()));
    cmds
}

fn env(policy: &CompatPolicy) -> DispatchEnv<'_> {
    DispatchEnv { policy, allow_oob: true, machine_ready: true }
}

fn run_env(cmds: &QmpCommandList<Ctx>, req: &str, env: &DispatchEnv<'_>) -> Option<QValue> {
    let ctx = Ctx::default();
    qmp_dispatch(cmds, &json::from_str(req).unwrap(), env, &ctx).map(QValue::Dict)
}

fn run(req: &str) -> QValue {
    let policy = CompatPolicy::default();
    run_env(&table(), req, &env(&policy)).expect("a response")
}

fn j(text: &str) -> QValue {
    json::from_str(text).unwrap()
}

fn desc(rsp: &QValue) -> String {
    let QValue::Dict(d) = rsp else { panic!("{rsp:?}") };
    let Some(QValue::Dict(e)) = d.get("error") else { panic!("not an error: {}", rsp.to_json()) };
    e.get_str("desc").unwrap().to_string()
}

#[test]
fn success_replies() {
    assert_eq!(run(r#"{"execute": "stop"}"#), j(r#"{"return": {}}"#));
    assert_eq!(
        run(r#"{"execute": "stop", "id": [1, "x"]}"#),
        j(r#"{"return": {}, "id": [1, "x"]}"#)
    );
    assert_eq!(
        run(r#"{"execute": "qom-list", "arguments": {"path": "/machine"}}"#),
        j(r#"{"return": [{"name": "type", "type": "string"}]}"#)
    );
    assert_eq!(
        run(r#"{"execute": "query-kvm"}"#),
        j(r#"{"return": {"enabled": false, "present": true}}"#)
    );
}

#[test]
fn handler_runs_with_context() {
    let cmds = table();
    let ctx = Ctx::default();
    let policy = CompatPolicy::default();
    for _ in 0..3 {
        qmp_dispatch(&cmds, &j(r#"{"execute": "stop"}"#), &env(&policy), &ctx);
    }
    assert_eq!(ctx.stops.load(Ordering::Relaxed), 3);
}

#[test]
fn malformed_requests() {
    let cases = [
        ("[1]", "QMP input must be a JSON object"),
        ("{}", "QMP input lacks member 'execute'"),
        (r#"{"execute": 1}"#, "QMP input member 'execute' must be a string"),
        (
            r#"{"execute": "stop", "arguments": []}"#,
            "QMP input member 'arguments' must be an object",
        ),
        (r#"{"execute": "stop", "extra": 1}"#, "QMP input member 'extra' is unexpected"),
        (
            r#"{"execute": "stop", "exec-oob": "stop"}"#,
            // QDict walks keys in QEMU's hash order, so this is the pair QEMU reports.
            "QMP input member 'execute' clashes with 'exec-oob'",
        ),
        (r#"{"execute": "nope"}"#, "The command nope has not been found"),
        (r#"{"exec-oob": "stop"}"#, "The command stop does not support OOB"),
    ];
    for (req, want) in cases {
        assert_eq!(desc(&run(req)), want, "{req}");
    }
    let rsp = run(r#"{"execute": "nope", "id": 7}"#);
    assert_eq!(
        rsp,
        j(
            r#"{"error": {"class": "CommandNotFound", "desc": "The command nope has not been found"}, "id": 7}"#
        )
    );
    // A non-object request has no id to echo back.
    assert_eq!(
        run("[1]"),
        j(r#"{"error": {"class": "GenericError", "desc": "QMP input must be a JSON object"}}"#)
    );
}

#[test]
fn exec_oob_needs_the_capability() {
    let policy = CompatPolicy::default();
    let mut e = env(&policy);
    e.allow_oob = false;
    let rsp =
        run_env(&table(), r#"{"exec-oob": "migrate-recover", "arguments": {"uri": "x"}}"#, &e);
    assert_eq!(desc(&rsp.unwrap()), "QMP input member 'exec-oob' is unexpected");
    assert_eq!(
        run(r#"{"exec-oob": "migrate-recover", "arguments": {"uri": "x"}}"#),
        j(r#"{"return": {}}"#)
    );
    let QValue::Dict(d) = j(r#"{"exec-oob": "stop"}"#) else { unreachable!() };
    assert!(qmp_is_oob(&d));
}

#[test]
fn argument_errors() {
    assert_eq!(desc(&run(r#"{"execute": "qom-list"}"#)), "Parameter 'path' is missing");
    assert_eq!(
        desc(&run(r#"{"execute": "qom-list", "arguments": {"path": "/", "x": 1}}"#)),
        "Parameter 'x' is unexpected"
    );
    assert_eq!(
        desc(&run(r#"{"execute": "stop", "arguments": {"x": 1}}"#)),
        "Parameter 'x' is unexpected"
    );
    assert_eq!(
        run(r#"{"execute": "qom-list", "arguments": {"path": "/nope"}}"#),
        j(r#"{"error": {"class": "DeviceNotFound", "desc": "Device '/nope' not found"}}"#)
    );
}

#[test]
fn disable_and_enable() {
    let mut cmds = table();
    let policy = CompatPolicy::default();
    cmds.disable("stop", None);
    let rsp = run_env(&cmds, r#"{"execute": "stop"}"#, &env(&policy)).unwrap();
    assert_eq!(desc(&rsp), "Command stop has been disabled");
    cmds.disable("stop", Some("not now"));
    let rsp = run_env(&cmds, r#"{"execute": "stop"}"#, &env(&policy)).unwrap();
    assert_eq!(desc(&rsp), "Command stop has been disabled: not now");
    cmds.enable("stop");
    let rsp = run_env(&cmds, r#"{"execute": "stop"}"#, &env(&policy)).unwrap();
    assert_eq!(rsp, j(r#"{"return": {}}"#));
}

#[test]
fn preconfig() {
    let policy = CompatPolicy::default();
    let mut e = env(&policy);
    e.machine_ready = false;
    let cmds = table();
    let rsp = run_env(&cmds, r#"{"execute": "stop"}"#, &e).unwrap();
    assert_eq!(
        desc(&rsp),
        "The command 'stop' is permitted only after machine initialization has completed"
    );
    let rsp = run_env(&cmds, r#"{"execute": "qom-list", "arguments": {"path": "/machine"}}"#, &e);
    assert!(matches!(rsp, Some(QValue::Dict(d)) if d.contains_key("return")));
    assert!(cmds.find("qom-list").unwrap().options.contains(QmpCommandOptions::ALLOW_PRECONFIG));
}

#[test]
fn deprecated_command_policy() {
    let policy = CompatPolicy { deprecated_input: CompatPolicyInput::Reject, ..Default::default() };
    let rsp = run_env(&table(), r#"{"execute": "query-kvm"}"#, &env(&policy)).unwrap();
    assert_eq!(
        rsp,
        j(
            r#"{"error": {"class": "CommandNotFound", "desc": "Deprecated command query-kvm disabled by policy"}}"#
        )
    );
}

#[test]
fn no_success_response() {
    let mut cmds = QmpCommandList::<()>::new();
    cmds.register(
        "quiet",
        Arc::new(|_: &(), _: QDict, _: &CompatPolicy| Ok(None)),
        QmpCommandOptions::NO_SUCCESS_RESP,
        0,
    );
    let policy = CompatPolicy::default();
    let req = j(r#"{"execute": "quiet"}"#);
    assert!(qmp_dispatch(&cmds, &req, &env(&policy), &()).is_none());
    assert!(!cmds.find("quiet").unwrap().has_success_response());
}

#[test]
#[should_panic(expected = "coroutine")]
fn coroutine_and_oob_do_not_mix() {
    let mut cmds = QmpCommandList::<()>::new();
    cmds.register(
        "bad",
        Arc::new(|_: &(), _: QDict, _: &CompatPolicy| Ok(None)),
        QmpCommandOptions::COROUTINE | QmpCommandOptions::ALLOW_OOB,
        0,
    );
}

#[test]
fn events() {
    let policy = CompatPolicy::default();
    let ev = event_stop(&policy).unwrap();
    assert_eq!(ev.get_str("event"), Some("STOP"));
    assert!(!ev.contains_key("data"));
    let Some(QValue::Dict(ts)) = ev.get("timestamp") else { panic!() };
    assert!(ts.contains_key("seconds") && ts.contains_key("microseconds"));

    let arg = DeviceDeletedArg { device: Some("d0".into()), path: "/machine/peripheral/d0".into() };
    let ev = event_device_deleted(&policy, arg).unwrap();
    assert_eq!(
        ev.get("data").cloned().unwrap(),
        j(r#"{"device": "d0", "path": "/machine/peripheral/d0"}"#)
    );

    assert_eq!(QapiEvent::from_name("DEVICE_DELETED"), Some(QapiEvent::DeviceDeleted));
    assert_eq!(QapiEvent::Stop.as_str(), "STOP");
}
