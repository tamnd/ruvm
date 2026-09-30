// SPDX-License-Identifier: GPL-2.0-or-later

//! Checks the generated types against what QEMU's generated visitors accept, produce and say
//! when the input is wrong.

use ruvm_qapi::types::*;
use ruvm_qapi::visit::{
    CompatPolicy, CompatPolicyInput, CompatPolicyOutput, QObjectInputVisitor, QObjectOutputVisitor,
    Visit,
};
use ruvm_qapi::{QValue, json};

fn input<T: Visit>(text: &str) -> ruvm_base::Result<T> {
    let mut v =
        QObjectInputVisitor::new_qmp(json::from_str(text).unwrap(), CompatPolicy::default());
    let mut obj = T::default();
    T::visit(&mut v, None, &mut obj)?;
    Ok(obj)
}

fn output<T: Visit>(obj: &mut T) -> String {
    let mut v = QObjectOutputVisitor::new();
    T::visit(&mut v, None, obj).unwrap();
    v.complete().to_json()
}

fn err<T: Visit + std::fmt::Debug>(text: &str) -> String {
    input::<T>(text).unwrap_err().message().to_string()
}

#[test]
fn enum_names_and_lookup() {
    assert_eq!(BlockdevDriver::Qcow2.as_str(), "qcow2");
    // host_device only exists where QEMU has HAVE_HOST_BLOCK_DEVICE, which leaves out Windows.
    #[cfg(not(windows))]
    assert_eq!(BlockdevDriver::from_name("host_device"), Some(BlockdevDriver::HostDevice));
    assert_eq!(BlockdevDriver::from_name("nope"), None);
    assert_eq!(RunState::default(), RunState::ALL[0]);
    assert_eq!(err::<RunState>("\"nope\""), "Parameter 'null' does not accept value 'nope'");
}

#[test]
fn union_round_trip() {
    let text = r#"{"driver": "qcow2", "node-name": "disk0", "file": {"driver": "file", "filename": "a.img"}, "backing": null}"#;
    let mut opts: BlockdevOptions = input(text).unwrap();
    assert_eq!(opts.node_name.as_deref(), Some("disk0"));
    let BlockdevOptionsU::Qcow2(q) = &opts.u else { panic!("{:?}", opts.u) };
    let BlockdevRef::Definition(file) = &*q.file else { panic!() };
    assert_eq!(file.u.tag(), BlockdevDriver::File);
    assert!(matches!(q.backing.as_deref(), Some(BlockdevRefOrNull::Null(()))));
    // Key order is the QDict's, as in QEMU, so compare values.
    assert_eq!(json::from_str(&output(&mut opts)).unwrap(), json::from_str(text).unwrap());
}

#[test]
fn alternate_picks_by_json_type() {
    let r: BlockdevRef = input(r#""node0""#).unwrap();
    assert_eq!(r, BlockdevRef::Reference("node0".into()));
    assert_eq!(
        err::<BlockdevRef>("42"),
        "Invalid parameter type for 'null', expected: BlockdevRef"
    );
    let e = err::<BlockdevOptions>(r#"{"driver": "raw", "file": 1}"#);
    assert_eq!(e, "Invalid parameter type for 'file', expected: BlockdevRef");
}

#[test]
fn struct_errors_match_qemu() {
    assert_eq!(err::<BlockdevOptions>("{}"), "Parameter 'driver' is missing");
    assert_eq!(
        err::<BlockdevOptions>(r#"{"driver": "nope"}"#),
        "Parameter 'driver' does not accept value 'nope'"
    );
    assert_eq!(
        err::<BlockdevOptions>(r#"{"driver": "null-co", "bogus": 1}"#),
        "Parameter 'bogus' is unexpected"
    );
    assert_eq!(err::<BlockdevOptions>(r#"{"driver": "file"}"#), "Parameter 'filename' is missing");
    assert_eq!(
        err::<BlockdevOptions>(r#"{"driver": "null-co", "read-only": "yes"}"#),
        "Invalid parameter type for 'read-only', expected: boolean"
    );
}

#[test]
fn failed_input_leaves_default() {
    let mut v = QObjectInputVisitor::new_qmp(
        json::from_str(r#"{"driver": "file", "filename": 3}"#).unwrap(),
        CompatPolicy::default(),
    );
    let mut obj = BlockdevOptions { node_name: Some("x".into()), ..Default::default() };
    assert!(BlockdevOptions::visit(&mut v, None, &mut obj).is_err());
    assert_eq!(obj, BlockdevOptions::default());
}

#[test]
fn lists_and_optional_members() {
    let text =
        r#"[{"name": "a", "type": "child<x>"}, {"name": "b", "type": "str", "description": "d"}]"#;
    let mut list: Vec<ObjectPropertyInfo> = Vec::new();
    let mut v = QObjectInputVisitor::new(json::from_str(text).unwrap());
    use ruvm_qapi::visit::VisitorExt;
    v.visit_list(None, &mut list, |v, e| ObjectPropertyInfo::visit(v, None, e)).unwrap();
    assert_eq!(list.len(), 2);
    assert_eq!(list[1].description.as_deref(), Some("d"));
    assert_eq!(list[0].description, None);
    let mut out = QObjectOutputVisitor::new();
    out.visit_list(None, &mut list, |v, e| ObjectPropertyInfo::visit(v, None, e)).unwrap();
    assert_eq!(out.complete(), json::from_str(text).unwrap());
}

#[test]
fn deprecated_members_follow_policy() {
    let reject = CompatPolicy { deprecated_input: CompatPolicyInput::Reject, ..Default::default() };
    let mut v =
        QObjectInputVisitor::new_qmp(json::from_str(r#"{"device": "ide0"}"#).unwrap(), reject);
    let mut obj = EjectArg::default();
    let e = ruvm_qapi::visit::Visitor::start_struct(&mut v, None)
        .and_then(|()| EjectArg::visit_members(&mut v, &mut obj))
        .unwrap_err();
    assert_eq!(e.message(), "Deprecated parameter device disabled by policy");

    let hide = CompatPolicy { deprecated_output: CompatPolicyOutput::Hide, ..Default::default() };
    let mut obj = EjectArg { device: Some("ide0".into()), id: Some("cd".into()), force: None };
    let mut v = QObjectOutputVisitor::new_qmp(hide);
    EjectArg::visit(&mut v, None, &mut obj).unwrap();
    assert_eq!(v.complete().to_json(), r#"{"id": "cd"}"#);
}

#[test]
fn any_members_keep_the_value() {
    let mut obj: QomSetArg = input(r#"{"path": "/", "property": "p", "value": [1, "x"]}"#).unwrap();
    assert_eq!(obj.value, json::from_str(r#"[1, "x"]"#).unwrap());
    let want = json::from_str(r#"{"path": "/", "property": "p", "value": [1, "x"]}"#).unwrap();
    assert_eq!(json::from_str(&output(&mut obj)).unwrap(), want);
    assert_eq!(QValue::default(), QValue::Null);
}
