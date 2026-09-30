// SPDX-License-Identifier: GPL-2.0-or-later

//! Cases from tests/unit/test-qobject-input-visitor.c, test-qobject-output-visitor.c,
//! test-string-input-visitor.c, test-string-output-visitor.c and test-forward-visitor.c.

use super::*;
use crate::json;
use crate::qvalue::QDict;

fn qin(s: &str) -> QObjectInputVisitor {
    QObjectInputVisitor::new(json::from_str(s).unwrap())
}

fn kv(pairs: &[(&str, QValue)]) -> QObjectInputVisitor {
    let mut d = QDict::new();
    for (k, v) in pairs {
        d.put(*k, v.clone());
    }
    QObjectInputVisitor::new_keyval(QValue::Dict(d))
}

fn msg<T: std::fmt::Debug>(r: Result<T>) -> String {
    r.unwrap_err().message().to_string()
}

#[derive(Debug, Default, PartialEq)]
struct UserDefOne {
    integer: i64,
    string: String,
    enum1: Option<usize>,
}

const ENUM_ONE: QEnumLookup = QEnumLookup::new(&["value1", "value2", "value3", "value4"]);

fn visit_user_def_one(v: &mut dyn Visitor, name: Option<&str>, obj: &mut UserDefOne) -> Result<()> {
    v.start_struct(name)?;
    let r = (|| {
        v.type_int64(Some("integer"), &mut obj.integer)?;
        v.type_str(Some("string"), &mut obj.string)?;
        let mut present = obj.enum1.is_some();
        if v.optional(Some("enum1"), present) {
            let mut e = obj.enum1.unwrap_or(0);
            v.type_enum(Some("enum1"), &mut e, &ENUM_ONE)?;
            obj.enum1 = Some(e);
            present = true;
        }
        let _ = present;
        v.check_struct()
    })();
    v.end_struct();
    r
}

#[test]
fn qobject_input_scalars() {
    let mut i = 0i64;
    qin("-42").type_int64(None, &mut i).unwrap();
    assert_eq!(i, -42);
    let mut u = 0u64;
    qin("-42").type_uint64(None, &mut u).unwrap();
    assert_eq!(u, (-42i64) as u64);
    qin("18446744073709551615").type_uint64(None, &mut u).unwrap();
    assert_eq!(u, u64::MAX);
    assert_eq!(
        msg(qin("18446744073709551615").type_int64(None, &mut i)),
        "Invalid parameter type for '<anonymous>', expected: integer"
    );
    assert_eq!(msg(qin("1.5").type_uint64(None, &mut u)), "Parameter '<anonymous>' expects uint64");
    assert_eq!(
        msg(qin("true").type_str(None, &mut String::new())),
        "Invalid parameter type for '<anonymous>', expected: string"
    );
    let mut n = 0.0;
    qin("3").type_number(None, &mut n).unwrap();
    assert_eq!(n, 3.0);
    let mut b = false;
    qin("true").type_bool(None, &mut b).unwrap();
    assert!(b);
    qin("null").type_null(None).unwrap();
    assert_eq!(
        msg(qin("1").type_null(None)),
        "Invalid parameter type for '<anonymous>', expected: null"
    );
    let mut i8v = 0i8;
    assert_eq!(msg(qin("128").type_int8(None, &mut i8v)), "Parameter 'null' expects int8_t");
    let mut u32v = 0u32;
    assert_eq!(msg(qin("-1").type_uint32(Some("x"), &mut u32v)), "Parameter 'x' expects uint32_t");
}

#[test]
fn qobject_input_structs_and_names() {
    let mut v = qin(r#"{"integer": -42, "string": "Hi!", "enum1": "value3"}"#);
    let mut o = UserDefOne::default();
    visit_user_def_one(&mut v, None, &mut o).unwrap();
    assert_eq!(o, UserDefOne { integer: -42, string: "Hi!".into(), enum1: Some(2) });

    let mut v = qin(r#"{"integer": -42, "string": "Hi!", "enum1": "value9"}"#);
    assert_eq!(
        msg(visit_user_def_one(&mut v, None, &mut o)),
        "Parameter 'enum1' does not accept value 'value9'"
    );

    let mut v = qin(r#"{"integer": -42}"#);
    assert_eq!(msg(visit_user_def_one(&mut v, None, &mut o)), "Parameter 'string' is missing");

    let mut v = qin(r#"{"integer": "x", "string": "Hi!"}"#);
    assert_eq!(
        msg(visit_user_def_one(&mut v, Some("arg"), &mut o)),
        "Invalid parameter type for 'arg.integer', expected: integer"
    );

    let mut v = qin(r#"{"integer": 1, "string": "Hi!", "extra": 1, "more": 2}"#);
    let err = msg(visit_user_def_one(&mut v, None, &mut o));
    assert!(err == "Parameter 'extra' is unexpected" || err == "Parameter 'more' is unexpected");

    let mut v = qin(r#"[1, 2]"#);
    assert_eq!(
        msg(visit_user_def_one(&mut v, None, &mut o)),
        "Invalid parameter type for '<anonymous>', expected: object"
    );

    // Nested names: a struct inside a list inside a struct.
    let mut v = qin(r#"{"a": [{"integer": 1, "string": "x"}, {"integer": 2}]}"#);
    v.start_struct(None).unwrap();
    let mut list: Vec<UserDefOne> = Vec::new();
    let e = v.visit_list(Some("a"), &mut list, |v, o| visit_user_def_one(v, None, o));
    assert_eq!(msg(e), "Parameter 'a[1].string' is missing");
    v.end_struct();
}

#[test]
fn qobject_input_lists() {
    let mut v = qin("[1, 2, 3]");
    let mut l: Vec<i64> = Vec::new();
    v.visit_list(None, &mut l, |v, x| v.type_int64(None, x)).unwrap();
    assert_eq!(l, [1, 2, 3]);

    let mut v = qin("[1, 2, 3]");
    v.start_list(None, 0).unwrap();
    let mut x = 0;
    v.type_int64(None, &mut x).unwrap();
    v.type_int64(None, &mut x).unwrap();
    assert_eq!(msg(v.check_list()), "Only 2 list elements expected in <anonymous>");
    v.end_list();

    let mut v = qin(r#"{"list": [1, "x"]}"#);
    v.start_struct(None).unwrap();
    let e = v.visit_list(Some("list"), &mut l, |v, x| v.type_int64(None, x));
    assert_eq!(msg(e), "Invalid parameter type for 'list[1]', expected: integer");
    v.end_struct();

    let mut v = qin(r#"{"list": 1}"#);
    v.start_struct(None).unwrap();
    let e = v.visit_list(Some("list"), &mut l, |v, x| v.type_int64(None, x));
    assert_eq!(msg(e), "Invalid parameter type for 'list', expected: array");
    v.end_struct();

    let mut v = qin("[]");
    v.visit_list(None, &mut l, |v, x| v.type_int64(None, x)).unwrap();
    assert!(l.is_empty());
}

#[test]
fn qobject_input_keyval() {
    let mut v = kv(&[
        ("i", "0x10".into()),
        ("u", "-1".into()),
        ("b", "on".into()),
        ("sz", "2k".into()),
        ("n", "1.5".into()),
        ("d", QValue::Dict(QDict::new().with("x", "1"))),
    ]);
    v.start_struct(None).unwrap();
    let (mut i, mut u, mut b, mut sz, mut n) = (0i64, 0u64, false, 0u64, 0f64);
    v.type_int64(Some("i"), &mut i).unwrap();
    v.type_uint64(Some("u"), &mut u).unwrap();
    v.type_bool(Some("b"), &mut b).unwrap();
    v.type_size(Some("sz"), &mut sz).unwrap();
    v.type_number(Some("n"), &mut n).unwrap();
    assert_eq!((i, u, b, sz, n), (16, u64::MAX, true, 2048, 1.5));
    assert_eq!(msg(v.type_str(Some("d"), &mut String::new())), "Parameters 'd.*' are unexpected");
    v.end_struct();

    let mut v = kv(&[("i", "x".into()), ("b", "maybe".into()), ("sz", "1Q".into())]);
    v.start_struct(None).unwrap();
    assert_eq!(msg(v.type_int64(Some("i"), &mut 0)), "Parameter 'i' expects integer");
    assert_eq!(msg(v.type_bool(Some("b"), &mut false)), "Parameter 'b' expects 'on' or 'off'");
    assert_eq!(msg(v.type_size(Some("sz"), &mut 0)), "Parameter 'sz' expects size");
    v.end_struct();

    // keyval names list elements with dots.
    let mut d = QDict::new();
    d.put("l", QValue::List(vec!["1".into(), "x".into()]));
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(d));
    v.start_struct(None).unwrap();
    let mut l: Vec<i64> = Vec::new();
    let e = v.visit_list(Some("l"), &mut l, |v, x| v.type_int64(None, x));
    assert_eq!(msg(e), "Parameter 'l.1' expects integer");
    v.end_struct();
}

#[test]
fn qobject_input_alternate_and_optional() {
    let mut v = qin(r#"{"a": 1}"#);
    v.start_struct(None).unwrap();
    assert!(v.optional(Some("a"), false));
    assert!(!v.optional(Some("b"), true));
    assert_eq!(v.start_alternate(Some("a")).unwrap(), Some(QType::QNum));
    let mut i = 0;
    v.type_int64(Some("a"), &mut i).unwrap();
    v.end_alternate();
    v.check_struct().unwrap();
    v.end_struct();
}

#[test]
fn compat_policy() {
    let policy = CompatPolicy {
        deprecated_input: CompatPolicyInput::Reject,
        unstable_output: CompatPolicyOutput::Hide,
        ..Default::default()
    };
    let mut v = QObjectInputVisitor::new_qmp(json::from_str(r#"{"a": 1}"#).unwrap(), policy);
    v.start_struct(None).unwrap();
    assert_eq!(
        msg(v.policy_reject(Some("a"), QAPI_DEPRECATED)),
        "Deprecated parameter a disabled by policy"
    );
    v.policy_reject(Some("a"), QAPI_UNSTABLE).unwrap();
    v.end_struct();

    static FEATURES: [u64; 2] = [0, QAPI_DEPRECATED];
    let lookup = QEnumLookup { array: &["old", "new"], features: Some(&FEATURES) };
    let mut v = QObjectInputVisitor::new_qmp(json::from_str(r#""new""#).unwrap(), policy);
    assert_eq!(msg(v.type_enum(None, &mut 0, &lookup)), "Deprecated value new disabled by policy");

    let mut o = QObjectOutputVisitor::new_qmp(policy);
    assert!(o.policy_skip(Some("x"), QAPI_UNSTABLE));
    assert!(!o.policy_skip(Some("x"), QAPI_DEPRECATED));
}

#[test]
fn qobject_output_round_trip() {
    let mut o = QObjectOutputVisitor::new();
    let mut x = UserDefOne { integer: -42, string: "Hi!".into(), enum1: Some(1) };
    visit_user_def_one(&mut o, None, &mut x).unwrap();
    let q = o.complete();
    assert_eq!(q.to_json(), r#"{"integer": -42, "enum1": "value2", "string": "Hi!"}"#);

    let mut back = UserDefOne::default();
    visit_user_def_one(&mut QObjectInputVisitor::new(q), None, &mut back).unwrap();
    assert_eq!(back, x);

    let mut o = QObjectOutputVisitor::new();
    let mut l = vec![1u64, u64::MAX];
    o.visit_list(None, &mut l, |v, x| v.type_uint64(None, x)).unwrap();
    assert_eq!(o.complete().to_json(), "[1, 18446744073709551615]");

    let mut o = QObjectOutputVisitor::new();
    o.type_number(None, &mut 0.1).unwrap();
    assert_eq!(o.complete().to_json(), "0.10000000000000001");
}

fn sin(s: &str) -> StringInputVisitor {
    StringInputVisitor::new(s)
}

fn ilist(s: &str) -> Result<Vec<i64>> {
    let mut l = Vec::new();
    sin(s).visit_list(None, &mut l, |v, x| v.type_int64(None, x))?;
    Ok(l)
}

fn ulist(s: &str) -> Result<Vec<u64>> {
    let mut l = Vec::new();
    sin(s).visit_list(None, &mut l, |v, x| v.type_uint64(None, x))?;
    Ok(l)
}

#[test]
fn string_input_scalars() {
    let mut i = 0;
    sin("-42").type_int64(None, &mut i).unwrap();
    assert_eq!(i, -42);
    assert_eq!(msg(sin("not an int").type_int64(None, &mut i)), "Parameter 'null' expects int64");
    assert_eq!(msg(sin("").type_int64(Some("x"), &mut i)), "Parameter 'x' expects int64");
    let mut b = false;
    for (s, want) in [
        ("true", true),
        ("yes", true),
        ("on", true),
        ("false", false),
        ("no", false),
        ("off", false),
    ] {
        sin(s).type_bool(None, &mut b).unwrap();
        assert_eq!(b, want);
    }
    assert_eq!(msg(sin("x").type_bool(Some("b"), &mut b)), "Parameter 'b' expects 'on' or 'off'");
    let mut n = 0.0;
    sin("2.75").type_number(None, &mut n).unwrap();
    assert_eq!(n, 2.75);
    assert_eq!(
        msg(sin("NaN").type_number(None, &mut n)),
        "Invalid parameter type for 'null', expected: number"
    );
    assert!(sin("inf").type_number(None, &mut n).is_err());
    let mut s = String::new();
    sin("Q E M U").type_str(None, &mut s).unwrap();
    assert_eq!(s, "Q E M U");
    let mut e = 0;
    for (idx, name) in ENUM_ONE.array.iter().enumerate() {
        sin(name).type_enum(None, &mut e, &ENUM_ONE).unwrap();
        assert_eq!(e, idx);
    }
    let mut sz = 0;
    sin("1M").type_size(Some("size"), &mut sz).unwrap();
    assert_eq!(sz, 1 << 20);
    let err = sin("abc").type_size(Some("size"), &mut sz).unwrap_err();
    assert_eq!(err.message(), "Parameter 'size' expects a non-negative number below 2^64");
    assert!(err.hint_text().unwrap().starts_with("Optional suffix k, M, G, T, P or E means"));
    assert_eq!(
        msg(sin("16E").type_size(Some("size"), &mut sz)),
        "Value '16E' is out of range for parameter 'size'"
    );
    assert_eq!(
        msg(sin("-1").type_size(Some("size"), &mut sz)),
        "Value '-1' is out of range for parameter 'size'"
    );
    sin("").type_null(None).unwrap();
    assert!(sin("x").type_null(None).is_err());
}

#[test]
fn string_input_int_lists() {
    assert_eq!(
        ilist("1,2,0,2-4,20,5-9,1-8").unwrap(),
        [1, 2, 0, 2, 3, 4, 20, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(ilist("32767,-32768--32767").unwrap(), [32767, -32768, -32767]);
    assert_eq!(ilist("-9223372036854775808,9223372036854775807").unwrap(), [i64::MIN, i64::MAX]);
    assert_eq!(ilist("1-1").unwrap(), [1]);
    assert_eq!(
        ilist("9223372036854775805-9223372036854775807").unwrap(),
        [i64::MAX - 2, i64::MAX - 1, i64::MAX]
    );
    for bad in
        ["9223372036854775808", "-9223372036854775809", "3-1", "9223372036854775807-0", "0-65536"]
    {
        assert!(ilist(bad).is_err(), "{bad}");
    }
    assert_eq!(ilist("").unwrap(), Vec::<i64>::new());
    assert_eq!(
        msg(ilist("not an int list")),
        "Parameter 'null' expects list of int64 values or ranges"
    );

    let mut v = sin("0,2-3");
    let mut x = 0;
    v.start_list(None, 0).unwrap();
    v.type_int64(None, &mut x).unwrap();
    assert_eq!(x, 0);
    v.type_int64(None, &mut x).unwrap();
    assert_eq!(x, 2);
    assert_eq!(msg(v.check_list()), "Fewer list elements expected");
    v.end_list();

    let mut v = sin("0");
    v.start_list(None, 0).unwrap();
    v.type_int64(None, &mut x).unwrap();
    assert_eq!(msg(v.type_int64(None, &mut x)), "Fewer list elements expected");
    v.check_list().unwrap();
    v.end_list();
}

#[test]
fn string_input_uint_lists() {
    assert_eq!(
        ulist("1,2,0,2-4,20,5-9,1-8").unwrap(),
        [1, 2, 0, 2, 3, 4, 20, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(ulist("32767,-32768--32767").unwrap(), [32767, -32768i64 as u64, -32767i64 as u64]);
    assert_eq!(
        ulist("-9223372036854775808,9223372036854775807").unwrap(),
        [i64::MIN as u64, i64::MAX as u64]
    );
    assert_eq!(ulist("18446744073709551615").unwrap(), [u64::MAX]);
    assert_eq!(
        ulist("18446744073709551613-18446744073709551615").unwrap(),
        [u64::MAX - 2, u64::MAX - 1, u64::MAX]
    );
    for bad in [
        "18446744073709551616",
        "-18446744073709551616",
        "3-1",
        "18446744073709551615-0",
        "0-65536",
    ] {
        assert!(ulist(bad).is_err(), "{bad}");
    }
    assert_eq!(
        msg(ulist("not an uint list")),
        "Parameter 'null' expects list of uint64 values or ranges"
    );
}

fn sout(human: bool, f: impl FnOnce(&mut StringOutputVisitor)) -> String {
    let mut v = StringOutputVisitor::new(human);
    f(&mut v);
    v.complete()
}

#[test]
fn string_output() {
    assert_eq!(sout(false, |v| v.type_int64(None, &mut 42).unwrap()), "42");
    assert_eq!(sout(true, |v| v.type_int64(None, &mut 42).unwrap()), "42 (0x2a)");
    assert_eq!(sout(true, |v| v.type_int64(None, &mut -1).unwrap()), "-1 (0xffffffffffffffff)");
    assert_eq!(sout(false, |v| v.type_bool(None, &mut true).unwrap()), "true");
    assert_eq!(
        sout(false, |v| v.type_number(None, &mut { std::f64::consts::PI }).unwrap()),
        "3.1415926535897931"
    );
    assert_eq!(sout(false, |v| v.type_str(None, &mut "Q E M U".into()).unwrap()), "Q E M U");
    assert_eq!(sout(true, |v| v.type_str(None, &mut "Q E M U".into()).unwrap()), "\"Q E M U\"");
    assert_eq!(sout(false, |v| v.type_size(None, &mut 1536).unwrap()), "1536");
    assert_eq!(sout(true, |v| v.type_size(None, &mut 1536).unwrap()), "1536 (1.5 KiB)");
    assert_eq!(sout(true, |v| v.type_null(None).unwrap()), "<null>");
    assert_eq!(sout(false, |v| v.type_null(None).unwrap()), "");
    let mut e = 2;
    assert_eq!(sout(false, |v| v.type_enum(None, &mut e, &ENUM_ONE).unwrap()), "value3");
    assert_eq!(sout(true, |v| v.type_enum(None, &mut e, &ENUM_ONE).unwrap()), "\"value3\"");

    let mut l: Vec<i64> =
        vec![0, 1, 9, 10, 16, 15, 14, 3, 4, 5, 6, 11, 12, 13, 21, 22, i64::MAX - 1, i64::MAX];
    let s = sout(false, |v| v.visit_list(None, &mut l, |v, x| v.type_int64(None, x)).unwrap());
    assert_eq!(s, "0-1,3-6,9-16,21-22,9223372036854775806-9223372036854775807");
    let s = sout(true, |v| v.visit_list(None, &mut l, |v, x| v.type_int64(None, x)).unwrap());
    assert_eq!(
        s,
        "0-1,3-6,9-16,21-22,9223372036854775806-9223372036854775807 \
         (0x0-0x1,0x3-0x6,0x9-0x10,0x15-0x16,0x7ffffffffffffffe-0x7fffffffffffffff)"
    );
    let mut l: Vec<String> = vec!["a".into(), "b".into()];
    let s = sout(false, |v| v.visit_list(None, &mut l, |v, x| v.type_str(None, x)).unwrap());
    assert_eq!(s, "a, b");
    let mut o = UserDefOne::default();
    assert_eq!(sout(false, |v| visit_user_def_one(v, None, &mut o).unwrap()), "<omitted>");
}

#[test]
fn forward_field() {
    let mut src = qin(r#"{"src": 42}"#);
    src.start_struct(None).unwrap();
    {
        let mut f = ForwardFieldVisitor::new(&mut src, "dst", "src");
        let mut x = 0;
        f.type_int64(Some("dst"), &mut x).unwrap();
        assert_eq!(x, 42);
        assert_eq!(msg(f.type_int64(Some("other"), &mut x)), "Parameter 'other' is missing");
    }
    src.check_struct().unwrap();
    src.end_struct();

    let mut src = qin(r#"{"src": {"integer": 1, "string": "s"}}"#);
    src.start_struct(None).unwrap();
    {
        let mut f = ForwardFieldVisitor::new(&mut src, "dst", "src");
        let mut o = UserDefOne::default();
        visit_user_def_one(&mut f, Some("dst"), &mut o).unwrap();
        assert_eq!(o.string, "s");
    }
    src.end_struct();

    let mut out = QObjectOutputVisitor::new();
    out.start_struct(None).unwrap();
    {
        let mut f = ForwardFieldVisitor::new(&mut out, "alias", "target");
        let mut l = vec![1i64, 2];
        f.visit_list(Some("alias"), &mut l, |v, x| v.type_int64(None, x)).unwrap();
    }
    out.end_struct();
    assert_eq!(out.complete().to_json(), r#"{"target": [1, 2]}"#);
}
