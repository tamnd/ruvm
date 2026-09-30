// SPDX-License-Identifier: GPL-2.0-or-later

//! The cases of QEMU's tests/unit/test-keyval.c.
//!
//! QEMU's test checks only that an error is set. The error texts are checked here as well, taken
//! from running the same inputs through QEMU 11.1. The alternate test uses types from the real
//! schema in place of the ones in tests/qapi-schema/qapi-schema-test.json, which are not
//! generated here: `BlockdevRef` is a string or an object like `AltStrObj`, `Qcow2OverlapChecks`
//! has an enum branch like `AltNumEnum` and `AltEnumBool`, and `StatsValue` has only non-string
//! scalars.

use ruvm_base::Result;
use ruvm_qapi::keyval::{keyval_merge, keyval_parse};
use ruvm_qapi::types::{BlockdevRef, Qcow2OverlapChecks, StatsValue};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit, Visitor};
use ruvm_qapi::{QDict, QValue};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
const TIB: u64 = 1 << 40;

fn err(params: &str, implied_key: Option<&str>) -> String {
    keyval_parse(params, implied_key, None).unwrap_err().message().to_string()
}

fn sub<'a>(d: &'a QDict, key: &str) -> &'a QDict {
    d.get(key).and_then(QValue::as_dict).unwrap()
}

fn visitor(params: &str) -> QObjectInputVisitor {
    QObjectInputVisitor::new_keyval(QValue::Dict(keyval_parse(params, None, None).unwrap()))
}

fn msg<T: std::fmt::Debug>(r: Result<T>) -> String {
    r.unwrap_err().message().to_string()
}

#[test]
fn keyval_parse_general() {
    // Nothing
    assert_eq!(keyval_parse("", None, None).unwrap().len(), 0);

    // Empty key (qemu_opts_parse() accepts this)
    assert_eq!(err("=val", None), "Invalid parameter ''");

    // Empty key fragment
    assert_eq!(err(".", None), "Invalid parameter '.'");
    assert_eq!(err("key.", None), "Invalid parameter 'key.'");

    // Invalid non-empty key (qemu_opts_parse() doesn't care)
    assert_eq!(err("7up=val", None), "Invalid parameter '7up'");

    // Overlong key
    let long_key = format!("{}z", "a".repeat(127));
    let params = format!("k.{long_key}=v");
    assert_eq!(err(&params[2..], None), format!("Parameter '{long_key}' is too long"));

    // Overlong key fragment
    assert_eq!(err(&params, None), format!("Parameter fragment '{long_key}' is too long"));

    // Long key (qemu_opts_parse() accepts and truncates silently)
    let params = format!("k.{}=v", &long_key[1..]);
    let qdict = keyval_parse(&params[2..], None, None).unwrap();
    assert_eq!(qdict.len(), 1);
    assert_eq!(qdict.get_str(&long_key[1..]), Some("v"));

    // Long key fragment
    let qdict = keyval_parse(&params, None, None).unwrap();
    assert_eq!(qdict.len(), 1);
    let sub_qdict = sub(&qdict, "k");
    assert_eq!(sub_qdict.len(), 1);
    assert_eq!(sub_qdict.get_str(&long_key[1..]), Some("v"));

    // Crap after valid key
    assert_eq!(err("key[0]=val", None), "Invalid parameter 'key[0]'");

    // Multiple keys, last one wins
    let qdict = keyval_parse("a=1,b=2,,x,a=3", None, None).unwrap();
    assert_eq!(qdict.len(), 2);
    assert_eq!(qdict.get_str("a"), Some("3"));
    assert_eq!(qdict.get_str("b"), Some("2,x"));

    // Even when it doesn't in qemu_opts_parse()
    let qdict = keyval_parse("id=foo,id=bar", None, None).unwrap();
    assert_eq!(qdict.len(), 1);
    assert_eq!(qdict.get_str("id"), Some("bar"));

    // Dotted keys
    let qdict = keyval_parse("a.b.c=1,a.b.c=2,d=3", None, None).unwrap();
    assert_eq!(qdict.len(), 2);
    let sub_qdict = sub(&qdict, "a");
    assert_eq!(sub_qdict.len(), 1);
    let sub_qdict = sub(sub_qdict, "b");
    assert_eq!(sub_qdict.len(), 1);
    assert_eq!(sub_qdict.get_str("c"), Some("2"));
    assert_eq!(qdict.get_str("d"), Some("3"));

    // Inconsistent dotted keys
    assert_eq!(err("a.b=1,a=2", None), "Parameters 'a.*' used inconsistently");
    assert_eq!(err("a.b=1,a.b.c=2", None), "Parameters 'a.b.*' used inconsistently");

    // Trailing comma is ignored
    let qdict = keyval_parse("x=y,", None, None).unwrap();
    assert_eq!(qdict.len(), 1);
    assert_eq!(qdict.get_str("x"), Some("y"));

    // Except when it isn't
    assert_eq!(err(",", None), "Invalid parameter ''");

    // Value containing ,id= not misinterpreted as qemu_opts_parse() does
    let qdict = keyval_parse("x=,,id=bar", None, None).unwrap();
    assert_eq!(qdict.len(), 1);
    assert_eq!(qdict.get_str("x"), Some(",id=bar"));

    // Anti-social ID is left to caller (qemu_opts_parse() rejects it)
    let qdict = keyval_parse("id=666", None, None).unwrap();
    assert_eq!(qdict.len(), 1);
    assert_eq!(qdict.get_str("id"), Some("666"));

    // Implied value not supported (unlike qemu_opts_parse())
    assert_eq!(err("an,noaus,noaus=", None), "Expected '=' after parameter 'an'");

    // Implied value, key "no" (qemu_opts_parse(): negated empty key)
    assert_eq!(err("no", None), "Expected '=' after parameter 'no'");

    // Implied key
    let qdict = keyval_parse("an,aus=off,noaus=", Some("implied"), None).unwrap();
    assert_eq!(qdict.len(), 3);
    assert_eq!(qdict.get_str("implied"), Some("an"));
    assert_eq!(qdict.get_str("aus"), Some("off"));
    assert_eq!(qdict.get_str("noaus"), Some(""));

    // Implied dotted key
    let qdict = keyval_parse("val", Some("eins.zwei"), None).unwrap();
    assert_eq!(qdict.len(), 1);
    let sub_qdict = sub(&qdict, "eins");
    assert_eq!(sub_qdict.len(), 1);
    assert_eq!(sub_qdict.get_str("zwei"), Some("val"));

    // Implied key with empty value (qemu_opts_parse() accepts this)
    assert_eq!(err(",", Some("implied")), "Invalid parameter ''");

    // Likewise (qemu_opts_parse(): implied key with comma value)
    assert_eq!(err(",,,a=1", Some("implied")), "Invalid parameter ''");

    // Implied key's value can't have comma (qemu_opts_parse(): it can)
    assert_eq!(err("val,,ue", Some("implied")), "Invalid parameter ''");

    // Empty key is not an implied key
    assert_eq!(err("=val", Some("implied")), "Invalid parameter ''");

    // "help" by itself, without implied key
    let mut help = false;
    let qdict = keyval_parse("help", None, Some(&mut help)).unwrap();
    assert_eq!(qdict.len(), 0);
    assert!(help);

    // "help" by itself, with implied key
    let mut help = false;
    let qdict = keyval_parse("help", Some("implied"), Some(&mut help)).unwrap();
    assert_eq!(qdict.len(), 0);
    assert!(help);

    // "help" when no help is available, without implied key
    assert_eq!(err("help", None), "Help is not available for this option");

    // "help" when no help is available, with implied key
    assert_eq!(err("help", Some("implied")), "Help is not available for this option");

    // Key "help"
    let mut help = true;
    let qdict = keyval_parse("help=on", None, Some(&mut help)).unwrap();
    assert_eq!(qdict.len(), 1);
    assert_eq!(qdict.get_str("help"), Some("on"));
    assert!(!help);

    // "help" followed by crap, without implied key
    let mut help = false;
    let e = keyval_parse("help.abc", None, Some(&mut help)).unwrap_err();
    assert_eq!(e.message(), "Expected '=' after parameter 'help.abc'");

    // "help" followed by crap, with implied key
    let mut help = true;
    let qdict = keyval_parse("help.abc", Some("implied"), Some(&mut help)).unwrap();
    assert_eq!(qdict.len(), 1);
    assert_eq!(qdict.get_str("implied"), Some("help.abc"));
    assert!(!help);

    // "help" with other stuff, without implied key
    let mut help = false;
    let qdict = keyval_parse("number=42,help,foo=bar", None, Some(&mut help)).unwrap();
    assert_eq!(qdict.len(), 2);
    assert_eq!(qdict.get_str("number"), Some("42"));
    assert_eq!(qdict.get_str("foo"), Some("bar"));
    assert!(help);

    // "help" with other stuff, with implied key
    let mut help = false;
    let qdict = keyval_parse("val,help,foo=bar", Some("implied"), Some(&mut help)).unwrap();
    assert_eq!(qdict.len(), 2);
    assert_eq!(qdict.get_str("implied"), Some("val"));
    assert_eq!(qdict.get_str("foo"), Some("bar"));
    assert!(help);
}

fn check_list012(qlist: Option<&QValue>) {
    let expected = [QValue::str("null"), QValue::str("eins"), QValue::str("zwei")];
    assert_eq!(qlist.and_then(QValue::as_list), Some(&expected[..]));
}

#[test]
fn keyval_parse_list() {
    // Root can't be a list
    assert_eq!(err("0=1", None), "Invalid parameter '0'");

    // List elements need not be in order
    let qdict = keyval_parse("list.0=null,list.2=zwei,list.1=eins", None, None).unwrap();
    assert_eq!(qdict.len(), 1);
    check_list012(qdict.get("list"));

    // Multiple indexes, last one wins
    let qdict =
        keyval_parse("list.1=goner,list.0=null,list.01=eins,list.2=zwei", None, None).unwrap();
    assert_eq!(qdict.len(), 1);
    check_list012(qdict.get("list"));

    // List at deeper nesting
    let qdict = keyval_parse("a.list.1=eins,a.list.00=null,a.list.2=zwei", None, None).unwrap();
    assert_eq!(qdict.len(), 1);
    let sub_qdict = sub(&qdict, "a");
    assert_eq!(sub_qdict.len(), 1);
    check_list012(sub_qdict.get("list"));

    // Inconsistent dotted keys: both list and dictionary
    assert_eq!(err("a.b.c=1,a.b.0=2", None), "Parameters 'a.b.*' used inconsistently");
    assert_eq!(err("a.0.c=1,a.b.c=2", None), "Parameters 'a.*' used inconsistently");

    // Missing list indexes
    assert_eq!(err("list.1=lonely", None), "Parameter 'list.0' missing");
    assert_eq!(err("list.0=null,list.2=eins,list.02=zwei", None), "Parameter 'list.1' missing");
}

#[test]
fn keyval_visit_bool() {
    let mut v = visitor("bool1=on,bool2=off");
    let mut b = false;
    v.start_struct(None).unwrap();
    v.type_bool(Some("bool1"), &mut b).unwrap();
    assert!(b);
    v.type_bool(Some("bool2"), &mut b).unwrap();
    assert!(!b);
    v.check_struct().unwrap();
    v.end_struct();

    let mut v = visitor("bool1=offer");
    v.start_struct(None).unwrap();
    assert_eq!(msg(v.type_bool(Some("bool1"), &mut b)), "Parameter 'bool1' expects 'on' or 'off'");
    v.end_struct();
}

#[test]
fn keyval_visit_number() {
    let mut u = 0u64;

    // Lower limit zero
    let mut v = visitor("number1=0");
    v.start_struct(None).unwrap();
    v.type_uint64(Some("number1"), &mut u).unwrap();
    assert_eq!(u, 0);
    v.check_struct().unwrap();
    v.end_struct();

    // Upper limit 2^64-1
    let mut v = visitor("number1=18446744073709551615,number2=-1");
    v.start_struct(None).unwrap();
    v.type_uint64(Some("number1"), &mut u).unwrap();
    assert_eq!(u, u64::MAX);
    v.type_uint64(Some("number2"), &mut u).unwrap();
    assert_eq!(u, u64::MAX);
    v.check_struct().unwrap();
    v.end_struct();

    // Above upper limit
    let mut v = visitor("number1=18446744073709551616");
    v.start_struct(None).unwrap();
    assert_eq!(msg(v.type_uint64(Some("number1"), &mut u)), "Parameter 'number1' expects integer");
    v.end_struct();

    // Below lower limit
    let mut v = visitor("number1=-18446744073709551616");
    v.start_struct(None).unwrap();
    assert!(v.type_uint64(Some("number1"), &mut u).is_err());
    v.end_struct();

    // Hex and octal
    let mut v = visitor("number1=0x2a,number2=052");
    v.start_struct(None).unwrap();
    v.type_uint64(Some("number1"), &mut u).unwrap();
    assert_eq!(u, 42);
    v.type_uint64(Some("number2"), &mut u).unwrap();
    assert_eq!(u, 42);
    v.check_struct().unwrap();
    v.end_struct();

    // Trailing crap
    let mut v = visitor("number1=3.14,number2=08");
    v.start_struct(None).unwrap();
    assert!(v.type_uint64(Some("number1"), &mut u).is_err());
    assert!(v.type_uint64(Some("number2"), &mut u).is_err());
    v.end_struct();
}

#[test]
fn keyval_visit_size() {
    let mut sz = 0u64;

    // Lower limit zero
    let mut v = visitor("sz1=0");
    v.start_struct(None).unwrap();
    v.type_size(Some("sz1"), &mut sz).unwrap();
    assert_eq!(sz, 0);
    v.check_struct().unwrap();
    v.end_struct();

    // Note: full 64 bits of precision

    // Around double limit of precision: 2^53-1, 2^53, 2^53+1
    let mut v = visitor("sz1=9007199254740991,sz2=9007199254740992,sz3=9007199254740993");
    v.start_struct(None).unwrap();
    v.type_size(Some("sz1"), &mut sz).unwrap();
    assert_eq!(sz, 0x1fffffffffffff);
    v.type_size(Some("sz2"), &mut sz).unwrap();
    assert_eq!(sz, 0x20000000000000);
    v.type_size(Some("sz3"), &mut sz).unwrap();
    assert_eq!(sz, 0x20000000000001);
    v.check_struct().unwrap();
    v.end_struct();

    // Close to signed integer limit 2^63
    let mut v = visitor("sz1=9223372036854775807,sz2=9223372036854775808,sz3=9223372036854775809");
    v.start_struct(None).unwrap();
    v.type_size(Some("sz1"), &mut sz).unwrap();
    assert_eq!(sz, 0x7fffffffffffffff);
    v.type_size(Some("sz2"), &mut sz).unwrap();
    assert_eq!(sz, 0x8000000000000000);
    v.type_size(Some("sz3"), &mut sz).unwrap();
    assert_eq!(sz, 0x8000000000000001);
    v.check_struct().unwrap();
    v.end_struct();

    // Close to actual upper limit 0xfffffffffffff800 (53 msbs set)
    let mut v = visitor("sz1=18446744073709549568,sz2=18446744073709550591");
    v.start_struct(None).unwrap();
    v.type_size(Some("sz1"), &mut sz).unwrap();
    assert_eq!(sz, 0xfffffffffffff800);
    v.type_size(Some("sz2"), &mut sz).unwrap();
    assert_eq!(sz, 0xfffffffffffffbff);
    v.check_struct().unwrap();
    v.end_struct();

    // Actual limit 2^64-1
    let mut v = visitor("sz1=18446744073709551615");
    v.start_struct(None).unwrap();
    v.type_size(Some("sz1"), &mut sz).unwrap();
    assert_eq!(sz, 0xffffffffffffffff);
    v.check_struct().unwrap();
    v.end_struct();

    // Beyond limits
    let mut v = visitor("sz1=-1,sz2=18446744073709551616");
    v.start_struct(None).unwrap();
    assert_eq!(msg(v.type_size(Some("sz1"), &mut sz)), "Parameter 'sz1' expects size");
    assert_eq!(msg(v.type_size(Some("sz2"), &mut sz)), "Parameter 'sz2' expects size");
    v.end_struct();

    // Suffixes
    let mut v = visitor("sz1=8b,sz2=1.5k,sz3=2M,sz4=0.1G,sz5=16777215T");
    v.start_struct(None).unwrap();
    v.type_size(Some("sz1"), &mut sz).unwrap();
    assert_eq!(sz, 8);
    v.type_size(Some("sz2"), &mut sz).unwrap();
    assert_eq!(sz, 1536);
    v.type_size(Some("sz3"), &mut sz).unwrap();
    assert_eq!(sz, 2 * MIB);
    v.type_size(Some("sz4"), &mut sz).unwrap();
    assert_eq!(sz, GIB / 10);
    v.type_size(Some("sz5"), &mut sz).unwrap();
    assert_eq!(sz, 16777215 * TIB);
    v.check_struct().unwrap();
    v.end_struct();

    // Beyond limit with suffix
    let mut v = visitor("sz1=16777216T");
    v.start_struct(None).unwrap();
    assert!(v.type_size(Some("sz1"), &mut sz).is_err());
    v.end_struct();

    // Trailing crap
    let mut v = visitor("sz1=0Z,sz2=16Gi");
    v.start_struct(None).unwrap();
    assert!(v.type_size(Some("sz1"), &mut sz).is_err());
    assert!(v.type_size(Some("sz2"), &mut sz).is_err());
    v.end_struct();
}

#[test]
fn keyval_visit_dict() {
    let mut i = 0i64;

    let mut v = visitor("a.b.c=1,a.b.c=2,d=3");
    v.start_struct(None).unwrap();
    v.start_struct(Some("a")).unwrap();
    v.start_struct(Some("b")).unwrap();
    v.type_int64(Some("c"), &mut i).unwrap();
    assert_eq!(i, 2);
    v.check_struct().unwrap();
    v.end_struct();
    v.check_struct().unwrap();
    v.end_struct();
    v.type_int64(Some("d"), &mut i).unwrap();
    assert_eq!(i, 3);
    v.check_struct().unwrap();
    v.end_struct();

    let mut v = visitor("a.b=");
    v.start_struct(None).unwrap();
    v.start_struct(Some("a")).unwrap();
    // a.c missing
    assert_eq!(msg(v.type_int64(Some("c"), &mut i)), "Parameter 'a.c' is missing");
    // a.b unexpected
    assert_eq!(msg(v.check_struct()), "Parameter 'a.b' is unexpected");
    v.end_struct();
    v.check_struct().unwrap();
    v.end_struct();
}

#[test]
fn keyval_visit_list() {
    let mut s = String::new();

    let mut v = visitor("a.0=,a.1=I,a.2.0=II");
    // TODO empty list
    v.start_struct(None).unwrap();
    v.start_list(Some("a"), 0).unwrap();
    v.type_str(None, &mut s).unwrap();
    assert_eq!(s, "");
    v.type_str(None, &mut s).unwrap();
    assert_eq!(s, "I");
    v.start_list(None, 0).unwrap();
    v.type_str(None, &mut s).unwrap();
    assert_eq!(s, "II");
    v.check_list().unwrap();
    v.end_list();
    v.check_list().unwrap();
    v.end_list();
    v.check_struct().unwrap();
    v.end_struct();

    let mut v = visitor("a.0=,b.0.0=head");
    v.start_struct(None).unwrap();
    v.start_list(Some("a"), 0).unwrap();
    // a[0] unexpected
    assert_eq!(msg(v.check_list()), "Only 0 list elements expected in a");
    v.end_list();
    v.start_list(Some("b"), 0).unwrap();
    v.start_list(None, 0).unwrap();
    v.type_str(None, &mut s).unwrap();
    assert_eq!(s, "head");
    // b[0][1] missing
    assert_eq!(msg(v.type_str(None, &mut s)), "Parameter 'b.0.1' is missing");
    v.end_list();
    v.end_list();
    v.check_struct().unwrap();
    v.end_struct();
}

#[test]
fn keyval_visit_optional() {
    let mut i = 0i64;
    let mut v = visitor("a.b=1");
    v.start_struct(None).unwrap();
    assert!(!v.optional(Some("b"), false)); // b missing
    assert!(v.optional(Some("a"), false)); // a present
    v.start_struct(Some("a")).unwrap();
    assert!(v.optional(Some("b"), false)); // a.b present
    v.type_int64(Some("b"), &mut i).unwrap();
    assert_eq!(i, 1);
    assert!(!v.optional(Some("a"), false)); // a.a missing
    v.check_struct().unwrap();
    v.end_struct();
    v.check_struct().unwrap();
    v.end_struct();
}

#[test]
fn keyval_visit_alternate() {
    // Can't do scalar alternate variants other than string. You get the string variant if there
    // is one, else an error.
    let mut v = visitor("a=1,b=2,c=on");
    v.start_struct(None).unwrap();
    let mut aso = BlockdevRef::default();
    BlockdevRef::visit(&mut v, Some("a"), &mut aso).unwrap();
    assert_eq!(aso, BlockdevRef::Reference("1".into()));
    let mut ane = Qcow2OverlapChecks::default();
    assert_eq!(
        msg(Qcow2OverlapChecks::visit(&mut v, Some("b"), &mut ane)),
        "Parameter 'b' does not accept value '2'"
    );
    let mut aeb = StatsValue::default();
    assert_eq!(
        msg(StatsValue::visit(&mut v, Some("c"), &mut aeb)),
        "Invalid parameter type for 'c', expected: StatsValue"
    );
    v.end_struct();
}

#[test]
fn keyval_visit_any() {
    let mut v = visitor("a.0=null,a.1=1");
    v.start_struct(None).unwrap();
    let mut any = QValue::Null;
    v.type_any(Some("a"), &mut any).unwrap();
    assert_eq!(any, QValue::List(vec![QValue::str("null"), QValue::str("1")]));
    v.check_struct().unwrap();
    v.end_struct();
}

#[test]
fn keyval_merge_dict() {
    let mut first =
        keyval_parse("opt1=abc,opt2.sub1=def,opt2.sub2=ghi,opt3=xyz", None, None).unwrap();
    let second = keyval_parse("opt1=ABC,opt2.sub2=GHI,opt2.sub3=JKL", None, None).unwrap();
    let combined =
        keyval_parse("opt1=ABC,opt2.sub1=def,opt2.sub2=GHI,opt2.sub3=JKL,opt3=xyz", None, None)
            .unwrap();
    keyval_merge(&mut first, &second).unwrap();
    assert_eq!(combined, first);
}

#[test]
fn keyval_merge_list() {
    let mut first = keyval_parse("opt1.0=abc,opt2.0=xyz", None, None).unwrap();
    let second = keyval_parse("opt1.0=def", None, None).unwrap();
    let combined = keyval_parse("opt1.0=abc,opt1.1=def,opt2.0=xyz", None, None).unwrap();
    keyval_merge(&mut first, &second).unwrap();
    assert_eq!(combined, first);
}

#[test]
fn keyval_merge_conflict() {
    let mut first = keyval_parse("opt2=ABC", None, None).unwrap();
    let mut second = keyval_parse("opt2.sub1=def,opt2.sub2=ghi", None, None).unwrap();
    let third = first.clone();
    assert_eq!(msg(keyval_merge(&mut first, &second)), "Parameter 'opt2' used inconsistently");
    assert_eq!(msg(keyval_merge(&mut second, &third)), "Parameter 'opt2' used inconsistently");
}
