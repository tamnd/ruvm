// SPDX-License-Identifier: GPL-2.0-or-later

//! The cases of QEMU's tests/unit/test-qemu-opts.c.
//!
//! QEMU registers the four lists once in `vm_config_groups` and every test finds them there. Rust
//! runs tests in parallel, so each test builds its own [`ConfigGroups`] instead. Where QEMU's test
//! only checks that an error is set, the error text is checked as well. A few tests at the end
//! cover what the C test leaves out: short-form boolean warnings, help requests, `merge_lists` and
//! `-set`.

use ruvm_qapi::QDict;
use ruvm_qapi::opts::{
    ConfigGroups, FlagWarning, ParseFailure, QemuOptDesc, QemuOptType, QemuOpts, QemuOptsList,
    has_help_option,
};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
const TIB: u64 = 1 << 40;

fn opts_list_01() -> QemuOptsList {
    QemuOptsList::new(
        "opts_list_01",
        &[
            QemuOptDesc::new("str1", QemuOptType::String)
                .help("Help texts are preserved in qemu_opts_append")
                .default_value("default"),
            QemuOptDesc::new("str2", QemuOptType::String),
            QemuOptDesc::new("str3", QemuOptType::String),
            QemuOptDesc::new("number1", QemuOptType::Number)
                .help("Having help texts only for some options is okay"),
            QemuOptDesc::new("number2", QemuOptType::Number),
        ],
    )
}

fn opts_list_02() -> QemuOptsList {
    QemuOptsList::new(
        "opts_list_02",
        &[
            QemuOptDesc::new("str1", QemuOptType::String),
            QemuOptDesc::new("str2", QemuOptType::String),
            QemuOptDesc::new("bool1", QemuOptType::Bool),
            QemuOptDesc::new("bool2", QemuOptType::Bool),
            QemuOptDesc::new("size1", QemuOptType::Size),
            QemuOptDesc::new("size2", QemuOptType::Size),
            QemuOptDesc::new("size3", QemuOptType::Size),
        ],
    )
}

fn opts_list_03() -> QemuOptsList {
    QemuOptsList::new("opts_list_03", &[]).with_implied_opt_name("implied")
}

fn opts_list_04() -> QemuOptsList {
    QemuOptsList::new("opts_list_04", &[QemuOptDesc::new("str3", QemuOptType::String)])
        .with_merge_lists()
}

/// `register_opts()`.
fn register_opts() -> ConfigGroups {
    let mut groups = ConfigGroups::new();
    groups.add(opts_list_01());
    groups.add(opts_list_02());
    groups.add(opts_list_03());
    groups.add(opts_list_04());
    groups
}

/// The start every test on a registered list has: the list is found, empty, and has no set
/// without an id.
fn find_empty<'a>(groups: &'a mut ConfigGroups, name: &str) -> &'a mut QemuOptsList {
    let list = groups.find_opts(name).unwrap();
    assert!(list.is_empty());
    assert_eq!(list.name(), Some(name));
    assert!(list.find(None).is_none());
    list
}

/// Deletes the set without an id and checks it is gone.
fn del_and_check(list: &mut QemuOptsList) {
    let handle = list.find(None).unwrap().handle();
    list.del(handle);
    assert!(list.find(None).is_none());
}

fn opts_count(opts: &QemuOpts) -> usize {
    opts.iter().count()
}

fn parse_err(list: &mut QemuOptsList, params: &str, permit_abbrev: bool) -> String {
    list.parse(params, permit_abbrev).unwrap_err().message().to_string()
}

#[test]
fn find_unknown_opts() {
    let mut groups = register_opts();
    // should not return anything, we don't have an "unknown" option
    let e = groups.find_opts_err("unknown").unwrap_err();
    assert_eq!(e.message(), "There is no option group 'unknown'");
}

#[test]
fn find_opts() {
    let mut groups = register_opts();
    // we have an "opts_list_01" option, should return it
    let list = groups.find_opts("opts_list_01").unwrap();
    assert_eq!(list.name(), Some("opts_list_01"));
}

#[test]
fn opts_create() {
    let mut groups = register_opts();
    let list = find_empty(&mut groups, "opts_list_01");

    // create the opts
    list.create(None, false).unwrap();
    assert!(!list.is_empty());

    // now we've create the opts, must find it
    assert!(list.find(None).is_some());

    del_and_check(list);
}

#[test]
fn opt_get() {
    let mut groups = register_opts();
    let list = find_empty(&mut groups, "opts_list_01");
    let opts = list.create(None, false).unwrap();

    // haven't set anything to str2 yet
    assert_eq!(opts.get("str2"), None);

    opts.set("str2", "value").unwrap();

    // now we have set str2, should know about it
    assert_eq!(opts.get("str2"), Some("value"));

    opts.set("str2", "value2").unwrap();

    // having reset the value, the returned should be the reset one
    assert_eq!(opts.get("str2"), Some("value2"));

    del_and_check(list);
}

#[test]
fn opt_get_bool() {
    let mut groups = register_opts();
    let list = find_empty(&mut groups, "opts_list_02");
    let opts = list.create(None, false).unwrap();

    // haven't set anything to bool1 yet, so defval should be returned
    assert!(!opts.get_bool("bool1", false));

    opts.set_bool("bool1", true).unwrap();

    // now we have set bool1, should know about it
    assert!(opts.get_bool("bool1", false));

    // having reset the value, opt should be the reset one not defval
    opts.set_bool("bool1", false).unwrap();
    assert!(!opts.get_bool("bool1", true));

    del_and_check(list);
}

#[test]
fn opt_get_number() {
    let mut groups = register_opts();
    let list = find_empty(&mut groups, "opts_list_01");
    let opts = list.create(None, false).unwrap();

    // haven't set anything to number1 yet, so defval should be returned
    assert_eq!(opts.get_number("number1", 5), 5);

    opts.set_number("number1", 10).unwrap();

    // now we have set number1, should know about it
    assert_eq!(opts.get_number("number1", 5), 10);

    // having reset it, the returned should be the reset one not defval
    opts.set_number("number1", 15).unwrap();
    assert_eq!(opts.get_number("number1", 5), 15);

    del_and_check(list);
}

#[test]
fn opt_get_size() {
    let mut groups = register_opts();
    let list = find_empty(&mut groups, "opts_list_02");
    let opts = list.create(None, false).unwrap();

    // haven't set anything to size1 yet, so defval should be returned
    assert_eq!(opts.get_size("size1", 5), 5);

    let mut dict = QDict::new();
    dict.put("size1", "10");
    opts.absorb_qdict(&mut dict).unwrap();

    // now we have set size1, should know about it
    assert_eq!(opts.get_size("size1", 5), 10);

    // reset value
    dict.put("size1", "15");
    opts.absorb_qdict(&mut dict).unwrap();

    // test the reset value
    assert_eq!(opts.get_size("size1", 5), 15);

    del_and_check(list);
}

#[test]
fn opt_unset() {
    let mut list = opts_list_03();

    // dynamically initialized (parsed) opts
    let opts = list.parse("key=value", false).unwrap();

    // check default/parsed value
    assert_eq!(opts.get("key"), Some("value"));

    // reset it to value2
    opts.set("key", "value2").unwrap();
    assert_eq!(opts.get("key"), Some("value2"));

    // unset, valid only for "accept any"
    assert!(opts.unset("key"));

    // after reset the value should be the parsed/default one
    assert_eq!(opts.get("key"), Some("value"));

    let handle = opts.handle();
    list.del(handle);
}

#[test]
fn opts_reset() {
    let mut groups = register_opts();
    let list = find_empty(&mut groups, "opts_list_01");
    let opts = list.create(None, false).unwrap();

    // haven't set anything to number1 yet, so defval should be returned
    assert_eq!(opts.get_number("number1", 5), 5);

    opts.set_number("number1", 10).unwrap();

    // now we have set number1, should know about it
    assert_eq!(opts.get_number("number1", 5), 10);

    list.reset();

    // should not find anything at this point
    assert!(list.find(None).is_none());
}

#[test]
fn opts_parse_general() {
    let mut list_01 = opts_list_01();
    let mut list_03 = opts_list_03();

    // Nothing
    let opts = list_03.parse("", false).unwrap();
    assert_eq!(opts_count(opts), 0);

    // Empty key
    let opts = list_03.parse("=val", false).unwrap();
    assert_eq!(opts_count(opts), 1);
    assert_eq!(opts.get(""), Some("val"));

    // Multiple keys, last one wins
    let opts = list_03.parse("a=1,b=2,,x,a=3", false).unwrap();
    assert_eq!(opts_count(opts), 3);
    assert_eq!(opts.get("a"), Some("3"));
    assert_eq!(opts.get("b"), Some("2,x"));

    // Except when it doesn't
    let opts = list_03.parse("id=foo,id=bar", false).unwrap();
    assert_eq!(opts_count(opts), 0);
    assert_eq!(opts.id(), Some("foo"));

    // TODO Cover low-level access to repeated keys

    // Trailing comma is ignored
    let opts = list_03.parse("x=y,", false).unwrap();
    assert_eq!(opts_count(opts), 1);
    assert_eq!(opts.get("x"), Some("y"));

    // Except when it isn't
    let opts = list_03.parse(",", false).unwrap();
    assert_eq!(opts_count(opts), 1);
    assert_eq!(opts.get(""), Some("on"));

    // Duplicate ID
    assert_eq!(parse_err(&mut list_03, "x=y,id=foo", false), "Duplicate ID 'foo' for opts_list_03");
    // TODO Cover .merge_lists = true

    // Buggy ID recognition (fixed)
    let opts = list_03.parse("x=,,id=bar", false).unwrap();
    assert_eq!(opts_count(opts), 1);
    assert_eq!(opts.id(), None);
    assert_eq!(opts.get("x"), Some(",id=bar"));

    // Anti-social ID
    let e = list_01.parse("id=666", false).unwrap_err();
    assert_eq!(e.message(), "Parameter 'id' expects an identifier");
    assert_eq!(
        e.hint_text(),
        Some("Identifiers consist of letters, digits, '-', '.', '_', starting with a letter.\n")
    );

    // Implied value (qemu_opts_parse warns but accepts it)
    let opts = list_03.parse("an,noaus,noaus=", false).unwrap();
    assert_eq!(opts_count(opts), 3);
    assert_eq!(opts.get("an"), Some("on"));
    assert_eq!(opts.get("aus"), Some("off"));
    assert_eq!(opts.get("noaus"), Some(""));

    // Implied value, negated empty key
    let opts = list_03.parse("no", false).unwrap();
    assert_eq!(opts_count(opts), 1);
    assert_eq!(opts.get(""), Some("off"));

    // Implied key
    let opts = list_03.parse("an,noaus,noaus=", true).unwrap();
    assert_eq!(opts_count(opts), 3);
    assert_eq!(opts.get("implied"), Some("an"));
    assert_eq!(opts.get("aus"), Some("off"));
    assert_eq!(opts.get("noaus"), Some(""));

    // Implied key with empty value
    let opts = list_03.parse(",", true).unwrap();
    assert_eq!(opts_count(opts), 1);
    assert_eq!(opts.get("implied"), Some(""));

    // Implied key with comma value
    let opts = list_03.parse(",,,a=1", true).unwrap();
    assert_eq!(opts_count(opts), 2);
    assert_eq!(opts.get("implied"), Some(","));
    assert_eq!(opts.get("a"), Some("1"));

    // Empty key is not an implied key
    let opts = list_03.parse("=val", true).unwrap();
    assert_eq!(opts_count(opts), 1);
    assert_eq!(opts.get(""), Some("val"));

    // Unknown key
    assert_eq!(parse_err(&mut list_01, "nonexistent=", false), "Invalid parameter 'nonexistent'");

    list_01.reset();
    list_03.reset();
}

#[test]
fn opts_parse_bool() {
    let mut list_02 = opts_list_02();

    let opts = list_02.parse("bool1=on,bool2=off", false).unwrap();
    assert_eq!(opts_count(opts), 2);
    assert!(opts.get_bool("bool1", false));
    assert!(!opts.get_bool("bool2", true));

    assert_eq!(
        parse_err(&mut list_02, "bool1=offer", false),
        "Parameter 'bool1' expects 'on' or 'off'"
    );

    list_02.reset();
}

#[test]
fn opts_parse_number() {
    let mut list_01 = opts_list_01();
    let expects_number = "Parameter 'number1' expects a number";

    // Lower limit zero
    let opts = list_01.parse("number1=0", false).unwrap();
    assert_eq!(opts_count(opts), 1);
    assert_eq!(opts.get_number("number1", 1), 0);

    // Upper limit 2^64-1
    let opts = list_01.parse("number1=18446744073709551615,number2=-1", false).unwrap();
    assert_eq!(opts_count(opts), 2);
    assert_eq!(opts.get_number("number1", 1), u64::MAX);
    assert_eq!(opts.get_number("number2", 0), u64::MAX);

    // Above upper limit
    assert_eq!(
        parse_err(&mut list_01, "number1=18446744073709551616", false),
        "Value '18446744073709551616' is too large for parameter 'number1'"
    );

    // Below lower limit
    assert_eq!(
        parse_err(&mut list_01, "number1=-18446744073709551616", false),
        "Value '-18446744073709551616' is too large for parameter 'number1'"
    );

    // Hex and octal
    let opts = list_01.parse("number1=0x2a,number2=052", false).unwrap();
    assert_eq!(opts_count(opts), 2);
    assert_eq!(opts.get_number("number1", 1), 42);
    assert_eq!(opts.get_number("number2", 0), 42);

    // Invalid
    assert_eq!(parse_err(&mut list_01, "number1=", false), expects_number);
    assert_eq!(parse_err(&mut list_01, "number1=eins", false), expects_number);

    // Leading whitespace
    let opts = list_01.parse("number1= \t42", false).unwrap();
    assert_eq!(opts_count(opts), 1);
    assert_eq!(opts.get_number("number1", 1), 42);

    // Trailing crap
    assert_eq!(parse_err(&mut list_01, "number1=3.14", false), expects_number);
    assert_eq!(parse_err(&mut list_01, "number1=08", false), expects_number);
    assert_eq!(parse_err(&mut list_01, "number1=0 ", false), expects_number);

    list_01.reset();
}

#[test]
fn opts_parse_size() {
    let mut list_02 = opts_list_02();

    // Lower limit zero
    let opts = list_02.parse("size1=0", false).unwrap();
    assert_eq!(opts_count(opts), 1);
    assert_eq!(opts.get_size("size1", 1), 0);

    // Note: full 64 bits of precision

    // Around double limit of precision: 2^53-1, 2^53, 2^53+1
    let opts = list_02
        .parse("size1=9007199254740991,size2=9007199254740992,size3=9007199254740993", false)
        .unwrap();
    assert_eq!(opts_count(opts), 3);
    assert_eq!(opts.get_size("size1", 1), 0x1fffffffffffff);
    assert_eq!(opts.get_size("size2", 1), 0x20000000000000);
    assert_eq!(opts.get_size("size3", 1), 0x20000000000001);

    // Close to signed int limit: 2^63-1, 2^63, 2^63+1
    let opts = list_02
        .parse(
            "size1=9223372036854775807,size2=9223372036854775808,size3=9223372036854775809",
            false,
        )
        .unwrap();
    assert_eq!(opts_count(opts), 3);
    assert_eq!(opts.get_size("size1", 1), 0x7fffffffffffffff);
    assert_eq!(opts.get_size("size2", 1), 0x8000000000000000);
    assert_eq!(opts.get_size("size3", 1), 0x8000000000000001);

    // Close to actual upper limit 0xfffffffffffff800 (53 msbs set)
    let opts =
        list_02.parse("size1=18446744073709549568,size2=18446744073709550591", false).unwrap();
    assert_eq!(opts_count(opts), 2);
    assert_eq!(opts.get_size("size1", 1), 0xfffffffffffff800);
    assert_eq!(opts.get_size("size2", 1), 0xfffffffffffffbff);

    // Actual limit, 2^64-1
    let opts = list_02.parse("size1=18446744073709551615", false).unwrap();
    assert_eq!(opts_count(opts), 1);
    assert_eq!(opts.get_size("size1", 1), 0xffffffffffffffff);

    // Beyond limits
    assert!(list_02.parse("size1=-1", false).is_err());
    assert!(list_02.parse("size1=18446744073709551616", false).is_err());

    // Suffixes
    let opts = list_02.parse("size1=8b,size2=1.5k,size3=2M", false).unwrap();
    assert_eq!(opts_count(opts), 3);
    assert_eq!(opts.get_size("size1", 0), 8);
    assert_eq!(opts.get_size("size2", 0), 1536);
    assert_eq!(opts.get_size("size3", 0), 2 * MIB);
    let opts = list_02.parse("size1=0.1G,size2=16777215T", false).unwrap();
    assert_eq!(opts_count(opts), 2);
    assert_eq!(opts.get_size("size1", 0), GIB / 10);
    assert_eq!(opts.get_size("size2", 0), 16777215 * TIB);

    // Beyond limit with suffix
    assert!(list_02.parse("size1=16777216T", false).is_err());

    // Trailing crap
    assert!(list_02.parse("size1=16E", false).is_err());
    assert!(list_02.parse("size1=16Gi", false).is_err());

    list_02.reset();
}

#[test]
fn has_help_option_table() {
    // params, expected with implied=false, expected with implied=true
    let test = [
        ("help", true, false),
        ("?", true, false),
        ("helpme", false, false),
        ("?me", false, false),
        ("a,help", true, true),
        ("a,?", true, true),
        ("a=0,help,b", true, true),
        ("a=0,?,b", true, true),
        ("help,b=1", true, false),
        ("?,b=1", true, false),
        ("a,b,,help", true, true),
        ("a,b,,?", true, true),
    ];
    let mut list_03 = opts_list_03();

    for (params, expect, expect_implied) in test {
        assert_eq!(has_help_option(params), expect, "{params}");
        let opts = list_03.parse(params, false).unwrap();
        assert_eq!(opts.has_help_opt(), expect, "{params}");
        let handle = opts.handle();
        list_03.del(handle);
        let opts = list_03.parse(params, true).unwrap();
        assert_eq!(opts.has_help_opt(), expect_implied, "{params} implied");
        let handle = opts.handle();
        list_03.del(handle);
    }
}

fn check_desc(
    desc: &QemuOptDesc,
    name: &str,
    ty: QemuOptType,
    help: Option<&str>,
    def_value_str: Option<&str>,
) {
    assert_eq!(desc.name, name);
    assert_eq!(desc.ty, ty);
    assert_eq!(desc.help, help);
    assert_eq!(desc.def_value_str, def_value_str);
}

fn append_verify_list_01(desc: &[QemuOptDesc], with_overlapping: bool) {
    let mut i = 0;

    if with_overlapping {
        check_desc(
            &desc[i],
            "str1",
            QemuOptType::String,
            Some("Help texts are preserved in qemu_opts_append"),
            Some("default"),
        );
        i += 1;

        check_desc(&desc[i], "str2", QemuOptType::String, None, None);
        i += 1;
    }

    check_desc(&desc[i], "str3", QemuOptType::String, None, None);
    i += 1;

    check_desc(
        &desc[i],
        "number1",
        QemuOptType::Number,
        Some("Having help texts only for some options is okay"),
        None,
    );
    i += 1;

    check_desc(&desc[i], "number2", QemuOptType::Number, None, None);
    i += 1;

    assert_eq!(desc.len(), i);
}

fn append_verify_list_02(desc: &[QemuOptDesc]) {
    check_desc(&desc[0], "str1", QemuOptType::String, None, None);
    check_desc(&desc[1], "str2", QemuOptType::String, None, None);
    check_desc(&desc[2], "bool1", QemuOptType::Bool, None, None);
    check_desc(&desc[3], "bool2", QemuOptType::Bool, None, None);
    check_desc(&desc[4], "size1", QemuOptType::Size, None, None);
    check_desc(&desc[5], "size2", QemuOptType::Size, None, None);
    check_desc(&desc[6], "size3", QemuOptType::Size, None, None);
}

#[test]
fn append_to_null() {
    let merged = QemuOptsList::append(None, &opts_list_01());

    assert_eq!(merged.name(), None);
    assert_eq!(merged.implied_opt_name(), None);
    assert!(!merged.merge_lists());

    append_verify_list_01(merged.desc(), true);
}

#[test]
fn append() {
    let first = QemuOptsList::append(None, &opts_list_02());
    let merged = QemuOptsList::append(Some(first), &opts_list_01());

    assert_eq!(merged.name(), None);
    assert_eq!(merged.implied_opt_name(), None);
    assert!(!merged.merge_lists());

    append_verify_list_02(&merged.desc()[..7]);
    append_verify_list_01(&merged.desc()[7..], false);
}

#[test]
fn to_qdict_basic() {
    let mut list_01 = opts_list_01();
    let opts = list_01.parse("str1=foo,str2=,str3=bar,number1=42", false).unwrap();

    let dict = opts.to_qdict();
    assert_eq!(dict.get_str("str1"), Some("foo"));
    assert_eq!(dict.get_str("str2"), Some(""));
    assert_eq!(dict.get_str("str3"), Some("bar"));
    assert_eq!(dict.get_str("number1"), Some("42"));
    assert!(!dict.contains_key("number2"));
}

#[test]
fn to_qdict_filtered() {
    let list_01 = opts_list_01();
    let list_02 = opts_list_02();
    let first = QemuOptsList::append(None, &list_02);
    let mut merged = QemuOptsList::append(Some(first), &list_01);

    let opts = merged.parse("str1=foo,str2=,str3=bar,bool1=off,number1=42", false).unwrap();

    // Convert to QDict without deleting from opts
    let mut dict = QDict::new();
    opts.to_qdict_filtered(&mut dict, Some(list_01.desc()), false);
    assert_eq!(dict.get_str("str1"), Some("foo"));
    assert_eq!(dict.get_str("str2"), Some(""));
    assert_eq!(dict.get_str("str3"), Some("bar"));
    assert_eq!(dict.get_str("number1"), Some("42"));
    assert!(!dict.contains_key("number2"));
    assert!(!dict.contains_key("bool1"));

    let mut dict = QDict::new();
    opts.to_qdict_filtered(&mut dict, Some(list_02.desc()), false);
    assert_eq!(dict.get_str("str1"), Some("foo"));
    assert_eq!(dict.get_str("str2"), Some(""));
    assert_eq!(dict.get_str("bool1"), Some("off"));
    assert!(!dict.contains_key("str3"));
    assert!(!dict.contains_key("number1"));
    assert!(!dict.contains_key("number2"));

    // Now delete converted options from opts
    let mut dict = QDict::new();
    opts.to_qdict_filtered(&mut dict, Some(list_01.desc()), true);
    assert_eq!(dict.get_str("str1"), Some("foo"));
    assert_eq!(dict.get_str("str2"), Some(""));
    assert_eq!(dict.get_str("str3"), Some("bar"));
    assert_eq!(dict.get_str("number1"), Some("42"));
    assert!(!dict.contains_key("number2"));
    assert!(!dict.contains_key("bool1"));

    let mut dict = QDict::new();
    opts.to_qdict_filtered(&mut dict, Some(list_02.desc()), true);
    assert_eq!(dict.get_str("bool1"), Some("off"));
    assert!(!dict.contains_key("str1"));
    assert!(!dict.contains_key("str2"));
    assert!(!dict.contains_key("str3"));
    assert!(!dict.contains_key("number1"));
    assert!(!dict.contains_key("number2"));

    assert!(opts.opts().is_empty());
}

#[test]
fn to_qdict_duplicates() {
    let mut list_03 = opts_list_03();
    let opts = list_03.parse("foo=a,foo=b", false).unwrap();

    // Verify that opts has two options with the same name
    let pairs: Vec<_> = opts.opts().iter().map(|o| (o.name(), o.value())).collect();
    assert_eq!(pairs, [("foo", "a"), ("foo", "b")]);

    // In the conversion to QDict, the last one wins
    let dict = opts.to_qdict();
    assert_eq!(dict.get_str("foo"), Some("b"));

    // The last one still wins if entries are deleted, and both are deleted
    let mut dict = QDict::new();
    opts.to_qdict_filtered(&mut dict, None, true);
    assert_eq!(dict.get_str("foo"), Some("b"));

    assert!(opts.opts().is_empty());
}

// What follows is not in test-qemu-opts.c.

#[test]
fn short_form_boolean_warnings() {
    let mut list_03 = opts_list_03();
    let mut warnings = Vec::new();
    let opts = list_03.parse_detailed("an,noaus,delay,nodelay,x=1", false, &mut warnings).unwrap();
    assert_eq!(opts.get("delay"), Some("off"));
    assert_eq!(warnings.len(), 4);
    let texts: Vec<_> = warnings.iter().map(|w| (w.message(), w.hint())).collect();
    assert_eq!(
        texts,
        [
            (
                "short-form boolean option 'an' deprecated".to_string(),
                "Please use an=on instead\n".to_string()
            ),
            (
                "short-form boolean option 'noaus' deprecated".to_string(),
                "Please use aus=off instead\n".to_string()
            ),
            (
                "short-form boolean option 'delay' deprecated".to_string(),
                "Please use nodelay=off instead\n".to_string()
            ),
            (
                "short-form boolean option 'nodelay' deprecated".to_string(),
                "Please use nodelay=on instead\n".to_string()
            ),
        ]
    );
    assert_eq!(warnings[1], FlagWarning { name: "aus".into(), negated: true });

    // The implied key takes the first value, so it draws no warning.
    let mut warnings = Vec::new();
    list_03.parse_detailed("an,x=1", true, &mut warnings).unwrap();
    assert!(warnings.is_empty());
}

#[test]
fn help_requests() {
    // A list with descriptions stops at a help request.
    let mut list_01 = opts_list_01();
    let mut warnings = Vec::new();
    let r = list_01.parse_detailed("str1=x,help", false, &mut warnings).map(|o| o.handle());
    assert!(matches!(r, Err(ParseFailure::HelpWanted)));
    assert!(list_01.is_empty());

    // One that accepts anything takes it as a plain option.
    let mut list_03 = opts_list_03();
    let opts = list_03.parse_detailed("help", false, &mut warnings).unwrap();
    assert_eq!(opts.get("help"), Some("on"));

    assert_eq!(
        list_01.help_text(true),
        "opts_list_01 options:\n\
         \x20 number1=<num>          - Having help texts only for some options is okay\n\
         \x20 number2=<num>\n\
         \x20 str1=<str>             - Help texts are preserved in qemu_opts_append\n\
         \x20 str2=<str>\n\
         \x20 str3=<str>\n"
    );
    assert_eq!(list_03.help_text(true), "There are no options for opts_list_03.\n");
    assert_eq!(list_03.help_text(false), "There are no options for opts_list_03.\n");
}

#[test]
fn merge_lists() {
    let mut groups = register_opts();
    let list = groups.find_opts("opts_list_04").unwrap();
    list.parse("str3=a", false).unwrap();
    let opts = list.parse("str3=b", false).unwrap();
    assert_eq!(opts_count(opts), 2);
    assert_eq!(opts.get("str3"), Some("b"));
    assert_eq!(list.iter().count(), 1);
    assert_eq!(parse_err(list, "id=x,str3=c", false), "Invalid parameter 'id'");
}

#[test]
fn set_option() {
    let mut groups = register_opts();
    groups.find_opts("opts_list_01").unwrap().parse("id=disk0,str1=a", false).unwrap();

    groups.set_option("opts_list_01.disk0.str2=b,c").unwrap();
    let list = groups.find_opts("opts_list_01").unwrap();
    assert_eq!(list.find(Some("disk0")).unwrap().get("str2"), Some("b,c"));

    let err = |groups: &mut ConfigGroups, s: &str| {
        groups.set_option(s).unwrap_err().message().to_string()
    };
    assert_eq!(err(&mut groups, "opts_list_01.disk0"), "can't parse: \"opts_list_01.disk0\"");
    assert_eq!(err(&mut groups, "nope.disk0.str2=b"), "There is no option group 'nope'");
    assert_eq!(
        err(&mut groups, "opts_list_01.disk1.str2=b"),
        "there is no opts_list_01 \"disk1\" defined"
    );
    assert_eq!(err(&mut groups, "opts_list_01.disk0.str9=b"), "Invalid parameter 'str9'");
}

#[test]
fn defaults_and_print() {
    let mut list_01 = opts_list_01();
    let opts = list_01.parse("id=a1,number1=7,str2=x,,y", false).unwrap();
    // str1 has a default value, which qemu_opt_get() falls back to.
    assert_eq!(opts.get("str1"), Some("default"));
    assert_eq!(opts.to_print_string(" "), "id=a1 str1=default str2=x,,y number1=7");
}
