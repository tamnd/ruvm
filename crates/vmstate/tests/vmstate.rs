// SPDX-License-Identifier: GPL-2.0-or-later

//! The cases from QEMU's tests/unit/test-vmstate.c, with the same descriptions, values and
//! expected wire bytes, followed by cases for the parts that file does not cover.

use std::sync::LazyLock;

use ruvm_vmstate::info::{Int32, Int32Equal, Uint32Equal};
use ruvm_vmstate::{
    EINVAL, EIO, QEMU_VM_EOF, StreamReader, StreamWriter, VMS_MARKER_PTR_NULL,
    VMS_MARKER_PTR_VALID, VmStateDescription, VmStateField, vmstate_load_state, vmstate_save_state,
};

type Vmsd<T> = LazyLock<VmStateDescription<T>>;

/// `save_vmstate()`: the saved bytes followed by `QEMU_VM_EOF`.
fn save_vmstate<T>(desc: &VmStateDescription<T>, obj: &mut T) -> Vec<u8> {
    let mut f = StreamWriter::new();
    if let Err(e) = vmstate_save_state(&mut f, desc, obj) {
        panic!("{}", e.message());
    }
    f.put_byte(QEMU_VM_EOF);
    assert_eq!(f.get_error(), 0);
    f.into_inner()
}

/// `load_vmstate_one()`. A failed load must leave an error in the stream and a good one must not.
fn load_vmstate_one<T>(
    desc: &VmStateDescription<T>,
    obj: &mut T,
    version: i32,
    wire: &[u8],
) -> Result<(), String> {
    let mut f = StreamReader::new(wire);
    match vmstate_load_state(&mut f, desc, obj, version) {
        Ok(()) => {
            assert_eq!(f.get_byte(), QEMU_VM_EOF);
            assert_eq!(f.get_error(), 0);
            Ok(())
        }
        Err(e) => {
            assert_ne!(f.get_error(), 0, "{}", e.message());
            Err(e.message().to_string())
        }
    }
}

/// `load_vmstate()`: loading an empty, a truncated or a half stream must fail, then the whole
/// stream is loaded into a fresh copy of `obj`.
fn load_vmstate<T: Clone>(
    desc: &VmStateDescription<T>,
    obj: &mut T,
    version: i32,
    wire: &[u8],
) -> Result<(), String> {
    let clone = obj.clone();
    let size = wire.len();
    assert!(load_vmstate_one(desc, obj, version, &wire[..0]).is_err());
    if size > 3 {
        *obj = clone.clone();
        assert!(load_vmstate_one(desc, obj, version, &wire[..size - 2]).is_err());
        *obj = clone.clone();
        assert!(load_vmstate_one(desc, obj, version, &wire[..size / 2]).is_err());
        *obj = clone.clone();
        assert!(load_vmstate_one(desc, obj, version, &wire[size / 2..size / 2 * 2]).is_err());
    }
    *obj = clone;
    load_vmstate_one(desc, obj, version, wire)
}

#[derive(Debug, Default, Clone, PartialEq)]
struct TestSimple {
    b_1: bool,
    b_2: bool,
    u8_1: u8,
    u16_1: u16,
    u32_1: u32,
    u64_1: u64,
    i8_1: i8,
    i8_2: i8,
    i16_1: i16,
    i16_2: i16,
    i32_1: i32,
    i32_2: i32,
    i64_1: i64,
    i64_2: i64,
}

fn obj_simple() -> TestSimple {
    TestSimple {
        b_1: true,
        b_2: false,
        u8_1: 130,
        u16_1: 512,
        u32_1: 70000,
        u64_1: 12121212,
        i8_1: 65,
        i8_2: -65,
        i16_1: 512,
        i16_2: -512,
        i32_1: 70000,
        i32_2: -70000,
        i64_1: 12121212,
        i64_2: -12121212,
    }
}

static VMSTATE_SIMPLE_PRIMITIVE: Vmsd<TestSimple> = LazyLock::new(|| {
    VmStateDescription::new("simple/primitive").version_id(1).minimum_version_id(1).fields([
        VmStateField::scalar("b_1", |s: &mut TestSimple| &mut s.b_1),
        VmStateField::scalar("b_2", |s: &mut TestSimple| &mut s.b_2),
        VmStateField::scalar("u8_1", |s: &mut TestSimple| &mut s.u8_1),
        VmStateField::scalar("u16_1", |s: &mut TestSimple| &mut s.u16_1),
        VmStateField::scalar("u32_1", |s: &mut TestSimple| &mut s.u32_1),
        VmStateField::scalar("u64_1", |s: &mut TestSimple| &mut s.u64_1),
        VmStateField::scalar("i8_1", |s: &mut TestSimple| &mut s.i8_1),
        VmStateField::scalar("i8_2", |s: &mut TestSimple| &mut s.i8_2),
        VmStateField::scalar("i16_1", |s: &mut TestSimple| &mut s.i16_1),
        VmStateField::scalar("i16_2", |s: &mut TestSimple| &mut s.i16_2),
        VmStateField::scalar("i32_1", |s: &mut TestSimple| &mut s.i32_1),
        VmStateField::scalar("i32_2", |s: &mut TestSimple| &mut s.i32_2),
        VmStateField::scalar("i64_1", |s: &mut TestSimple| &mut s.i64_1),
        VmStateField::scalar("i64_2", |s: &mut TestSimple| &mut s.i64_2),
    ])
});

#[rustfmt::skip]
const WIRE_SIMPLE_PRIMITIVE: &[u8] = &[
    /* b_1 */   0x01,
    /* b_2 */   0x00,
    /* u8_1 */  0x82,
    /* u16_1 */ 0x02, 0x00,
    /* u32_1 */ 0x00, 0x01, 0x11, 0x70,
    /* u64_1 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0xb8, 0xf4, 0x7c,
    /* i8_1 */  0x41,
    /* i8_2 */  0xbf,
    /* i16_1 */ 0x02, 0x00,
    /* i16_2 */ 0xfe, 0x0,
    /* i32_1 */ 0x00, 0x01, 0x11, 0x70,
    /* i32_2 */ 0xff, 0xfe, 0xee, 0x90,
    /* i64_1 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0xb8, 0xf4, 0x7c,
    /* i64_2 */ 0xff, 0xff, 0xff, 0xff, 0xff, 0x47, 0x0b, 0x84,
    QEMU_VM_EOF,
];

#[test]
fn simple_primitive() {
    let wire = save_vmstate(&VMSTATE_SIMPLE_PRIMITIVE, &mut obj_simple());
    assert_eq!(wire, WIRE_SIMPLE_PRIMITIVE);

    let mut obj = TestSimple::default();
    load_vmstate(&VMSTATE_SIMPLE_PRIMITIVE, &mut obj, 1, WIRE_SIMPLE_PRIMITIVE).unwrap();
    assert_eq!(obj, obj_simple());
}

#[derive(Debug, Default, Clone, PartialEq)]
struct TestSimpleArray {
    u16_1: [u16; 3],
}

static VMSTATE_SIMPLE_ARR: Vmsd<TestSimpleArray> = LazyLock::new(|| {
    VmStateDescription::new("simple/array")
        .version_id(1)
        .minimum_version_id(1)
        .field(VmStateField::array("u16_1", |s: &mut TestSimpleArray| &mut s.u16_1))
});

#[rustfmt::skip]
const WIRE_SIMPLE_ARR: &[u8] = &[
    /* u16_1 */ 0x00, 0x42,
    /* u16_1 */ 0x00, 0x43,
    /* u16_1 */ 0x00, 0x44,
    QEMU_VM_EOF,
];

#[test]
fn simple_array() {
    let mut src = TestSimpleArray { u16_1: [0x42, 0x43, 0x44] };
    assert_eq!(save_vmstate(&VMSTATE_SIMPLE_ARR, &mut src), WIRE_SIMPLE_ARR);

    let mut obj = TestSimpleArray::default();
    load_vmstate(&VMSTATE_SIMPLE_ARR, &mut obj, 1, WIRE_SIMPLE_ARR).unwrap();
    assert_eq!(obj, src);
}

#[derive(Debug, Default, Clone, PartialEq)]
struct TestStruct {
    a: u32,
    b: u32,
    c: u32,
    e: u32,
    d: u64,
    f: u64,
    skip_c_e: bool,
}

static VMSTATE_VERSIONED: Vmsd<TestStruct> = LazyLock::new(|| {
    VmStateDescription::new("test/versioned").version_id(2).minimum_version_id(1).fields([
        VmStateField::scalar("a", |s: &mut TestStruct| &mut s.a),
        // A versioned field in the middle, so bugs are easier to catch.
        VmStateField::scalar("b", |s: &mut TestStruct| &mut s.b).version(2),
        VmStateField::scalar("c", |s: &mut TestStruct| &mut s.c),
        VmStateField::scalar("d", |s: &mut TestStruct| &mut s.d),
        VmStateField::scalar("e", |s: &mut TestStruct| &mut s.e).version(2),
        VmStateField::scalar("f", |s: &mut TestStruct| &mut s.f).version(2),
    ])
});

#[test]
fn versioned_load_v1() {
    #[rustfmt::skip]
    let buf = [
        0, 0, 0, 10,             /* a */
        0, 0, 0, 30,             /* c */
        0, 0, 0, 0, 0, 0, 0, 40, /* d */
        QEMU_VM_EOF,
    ];
    let mut obj = TestStruct { b: 200, e: 500, f: 600, ..Default::default() };
    let mut f = StreamReader::new(&buf);
    vmstate_load_state(&mut f, &VMSTATE_VERSIONED, &mut obj, 1).unwrap();
    assert_eq!(f.get_error(), 0);
    assert_eq!((obj.a, obj.b, obj.c, obj.d, obj.e, obj.f), (10, 200, 30, 40, 500, 600));
}

#[test]
fn versioned_load_v2() {
    #[rustfmt::skip]
    let buf = [
        0, 0, 0, 10,             /* a */
        0, 0, 0, 20,             /* b */
        0, 0, 0, 30,             /* c */
        0, 0, 0, 0, 0, 0, 0, 40, /* d */
        0, 0, 0, 50,             /* e */
        0, 0, 0, 0, 0, 0, 0, 60, /* f */
        QEMU_VM_EOF,
    ];
    let mut obj = TestStruct::default();
    let mut f = StreamReader::new(&buf);
    vmstate_load_state(&mut f, &VMSTATE_VERSIONED, &mut obj, 2).unwrap();
    assert_eq!((obj.a, obj.b, obj.c, obj.d, obj.e, obj.f), (10, 20, 30, 40, 50, 60));
}

fn test_skip(t: &TestStruct, _version_id: i32) -> bool {
    !t.skip_c_e
}

static VMSTATE_SKIPPING: Vmsd<TestStruct> = LazyLock::new(|| {
    VmStateDescription::new("test/skip").version_id(2).minimum_version_id(1).fields([
        VmStateField::scalar("a", |s: &mut TestStruct| &mut s.a),
        VmStateField::scalar("b", |s: &mut TestStruct| &mut s.b),
        VmStateField::scalar("c", |s: &mut TestStruct| &mut s.c).test(test_skip),
        VmStateField::scalar("d", |s: &mut TestStruct| &mut s.d),
        VmStateField::scalar("e", |s: &mut TestStruct| &mut s.e).test(test_skip),
        VmStateField::scalar("f", |s: &mut TestStruct| &mut s.f).version(2),
    ])
});

fn abcdef(skip_c_e: bool) -> TestStruct {
    TestStruct { a: 1, b: 2, c: 3, d: 4, e: 5, f: 6, skip_c_e }
}

#[test]
fn field_exists_save_noskip() {
    let mut f = StreamWriter::new();
    vmstate_save_state(&mut f, &VMSTATE_SKIPPING, &mut abcdef(false)).unwrap();
    assert_eq!(f.get_error(), 0);
    #[rustfmt::skip]
    let expected = [
        0, 0, 0, 1,             /* a */
        0, 0, 0, 2,             /* b */
        0, 0, 0, 3,             /* c */
        0, 0, 0, 0, 0, 0, 0, 4, /* d */
        0, 0, 0, 5,             /* e */
        0, 0, 0, 0, 0, 0, 0, 6, /* f */
    ];
    assert_eq!(f.as_bytes(), expected);
}

#[test]
fn field_exists_save_skip() {
    let mut f = StreamWriter::new();
    vmstate_save_state(&mut f, &VMSTATE_SKIPPING, &mut abcdef(true)).unwrap();
    assert_eq!(f.get_error(), 0);
    #[rustfmt::skip]
    let expected = [
        0, 0, 0, 1,             /* a */
        0, 0, 0, 2,             /* b */
        0, 0, 0, 0, 0, 0, 0, 4, /* d */
        0, 0, 0, 0, 0, 0, 0, 6, /* f */
    ];
    assert_eq!(f.as_bytes(), expected);
}

#[test]
fn field_exists_load_noskip() {
    #[rustfmt::skip]
    let buf = [
        0, 0, 0, 10,             /* a */
        0, 0, 0, 20,             /* b */
        0, 0, 0, 30,             /* c */
        0, 0, 0, 0, 0, 0, 0, 40, /* d */
        0, 0, 0, 50,             /* e */
        0, 0, 0, 0, 0, 0, 0, 60, /* f */
        QEMU_VM_EOF,
    ];
    let mut obj = TestStruct { skip_c_e: false, ..Default::default() };
    let mut f = StreamReader::new(&buf);
    vmstate_load_state(&mut f, &VMSTATE_SKIPPING, &mut obj, 2).unwrap();
    assert_eq!(f.get_error(), 0);
    assert_eq!((obj.a, obj.b, obj.c, obj.d, obj.e, obj.f), (10, 20, 30, 40, 50, 60));
}

#[test]
fn field_exists_load_skip() {
    #[rustfmt::skip]
    let buf = [
        0, 0, 0, 10,             /* a */
        0, 0, 0, 20,             /* b */
        0, 0, 0, 0, 0, 0, 0, 40, /* d */
        0, 0, 0, 0, 0, 0, 0, 60, /* f */
        QEMU_VM_EOF,
    ];
    let mut obj = TestStruct { skip_c_e: true, c: 300, e: 500, ..Default::default() };
    let mut f = StreamReader::new(&buf);
    vmstate_load_state(&mut f, &VMSTATE_SKIPPING, &mut obj, 2).unwrap();
    assert_eq!(f.get_error(), 0);
    assert_eq!((obj.a, obj.b, obj.c, obj.d, obj.e, obj.f), (10, 20, 300, 40, 500, 60));
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct TestStructTriv {
    i: i32,
}

static VMSD_TST: Vmsd<TestStructTriv> = LazyLock::new(|| {
    VmStateDescription::new("test/tst")
        .version_id(1)
        .minimum_version_id(1)
        .field(VmStateField::scalar("i", |s: &mut TestStructTriv| &mut s.i))
});

const AR_SIZE: usize = 4;

#[derive(Debug, Default, Clone, PartialEq)]
struct TestArrayOfPtrToStuct {
    ar: [Option<TestStructTriv>; AR_SIZE],
}

static VMSD_ARPS: Vmsd<TestArrayOfPtrToStuct> = LazyLock::new(|| {
    VmStateDescription::new("test/arps").version_id(1).minimum_version_id(1).field(
        VmStateField::array_of_pointer_to_struct(
            "ar",
            &VMSD_TST,
            |s: &mut TestArrayOfPtrToStuct| &mut s.ar,
        ),
    )
});

#[rustfmt::skip]
const WIRE_ARR_PTR_NO0: &[u8] = &[
    0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x02,
    0x00, 0x00, 0x00, 0x03,
    QEMU_VM_EOF,
];

fn triv(i: i32) -> Option<TestStructTriv> {
    Some(TestStructTriv { i })
}

#[test]
fn array_ptr_str_no0_save() {
    let mut sample = TestArrayOfPtrToStuct { ar: [triv(0), triv(1), triv(2), triv(3)] };
    assert_eq!(save_vmstate(&VMSD_ARPS, &mut sample), WIRE_ARR_PTR_NO0);
}

#[test]
fn array_ptr_str_no0_load() {
    let mut obj = TestArrayOfPtrToStuct { ar: [triv(0); AR_SIZE] };
    load_vmstate_one(&VMSD_ARPS, &mut obj, 1, WIRE_ARR_PTR_NO0).unwrap();
    assert_eq!(obj.ar, [triv(0), triv(1), triv(2), triv(3)]);
}

#[rustfmt::skip]
const WIRE_ARR_PTR_0: &[u8] = &[
    0x00, 0x00, 0x00, 0x00,
    VMS_MARKER_PTR_NULL,
    0x00, 0x00, 0x00, 0x02,
    0x00, 0x00, 0x00, 0x03,
    QEMU_VM_EOF,
];

#[test]
fn array_ptr_str_0_save() {
    let mut sample = TestArrayOfPtrToStuct { ar: [triv(0), None, triv(2), triv(3)] };
    assert_eq!(save_vmstate(&VMSD_ARPS, &mut sample), WIRE_ARR_PTR_0);
}

#[test]
fn array_ptr_str_0_load() {
    let mut obj = TestArrayOfPtrToStuct { ar: [triv(0), None, triv(0), triv(0)] };
    load_vmstate_one(&VMSD_ARPS, &mut obj, 1, WIRE_ARR_PTR_0).unwrap();
    assert_eq!(obj.ar, [triv(0), None, triv(2), triv(3)]);
}

#[derive(Debug, Default, Clone, PartialEq)]
struct TestArrayOfPtrToInt {
    ar: [Option<i32>; AR_SIZE],
}

static VMSD_ARPP: Vmsd<TestArrayOfPtrToInt> = LazyLock::new(|| {
    VmStateDescription::new("test/arps").version_id(1).minimum_version_id(1).field(
        VmStateField::array_of_pointer("ar", &Int32, |s: &mut TestArrayOfPtrToInt| &mut s.ar),
    )
});

#[test]
fn array_ptr_prim_0_save() {
    let mut sample = TestArrayOfPtrToInt { ar: [Some(0), None, Some(2), Some(3)] };
    assert_eq!(save_vmstate(&VMSD_ARPP, &mut sample), WIRE_ARR_PTR_0);
}

#[test]
fn array_ptr_prim_0_load() {
    let mut obj = TestArrayOfPtrToInt { ar: [Some(3), None, Some(1), Some(0)] };
    load_vmstate_one(&VMSD_ARPP, &mut obj, 1, WIRE_ARR_PTR_0).unwrap();
    assert_eq!(obj.ar, [Some(0), None, Some(2), Some(3)]);
}

#[rustfmt::skip]
const WIRE_ARR_PTR_WITH_NULLS: &[u8] = &[
    VMS_MARKER_PTR_VALID,
    0x00, 0x00, 0x00, 0x00,
    VMS_MARKER_PTR_NULL,
    VMS_MARKER_PTR_VALID,
    0x00, 0x00, 0x00, 0x02,
    VMS_MARKER_PTR_VALID,
    0x00, 0x00, 0x00, 0x03,
    QEMU_VM_EOF,
];

#[derive(Debug, Default, Clone, PartialEq)]
struct TestVArrayOfPtrToStuctWithNulls {
    ar_items_num: u32,
    ar: Vec<Option<TestStructTriv>>,
}

static VMSD_ARPS_WITH_NULLS: Vmsd<TestVArrayOfPtrToStuctWithNulls> = LazyLock::new(|| {
    VmStateDescription::new("test/arps_with_nulls").version_id(1).minimum_version_id(1).field(
        VmStateField::varray_of_pointer_to_struct_alloc(
            "ar",
            |s: &TestVArrayOfPtrToStuctWithNulls| s.ar_items_num as usize,
            &VMSD_TST,
            |s: &mut TestVArrayOfPtrToStuctWithNulls| &mut s.ar,
        ),
    )
});

#[test]
fn array_ptr_nulls_str_save() {
    let mut sample = TestVArrayOfPtrToStuctWithNulls {
        ar_items_num: AR_SIZE as u32,
        ar: vec![triv(0), None, triv(2), triv(3)],
    };
    assert_eq!(save_vmstate(&VMSD_ARPS_WITH_NULLS, &mut sample), WIRE_ARR_PTR_WITH_NULLS);
}

#[test]
fn array_ptr_nulls_str_load() {
    let mut obj =
        TestVArrayOfPtrToStuctWithNulls { ar_items_num: AR_SIZE as u32, ar: vec![None; AR_SIZE] };
    load_vmstate_one(&VMSD_ARPS_WITH_NULLS, &mut obj, 1, WIRE_ARR_PTR_WITH_NULLS).unwrap();
    assert_eq!(obj.ar, [triv(0), None, triv(2), triv(3)]);
}

#[test]
fn array_ptr_bad_marker() {
    let mut obj =
        TestVArrayOfPtrToStuctWithNulls { ar_items_num: AR_SIZE as u32, ar: vec![None; AR_SIZE] };
    let mut f = StreamReader::new(&[0x32, 0, 0, 0, 0]);
    let e = vmstate_load_state(&mut f, &VMSD_ARPS_WITH_NULLS, &mut obj, 1).unwrap_err();
    assert_eq!(e.message(), "Unexpected ptr marker: 50");
}

/// `TmpTestStruct`. QEMU's temporary holds a pointer to its parent; this one holds a copy that is
/// handed back after loading.
#[derive(Debug, Default)]
struct TmpTestStruct {
    parent: TestStruct,
    diff: i64,
}

static VMSTATE_TMP_BACK_TO_PARENT: Vmsd<TestStruct> = LazyLock::new(|| {
    VmStateDescription::new("test/tmp_child_parent")
        .field(VmStateField::scalar("f", |s: &mut TestStruct| &mut s.f))
});

static VMSTATE_TMP_CHILD: Vmsd<TmpTestStruct> = LazyLock::new(|| {
    VmStateDescription::new("test/tmp_child")
        .pre_save(|tts: &mut TmpTestStruct| {
            tts.diff = i64::from(tts.parent.b) - i64::from(tts.parent.a);
            0
        })
        .post_load(|tts: &mut TmpTestStruct, _version_id| {
            tts.parent.b = (i64::from(tts.parent.a) + tts.diff) as u32;
            0
        })
        .field(VmStateField::scalar("diff", |s: &mut TmpTestStruct| &mut s.diff))
        .field(VmStateField::structure(
            "parent",
            &VMSTATE_TMP_BACK_TO_PARENT,
            |s: &mut TmpTestStruct| &mut s.parent,
        ))
});

static VMSTATE_WITH_TMP: Vmsd<TestStruct> = LazyLock::new(|| {
    VmStateDescription::new("test/with_tmp").version_id(1).fields([
        VmStateField::scalar("a", |s: &mut TestStruct| &mut s.a),
        VmStateField::scalar("d", |s: &mut TestStruct| &mut s.d),
        VmStateField::with_tmp(
            &VMSTATE_TMP_CHILD,
            |parent: &TestStruct| TmpTestStruct { parent: parent.clone(), diff: 0 },
            |parent: &mut TestStruct, tmp: TmpTestStruct| *parent = tmp.parent,
        ),
    ])
});

#[test]
fn tmp_struct() {
    #[rustfmt::skip]
    let wire_with_tmp: &[u8] = &[
        /* u32 a */ 0x00, 0x00, 0x00, 0x02,
        /* u64 d */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        /* diff  */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
        /* u64 f */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08,
        QEMU_VM_EOF,
    ];

    let mut obj = TestStruct { a: 2, b: 4, d: 1, f: 8, ..Default::default() };
    assert_eq!(save_vmstate(&VMSTATE_WITH_TMP, &mut obj), wire_with_tmp);

    let mut obj = TestStruct::default();
    load_vmstate(&VMSTATE_WITH_TMP, &mut obj, 1, wire_with_tmp).unwrap();
    assert_eq!(obj.a, 2); // from the top level description
    assert_eq!(obj.b, 4); // from the child's post_load
    assert_eq!(obj.d, 1); // from the top level description
    assert_eq!(obj.f, 8); // from the child back to the parent
}

// What follows is not in test-vmstate.c. The expected bytes are written out by hand from the
// stream format in vmstate.c: fields back to back, then for each subsection 0x05, a length byte,
// the name, a big-endian version and the subsection's own fields.

#[derive(Debug, Default, Clone, PartialEq)]
struct Dev {
    reg: u32,
    extra: u16,
    send_extra: bool,
    inner: Inner,
}

#[derive(Debug, Default, Clone, PartialEq)]
struct Inner {
    x: u8,
    y: i16,
}

static VMSTATE_INNER: Vmsd<Inner> = LazyLock::new(|| {
    VmStateDescription::new("inner").version_id(1).minimum_version_id(1).fields([
        VmStateField::scalar("x", |s: &mut Inner| &mut s.x),
        VmStateField::scalar("y", |s: &mut Inner| &mut s.y),
    ])
});

static VMSTATE_DEV_EXTRA: Vmsd<Dev> = LazyLock::new(|| {
    VmStateDescription::new("dev/extra")
        .version_id(3)
        .minimum_version_id(3)
        .needed(|s: &Dev| s.send_extra)
        .field(VmStateField::scalar("extra", |s: &mut Dev| &mut s.extra))
});

static VMSTATE_DEV: Vmsd<Dev> = LazyLock::new(|| {
    VmStateDescription::new("dev")
        .version_id(1)
        .minimum_version_id(1)
        .field(VmStateField::scalar("reg", |s: &mut Dev| &mut s.reg))
        .field(VmStateField::structure("inner", &VMSTATE_INNER, |s: &mut Dev| &mut s.inner))
        .subsection(&VMSTATE_DEV_EXTRA)
});

fn dev(send_extra: bool) -> Dev {
    Dev { reg: 0x11223344, extra: 0xbeef, send_extra, inner: Inner { x: 7, y: -2 } }
}

#[rustfmt::skip]
const WIRE_DEV_WITH_SUBSECTION: &[u8] = &[
    /* reg */        0x11, 0x22, 0x33, 0x44,
    /* inner.x */    0x07,
    /* inner.y */    0xff, 0xfe,
    /* subsection */ 0x05,
    /* name len */   0x09,
    /* name */       b'd', b'e', b'v', b'/', b'e', b'x', b't', b'r', b'a',
    /* version */    0x00, 0x00, 0x00, 0x03,
    /* extra */      0xbe, 0xef,
    QEMU_VM_EOF,
];

#[test]
fn subsection_present() {
    assert_eq!(save_vmstate(&VMSTATE_DEV, &mut dev(true)), WIRE_DEV_WITH_SUBSECTION);

    // Not load_vmstate(): the second half of this stream happens to be a valid section itself.
    let mut obj = Dev::default();
    load_vmstate_one(&VMSTATE_DEV, &mut obj, 1, WIRE_DEV_WITH_SUBSECTION).unwrap();
    assert_eq!(obj, Dev { send_extra: false, ..dev(true) });
}

#[test]
fn subsection_absent() {
    let wire = save_vmstate(&VMSTATE_DEV, &mut dev(false));
    assert_eq!(wire, [0x11, 0x22, 0x33, 0x44, 0x07, 0xff, 0xfe, QEMU_VM_EOF]);

    // A subsection the source did not send keeps the destination's value.
    let mut obj = Dev { extra: 0x1234, ..Default::default() };
    load_vmstate(&VMSTATE_DEV, &mut obj, 1, &wire).unwrap();
    assert_eq!(obj.extra, 0x1234);
    assert_eq!(obj.inner, Inner { x: 7, y: -2 });
}

#[test]
fn subsection_unknown() {
    let mut wire = WIRE_DEV_WITH_SUBSECTION.to_vec();
    wire[17] = b'X';
    let mut obj = Dev::default();
    let e = load_vmstate_one(&VMSTATE_DEV, &mut obj, 1, &wire).unwrap_err();
    assert_eq!(e, "VM subsection 'dev/extrX' in 'dev' does not exist");
}

#[test]
fn subsection_with_other_prefix_is_left_alone() {
    // A subsection marker for a name that does not start with "dev" belongs to someone else:
    // the load stops there without consuming it.
    let wire = [0x11, 0x22, 0x33, 0x44, 0x07, 0xff, 0xfe, 0x05, 0x05, b'o', b't', b'h', b'e', b'r'];
    let mut obj = Dev::default();
    let mut f = StreamReader::new(&wire);
    vmstate_load_state(&mut f, &VMSTATE_DEV, &mut obj, 1).unwrap();
    assert_eq!(f.position(), 7);
    assert_eq!(f.get_error(), 0);
}

#[test]
fn subsection_too_old() {
    let mut wire = WIRE_DEV_WITH_SUBSECTION.to_vec();
    wire[21] = 2;
    let mut obj = Dev::default();
    let e = load_vmstate_one(&VMSTATE_DEV, &mut obj, 1, &wire).unwrap_err();
    assert_eq!(
        e,
        "Loading VM subsection 'dev/extra' in 'dev' failed: \
         dev/extra: incoming version_id 2 is too old for local minimum version_id 3"
    );
}

#[test]
fn subsection_version_is_one_byte() {
    // vmstate_subsection_load() keeps the version in a uint8_t, so 0x103 reads as 3.
    let mut wire = WIRE_DEV_WITH_SUBSECTION.to_vec();
    wire[20] = 1;
    let mut obj = Dev::default();
    load_vmstate_one(&VMSTATE_DEV, &mut obj, 1, &wire).unwrap();
    assert_eq!(obj.extra, 0xbeef);
}

#[test]
fn section_too_old_and_too_new() {
    let mut obj = TestStruct::default();
    let mut f = StreamReader::new(&[]);
    let e = vmstate_load_state(&mut f, &VMSTATE_VERSIONED, &mut obj, 0).unwrap_err();
    assert_eq!(
        e.message(),
        "test/versioned: incoming version_id 0 is too old for local minimum version_id 1"
    );
    let e = vmstate_load_state(&mut f, &VMSTATE_VERSIONED, &mut obj, 3).unwrap_err();
    assert_eq!(
        e.message(),
        "test/versioned: incoming version_id 3 is too new for local version_id 2"
    );
}

#[test]
fn nested_struct_errors_pass_through() {
    // The nested description is loaded at its own version, which here is too new for it.
    static NEWER_INNER: Vmsd<Inner> =
        LazyLock::new(|| VmStateDescription::new("inner").version_id(2).minimum_version_id(2));
    static OUTER: Vmsd<Dev> = LazyLock::new(|| {
        VmStateDescription::new("outer").version_id(1).field(VmStateField::vstruct(
            "inner",
            &NEWER_INNER,
            1,
            |s: &mut Dev| &mut s.inner,
        ))
    });
    let mut obj = Dev::default();
    let e = load_vmstate_one(&OUTER, &mut obj, 1, &[QEMU_VM_EOF]).unwrap_err();
    assert_eq!(e, "inner: incoming version_id 1 is too old for local minimum version_id 2");
}

#[test]
fn truncated_stream_reports_eio() {
    let mut obj = TestSimple::default();
    let mut f = StreamReader::new(&WIRE_SIMPLE_PRIMITIVE[..4]);
    let e = vmstate_load_state(&mut f, &VMSTATE_SIMPLE_PRIMITIVE, &mut obj, 1).unwrap_err();
    assert_eq!(e.message(), "Failed to load simple/primitive state: stream error: -5");
    assert_eq!(f.get_error(), -EIO);
}

#[derive(Debug, Default, Clone, PartialEq)]
struct Misc {
    magic: u32,
    count: i32,
    len: u32,
    data: Vec<u16>,
    blob_len: u32,
    blob: Vec<u8>,
    fixed: [u8; 3],
    limit: i32,
}

static VMSTATE_MISC: Vmsd<Misc> = LazyLock::new(|| {
    VmStateDescription::new("misc").version_id(1).minimum_version_id(1).fields([
        VmStateField::single("magic", &Uint32Equal, |s: &mut Misc| &mut s.magic),
        VmStateField::unused(3),
        VmStateField::single("count", &Int32Equal, |s: &mut Misc| &mut s.count),
        VmStateField::scalar("len", |s: &mut Misc| &mut s.len),
        VmStateField::validate("len_ok", |s: &Misc, _| s.len <= 4),
        VmStateField::varray_alloc("data", |s: &Misc| s.len as usize, |s: &mut Misc| &mut s.data),
        VmStateField::scalar("blob_len", |s: &mut Misc| &mut s.blob_len),
        VmStateField::vbuffer_alloc(
            "blob",
            |s: &Misc| s.blob_len as usize,
            |s: &mut Misc| &mut s.blob,
        ),
        VmStateField::buffer("fixed", |s: &mut Misc| &mut s.fixed),
        VmStateField::single("limit", &ruvm_vmstate::info::Int32Le, |s: &mut Misc| &mut s.limit),
    ])
});

fn misc() -> Misc {
    Misc {
        magic: 0xcafe,
        count: -1,
        len: 2,
        data: vec![0x0102, 0x0304],
        blob_len: 3,
        blob: vec![9, 8, 7],
        fixed: *b"abc",
        limit: 5,
    }
}

#[rustfmt::skip]
const WIRE_MISC: &[u8] = &[
    /* magic */    0x00, 0x00, 0xca, 0xfe,
    /* unused */   0x00, 0x00, 0x00,
    /* count */    0xff, 0xff, 0xff, 0xff,
    /* len */      0x00, 0x00, 0x00, 0x02,
    /* data */     0x01, 0x02, 0x03, 0x04,
    /* blob_len */ 0x00, 0x00, 0x00, 0x03,
    /* blob */     0x09, 0x08, 0x07,
    /* fixed */    b'a', b'b', b'c',
    /* limit */    0x00, 0x00, 0x00, 0x05,
    QEMU_VM_EOF,
];

#[test]
fn buffers_varrays_and_checks() {
    assert_eq!(save_vmstate(&VMSTATE_MISC, &mut misc()), WIRE_MISC);

    let mut obj = Misc { magic: 0xcafe, count: -1, limit: 10, ..Default::default() };
    let mut wire = WIRE_MISC.to_vec();
    wire[4..7].copy_from_slice(&[1, 2, 3]); // whatever is in the padding is ignored
    load_vmstate(&VMSTATE_MISC, &mut obj, 1, &wire).unwrap();
    assert_eq!(obj, misc());
}

#[test]
fn equal_mismatch() {
    let mut obj = Misc { magic: 0xcafe, count: 0, ..Default::default() };
    let e = load_vmstate_one(&VMSTATE_MISC, &mut obj, 1, WIRE_MISC).unwrap_err();
    assert_eq!(e, "0 != ffffffff");

    let mut f = StreamReader::new(WIRE_MISC);
    let mut obj = Misc { magic: 1, ..Default::default() };
    vmstate_load_state(&mut f, &VMSTATE_MISC, &mut obj, 1).unwrap_err();
    assert_eq!(f.get_error(), -EINVAL);
}

#[test]
fn int32_le_rejects_larger_value() {
    let mut obj = Misc { magic: 0xcafe, count: -1, limit: 4, ..Default::default() };
    let e = load_vmstate_one(&VMSTATE_MISC, &mut obj, 1, WIRE_MISC).unwrap_err();
    assert_eq!(e, "Invalid value 5 expecting positive value <= 4");
}

#[test]
fn validate_failure() {
    let mut wire = WIRE_MISC.to_vec();
    wire[14] = 5;
    let mut obj = Misc { magic: 0xcafe, count: -1, limit: 10, ..Default::default() };
    let mut f = StreamReader::new(&wire);
    let e = vmstate_load_state(&mut f, &VMSTATE_MISC, &mut obj, 1).unwrap_err();
    assert_eq!(e.message(), "Input validation failed: misc/len_ok version_id: 1");
}

#[test]
fn hook_errors() {
    static PRE_LOAD: Vmsd<u32> =
        LazyLock::new(|| VmStateDescription::new("hooked").version_id(2).pre_load(|_| -22));
    static POST_LOAD: Vmsd<u32> = LazyLock::new(|| {
        VmStateDescription::new("hooked")
            .version_id(2)
            .minimum_version_id(1)
            .post_load_errp(|_, v| Err(ruvm_base::err!("bad version {v}")))
    });
    static PRE_SAVE: Vmsd<u32> =
        LazyLock::new(|| VmStateDescription::new("hooked").pre_save(|_| -1));

    let mut f = StreamReader::new(&[]);
    let e = vmstate_load_state(&mut f, &PRE_LOAD, &mut 0, 2).unwrap_err();
    assert_eq!(
        e.message(),
        "pre load hook failed for: 'hooked', version_id: 2, minimum version_id: 0, ret: -22"
    );
    let e = vmstate_load_state(&mut f, &POST_LOAD, &mut 0, 1).unwrap_err();
    assert_eq!(
        e.message(),
        "post load hook failed for: hooked, version_id: 2, minimum_version: 1: bad version 1"
    );
    let e = vmstate_save_state(&mut StreamWriter::new(), &PRE_SAVE, &mut 0).unwrap_err();
    assert_eq!(e.message(), "pre-save failed: hooked");
}

#[test]
fn post_save_runs_after_a_failed_save() {
    static SHORT: Vmsd<(bool, Vec<u8>)> = LazyLock::new(|| {
        VmStateDescription::new("short")
            .post_save(|s: &mut (bool, Vec<u8>)| s.0 = true)
            .field(VmStateField::partial_buffer("buf", 4, |s: &mut (bool, Vec<u8>)| &mut s.1))
    });
    let mut state = (false, vec![1, 2]);
    let e = vmstate_save_state(&mut StreamWriter::new(), &SHORT, &mut state).unwrap_err();
    assert_eq!(e.message(), "Save of field short/buf failed: buffer of 2 bytes cannot supply 4");
    assert!(state.0);
}

#[derive(Default)]
struct DescSeg {
    selector: u32,
    base: u64,
}

#[derive(Default)]
struct DescCpu {
    segs: [DescSeg; 3],
    regs: [u64; 2],
    flags: u32,
    extra: u8,
}

static VMSTATE_DESC_SEG: Vmsd<DescSeg> = LazyLock::new(|| {
    VmStateDescription::new("segment").version_id(1).minimum_version_id(1).fields([
        VmStateField::scalar("selector", |s: &mut DescSeg| &mut s.selector),
        VmStateField::scalar("base", |s: &mut DescSeg| &mut s.base),
    ])
});

static VMSTATE_DESC_CPU_EXTRA: Vmsd<DescCpu> = LazyLock::new(|| {
    VmStateDescription::new("cpu/extra")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &DescCpu| s.extra != 0)
        .field(VmStateField::scalar("extra", |s: &mut DescCpu| &mut s.extra))
});

static VMSTATE_DESC_CPU: Vmsd<DescCpu> = LazyLock::new(|| {
    VmStateDescription::new("cpu")
        .version_id(3)
        .minimum_version_id(1)
        .fields([
            VmStateField::struct_array("segs", &VMSTATE_DESC_SEG, |s: &mut DescCpu| &mut s.segs),
            VmStateField::array("regs", |s: &mut DescCpu| &mut s.regs),
            VmStateField::scalar("flags", |s: &mut DescCpu| &mut s.flags),
            VmStateField::scalar("flags", |s: &mut DescCpu| &mut s.flags).test(|_, _| true),
        ])
        .subsection(&VMSTATE_DESC_CPU_EXTRA)
});

/// The vmdesc follows `vmstate_save_vmsd_v()`: compressed arrays, nested structs, duplicate
/// names numbered, a test field described per element and subsections only when sent.
#[test]
fn vmdesc_describes_what_went_out() {
    let mut cpu = DescCpu { extra: 1, ..DescCpu::default() };
    let mut f = StreamWriter::new();
    let mut desc = ruvm_vmstate::JsonWriter::new();
    desc.start_object(None);
    ruvm_vmstate::vmstate_save_state_vmdesc(&mut f, &VMSTATE_DESC_CPU, &mut cpu, Some(&mut desc))
        .unwrap();
    desc.end_object();
    assert_eq!(
        desc.as_str(),
        concat!(
            r#"{"vmsd_name": "cpu", "version": 3, "fields": ["#,
            r#"{"name": "segs", "array_len": 3, "type": "struct", "struct": {"#,
            r#""vmsd_name": "segment", "version": 1, "fields": ["#,
            r#"{"name": "selector", "type": "uint32", "size": 4}, "#,
            r#"{"name": "base", "type": "uint64", "size": 8}]}, "size": 12}, "#,
            r#"{"name": "regs", "array_len": 2, "type": "uint64", "size": 8}, "#,
            r#"{"name": "flags[0]", "type": "uint32", "size": 4}, "#,
            r#"{"name": "flags[1]", "type": "uint32", "size": 4}], "#,
            r#""subsections": [{"vmsd_name": "cpu/extra", "version": 1, "fields": ["#,
            r#"{"name": "extra", "type": "uint8", "size": 1}]}]}"#,
        )
    );
    // 3 * 12 + 2 * 8 + 2 * 4, then the subsection header and its byte.
    assert_eq!(f.as_bytes().len(), 60 + 1 + 1 + 9 + 4 + 1);
}
