// SPDX-License-Identifier: GPL-2.0-or-later

//! The AML byte code builder from hw/acpi/aml-build.c.
//!
//! This follows the C code closely on purpose. The expected tables in `vendor-qemu/acpi-expected`
//! are compared byte for byte, so every encoding choice QEMU makes (which integer prefix, where a
//! package length goes, how a resource template ends) has to be made the same way here. The
//! functions drop the `aml_` prefix, and the few that clash with Rust keywords get a trailing
//! underscore: [`if_`], [`else_`], [`while_`], [`return_`] and [`break_`].
//!
//! An [`Aml`] is a term under construction. Appending a child copies the child's encoding into
//! the parent, the same as `aml_append()`, so one term can be appended in several places.

/// `AmlBlockFlags`, how a term is framed when it is appended to its parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Block {
    /// Only data.
    NoOpcode,
    /// An opcode optionally followed by data.
    Opcode,
    /// An opcode and a PkgLength.
    Package,
    /// Like `Package` with the 0x5B ExtOpPrefix in front.
    ExtPackage,
    /// Data encoded as a DefBuffer.
    Buffer,
    /// A ResourceTemplate, a buffer that ends with an EndTag.
    ResTemplate,
}

/// One AML term.
#[derive(Clone, Debug)]
pub struct Aml {
    op: u8,
    block: Block,
    buf: Vec<u8>,
}

impl Default for Aml {
    fn default() -> Self {
        Self::new()
    }
}

impl Aml {
    /// An empty term that only holds data, what `init_aml_allocator()` hands back for a table.
    pub fn new() -> Self {
        Self { op: 0, block: Block::NoOpcode, buf: Vec::new() }
    }

    fn opcode(op: u8) -> Self {
        Self { op, block: Block::Opcode, buf: Vec::new() }
    }

    fn bundle(op: u8, block: Block) -> Self {
        Self { op, block, buf: Vec::new() }
    }

    fn raw(buf: Vec<u8>) -> Self {
        Self { op: 0, block: Block::NoOpcode, buf }
    }

    /// `aml_append()`.
    pub fn append(&mut self, child: &Aml) {
        let mut buf = child.buf.clone();
        match child.block {
            Block::Opcode => self.buf.push(child.op),
            Block::ExtPackage => {
                build_package(&mut buf, child.op);
                buf.insert(0, 0x5B);
            }
            Block::Package => build_package(&mut buf, child.op),
            Block::ResTemplate => {
                // EndTag. A zero checksum counts as a good one (ACPI 1.0b, 6.4.2.8).
                buf.extend_from_slice(&[0x79, 0]);
                build_buffer(&mut buf, child.op);
            }
            Block::Buffer => build_buffer(&mut buf, child.op),
            Block::NoOpcode => {}
        }
        self.buf.extend_from_slice(&buf);
    }

    /// The encoded bytes, without this term's own framing.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Takes the encoded bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

const PACKAGE_LENGTH_1BYTE_SHIFT: u32 = 6;
const PACKAGE_LENGTH_2BYTE_SHIFT: u32 = 4;
const PACKAGE_LENGTH_3BYTE_SHIFT: u32 = 12;
const PACKAGE_LENGTH_4BYTE_SHIFT: u32 = 20;

/// `build_prepend_package_length()`, returned as the bytes to put in front.
fn package_length(length: usize, incl_self: bool) -> Vec<u8> {
    let mut length = u32::try_from(length).expect("AML package fits in 32 bits");
    let length_bytes: u32 = if length + 1 < (1 << PACKAGE_LENGTH_1BYTE_SHIFT) {
        1
    } else if length + 2 < (1 << PACKAGE_LENGTH_3BYTE_SHIFT) {
        2
    } else if length + 3 < (1 << PACKAGE_LENGTH_4BYTE_SHIFT) {
        3
    } else {
        4
    };
    // A NamedField uses the PkgLength encoding without counting the PkgLength itself.
    if incl_self {
        length += length_bytes;
    }
    if length_bytes == 1 {
        return vec![length as u8];
    }
    let mut tail = Vec::with_capacity(3);
    if length_bytes == 4 {
        tail.push((length >> PACKAGE_LENGTH_4BYTE_SHIFT) as u8);
        length &= (1 << PACKAGE_LENGTH_4BYTE_SHIFT) - 1;
    }
    if length_bytes >= 3 {
        tail.push((length >> PACKAGE_LENGTH_3BYTE_SHIFT) as u8);
        length &= (1 << PACKAGE_LENGTH_3BYTE_SHIFT) - 1;
    }
    tail.push((length >> PACKAGE_LENGTH_2BYTE_SHIFT) as u8);
    length &= (1 << PACKAGE_LENGTH_2BYTE_SHIFT) - 1;
    // The top two bits of the lead byte say how many bytes follow it.
    let mut out = vec![(((length_bytes - 1) << PACKAGE_LENGTH_1BYTE_SHIFT) | length) as u8];
    out.extend(tail.iter().rev());
    out
}

fn prepend(buf: &mut Vec<u8>, bytes: &[u8]) {
    buf.splice(0..0, bytes.iter().copied());
}

fn build_package(buf: &mut Vec<u8>, op: u8) {
    let len = package_length(buf.len(), true);
    prepend(buf, &len);
    buf.insert(0, op);
}

fn build_buffer(buf: &mut Vec<u8>, op: u8) {
    let mut size = Vec::new();
    append_int(&mut size, buf.len() as u64);
    prepend(buf, &size);
    build_package(buf, op);
}

/// `build_append_int_noprefix()`: `size` little endian bytes of `value`.
pub fn append_int_noprefix(buf: &mut Vec<u8>, value: u64, size: usize) {
    let mut value = value;
    for _ in 0..size {
        buf.push(value as u8);
        value >>= 8;
    }
}

/// `build_append_int()`: the shortest of ZeroOp, OneOp and the byte, word, dword and qword forms.
fn append_int(buf: &mut Vec<u8>, value: u64) {
    match value {
        0 => buf.push(0x00),
        1 => buf.push(0x01),
        2..=0xFF => {
            buf.push(0x0A);
            append_int_noprefix(buf, value, 1);
        }
        0x100..=0xFFFF => {
            buf.push(0x0B);
            append_int_noprefix(buf, value, 2);
        }
        0x1_0000..=0xFFFF_FFFF => {
            buf.push(0x0C);
            append_int_noprefix(buf, value, 4);
        }
        _ => {
            buf.push(0x0E);
            append_int_noprefix(buf, value, 8);
        }
    }
}

const ACPI_NAMESEG_LEN: usize = 4;

fn append_nameseg(buf: &mut Vec<u8>, seg: &str) {
    assert!(seg.len() <= ACPI_NAMESEG_LEN, "name segment {seg:?} is longer than 4");
    buf.extend_from_slice(seg.as_bytes());
    buf.extend(std::iter::repeat_n(b'_', ACPI_NAMESEG_LEN - seg.len()));
}

/// `build_append_namestring()`.
fn append_namestring(buf: &mut Vec<u8>, name: &str) {
    let segs: Vec<&str> = name.split('.').collect();
    assert!(!segs.is_empty() && segs.len() <= 255, "bad AML name {name:?}");
    // RootPath or PrefixPath.
    let first = segs[0].trim_start_matches(['\\', '^']);
    buf.extend_from_slice(&segs[0].as_bytes()[..segs[0].len() - first.len()]);
    match segs.len() {
        1 if first.is_empty() => buf.push(0x00), // NullName
        1 => append_nameseg(buf, first),
        2 => {
            buf.push(0x2E); // DualNamePrefix
            append_nameseg(buf, first);
            append_nameseg(buf, segs[1]);
        }
        n => {
            buf.push(0x2F); // MultiNamePrefix
            buf.push(n as u8);
            append_nameseg(buf, first);
            for seg in &segs[1..] {
                append_nameseg(buf, seg);
            }
        }
    }
}

/// `build_append_named_dword()`: `Name(XXXX, 0x00000000)` with the zero encoded as a dword.
/// Returns the offset of the dword so it can be patched later.
pub fn append_named_dword(buf: &mut Vec<u8>, name: &str) -> usize {
    buf.push(0x08); // NameOp
    append_namestring(buf, name);
    buf.push(0x0C); // DWordPrefix
    let offset = buf.len();
    append_int_noprefix(buf, 0, 4);
    offset
}

/// `AmlIODecode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum IoDecode {
    Decode10 = 0,
    Decode16 = 1,
}

/// `AmlAccessType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AccessType {
    Any = 0,
    Byte = 1,
    Word = 2,
    Dword = 3,
    Qword = 4,
    Buffer = 5,
}

/// `AmlLockRule`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LockRule {
    NoLock = 0,
    Lock = 1,
}

/// `AmlUpdateRule`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum UpdateRule {
    Preserve = 0,
    WriteAsOnes = 1,
    WriteAsZeros = 2,
}

/// `AmlAddressSpace`, as used in a Generic Address Structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AddressSpace {
    SystemMemory = 0x00,
    SystemIo = 0x01,
    PciConfig = 0x02,
    EmbeddedCtrl = 0x03,
    Smbus = 0x04,
    Ffh = 0x7F,
}

/// `AmlRegionSpace`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RegionSpace {
    SystemMemory = 0x00,
    SystemIo = 0x01,
    PciConfig = 0x02,
}

/// `AmlResourceType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResourceType {
    Memory = 0,
    Io = 1,
    BusNumber = 2,
}

/// `AmlDecode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Decode {
    Pos = 0,
    Sub = 1 << 1,
}

/// `AmlMinFixed`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MinFixed {
    NotFixed = 0,
    Fixed = 1 << 2,
}

/// `AmlMaxFixed`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MaxFixed {
    NotFixed = 0,
    Fixed = 1 << 3,
}

/// `AmlISARanges`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum IsaRanges {
    IsaOnly = 1,
    NonIsaOnly = 2,
    EntireRange = 3,
}

/// `AmlCacheable`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cacheable {
    NonCacheable = 0,
    Cacheable = 1,
    WriteCombining = 2,
    Prefetchable = 3,
}

/// `AmlReadAndWrite`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ReadWrite {
    ReadOnly = 0,
    ReadWrite = 1,
}

/// `AmlConsumerAndProducer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ConsumerProducer {
    ConsumerProducer = 0,
    Consumer = 1,
}

/// `AmlLevelAndEdge`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Trigger {
    Level = 0,
    Edge = 1,
}

/// `AmlActiveHighAndLow`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Polarity {
    ActiveHigh = 0,
    ActiveLow = 1,
}

/// `AmlShared`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Shared {
    Exclusive = 0,
    Shared = 1,
    ExclusiveAndWake = 2,
    SharedAndWake = 3,
}

/// `AmlSerializeFlag`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Serialize {
    NotSerialized = 0,
    Serialized = 1,
}

/// `AmlDmaType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DmaType {
    Compatibility = 0,
    TypeA = 1,
    TypeB = 2,
    TypeF = 3,
}

/// `AmlDmaBusMaster`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DmaBusMaster {
    NotBusMaster = 0,
    BusMaster = 1,
}

/// `AmlTransferSize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TransferSize {
    Transfer8 = 0,
    Transfer8And16 = 1,
    Transfer16 = 2,
}

fn with_args(op: u8, args: &[&Aml]) -> Aml {
    let mut var = Aml::opcode(op);
    for a in args {
        var.append(a);
    }
    var
}

/// `build_opcode_2arg_dst()`: "Op Operand Operand Target", with NullName when there is no target.
fn opcode_2arg_dst(op: u8, arg1: &Aml, arg2: &Aml, dst: Option<&Aml>) -> Aml {
    let mut var = with_args(op, &[arg1, arg2]);
    match dst {
        Some(d) => var.append(d),
        None => var.buf.push(0x00),
    }
    var
}

/// `aml_scope()`, DefScope.
pub fn scope(name: &str) -> Aml {
    let mut var = Aml::bundle(0x10, Block::Package);
    append_namestring(&mut var.buf, name);
    var
}

/// `aml_return()`.
pub fn return_(val: &Aml) -> Aml {
    with_args(0xA4, &[val])
}

/// `aml_debug()`, the Debug object.
pub fn debug() -> Aml {
    Aml::raw(vec![0x5B, 0x31])
}

/// `aml_int()`: ZeroOp, OneOp or the shortest constant that holds `val`.
pub fn int(val: u64) -> Aml {
    let mut var = Aml::new();
    append_int(&mut var.buf, val);
    var
}

/// `aml_name()`, a NameString.
pub fn name(name: &str) -> Aml {
    let mut var = Aml::new();
    append_namestring(&mut var.buf, name);
    var
}

/// `aml_name_decl()`, DefName.
pub fn name_decl(name: &str, val: &Aml) -> Aml {
    let mut var = Aml::opcode(0x08);
    append_namestring(&mut var.buf, name);
    var.append(val);
    var
}

/// `aml_arg()`, Arg0 to Arg6.
pub fn arg(pos: u8) -> Aml {
    assert!(pos <= 6, "there is no Arg{pos}");
    Aml::opcode(0x68 + pos)
}

/// `aml_local()`, Local0 to Local7.
pub fn local(num: u8) -> Aml {
    assert!(num <= 7, "there is no Local{num}");
    Aml::opcode(0x60 + num)
}

/// `aml_to_integer()`.
pub fn to_integer(arg: &Aml) -> Aml {
    let mut var = with_args(0x99, &[arg]);
    var.buf.push(0x00);
    var
}

fn convert(op: u8, src: &Aml, dst: Option<&Aml>) -> Aml {
    let mut var = with_args(op, &[src]);
    match dst {
        Some(d) => var.append(d),
        None => var.buf.push(0x00),
    }
    var
}

/// `aml_to_hexstring()`.
pub fn to_hexstring(src: &Aml, dst: Option<&Aml>) -> Aml {
    convert(0x98, src, dst)
}

/// `aml_to_buffer()`.
pub fn to_buffer(src: &Aml, dst: Option<&Aml>) -> Aml {
    convert(0x96, src, dst)
}

/// `aml_to_decimalstring()`.
pub fn to_decimalstring(src: &Aml, dst: Option<&Aml>) -> Aml {
    convert(0x97, src, dst)
}

/// `aml_store()`.
pub fn store(val: &Aml, target: &Aml) -> Aml {
    with_args(0x70, &[val, target])
}

/// `aml_and()`.
pub fn and(arg1: &Aml, arg2: &Aml, dst: Option<&Aml>) -> Aml {
    opcode_2arg_dst(0x7B, arg1, arg2, dst)
}

/// `aml_or()`.
pub fn or(arg1: &Aml, arg2: &Aml, dst: Option<&Aml>) -> Aml {
    opcode_2arg_dst(0x7D, arg1, arg2, dst)
}

/// `aml_land()`.
pub fn land(arg1: &Aml, arg2: &Aml) -> Aml {
    with_args(0x90, &[arg1, arg2])
}

/// `aml_lor()`.
pub fn lor(arg1: &Aml, arg2: &Aml) -> Aml {
    with_args(0x91, &[arg1, arg2])
}

/// `aml_shiftleft()`.
pub fn shiftleft(arg1: &Aml, count: &Aml) -> Aml {
    opcode_2arg_dst(0x79, arg1, count, None)
}

/// `aml_shiftright()`.
pub fn shiftright(arg1: &Aml, count: &Aml, dst: Option<&Aml>) -> Aml {
    opcode_2arg_dst(0x7A, arg1, count, dst)
}

/// `aml_lless()`.
pub fn lless(arg1: &Aml, arg2: &Aml) -> Aml {
    with_args(0x95, &[arg1, arg2])
}

/// `aml_add()`.
pub fn add(arg1: &Aml, arg2: &Aml, dst: Option<&Aml>) -> Aml {
    opcode_2arg_dst(0x72, arg1, arg2, dst)
}

/// `aml_subtract()`.
pub fn subtract(arg1: &Aml, arg2: &Aml, dst: Option<&Aml>) -> Aml {
    opcode_2arg_dst(0x74, arg1, arg2, dst)
}

/// `aml_increment()`.
pub fn increment(arg: &Aml) -> Aml {
    with_args(0x75, &[arg])
}

/// `aml_decrement()`.
pub fn decrement(arg: &Aml) -> Aml {
    with_args(0x76, &[arg])
}

/// `aml_index()`.
pub fn index(arg1: &Aml, idx: &Aml) -> Aml {
    opcode_2arg_dst(0x88, arg1, idx, None)
}

/// `aml_notify()`.
pub fn notify(arg1: &Aml, arg2: &Aml) -> Aml {
    with_args(0x86, &[arg1, arg2])
}

/// `aml_break()`.
pub fn break_() -> Aml {
    Aml::opcode(0xA5)
}

/// `aml_call0()` through `aml_call6()`: a method invocation is its name followed by the arguments.
pub fn call(method: &str, args: &[&Aml]) -> Aml {
    assert!(args.len() <= 7, "a method takes at most 7 arguments");
    let mut var = Aml::new();
    append_namestring(&mut var.buf, method);
    for a in args {
        var.append(a);
    }
    var
}

/// `aml_memory32_fixed()`.
pub fn memory32_fixed(addr: u32, size: u32, rw: ReadWrite) -> Aml {
    let mut buf = vec![0x86, 9, 0, rw as u8];
    buf.extend_from_slice(&addr.to_le_bytes());
    buf.extend_from_slice(&size.to_le_bytes());
    Aml::raw(buf)
}

/// `aml_interrupt()`, the Extended Interrupt descriptor.
pub fn interrupt(
    con: ConsumerProducer,
    trigger: Trigger,
    polarity: Polarity,
    shared: Shared,
    irqs: &[u32],
) -> Aml {
    assert!(!irqs.is_empty() && irqs.len() <= 255, "bad interrupt list");
    let flags = con as u8 | (trigger as u8) << 1 | (polarity as u8) << 2 | (shared as u8) << 3;
    let len = (2 + irqs.len() * 4) as u16;
    let mut buf = vec![0x89];
    buf.extend_from_slice(&len.to_le_bytes());
    buf.push(flags);
    buf.push(irqs.len() as u8);
    for irq in irqs {
        buf.extend_from_slice(&irq.to_le_bytes());
    }
    Aml::raw(buf)
}

/// `aml_io()`, the I/O port descriptor.
pub fn io(dec: IoDecode, min_base: u16, max_base: u16, aln: u8, len: u8) -> Aml {
    let mut buf = vec![0x47, dec as u8];
    buf.extend_from_slice(&min_base.to_le_bytes());
    buf.extend_from_slice(&max_base.to_le_bytes());
    buf.extend_from_slice(&[aln, len]);
    Aml::raw(buf)
}

/// `aml_irq_no_flags()`, the two byte IRQ descriptor.
pub fn irq_no_flags(irq: u8) -> Aml {
    assert!(irq < 16, "ISA irq {irq} out of range");
    let mask = 1u16 << irq;
    let mut buf = vec![0x22];
    buf.extend_from_slice(&mask.to_le_bytes());
    Aml::raw(buf)
}

/// `aml_irq()`, the three byte IRQ descriptor.
pub fn irq(irq: u8, trigger: Trigger, polarity: Polarity, shared: Shared) -> Aml {
    assert!(
        (trigger == Trigger::Edge && polarity == Polarity::ActiveHigh)
            || (trigger == Trigger::Level && polarity == Polarity::ActiveLow),
        "ISA irq descriptors are edge high or level low"
    );
    assert!(irq < 16, "ISA irq {irq} out of range");
    let mask = 1u16 << irq;
    let flags = trigger as u8 | (polarity as u8) << 3 | (shared as u8) << 4;
    let mut buf = vec![0x23];
    buf.extend_from_slice(&mask.to_le_bytes());
    buf.push(flags);
    Aml::raw(buf)
}

/// `aml_lnot()`.
pub fn lnot(arg: &Aml) -> Aml {
    with_args(0x92, &[arg])
}

/// `aml_equal()`, LEqual.
pub fn equal(arg1: &Aml, arg2: &Aml) -> Aml {
    with_args(0x93, &[arg1, arg2])
}

/// `aml_lgreater()`.
pub fn lgreater(arg1: &Aml, arg2: &Aml) -> Aml {
    with_args(0x94, &[arg1, arg2])
}

/// `aml_lgreater_equal()`, encoded as LNot LLess.
pub fn lgreater_equal(arg1: &Aml, arg2: &Aml) -> Aml {
    let mut var = Aml::opcode(0x92);
    var.buf.push(0x95);
    var.append(arg1);
    var.append(arg2);
    var
}

/// `aml_if()`.
pub fn if_(predicate: &Aml) -> Aml {
    let mut var = Aml::bundle(0xA0, Block::Package);
    var.append(predicate);
    var
}

/// `aml_else()`.
pub fn else_() -> Aml {
    Aml::bundle(0xA1, Block::Package)
}

/// `aml_while()`.
pub fn while_(predicate: &Aml) -> Aml {
    let mut var = Aml::bundle(0xA2, Block::Package);
    var.append(predicate);
    var
}

/// `aml_method()`.
pub fn method(name: &str, arg_count: u8, serialize: Serialize) -> Aml {
    assert!(arg_count < 8, "a method takes at most 7 arguments");
    let mut var = Aml::bundle(0x14, Block::Package);
    append_namestring(&mut var.buf, name);
    var.buf.push(arg_count | (serialize as u8) << 3);
    var
}

/// `aml_device()`.
pub fn device(name: &str) -> Aml {
    let mut var = Aml::bundle(0x82, Block::ExtPackage);
    append_namestring(&mut var.buf, name);
    var
}

/// `aml_resource_template()`.
pub fn resource_template() -> Aml {
    Aml::bundle(0x11, Block::ResTemplate)
}

/// `aml_buffer()`. `None` gives a zero filled buffer of `size` bytes.
pub fn buffer(size: usize, bytes: Option<&[u8]>) -> Aml {
    let mut var = Aml::bundle(0x11, Block::Buffer);
    match bytes {
        Some(b) => var.buf.extend_from_slice(&b[..size]),
        None => var.buf.resize(size, 0),
    }
    var
}

/// `aml_package()`.
pub fn package(num_elements: u8) -> Aml {
    let mut var = Aml::bundle(0x12, Block::Package);
    var.buf.push(num_elements);
    var
}

/// `aml_varpackage()`.
pub fn varpackage(num_elements: u32) -> Aml {
    let mut var = Aml::bundle(0x13, Block::Package);
    append_int(&mut var.buf, num_elements.into());
    var
}

/// `aml_operation_region()`.
pub fn operation_region(name: &str, rs: RegionSpace, offset: &Aml, len: u32) -> Aml {
    let mut var = Aml::raw(vec![0x5B, 0x80]);
    append_namestring(&mut var.buf, name);
    var.buf.push(rs as u8);
    var.append(offset);
    append_int(&mut var.buf, len.into());
    var
}

/// `aml_named_field()`. The length is in bits.
pub fn named_field(name: &str, length: usize) -> Aml {
    let mut var = Aml::new();
    append_nameseg(&mut var.buf, name);
    var.buf.extend(package_length(length, false));
    var
}

/// `aml_reserved_field()`. The length is in bits.
pub fn reserved_field(length: usize) -> Aml {
    let mut var = Aml::raw(vec![0x00]);
    var.buf.extend(package_length(length, false));
    var
}

/// `aml_field()`.
pub fn field(name: &str, access: AccessType, lock: LockRule, rule: UpdateRule) -> Aml {
    let mut var = Aml::bundle(0x81, Block::ExtPackage);
    append_namestring(&mut var.buf, name);
    var.buf.push((rule as u8) << 5 | (lock as u8) << 4 | access as u8);
    var
}

fn create_field_common(op: u8, srcbuf: &Aml, index: &Aml, name: &str) -> Aml {
    let mut var = with_args(op, &[srcbuf, index]);
    append_namestring(&mut var.buf, name);
    var
}

/// `aml_create_field()`.
pub fn create_field(srcbuf: &Aml, bit_index: &Aml, num_bits: &Aml, name: &str) -> Aml {
    let mut var = Aml::raw(vec![0x5B, 0x13]);
    var.append(srcbuf);
    var.append(bit_index);
    var.append(num_bits);
    append_namestring(&mut var.buf, name);
    var
}

/// `aml_create_dword_field()`.
pub fn create_dword_field(srcbuf: &Aml, index: &Aml, name: &str) -> Aml {
    create_field_common(0x8A, srcbuf, index, name)
}

/// `aml_create_qword_field()`.
pub fn create_qword_field(srcbuf: &Aml, index: &Aml, name: &str) -> Aml {
    create_field_common(0x8F, srcbuf, index, name)
}

/// `aml_string()`.
pub fn string(s: &str) -> Aml {
    let mut var = Aml::opcode(0x0D);
    var.buf.extend_from_slice(s.as_bytes());
    var.buf.push(0);
    var
}

/// `aml_processor()`.
pub fn processor(proc_id: u8, pblk_addr: u32, pblk_len: u8, name: &str) -> Aml {
    let mut var = Aml::bundle(0x83, Block::ExtPackage);
    append_namestring(&mut var.buf, name);
    var.buf.push(proc_id);
    var.buf.extend_from_slice(&pblk_addr.to_le_bytes());
    var.buf.push(pblk_len);
    var
}

fn hex_digit(c: u8) -> u32 {
    if c >= b'A' { u32::from(c - b'A' + 10) } else { u32::from(c.wrapping_sub(b'0')) }
}

fn hex_byte(s: &[u8]) -> u8 {
    (hex_digit(s[0]) << 4 | hex_digit(s[1])) as u8
}

/// `aml_eisaid()`: a seven character EISA id such as `PNP0501` packed into a dword.
pub fn eisaid(s: &str) -> Aml {
    let b = s.as_bytes();
    assert_eq!(b.len(), 7, "EISA id {s:?} is not 7 characters");
    let id = u32::from(b[0] - 0x40) << 26
        | u32::from(b[1] - 0x40) << 21
        | u32::from(b[2] - 0x40) << 16
        | hex_digit(b[3]) << 12
        | hex_digit(b[4]) << 8
        | hex_digit(b[5]) << 4
        | hex_digit(b[6]);
    let mut buf = vec![0x0C];
    buf.extend_from_slice(&id.to_be_bytes());
    Aml::raw(buf)
}

#[allow(clippy::too_many_arguments)]
fn address_space_desc(
    tag: u8,
    width: usize,
    ty: ResourceType,
    min_fixed: MinFixed,
    max_fixed: MaxFixed,
    dec: Decode,
    values: [u64; 5],
    type_flags: u8,
) -> Aml {
    let len: u16 = match width {
        2 => 0x0D,
        4 => 23,
        _ => 0x2B,
    };
    let mut buf = vec![tag];
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&[ty as u8, max_fixed as u8 | min_fixed as u8 | dec as u8, type_flags]);
    for v in values {
        append_int_noprefix(&mut buf, v, width);
    }
    Aml::raw(buf)
}

/// `aml_word_bus_number()`.
#[allow(clippy::too_many_arguments)]
pub fn word_bus_number(
    min_fixed: MinFixed,
    max_fixed: MaxFixed,
    dec: Decode,
    gran: u16,
    min: u16,
    max: u16,
    trans: u16,
    len: u16,
) -> Aml {
    let v = [gran, min, max, trans, len].map(u64::from);
    address_space_desc(0x88, 2, ResourceType::BusNumber, min_fixed, max_fixed, dec, v, 0)
}

/// `aml_word_io()`.
#[allow(clippy::too_many_arguments)]
pub fn word_io(
    min_fixed: MinFixed,
    max_fixed: MaxFixed,
    dec: Decode,
    isa: IsaRanges,
    gran: u16,
    min: u16,
    max: u16,
    trans: u16,
    len: u16,
) -> Aml {
    let v = [gran, min, max, trans, len].map(u64::from);
    address_space_desc(0x88, 2, ResourceType::Io, min_fixed, max_fixed, dec, v, isa as u8)
}

/// `aml_dword_io()`.
#[allow(clippy::too_many_arguments)]
pub fn dword_io(
    min_fixed: MinFixed,
    max_fixed: MaxFixed,
    dec: Decode,
    isa: IsaRanges,
    gran: u32,
    min: u32,
    max: u32,
    trans: u32,
    len: u32,
) -> Aml {
    let v = [gran, min, max, trans, len].map(u64::from);
    address_space_desc(0x87, 4, ResourceType::Io, min_fixed, max_fixed, dec, v, isa as u8)
}

/// `aml_dword_memory()`.
#[allow(clippy::too_many_arguments)]
pub fn dword_memory(
    dec: Decode,
    min_fixed: MinFixed,
    max_fixed: MaxFixed,
    cacheable: Cacheable,
    rw: ReadWrite,
    gran: u32,
    min: u32,
    max: u32,
    trans: u32,
    len: u32,
) -> Aml {
    let v = [gran, min, max, trans, len].map(u64::from);
    let flags = rw as u8 | (cacheable as u8) << 1;
    address_space_desc(0x87, 4, ResourceType::Memory, min_fixed, max_fixed, dec, v, flags)
}

/// `aml_qword_memory()`.
#[allow(clippy::too_many_arguments)]
pub fn qword_memory(
    dec: Decode,
    min_fixed: MinFixed,
    max_fixed: MaxFixed,
    cacheable: Cacheable,
    rw: ReadWrite,
    gran: u64,
    min: u64,
    max: u64,
    trans: u64,
    len: u64,
) -> Aml {
    let flags = rw as u8 | (cacheable as u8) << 1;
    let v = [gran, min, max, trans, len];
    address_space_desc(0x8A, 8, ResourceType::Memory, min_fixed, max_fixed, dec, v, flags)
}

/// `aml_dma()`.
pub fn dma(ty: DmaType, bm: DmaBusMaster, sz: TransferSize, channel: u8) -> Aml {
    Aml::raw(vec![0x2A, 1 << channel, sz as u8 | (bm as u8) << 2 | (ty as u8) << 5])
}

/// `aml_sleep()`.
pub fn sleep(msec: u64) -> Aml {
    let mut var = Aml::raw(vec![0x5B, 0x22]);
    var.append(&int(msec));
    var
}

/// `aml_touuid()`: a UUID string turned into the mixed endian buffer ToUUID produces.
pub fn touuid(uuid: &str) -> Aml {
    let u = uuid.as_bytes();
    assert_eq!(u.len(), 36, "bad UUID {uuid:?}");
    let mut var = Aml::bundle(0x11, Block::Buffer);
    for off in [6, 4, 2, 0, 11, 9, 16, 14, 19, 21, 24, 26, 28, 30, 32, 34] {
        var.buf.push(hex_byte(&u[off..]));
    }
    var
}

/// `aml_unicode()`: a buffer holding the string as UTF-16LE with its terminating NUL.
pub fn unicode(s: &str) -> Aml {
    let mut var = Aml::bundle(0x11, Block::Buffer);
    for &c in s.as_bytes().iter().chain(std::iter::once(&0)) {
        var.buf.extend_from_slice(&[c, 0]);
    }
    var
}

/// `aml_refof()`.
pub fn refof(arg: &Aml) -> Aml {
    with_args(0x71, &[arg])
}

/// `aml_derefof()`.
pub fn derefof(arg: &Aml) -> Aml {
    with_args(0x83, &[arg])
}

/// `aml_sizeof()`.
pub fn sizeof(arg: &Aml) -> Aml {
    with_args(0x87, &[arg])
}

/// `aml_mutex()`.
pub fn mutex(name: &str, sync_level: u8) -> Aml {
    let mut var = Aml::raw(vec![0x5B, 0x01]);
    append_namestring(&mut var.buf, name);
    var.buf.push(sync_level);
    var
}

/// `aml_acquire()`.
pub fn acquire(mutex: &Aml, timeout: u16) -> Aml {
    let mut var = Aml::raw(vec![0x5B, 0x23]);
    var.append(mutex);
    var.buf.extend_from_slice(&timeout.to_le_bytes());
    var
}

/// `aml_release()`.
pub fn release(mutex: &Aml) -> Aml {
    let mut var = Aml::raw(vec![0x5B, 0x27]);
    var.append(mutex);
    var
}

/// `aml_alias()`.
pub fn alias(source: &str, alias: &str) -> Aml {
    with_args(0x06, &[&name(source), &name(alias)])
}

/// `aml_concatenate()`.
pub fn concatenate(source1: &Aml, source2: &Aml, target: Option<&Aml>) -> Aml {
    opcode_2arg_dst(0x73, source1, source2, target)
}

/// `aml_object_type()`.
pub fn object_type(object: &Aml) -> Aml {
    with_args(0x8E, &[object])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(child: &Aml) -> Vec<u8> {
        let mut root = Aml::new();
        root.append(child);
        root.into_bytes()
    }

    #[test]
    fn integers_pick_the_shortest_form() {
        assert_eq!(enc(&int(0)), [0x00]);
        assert_eq!(enc(&int(1)), [0x01]);
        assert_eq!(enc(&int(0xB)), [0x0A, 0x0B]);
        assert_eq!(enc(&int(0x1234)), [0x0B, 0x34, 0x12]);
        assert_eq!(enc(&int(0xfea0_0000)), [0x0C, 0x00, 0x00, 0xa0, 0xfe]);
        assert_eq!(enc(&int(1 << 32)), [0x0E, 0, 0, 0, 0, 1, 0, 0, 0]);
    }

    #[test]
    fn names_follow_the_namestring_grammar() {
        assert_eq!(enc(&name("PWRB")), b"PWRB");
        assert_eq!(enc(&name("RTC")), b"RTC_");
        assert_eq!(enc(&name("\\")), b"\\\0");
        assert_eq!(enc(&name("\\_SB.PCI0")), b"\\\x2e_SB_PCI0");
        assert_eq!(enc(&name("^^A.B.C")), b"^^\x2f\x03A___B___C___");
    }

    #[test]
    fn package_length_sizes() {
        assert_eq!(package_length(10, true), [11]);
        assert_eq!(package_length(62, true), [63]);
        // 63 bytes of data needs the two byte form, which then counts itself.
        assert_eq!(package_length(63, true), [0x41, 0x04]);
        assert_eq!(package_length(4093, true), [0x4F, 0xFF]);
        assert_eq!(package_length(4094, true), [0x81, 0x00, 0x01]);
        assert_eq!(package_length(1 << 20, true), [0xC4, 0x00, 0x00, 0x01]);
        // NamedField lengths do not include themselves.
        assert_eq!(package_length(32, false), [32]);
    }

    #[test]
    fn eisaid_is_big_endian_packed() {
        assert_eq!(enc(&eisaid("PNP0501")), [0x0C, 0x41, 0xD0, 0x05, 0x01]);
        assert_eq!(enc(&eisaid("PNP0C0C")), [0x0C, 0x41, 0xD0, 0x0C, 0x0C]);
    }

    #[test]
    fn resource_template_ends_with_end_tag() {
        let mut crs = resource_template();
        crs.append(&io(IoDecode::Decode16, 0x510, 0x510, 1, 0x0c));
        assert_eq!(
            enc(&crs),
            [0x11, 0x0D, 0x0A, 0x0A, 0x47, 0x01, 0x10, 0x05, 0x10, 0x05, 0x01, 0x0C, 0x79, 0x00]
        );
    }

    #[test]
    fn uuid_bytes_are_mixed_endian() {
        let b = enc(&touuid("E5C937D0-3553-4D7A-9117-EA4D19C3434D"));
        assert_eq!(&b[..4], [0x11, 0x13, 0x0A, 0x10]);
        assert_eq!(
            &b[4..],
            [
                0xD0, 0x37, 0xC9, 0xE5, 0x53, 0x35, 0x7A, 0x4D, 0x91, 0x17, 0xEA, 0x4D, 0x19, 0xC3,
                0x43, 0x4D
            ]
        );
    }
}
