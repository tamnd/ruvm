// SPDX-License-Identifier: GPL-2.0-or-later

//! `VMStateField` and the `VMSTATE_*` field macros from include/migration/vmstate.h.
//!
//! QEMU finds a field by its byte offset in the device struct and describes its shape with flags
//! (`VMS_ARRAY`, `VMS_VARRAY_UINT32`, `VMS_POINTER`, `VMS_STRUCT` and so on). Here a field reaches
//! its value through a closure that borrows it out of the state, and the shape is picked by the
//! constructor, one per macro family. The interpreter in `vmstate.rs` only sees a count of
//! elements and a way to load or save element `i`, which is also all `vmstate_load_vmsd()` needs.

use std::fmt;

use ruvm_base::{Result, bail};

use crate::file::{StreamReader, StreamWriter};
use crate::info::{Buffer, UnusedBuffer, VmStateInfo, VmStateType};
use crate::json::JsonWriter;
use crate::vmsd::VmStateDescription;
use crate::vmstate::{load_vmsd, save_vmsd_v};
use crate::{VMS_MARKER_PTR_NULL, VMS_MARKER_PTR_VALID};

type Getter<T, V> = Box<dyn Fn(&mut T) -> &mut V + Send + Sync>;
type Count<T> = Box<dyn Fn(&T) -> usize + Send + Sync>;
type Alloc<T> = Box<dyn Fn(&mut T, usize) + Send + Sync>;
fn getter<T, V: ?Sized>(get: impl Fn(&mut T) -> &mut V + Send + Sync + 'static) -> Getter<T, V> {
    Box::new(get)
}

pub(crate) type Exists<T> = Box<dyn Fn(&T, i32) -> bool + Send + Sync>;

/// One entry of a `VMStateDescription` field list.
///
/// Build one with the constructor that matches the QEMU macro, then narrow it with
/// [`version`](Self::version), [`test`](Self::test) or [`must_exist`](Self::must_exist).
pub struct VmStateField<T: 'static> {
    pub(crate) name: &'static str,
    pub(crate) version_id: i32,
    pub(crate) field_exists: Option<Exists<T>>,
    pub(crate) must_exist: bool,
    pub(crate) body: Box<dyn FieldBody<T>>,
}

impl<T: 'static> fmt::Debug for VmStateField<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VmStateField")
            .field("name", &self.name)
            .field("type", &self.body.type_name())
            .field("version_id", &self.version_id)
            .field("field_exists", &self.field_exists.is_some())
            .field("must_exist", &self.must_exist)
            .finish()
    }
}

impl<T: 'static> VmStateField<T> {
    fn with_body(name: &'static str, body: impl FieldBody<T> + 'static) -> Self {
        VmStateField {
            name,
            version_id: 0,
            field_exists: None,
            must_exist: false,
            body: Box::new(body),
        }
    }

    /// The field name as it appears in vmdesc.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The first version of the section that carries this field.
    pub fn version_id(&self) -> i32 {
        self.version_id
    }

    /// The type name vmdesc reports, `vmfield_get_type_name()`.
    pub fn type_name(&self) -> &'static str {
        self.body.type_name()
    }

    /// `vmsd_can_compress()`: whether vmdesc may describe all elements of the array with one
    /// entry.
    pub(crate) fn can_compress(&self) -> bool {
        self.field_exists.is_none() && self.body.can_compress()
    }

    /// The `_V` macro variants: the field is only in the stream from section version `version_id`
    /// on.
    pub fn version(mut self, version_id: i32) -> Self {
        self.version_id = version_id;
        self
    }

    /// The `_TEST` macro variants. When a test is set it alone decides whether the field is in the
    /// stream, and the field version is ignored, as in `vmstate_field_exists()`.
    pub fn test(mut self, test: impl Fn(&T, i32) -> bool + Send + Sync + 'static) -> Self {
        self.field_exists = Some(Box::new(test));
        self
    }

    /// `VMS_MUST_EXIST`: loading fails when the field does not exist.
    pub fn must_exist(mut self) -> Self {
        self.must_exist = true;
        self
    }

    /// `VMSTATE_SINGLE` with the info picked by the Rust type, which covers `VMSTATE_BOOL`,
    /// `VMSTATE_INT8` through `VMSTATE_INT64` and `VMSTATE_UINT8` through `VMSTATE_UINT64`.
    ///
    /// A getter that goes through a `Box` gives the `VMSTATE_POINTER` flavour.
    pub fn scalar<V: VmStateType>(
        name: &'static str,
        get: impl Fn(&mut T) -> &mut V + Send + Sync + 'static,
    ) -> Self {
        Self::single(name, V::info(), get)
    }

    /// `VMSTATE_SINGLE` with an explicit info, for example [`Uint32Equal`](crate::info::Uint32Equal).
    pub fn single<V: 'static>(
        name: &'static str,
        info: &'static dyn VmStateInfo<V>,
        get: impl Fn(&mut T) -> &mut V + Send + Sync + 'static,
    ) -> Self {
        Self::with_body(
            name,
            Single { get: getter(get), codec: InfoCodec(info), size: Size::Fixed(0), alloc: None },
        )
    }

    /// `VMSTATE_ARRAY` and the typed `VMSTATE_UINT16_ARRAY` style macros. The element count is the
    /// length of the Rust array.
    pub fn array<V: VmStateType, const N: usize>(
        name: &'static str,
        get: impl Fn(&mut T) -> &mut [V; N] + Send + Sync + 'static,
    ) -> Self {
        Self::array_info(name, V::info(), get)
    }

    /// `VMSTATE_ARRAY` with an explicit info.
    pub fn array_info<V: 'static, const N: usize>(
        name: &'static str,
        info: &'static dyn VmStateInfo<V>,
        get: impl Fn(&mut T) -> &mut [V; N] + Send + Sync + 'static,
    ) -> Self {
        Self::with_body(
            name,
            Array {
                get: getter::<T, [_]>(move |s| get(s)),
                num: Num::Fixed(N),
                codec: InfoCodec(info),
                alloc: None,
            },
        )
    }

    /// `VMSTATE_VARRAY_UINT32` and friends: the element count comes from another field of the
    /// state, read by `len`. The storage must already be long enough.
    pub fn varray<V: VmStateType, A: AsMut<[V]> + ?Sized + 'static>(
        name: &'static str,
        len: impl Fn(&T) -> usize + Send + Sync + 'static,
        get: impl Fn(&mut T) -> &mut A + Send + Sync + 'static,
    ) -> Self {
        Self::varray_info(name, V::info(), len, get)
    }

    /// `VMSTATE_VARRAY_UINT32` with an explicit info.
    pub fn varray_info<V: 'static, A: AsMut<[V]> + ?Sized + 'static>(
        name: &'static str,
        info: &'static dyn VmStateInfo<V>,
        len: impl Fn(&T) -> usize + Send + Sync + 'static,
        get: impl Fn(&mut T) -> &mut A + Send + Sync + 'static,
    ) -> Self {
        Self::with_body(
            name,
            Array {
                get: getter(move |s| get(s).as_mut()),
                num: Num::Var(Box::new(len)),
                codec: InfoCodec(info),
                alloc: None,
            },
        )
    }

    /// `VMSTATE_VARRAY_UINT32_ALLOC` and friends: like [`varray`](Self::varray), but on load the
    /// vector is resized to the incoming count first.
    pub fn varray_alloc<V: VmStateType + Default>(
        name: &'static str,
        len: impl Fn(&T) -> usize + Send + Sync + 'static,
        get: impl Fn(&mut T) -> &mut Vec<V> + Send + Sync + 'static,
    ) -> Self {
        let get = std::sync::Arc::new(get);
        let get2 = get.clone();
        Self::with_body(
            name,
            Array {
                get: getter(move |s| get(s).as_mut_slice()),
                num: Num::Var(Box::new(len)),
                codec: InfoCodec(V::info()),
                alloc: Some(Box::new(move |s: &mut T, n| get2(s).resize_with(n, V::default))),
            },
        )
    }

    /// `VMSTATE_BUFFER`: every byte of a fixed size array.
    pub fn buffer<const N: usize>(
        name: &'static str,
        get: impl Fn(&mut T) -> &mut [u8; N] + Send + Sync + 'static,
    ) -> Self {
        Self::partial_buffer(name, N, get)
    }

    /// `VMSTATE_PARTIAL_BUFFER`: the first `size` bytes of a buffer.
    pub fn partial_buffer<A: AsMut<[u8]> + ?Sized + 'static>(
        name: &'static str,
        size: usize,
        get: impl Fn(&mut T) -> &mut A + Send + Sync + 'static,
    ) -> Self {
        Self::with_body(
            name,
            Single {
                get: getter(move |s| get(s).as_mut()),
                codec: InfoCodec(&Buffer),
                size: Size::Fixed(size),
                alloc: None,
            },
        )
    }

    /// `VMSTATE_VBUFFER_UINT32` and friends: the byte count comes from another field. With
    /// `VMS_MULTIPLY` the `size` closure does the multiplication itself.
    pub fn vbuffer<A: AsMut<[u8]> + ?Sized + 'static>(
        name: &'static str,
        size: impl Fn(&T) -> usize + Send + Sync + 'static,
        get: impl Fn(&mut T) -> &mut A + Send + Sync + 'static,
    ) -> Self {
        Self::with_body(
            name,
            Single {
                get: getter(move |s| get(s).as_mut()),
                codec: InfoCodec(&Buffer),
                size: Size::Var(Box::new(size)),
                alloc: None,
            },
        )
    }

    /// `VMSTATE_VBUFFER_ALLOC_UINT32`: like [`vbuffer`](Self::vbuffer), but on load the vector is
    /// resized to the incoming byte count first.
    pub fn vbuffer_alloc(
        name: &'static str,
        size: impl Fn(&T) -> usize + Send + Sync + 'static,
        get: impl Fn(&mut T) -> &mut Vec<u8> + Send + Sync + 'static,
    ) -> Self {
        let get = std::sync::Arc::new(get);
        let get2 = get.clone();
        Self::with_body(
            name,
            Single {
                get: getter(move |s| get(s).as_mut_slice()),
                codec: InfoCodec(&Buffer),
                size: Size::Var(Box::new(size)),
                alloc: Some(Box::new(move |s: &mut T, n| get2(s).resize(n, 0))),
            },
        )
    }

    /// `VMSTATE_STRUCT`: a nested structure with its own description, loaded at that description's
    /// version. A getter that goes through a `Box` gives `VMSTATE_STRUCT_POINTER`.
    pub fn structure<U: 'static>(
        name: &'static str,
        vmsd: &'static VmStateDescription<U>,
        get: impl Fn(&mut T) -> &mut U + Send + Sync + 'static,
    ) -> Self {
        Self::with_body(
            name,
            Single {
                get: getter(get),
                codec: StructCodec { vmsd, struct_version_id: None },
                size: Size::Fixed(0),
                alloc: None,
            },
        )
    }

    /// `VMSTATE_VSTRUCT`: like [`structure`](Self::structure), but saved and loaded at
    /// `struct_version_id` rather than the nested description's own version.
    pub fn vstruct<U: 'static>(
        name: &'static str,
        vmsd: &'static VmStateDescription<U>,
        struct_version_id: i32,
        get: impl Fn(&mut T) -> &mut U + Send + Sync + 'static,
    ) -> Self {
        Self::with_body(
            name,
            Single {
                get: getter(get),
                codec: StructCodec { vmsd, struct_version_id: Some(struct_version_id) },
                size: Size::Fixed(0),
                alloc: None,
            },
        )
    }

    /// `VMSTATE_STRUCT_ARRAY`.
    pub fn struct_array<U: 'static, const N: usize>(
        name: &'static str,
        vmsd: &'static VmStateDescription<U>,
        get: impl Fn(&mut T) -> &mut [U; N] + Send + Sync + 'static,
    ) -> Self {
        Self::with_body(
            name,
            Array {
                get: getter::<T, [_]>(move |s| get(s)),
                num: Num::Fixed(N),
                codec: StructCodec { vmsd, struct_version_id: None },
                alloc: None,
            },
        )
    }

    /// `VMSTATE_STRUCT_VARRAY_UINT32` and friends. The storage must already be long enough.
    pub fn struct_varray<U: 'static, A: AsMut<[U]> + ?Sized + 'static>(
        name: &'static str,
        len: impl Fn(&T) -> usize + Send + Sync + 'static,
        vmsd: &'static VmStateDescription<U>,
        get: impl Fn(&mut T) -> &mut A + Send + Sync + 'static,
    ) -> Self {
        Self::with_body(
            name,
            Array {
                get: getter(move |s| get(s).as_mut()),
                num: Num::Var(Box::new(len)),
                codec: StructCodec { vmsd, struct_version_id: None },
                alloc: None,
            },
        )
    }

    /// `VMSTATE_STRUCT_VARRAY_ALLOC`: the vector is resized to the incoming count before loading.
    pub fn struct_varray_alloc<U: Default + 'static>(
        name: &'static str,
        len: impl Fn(&T) -> usize + Send + Sync + 'static,
        vmsd: &'static VmStateDescription<U>,
        get: impl Fn(&mut T) -> &mut Vec<U> + Send + Sync + 'static,
    ) -> Self {
        let get = std::sync::Arc::new(get);
        let get2 = get.clone();
        Self::with_body(
            name,
            Array {
                get: getter(move |s| get(s).as_mut_slice()),
                num: Num::Var(Box::new(len)),
                codec: StructCodec { vmsd, struct_version_id: None },
                alloc: Some(Box::new(move |s: &mut T, n| get2(s).resize_with(n, U::default))),
            },
        )
    }

    /// `VMSTATE_ARRAY_OF_POINTER`: an array whose elements may be missing. A missing element on
    /// the source goes out as a single `VMS_MARKER_PTR_NULL` byte, and a missing element on the
    /// destination expects that byte.
    pub fn array_of_pointer<V: 'static, const N: usize>(
        name: &'static str,
        info: &'static dyn VmStateInfo<V>,
        get: impl Fn(&mut T) -> &mut [Option<V>; N] + Send + Sync + 'static,
    ) -> Self {
        Self::with_body(
            name,
            PtrArray {
                get: getter::<T, [_]>(move |s| get(s)),
                num: Num::Fixed(N),
                codec: InfoCodec(info),
                auto_alloc: None,
            },
        )
    }

    /// `VMSTATE_ARRAY_OF_POINTER_TO_STRUCT`.
    pub fn array_of_pointer_to_struct<U: 'static, const N: usize>(
        name: &'static str,
        vmsd: &'static VmStateDescription<U>,
        get: impl Fn(&mut T) -> &mut [Option<U>; N] + Send + Sync + 'static,
    ) -> Self {
        Self::with_body(
            name,
            PtrArray {
                get: getter::<T, [_]>(move |s| get(s)),
                num: Num::Fixed(N),
                codec: StructCodec { vmsd, struct_version_id: None },
                auto_alloc: None,
            },
        )
    }

    /// `VMSTATE_VARRAY_OF_POINTER_TO_STRUCT_UINT32_ALLOC`: every element is preceded by a marker
    /// byte, and elements the source had are created with `U::default()` before loading. The
    /// storage must hold at least `len` slots, which should all be empty before a load.
    pub fn varray_of_pointer_to_struct_alloc<U: Default + 'static, A>(
        name: &'static str,
        len: impl Fn(&T) -> usize + Send + Sync + 'static,
        vmsd: &'static VmStateDescription<U>,
        get: impl Fn(&mut T) -> &mut A + Send + Sync + 'static,
    ) -> Self
    where
        A: AsMut<[Option<U>]> + ?Sized + 'static,
    {
        Self::with_body(
            name,
            PtrArray {
                get: getter(move |s| get(s).as_mut()),
                num: Num::Var(Box::new(len)),
                codec: StructCodec { vmsd, struct_version_id: None },
                auto_alloc: Some(Box::new(U::default)),
            },
        )
    }

    /// `VMSTATE_UNUSED`: `size` bytes of padding where a field used to be.
    pub fn unused(size: usize) -> Self {
        Self::with_body("unused", Unused { size })
    }

    /// `VMSTATE_VALIDATE`: moves no data, but loading fails with "Input validation failed" when
    /// `test` returns false.
    pub fn validate(
        name: &'static str,
        test: impl Fn(&T, i32) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self::with_body(name, Validate).test(test).must_exist()
    }

    /// `VMSTATE_WITH_TMP`: state that only exists in the stream.
    ///
    /// QEMU allocates a temporary struct whose first member points back at the parent, and the
    /// temporary's `pre_save` and `post_load` hooks move data between the two. Here `make` builds
    /// the temporary from the parent, both before saving and before loading, and `apply` hands the
    /// loaded temporary back. The temporary is loaded at this field's version, as `load_tmp()`
    /// does.
    pub fn with_tmp<U: 'static>(
        vmsd: &'static VmStateDescription<U>,
        make: impl Fn(&T) -> U + Send + Sync + 'static,
        apply: impl Fn(&mut T, U) + Send + Sync + 'static,
    ) -> Self {
        Self::with_body("tmp", Tmp { vmsd, make: Box::new(make), apply: Box::new(apply) })
    }
}

/// What the interpreter needs to know about one field, whatever its shape.
pub(crate) trait FieldBody<T>: Send + Sync {
    fn type_name(&self) -> &'static str;

    /// `VMS_STRUCT`: vmdesc wraps the nested description in a "struct" object.
    fn is_struct(&self) -> bool {
        false
    }

    /// The flag half of `vmsd_can_compress()`.
    fn can_compress(&self) -> bool {
        true
    }

    /// `vmstate_n_elems()`.
    fn n_elems(&self, s: &T) -> usize;

    /// `vmstate_size()`.
    fn size(&self, _s: &T) -> usize {
        0
    }

    /// `vmstate_handle_alloc()`.
    fn handle_alloc(&self, _s: &mut T, _size: usize, _n_elems: usize) {}

    /// The pointer marker half of `vmstate_load_next()`. `Ok(false)` means the element is absent
    /// and is skipped.
    fn load_next(&self, _f: &mut StreamReader<'_>, _s: &mut T, _i: usize) -> Result<bool> {
        Ok(true)
    }

    fn load_elem(
        &self,
        f: &mut StreamReader<'_>,
        s: &mut T,
        i: usize,
        size: usize,
        field_version: i32,
    ) -> Result<()>;

    fn save_elem(
        &self,
        f: &mut StreamWriter,
        s: &mut T,
        i: usize,
        size: usize,
        vmdesc: Option<&mut JsonWriter>,
    ) -> Result<()>;
}

/// How one element is coded: a leaf info or a nested description.
pub(crate) trait ElemCodec<V: ?Sized>: Send + Sync {
    fn type_name(&self) -> &'static str;
    fn is_struct(&self) -> bool {
        false
    }
    fn can_compress(&self) -> bool {
        true
    }
    fn load(&self, f: &mut StreamReader<'_>, v: &mut V, size: usize) -> Result<()>;
    fn save(
        &self,
        f: &mut StreamWriter,
        v: &mut V,
        size: usize,
        vmdesc: Option<&mut JsonWriter>,
    ) -> Result<()>;
}

struct InfoCodec<V: ?Sized + 'static>(&'static dyn VmStateInfo<V>);

impl<V: ?Sized + 'static> ElemCodec<V> for InfoCodec<V> {
    fn type_name(&self) -> &'static str {
        self.0.name()
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut V, size: usize) -> Result<()> {
        self.0.load(f, v, size)
    }

    fn save(
        &self,
        f: &mut StreamWriter,
        v: &mut V,
        size: usize,
        _vmdesc: Option<&mut JsonWriter>,
    ) -> Result<()> {
        self.0.save(f, v, size)
    }
}

/// `VMS_STRUCT`, or `VMS_VSTRUCT` when `struct_version_id` is set.
struct StructCodec<U: 'static> {
    vmsd: &'static VmStateDescription<U>,
    struct_version_id: Option<i32>,
}

impl<U: 'static> ElemCodec<U> for StructCodec<U> {
    fn type_name(&self) -> &'static str {
        if self.struct_version_id.is_some() { "vstruct" } else { "struct" }
    }

    fn is_struct(&self) -> bool {
        self.struct_version_id.is_none()
    }

    fn can_compress(&self) -> bool {
        !self.is_struct()
            || (self.vmsd.subsections.is_empty()
                && self.vmsd.fields.iter().all(VmStateField::can_compress))
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut U, _size: usize) -> Result<()> {
        let version_id = self.struct_version_id.unwrap_or(self.vmsd.version_id);
        load_vmsd(f, self.vmsd, v, version_id)
    }

    fn save(
        &self,
        f: &mut StreamWriter,
        v: &mut U,
        _size: usize,
        vmdesc: Option<&mut JsonWriter>,
    ) -> Result<()> {
        let version_id = self.struct_version_id.unwrap_or(self.vmsd.version_id);
        save_vmsd_v(f, self.vmsd, v, version_id, vmdesc)
    }
}

enum Size<T> {
    Fixed(usize),
    Var(Count<T>),
}

enum Num<T> {
    Fixed(usize),
    Var(Count<T>),
}

impl<T> Num<T> {
    fn get(&self, s: &T) -> usize {
        match self {
            Num::Fixed(n) => *n,
            Num::Var(f) => f(s),
        }
    }
}

/// `VMS_SINGLE`, `VMS_BUFFER` and `VMS_VBUFFER` fields: one element.
struct Single<T, V: ?Sized, C> {
    get: Getter<T, V>,
    codec: C,
    size: Size<T>,
    alloc: Option<Alloc<T>>,
}

impl<T, V: ?Sized, C: ElemCodec<V>> FieldBody<T> for Single<T, V, C> {
    fn type_name(&self) -> &'static str {
        self.codec.type_name()
    }

    fn is_struct(&self) -> bool {
        self.codec.is_struct()
    }

    fn can_compress(&self) -> bool {
        self.codec.can_compress()
    }

    fn n_elems(&self, _s: &T) -> usize {
        1
    }

    fn size(&self, s: &T) -> usize {
        match &self.size {
            Size::Fixed(n) => *n,
            Size::Var(f) => f(s),
        }
    }

    fn handle_alloc(&self, s: &mut T, size: usize, n_elems: usize) {
        if let Some(alloc) = &self.alloc {
            alloc(s, size * n_elems);
        }
    }

    fn load_elem(
        &self,
        f: &mut StreamReader<'_>,
        s: &mut T,
        _i: usize,
        size: usize,
        _field_version: i32,
    ) -> Result<()> {
        self.codec.load(f, (self.get)(s), size)
    }

    fn save_elem(
        &self,
        f: &mut StreamWriter,
        s: &mut T,
        _i: usize,
        size: usize,
        vmdesc: Option<&mut JsonWriter>,
    ) -> Result<()> {
        self.codec.save(f, (self.get)(s), size, vmdesc)
    }
}

fn element<V>(slice: &mut [V], i: usize) -> Result<&mut V> {
    let len = slice.len();
    match slice.get_mut(i) {
        Some(v) => Ok(v),
        None => bail!("array of {len} elements has no element {i}"),
    }
}

/// `VMS_ARRAY` and `VMS_VARRAY_*` fields.
struct Array<T, V, C> {
    get: Getter<T, [V]>,
    num: Num<T>,
    codec: C,
    alloc: Option<Alloc<T>>,
}

impl<T, V, C: ElemCodec<V>> FieldBody<T> for Array<T, V, C> {
    fn type_name(&self) -> &'static str {
        self.codec.type_name()
    }

    fn is_struct(&self) -> bool {
        self.codec.is_struct()
    }

    fn can_compress(&self) -> bool {
        self.codec.can_compress()
    }

    fn n_elems(&self, s: &T) -> usize {
        self.num.get(s)
    }

    fn handle_alloc(&self, s: &mut T, _size: usize, n_elems: usize) {
        if let Some(alloc) = &self.alloc {
            alloc(s, n_elems);
        }
    }

    fn load_elem(
        &self,
        f: &mut StreamReader<'_>,
        s: &mut T,
        i: usize,
        size: usize,
        _field_version: i32,
    ) -> Result<()> {
        self.codec.load(f, element((self.get)(s), i)?, size)
    }

    fn save_elem(
        &self,
        f: &mut StreamWriter,
        s: &mut T,
        i: usize,
        size: usize,
        vmdesc: Option<&mut JsonWriter>,
    ) -> Result<()> {
        self.codec.save(f, element((self.get)(s), i)?, size, vmdesc)
    }
}

/// `VMS_ARRAY_OF_POINTER` fields, with `VMS_ARRAY_OF_POINTER_AUTO_ALLOC` when `auto_alloc` is
/// set.
struct PtrArray<T, V, C> {
    get: Getter<T, [Option<V>]>,
    num: Num<T>,
    codec: C,
    auto_alloc: Option<Box<dyn Fn() -> V + Send + Sync>>,
}

impl<T, V, C: ElemCodec<V>> FieldBody<T> for PtrArray<T, V, C> {
    fn type_name(&self) -> &'static str {
        self.codec.type_name()
    }

    fn is_struct(&self) -> bool {
        self.codec.is_struct()
    }

    fn can_compress(&self) -> bool {
        self.auto_alloc.is_none() && self.codec.can_compress()
    }

    fn n_elems(&self, s: &T) -> usize {
        self.num.get(s)
    }

    fn load_next(&self, f: &mut StreamReader<'_>, s: &mut T, i: usize) -> Result<bool> {
        let elem = element((self.get)(s), i)?;
        // QEMU asserts that an auto allocated array starts out empty. Reading the marker
        // whenever auto allocation is on gives the same stream handling without the assert.
        if elem.is_some() && self.auto_alloc.is_none() {
            return Ok(true);
        }
        // vmstate_ptr_marker_load()
        let byte = f.get_byte();
        if byte == VMS_MARKER_PTR_NULL {
            if self.auto_alloc.is_some() {
                *elem = None;
            }
            return Ok(false);
        }
        if byte == VMS_MARKER_PTR_VALID {
            let Some(new) = &self.auto_alloc else {
                bail!("Unexpected ptr marker: {byte}");
            };
            *elem = Some(new());
            return Ok(true);
        }
        bail!("Unexpected ptr marker: {byte}")
    }

    fn load_elem(
        &self,
        f: &mut StreamReader<'_>,
        s: &mut T,
        i: usize,
        size: usize,
        _field_version: i32,
    ) -> Result<()> {
        match element((self.get)(s), i)? {
            Some(v) => self.codec.load(f, v, size),
            None => Ok(()),
        }
    }

    fn save_elem(
        &self,
        f: &mut StreamWriter,
        s: &mut T,
        i: usize,
        size: usize,
        vmdesc: Option<&mut JsonWriter>,
    ) -> Result<()> {
        match element((self.get)(s), i)? {
            // vmstate_info_ptr_marker
            None => {
                f.put_byte(VMS_MARKER_PTR_NULL);
                Ok(())
            }
            Some(v) => {
                if self.auto_alloc.is_some() {
                    f.put_byte(VMS_MARKER_PTR_VALID);
                }
                self.codec.save(f, v, size, vmdesc)
            }
        }
    }
}

/// `VMSTATE_UNUSED_BUFFER`.
struct Unused {
    size: usize,
}

impl<T> FieldBody<T> for Unused {
    fn type_name(&self) -> &'static str {
        UnusedBuffer.name()
    }

    fn n_elems(&self, _s: &T) -> usize {
        1
    }

    fn size(&self, _s: &T) -> usize {
        self.size
    }

    fn load_elem(
        &self,
        f: &mut StreamReader<'_>,
        _s: &mut T,
        _i: usize,
        size: usize,
        _field_version: i32,
    ) -> Result<()> {
        UnusedBuffer.load(f, &mut (), size)
    }

    fn save_elem(
        &self,
        f: &mut StreamWriter,
        _s: &mut T,
        _i: usize,
        size: usize,
        _vmdesc: Option<&mut JsonWriter>,
    ) -> Result<()> {
        UnusedBuffer.save(f, &(), size)
    }
}

/// `VMSTATE_VALIDATE`: an array of zero elements whose `field_exists` test is the check.
struct Validate;

impl<T> FieldBody<T> for Validate {
    fn type_name(&self) -> &'static str {
        "unknown"
    }

    fn n_elems(&self, _s: &T) -> usize {
        0
    }

    fn load_elem(
        &self,
        _f: &mut StreamReader<'_>,
        _s: &mut T,
        _i: usize,
        _size: usize,
        _field_version: i32,
    ) -> Result<()> {
        Ok(())
    }

    fn save_elem(
        &self,
        _f: &mut StreamWriter,
        _s: &mut T,
        _i: usize,
        _size: usize,
        _vmdesc: Option<&mut JsonWriter>,
    ) -> Result<()> {
        Ok(())
    }
}

type Apply<T, U> = Box<dyn Fn(&mut T, U) + Send + Sync>;

/// `vmstate_info_tmp`.
struct Tmp<T, U: 'static> {
    vmsd: &'static VmStateDescription<U>,
    make: Box<dyn Fn(&T) -> U + Send + Sync>,
    apply: Apply<T, U>,
}

impl<T, U: 'static> FieldBody<T> for Tmp<T, U> {
    fn type_name(&self) -> &'static str {
        "tmp"
    }

    fn n_elems(&self, _s: &T) -> usize {
        1
    }

    fn load_elem(
        &self,
        f: &mut StreamReader<'_>,
        s: &mut T,
        _i: usize,
        _size: usize,
        field_version: i32,
    ) -> Result<()> {
        let mut tmp = (self.make)(s);
        load_vmsd(f, self.vmsd, &mut tmp, field_version)?;
        (self.apply)(s, tmp);
        Ok(())
    }

    fn save_elem(
        &self,
        f: &mut StreamWriter,
        s: &mut T,
        _i: usize,
        _size: usize,
        vmdesc: Option<&mut JsonWriter>,
    ) -> Result<()> {
        let mut tmp = (self.make)(s);
        save_vmsd_v(f, self.vmsd, &mut tmp, self.vmsd.version_id, vmdesc)
    }
}
