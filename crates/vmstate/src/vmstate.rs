// SPDX-License-Identifier: GPL-2.0-or-later

//! The VMState interpreter, migration/vmstate.c.
//!
//! Saving walks the field list, writes each field that exists at the section version, then writes
//! every subsection whose `needed` test passes. Loading walks the same list at the incoming
//! version, then peeks for subsection markers. The stream has no framing between fields, so both
//! sides have to agree on the exact list, which is why the checks here follow QEMU closely.

use ruvm_base::{Result, bail, err};

use crate::QEMU_VM_SUBSECTION;
use crate::field::VmStateField;
use crate::file::{EINVAL, StreamReader, StreamWriter};
use crate::json::JsonWriter;
use crate::vmsd::{Hook, VmStateDescription};

/// `vmstate_field_exists()`: a `field_exists` test decides alone, otherwise the field exists from
/// its version on.
fn field_exists<T>(field: &VmStateField<T>, opaque: &T, version_id: i32) -> bool {
    match &field.field_exists {
        Some(test) => test(opaque, version_id),
        None => field.version_id <= version_id,
    }
}

/// `vmstate_pre_load()`.
fn pre_load<T>(vmsd: &VmStateDescription<T>, opaque: &mut T) -> Result<()> {
    match &vmsd.pre_load {
        Some(Hook::Errp(hook)) => hook(opaque).map_err(|e| {
            e.prepend(format_args!(
                "pre load hook failed for: '{}', version_id: {}, minimum version_id: {}: ",
                vmsd.name, vmsd.version_id, vmsd.minimum_version_id
            ))
        }),
        Some(Hook::Ret(hook)) => {
            let ret = hook(opaque);
            if ret != 0 {
                bail!(
                    "pre load hook failed for: '{}', version_id: {}, minimum version_id: {}, ret: {}",
                    vmsd.name,
                    vmsd.version_id,
                    vmsd.minimum_version_id,
                    ret
                );
            }
            Ok(())
        }
        None => Ok(()),
    }
}

/// `vmstate_post_load()`.
fn post_load<T>(vmsd: &VmStateDescription<T>, opaque: &mut T, version_id: i32) -> Result<()> {
    match &vmsd.post_load {
        Some(Hook::Errp(hook)) => hook(opaque, version_id).map_err(|e| {
            e.prepend(format_args!(
                "post load hook failed for: {}, version_id: {}, minimum_version: {}: ",
                vmsd.name, vmsd.version_id, vmsd.minimum_version_id
            ))
        }),
        Some(Hook::Ret(hook)) => {
            let ret = hook(opaque, version_id);
            if ret < 0 {
                bail!(
                    "post load hook failed for: {}, version_id: {}, minimum_version: {}, ret: {}",
                    vmsd.name,
                    vmsd.version_id,
                    vmsd.minimum_version_id,
                    ret
                );
            }
            Ok(())
        }
        None => Ok(()),
    }
}

/// `vmstate_pre_save()`.
fn pre_save<T>(vmsd: &VmStateDescription<T>, opaque: &mut T) -> Result<()> {
    match &vmsd.pre_save {
        Some(Hook::Errp(hook)) => {
            hook(opaque).map_err(|e| e.prepend(format_args!("pre-save for {} failed: ", vmsd.name)))
        }
        Some(Hook::Ret(hook)) => {
            if hook(opaque) < 0 {
                bail!("pre-save failed: {}", vmsd.name);
            }
            Ok(())
        }
        None => Ok(()),
    }
}

/// `vmstate_load_vmsd()`.
pub(crate) fn load_vmsd<T>(
    f: &mut StreamReader<'_>,
    vmsd: &VmStateDescription<T>,
    opaque: &mut T,
    version_id: i32,
) -> Result<()> {
    if version_id > vmsd.version_id {
        bail!(
            "{}: incoming version_id {} is too new for local version_id {}",
            vmsd.name,
            version_id,
            vmsd.version_id
        );
    }
    if version_id < vmsd.minimum_version_id {
        bail!(
            "{}: incoming version_id {} is too old for local minimum version_id {}",
            vmsd.name,
            version_id,
            vmsd.minimum_version_id
        );
    }

    pre_load(vmsd, opaque)?;

    for field in &vmsd.fields {
        if field_exists(field, opaque, version_id) {
            let body = &field.body;
            let n_elems = body.n_elems(opaque);
            let size = body.size(opaque);
            body.handle_alloc(opaque, size, n_elems);

            for i in 0..n_elems {
                // A marker error leaves the stream error alone, as in QEMU.
                if !body.load_next(f, opaque, i)? {
                    continue;
                }
                match body.load_elem(f, opaque, i, size, field.version_id) {
                    Ok(()) => {
                        let ret = f.get_error();
                        if ret < 0 {
                            bail!("Failed to load {} state: stream error: {}", vmsd.name, ret);
                        }
                    }
                    Err(e) => {
                        f.set_error(-EINVAL);
                        return Err(e);
                    }
                }
            }
        } else if field.must_exist {
            bail!(
                "Input validation failed: {}/{} version_id: {}",
                vmsd.name,
                field.name,
                vmsd.version_id
            );
        }
    }

    if let Err(e) = subsection_load(f, vmsd, opaque) {
        f.set_error(-EINVAL);
        return Err(e);
    }

    post_load(vmsd, opaque, version_id)
}

/// `vmstate_save_vmsd_v()`. With `vmdesc`, the description of what went out is added to the
/// object the caller has open.
pub(crate) fn save_vmsd_v<T>(
    f: &mut StreamWriter,
    vmsd: &VmStateDescription<T>,
    opaque: &mut T,
    version_id: i32,
    vmdesc: Option<&mut JsonWriter>,
) -> Result<()> {
    pre_save(vmsd, opaque)?;
    let ret = save_fields(f, vmsd, opaque, version_id, vmdesc);
    if let Some(post_save) = &vmsd.post_save {
        post_save(opaque);
    }
    ret
}

/// `vmsd_desc_field_start()`.
fn desc_field_start<T>(
    vmdesc: &mut JsonWriter,
    vmsd: &VmStateDescription<T>,
    field: &VmStateField<T>,
    i: usize,
    max: usize,
) {
    let same: Vec<_> = vmsd.fields.iter().filter(|f| f.name == field.name).collect();
    let name = if same.len() > 1 {
        let num = same.iter().position(|f| std::ptr::eq(*f, field)).unwrap_or(0);
        format!("{}[{num}]", field.name)
    } else {
        field.name.to_owned()
    };
    vmdesc.start_object(None);
    vmdesc.str(Some("name"), &name);
    if max > 1 {
        if field.can_compress() {
            vmdesc.int64(Some("array_len"), max as i64);
        } else {
            vmdesc.int64(Some("index"), i as i64);
        }
    }
    vmdesc.str(Some("type"), field.body.type_name());
    if field.body.is_struct() {
        vmdesc.start_object(Some("struct"));
    }
}

/// `vmsd_desc_field_end()`.
fn desc_field_end<T>(vmdesc: &mut JsonWriter, field: &VmStateField<T>, size: u64) {
    if field.body.is_struct() {
        vmdesc.end_object();
    }
    vmdesc.uint64(Some("size"), size);
    vmdesc.end_object();
}

fn save_fields<T>(
    f: &mut StreamWriter,
    vmsd: &VmStateDescription<T>,
    opaque: &mut T,
    version_id: i32,
    mut vmdesc: Option<&mut JsonWriter>,
) -> Result<()> {
    if let Some(d) = vmdesc.as_deref_mut() {
        d.str(Some("vmsd_name"), vmsd.name);
        d.int64(Some("version"), i64::from(version_id));
        d.start_array(Some("fields"));
    }
    for field in &vmsd.fields {
        if field_exists(field, opaque, version_id) {
            let body = &field.body;
            let n_elems = body.n_elems(opaque);
            let size = body.size(opaque);
            let mut desc = vmdesc.as_deref_mut();
            for i in 0..n_elems {
                let max_elems = if field.can_compress() { n_elems - i } else { 1 };
                if let Some(d) = desc.as_deref_mut() {
                    desc_field_start(d, vmsd, field, i, max_elems);
                }
                let before = f.transferred();
                body.save_elem(f, opaque, i, size, desc.as_deref_mut()).map_err(|e| {
                    e.prepend(format_args!("Save of field {}/{} failed: ", vmsd.name, field.name))
                })?;
                if let Some(d) = desc.as_deref_mut() {
                    desc_field_end(d, field, f.transferred() - before);
                }
                // A compressed array is described by its first element.
                if max_elems > 1 {
                    desc = None;
                }
            }
        } else if field.must_exist {
            // QEMU reports this and then asserts. A library should not abort the process over
            // it, so the save fails instead.
            bail!("Output state validation failed: {}/{}", vmsd.name, field.name);
        }
    }
    if let Some(d) = vmdesc.as_deref_mut() {
        d.end_array();
    }
    subsection_save(f, vmsd, opaque, vmdesc)
}

/// `vmstate_get_subsection()`.
fn get_subsection<'a, T>(
    subs: &[&'a VmStateDescription<T>],
    idstr: &[u8],
) -> Option<&'a VmStateDescription<T>> {
    subs.iter().copied().find(|s| s.name.as_bytes() == idstr)
}

/// `vmstate_subsection_load()`.
///
/// Anything after the fields that does not look like one of this section's subsections is left
/// for the caller, which is how a section footer or the next section is recognised. A well formed
/// subsection with an unknown name is an error.
fn subsection_load<T>(
    f: &mut StreamReader<'_>,
    vmsd: &VmStateDescription<T>,
    opaque: &mut T,
) -> Result<()> {
    while f.peek_byte(0) == QEMU_VM_SUBSECTION {
        let len = usize::from(f.peek_byte(1));
        if len < vmsd.name.len() + 1 {
            // A subsection name has to be "section_name/a".
            return Ok(());
        }
        let idstr = f.peek_buffer(len, 2);
        if idstr.len() != len {
            return Ok(());
        }
        // QEMU compares only the parent's name as a prefix, not the slash after it.
        if !idstr.starts_with(vmsd.name.as_bytes()) {
            return Ok(());
        }
        let name = String::from_utf8_lossy(idstr).into_owned();
        let Some(sub) = get_subsection(&vmsd.subsections, idstr) else {
            bail!("VM subsection '{}' in '{}' does not exist", name, vmsd.name);
        };
        f.skip(1); // subsection
        f.skip(1); // len
        f.skip(len); // idstr
        // QEMU reads the version into a uint8_t, so only the low byte counts.
        let version_id = i32::from(f.get_be32() as u8);

        load_vmsd(f, sub, opaque, version_id).map_err(|e| {
            e.prepend(format_args!("Loading VM subsection '{}' in '{}' failed: ", name, vmsd.name))
        })?;
    }
    Ok(())
}

/// `vmstate_subsection_save()`.
fn subsection_save<T>(
    f: &mut StreamWriter,
    vmsd: &VmStateDescription<T>,
    opaque: &mut T,
    mut vmdesc: Option<&mut JsonWriter>,
) -> Result<()> {
    let mut has_subsections = false;
    for sub in &vmsd.subsections {
        if !sub.section_needed(opaque) {
            continue;
        }
        let Ok(len) = u8::try_from(sub.name.len()) else {
            return Err(err!("subsection name '{}' is longer than 255 bytes", sub.name));
        };
        if let Some(d) = vmdesc.as_deref_mut() {
            if !has_subsections {
                d.start_array(Some("subsections"));
                has_subsections = true;
            }
            d.start_object(None);
        }
        f.put_byte(QEMU_VM_SUBSECTION);
        f.put_byte(len);
        f.put_buffer(sub.name.as_bytes());
        f.put_be32(sub.version_id as u32);
        save_vmsd_v(f, sub, opaque, sub.version_id, vmdesc.as_deref_mut())?;
        if let Some(d) = vmdesc.as_deref_mut() {
            d.end_object();
        }
    }
    if let Some(d) = vmdesc.filter(|_| has_subsections) {
        d.end_array();
    }
    Ok(())
}

/// `vmstate_save_state()`: writes `opaque` at the description's own version.
pub fn vmstate_save_state<T>(
    f: &mut StreamWriter,
    vmsd: &VmStateDescription<T>,
    opaque: &mut T,
) -> Result<()> {
    save_vmsd_v(f, vmsd, opaque, vmsd.version_id, None)
}

/// `vmstate_save_vmsd()` with a vmdesc writer: like [`vmstate_save_state`], and also adds
/// `vmsd_name`, `version`, `fields` and `subsections` to the JSON object the caller has open.
pub fn vmstate_save_state_vmdesc<T>(
    f: &mut StreamWriter,
    vmsd: &VmStateDescription<T>,
    opaque: &mut T,
    vmdesc: Option<&mut JsonWriter>,
) -> Result<()> {
    save_vmsd_v(f, vmsd, opaque, vmsd.version_id, vmdesc)
}

/// `vmstate_load_state()`: reads `opaque` from a stream that carries `version_id`.
///
/// On failure the stream usually has an error recorded as well, `-EINVAL` when a field or
/// subsection failed to load and `-EIO` when the stream ended early.
pub fn vmstate_load_state<T>(
    f: &mut StreamReader<'_>,
    vmsd: &VmStateDescription<T>,
    opaque: &mut T,
    version_id: i32,
) -> Result<()> {
    load_vmsd(f, vmsd, opaque, version_id)
}
