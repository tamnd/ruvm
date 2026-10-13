// SPDX-License-Identifier: GPL-2.0-or-later

//! The entropy backend objects from backends/rng.c, backends/rng-builtin.c and
//! backends/rng-random.c: `rng-backend`, `rng-builtin` and, on POSIX hosts, `rng-random`.
//!
//! `-object` and `object-add` create them with QEMU's properties and checks. virtio-rng does
//! not take the `rng` link yet and keeps reading the host's random source on its own, so the
//! objects hold no requests. `rng-egd` is not here, because it needs a chardev backend.

use std::sync::{Arc, Mutex};

use ruvm_base::Result;
use ruvm_qom::{Object, Registry, TYPE_OBJECT, TYPE_USER_CREATABLE, TypeInfo, UserCreatableClass};

/// `TYPE_RNG_BACKEND`.
pub(crate) const TYPE_RNG_BACKEND: &str = "rng-backend";
/// `TYPE_RNG_BUILTIN`.
pub(crate) const TYPE_RNG_BUILTIN: &str = "rng-builtin";
/// `TYPE_RNG_RANDOM`.
#[cfg(unix)]
pub(crate) const TYPE_RNG_RANDOM: &str = "rng-random";

/// `RngBackend` with the `filename` of `RngRandom`, which only `rng-random` shows.
#[derive(Debug)]
struct Fields {
    opened: bool,
    #[cfg(unix)]
    filename: String,
    #[cfg(unix)]
    file: Option<std::fs::File>,
}

impl Default for Fields {
    /// `rng_random_init()` starts with `/dev/urandom`.
    fn default() -> Self {
        Fields {
            opened: false,
            #[cfg(unix)]
            filename: "/dev/urandom".to_string(),
            #[cfg(unix)]
            file: None,
        }
    }
}

#[derive(Debug, Default)]
struct RngState(Mutex<Fields>);

fn with_fields<T>(obj: &Object, f: impl FnOnce(&mut Fields) -> T) -> T {
    let st = obj.state::<RngState>().expect("an rng backend has rng state");
    let mut g = st.0.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut g)
}

/// `rng_backend_complete()`: the class `opened` hook, then the `opened` flag.
fn complete(obj: &Object, opened: Option<fn(&Object) -> Result<()>>) -> Result<()> {
    if let Some(open) = opened {
        open(obj)?;
    }
    with_fields(obj, |f| f.opened = true);
    Ok(())
}

fn set_complete(k: &Arc<ruvm_qom::ObjectClass>, opened: Option<fn(&Object) -> Result<()>>) {
    let uc = k.interface(TYPE_USER_CREATABLE).expect("rng backends are user creatable");
    uc.set_ext(UserCreatableClass {
        complete: Some(Arc::new(move |o| complete(o, opened))),
        prepare_delete: None,
    });
}

/// `rng_random_opened()`: the file opened with `O_RDONLY | O_NONBLOCK`, as `qemu_open()` does.
#[cfg(unix)]
fn random_opened(obj: &Object) -> Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let filename = with_fields(obj, |f| f.filename.clone());
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&filename)
        .map_err(|e| {
            ruvm_base::Error::generic(format!(
                "Could not open '{filename}': {}",
                ruvm_base::error::strerror(&e)
            ))
        })?;
    with_fields(obj, |f| f.file = Some(file));
    Ok(())
}

/// Registers `rng-backend`, `rng-builtin` and, on POSIX hosts, `rng-random`.
pub(crate) fn register_types(registry: &Registry) {
    let backend = TypeInfo::new(TYPE_RNG_BACKEND)
        .parent(TYPE_OBJECT)
        .abstract_()
        .interface(TYPE_USER_CREATABLE)
        .instance_state(RngState::default)
        .class_init(|k| {
            set_complete(k, None);
            k.property_add_bool(
                "opened",
                Some(Arc::new(|o: &Object| Ok(with_fields(o, |f| f.opened)))),
                None,
            );
        });
    let builtin = TypeInfo::new(TYPE_RNG_BUILTIN).parent(TYPE_RNG_BACKEND);
    #[cfg(unix)]
    let random = TypeInfo::new(TYPE_RNG_RANDOM).parent(TYPE_RNG_BACKEND).class_init(|k| {
        set_complete(k, Some(random_opened));
        k.property_add_str(
            "filename",
            Some(Arc::new(|o: &Object| Ok(with_fields(o, |f| f.filename.clone())))),
            Some(Arc::new(|o: &Object, v: &str| {
                with_fields(o, |f| {
                    if f.opened {
                        return Err(ruvm_base::Error::generic(
                            "Property 'filename' can no longer be set",
                        ));
                    }
                    f.filename = v.to_string();
                    Ok(())
                })
            })),
        );
    });
    #[cfg(unix)]
    registry.register_all([backend, builtin, random]);
    #[cfg(not(unix))]
    registry.register_all([backend, builtin]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_qapi::QDict;

    fn add(registry: &Registry, ty: &str, id: &str, props: &[(&str, &str)]) -> Result<Object> {
        let mut args = QDict::new().with("qom-type", ty).with("id", id);
        for (k, v) in props {
            args = args.with(*k, *v);
        }
        registry.user_creatable_add(&args, true)
    }

    #[test]
    fn backends() {
        let registry = Registry::new();
        register_types(&registry);
        let o = add(&registry, TYPE_RNG_BUILTIN, "rng0", &[]).unwrap();
        assert!(o.property_get_bool("opened").unwrap());
        let e = add(&registry, TYPE_RNG_BACKEND, "rng1", &[]).unwrap_err();
        assert_eq!(e.message(), "object type 'rng-backend' is abstract");
    }

    #[cfg(unix)]
    #[test]
    fn random() {
        let registry = Registry::new();
        register_types(&registry);
        let o = add(&registry, TYPE_RNG_RANDOM, "r0", &[]).unwrap();
        assert_eq!(o.property_get_str("filename").unwrap(), "/dev/urandom");
        let e = o.property_set_str("filename", "/dev/random").unwrap_err();
        assert_eq!(e.message(), "Property 'filename' can no longer be set");
        let e =
            add(&registry, TYPE_RNG_RANDOM, "r1", &[("filename", "/nonexistent/rng")]).unwrap_err();
        assert_eq!(e.message(), "Could not open '/nonexistent/rng': No such file or directory");
    }
}
