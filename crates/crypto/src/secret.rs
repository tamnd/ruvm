// SPDX-License-Identifier: GPL-2.0-or-later

//! The `secret` object, from crypto/secret_common.c and crypto/secret.c.
//!
//! [`register_types`] adds the abstract `secret_common` type and the `secret` type to a QOM
//! registry. A `secret` takes its data from `data` or `file`, optionally base64 encoded
//! (`format=base64`) and optionally encrypted with AES-256-CBC under another secret (`keyid`
//! and `iv`). The rest of ruvm looks secrets up by id with [`secret_lookup`] and its variants,
//! which search `/objects` of a registry.
//!
//! Differences from QEMU:
//!
//! - `secret_keyring` (the Linux kernel keyring) is not implemented.
//! - The string properties read back as the empty string when unset, where QEMU returns NULL.

use std::sync::{Arc, Mutex};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{QCryptoCipherAlgo, QCryptoCipherMode, QCryptoSecretFormat};
use ruvm_qom::{Object, Registry, TYPE_OBJECT, TYPE_USER_CREATABLE, TypeInfo, UserCreatableClass};

use crate::base64;
use crate::cipher::Cipher;

/// `TYPE_QCRYPTO_SECRET_COMMON`.
pub const TYPE_QCRYPTO_SECRET_COMMON: &str = "secret_common";
/// `TYPE_QCRYPTO_SECRET`.
pub const TYPE_QCRYPTO_SECRET: &str = "secret";

#[derive(Default)]
struct Fields {
    format: QCryptoSecretFormat,
    keyid: Option<String>,
    iv: Option<String>,
    data: Option<String>,
    file: Option<String>,
    rawdata: Option<Vec<u8>>,
}

/// The instance state of a secret: `QCryptoSecretCommon` and `QCryptoSecret` together.
#[derive(Default)]
struct SecretState(Mutex<Fields>);

fn with_fields<T>(obj: &Object, f: impl FnOnce(&mut Fields) -> T) -> T {
    let st = obj.state::<SecretState>().expect("a secret has secret state");
    let mut g = st.0.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut g)
}

/// `qcrypto_secret_load_data()` of the `secret` type.
fn load_data(data: Option<&str>, file: Option<&str>) -> Result<Vec<u8>> {
    if let Some(file) = file {
        if data.is_some() {
            return Err(Error::generic("'file' and 'data' are mutually exclusive"));
        }
        return std::fs::read(file).map_err(|e| {
            let text = ruvm_base::error::strerror(&e);
            Error::generic(format!(
                "Unable to read {file}: Failed to open file \u{201c}{file}\u{201d}: {text}"
            ))
        });
    }
    match data {
        Some(d) => Ok(d.as_bytes().to_vec()),
        None => Err(Error::generic("Either 'file' or 'data' must be provided")),
    }
}

/// `qcrypto_secret_decrypt()`.
fn decrypt(
    reg: &Registry,
    format: QCryptoSecretFormat,
    keyid: &str,
    iv: Option<&str>,
    input: &[u8],
) -> Result<Vec<u8>> {
    let key = secret_lookup_in(reg, keyid)?;
    if key.len() != 32 {
        return Err(Error::generic("Key should be 32 bytes in length"));
    }
    let Some(iv) = iv else {
        return Err(Error::generic("IV is required to decrypt secret"));
    };
    let iv = base64::decode(iv.as_bytes())?;
    if iv.len() != 16 {
        return Err(Error::generic(format!("IV should be 16 bytes in length not {}", iv.len())));
    }
    let mut aes = Cipher::new(QCryptoCipherAlgo::Aes256, QCryptoCipherMode::Cbc, &key)?;
    aes.setiv(&iv)?;
    let mut text = match format {
        QCryptoSecretFormat::Base64 => base64::decode(input)?,
        QCryptoSecretFormat::Raw => input.to_vec(),
    };
    aes.decrypt(&mut text)?;
    // QEMU reads plaintext[len - 1] even for empty input, where it is the byte before the
    // buffer; an empty ciphertext is refused here instead.
    let pad = text.last().copied().unwrap_or(0);
    if text.is_empty() || usize::from(pad) > 16 || usize::from(pad) > text.len() {
        return Err(Error::generic(format!(
            "Incorrect number of padding bytes ({pad}) found on decrypted data"
        )));
    }
    text.truncate(text.len() - usize::from(pad));
    Ok(text)
}

/// `qcrypto_secret_complete()`.
fn complete(obj: &Object) -> Result<()> {
    let (format, keyid, iv, data, file) = with_fields(obj, |f| {
        (f.format, f.keyid.clone(), f.iv.clone(), f.data.clone(), f.file.clone())
    });
    let input = load_data(data.as_deref(), file.as_deref())?;
    let raw = if let Some(keyid) = keyid {
        decrypt(&obj.registry(), format, &keyid, iv.as_deref(), &input)?
    } else if format == QCryptoSecretFormat::Base64 {
        base64::decode(&input)?
    } else {
        input
    };
    with_fields(obj, |f| f.rawdata = Some(raw));
    Ok(())
}

fn str_prop(
    k: &Arc<ruvm_qom::ObjectClass>,
    name: &str,
    field: fn(&mut Fields) -> &mut Option<String>,
) {
    k.property_add_str(
        name,
        Some(Arc::new(move |o: &Object| {
            Ok(with_fields(o, |f| field(f).clone().unwrap_or_default()))
        })),
        Some(Arc::new(move |o: &Object, v: &str| {
            with_fields(o, |f| *field(f) = Some(v.to_string()));
            Ok(())
        })),
    );
}

/// Registers `secret_common` and `secret` with `registry`, unless they are there already.
pub fn register_types(registry: &Registry) {
    if registry.type_exists(TYPE_QCRYPTO_SECRET_COMMON) {
        return;
    }
    let common = TypeInfo::new(TYPE_QCRYPTO_SECRET_COMMON)
        .parent(TYPE_OBJECT)
        .abstract_()
        .interface(TYPE_USER_CREATABLE)
        .instance_state(SecretState::default)
        .class_init(|k| {
            let uc = k.interface(TYPE_USER_CREATABLE).expect("secrets are user creatable");
            uc.set_ext(UserCreatableClass {
                complete: Some(Arc::new(complete)),
                prepare_delete: None,
            });
            k.property_add_enum(
                "format",
                "QCryptoSecretFormat",
                QCryptoSecretFormat::LOOKUP,
                Some(Arc::new(|o: &Object| Ok(with_fields(o, |f| f.format as usize)))),
                Some(Arc::new(|o: &Object, v: usize| {
                    with_fields(o, |f| f.format = QCryptoSecretFormat::ALL[v]);
                    Ok(())
                })),
            );
            str_prop(k, "keyid", |f| &mut f.keyid);
            str_prop(k, "iv", |f| &mut f.iv);
        });
    let secret =
        TypeInfo::new(TYPE_QCRYPTO_SECRET).parent(TYPE_QCRYPTO_SECRET_COMMON).class_init(|k| {
            str_prop(k, "data", |f| &mut f.data);
            str_prop(k, "file", |f| &mut f.file);
        });
    registry.register_all([common, secret]);
}

/// `qcrypto_secret_lookup()` in the global registry.
pub fn secret_lookup(secretid: &str) -> Result<Vec<u8>> {
    secret_lookup_in(Registry::global(), secretid)
}

/// `qcrypto_secret_lookup()`: the raw data of the secret `secretid` in `/objects`.
pub fn secret_lookup_in(registry: &Registry, secretid: &str) -> Result<Vec<u8>> {
    let Some(obj) = registry.objects_root().resolve_path_component(secretid) else {
        return Err(Error::generic(format!("No secret with id '{secretid}'")));
    };
    if obj.dynamic_cast(TYPE_QCRYPTO_SECRET_COMMON).is_none() {
        return Err(Error::generic(format!("Object with id '{secretid}' is not a secret")));
    }
    with_fields(&obj, |f| f.rawdata.clone())
        .ok_or_else(|| Error::generic(format!("Secret with id '{secretid}' has no data")))
}

/// `qcrypto_secret_lookup_as_utf8()` in the global registry.
pub fn secret_lookup_as_utf8(secretid: &str) -> Result<String> {
    secret_lookup_as_utf8_in(Registry::global(), secretid)
}

/// `qcrypto_secret_lookup_as_utf8()`. Like `g_utf8_validate()` with a length, a NUL byte makes
/// the data invalid.
pub fn secret_lookup_as_utf8_in(registry: &Registry, secretid: &str) -> Result<String> {
    let data = secret_lookup_in(registry, secretid)?;
    match String::from_utf8(data) {
        Ok(s) if !s.contains('\0') => Ok(s),
        _ => Err(Error::generic(format!("Data from secret {secretid} is not valid UTF-8"))),
    }
}

/// `qcrypto_secret_lookup_as_base64()` in the global registry.
pub fn secret_lookup_as_base64(secretid: &str) -> Result<String> {
    secret_lookup_as_base64_in(Registry::global(), secretid)
}

/// `qcrypto_secret_lookup_as_base64()`.
pub fn secret_lookup_as_base64_in(registry: &Registry, secretid: &str) -> Result<String> {
    Ok(base64::encode(&secret_lookup_in(registry, secretid)?))
}

/// Creates a secret from `--object` style text such as `secret,id=sec0,data=123456`, the way
/// `qemu-img --object` does: the text is parsed by `keyval_parse()` with `qom-type` as the
/// implied key and handed to `user_creatable_add()`. The secret types are registered first.
pub fn secret_object_add(registry: &Registry, spec: &str) -> Result<Object> {
    register_types(registry);
    let args = ruvm_qapi::keyval::keyval_parse(spec, Some("qom-type"), None)?;
    registry.user_creatable_add(&args, true)
}

/// [`secret_object_add`] on the global registry, where `--object` puts objects and where
/// [`secret_lookup`] finds them.
pub fn secret_object_add_global(spec: &str) -> Result<Object> {
    secret_object_add(Registry::global(), spec)
}
