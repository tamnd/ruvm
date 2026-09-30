// SPDX-License-Identifier: GPL-2.0-or-later

//! The `secret` object against QEMU's tests/unit/test-crypto-secret.c, plus the error messages.

use ruvm_crypto::secret::{
    secret_lookup_as_base64_in, secret_lookup_as_utf8_in, secret_lookup_in, secret_object_add,
};
use ruvm_qom::Registry;

fn reg() -> Registry {
    Registry::new()
}

fn add_err(r: &Registry, spec: &str) -> String {
    secret_object_add(r, spec).map(|_| ()).unwrap_err().message().to_string()
}

const MASTER: &str =
    "secret,id=master,data=9miloPQCzGy+TL6aonfzVcptibCmCIhKzrnlfwiWivk=,format=base64";

#[test]
fn direct() {
    let r = reg();
    secret_object_add(&r, "secret,id=sec0,data=123456").unwrap();
    assert_eq!(secret_lookup_as_utf8_in(&r, "sec0").unwrap(), "123456");
}

#[test]
fn indirect_good_and_empty() {
    let dir = std::env::temp_dir().join(format!("ruvm-secret-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let good = dir.join("good");
    std::fs::write(&good, "123456").unwrap();
    let empty = dir.join("empty");
    std::fs::write(&empty, "").unwrap();
    let r = reg();
    secret_object_add(&r, &format!("secret,id=sec0,file={}", good.display())).unwrap();
    assert_eq!(secret_lookup_as_utf8_in(&r, "sec0").unwrap(), "123456");
    secret_object_add(&r, &format!("secret,id=sec1,file={}", empty.display())).unwrap();
    assert_eq!(secret_lookup_as_utf8_in(&r, "sec1").unwrap(), "");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn indirect_badfile() {
    let r = reg();
    let e = add_err(&r, "secret,id=sec0,file=does-not-exist");
    assert_eq!(
        e,
        "Unable to read does-not-exist: Failed to open file \u{201c}does-not-exist\u{201d}: No such file or directory"
    );
}

#[test]
fn data_and_file() {
    let r = reg();
    assert_eq!(
        add_err(&r, "secret,id=s,data=a,file=b"),
        "'file' and 'data' are mutually exclusive"
    );
    assert_eq!(add_err(&r, "secret,id=s"), "Either 'file' or 'data' must be provided");
}

#[test]
fn noconv_base64() {
    let r = reg();
    secret_object_add(&r, "secret,id=sec0,data=MTIzNDU2,format=base64").unwrap();
    assert_eq!(secret_lookup_as_base64_in(&r, "sec0").unwrap(), "MTIzNDU2");
    assert_eq!(
        add_err(&r, "secret,id=sec1,data=MTI$NDU2,format=base64"),
        "Base64 data contains invalid characters"
    );
}

#[test]
fn noconv_utf8_and_conversions() {
    let r = reg();
    secret_object_add(&r, "secret,id=sec0,data=123456,format=raw").unwrap();
    assert_eq!(secret_lookup_as_utf8_in(&r, "sec0").unwrap(), "123456");
    assert_eq!(secret_lookup_as_base64_in(&r, "sec0").unwrap(), "MTIzNDU2");
    secret_object_add(&r, "secret,id=sec1,data=MTIzNDU2,format=base64").unwrap();
    assert_eq!(secret_lookup_as_utf8_in(&r, "sec1").unwrap(), "123456");
    secret_object_add(&r, "secret,id=sec2,data=f0VMRgIBAQAAAA==,format=base64").unwrap();
    assert_eq!(
        secret_lookup_as_utf8_in(&r, "sec2").unwrap_err().message(),
        "Data from secret sec2 is not valid UTF-8"
    );
    assert_eq!(secret_lookup_in(&r, "sec2").unwrap(), b"\x7fELF\x02\x01\x01\0\0\0");
}

#[test]
fn crypt_base64() {
    let r = reg();
    secret_object_add(&r, MASTER).unwrap();
    secret_object_add(
        &r,
        "secret,id=sec0,data=zL/3CUYZC1IqOrRrzXqwsA==,format=base64,keyid=master,iv=0I7Gw/TKuA+Old2W2apQ3g==",
    )
    .unwrap();
    assert_eq!(secret_lookup_as_utf8_in(&r, "sec0").unwrap(), "123456");
}

#[test]
fn crypt_errors() {
    let r = reg();
    secret_object_add(&r, MASTER).unwrap();
    secret_object_add(&r, "secret,id=short,data=9miloPQCzGy+TL6aonfzVc,format=base64").unwrap();
    let sec = "secret,id=sec0,data=zL/3CUYZC1IqOrRrzXqwsA==,format=raw";
    assert_eq!(
        add_err(&r, &format!("{sec},keyid=short,iv=0I7Gw/TKuA+Old2W2apQ3g==")),
        "Key should be 32 bytes in length"
    );
    assert_eq!(
        add_err(&r, &format!("{sec},keyid=master,iv=0I7Gw/TKuA+Old2W2a")),
        "IV should be 16 bytes in length not 12"
    );
    assert_eq!(add_err(&r, &format!("{sec},keyid=master")), "IV is required to decrypt secret");
    assert_eq!(
        add_err(&r, &format!("{sec},keyid=master,iv=0I7Gw/TK$$uA+Old2W2a")),
        "Base64 data contains invalid characters"
    );
}

#[test]
fn lookup_errors() {
    let r = reg();
    secret_object_add(&r, "secret,id=sec0,data=x").unwrap();
    assert_eq!(secret_lookup_in(&r, "nope").unwrap_err().message(), "No secret with id 'nope'");
    // Raw data is not base64 decoded, so a raw ciphertext that is not base64 text fails the same
    // way as a base64 one in QEMU: through the key lookup.
    assert_eq!(
        add_err(&r, "secret,id=sec1,data=abc,keyid=nope,iv=0I7Gw/TKuA+Old2W2apQ3g=="),
        "No secret with id 'nope'"
    );
}
