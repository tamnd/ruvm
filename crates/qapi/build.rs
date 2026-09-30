// SPDX-License-Identifier: GPL-2.0-or-later

//! Builds the `query-qmp-schema` reply from the vendored QAPI schema.
//!
//! The QObject, JSON and number formatting code is shared with the library through `#[path]`, so the reply is
//! printed by the same writer QMP uses and comes out byte for byte as QEMU prints it.

use std::path::Path;

#[allow(dead_code, unreachable_pub)]
#[path = "src/cutils.rs"]
mod cutils;
#[allow(dead_code, unreachable_pub)]
#[path = "src/json.rs"]
mod json;
#[allow(dead_code, unreachable_pub)]
#[path = "src/qvalue.rs"]
mod qvalue;

use qvalue::{QDict, QValue};
use ruvm_qapi_gen::{Lit, Schema, introspect};

/// How each build condition in the schema is decided. A condition missing from this table
/// fails the build, so a QEMU update that adds one has to be looked at.
enum Rule {
    /// Set when building for this `target_os`.
    Os(&'static [&'static str]),
    /// Set on every unix target.
    Unix,
    /// Always set.
    On,
    /// Not set until ruvm has the feature behind it.
    Off,
}

const CONDITIONS: &[(&str, Rule)] = &[
    ("CONFIG_AF_XDP", Rule::Off),
    ("CONFIG_AUDIO_ALSA", Rule::Off),
    ("CONFIG_AUDIO_COREAUDIO", Rule::Off),
    ("CONFIG_AUDIO_DSOUND", Rule::Off),
    ("CONFIG_AUDIO_JACK", Rule::Off),
    ("CONFIG_AUDIO_OSS", Rule::Off),
    ("CONFIG_AUDIO_PA", Rule::Off),
    ("CONFIG_AUDIO_PIPEWIRE", Rule::Off),
    ("CONFIG_AUDIO_SDL", Rule::Off),
    ("CONFIG_AUDIO_SNDIO", Rule::Off),
    ("CONFIG_BLKIO", Rule::Off),
    ("CONFIG_BLKIO_VHOST_VDPA_FD", Rule::Off),
    ("CONFIG_BRLAPI", Rule::Off),
    ("CONFIG_COCOA", Rule::Off),
    ("CONFIG_CURSES", Rule::Off),
    ("CONFIG_DBUS_DISPLAY", Rule::Off),
    ("CONFIG_EBPF", Rule::Off),
    ("CONFIG_FDT", Rule::On),
    ("CONFIG_FUSE", Rule::Off),
    ("CONFIG_GTK", Rule::Off),
    ("CONFIG_IGVM", Rule::Off),
    ("CONFIG_LIBPMEM", Rule::Off),
    ("CONFIG_LINUX", Rule::Os(&["linux"])),
    ("CONFIG_LINUX_IO_URING", Rule::Off),
    ("CONFIG_OPENGL", Rule::Off),
    ("CONFIG_PASST", Rule::Off),
    ("CONFIG_PIXMAN", Rule::Off),
    ("CONFIG_POSIX", Rule::Unix),
    ("CONFIG_QATZIP", Rule::Off),
    ("CONFIG_QPL", Rule::Off),
    ("CONFIG_REPLICATION", Rule::Off),
    ("CONFIG_SDL", Rule::Off),
    ("CONFIG_SECRET_KEYRING", Rule::Off),
    ("CONFIG_SPICE", Rule::Off),
    ("CONFIG_SPICE_PROTOCOL", Rule::Off),
    ("CONFIG_TCG", Rule::On),
    ("CONFIG_TPM", Rule::Off),
    ("CONFIG_UADK", Rule::Off),
    ("CONFIG_VDUSE_BLK_EXPORT", Rule::Off),
    ("CONFIG_VHOST_CRYPTO", Rule::Off),
    ("CONFIG_VHOST_USER_BLK_SERVER", Rule::Off),
    ("CONFIG_VMNET", Rule::Off),
    ("CONFIG_VNC", Rule::Off),
    ("CONFIG_WIN32", Rule::Os(&["windows"])),
    ("CONFIG_ZSTD", Rule::Off),
    ("HAVE_CHARDEV_PARALLEL", Rule::Os(&["linux", "freebsd", "dragonfly"])),
    ("HAVE_CHARDEV_SERIAL", Rule::Unix),
    ("HAVE_HOST_BLOCK_DEVICE", Rule::Unix),
    ("HAVE_IPPROTO_MPTCP", Rule::Os(&["linux"])),
    ("HAVE_TCP_KEEPCNT", Rule::Unix),
    ("HAVE_TCP_KEEPIDLE", Rule::Unix),
    ("HAVE_TCP_KEEPINTVL", Rule::Unix),
];

fn to_qvalue(l: &Lit) -> QValue {
    match l {
        Lit::Null => QValue::Null,
        Lit::Bool(b) => QValue::Bool(*b),
        Lit::Str(s) => QValue::str(s),
        Lit::List(items) => QValue::List(items.iter().map(|a| to_qvalue(&a.value)).collect()),
        Lit::Dict(d) => {
            // QEMU's C literal lists keys sorted, and qobject_from_qlit() inserts them in that order.
            let mut d: Vec<_> = d.iter().collect();
            d.sort_by(|a, b| a.0.cmp(&b.0));
            let mut q = QDict::new();
            for (k, v) in d {
                q.put(k.clone(), to_qvalue(v));
            }
            QValue::Dict(q)
        }
    }
}

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let qapi = Path::new(&manifest).join("../../vendor-qemu/qapi");
    println!("cargo::rerun-if-changed={}", qapi.display());
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let unix = std::env::var("CARGO_CFG_UNIX").is_ok();

    let schema = Schema::load(&qapi.join("qapi-schema.json")).unwrap_or_else(|e| panic!("{e}"));
    let trees = introspect(&schema, false);
    for sym in introspect::symbols(&trees) {
        if !CONDITIONS.iter().any(|(n, _)| *n == sym) {
            panic!(
                "the QAPI schema uses condition {sym}, which crates/qapi/build.rs does not know about"
            );
        }
    }
    let is_set = |sym: &str| match CONDITIONS.iter().find(|(n, _)| *n == sym).map(|(_, r)| r) {
        Some(Rule::On) => true,
        Some(Rule::Unix) => unix,
        Some(Rule::Os(list)) => list.contains(&os.as_str()),
        Some(Rule::Off) | None => false,
    };
    let list = introspect::resolve(&trees, &is_set);
    let json = QValue::List(list.iter().map(to_qvalue).collect()).to_json();
    let out =
        Path::new(&std::env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join("qmp-schema.json");
    std::fs::write(out, json).expect("write the schema");
}
