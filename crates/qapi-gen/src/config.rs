// SPDX-License-Identifier: GPL-2.0-or-later

//! The build conditions QEMU's inputs test, and how ruvm decides each one.
//!
//! The QAPI schema has `'if'` conditions and the `.hx` tables have `#ifdef` lines. Both name
//! meson's `CONFIG_*` symbols or a compiler's host macros, and both generators look them up
//! here. A symbol missing from the table is an error, so a QEMU update that adds one fails the
//! build until someone decides what it means for ruvm.

/// How one build condition is decided.
#[derive(Debug, Clone, Copy)]
pub enum Rule {
    /// Set when building for one of these `target_os` values.
    Os(&'static [&'static str]),
    /// Set on every unix target.
    Unix,
    /// Always set.
    On,
    /// Not set until ruvm has the feature behind it.
    Off,
    /// Set when the crate whose build script asks has this cargo feature.
    Feature(&'static str),
    /// Set when the crate whose build script asks has this cargo feature and builds for one of
    /// these `target_os` values.
    FeatureOs(&'static str, &'static [&'static str]),
}

const CONDITIONS: &[(&str, Rule)] = &[
    ("CONFIG_AF_XDP", Rule::Off),
    ("CONFIG_AUDIO_ALSA", Rule::Feature("audio-alsa")),
    ("CONFIG_AUDIO_COREAUDIO", Rule::FeatureOs("audio-coreaudio", &["macos"])),
    ("CONFIG_AUDIO_DSOUND", Rule::Off),
    ("CONFIG_AUDIO_JACK", Rule::Off),
    ("CONFIG_AUDIO_OSS", Rule::Off),
    ("CONFIG_AUDIO_PA", Rule::Feature("audio-pa")),
    ("CONFIG_AUDIO_PIPEWIRE", Rule::Feature("audio-pipewire")),
    ("CONFIG_AUDIO_SDL", Rule::Off),
    ("CONFIG_AUDIO_SNDIO", Rule::Off),
    ("CONFIG_BLKIO", Rule::Off),
    ("CONFIG_BLKIO_VHOST_VDPA_FD", Rule::Off),
    ("CONFIG_BRLAPI", Rule::Off),
    ("CONFIG_COCOA", Rule::FeatureOs("ui-cocoa", &["macos"])),
    ("CONFIG_CURSES", Rule::Off),
    ("CONFIG_DBUS_DISPLAY", Rule::Feature("ui-dbus")),
    ("CONFIG_EBPF", Rule::Off),
    ("CONFIG_FDT", Rule::On),
    ("CONFIG_FUSE", Rule::Off),
    ("CONFIG_GTK", Rule::Feature("ui-gtk")),
    ("CONFIG_IGVM", Rule::Off),
    ("CONFIG_LIBPMEM", Rule::Off),
    ("CONFIG_LINUX", Rule::Os(&["linux"])),
    ("CONFIG_LINUX_IO_URING", Rule::Off),
    ("CONFIG_NETMAP", Rule::Off),
    ("CONFIG_OPENGL", Rule::Off),
    ("CONFIG_PASST", Rule::Off),
    ("CONFIG_PIXMAN", Rule::On),
    ("CONFIG_POSIX", Rule::Unix),
    ("CONFIG_QATZIP", Rule::Off),
    ("CONFIG_QPL", Rule::Off),
    ("CONFIG_REPLICATION", Rule::Off),
    ("CONFIG_SDL", Rule::Feature("ui-sdl")),
    ("CONFIG_SECRET_KEYRING", Rule::Off),
    ("CONFIG_SLIRP", Rule::Off),
    ("CONFIG_SPICE", Rule::Off),
    ("CONFIG_SPICE_PROTOCOL", Rule::Off),
    ("CONFIG_TCG", Rule::On),
    ("CONFIG_TPM", Rule::Off),
    ("CONFIG_TRACE_SIMPLE", Rule::Off),
    ("CONFIG_UADK", Rule::Off),
    ("CONFIG_VDE", Rule::Off),
    ("CONFIG_VDUSE_BLK_EXPORT", Rule::Off),
    ("CONFIG_VHOST_CRYPTO", Rule::Off),
    ("CONFIG_VHOST_USER_BLK_SERVER", Rule::Off),
    ("CONFIG_VMNET", Rule::Off),
    ("CONFIG_VNC", Rule::On),
    ("CONFIG_WIN32", Rule::Os(&["windows"])),
    ("CONFIG_ZSTD", Rule::Off),
    ("EMSCRIPTEN", Rule::Os(&["emscripten"])),
    ("HAVE_CHARDEV_PARALLEL", Rule::Os(&["linux", "freebsd", "dragonfly"])),
    ("HAVE_CHARDEV_SERIAL", Rule::Unix),
    ("HAVE_HOST_BLOCK_DEVICE", Rule::Unix),
    ("HAVE_IPPROTO_MPTCP", Rule::Os(&["linux"])),
    ("HAVE_TCP_KEEPCNT", Rule::Unix),
    ("HAVE_TCP_KEEPIDLE", Rule::Unix),
    ("HAVE_TCP_KEEPINTVL", Rule::Unix),
    ("_WIN32", Rule::Os(&["windows"])),
    ("__DragonFly__", Rule::Os(&["dragonfly"])),
    ("__FreeBSD__", Rule::Os(&["freebsd"])),
    ("__NetBSD__", Rule::Os(&["netbsd"])),
    ("__OpenBSD__", Rule::Os(&["openbsd"])),
    ("__linux__", Rule::Os(&["linux", "android"])),
    ("__sun__", Rule::Os(&["solaris", "illumos"])),
];

/// The rule for `sym`, or `None` if the table does not know it.
pub fn rule(sym: &str) -> Option<Rule> {
    CONDITIONS.iter().find(|(n, _)| *n == sym).map(|(_, r)| *r)
}

/// The conditions of one build, from the `target_os` and whether the target is unix.
#[derive(Debug, Clone)]
pub struct Config {
    os: String,
    unix: bool,
    features: Vec<String>,
}

impl Config {
    pub fn new(os: &str, unix: bool) -> Self {
        Config { os: os.to_string(), unix, features: Vec::new() }
    }

    /// The same conditions with these cargo features on.
    pub fn with_features(mut self, features: &[&str]) -> Self {
        self.features = features.iter().map(|f| f.to_string()).collect();
        self
    }

    /// The target of the build script calling this, from the variables cargo sets.
    pub fn from_cargo_env() -> Self {
        let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let features: Vec<String> = std::env::vars()
            .filter_map(|(k, _)| {
                k.strip_prefix("CARGO_FEATURE_").map(|f| f.to_lowercase().replace('_', "-"))
            })
            .collect();
        let features: Vec<&str> = features.iter().map(String::as_str).collect();
        Config::new(&os, std::env::var("CARGO_CFG_UNIX").is_ok()).with_features(&features)
    }

    /// Whether `sym` is set. Unknown symbols are not, and [`rule`] is how a caller finds them
    /// first.
    pub fn is_set(&self, sym: &str) -> bool {
        match rule(sym) {
            Some(Rule::On) => true,
            Some(Rule::Unix) => self.unix,
            Some(Rule::Os(list)) => list.contains(&self.os.as_str()),
            Some(Rule::Feature(f)) => self.features.iter().any(|x| x == f),
            Some(Rule::FeatureOs(f, list)) => {
                self.features.iter().any(|x| x == f) && list.contains(&self.os.as_str())
            }
            Some(Rule::Off) | None => false,
        }
    }
}
