// SPDX-License-Identifier: GPL-2.0-or-later

//! `-readconfig`: `qemu_read_config_file()` from util/qemu-config.c with the
//! `qemu_parse_config_group()` callback from system/vl.c.
//!
//! Each group of the file becomes the command line option it stands for, and
//! `parse_options()` handles those options where `-readconfig` was. QEMU only records a
//! QemuOpts group and does not run the option's code, so the caller tells the options from a
//! file apart where that shows: `-mon` warns and drops the default monitor, a `[mon]` group
//! does neither.

use std::fs;

use ruvm_base::error::strerror;

/// The groups `qemu_find_opts()` knows after `qemu_init()` adds them, with the option each
/// one is on the command line, and the five that `is_qemuopts_group()` leaves to
/// `qemu_record_config_group()`.
const GROUPS: &[(&str, &str)] = &[
    ("drive", "drive"),
    ("chardev", "chardev"),
    ("device", "device"),
    ("netdev", "netdev"),
    ("nic", "nic"),
    ("net", "net"),
    ("rtc", "rtc"),
    ("global", "global"),
    ("mon", "mon"),
    ("trace", "trace"),
    ("option-rom", "option-rom"),
    ("accel", "accel"),
    ("memory", "m"),
    ("add-fd", "add-fd"),
    ("tpmdev", "tpmdev"),
    ("overcommit", "overcommit"),
    ("msg", "msg"),
    ("name", "name"),
    ("numa", "numa"),
    ("icount", "icount"),
    ("semihosting-config", "semihosting-config"),
    ("fw_cfg", "fw_cfg"),
    ("action", "action"),
    ("run-with", "run-with"),
    ("object", "object"),
    ("audiodev", "audiodev"),
    ("machine", "machine"),
    ("smp-opts", "smp"),
    ("boot-opts", "boot"),
];

/// Why a config file was not read. `line` is the line `loc_set_file()` last set, which the
/// message is printed against instead of the command line option. Without one the message
/// goes against `-readconfig`, where QEMU prints it with no location at all.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ConfigError {
    pub line: Option<u32>,
    pub msg: String,
}

/// A group and its `key = "value"` pairs in file order, the QDict `qemu_config_foreach()`
/// builds. A later key replaces an earlier one, as `qdict_put_str()` does.
#[derive(Debug, Default)]
struct Group {
    name: String,
    pairs: Vec<(String, String)>,
}

impl Group {
    fn put(&mut self, key: String, value: String) {
        match self.pairs.iter_mut().find(|(k, _)| *k == key) {
            Some(pair) => pair.1 = value,
            None => self.pairs.push((key, value)),
        }
    }

    /// The command line option and its argument, with the commas in values doubled the way
    /// QemuOpts and keyval read them.
    fn to_option(&self, line: Option<u32>) -> Result<[String; 2], ConfigError> {
        let Some((_, option)) = GROUPS.iter().find(|(g, _)| *g == self.name) else {
            return Err(ConfigError {
                line,
                msg: format!("There is no option group '{}'", self.name),
            });
        };
        let arg: Vec<String> =
            self.pairs.iter().map(|(k, v)| format!("{k}={}", v.replace(',', ",,"))).collect();
        Ok([format!("-{option}"), arg.join(",")])
    }
}

/// `qemu_read_config_file()`: the options the groups of `path` stand for, in file order.
pub(crate) fn read_config_file(path: &str) -> Result<Vec<String>, ConfigError> {
    let text = fs::read(path).map_err(|e| ConfigError {
        line: None,
        msg: format!("Could not open '{path}': {}", strerror(&e)),
    })?;
    parse(&String::from_utf8_lossy(&text))
}

/// `qemu_config_foreach()`, which reads the lines with `sscanf()`.
fn parse(text: &str) -> Result<Vec<String>, ConfigError> {
    let mut out = Vec::new();
    let mut group: Option<Group> = None;
    let mut loc = None;
    for (i, line) in text.split_inclusive('\n').enumerate() {
        let lno = i as u32 + 1;
        if line.starts_with('\n') || line.starts_with('#') {
            continue;
        }
        if let Some(name) = line.strip_prefix('[') {
            let header = header_with_id(name).or_else(|| header_bare(name));
            if let Some((name, id)) = header {
                if let Some(prev) = group.take() {
                    out.extend(prev.to_option(loc)?);
                }
                let mut g = Group { name, pairs: Vec::new() };
                if let Some(id) = id {
                    g.put("id".to_string(), id);
                }
                group = Some(g);
                continue;
            }
        }
        loc = Some(lno);
        let Some((key, value)) = key_value(line) else {
            return Err(ConfigError { line: loc, msg: "parse error".to_string() });
        };
        let Some(g) = group.as_mut() else {
            return Err(ConfigError { line: loc, msg: "no group defined".to_string() });
        };
        g.put(key, value);
    }
    if let Some(g) = group {
        out.extend(g.to_option(loc)?);
    }
    Ok(out)
}

/// `%Ns`: up to `n` characters that are not white space, after skipping white space.
fn scan_word(s: &str, n: usize) -> Option<(&str, &str)> {
    let s = s.trim_start();
    let end = s
        .char_indices()
        .take(n)
        .find(|(_, c)| c.is_ascii_whitespace())
        .map_or_else(|| s.char_indices().nth(n).map_or(s.len(), |(i, _)| i), |(i, _)| i);
    (end > 0).then(|| s.split_at(end))
}

/// `%N[^c]`: one to `n` characters other than `stop`.
fn scan_until(s: &str, stop: char, n: usize) -> Option<(&str, &str)> {
    let end = s
        .char_indices()
        .take(n)
        .find(|&(_, c)| c == stop)
        .map_or_else(|| s.char_indices().nth(n).map_or(s.len(), |(i, _)| i), |(i, _)| i);
    (end > 0).then(|| s.split_at(end))
}

/// `[%63s "%63[^"]"]`, which needs both conversions.
fn header_with_id(s: &str) -> Option<(String, Option<String>)> {
    let (name, rest) = scan_word(s, 63)?;
    let rest = rest.trim_start().strip_prefix('"')?;
    let (id, _) = scan_until(rest, '"', 63)?;
    Some((name.to_string(), Some(id.to_string())))
}

/// `[%63[^]]]`.
fn header_bare(s: &str) -> Option<(String, Option<String>)> {
    let (name, _) = scan_until(s, ']', 63)?;
    Some((name.to_string(), None))
}

/// ` %63s = "%1023[^"]"`, or ` %63s = ""` with an empty value. Either gives the key once
/// `%63s` matched, so a line with a word on it is always a key.
fn key_value(line: &str) -> Option<(String, String)> {
    let (key, rest) = scan_word(line, 63)?;
    let value = rest
        .trim_start()
        .strip_prefix('=')
        .and_then(|r| r.trim_start().strip_prefix('"'))
        .and_then(|r| scan_until(r, '"', 1023))
        .map_or("", |(v, _)| v);
    Some((key.to_string(), value.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(line: Option<u32>, msg: &str) -> ConfigError {
        ConfigError { line, msg: msg.to_string() }
    }

    #[test]
    fn groups_become_options() {
        let text = "# comment\n\n[object]\nqom-type = \"rng-builtin\"\nid = \"rng0\"\n\
                    [device \"rng\"]\n  driver = \"virtio-rng-pci\"\nrng = \"rng0\"\n\
                    [memory]\nsize = \"200M\"\n[smp-opts]\ncpus = \"2\"\n\
                    [name]\nguest = \"a,b\"\ndebug-threads = \"\"\n";
        assert_eq!(
            parse(text).unwrap(),
            [
                "-object",
                "qom-type=rng-builtin,id=rng0",
                "-device",
                "id=rng,driver=virtio-rng-pci,rng=rng0",
                "-m",
                "size=200M",
                "-smp",
                "cpus=2",
                "-name",
                "guest=a,,b,debug-threads=",
            ]
        );
    }

    #[test]
    fn sscanf_quirks() {
        // A later key replaces an earlier one, and a word alone is a key with no value.
        let text = "[machine]\ntype = \"virt\"\ntype = \"none\"\nusb=\"on\"\n";
        assert_eq!(parse(text).unwrap(), ["-machine", "type=none,usb=\"on\"="]);
        // The closing quote and bracket are not checked, and the value runs on to the newline.
        assert_eq!(parse("[device \"d0\"\nx = \"y\n").unwrap(), ["-device", "id=d0,x=y\n"]);
        assert_eq!(parse("").unwrap(), Vec::<String>::new());
    }

    #[test]
    fn errors() {
        assert_eq!(parse("x = \"1\"\n").unwrap_err(), err(Some(1), "no group defined"));
        assert_eq!(parse("[device]\n   \n").unwrap_err(), err(Some(2), "parse error"));
        // "[]" is no header, so it is read as a key.
        assert_eq!(parse("[]\n").unwrap_err(), err(Some(1), "no group defined"));
        assert_eq!(parse("[device]\n[]\n").unwrap(), ["-device", "[]="]);
        assert_eq!(
            parse("[foo]\na = \"1\"\n\n[device]\n").unwrap_err(),
            err(Some(2), "There is no option group 'foo'")
        );
        assert_eq!(parse("[foo]\n").unwrap_err(), err(None, "There is no option group 'foo'"));
        let e = read_config_file("/nonexistent/ruvm.cfg").unwrap_err();
        assert_eq!(
            e,
            err(None, "Could not open '/nonexistent/ruvm.cfg': No such file or directory")
        );
    }
}
