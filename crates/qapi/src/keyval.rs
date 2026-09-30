// SPDX-License-Identifier: GPL-2.0-or-later

//! The KEY=VALUE,... parser from util/keyval.c, used by `-blockdev`, `-object`, `-audiodev`,
//! `-display` and the other options that go through QAPI instead of QemuOpts.
//!
//! The grammar, as QEMU's comment gives it:
//!
//! ```text
//! key-vals     = [ key-val { ',' key-val } [ ',' ] ]
//! key-val      = key '=' val | help
//! key          = key-fragment { '.' key-fragment }
//! key-fragment = qapi-name | index
//! qapi-name    = '__' / [a-z0-9.-]+ / '_' / [A-Za-z][A-Za-z0-9_-]* /
//! index        = / [0-9]+ /
//! val          = { / [^,]+ / | ',,' }
//! help         = 'help' | '?'
//! ```
//!
//! Dotted keys build nested dicts, and a dict whose keys are all numbers becomes a list, with no
//! gaps allowed. Every leaf is a string. A key fragment is 1 to 127 characters long, and when the
//! last key-val names the same leaf as an earlier one, the last one wins.
//!
//! With an implied key the first key-val may leave out `key=`. Its value then can't be empty and
//! can't contain `,` or `=`.
//!
//! The strings are turned into numbers, booleans and so on by the keyval flavour of the QObject
//! input visitor, [`crate::visit::QObjectInputVisitor::new_keyval`].

use ruvm_base::{Error, Result};

use crate::qvalue::{QDict, QValue};

/// `key_to_index()`: the leading decimal digits of `key` as a list index, capped at `i32::MAX`.
///
/// With `need_all` any text after the digits is an error, otherwise the number of digits comes
/// back with the index. `None` means `key` does not start with a digit, or has text after the
/// digits when that is not allowed.
fn key_to_index(key: &[u8], need_all: bool) -> Option<(u32, usize)> {
    let digits = key.iter().take_while(|b| b.is_ascii_digit()).count();
    if digits == 0 || (need_all && digits != key.len()) {
        return None;
    }
    let mut index: u64 = 0;
    for &b in &key[..digits] {
        index = index.saturating_mul(10).saturating_add(u64::from(b - b'0'));
    }
    Some((index.min(i32::MAX as u64) as u32, digits))
}

/// `parse_qapi_name()` from qapi/qapi-util.c without `complete`: the length of the QAPI name at
/// the start of `s`, if there is one. A name is letters, digits, `-` and `_` starting with a
/// letter, optionally behind a `__RFQDN_` downstream prefix.
fn parse_qapi_name(s: &[u8]) -> Option<usize> {
    let at = |i: usize| s.get(i).copied().unwrap_or(0);
    let mut p = 0;
    if at(p) == b'_' {
        p += 1;
        if at(p) != b'_' {
            return None;
        }
        p += 1;
        while at(p).is_ascii_alphanumeric() || at(p) == b'-' || at(p) == b'.' {
            p += 1;
        }
        if at(p) != b'_' {
            return None;
        }
        p += 1;
    }
    if !at(p).is_ascii_alphabetic() {
        return None;
    }
    p += 1;
    while at(p).is_ascii_alphanumeric() || at(p) == b'-' || at(p) == b'_' {
        p += 1;
    }
    Some(p)
}

/// `starts_with_help_option()` from include/qemu/help_option.h.
fn starts_with_help_option(s: &str) -> usize {
    if s.starts_with('?') {
        1
    } else if s.starts_with("help") {
        4
    } else {
        0
    }
}

/// The part of `key` before `end`. Every place this is called with ends on an ASCII byte or at
/// the end of the string, so the slice is always on a character boundary.
fn prefix(key: &str, end: usize) -> &str {
    &key[..end]
}

/// `keyval_parse_put()` for an inner node: makes sure `cur[key_in_cur]` is a dict and returns it.
/// `key[..cursor]` names the node in the error message.
fn put_dict<'a>(
    cur: &'a mut QDict,
    key_in_cur: &str,
    key: &str,
    cursor: usize,
) -> Result<&'a mut QDict> {
    if !cur.contains_key(key_in_cur) {
        cur.put(key_in_cur, QDict::new());
    }
    match cur.get_mut(key_in_cur) {
        Some(QValue::Dict(d)) => Ok(d),
        _ => Err(Error::generic(format!(
            "Parameters '{}.*' used inconsistently",
            prefix(key, cursor)
        ))),
    }
}

/// `keyval_parse_put()` for a leaf: puts `value` unless `cur[key_in_cur]` is already something
/// other than a string.
fn put_str(
    cur: &mut QDict,
    key_in_cur: &str,
    value: String,
    key: &str,
    cursor: usize,
) -> Result<()> {
    match cur.get(key_in_cur) {
        None | Some(QValue::Str(_)) => {
            cur.put(key_in_cur, value);
            Ok(())
        }
        Some(_) => Err(Error::generic(format!(
            "Parameters '{}.*' used inconsistently",
            prefix(key, cursor)
        ))),
    }
}

/// `keyval_parse_one()`: parses the key-val at the start of `params` into `qdict` and returns what
/// follows it, or sets `help` for a bare `help` or `?`.
fn keyval_parse_one<'a>(
    qdict: &mut QDict,
    params: &'a str,
    implied_key: Option<&str>,
    help: &mut bool,
) -> Result<&'a str> {
    let mut key = params;
    let mut val_end = None;
    let mut len = params.find(['=', ',']).unwrap_or(params.len());
    if len != 0 && params.as_bytes().get(len) != Some(&b'=') {
        if starts_with_help_option(key) == len {
            *help = true;
            let s = &params[len..];
            return Ok(s.strip_prefix(',').unwrap_or(s));
        }
        if let Some(implied) = implied_key {
            key = implied;
            val_end = Some(len);
            len = implied.len();
        }
    }
    let implied = val_end.is_some();
    let key_end = len;
    let kb = key.as_bytes();
    let at = |i: usize| kb.get(i).copied().unwrap_or(0);

    // Loop over the key fragments. `s` is where the current one starts, `key_in_cur` is the one
    // before it, which names the member of `cur` the current fragment goes into.
    let mut cur = qdict;
    let mut s = 0;
    let mut key_in_cur = String::new();
    loop {
        let index = if s != 0 { key_to_index(&kb[s..], false) } else { None };
        let len = match index {
            Some((_, digits)) => digits,
            None => parse_qapi_name(&kb[s..]).unwrap_or(0),
        };
        assert!(s + len <= key_end);
        if len == 0 || (s + len < key_end && kb[s + len] != b'.') {
            assert!(!implied);
            return Err(Error::generic(format!("Invalid parameter '{}'", prefix(key, key_end))));
        }
        if len >= 128 {
            assert!(!implied);
            let what = if s != 0 || s + len != key_end { " fragment" } else { "" };
            return Err(Error::generic(format!(
                "Parameter{what} '{}' is too long",
                &key[s..s + len]
            )));
        }

        if s != 0 {
            cur = put_dict(cur, &key_in_cur, key, s - 1)?;
        }

        key_in_cur = key[s..s + len].to_string();
        s += len;

        if at(s) != b'.' {
            break;
        }
        s += 1;
    }

    let (val, rest) = if let Some(val_end) = val_end {
        assert_eq!(s, key.len());
        let rest = &params[val_end..];
        (params[..val_end].to_string(), rest.strip_prefix(',').unwrap_or(rest))
    } else {
        if at(s) != b'=' {
            return Err(Error::generic(format!(
                "Expected '=' after parameter '{}'",
                prefix(key, s)
            )));
        }
        let mut val = String::new();
        let mut rest = &params[s + 1..];
        loop {
            let end = rest.find(',').unwrap_or(rest.len());
            val.push_str(&rest[..end]);
            rest = &rest[end..];
            if rest.starts_with(",,") {
                val.push(',');
                rest = &rest[2..];
            } else {
                rest = rest.strip_prefix(',').unwrap_or(rest);
                break;
            }
        }
        (val, rest)
    };

    put_str(cur, &key_in_cur, val, key, key_end)?;
    Ok(rest)
}

/// `keyval_listify()`: turns every dict below `cur` whose keys are all list indexes into a list.
/// `path` is the dotted path to `cur` with a trailing dot, for error messages. Returns the
/// elements `cur` itself should be replaced with, if it is a list.
fn keyval_listify(cur: &mut QDict, path: &str) -> Result<Option<Vec<QValue>>> {
    let mut has_index = false;
    let mut has_member = false;
    let keys: Vec<String> = cur.keys().map(str::to_string).collect();
    for key in &keys {
        if key_to_index(key.as_bytes(), true).is_some() {
            has_index = true;
        } else {
            has_member = true;
        }
        let list = match cur.get_mut(key) {
            Some(QValue::Dict(d)) => keyval_listify(d, &format!("{path}{key}."))?,
            _ => None,
        };
        if let Some(list) = list {
            cur.put(key.as_str(), list);
        }
    }

    if has_index && has_member {
        return Err(Error::generic(format!("Parameters '{path}*' used inconsistently")));
    }
    if !has_index {
        return Ok(None);
    }

    // One slot more than there are entries. An index that does not fit is dropped, which leaves a
    // hole the loop below reports.
    let nelt = cur.len() + 1;
    let mut elt: Vec<Option<&QValue>> = vec![None; nelt];
    let mut max_index: i64 = -1;
    for (key, value) in cur.iter() {
        let (index, _) = key_to_index(key.as_bytes(), true).expect("all keys are indexes");
        let index = index as usize;
        max_index = max_index.max(index as i64);
        if index >= nelt - 1 {
            continue;
        }
        // Keys are distinct but indexes need not be: "1" and "01" are the same element.
        elt[index] = Some(value);
    }

    let n = nelt.min((max_index + 1) as usize);
    let mut list = Vec::with_capacity(n);
    for (i, e) in elt.iter().take(n).enumerate() {
        match e {
            Some(v) => list.push((*v).clone()),
            None => return Err(Error::generic(format!("Parameter '{path}{i}' missing"))),
        }
    }
    Ok(Some(list))
}

/// `keyval_parse_into()`: parses `params` into an existing dict.
///
/// A bare `help` or `?` is not stored. It sets `*p_help` when `p_help` is given and is an error
/// otherwise. On failure the keys parsed so far stay in `qdict`, as in QEMU.
pub fn keyval_parse_into(
    qdict: &mut QDict,
    params: &str,
    mut implied_key: Option<&str>,
    p_help: Option<&mut bool>,
) -> Result<()> {
    let mut help = false;
    let mut s = params;
    while !s.is_empty() {
        s = keyval_parse_one(qdict, s, implied_key, &mut help)?;
        implied_key = None;
    }

    match p_help {
        Some(p) => *p = help,
        None if help => return Err(Error::generic("Help is not available for this option")),
        None => {}
    }

    let listified = keyval_listify(qdict, "")?;
    assert!(listified.is_none(), "the root is never a list");
    Ok(())
}

/// `keyval_parse()`: parses `params` in QEMU's KEY=VALUE,... syntax into a new dict.
pub fn keyval_parse(
    params: &str,
    implied_key: Option<&str>,
    p_help: Option<&mut bool>,
) -> Result<QDict> {
    let mut qdict = QDict::new();
    keyval_parse_into(&mut qdict, params, implied_key, p_help)?;
    Ok(qdict)
}

/// `keyval_do_merge()`. `path` is the dotted path to `dest` for error messages.
fn keyval_do_merge(dest: &mut QDict, merged: &QDict, path: &mut String) -> Result<()> {
    for (key, value) in merged.iter() {
        match (dest.get_mut(key), value) {
            (None, _) => {}
            (Some(old), _) if old.qtype() != value.qtype() => {
                return Err(Error::generic(format!("Parameter '{path}{key}' used inconsistently")));
            }
            (Some(QValue::Dict(old)), QValue::Dict(new)) => {
                let save_len = path.len();
                path.push_str(key);
                path.push('.');
                let r = keyval_do_merge(old, new, path);
                path.truncate(save_len);
                r?;
                continue;
            }
            (Some(QValue::List(old)), QValue::List(new)) => {
                old.extend(new.iter().cloned());
                continue;
            }
            (Some(old), _) => assert!(matches!(old, QValue::Str(_))),
        }
        dest.put(key, value.clone());
    }
    Ok(())
}

/// `keyval_merge()`: merges `merged`, a dict from [`keyval_parse`], into `dest`.
///
/// Lists are concatenated, dicts are merged recursively and for strings `merged` wins. On error
/// `dest` may already have been changed.
pub fn keyval_merge(dest: &mut QDict, merged: &QDict) -> Result<()> {
    keyval_do_merge(dest, merged, &mut String::new())
}
