// SPDX-License-Identifier: GPL-2.0-or-later

//! QemuOpts, the legacy option parser from util/qemu-option.c, and the option group table and
//! `-set` from util/qemu-config.c and system/vl.c.
//!
//! This is what `-device`, `-drive`, `-chardev`, `-machine` and many other options are parsed
//! with. It is older than QAPI and keeps its quirks, because command lines depend on them: `,,` is
//! a literal comma, the first value may leave out its key, a bare `foo` means `foo=on` and `nofoo`
//! means `foo=off`, `id` lives outside the options, and a key given twice is kept twice with the
//! last one winning on lookup.
//!
//! A [`QemuOptsList`] is QEMU's `QemuOptsList`: a name, the description table and the option sets
//! parsed so far. Each [`QemuOpts`] is one set, for example one `-device` argument. The C code
//! hands out pointers to option sets. Here the list owns them, the methods that make one return a
//! reference to it, and [`QemuOpts::handle`] names one for [`QemuOptsList::del`].

use std::sync::Arc;

use ruvm_base::{Error, Result, report};

use crate::cutils::{self, Errno};
use crate::json;
use crate::qvalue::{QDict, QValue};
use crate::visit::{parse_option_size, qapi_bool_parse};

/// `enum QemuOptType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QemuOptType {
    /// `QEMU_OPT_STRING`: the value is used as it is.
    String,
    /// `QEMU_OPT_BOOL`: on or off.
    Bool,
    /// `QEMU_OPT_NUMBER`: an unsigned number.
    Number,
    /// `QEMU_OPT_SIZE`: a size with an optional k, M, G, T, P or E suffix.
    Size,
}

impl QemuOptType {
    /// `opt_type_to_string()`, as `-device foo,help` prints it.
    pub fn as_str(self) -> &'static str {
        match self {
            QemuOptType::String => "str",
            QemuOptType::Bool => "bool (on/off)",
            QemuOptType::Number => "num",
            QemuOptType::Size => "size",
        }
    }
}

/// `QemuOptDesc`: one option a list accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QemuOptDesc {
    pub name: &'static str,
    pub ty: QemuOptType,
    pub help: Option<&'static str>,
    /// The value `qemu_opt_get()` and friends return when the option was not given.
    pub def_value_str: Option<&'static str>,
}

impl QemuOptDesc {
    pub const fn new(name: &'static str, ty: QemuOptType) -> Self {
        QemuOptDesc { name, ty, help: None, def_value_str: None }
    }

    pub const fn help(mut self, help: &'static str) -> Self {
        self.help = Some(help);
        self
    }

    pub const fn default_value(mut self, def: &'static str) -> Self {
        self.def_value_str = Some(def);
        self
    }
}

/// `find_desc_by_name()`.
fn find_desc_by_name(desc: &[QemuOptDesc], name: &str) -> Option<QemuOptDesc> {
    desc.iter().find(|d| d.name == name).copied()
}

/// `is_help_option()` from include/qemu/help_option.h.
pub fn is_help_option(s: &str) -> bool {
    s == "?" || s == "help"
}

/// `get_opt_value()`: the value at the start of `p` with each `,,` turned into `,`, and the rest
/// of `p` from the comma that ended the value, or an empty string.
pub fn get_opt_value(p: &str) -> (String, &str) {
    let mut value = String::new();
    let mut p = p;
    loop {
        let offset = p.find(',').unwrap_or(p.len());
        value.push_str(&p[..offset]);
        if p[offset..].starts_with(",,") {
            value.push(',');
            p = &p[offset + 2..];
        } else {
            return (value, &p[offset..]);
        }
    }
}

/// `parse_option_number()`.
fn parse_option_number(name: &str, value: &str) -> Result<u64> {
    match cutils::strtou64(value, 0, true) {
        Ok((v, _)) => Ok(v),
        Err((Errno::Range, _)) => {
            Err(Error::generic(format!("Value '{value}' is too large for parameter '{name}'")))
        }
        Err(_) => Err(Error::generic(format!("Parameter '{name}' expects a number"))),
    }
}

/// The deprecation warning for a short-form boolean such as `foo` or `nofoo`, which
/// `qemu_opts_parse_noisily()` prints with `warn_report()` and `error_printf()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlagWarning {
    /// The option name without the `no` prefix.
    pub name: String,
    /// Whether it was written with the `no` prefix.
    pub negated: bool,
}

impl FlagWarning {
    /// The line that goes through `warn_report()`.
    pub fn message(&self) -> String {
        let prefix = if self.negated { "no" } else { "" };
        format!("short-form boolean option '{prefix}{}' deprecated", self.name)
    }

    /// The line after it, with its newline.
    pub fn hint(&self) -> String {
        if self.name == "delay" {
            format!("Please use nodelay={} instead\n", if self.negated { "on" } else { "off" })
        } else {
            format!(
                "Please use {}={} instead\n",
                self.name,
                if self.negated { "off" } else { "on" }
            )
        }
    }

    /// Prints the warning the way QEMU does.
    pub fn report(&self) {
        report::warn_report(&self.message());
        eprint!("{}", self.hint());
    }
}

/// Why [`QemuOptsList::parse_detailed`] gave up.
#[derive(Debug)]
pub enum ParseFailure {
    /// The string asked for help with a bare `help` or `?`. This is not an error, the caller
    /// prints the help text.
    HelpWanted,
    Error(Error),
}

impl From<Error> for ParseFailure {
    fn from(e: Error) -> Self {
        ParseFailure::Error(e)
    }
}

/// One `name=value` from `get_opt_name_value()`.
struct NameValue<'a> {
    name: String,
    value: String,
    rest: &'a str,
    is_help: bool,
    flag: Option<FlagWarning>,
}

/// `get_opt_name_value()`: splits off the first `name=value` of `params`. With `firstname` a value
/// without a name is given that name, otherwise it is a flag.
fn get_opt_name_value<'a>(params: &'a str, firstname: Option<&str>) -> NameValue<'a> {
    let len = params.find(['=', ',']).unwrap_or(params.len());
    let mut is_help = false;
    let mut flag = None;
    let (name, value, p) = if params.as_bytes().get(len) != Some(&b'=') {
        if let Some(firstname) = firstname {
            // An implicitly named first option.
            let (value, p) = get_opt_value(params);
            (firstname.to_string(), value, p)
        } else {
            // An option without a value, which must be a flag.
            let raw = &params[..len];
            let (name, value, negated) = match raw.strip_prefix("no") {
                Some(name) => (name, "off", true),
                None => {
                    is_help = is_help_option(raw);
                    (raw, "on", false)
                }
            };
            if !is_help {
                flag = Some(FlagWarning { name: name.to_string(), negated });
            }
            (name.to_string(), value.to_string(), &params[len..])
        }
    } else {
        let (value, p) = get_opt_value(&params[len + 1..]);
        (params[..len].to_string(), value, p)
    };
    assert!(p.is_empty() || p.starts_with(','));
    NameValue { name, value, rest: p.strip_prefix(',').unwrap_or(p), is_help, flag }
}

/// `opts_parse_id()`: the value of the first `id=`, found without an implied first key.
fn opts_parse_id(params: &str) -> Option<String> {
    let mut p = params;
    while !p.is_empty() {
        let nv = get_opt_name_value(p, None);
        if nv.name == "id" {
            return Some(nv.value);
        }
        p = nv.rest;
    }
    None
}

/// `has_help_option()`: whether `params` contains a bare `help` or `?`.
pub fn has_help_option(params: &str) -> bool {
    let mut p = params;
    while !p.is_empty() {
        let nv = get_opt_name_value(p, None);
        if nv.is_help {
            return true;
        }
        p = nv.rest;
    }
    false
}

/// `QemuOpt`: one `name=value` in an option set.
#[derive(Clone, Debug)]
pub struct QemuOpt {
    name: String,
    str: String,
    desc: Option<QemuOptDesc>,
    boolean: bool,
    uint: u64,
}

impl QemuOpt {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The value as it was written.
    pub fn value(&self) -> &str {
        &self.str
    }

    /// The description the value was checked against, if the list has one for it.
    pub fn desc(&self) -> Option<&QemuOptDesc> {
        self.desc.as_ref()
    }

    /// `qemu_opt_parse()`: checks and converts the value according to the description.
    fn parse(&mut self) -> Result<()> {
        let Some(desc) = self.desc else { return Ok(()) };
        match desc.ty {
            QemuOptType::String => {}
            QemuOptType::Bool => self.boolean = qapi_bool_parse(&self.name, &self.str)?,
            QemuOptType::Number => self.uint = parse_option_number(&self.name, &self.str)?,
            QemuOptType::Size => self.uint = parse_option_size(&self.name, &self.str)?,
        }
        Ok(())
    }
}

/// Names one [`QemuOpts`] in its list, for [`QemuOptsList::get`] and [`QemuOptsList::del`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OptsHandle(u64);

/// `QemuOpts`: one set of options, such as the ones from a single `-device` argument.
#[derive(Clone, Debug)]
pub struct QemuOpts {
    id: Option<String>,
    list_desc: Arc<[QemuOptDesc]>,
    head: Vec<QemuOpt>,
    handle: OptsHandle,
}

impl QemuOpts {
    pub fn handle(&self) -> OptsHandle {
        self.handle
    }

    /// `qemu_opts_id()`.
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// `qemu_opts_set_id()`.
    pub fn set_id(&mut self, id: Option<String>) {
        self.id = id;
    }

    /// Every option in the order it was given, repeated names included.
    pub fn opts(&self) -> &[QemuOpt] {
        &self.head
    }

    /// `qemu_opt_foreach()` as an iterator of names and values.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.head.iter().map(|o| (o.name.as_str(), o.str.as_str()))
    }

    /// `qemu_opt_iter_init()` and `qemu_opt_iter_next()`: the values of every option called
    /// `name`, or of every option, in order.
    pub fn iter_values<'a>(&'a self, name: Option<&'a str>) -> impl Iterator<Item = &'a str> {
        self.head.iter().filter(move |o| name.is_none_or(|n| n == o.name)).map(|o| o.str.as_str())
    }

    fn accepts_any(&self) -> bool {
        self.list_desc.is_empty()
    }

    fn find_default_by_name(&self, name: &str) -> Option<&'static str> {
        find_desc_by_name(&self.list_desc, name).and_then(|d| d.def_value_str)
    }

    fn find_index(&self, name: &str) -> Option<usize> {
        self.head.iter().rposition(|o| o.name == name)
    }

    /// `qemu_opt_find()`: the last option called `name`.
    pub fn find(&self, name: &str) -> Option<&QemuOpt> {
        self.find_index(name).map(|i| &self.head[i])
    }

    /// `qemu_opt_del_all()`.
    fn del_all(&mut self, name: &str) {
        self.head.retain(|o| o.name != name);
    }

    /// `qemu_opt_get()`: the value of the last option called `name`, or its default.
    pub fn get(&self, name: &str) -> Option<&str> {
        match self.find(name) {
            Some(opt) => Some(&opt.str),
            None => self.find_default_by_name(name),
        }
    }

    /// `qemu_opt_get_del()`: like [`QemuOpts::get`], and removes every option called `name`.
    pub fn get_del(&mut self, name: &str) -> Option<String> {
        let Some(i) = self.find_index(name) else {
            return self.find_default_by_name(name).map(str::to_string);
        };
        let s = std::mem::take(&mut self.head[i].str);
        self.del_all(name);
        Some(s)
    }

    /// `qemu_opt_has_any()`.
    pub fn has_any(&self, names: &[&str]) -> bool {
        names.iter().any(|n| self.get(n).is_some())
    }

    /// `qemu_opt_has_help_opt()`: whether one of the options is called `help` or `?`.
    pub fn has_help_opt(&self) -> bool {
        self.head.iter().rev().any(|o| is_help_option(&o.name))
    }

    /// `qemu_opt_get_bool()`. The option must be described as a boolean.
    pub fn get_bool(&self, name: &str, defval: bool) -> bool {
        let Some(opt) = self.find(name) else {
            return match self.find_default_by_name(name) {
                Some(def) => qapi_bool_parse(name, def).expect("the default is a boolean"),
                None => defval,
            };
        };
        assert!(opt.desc.is_some_and(|d| d.ty == QemuOptType::Bool));
        opt.boolean
    }

    /// `qemu_opt_get_bool_del()`: [`QemuOpts::get_bool`], then removes every option called
    /// `name`.
    pub fn get_bool_del(&mut self, name: &str, defval: bool) -> bool {
        let ret = self.get_bool(name, defval);
        self.del_all(name);
        ret
    }

    fn get_uint(&self, name: &str, defval: u64, ty: QemuOptType) -> u64 {
        let Some(opt) = self.find(name) else {
            return match self.find_default_by_name(name) {
                Some(def) if ty == QemuOptType::Number => {
                    parse_option_number(name, def).expect("the default is a number")
                }
                Some(def) => parse_option_size(name, def).expect("the default is a size"),
                None => defval,
            };
        };
        assert!(opt.desc.is_some_and(|d| d.ty == ty));
        opt.uint
    }

    /// `qemu_opt_get_number()`. The option must be described as a number.
    pub fn get_number(&self, name: &str, defval: u64) -> u64 {
        self.get_uint(name, defval, QemuOptType::Number)
    }

    /// `qemu_opt_get_number_del()`.
    pub fn get_number_del(&mut self, name: &str, defval: u64) -> u64 {
        let ret = self.get_number(name, defval);
        self.del_all(name);
        ret
    }

    /// `qemu_opt_get_size()`. The option must be described as a size.
    pub fn get_size(&self, name: &str, defval: u64) -> u64 {
        self.get_uint(name, defval, QemuOptType::Size)
    }

    /// `qemu_opt_get_size_del()`.
    pub fn get_size_del(&mut self, name: &str, defval: u64) -> u64 {
        let ret = self.get_size(name, defval);
        self.del_all(name);
        ret
    }

    /// `qemu_opt_unset()`: removes the last option called `name`. Only lists that accept any
    /// option allow this. Returns whether there was one.
    pub fn unset(&mut self, name: &str) -> bool {
        assert!(self.accepts_any());
        match self.find_index(name) {
            Some(i) => {
                self.head.remove(i);
                true
            }
            None => false,
        }
    }

    /// `opt_validate()` on a new option, which is kept only if it passes.
    fn add_validated(&mut self, name: &str, value: String) -> Result<()> {
        let desc = find_desc_by_name(&self.list_desc, name);
        if desc.is_none() && !self.accepts_any() {
            return Err(Error::generic(format!("Invalid parameter '{name}'")));
        }
        let mut opt = QemuOpt { name: name.to_string(), str: value, desc, boolean: false, uint: 0 };
        opt.parse()?;
        self.head.push(opt);
        Ok(())
    }

    fn check_known(&self, name: &str) -> Result<Option<QemuOptDesc>> {
        let desc = find_desc_by_name(&self.list_desc, name);
        if desc.is_none() && !self.accepts_any() {
            return Err(Error::generic(format!("Invalid parameter '{name}'")));
        }
        Ok(desc)
    }

    /// `qemu_opt_set()`: adds `name=value`, which is also what `-set` does.
    pub fn set(&mut self, name: &str, value: &str) -> Result<()> {
        self.add_validated(name, value.to_string())
    }

    /// `qemu_opt_set_bool()`.
    pub fn set_bool(&mut self, name: &str, val: bool) -> Result<()> {
        let desc = self.check_known(name)?;
        let str = if val { "on" } else { "off" }.to_string();
        self.head.push(QemuOpt { name: name.to_string(), str, desc, boolean: val, uint: 0 });
        Ok(())
    }

    /// `qemu_opt_set_number()`.
    pub fn set_number(&mut self, name: &str, val: i64) -> Result<()> {
        let desc = self.check_known(name)?;
        let str = val.to_string();
        self.head.push(QemuOpt {
            name: name.to_string(),
            str,
            desc,
            boolean: false,
            uint: val as u64,
        });
        Ok(())
    }

    /// `opts_do_parse()`.
    fn do_parse_inner(
        &mut self,
        params: &str,
        mut firstname: Option<&str>,
        mut warnings: Option<&mut Vec<FlagWarning>>,
        mut help_wanted: Option<&mut bool>,
    ) -> Result<(), ParseFailure> {
        let mut p = params;
        while !p.is_empty() {
            let nv = get_opt_name_value(p, firstname);
            if let (Some(w), Some(flag)) = (warnings.as_deref_mut(), nv.flag) {
                w.push(flag);
            }
            if let Some(h) = help_wanted.as_deref_mut() {
                if nv.is_help {
                    *h = true;
                }
                if *h {
                    return Err(ParseFailure::HelpWanted);
                }
            }
            p = nv.rest;
            firstname = None;

            if nv.name == "id" {
                continue;
            }
            self.add_validated(&nv.name, nv.value)?;
        }
        Ok(())
    }

    /// `qemu_opts_do_parse()`: parses `params` into this set. With `firstname`, the first value
    /// may leave out its key, which is then `firstname`.
    pub fn do_parse(&mut self, params: &str, firstname: Option<&str>) -> Result<()> {
        match self.do_parse_inner(params, firstname, None, None) {
            Ok(()) => Ok(()),
            Err(ParseFailure::Error(e)) => Err(e),
            Err(ParseFailure::HelpWanted) => unreachable!("help is not looked for"),
        }
    }

    /// `qemu_opts_from_qdict_entry()`.
    fn set_from_qvalue(&mut self, key: &str, obj: &QValue) -> Result<()> {
        if key == "id" {
            return Ok(());
        }
        let value = match obj {
            QValue::Str(s) => s.clone(),
            QValue::Int(v) => v.to_string(),
            QValue::Uint(v) => v.to_string(),
            QValue::Double(v) => json::format_g17(*v),
            QValue::Bool(b) => if *b { "on" } else { "off" }.to_string(),
            _ => return Ok(()),
        };
        self.set(key, &value)
    }

    /// `qemu_opts_absorb_qdict()`: adds every entry of `qdict` this list accepts and removes it
    /// from `qdict`, which keeps the rest.
    pub fn absorb_qdict(&mut self, qdict: &mut QDict) -> Result<()> {
        let keys: Vec<String> = qdict.keys().map(str::to_string).collect();
        for key in keys {
            if self.accepts_any() || find_desc_by_name(&self.list_desc, &key).is_some() {
                let value = qdict.get(&key).expect("key taken from the dict").clone();
                self.set_from_qvalue(&key, &value)?;
                qdict.remove(&key);
            }
        }
        Ok(())
    }

    /// `qemu_opts_to_qdict_filtered()`: puts the id and every option into `qdict` as strings.
    ///
    /// With `filter`, only options it describes are copied. With `del`, copied options are
    /// removed from this set. Of repeated options the last one wins in the dict, and with `del`
    /// all of them go.
    pub fn to_qdict_filtered(
        &mut self,
        qdict: &mut QDict,
        filter: Option<&[QemuOptDesc]>,
        del: bool,
    ) {
        if let Some(id) = &self.id {
            qdict.put("id", id.as_str());
        }
        let wanted = |o: &QemuOpt| filter.is_none_or(|f| f.iter().any(|d| d.name == o.name));
        for opt in self.head.iter().filter(|o| wanted(o)) {
            qdict.put(opt.name.as_str(), opt.str.as_str());
        }
        if del {
            self.head.retain(|o| !wanted(o));
        }
    }

    /// `qemu_opts_to_qdict()`: the id and every option as a dict of strings.
    pub fn to_qdict(&self) -> QDict {
        let mut qdict = QDict::new();
        if let Some(id) = &self.id {
            qdict.put("id", id.as_str());
        }
        for opt in &self.head {
            qdict.put(opt.name.as_str(), opt.str.as_str());
        }
        qdict
    }

    /// `qemu_opts_validate()`: checks the options against `desc`, for a list that accepts any
    /// option and whose user knows more than the list does.
    pub fn validate(&mut self, desc: &[QemuOptDesc]) -> Result<()> {
        assert!(self.accepts_any());
        for opt in &mut self.head {
            opt.desc = find_desc_by_name(desc, &opt.name);
            if opt.desc.is_none() {
                return Err(Error::generic(format!("Invalid parameter '{}'", opt.name)));
            }
            opt.parse()?;
        }
        Ok(())
    }

    /// `qemu_opts_print()` into a string: the id and options joined by `separator`, commas in
    /// values doubled. For a list with descriptions, options come in description order and
    /// defaults are included.
    pub fn to_print_string(&self, separator: &str) -> String {
        fn escaped(out: &mut String, value: &str) {
            out.push_str(&value.replace(',', ",,"));
        }
        let mut out = String::new();
        let mut sep = "";
        if let Some(id) = &self.id {
            out.push_str(&format!("id={id}"));
            sep = separator;
        }
        if self.accepts_any() {
            for opt in &self.head {
                out.push_str(&format!("{sep}{}=", opt.name));
                escaped(&mut out, &opt.str);
                sep = separator;
            }
            return out;
        }
        for desc in self.list_desc.iter() {
            let opt = self.find(desc.name);
            let Some(value) = opt.map(|o| o.str.as_str()).or(desc.def_value_str) else {
                continue;
            };
            match (desc.ty, opt) {
                (QemuOptType::String, _) => {
                    out.push_str(&format!("{sep}{}=", desc.name));
                    escaped(&mut out, value);
                }
                (QemuOptType::Size | QemuOptType::Number, Some(opt)) => {
                    out.push_str(&format!("{sep}{}={}", desc.name, opt.uint as i64));
                }
                _ => out.push_str(&format!("{sep}{}={value}", desc.name)),
            }
            sep = separator;
        }
        out
    }

    /// `qemu_opts_print()`, to standard output.
    pub fn print(&self, separator: &str) {
        print!("{}", self.to_print_string(separator));
    }
}

/// `QemuOptsList`: a group of options such as `drive` or `device`, with the option sets parsed
/// for it so far.
#[derive(Clone, Debug)]
pub struct QemuOptsList {
    name: Option<&'static str>,
    implied_opt_name: Option<&'static str>,
    merge_lists: bool,
    desc: Arc<[QemuOptDesc]>,
    head: Vec<QemuOpts>,
    next_handle: u64,
}

impl QemuOptsList {
    /// A list called `name` that accepts the options in `desc`, or any option if `desc` is empty.
    pub fn new(name: &'static str, desc: &[QemuOptDesc]) -> Self {
        QemuOptsList {
            name: Some(name),
            implied_opt_name: None,
            merge_lists: false,
            desc: desc.into(),
            head: Vec::new(),
            next_handle: 0,
        }
    }

    /// Sets `implied_opt_name`, the key of a first value written without one.
    pub fn with_implied_opt_name(mut self, name: &'static str) -> Self {
        self.implied_opt_name = Some(name);
        self
    }

    /// Sets `merge_lists`: every use of the option adds to one set instead of making a new one.
    pub fn with_merge_lists(mut self) -> Self {
        self.merge_lists = true;
        self
    }

    pub fn name(&self) -> Option<&'static str> {
        self.name
    }

    pub fn implied_opt_name(&self) -> Option<&'static str> {
        self.implied_opt_name
    }

    pub fn merge_lists(&self) -> bool {
        self.merge_lists
    }

    pub fn desc(&self) -> &[QemuOptDesc] {
        &self.desc
    }

    /// `opts_accepts_any()`: an empty description table means any option is accepted.
    pub fn accepts_any(&self) -> bool {
        self.desc.is_empty()
    }

    /// `qemu_opts_append()`: `dst` (or an unnamed empty list) with the descriptions of `list`
    /// added after its own, skipping names it already has.
    pub fn append(dst: Option<QemuOptsList>, list: &QemuOptsList) -> QemuOptsList {
        let mut dst = dst.unwrap_or_else(|| QemuOptsList {
            name: None,
            implied_opt_name: None,
            merge_lists: false,
            desc: Arc::new([]),
            head: Vec::new(),
            next_handle: 0,
        });
        let mut desc = dst.desc.to_vec();
        for d in list.desc.iter() {
            if find_desc_by_name(&desc, d.name).is_none() {
                desc.push(*d);
            }
        }
        dst.desc = desc.into();
        dst
    }

    /// `qemu_opts_foreach()` as an iterator, in the order the sets were made.
    pub fn iter(&self) -> impl Iterator<Item = &QemuOpts> {
        self.head.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut QemuOpts> {
        self.head.iter_mut()
    }

    pub fn is_empty(&self) -> bool {
        self.head.is_empty()
    }

    pub fn get(&self, handle: OptsHandle) -> Option<&QemuOpts> {
        self.head.iter().find(|o| o.handle == handle)
    }

    pub fn get_mut(&mut self, handle: OptsHandle) -> Option<&mut QemuOpts> {
        self.head.iter_mut().find(|o| o.handle == handle)
    }

    fn position(&self, id: Option<&str>) -> Option<usize> {
        self.head.iter().position(|o| o.id.as_deref() == id)
    }

    /// `qemu_opts_find()`: the first set whose id is `id`, where `None` finds the first set
    /// without one.
    pub fn find(&self, id: Option<&str>) -> Option<&QemuOpts> {
        self.position(id).map(|i| &self.head[i])
    }

    pub fn find_mut(&mut self, id: Option<&str>) -> Option<&mut QemuOpts> {
        self.position(id).map(|i| &mut self.head[i])
    }

    /// `qemu_opts_create()`: a new, empty set with the given id.
    ///
    /// For a `merge_lists` list there is only ever one set, which has no id, and it is returned if
    /// it exists. Otherwise the id must be well formed and not used yet. `fail_if_exists` must be
    /// set when there is an id, as in QEMU.
    pub fn create(&mut self, id: Option<&str>, fail_if_exists: bool) -> Result<&mut QemuOpts> {
        if self.merge_lists {
            if id.is_some() {
                return Err(Error::generic("Invalid parameter 'id'"));
            }
            if let Some(i) = self.position(None) {
                return Ok(&mut self.head[i]);
            }
        } else if let Some(id) = id {
            assert!(fail_if_exists);
            if !cutils::id_wellformed(id) {
                return Err(Error::generic("Parameter 'id' expects an identifier").hint(
                    "Identifiers consist of letters, digits, '-', '.', '_', starting with a letter.\n",
                ));
            }
            if self.position(Some(id)).is_some() {
                return Err(Error::generic(format!(
                    "Duplicate ID '{id}' for {}",
                    self.name.unwrap_or("(null)")
                )));
            }
        }
        let handle = OptsHandle(self.next_handle);
        self.next_handle += 1;
        self.head.push(QemuOpts {
            id: id.map(str::to_string),
            list_desc: self.desc.clone(),
            head: Vec::new(),
            handle,
        });
        Ok(self.head.last_mut().expect("just pushed"))
    }

    /// `qemu_opts_del()`.
    pub fn del(&mut self, handle: OptsHandle) {
        self.head.retain(|o| o.handle != handle);
    }

    /// `qemu_opts_reset()`: deletes every set.
    pub fn reset(&mut self) {
        self.head.clear();
    }

    /// `opts_parse()`.
    fn opts_parse(
        &mut self,
        params: &str,
        permit_abbrev: bool,
        warnings: Option<&mut Vec<FlagWarning>>,
        help_wanted: Option<&mut bool>,
    ) -> Result<&mut QemuOpts, ParseFailure> {
        assert!(!permit_abbrev || self.implied_opt_name.is_some());
        let firstname = if permit_abbrev { self.implied_opt_name } else { None };
        let id = opts_parse_id(params);
        let merge = self.merge_lists;
        let handle = self.create(id.as_deref(), !merge)?.handle;
        let opts = self.get_mut(handle).expect("just made");
        match opts.do_parse_inner(params, firstname, warnings, help_wanted) {
            Ok(()) => Ok(self.get_mut(handle).expect("just made")),
            Err(e) => {
                self.del(handle);
                Err(e)
            }
        }
    }

    /// `qemu_opts_parse()`: a set with the options in `params`. With `permit_abbrev`, the first
    /// value may leave out its key, which is then the list's implied option name.
    ///
    /// On error the set is deleted. For a `merge_lists` list that is the one shared set, with
    /// everything merged into it before, which is what QEMU does too.
    pub fn parse(&mut self, params: &str, permit_abbrev: bool) -> Result<&mut QemuOpts> {
        match self.opts_parse(params, permit_abbrev, None, None) {
            Ok(opts) => Ok(opts),
            Err(ParseFailure::Error(e)) => Err(e),
            Err(ParseFailure::HelpWanted) => unreachable!("help is not looked for"),
        }
    }

    /// [`QemuOptsList::parse_noisily`] without the printing: the short-form boolean warnings are
    /// collected into `warnings`, and a help request is a [`ParseFailure::HelpWanted`]. Help is
    /// only looked for when the list has descriptions.
    pub fn parse_detailed(
        &mut self,
        params: &str,
        permit_abbrev: bool,
        warnings: &mut Vec<FlagWarning>,
    ) -> Result<&mut QemuOpts, ParseFailure> {
        let mut help_wanted = false;
        let help = if self.accepts_any() { None } else { Some(&mut help_wanted) };
        self.opts_parse(params, permit_abbrev, Some(warnings), help)
    }

    /// `qemu_opts_parse_noisily()`: [`QemuOptsList::parse`] that warns about short-form booleans,
    /// prints the help text when asked for it and reports errors itself.
    pub fn parse_noisily(&mut self, params: &str, permit_abbrev: bool) -> Option<&mut QemuOpts> {
        let mut warnings = Vec::new();
        let r = self.parse_detailed(params, permit_abbrev, &mut warnings).map(|o| o.handle);
        for w in &warnings {
            w.report();
        }
        match r {
            Ok(handle) => self.get_mut(handle),
            Err(ParseFailure::HelpWanted) => {
                self.print_help(true);
                None
            }
            Err(ParseFailure::Error(e)) => {
                report::report_error(&e);
                None
            }
        }
    }

    /// `qemu_opts_from_qdict()`: a set with the entries of `qdict`. `id` becomes the set's id if
    /// it is a string, and only strings, numbers and booleans are copied.
    pub fn from_qdict(&mut self, qdict: &QDict) -> Result<&mut QemuOpts> {
        let handle = self.create(qdict.get_str("id"), true)?.handle;
        let opts = self.get_mut(handle).expect("just made");
        for (key, value) in qdict.iter() {
            if let Err(e) = opts.set_from_qvalue(key, value) {
                self.del(handle);
                return Err(e);
            }
        }
        Ok(self.get_mut(handle).expect("just made"))
    }

    /// `qemu_opts_print_help()` into a string. With `print_caption` the options are preceded by a
    /// line naming the list.
    pub fn help_text(&self, print_caption: bool) -> String {
        let mut lines: Vec<String> = self
            .desc
            .iter()
            .map(|d| {
                let mut s = format!("  {}=<{}>", d.name, d.ty.as_str());
                if let Some(help) = d.help {
                    if s.len() < 24 {
                        s.push_str(&" ".repeat(24 - s.len()));
                    }
                    s.push_str(&format!(" - {help}"));
                }
                s
            })
            .collect();
        lines.sort();
        let mut out = String::new();
        if print_caption && !lines.is_empty() {
            match self.name {
                Some(name) => out.push_str(&format!("{name} options:\n")),
                None => out.push_str("Options:\n"),
            }
        } else if lines.is_empty() {
            match self.name {
                Some(name) => out.push_str(&format!("There are no options for {name}.\n")),
                None => out.push_str("No options available.\n"),
            }
        }
        for line in lines {
            out.push_str(&line);
            out.push('\n');
        }
        out
    }

    /// `qemu_opts_print_help()`, to standard output.
    pub fn print_help(&self, print_caption: bool) {
        print!("{}", self.help_text(print_caption));
    }
}

/// `vm_config_groups` from util/qemu-config.c: the option lists known by group name.
#[derive(Debug, Default)]
pub struct ConfigGroups {
    lists: Vec<QemuOptsList>,
}

impl ConfigGroups {
    pub fn new() -> Self {
        Self::default()
    }

    /// `qemu_add_opts()`.
    pub fn add(&mut self, list: QemuOptsList) {
        self.lists.push(list);
    }

    /// `qemu_find_opts_err()`.
    pub fn find_opts_err(&mut self, group: &str) -> Result<&mut QemuOptsList> {
        self.lists
            .iter_mut()
            .find(|l| l.name == Some(group))
            .ok_or_else(|| Error::generic(format!("There is no option group '{group}'")))
    }

    /// `qemu_find_opts()`: like [`ConfigGroups::find_opts_err`] but reports the error.
    pub fn find_opts(&mut self, group: &str) -> Option<&mut QemuOptsList> {
        match self.find_opts_err(group) {
            Ok(list) => Some(list),
            Err(e) => {
                report::report_error(&e);
                None
            }
        }
    }

    /// `qemu_find_opts_singleton()`: the set without an id in `group`, made if needed. The group
    /// must exist.
    pub fn find_opts_singleton(&mut self, group: &str) -> &mut QemuOpts {
        let list = self.find_opts(group).expect("the group exists");
        if list.position(None).is_none() {
            list.create(None, false).expect("a set without an id can always be made");
        }
        list.find_mut(None).expect("just made")
    }

    /// `qemu_set_option()` from system/vl.c, which implements `-set group.id.arg=value`.
    ///
    /// QEMU first refuses groups that are not parsed with QemuOpts ("-set is not supported with
    /// %s"). That check needs the table of options in the command line parser, so the caller
    /// makes it.
    pub fn set_option(&mut self, s: &str) -> Result<()> {
        let parsed = parse_set_option(s);
        let Some((group, id, arg, value)) = parsed else {
            return Err(Error::generic(format!("can't parse: \"{s}\"")));
        };
        let list = self.find_opts_err(group)?;
        let Some(opts) = list.find_mut(Some(id)) else {
            return Err(Error::generic(format!("there is no {group} \"{id}\" defined")));
        };
        opts.set(arg, value)
    }
}

/// The `sscanf(str, "%63[^.].%63[^.].%63[^=]%n", ...)` of `qemu_set_option()` followed by the
/// check for `=`: group, id, option name and value.
fn parse_set_option(s: &str) -> Option<(&str, &str, &str, &str)> {
    // One `%63[^x]` conversion: 1 to 63 bytes that are not `stop`.
    fn field(s: &str, stop: u8) -> Option<(&str, &str)> {
        let n = s.bytes().take(63).take_while(|&b| b != stop).count();
        if n == 0 {
            return None;
        }
        // The conversion stops at `stop`, at the end or after 63 bytes. Only the first case can go
        // on to match the literal that follows, and there the split is on a character boundary.
        if s.as_bytes().get(n) != Some(&stop) {
            return None;
        }
        Some((&s[..n], &s[n..]))
    }
    let (group, rest) = field(s, b'.')?;
    let rest = rest.strip_prefix('.')?;
    let (id, rest) = field(rest, b'.')?;
    let rest = rest.strip_prefix('.')?;
    let (arg, rest) = field(rest, b'=')?;
    let value = rest.strip_prefix('=')?;
    Some((group, id, arg, value))
}
