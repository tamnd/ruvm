// SPDX-License-Identifier: GPL-2.0-or-later

//! `JSONWriter`, qobject/json-writer.c, in its compact form.
//!
//! The migration code builds the vmdesc that ends a stream with it: one object per section, with
//! the name, type and size of every field that went out.

use std::fmt::Write as _;

/// A streaming JSON writer. Every value either has a name, inside an object, or not, inside an
/// array or at the top.
#[derive(Debug, Default, Clone)]
pub struct JsonWriter {
    out: String,
    // One entry per open container: whether something has been written into it yet.
    nonempty: Vec<bool>,
}

impl JsonWriter {
    /// `json_writer_new(false)`.
    pub fn new() -> Self {
        JsonWriter::default()
    }

    /// The text written so far.
    pub fn as_str(&self) -> &str {
        &self.out
    }

    /// Takes the text out of the writer.
    pub fn into_string(self) -> String {
        self.out
    }

    fn quoted(&mut self, s: &str) {
        self.out.push('"');
        for c in s.chars() {
            match c {
                '"' => self.out.push_str("\\\""),
                '\\' => self.out.push_str("\\\\"),
                '\n' => self.out.push_str("\\n"),
                '\r' => self.out.push_str("\\r"),
                '\t' => self.out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    let _ = write!(self.out, "\\u{:04X}", c as u32);
                }
                c => self.out.push(c),
            }
        }
        self.out.push('"');
    }

    fn member(&mut self, name: Option<&str>) {
        if let Some(open) = self.nonempty.last_mut() {
            if *open {
                self.out.push_str(", ");
            }
            *open = true;
        }
        if let Some(name) = name {
            self.quoted(name);
            self.out.push_str(": ");
        }
    }

    /// `json_writer_start_object()`.
    pub fn start_object(&mut self, name: Option<&str>) {
        self.member(name);
        self.out.push('{');
        self.nonempty.push(false);
    }

    /// `json_writer_end_object()`.
    pub fn end_object(&mut self) {
        self.nonempty.pop();
        self.out.push('}');
    }

    /// `json_writer_start_array()`.
    pub fn start_array(&mut self, name: Option<&str>) {
        self.member(name);
        self.out.push('[');
        self.nonempty.push(false);
    }

    /// `json_writer_end_array()`.
    pub fn end_array(&mut self) {
        self.nonempty.pop();
        self.out.push(']');
    }

    /// `json_writer_str()`.
    pub fn str(&mut self, name: Option<&str>, value: &str) {
        self.member(name);
        self.quoted(value);
    }

    /// `json_writer_int64()`.
    pub fn int64(&mut self, name: Option<&str>, value: i64) {
        self.member(name);
        let _ = write!(self.out, "{value}");
    }

    /// `json_writer_uint64()`.
    pub fn uint64(&mut self, name: Option<&str>, value: u64) {
        self.member(name);
        let _ = write!(self.out, "{value}");
    }

    /// `json_writer_bool()`.
    pub fn bool(&mut self, name: Option<&str>, value: bool) {
        self.member(name);
        self.out.push_str(if value { "true" } else { "false" });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_containers() {
        let mut w = JsonWriter::new();
        w.start_object(None);
        w.str(Some("name"), "a\"b");
        w.start_array(Some("list"));
        w.int64(None, -1);
        w.start_object(None);
        w.bool(Some("x"), true);
        w.end_object();
        w.end_array();
        w.uint64(Some("n"), 7);
        w.end_object();
        assert_eq!(w.as_str(), r#"{"name": "a\"b", "list": [-1, {"x": true}], "n": 7}"#);
    }
}
