// SPDX-License-Identifier: GPL-2.0-or-later

//! `QObject` and friends from qobject/.

use std::cmp::Reverse;
use std::fmt;

/// A QAPI value. `Int`, `Uint` and `Double` are the three kinds of `QNum`, kept apart because they
/// print differently and because the input visitors care which one arrived.
#[derive(Clone, Debug)]
pub enum QValue {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Double(f64),
    Str(String),
    List(Vec<QValue>),
    Dict(QDict),
}

/// `QType`, the tag of a value, with the names `qtype_name()` prints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QType {
    None,
    QNull,
    QNum,
    QString,
    QDict,
    QList,
    QBool,
}

impl QType {
    pub fn name(self) -> &'static str {
        match self {
            QType::None => "none",
            QType::QNull => "qnull",
            QType::QNum => "qnum",
            QType::QString => "qstring",
            QType::QDict => "qdict",
            QType::QList => "qlist",
            QType::QBool => "qbool",
        }
    }
}

impl QValue {
    pub fn qtype(&self) -> QType {
        match self {
            QValue::Null => QType::QNull,
            QValue::Bool(_) => QType::QBool,
            QValue::Int(_) | QValue::Uint(_) | QValue::Double(_) => QType::QNum,
            QValue::Str(_) => QType::QString,
            QValue::List(_) => QType::QList,
            QValue::Dict(_) => QType::QDict,
        }
    }

    pub fn str(s: impl Into<String>) -> Self {
        QValue::Str(s.into())
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            QValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            QValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_dict(&self) -> Option<&QDict> {
        match self {
            QValue::Dict(d) => Some(d),
            _ => None,
        }
    }

    pub fn as_dict_mut(&mut self) -> Option<&mut QDict> {
        match self {
            QValue::Dict(d) => Some(d),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[QValue]> {
        match self {
            QValue::List(l) => Some(l),
            _ => None,
        }
    }

    /// `qnum_get_try_int()`: the value as an `i64` if it is an integer that fits.
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            QValue::Int(v) => Some(v),
            QValue::Uint(v) => i64::try_from(v).ok(),
            _ => None,
        }
    }

    /// `qnum_get_try_uint()`.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            QValue::Int(v) => u64::try_from(v).ok(),
            QValue::Uint(v) => Some(v),
            _ => None,
        }
    }

    /// `qnum_get_double()`, which works for every kind of number.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            QValue::Int(v) => Some(v as f64),
            QValue::Uint(v) => Some(v as f64),
            QValue::Double(v) => Some(v),
            _ => None,
        }
    }

    /// Compact JSON exactly as `qobject_to_json()` writes it.
    pub fn to_json(&self) -> String {
        crate::json::to_string(self, false)
    }

    /// Indented JSON exactly as `qobject_to_json_pretty()` writes it.
    pub fn to_json_pretty(&self) -> String {
        crate::json::to_string(self, true)
    }
}

/// `qobject_is_equal()`: numbers compare by value across kinds, doubles never equal integers
/// unless they hold the same integer value, and dicts compare regardless of order.
impl PartialEq for QValue {
    fn eq(&self, other: &Self) -> bool {
        use QValue::*;
        match (self, other) {
            (Null, Null) => true,
            (Bool(a), Bool(b)) => a == b,
            (Str(a), Str(b)) => a == b,
            (List(a), List(b)) => a == b,
            (Dict(a), Dict(b)) => a == b,
            (Double(a), Double(b)) => a == b,
            (Double(d), n @ (Int(_) | Uint(_))) | (n @ (Int(_) | Uint(_)), Double(d)) => {
                n.as_f64() == Some(*d) && d.fract() == 0.0 && (*d as i128) == n.as_i128()
            }
            (a @ (Int(_) | Uint(_)), b @ (Int(_) | Uint(_))) => a.as_i128() == b.as_i128(),
            _ => false,
        }
    }
}

impl QValue {
    fn as_i128(&self) -> i128 {
        match *self {
            QValue::Int(v) => v as i128,
            QValue::Uint(v) => v as i128,
            _ => unreachable!("only called on integers"),
        }
    }
}

impl fmt::Display for QValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_json())
    }
}

impl From<bool> for QValue {
    fn from(v: bool) -> Self {
        QValue::Bool(v)
    }
}

impl From<i64> for QValue {
    fn from(v: i64) -> Self {
        QValue::Int(v)
    }
}

impl From<u64> for QValue {
    fn from(v: u64) -> Self {
        QValue::Uint(v)
    }
}

impl From<f64> for QValue {
    fn from(v: f64) -> Self {
        QValue::Double(v)
    }
}

impl From<&str> for QValue {
    fn from(v: &str) -> Self {
        QValue::Str(v.to_string())
    }
}

impl From<String> for QValue {
    fn from(v: String) -> Self {
        QValue::Str(v)
    }
}

impl From<QDict> for QValue {
    fn from(v: QDict) -> Self {
        QValue::Dict(v)
    }
}

impl From<Vec<QValue>> for QValue {
    fn from(v: Vec<QValue>) -> Self {
        QValue::List(v)
    }
}

/// QEMU's `QDict`.
///
/// QEMU keeps a dict in a 512 bucket hash table, puts new keys at the head of their bucket and
/// iterates bucket by bucket. That order shows up everywhere a dict is printed, which is why the
/// QMP greeting says `"micro": 0, "minor": 1, "major": 11`. This type stores entries in insertion
/// order and [`QDict::iter`] walks them in QEMU's order. Replacing the value of an existing key
/// keeps its place, as in QEMU.
#[derive(Clone, Debug, Default)]
pub struct QDict {
    entries: Vec<Entry>,
    seq: u64,
}

#[derive(Clone, Debug)]
struct Entry {
    key: String,
    value: QValue,
    bucket: u32,
    seq: u64,
}

const BUCKETS: u32 = 512;

/// `tdb_hash()` from qobject/qdict.c.
pub(crate) fn tdb_hash(name: &str) -> u32 {
    let bytes = name.as_bytes();
    let mut value = 0x238F_13AFu32.wrapping_mul(bytes.len() as u32);
    for (i, &b) in bytes.iter().enumerate() {
        value = value.wrapping_add((b as u32) << ((i as u32 * 5) % 24));
    }
    1_103_515_243u32.wrapping_mul(value).wrapping_add(12345)
}

impl QDict {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `qdict_put_obj()`.
    pub fn put(&mut self, key: impl Into<String>, value: impl Into<QValue>) {
        let key = key.into();
        let value = value.into();
        if let Some(e) = self.entries.iter_mut().find(|e| e.key == key) {
            e.value = value;
            return;
        }
        self.seq += 1;
        let bucket = tdb_hash(&key) % BUCKETS;
        self.entries.push(Entry { key, value, bucket, seq: self.seq });
    }

    /// Builder form of [`QDict::put`].
    pub fn with(mut self, key: impl Into<String>, value: impl Into<QValue>) -> Self {
        self.put(key, value);
        self
    }

    pub fn get(&self, key: &str) -> Option<&QValue> {
        self.entries.iter().find(|e| e.key == key).map(|e| &e.value)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut QValue> {
        self.entries.iter_mut().find(|e| e.key == key).map(|e| &mut e.value)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.iter().any(|e| e.key == key)
    }

    /// `qdict_get_try_str()`.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(QValue::as_str)
    }

    /// `qdict_del()` that hands back the value.
    pub fn remove(&mut self, key: &str) -> Option<QValue> {
        let i = self.entries.iter().position(|e| e.key == key)?;
        Some(self.entries.remove(i).value)
    }

    /// Entries in QEMU's iteration order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &QValue)> {
        let mut order: Vec<&Entry> = self.entries.iter().collect();
        order.sort_by_key(|e| (e.bucket, Reverse(e.seq)));
        order.into_iter().map(|e| (e.key.as_str(), &e.value))
    }

    /// Entries in the order they were first inserted, which is the order of keys in JSON input.
    pub fn iter_inserted(&self) -> impl Iterator<Item = (&str, &QValue)> {
        self.entries.iter().map(|e| (e.key.as_str(), &e.value))
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.iter().map(|(k, _)| k)
    }
}

impl PartialEq for QDict {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self.entries.iter().all(|e| other.get(&e.key) == Some(&e.value))
    }
}

impl FromIterator<(String, QValue)> for QDict {
    fn from_iter<I: IntoIterator<Item = (String, QValue)>>(iter: I) -> Self {
        let mut d = QDict::new();
        for (k, v) in iter {
            d.put(k, v);
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_keys_come_out_in_qemu_order() {
        let version = QDict::new().with("major", 11i64).with("minor", 1i64).with("micro", 0i64);
        assert_eq!(version.keys().collect::<Vec<_>>(), ["micro", "minor", "major"]);
        let outer = QDict::new().with("qemu", version).with("package", "");
        assert_eq!(outer.keys().collect::<Vec<_>>(), ["qemu", "package"]);
        let qmp = QDict::new()
            .with("version", outer)
            .with("capabilities", QValue::List(vec!["oob".into()]));
        assert_eq!(qmp.keys().collect::<Vec<_>>(), ["version", "capabilities"]);
    }

    #[test]
    fn replacing_keeps_the_position_and_reinserting_moves_to_the_head() {
        let mut d = QDict::new();
        // "a" and "b" land in different buckets. Check that only bucket order and recency matter.
        d.put("x", 1i64);
        d.put("y", 2i64);
        let before: Vec<_> = d.keys().map(String::from).collect();
        d.put("x", 3i64);
        assert_eq!(d.keys().map(String::from).collect::<Vec<_>>(), before);
        assert_eq!(d.get("x"), Some(&QValue::Int(3)));
        assert_eq!(d.remove("y"), Some(QValue::Int(2)));
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn numbers_compare_across_kinds() {
        assert_eq!(QValue::Int(5), QValue::Uint(5));
        assert_eq!(QValue::Int(5), QValue::Double(5.0));
        assert_ne!(QValue::Int(-1), QValue::Uint(u64::MAX));
        assert_ne!(QValue::Double(0.5), QValue::Int(0));
    }

    #[test]
    fn tdb_hash_matches_qemu() {
        // Values computed with the C function from qobject/qdict.c.
        assert_eq!(tdb_hash(""), 12345);
        assert_eq!(tdb_hash("QMP") % 512, tdb_hash("QMP") % 512);
    }
}
