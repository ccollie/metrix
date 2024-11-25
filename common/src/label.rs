use std::cmp::Ordering;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use get_size::GetSize;
use integer_encoding::VarInt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[derive(GetSize)]
pub struct Label {
    pub name: String,
    pub value: String,
}

impl Label {
    pub fn new<S: Into<String>>(key: S, value: String) -> Self {
        Self {
            name: key.into(),
            value,
        }
    }

    pub fn marshal(&self, dest: &mut Vec<u8>) {
        write_string(dest, &self.name);
        write_string(dest, &self.value);
    }

    pub fn unmarshal(src: &[u8]) -> (Self, &[u8]) {
        let (name, src) = read_string(src);
        let (value, src) = read_string(src);
        (Self { name: name.to_string(), value: value.to_string() }, src)
    }
}

impl PartialOrd for Label {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let cmp = self.cmp(other);
        Some(cmp)
    }
}

impl Ord for Label {
    fn cmp(&self, other: &Self) -> Ordering {
        let cmp = self.name.cmp(&other.name);
        if cmp!= Ordering::Equal {
            cmp
        } else {
            self.value.cmp(&other.value)
        }
    }
}

impl Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{name}={value}", name = self.name, value = self.value)
    }
}

const SEP: u8 = 0xfe;

impl Hash for Label {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(self.name.as_bytes());
        state.write_u8(SEP);
        state.write(self.value.as_bytes());
    }
}


fn write_string(buf: &mut Vec<u8>, s: &str) {
    write_int(buf, s.len());
    if !s.is_empty() {
        buf.extend_from_slice(s.as_bytes());
    }
}

fn read_string(slice: &[u8]) -> (&str, &[u8]) {
    let (len, n) = u64::decode_var(slice).unwrap();
    let len = len as usize;
    let tail = &slice[n..];
    let s = std::str::from_utf8(&tail[..len]).unwrap();
    let tail = &tail[len..];
    (s, tail)
}

pub fn write_int<T: VarInt>(dst: &mut Vec<u8>, v: T) {
    let len = dst.len();
    dst.resize(len + v.required_space(), 0);
    let _ = v.encode_var(&mut dst[len..]);
}
