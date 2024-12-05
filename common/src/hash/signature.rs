use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use crate::hash::FastHasher;
use crate::label::Label;

#[derive(Debug, Default, Clone, PartialEq, Eq, Copy, Ord, PartialOrd)]
pub struct Signature(u64);

/// implement hash which returns the value of the inner u64
impl Hash for Signature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl Deref for Signature {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

const EMPTY_LIST_SIGNATURE: u64 = 0x9e3779b97f4a7c15;
const EMPTY_NAME_VALUE: u64 = 0x9e3779b97f4a7c16;

impl Signature {
    pub fn new(s: &str) -> Signature {
        let mut hasher = FastHasher::default();
        if !s.is_empty() {
            hasher.write(s.as_bytes());
        } else {
            hasher.write_u64(EMPTY_NAME_VALUE);
        }
        Signature(hasher.finish())
    }

    pub fn from_name_and_labels<'a>(name: &str, iter: impl Iterator<Item = &'a Label>) -> Self {
        let mut hasher = FastHasher::default();
        let mut has_tags = false;

        if !name.is_empty() {
            hasher.write(name.as_bytes());
        } else {
            hasher.write_u64(EMPTY_NAME_VALUE);
        }
        for tag in iter {
            tag.hash(&mut hasher);
            has_tags = true;
        }
        if !has_tags {
            hasher.write_u64(EMPTY_LIST_SIGNATURE);
        }
        let sig = hasher.finish();
        Signature(sig)
    }

    pub fn from_vec<T: Hash>(items: &Vec<T>) -> Self {
        let mut hasher = FastHasher::default();
        for item in items {
            item.hash(&mut hasher);
        }
        Signature(hasher.finish())
    }

    pub fn create_from_iter<'a, T: Hash + 'a>(iter: impl Iterator<Item = &'a T>) -> Self {
        let mut hasher = FastHasher::default();
        for item in iter {
            item.hash(&mut hasher);
        }
        Signature(hasher.finish())
    }
}

impl From<Signature> for u64 {
    fn from(sig: Signature) -> Self {
        sig.0
    }
}

impl From<u64> for Signature {
    fn from(sig: u64) -> Self {
        Signature(sig)
    }
}

impl From<Vec<String>> for Signature {
    fn from(labels: Vec<String>) -> Self {
        Signature::from_vec(&labels)
    }
}

impl From<&Vec<String>> for Signature {
    fn from(labels: &Vec<String>) -> Self {
        Signature::from_vec(labels)
    }
}

impl From<&str> for Signature {
    fn from(s: &str) -> Self {
        let mut hasher = FastHasher::default();
        if !s.is_empty() {
            hasher.write(s.as_bytes());
        } else {
            hasher.write_u64(EMPTY_NAME_VALUE);
        }
        Signature(hasher.finish())
    }
}

impl Display for Signature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:x}", self.0)
    }
}