use blart::AsBytes;
use std::borrow::Borrow;
use std::fmt::Display;
use std::fmt::Write;
use std::ops::Deref;
use crate::types::METRIC_NAME_LABEL;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndexKey(Box<[u8]>);

const SENTINEL: u8 = 0;

impl IndexKey {
    pub fn new(key: &str) -> Self {
        Self::from(key)
    }

    pub fn for_metric_name(metric_name: &str) -> Self {
        Self::for_label_value(METRIC_NAME_LABEL, metric_name)
    }

    pub fn for_label_value(label_name: &str, value: &str) -> Self {
        Self::from(format!("{label_name}={value}"))
    }

    pub fn as_str(&self) -> &str {
        let buf = &self.0[..self.0.len() - 1];
        std::str::from_utf8(buf).unwrap()
    }

    pub fn split(&self) -> Option<(&str, &str)> {
        let key = self.as_str();
        key.find('=')
            .map(|index| (&key[..index], &key[index + 1..self.len()]))
    }

    pub(crate) fn sub_string(&self, start: usize) -> &str {
        let buf = &self.0[start..self.0.len() - 1];
        std::str::from_utf8(buf).expect("invalid utf8")
    }

    pub fn into_vec(self) -> Vec<u8> {
        self.0.into_vec()
    }

    pub fn len(&self) -> usize {
        self.0.len() - 1
    }
}

impl Display for IndexKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Deref for IndexKey {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsBytes for IndexKey {
    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl AsRef<[u8]> for IndexKey {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl From<&[u8]> for IndexKey {
    fn from(key: &[u8]) -> Self {
        let utf8 = String::from_utf8_lossy(key);
        let mut v = utf8.as_bytes().to_vec();
        v.push(SENTINEL);
        IndexKey(v.into_boxed_slice())
    }
}

impl From<Vec<u8>> for IndexKey {
    fn from(key: Vec<u8>) -> Self {
        Self::from(key.as_bytes())
    }
}

impl From<&str> for IndexKey {
    fn from(key: &str) -> Self {
        let mut key = key.as_bytes().to_vec();
        key.push(SENTINEL);
        IndexKey(key.into_boxed_slice())
    }
}

impl From<String> for IndexKey {
    fn from(key: String) -> Self {
        Self::from(key.as_str())
    }
}

impl Borrow<[u8]> for IndexKey {
    fn borrow(&self) -> &[u8] {
        self.as_bytes()
    }
}

pub fn format_key_for_label_prefix(dest: &mut String, label_name: &str) {
    dest.clear();
    // Safety: according to the source, write! does not panic
    write!(dest, "{label_name}=").expect("write! macro failed");
}

pub fn format_key_for_label_value(dest: &mut String, label_name: &str, value: &str) {
    dest.clear();
    // according to https://github.com/rust-lang/rust/blob/1.47.0/library/alloc/src/string.rs#L2414-L2427
    // write! will not return an Err, so the unwrap is safe
    write!(dest, "{label_name}={value}\0").expect("write! macro failed");
}

pub fn format_key_for_metric_name(dest: &mut String, metric_name: &str) {
    format_key_for_label_value(dest, METRIC_NAME_LABEL, metric_name);
}

pub fn get_key_for_label_prefix(label_name: &str) -> String {
    let mut value = String::with_capacity(label_name.len() + 1);
    format_key_for_label_prefix(&mut value, label_name);
    value
}

pub fn get_key_for_label_value(label_name: &str, value: &str) -> String {
    let mut res = String::with_capacity(label_name.len() + value.len() + 1);
    format_key_for_label_value(&mut res, label_name, value);
    res
}

pub fn get_key_for_metric_name(metric_name: &str) -> String {
    get_key_for_label_value(METRIC_NAME_LABEL, metric_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use blart::TreeMap;

    #[test]
    fn test_new() {
        let key = IndexKey::new("test_key");
        assert_eq!(key.as_str(), "test_key");
    }

    #[test]
    fn test_for_metric_name() {
        let key = IndexKey::for_metric_name("metric");
        assert_eq!(key.as_str(), "__name__=metric");
    }

    #[test]
    fn test_for_label_value() {
        let key = IndexKey::for_label_value("label", "value");
        assert_eq!(key.as_str(), "label=value");
    }

    #[test]
    fn test_as_str() {
        let key = IndexKey::new("test_key");
        assert_eq!(key.as_str(), "test_key");
    }

    #[test]
    fn test_split() {
        let key = IndexKey::new("label=value");
        let (label, value) = key.split().unwrap();
        assert_eq!(label, "label");
        assert_eq!(value, "value");
    }

    #[test]
    fn test_sub_string() {
        let key = IndexKey::new("label=value");
        assert_eq!(key.sub_string(6), "value");
    }

    #[test]
    fn test_into_vec() {
        let key = IndexKey::new("test_key");
        assert_eq!(key.into_vec(), b"test_key\0".to_vec());
    }

    #[test]
    fn test_len() {
        let key = IndexKey::new("test_key");
        assert_eq!(key.len(), 8);
    }

    #[test]
    fn test_display() {
        let key = IndexKey::new("test_key");
        assert_eq!(format!("{}", key), "test_key");
    }

    #[test]
    fn test_as_bytes() {
        let key = IndexKey::new("test_key");
        assert_eq!(key.as_bytes(), b"test_key\0");
    }

    #[test]
    fn test_from_u8_slice() {
        let key = IndexKey::from(b"test_key".as_ref());
        assert_eq!(key.as_str(), "test_key");
    }

    #[test]
    fn test_from_vec_u8() {
        let key = IndexKey::from(b"test_key".to_vec());
        assert_eq!(key.as_str(), "test_key");
    }

    #[test]
    fn test_from_str() {
        let key = IndexKey::from("test_key");
        assert_eq!(key.as_str(), "test_key");
    }

    #[test]
    fn test_borrow() {
        let key = IndexKey::new("test_key");
        let borrowed: &[u8] = key.borrow();
        assert_eq!(borrowed, b"test_key\0");
    }


    #[test]
    fn test_with_collection() {
        let mut tree: TreeMap<IndexKey, String> = TreeMap::new();

        let regions = ["US", "EU", "APAC"];
        let services = ["web", "api", "db"];
        let environments = ["prod", "staging", "dev"];

        for region in regions.iter() {
            let region_key = IndexKey::for_label_value("region", region);
            let _ = tree.try_insert(region_key, region.to_string()).unwrap();
        }

        for service in services.iter() {
            let key = IndexKey::for_label_value("service", service);
            let _ = tree.try_insert(key, service.to_string()).unwrap();
        }

        for environment in environments.iter() {
            let key = IndexKey::for_label_value("environment", environment);
            let _ = tree.try_insert(key, environment.to_string()).unwrap();
        }

        for region in regions.iter() {
            let search_key = get_key_for_label_value("region", region);
            let value = tree.get(search_key.as_bytes()).map(|v| v.as_str());
            assert_eq!(value, Some(*region));
        }

        for service in services.iter() {
            let search_key = get_key_for_label_value("service", service);
            let value = tree.get(search_key.as_bytes()).map(|v| v.as_str());
            assert_eq!(value, Some(*service));
        }

        for environment in environments.iter() {
            let search_key = get_key_for_label_value("environment", environment);
            let value = tree.get(search_key.as_bytes()).map(|v| v.as_str());
            assert_eq!(value, Some(*environment));
        }
    }
}
