use ahash::{AHashMap, AHashSet, AHasher};
use xxhash_rust::xxh3::xxh3_64;

pub type FastHasher = AHasher;
pub type FastHashMap<K, V> = AHashMap<K, V>;
pub type FastHashSet<T> = AHashSet<T>;

pub fn fast_hash64(bytes: &[u8]) -> u64 {
    xxh3_64(bytes)
}

pub use ahash::HashSetExt;
