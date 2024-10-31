use gxhash::{gxhash64, GxHasher, HashMap, HashSet};

pub type FastHasher = GxHasher;
pub type FastHashMap<K, V> = HashMap<K, V>;
pub type FastHashSet<T> = HashSet<T>;

pub fn fast_hash64(bytes: &[u8]) -> u64 {
    // gxhash is not a cryptographic hash in any case so a fixed seed
    // is not an issue for our purposes
    gxhash64(bytes, 52731)
}

// export HashSetExt for convenience
pub use gxhash::HashSetExt;
