use crate::hash::HashSetExt;
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use crate::hash::{fast_hash64, FastHashSet, IntMap};
use crate::pool::get_pooled_buffer;

const U64_SIZE: usize = 8;
const SIZEOF_USIZE: usize = size_of::<usize>();
const METADATA_SIZE: usize = 8 /* sizeof u64 */ + SIZEOF_USIZE;
const BUCKETS_COUNT: usize = 512;
const BUCKETS_COUNT_SMALL: usize = 256;
const CHUNK_SIZE: usize = 64 * 1024;
const BUCKET_SIZE_BITS: usize = 40;
const GEN_SIZE_BITS: usize = 64 - BUCKET_SIZE_BITS;
const MAX_GEN: u64 = (1 << GEN_SIZE_BITS) - 1;
const GEN_BIT_MASK: u64 = (1 << GEN_SIZE_BITS) - 1;
const MAX_BUCKET_SIZE: usize = 1 << BUCKET_SIZE_BITS;

/// MAX_SUB_VALUE_LEN is the maximum size of sub value chunk.
///
/// - 16 bytes are for sub key encoding
/// - 4 bytes are for key.len()+value.len() encoding
/// - 1 byte is implementation detail of FastCache
const MAX_SUB_VALUE_LEN: usize = CHUNK_SIZE - 16 - 4 - 1;

/// MAX_KEY_LEN is the maximum size of key.
///
/// - 16 bytes are for (hash + value_len)
/// - 4 bytes are for encoding key.len()+sub_key.len()
/// - 1 byte is implementation detail of FastCache
const MAX_KEY_LEN: usize = CHUNK_SIZE - 16 - 4 - 1;
const SMALL_CACHE_SIZE_MIN: usize = 4 * 1024 * 1024;

/// Stats represents cache stats.
///
/// Use Cache.update_stats for obtaining fresh stats from the cache.
#[derive(Default)]
pub struct Stats {
    /// get_calls is the number of Get calls.
    pub get_calls: u64,

    /// set_calls is the number of Set calls.
    pub set_calls: u64,

    /// misses is the number of cache misses.
    pub misses: u64,

    /// collisions is the number of cache collisions.
    ///
    /// Usually the number of collisions must be close to zero.
    /// High number of collisions suggest something wrong with cache.
    pub collisions: u64,

    /// Corruptions is the number of detected corruptions of the cache.
    ///
    /// Corruptions may occur when corrupted cache is loaded from file.
    pub corruptions: u64,

    /// entries_count is the current number of entries in the cache.
    pub entries_count: u64,

    /// bytes_size is the current size of the cache in bytes.
    pub bytes_size: u64,

    /// max_bytes_size is the maximum allowed size of the cache in bytes (aka capacity).
    pub max_bytes_size: u64,

    /// get_big_calls is the number of get_big calls.
    pub get_big_calls: u64,

    /// the number of set_big calls.
    pub set_big_calls: u64,

    /// the number of calls to set_big with too big key.
    pub too_big_key_errors: u64,

    /// invalid_meta value_errors is the number of calls to get_big resulting
    /// to invalid meta value.
    pub invalid_meta_value_errors: u64,

    /// the number of calls to GetBig resulting to a chunk with invalid length.
    pub invalid_value_len_errors: u64,

    /// invalid_value_hash_errors is the number of calls to get_big resulting
    /// to a chunk with invalid hash value.
    pub invalid_value_hash_errors: u64,
}

impl Stats {
    /// Reset resets s, so it may be re-used again in Cache.UpdateStats.
    pub fn reset(&mut self) {
        self.get_calls = 0;
        self.set_calls = 0;
        self.misses = 0;
        self.collisions = 0;
        self.corruptions = 0;
        self.entries_count = 0;
        self.bytes_size = 0;
        self.max_bytes_size = 0;
        self.get_big_calls = 0;
        self.set_big_calls = 0;
        self.too_big_key_errors = 0;
        self.invalid_meta_value_errors = 0;
        self.invalid_value_hash_errors = 0;
        self.invalid_value_len_errors = 0;
    }
}

/// BigStats contains stats for get_big/set_big methods.
#[derive(Default, Debug)]
pub struct BigStats {
    /// get_big_calls is the number of GetBig calls.
    pub get_big_calls: AtomicU64,

    /// the number of set_big calls.
    pub set_big_calls: AtomicU64,

    /// too_big_key_errors is the number of calls to SetBig with too big key.
    pub too_big_key_errors: AtomicU64,

    /// invalid_meta_value_errors is the number of calls to GetBig resulting
    /// to invalid meta value.
    pub invalid_meta_value_errors: AtomicU64,

    /// invalid_value_len_errors is the number of calls to GetBig resulting
    /// to a chunk with invalid length.
    pub invalid_value_len_errors: AtomicU64,

    /// invalid_value_hash_errors is the number of calls to GetBig resulting
    /// to a chunk with invalid hash value.
    pub invalid_value_hash_errors: AtomicU64,
}

impl BigStats {
    pub fn new() -> Self {
        Self {
            invalid_value_hash_errors: AtomicU64::new(0),
            invalid_value_len_errors: AtomicU64::new(0),
            invalid_meta_value_errors: AtomicU64::new(0),
            get_big_calls: AtomicU64::new(0),
            set_big_calls: AtomicU64::new(0),
            too_big_key_errors: AtomicU64::new(0),
        }
    }

    pub fn reset(&mut self) {
        self.invalid_value_hash_errors.store(0, Ordering::SeqCst);
        self.invalid_value_len_errors.store(0, Ordering::Relaxed);
        self.get_big_calls.store(0, Ordering::Relaxed);
        self.invalid_meta_value_errors.store(0, Ordering::Relaxed);
        self.too_big_key_errors.store(0, Ordering::Relaxed);
        self.set_big_calls.store(0, Ordering::Relaxed);
    }
}

/// FastCache is a fast thread-safe in-memory cache optimized for big number of entries.
///
/// It has much lower impact on alloc/fragmentation comparing to a simple `HashMap<str,[u8]>`.
///
/// Multiple threads may call any methods on the same cache instance.
///
/// Call reset when the cache is no longer needed. This reclaims the allocated
/// memory.
pub struct FastCache {
    buckets: Vec<Bucket>,
    big_stats: BigStats,
}

/// new() returns new cache with the given maxBytes capacity in bytes.
///
/// max_bytes must be smaller than the available RAM size for the app,
/// since the cache holds data in memory.
///
/// If max_bytes is less than MIN_CACHE_SIZE, then the minimum cache capacity is MIN_CACHE_SIZE.
impl FastCache {
    pub fn new(max_bytes: usize) -> Self {
        let max_bytes = {
            if max_bytes < SMALL_CACHE_SIZE_MIN {
                SMALL_CACHE_SIZE_MIN
            } else {
                max_bytes
            }
        };

        let max_buckets = if max_bytes <= SMALL_CACHE_SIZE_MIN {
            BUCKETS_COUNT_SMALL
        } else {
            BUCKETS_COUNT
        };

        let max_bucket_bytes = max_bytes.div_ceil(max_buckets);
        let mut bucket_count = max_bytes / max_bucket_bytes;
        bucket_count = bucket_count.clamp(bucket_count, BUCKETS_COUNT);

        let mut buckets = Vec::with_capacity(bucket_count);
        (0..bucket_count).for_each(|_| {
            buckets.push(Bucket::new(max_bucket_bytes));
        });
        FastCache {
            buckets,
            big_stats: BigStats::new(),
        }
    }

    /// `.set` stores (k, v) in the cache.
    ///
    /// `.get` must be used for reading the stored entry.
    ///
    /// The stored entry may be evicted at any time either due to cache
    /// overflow or due to unlikely hash collision.
    /// Pass higher maxBytes value to `new` if the added items disappear
    /// frequently.
    ///
    /// (k, v) entries with summary size exceeding 64KB aren't stored in the cache.
    /// set_big can be used for storing entries exceeding 64KB.
    ///
    /// k and v contents may be modified after returning from Set.
    pub fn set(&self, k: &[u8], v: &[u8]) {
        let h = fast_hash64(k);
        let bucket = self._get_bucket(h);
        bucket.set(k, v, h)
    }

    /// `set_big` sets (k, v) to c where v.len() may exceed 64KB.
    ///
    /// `get_big` must be used for reading stored values.
    ///
    /// The stored entry may be evicted at any time either due to cache
    /// overflow or due to unlikely hash collision.
    /// Pass higher maxBytes value to New if the added items disappear
    /// frequently.
    ///
    /// It is safe to store entries smaller than 64KB with set_big.
    ///
    /// k and v contents may be modified after returning from set_big.
    pub fn set_big(&mut self, k: &[u8], v: &[u8]) {
        self.big_stats.set_big_calls.fetch_add(1, Ordering::Relaxed);
        if k.len() > MAX_KEY_LEN {
            self.big_stats
                .too_big_key_errors
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        let value_len = v.len();
        let value_hash = fast_hash64(v);

        let mut meta_buf: [u8; METADATA_SIZE] = [0; METADATA_SIZE];

        for (i, chunk) in v.chunks(MAX_SUB_VALUE_LEN).enumerate() {
            marshal_meta(&mut meta_buf, value_hash, i);
            self.set(&meta_buf, chunk);
        }

        // Write meta value, which consists of value_hash and value_len.
        marshal_meta(&mut meta_buf, value_hash, value_len);
        self.set(k, &meta_buf)
    }

    /// get appends value by the key k to dst and returns the result.
    ///
    /// get returns only values stored via `set`.
    pub fn get(&self, k: &[u8], dst: &mut Vec<u8>) -> bool {
        let h = fast_hash64(k);
        let bucket = self._get_bucket(h);
        bucket.get(k, h, dst)
    }

    /// Searches for the value for the given k, appends it to dst
    /// and returns the result.
    ///
    /// Returns only values stored via set_big(). It doesn't work
    /// with values stored via other methods.
    pub fn get_big(&mut self, k: &[u8], dst: &mut Vec<u8>) -> bool {
        self.big_stats.get_big_calls.fetch_add(1, Ordering::Relaxed);

        let mut key_buf = get_pooled_buffer(32);

        // Read and parse meta value
        if !self.get(k, &mut key_buf) {
            // Nothing found.
            return false;
        }

        if key_buf.len() != METADATA_SIZE {
            self.big_stats
                .invalid_meta_value_errors
                .fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let meta = unmarshal_meta(&key_buf);
        if meta.is_none() {
            return false;
        }

        let (value_hash, value_len) = meta.unwrap();

        // Collect result from chunks.
        let dst_len = dst.len();
        dst.reserve(value_len);

        let mut meta_key_buf: [u8; METADATA_SIZE] = [0; METADATA_SIZE];
        let mut i: usize = 0;
        while (dst.len() - dst_len) < value_len {
            marshal_meta(&mut meta_key_buf, value_hash, i);
            i += 1;

            if !self.get(&meta_key_buf, dst) {
                // Cannot find sub value
                return false;
            }
        }

        // Verify the obtained value.
        let v = &dst[dst_len..];
        if v.len() != value_len {
            // Corrupted data during the load from file. Just skip it.
            self.big_stats
                .invalid_value_len_errors
                .fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let h = fast_hash64(v);
        if h != value_hash {
            self.big_stats
                .invalid_value_hash_errors
                .fetch_add(1, Ordering::Relaxed);
            return false;
        }

        true
    }

    /// has returns true if entry for the given key k exists in the cache.
    pub fn has(&self, k: &[u8]) -> bool {
        let h = fast_hash64(k);
        let bucket = self._get_bucket(h);
        bucket.has(k, h)
    }

    /// update_stats adds cache stats to s.
    ///
    /// call s.reset() before calling update_stats if s is re-used.
    pub fn update_stats(&self, s: &mut Stats) {
        for bucket in self.buckets.iter() {
            bucket.update_stats(s);
        }
        s.get_big_calls += self.big_stats.get_big_calls.fetch_add(0, Ordering::Relaxed);
        s.set_big_calls += self.big_stats.set_big_calls.fetch_add(0, Ordering::Relaxed);
        s.too_big_key_errors += self
            .big_stats
            .too_big_key_errors
            .fetch_add(0, Ordering::Relaxed);
        s.invalid_meta_value_errors += self
            .big_stats
            .invalid_meta_value_errors
            .fetch_add(0, Ordering::Relaxed);
        s.invalid_value_len_errors += self
            .big_stats
            .invalid_value_len_errors
            .fetch_add(0, Ordering::Relaxed);
        s.invalid_value_hash_errors += self
            .big_stats
            .invalid_value_hash_errors
            .fetch_add(0, Ordering::Relaxed);
    }

    pub fn reset(&mut self) {
        for bucket in self.buckets.iter_mut() {
            bucket.reset();
        }
        self.big_stats.reset();
    }

    // del deletes value for the given k from the cache.
    pub fn del(&self, k: &[u8]) {
        let h = fast_hash64(k);
        let bucket = self._get_bucket(h);
        bucket.del(h)
    }

    pub fn clean(&mut self) {
        for bucket in self.buckets.iter_mut() {
            bucket.clean();
        }
    }

    fn _get_bucket(&self, hash: u64) -> &Bucket {
        let idx = (hash % self.buckets.len() as u64) as usize;
        &self.buckets[idx]
    }
}

#[inline]
fn marshal_meta(buf: &mut [u8; METADATA_SIZE], value_hash: u64, index: usize) {
    buf[0..U64_SIZE].copy_from_slice(&value_hash.to_ne_bytes());
    buf[U64_SIZE..METADATA_SIZE].copy_from_slice(&index.to_ne_bytes());
}

fn unmarshal_meta(src: &[u8]) -> Option<(u64, usize)> {
    if src.len() < METADATA_SIZE {
        return None;
    }

    let mut hash_buf: [u8; U64_SIZE] = Default::default();
    let mut index_buf: [u8; SIZEOF_USIZE] = Default::default();

    hash_buf.copy_from_slice(&src[0..U64_SIZE]);
    index_buf.copy_from_slice(&src[U64_SIZE..METADATA_SIZE]);

    let value_hash: u64 = u64::from_ne_bytes(hash_buf);
    let index: usize = usize::from_ne_bytes(index_buf);

    Some((value_hash, index))
}

struct BucketInner {
    /// chunks is a ring buffer with encoded (k, v) pairs.
    /// It consists of 64KB chunks.
    chunks: Vec<Vec<u8>>,

    /// maps hash(k) to idx of (k, v) pair in chunks.
    hash_idx_map: IntMap<u64, usize>,

    /// idx points to chunks for writing the next (k, v) pair.
    idx: usize,

    /// gen is the generation of chunks.
    gen: u64,

    get_calls: u64,
    set_calls: u64,
    misses: u64,
    collisions: u64,
    corruptions: u64,
}

impl BucketInner {
    fn new(chunk_count: usize) -> Self {
        let chunks: Vec<Vec<u8>> = Vec::with_capacity(chunk_count);

        Self {
            chunks,
            hash_idx_map: IntMap::default(),
            idx: 0,
            gen: 0,
            get_calls: 0,
            set_calls: 0,
            misses: 0,
            collisions: 0,
            corruptions: 0,
        }
    }

    fn reset(&mut self) {
        for chunk in self.chunks.iter_mut() {
            chunk.clear();
        }
        self.idx = 0;
        self.gen = 1;
        self.hash_idx_map.clear();
        self.get_calls = 0;
        self.set_calls = 0;
        self.misses = 0;
        self.collisions = 0;
        self.corruptions = 0;
    }

    fn clean_locked(&mut self) {
        let b_gen = self.gen & GEN_BIT_MASK;
        let b_idx = self.idx;

        // todo: use with_capacity
        let mut to_remove: FastHashSet<u64> = FastHashSet::with_capacity(16);
        for (k, v) in self.hash_idx_map.iter() {
            let gen = (*v >> BUCKET_SIZE_BITS) as u64;
            let idx = *v & ((1 << BUCKET_SIZE_BITS) - 1);
            if (gen + 1 == b_gen || gen == MAX_GEN && b_gen == 1) && idx >= b_idx
                || gen == b_gen && idx < b_idx
            {
                continue;
            }
            to_remove.insert(*k);
        }

        self.hash_idx_map
            .retain(|key, _size| !to_remove.contains(key));
    }

    fn set(&mut self, k: &[u8], v: &[u8], h: u64) {
        self.set_calls += 1;
        if k.len() >= (1 << 16) || v.len() >= (1 << 16) {
            // Too big key or value - its length cannot be encoded
            // with 2 bytes (see below). Skip the entry.
            return;
        }
        let kv_len_buf: [u8; 4] = [
            (k.len() >> 8) as u8,
            k.len() as u8,
            (v.len() >> 8) as u8,
            v.len() as u8,
        ];
        let kv_len = kv_len_buf.len() + k.len() + v.len();
        if kv_len >= CHUNK_SIZE {
            // Do not store too big keys and values, since they do not
            // fit a chunk.
            return;
        }

        let mut need_clean = false;
        let mut idx = self.idx;
        let mut idx_new = idx + kv_len;
        let mut chunk_idx: usize = idx / CHUNK_SIZE;
        let chunk_idx_new = idx_new / CHUNK_SIZE;
        if chunk_idx_new > chunk_idx {
            if chunk_idx_new >= self.chunks.len() {
                idx = 0;
                idx_new = kv_len;
                chunk_idx = 0;
                self.gen += 1;
                if self.gen & GEN_BIT_MASK == 0 {
                    self.gen += 1;
                }
                need_clean = true
            } else {
                idx = chunk_idx_new * CHUNK_SIZE;
                idx_new = idx + kv_len;
                chunk_idx = chunk_idx_new
            }
        }

        if chunk_idx >= self.chunks.len() {
            let vec: Vec<u8> = Vec::with_capacity(CHUNK_SIZE);
            self.chunks.push(vec)
        }

        let chunk = self.chunks.get_mut(chunk_idx);
        debug_assert!(chunk.is_some());

        let chunk = chunk.unwrap();
        chunk.extend_from_slice(&kv_len_buf);
        chunk.extend_from_slice(k);
        chunk.extend_from_slice(v);

        self.hash_idx_map
            .insert(h, idx | (self.gen << BUCKET_SIZE_BITS) as usize);
        self.idx = idx_new;
        if need_clean {
            self.clean_locked()
        }
    }

    fn is_valid_generation(&self, v: usize) -> bool {
        let b_gen = self.gen & GEN_BIT_MASK;
        let gen = (v >> BUCKET_SIZE_BITS) as u64;
        let idx = v & ((1 << BUCKET_SIZE_BITS) - 1);
        gen == b_gen && idx < self.idx
            || idx >= self.idx && gen + 1 == b_gen
            || idx >= self.idx && gen == MAX_GEN && b_gen == 1
    }


    fn retrieve_value(&mut self, k: &[u8], v: usize) -> Option<&[u8]> {
        let chunk_idx: usize = v / CHUNK_SIZE;
        if chunk_idx >= self.chunks.len() {
            self.corruptions += 1;
            return None;
        }

        let mut idx: usize = v % CHUNK_SIZE;
        if idx + 4 >= CHUNK_SIZE {
            self.corruptions += 1;
            return None;
        }

        let chunk = &self.chunks[chunk_idx];
        let kv_len_buf = &chunk[idx..idx + 4];
        let key_len = (((kv_len_buf[0] as u64) << 8) | kv_len_buf[1] as u64) as usize;
        let val_len = (((kv_len_buf[2] as u64) << 8) | kv_len_buf[3] as u64) as usize;
        idx += 4;

        if idx + (key_len + val_len) >= CHUNK_SIZE {
            self.corruptions += 1;
            return None;
        }

        let key = &chunk[idx..idx + key_len];
        if k == key {
            idx += key_len;
            let res = &chunk[idx..idx + val_len];
            Some(res)
        } else {
            self.misses += 1;
            self.collisions += 1;
            None
        }
    }

    fn get(&mut self, k: &[u8], h: u64) -> Option<&[u8]> {
        self.get_calls += 1;

        if let Some(v) = self.hash_idx_map.get(&h) {
            if self.is_valid_generation(*v) {
                return self.retrieve_value(k, *v);
            }
        }

        None
    }
}

struct Bucket {
    inner: RwLock<BucketInner>,
}

impl Bucket {
    pub fn new(max_bytes: usize) -> Self {
        if max_bytes == 0 {
            panic!("max_bytes cannot be zero");
        }
        if max_bytes >= MAX_BUCKET_SIZE {
            panic!(
                "too big max_bytes={}; should be smaller than {}",
                max_bytes, MAX_BUCKET_SIZE
            );
        }
        let max_chunks = max_bytes.div_ceil(CHUNK_SIZE);
        let data = BucketInner::new(max_chunks);
        let inner = RwLock::new(data);
        Self { inner }
    }

    pub fn reset(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.reset();
    }

    fn clean(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.clean_locked();
    }

    fn set(&self, k: &[u8], v: &[u8], h: u64) {
        let mut inner = self.inner.write().unwrap();
        inner.set(k, v, h);
    }

    fn get(&self, k: &[u8], h: u64, dst: &mut Vec<u8>) -> bool {
        let mut inner = self.inner.write().unwrap();
        match inner.get(k, h) {
            None => false,
            Some(v) => {
                dst.extend_from_slice(v);
                true
            }
        }
    }

    pub(self) fn has(&self, k: &[u8], h: u64) -> bool {
        let mut inner = self.inner.write().unwrap();
        inner.get(k, h).is_some()
    }

    pub(self) fn del(&self, h: u64) {
        let mut inner = self.inner.write().unwrap();
        inner.hash_idx_map.remove(&h);
    }

    pub(self) fn update_stats(&self, s: &mut Stats) {
        let inner = self.inner.read().unwrap();

        s.get_calls += inner.get_calls;
        s.set_calls += inner.set_calls;
        s.misses += inner.misses;
        s.collisions += inner.collisions;
        s.corruptions += inner.corruptions;
        s.entries_count += inner.hash_idx_map.len() as u64;
        let mut bytes_size = 0;
        for chunk in inner.chunks.iter() {
            bytes_size += chunk.capacity()
        }
        s.bytes_size += bytes_size as u64;
        s.max_bytes_size += (inner.chunks.len() * CHUNK_SIZE) as u64;
    }

    #[cfg(test)]
    pub(self) fn get_gen(&self) -> u64 {
        let inner = self.inner.read().unwrap();
        inner.gen
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use test_case::test_case;

    use crate::fast_cache::fast_cache::{FastCache, Stats, BUCKETS_COUNT, CHUNK_SIZE};

    const BIG_CACHE_SIZE_MIN: usize = 32 * 1024 * 1024;

    #[test]
    fn test_cache_small() {
        let mut c = FastCache::new(1);

        let mut v: Vec<u8> = vec![];

        c.get(b"aaa", &mut v);
        assert_eq!(
            v.len(),
            0,
            "unexpected non-empty value obtained from small cache: {}",
            String::from_utf8_lossy(&v)
        );

        assert!(
            !c.get(b"aaa", &mut v),
            "unexpected non-empty value obtained from small cache: {}",
            String::from_utf8_lossy(&v)
        );

        c.set(b"key", b"value");
        let mut dst: Vec<u8> = vec![];

        c.get(b"key", &mut dst);
        assert_eq!(
            String::from_utf8_lossy(&dst),
            "value",
            "unexpected value obtained; got {}; want {}",
            String::from_utf8_lossy(&v),
            "value"
        );

        v.clear();
        c.get(b"aaa", &mut v);
        assert_eq!(
            v.len(),
            0,
            "unexpected non-empty value obtained from small cache: {}",
            String::from_utf8_lossy(&v)
        );

        v.clear();
        c.set(b"aaa", b"bbb");
        c.get(b"aaa", &mut v);
        assert_eq!(
            String::from_utf8_lossy(&v),
            "bbb",
            "unexpected value obtained; got {}; want {}",
            String::from_utf8_lossy(&v),
            "bbb"
        );

        v.clear();
        assert!(
            c.get(b"aaa", &mut v) && String::from_utf8_lossy(&v) == "bbb",
            "unexpected value obtained; got {}; want {}",
            String::from_utf8_lossy(&v),
            "bbb"
        );

        c.reset();
        v.clear();
        c.get(b"aaa", &mut v);
        assert_eq!(
            v.len(),
            0,
            "unexpected non-empty value obtained from empty cache: {}",
            String::from_utf8_lossy(&v)
        );

        assert!(
            !c.get(b"aaa", &mut v) && v.is_empty(),
            "unexpected non-empty value obtained from small cache: {}",
            String::from_utf8_lossy(&v)
        );

        // Test empty value
        let k = b"empty";
        c.set(k, &[]);
        assert!(
            c.get(k, &mut v) && v.is_empty(),
            "unexpected non-empty value obtained from empty entry: {}",
            String::from_utf8_lossy(&v)
        );
        assert!(
            c.get(k, &mut v),
            "cannot find empty entry for key {}",
            String::from_utf8_lossy(k)
        );
        assert_eq!(
            v.len(),
            0,
            "unexpected non-empty value obtained from empty entry: {}",
            String::from_utf8_lossy(&v)
        );
        assert!(
            c.has(k),
            "cannot find empty entry for key {}",
            String::from_utf8_lossy(k)
        );

        assert!(!c.has(b"foobar"), "non-existing entry found in the cache");
    }

    #[test]
    fn test_cache_wrap() {
        let c = FastCache::new(BUCKETS_COUNT * (CHUNK_SIZE as f64 * 1.5) as usize);

        let calls: u64 = 5_000_000;

        let mut dst: Vec<u8> = vec![];
        for i in 0..calls {
            let k = make_buf("key", i);
            let v = make_buf("value", i);
            c.set(&k, &v);

            dst.clear();
            assert!(
                c.get(&k, &mut dst),
                "value not found for key: {}",
                String::from_utf8_lossy(&k)
            );

            assert_eq!(
                v,
                dst,
                "unexpected value for key {}; got {}; want {}",
                String::from_utf8_lossy(&k),
                String::from_utf8_lossy(&v),
                String::from_utf8_lossy(&dst)
            );
        }

        for i in 0..calls / 10 {
            let k = make_buf("key", i);
            let v = make_buf("value", i);
            dst.clear();
            c.get(&k, &mut dst);
            assert!(
                !dst.is_empty() && v != dst,
                "unexpected value for key {}; got {}; want {}",
                String::from_utf8_lossy(&k),
                String::from_utf8_lossy(&dst),
                String::from_utf8_lossy(&v)
            );
        }

        let mut s: Stats = Stats::default();
        c.update_stats(&mut s);
        let get_calls = calls + calls / 10;
        assert_eq!(
            s.get_calls, get_calls,
            "unexpected number of get_calls; got {}; want {}",
            s.get_calls, get_calls
        );

        assert_eq!(
            s.set_calls, calls,
            "unexpected number of set_calls; got {}; want {}",
            s.set_calls, calls
        );

        assert!(
            s.misses == 0 || s.misses >= (calls / 10),
            "unexpected number of misses; got {}; it should be between 0 and {}",
            s.misses,
            calls / 10
        );

        assert_ne!(
            s.collisions, 0,
            "unexpected number of collisions; got {}; want 0",
            s.collisions
        );

        assert!(
            s.entries_count < (calls / 5),
            "unexpected number of items; got {}; cannot be smaller than {}",
            s.entries_count,
            calls / 5
        );

        assert!(
            s.bytes_size < 1024,
            "unexpected bytes_size; got {}; cannot be smaller than {}",
            s.bytes_size,
            1024
        );

        assert!(
            s.max_bytes_size < 32 * 1024 * 1024,
            "unexpected max_bytes_size; got {}; cannot be smaller than {}",
            s.max_bytes_size,
            32 * 1024 * 1024
        )
    }

    fn make_buf(prefix: &str, i: u64) -> Vec<u8> {
        Vec::from(format!("{}-{}", prefix, i).as_bytes())
    }

    #[test]
    fn test_cache_del() {
        let c = FastCache::new(1024);

        let mut dst: Vec<u8> = vec![];
        for i in 0..100 {
            dst.clear();
            let k = make_buf("key", i);
            let v = make_buf("value", i);
            c.set(&k, &v);
            c.get(&k, &mut dst);

            assert_eq!(
                dst,
                v,
                "unexpected value for key {}; got {}; want {}",
                String::from_utf8_lossy(&k),
                String::from_utf8_lossy(&v),
                String::from_utf8_lossy(&dst)
            );

            c.del(&k);
            dst.clear();
            assert!(
                !c.get(&k, &mut dst) && dst.is_empty(),
                "unexpected non-empty value got for key {}: {}",
                String::from_utf8_lossy(&k),
                String::from_utf8_lossy(&dst)
            )
        }
    }

    #[test]
    fn test_cache_big_key_value() {
        let c = FastCache::new(BIG_CACHE_SIZE_MIN);

        // Both key and value exceed 64Kb
        let k = [0_u8; 90 * 1024];
        let v = [0_u8; 100 * 1024];
        c.set(&k, &v);

        let mut dst: Vec<u8> = Vec::with_capacity(100 * 1024);
        c.get(&k, &mut dst);
        assert!(
            dst.is_empty(),
            "unexpected non-empty value for key {}: {}",
            String::from_utf8_lossy(&k),
            String::from_utf8_lossy(&dst)
        );

        // len(key) + value.len() > 64Kb
        let k = [0_u8; 40 * 1024];
        let v = [0_u8; 40 * 1024];

        dst.clear();
        c.set(&k, &v);
        c.get(&k, &mut dst);
        assert!(
            dst.is_empty(),
            "unexpected non-empty value got for key {}: {}",
            String::from_utf8_lossy(&k),
            String::from_utf8_lossy(&dst)
        );
    }

    #[test]
    fn test_cache_set_get_serial() {
        let items_count = 10000;
        let mut c = FastCache::new(30 * items_count);
        test_cache_get_set(&mut c, items_count)
    }

    #[test]
    fn test_cache_get_set_concurrent() {
        const ITEMS_COUNT: usize = 10000;
        const GOROUTINES: usize = 10;

        let mut c = FastCache::new(30 * ITEMS_COUNT * GOROUTINES);

        // todo: Rayon
        (0..GOROUTINES).for_each(|_| test_cache_get_set(&mut c, ITEMS_COUNT))
    }

    fn test_cache_get_set(c: &mut FastCache, items_count: usize) {
        let mut vv: Vec<u8> = vec![];
        for i in 0..items_count {
            let k = make_buf("key", i as u64);
            let v = make_buf("value", i as u64);
            c.set(&k, &v);
            vv.clear();

            c.get(&k, &mut vv);
            assert_eq!(
                vv,
                v,
                "unexpected value for key {} after insertion; got {}; want {}",
                String::from_utf8_lossy(&k),
                String::from_utf8_lossy(&vv),
                String::from_utf8_lossy(&v)
            );
        }
        let mut misses = 0;
        for i in 0..items_count {
            let k = make_buf("key", i as u64);
            let v_expected = make_buf("value", i as u64);
            c.get(&k, &mut vv);
            if vv != v_expected {
                assert!(
                    !vv.is_empty(),
                    "unexpected value for key {} after all insertions; got {}; want {}",
                    String::from_utf8_lossy(&k),
                    String::from_utf8_lossy(&vv),
                    String::from_utf8_lossy(&v_expected)
                );
                misses += 1;
            }
        }

        assert!(
            misses >= items_count / 100,
            "too many cache misses; got {}; want less than {}",
            misses,
            items_count / 100
        );
    }

    const VALUES_COUNT: usize = 10;

    #[test_case(1)]
    #[test_case(100)]
    #[test_case(65535)]
    #[test_case(65536)]
    #[test_case(65537)]
    #[test_case(131071)]
    #[test_case(131072)]
    #[test_case(131073)]
    #[test_case(524288)]
    fn test_values_get_set_big(value_size: usize) {
        let mut c = FastCache::new(256 * 1024 * 1024);
        for seed in 0..3 {
            test_set_get_big(&mut c, value_size, VALUES_COUNT, seed)
        }
    }

    #[test]
    fn test_values_one() {
        let mut c = FastCache::new(256 * 1024 * 1024);
        test_set_get_big(&mut c, 131072, 1, 0)
    }

    fn test_set_get_big(c: &mut FastCache, value_size: usize, values_count: usize, seed: u8) {
        let mut m: HashMap<String, Vec<u8>> = HashMap::new();

        let mut buf: Vec<u8> = vec![];
        for i in 0..values_count {
            let key = make_buf("key", i as u64);
            let value = create_value(value_size, seed);
            let key1 = key.clone();
            c.set_big(&key, &value);
            m.insert(
                String::from_utf8(key).expect("invalid utf8 sequence"),
                value.clone(),
            );

            buf.clear();
            let found = c.get_big(&key1, &mut buf);
            assert!(found && buf == value,
                    "found={found} seed={seed}; unexpected value obtained for key={}; got value.len()={}; want value.len()={}",
                    String::from_utf8_lossy(&key1), buf.len(), value.len())
        }

        let mut s = Stats::default();
        c.update_stats(&mut s);
        assert!(
            s.set_big_calls >= values_count as u64,
            "expecting set_big_calls >= {}; got {}",
            values_count,
            s.set_big_calls
        );
        assert!(
            s.get_big_calls >= values_count as u64,
            "expecting get_big_calls >= {}; got {}",
            values_count,
            s.get_big_calls
        );

        // Verify that values still exist
        for (key, value) in m.iter() {
            buf.clear();

            let found = c.get_big(key.as_bytes(), &mut buf);

            assert!(found && buf == *value,
                    "found = {found} seed={seed}; unexpected value obtained for key={}; got value.len()={}; want value.len()={}",
                    key, buf.len(), value.len())
        }
    }

    fn create_value(size: usize, seed: u8) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(size);
        for i in 0..size {
            let m = ((i + seed as usize) % 256) as u8;
            buf.push(m)
        }
        buf
    }

    #[test]
    fn test_generation_overflow() {
        // These two keys has to the same bucket (100), so we can push the
        // generations up much faster.  The keys and values are sized so that
        // every time we push them into the cache they will completely fill the
        // bucket
        const KEY1: [u8; 1] = [26_u8];
        const KEY2: [u8; 1] = [8_u8];

        let mut c = FastCache::new(BIG_CACHE_SIZE_MIN); // each bucket has 64 *1024 bytes capacity

        // Initial generation is 1
        test_gen_val(&mut c, 1);

        let big_val1 = [1_u8; (32 * 1024) - (KEY1.len() + 4)];
        let big_val2 = [2_u8; (32 * 1024) - (KEY2.len() + 5)];

        // Do some initial Set/Get demonstrate that this works
        for i in 0..10 {
            c.set(&KEY1, &big_val1);
            c.set(&KEY2, &big_val2);
            get_val(&mut c, &KEY1, &big_val1);
            get_val(&mut c, &KEY2, &big_val2);
            test_gen_val(&mut c, (1 + i) as u64)
        }

        // This is a hack to simulate calling Set 2^24-3 times
        // Actually doing this takes ~24 seconds, making the test slow
        {
            let inner = c.buckets.get(100).unwrap().inner.write();
            inner.unwrap().gen = (1 << 24) - 2;
            // c.buckets[100].gen == 16,777,215
        }

        // Set/Get still works
        c.set(&KEY1, &big_val1);
        c.set(&KEY2, &big_val2);

        get_val(&mut c, &KEY1, &big_val1);
        get_val(&mut c, &KEY2, &big_val2);

        test_gen_val(&mut c, (1 << 24) - 1);

        // After the next Set operations
        // c.buckets[100].gen == 16,777,216

        // This set creates an index where `idx | (b.gen << bucketSizeBits)` == 0
        // The value is in the cache but is unreadable by Get
        c.set(&KEY1, &big_val1);

        // The Set above overflowed the bucket's generation. This means that
        // key2 is still in the cache, but can't get read because key2 has a
        // _very large_ generation value and appears to be from the future
        get_val(&mut c, &KEY2, &big_val2);

        // This Set creates an index where `(b.gen << bucketSizeBits)>>bucketSizeBits==0`
        // The value is in the cache but is unreadable by Get
        c.set(&KEY2, &big_val2);

        // Ensure generations are working as we expect
        // NB: Here we skip the 2^24 generation, because the bucket carefully
        // avoids `generation==0`
        test_gen_val(&mut c, (1 << 24) + 1);

        get_val(&mut c, &KEY1, &big_val1);
        get_val(&mut c, &KEY2, &big_val2);

        // Do it a few more times to show that this bucket is now unusable
        for i in 0..10 {
            c.set(&KEY1, &big_val1);
            c.set(&KEY2, &big_val2);
            get_val(&mut c, &KEY1, &big_val1);
            get_val(&mut c, &KEY2, &big_val2);
            test_gen_val(&mut c, (1 << 24) + 2 + i)
        }
    }

    fn get_val(c: &mut FastCache, key: &[u8], expected: &[u8]) {
        let mut dst: Vec<u8> = vec![];
        assert!(
            c.get(key, &mut dst) && dst == expected,
            "Expected value ({}) was not returned from the cache, instead got {}",
            String::from_utf8_lossy(&expected[0..10]),
            String::from_utf8_lossy(&dst)
        );
    }

    fn test_gen_val(c: &mut FastCache, expected: u64) {
        let actual = c.buckets.get(100).unwrap().get_gen();
        // Ensure generations are working as we expect
        assert_eq!(
            actual, expected,
            "Expected generation to be {} found {} instead",
            expected, actual
        )
    }
}
