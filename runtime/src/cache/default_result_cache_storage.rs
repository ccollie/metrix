use metricsql_common::cache::FastCache;

use crate::cache::RollupResultCacheStorage;

pub struct DefaultResultCacheStorage {
    cache: FastCache,
}

impl DefaultResultCacheStorage {
    pub fn new(cache_size: usize) -> Self {
        let cache = FastCache::new(cache_size);
        Self { cache }
    }
}

impl RollupResultCacheStorage for DefaultResultCacheStorage {
    fn get(&mut self, k: &[u8], dst: &mut Vec<u8>) -> bool {
        self.cache.get(k, dst)
    }

    fn get_big(&mut self, k: &[u8], dst: &mut Vec<u8>) -> bool {
        self.cache.get_big(k, dst)
    }

    fn set(&mut self, k: &[u8], v: &[u8]) {
        self.cache.set(k, v)
    }

    fn set_big(&mut self, k: &[u8], v: &[u8]) {
        self.cache.set_big(k, v)
    }

    fn clear(&mut self) {
        self.cache.reset()
    }

    fn reset(&mut self) {
        self.cache.reset()
    }

    fn len(&self) -> usize {
        todo!()
    }
}
