use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use get_size::GetSize;
use lru_time_cache::LruCache;
use crate::prelude::{get_optimized_re_match_func, EMPTY_MATCH_COST};
use super::StringMatchHandler;

// todo: read from env
static USE_REGEXP_CACHE: LazyLock<bool> = LazyLock::new(|| false);

const DEFAULT_MAX_SIZE_BYTES: usize = 1024 * 1024 * 1024;
const DEFAULT_CACHE_SIZE: usize = 100;

#[derive(Clone, Debug)]
pub struct RegexpCacheValue {
    pub re_match: StringMatchHandler,
    pub re_cost: usize,
    pub size_bytes: usize,
}

pub struct RegexpCache {
    requests: AtomicU64,
    misses: AtomicU64,
    inner: Mutex<LruCache<String, Arc<RegexpCacheValue>>>,
    max_size_bytes: usize,
}

impl RegexpCache {
    pub fn new(max_size_bytes: usize) -> Self {
        Self {
            requests: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            inner: Mutex::new(LruCache::with_capacity(DEFAULT_CACHE_SIZE)),
            max_size_bytes
        }
    }

    pub fn get(&self, key: &str) -> Option<Arc<RegexpCacheValue>> {
        let mut inner = self.inner.lock().unwrap();
        let item = inner.get(key);
        if item.is_some() {
            self.requests.fetch_add(1, Ordering::Relaxed);
            Some(item?.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn put(&self, key: &str, value: Arc<RegexpCacheValue>) {
        self.inner.lock().unwrap().insert(key.to_string(), value);
    }

    /// returns the number of cached regexps for tag filters.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().is_empty()
    }

    pub fn clear(&self) {
        self.inner.lock().unwrap().clear();
    }

    pub fn remove(&self, key: &str) -> Option<Arc<RegexpCacheValue>> {
        self.inner.lock().unwrap().remove(key)
    }

    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    pub fn requests(&self) -> u64 {
        self.requests.load(Ordering::Relaxed)
    }

    pub fn max_size_bytes(&self) -> usize {
        self.max_size_bytes
    }
}

fn matcher_size_bytes(m: &StringMatchHandler) -> usize {
    use StringMatchHandler::*;
    let base = size_of::<StringMatchHandler>();
    let extra = match m {
        Alternates(alts) => alts.get_size(),
        OrderedAlternates(alts) => {
            alts.get_size()
        },
        And(first, second) => {
            matcher_size_bytes(first) + matcher_size_bytes(second)
        }
        MatchAll | MatchNone | Empty  => 0,
        NotEmpty(m) => 1,
        Literal(s) |
        Contains(s) |
        StartsWith(s) |
        EndsWith(s) => s.get_size(),
        Regex(fr) => fr.get_size(),
        MatchFn(_) => {
            size_of::<fn(&str, &str) -> bool>()
        }
        AnyWithoutNewline => 1,
        EqualsMulti(m) => m.get_size(),
        EqualMultiMap(m) => m.get_size(),
        ContainsMulti(m) => m.get_size(),
        Prefix(p) => size_of::<String>() + p.get_size(),
        Suffix(s) => s.get_size(),
        Or(m) => m.get_size(),
        ZeroOrOneChars(m) => m.get_size()
    };
    base + extra
}

pub fn compile_regexp(expr: &str) -> Result<(StringMatchHandler, usize), String> {
    if *USE_REGEXP_CACHE {
        let cached = get_regexp_from_cache(expr)?;
        Ok((cached.re_match.clone(), cached.re_cost))
    } else {
        let compiled = compile_regexp_ex(expr)?;
        Ok((compiled.re_match, compiled.re_cost))
    }
}

pub fn compile_regexp_anchored(expr: &str) -> Result<(StringMatchHandler, usize), String> {
    // all this is to ensure start and end anchors, avoiding allocation if possible
    let mut has_start_anchor = false;
    let mut has_end_anchor = false;

    if expr.is_empty() {
        return Ok((StringMatchHandler::empty_string_match(), EMPTY_MATCH_COST));
    }

    let mut cursor = expr;
    while let Some(t) = cursor.strip_prefix('^') {
        cursor = t;
        has_start_anchor = true;
    }
    while cursor.ends_with("$") && !cursor.ends_with("\\$") {
        if let Some(t) = cursor.strip_suffix("$") {
            cursor = t;
            has_end_anchor = true;
        } else {
            break;
        }
    }

    if has_start_anchor && has_end_anchor {
        // no need to allocate
        compile_regexp(expr)
    } else {
        let anchored = format!("^{}$", cursor);
        compile_regexp(&anchored)
    }
}

fn compile_regexp_ex(expr: &str) -> Result<RegexpCacheValue, String> {
    let (matcher, cost) =
        get_optimized_re_match_func(expr)
            .map_err(|_| {
                format!("cannot build regexp from {}", expr)
            })?;

    // heuristic for rcv in-memory size
    let size_bytes = matcher_size_bytes(&matcher);

    // Put the re_match in the cache.
    Ok(RegexpCacheValue {
        re_match: matcher,
        re_cost: cost,
        size_bytes,
    })
}

pub fn get_regexp_from_cache(expr: &str) -> Result<Arc<RegexpCacheValue>, String> {
    let cache = get_regexp_cache();
    if let Some(rcv) = cache.get(expr) {
        // Fast path - the regexp found in the cache.
        return Ok(rcv);
    }

    // Put the re_match in the cache.
    let (re_match, re_cost) = compile_regexp_anchored(expr)?;
    // heuristic for rcv in-memory size
    let size_bytes = matcher_size_bytes(&re_match);

    let rcv = RegexpCacheValue {
        re_match,
        re_cost,
        size_bytes,
    };
    let result = Arc::new(rcv);
    cache.put(expr, result.clone());

    Ok(result)
}

const DEFAULT_MAX_REGEXP_CACHE_SIZE: usize = 2048;
const DEFAULT_MAX_PREFIX_CACHE_SIZE: usize = 2048;

fn get_regexp_cache_max_size() -> &'static usize {
    static REGEXP_CACHE_MAX_SIZE: OnceLock<usize> = OnceLock::new();
    REGEXP_CACHE_MAX_SIZE.get_or_init(|| {
        // todo: read value from env
        DEFAULT_MAX_REGEXP_CACHE_SIZE
    })
}

fn get_prefix_cache_max_size() -> &'static usize {
    static REGEXP_CACHE_MAX_SIZE: OnceLock<usize> = OnceLock::new();
    REGEXP_CACHE_MAX_SIZE.get_or_init(|| {
        // todo: read value from env
        DEFAULT_MAX_PREFIX_CACHE_SIZE
    })
}

static REGEX_CACHE: LazyLock<RegexpCache> = LazyLock::new(|| {
    let size = get_regexp_cache_max_size();
    RegexpCache::new(*size)
});

// todo: get from env

pub fn get_regexp_cache() -> &'static RegexpCache {
    &REGEX_CACHE
}
