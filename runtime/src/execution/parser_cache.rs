use metricsql_parser::parse;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use lru_time_cache::LruCache;

use metricsql_parser::ast::{Expr, Operator};
use metricsql_parser::prelude::{
    adjust_comparison_ops,
    optimize as optimize_expr,
    ParseError
};

const PARSE_CACHE_MAX_LEN: usize = 500;

pub struct ParseCacheValue {
    pub expr: Option<Expr>,
    pub err: Option<ParseError>,
    pub optimized: Option<Expr>,
    pub has_subquery: bool,
    pub sort_results: bool,
}

pub struct ParseCache {
    requests: AtomicU64,
    misses: AtomicU64,
    lru: Mutex<LruCache<String, Arc<ParseCacheValue>>>, // todo:
}

#[derive(PartialEq)]
pub enum ParseCacheResult {
    CacheHit,
    CacheMiss,
}

impl Default for ParseCache {
    fn default() -> Self {
        ParseCache::new(PARSE_CACHE_MAX_LEN)
    }
}

impl ParseCache {
    pub fn new(capacity: usize) -> Self {
        ParseCache {
            requests: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            lru: Mutex::new(LruCache::with_capacity(capacity)),
        }
    }

    pub fn len(&self) -> usize {
        self.lru.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.lru.lock().unwrap().is_empty()
    }

    pub fn misses(&self) -> u64 {
        self.misses.fetch_add(0, Ordering::Relaxed)
    }

    pub fn requests(&self) -> u64 {
        self.requests.fetch_add(0, Ordering::Relaxed)
    }

    pub fn clear(&mut self) {
        self.lru.lock().unwrap().clear()
    }

    pub fn parse(&self, q: &str, optimize: bool) -> (Arc<ParseCacheValue>, ParseCacheResult) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        match self.get(q) {
            Some(value) => (value, ParseCacheResult::CacheHit),
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                let parsed = Self::parse_internal(q, optimize);
                let k = q.to_string();
                let to_insert = Arc::new(parsed);

                let mut lru = self.lru.lock().unwrap();
                lru.insert(k, to_insert.clone());
                (to_insert, ParseCacheResult::CacheMiss)
            }
        }
    }

    pub fn get(&self, q: &str) -> Option<Arc<ParseCacheValue>> {
        // use interior mutability here
        let mut lru = self.lru.lock().unwrap();
        lru.get(q).map(Arc::clone)
    }

    // todo: pass options
    pub(super) fn parse_internal(q: &str, optimize: bool) -> ParseCacheValue {
        match parse(q) {
            Ok(mut expr) => {
                adjust_comparison_ops(&mut expr);
                let has_subquery = expr.contains_subquery();
                let sort_results = should_sort_results(&expr);

                if optimize {
                    match optimize_expr(expr) {
                        Ok(e) => expr = e,
                        Err(e) => {
                            return ParseCacheValue {
                                expr: None,
                                optimized: None,
                                err: Some(ParseError::General(format!(
                                    "Error optimizing expression: {:?}",
                                    e
                                ))),
                                has_subquery: false,
                                sort_results: false,
                            }
                        }
                    }
                };

                ParseCacheValue {
                    expr: Some(expr),
                    optimized: None,
                    err: None,
                    has_subquery,
                    sort_results,
                }
            }
            Err(e) => ParseCacheValue {
                expr: None,
                optimized: None,
                err: Some(e.clone()),
                has_subquery: false,
                sort_results: false,
            },
        }
    }
}

fn should_sort_results(e: &Expr) -> bool {
    match e {
        Expr::Function(fe) => !fe.function.may_sort_results(),
        Expr::Aggregation(ae) => !ae.function.may_sort_results(),
        Expr::BinaryOperator(be) => {
            // Do not sort results for `a or b` in the same way as Prometheus does.
            // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/4763
            be.op != Operator::Or
        }
        _ => true,
    }
}
