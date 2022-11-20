use std::borrow::{Cow};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use lru_time_cache::LruCache;

use metricsql::ast::Expression;
use metricsql::optimizer::optimize;
use metricsql::parser::{ParseError, visit_all};

use crate::eval::{create_evaluator, ExprEvaluator};

const PARSE_CACHE_MAX_LEN: usize = 500;

pub struct ParseCacheValue {
    pub expr: Option<Expression>,
    pub evaluator: Option<ExprEvaluator>,
    pub err: Option<ParseError>,
    pub has_subquery: bool
}

pub struct ParseCache {
    requests: AtomicU64,
    misses: AtomicU64,
    lru: Mutex<LruCache<String, Arc<ParseCacheValue>>>, // todo: use parking_lot rwLock
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

    pub fn misses(&self) -> u64 {
        self.misses.fetch_add(0, Ordering::Relaxed)
    }

    pub fn requests(&self) -> u64 {
        self.requests.fetch_add(0, Ordering::Relaxed)
    }

    pub fn clear(&mut self) {
        self.lru.lock().unwrap().clear()
    }

    pub fn parse(&self, q: &str) -> Arc<ParseCacheValue> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        match self.get(q) {
            Some(value) => value,
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                let parsed = Self::parse_internal(q);
                let k = q.to_string();

                let mut lru = self.lru.lock().unwrap();
                let value = lru.insert(k, Arc::new(parsed));
                value.unwrap().clone()
            }
        }
    }

    pub fn get(&self, q: &str) -> Option<Arc<ParseCacheValue>> {
        // use interior mutability here
        let mut lru = self.lru.lock().unwrap();
        match lru.get(q) {
            None => None,
            Some(v) => Some(Arc::clone(v))
        }
    }

    fn parse_internal(q: &str) -> ParseCacheValue {
        match metricsql::parser::parse(q) {
            Ok(expr) => {
                let mut expression = match optimize(&expr) {
                    Cow::Owned(exp) => exp,
                    Cow::Borrowed(exp) => exp.clone()
                };
                adjust_cmp_ops(&mut expression);
                match create_evaluator(&expression) {
                    Ok(evaluator) => {
                        ParseCacheValue {
                            expr: Some(expr.clone()),
                            evaluator: Some(evaluator),
                            err: None,
                            has_subquery: expression.contains_subquery()
                        }
                    },
                    Err(e) => {
                        ParseCacheValue {
                            expr: Some(expr),
                            evaluator: Some(ExprEvaluator::default()),
                            has_subquery: false,
                            err: Some(
                                ParseError::General(format!("Error creating evaluator: {:?}", e))
                            ),
                        }
                    }
                }
            },
            Err(e) => {
                ParseCacheValue {
                    expr: None,
                    evaluator: Some(ExprEvaluator::default()),
                    err: Some(e.clone()),
                    has_subquery: false
                }
            }
        }
    }
}


// todo: put in optimize phase
pub(crate) fn adjust_cmp_ops(e: &mut Expression) {
    visit_all(e, |expr: &mut Expression|
        {
            match expr {
                Expression::BinaryOperator(be) => {
                    let _ = be.adjust_comparison_op();
                },
                _ => {}
            }
        });
}
