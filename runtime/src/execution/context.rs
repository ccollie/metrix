use std::time::Duration;
use std::sync::Arc;
use tracing::{span_enabled, Level};

use crate::cache::rollup_result_cache::RollupResultCache;
use crate::execution::active_queries::{ActiveQueries, ActiveQueryEntry};
use crate::execution::parser_cache::{ParseCache, ParseCacheResult, ParseCacheValue};
use crate::provider::{Deadline, QueryResults, SearchQuery};
use crate::query_stats::QueryStatsTracker;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::{MetricStorage, NullMetricStorage};

const DEFAULT_MAX_QUERY_LEN: usize = 16 * 1024;
const DEFAULT_MAX_UNIQUE_TIMESERIES: usize = 1000;
const DEFAULT_LATENCY_OFFSET: usize = 30 * 1000;

// todo; should this be a trait ?
pub struct Context {
    pub config: SessionConfig,
    pub parse_cache: ParseCache,
    pub rollup_result_cache: RollupResultCache,
    pub(crate) active_queries: ActiveQueries,
    pub query_stats: QueryStatsTracker,
    pub storage: Arc<dyn MetricStorage>,
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_metric_storage(mut self, storage: Arc<dyn MetricStorage>) -> Self {
        self.storage = storage;
        self
    }
    
    pub fn with_config(mut self, config: SessionConfig) -> Self {
        self.config = config;
        self
    }

    pub fn search(&self, sq: SearchQuery, deadline: Deadline) -> RuntimeResult<QueryResults> {
        use metricsql_common::async_runtime::*;

        let storage = self.storage.clone();
        // todo: use std::time::Duration for deadline
        let timeout_ms = deadline.timeout.as_millis() as u64;
        let duration = Duration::from_millis(timeout_ms);

        fn handle_error(err: Error) -> RuntimeError {
            match err {
                Error::Join(msg) => RuntimeError::General(msg),
                Error::Timeout { .. } => {
                    RuntimeError::DeadlineExceededError("search timeout".to_string())
                }
                Error::Execution { source } => RuntimeError::ExecutionError(source.to_string()),
            }
        }

        let res = block_sync(async move {
            if duration.is_zero() {
                storage.search(sq, deadline).await
            } else {
                let res =
                    timeout(duration, async move { storage.search(sq, deadline).await }).await;
                match res {
                    Ok(res) => res,
                    Err(_elapsed) => {
                        let msg = format!("search timeout after {} ms", timeout_ms);
                        Err(RuntimeError::DeadlineExceededError(msg))
                    }
                }
            }
        });
        res.unwrap_or_else(|e| Err(handle_error(e)))
    }

    // todo: pass in tracer
    pub fn parse_promql(&self, q: &str) -> RuntimeResult<(Arc<ParseCacheValue>, ParseCacheResult)> {
        let optimize = self.config.optimize_queries;
        let (res, cached) = if self.config.disable_cache {
            let val = ParseCache::parse_internal(q, optimize);
            (Arc::new(val), ParseCacheResult::CacheMiss)
        } else {
            self.parse_cache.parse(q, optimize)
        };
        if let Some(err) = &res.err {
            return Err(RuntimeError::ParseError(err.clone()));
        }
        Ok((res, cached))
    }

    #[inline]
    pub fn stats_enabled(&self) -> bool {
        self.config.stats_enabled
    }

    #[inline]
    pub fn trace_enabled(&self) -> bool {
        self.config.trace_enabled && span_enabled!(Level::TRACE)
    }

    pub fn get_active_queries(&self) -> Vec<ActiveQueryEntry> {
        self.active_queries.get_all()
    }
}

impl Default for Context {
    fn default() -> Self {
        Self {
            config: Default::default(),
            parse_cache: Default::default(),
            rollup_result_cache: Default::default(),
            active_queries: ActiveQueries::new(),
            query_stats: Default::default(),
            storage: Arc::new(NullMetricStorage {}),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SessionState {
    pub config: SessionConfig,
}

/// Global configuration options for request context
#[derive(Clone, Debug)]
pub struct SessionConfig {
    /// should we log query stats
    pub stats_enabled: bool,

    /// Whether to disable response caching. This may be useful during data back filling
    pub disable_cache: bool,

    /// Whether query tracing is enabled.
    pub trace_enabled: bool,

    /// The maximum provider query length in bytes
    pub max_query_len: usize,

    /// The time when data points become visible in query results after the collection.
    /// Too small value can result in incomplete last points for query results
    pub latency_offset: Duration,

    /// The maximum amount of memory a single query may consume. Queries requiring more memory are
    /// rejected. The total memory limit for concurrently executed queries can be estimated as
    /// `max_memory_per_query` multiplied by -provider.maxConcurrentQueries
    pub max_memory_per_query: usize,

    /// Set this flag to true if the database doesn't contain Prometheus stale markers, so there is
    /// no need in spending additional CPU time on its handling. Staleness markers may exist only in
    /// data obtained from Prometheus scrape targets
    pub no_stale_markers: bool,

    /// The maximum number of points per series which can be generated by subquery.
    /// See https://valyala.medium.com/prometheus-subqueries-in-victoriametrics-9b1492b720b3
    pub max_points_subquery_per_timeseries: usize,

    /// The maximum interval for staleness calculations. By default, it is automatically calculated from
    /// the median interval between samples. This could be useful for tuning Prometheus data model
    /// closer to Influx-style data model.
    /// See https://prometheus.io/docs/prometheus/latest/querying/basics/#staleness for details.
    /// See also `set_lookback_to_step` flag
    pub max_staleness_interval: Duration,

    /// The minimum interval for staleness calculations. This could be useful for removing gaps on
    /// graphs generated from time series with irregular intervals between samples.
    pub min_staleness_interval: Duration,

    /// The maximum number of unique time series to be returned from instant or range queries
    /// This option allows limiting memory usage
    pub max_response_series: usize,

    /// Synonym to -provider.lookback-delta from Prometheus.
    /// The value is dynamically detected from interval between time series data-points if not set.
    /// It can be overridden on per-query basis via max_lookback arg.
    /// See also `max_staleness_interval` flag, which has the same meaning due to historical reasons
    pub max_lookback: Duration,

    /// Whether to fix lookback interval to `step` query arg value.
    /// If set to true, the query model becomes closer to InfluxDB data model. If set to true,
    /// then `max_lookback` and `max_staleness_interval` are ignored. Defaults to `false`
    pub set_lookback_to_step: bool,

    /// The maximum step when the range query handler adjusts points with timestamps closer than
    /// `latency_offset` to the current time. The adjustment is needed because such points may contain
    /// incomplete data
    pub max_step_for_points_adjustment: Duration,

    /// The maximum duration for query execution (default 30 secs)
    pub max_query_duration: Duration,

    /// Whether to optimize the query before execution
    pub optimize_queries: bool
}

impl SessionConfig {
    /// Create an execution config with default setting
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_cache(mut self, caching: bool) -> Self {
        self.disable_cache = !caching;
        self
    }

    pub fn with_stale_markers(mut self, has_markers: bool) -> Self {
        self.no_stale_markers = !has_markers;
        self
    }

    pub fn with_stats_enabled(mut self, stats_enabled: bool) -> Self {
        self.stats_enabled = stats_enabled;
        self
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        SessionConfig {
            stats_enabled: false,
            disable_cache: false,
            trace_enabled: false,
            max_query_len: DEFAULT_MAX_QUERY_LEN,
            latency_offset: Duration::from_millis(DEFAULT_LATENCY_OFFSET as u64),
            max_memory_per_query: 0,
            no_stale_markers: true,
            max_points_subquery_per_timeseries: 0,
            max_staleness_interval: Duration::ZERO,
            min_staleness_interval: Duration::ZERO,
            max_response_series: DEFAULT_MAX_UNIQUE_TIMESERIES,
            max_lookback: Duration::ZERO,
            set_lookback_to_step: false,
            max_step_for_points_adjustment: Duration::from_millis(1000),
            max_query_duration: Duration::from_secs(30),
            optimize_queries: false,
        }
    }
}
