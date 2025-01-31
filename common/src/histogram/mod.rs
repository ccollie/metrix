use std::sync::{LazyLock, RwLock};

const E10_MIN: i32 = -9;
const E10_MAX: i32 = 18;
const BUCKETS_PER_DECIMAL: usize = 18;
const DECIMAL_BUCKETS_COUNT: usize = (E10_MAX - E10_MIN) as usize;
const BUCKETS_COUNT: usize = DECIMAL_BUCKETS_COUNT * BUCKETS_PER_DECIMAL;

static BUCKET_RANGES: LazyLock<Box<[String; BUCKETS_COUNT]>> = LazyLock::new(init_bucket_ranges);
static BUCKET_MULTIPLIER: LazyLock<f64> =
    LazyLock::new(|| 10_f64.powf(1.0 / BUCKETS_PER_DECIMAL as f64));
static UPPER_BUCKET_RANGE: LazyLock<String> =
    LazyLock::new(|| format!("{}...+Inf", format_float(10_f64.powi(E10_MAX))));
static LOWER_BUCKET_RANGE: LazyLock<String> =
    LazyLock::new(|| format!("0...{}", format_float(10_f64.powi(E10_MIN))));

/// `Histogram` is a histogram for non-negative values with automatically created buckets.
///
/// See https://medium.com/@valyala/improving-histogram-usability-for-prometheus-and-grafana-bc7e5df0e350
///
/// Each bucket contains a counter for values in the given range.
/// Each non-empty bucket is exposed via the following metric:
///
///    `<metric_name>_bucket{<optional_tags>,vmrange="<start>...<end>"} <counter>`
///
/// Where:
///
///   - <metric_name> is the metric name passed to NewHistogram
///   - <optional_tags> is optional tags for the <metric_name>, which are passed to NewHistogram
///   - <start> and <end> - start and end values for the given bucket
///   - <counter> - the number of hits to the given bucket during Update* calls
///
/// Histogram buckets can be converted to Prometheus-like buckets with `le` labels
/// with `prometheus_buckets(<metric_name>_bucket)` function from PromQL extensions in VictoriaMetrics.
/// (see https://docs.victoriametrics.com/metricsql/ ):
///
///    `prometheus_buckets(request_duration_bucket)`
///
/// Time series produced by the Histogram have better compression ratio comparing to
/// Prometheus histogram buckets with `le` labels, since they don't include counters
/// for all the previous buckets.
///
/// Zero histogram is usable.
pub struct Histogram {
    inner: RwLock<Inner>,
}

struct Inner {
    decimal_buckets: [Option<[u64; BUCKETS_PER_DECIMAL]>; DECIMAL_BUCKETS_COUNT],
    lower: u64,
    upper: u64,
    sum: f64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Inner {
                decimal_buckets: [None; DECIMAL_BUCKETS_COUNT],
                lower: 0,
                upper: 0,
                sum: 0.0,
            }),
        }
    }

    pub fn reset(&self) {
        let mut inner = self.inner.write().unwrap();
        for db in inner.decimal_buckets.iter_mut().flatten() {
            for c in db {
                *c = 0;
            }
        }
        inner.lower = 0;
        inner.upper = 0;
        inner.sum = 0.0;
    }

    pub fn update(&self, v: f64) {
        if v.is_nan() || v < 0.0 {
            return;
        }
        let bucket_idx =
            ((v.log10() - E10_MIN as f64) * BUCKETS_PER_DECIMAL as f64).floor() as isize;

        let mut inner = self.inner.write().unwrap();
        inner.sum += v;
        if bucket_idx < 0 {
            inner.lower += 1;
        } else if bucket_idx >= BUCKETS_COUNT as isize {
            inner.upper += 1;
        } else {
            let idx = bucket_idx as usize;
            let decimal_bucket_idx = idx / BUCKETS_PER_DECIMAL;
            let offset = idx % BUCKETS_PER_DECIMAL;
            let db = inner.decimal_buckets[decimal_bucket_idx]
                .get_or_insert([0; BUCKETS_PER_DECIMAL]);
            db[offset] = 1;
        }
    }

    pub fn merge(&self, src: &Histogram) {
        let mut inner = self.inner.write().unwrap();
        let src_inner = src.inner.read().unwrap();
        inner.lower += src_inner.lower;
        inner.upper += src_inner.upper;
        inner.sum += src_inner.sum;

        for (i, db_src) in src_inner.decimal_buckets.iter().enumerate() {
            if let Some(db_src) = db_src {
                let db_dst =
                    inner.decimal_buckets[i].get_or_insert([0; BUCKETS_PER_DECIMAL]);
                for (j, c) in db_src.iter().enumerate() {
                    db_dst[j] = *c;
                }
            }
        }
    }

    pub fn visit_non_zero_buckets<F>(&self, mut f: F)
    where
        F: FnMut(&str, u64),
    {
        let inner = self.inner.read().unwrap();
        if inner.lower > 0 {
            f(LOWER_BUCKET_RANGE.as_str(), inner.lower);
        }
        for (decimal_bucket_idx, db) in inner.decimal_buckets.iter().enumerate() {
            if let Some(db) = db {
                for (offset, count) in db.iter().enumerate() {
                    if *count > 0 {
                        let bucket_idx = decimal_bucket_idx * BUCKETS_PER_DECIMAL + offset;
                        let vm_range = get_vm_range(bucket_idx);
                        f(&vm_range, *count);
                    }
                }
            }
        }
        if inner.upper > 0 {
            f(UPPER_BUCKET_RANGE.as_str(), inner.upper);
        }
    }

    pub fn update_duration(&self, start_time: std::time::Instant) {
        let d = start_time.elapsed().as_secs_f64();
        self.update(d);
    }

    pub fn marshal_to<W: std::io::Write>(&self, prefix: &str, w: &mut W) {
        let mut count_total = 0;
        self.visit_non_zero_buckets(|vmrange, count| {
            let tag = format!("vmrange=\"{}\"", vmrange);
            let metric_name = add_tag(prefix, &tag);
            let (name, labels) = split_metric_name(&metric_name);
            writeln!(w, "{}_bucket{} {}", name, labels, count).unwrap();
            count_total += count;
        });
        if count_total == 0 {
            return;
        }
        let (name, labels) = split_metric_name(prefix);
        let sum = self.get_sum();
        if sum.floor() == sum {
            writeln!(w, "{}_sum{} {}", name, labels, sum as i64).unwrap();
        } else {
            writeln!(w, "{}_sum{} {:.6}", name, labels, sum).unwrap();
        }
        writeln!(w, "{}_count{} {}", name, labels, count_total).unwrap();
    }

    pub fn get_sum(&self) -> f64 {
        let inner = self.inner.read().unwrap();
        inner.sum
    }
}

#[inline]
fn format_float(f: f64) -> String {
    format!("{:.3e}", f)
}

fn init_bucket_ranges() -> Box<[String; BUCKETS_COUNT]> {
    let mut v = 10_f64.powi(E10_MIN);
    let mut start = format_float(v);
    const ARRAY_REPEAT_VALUE: String = String::new();
    let mut ranges = Box::new([ARRAY_REPEAT_VALUE; BUCKETS_COUNT]);
    for i in 0..BUCKETS_COUNT {
        v *= *BUCKET_MULTIPLIER;
        let end = format_float(v);
        ranges[i] = format!("{}...{}", start, end);
        start = end;
    }
    ranges
}

fn get_vm_range(bucket_idx: usize) -> String {
    BUCKET_RANGES[bucket_idx].clone()
}

fn add_tag(prefix: &str, tag: &str) -> String {
    if prefix.contains('{') {
        format!("{},{}", prefix, tag)
    } else {
        format!("{{{},{}}}", prefix, tag)
    }
}

fn split_metric_name(metric_name: &str) -> (&str, &str) {
    if let Some(pos) = metric_name.find('{') {
        (&metric_name[..pos], &metric_name[pos..])
    } else {
        (metric_name, "")
    }
}
