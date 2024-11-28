use std::fmt;
use std::sync::OnceLock;

use lockfree_object_pool::{LinearObjectPool, LinearReusable};

const E10MIN: i32 = -9;
const E10MAX: i32 = 18;
const BUCKETS_PER_DECIMAL: usize = 18;
const DECIMAL_BUCKETS_COUNT: i32 = E10MAX - E10MIN;
const BUCKETS_COUNT: usize = (DECIMAL_BUCKETS_COUNT * BUCKETS_PER_DECIMAL as i32) as usize;
const LOWER_MIN: f64 = 1e-9;

static LOWER_BUCKET_RANGE: &str = "0...0.000";
static UPPER_BUCKET_RANGE: &str = "1000000000000000000.000...+Inf";


/// `Histogram` is a histogram for non-negative values with automatically created buckets.
///
/// See https://medium.com/@valyala/improving-histogram-usability-for-prometheus-and-grafana-bc7e5df0e350
///
/// Each bucket contains a counter for values in the given range.
/// Each non-empty bucket is exposed via the following metric:
///
///     <metric_name>_bucket{<optional_tags>,vmrange="<start>...<end>"} <counter>
///
/// Where:
///
///     - <metric_name> is the metric name passed to NewHistogram
///     - <optional_tags> is optional tags for the <metric_name>, which are passed to NewHistogram
///     - <start> and <end> - start and end values for the given bucket
///     - <counter> - the number of hits to the given bucket during Update* calls
///
/// Histogram buckets can be converted to Prometheus-like buckets with `le` labels
/// with `prometheus_buckets(<metric_name>_bucket)` function from PromQL extensions in VictoriaMetrics.
/// (see https://github.com/VictoriaMetrics/VictoriaMetrics/wiki/MetricsQL ):
///
///     prometheus_buckets(request_duration_bucket)
///
/// Time series produced by the Histogram have better compression ratio comparing to
/// Prometheus histogram buckets with `le` labels, since they don't include counters
/// for all the previous buckets.
///
/// Zero histogram is usable.
#[derive(Clone, Debug, Default)]
pub struct Histogram {
    count: u64,
    lower: u64,
    upper: u64,
    sum: f64,
    decimal_buckets: Vec<[u64; BUCKETS_PER_DECIMAL]>,
    values: Vec<f64>,
}

impl Histogram {
    // returns new fast histogram.
    pub fn new() -> Self {
        Self::with_capacity(DECIMAL_BUCKETS_COUNT as usize)
    }

    pub fn with_capacity(n: usize) -> Self {
        let mut buckets: Vec<[u64; BUCKETS_PER_DECIMAL]> = Vec::with_capacity(n);
        buckets.resize_with(n, || [0u64; BUCKETS_PER_DECIMAL]);
        Histogram {
            count: 0,
            lower: 0,
            upper: 0,
            sum: 0.0,
            decimal_buckets: buckets,
            values: Vec::with_capacity(16),
        }
        // reset
    }

    // Reset resets the histogram.
    pub fn reset(&mut self) {
        for bucket in self.decimal_buckets.iter_mut() {
            bucket.fill(0);
        }
        self.count = 0;
        self.lower = 0;
        self.upper = 0;
        self.values.clear();
    }

    // Update updates h with v.
    //
    // Negative values and NaNs are ignored.
    pub fn update(&mut self, v: f64) {
        if v.is_nan() || v < 0.0 {
            // Skip NaNs and negative values.
            return;
        }
        self.count += 1;
        let bucket_idx = (v.log10() - E10MIN as f64) * BUCKETS_PER_DECIMAL as f64;
        self.sum += v;
        self.values.push(v);
        if bucket_idx < 0_f64 {
            self.lower += 1;
        } else if bucket_idx >= BUCKETS_COUNT as f64 {
            self.upper += 1;
        } else {
            let mut idx = bucket_idx.floor() as usize;
            if bucket_idx == idx as f64 {
                // Edge case for 10^n values, which must go to the lower bucket
                // according to Prometheus logic for `le`-based histograms.
                idx -= 1;
            }
            let decimal_bucket_idx = idx / BUCKETS_PER_DECIMAL;
            let offset = idx % BUCKETS_PER_DECIMAL;
            if self.decimal_buckets.len() <= decimal_bucket_idx {
                self.decimal_buckets
                    .resize_with(decimal_bucket_idx + 1, || [0u64; BUCKETS_PER_DECIMAL]);
            }
            let bucket = &mut self.decimal_buckets[decimal_bucket_idx];
            bucket[offset] += 1;
        }
    }

    #[allow(dead_code)]
    pub fn marshal_to(&self, prefix: &str, dst: &mut String) {
        let mut count_total = 0;

        for item in self.non_zero_buckets() {
            let count = item.count;
            let tag = format!("vmrange={}", item.vm_range);
            let metric_name = add_tag(prefix, tag);
            let (name, labels) = split_metric_name(&metric_name);
            dst.push_str(format!("{}_bucket${} {}\n", name, labels, count).as_str());
            count_total += count;
        }

        if count_total == 0 {
            return;
        }
        let (name, labels) = split_metric_name(prefix);
        let sum = self.sum;
        if sum.floor() == sum {
            let msg = format!("{}_sum{} {}\n", name, labels, sum.floor());
            dst.push_str(&msg);
        } else {
            let msg = format!("{}_sum{} {}\n", name, labels, sum);
            dst.push_str(&msg);
        }
        dst.push_str(&format!("{}_count{} {}\n", name, labels, count_total));
    }

    /// Get an iterator over this histogram's buckets.
    pub fn non_zero_buckets(&self) -> NonZeroBuckets {
        NonZeroBuckets {
            histogram: self,
            index: 0,
            offset: 0,
            lower_handled: false,
        }
    }

    /// visit_non_zero_buckets calls f for all buckets with non-zero counters.
    ///
    /// vmrange contains "<start>...<end>" string with bucket bounds. The lower bound
    /// isn't included in the bucket, while the upper bound is included.
    /// This is required to be compatible with Prometheus-style histogram buckets
    /// with `le` (less or equal) labels.
    #[allow(unused)]
    pub fn visit_non_zero_buckets<'a, F, C>(&self, context: &mut C, f: F)
    where
        F: Fn(&'a str, u64, &mut C),
    {
        if self.lower > 0 {
            f(LOWER_BUCKET_RANGE, self.lower, context)
        }

        let ranges = get_bucket_ranges();

        for (decimal_bucket_idx, db) in self.decimal_buckets.iter().enumerate() {
            for (offset, count) in db.iter().enumerate() {
                if *count > 0u64 {
                    let bucket_idx = decimal_bucket_idx * BUCKETS_PER_DECIMAL + offset;
                    let vmrange = &ranges[bucket_idx];
                    f(vmrange, *count, context)
                }
            }
        }

        if self.upper > 0 {
            f(UPPER_BUCKET_RANGE, self.upper, context)
        }
    }
}

/// Iterator return type.
#[derive(Clone)]
pub struct NonZeroBucket<'a> {
    pub vm_range: &'a str,
    /// The number of samples in this bucket's range.
    pub count: u64,
}

impl<'a> fmt::Debug for NonZeroBucket<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Bucket {{ {}..{} }}", self.vm_range, self.count)
    }
}

/// An iterator over the non-zero buckets in a histogram.
#[derive(Debug, Clone)]
pub struct NonZeroBuckets<'a> {
    histogram: &'a Histogram,
    index: usize,
    offset: usize,
    lower_handled: bool,
}

impl<'a> Iterator for NonZeroBuckets<'a> {
    type Item = NonZeroBucket<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.lower_handled {
            self.lower_handled = true;
            if self.histogram.lower > 0 {
                return Some(NonZeroBucket {
                    vm_range: LOWER_BUCKET_RANGE,
                    count: self.histogram.lower,
                });
            }
        }

        let buckets = &self.histogram.decimal_buckets;

        if self.index >= buckets.len() {
            if self.histogram.upper > 0 {
                return Some(NonZeroBucket {
                    vm_range: UPPER_BUCKET_RANGE,
                    count: self.histogram.upper,
                });
            }
            return None;
        }

        loop {
            let bucket = &buckets[self.index];
            while self.offset < BUCKETS_PER_DECIMAL && bucket[self.offset] == 0 {
                self.offset += 1;
            }

            if self.offset >= bucket.len() {
                self.index += 1;
                self.offset = 0;
                if self.index >= buckets.len() {
                    if self.histogram.upper > 0 {
                        break Some(NonZeroBucket {
                            vm_range: UPPER_BUCKET_RANGE,
                            count: self.histogram.upper,
                        });
                    }
                    break None;
                }
                continue;
            }

            let bucket_idx = self.index * BUCKETS_PER_DECIMAL + self.offset;
            let ranges = get_bucket_ranges();
            let vm_range = &ranges[bucket_idx];
            let count = bucket[self.offset];
            self.offset += 1;

            break Some(NonZeroBucket { vm_range, count });
        }
    }
}

#[inline]
fn format_float(v: f64) -> String {
    format!("{:.3e}", v)
}

static BUCKET_RANGES: OnceLock<[String; BUCKETS_COUNT]> = OnceLock::new();

fn create_bucket_ranges() -> [String; BUCKETS_COUNT] {
    let bucket_multiplier: f64 = 10_f64.powf(1.0 / BUCKETS_PER_DECIMAL as f64);
    let mut ranges: Vec<String> = Vec::with_capacity(BUCKETS_COUNT);
    let mut v: f64 = LOWER_MIN;
    let mut start = format_float(v);

    for _ in 0..BUCKETS_COUNT {
        v *= bucket_multiplier;
        let end = format_float(v);
        ranges.push(format!("{}...{}", start, end).to_string());
        start = end;
    }
    ranges.try_into().unwrap_or_else(|v: Vec<String>| {
        panic!(
            "Expected a Vec of length {BUCKETS_COUNT} but it was {}",
            v.len()
        )
    })
}

fn get_bucket_ranges() -> &'static [String; BUCKETS_COUNT] {
    BUCKET_RANGES.get_or_init(create_bucket_ranges)
}

// todo: move to utils ?
// todo - return slices instead
#[allow(dead_code)]
fn split_metric_name(name: &str) -> (&str, &str) {
    if let Some(index) = name.find('{') {
        if let Some(tailing_index) = name.find('}') {
            let metric_name = &name[1..index];
            let labels = &name[index + 1..tailing_index];
            return (metric_name, labels);
        }
    }
    (name, "")
}

fn add_tag(name: &str, tag: String) -> String {
    let need_braces = if name.is_empty() {
        true
    } else {
        let last_ch = name.chars().last().unwrap();
        last_ch == '}'
    };

    if need_braces {
        return format!("{}{{{}}}", name, tag);
    }
    let mut chars = name.chars();
    chars.next_back();
    format!("{},{}}}", chars.as_str(), tag)
}

fn get_pool() -> &'static LinearObjectPool<Histogram> {
    static INSTANCE: OnceLock<LinearObjectPool<Histogram>> = OnceLock::new();
    INSTANCE.get_or_init(|| LinearObjectPool::<Histogram>::new(Histogram::new, |v| v.reset()))
}

pub fn get_pooled_histogram() -> LinearReusable<'static, Histogram> {
    get_pool().pull()
}
