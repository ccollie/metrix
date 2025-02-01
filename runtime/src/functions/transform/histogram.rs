use std::cell::RefCell;
use std::rc::Rc;

use ahash::{AHashMap, HashMapExt};
use smallvec::smallvec;
use metricsql_common::hash::{IntMap, Signature};
use metricsql_parser::parse_number;

use crate::execution::merge_non_overlapping_timeseries;
use crate::functions::arg_parse::{
    get_float_arg, get_int_arg, get_scalar_arg_as_vec, get_series_arg,
};
use crate::functions::transform::utils::{copy_timeseries, is_inf};
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeError, RuntimeResult};
use crate::types::{FunctionArgs, MetricName, QueryValue, Timeseries};

static ELLIPSIS: &str = "...";
static LE: &str = "le";

pub(crate) fn buckets_limit(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut limit = get_int_arg(&tfa.args, 0)?;

    if limit <= 0 {
        return Ok(vec![]);
    }
    if limit < 3 {
        // Preserve the first and the last bucket for better accuracy for min and max values.
        limit = 3
    }
    let series = get_series_arg(&tfa.args, 1, tfa.ec)?;
    let tss = vmrange_buckets_to_le(series);
    let tss_len = tss.len();

    if tss_len == 0 {
        return Ok(vec![]);
    }

    let points_count = tss[0].values.len();

    // Group timeseries by all MetricGroup+tags excluding `le` tag.
    struct Bucket {
        le: f64,
        hits: f64,
        ts: Timeseries,
    }

    let mut bucket_map: IntMap<Signature, Vec<Bucket>> = IntMap::new();

    let mut mn: MetricName = MetricName::default();
    let empty_str = "".to_string();

    for ts in tss.into_iter() {
        let le_str = ts.metric_name.label_value(LE).unwrap_or(&empty_str);

        // Skip time series without `le` tag.
        if le_str.is_empty() {
            continue;
        }

        if let Ok(le) = le_str.parse::<f64>() {
            mn.copy_from(&ts.metric_name);
            mn.remove_label(LE);

            let key = mn.signature();

            bucket_map
                .entry(key)
                .or_default()
                .push(Bucket { le, hits: 0.0, ts });
        } else {
            // Skip time series with invalid `le` tag.
            continue;
        }
    }

    // Remove buckets with the smallest counters.
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tss_len);
    for le_group in bucket_map.values_mut() {
        if le_group.len() <= limit as usize {
            // Fast path - the number of buckets doesn't exceed the given limit.
            // Keep all the buckets as is.

            // To properly remove items by index, we need to do it in reverse order.
            let series = le_group
                .iter_mut()
                .map(|x| std::mem::take(&mut x.ts))
                .collect::<Vec<_>>();

            rvs.extend(series);
            continue;
        }
        // Slow path - remove buckets with the smallest number of hits until their count reaches the limit.

        // Calculate per-bucket hits.
        le_group.sort_by(|a, b| a.le.total_cmp(&b.le));
        for n in 0..points_count {
            let mut prev_value: f64 = 0.0;
            for bucket in le_group.iter_mut() {
                let value = bucket.ts.values[n];
                bucket.hits += value - prev_value;
                prev_value = value
            }
        }

        while le_group.len() > limit as usize {
            // Preserve the first and the last bucket for better accuracy for min and max values
            let mut xx_min_idx = 1;
            let mut min_merge_hits = le_group[1].hits + le_group[2].hits;
            for i in 0..le_group[1..le_group.len() - 2].len() {
                let merge_hits = le_group[i + 1].hits + le_group[i + 2].hits;
                if merge_hits < min_merge_hits {
                    xx_min_idx = i + 1;
                    min_merge_hits = merge_hits
                }
            }
            le_group[xx_min_idx + 1].hits += le_group[xx_min_idx].hits;
            // remove item at xx_min_idx ?
            // leGroup = append(leGroup[: xx_min_idx], leGroup[xx_min_idx + 1: ]...)
            le_group.remove(xx_min_idx);
        }

        let ts_iter = le_group
            .iter_mut()
            .map(|bucket| std::mem::take(&mut bucket.ts))
            .collect::<Vec<Timeseries>>();

        rvs.extend(ts_iter);
    }

    Ok(rvs)
}

pub(crate) fn prometheus_buckets(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    let rvs = vmrange_buckets_to_le(series);
    Ok(rvs)
}

type SharedTimeseries = Rc<RefCell<Timeseries>>;

/// Group timeseries by MetricGroup+tags excluding `vmrange` tag.
#[derive(Clone, Default)]
struct Bucket {
    start_str: String,
    end_str: String,
    start: f64,
    end: f64,
    ts: SharedTimeseries,
}

impl Bucket {
    fn is_set(&self) -> bool {
        !self.start_str.is_empty()
            || !self.end_str.is_empty() && self.start != 0.0 && self.end != 0.0
    }

    fn is_zero_ts(&self) -> bool {
        let ts = self.ts.borrow();
        ts.values.iter().all(|x| *x <= 0.0)
    }

    /// Convert `vmrange` label in each group of time series to `le` label.
    fn copy_ts(&self, le_str: &str) -> SharedTimeseries {
        let src = self.ts.borrow();
        let mut ts: Timeseries = src.clone();
        ts.values.fill(0.0);
        ts.metric_name.set(LE, le_str);
        Rc::new(RefCell::new(ts))
    }

    fn set_le(&mut self, end_str: &str) {
        let mut ts = self.ts.borrow_mut();
        ts.metric_name.set(LE, end_str);
    }

    pub fn pop_timeseries(&mut self) -> Option<Timeseries> {
        match self.ts.try_borrow_mut() {
            Ok(refcell) => {
                // you can do refcell.into_inner here
                Some(refcell.to_owned())
            }
            Err(_) => {
                // another Rc still exists, so you cannot take ownership of the value
                debug_assert!(false, "Cannot borrow mutably");
                None
            }
        }
    }
}

pub(crate) fn vmrange_buckets_to_le(tss: Vec<Timeseries>) -> Vec<Timeseries> {
    if tss.is_empty() {
        return tss;
    }
    
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tss.len());

    let mut buckets: IntMap<Signature, Vec<Bucket>> = IntMap::with_capacity(tss.len());

    let empty_str = "".to_string();
    let values_count = tss[0].values.len();

    for ts in tss.into_iter() {
        let vm_range = ts.metric_name.label_value("vmrange").unwrap_or(&empty_str);

        if vm_range.is_empty() {
            if let Some(le) = ts.metric_name.label_value(LE) {
                if !le.is_empty() {
                    // Keep Prometheus-compatible buckets.
                    rvs.push(ts);
                }
            }
            continue;
        }

        let n = match vm_range.find(ELLIPSIS) {
            Some(pos) => pos,
            None => continue,
        };

        let start_str = &vm_range[0..n];
        let start = match start_str.parse::<f64>() {
            Err(_) => continue,
            Ok(n) => n,
        };

        let end_str = &vm_range[(n + ELLIPSIS.len())..vm_range.len()];
        let end = match end_str.parse::<f64>() {
            Err(_) => continue,
            Ok(n) => n,
        };

        // prevent borrow of value from ts.metric_name
        let start_string = start_str.to_string();
        let end_string = end_str.to_string();

        let mut _ts = ts;
        _ts.metric_name.remove_label(LE);
        _ts.metric_name.remove_label("vmrange");

        let key = _ts.metric_name.signature();
        // series.push(_ts);
        let shared_ts = Rc::new(RefCell::new(_ts));

        buckets.entry(key).or_default().push(Bucket {
            start_str: start_string,
            end_str: end_string,
            start,
            end,
            ts: shared_ts,
        });
    }

    let default_bucket: Bucket = Default::default();

    let mut uniq_ts: AHashMap<String, SharedTimeseries> = AHashMap::with_capacity(8);

    for xss in buckets.values_mut() {
        xss.sort_by(|a, b| a.end.total_cmp(&b.end));
        let mut xss_new: Vec<Bucket> = Vec::with_capacity(xss.len() + 2);
        let mut xs_prev: &Bucket = &default_bucket;

        uniq_ts.clear();

        for xs in xss.iter_mut() {
            if xs.is_zero_ts() {
                // Skip time series with zeros. They are substituted by xss_new below.
                // Skip buckets with zero values - they will be merged into a single bucket
                // when the next non-zero bucket appears.

                // Do not store xs in xsPrev in order to properly create `le` time series
                // for zero buckets.
                // See https://github.com/VictoriaMetrics/VictoriaMetrics/pull/4021
                continue;
            }

            if xs.start != xs_prev.end {
                // There is a gap between the previous bucket and the current bucket
                // or the previous bucket is skipped because it was zero.
                // Fill it with a time series with le=xs.start.
                if !uniq_ts.contains_key(&xs.start_str) {
                    let copy = xs.copy_ts(&xs.start_str);

                    uniq_ts.insert(xs.start_str.to_string(), xs.ts.clone());
                    xss_new.push(Bucket {
                        start_str: "".to_string(),
                        start: 0.0,
                        end_str: xs.start_str.clone(),
                        end: xs.start,
                        ts: copy,
                    });
                }
            }

            // ugly, but otherwise we get a borrow error if we do xs.set_le(&xs.end_str);
            let end_str = xs.end_str.clone();
            // Convert the current time series to a time series with le=xs.end
            xs.set_le(&end_str);

            if let Some(shared_ts) = uniq_ts.get(&end_str) {
                let old_ts: &mut Timeseries = &mut shared_ts.borrow_mut();
                let ts = xs.ts.borrow();
                merge_non_overlapping_timeseries(old_ts, &ts);
            } else {
                uniq_ts.insert(end_str, xs.ts.clone());
                xss_new.push(xs.clone());
            }

            xs_prev = xs;
        }

        if xs_prev.is_set() && !is_inf(xs_prev.end, 1) && !xs_prev.is_zero_ts() {
            let ts = xs_prev.copy_ts("+Inf");

            xss_new.push(Bucket {
                start_str: "".to_string(),
                end_str: "+Inf".to_string(),
                start: 0.0,
                end: f64::INFINITY,
                ts,
            })
        }

        *xss = xss_new;
        if xss.is_empty() {
            continue;
        }

        for i in 0..values_count {
            let mut count: f64 = 0.0;
            for xs in xss.iter_mut() {
                let mut ts = xs.ts.borrow_mut();
                let v = ts.values[i];
                if v > 0.0 {
                    count += v
                }
                ts.values[i] = count
            }
        }

        for xs in xss.iter_mut() {
            if let Some(ts) = xs.pop_timeseries() {
                rvs.push(ts);
            }
        }
    }

    rvs
}

pub(crate) fn histogram_share(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let les: Vec<f64> = get_scalar_arg_as_vec(&tfa.args, 0, tfa.ec)?;

    // Convert buckets with `vmrange` labels to buckets with `le` labels.
    let series = get_series_arg(&tfa.args, 1, tfa.ec)?;
    let mut tss = vmrange_buckets_to_le(series);

    // Parse bounds_label. See https://github.com/prometheus/prometheus/issues/5706 for details.
    let bounds_label = if tfa.args.len() > 2 {
        tfa.args[2].get_string()?
    } else {
        "".to_string()
    };

    // Group metrics by all tags excluding "le"
    let m = group_le_timeseries(&mut tss);

    // Calculate share for les
    let share = |i: usize, les: &[f64], xss: &mut Vec<LeTimeseries>| -> (f64, f64, f64) {
        let le_req = les[i];
        if le_req.is_nan() || xss.is_empty() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        fix_broken_buckets(i, xss);
        if le_req < 0.0 {
            return (0.0, 0.0, 0.0);
        }
        if is_inf(le_req, 1) {
            return (1.0, 1.0, 1.0);
        }
        let mut v_prev: f64 = 0.0;
        let mut le_prev: f64 = 0.0;

        for xs in xss.iter() {
            let v = xs.ts.values[i];
            let le = xs.le;
            if le_req >= le {
                v_prev = v;
                le_prev = le;
                continue;
            }
            // precondition: le_prev <= le_req < le
            let v_last = xss[xss.len() - 1].ts.values[i];
            let lower = v_prev / v_last;
            if is_inf(le, 1) {
                return (lower, lower, 1.0);
            }
            if le_prev == le_req {
                return (lower, lower, lower);
            }
            let upper = v / v_last;
            let q = lower + (v - v_prev) / v_last * (le_req - le_prev) / (le - le_prev);
            return (q, lower, upper);
        }
        // precondition: le_req > leLast
        (1.0, 1.0, 1.0)
    };

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());
    for (_, mut xss) in m.into_iter() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));

        xss = merge_same_le(&mut xss);

        let mut ts_lower: Timeseries;
        let mut ts_upper: Timeseries;

        if !bounds_label.is_empty() {
            ts_lower = xss[0].ts.clone();
            ts_lower.metric_name.remove_label(&bounds_label);
            ts_lower.metric_name.set(&bounds_label, "lower");

            ts_upper = xss[0].ts.clone();
            ts_upper.metric_name.remove_label(&bounds_label);
            ts_upper.metric_name.set(&bounds_label, "upper")
        } else {
            ts_lower = Timeseries::default();
            ts_upper = Timeseries::default();
        }

        for i in 0..xss[0].ts.values.len() {
            let (q, lower, upper) = share(i, &les, &mut xss);
            xss[0].ts.values[i] = q;
            if !bounds_label.is_empty() {
                ts_lower.values[i] = lower;
                ts_upper.values[i] = upper
            }
        }

        rvs.push(std::mem::take(&mut xss[0].ts));
        if !bounds_label.is_empty() {
            rvs.push(ts_lower);
            rvs.push(ts_upper);
        }
    }

    Ok(rvs)
}

pub(crate) fn histogram_avg(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    let mut tss = vmrange_buckets_to_le(series);
    let mut m = group_le_timeseries(&mut tss);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());

    for (_, xss) in m.iter_mut() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));
        for i in 0..xss[0].ts.values.len() {
            xss[0].ts.values[i] = avg_for_le_timeseries(i, xss)
        }
        rvs.push(std::mem::take(&mut xss[0].ts));
    }
    Ok(rvs)
}

pub(crate) fn histogram_stddev(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    let mut tss = vmrange_buckets_to_le(series);
    let m = group_le_timeseries(&mut tss);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());

    for (_, mut xss) in m.into_iter() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));
        for i in 0..xss[0].ts.values.len() {
            let v = stdvar_for_le_timeseries(i, &xss);
            xss[0].ts.values[i] = v.sqrt();
        }
        rvs.push(std::mem::take(&mut xss[0].ts));
    }
    Ok(rvs)
}

pub(crate) fn histogram_stdvar(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    let mut tss = vmrange_buckets_to_le(series);
    let m = group_le_timeseries(&mut tss);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());
    for (_, mut xss) in m.into_iter() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));
        for i in 0..xss[0].ts.values.len() {
            xss[0].ts.values[i] = stdvar_for_le_timeseries(i, &xss)
        }
        rvs.push(std::mem::take(&mut xss[0].ts));
    }
    Ok(rvs)
}

fn avg_for_le_timeseries(i: usize, xss: &[LeTimeseries]) -> f64 {
    let mut le_prev: f64 = 0.0;
    let mut v_prev: f64 = 0.0;
    let mut sum: f64 = 0.0;
    let mut weight_total: f64 = 0.0;
    for xs in xss {
        if is_inf(xs.le, 0) {
            continue;
        }
        let le = xs.le;
        let n = (le + le_prev) / 2_f64;
        let v = xs.ts.values[i];
        let weight = v - v_prev;
        sum += n * weight;
        weight_total += weight;
        le_prev = le;
        v_prev = v;
    }
    if weight_total == 0.0 {
        return f64::NAN;
    }
    sum / weight_total
}

fn stdvar_for_le_timeseries(i: usize, xss: &[LeTimeseries]) -> f64 {
    let mut le_prev: f64 = 0.0;
    let mut v_prev: f64 = 0.0;
    let mut sum: f64 = 0.0;
    let mut sum2: f64 = 0.0;
    let mut weight_total: f64 = 0.0;
    for xs in xss {
        if is_inf(xs.le, 0) {
            continue;
        }
        let le = xs.le;
        let n = (le + le_prev) / 2.0;
        let v = xs.ts.values[i];
        let weight = v - v_prev;
        sum += n * weight;
        sum2 += n * n * weight;
        weight_total += weight;
        le_prev = le;
        v_prev = v
    }
    if weight_total == 0.0 {
        return f64::NAN;
    }
    let avg = sum / weight_total;
    let avg2 = sum2 / weight_total;
    let mut stdvar = avg2 - avg * avg;
    if stdvar < 0.0 {
        // Correct possible calculation error.
        stdvar = 0.0
    }
    stdvar
}

pub(crate) fn histogram_quantiles(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let dst_label = tfa.args[0].get_string()?;

    let len = tfa.args.len();
    let tss_orig = tfa.args[len - 1].as_instant_vec(tfa.ec)?;
    // Calculate quantile individually per each phi.
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tfa.args.len());

    let mut tfa_tmp = TransformFuncArg {
        ec: tfa.ec,
        args: FunctionArgs::new(),
        fe: tfa.fe,
    };

    for i in 1..len - 1 {
        let phi_arg = get_float_arg(&tfa.args, i, Some(0_f64))?;
        if !(0.0..=1.0).contains(&phi_arg) {
            let msg = "got unexpected phi arg. It should contain only numbers in the range [0..1]";
            return Err(RuntimeError::ArgumentError(msg.to_string()));
        }
        let phi_str = phi_arg.to_string();
        let tss = copy_timeseries(&tss_orig);

        tfa_tmp.args = smallvec![QueryValue::Scalar(phi_arg), QueryValue::InstantVector(tss)];

        match histogram_quantile(&mut tfa_tmp) {
            Err(e) => {
                let msg = format!("cannot calculate quantile {}: {:?}", phi_str, e);
                return Err(RuntimeError::General(msg));
            }
            Ok(mut tss_tmp) => {
                for ts in tss_tmp.iter_mut() {
                    // ts.metric_name.remove_tag(&dst_label);
                    ts.metric_name.set(&dst_label, &phi_str);
                }
                rvs.extend(tss_tmp)
            }
        }
    }

    Ok(rvs)
}

pub(crate) fn histogram_quantile(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    // Convert buckets with `vmrange` labels to buckets with `le` labels.
    let series = get_series_arg(&tfa.args, 1, tfa.ec)?;
    if series.is_empty() {
        return Ok(Vec::new());    
    }
    
    let mut tss = vmrange_buckets_to_le(series);
    
    let phis: Vec<f64> = get_scalar_arg_as_vec(&tfa.args, 0, tfa.ec)?;

    // Parse bounds_label. See https://github.com/prometheus/prometheus/issues/5706 for details.
    let bounds_label = if tfa.args.len() > 2 {
        match tfa.args[2].get_string() {
            Ok(s) => s,
            Err(err) => {
                let msg = format!("cannot parse boundsLabel (arg #3): {:?}", err);
                return Err(RuntimeError::ArgumentError(msg));
            }
        }
    } else {
        "".to_string()
    };

    // Group metrics by all tags excluding "le"
    let mut m = group_le_timeseries(&mut tss);

    // Calculate quantile for each group in m
    let last_non_inf = |_i: usize, xss: &[LeTimeseries]| -> f64 {
        if let Some(v) = xss.iter().rev().find(|x| x.le.is_finite()) {
            v.le
        } else {
            f64::NAN
        }
    };

    let quantile = |i: usize, phis: &[f64], xss: &mut Vec<LeTimeseries>| -> (f64, f64, f64) {
        let phi = phis[i];
        if phi.is_nan() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        fix_broken_buckets(i, xss);
        let mut v_last: f64 = 0.0;
        if !xss.is_empty() {
            v_last = xss[xss.len() - 1].ts.values[i]
        }
        if v_last == 0.0 {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        if phi < 0.0 {
            return (f64::NEG_INFINITY, f64::NEG_INFINITY, xss[0].ts.values[i]);
        }
        if phi > 1.0 {
            return (f64::INFINITY, v_last, f64::INFINITY);
        }
        let v_req = v_last * phi;
        let mut v_prev: f64 = 0.0;
        let mut le_prev: f64 = 0.0;
        for xs in xss.iter() {
            let v = xs.ts.values[i];
            let le = xs.le;
            if v <= 0.0 {
                // Skip zero buckets.
                le_prev = le;
                continue;
            }
            if v < v_req {
                v_prev = v;
                le_prev = le;
                continue;
            }
            if is_inf(le, 0) {
                break;
            }
            if v == v_prev {
                return (le_prev, le_prev, v);
            }
            let vv = le_prev + (le - le_prev) * (v_req - v_prev) / (v - v_prev);
            return (vv, le_prev, le);
        }
        let vv = last_non_inf(i, xss);
        (vv, vv, f64::INFINITY)
    };

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());
    for (_, xss) in m.iter_mut() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));

        let mut xss = merge_same_le(xss);

        let mut ts_lower: Timeseries;
        let mut ts_upper: Timeseries;

        if !bounds_label.is_empty() {
            ts_lower = xss[0].ts.clone(); // todo: use take and clone instead of 2 clones ?
            ts_lower.metric_name.set(&bounds_label, "lower");

            ts_upper = xss[0].ts.clone();
            ts_upper.metric_name.set(&bounds_label, "upper");
        } else {
            ts_lower = Timeseries::default();
            ts_upper = Timeseries::default();
        }

        for i in 0..xss[0].ts.values.len() {
            let (v, lower, upper) = quantile(i, &phis, &mut xss);
            xss[0].ts.values[i] = v;
            if !bounds_label.is_empty() {
                ts_lower.values[i] = lower;
                ts_upper.values[i] = upper;
            }
        }

        let mut dst: LeTimeseries = if xss.len() == 1 {
            xss.remove(0)
        } else {
            xss.swap_remove(0)
        };

        rvs.push(std::mem::take(&mut dst.ts));
        if !bounds_label.is_empty() {
            rvs.push(ts_lower);
            rvs.push(ts_upper);
        }
    }

    Ok(rvs)
}

#[derive(Default)]
pub(super) struct LeTimeseries {
    pub le: f64,
    pub ts: Timeseries,
}

fn group_le_timeseries(tss: &mut [Timeseries]) -> IntMap<Signature, Vec<LeTimeseries>> {
    let mut m: IntMap<Signature, Vec<LeTimeseries>> = IntMap::new();

    for ts in tss.iter_mut() {
        if let Some(tag_value) = ts.metric_name.label_value(LE) {
            if tag_value.is_empty() {
                continue;
            }

            if let Ok(le) = parse_number(tag_value) {
                ts.metric_name.reset_measurement();
                ts.metric_name.remove_label("le");
                let key = ts.metric_name.signature();

                m.entry(key).or_default().push(LeTimeseries {
                    le,
                    ts: std::mem::take(ts),
                });
            }
        }
    }

    m
}

pub(super) fn fix_broken_buckets(i: usize, xss: &mut [LeTimeseries]) {
    // Buckets are already sorted by le, so their values must be in ascending order,
    // since the next bucket includes all the previous buckets.
    // If the next bucket has lower value than the current bucket,
    // then the current bucket must be substituted with the next bucket value.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2819
    if xss.len() < 2 {
        return;
    }

    // Substitute upper bucket values with lower bucket values if the upper values are NaN
    // or are bigger than the lower bucket values.
    let mut v_next = xss[0].ts.values[i]; // todo: check i to avoid panic
    for lts in xss.iter_mut().skip(1) {
        if let Some(v) = lts.ts.values.get_mut(i) {
            if v.is_nan() || v_next > *v {
                *v = v_next;
            } else {
                v_next = *v;
            }
        }
    }
}

fn merge_same_le(xss: &mut [LeTimeseries]) -> Vec<LeTimeseries> {
    // Merge buckets with identical le values.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/pull/3225
    let mut prev_le = xss[0].le;
    let mut dst = Vec::with_capacity(xss.len());
    let mut iter = xss.iter_mut();
    let first = iter.next();
    if first.is_none() {
        return dst;
    }
    dst.push(std::mem::take(first.unwrap()));
    let mut dst_index = 0;

    for xs in iter {
        if xs.le != prev_le {
            prev_le = xs.le;
            dst.push(std::mem::take(xs));
            dst_index = dst.len() - 1;
            continue;
        }

        if let Some(dst) = dst.get_mut(dst_index) {
            for (v, dst_val) in xs.ts.values.iter().zip(dst.ts.values.iter_mut()) {
                *dst_val += v;
            }
        }
    }
    dst
}
