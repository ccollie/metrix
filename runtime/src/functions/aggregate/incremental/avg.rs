use itertools::izip;

use super::context::{IncrementalAggrContext, IncrementalAggrHandler};

pub struct IncrementalAggrAvg {}

impl IncrementalAggrHandler for IncrementalAggrAvg {
    fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]) {
        // do not use `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation,
        // since it is slower and has no obvious benefits in increased precision.

        for (v, count, dst) in izip!(
            values.iter(),
            iac.values.iter_mut(),
            iac.ts.values.iter_mut()
        ).filter(|(v, _, _)| !v.is_nan()) {

            if *count == 0.0 {
                *dst = *v;
                *count = 1.0;
                continue;
            }

            *dst += v;
            *count += 1.0;
        }
    }

    fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
        let iter = izip!(
            src.values.iter(),
            dst.values.iter_mut(),
            src.ts.values.iter(),
            dst.ts.values.iter_mut()
        );
        for (src_count, dst_count, v, dst) in iter {
            if *src_count == 0.0 {
                continue;
            }

            if *dst_count == 0.0 {
                *dst = *v;
                *dst_count = *src_count;
                continue;
            }

            *dst += v;
            *dst_count += *src_count;
        }
    }

    fn finalize(&self, iac: &mut IncrementalAggrContext) {
        let counts = &iac.values;

        for (count, dst_value) in counts.iter().zip(iac.ts.values.iter_mut()) {
            if *count == 0.0 {
                *dst_value = f64::NAN;
                continue;
            }
            *dst_value /= count;
        }
    }

    fn keep_original(&self) -> bool {
        false
    }
}
