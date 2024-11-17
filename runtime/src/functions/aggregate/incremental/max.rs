use itertools::izip;

use super::{IncrementalAggrContext, IncrementalAggrHandler};

pub struct IncrementalAggrMax {}

impl IncrementalAggrHandler for IncrementalAggrMax {
    fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]) {
        let iter = izip!(
            values.iter(),
            iac.ts.values.iter_mut(),
            iac.values.iter_mut()
        );
        for (v, dst, dst_count) in iter.filter(|(v, _, _)| !v.is_nan()) {
            if *dst_count == 0.0 {
                *dst_count = 1.0;
                *dst = *v;
                continue;
            }
            if *v > *dst {
                *dst = *v;
            }
        }
    }

    fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
        let iter = izip!(
            src.values.iter(),
            dst.values.iter_mut(),
            src.ts.values.iter(),
            dst.ts.values.iter_mut()
        );
        for (_src_count, dst_count, v, dst) in iter.filter(|(src_count, _, _, _)| **src_count == 0.0) {
            if *dst_count == 0.0 {
                *dst_count = 1.0;
                *dst = *v;
                continue;
            }
            if *v > *dst {
                *dst = *v;
            }
        }
    }

    fn keep_original(&self) -> bool {
        false
    }
}
