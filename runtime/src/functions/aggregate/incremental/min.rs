use super::{IncrementalAggrContext, IncrementalAggrHandler};

pub struct IncrementalAggrMin {}

impl IncrementalAggrHandler for IncrementalAggrMin {
    fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]) {
        for (v, (dst, dst_count)) in values
            .iter().cloned()
            .zip(iac.ts.values.iter_mut().zip(iac.values.iter_mut()))
            .filter(|(v, (_, _))| !v.is_nan()) {

            if *dst_count == 0.0 {
                *dst_count = 1.0;
                *dst = v;
                continue;
            }
            if v < *dst {
                *dst = v;
            }

        }
    }

    fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
        for ((_src_count, dst_count), (v, dst)) in
            src.values.iter().cloned().zip(dst.values.iter_mut())
                .zip(src.ts.values.iter().cloned().zip(dst.ts.values.iter_mut())
                ).filter(|((src_count, _), (_, _))| *src_count != 0.0) {

            if *dst_count == 0.0 {
                *dst_count = 1.0;
                *dst = v;
                continue;
            }
            if v < *dst {
                *dst = v;
            }
        }
    }

    fn keep_original(&self) -> bool {
        false
    }
}
