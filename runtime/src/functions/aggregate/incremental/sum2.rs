use super::context::{IncrementalAggrContext, IncrementalAggrHandler};

pub struct IncrementalAggrSum2 {}

impl IncrementalAggrHandler for IncrementalAggrSum2 {
    fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]) {
        for ((v, count), dst) in values
            .iter().cloned()
            .zip(iac.values.iter_mut())
            .zip(iac.ts.values.iter_mut())
        {
            if v.is_nan() {
                continue;
            }
            let v_squared = v * v;
            if *count == 0.0 {
                *count = 1.0;
                *dst = v_squared;
                continue;
            }

            *dst += v_squared;
        }
    }

    fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
        for ((_src_count, dst_count), (v, dst)) in
            src.values.iter().cloned()
                .zip(dst.values.iter_mut())
                .zip(src.ts.values.iter().cloned().zip(dst.ts.values.iter_mut()))
                .filter(|((src_count, _), _)| *src_count != 0.0) {
            if *dst_count == 0.0 {
                *dst_count = 1.0;
                *dst = v;
                continue;
            }
            *dst += v;
        }
    }

    fn keep_original(&self) -> bool {
        false
    }
}
