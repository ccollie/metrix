use super::context::{IncrementalAggrContext, IncrementalAggrHandler};

pub struct IncrementalAggrGeomean {}

impl IncrementalAggrHandler for IncrementalAggrGeomean {
    fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]) {
        for (i, v) in values.iter().enumerate().filter(|(_, v)| !v.is_nan()) {
            if iac.values[i] == 0.0 {
                iac.ts.values[i] = *v;
                iac.values[i] = 1.0;
                continue;
            }

            iac.ts.values[i] *= v;
            iac.values[i] += 1.0;
        }
    }

    fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
        let src_values = &src.ts.values;
        let src_counts = &src.values;

        for (i, v) in src_values.iter().enumerate() {
            if src_counts[i] == 0.0 {
                continue;
            }

            if dst.values[i] == 0.0 {
                dst.ts.values[i] = *v;
                dst.values[i] = src_counts[i];
                continue;
            }

            dst.ts.values[i] *= v;
            dst.values[i] += src_counts[i]
        }
    }

    fn finalize(&self, iac: &mut IncrementalAggrContext) {
        let counts = &iac.values;

        for (count, v) in counts.iter().zip(iac.ts.values.iter_mut()) {
            if *count == 0.0 {
                *v = f64::NAN;
                continue;
            }
            *v = v.powf(1.0 / count);
        }
    }

    fn keep_original(&self) -> bool {
        false
    }
}
