use super::{IncrementalAggrContext, IncrementalAggrHandler};

pub struct IncrementalAggrCount {}

impl IncrementalAggrHandler for IncrementalAggrCount {
    fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]) {
        update_count(iac, values);
    }

    fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
        merge_count(dst, src);
    }

    fn finalize(&self, iac: &mut IncrementalAggrContext) {
        for v in iac.ts.values.iter_mut() {
            if *v == 0.0 {
                *v = f64::NAN
            }
        }
    }

    fn keep_original(&self) -> bool {
        false
    }
}

pub fn update_count(iac: &mut IncrementalAggrContext, values: &[f64]) {
    for (_v, dst) in values.iter().cloned()
        .zip(iac.ts.values.iter_mut())
        .filter(|(v, _)| !v.is_nan()) {
        *dst += 1.0;
    }
}

pub fn merge_count(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    for (v, dst) in src.ts.values.iter().zip(dst.ts.values.iter_mut()) {
        *dst += v;
    }
}
