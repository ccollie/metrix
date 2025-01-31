use metricsql_parser::label::Matchers;

pub(crate) fn is_empty_extra_matchers(matchers: &Option<Matchers>) -> bool {
    if let Some(matchers) = matchers {
        return matchers.is_empty();
    }
    true
}

pub(crate) fn join_matchers_with_extra_filters_owned(
    src: &Matchers,
    etfs: &Option<Matchers>,
) -> Matchers {
    if src.is_empty() {
        if let Some(etfs) = etfs {
            return etfs.clone();
        }
        return src.clone();
    }

    if let Some(etfs) = etfs {
        if etfs.is_empty() {
            return src.clone();
        }
        let mut dst = src.clone();

        if !etfs.or_matchers.is_empty() {
            if !dst.matchers.is_empty() {
                dst.or_matchers.push(std::mem::take(&mut dst.matchers));
            }
            etfs.or_matchers.iter().for_each(|m| {
                dst.or_matchers.push(m.clone());
            });
        }
        if !etfs.matchers.is_empty() {
            if !dst.matchers.is_empty() {
                dst.or_matchers.push(std::mem::take(&mut dst.matchers));
            }
            dst.or_matchers.push(etfs.matchers.clone());
        }
        return dst;
    }
    src.clone()
}
