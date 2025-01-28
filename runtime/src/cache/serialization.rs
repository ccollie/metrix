use std::mem::size_of;
use std::sync::Arc;

use pco::data_types::Number;
use pco::errors::PcoError;
use pco::standalone::{simple_compress, simple_decompress_into};
use pco::{ChunkConfig, DeltaSpec, PagingSpec};

use crate::common::encoding::marshal_var_i64;
use crate::types::{MetricName, SeriesSlice, Timeseries, Timestamp};
use crate::{RuntimeError, RuntimeResult};

// todo: move elsewhere
#[allow(dead_code)]
pub(crate) fn compress_series(series: &[Timeseries], buf: &mut Vec<u8>) -> RuntimeResult<()> {
    let series_slices: Vec<SeriesSlice> = series
        .iter()
        .map(|s| SeriesSlice {
            metric_name: &s.metric_name,
            timestamps: &s.timestamps,
            values: &s.values,
        })
        .collect();

    compress_series_slice(&series_slices, buf)
}

pub(crate) fn compress_series_slice(
    series: &[SeriesSlice],
    buf: &mut Vec<u8>,
) -> RuntimeResult<()> {
    const DATA_PAGE_SIZE: usize = 3000;

    let series_count = series.len();

    // write out series count
    write_usize(buf, series_count);

    if series_count == 0 {
        return Ok(());
    }

    // Invariant: the length of all series is the same
    // Invariant: the timestamps are equal between series
    let q_timestamps = &series[0].timestamps;
    let mut count = q_timestamps.len();

    // todo: smallvec
    let mut data_page_sizes = Vec::with_capacity(count / DATA_PAGE_SIZE + 1);

    while count > 0 {
        let page_size = count.min(DATA_PAGE_SIZE);
        data_page_sizes.push(page_size);
        count -= page_size;
    }

    // todo: avoid clone
    let config =
        ChunkConfig::default().with_paging_spec(PagingSpec::Exact(data_page_sizes.clone()));

    // the caller ensures that timestamps are equally spaced, so we can use delta encoding
    let ts_config = ChunkConfig::default().with_delta_spec(DeltaSpec::TryConsecutive(2));

    // write out value chunk metadata

    // write out series labels
    for series in series.iter() {
        series.metric_name.marshal(buf);
    }

    let mut idx = 0;
    let mut value_offset = 0;
    for page_size in data_page_sizes.iter() {
        // Each page consists of
        // 1. count
        // 2. timestamp min and max (for fast decompression filtering)
        // 3. Total data size (timestamps and values). Allows for fast seeking by date range.
        // 4. compressed timestamps body size
        // 5. timestamp page
        // 6. series compressed values size
        // 7. compressed values page

        // 6/7 are repeated for each series

        // 1.
        write_usize(buf, *page_size);

        // 2.
        // There's no reason you have to serialize the timestamp metadata in the
        // same way as q_compress. Here we do it out of convenience.
        let t_min = q_timestamps[idx];
        idx += page_size;
        let t_max = q_timestamps[idx - 1];
        write_timestamp(buf, t_min);
        write_timestamp(buf, t_max);

        // 3.
        // add placeholder for total data size
        let placeholder_offset = buf.len();

        write_usize(buf, 0);
        let data_size_offset = buf.len();

        if *page_size == 0 {
            continue;
        }

        let end_offset = value_offset + *page_size;
        // for ease of seeking by date range, we write out the full compressed data page size (timestamps and data)
        let timestamps = &q_timestamps[value_offset..end_offset];

        // 4/5
        write_data(buf, timestamps, &ts_config)?;

        for slice in series.iter() {
            let values = &slice.values[value_offset..end_offset];
            // 6/7
            write_data(buf, values, &config)?;
        }

        // patch in the data size
        let data_size = buf.len() - data_size_offset;
        buf[placeholder_offset..placeholder_offset + size_of::<usize>()]
            .copy_from_slice(&data_size.to_le_bytes());

        write_usize(buf, data_size);

        value_offset += *page_size;
    }

    Ok(())
}

pub(crate) fn deserialize_series_between(
    compressed: &[u8],
    t0: i64,
    t1: i64,
) -> RuntimeResult<Vec<Timeseries>> {
    if compressed.is_empty() {
        return Ok(vec![]);
    }

    let mut compressed = compressed;
    // read series count
    let series_count = read_usize(&mut compressed, "series count")?;
    if series_count == 0 {
        return Ok(vec![]);
    }

    // read series labels
    let mut res: Vec<Timeseries> = Vec::with_capacity(series_count);
    for _ in 0..series_count {
        let (c, s) = MetricName::unmarshal(compressed)?;
        compressed = c;
        res.push(Timeseries {
            metric_name: s,
            timestamps: Arc::new(vec![]),
            values: vec![],
        });
    }

    // todo: init capacity
    let mut timestamps = Vec::new();

    // decompression scratch buffers to minimize allocations
    let mut page_t: Vec<i64> = Vec::new();
    let mut page_v: Vec<f64> = Vec::new();

    while !compressed.is_empty() {
        let size = read_usize(&mut compressed, "data length")?;

        if size == 0 {
            break;
        }

        page_v.resize(size, 0.0);
        page_t.resize(size, 0);

        let mut ts = read_timestamp(&mut compressed)?;

        if ts > t1 {
            break;
        }

        ts = read_timestamp(&mut compressed)?;

        // size of data segment (timestamps and values)
        let size = read_usize(&mut compressed, "data segment length")?;
        let data_size = size;

        if ts < t0 {
            // we can skip this data
            compressed = &compressed[data_size..];
        } else {
            // we need to filter and append this data
            let count = read_timestamp_page(&mut compressed, &mut page_t)?;
            if count != size {
                return Err(RuntimeError::SerializationError(
                    "incomplete timestamp page".to_string(),
                ));
            }

            if page_t.is_empty() {
                continue;
            }

            let first = page_t[0];
            if first > t1 {
                break;
            }

            let (ts_start_index, ts_end_index) = get_timestamp_index_bounds(&page_t, t0, t1);
            timestamps.extend_from_slice(&page_t[ts_start_index..=ts_end_index]);

            for series in res.iter_mut() {
                let count = read_values_page(&mut compressed, &mut page_v)?;

                if count != size {
                    return Err(RuntimeError::SerializationError(
                        "incomplete data page".to_string(),
                    ));
                }

                series
                    .values
                    .extend_from_slice(&page_v[ts_start_index..=ts_end_index]);
            }

            let end = page_t[page_t.len() - 1];
            if end > t1 {
                break;
            }
        }
    }

    let timestamps = Arc::new(timestamps);
    for series in res.iter_mut() {
        series.timestamps = Arc::clone(&timestamps);
    }

    Ok(res)
}

pub(crate) fn get_timestamp_index_bounds(
    timestamps: &[i64],
    start_ts: Timestamp,
    end_ts: Timestamp,
) -> (usize, usize) {
    let start_idx = timestamps
        .binary_search(&start_ts)
        .unwrap_or_else(|idx| idx);
    let end_idx = timestamps
        .binary_search(&end_ts)
        .unwrap_or_else(|idx| idx - 1);
    (start_idx, end_idx)
}

// todo: simple_compress_into
fn write_data<T: Number>(
    dest: &mut Vec<u8>,
    values: &[T],
    config: &ChunkConfig,
) -> RuntimeResult<usize> {
    let buf = simple_compress(values, config).map_err(map_err)?;
    write_usize(dest, buf.len());
    dest.extend(&buf);
    Ok(buf.len())
}

fn read_timestamp_page(compressed: &mut &[u8], dst: &mut [i64]) -> RuntimeResult<usize> {
    let size = read_usize(compressed, "timestamp data size")?;
    let progress = simple_decompress_into(compressed, dst).map_err(map_err)?;
    if !progress.finished {
        return Err(RuntimeError::SerializationError("incomplete timestamp data".to_string()));
    }
    *compressed = &compressed[size..];
    Ok(progress.n_processed)
}

fn read_values_page(compressed: &mut &[u8], dst: &mut [f64]) -> RuntimeResult<usize> {
    let size = read_usize(compressed, "value data size")?;
    let progress = simple_decompress_into(compressed, dst).map_err(map_err)?;
    if !progress.finished {
        return Err(RuntimeError::SerializationError("incomplete value data".to_string()));
    }

    *compressed = &compressed[size..];
    Ok(progress.n_processed)
}

fn map_err(e: PcoError) -> RuntimeError {
    RuntimeError::SerializationError(e.to_string())
}

fn read_timestamp(compressed: &mut &[u8]) -> RuntimeResult<i64> {
    let (remaining, value) = crate::common::encoding::read_i64(compressed, "timestamp")?;
    *compressed = remaining;
    Ok(value)
}

fn write_timestamp(dest: &mut Vec<u8>, ts: i64) {
    marshal_var_i64(dest, ts);
}

fn write_usize(slice: &mut Vec<u8>, size: usize) {
    slice.extend_from_slice(&size.to_le_bytes());
}

fn read_usize(input: &mut &[u8], field: &str) -> RuntimeResult<usize> {
    let (int_bytes, rest) = input.split_at(size_of::<usize>());
    let buf = int_bytes.try_into().map_err(|_| {
        RuntimeError::SerializationError(format!("invalid usize reading {}", field).to_string())
    })?;

    *input = rest;
    Ok(usize::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_series(tss: &[Timeseries]) {
        let mut buffer: Vec<u8> = Vec::new();
        compress_series(tss, &mut buffer).unwrap();
        let tss2 = deserialize_series_between(&buffer, 0, 100000).unwrap();

        assert_eq!(
            tss, tss2,
            "unexpected timeseries unmarshalled\ngot\n{:?}\nwant\n{:?}",
            tss2[0], tss[0]
        )
    }

    // Single series
    #[test]
    fn single_series() {
        let series = Timeseries::default();
        test_series(&[series]);

        let mut series = Timeseries::default();
        series.metric_name.measurement = "foobar".to_string();
        series.metric_name.add_label("tag1", "value1");
        series.metric_name.add_label("tag2", "value2");

        series.values = vec![1.0, 2.0, 3.234];
        series.timestamps = Arc::new(vec![10, 20, 30]);
        test_series(&[series]);
    }

    #[test]
    fn multiple_series() {
        let mut series = Timeseries::default();
        series.metric_name.measurement = "foobar".to_string();
        series.metric_name.add_label("tag1", "value1");
        series.metric_name.add_label("tag2", "value2");

        series.values = vec![1.0, 2.34, -33.0];
        series.timestamps = Arc::new(vec![0, 10, 20]);

        let mut series2 = Timeseries::default();
        series2.metric_name.measurement = "baz".to_string();
        series2.metric_name.add_label("tag12", "value13");

        series2.values = vec![4.0, 1.0, -2.34];
        series2.timestamps = Arc::new(vec![0, 10, 20]);

        test_series(&[series, series2]);
    }
}
