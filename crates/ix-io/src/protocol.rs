//! Common protocol traits for data sources and sinks.
//!
//! [`DataSource`] and [`DataSink`] are the **synchronous, pull-based** record
//! interface: [`pump`] moves records from any source into any sink without
//! knowing either concrete type. That generic consumer is what makes the pair
//! a seam rather than a declaration.
//!
//! Not every backend in this crate implements them, and the ones that do not
//! say so in their own module docs. The dividing line is the shape of the
//! backend, not an oversight:
//!
//! | backend | implements | why |
//! | --- | --- | --- |
//! | [`crate::csv_io`] | [`DataSource`] + [`DataSink`] | blocking reader/writer over a record stream |
//! | [`crate::json_io`] | [`DataSource`] + [`DataSink`] (NDJSON) | ditto — whole-document JSON is a batch, not a stream |
//! | [`crate::http`] | through [`BatchSource`] | `async`; see below |
//! | [`crate::tcp`], [`crate::websocket`] | through [`BatchSource`] | `async`; see below |
//! | [`crate::pipe`] | no | delivers unframed bytes with no record or EOF semantics |
//! | [`crate::watcher`] | no | emits filesystem events, which carry no data |
//! | [`crate::trace_bridge`] | no | loads GA `Trace` documents, a domain type |
//!
//! # Why the network backends do not implement [`DataSource`] directly
//!
//! [`DataSource::read_batch`] is synchronous; `http`, `tcp` and `websocket`
//! are `async`. Bridging them inside `&mut self` would mean blocking on a
//! future, which **panics** when called from within a Tokio runtime — the
//! runtime every caller of those backends is already inside. They therefore
//! *acquire* asynchronously and *serve* synchronously: `await` a [`DataBatch`],
//! then wrap it in [`BatchSource`].
//!
//! A parallel `AsyncDataSource` trait was considered and rejected. Declaring a
//! second contract that nothing consumes generically is the defect ix#299 is
//! about, not its cure.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// A record of data — the common unit exchanged between sources and sinks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataRecord {
    /// A single row of f64 values (for numeric data).
    Row(Vec<f64>),
    /// A named row (column_name -> value).
    Named(std::collections::HashMap<String, f64>),
    /// Raw text line.
    Text(String),
    /// Raw bytes.
    Bytes(Vec<u8>),
}

impl DataRecord {
    /// Convert to f64 vec (only for Row variant).
    pub fn as_row(&self) -> Option<&[f64]> {
        match self {
            DataRecord::Row(v) => Some(v),
            _ => None,
        }
    }
}

/// A batch of records, convertible to ndarray.
#[derive(Debug, Clone)]
pub struct DataBatch {
    pub records: Vec<DataRecord>,
    pub column_names: Option<Vec<String>>,
}

impl DataBatch {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            column_names: None,
        }
    }

    pub fn with_columns(mut self, names: Vec<String>) -> Self {
        self.column_names = Some(names);
        self
    }

    pub fn push(&mut self, record: DataRecord) {
        self.records.push(record);
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Convert numeric rows to an Array2<f64>.
    /// Ignores non-Row records.
    pub fn to_array2(&self) -> Option<Array2<f64>> {
        let rows: Vec<&[f64]> = self.records.iter().filter_map(|r| r.as_row()).collect();

        if rows.is_empty() {
            return None;
        }

        let ncols = rows[0].len();
        let nrows = rows.len();
        let flat: Vec<f64> = rows.into_iter().flat_map(|r| r.iter().copied()).collect();
        Array2::from_shape_vec((nrows, ncols), flat).ok()
    }

    /// Extract a single column as Array1<f64>.
    pub fn column(&self, idx: usize) -> Option<Array1<f64>> {
        let values: Vec<f64> = self
            .records
            .iter()
            .filter_map(|r| r.as_row().and_then(|row| row.get(idx).copied()))
            .collect();

        if values.is_empty() {
            None
        } else {
            Some(Array1::from_vec(values))
        }
    }

    /// Split into features (X) and target (y) by column index.
    pub fn split_xy(&self, target_col: usize) -> Option<(Array2<f64>, Array1<f64>)> {
        let rows: Vec<&[f64]> = self.records.iter().filter_map(|r| r.as_row()).collect();
        if rows.is_empty() {
            return None;
        }

        let ncols = rows[0].len();
        let nrows = rows.len();

        let mut x_flat = Vec::with_capacity(nrows * (ncols - 1));
        let mut y_vec = Vec::with_capacity(nrows);

        for row in &rows {
            for (j, &val) in row.iter().enumerate() {
                if j == target_col {
                    y_vec.push(val);
                } else {
                    x_flat.push(val);
                }
            }
        }

        let x = Array2::from_shape_vec((nrows, ncols - 1), x_flat).ok()?;
        let y = Array1::from_vec(y_vec);
        Some((x, y))
    }
}

impl Default for DataBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for synchronous data producers.
pub trait DataSource {
    /// Read up to `max_records` records.
    ///
    /// Returning fewer than `max_records` — including zero — does **not** mean
    /// the source is exhausted; only [`has_more`](DataSource::has_more)
    /// answers that. Implementations should carry `column_names` through onto
    /// every batch they emit so a downstream [`DataSink`] can write a header.
    ///
    /// `max_records == 0` must return an empty batch rather than looping.
    fn read_batch(&mut self, max_records: usize) -> Result<DataBatch, crate::error::IoError>;

    /// Whether a further [`read_batch`](DataSource::read_batch) may yield records.
    ///
    /// **This is a one-sided guarantee.** `false` is definitive: the source is
    /// exhausted. `true` means "not known to be exhausted" — a pull-based
    /// reader cannot answer otherwise without look-ahead it does not have, and
    /// a freshly opened empty file legitimately reports `true` until the first
    /// read. [`pump`] and [`drain`] are written against that weaker contract.
    fn has_more(&self) -> bool;
}

/// Trait for synchronous data consumers.
pub trait DataSink {
    /// Write a batch of records.
    ///
    /// A sink with a header (CSV) takes its column names from the **first**
    /// batch it is handed, so a caller mixing batches with different
    /// `column_names` gets the first one's header.
    fn write_batch(&mut self, batch: &DataBatch) -> Result<(), crate::error::IoError>;

    /// Flush buffered data. [`pump`] calls this once at the end.
    fn flush(&mut self) -> Result<(), crate::error::IoError>;
}

/// A [`DataSource`] over an already-materialised [`DataBatch`].
///
/// This is the adapter the `async` backends reach the synchronous protocol
/// through: `await` a batch from `http`/`tcp`/`websocket`, wrap it here, and
/// it is interchangeable with a streaming CSV or NDJSON source at every call
/// site that takes `impl DataSource`.
///
/// It buffers the whole batch by construction. That is a property of the
/// caller's fetch, not of this type — bound the fetch (see
/// [`crate::http::FetchLimits`]) rather than expecting the cursor to.
#[derive(Debug, Clone)]
pub struct BatchSource {
    batch: DataBatch,
    cursor: usize,
}

impl BatchSource {
    /// Wrap a batch, positioned at its first record.
    pub fn new(batch: DataBatch) -> Self {
        Self { batch, cursor: 0 }
    }

    /// Records not yet handed out.
    pub fn remaining(&self) -> usize {
        self.batch.records.len().saturating_sub(self.cursor)
    }
}

impl From<DataBatch> for BatchSource {
    fn from(batch: DataBatch) -> Self {
        Self::new(batch)
    }
}

impl DataSource for BatchSource {
    fn read_batch(&mut self, max_records: usize) -> Result<DataBatch, crate::error::IoError> {
        let end = self
            .cursor
            .saturating_add(max_records)
            .min(self.batch.records.len());
        let mut out = DataBatch::new();
        out.column_names = self.batch.column_names.clone();
        out.records
            .extend_from_slice(&self.batch.records[self.cursor..end]);
        self.cursor = end;
        Ok(out)
    }

    fn has_more(&self) -> bool {
        self.remaining() > 0
    }
}

/// Move every remaining record from `source` into `sink`, `batch_size` at a time.
///
/// Streaming: nothing larger than one batch is held at once, so there is no
/// record cap here — unlike [`drain`], which accumulates.
///
/// Returns the number of records written. The sink is flushed once, at the end.
///
/// # Errors
///
/// `batch_size == 0` is rejected rather than spinning forever on a source that
/// keeps reporting [`has_more`](DataSource::has_more).
pub fn pump<S, K>(
    source: &mut S,
    sink: &mut K,
    batch_size: usize,
) -> Result<usize, crate::error::IoError>
where
    S: DataSource + ?Sized,
    K: DataSink + ?Sized,
{
    if batch_size == 0 {
        return Err(crate::error::IoError::Limit(
            "pump: batch_size must be > 0".into(),
        ));
    }

    let mut moved = 0usize;
    while source.has_more() {
        let batch = source.read_batch(batch_size)?;
        if batch.is_empty() {
            // `has_more() == true` is provisional; an empty read is the EOF
            // signal for sources that cannot look ahead.
            break;
        }
        moved += batch.len();
        sink.write_batch(&batch)?;
    }
    sink.flush()?;
    Ok(moved)
}

/// Read a source to exhaustion into one batch, refusing to exceed `max_records`.
///
/// Unlike [`pump`], this accumulates in memory, so it takes an explicit ceiling
/// and **fails loudly** when the source has more than that. Silently truncating
/// would hand back a short batch indistinguishable from a complete one.
///
/// # Errors
///
/// `batch_size == 0`, or a source holding more than `max_records` records.
pub fn drain<S>(
    source: &mut S,
    batch_size: usize,
    max_records: usize,
) -> Result<DataBatch, crate::error::IoError>
where
    S: DataSource + ?Sized,
{
    if batch_size == 0 {
        return Err(crate::error::IoError::Limit(
            "drain: batch_size must be > 0".into(),
        ));
    }

    let mut out = DataBatch::new();
    while source.has_more() {
        let batch = source.read_batch(batch_size)?;
        if batch.is_empty() {
            break;
        }
        if out.column_names.is_none() {
            out.column_names = batch.column_names.clone();
        }
        if out.records.len() + batch.records.len() > max_records {
            return Err(crate::error::IoError::Limit(format!(
                "drain: source yields more than max_records = {max_records}; \
                 raise the ceiling or use pump() to stream instead"
            )));
        }
        out.records.extend(batch.records);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_batch_to_array() {
        let mut batch = DataBatch::new();
        batch.push(DataRecord::Row(vec![1.0, 2.0, 3.0]));
        batch.push(DataRecord::Row(vec![4.0, 5.0, 6.0]));

        let arr = batch.to_array2().unwrap();
        assert_eq!(arr.dim(), (2, 3));
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 2]], 6.0);
    }

    #[test]
    fn test_split_xy() {
        let mut batch = DataBatch::new();
        batch.push(DataRecord::Row(vec![1.0, 2.0, 10.0]));
        batch.push(DataRecord::Row(vec![3.0, 4.0, 20.0]));

        let (x, y) = batch.split_xy(2).unwrap();
        assert_eq!(x.dim(), (2, 2));
        assert_eq!(y.len(), 2);
        assert_eq!(y[0], 10.0);
        assert_eq!(y[1], 20.0);
    }

    #[test]
    fn test_column_extraction() {
        let mut batch = DataBatch::new();
        batch.push(DataRecord::Row(vec![1.0, 10.0]));
        batch.push(DataRecord::Row(vec![2.0, 20.0]));
        batch.push(DataRecord::Row(vec![3.0, 30.0]));

        let col = batch.column(1).unwrap();
        assert_eq!(col.len(), 3);
        assert_eq!(col[0], 10.0);
    }
}

#[cfg(test)]
mod protocol_tests {
    use super::*;
    use crate::error::IoError;

    /// A source that reports `has_more() == true` forever but yields nothing —
    /// the exact case the weak [`DataSource::has_more`] contract permits, and
    /// the reason [`pump`] and [`drain`] break on an empty read.
    struct AlwaysHungry;

    impl DataSource for AlwaysHungry {
        fn read_batch(&mut self, _max_records: usize) -> Result<DataBatch, IoError> {
            Ok(DataBatch::new())
        }
        fn has_more(&self) -> bool {
            true
        }
    }

    /// A sink that counts, so `pump`'s flush contract can be asserted.
    #[derive(Default)]
    struct CountingSink {
        records: usize,
        batches: usize,
        flushes: usize,
    }

    impl DataSink for CountingSink {
        fn write_batch(&mut self, batch: &DataBatch) -> Result<(), IoError> {
            self.records += batch.len();
            self.batches += 1;
            Ok(())
        }
        fn flush(&mut self) -> Result<(), IoError> {
            self.flushes += 1;
            Ok(())
        }
    }

    fn rows(n: usize) -> DataBatch {
        let mut batch = DataBatch::new().with_columns(vec!["i".into()]);
        for i in 0..n {
            batch.push(DataRecord::Row(vec![i as f64]));
        }
        batch
    }

    #[test]
    fn batch_source_hands_out_every_record_exactly_once() {
        let mut src = BatchSource::new(rows(5));
        assert_eq!(src.remaining(), 5);

        let mut seen = Vec::new();
        while src.has_more() {
            let batch = src.read_batch(2).unwrap();
            for r in &batch.records {
                seen.push(r.as_row().unwrap()[0]);
            }
        }
        assert_eq!(seen, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(src.remaining(), 0);
        assert!(src.read_batch(2).unwrap().is_empty());
    }

    #[test]
    fn batch_source_propagates_column_names() {
        let mut src = BatchSource::new(rows(3));
        let batch = src.read_batch(1).unwrap();
        assert_eq!(batch.column_names.as_deref(), Some(&["i".to_string()][..]));
    }

    #[test]
    fn batch_source_read_of_zero_returns_empty_without_advancing() {
        let mut src = BatchSource::new(rows(3));
        assert!(src.read_batch(0).unwrap().is_empty());
        assert_eq!(src.remaining(), 3, "a zero-sized read must not consume");
    }

    #[test]
    fn pump_moves_every_record_and_flushes_once() {
        let mut src = BatchSource::new(rows(7));
        let mut sink = CountingSink::default();

        assert_eq!(pump(&mut src, &mut sink, 3).unwrap(), 7);
        assert_eq!(sink.records, 7);
        assert_eq!(sink.batches, 3, "7 records at 3 per batch is 3+3+1");
        assert_eq!(sink.flushes, 1, "flush belongs at the end, not per batch");
    }

    #[test]
    fn pump_terminates_on_an_empty_read_despite_has_more() {
        let mut sink = CountingSink::default();
        assert_eq!(pump(&mut AlwaysHungry, &mut sink, 4).unwrap(), 0);
        assert_eq!(sink.flushes, 1);
    }

    #[test]
    fn drain_terminates_on_an_empty_read_despite_has_more() {
        let batch = drain(&mut AlwaysHungry, 4, 10).unwrap();
        assert!(batch.is_empty());
    }

    #[test]
    fn drain_collects_everything_under_the_ceiling() {
        let mut src = BatchSource::new(rows(6));
        let batch = drain(&mut src, 4, 6).unwrap();
        assert_eq!(batch.len(), 6);
        assert_eq!(batch.column_names.as_deref(), Some(&["i".to_string()][..]));
        assert_eq!(batch.to_array2().unwrap().dim(), (6, 1));
    }

    #[test]
    fn drain_fails_loudly_rather_than_returning_a_short_batch() {
        let mut src = BatchSource::new(rows(6));
        let err = drain(&mut src, 4, 5).expect_err("6 records must not fit under a ceiling of 5");
        assert!(err.to_string().contains("max_records = 5"), "got: {err}");
    }

    #[test]
    fn zero_batch_size_is_rejected_by_both_drivers() {
        let mut src = BatchSource::new(rows(2));
        assert!(drain(&mut src, 0, 10).is_err());

        let mut src = BatchSource::new(rows(2));
        let mut sink = CountingSink::default();
        assert!(pump(&mut src, &mut sink, 0).is_err());
    }

    /// `pump` is generic over both sides, so a source and a sink that have
    /// never heard of each other compose. That generic consumer is what makes
    /// [`DataSource`] / [`DataSink`] a seam rather than a declaration.
    #[test]
    fn pump_composes_a_source_and_a_sink_through_dyn_dispatch() {
        let mut src: Box<dyn DataSource> = Box::new(BatchSource::new(rows(4)));
        let mut sink: Box<dyn DataSink> = Box::new(CountingSink::default());
        assert_eq!(pump(src.as_mut(), sink.as_mut(), 2).unwrap(), 4);
    }
}
