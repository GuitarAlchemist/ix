//! Common protocol trait for data sources and sinks.
//!
//! Every I/O backend implements DataSource and/or DataSink,
//! giving a uniform interface for the skill layer.

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
    /// Read the next batch of records.
    fn read_batch(&mut self, max_records: usize) -> Result<DataBatch, crate::error::IoError>;

    /// Check if the source has more data.
    fn has_more(&self) -> bool;
}

/// Trait for synchronous data consumers.
pub trait DataSink {
    /// Write a batch of records.
    fn write_batch(&mut self, batch: &DataBatch) -> Result<(), crate::error::IoError>;

    /// Flush buffered data.
    fn flush(&mut self) -> Result<(), crate::error::IoError>;
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
