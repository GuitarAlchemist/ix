//! CSV reading and writing with ndarray integration.

use std::path::Path;

use ndarray::{Array1, Array2};

use crate::error::IoError;
use crate::protocol::{DataBatch, DataRecord};

/// Read a CSV file into a DataBatch.
/// Assumes all columns are numeric (f64). Non-numeric values become NaN.
pub fn read_csv(path: &Path, has_header: bool) -> Result<DataBatch, IoError> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(has_header)
        .from_path(path)?;

    let column_names = if has_header {
        Some(reader.headers()?.iter().map(|s| s.to_string()).collect())
    } else {
        None
    };

    let mut batch = DataBatch::new();
    if let Some(names) = column_names {
        batch = batch.with_columns(names);
    }

    for result in reader.records() {
        let record = result?;
        let row: Vec<f64> = record
            .iter()
            .map(|s| s.trim().parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        batch.push(DataRecord::Row(row));
    }

    Ok(batch)
}

/// Read CSV from a string.
pub fn read_csv_string(data: &str, has_header: bool) -> Result<DataBatch, IoError> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(has_header)
        .from_reader(data.as_bytes());

    let column_names = if has_header {
        Some(reader.headers()?.iter().map(|s| s.to_string()).collect())
    } else {
        None
    };

    let mut batch = DataBatch::new();
    if let Some(names) = column_names {
        batch = batch.with_columns(names);
    }

    for result in reader.records() {
        let record = result?;
        let row: Vec<f64> = record
            .iter()
            .map(|s| s.trim().parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        batch.push(DataRecord::Row(row));
    }

    Ok(batch)
}

/// Write an Array2 to CSV.
pub fn write_csv(
    path: &Path,
    data: &Array2<f64>,
    headers: Option<&[String]>,
) -> Result<(), IoError> {
    let mut writer = csv::Writer::from_path(path)?;

    if let Some(h) = headers {
        writer.write_record(h)?;
    }

    for row in data.rows() {
        let fields: Vec<String> = row.iter().map(|v| v.to_string()).collect();
        writer.write_record(&fields)?;
    }

    writer.flush()?;
    Ok(())
}

/// Write a DataBatch to CSV.
pub fn write_batch_csv(path: &Path, batch: &DataBatch) -> Result<(), IoError> {
    let mut writer = csv::Writer::from_path(path)?;

    if let Some(ref names) = batch.column_names {
        writer.write_record(names)?;
    }

    for record in &batch.records {
        if let DataRecord::Row(row) = record {
            let fields: Vec<String> = row.iter().map(|v| v.to_string()).collect();
            writer.write_record(&fields)?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Quick helper: load CSV directly to Array2 + optional header names.
pub fn load_csv_matrix(
    path: &Path,
    has_header: bool,
) -> Result<(Array2<f64>, Option<Vec<String>>), IoError> {
    let batch = read_csv(path, has_header)?;
    let names = batch.column_names.clone();
    let matrix = batch
        .to_array2()
        .ok_or_else(|| IoError::Parse("No numeric data found".into()))?;
    Ok((matrix, names))
}

/// Quick helper: load CSV and split into X (features) and y (target).
pub fn load_csv_xy(
    path: &Path,
    target_col: usize,
    has_header: bool,
) -> Result<(Array2<f64>, Array1<f64>), IoError> {
    let batch = read_csv(path, has_header)?;
    batch
        .split_xy(target_col)
        .ok_or_else(|| IoError::Parse("Cannot split into X and y".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_csv_string() {
        let data = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n";
        let batch = read_csv_string(data, true).unwrap();

        assert_eq!(batch.len(), 3);
        assert_eq!(
            batch.column_names.as_ref().unwrap(),
            &["a".to_string(), "b".to_string(), "c".to_string()]
        );

        let arr = batch.to_array2().unwrap();
        assert_eq!(arr.dim(), (3, 3));
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[2, 2]], 9.0);
    }

    #[test]
    fn test_split_xy_from_csv() {
        let data = "f1,f2,target\n1,2,10\n3,4,20\n5,6,30\n";
        let batch = read_csv_string(data, true).unwrap();
        let (x, y) = batch.split_xy(2).unwrap();

        assert_eq!(x.dim(), (3, 2));
        assert_eq!(y.len(), 3);
        assert_eq!(y[2], 30.0);
    }

    #[test]
    fn test_nan_handling() {
        let data = "1,abc,3\n4,5,xyz\n";
        let batch = read_csv_string(data, false).unwrap();
        let arr = batch.to_array2().unwrap();

        assert!(arr[[0, 1]].is_nan());
        assert!(arr[[1, 2]].is_nan());
        assert_eq!(arr[[0, 0]], 1.0);
    }
}
