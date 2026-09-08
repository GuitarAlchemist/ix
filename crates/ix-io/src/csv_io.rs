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

// ── Protocol backends ────────────────────────────────────────────

use std::fs::File;
use std::io::{BufWriter, Read, Write};

use crate::protocol::{DataSink, DataSource};

/// A streaming [`DataSource`] over CSV.
///
/// Unlike [`read_csv`], this does not materialise the file: each
/// [`read_batch`](DataSource::read_batch) pulls only the records asked for.
/// Non-numeric fields become `NaN`, matching [`read_csv`].
pub struct CsvSource<R: Read> {
    reader: csv::Reader<R>,
    column_names: Option<Vec<String>>,
    exhausted: bool,
}

impl<R: Read> CsvSource<R> {
    /// Wrap any reader. Reads the header row eagerly when `has_header`.
    pub fn new(inner: R, has_header: bool) -> Result<Self, IoError> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(has_header)
            .from_reader(inner);

        let column_names = if has_header {
            Some(reader.headers()?.iter().map(|s| s.to_string()).collect())
        } else {
            None
        };

        Ok(Self {
            reader,
            column_names,
            exhausted: false,
        })
    }

    /// Column names, if the source was opened with a header row.
    pub fn column_names(&self) -> Option<&[String]> {
        self.column_names.as_deref()
    }
}

impl CsvSource<File> {
    /// Open a CSV file as a streaming source.
    pub fn from_path(path: &Path, has_header: bool) -> Result<Self, IoError> {
        Self::new(File::open(path)?, has_header)
    }
}

impl<R: Read> DataSource for CsvSource<R> {
    fn read_batch(&mut self, max_records: usize) -> Result<DataBatch, IoError> {
        let mut batch = DataBatch::new();
        batch.column_names = self.column_names.clone();

        let mut record = csv::StringRecord::new();
        while batch.len() < max_records {
            if !self.reader.read_record(&mut record)? {
                self.exhausted = true;
                break;
            }
            let row: Vec<f64> = record
                .iter()
                .map(|s| s.trim().parse::<f64>().unwrap_or(f64::NAN))
                .collect();
            batch.push(DataRecord::Row(row));
        }

        Ok(batch)
    }

    fn has_more(&self) -> bool {
        !self.exhausted
    }
}

/// A [`DataSink`] writing CSV rows.
///
/// Non-`Row` records are skipped, matching [`write_batch_csv`]. The header is
/// taken from the first batch that carries `column_names` and written once.
pub struct CsvSink<W: Write> {
    writer: csv::Writer<W>,
    header_written: bool,
}

impl<W: Write> CsvSink<W> {
    /// Wrap any writer.
    pub fn new(inner: W) -> Self {
        Self {
            writer: csv::Writer::from_writer(inner),
            header_written: false,
        }
    }
}

impl CsvSink<BufWriter<File>> {
    /// Create (or truncate) a CSV file to write into.
    pub fn create(path: &Path) -> Result<Self, IoError> {
        Ok(Self::new(BufWriter::new(File::create(path)?)))
    }
}

impl<W: Write> DataSink for CsvSink<W> {
    fn write_batch(&mut self, batch: &DataBatch) -> Result<(), IoError> {
        if !self.header_written {
            if let Some(ref names) = batch.column_names {
                self.writer.write_record(names)?;
            }
            // Set unconditionally: a first batch without names means this file
            // has no header, and a later batch must not inject one mid-stream.
            self.header_written = true;
        }

        for record in &batch.records {
            if let DataRecord::Row(row) = record {
                let fields: Vec<String> = row.iter().map(|v| v.to_string()).collect();
                self.writer.write_record(&fields)?;
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), IoError> {
        self.writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod protocol_tests {
    use super::*;
    use crate::protocol::{drain, pump};
    use std::io::Cursor;

    #[test]
    fn csv_source_streams_in_bounded_batches() {
        let data = "a,b\n1,2\n3,4\n5,6\n";
        let mut src = CsvSource::new(Cursor::new(data), true).unwrap();

        assert_eq!(
            src.column_names(),
            Some(&["a".to_string(), "b".to_string()][..])
        );

        let first = src.read_batch(2).unwrap();
        assert_eq!(first.len(), 2, "must hand back exactly the requested count");
        assert!(src.has_more());

        let second = src.read_batch(2).unwrap();
        assert_eq!(second.len(), 1, "only one record was left");
        assert!(!src.has_more(), "EOF was reached inside the second read");

        let third = src.read_batch(2).unwrap();
        assert!(third.is_empty());
    }

    #[test]
    fn csv_source_carries_column_names_onto_every_batch() {
        let data = "x,y\n1,2\n3,4\n";
        let mut src = CsvSource::new(Cursor::new(data), true).unwrap();
        for _ in 0..2 {
            let batch = src.read_batch(1).unwrap();
            assert_eq!(
                batch.column_names.as_deref(),
                Some(&["x".to_string(), "y".to_string()][..])
            );
        }
    }

    #[test]
    fn pump_csv_to_csv_round_trips_through_the_traits() {
        let data = "a,b\n1,2\n3,4\n5,6\n";
        let mut src = CsvSource::new(Cursor::new(data), true).unwrap();
        let mut sink = CsvSink::new(Vec::<u8>::new());

        let moved = pump(&mut src, &mut sink, 2).unwrap();
        assert_eq!(moved, 3);

        let out = String::from_utf8(sink.writer.into_inner().unwrap()).unwrap();
        let reparsed = read_csv_string(&out, true).unwrap();
        assert_eq!(reparsed.column_names.as_ref().unwrap(), &["a", "b"]);
        assert_eq!(
            reparsed.to_array2().unwrap(),
            read_csv_string(data, true).unwrap().to_array2().unwrap()
        );
    }

    #[test]
    fn pump_rejects_zero_batch_size_instead_of_spinning() {
        let mut src = CsvSource::new(Cursor::new("1,2\n"), false).unwrap();
        let mut sink = CsvSink::new(Vec::<u8>::new());
        let err = pump(&mut src, &mut sink, 0).expect_err("batch_size 0 must be an error");
        assert!(
            err.to_string().contains("batch_size must be > 0"),
            "got: {err}"
        );
    }

    #[test]
    fn drain_refuses_to_silently_truncate() {
        let data = "1\n2\n3\n4\n";
        let mut src = CsvSource::new(Cursor::new(data), false).unwrap();
        let err = drain(&mut src, 2, 3).expect_err("4 records must not fit under a ceiling of 3");
        assert!(err.to_string().contains("max_records = 3"), "got: {err}");

        let mut src = CsvSource::new(Cursor::new(data), false).unwrap();
        let ok = drain(&mut src, 2, 4).unwrap();
        assert_eq!(ok.len(), 4);
    }

    #[test]
    fn sink_without_column_names_writes_no_header() {
        let mut batch = DataBatch::new();
        batch.push(DataRecord::Row(vec![1.0, 2.0]));
        let mut sink = CsvSink::new(Vec::<u8>::new());
        sink.write_batch(&batch).unwrap();
        sink.flush().unwrap();
        let out = String::from_utf8(sink.writer.into_inner().unwrap()).unwrap();
        assert_eq!(out, "1,2\n");
    }
}
