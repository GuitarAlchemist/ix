//! JSON data reading, writing, and streaming.

use std::path::Path;

use crate::error::IoError;
use crate::protocol::{DataBatch, DataRecord};

/// Read a JSON array of objects into a DataBatch.
/// Each object should have numeric values.
pub fn read_json_file(path: &Path) -> Result<DataBatch, IoError> {
    let content = std::fs::read_to_string(path)?;
    read_json_string(&content)
}

/// Read JSON from a string.
pub fn read_json_string(data: &str) -> Result<DataBatch, IoError> {
    let value: serde_json::Value = serde_json::from_str(data)?;

    match value {
        serde_json::Value::Array(arr) => parse_json_array(&arr),
        serde_json::Value::Object(_) => {
            // Single object — wrap in array
            parse_json_array(&[value])
        }
        _ => Err(IoError::Parse("Expected JSON array or object".into())),
    }
}

fn parse_json_array(arr: &[serde_json::Value]) -> Result<DataBatch, IoError> {
    let mut batch = DataBatch::new();

    // Extract column names from first object
    if let Some(serde_json::Value::Object(first)) = arr.first() {
        let names: Vec<String> = first.keys().cloned().collect();
        batch = batch.with_columns(names.clone());

        for item in arr {
            if let serde_json::Value::Object(obj) = item {
                let row: Vec<f64> = names
                    .iter()
                    .map(|key| obj.get(key).and_then(|v| v.as_f64()).unwrap_or(f64::NAN))
                    .collect();
                batch.push(DataRecord::Row(row));
            }
        }
    } else {
        // Array of arrays
        for item in arr {
            if let serde_json::Value::Array(inner) = item {
                let row: Vec<f64> = inner
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(f64::NAN))
                    .collect();
                batch.push(DataRecord::Row(row));
            }
        }
    }

    Ok(batch)
}

/// Write a DataBatch to JSON.
pub fn write_json_file(path: &Path, batch: &DataBatch) -> Result<(), IoError> {
    let json = batch_to_json(batch)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Serialize a DataBatch to a JSON string.
pub fn batch_to_json(batch: &DataBatch) -> Result<String, IoError> {
    let mut items = Vec::new();

    for record in &batch.records {
        if let DataRecord::Row(row) = record {
            if let Some(ref names) = batch.column_names {
                let obj: serde_json::Map<String, serde_json::Value> = names
                    .iter()
                    .zip(row.iter())
                    .map(|(name, &val)| (name.clone(), serde_json::json!(val)))
                    .collect();
                items.push(serde_json::Value::Object(obj));
            } else {
                let arr: Vec<serde_json::Value> =
                    row.iter().map(|&v| serde_json::json!(v)).collect();
                items.push(serde_json::Value::Array(arr));
            }
        }
    }

    Ok(serde_json::to_string_pretty(&items)?)
}

/// Read newline-delimited JSON (NDJSON/JSONL) — one JSON object per line.
pub fn read_ndjson_string(data: &str) -> Result<DataBatch, IoError> {
    let mut batch = DataBatch::new();
    let mut names_set = false;

    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)?;

        if let serde_json::Value::Object(obj) = &value {
            if !names_set {
                let names: Vec<String> = obj.keys().cloned().collect();
                batch = batch.with_columns(names);
                names_set = true;
            }

            if let Some(ref names) = batch.column_names {
                let row: Vec<f64> = names
                    .iter()
                    .map(|key| obj.get(key).and_then(|v| v.as_f64()).unwrap_or(f64::NAN))
                    .collect();
                batch.push(DataRecord::Row(row));
            }
        }
    }

    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_json_array_of_objects() {
        let data = r#"[{"x": 1, "y": 2}, {"x": 3, "y": 4}]"#;
        let batch = read_json_string(data).unwrap();

        assert_eq!(batch.len(), 2);
        let arr = batch.to_array2().unwrap();
        assert_eq!(arr.dim(), (2, 2));
    }

    #[test]
    fn test_read_json_array_of_arrays() {
        let data = r#"[[1, 2, 3], [4, 5, 6]]"#;
        let batch = read_json_string(data).unwrap();

        let arr = batch.to_array2().unwrap();
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 2]], 6.0);
    }

    #[test]
    fn test_ndjson() {
        let data = r#"{"a": 1, "b": 2}
{"a": 3, "b": 4}
{"a": 5, "b": 6}"#;
        let batch = read_ndjson_string(data).unwrap();
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_roundtrip_json() {
        let mut batch = DataBatch::new().with_columns(vec!["x".into(), "y".into()]);
        batch.push(DataRecord::Row(vec![1.0, 2.0]));
        batch.push(DataRecord::Row(vec![3.0, 4.0]));

        let json = batch_to_json(&batch).unwrap();
        let batch2 = read_json_string(&json).unwrap();

        let arr1 = batch.to_array2().unwrap();
        let arr2 = batch2.to_array2().unwrap();
        assert_eq!(arr1, arr2);
    }
}

// ── Protocol backends ────────────────────────────────────────────

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use crate::protocol::{DataSink, DataSource};

/// A streaming [`DataSource`] over newline-delimited JSON.
///
/// NDJSON is the only JSON shape that is a *stream*: a whole-document JSON
/// array has to be parsed in full before its first record is known, so
/// [`read_json_string`] returns a [`DataBatch`] and callers wrap it in
/// [`crate::protocol::BatchSource`] instead.
///
/// Column names are fixed by the first object seen; later objects are read
/// through those keys, and a missing key yields `NaN` — the same rule as
/// [`read_ndjson_string`].
pub struct NdjsonSource<R: BufRead> {
    reader: R,
    column_names: Option<Vec<String>>,
    exhausted: bool,
}

impl<R: BufRead> NdjsonSource<R> {
    /// Wrap any buffered reader.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            column_names: None,
            exhausted: false,
        }
    }

    /// Column names, once the first object has been read.
    pub fn column_names(&self) -> Option<&[String]> {
        self.column_names.as_deref()
    }
}

impl NdjsonSource<BufReader<File>> {
    /// Open an NDJSON file as a streaming source.
    pub fn from_path(path: &Path) -> Result<Self, IoError> {
        Ok(Self::new(BufReader::new(File::open(path)?)))
    }
}

impl<R: BufRead> DataSource for NdjsonSource<R> {
    fn read_batch(&mut self, max_records: usize) -> Result<DataBatch, IoError> {
        let mut batch = DataBatch::new();

        let mut line = String::new();
        while batch.len() < max_records {
            line.clear();
            if self.reader.read_line(&mut line)? == 0 {
                self.exhausted = true;
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let value: serde_json::Value = serde_json::from_str(trimmed)?;
            let serde_json::Value::Object(obj) = value else {
                return Err(IoError::Parse(
                    "NDJSON line is not a JSON object".to_string(),
                ));
            };

            let names = self
                .column_names
                .get_or_insert_with(|| obj.keys().cloned().collect());
            let row: Vec<f64> = names
                .iter()
                .map(|key| obj.get(key).and_then(|v| v.as_f64()).unwrap_or(f64::NAN))
                .collect();
            batch.push(DataRecord::Row(row));
        }

        batch.column_names = self.column_names.clone();
        Ok(batch)
    }

    fn has_more(&self) -> bool {
        !self.exhausted
    }
}

/// A [`DataSink`] writing newline-delimited JSON objects.
///
/// Truly streaming — each batch is written and forgotten. Rows are keyed by
/// the batch's `column_names` when present, and emitted as bare JSON arrays
/// otherwise. Non-`Row` records are skipped, matching [`batch_to_json`].
pub struct NdjsonSink<W: Write> {
    writer: W,
    column_names: Option<Vec<String>>,
    names_fixed: bool,
}

impl<W: Write> NdjsonSink<W> {
    /// Wrap any writer.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            column_names: None,
            names_fixed: false,
        }
    }
}

impl NdjsonSink<BufWriter<File>> {
    /// Create (or truncate) an NDJSON file to write into.
    pub fn create(path: &Path) -> Result<Self, IoError> {
        Ok(Self::new(BufWriter::new(File::create(path)?)))
    }
}

impl<W: Write> DataSink for NdjsonSink<W> {
    fn write_batch(&mut self, batch: &DataBatch) -> Result<(), IoError> {
        if !self.names_fixed {
            self.column_names = batch.column_names.clone();
            self.names_fixed = true;
        }

        for record in &batch.records {
            let DataRecord::Row(row) = record else {
                continue;
            };
            let value = match self.column_names {
                Some(ref names) => {
                    let obj: serde_json::Map<String, serde_json::Value> = names
                        .iter()
                        .zip(row.iter())
                        .map(|(name, &val)| (name.clone(), serde_json::json!(val)))
                        .collect();
                    serde_json::Value::Object(obj)
                }
                None => {
                    serde_json::Value::Array(row.iter().map(|&v| serde_json::json!(v)).collect())
                }
            };
            writeln!(self.writer, "{}", serde_json::to_string(&value)?)?;
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
    use crate::protocol::{pump, BatchSource};
    use std::io::Cursor;

    const SAMPLE: &str = "{\"a\":1,\"b\":2}\n\n{\"a\":3,\"b\":4}\n{\"a\":5,\"b\":6}\n";

    #[test]
    fn ndjson_source_streams_and_skips_blank_lines() {
        let mut src = NdjsonSource::new(Cursor::new(SAMPLE));

        let first = src.read_batch(2).unwrap();
        assert_eq!(
            first.len(),
            2,
            "the blank line must not end the batch early"
        );
        assert_eq!(
            first.column_names.as_deref(),
            Some(&["a".to_string(), "b".to_string()][..])
        );
        assert!(src.has_more());

        let second = src.read_batch(2).unwrap();
        assert_eq!(second.len(), 1);
        assert!(!src.has_more());
    }

    #[test]
    fn ndjson_source_rejects_a_non_object_line() {
        let mut src = NdjsonSource::new(Cursor::new("[1,2,3]\n"));
        let err = src
            .read_batch(4)
            .expect_err("a bare array is not an NDJSON record");
        assert!(err.to_string().contains("not a JSON object"), "got: {err}");
    }

    #[test]
    fn pump_ndjson_to_ndjson_round_trips() {
        let mut src = NdjsonSource::new(Cursor::new(SAMPLE));
        let mut sink = NdjsonSink::new(Vec::<u8>::new());

        assert_eq!(pump(&mut src, &mut sink, 2).unwrap(), 3);

        let out = String::from_utf8(sink.writer).unwrap();
        let reparsed = read_ndjson_string(&out).unwrap();
        assert_eq!(reparsed.len(), 3);
        assert_eq!(reparsed.column_names.as_ref().unwrap(), &["a", "b"]);
        assert_eq!(
            reparsed.to_array2().unwrap(),
            read_ndjson_string(SAMPLE).unwrap().to_array2().unwrap()
        );
    }

    /// The documented adaptation path for the batch-shaped readers: parse a
    /// whole JSON document, then serve it through the same trait a streaming
    /// source is used through.
    #[test]
    fn batch_source_bridges_whole_document_json_into_the_protocol() {
        let batch = read_json_string(r#"[{"x":1,"y":2},{"x":3,"y":4}]"#).unwrap();
        let mut src = BatchSource::new(batch);
        let mut sink = NdjsonSink::new(Vec::<u8>::new());

        assert_eq!(pump(&mut src, &mut sink, 1).unwrap(), 2);
        assert!(!src.has_more());

        let out = String::from_utf8(sink.writer).unwrap();
        assert_eq!(out.lines().count(), 2);
        assert_eq!(read_ndjson_string(&out).unwrap().len(), 2);
    }
}
