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
