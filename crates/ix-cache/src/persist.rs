//! Disk persistence — snapshot and restore cache state.
//!
//! Similar to Redis RDB snapshots: serialize all entries to a file
//! and reload on startup.

use std::fs;
use std::io::{self, BufReader, BufWriter};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::store::Cache;

/// Serializable snapshot of a cache entry.
#[derive(Serialize, Deserialize)]
struct SnapshotEntry {
    key: String,
    data: Vec<u8>,
    /// TTL remaining in milliseconds (None = no expiry).
    ttl_ms: Option<u64>,
}

/// Snapshot metadata.
#[derive(Serialize, Deserialize)]
struct Snapshot {
    version: u32,
    created_at_unix_ms: u64,
    num_entries: usize,
    entries: Vec<SnapshotEntry>,
}

/// Save the cache to a file.
///
/// Serializes all non-expired entries with their remaining TTL.
pub fn save_snapshot(cache: &Cache, path: &Path) -> io::Result<usize> {
    let keys = cache.keys("*");
    let _now = Instant::now();
    let mut entries = Vec::new();

    for key in &keys {
        // Read the raw entry data by getting the JSON bytes
        // We use get::<serde_json::Value> to preserve the original type
        if let Some(val) = cache.get::<serde_json::Value>(key) {
            let data = serde_json::to_vec(&val).unwrap_or_default();
            let ttl_ms = cache.ttl(key).map(|d| d.as_millis() as u64);

            entries.push(SnapshotEntry {
                key: key.clone(),
                data,
                ttl_ms,
            });
        }
    }

    let num = entries.len();
    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let snapshot = Snapshot {
        version: 1,
        created_at_unix_ms: now_unix,
        num_entries: num,
        entries,
    };

    let file = fs::File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, &snapshot).map_err(io::Error::other)?;

    Ok(num)
}

/// Load a snapshot into the cache.
///
/// Restores entries with their remaining TTLs.
/// Does NOT flush existing cache entries — merges on top.
pub fn load_snapshot(cache: &Cache, path: &Path) -> io::Result<usize> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);

    let snapshot: Snapshot = serde_json::from_reader(reader)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    if snapshot.version != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported snapshot version: {}", snapshot.version),
        ));
    }

    let mut loaded = 0;
    for entry in &snapshot.entries {
        let val: serde_json::Value =
            serde_json::from_slice(&entry.data).unwrap_or(serde_json::Value::Null);

        let ttl = entry.ttl_ms.map(Duration::from_millis);
        cache.set_with_ttl(&entry.key, &val, ttl);
        loaded += 1;
    }

    Ok(loaded)
}

/// Snapshot info without loading all data.
pub fn snapshot_info(path: &Path) -> io::Result<SnapshotInfo> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);

    let snapshot: Snapshot = serde_json::from_reader(reader)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let total_bytes: usize = snapshot.entries.iter().map(|e| e.data.len()).sum();

    Ok(SnapshotInfo {
        version: snapshot.version,
        created_at_unix_ms: snapshot.created_at_unix_ms,
        num_entries: snapshot.num_entries,
        total_data_bytes: total_bytes,
    })
}

/// Metadata about a snapshot file.
#[derive(Debug, Clone)]
pub struct SnapshotInfo {
    pub version: u32,
    pub created_at_unix_ms: u64,
    pub num_entries: usize,
    pub total_data_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CacheConfig;
    use std::time::Duration;

    #[test]
    fn test_save_and_load_snapshot() {
        let cache = Cache::new(CacheConfig::default());
        cache.set("name", &"ix".to_string());
        cache.set("version", &42i64);
        cache.set("tags", &vec!["rust", "ml", "gpu"]);

        let tmp = std::env::temp_dir().join("ix_cache_test.snapshot");

        // Save
        let saved = save_snapshot(&cache, &tmp).unwrap();
        assert_eq!(saved, 3);

        // Load into fresh cache
        let cache2 = Cache::new(CacheConfig::default());
        let loaded = load_snapshot(&cache2, &tmp).unwrap();
        assert_eq!(loaded, 3);

        assert_eq!(cache2.get::<String>("name"), Some("ix".to_string()));
        assert_eq!(cache2.get::<i64>("version"), Some(42));

        // Cleanup
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_snapshot_preserves_ttl() {
        let cache = Cache::new(CacheConfig::default());
        cache.set_with_ttl("temp", &"data", Some(Duration::from_secs(300)));

        let tmp = std::env::temp_dir().join("ix_cache_ttl_test.snapshot");
        save_snapshot(&cache, &tmp).unwrap();

        let cache2 = Cache::new(CacheConfig::default());
        load_snapshot(&cache2, &tmp).unwrap();

        // Should still have a TTL
        let ttl = cache2.ttl("temp");
        assert!(ttl.is_some());
        assert!(ttl.unwrap().as_secs() > 0);

        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_snapshot_info() {
        let cache = Cache::new(CacheConfig::default());
        cache.set("a", &1);
        cache.set("b", &2);

        let tmp = std::env::temp_dir().join("ix_cache_info_test.snapshot");
        save_snapshot(&cache, &tmp).unwrap();

        let info = snapshot_info(&tmp).unwrap();
        assert_eq!(info.version, 1);
        assert_eq!(info.num_entries, 2);
        assert!(info.total_data_bytes > 0);

        let _ = fs::remove_file(&tmp);
    }
}
