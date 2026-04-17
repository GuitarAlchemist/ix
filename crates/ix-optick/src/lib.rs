//! # ix-optick
//!
//! Memory-mapped OPTIC-K v3 index reader with brute-force cosine search.
//!
//! The OPTK binary format stores pre-scaled, L2-normalized voicing vectors
//! (228 dimensions) partitioned by instrument (guitar, bass, ukulele).
//! Since vectors are pre-normalized, cosine similarity reduces to a dot product.
//!
//! ## Usage
//!
//! ```no_run
//! use ix_optick::OptickIndex;
//! use std::path::Path;
//!
//! let index = OptickIndex::open(Path::new("voicings.optk")).unwrap();
//! println!("Total voicings: {}", index.count());
//!
//! // Search across all instruments
//! let query = vec![0.0f32; 228];
//! let results = index.search(&query, None, 5).unwrap();
//! for r in &results {
//!     println!("{}: score={:.4}, diagram={}", r.index, r.score, r.metadata.diagram);
//! }
//! ```

use std::path::Path;

use memmap2::Mmap;
use serde::Deserialize;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"OPTK";
const SUPPORTED_VERSION: u32 = 3;
const ENDIAN_MARKER: u16 = 0xFEFF;
const DIMENSION: u32 = 228;
const NUM_INSTRUMENTS: usize = 3;

/// The canonical string whose CRC32 defines the expected schema hash.
/// This encodes the partition layout so readers can detect format drift.
/// MUST match GA's `OptickIndexWriter.PartitionLayout` byte-for-byte.
const SCHEMA_SEED: &str = "IDENTITY:0-5,STRUCTURE:6-29,MORPHOLOGY:30-53,CONTEXT:54-65,\
SYMBOLIC:66-77,EXTENSIONS:78-95,SPECTRAL:96-108,MODAL:109-148,\
HIERARCHY:149-163,ATONAL_MODAL:164-227";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur when opening or querying an OPTK index.
#[derive(Debug, Error)]
pub enum OptickError {
    #[error("invalid magic bytes (expected OPTK)")]
    InvalidMagic,

    #[error("unsupported version {0} (expected {SUPPORTED_VERSION})")]
    UnsupportedVersion(u32),

    #[error("endian marker mismatch (expected little-endian 0xFEFF)")]
    EndianMismatch,

    #[error("schema hash mismatch (got {got:#010x}, expected {expected:#010x})")]
    SchemaMismatch { got: u32, expected: u32 },

    #[error("invalid dimension {0} (expected {DIMENSION})")]
    InvalidDimension(u32),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("metadata parse error: {0}")]
    MetadataParseError(String),

    #[error("query dimension mismatch: got {got}, expected {expected}")]
    QueryDimensionMismatch { got: usize, expected: usize },

    #[error("unknown instrument: {0}")]
    UnknownInstrument(String),

    #[error("file too small ({size} bytes) to contain a valid header")]
    FileTooSmall { size: usize },
}

// ---------------------------------------------------------------------------
// Header types
// ---------------------------------------------------------------------------

/// Byte offset and count for a single instrument partition.
#[derive(Debug, Clone, Copy)]
pub struct InstrumentSlice {
    pub byte_offset: u64,
    pub count: u64,
}

/// Parsed OPTK v3 file header.
#[derive(Debug, Clone)]
pub struct OptickHeader {
    pub version: u32,
    pub header_size: u32,
    pub schema_hash: u32,
    pub dimension: u32,
    pub count: u64,
    pub instruments: u8,
    pub instrument_slices: [InstrumentSlice; NUM_INSTRUMENTS],
    pub vectors_offset: u64,
    pub metadata_offset: u64,
    pub metadata_length: u64,
    pub partition_weights: [f32; DIMENSION as usize],
}

// ---------------------------------------------------------------------------
// Search result / metadata
// ---------------------------------------------------------------------------

/// A single voicing metadata record deserialized from msgpack.
#[derive(Debug, Clone, Deserialize)]
pub struct VoicingMetadata {
    pub diagram: String,
    pub instrument: String,
    #[serde(rename = "midiNotes")]
    pub midi_notes: Vec<i32>,
    pub quality_inferred: Option<String>,
}

/// A search result containing the index, similarity score, and metadata.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub index: usize,
    pub score: f32,
    pub metadata: VoicingMetadata,
}

// ---------------------------------------------------------------------------
// Reader helpers
// ---------------------------------------------------------------------------

/// Read a little-endian `u16` from `buf` at `offset`. Advances `offset`.
fn read_u16(buf: &[u8], offset: &mut usize) -> u16 {
    let v = u16::from_le_bytes([buf[*offset], buf[*offset + 1]]);
    *offset += 2;
    v
}

/// Read a little-endian `u32` from `buf` at `offset`. Advances `offset`.
fn read_u32(buf: &[u8], offset: &mut usize) -> u32 {
    let v = u32::from_le_bytes(buf[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    v
}

/// Read a little-endian `u64` from `buf` at `offset`. Advances `offset`.
fn read_u64(buf: &[u8], offset: &mut usize) -> u64 {
    let v = u64::from_le_bytes(buf[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    v
}

/// Read a little-endian `f32` from `buf` at `offset`. Advances `offset`.
fn read_f32(buf: &[u8], offset: &mut usize) -> f32 {
    let v = f32::from_le_bytes(buf[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    v
}

// ---------------------------------------------------------------------------
// OptickIndex
// ---------------------------------------------------------------------------

/// A memory-mapped OPTK v3 index with brute-force cosine search.
pub struct OptickIndex {
    mmap: Mmap,
    header: OptickHeader,
}

impl std::fmt::Debug for OptickIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OptickIndex")
            .field("count", &self.header.count)
            .field("dimension", &self.header.dimension)
            .field("mmap_len", &self.mmap.len())
            .finish()
    }
}

impl OptickIndex {
    /// Open and validate an OPTK v3 index file.
    pub fn open(path: &Path) -> Result<Self, OptickError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the file is opened read-only and we treat the mapping as
        // immutable for the lifetime of `OptickIndex`.
        let mmap = unsafe { Mmap::map(&file)? };
        let header = Self::parse_header(&mmap)?;
        Ok(Self { mmap, header })
    }

    /// Create an index directly from raw bytes (useful for testing).
    #[cfg(test)]
    fn from_bytes(data: Vec<u8>) -> Result<Self, OptickError> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("optick_test_{}_{}.optk", std::process::id(), id));
        std::fs::write(&path, &data)?;
        Self::open(&path)
        // Leave the file; OS cleans up temp. Deleting while mmap'd fails on Windows.
    }

    /// Total number of voicings in the index.
    pub fn count(&self) -> u64 {
        self.header.count
    }

    /// Vector dimensionality (always 228 for OPTK v3).
    pub fn dimension(&self) -> u32 {
        self.header.dimension
    }

    /// Access the parsed header.
    pub fn header(&self) -> &OptickHeader {
        &self.header
    }

    /// Brute-force cosine search (dot product on pre-normalized vectors).
    ///
    /// - `query`: a 228-dimensional f32 vector (will be L2-normalized internally).
    /// - `instrument`: optional filter (`"guitar"`, `"bass"`, or `"ukulele"`).
    /// - `top_k`: number of results to return.
    ///
    /// Returns results sorted by descending similarity score.
    pub fn search(
        &self,
        query: &[f32],
        instrument: Option<&str>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, OptickError> {
        let dim = self.header.dimension as usize;
        if query.len() != dim {
            return Err(OptickError::QueryDimensionMismatch {
                got: query.len(),
                expected: dim,
            });
        }

        // L2-normalize the query
        let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_norm: Vec<f32> = if norm > 0.0 {
            query.iter().map(|x| x / norm).collect()
        } else {
            query.to_vec()
        };

        // Determine scan range
        let (global_start, scan_count) = match instrument {
            Some(name) => {
                let idx = instrument_index(name)
                    .ok_or_else(|| OptickError::UnknownInstrument(name.to_string()))?;
                let slice = &self.header.instrument_slices[idx];
                // The instrument byte_offset is relative to vectors_offset.
                // Compute the voicing-index start from the byte offset.
                let start = (slice.byte_offset / (dim as u64 * 4)) as usize;
                (start, slice.count as usize)
            }
            None => (0, self.header.count as usize),
        };

        // Get the vectors region as f32 slice
        let vectors_start = self.header.vectors_offset as usize;
        let vectors_end = vectors_start + (self.header.count as usize) * dim * 4;
        if vectors_end > self.mmap.len() {
            return Err(OptickError::IoError(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "vectors extend past end of file",
            )));
        }
        let vectors_bytes = &self.mmap[vectors_start..vectors_end];
        let vectors: &[f32] = bytemuck::cast_slice(vectors_bytes);

        // Brute-force dot product scan
        let k = top_k.min(scan_count);
        // Use a min-heap of size k (store negative score so BinaryHeap acts as min-heap)
        let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(OrderedF32, usize)>> =
            std::collections::BinaryHeap::with_capacity(k + 1);

        for i in 0..scan_count {
            let global_idx = global_start + i;
            let vec_offset = global_idx * dim;
            let v = &vectors[vec_offset..vec_offset + dim];
            let dot: f32 = v.iter().zip(query_norm.iter()).map(|(a, b)| a * b).sum();

            if heap.len() < k {
                heap.push(std::cmp::Reverse((OrderedF32(dot), global_idx)));
            } else if let Some(&std::cmp::Reverse((OrderedF32(min_score), _))) = heap.peek() {
                if dot > min_score {
                    heap.pop();
                    heap.push(std::cmp::Reverse((OrderedF32(dot), global_idx)));
                }
            }
        }

        // Extract and sort descending
        let mut top: Vec<(f32, usize)> = heap
            .into_iter()
            .map(|std::cmp::Reverse((OrderedF32(s), idx))| (s, idx))
            .collect();
        top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Parse metadata only for top-k
        let mut results = Vec::with_capacity(top.len());
        for (score, idx) in top {
            let metadata = self.read_metadata(idx)?;
            results.push(SearchResult {
                index: idx,
                score,
                metadata,
            });
        }

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn parse_header(buf: &[u8]) -> Result<OptickHeader, OptickError> {
        // Minimum header: magic(4) + version(4) + header_size(4) + schema_hash(4)
        // + endian(2) + reserved(2) + dimension(4) + count(8) + instruments(1) + pad(7)
        // + 3*(8+8) + vectors_offset(8) + metadata_offset(8) + metadata_length(8)
        // + 228*4 = 4+4+4+4+2+2+4+8+1+7+48+8+8+8+912 = 1024 (approximately)
        let min_header = 4 + 4 + 4 + 4 + 2 + 2 + 4 + 8 + 1 + 7
            + NUM_INSTRUMENTS * 16
            + 8 + 8 + 8
            + DIMENSION as usize * 4;

        if buf.len() < min_header {
            return Err(OptickError::FileTooSmall { size: buf.len() });
        }

        let mut off = 0usize;

        // Magic
        if &buf[0..4] != MAGIC {
            return Err(OptickError::InvalidMagic);
        }
        off += 4;

        // Version
        let version = read_u32(buf, &mut off);
        if version != SUPPORTED_VERSION {
            return Err(OptickError::UnsupportedVersion(version));
        }

        // Header size
        let header_size = read_u32(buf, &mut off);

        // Schema hash
        let schema_hash = read_u32(buf, &mut off);
        let expected_hash = compute_schema_hash();
        if schema_hash != expected_hash {
            return Err(OptickError::SchemaMismatch {
                got: schema_hash,
                expected: expected_hash,
            });
        }

        // Endian marker
        let endian = read_u16(buf, &mut off);
        if endian != ENDIAN_MARKER {
            return Err(OptickError::EndianMismatch);
        }

        // Reserved
        let _reserved = read_u16(buf, &mut off);

        // Dimension
        let dimension = read_u32(buf, &mut off);
        if dimension != DIMENSION {
            return Err(OptickError::InvalidDimension(dimension));
        }

        // Count
        let count = read_u64(buf, &mut off);

        // Instruments
        let instruments = buf[off];
        off += 1;

        // Padding (7 bytes)
        off += 7;

        // Instrument offsets
        let mut instrument_slices = [InstrumentSlice {
            byte_offset: 0,
            count: 0,
        }; NUM_INSTRUMENTS];
        for slot in instrument_slices.iter_mut() {
            slot.byte_offset = read_u64(buf, &mut off);
            slot.count = read_u64(buf, &mut off);
        }

        // vectors_offset, metadata_offset, metadata_length
        let vectors_offset = read_u64(buf, &mut off);
        let metadata_offset = read_u64(buf, &mut off);
        let metadata_length = read_u64(buf, &mut off);

        // Partition weights
        let mut partition_weights = [0.0f32; DIMENSION as usize];
        for w in partition_weights.iter_mut() {
            *w = read_f32(buf, &mut off);
        }

        Ok(OptickHeader {
            version,
            header_size,
            schema_hash,
            dimension,
            count,
            instruments,
            instrument_slices,
            vectors_offset,
            metadata_offset,
            metadata_length,
            partition_weights,
        })
    }

    /// Read and deserialize the msgpack metadata record at the given voicing index.
    fn read_metadata(&self, voicing_index: usize) -> Result<VoicingMetadata, OptickError> {
        let meta_start = self.header.metadata_offset as usize;
        let meta_end = meta_start + self.header.metadata_length as usize;

        if meta_end > self.mmap.len() {
            return Err(OptickError::IoError(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "metadata extends past end of file",
            )));
        }

        let meta_buf = &self.mmap[meta_start..meta_end];

        // msgpack records are concatenated sequentially; skip to the target index.
        let mut cursor = std::io::Cursor::new(meta_buf);
        for _ in 0..voicing_index {
            let _skip: rmp_serde::decode::Error = match rmp_serde::from_read::<_, VoicingMetadata>(&mut cursor) {
                Ok(_) => continue,
                Err(e) => {
                    return Err(OptickError::MetadataParseError(format!(
                        "failed to skip to index {voicing_index}: {e}"
                    )));
                }
            };
        }

        rmp_serde::from_read::<_, VoicingMetadata>(&mut cursor)
            .map_err(|e| OptickError::MetadataParseError(format!("index {voicing_index}: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the expected schema hash from the canonical seed string.
pub fn compute_schema_hash() -> u32 {
    crc32fast::hash(SCHEMA_SEED.as_bytes())
}

/// Map instrument name to its partition index.
fn instrument_index(name: &str) -> Option<usize> {
    match name.to_ascii_lowercase().as_str() {
        "guitar" => Some(0),
        "bass" => Some(1),
        "ukulele" => Some(2),
        _ => None,
    }
}

/// Wrapper to make f32 usable in BinaryHeap (total ordering).
#[derive(Clone, Copy, PartialEq)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 228;

    /// Instrument names for the 3 partitions.
    const INSTRUMENTS: [&str; 3] = ["guitar", "bass", "ukulele"];

    /// Build a synthetic OPTK v3 binary image with `voicings` vectors.
    /// Each voicing gets a metadata record with a diagram like "V0", "V1", etc.
    fn build_test_index(voicings: &[[f32; DIM]], instrument_counts: [usize; 3]) -> Vec<u8> {
        assert_eq!(
            voicings.len(),
            instrument_counts.iter().sum::<usize>(),
            "voicing count must match sum of instrument counts"
        );

        let mut buf = Vec::new();

        // --- Header ---
        // magic
        buf.extend_from_slice(b"OPTK");
        // version
        buf.extend_from_slice(&3u32.to_le_bytes());
        // header_size (placeholder, we patch it later)
        let header_size_pos = buf.len();
        buf.extend_from_slice(&0u32.to_le_bytes());
        // schema_hash
        buf.extend_from_slice(&compute_schema_hash().to_le_bytes());
        // endian marker
        buf.extend_from_slice(&0xFEFFu16.to_le_bytes());
        // reserved
        buf.extend_from_slice(&0u16.to_le_bytes());
        // dimension
        buf.extend_from_slice(&(DIM as u32).to_le_bytes());
        // count
        buf.extend_from_slice(&(voicings.len() as u64).to_le_bytes());
        // instruments
        buf.push(3u8);
        // pad (7 bytes)
        buf.extend_from_slice(&[0u8; 7]);

        // instrument offsets: byte_offset relative to vectors start, count
        let mut running = 0usize;
        for &c in &instrument_counts {
            let byte_off = (running * DIM * 4) as u64;
            buf.extend_from_slice(&byte_off.to_le_bytes());
            buf.extend_from_slice(&(c as u64).to_le_bytes());
            running += c;
        }

        // vectors_offset (placeholder)
        let vec_off_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes());
        // metadata_offset (placeholder)
        let meta_off_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes());
        // metadata_length (placeholder)
        let meta_len_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes());

        // partition_weights (all 1.0 for testing)
        for _ in 0..DIM {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }

        // Patch header_size
        let header_size = buf.len() as u32;
        buf[header_size_pos..header_size_pos + 4].copy_from_slice(&header_size.to_le_bytes());

        // --- Vectors ---
        let vectors_offset = buf.len() as u64;
        buf[vec_off_pos..vec_off_pos + 8].copy_from_slice(&vectors_offset.to_le_bytes());

        for v in voicings {
            for &x in v.iter() {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }

        // --- Metadata (msgpack) ---
        let metadata_offset = buf.len() as u64;
        buf[meta_off_pos..meta_off_pos + 8].copy_from_slice(&metadata_offset.to_le_bytes());

        let mut inst_idx = 0usize;
        let mut count_in_inst = 0usize;
        for (i, _v) in voicings.iter().enumerate() {
            while inst_idx < 3 && count_in_inst >= instrument_counts[inst_idx] {
                inst_idx += 1;
                count_in_inst = 0;
            }
            let inst_name = INSTRUMENTS[inst_idx.min(2)];
            count_in_inst += 1;

            let meta = serde_json::json!({
                "diagram": format!("V{i}"),
                "instrument": inst_name,
                "midiNotes": [60 + i as i32],
                "quality_inferred": null,
            });
            // Serialize to msgpack
            let packed = rmp_serde::to_vec(&meta).unwrap();
            buf.extend_from_slice(&packed);
        }

        let metadata_length = buf.len() as u64 - metadata_offset;
        buf[meta_len_pos..meta_len_pos + 8].copy_from_slice(&metadata_length.to_le_bytes());

        buf
    }

    /// Create a normalized f32 vector with a spike at a given dimension.
    fn spike_vector(spike_dim: usize, value: f32) -> [f32; DIM] {
        let mut v = [0.0f32; DIM];
        v[spike_dim] = value;
        // L2 normalize
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        v
    }

    #[test]
    fn test_header_parsing() {
        let v0 = spike_vector(0, 1.0);
        let v1 = spike_vector(1, 1.0);
        let v2 = spike_vector(2, 1.0);
        let data = build_test_index(&[v0, v1, v2], [1, 1, 1]);

        let index = OptickIndex::from_bytes(data).expect("should parse header");
        assert_eq!(index.count(), 3);
        assert_eq!(index.dimension(), 228);
        assert_eq!(index.header().version, 3);
        assert_eq!(index.header().instruments, 3);
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = build_test_index(&[spike_vector(0, 1.0)], [1, 0, 0]);
        data[0] = b'X'; // corrupt magic
        let err = OptickIndex::from_bytes(data).unwrap_err();
        assert!(matches!(err, OptickError::InvalidMagic));
    }

    #[test]
    fn test_cosine_search_ordering() {
        // Create 3 vectors with spikes in different dimensions
        let v0 = spike_vector(0, 1.0); // guitar
        let v1 = spike_vector(1, 1.0); // bass
        let v2 = spike_vector(2, 1.0); // ukulele
        let data = build_test_index(&[v0, v1, v2], [1, 1, 1]);

        let index = OptickIndex::from_bytes(data).expect("should open");

        // Query aligned with v0 should rank v0 first
        let mut query = [0.0f32; DIM];
        query[0] = 1.0;
        let results = index.search(&query, None, 3).expect("search");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].index, 0);
        assert!((results[0].score - 1.0).abs() < 1e-5, "perfect match score should be ~1.0");

        // Query aligned with v2 should rank v2 first
        let mut query2 = [0.0f32; DIM];
        query2[2] = 1.0;
        let results2 = index.search(&query2, None, 1).expect("search");
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0].index, 2);
    }

    #[test]
    fn test_instrument_filtering() {
        let v0 = spike_vector(0, 1.0); // guitar
        let v1 = spike_vector(1, 1.0); // guitar
        let v2 = spike_vector(2, 1.0); // bass
        let data = build_test_index(&[v0, v1, v2], [2, 1, 0]);

        let index = OptickIndex::from_bytes(data).expect("should open");

        // Search only bass partition
        let mut query = [0.0f32; DIM];
        query[2] = 1.0;
        let results = index.search(&query, Some("bass"), 5).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 2);
        assert_eq!(results[0].metadata.instrument, "bass");

        // Search only guitar partition
        let guitar_results = index.search(&query, Some("guitar"), 5).expect("search");
        assert_eq!(guitar_results.len(), 2);
        // All results should be from guitar
        for r in &guitar_results {
            assert_eq!(r.metadata.instrument, "guitar");
        }
    }

    #[test]
    fn test_query_dimension_mismatch() {
        let data = build_test_index(&[spike_vector(0, 1.0)], [1, 0, 0]);
        let index = OptickIndex::from_bytes(data).expect("should open");

        let bad_query = vec![0.0f32; 100]; // wrong dimension
        let err = index.search(&bad_query, None, 1).unwrap_err();
        assert!(matches!(
            err,
            OptickError::QueryDimensionMismatch { got: 100, expected: 228 }
        ));
    }

    #[test]
    fn test_unknown_instrument() {
        let data = build_test_index(&[spike_vector(0, 1.0)], [1, 0, 0]);
        let index = OptickIndex::from_bytes(data).expect("should open");

        let query = vec![0.0f32; DIM];
        let err = index.search(&query, Some("banjo"), 1).unwrap_err();
        assert!(matches!(err, OptickError::UnknownInstrument(_)));
    }

    #[test]
    fn test_schema_hash_is_deterministic() {
        let h1 = compute_schema_hash();
        let h2 = compute_schema_hash();
        assert_eq!(h1, h2);
        assert_ne!(h1, 0); // sanity: should not be zero
    }

    #[test]
    fn test_metadata_content() {
        let v0 = spike_vector(0, 1.0);
        let data = build_test_index(&[v0], [1, 0, 0]);
        let index = OptickIndex::from_bytes(data).expect("should open");

        let mut query = [0.0f32; DIM];
        query[0] = 1.0;
        let results = index.search(&query, None, 1).expect("search");
        assert_eq!(results[0].metadata.diagram, "V0");
        assert_eq!(results[0].metadata.instrument, "guitar");
        assert_eq!(results[0].metadata.midi_notes, vec![60]);
        assert!(results[0].metadata.quality_inferred.is_none());
    }
}
