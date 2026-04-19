//! # ix-optick
//!
//! Memory-mapped OPTIC-K v4 index reader with brute-force cosine search.
//!
//! The OPTK v4 binary format stores pre-scaled, L2-normalized voicing vectors
//! (112 dimensions — search-relevant partitions only) partitioned by instrument
//! (guitar, bass, ukulele). Since vectors are pre-normalized, cosine similarity
//! reduces to a dot product.
//!
//! ## V4 improvements over v3
//!
//! - Dimension reduced 228 → 112 (info-only partitions dropped)
//! - Per-voicing metadata offset table enables O(1) metadata fetch
//! - ~55% smaller files, no search-quality regression
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
//! let query = vec![0.0f32; 112];
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
const SUPPORTED_VERSION: u32 = 4;
const ENDIAN_MARKER: u16 = 0xFEFF;
const DIMENSION: u32 = 112;
const NUM_INSTRUMENTS: usize = 3;

/// The canonical string whose CRC32 defines the expected schema hash.
/// V4 drops info-only partitions and uses dense (compact) indexing.
/// The `-pp` suffix (per-partition normalization) distinguishes v4-pp from
/// the original v4: partition layout is identical, but each partition slice
/// is L2-normalized independently before sqrt-weight scaling. This closes
/// cross-instrument STRUCTURE leaks that failed invariants #25/#28/#32.
/// MUST match GA's `EmbeddingSchema.BuildCompactLayoutV4` byte-for-byte.
const SCHEMA_SEED: &str =
    "optk-v4-pp:STRUCTURE:0-23,MORPHOLOGY:24-47,CONTEXT:48-59,\
SYMBOLIC:60-71,MODAL:72-111";

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

/// Parsed OPTK v4 file header.
#[derive(Debug, Clone)]
pub struct OptickHeader {
    pub version: u32,
    pub header_size: u32,
    pub schema_hash: u32,
    pub dimension: u32,
    pub count: u64,
    pub instruments: u8,
    pub instrument_slices: [InstrumentSlice; NUM_INSTRUMENTS],
    pub metadata_offsets_offset: u64,
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

    /// Vector dimensionality (always 112 for OPTK v4).
    pub fn dimension(&self) -> u32 {
        self.header.dimension
    }

    /// Access the parsed header.
    pub fn header(&self) -> &OptickHeader {
        &self.header
    }

    /// Access the raw vector data as a flat `&[f32]` slice of length
    /// `count * dimension`. Vector `i` is at `[i * dim .. (i + 1) * dim]`.
    ///
    /// Vectors are pre-scaled (partition weights applied) and L2-normalized
    /// at write time. Useful for offline analysis (clustering, classification,
    /// topological diagnostics) without going through brute-force search.
    pub fn vectors(&self) -> &[f32] {
        let dim = self.header.dimension as usize;
        let vectors_start = self.header.vectors_offset as usize;
        let vectors_end = vectors_start + (self.header.count as usize) * dim * 4;
        let vectors_bytes = &self.mmap[vectors_start..vectors_end];
        bytemuck::cast_slice(vectors_bytes)
    }

    /// Return a read-only view over a single voicing's vector.
    pub fn vector(&self, voicing_index: usize) -> Option<&[f32]> {
        let count = self.header.count as usize;
        if voicing_index >= count {
            return None;
        }
        let dim = self.header.dimension as usize;
        let all = self.vectors();
        Some(&all[voicing_index * dim..(voicing_index + 1) * dim])
    }

    /// Public metadata accessor — reads and deserializes the msgpack record
    /// for a single voicing using the v4 O(1) offset table.
    pub fn metadata(&self, voicing_index: usize) -> Result<VoicingMetadata, OptickError> {
        self.read_metadata(voicing_index)
    }

    /// Return the instrument label for a given global voicing index by looking
    /// it up in the instrument slices (no msgpack decode).
    ///
    /// Voicings are contiguous per instrument in the v4 layout.
    pub fn instrument_of(&self, voicing_index: usize) -> Option<&'static str> {
        let dim = self.header.dimension as usize;
        let vec_bytes_per = (dim as u64) * 4;
        let vectors_offset = self.header.vectors_offset;
        for (i, slice) in self.header.instrument_slices.iter().enumerate() {
            if slice.count == 0 {
                continue;
            }
            let rel_bytes = slice.byte_offset.saturating_sub(vectors_offset);
            let start = (rel_bytes / vec_bytes_per) as usize;
            let end = start + slice.count as usize;
            if voicing_index >= start && voicing_index < end {
                return Some(match i {
                    0 => "guitar",
                    1 => "bass",
                    2 => "ukulele",
                    _ => return None,
                });
            }
        }
        None
    }

    /// Brute-force cosine search (dot product on pre-normalized vectors).
    ///
    /// - `query`: a 112-dimensional f32 vector (will be L2-normalized internally).
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

        // Determine scan range.
        // V4: instrument byte_offsets are ABSOLUTE (relative to start of file).
        // Convert to voicing-index by subtracting vectors_offset.
        let vectors_offset = self.header.vectors_offset;
        let (global_start, scan_count) = match instrument {
            Some(name) => {
                let idx = instrument_index(name)
                    .ok_or_else(|| OptickError::UnknownInstrument(name.to_string()))?;
                let slice = &self.header.instrument_slices[idx];
                let rel_bytes = slice.byte_offset.saturating_sub(vectors_offset);
                let start = (rel_bytes / (dim as u64 * 4)) as usize;
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
        // V4 header: magic(4) + version(4) + header_size(4) + schema_hash(4)
        // + endian(2) + reserved(2) + dimension(4) + count(8) + instruments(1) + pad(7)
        // + 3*(8+8) + metadata_offsets_offset(8) + vectors_offset(8) + metadata_offset(8)
        // + metadata_length(8) + 112*4 = 4+4+4+4+2+2+4+8+1+7+48+8+8+8+8+448 = 568 bytes
        let min_header = 4 + 4 + 4 + 4 + 2 + 2 + 4 + 8 + 1 + 7
            + NUM_INSTRUMENTS * 16
            + 8 + 8 + 8 + 8
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

        // v4: metadata_offsets_offset, then vectors_offset, metadata_offset, metadata_length
        let metadata_offsets_offset = read_u64(buf, &mut off);
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
            metadata_offsets_offset,
            vectors_offset,
            metadata_offset,
            metadata_length,
            partition_weights,
        })
    }

    /// Read and deserialize the msgpack metadata record at the given voicing index.
    /// Uses the v4 per-record offset table for O(1) access.
    fn read_metadata(&self, voicing_index: usize) -> Result<VoicingMetadata, OptickError> {
        let count = self.header.count as usize;
        if voicing_index >= count {
            return Err(OptickError::MetadataParseError(format!(
                "voicing index {voicing_index} out of range (count={count})"
            )));
        }

        // Read the relative offset for this voicing from the offset table.
        let offsets_base = self.header.metadata_offsets_offset as usize;
        let entry_pos = offsets_base + voicing_index * 8;
        if entry_pos + 8 > self.mmap.len() {
            return Err(OptickError::IoError(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "metadata offsets table extends past end of file",
            )));
        }
        let rel_off = u64::from_le_bytes(
            self.mmap[entry_pos..entry_pos + 8].try_into().unwrap()
        ) as usize;

        // The msgpack record starts at metadata_offset + rel_off.
        let meta_start = self.header.metadata_offset as usize;
        let meta_end = meta_start + self.header.metadata_length as usize;
        if meta_end > self.mmap.len() {
            return Err(OptickError::IoError(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "metadata extends past end of file",
            )));
        }

        let record_start = meta_start + rel_off;
        if record_start >= meta_end {
            return Err(OptickError::MetadataParseError(format!(
                "record offset {rel_off} out of metadata range"
            )));
        }

        let mut cursor = std::io::Cursor::new(&self.mmap[record_start..meta_end]);
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

    const DIM: usize = 112;

    /// Instrument names for the 3 partitions.
    const INSTRUMENTS: [&str; 3] = ["guitar", "bass", "ukulele"];

    /// Build a synthetic OPTK v4 binary image with `voicings` vectors.
    /// Each voicing gets a metadata record with a diagram like "V0", "V1", etc.
    fn build_test_index(voicings: &[[f32; DIM]], instrument_counts: [usize; 3]) -> Vec<u8> {
        assert_eq!(
            voicings.len(),
            instrument_counts.iter().sum::<usize>(),
            "voicing count must match sum of instrument counts"
        );

        let mut buf = Vec::new();
        let count = voicings.len();

        // --- Header ---
        buf.extend_from_slice(b"OPTK");                          // magic
        buf.extend_from_slice(&4u32.to_le_bytes());              // version
        let header_size_pos = buf.len();
        buf.extend_from_slice(&0u32.to_le_bytes());              // header_size (patched)
        buf.extend_from_slice(&compute_schema_hash().to_le_bytes()); // schema_hash
        buf.extend_from_slice(&0xFEFFu16.to_le_bytes());         // endian marker
        buf.extend_from_slice(&0u16.to_le_bytes());              // reserved
        buf.extend_from_slice(&(DIM as u32).to_le_bytes());      // dimension
        buf.extend_from_slice(&(count as u64).to_le_bytes());    // count
        buf.push(3u8);                                            // instruments
        buf.extend_from_slice(&[0u8; 7]);                        // pad

        // Instrument offsets — ABSOLUTE byte offsets (patched once we know header size)
        let inst_offsets_pos = buf.len();
        for _ in 0..3 {
            buf.extend_from_slice(&0u64.to_le_bytes());          // byte_offset (patched)
            buf.extend_from_slice(&0u64.to_le_bytes());          // count (patched)
        }

        let metadata_offsets_offset_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes());              // metadata_offsets_offset (patched)
        let vec_off_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes());              // vectors_offset (patched)
        let meta_off_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes());              // metadata_offset (patched)
        let meta_len_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes());              // metadata_length (patched)

        // partition_weights (all 1.0 for testing — vectors already normalized)
        for _ in 0..DIM {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }

        // Patch header_size
        let header_size = buf.len() as u32;
        buf[header_size_pos..header_size_pos + 4].copy_from_slice(&header_size.to_le_bytes());

        // --- Metadata offsets table ---
        let metadata_offsets_offset = buf.len() as u64;
        buf[metadata_offsets_offset_pos..metadata_offsets_offset_pos + 8]
            .copy_from_slice(&metadata_offsets_offset.to_le_bytes());

        // Reserve space; patch after we compute metadata record offsets.
        let offsets_table_start = buf.len();
        for _ in 0..count {
            buf.extend_from_slice(&0u64.to_le_bytes());
        }

        // --- Vectors ---
        let vectors_offset = buf.len() as u64;
        buf[vec_off_pos..vec_off_pos + 8].copy_from_slice(&vectors_offset.to_le_bytes());

        for v in voicings {
            for &x in v.iter() {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }

        // Patch instrument offsets (absolute)
        let mut running = 0u64;
        let mut pos = inst_offsets_pos;
        for &c in &instrument_counts {
            let byte_off = vectors_offset + running;
            buf[pos..pos + 8].copy_from_slice(&byte_off.to_le_bytes());
            buf[pos + 8..pos + 16].copy_from_slice(&(c as u64).to_le_bytes());
            running += (c * DIM * 4) as u64;
            pos += 16;
        }

        // --- Metadata (msgpack) + patch offsets table ---
        let metadata_offset = buf.len() as u64;
        buf[meta_off_pos..meta_off_pos + 8].copy_from_slice(&metadata_offset.to_le_bytes());

        let mut inst_idx = 0usize;
        let mut count_in_inst = 0usize;
        let mut record_rel_offsets = Vec::with_capacity(count);
        for (i, _v) in voicings.iter().enumerate() {
            while inst_idx < 3 && count_in_inst >= instrument_counts[inst_idx] {
                inst_idx += 1;
                count_in_inst = 0;
            }
            let inst_name = INSTRUMENTS[inst_idx.min(2)];
            count_in_inst += 1;

            let rel = (buf.len() as u64) - metadata_offset;
            record_rel_offsets.push(rel);

            let meta = serde_json::json!({
                "diagram": format!("V{i}"),
                "instrument": inst_name,
                "midiNotes": [60 + i as i32],
                "quality_inferred": null,
            });
            let packed = rmp_serde::to_vec(&meta).unwrap();
            buf.extend_from_slice(&packed);
        }

        let metadata_length = (buf.len() as u64) - metadata_offset;
        buf[meta_len_pos..meta_len_pos + 8].copy_from_slice(&metadata_length.to_le_bytes());

        // Patch the offsets table
        for (i, rel) in record_rel_offsets.iter().enumerate() {
            let p = offsets_table_start + i * 8;
            buf[p..p + 8].copy_from_slice(&rel.to_le_bytes());
        }

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
        assert_eq!(index.dimension(), 112);
        assert_eq!(index.header().version, 4);
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
            OptickError::QueryDimensionMismatch { got: 100, expected: 112 }
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
