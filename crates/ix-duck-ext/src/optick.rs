//! `ix_optick_scan(index_path)` — the production OPTIC-K voicing index (a binary
//! mmap) exposed as a DuckDB **table function**, the differentiated Tier-3 payload.
//!
//! `optick.index` is the search engine's mmap, not DuckDB-readable; this opens it
//! read-only via `ix_optick::OptickIndex` (once, in `bind`) and streams one row per
//! voicing: `(voicing BIGINT, instrument VARCHAR, embedding DOUBLE[], midi_notes BIGINT[])`.
//!
//! Voicing distance needs no dedicated UDF — compose the already-registered
//! `ix_euclidean` over two scanned embeddings:
//!
//! ```sql
//! SELECT ix_euclidean(a.embedding, b.embedding)
//! FROM ix_optick_scan('…/optick.index') a, ix_optick_scan('…/optick.index') b
//! WHERE a.voicing = 100 AND b.voicing = 200;
//! ```
//!
//! `midi_notes` (the decoded voicing pitches) makes the corpus crunchable by the
//! set-theory UDFs without any export — e.g. realization density per set class, or
//! voice-leading cost to a target chord:
//!
//! ```sql
//! SELECT ix_forte_number(midi_notes) AS set_class, count(*) AS voicings
//! FROM ix_optick_scan('…/optick.index') GROUP BY 1 ORDER BY voicings DESC;
//!
//! SELECT voicing, ix_icv_l1(midi_notes, [0,4,7,11]) AS cost
//! FROM ix_optick_scan('…/optick.index') ORDER BY cost LIMIT 20;
//! ```

use std::ffi::CString;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use ix_optick::OptickIndex;

#[repr(C)]
pub struct OptickScanBind {
    index: OptickIndex,
}
#[repr(C)]
pub struct OptickScanInit {
    cursor: AtomicUsize,
}

pub struct IxOptickScan;

impl VTab for IxOptickScan {
    type InitData = OptickScanInit;
    type BindData = OptickScanBind;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let path = bind.get_parameter(0).to_string();
        let index = OptickIndex::open(Path::new(&path))?;
        bind.add_result_column("voicing", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        bind.add_result_column("instrument", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column(
            "embedding",
            LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Double)),
        );
        bind.add_result_column(
            "midi_notes",
            LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Bigint)),
        );
        Ok(OptickScanBind { index })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        Ok(OptickScanInit { cursor: AtomicUsize::new(0) })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bind = func.get_bind_data();
        let init = func.get_init_data();
        let n = bind.index.count() as usize;
        let dim = bind.index.dimension() as usize;
        let start = init.cursor.load(Ordering::Relaxed);
        if start >= n {
            output.set_len(0);
            return Ok(());
        }
        let cap = output.flat_vector(0).capacity();
        let rows = (n - start).min(cap);

        // col 0: voicing index (BIGINT)
        {
            let mut v = output.flat_vector(0);
            let s = unsafe { v.as_mut_slice_with_len::<i64>(rows) };
            for (i, slot) in s.iter_mut().enumerate().take(rows) {
                *slot = (start + i) as i64;
            }
        }
        // col 1: instrument (VARCHAR)
        {
            let v = output.flat_vector(1);
            for i in 0..rows {
                let instr = bind.index.instrument_of(start + i).unwrap_or("unknown");
                v.insert(i, CString::new(instr)?);
            }
        }
        // col 2: embedding (LIST<DOUBLE>) — flatten this chunk into the child buffer,
        // then point each row's (offset, length) entry at its slice (cf. ix_pca_project).
        {
            let total = rows * dim;
            let flat: Vec<f64> = (0..rows)
                .flat_map(|i| {
                    bind.index
                        .vector(start + i)
                        .unwrap_or(&[])
                        .iter()
                        .map(|&x| x as f64)
                })
                .collect();
            let mut lv = output.list_vector(2);
            {
                let mut child = lv.child(total);
                let cslice = unsafe { child.as_mut_slice_with_len::<f64>(total) };
                cslice[..total].copy_from_slice(&flat);
            }
            lv.set_len(total);
            for i in 0..rows {
                lv.set_entry(i, i * dim, dim);
            }
        }
        // col 3: midi_notes (LIST<BIGINT>) — decoded from the msgpack metadata per
        // voicing (variable length, unlike the fixed-dim embedding). A row whose
        // metadata fails to decode yields an empty list rather than aborting the scan.
        {
            let per_row: Vec<Vec<i64>> = (0..rows)
                .map(|i| {
                    bind.index
                        .metadata(start + i)
                        .map(|m| m.midi_notes.iter().map(|&x| x as i64).collect())
                        .unwrap_or_default()
                })
                .collect();
            let total: usize = per_row.iter().map(Vec::len).sum();
            let mut lv = output.list_vector(3);
            {
                let mut child = lv.child(total);
                let cslice = unsafe { child.as_mut_slice_with_len::<i64>(total) };
                let mut off = 0usize;
                for notes in &per_row {
                    cslice[off..off + notes.len()].copy_from_slice(notes);
                    off += notes.len();
                }
            }
            lv.set_len(total);
            let mut off = 0usize;
            for (i, notes) in per_row.iter().enumerate() {
                lv.set_entry(i, off, notes.len());
                off += notes.len();
            }
        }
        output.set_len(rows);
        init.cursor.store(start + rows, Ordering::Relaxed);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}
