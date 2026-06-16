//! Probabilistic data-structure UDFs over `ix-probabilistic`.
//!
//! DuckDB natively has `approx_count_distinct` (a HyperLogLog) and uses Bloom
//! filters internally for joins, but it exposes **none** of them as queryable,
//! persistable, mergeable objects. These UDFs do: each sketch is a *portable
//! column value* you build once, store as a tiny JSON blob, then probe cheaply
//! with a scalar — or merge across partitions / repos.
//!
//! The duckdb-rs build has no aggregate-UDF support, so every structure follows a
//! **build → blob → probe/merge** triad of scalar functions (build materialises a
//! column via `list()`):
//!
//! | structure | build | query | combine |
//! |---|---|---|---|
//! | Bloom    | `ix_bloom_build(items, capacity, fp_rate)` | `ix_bloom_contains(sketch, item)` | `ix_bloom_union(a, b)` |
//! | HLL      | `ix_hll_build(items, precision)`          | `ix_hll_count(sketch)`            | `ix_hll_merge(a, b)`   |
//! | Count-Min| `ix_cms_build(items, epsilon, delta)`     | `ix_cms_estimate(sketch, item)`   | `ix_cms_merge(a, b)`   |
//! | Cuckoo   | `ix_cuckoo_build(items, capacity)`        | `ix_cuckoo_contains(sketch, item)`| `ix_cuckoo_remove(sketch, item)` |
//!
//! Items are `BIGINT` so build-vs-probe hashing is identical; for text keys bridge
//! through DuckDB's own deterministic `hash()`: `ix_bloom_contains(s, hash(q)::BIGINT)`.
//! Blobs are JSON (`serde`); `ix-probabilistic` hashes with a fixed-seed
//! `DefaultHasher`, so a blob probes identically on any machine (same Rust std).
//! Pure wraps of `ix-probabilistic` — no algorithm code here.

use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::ffi::duckdb_string_t;
use duckdb::types::DuckString;
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use duckdb::Connection;
use ix_probabilistic::bloom::BloomFilter;
use ix_probabilistic::count_min::CountMinSketch;
use ix_probabilistic::cuckoo::CuckooFilter;
use ix_probabilistic::hyperloglog::HyperLogLog;
use std::error::Error;
use std::ffi::CString;

type Res = Result<(), Box<dyn Error>>;

fn list_bigint() -> LogicalTypeHandle {
    LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Bigint))
}
fn ty(id: LogicalTypeId) -> LogicalTypeHandle {
    LogicalTypeHandle::from(id)
}

// ── readers ──────────────────────────────────────────────────────────────────

/// Read a `LIST<BIGINT>` column into one owned `Vec<i64>` per row (raw values, no
/// reduction). Reads the per-row `(offset, length)` entries before borrowing the
/// child buffer so the slices never alias; `cap = max(offset + length)` keeps
/// every `all[o..o+l]` in bounds.
fn read_bigint_list(input: &mut DataChunkHandle, col: usize, n: usize) -> Vec<Vec<i64>> {
    let lv = input.list_vector(col);
    let entries: Vec<(usize, usize)> = (0..n).map(|i| lv.get_entry(i)).collect();
    let cap = entries.iter().map(|(o, l)| o + l).max().unwrap_or(0);
    let child = lv.child(cap);
    let all = unsafe { child.as_slice_with_len::<i64>(cap) };
    entries.iter().map(|&(o, l)| all[o..o + l].to_vec()).collect()
}

/// Read a `VARCHAR` column into owned `String`s (one per row).
fn read_varchar_col(input: &mut DataChunkHandle, col: usize, n: usize) -> Vec<String> {
    let v = input.flat_vector(col);
    let slice = unsafe { v.as_slice_with_len::<duckdb_string_t>(n) };
    slice
        .iter()
        .map(|ptr| DuckString::new(&mut { *ptr }).as_str().to_string())
        .collect()
}

/// Read a flat `BIGINT` column into `Vec<i64>` (copied out of the borrowed buffer).
fn read_i64_col(input: &mut DataChunkHandle, col: usize, n: usize) -> Vec<i64> {
    let v = input.flat_vector(col);
    unsafe { v.as_slice_with_len::<i64>(n) }.to_vec()
}

/// Read a flat `DOUBLE` column into `Vec<f64>`.
fn read_f64_col(input: &mut DataChunkHandle, col: usize, n: usize) -> Vec<f64> {
    let v = input.flat_vector(col);
    unsafe { v.as_slice_with_len::<f64>(n) }.to_vec()
}

// ── writers ──────────────────────────────────────────────────────────────────

fn write_varchar(output: &mut dyn WritableVector, vals: &[String]) -> Res {
    let out = output.flat_vector();
    for (i, v) in vals.iter().enumerate() {
        out.insert(i, CString::new(v.as_str())?);
    }
    Ok(())
}

fn write_bool(output: &mut dyn WritableVector, vals: &[bool]) {
    let mut out = output.flat_vector();
    let slice = unsafe { out.as_mut_slice_with_len::<bool>(vals.len()) };
    slice.copy_from_slice(vals);
}

fn write_i64(output: &mut dyn WritableVector, vals: &[i64]) {
    let mut out = output.flat_vector();
    let slice = unsafe { out.as_mut_slice_with_len::<i64>(vals.len()) };
    slice.copy_from_slice(vals);
}

// ── signatures ───────────────────────────────────────────────────────────────

/// `(VARCHAR sketch, BIGINT item) -> ret`.
fn probe_sig(ret: LogicalTypeId) -> Vec<ScalarFunctionSignature> {
    vec![ScalarFunctionSignature::exact(
        vec![ty(LogicalTypeId::Varchar), ty(LogicalTypeId::Bigint)],
        ty(ret),
    )]
}
/// `(VARCHAR a, VARCHAR b) -> VARCHAR`.
fn combine_sig() -> Vec<ScalarFunctionSignature> {
    vec![ScalarFunctionSignature::exact(
        vec![ty(LogicalTypeId::Varchar), ty(LogicalTypeId::Varchar)],
        ty(LogicalTypeId::Varchar),
    )]
}

// ── Bloom filter ─────────────────────────────────────────────────────────────

struct IxBloomBuild;
impl VScalar for IxBloomBuild {
    type State = ();
    // @ai:invariant ix_bloom_build wraps ix_probabilistic BloomFilter::new(capacity, fp_rate) + insert each BIGINT item, returns a JSON blob; capacity<1 or fp_rate∉(0,1) -> SQL error (no panic on size 0) [T:test conf:0.85 src:ix_duck::sketch::tests::bloom_roundtrip_no_false_negative]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let lists = read_bigint_list(input, 0, n);
        let caps = read_i64_col(input, 1, n);
        let fps = read_f64_col(input, 2, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            if caps[i] < 1 {
                return Err("ix_bloom_build: capacity must be >= 1".into());
            }
            if !(fps[i] > 0.0 && fps[i] < 1.0) {
                return Err("ix_bloom_build: fp_rate must be in (0, 1)".into());
            }
            let mut bf = BloomFilter::new(caps[i] as usize, fps[i]);
            for v in &lists[i] {
                bf.insert(v);
            }
            out.push(serde_json::to_string(&bf)?);
        }
        write_varchar(output, &out)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![list_bigint(), ty(LogicalTypeId::Bigint), ty(LogicalTypeId::Double)],
            ty(LogicalTypeId::Varchar),
        )]
    }
}

struct IxBloomContains;
impl VScalar for IxBloomContains {
    type State = ();
    // @ai:invariant ix_bloom_contains deserializes the blob and returns BloomFilter::contains(item); every inserted item -> true (no false negatives); malformed blob -> SQL error [T:test conf:0.9 src:ix_duck::sketch::tests::bloom_roundtrip_no_false_negative]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let sketches = read_varchar_col(input, 0, n);
        let items = read_i64_col(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let bf: BloomFilter = serde_json::from_str(&sketches[i])
                .map_err(|e| format!("ix_bloom_contains: invalid sketch blob: {e}"))?;
            out.push(bf.contains(&items[i]));
        }
        write_bool(output, &out);
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        probe_sig(LogicalTypeId::Boolean)
    }
}

struct IxBloomUnion;
impl VScalar for IxBloomUnion {
    type State = ();
    // @ai:invariant ix_bloom_union returns BloomFilter::union of two blobs; mismatched parameters (capacity/fp_rate) -> SQL error; result contains every item from both [T:test conf:0.85 src:ix_duck::sketch::tests::bloom_union]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let a = read_varchar_col(input, 0, n);
        let b = read_varchar_col(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let fa: BloomFilter = serde_json::from_str(&a[i])
                .map_err(|e| format!("ix_bloom_union: invalid sketch blob (arg 1): {e}"))?;
            let fb: BloomFilter = serde_json::from_str(&b[i])
                .map_err(|e| format!("ix_bloom_union: invalid sketch blob (arg 2): {e}"))?;
            let u = fa
                .union(&fb)
                .ok_or("ix_bloom_union: filters differ in capacity/fp_rate (must match to union)")?;
            out.push(serde_json::to_string(&u)?);
        }
        write_varchar(output, &out)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        combine_sig()
    }
}

// ── HyperLogLog ──────────────────────────────────────────────────────────────

struct IxHllBuild;
impl VScalar for IxHllBuild {
    type State = ();
    // @ai:invariant ix_hll_build wraps HyperLogLog::new(precision) + add each BIGINT item; precision is clamped to [4,18] by the lib; precision<1 -> SQL error [T:test conf:0.8 src:ix_duck::sketch::tests::hll_count_estimate]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let lists = read_bigint_list(input, 0, n);
        let precisions = read_i64_col(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            if precisions[i] < 1 {
                return Err("ix_hll_build: precision must be >= 1".into());
            }
            let mut hll = HyperLogLog::new(precisions[i] as usize);
            for v in &lists[i] {
                hll.add(v);
            }
            out.push(serde_json::to_string(&hll)?);
        }
        write_varchar(output, &out)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![list_bigint(), ty(LogicalTypeId::Bigint)],
            ty(LogicalTypeId::Varchar),
        )]
    }
}

struct IxHllCount;
impl VScalar for IxHllCount {
    type State = ();
    // @ai:invariant ix_hll_count returns the rounded HyperLogLog::count cardinality estimate of the blob; ~within the lib's error bound of the true distinct count [T:test conf:0.8 src:ix_duck::sketch::tests::hll_count_estimate]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let sketches = read_varchar_col(input, 0, n);
        let mut out = Vec::with_capacity(n);
        for s in &sketches {
            let hll: HyperLogLog = serde_json::from_str(s)
                .map_err(|e| format!("ix_hll_count: invalid sketch blob: {e}"))?;
            out.push(hll.count().round() as i64);
        }
        write_i64(output, &out);
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![ty(LogicalTypeId::Varchar)],
            ty(LogicalTypeId::Bigint),
        )]
    }
}

struct IxHllMerge;
impl VScalar for IxHllMerge {
    type State = ();
    // @ai:invariant ix_hll_merge returns HyperLogLog::merge of two blobs (bucket-wise max); differing precision -> SQL error; merged count estimates the union cardinality [T:test conf:0.8 src:ix_duck::sketch::tests::hll_merge]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let a = read_varchar_col(input, 0, n);
        let b = read_varchar_col(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut fa: HyperLogLog = serde_json::from_str(&a[i])
                .map_err(|e| format!("ix_hll_merge: invalid sketch blob (arg 1): {e}"))?;
            let fb: HyperLogLog = serde_json::from_str(&b[i])
                .map_err(|e| format!("ix_hll_merge: invalid sketch blob (arg 2): {e}"))?;
            fa.merge(&fb).map_err(|e| format!("ix_hll_merge: {e}"))?;
            out.push(serde_json::to_string(&fa)?);
        }
        write_varchar(output, &out)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        combine_sig()
    }
}

// ── Count-Min Sketch ─────────────────────────────────────────────────────────

struct IxCmsBuild;
impl VScalar for IxCmsBuild {
    type State = ();
    // @ai:invariant ix_cms_build wraps CountMinSketch::with_error(epsilon, delta) + add each BIGINT item; epsilon<=0 or delta∉(0,1) -> SQL error [T:test conf:0.85 src:ix_duck::sketch::tests::cms_estimate]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let lists = read_bigint_list(input, 0, n);
        let eps = read_f64_col(input, 1, n);
        let delta = read_f64_col(input, 2, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            if eps[i] <= 0.0 {
                return Err("ix_cms_build: epsilon must be > 0".into());
            }
            if !(delta[i] > 0.0 && delta[i] < 1.0) {
                return Err("ix_cms_build: delta must be in (0, 1)".into());
            }
            let mut cms = CountMinSketch::with_error(eps[i], delta[i]);
            for v in &lists[i] {
                cms.add(v);
            }
            out.push(serde_json::to_string(&cms)?);
        }
        write_varchar(output, &out)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![list_bigint(), ty(LogicalTypeId::Double), ty(LogicalTypeId::Double)],
            ty(LogicalTypeId::Varchar),
        )]
    }
}

struct IxCmsEstimate;
impl VScalar for IxCmsEstimate {
    type State = ();
    // @ai:invariant ix_cms_estimate returns CountMinSketch::estimate(item) frequency from the blob; always >= the true count (Count-Min never undercounts) [T:test conf:0.85 src:ix_duck::sketch::tests::cms_estimate]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let sketches = read_varchar_col(input, 0, n);
        let items = read_i64_col(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let cms: CountMinSketch = serde_json::from_str(&sketches[i])
                .map_err(|e| format!("ix_cms_estimate: invalid sketch blob: {e}"))?;
            out.push(cms.estimate(&items[i]) as i64);
        }
        write_i64(output, &out);
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        probe_sig(LogicalTypeId::Bigint)
    }
}

struct IxCmsMerge;
impl VScalar for IxCmsMerge {
    type State = ();
    // @ai:invariant ix_cms_merge returns CountMinSketch::merge of two blobs (cell-wise add); differing dimensions -> SQL error; estimates of the combined stream [T:test conf:0.85 src:ix_duck::sketch::tests::cms_merge]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let a = read_varchar_col(input, 0, n);
        let b = read_varchar_col(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut fa: CountMinSketch = serde_json::from_str(&a[i])
                .map_err(|e| format!("ix_cms_merge: invalid sketch blob (arg 1): {e}"))?;
            let fb: CountMinSketch = serde_json::from_str(&b[i])
                .map_err(|e| format!("ix_cms_merge: invalid sketch blob (arg 2): {e}"))?;
            fa.merge(&fb).map_err(|e| format!("ix_cms_merge: {e}"))?;
            out.push(serde_json::to_string(&fa)?);
        }
        write_varchar(output, &out)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        combine_sig()
    }
}

// ── Cuckoo filter (supports deletion) ────────────────────────────────────────

struct IxCuckooBuild;
impl VScalar for IxCuckooBuild {
    type State = ();
    // @ai:invariant ix_cuckoo_build wraps CuckooFilter::new(capacity) + insert each BIGINT item; capacity<1 -> SQL error; a full filter (insert returns false) -> SQL error asking for more capacity (never silently drops -> no false negatives) [T:test conf:0.85 src:ix_duck::sketch::tests::cuckoo_build_contains_remove]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let lists = read_bigint_list(input, 0, n);
        let caps = read_i64_col(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            if caps[i] < 1 {
                return Err("ix_cuckoo_build: capacity must be >= 1".into());
            }
            let mut cf = CuckooFilter::new(caps[i] as usize);
            for v in &lists[i] {
                if !cf.insert(v) {
                    return Err(format!(
                        "ix_cuckoo_build: filter full at capacity {} ({} items); increase capacity",
                        caps[i], lists[i].len()
                    )
                    .into());
                }
            }
            out.push(serde_json::to_string(&cf)?);
        }
        write_varchar(output, &out)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![list_bigint(), ty(LogicalTypeId::Bigint)],
            ty(LogicalTypeId::Varchar),
        )]
    }
}

struct IxCuckooContains;
impl VScalar for IxCuckooContains {
    type State = ();
    // @ai:invariant ix_cuckoo_contains deserializes the blob and returns CuckooFilter::contains(item); inserted-and-not-removed items -> true [T:test conf:0.9 src:ix_duck::sketch::tests::cuckoo_build_contains_remove]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let sketches = read_varchar_col(input, 0, n);
        let items = read_i64_col(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let cf: CuckooFilter = serde_json::from_str(&sketches[i])
                .map_err(|e| format!("ix_cuckoo_contains: invalid sketch blob: {e}"))?;
            out.push(cf.contains(&items[i]));
        }
        write_bool(output, &out);
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        probe_sig(LogicalTypeId::Boolean)
    }
}

struct IxCuckooRemove;
impl VScalar for IxCuckooRemove {
    type State = ();
    // @ai:invariant ix_cuckoo_remove returns a new blob with item deleted (CuckooFilter::remove — the deletion the Bloom filter can't do); removing an absent item is a no-op [T:test conf:0.85 src:ix_duck::sketch::tests::cuckoo_build_contains_remove]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let sketches = read_varchar_col(input, 0, n);
        let items = read_i64_col(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut cf: CuckooFilter = serde_json::from_str(&sketches[i])
                .map_err(|e| format!("ix_cuckoo_remove: invalid sketch blob: {e}"))?;
            cf.remove(&items[i]);
            out.push(serde_json::to_string(&cf)?);
        }
        write_varchar(output, &out)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        probe_sig(LogicalTypeId::Varchar)
    }
}

/// Register the probabilistic-sketch scalar UDFs (build / probe / merge triads).
pub(crate) fn register(conn: &Connection) -> duckdb::Result<()> {
    conn.register_scalar_function::<IxBloomBuild>("ix_bloom_build")?;
    conn.register_scalar_function::<IxBloomContains>("ix_bloom_contains")?;
    conn.register_scalar_function::<IxBloomUnion>("ix_bloom_union")?;
    conn.register_scalar_function::<IxHllBuild>("ix_hll_build")?;
    conn.register_scalar_function::<IxHllCount>("ix_hll_count")?;
    conn.register_scalar_function::<IxHllMerge>("ix_hll_merge")?;
    conn.register_scalar_function::<IxCmsBuild>("ix_cms_build")?;
    conn.register_scalar_function::<IxCmsEstimate>("ix_cms_estimate")?;
    conn.register_scalar_function::<IxCmsMerge>("ix_cms_merge")?;
    conn.register_scalar_function::<IxCuckooBuild>("ix_cuckoo_build")?;
    conn.register_scalar_function::<IxCuckooContains>("ix_cuckoo_contains")?;
    conn.register_scalar_function::<IxCuckooRemove>("ix_cuckoo_remove")?;
    Ok(())
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use crate::open_bench;

    fn b(sql: &str) -> bool {
        open_bench()
            .unwrap()
            .query_row(sql, [], |r| r.get::<_, bool>(0))
            .unwrap()
    }
    fn i(sql: &str) -> i64 {
        open_bench()
            .unwrap()
            .query_row(sql, [], |r| r.get::<_, i64>(0))
            .unwrap()
    }

    #[test]
    fn bloom_roundtrip_no_false_negative() {
        // Every inserted item must probe true (Bloom has no false negatives).
        assert!(b("SELECT ix_bloom_contains(ix_bloom_build([10, 20, 30], 100, 0.01), 20)"));
        assert!(b("SELECT ix_bloom_contains(ix_bloom_build([10, 20, 30], 100, 0.01), 30)"));
    }

    #[test]
    fn bloom_union() {
        // Same params → unionable; the union contains items from both filters.
        let sql = "SELECT ix_bloom_contains(
                       ix_bloom_union(ix_bloom_build([1], 100, 0.01),
                                      ix_bloom_build([2], 100, 0.01)), {})";
        assert!(b(&sql.replace("{}", "1")));
        assert!(b(&sql.replace("{}", "2")));
    }

    #[test]
    fn hll_count_estimate() {
        // 1000 distinct values, p=12 → estimate within ~10%.
        let est = i("SELECT ix_hll_count(ix_hll_build(
                        (SELECT list(r)::BIGINT[] FROM range(1000) t(r)), 12))");
        assert!((est - 1000).abs() < 150, "HLL estimate {est} too far from 1000");
    }

    #[test]
    fn hll_merge() {
        // Disjoint halves merged → estimates the 1000-cardinality union.
        let est = i("SELECT ix_hll_count(ix_hll_merge(
                        ix_hll_build((SELECT list(r)::BIGINT[] FROM range(0, 500) t(r)), 12),
                        ix_hll_build((SELECT list(r)::BIGINT[] FROM range(500, 1000) t(r)), 12)))");
        assert!((est - 1000).abs() < 150, "merged HLL estimate {est} too far from 1000");
    }

    #[test]
    fn cms_estimate() {
        // "7" added 4×, "9" once → estimates never undercount.
        assert!(i("SELECT ix_cms_estimate(ix_cms_build([7,7,7,7,9], 0.01, 0.01), 7)") >= 4);
        assert!(i("SELECT ix_cms_estimate(ix_cms_build([7,7,7,7,9], 0.01, 0.01), 9)") >= 1);
    }

    #[test]
    fn cms_merge() {
        // Same item counted in both sketches → merged estimate sums.
        let est = i("SELECT ix_cms_estimate(ix_cms_merge(
                        ix_cms_build([5,5,5], 0.01, 0.01),
                        ix_cms_build([5,5], 0.01, 0.01)), 5)");
        assert!(est >= 5, "merged CMS estimate {est} should be >= 5");
    }

    #[test]
    fn cuckoo_build_contains_remove() {
        // Present after build; absent after remove (Cuckoo's deletion).
        assert!(b("SELECT ix_cuckoo_contains(ix_cuckoo_build([1,2,3], 100), 2)"));
        assert!(!b("SELECT ix_cuckoo_contains(
                        ix_cuckoo_remove(ix_cuckoo_build([1,2,3], 100), 2), 2)"));
        // Removing 2 leaves the others intact.
        assert!(b("SELECT ix_cuckoo_contains(
                       ix_cuckoo_remove(ix_cuckoo_build([1,2,3], 100), 2), 1)"));
    }
}
