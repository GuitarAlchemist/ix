---
title: "feat: Add machin-code crate — tree-sitter based ML on source code"
type: feat
status: active
date: 2026-03-13
deepened: 2026-03-13
origin: docs/brainstorms/2026-03-13-machin-code-tree-sitter-brainstorm.md
---

# feat: Add machin-code crate — tree-sitter ML on source code

## Enhancement Summary

**Deepened on:** 2026-03-13
**Research agents used:** best-practices-researcher (tree-sitter Rust patterns), architecture-strategist, performance-oracle, security-sentinel, code-simplicity-reviewer, Context7 (tree-sitter docs)

### Key Improvements from Deepening

1. **Simplified architecture**: Cut trait, builder, types.rs, similarity.rs — use free functions and inline structs (saves ~185 LOC of premature abstraction)
2. **Critical performance fix**: Adjacency matrix cap lowered from 10K to 1K nodes dense (800MB → 8MB), with auto-sparse above that
3. **Security hardening**: Parse timeout, file size limits, path sanitization, supply chain pinning
4. **Correct encapsulation**: `CodeTree` uses private fields + accessors (not pub fields) to enforce source/tree consistency
5. **Version pinning**: tree-sitter 0.26.x confirmed for Windows/MSVC support (MSVC detection fix in PR #4742)
6. **Performance optimization**: `kind_id()` array indexing instead of HashMap for 3-5x histogram speedup

### Simplifications Applied

| Cut | Reason |
|-----|--------|
| `CodeFeatureExtractor` trait | One implementation — use free functions, extract trait later if needed |
| `types.rs` file | Inline structs into the modules that create them |
| `similarity.rs` module | One-liner wrapping `machin_math::distance::cosine()` — users do this directly |
| `CodeParser` builder pattern | `parse("rust", source)` free function is sufficient |
| `NormalizationMode` enum | Users normalize ndarray themselves: `hist / hist.sum()` |
| `to_f32()` helper | `.mapv(|x| x as f32)` is idiomatic |
| 4 phases → 2 phases | Phase 1 ships parse + histogram + adjacency together (value requires both) |

---

## Overview

Add a new `machin-code` crate that uses tree-sitter to parse source code (any language) into concrete syntax trees, then extracts numerical feature representations (`Array1`/`Array2<f64>`) that feed directly into existing MachinDeOuf algorithms. This extends the toolkit from "ML on numerical data" to "ML on code as a first-class data type."

Language grammars are Cargo features — users compile only the languages they need.

(See brainstorm: `docs/brainstorms/2026-03-13-machin-code-tree-sitter-brainstorm.md`)

## Problem Statement / Motivation

MachinDeOuf has 22 crates covering optimization, supervised/unsupervised learning, neural networks, graph algorithms, signal processing, and more — but all operate exclusively on numerical data. Source code is one of the richest structured data types, and there is no way to apply MachinDeOuf's algorithms to it without manual feature engineering.

Tree-sitter provides language-agnostic AST parsing for 100+ languages. By bridging tree-sitter's syntax trees to ndarray, every existing crate becomes usable on code: cluster similar functions with DBSCAN, classify code style with KNN, detect clones with Bloom filters, analyze structure with graph algorithms.

**Use cases unlocked:** code clone detection, style classification, cross-language similarity, codebase structural analysis, vulnerability pattern matching, Guitar Alchemist C# analysis, TARS F# analysis.

## Proposed Solution

### Architecture (Simplified)

```
machin-code/src/
├── lib.rs          — Public API, re-exports, crate docs
├── error.rs        — CodeError enum (thiserror)
├── parse.rs        — pub fn parse(lang, src) → CodeTree (private fields + accessors)
├── extract.rs      — pub fn histogram(tree) → Array1<f64>, pub fn adjacency(tree) → Array2<f64>
└── query.rs        — Structural pattern matching (Phase 2, thin tree-sitter Query wrapper)
```

**Why 4 files, not 7:** Simplicity review found that types.rs, similarity.rs, and a trait were premature abstractions. Structs live in the module that creates them. Similarity is one line (`machin_math::distance::cosine()`). The trait has one impl — use free functions.

### Core Types

```rust
// parse.rs

/// Wraps a tree-sitter Tree with metadata.
/// Private fields enforce source/tree consistency (architecture review).
pub struct CodeTree {
    tree: tree_sitter::Tree,
    source: String,       // Owned: tree-sitter Tree does NOT borrow source
    language: String,
    has_errors: bool,
}

impl CodeTree {
    pub fn tree(&self) -> &tree_sitter::Tree { &self.tree }
    pub fn source(&self) -> &str { &self.source }
    pub fn language(&self) -> &str { &self.language }
    pub fn has_errors(&self) -> bool { self.has_errors }

    /// Extract text for a node by slicing the owned source.
    pub fn node_text(&self, node: tree_sitter::Node) -> &str {
        &self.source[node.byte_range()]
    }
}
```

### Research Insights: Why Private Fields

The architecture review found that `CodeTree`'s fields have an **invariant**: `tree` was produced by parsing `source` with the grammar for `language`. Making fields public would let callers replace `source` without re-parsing, making byte offsets invalid. This follows the pattern of `BloomFilter` and `CuckooFilter` in `crates/machin-probabilistic/` — fields with invariants are private.

### Research Insights: Why Owned `String`

The architecture review confirmed owning `source: String` is correct:
- tree-sitter's `Tree` does **not** borrow source text (it stores byte offsets, not references)
- Borrowing would add a lifetime parameter that infects every struct and trait (`CodeTree<'a>`) — no workspace type uses lifetime parameters
- Source files are 1-100KB — negligible vs parsing/extraction cost
- For batch processing (1000+ files), use a streaming API that drops source after extraction

### Public API (Free Functions)

```rust
// parse.rs

/// Parse source code into a CodeTree. Language is selected by name.
/// Uses #[cfg] feature gates internally.
pub fn parse(language: &str, source: &str) -> Result<CodeTree, CodeError>

// extract.rs

/// Extract a node-kind histogram. Uses kind_id() array indexing (O(N), cache-friendly).
/// Returns Array1<f64> of length language.node_kind_count().
pub fn histogram(tree: &CodeTree) -> Array1<f64>

/// Vocabulary: maps histogram index → node kind name.
pub fn histogram_vocabulary(tree: &CodeTree) -> Vec<String>

/// Extract adjacency matrix for named nodes only.
/// Dense for ≤1,000 nodes, returns error above 1,000 (use adjacency_sparse instead).
pub fn adjacency(tree: &CodeTree) -> Result<Array2<f64>, CodeError>

/// Sparse adjacency as machin-graph Graph (handles any size tree).
/// Requires `graph` feature flag.
#[cfg(feature = "graph")]
pub fn adjacency_graph(tree: &CodeTree) -> machin_graph::Graph
```

### Key Design Decisions

(Carried forward from brainstorm + refined by research agents)

| Decision | Choice | Rationale | Source |
|----------|--------|-----------|--------|
| Histogram vocabulary | **Language-specific** | Each grammar has different node kinds (~200-300). Cross-language mapping later. | Brainstorm |
| Histogram implementation | **`kind_id()` array indexing** | 3-5x faster than HashMap. `kind_id()` returns `u16`, pre-allocate `vec![0u32; node_kind_count()]`. Fits in L1 cache (~1.2KB). | Performance oracle |
| Parse error handling | **Include ERROR/MISSING nodes + `has_errors` flag** | Use `root.has_error()` as fast filter. ERROR nodes get dedicated histogram indices. | Best practices research |
| Adjacency matrix scope | **Named nodes only, cap at 1,000 dense** | 1K nodes = 8MB (fits L3 cache). Old 10K cap = 800MB (OOM risk). Auto-switch to sparse `Graph` above 1K. | Performance oracle |
| Struct encapsulation | **Private fields + accessors on CodeTree** | Source/tree invariant must be enforced. Follows `BloomFilter` pattern in workspace. | Architecture review |
| API style | **Free functions, no trait** | One implementation. Trait adds 40 LOC of abstraction with zero polymorphism. Extract trait later if needed. | Simplicity review |
| Numeric type | **f64** (workspace convention) | Consistent with all other crates. GPU users cast with `.mapv(\|x\| x as f32)`. | Architecture review |
| Thread safety | **Parser per-thread** | `Parser` is not `Send`/`Sync`. `Tree` is `Clone + Send`. Create parser per worker in pipelines. | Best practices research |
| Traversal | **TreeCursor exclusively** | `TreeCursor::goto_parent()` is O(1). `Node::parent()` traverses from root every time (cache was removed for thread safety). Never mix cursor and node APIs in the same walk. | Best practices research |
| Parse timeout | **`Parser::set_timeout_micros(30_000_000)`** | 30s timeout prevents pathological grammar hangs. | Security audit |

### Error Hierarchy

```rust
#[derive(Debug, thiserror::Error)]
pub enum CodeError {
    #[error("unsupported language: {0} (compile with the `lang-{0}` feature)")]
    UnsupportedLanguage(String),
    #[error("parse failed: tree-sitter returned None (possible timeout)")]
    ParseFailed,
    #[error("empty input: source code is empty or whitespace-only")]
    EmptyInput,
    #[error("tree too large: {0} named nodes exceeds limit of {1} for dense adjacency (use adjacency_graph instead)")]
    TreeTooLarge(usize, usize),
    #[error("file too large: {0} bytes exceeds limit of {1}")]
    FileTooLarge(u64, u64),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
```

## Technical Considerations

### Build System — tree-sitter C Compilation on Windows

Tree-sitter grammars compile C code via the `cc` crate at build time. On Windows 11 with MSVC:

- **Fixed in 0.26.x**: MSVC compiler detection bug was fixed in [PR #4742](https://github.com/tree-sitter/tree-sitter/pull/4742) (Aug 2025). Pin to 0.26.1+.
- **Warning suppression**: Grammar-generated C code triggers MSVC warnings. Grammar crates handle this in their own `build.rs` — no action needed for consumers.
- **UTF-8 source**: Add `/utf-8` flag if vendoring grammars.
- **Alternative**: MinGW-w64 support added in [PR #4201](https://github.com/tree-sitter/tree-sitter/pull/4201) as a fallback.

**Action:** Test `cargo add tree-sitter tree-sitter-rust` in a scratch project on Windows before implementation.

### Research Insights: Version Compatibility

| tree-sitter version | ABI | Grammar compatibility |
|---------------------|-----|----------------------|
| 0.24.x | 14 | Older grammars without `tree-sitter.json` |
| 0.25.x | **15** | Breaking: requires `tree-sitter.json` in grammars |
| 0.26.x | 15 | Non-breaking addition (MSVC fix, MinGW, WASI) |

**Pin to 0.26.x** — latest stable with best Windows support. Grammar crates must target ABI 15. Verify each grammar crate's compatibility before adding.

### Memory — Adjacency Matrix Limits (Revised)

| Named Nodes | Dense Memory | Recommendation |
|-------------|-------------|----------------|
| ≤500 | ≤2 MB | Dense `Array2<f64>` |
| 501-1,000 | 2-8 MB | Dense (still fits L3 cache) |
| 1,001-2,000 | 8-32 MB | Sparse only (`adjacency_graph()`) |
| >2,000 | >32 MB | Sparse only, warn user |

AST adjacency is inherently sparse — each node has ~3-10 children, so a 2,000-node tree has ~4,000-20,000 edges. Sparse representation uses ~100-500KB vs 32MB dense.

### Research Insights: Memory Management

Key lifetime rules from best-practices research:
- `Tree` is `Clone + Send` — safe to send to other threads
- `Node<'tree>` borrows `Tree` — cannot outlive it
- **Extract owned data during traversal** (kind, byte range, field names) rather than keeping `Node` references
- `TreeCursor` is allocation-free traversal — no heap allocation per step

### Security Considerations

From the security audit (3 Critical, 4 High findings):

**Resource limits (must implement):**
```rust
const DEFAULT_MAX_FILE_SIZE: u64 = 10 * 1024 * 1024; // 10 MB
const DEFAULT_PARSE_TIMEOUT_US: u64 = 30_000_000;     // 30 seconds
const DEFAULT_MAX_NAMED_NODES_DENSE: usize = 1_000;
```

**Supply chain:**
- Pin exact grammar crate versions with `=x.y.z` in workspace deps
- Run `cargo-audit` in CI
- Consider vendoring grammar C sources for high-assurance environments

**File I/O (Phase 2 CLI):**
- Canonicalize paths before reading
- Check file type (reject device files, FIFOs)
- Validate file size before reading into memory

**FFI boundary:**
- tree-sitter is C — no Rust memory safety guarantees inside the parser
- `set_timeout_micros()` on every parse to prevent hangs
- Use iterative `TreeCursor` traversal (not recursive) to prevent stack overflow on deeply nested trees

## Implementation Phases (Collapsed to 2)

### Phase 1: Parse + Histogram + Adjacency (MVP)

**Goal:** Parse any supported language, extract histogram and adjacency features, feed to clustering and graph algorithms. End-to-end proof of "ML on code."

**Files to create:**

| File | Purpose |
|------|---------|
| `crates/machin-code/Cargo.toml` | Crate manifest with feature-gated grammar deps |
| `crates/machin-code/src/lib.rs` | Crate docs + pub mod declarations |
| `crates/machin-code/src/error.rs` | `CodeError` enum |
| `crates/machin-code/src/parse.rs` | `pub fn parse()` + `CodeTree` struct (private fields) |
| `crates/machin-code/src/extract.rs` | `histogram()`, `histogram_vocabulary()`, `adjacency()`, `adjacency_graph()` |

**Files to modify:**

| File | Change |
|------|--------|
| `Cargo.toml` (root) | Add `"crates/machin-code"` to members; add `tree-sitter`, `tree-sitter-rust`, etc. to `[workspace.dependencies]` |

**Implementation details (from research):**

Histogram extraction — use `kind_id()` array indexing:
```rust
pub fn histogram(tree: &CodeTree) -> Array1<f64> {
    let lang: tree_sitter::Language = /* from tree */;
    let vocab = lang.node_kind_count();
    let mut counts = vec![0u32; vocab];
    let mut cursor = tree.tree().walk();

    // TreeCursor depth-first walk — zero allocation
    loop {
        let node = cursor.node();
        if node.is_named() {
            counts[node.kind_id() as usize] += 1;
        }
        if cursor.goto_first_child() { continue; }
        while !cursor.goto_next_sibling() {
            if !cursor.goto_parent() {
                return Array1::from_vec(counts.into_iter().map(|c| c as f64).collect());
            }
        }
    }
}
```

Language registry — `#[cfg]` gated:
```rust
pub fn parse(language: &str, source: &str) -> Result<CodeTree, CodeError> {
    if source.trim().is_empty() {
        return Err(CodeError::EmptyInput);
    }

    let ts_lang: tree_sitter::Language = match language {
        #[cfg(feature = "lang-rust")]
        "rust" => tree_sitter_rust::LANGUAGE.into(),
        #[cfg(feature = "lang-python")]
        "python" => tree_sitter_python::LANGUAGE.into(),
        #[cfg(feature = "lang-csharp")]
        "csharp" | "c#" => tree_sitter_c_sharp::LANGUAGE.into(),
        // ...
        _ => return Err(CodeError::UnsupportedLanguage(language.to_string())),
    };

    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&ts_lang).expect("language version mismatch");
    parser.set_timeout_micros(DEFAULT_PARSE_TIMEOUT_US);

    let tree = parser.parse(source, None).ok_or(CodeError::ParseFailed)?;
    let has_errors = tree.root_node().has_error();

    Ok(CodeTree { tree, source: source.to_string(), language: language.to_string(), has_errors })
}
```

**Tests:**

- Parse `fn main() { let x = 42; }` → root kind is `"source_file"`, `has_errors == false`
- Parse `fn foo( {` → `has_errors == true`, tree still produced
- Parse `""` → `CodeError::EmptyInput`
- Histogram from Rust snippet → non-zero entries at `kind_id` for `function_item`, `let_declaration`, `integer_literal`
- Histogram length == `language.node_kind_count()`
- Adjacency from small snippet → matrix dimensions = named node count
- Adjacency from 1,001+ named nodes → `CodeError::TreeTooLarge`
- `adjacency_graph()` from any size → valid `Graph` instance (with `graph` feature)

**Integration test:** Parse 3 Rust snippets (function, struct, impl block), extract histograms, stack into `Array2`, feed to `KMeans::new(2).fit_predict()`, verify output shape.

**Acceptance criteria:**
- [x] `cargo build -p machin-code` compiles on Windows 11 (tree-sitter C compilation works)
- [x] `cargo test -p machin-code` passes all tests
- [ ] Histogram → KMeans clustering works end-to-end without manual conversion
- [ ] Adjacency → `machin-graph` algorithms work (with `graph` feature)
- [x] `has_errors` flag correctly identifies partial parses
- [x] Parse timeout prevents hangs on pathological input
- [x] File size check prevents OOM (when parsing from file path)

### Phase 2: Query + CLI + Path-Contexts (Future)

**Goal:** Structural pattern matching, CLI exposure, code2vec-style embeddings, batch processing.

**Scope:**

| Feature | Files | Notes |
|---------|-------|-------|
| Query wrapper | `query.rs` | Thin wrapper around tree-sitter `Query`/`QueryCursor`, S-expression patterns |
| Per-function extraction | `query.rs` | Use queries to find function nodes, extract histogram per function |
| CLI | `machin-skill/src/main.rs` | `machin code --lang rust --input file.rs --extract histogram` |
| Path-contexts | `extract.rs` | Cap at 500 leaves (sampled), max_path_length=8, hash-based encoding |
| Batch mode | CLI | `--input-dir src/ --glob "**/*.rs"` |
| Language auto-detect | `parse.rs` | File extension → language name mapping |

**Path-context performance budget (from performance oracle):**

| Leaves | Pairs | With max_length=8 filter | With O(1) LCA | Time |
|--------|-------|--------------------------|----------------|------|
| 500 (capped) | 125K | ~12K effective | ~12K × O(1) | ~2-5ms |
| 1,000 (uncapped) | 500K | ~50K effective | ~50K × O(1) | ~8-15ms |
| 2,000 (uncapped) | 2M | ~200K effective | ~200K × O(1) | ~30-50ms |

**Recommendation:** Default cap at 500 leaves with seeded random sampling. Precompute LCA with Euler tour + sparse table for O(1) queries.

**Not planned:**
- Learned embeddings (requires training loop — separate project)
- Incremental parsing integration (API accommodates it via `parse(source, Some(&previous))`)
- Cross-language vocabulary mapping (Phase 3 if needed)

## Cargo.toml Design

```toml
[package]
name = "machin-code"
version.workspace = true
edition.workspace = true
license.workspace = true
description = "ML on source code: tree-sitter parsing + feature extraction for any language"

[dependencies]
machin-math = { workspace = true }
ndarray = { workspace = true }
thiserror = { workspace = true }
tree-sitter = { workspace = true }

# Graph integration (optional — architecture review recommendation)
machin-graph = { workspace = true, optional = true }

# Language grammars — opt-in via features
tree-sitter-rust = { workspace = true, optional = true }
tree-sitter-python = { workspace = true, optional = true }
tree-sitter-javascript = { workspace = true, optional = true }
tree-sitter-java = { workspace = true, optional = true }
tree-sitter-c-sharp = { workspace = true, optional = true }
tree-sitter-cpp = { workspace = true, optional = true }
tree-sitter-go = { workspace = true, optional = true }
tree-sitter-typescript = { workspace = true, optional = true }

[features]
default = ["lang-rust"]
graph = ["dep:machin-graph"]
lang-rust = ["dep:tree-sitter-rust"]
lang-python = ["dep:tree-sitter-python"]
lang-javascript = ["dep:tree-sitter-javascript"]
lang-java = ["dep:tree-sitter-java"]
lang-csharp = ["dep:tree-sitter-c-sharp"]
lang-cpp = ["dep:tree-sitter-cpp"]
lang-go = ["dep:tree-sitter-go"]
lang-typescript = ["dep:tree-sitter-typescript"]
all-languages = [
    "lang-rust", "lang-python", "lang-javascript", "lang-java",
    "lang-csharp", "lang-cpp", "lang-go", "lang-typescript"
]
```

**Workspace Cargo.toml additions:**
```toml
# Add to [workspace.dependencies] — pin exact versions for supply chain safety
tree-sitter = "=0.26.6"
tree-sitter-rust = { version = "=0.23.2", optional = true }
tree-sitter-python = { version = "=0.23.5", optional = true }
# ... (verify exact versions before implementation)
```

## System-Wide Impact

- **Workspace build**: New C compilation step for tree-sitter grammars. First build adds ~10-15s per grammar. Incremental builds unaffected.
- **Binary size**: `machin-skill` binary grows by ~200KB (Rust grammar only) to ~5MB (all languages).
- **No breaking changes**: Pure addition — no existing crate is modified (except workspace Cargo.toml and machin-skill for CLI in Phase 2).
- **Pipeline integration**: `machin-pipeline` DAG executor can orchestrate code analysis stages. `Parser` is not `Send` — create per-worker. `Tree` is `Clone + Send` — safe to share.
- **machin-graph dependency**: Optional, behind `graph` feature flag. This is the first cross-dependency between algorithm-layer crates, but it follows a directed edge with no cycle risk.

## Acceptance Criteria

### Phase 1 (MVP)
- [x] `crates/machin-code/` exists with `parse.rs`, `extract.rs`, `error.rs`, `lib.rs`
- [x] `cargo build -p machin-code` compiles on Windows 11 (tree-sitter C compilation works)
- [ ] Rust source code → histogram → KMeans clustering works end-to-end
- [ ] Rust source code → adjacency → `machin-graph::Graph` works (with `graph` feature)
- [x] Partial parse trees (syntax errors) handled gracefully with `has_errors` flag
- [x] Empty input returns `CodeError::EmptyInput`
- [x] Dense adjacency capped at 1,000 named nodes with clear error above
- [x] Parse timeout set (30s default)
- [x] All tests pass: `cargo test -p machin-code`

### Phase 2
- [ ] Tree-sitter query wrapper with `find()` and per-function extraction
- [ ] CLI: `machin code --lang rust --input file.rs --extract histogram` with JSON output
- [ ] Path-context extraction with 500-leaf cap and max_path_length=8
- [ ] File size validation and path sanitization for CLI

## Dependencies & Risks

| Risk | Impact | Mitigation | Source |
|------|--------|------------|--------|
| tree-sitter C compilation fails on Windows | Blocks entire crate | Test in scratch project first; MSVC fix confirmed in 0.26.x | Best practices |
| Grammar version ABI mismatch | Compile error or runtime panic | Pin exact versions; verify ABI 15 compatibility | Best practices |
| Large file OOM (adjacency matrix) | Crash | Named-node-only + 1K dense cap + sparse fallback | Performance oracle |
| Pathological input hangs parser | DoS | `set_timeout_micros(30_000_000)` on every parse | Security audit |
| Supply chain: compromised grammar crate | Binary compromise | Pin `=x.y.z` versions, `cargo-audit` in CI | Security audit |
| Stack overflow on deeply nested AST | Crash | TreeCursor iterative traversal only, never recursive | Best practices |
| Build time regression | Developer friction | Feature-gate languages, default to Rust-only | Architecture review |

## Success Metrics

- Parse + histogram extraction in **< 2ms** for a 1,000-line file (revised from 10ms based on performance analysis: tree-sitter ~1ms + histogram ~0.1ms)
- Parse + histogram + sparse adjacency in **< 5ms** for a 1,000-line file
- Histogram → KMeans clustering produces meaningful code groupings (e.g., separates tests from implementations)
- Zero additional `cargo test --workspace` failures after integration
- Binary size increase < 500KB with default features (Rust grammar only)

## Testing Strategy

From best practices research:

- **Test with real grammars** — do not mock. Tree-sitter grammars are fast and deterministic. Mocking only tests the mock.
- **Embed small source snippets in tests** — keep tests self-contained, no external fixture files.
- **Error node tests**: Parse invalid source, verify `has_errors == true`, verify feature extraction still produces reasonable output.
- **Consider `insta` snapshot tests** for S-expression output — catches regressions when upgrading grammar versions.
- **Float comparisons**: Use `assert!((a - b).abs() < 1e-10)` pattern consistent with workspace (only machin-math uses `approx`).

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-03-13-machin-code-tree-sitter-brainstorm.md](../brainstorms/2026-03-13-machin-code-tree-sitter-brainstorm.md) — Key decisions: language-specific histograms, three extraction tiers, feature-gated grammars, ndarray output always

### Internal References

- Crate structure pattern: `crates/machin-signal/` (algorithmic crate model)
- CLI integration: `crates/machin-skill/src/main.rs:14-59` (Commands enum)
- Graph types: `crates/machin-graph/src/graph.rs` (adjacency list format, sparse)
- Clustering API: `crates/machin-unsupervised/` (Clusterer trait, KMeans)
- Distance functions: `crates/machin-math/src/distance.rs` (cosine, euclidean)
- Encapsulation pattern: `crates/machin-probabilistic/src/bloom.rs` (private fields + accessors)
- Feature flag pattern: `crates/machin-cache/Cargo.toml` (optional `resp-server` feature)

### External References

- tree-sitter Rust crate: https://crates.io/crates/tree-sitter (v0.26.6)
- tree-sitter docs: https://tree-sitter.github.io/tree-sitter/
- code2vec paper (path-context approach): https://arxiv.org/pdf/1803.09473
- TreeCursor vs Node traversal: https://github.com/tree-sitter/tree-sitter/discussions/2018
- Error node semantics: https://github.com/tree-sitter/tree-sitter/issues/396
- MSVC detection fix: https://github.com/tree-sitter/tree-sitter/pull/4742
- MinGW-w64 support: https://github.com/tree-sitter/tree-sitter/pull/4201
- Zed tree-sitter upgrade patterns: https://github.com/zed-industries/zed/pull/24340
- tree-sitter-traversal crate (ergonomic iterators): https://crates.io/crates/tree-sitter-traversal

### Research Agents

| Agent | Key Finding |
|-------|-------------|
| best-practices-researcher | Pin 0.26.x, use TreeCursor exclusively, test with real grammars, ERROR node handling via `has_error()`/`is_error()`/`is_missing()` |
| architecture-strategist | Private fields on CodeTree, optional machin-graph dep, owned String is correct, add tree-sitter to workspace deps |
| performance-oracle | kind_id() array indexing 3-5x faster, adjacency cap at 1K dense, path-context cap at 500 leaves, streaming API for batch |
| security-sentinel | C FFI risk (timeout), supply chain (pin versions), file size limits, path sanitization |
| code-simplicity-reviewer | Cut trait, builder, types.rs, similarity.rs, normalization enum, to_f32() — saves ~185 LOC |
| Context7 | Confirmed Query API with captures, S-expression pattern syntax |
