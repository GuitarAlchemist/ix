---
date: 2026-03-13
topic: machin-code-tree-sitter
---

# machin-code: ML on Source Code via Tree-sitter

## What We're Building

A new `machin-code` crate that uses tree-sitter to parse source code (any language) into concrete syntax trees, then extracts numerical representations (`Array1`/`Array2<f64>`) that feed directly into existing MachinDeOuf algorithms. This extends MachinDeOuf from "ML on data" to "ML on code as a first-class data type."

Language grammars are Cargo features — users opt in to only the languages they need (Rust, Python, C#, Java, JavaScript, Go, C++, etc.).

## Why This Approach

**Tree-sitter** was chosen over alternatives because:
- Battle-tested in production (Zed, Helix, GitHub code nav)
- Language-agnostic — one API for all languages, grammars are plugins
- Rust-native crate (`tree-sitter 0.26`, ~14.8M downloads)
- Nodes have `kind_id() -> u16` — a built-in integer vocabulary for embeddings
- Incremental parsing for real-time use cases
- Query system for structural pattern matching

**Rejected alternatives:**
- `syn` (Rust-only, no multi-language)
- `rust-analyzer` (Rust-only, heavy)
- Custom parsers (massive effort, fragile)
- Roslyn/.NET analyzers (C#-only, wrong ecosystem)

## Key Decisions

- **Crate name: `machin-code`** (not `machin-ast`) — broader scope, code is the domain
- **Feature-gated languages** — `lang-rust`, `lang-python`, `lang-csharp`, etc. Default: `lang-rust`
- **Three extraction tiers** (build incrementally):
  1. **Histograms** (`Array1<f64>`) — node-kind counts, cheapest, works with clustering/classification immediately
  2. **Adjacency matrices** (`Array2<f64>`) — parent-child graph, feeds `machin-graph`
  3. **Path-context embeddings** (`Array2<f64>`) — code2vec-style, feeds `machin-nn` and `machin-gpu` similarity
- **Output is always ndarray** — no new types, immediate compatibility with all existing crates
- **Query helpers** — thin wrappers around tree-sitter queries for common patterns (find functions, find imports, find types)

## Architecture

```
machin-code
├── parse.rs        — Parser wrapper, language registry, multi-language support
├── extract.rs      — AST → ndarray feature extraction (histogram, adjacency, path-context)
├── query.rs        — Structural pattern matching helpers (find functions, classes, etc.)
├── similarity.rs   — Code similarity using extracted features + machin-math::distance
├── diff.rs         — Structural diff between two ASTs (optional, later)
└── lib.rs          — Public API, re-exports
```

## Integration Points

| Existing Crate | How machin-code Feeds It |
|----------------|--------------------------|
| `machin-unsupervised` | Cluster similar functions (histogram → K-Means/DBSCAN) |
| `machin-supervised` | Classify code style/language/quality (histogram → classifiers) |
| `machin-graph` | Analyze code structure (adjacency matrix → graph algorithms) |
| `machin-nn` | Train code embeddings (path-contexts → neural nets) |
| `machin-gpu` | Fast similarity search across codebases (embeddings → GPU cosine) |
| `machin-probabilistic` | Approximate code dedup (subtree hashes → Bloom filters) |
| `machin-signal` | Analyze code complexity patterns over time (metrics → FFT) |
| `machin-search` | Navigate code graphs (AST → A*, pattern search) |

## Use Cases Unlocked

1. **Code clone detection** — extract histograms, cluster with DBSCAN, find duplicates
2. **Style classification** — train classifier on labeled code (clean vs messy, idiomatic vs not)
3. **Cross-language similarity** — compare C# and Rust implementations via structural features
4. **Guitar Alchemist analysis** — analyze GA's C# codebase structure, find refactoring targets
5. **Vulnerability pattern matching** — tree-sitter queries for known dangerous patterns
6. **Code complexity metrics** — AST depth, branching factor, cyclomatic complexity from tree structure
7. **Codebase evolution** — track structural changes over git history via FFT on metric time series

## Open Questions

- Should `machin-code` depend on `machin-math` for distance functions, or keep it zero-dependency (just tree-sitter + ndarray) and let users compose?
- Should we include a `CodeSnippet` struct that bundles source + tree + language, or keep it functional (parse → extract → done)?
- How to handle the `all-languages` feature flag — binary size vs convenience?

## Next Steps

→ `/ce:plan` for implementation details — start with Tier 1 (parse + histogram) and one use case (code clustering)
