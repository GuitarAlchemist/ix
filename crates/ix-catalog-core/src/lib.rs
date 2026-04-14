//! # ix-catalog-core
//!
//! Shared substrate for ix's queryable, MCP-surfaced catalogs. A
//! "catalog" in ix is a curated static inventory of named things
//! (tools, grammars, RFCs, datasets, ...) that agents can filter and
//! browse through a uniform JSON interface. The pattern is:
//!
//! 1. A `const &[Entry]` in some sub-crate, populated by hand.
//! 2. A zero-sized struct implementing the [`Catalog`] trait defined
//!    here, so the MCP dispatcher can treat every catalog the same
//!    way.
//! 3. A single MCP tool per catalog (`ix_code_catalog`,
//!    `ix_grammar_catalog`, `ix_rfc_catalog`, ...) that delegates
//!    straight through to [`Catalog::query`].
//! 4. A meta-tool `ix_catalog_list` that walks every registered
//!    `&dyn Catalog` and returns their [`CatalogSummary`]s.
//!
//! The goal is that adding a new catalog is ~50 lines of glue code:
//! define the entry struct, write a `CATALOG: &[Entry]`, implement
//! [`Catalog`], register the MCP tool. No bespoke handler logic,
//! no drift between catalogs.
//!
//! This crate is intentionally tiny and has no ix dependencies. It
//! is imported by every catalog-bearing crate and by `ix-agent`.

use serde::Serialize;
use serde_json::Value;

/// Short, uniform description of a single catalog. Used by the
/// `ix_catalog_list` meta-tool so agents can discover what catalogs
/// exist without knowing their shape in advance.
#[derive(Debug, Clone, Serialize)]
pub struct CatalogSummary {
    /// Snake_case identifier, matching the MCP tool name after the
    /// `ix_` prefix and before `_catalog`. E.g. `"code_analysis"`
    /// for `ix_code_catalog`, `"rfc"` for `ix_rfc_catalog`.
    pub name: &'static str,

    /// One-sentence statement of what the catalog covers and what
    /// it deliberately excludes. Agents read this to decide whether
    /// to route a question here.
    pub scope: &'static str,

    /// Total number of entries in the catalog.
    pub entry_count: usize,
}

/// Every ix catalog implements this trait. It is deliberately small
/// — four associated methods for metadata, one for counts, and one
/// for the actual query. The query input/output is `serde_json::Value`
/// so the MCP dispatcher can pass JSON straight through without
/// caring about the catalog-specific entry shape.
///
/// # Implementing
///
/// Implementations are typically zero-sized types:
///
/// ```
/// use ix_catalog_core::Catalog;
/// use serde_json::{json, Value};
///
/// pub struct MyCatalog;
///
/// impl Catalog for MyCatalog {
///     fn name(&self) -> &'static str { "my_catalog" }
///     fn scope(&self) -> &'static str {
///         "Example catalog; demonstrates the Catalog trait shape."
///     }
///     fn entry_count(&self) -> usize { 0 }
///     fn counts(&self) -> Value { json!({ "total": 0 }) }
///     fn query(&self, _filter: Value) -> Result<Value, String> {
///         Ok(json!({ "matched": 0, "entries": [] }))
///     }
/// }
///
/// let cat = MyCatalog;
/// assert_eq!(cat.name(), "my_catalog");
/// ```
pub trait Catalog: Sync {
    /// Snake_case identifier. Must be stable — agents key off this.
    fn name(&self) -> &'static str;

    /// One-sentence scope statement. Shown to agents via the
    /// [`summary`](Catalog::summary) helper and the `ix_catalog_list`
    /// meta-tool.
    fn scope(&self) -> &'static str;

    /// Total number of entries. Must match what `query` with an
    /// empty filter would return as `matched`.
    fn entry_count(&self) -> usize;

    /// Per-category or per-facet breakdown. The exact shape is
    /// catalog-specific, but every catalog should emit a `total`
    /// field so a uniform summary tool can pick it up. Default
    /// implementation returns just `{ "total": entry_count() }`.
    fn counts(&self) -> Value {
        serde_json::json!({ "total": self.entry_count() })
    }

    /// Apply a filter and return matched entries. The filter shape
    /// is catalog-specific; conventions:
    ///
    /// - An empty object `{}` returns every entry.
    /// - String filters are case-insensitive where plausible.
    /// - Unknown filter fields are ignored, NOT errored on — this
    ///   keeps agents forward-compatible when the filter shape grows.
    /// - Returned JSON must include a `matched` integer and an
    ///   `entries` array. Implementations may add catalog-specific
    ///   top-level fields (e.g. `counts`) alongside.
    ///
    /// Errors are returned as `Err(String)` with a message that names
    /// the offending filter field and the expected shape.
    fn query(&self, filter: Value) -> Result<Value, String>;

    /// Convenience: build a [`CatalogSummary`] from this catalog's
    /// metadata. Default implementation wires up the fields — override
    /// only if a catalog wants a different name / scope at runtime,
    /// which is extremely unusual.
    fn summary(&self) -> CatalogSummary {
        CatalogSummary {
            name: self.name(),
            scope: self.scope(),
            entry_count: self.entry_count(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers — extracted from ix-code::catalog so every catalog can use them.
// ---------------------------------------------------------------------------

/// Lowercase a string and convert `-` to `_`. Used by every catalog
/// that accepts string inputs it wants to normalise before matching
/// against snake_case enum variants.
///
/// ```
/// use ix_catalog_core::normalize_snake_case;
/// assert_eq!(normalize_snake_case("Static-Analysis"), "static_analysis");
/// assert_eq!(normalize_snake_case("RFC-9110"), "rfc_9110");
/// ```
pub fn normalize_snake_case(s: &str) -> String {
    s.chars()
        .map(|c| if c == '-' { '_' } else { c.to_ascii_lowercase() })
        .collect()
}

/// Case-insensitive substring match. Avoids allocating more than
/// necessary — `to_ascii_lowercase` on both sides is cheap for the
/// short strings catalogs deal with, and the predicate shape is
/// easy to reason about.
///
/// ```
/// use ix_catalog_core::string_contains_ci;
/// assert!(string_contains_ci("Abstract Interpretation", "abstract"));
/// assert!(string_contains_ci("CYCLOMATIC COMPLEXITY", "CyCloMatic"));
/// assert!(!string_contains_ci("foo", "bar"));
/// ```
pub fn string_contains_ci(haystack: &str, needle: &str) -> bool {
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

/// Error type for catalog query failures. Catalogs that prefer to
/// return rich error information can use this, but returning a plain
/// `String` from `query` is also fine — the trait is intentionally
/// permissive about error types in exchange for simplicity.
#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    /// The filter referenced a field the catalog doesn't support.
    #[error("unknown filter field '{field}' (known fields: {known})")]
    UnknownFilterField {
        /// Name of the unknown field.
        field: String,
        /// Comma-separated list of supported fields, for the error
        /// message.
        known: String,
    },

    /// A filter field was the wrong JSON type.
    #[error("filter field '{field}' must be {expected}, got {actual}")]
    WrongFilterType {
        /// Name of the offending field.
        field: String,
        /// Expected JSON type ("string", "integer", ...).
        expected: String,
        /// Actual JSON type observed.
        actual: String,
    },

    /// A filter enum value didn't match any known variant.
    #[error("filter field '{field}' = '{value}' is not a valid enum; expected one of: {expected}")]
    UnknownEnumValue {
        /// Name of the offending field.
        field: String,
        /// Value the user supplied.
        value: String,
        /// Comma-separated list of valid enum values.
        expected: String,
    },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Minimal stub implementation exercising the default trait
    /// methods + an override of `query`. Used only in the tests
    /// below.
    struct StubCatalog;

    impl Catalog for StubCatalog {
        fn name(&self) -> &'static str {
            "stub"
        }
        fn scope(&self) -> &'static str {
            "Stub catalog for ix-catalog-core trait tests."
        }
        fn entry_count(&self) -> usize {
            3
        }
        fn query(&self, filter: Value) -> Result<Value, String> {
            if filter.get("fail").is_some() {
                return Err("stub: filter contained 'fail'".to_string());
            }
            Ok(json!({ "matched": 3, "entries": ["a", "b", "c"] }))
        }
    }

    #[test]
    fn default_counts_reports_total() {
        let c = StubCatalog;
        let counts = c.counts();
        assert_eq!(counts["total"].as_u64(), Some(3));
    }

    #[test]
    fn default_summary_wires_metadata() {
        let c = StubCatalog;
        let s = c.summary();
        assert_eq!(s.name, "stub");
        assert_eq!(s.entry_count, 3);
        assert!(s.scope.starts_with("Stub catalog"));
    }

    #[test]
    fn query_errors_are_propagated() {
        let c = StubCatalog;
        let ok = c.query(json!({})).expect("empty filter must succeed");
        assert_eq!(ok["matched"].as_u64(), Some(3));

        let err = c.query(json!({ "fail": true })).expect_err("should fail");
        assert!(err.contains("stub: filter contained 'fail'"));
    }

    #[test]
    fn normalize_snake_case_basic_forms() {
        assert_eq!(normalize_snake_case("static_analysis"), "static_analysis");
        assert_eq!(normalize_snake_case("Static-Analysis"), "static_analysis");
        assert_eq!(normalize_snake_case("RFC"), "rfc");
        assert_eq!(normalize_snake_case(""), "");
    }

    #[test]
    fn string_contains_ci_is_case_insensitive() {
        assert!(string_contains_ci("HTTP/2 over TLS", "http"));
        assert!(string_contains_ci("Abstract Interpretation", "INTERPRETATION"));
        assert!(!string_contains_ci("HTTP/2", "http3"));
    }

    #[test]
    fn catalog_error_has_useful_messages() {
        let e = CatalogError::UnknownFilterField {
            field: "colour".into(),
            known: "category, language".into(),
        };
        let msg = format!("{e}");
        assert!(msg.contains("colour"));
        assert!(msg.contains("category, language"));
    }
}
