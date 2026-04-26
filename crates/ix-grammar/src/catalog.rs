//! # Grammar catalog
//!
//! Curated, MCP-queryable inventory of real-world EBNF / ABNF / PEG /
//! ANTLR-G4 grammar sources. Agents ask "where is the authoritative
//! Python grammar" or "which RFC defines the URI ABNF" and the
//! catalog returns a pointer to the canonical source, the format
//! (so the agent knows whether to use `ix_grammar::ebnf::parse` or
//! `ix_grammar::abnf::parse`), and a short description.
//!
//! This is not the grammars themselves — it is an index of where
//! they live. Fetching the actual grammar file is the agent's job
//! (via HTTP or a local checkout); parsing it into an
//! [`EbnfGrammar`](crate::constrained::EbnfGrammar) is ours.
//!
//! See also `ix-code::catalog` (external code-analysis tools) and
//! `ix-net::rfc_catalog` (RFCs); all three implement the shared
//! [`Catalog`](ix_catalog_core::Catalog) trait.

use ix_catalog_core::{string_contains_ci, Catalog};
use serde::Serialize;
use serde_json::{json, Value};

/// Notation a grammar is written in. Determines which of ix-grammar's
/// parsers (if any) can consume it directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GrammarFormat {
    /// Standard ISO/IEC 14977 EBNF with `=`, `|`, `[...]`, `{...}`,
    /// `(...)`. Consumed by `ix_grammar::ebnf::parse` (subset).
    Ebnf,
    /// W3C EBNF variant used by XML, XPath, and similar web specs.
    /// Adopts regex-style `+`, `*`, `?` postfix operators. Not yet
    /// consumed natively by ix-grammar; see the ebnf parser for a
    /// compatible subset.
    W3cEbnf,
    /// RFC 5234 Augmented Backus-Naur Form used by IETF specs:
    /// HTTP, SMTP, DNS, TLS, URI, OAuth, JSON, etc. Uses `=`, `/`
    /// for alternation, `*element`, `N*M element` for repetition.
    /// Consumed by `ix_grammar::abnf::parse` (subset).
    Abnf,
    /// Parsing Expression Grammar — ordered choice, no ambiguity.
    /// Not natively supported by ix-grammar; use the `pest` crate
    /// directly.
    Peg,
    /// ANTLR's grammar DSL (`.g4` files). The `antlr/grammars-v4`
    /// repo hosts hundreds. Not natively supported; ANTLR itself is
    /// the reference implementation.
    AntlrG4,
    /// Plain Backus-Naur Form — the historical ancestor, rarely
    /// used in modern specs but defines some classic languages.
    Bnf,
}

/// One grammar source in the catalog.
#[derive(Debug, Clone, Serialize)]
pub struct GrammarEntry {
    /// Short name — the language, protocol, or meta-grammar.
    pub name: &'static str,
    /// Programming language, protocol, or format this grammar
    /// defines (e.g. `"python"`, `"http"`, `"rfc"`).
    pub language: &'static str,
    /// Notation the source grammar is written in.
    pub format: GrammarFormat,
    /// Topical tags: `["web", "api"]`, `["crypto", "pki"]`, etc.
    /// Used by `by_topic` queries.
    pub topics: &'static [&'static str],
    /// Organisation or standards body that maintains the grammar.
    /// E.g. `"python.org"`, `"IETF"`, `"ECMA"`, `"ISO"`.
    pub source_org: &'static str,
    /// Year the cited revision was published (or last revised).
    pub year: u16,
    /// Canonical URL — direct link to the grammar file when
    /// possible, otherwise the spec document that contains it.
    pub url: &'static str,
    /// One-sentence description.
    pub description: &'static str,
}

/// The full catalog. Sorted by language, then alphabetically.
pub const CATALOG: &[GrammarEntry] = &[
    // ─── Programming languages ──────────────────────────────────
    GrammarEntry {
        name: "Python 3 grammar (PEP 617)",
        language: "python",
        format: GrammarFormat::Peg,
        topics: &["language", "dynamic"],
        source_org: "python.org",
        year: 2020,
        url: "https://docs.python.org/3/reference/grammar.html",
        description: "CPython's current PEG-based grammar, used by the PEP 617 parser.",
    },
    GrammarEntry {
        name: "Go language specification",
        language: "go",
        format: GrammarFormat::Ebnf,
        topics: &["language", "static"],
        source_org: "golang.org",
        year: 2012,
        url: "https://go.dev/ref/spec",
        description: "Go's complete language spec is written in a clean ISO-style EBNF.",
    },
    GrammarEntry {
        name: "ECMA-262 (JavaScript/ECMAScript)",
        language: "javascript",
        format: GrammarFormat::W3cEbnf,
        topics: &["language", "web", "dynamic"],
        source_org: "ECMA International",
        year: 2024,
        url: "https://tc39.es/ecma262/",
        description: "ECMAScript specification including lexical + syntactic grammars for JavaScript.",
    },
    GrammarEntry {
        name: "Rust reference grammar",
        language: "rust",
        format: GrammarFormat::Ebnf,
        topics: &["language", "systems", "static"],
        source_org: "rust-lang.org",
        year: 2024,
        url: "https://doc.rust-lang.org/reference/",
        description: "Rust language reference with EBNF-style rule blocks for every construct.",
    },
    GrammarEntry {
        name: "Ruby parser gem grammar",
        language: "ruby",
        format: GrammarFormat::Bnf,
        topics: &["language", "dynamic"],
        source_org: "whitequark/parser",
        year: 2023,
        url: "https://github.com/whitequark/parser",
        description: "The canonical Ruby parser; its Racc grammar is the closest thing to a Ruby reference grammar.",
    },
    GrammarEntry {
        name: "Haskell 2010 report grammar",
        language: "haskell",
        format: GrammarFormat::Bnf,
        topics: &["language", "functional", "static"],
        source_org: "haskell.org",
        year: 2010,
        url: "https://www.haskell.org/onlinereport/haskell2010/haskellch10.html",
        description: "The Haskell 2010 report's concrete syntax in traditional BNF.",
    },
    GrammarEntry {
        name: "R7RS Scheme",
        language: "scheme",
        format: GrammarFormat::Bnf,
        topics: &["language", "functional", "lisp"],
        source_org: "scheme-reports.org",
        year: 2013,
        url: "https://small.r7rs.org/",
        description: "R7RS Scheme's formal syntax appendix with BNF-style production rules.",
    },
    GrammarEntry {
        name: "C11 (ISO/IEC 9899:2011)",
        language: "c",
        format: GrammarFormat::Ebnf,
        topics: &["language", "systems", "static"],
        source_org: "ISO",
        year: 2011,
        url: "https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf",
        description: "C11 draft N1570 Annex A — the normative grammar of C in ISO-style EBNF.",
    },
    GrammarEntry {
        name: "WebAssembly core specification",
        language: "webassembly",
        format: GrammarFormat::Ebnf,
        topics: &["web", "bytecode", "binary"],
        source_org: "W3C",
        year: 2022,
        url: "https://webassembly.github.io/spec/core/",
        description: "Wasm core spec with both text (.wat) and binary-format grammars.",
    },

    // ─── Data formats ────────────────────────────────────────────
    GrammarEntry {
        name: "JSON (RFC 8259)",
        language: "json",
        format: GrammarFormat::Abnf,
        topics: &["data", "serialization", "rfc"],
        source_org: "IETF",
        year: 2017,
        url: "https://www.rfc-editor.org/rfc/rfc8259",
        description: "The definitive JSON grammar in RFC 5234 ABNF.",
    },
    GrammarEntry {
        name: "TOML v1.0.0",
        language: "toml",
        format: GrammarFormat::Abnf,
        topics: &["data", "config"],
        source_org: "toml.io",
        year: 2021,
        url: "https://toml.io/en/v1.0.0",
        description: "TOML's reference grammar in ABNF.",
    },
    GrammarEntry {
        name: "YAML 1.2",
        language: "yaml",
        format: GrammarFormat::Ebnf,
        topics: &["data", "config"],
        source_org: "yaml.org",
        year: 2021,
        url: "https://yaml.org/spec/1.2.2/",
        description: "YAML 1.2.2 specification; indentation-sensitive productions in an EBNF-like notation.",
    },
    GrammarEntry {
        name: "CSS3 module grammars",
        language: "css",
        format: GrammarFormat::W3cEbnf,
        topics: &["web", "styling"],
        source_org: "W3C",
        year: 2023,
        url: "https://www.w3.org/TR/css-syntax-3/",
        description: "CSS Syntax Module Level 3 — tokenizer and parser grammar for CSS3.",
    },
    GrammarEntry {
        name: "SQL-2016 (ISO/IEC 9075)",
        language: "sql",
        format: GrammarFormat::Ebnf,
        topics: &["database", "query"],
        source_org: "ISO",
        year: 2016,
        url: "https://github.com/ronsavage/SQL/blob/master/sql-2016.ebnf",
        description: "Community-maintained EBNF extracted from the ISO/IEC 9075:2016 SQL standard.",
    },
    GrammarEntry {
        name: "GraphQL June 2018",
        language: "graphql",
        format: GrammarFormat::Ebnf,
        topics: &["web", "api", "query"],
        source_org: "Meta / GraphQL Foundation",
        year: 2018,
        url: "https://spec.graphql.org/June2018/",
        description: "GraphQL query language specification with lexical + syntactic grammar.",
    },

    // ─── IETF protocol grammars (ABNF) ───────────────────────────
    GrammarEntry {
        name: "HTTP/1.1 message syntax (RFC 9112)",
        language: "http",
        format: GrammarFormat::Abnf,
        topics: &["web", "protocol", "rfc"],
        source_org: "IETF",
        year: 2022,
        url: "https://www.rfc-editor.org/rfc/rfc9112",
        description: "Canonical ABNF for the HTTP/1.1 wire format, including start-line and fields.",
    },
    GrammarEntry {
        name: "HTTP semantics (RFC 9110)",
        language: "http",
        format: GrammarFormat::Abnf,
        topics: &["web", "protocol", "rfc"],
        source_org: "IETF",
        year: 2022,
        url: "https://www.rfc-editor.org/rfc/rfc9110",
        description: "HTTP semantics spec — methods, status codes, header field grammars in ABNF.",
    },
    GrammarEntry {
        name: "URI generic syntax (RFC 3986)",
        language: "uri",
        format: GrammarFormat::Abnf,
        topics: &["web", "protocol", "rfc"],
        source_org: "IETF",
        year: 2005,
        url: "https://www.rfc-editor.org/rfc/rfc3986",
        description: "The URI generic syntax — parse any URL/URI with this ABNF.",
    },
    GrammarEntry {
        name: "TLS 1.3 (RFC 8446)",
        language: "tls",
        format: GrammarFormat::Abnf,
        topics: &["crypto", "protocol", "rfc"],
        source_org: "IETF",
        year: 2018,
        url: "https://www.rfc-editor.org/rfc/rfc8446",
        description: "TLS 1.3 handshake and record layer grammars.",
    },
    GrammarEntry {
        name: "DNS message format (RFC 1035)",
        language: "dns",
        format: GrammarFormat::Abnf,
        topics: &["network", "protocol", "rfc"],
        source_org: "IETF",
        year: 1987,
        url: "https://www.rfc-editor.org/rfc/rfc1035",
        description: "DNS wire protocol and zone file grammar.",
    },
    GrammarEntry {
        name: "SMTP (RFC 5321)",
        language: "smtp",
        format: GrammarFormat::Abnf,
        topics: &["email", "protocol", "rfc"],
        source_org: "IETF",
        year: 2008,
        url: "https://www.rfc-editor.org/rfc/rfc5321",
        description: "Simple Mail Transfer Protocol command and reply syntax in ABNF.",
    },
    GrammarEntry {
        name: "IMAP4rev2 (RFC 9051)",
        language: "imap",
        format: GrammarFormat::Abnf,
        topics: &["email", "protocol", "rfc"],
        source_org: "IETF",
        year: 2021,
        url: "https://www.rfc-editor.org/rfc/rfc9051",
        description: "IMAP4rev2 command, response, and envelope grammar.",
    },
    GrammarEntry {
        name: "MIME (RFC 2045)",
        language: "mime",
        format: GrammarFormat::Abnf,
        topics: &["email", "format", "rfc"],
        source_org: "IETF",
        year: 1996,
        url: "https://www.rfc-editor.org/rfc/rfc2045",
        description: "MIME header field and content-type grammar.",
    },
    GrammarEntry {
        name: "OAuth 2.0 (RFC 6749)",
        language: "oauth",
        format: GrammarFormat::Abnf,
        topics: &["auth", "protocol", "rfc"],
        source_org: "IETF",
        year: 2012,
        url: "https://www.rfc-editor.org/rfc/rfc6749",
        description: "OAuth 2.0 authorization framework — request and response grammars.",
    },
    GrammarEntry {
        name: "WebSockets (RFC 6455)",
        language: "websocket",
        format: GrammarFormat::Abnf,
        topics: &["web", "protocol", "rfc"],
        source_org: "IETF",
        year: 2011,
        url: "https://www.rfc-editor.org/rfc/rfc6455",
        description: "WebSocket handshake and frame format grammar.",
    },

    // ─── Meta-grammars (the notations themselves) ────────────────
    GrammarEntry {
        name: "ABNF (RFC 5234)",
        language: "abnf",
        format: GrammarFormat::Abnf,
        topics: &["meta", "grammar", "rfc"],
        source_org: "IETF",
        year: 2008,
        url: "https://www.rfc-editor.org/rfc/rfc5234",
        description: "The ABNF specification itself, written in ABNF (self-referential).",
    },
    GrammarEntry {
        name: "ISO/IEC 14977 EBNF",
        language: "ebnf",
        format: GrammarFormat::Ebnf,
        topics: &["meta", "grammar", "iso"],
        source_org: "ISO",
        year: 1996,
        url: "https://www.cl.cam.ac.uk/~mgk25/iso-14977.pdf",
        description: "The ISO/IEC 14977 Extended Backus-Naur Form standard.",
    },

    // ─── Meta-pointers (aggregator repos and parser-generator crates) ─
    GrammarEntry {
        name: "ANTLR grammars-v4",
        language: "many",
        format: GrammarFormat::AntlrG4,
        topics: &["meta", "aggregator", "parser-generator"],
        source_org: "antlr.org",
        year: 2024,
        url: "https://github.com/antlr/grammars-v4",
        description: "Canonical aggregator: ~200 ANTLR .g4 grammars for mainstream programming languages.",
    },
    GrammarEntry {
        name: "GitHub Linguist vendor grammars",
        language: "many",
        format: GrammarFormat::AntlrG4,
        topics: &["meta", "aggregator", "syntax-highlighting"],
        source_org: "github/linguist",
        year: 2024,
        url: "https://github.com/github-linguist/linguist/blob/main/vendor/README.md",
        description: "GitHub Linguist's language definitions — TextMate grammars powering all of github.com syntax highlighting.",
    },
    GrammarEntry {
        name: "pest (Rust PEG parser)",
        language: "rust",
        format: GrammarFormat::Peg,
        topics: &["meta", "parser-generator", "library"],
        source_org: "pest-parser",
        year: 2024,
        url: "https://pest.rs/",
        description: "Canonical Rust PEG parser generator — use this for any grammar in PEG notation.",
    },
    GrammarEntry {
        name: "nom parser combinators",
        language: "rust",
        format: GrammarFormat::Peg,
        topics: &["meta", "parser-combinator", "library"],
        source_org: "rust-bakery",
        year: 2024,
        url: "https://github.com/rust-bakery/nom",
        description: "Rust's canonical parser-combinator library — ad-hoc grammars expressed in code.",
    },
];

// ──────────────────────────────────────────────────────────────────
// Query API
// ──────────────────────────────────────────────────────────────────

/// Return the full catalog as a static slice.
pub fn all() -> &'static [GrammarEntry] {
    CATALOG
}

/// Case-insensitive language filter. Entries whose `language` is
/// `"many"` (meta-pointers covering multiple languages) always pass.
pub fn by_language(lang: &str) -> Vec<GrammarEntry> {
    let q = lang.to_ascii_lowercase();
    CATALOG
        .iter()
        .filter(|e| e.language.eq_ignore_ascii_case(&q) || e.language == "many")
        .cloned()
        .collect()
}

/// Filter by grammar notation format.
pub fn by_format(format: GrammarFormat) -> Vec<GrammarEntry> {
    CATALOG
        .iter()
        .filter(|e| e.format == format)
        .cloned()
        .collect()
}

/// Filter by topic substring (case-insensitive). An entry matches
/// if any of its `topics` contains the needle.
pub fn by_topic(topic: &str) -> Vec<GrammarEntry> {
    let q = topic.to_ascii_lowercase();
    CATALOG
        .iter()
        .filter(|e| e.topics.iter().any(|t| string_contains_ci(t, &q)))
        .cloned()
        .collect()
}

/// Per-format count summary.
#[derive(Debug, Default, Serialize)]
pub struct GrammarCatalogCounts {
    pub total: usize,
    pub ebnf: usize,
    pub w3c_ebnf: usize,
    pub abnf: usize,
    pub peg: usize,
    pub antlr_g4: usize,
    pub bnf: usize,
}

/// Count entries per format.
pub fn counts() -> GrammarCatalogCounts {
    let mut c = GrammarCatalogCounts {
        total: CATALOG.len(),
        ..Default::default()
    };
    for e in CATALOG {
        match e.format {
            GrammarFormat::Ebnf => c.ebnf += 1,
            GrammarFormat::W3cEbnf => c.w3c_ebnf += 1,
            GrammarFormat::Abnf => c.abnf += 1,
            GrammarFormat::Peg => c.peg += 1,
            GrammarFormat::AntlrG4 => c.antlr_g4 += 1,
            GrammarFormat::Bnf => c.bnf += 1,
        }
    }
    c
}

// ──────────────────────────────────────────────────────────────────
// Catalog trait implementation
// ──────────────────────────────────────────────────────────────────

/// Zero-sized handle implementing [`Catalog`] for the grammar catalog.
pub struct GrammarCatalog;

impl GrammarFormat {
    fn parse_filter(s: &str) -> Option<Self> {
        let normalised = ix_catalog_core::normalize_snake_case(s);
        match normalised.as_str() {
            "ebnf" => Some(Self::Ebnf),
            "w3c_ebnf" | "w3cebnf" => Some(Self::W3cEbnf),
            "abnf" => Some(Self::Abnf),
            "peg" => Some(Self::Peg),
            "antlr_g4" | "antlr" | "g4" => Some(Self::AntlrG4),
            "bnf" => Some(Self::Bnf),
            _ => None,
        }
    }
}

impl Catalog for GrammarCatalog {
    fn name(&self) -> &'static str {
        "grammar"
    }

    fn scope(&self) -> &'static str {
        "Curated catalog of real-world grammar sources across EBNF, ABNF, \
         PEG, ANTLR G4, W3C EBNF, and BNF notations. Covers programming \
         languages, data formats, IETF protocols (HTTP, TLS, DNS, SMTP, \
         IMAP, OAuth, ...), and meta-grammars (ABNF, EBNF, ANTLR \
         grammars-v4). Not a mirror of the grammars themselves — a \
         curated index of where they live."
    }

    fn entry_count(&self) -> usize {
        CATALOG.len()
    }

    fn counts(&self) -> Value {
        let c = counts();
        json!({
            "total": c.total,
            "ebnf": c.ebnf,
            "w3c_ebnf": c.w3c_ebnf,
            "abnf": c.abnf,
            "peg": c.peg,
            "antlr_g4": c.antlr_g4,
            "bnf": c.bnf,
        })
    }

    fn query(&self, filter: Value) -> Result<Value, String> {
        let mut matched: Vec<GrammarEntry> = all().to_vec();

        if let Some(lang) = filter.get("language").and_then(|v| v.as_str()) {
            let filtered = by_language(lang);
            matched.retain(|e| filtered.iter().any(|f| f.name == e.name));
        }

        if let Some(fmt_str) = filter.get("format").and_then(|v| v.as_str()) {
            let fmt = GrammarFormat::parse_filter(fmt_str).ok_or_else(|| {
                format!(
                    "ix_grammar_catalog: unknown format '{fmt_str}' — expected one of: \
                     ebnf, w3c_ebnf, abnf, peg, antlr_g4, bnf"
                )
            })?;
            let filtered = by_format(fmt);
            matched.retain(|e| filtered.iter().any(|f| f.name == e.name));
        }

        if let Some(topic) = filter.get("topic").and_then(|v| v.as_str()) {
            let filtered = by_topic(topic);
            matched.retain(|e| filtered.iter().any(|f| f.name == e.name));
        }

        Ok(json!({
            "counts": self.counts(),
            "matched": matched.len(),
            "entries": matched,
        }))
    }
}

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_all_formats_represented() {
        let c = counts();
        assert!(c.total >= 25);
        assert!(c.abnf > 0, "no ABNF entries");
        assert!(c.ebnf > 0, "no EBNF entries");
        assert!(c.peg > 0, "no PEG entries");
        assert!(c.antlr_g4 > 0, "no ANTLR G4 entries");
        assert!(c.bnf > 0, "no BNF entries");
    }

    #[test]
    fn every_entry_is_well_formed() {
        for e in CATALOG {
            assert!(!e.name.is_empty(), "empty name");
            assert!(!e.url.is_empty(), "empty url for {}", e.name);
            assert!(
                !e.description.is_empty(),
                "empty description for {}",
                e.name
            );
            assert!(!e.language.is_empty(), "empty language for {}", e.name);
            assert!(e.year >= 1960, "bogus year for {}: {}", e.name, e.year);
            assert!(
                e.url.starts_with("http://") || e.url.starts_with("https://"),
                "{}: url must be http/https, got {}",
                e.name,
                e.url
            );
        }
    }

    #[test]
    fn python_query_returns_python_grammar() {
        let python = by_language("python");
        assert!(
            python.iter().any(|e| e.name.contains("Python")),
            "no Python grammar in language=python query"
        );
    }

    #[test]
    fn abnf_format_query_catches_the_rfc_protocol_block() {
        let abnf = by_format(GrammarFormat::Abnf);
        let names: Vec<&str> = abnf.iter().map(|e| e.name).collect();
        let required = [
            "HTTP/1.1 message syntax (RFC 9112)",
            "URI generic syntax (RFC 3986)",
            "JSON (RFC 8259)",
        ];
        for r in required {
            assert!(names.contains(&r), "missing {r} in ABNF query");
        }
    }

    #[test]
    fn counts_sum_to_total() {
        let c = counts();
        let sum = c.ebnf + c.w3c_ebnf + c.abnf + c.peg + c.antlr_g4 + c.bnf;
        assert_eq!(sum, c.total);
    }

    #[test]
    fn catalog_trait_query_round_trips() {
        let cat = GrammarCatalog;
        assert_eq!(cat.name(), "grammar");
        assert!(cat.entry_count() >= 25);

        // Empty filter returns everything.
        let all_json = cat.query(json!({})).expect("empty filter");
        let matched = all_json["matched"].as_u64().unwrap() as usize;
        assert_eq!(matched, cat.entry_count());

        // Format filter narrows.
        let abnf = cat.query(json!({ "format": "abnf" })).expect("abnf filter");
        let abnf_count = abnf["matched"].as_u64().unwrap() as usize;
        assert!(abnf_count < matched);
        assert!(abnf_count > 0);

        // Unknown format is a clean error.
        let err = cat
            .query(json!({ "format": "nope" }))
            .expect_err("unknown format");
        assert!(err.contains("unknown format"));
    }

    #[test]
    fn topic_filter_finds_rfc_entries() {
        let rfc = by_topic("rfc");
        assert!(
            rfc.len() >= 10,
            "expected 10+ RFC-tagged entries, got {}",
            rfc.len()
        );
    }
}
