//! # ix-net
//!
//! Networking-related catalogs and helpers for ix. Currently hosts
//! [`rfc_catalog`] — a curated inventory of IETF RFCs covering the
//! current internet stack, wired through the shared
//! [`ix_catalog_core::Catalog`](ix_catalog_core::Catalog) trait so
//! agents can query it uniformly via the `ix_rfc_catalog` MCP tool.
//!
//! Scope: the ~150 RFCs that define the modern internet (HTTP, TLS,
//! DNS, SMTP, IMAP, OAuth, TCP, IP, JSON, CBOR, URI, ABNF, etc.)
//! plus the obsolescence graph wiring them to their historical
//! predecessors. This is NOT a mirror of all ~9,000 RFCs —
//! rfc-editor.org is the authoritative full index. ix-net covers
//! what agents doing protocol work actually need to cite.

pub mod rfc_catalog;
