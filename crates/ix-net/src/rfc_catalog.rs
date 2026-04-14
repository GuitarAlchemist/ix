//! # Curated RFC catalog
//!
//! Queryable inventory of the ~100 IETF RFCs that define the
//! modern internet stack — from IP through HTTP/3, TLS 1.3, JSON,
//! OAuth, and the BCPs that ground the rest. Every entry includes
//! the authoritative URL on `rfc-editor.org`, plus an obsolescence
//! graph so agents can distinguish current standards from the
//! documents they replaced (the canonical example: quoting RFC 2616
//! when the current HTTP spec is RFC 9110).
//!
//! This catalog is **not** a mirror of all ~9,000 RFCs.
//! rfc-editor.org indexes the full series; ix-net indexes the
//! subset that actually shows up in agent work. Adding a new RFC
//! is a PR against this file — see the contribution notes in
//! [`docs/guides/rfc-catalog.md`](../../../docs/guides/rfc-catalog.md).
//!
//! ## Query API
//!
//! - [`all`] — full slice
//! - [`by_number`] — exact lookup
//! - [`by_topic`] — topic substring match
//! - [`by_status`] — filter by publication status
//! - [`current_standard_for`] — return non-obsoleted entries matching a topic
//! - [`obsolescence_chain`] — walk obsoletes/obsoleted_by in both
//!   directions to produce the complete chain for a given entry

use ix_catalog_core::{string_contains_ci, Catalog};
use serde::Serialize;
use serde_json::{json, Value};

/// Broad RFC category. Maps roughly to IETF stream classifications
/// but simplified so agents can filter usefully.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RfcCategory {
    /// Defines a wire protocol or data format.
    Protocol,
    /// Defines an architectural framework or operational model.
    Framework,
    /// Best Current Practice — IETF operational guidance.
    Bcp,
    /// Informational; does not define a standard.
    Informational,
    /// Experimental track — not yet a standard.
    Experimental,
    /// Historic — explicitly retired.
    Historic,
}

/// Publication status on the standards track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RfcStatus {
    /// Internet Standard (the highest level).
    InternetStandard,
    /// Proposed Standard — stable specification, widely implemented.
    ProposedStandard,
    /// Draft Standard — retired as a maturity level in RFC 6410
    /// but some older RFCs still carry it.
    DraftStandard,
    /// Experimental track.
    Experimental,
    /// Informational / historic / unknown — not on the standards track.
    Informational,
    /// Obsoleted by a later RFC. Agents should prefer the current
    /// spec via [`current_standard_for`].
    Obsoleted,
}

/// One RFC entry in the catalog.
#[derive(Debug, Clone, Serialize)]
pub struct RfcEntry {
    /// RFC number (e.g. 9110).
    pub number: u32,
    /// Title as published by the RFC Editor.
    pub title: &'static str,
    /// Broad category.
    pub category: RfcCategory,
    /// Current status. `Obsoleted` means superseded; the replacement
    /// is in the `obsoleted_by` list.
    pub status: RfcStatus,
    /// Year first published.
    pub year: u16,
    /// Topical tags used by [`by_topic`] queries.
    pub topics: &'static [&'static str],
    /// RFC numbers this RFC obsoletes (replaces).
    pub obsoletes: &'static [u32],
    /// RFC numbers that have obsoleted this RFC. If non-empty, this
    /// entry's `status` should be `Obsoleted`.
    pub obsoleted_by: &'static [u32],
    /// Canonical URL on rfc-editor.org.
    pub url: &'static str,
    /// One-sentence summary, written to be quotable in agent
    /// explanations.
    pub summary: &'static str,
}

/// The curated catalog. Ordered roughly by topic: IP layer first,
/// then transport, HTTP stack, DNS, email, security, formats, and
/// historical entries.
pub const CATALOG: &[RfcEntry] = &[
    // ──────────────────────────────────────────────────────────────
    // IP layer
    // ──────────────────────────────────────────────────────────────
    RfcEntry {
        number: 791,
        title: "Internet Protocol",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 1981,
        topics: &["ip", "network", "ipv4"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc791",
        summary: "The original IPv4 specification — still the definitive reference for the v4 header format, fragmentation, and routing semantics.",
    },
    RfcEntry {
        number: 8200,
        title: "Internet Protocol, Version 6 (IPv6) Specification",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 2017,
        topics: &["ip", "network", "ipv6"],
        obsoletes: &[2460],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc8200",
        summary: "Current IPv6 core specification; defines header format, extension headers, and addressing.",
    },
    RfcEntry {
        number: 2460,
        title: "Internet Protocol, Version 6 (IPv6) Specification",
        category: RfcCategory::Protocol,
        status: RfcStatus::Obsoleted,
        year: 1998,
        topics: &["ip", "network", "ipv6", "historical"],
        obsoletes: &[],
        obsoleted_by: &[8200],
        url: "https://www.rfc-editor.org/rfc/rfc2460",
        summary: "Original IPv6 spec, superseded by RFC 8200 in 2017.",
    },
    RfcEntry {
        number: 792,
        title: "Internet Control Message Protocol",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 1981,
        topics: &["icmp", "network", "ipv4"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc792",
        summary: "ICMP for IPv4: error reporting, echo request/reply, traceroute underpinnings.",
    },
    RfcEntry {
        number: 4443,
        title: "Internet Control Message Protocol (ICMPv6) for the IPv6 Specification",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 2006,
        topics: &["icmp", "network", "ipv6"],
        obsoletes: &[2463],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc4443",
        summary: "ICMP for IPv6 — error messages, neighbor discovery signaling.",
    },
    // ──────────────────────────────────────────────────────────────
    // Transport layer
    // ──────────────────────────────────────────────────────────────
    RfcEntry {
        number: 9293,
        title: "Transmission Control Protocol (TCP)",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 2022,
        topics: &["tcp", "transport"],
        obsoletes: &[793, 879, 2873, 6093, 6429, 6528, 6691],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc9293",
        summary: "Modern consolidated TCP specification, replacing RFC 793 and six incremental updates.",
    },
    RfcEntry {
        number: 793,
        title: "Transmission Control Protocol",
        category: RfcCategory::Protocol,
        status: RfcStatus::Obsoleted,
        year: 1981,
        topics: &["tcp", "transport", "historical"],
        obsoletes: &[],
        obsoleted_by: &[9293],
        url: "https://www.rfc-editor.org/rfc/rfc793",
        summary: "The original Postel TCP spec; the canonical historical citation, obsoleted in 2022 by RFC 9293.",
    },
    RfcEntry {
        number: 768,
        title: "User Datagram Protocol",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 1980,
        topics: &["udp", "transport"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc768",
        summary: "Two-page UDP specification; the oldest still-current internet standard.",
    },
    RfcEntry {
        number: 9000,
        title: "QUIC: A UDP-Based Multiplexed and Secure Transport",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2021,
        topics: &["quic", "transport", "http3"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc9000",
        summary: "QUIC core protocol — stream multiplexing, connection migration, built-in encryption.",
    },
    RfcEntry {
        number: 9001,
        title: "Using TLS to Secure QUIC",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2021,
        topics: &["quic", "tls", "transport"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc9001",
        summary: "How QUIC uses TLS 1.3 for its handshake and key schedule.",
    },
    // ──────────────────────────────────────────────────────────────
    // HTTP stack
    // ──────────────────────────────────────────────────────────────
    RfcEntry {
        number: 9110,
        title: "HTTP Semantics",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 2022,
        topics: &["http", "web", "semantics"],
        // 7234 (HTTP Caching) is obsoleted by 9111, not 9110.
        // 7230 is jointly obsoleted by 9110 (semantics) and 9112
        // (message syntax) per the 9110 and 9112 abstracts.
        obsoletes: &[2818, 7230, 7231, 7232, 7233, 7235],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc9110",
        summary: "The current HTTP semantics specification — methods, status codes, headers, content negotiation.",
    },
    RfcEntry {
        number: 9111,
        title: "HTTP Caching",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 2022,
        topics: &["http", "caching", "web"],
        obsoletes: &[7234],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc9111",
        summary: "HTTP caching semantics: freshness, validation, cache control, Vary header.",
    },
    RfcEntry {
        number: 9112,
        title: "HTTP/1.1",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 2022,
        topics: &["http", "http1", "web"],
        obsoletes: &[7230],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc9112",
        summary: "HTTP/1.1 message syntax and connection management in ABNF.",
    },
    RfcEntry {
        number: 9113,
        title: "HTTP/2",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2022,
        topics: &["http", "http2", "web"],
        obsoletes: &[7540, 8740],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc9113",
        summary: "HTTP/2 — binary framing, multiplexing, header compression (HPACK).",
    },
    RfcEntry {
        number: 9114,
        title: "HTTP/3",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2022,
        topics: &["http", "http3", "quic", "web"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc9114",
        summary: "HTTP/3 over QUIC — the modern web transport replacing HTTP/2 over TCP.",
    },
    RfcEntry {
        number: 2616,
        title: "Hypertext Transfer Protocol -- HTTP/1.1",
        category: RfcCategory::Protocol,
        status: RfcStatus::Obsoleted,
        year: 1999,
        topics: &["http", "historical"],
        obsoletes: &[2068],
        obsoleted_by: &[7230, 7231, 7232, 7233, 7234, 7235],
        url: "https://www.rfc-editor.org/rfc/rfc2616",
        summary: "Legacy HTTP/1.1 spec. Obsoleted by RFC 7230-7235 in 2014, which themselves were consolidated into RFC 9110+9111+9112 in 2022.",
    },
    RfcEntry {
        number: 7230,
        title: "Hypertext Transfer Protocol (HTTP/1.1): Message Syntax and Routing",
        category: RfcCategory::Protocol,
        status: RfcStatus::Obsoleted,
        year: 2014,
        topics: &["http", "historical"],
        obsoletes: &[2616],
        obsoleted_by: &[9110, 9112],
        url: "https://www.rfc-editor.org/rfc/rfc7230",
        summary: "HTTP/1.1 message syntax. Obsoleted by RFC 9110 (semantics) + RFC 9112 (HTTP/1.1).",
    },
    RfcEntry {
        number: 7231,
        title: "HTTP/1.1 Semantics and Content",
        category: RfcCategory::Protocol,
        status: RfcStatus::Obsoleted,
        year: 2014,
        topics: &["http", "historical"],
        obsoletes: &[2616],
        obsoleted_by: &[9110],
        url: "https://www.rfc-editor.org/rfc/rfc7231",
        summary: "HTTP/1.1 semantics. Obsoleted by RFC 9110.",
    },
    RfcEntry {
        number: 7232,
        title: "HTTP/1.1 Conditional Requests",
        category: RfcCategory::Protocol,
        status: RfcStatus::Obsoleted,
        year: 2014,
        topics: &["http", "historical"],
        obsoletes: &[2616],
        obsoleted_by: &[9110],
        url: "https://www.rfc-editor.org/rfc/rfc7232",
        summary: "HTTP conditional requests (If-Match, If-Modified-Since, etc). Obsoleted by RFC 9110.",
    },
    RfcEntry {
        number: 7233,
        title: "HTTP/1.1 Range Requests",
        category: RfcCategory::Protocol,
        status: RfcStatus::Obsoleted,
        year: 2014,
        topics: &["http", "historical"],
        obsoletes: &[2616],
        obsoleted_by: &[9110],
        url: "https://www.rfc-editor.org/rfc/rfc7233",
        summary: "HTTP range requests. Obsoleted by RFC 9110.",
    },
    RfcEntry {
        number: 7234,
        title: "HTTP/1.1 Caching",
        category: RfcCategory::Protocol,
        status: RfcStatus::Obsoleted,
        year: 2014,
        topics: &["http", "caching", "historical"],
        obsoletes: &[2616],
        obsoleted_by: &[9111],
        url: "https://www.rfc-editor.org/rfc/rfc7234",
        summary: "HTTP/1.1 caching. Obsoleted by RFC 9111.",
    },
    RfcEntry {
        number: 7235,
        title: "HTTP/1.1 Authentication",
        category: RfcCategory::Protocol,
        status: RfcStatus::Obsoleted,
        year: 2014,
        topics: &["http", "auth", "historical"],
        obsoletes: &[2616],
        obsoleted_by: &[9110],
        url: "https://www.rfc-editor.org/rfc/rfc7235",
        summary: "HTTP authentication framework. Obsoleted by RFC 9110.",
    },
    RfcEntry {
        number: 7540,
        title: "Hypertext Transfer Protocol Version 2 (HTTP/2)",
        category: RfcCategory::Protocol,
        status: RfcStatus::Obsoleted,
        year: 2015,
        topics: &["http", "http2", "historical"],
        obsoletes: &[],
        obsoleted_by: &[9113],
        url: "https://www.rfc-editor.org/rfc/rfc7540",
        summary: "Original HTTP/2 specification. Obsoleted by RFC 9113.",
    },
    RfcEntry {
        number: 3986,
        title: "Uniform Resource Identifier (URI): Generic Syntax",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 2005,
        topics: &["uri", "url", "web"],
        obsoletes: &[2732, 2396, 1808],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc3986",
        summary: "The URI generic syntax — the ABNF every URL parser must implement.",
    },
    RfcEntry {
        number: 6455,
        title: "The WebSocket Protocol",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2011,
        topics: &["websocket", "web"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc6455",
        summary: "WebSockets: handshake, frame format, extensions — full-duplex over HTTP upgrade.",
    },
    // ──────────────────────────────────────────────────────────────
    // TLS / security
    // ──────────────────────────────────────────────────────────────
    RfcEntry {
        number: 8446,
        title: "The Transport Layer Security (TLS) Protocol Version 1.3",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2018,
        topics: &["tls", "crypto", "security"],
        obsoletes: &[5077, 5246, 6961],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc8446",
        summary: "TLS 1.3 — the current internet-wide transport security protocol, mandatory for HTTPS.",
    },
    RfcEntry {
        number: 5246,
        title: "The Transport Layer Security (TLS) Protocol Version 1.2",
        category: RfcCategory::Protocol,
        status: RfcStatus::Obsoleted,
        year: 2008,
        topics: &["tls", "crypto", "historical"],
        obsoletes: &[3268, 4346],
        obsoleted_by: &[8446],
        url: "https://www.rfc-editor.org/rfc/rfc5246",
        summary: "TLS 1.2 — superseded by RFC 8446 (TLS 1.3) in 2018 but still widely deployed.",
    },
    RfcEntry {
        number: 9147,
        title: "The Datagram Transport Layer Security (DTLS) Protocol Version 1.3",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2022,
        topics: &["dtls", "tls", "crypto", "udp"],
        obsoletes: &[6347],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc9147",
        summary: "DTLS 1.3 — TLS 1.3 adapted for unreliable datagram transports (UDP, QUIC).",
    },
    RfcEntry {
        number: 5280,
        title: "Internet X.509 Public Key Infrastructure Certificate and Certificate Revocation List (CRL) Profile",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2008,
        topics: &["pki", "x509", "crypto"],
        obsoletes: &[3280],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc5280",
        summary: "X.509 certificate profile — the format every HTTPS certificate conforms to.",
    },
    RfcEntry {
        number: 6749,
        title: "The OAuth 2.0 Authorization Framework",
        category: RfcCategory::Framework,
        status: RfcStatus::ProposedStandard,
        year: 2012,
        topics: &["oauth", "auth", "security"],
        obsoletes: &[5849],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc6749",
        summary: "OAuth 2.0 authorization framework — the foundation of modern delegated access.",
    },
    RfcEntry {
        number: 7519,
        title: "JSON Web Token (JWT)",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2015,
        topics: &["jwt", "jose", "auth", "security"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc7519",
        summary: "JWT — the compact, URL-safe token format used by OAuth, OIDC, and countless auth systems.",
    },
    RfcEntry {
        number: 7515,
        title: "JSON Web Signature (JWS)",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2015,
        topics: &["jws", "jose", "crypto"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc7515",
        summary: "JWS — integrity-protected content format, the signature layer of JWT.",
    },
    RfcEntry {
        number: 7516,
        title: "JSON Web Encryption (JWE)",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2015,
        topics: &["jwe", "jose", "crypto"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc7516",
        summary: "JWE — encrypted content format in the JOSE family.",
    },
    RfcEntry {
        number: 7517,
        title: "JSON Web Key (JWK)",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2015,
        topics: &["jwk", "jose", "crypto"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc7517",
        summary: "JWK — JSON representation of cryptographic keys used by JOSE / JWT.",
    },
    RfcEntry {
        number: 7518,
        title: "JSON Web Algorithms (JWA)",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2015,
        topics: &["jwa", "jose", "crypto"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc7518",
        summary: "JWA — algorithm identifiers (RS256, ES256, HS256, ...) used across JOSE.",
    },
    // ──────────────────────────────────────────────────────────────
    // DNS
    // ──────────────────────────────────────────────────────────────
    RfcEntry {
        number: 1034,
        title: "Domain Names - Concepts and Facilities",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 1987,
        topics: &["dns", "naming"],
        obsoletes: &[882, 883, 973],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc1034",
        summary: "DNS concepts — the architecture document companion to RFC 1035.",
    },
    RfcEntry {
        number: 1035,
        title: "Domain Names - Implementation and Specification",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 1987,
        topics: &["dns", "naming", "protocol"],
        obsoletes: &[882, 883, 973],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc1035",
        summary: "DNS wire protocol — message format, zone files, record types. Every resolver implements this.",
    },
    RfcEntry {
        number: 4033,
        title: "DNS Security Introduction and Requirements",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2005,
        topics: &["dns", "dnssec", "security"],
        obsoletes: &[2535, 3008, 3090, 3445, 3655, 3658, 3755, 3757, 3845],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc4033",
        summary: "DNSSEC overview — why DNS needs cryptographic authentication.",
    },
    RfcEntry {
        number: 4034,
        title: "Resource Records for the DNS Security Extensions",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2005,
        topics: &["dns", "dnssec", "security"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc4034",
        summary: "DNSSEC record types: DNSKEY, RRSIG, NSEC, DS.",
    },
    RfcEntry {
        number: 4035,
        title: "Protocol Modifications for the DNS Security Extensions",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2005,
        topics: &["dns", "dnssec", "security"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc4035",
        summary: "DNSSEC protocol behavior — how resolvers and servers handle signed responses.",
    },
    RfcEntry {
        number: 8484,
        title: "DNS Queries over HTTPS (DoH)",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2018,
        topics: &["dns", "doh", "privacy", "http"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc8484",
        summary: "DoH — tunnels DNS queries over HTTPS for privacy and censorship resistance.",
    },
    RfcEntry {
        number: 7858,
        title: "Specification for DNS over Transport Layer Security (TLS)",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2016,
        topics: &["dns", "dot", "privacy", "tls"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc7858",
        summary: "DoT — encrypted DNS over TLS on port 853.",
    },
    RfcEntry {
        number: 6891,
        title: "Extension Mechanisms for DNS (EDNS(0))",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 2013,
        topics: &["dns", "edns"],
        obsoletes: &[2671],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc6891",
        summary: "EDNS(0) — the extension mechanism that makes DNS carry payloads larger than 512 bytes.",
    },
    // ──────────────────────────────────────────────────────────────
    // Email
    // ──────────────────────────────────────────────────────────────
    RfcEntry {
        number: 5321,
        title: "Simple Mail Transfer Protocol",
        category: RfcCategory::Protocol,
        status: RfcStatus::DraftStandard,
        year: 2008,
        topics: &["smtp", "email"],
        obsoletes: &[2821],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc5321",
        summary: "Current SMTP specification. Obsoletes RFC 2821 (which obsoleted 821).",
    },
    RfcEntry {
        number: 5322,
        title: "Internet Message Format",
        category: RfcCategory::Protocol,
        status: RfcStatus::DraftStandard,
        year: 2008,
        topics: &["email", "format"],
        obsoletes: &[2822],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc5322",
        summary: "Email message format — the From, To, Subject, date, and body rules every mail client follows.",
    },
    RfcEntry {
        number: 9051,
        title: "Internet Message Access Protocol (IMAP) - Version 4rev2",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2021,
        topics: &["imap", "email"],
        obsoletes: &[3501],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc9051",
        summary: "IMAP4rev2 — current remote mailbox access protocol, obsoletes the long-running RFC 3501.",
    },
    RfcEntry {
        number: 3501,
        title: "Internet Message Access Protocol - Version 4rev1",
        category: RfcCategory::Protocol,
        status: RfcStatus::Obsoleted,
        year: 2003,
        topics: &["imap", "email", "historical"],
        obsoletes: &[2060],
        obsoleted_by: &[9051],
        url: "https://www.rfc-editor.org/rfc/rfc3501",
        summary: "IMAP4rev1 — the long-running IMAP spec, obsoleted by IMAP4rev2 (RFC 9051) in 2021.",
    },
    RfcEntry {
        number: 6376,
        title: "DomainKeys Identified Mail (DKIM) Signatures",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 2011,
        topics: &["dkim", "email", "auth"],
        obsoletes: &[4871, 5672],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc6376",
        summary: "DKIM signs outgoing email with a domain-owned key; receivers verify.",
    },
    RfcEntry {
        number: 7208,
        title: "Sender Policy Framework (SPF)",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2014,
        topics: &["spf", "email", "auth"],
        obsoletes: &[4408],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc7208",
        summary: "SPF — DNS TXT records declaring which IPs may send mail for a domain.",
    },
    RfcEntry {
        number: 7489,
        title: "Domain-based Message Authentication, Reporting, and Conformance (DMARC)",
        category: RfcCategory::Informational,
        status: RfcStatus::Informational,
        year: 2015,
        topics: &["dmarc", "email", "auth"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc7489",
        summary: "DMARC ties SPF and DKIM together with a published policy and aggregate reports.",
    },
    RfcEntry {
        number: 2045,
        title: "Multipurpose Internet Mail Extensions (MIME) Part One: Format of Internet Message Bodies",
        category: RfcCategory::Protocol,
        status: RfcStatus::DraftStandard,
        year: 1996,
        topics: &["mime", "email", "format"],
        obsoletes: &[1521, 1522, 1590],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc2045",
        summary: "MIME message body format — the foundation of attachments, multipart, and content-type.",
    },
    RfcEntry {
        number: 2046,
        title: "MIME Part Two: Media Types",
        category: RfcCategory::Protocol,
        status: RfcStatus::DraftStandard,
        year: 1996,
        topics: &["mime", "email", "format"],
        obsoletes: &[1521, 1522, 1590],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc2046",
        summary: "MIME media types — text/*, image/*, multipart/*, application/*, etc.",
    },
    // ──────────────────────────────────────────────────────────────
    // Realtime
    // ──────────────────────────────────────────────────────────────
    RfcEntry {
        number: 3261,
        title: "SIP: Session Initiation Protocol",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2002,
        topics: &["sip", "voip", "realtime"],
        obsoletes: &[2543],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc3261",
        summary: "SIP — session initiation for voice-over-IP, video calling, instant messaging.",
    },
    RfcEntry {
        number: 3550,
        title: "RTP: A Transport Protocol for Real-Time Applications",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 2003,
        topics: &["rtp", "voip", "realtime"],
        obsoletes: &[1889],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc3550",
        summary: "RTP + RTCP — the packet format + control protocol for real-time media streams.",
    },
    RfcEntry {
        number: 8825,
        title: "Overview: Real-Time Protocols for Browser-Based Applications",
        category: RfcCategory::Informational,
        status: RfcStatus::Informational,
        year: 2021,
        topics: &["webrtc", "realtime"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc8825",
        summary: "WebRTC protocol suite overview — the architecture doc tying together ICE, DTLS, SRTP, SDP.",
    },
    // ──────────────────────────────────────────────────────────────
    // Data formats
    // ──────────────────────────────────────────────────────────────
    RfcEntry {
        number: 8259,
        title: "The JavaScript Object Notation (JSON) Data Interchange Format",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 2017,
        topics: &["json", "format"],
        obsoletes: &[7159],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc8259",
        summary: "The definitive JSON specification, in ABNF. Simultaneously an IETF RFC and an ECMA standard.",
    },
    RfcEntry {
        number: 8949,
        title: "Concise Binary Object Representation (CBOR)",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 2020,
        topics: &["cbor", "format", "binary"],
        obsoletes: &[7049],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc8949",
        summary: "CBOR — a JSON-like binary format, designed for small code size and small message size.",
    },
    RfcEntry {
        number: 9562,
        title: "Universally Unique IDentifiers (UUIDs)",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2024,
        topics: &["uuid", "format"],
        obsoletes: &[4122],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc9562",
        summary: "Modern UUID specification including v6, v7, v8 variants. Obsoletes RFC 4122.",
    },
    RfcEntry {
        number: 4122,
        title: "A Universally Unique IDentifier (UUID) URN Namespace",
        category: RfcCategory::Protocol,
        status: RfcStatus::Obsoleted,
        year: 2005,
        topics: &["uuid", "format", "historical"],
        obsoletes: &[],
        obsoleted_by: &[9562],
        url: "https://www.rfc-editor.org/rfc/rfc4122",
        summary: "Original UUID spec defining v1, v3, v4, v5. Obsoleted by RFC 9562.",
    },
    // ──────────────────────────────────────────────────────────────
    // Meta / grammars / keywords
    // ──────────────────────────────────────────────────────────────
    RfcEntry {
        number: 5234,
        title: "Augmented BNF for Syntax Specifications: ABNF",
        category: RfcCategory::Protocol,
        status: RfcStatus::InternetStandard,
        year: 2008,
        topics: &["abnf", "grammar", "meta"],
        obsoletes: &[4234],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc5234",
        summary: "ABNF — the grammar notation every modern IETF spec uses. Self-referential (defined in ABNF).",
    },
    RfcEntry {
        number: 2119,
        title: "Key words for use in RFCs to Indicate Requirement Levels",
        category: RfcCategory::Bcp,
        status: RfcStatus::InternetStandard,
        year: 1997,
        topics: &["bcp", "keywords", "meta"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc2119",
        summary: "BCP 14 — defines MUST / SHOULD / MAY semantics used by every IETF spec.",
    },
    RfcEntry {
        number: 8174,
        title: "Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words",
        category: RfcCategory::Bcp,
        status: RfcStatus::InternetStandard,
        year: 2017,
        topics: &["bcp", "keywords", "meta"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc8174",
        summary: "Clarifies that only UPPERCASE RFC 2119 keywords carry normative weight.",
    },
    RfcEntry {
        number: 5646,
        title: "Tags for Identifying Languages",
        category: RfcCategory::Bcp,
        status: RfcStatus::InternetStandard,
        year: 2009,
        topics: &["bcp", "languages", "i18n"],
        obsoletes: &[4646],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc5646",
        summary: "BCP 47 — the IETF language tag grammar (en, en-US, zh-Hant, ...).",
    },
    // ──────────────────────────────────────────────────────────────
    // SSH
    // ──────────────────────────────────────────────────────────────
    RfcEntry {
        number: 4251,
        title: "The Secure Shell (SSH) Protocol Architecture",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2006,
        topics: &["ssh", "security"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc4251",
        summary: "SSH protocol architecture — transport, authentication, and connection layers.",
    },
    RfcEntry {
        number: 4252,
        title: "The Secure Shell (SSH) Authentication Protocol",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2006,
        topics: &["ssh", "auth"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc4252",
        summary: "SSH authentication methods: publickey, password, hostbased.",
    },
    RfcEntry {
        number: 4253,
        title: "The Secure Shell (SSH) Transport Layer Protocol",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2006,
        topics: &["ssh", "crypto"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc4253",
        summary: "SSH transport layer — packet format, key exchange, symmetric encryption.",
    },
    RfcEntry {
        number: 4254,
        title: "The Secure Shell (SSH) Connection Protocol",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2006,
        topics: &["ssh"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc4254",
        summary: "SSH connection protocol — channel multiplexing, sessions, port forwarding.",
    },
    // ──────────────────────────────────────────────────────────────
    // NTP, misc
    // ──────────────────────────────────────────────────────────────
    RfcEntry {
        number: 5905,
        title: "Network Time Protocol Version 4: Protocol and Algorithms Specification",
        category: RfcCategory::Protocol,
        status: RfcStatus::ProposedStandard,
        year: 2010,
        topics: &["ntp", "time"],
        obsoletes: &[1305, 4330],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc5905",
        summary: "NTPv4 — time synchronization across the internet; foundation for everything time-sensitive.",
    },
    // ──────────────────────────────────────────────────────────────
    // Historical honorable mentions
    // ──────────────────────────────────────────────────────────────
    RfcEntry {
        number: 1,
        title: "Host Software",
        category: RfcCategory::Historic,
        status: RfcStatus::Informational,
        year: 1969,
        topics: &["historical", "history"],
        obsoletes: &[],
        obsoleted_by: &[],
        url: "https://www.rfc-editor.org/rfc/rfc1",
        summary: "The first RFC, by Steve Crocker, April 1969. ARPANET host-to-host protocol design notes.",
    },
];

// ──────────────────────────────────────────────────────────────────
// Query API
// ──────────────────────────────────────────────────────────────────

/// Full catalog slice.
pub fn all() -> &'static [RfcEntry] {
    CATALOG
}

/// Look up an RFC by number. Returns `None` if the RFC is not in
/// the curated catalog — check rfc-editor.org for the full index.
pub fn by_number(n: u32) -> Option<RfcEntry> {
    CATALOG.iter().find(|e| e.number == n).cloned()
}

/// Topic substring match (case-insensitive).
pub fn by_topic(topic: &str) -> Vec<RfcEntry> {
    CATALOG
        .iter()
        .filter(|e| e.topics.iter().any(|t| string_contains_ci(t, topic)))
        .cloned()
        .collect()
}

/// Filter by publication status.
pub fn by_status(status: RfcStatus) -> Vec<RfcEntry> {
    CATALOG.iter().filter(|e| e.status == status).cloned().collect()
}

/// Non-obsoleted entries matching a topic. Use this when you want
/// "the current spec for X" and don't want to accidentally cite a
/// retired RFC.
pub fn current_standard_for(topic: &str) -> Vec<RfcEntry> {
    by_topic(topic)
        .into_iter()
        .filter(|e| e.status != RfcStatus::Obsoleted && e.obsoleted_by.is_empty())
        .collect()
}

/// Walk the obsolescence graph in both directions starting from
/// `n`. Returns every entry in the chain of obsoletes /
/// obsoleted_by, deduplicated and starting with the seed entry.
///
/// Useful for answering "which RFC is the live version of RFC X"
/// and "what did RFC X replace".
pub fn obsolescence_chain(n: u32) -> Vec<RfcEntry> {
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<RfcEntry> = Vec::new();
    let mut stack: Vec<u32> = vec![n];

    while let Some(num) = stack.pop() {
        if !seen.insert(num) {
            continue;
        }
        if let Some(entry) = by_number(num) {
            for prev in entry.obsoletes {
                stack.push(*prev);
            }
            for next in entry.obsoleted_by {
                stack.push(*next);
            }
            out.push(entry);
        }
    }
    out.sort_by_key(|e| e.number);
    out
}

/// Per-status count summary.
#[derive(Debug, Default, Serialize)]
pub struct RfcCatalogCounts {
    pub total: usize,
    pub internet_standard: usize,
    pub proposed_standard: usize,
    pub draft_standard: usize,
    pub experimental: usize,
    pub informational: usize,
    pub obsoleted: usize,
}

/// Count entries per status.
pub fn counts() -> RfcCatalogCounts {
    let mut c = RfcCatalogCounts {
        total: CATALOG.len(),
        ..Default::default()
    };
    for e in CATALOG {
        match e.status {
            RfcStatus::InternetStandard => c.internet_standard += 1,
            RfcStatus::ProposedStandard => c.proposed_standard += 1,
            RfcStatus::DraftStandard => c.draft_standard += 1,
            RfcStatus::Experimental => c.experimental += 1,
            RfcStatus::Informational => c.informational += 1,
            RfcStatus::Obsoleted => c.obsoleted += 1,
        }
    }
    c
}

// ──────────────────────────────────────────────────────────────────
// Catalog trait implementation
// ──────────────────────────────────────────────────────────────────

/// Zero-sized handle implementing [`Catalog`] for the RFC catalog.
pub struct RfcCatalog;

impl RfcStatus {
    fn parse_filter(s: &str) -> Option<Self> {
        let normalised = ix_catalog_core::normalize_snake_case(s);
        match normalised.as_str() {
            "internet_standard" | "std" => Some(Self::InternetStandard),
            "proposed_standard" | "proposed" => Some(Self::ProposedStandard),
            "draft_standard" | "draft" => Some(Self::DraftStandard),
            "experimental" => Some(Self::Experimental),
            "informational" | "info" => Some(Self::Informational),
            "obsoleted" | "obsolete" => Some(Self::Obsoleted),
            _ => None,
        }
    }
}

impl Catalog for RfcCatalog {
    fn name(&self) -> &'static str {
        "rfc"
    }

    fn scope(&self) -> &'static str {
        "Curated catalog of IETF RFCs covering the modern internet stack: \
         IP, TCP, UDP, QUIC, HTTP/1.1/2/3, TLS 1.3, DNS + DNSSEC, SMTP, \
         IMAP, OAuth, JOSE/JWT, SSH, SIP, RTP, JSON, CBOR, UUID, ABNF, \
         NTP, plus BCPs and key historical references. Includes the \
         obsolescence graph so agents can distinguish current standards \
         from the documents they replaced (e.g. RFC 9110 obsoletes \
         2616 + 7230-7235). NOT a mirror of all ~9000 RFCs — see \
         rfc-editor.org for the full index."
    }

    fn entry_count(&self) -> usize {
        CATALOG.len()
    }

    fn counts(&self) -> Value {
        let c = counts();
        json!({
            "total": c.total,
            "internet_standard": c.internet_standard,
            "proposed_standard": c.proposed_standard,
            "draft_standard": c.draft_standard,
            "experimental": c.experimental,
            "informational": c.informational,
            "obsoleted": c.obsoleted,
        })
    }

    fn query(&self, filter: Value) -> Result<Value, String> {
        let mut matched: Vec<RfcEntry> = all().to_vec();

        if let Some(n) = filter.get("number").and_then(|v| v.as_u64()) {
            matched.retain(|e| e.number as u64 == n);
        }

        if let Some(topic) = filter.get("topic").and_then(|v| v.as_str()) {
            let filtered = by_topic(topic);
            matched.retain(|e| filtered.iter().any(|f| f.number == e.number));
        }

        if let Some(status_str) = filter.get("status").and_then(|v| v.as_str()) {
            let status = RfcStatus::parse_filter(status_str).ok_or_else(|| {
                format!(
                    "ix_rfc_catalog: unknown status '{status_str}' — expected one of: \
                     internet_standard, proposed_standard, draft_standard, \
                     experimental, informational, obsoleted"
                )
            })?;
            matched.retain(|e| e.status == status);
        }

        // Optional: current_standard=true filters out obsoleted entries.
        if filter
            .get("current_standard")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            matched.retain(|e| e.status != RfcStatus::Obsoleted && e.obsoleted_by.is_empty());
        }

        // Optional: obsolescence_chain=N returns just the chain.
        if let Some(n) = filter
            .get("obsolescence_chain")
            .and_then(|v| v.as_u64())
        {
            let chain = obsolescence_chain(n as u32);
            return Ok(json!({
                "counts": self.counts(),
                "matched": chain.len(),
                "entries": chain,
                "chain_for": n,
            }));
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
    fn catalog_has_reasonable_size() {
        let c = counts();
        assert!(c.total >= 60, "expected 60+ entries, got {}", c.total);
        assert!(c.internet_standard > 0);
        assert!(c.proposed_standard > 0);
        assert!(c.obsoleted > 0);
    }

    #[test]
    fn every_entry_is_well_formed() {
        let mut seen_numbers = std::collections::HashSet::new();
        for e in CATALOG {
            assert!(e.number > 0, "RFC number must be positive");
            assert!(
                seen_numbers.insert(e.number),
                "RFC {} is duplicated in the catalog",
                e.number
            );
            assert!(!e.title.is_empty(), "RFC {} has empty title", e.number);
            assert!(
                e.url.starts_with("https://www.rfc-editor.org/rfc/rfc"),
                "RFC {} URL must be canonical rfc-editor, got {}",
                e.number,
                e.url
            );
            assert!(
                e.url.contains(&e.number.to_string()),
                "RFC {} URL must reference its own number",
                e.number
            );
            assert!(!e.summary.is_empty(), "RFC {} has empty summary", e.number);
            assert!(e.year >= 1969, "RFC {} has year < 1969", e.number);
        }
    }

    #[test]
    fn obsolescence_graph_is_symmetric_for_recorded_entries() {
        // Build a lookup so we can cross-check every edge.
        let by_num: std::collections::HashMap<u32, &RfcEntry> =
            CATALOG.iter().map(|e| (e.number, e)).collect();

        for e in CATALOG {
            // Every RFC this entry obsoletes should list us in its
            // obsoleted_by field — if that predecessor is in the
            // catalog.
            for prev in e.obsoletes {
                if let Some(pe) = by_num.get(prev) {
                    assert!(
                        pe.obsoleted_by.contains(&e.number),
                        "RFC {} obsoletes {}, but {} does not list {} in obsoleted_by",
                        e.number,
                        prev,
                        prev,
                        e.number
                    );
                }
            }
            // Every RFC that obsoletes this one should list us in
            // its obsoletes field.
            for next in e.obsoleted_by {
                if let Some(ne) = by_num.get(next) {
                    assert!(
                        ne.obsoletes.contains(&e.number),
                        "RFC {} lists {} in obsoleted_by, but {} does not list {} in obsoletes",
                        e.number,
                        next,
                        next,
                        e.number
                    );
                }
            }
        }
    }

    #[test]
    fn current_standard_for_http_returns_9110_and_not_2616() {
        let http = current_standard_for("http");
        let numbers: Vec<u32> = http.iter().map(|e| e.number).collect();
        assert!(
            numbers.contains(&9110),
            "current HTTP spec should include 9110; got {:?}",
            numbers
        );
        assert!(
            !numbers.contains(&2616),
            "current HTTP spec must NOT include obsoleted 2616; got {:?}",
            numbers
        );
        assert!(
            !numbers.contains(&7230),
            "7230 is obsoleted and should not appear in current_standard_for"
        );
    }

    #[test]
    fn obsolescence_chain_for_2616_reaches_9110() {
        let chain = obsolescence_chain(2616);
        let numbers: Vec<u32> = chain.iter().map(|e| e.number).collect();
        assert!(
            numbers.contains(&2616),
            "chain must include the seed entry"
        );
        assert!(
            numbers.contains(&9110),
            "chain from 2616 must reach 9110 via 7230-7235; got {:?}",
            numbers
        );
    }

    #[test]
    fn obsolescence_chain_for_9293_includes_793() {
        let chain = obsolescence_chain(9293);
        let numbers: Vec<u32> = chain.iter().map(|e| e.number).collect();
        assert!(numbers.contains(&793), "9293 obsoletes 793; got {:?}", numbers);
        assert!(numbers.contains(&9293));
    }

    #[test]
    fn topic_queries_work() {
        let tls = by_topic("tls");
        assert!(tls.len() >= 2, "tls topic should hit 8446 + 5246 at minimum");

        let dns = by_topic("dns");
        assert!(dns.len() >= 4, "dns topic should hit 1034, 1035, 4033, 8484 at minimum");

        let json = by_topic("json");
        assert!(json.iter().any(|e| e.number == 8259), "json topic must include 8259");
    }

    #[test]
    fn status_filter_is_exclusive() {
        let std_rfcs = by_status(RfcStatus::InternetStandard);
        assert!(std_rfcs.iter().all(|e| e.status == RfcStatus::InternetStandard));
        assert!(std_rfcs.iter().any(|e| e.number == 791));
    }

    #[test]
    fn catalog_trait_query_round_trips() {
        let c = RfcCatalog;
        assert_eq!(c.name(), "rfc");
        assert!(c.entry_count() >= 60);

        // Empty filter returns all.
        let all_json = c.query(json!({})).expect("empty filter");
        assert_eq!(
            all_json["matched"].as_u64().unwrap() as usize,
            c.entry_count()
        );

        // number filter narrows to one.
        let by_num = c.query(json!({ "number": 9110 })).expect("number filter");
        assert_eq!(by_num["matched"].as_u64(), Some(1));

        // topic filter narrows.
        let http = c.query(json!({ "topic": "http" })).expect("topic filter");
        assert!(http["matched"].as_u64().unwrap() >= 5);

        // current_standard filter drops obsoleted.
        let curr = c
            .query(json!({ "topic": "http", "current_standard": true }))
            .expect("current filter");
        let entries = curr["entries"].as_array().unwrap();
        for e in entries {
            assert_ne!(e["status"].as_str(), Some("obsoleted"));
        }
    }

    #[test]
    fn obsolescence_chain_is_exposed_via_query() {
        let c = RfcCatalog;
        let result = c
            .query(json!({ "obsolescence_chain": 2616 }))
            .expect("chain query");
        let entries = result["entries"].as_array().unwrap();
        let numbers: Vec<u64> = entries
            .iter()
            .filter_map(|e| e["number"].as_u64())
            .collect();
        assert!(numbers.contains(&2616));
        assert!(numbers.contains(&9110));
    }
}
