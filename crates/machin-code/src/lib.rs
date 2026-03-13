//! # machin-code
//!
//! ML on source code: tree-sitter parsing + feature extraction for any language.
//!
//! This crate bridges tree-sitter's language-agnostic syntax trees to ndarray,
//! enabling MachinDeOuf's ML algorithms to work on code as a first-class data type.
//!
//! ## Modules
//!
//! - **parse**: Parse source code into a `CodeTree` wrapper
//! - **extract**: Extract numerical features (histograms, adjacency matrices) from parsed code
//! - **error**: Error types for parsing and extraction
//!
//! ## Quick Start
//!
//! ```rust
//! use machin_code::{parse, extract};
//!
//! let tree = parse::parse("rust", "fn main() { let x = 42; }").unwrap();
//! let hist = extract::histogram(&tree);
//! // hist is an Array1<f64> ready for clustering, classification, etc.
//! ```

pub mod error;
pub mod parse;
pub mod extract;
