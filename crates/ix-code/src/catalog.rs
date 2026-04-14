//! # Catalog of mathematical tools for analysing programming-language repositories.
//!
//! This module is a curated, queryable inventory of third-party tools
//! that apply mathematical techniques to source code, dependency
//! graphs, git history, and build artifacts. It is not a registry of
//! `ix-code`'s own capabilities — it is ix's answer to the question
//! *"what else exists?"*, so agents can route users to the right
//! specialised tool rather than forcing every analysis through
//! `ix_code_analyze`.
//!
//! Every entry has:
//! - `name` — tool name as it would appear on GitHub or crates.io
//! - `category` — one of six buckets (see [`ToolCategory`])
//! - `technique` — the underlying mathematical technique
//! - `languages` — programming languages the tool supports
//! - `description` — one-sentence summary, no marketing copy
//! - `url` — canonical home page when one exists
//!
//! Populated from user-curated research — see the commit history on
//! this file for the specific sources. Updates are welcome as pull
//! requests.
//!
//! ## Query API
//!
//! - [`all`] — full inventory
//! - [`by_language`] — filter by a language string (case-insensitive)
//! - [`by_category`] — filter by category
//! - [`by_technique`] — filter by technique substring
//!
//! The data is a `&'static [CodeAnalysisTool]`, so queries are just
//! iterator filters with zero allocation for the tool slice itself.

use ix_catalog_core::{normalize_snake_case, string_contains_ci, Catalog};
use serde::Serialize;
use serde_json::{json, Value};

/// Broad category a tool belongs to. The six categories cover
/// disjoint concerns — a tool with multiple roles gets its primary
/// category here and additional techniques described in the
/// `technique` free-text field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCategory {
    /// Cyclomatic complexity, Halstead metrics, SLOC, linters,
    /// dependency graphs, call graphs, AST queries, abstract
    /// interpretation that runs on source.
    StaticAnalysis,

    /// Theorem provers, proof assistants, model checkers, SMT solvers
    /// — tools that prove properties about code against a formal
    /// specification.
    FormalVerification,

    /// Tools that find undefined behaviour, aliasing violations, race
    /// conditions, or concurrency bugs at runtime or via instrumented
    /// interpretation.
    SafetyMemory,

    /// Hotspot detection, trend analysis, productivity metrics,
    /// behavioral code analysis — everything that treats the
    /// repository as a time-varying data source.
    StatisticalAnalysis,

    /// Generators for API docs, architecture diagrams, user manuals,
    /// and other documentation produced mechanically from source code
    /// or repository metadata.
    Documentation,

    /// Math libraries used INSIDE analysis code (linear algebra,
    /// numerical methods, optimisation). Not analysis tools
    /// themselves, but the primitives they build on.
    NumericLibrary,

    /// Full machine-learning frameworks: deep-learning stacks,
    /// classical-ML libraries, and the dataframe / inference tooling
    /// that sits next to them. These are the libraries you reach for
    /// when the analysis itself is an ML task, not when you want a
    /// single primitive.
    MlFramework,
}

impl ToolCategory {
    /// Return the snake_case identifier matching the serde
    /// representation. Used by the MCP tool query input.
    pub fn as_str(&self) -> &'static str {
        match self {
            ToolCategory::StaticAnalysis => "static_analysis",
            ToolCategory::FormalVerification => "formal_verification",
            ToolCategory::SafetyMemory => "safety_memory",
            ToolCategory::StatisticalAnalysis => "statistical_analysis",
            ToolCategory::Documentation => "documentation",
            ToolCategory::NumericLibrary => "numeric_library",
            ToolCategory::MlFramework => "ml_framework",
        }
    }

    /// Parse a snake_case or lowercase hyphen form. Rejects anything
    /// not in the enum.
    pub fn parse(s: &str) -> Option<Self> {
        let normalised = normalize_snake_case(s);
        match normalised.as_str() {
            "static_analysis" | "static" => Some(Self::StaticAnalysis),
            "formal_verification" | "formal" | "verification" => {
                Some(Self::FormalVerification)
            }
            "safety_memory" | "safety" | "memory" => Some(Self::SafetyMemory),
            "statistical_analysis" | "statistical" | "stats" => {
                Some(Self::StatisticalAnalysis)
            }
            "documentation" | "docs" => Some(Self::Documentation),
            "numeric_library" | "numeric" | "library" => {
                Some(Self::NumericLibrary)
            }
            "ml_framework" | "ml" | "ml-framework" | "framework" => {
                Some(Self::MlFramework)
            }
            _ => None,
        }
    }
}

/// One entry in the catalog.
#[derive(Debug, Clone, Serialize)]
pub struct CodeAnalysisTool {
    pub name: &'static str,
    pub category: ToolCategory,
    pub technique: &'static str,
    pub languages: &'static [&'static str],
    pub description: &'static str,
    pub url: Option<&'static str>,
}

/// The full inventory. Sorted roughly by category then alphabetically
/// within each category. New entries should preserve that ordering
/// to keep diffs clean.
pub const CATALOG: &[CodeAnalysisTool] = &[
    // ──────────────────────────────────────────────────────────────
    // Static analysis and complexity metrics
    // ──────────────────────────────────────────────────────────────
    CodeAnalysisTool {
        name: "Radon",
        category: ToolCategory::StaticAnalysis,
        technique: "cyclomatic complexity, Halstead metrics, maintainability index",
        languages: &["python"],
        description: "McCabe complexity and raw size metrics for Python source.",
        url: Some("https://radon.readthedocs.io/"),
    },
    CodeAnalysisTool {
        name: "gocyclo",
        category: ToolCategory::StaticAnalysis,
        technique: "cyclomatic complexity",
        languages: &["go"],
        description: "Per-function cyclomatic complexity measurement for Go.",
        url: Some("https://github.com/fzipp/gocyclo"),
    },
    CodeAnalysisTool {
        name: "CodeQL",
        category: ToolCategory::StaticAnalysis,
        technique: "AST / CFG / DFG queries, graph theory",
        languages: &[
            "c", "cpp", "csharp", "go", "java", "javascript", "typescript", "python", "ruby",
            "swift", "kotlin",
        ],
        description: "Query language over AST, CFG, and data-flow graphs for bug finding.",
        url: Some("https://codeql.github.com/"),
    },
    CodeAnalysisTool {
        name: "Include-gardener",
        category: ToolCategory::StaticAnalysis,
        technique: "dependency graph extraction",
        languages: &["c", "cpp"],
        description: "Generates graphs of #include relations for C/C++ projects.",
        url: Some("https://github.com/feddischson/include_gardener"),
    },
    CodeAnalysisTool {
        name: "Astrée",
        category: ToolCategory::StaticAnalysis,
        technique: "abstract interpretation",
        languages: &["c", "cpp"],
        description: "Proves the absence of runtime errors and data races via abstract interpretation.",
        url: Some("https://www.absint.com/astree/index.htm"),
    },
    CodeAnalysisTool {
        name: "Polyspace",
        category: ToolCategory::StaticAnalysis,
        technique: "abstract interpretation",
        languages: &["c", "cpp", "ada"],
        description: "Proves the absence of overflows, out-of-bounds accesses, and other runtime errors.",
        url: Some("https://www.mathworks.com/products/polyspace.html"),
    },
    CodeAnalysisTool {
        name: "MIRAI",
        category: ToolCategory::StaticAnalysis,
        technique: "abstract interpretation, function summaries",
        languages: &["rust"],
        description: "Abstract interpreter for Rust that tracks data flow and detects potential panics.",
        url: Some("https://github.com/facebookexperimental/MIRAI"),
    },
    CodeAnalysisTool {
        name: "ix_code_analyze",
        category: ToolCategory::StaticAnalysis,
        technique: "cyclomatic, cognitive, Halstead, maintainability index, SLOC",
        languages: &[
            "rust", "python", "javascript", "typescript", "cpp", "c", "java", "go", "csharp",
            "fsharp",
        ],
        description: "ix's own lightweight code analyzer — 20 metrics per file, ML-ready feature vectors.",
        url: Some("https://github.com/GuitarAlchemist/ix"),
    },
    // ──────────────────────────────────────────────────────────────
    // Formal verification and logic
    // ──────────────────────────────────────────────────────────────
    CodeAnalysisTool {
        name: "Lean",
        category: ToolCategory::FormalVerification,
        technique: "dependent type theory, proof assistant",
        languages: &["lean"],
        description: "Programming language and proof assistant used to verify cryptographic routines and policy engines.",
        url: Some("https://leanprover.github.io/"),
    },
    CodeAnalysisTool {
        name: "Coq",
        category: ToolCategory::FormalVerification,
        technique: "calculus of inductive constructions, proof assistant",
        languages: &["coq", "ocaml"],
        description: "Proof assistant for formalising mathematical theories and developing verified software.",
        url: Some("https://coq.inria.fr/"),
    },
    CodeAnalysisTool {
        name: "Z3",
        category: ToolCategory::FormalVerification,
        technique: "SMT solving",
        languages: &["c", "cpp", "python", "dotnet", "java"],
        description: "High-performance theorem prover for automated verification and SMT problems.",
        url: Some("https://github.com/Z3Prover/z3"),
    },
    CodeAnalysisTool {
        name: "CBMC",
        category: ToolCategory::FormalVerification,
        technique: "bounded model checking",
        languages: &["c", "cpp"],
        description: "Bounded model checker for C programs verifying user-defined and standard assertions.",
        url: Some("https://www.cprover.org/cbmc/"),
    },
    CodeAnalysisTool {
        name: "CPAchecker",
        category: ToolCategory::FormalVerification,
        technique: "configurable program analysis, model checking",
        languages: &["c"],
        description: "Open-source tool for checking execution paths in C programs.",
        url: Some("https://cpachecker.sosy-lab.org/"),
    },
    CodeAnalysisTool {
        name: "Kani",
        category: ToolCategory::FormalVerification,
        technique: "bounded model checking over Rust MIR",
        languages: &["rust"],
        description: "Symbolic model checker for Rust that evaluates all possible inputs via MIR.",
        url: Some("https://model-checking.github.io/kani/"),
    },
    CodeAnalysisTool {
        name: "Verus",
        category: ToolCategory::FormalVerification,
        technique: "SMT-based deductive verification",
        languages: &["rust"],
        description: "Write proofs directly in Rust syntax; uses Z3 for functional correctness on low-level or concurrent code.",
        url: Some("https://verus-lang.github.io/verus/"),
    },
    CodeAnalysisTool {
        name: "Aeneas",
        category: ToolCategory::FormalVerification,
        technique: "purely functional translation for proof assistants",
        languages: &["rust", "lean", "coq", "fstar"],
        description: "Translates Rust into a purely functional form, eliminating the need to reason about memory addresses.",
        url: Some("https://github.com/AeneasVerif/aeneas"),
    },
    CodeAnalysisTool {
        name: "Creusot",
        category: ToolCategory::FormalVerification,
        technique: "deductive verification, Pearlite specification",
        languages: &["rust"],
        description: "Deductive verifier for safe Rust using the Pearlite specification language.",
        url: Some("https://github.com/creusot-rs/creusot"),
    },
    CodeAnalysisTool {
        name: "Klee",
        category: ToolCategory::FormalVerification,
        technique: "symbolic execution",
        languages: &["c", "cpp", "rust"],
        description: "Symbolic execution engine that converts program variables into mathematical expressions to find bugs.",
        url: Some("https://klee.github.io/"),
    },
    CodeAnalysisTool {
        name: "Haybale",
        category: ToolCategory::FormalVerification,
        technique: "symbolic execution over LLVM IR",
        languages: &["rust", "c", "cpp"],
        description: "Symbolic execution engine operating on LLVM bitcode; used for Rust via rustc LLVM output.",
        url: Some("https://github.com/PLSysSec/haybale"),
    },
    // ──────────────────────────────────────────────────────────────
    // Safety and memory analysis
    // ──────────────────────────────────────────────────────────────
    CodeAnalysisTool {
        name: "Miri",
        category: ToolCategory::SafetyMemory,
        technique: "instrumented interpretation of Rust MIR",
        languages: &["rust"],
        description: "Checks unsafe Rust blocks for undefined behaviour against the aliasing and memory rules.",
        url: Some("https://github.com/rust-lang/miri"),
    },
    CodeAnalysisTool {
        name: "Loom",
        category: ToolCategory::SafetyMemory,
        technique: "exhaustive thread-interleaving model checking",
        languages: &["rust"],
        description: "Exhaustively explores thread interleavings to find concurrency races missed by normal tests.",
        url: Some("https://github.com/tokio-rs/loom"),
    },
    CodeAnalysisTool {
        name: "AddressSanitizer",
        category: ToolCategory::SafetyMemory,
        technique: "shadow memory instrumentation",
        languages: &["c", "cpp", "rust", "go"],
        description: "Detects memory errors (out-of-bounds, use-after-free, double-free) at runtime via shadow memory.",
        url: Some("https://clang.llvm.org/docs/AddressSanitizer.html"),
    },
    CodeAnalysisTool {
        name: "ThreadSanitizer",
        category: ToolCategory::SafetyMemory,
        technique: "happens-before race detection",
        languages: &["c", "cpp", "rust", "go"],
        description: "Detects data races at runtime by tracking happens-before relationships.",
        url: Some("https://clang.llvm.org/docs/ThreadSanitizer.html"),
    },
    // ──────────────────────────────────────────────────────────────
    // Statistical and behavioral analysis
    // ──────────────────────────────────────────────────────────────
    CodeAnalysisTool {
        name: "CodeScene",
        category: ToolCategory::StatisticalAnalysis,
        technique: "behavioural code analysis, hotspot detection",
        languages: &[
            "rust", "python", "javascript", "typescript", "c", "cpp", "java", "csharp", "go",
            "ruby", "swift",
        ],
        description: "Behavioural analysis that prioritises technical debt and visualises code-health trends.",
        url: Some("https://codescene.com/"),
    },
    CodeAnalysisTool {
        name: "git-trend",
        category: ToolCategory::StatisticalAnalysis,
        technique: "git history time-series analysis",
        languages: &["language-agnostic"],
        description: "Mines git history for churn, authorship, and co-change patterns to surface at-risk files.",
        url: Some("https://github.com/rs/git-trend"),
    },
    CodeAnalysisTool {
        name: "ix_git_log",
        category: ToolCategory::StatisticalAnalysis,
        technique: "commit cadence time series",
        languages: &["language-agnostic"],
        description: "ix's own git log adapter — bucket commits per day or per week for downstream FFT / Lyapunov / clustering.",
        url: Some("https://github.com/GuitarAlchemist/ix"),
    },
    CodeAnalysisTool {
        name: "ix_cargo_deps",
        category: ToolCategory::StatisticalAnalysis,
        technique: "workspace dependency graph extraction",
        languages: &["rust"],
        description: "ix's own Cargo.toml parser — emits node+edge+feature matrix ready for ix_graph / ix_kmeans.",
        url: Some("https://github.com/GuitarAlchemist/ix"),
    },
    // ──────────────────────────────────────────────────────────────
    // Documentation generation
    // ──────────────────────────────────────────────────────────────
    CodeAnalysisTool {
        name: "rustdoc",
        category: ToolCategory::Documentation,
        technique: "doc-comment extraction, compiled examples",
        languages: &["rust"],
        description: "Rust's built-in doc generator: extracts /// comments into searchable HTML and runs every code example as a test.",
        url: Some("https://doc.rust-lang.org/rustdoc/"),
    },
    CodeAnalysisTool {
        name: "mdBook",
        category: ToolCategory::Documentation,
        technique: "markdown book compilation",
        languages: &["language-agnostic"],
        description: "Markdown-based book generator, ideal for user manuals and tutorials (e.g. \"The Rust Programming Language\" book).",
        url: Some("https://rust-lang.github.io/mdBook/"),
    },
    CodeAnalysisTool {
        name: "Litho (deepwiki-rs)",
        category: ToolCategory::Documentation,
        technique: "C4 architecture extraction, dependency analysis",
        languages: &["rust"],
        description: "Analyses crate structure and dependencies to auto-generate C4 context / container / component diagrams.",
        url: Some("https://github.com/deepwiki-rs/litho"),
    },
    CodeAnalysisTool {
        name: "Auto-UML",
        category: ToolCategory::Documentation,
        technique: "tree-sitter AST parsing, Mermaid generation",
        languages: &["rust"],
        description: "CLI that parses Rust source via tree-sitter and emits Mermaid class diagrams.",
        url: Some("https://github.com/rsblabs/auto-uml"),
    },
    CodeAnalysisTool {
        name: "Oxdraw",
        category: ToolCategory::Documentation,
        technique: "diagram-as-code, code-linked Mermaid",
        languages: &["rust"],
        description: "Declarative diagrams (Mermaid syntax) linked to specific code segments for onboarding and architectural overviews.",
        url: Some("https://github.com/oxdraw/oxdraw"),
    },
    CodeAnalysisTool {
        name: "Utoipa",
        category: ToolCategory::Documentation,
        technique: "macro-driven OpenAPI generation",
        languages: &["rust"],
        description: "Auto-generates OpenAPI (Swagger) specifications from Rust types and handlers via attribute macros.",
        url: Some("https://github.com/juhaku/utoipa"),
    },
    CodeAnalysisTool {
        name: "simple-mermaid",
        category: ToolCategory::Documentation,
        technique: "rustdoc Mermaid embedding",
        languages: &["rust"],
        description: "Embeds Mermaid diagrams directly into #[doc] attributes so they render in rustdoc HTML output.",
        url: Some("https://crates.io/crates/simple-mermaid"),
    },
    CodeAnalysisTool {
        name: "Doxygen",
        category: ToolCategory::Documentation,
        technique: "doc-comment extraction, call-graph rendering",
        languages: &["c", "cpp", "python", "java", "csharp", "php", "fortran"],
        description: "The canonical cross-language doc generator for C/C++ and many others; produces call graphs via Graphviz.",
        url: Some("https://www.doxygen.nl/"),
    },
    CodeAnalysisTool {
        name: "Sphinx",
        category: ToolCategory::Documentation,
        technique: "reStructuredText, autodoc",
        languages: &["python", "c", "cpp"],
        description: "Python's canonical doc generator; used for most major Python library docs.",
        url: Some("https://www.sphinx-doc.org/"),
    },
    CodeAnalysisTool {
        name: "godoc",
        category: ToolCategory::Documentation,
        technique: "doc-comment extraction",
        languages: &["go"],
        description: "Go's built-in doc server; extracts package-level comments into HTML.",
        url: Some("https://pkg.go.dev/golang.org/x/tools/cmd/godoc"),
    },
    CodeAnalysisTool {
        name: "TypeDoc",
        category: ToolCategory::Documentation,
        technique: "TypeScript AST walking",
        languages: &["typescript"],
        description: "Converts TypeScript source comments into HTML / JSON documentation.",
        url: Some("https://typedoc.org/"),
    },
    CodeAnalysisTool {
        name: "Kodesage",
        category: ToolCategory::Documentation,
        technique: "AI-powered multi-source synthesis",
        languages: &["language-agnostic"],
        description: "AI platform that consolidates codebase, issue tracker, and test data into system-level docs and compliance reports.",
        url: Some("https://kodesage.ai/"),
    },
    // ──────────────────────────────────────────────────────────────
    // Numeric libraries used inside analysis code
    // ──────────────────────────────────────────────────────────────
    CodeAnalysisTool {
        name: "nalgebra",
        category: ToolCategory::NumericLibrary,
        technique: "linear algebra, geometry, statistics",
        languages: &["rust"],
        description: "Rust linear algebra library with static and dynamic matrices, decompositions, and geometric primitives.",
        url: Some("https://nalgebra.org/"),
    },
    CodeAnalysisTool {
        name: "ndarray",
        category: ToolCategory::NumericLibrary,
        technique: "n-dimensional arrays, BLAS integration",
        languages: &["rust"],
        description: "Rust's numpy-equivalent: multi-dimensional array container with slicing, broadcasting, and BLAS.",
        url: Some("https://docs.rs/ndarray/"),
    },
    CodeAnalysisTool {
        name: "Peroxide",
        category: ToolCategory::NumericLibrary,
        technique: "numerical methods, autodiff, statistics",
        languages: &["rust"],
        description: "Comprehensive Rust numeric library similar to NumPy/MATLAB, with automatic differentiation.",
        url: Some("https://github.com/Axect/Peroxide"),
    },
    CodeAnalysisTool {
        name: "MathOpt",
        category: ToolCategory::NumericLibrary,
        technique: "LP / MIP modeling and solving",
        languages: &["cpp", "python"],
        description: "Library for modeling and solving linear and mixed-integer programs.",
        url: Some("https://developers.google.com/optimization"),
    },
    CodeAnalysisTool {
        name: "SciPy",
        category: ToolCategory::NumericLibrary,
        technique: "scientific computing, statistics, optimisation",
        languages: &["python"],
        description: "Python's canonical scientific computing library, used in most quantitative repo analysis pipelines.",
        url: Some("https://scipy.org/"),
    },
    CodeAnalysisTool {
        name: "MATLAB",
        category: ToolCategory::NumericLibrary,
        technique: "numerical computation, algorithm visualisation",
        languages: &["matlab"],
        description: "Commercial numerical computing environment with extensive toolboxes for control, signal, and optimisation.",
        url: Some("https://www.mathworks.com/products/matlab.html"),
    },
    CodeAnalysisTool {
        name: "GNU Octave",
        category: ToolCategory::NumericLibrary,
        technique: "numerical computation",
        languages: &["octave", "matlab"],
        description: "Free MATLAB-compatible numerical computing environment.",
        url: Some("https://octave.org/"),
    },
    // ──────────────────────────────────────────────────────────────
    // Machine-learning frameworks (Rust ecosystem)
    // ──────────────────────────────────────────────────────────────
    CodeAnalysisTool {
        name: "Burn",
        category: ToolCategory::MlFramework,
        technique: "deep-learning framework, dynamic graph, custom kernels",
        languages: &["rust"],
        description: "Comprehensive Rust deep-learning framework: data loading, model definition, training, hyperparameter optimisation, custom kernel code for fine-grained control.",
        url: Some("https://burn.dev/"),
    },
    CodeAnalysisTool {
        name: "Candle",
        category: ToolCategory::MlFramework,
        technique: "minimalist deep-learning, LLM inference",
        languages: &["rust"],
        description: "Minimalist ML framework from Hugging Face; optimised for LLM inference performance on NVIDIA GPUs via cuTENSOR and cuDNN.",
        url: Some("https://github.com/huggingface/candle"),
    },
    CodeAnalysisTool {
        name: "tch-rs",
        category: ToolCategory::MlFramework,
        technique: "PyTorch C++ bindings",
        languages: &["rust"],
        description: "High-performance Rust bindings for the PyTorch C++ API; the bridge for teams moving model serving from Python to Rust.",
        url: Some("https://github.com/LaurentMazare/tch-rs"),
    },
    CodeAnalysisTool {
        name: "dfdx",
        category: ToolCategory::MlFramework,
        technique: "differentiable programming, functional style",
        languages: &["rust"],
        description: "Declarative, functional-style differentiable programming library with automatic differentiation and CUDA support.",
        url: Some("https://github.com/coreylowman/dfdx"),
    },
    CodeAnalysisTool {
        name: "Linfa",
        category: ToolCategory::MlFramework,
        technique: "classical ML, standardised API",
        languages: &["rust"],
        description: "The 'Scikit-learn of Rust': standardised classical-ML framework with SVM, k-means, logistic regression, Gaussian mixture models.",
        url: Some("https://rust-ml.github.io/linfa/"),
    },
    CodeAnalysisTool {
        name: "SmartCore",
        category: ToolCategory::MlFramework,
        technique: "classical ML, interpretability",
        languages: &["rust"],
        description: "Low-level classical-ML library covering classification, regression, clustering; emphasises interpretability (feature importance, decision paths).",
        url: Some("https://smartcorelib.org/"),
    },
    CodeAnalysisTool {
        name: "Polars",
        category: ToolCategory::MlFramework,
        technique: "columnar dataframe, Arrow-backed",
        languages: &["rust", "python"],
        description: "Blazingly fast DataFrame library similar to Pandas; used for heavy data preprocessing in Rust + Python ML pipelines.",
        url: Some("https://www.pola.rs/"),
    },
    CodeAnalysisTool {
        name: "XGBoost-RS",
        category: ToolCategory::MlFramework,
        technique: "gradient boosting, tabular prediction",
        languages: &["rust"],
        description: "Rust bindings for the XGBoost gradient-boosting library; the go-to for structured-data classification and regression.",
        url: Some("https://github.com/davechallis/rust-xgboost"),
    },
    CodeAnalysisTool {
        name: "Tract",
        category: ToolCategory::MlFramework,
        technique: "edge inference, multi-format model loader",
        languages: &["rust"],
        description: "Edge and embedded inference engine supporting ONNX, TensorFlow, and PyTorch models; the standard Rust choice for on-device ML.",
        url: Some("https://github.com/sonos/tract"),
    },
];

// ──────────────────────────────────────────────────────────────────
// Query API
// ──────────────────────────────────────────────────────────────────

/// Return the full catalog as a static slice. Callers that need an
/// owned `Vec` should `.iter().cloned().collect()` — the entries are
/// small and `Clone` is cheap.
pub fn all() -> &'static [CodeAnalysisTool] {
    CATALOG
}

/// Filter the catalog by language (case-insensitive). Matches against
/// every language in the `languages` slice of each entry. The special
/// pseudo-language `"language-agnostic"` matches every query.
pub fn by_language(lang: &str) -> Vec<CodeAnalysisTool> {
    let q = lang.to_ascii_lowercase();
    CATALOG
        .iter()
        .filter(|t| {
            t.languages
                .iter()
                .any(|l| l.eq_ignore_ascii_case(&q) || *l == "language-agnostic")
        })
        .cloned()
        .collect()
}

/// Filter the catalog by category. Input may be either a
/// [`ToolCategory`] or the snake_case string form — the MCP handler
/// uses the string form.
pub fn by_category(category: ToolCategory) -> Vec<CodeAnalysisTool> {
    CATALOG
        .iter()
        .filter(|t| t.category == category)
        .cloned()
        .collect()
}

/// Filter the catalog by free-text substring match on the `technique`
/// field. Case-insensitive. Useful for queries like "cyclomatic",
/// "abstract interpretation", "symbolic execution".
pub fn by_technique(technique: &str) -> Vec<CodeAnalysisTool> {
    let q = technique.to_ascii_lowercase();
    CATALOG
        .iter()
        .filter(|t| t.technique.to_ascii_lowercase().contains(&q))
        .cloned()
        .collect()
}

/// Summary counts — useful for MCP responses that want a top-level
/// view before the full listing.
pub fn counts() -> CatalogCounts {
    let mut counts = CatalogCounts {
        total: CATALOG.len(),
        ..Default::default()
    };
    for tool in CATALOG {
        match tool.category {
            ToolCategory::StaticAnalysis => counts.static_analysis += 1,
            ToolCategory::FormalVerification => counts.formal_verification += 1,
            ToolCategory::SafetyMemory => counts.safety_memory += 1,
            ToolCategory::StatisticalAnalysis => counts.statistical_analysis += 1,
            ToolCategory::Documentation => counts.documentation += 1,
            ToolCategory::NumericLibrary => counts.numeric_library += 1,
            ToolCategory::MlFramework => counts.ml_framework += 1,
        }
    }
    counts
}

/// Per-category count summary.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct CatalogCounts {
    pub total: usize,
    pub static_analysis: usize,
    pub formal_verification: usize,
    pub safety_memory: usize,
    pub statistical_analysis: usize,
    pub documentation: usize,
    pub numeric_library: usize,
    pub ml_framework: usize,
}

// ──────────────────────────────────────────────────────────────────
// Catalog trait implementation — the uniform façade every ix
// catalog shares via ix-catalog-core. Keeps the free functions
// above (all/by_language/by_category/by_technique/counts) as the
// canonical API for Rust callers while giving the MCP dispatcher a
// single `query` entry point.
// ──────────────────────────────────────────────────────────────────

/// Zero-sized handle used to dispatch the code-analysis catalog
/// through the [`Catalog`] trait. The actual data lives in
/// [`CATALOG`]; this struct exists only so the MCP layer can hold
/// `&dyn Catalog` references uniformly.
pub struct CodeAnalysisCatalog;

impl Catalog for CodeAnalysisCatalog {
    fn name(&self) -> &'static str {
        "code_analysis"
    }

    fn scope(&self) -> &'static str {
        "External mathematical tools for analysing programming-language \
         repositories: static analysers, formal verifiers, safety/memory \
         checkers, statistical + behavioural analysis, documentation \
         generators, numeric libraries, and ML frameworks. Does NOT \
         include ix's own code-analysis primitives for their own sake \
         (those live in other crates); the catalog exists so agents can \
         route to the right specialist rather than over-stretching \
         ix_code_analyze."
    }

    fn entry_count(&self) -> usize {
        CATALOG.len()
    }

    fn counts(&self) -> Value {
        let c = counts();
        json!({
            "total": c.total,
            "static_analysis": c.static_analysis,
            "formal_verification": c.formal_verification,
            "safety_memory": c.safety_memory,
            "statistical_analysis": c.statistical_analysis,
            "documentation": c.documentation,
            "numeric_library": c.numeric_library,
            "ml_framework": c.ml_framework,
        })
    }

    fn query(&self, filter: Value) -> Result<Value, String> {
        // Start from the full catalog and apply each optional filter
        // in turn. Each filter is case-insensitive where plausible.
        let mut matched: Vec<CodeAnalysisTool> = all().to_vec();

        if let Some(lang) = filter.get("language").and_then(|v| v.as_str()) {
            let filtered = by_language(lang);
            matched.retain(|t| filtered.iter().any(|f| f.name == t.name));
        }

        if let Some(cat_str) = filter.get("category").and_then(|v| v.as_str()) {
            let cat = ToolCategory::parse(cat_str).ok_or_else(|| {
                format!(
                    "ix_code_catalog: unknown category '{cat_str}' — expected one of: \
                     static_analysis, formal_verification, safety_memory, \
                     statistical_analysis, documentation, numeric_library, ml_framework"
                )
            })?;
            let filtered = by_category(cat);
            matched.retain(|t| filtered.iter().any(|f| f.name == t.name));
        }

        if let Some(tech) = filter.get("technique").and_then(|v| v.as_str()) {
            matched.retain(|t| string_contains_ci(t.technique, tech));
        }

        Ok(json!({
            "counts": self.counts(),
            "matched": matched.len(),
            "entries": matched,
            // Back-compat alias — the pre-refactor handler returned a
            // `tools` array. Keep emitting it so existing smoke tests
            // and any client that already references it do not break.
            "tools": matched,
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
    fn catalog_is_not_empty_and_covers_every_category() {
        let c = counts();
        assert!(c.total >= 40, "expected at least 40 entries, got {}", c.total);
        assert!(c.static_analysis > 0);
        assert!(c.formal_verification > 0);
        assert!(c.safety_memory > 0);
        assert!(c.statistical_analysis > 0);
        assert!(c.documentation > 0);
        assert!(c.numeric_library > 0);
        assert!(c.ml_framework > 0);
    }

    #[test]
    fn every_entry_is_well_formed() {
        for tool in CATALOG {
            assert!(!tool.name.is_empty(), "name must be non-empty");
            assert!(!tool.technique.is_empty(), "technique must be non-empty");
            assert!(!tool.description.is_empty(), "description must be non-empty");
            assert!(
                !tool.languages.is_empty(),
                "{}: languages list must be non-empty",
                tool.name
            );
            for lang in tool.languages {
                assert!(
                    lang.chars().all(|c| c.is_ascii_lowercase() || c == '-'),
                    "{}: language '{}' must be lowercase-with-hyphens",
                    tool.name,
                    lang
                );
            }
        }
    }

    #[test]
    fn rust_query_includes_the_rust_specific_suite() {
        let rust_tools: Vec<&str> = by_language("rust")
            .iter()
            .map(|t| t.name)
            .collect();
        for required in ["Kani", "Verus", "Creusot", "Miri", "Loom", "MIRAI", "rustdoc", "mdBook"] {
            assert!(
                rust_tools.contains(&required),
                "Rust query missing expected tool: {required}"
            );
        }
    }

    #[test]
    fn language_agnostic_tools_show_up_in_every_language_query() {
        // mdBook, CodeScene, git-trend, ix_git_log, and Kodesage are
        // language-agnostic; each must appear in a language query.
        let for_python = by_language("python");
        assert!(for_python.iter().any(|t| t.name == "mdBook"));
        let for_rust = by_language("rust");
        assert!(for_rust.iter().any(|t| t.name == "mdBook"));
        let for_c = by_language("c");
        assert!(for_c.iter().any(|t| t.name == "mdBook"));
    }

    #[test]
    fn category_filter_is_exclusive() {
        let formal = by_category(ToolCategory::FormalVerification);
        assert!(formal.iter().all(|t| t.category == ToolCategory::FormalVerification));
        let static_ = by_category(ToolCategory::StaticAnalysis);
        assert!(static_.iter().all(|t| t.category == ToolCategory::StaticAnalysis));
    }

    #[test]
    fn technique_substring_match_finds_cyclomatic_and_abstract_interpretation() {
        let cyclomatic = by_technique("cyclomatic");
        assert!(!cyclomatic.is_empty());
        assert!(cyclomatic.iter().any(|t| t.name == "Radon" || t.name == "gocyclo"));

        let abstract_interp = by_technique("abstract interpretation");
        assert!(!abstract_interp.is_empty());
        assert!(abstract_interp.iter().any(|t| t.name == "Astrée" || t.name == "MIRAI"));
    }

    #[test]
    fn category_parse_accepts_canonical_and_short_forms() {
        assert_eq!(
            ToolCategory::parse("static_analysis"),
            Some(ToolCategory::StaticAnalysis)
        );
        assert_eq!(
            ToolCategory::parse("static"),
            Some(ToolCategory::StaticAnalysis)
        );
        assert_eq!(
            ToolCategory::parse("formal-verification"),
            Some(ToolCategory::FormalVerification)
        );
        assert_eq!(ToolCategory::parse("nope"), None);
    }

    #[test]
    fn counts_match_catalog_length() {
        let c = counts();
        let sum = c.static_analysis
            + c.formal_verification
            + c.safety_memory
            + c.statistical_analysis
            + c.documentation
            + c.numeric_library
            + c.ml_framework;
        assert_eq!(sum, c.total);
        assert_eq!(c.total, CATALOG.len());
    }

    #[test]
    fn ml_framework_query_includes_burn_candle_linfa() {
        let ml = by_category(ToolCategory::MlFramework);
        let names: Vec<&str> = ml.iter().map(|t| t.name).collect();
        for required in ["Burn", "Candle", "tch-rs", "dfdx", "Linfa", "SmartCore", "Polars", "Tract"] {
            assert!(
                names.contains(&required),
                "ml_framework missing expected entry {required}"
            );
        }
    }
}
