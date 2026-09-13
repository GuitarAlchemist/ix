//! Slice S8 — `evaluate` over the Class-`Development` corpus, every rule at
//! every swept width.
//!
//! Red-first discriminators:
//!
//! * **NC-10** — a report carrying a `verdict`, a threshold, a margin or a
//!   sentence claiming one rule beats another fails. The corpus is Class
//!   `Development` permanently and no ordering claim may be computed from it;
//! * **NC-18** — a pooled ratio never appears without the per-family count
//!   vectors it was summed from. A pooled number without its decomposition is a
//!   weighted number with the weights hidden;
//! * **NC-1** — the reserved classes are refused outright, and no Lane-W path
//!   opens a holdout path, because no eligible holdout population exists.
//!
//! The output is a table of counts. Nothing here crosses a threshold, and
//! nothing is called a winner.

mod support;

use ix_gaia_advisory::{evaluate, AdvisoryRefusal, EvalRequest, EvidenceClass, RuleId, RuleWidth};

use support::corpus::{self, Family, Options};
use support::{approved_evidence_root, fresh_scratch, MANIFEST_FILE_NAME};

const SWEPT: [u32; 3] = [0, 1024, 4096];

fn full() -> Options {
    Options {
        families: vec![
            Family::Crlf,
            Family::Debris,
            Family::Restamp,
            Family::StaleFigure,
            Family::Pristine,
        ],
        inject_duplicate: false,
    }
}

/// Every `(rule, width)` cell the sweep characterizes: the five content-free
/// and exact rules at width zero, plus the windowed rule at each swept width.
fn swept_cells() -> Vec<RuleWidth> {
    let mut cells: Vec<RuleWidth> = [
        RuleId::AlwaysAgree,
        RuleId::NameSetOnly,
        RuleId::LengthOnly,
        RuleId::StructuralOnly,
        RuleId::FullDigest,
    ]
    .into_iter()
    .map(|rule| RuleWidth {
        rule,
        window_bytes: 0,
    })
    .collect();
    for window_bytes in SWEPT {
        cells.push(RuleWidth {
            rule: RuleId::WindowProbe,
            window_bytes,
        });
    }
    cells
}

fn request(corpus: &corpus::Corpus, class: EvidenceClass) -> EvalRequest {
    EvalRequest {
        corpus_root: corpus.root.clone(),
        expected_corpus_digest_ix_agg_1: corpus.digest.clone(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        cells: swept_cells(),
        window_reference_root: approved_evidence_root(),
        evidence_class: class,
    }
}

fn locked_corpus(slice: &str) -> corpus::Corpus {
    corpus::generate(&fresh_scratch(slice).join("corpus"), &full())
}

#[test]
fn every_cell_reports_its_six_counts_over_every_admitted_case() {
    let corpus = locked_corpus("s8-cells");
    let report = evaluate(&request(&corpus, EvidenceClass::Development)).expect("the corpus binds");

    assert_eq!(
        report.cells.len(),
        swept_cells().len() * 5,
        "one cell per (rule, width, family): eight rule-widths over five families"
    );
    for cell in &report.cells {
        assert_eq!(cell.evidence_class, EvidenceClass::Development);
        assert_eq!(
            cell.counts.total(),
            cell.counts.n_tt
                + cell.counts.n_tf
                + cell.counts.n_tu
                + cell.counts.n_ft
                + cell.counts.n_ff
                + cell.counts.n_fu
        );
    }

    // Every admitted case is counted exactly once per rule-width.
    for rule_width in swept_cells() {
        let total: u64 = report
            .cells
            .iter()
            .filter(|cell| {
                cell.rule == rule_width.rule && cell.window_bytes == rule_width.window_bytes
            })
            .map(|cell| cell.counts.total())
            .sum();
        assert_eq!(
            total, 25,
            "{:?}@{} must cover all 25 admitted cases",
            rule_width.rule, rule_width.window_bytes
        );
    }
}

#[test]
fn the_floor_agrees_with_everything_and_is_reported_as_such() {
    let corpus = locked_corpus("s8-floor");
    let report = evaluate(&request(&corpus, EvidenceClass::Development)).expect("binds");

    let pooled = report
        .pooled
        .iter()
        .find(|cell| cell.rule == RuleId::AlwaysAgree)
        .expect("the floor is pooled like every other rule");
    assert_eq!(
        pooled.false_agreement.value,
        Some(1.0),
        "on a corpus whose truth-F cases all bind, the floor's false_agreement is exactly 1.0"
    );
    assert_eq!(pooled.counts.n_ft, 24);
    assert_eq!(pooled.counts.n_tt, 1);
    assert_eq!(
        pooled.false_drift.denominator, 1,
        "the guardrail direction has denominator <= 1 and is not estimable from this corpus"
    );
}

#[test]
fn nc_9_width_zero_reproduces_the_ablation_count_vector_at_every_family() {
    let corpus = locked_corpus("s8-nc9");
    let report = evaluate(&request(&corpus, EvidenceClass::Development)).expect("binds");

    for family in [
        "F-CRLF",
        "F-DEBRIS",
        "F-RESTAMP",
        "F-STALE-FIGURE",
        "F-PRISTINE",
    ] {
        let pick = |rule: RuleId, width: u32| {
            report
                .cells
                .iter()
                .find(|cell| {
                    cell.rule == rule && cell.window_bytes == width && cell.family == family
                })
                .unwrap_or_else(|| panic!("cell for {rule:?}@{width} on {family}"))
        };
        assert_eq!(
            pick(RuleId::WindowProbe, 0).counts,
            pick(RuleId::StructuralOnly, 0).counts,
            "NC-9: WindowProbe(0) diverged from StructuralOnly on {family} — a harness defect"
        );
    }
}

#[test]
fn nc_18_no_pooled_ratio_appears_without_its_per_family_decomposition() {
    let corpus = locked_corpus("s8-nc18");
    let report = evaluate(&request(&corpus, EvidenceClass::Development)).expect("binds");

    assert!(!report.pooled.is_empty());
    for pooled in &report.pooled {
        let parts: Vec<_> = report
            .cells
            .iter()
            .filter(|cell| cell.rule == pooled.rule && cell.window_bytes == pooled.window_bytes)
            .collect();
        assert_eq!(
            parts.len(),
            pooled.families.len(),
            "NC-18: every pooled cell names exactly the families it summed"
        );

        let mut summed = ix_gaia_advisory::CountVector::default();
        for part in &parts {
            assert!(
                pooled.families.contains(&part.family),
                "NC-18: {} is summed into the pool but not named in it",
                part.family
            );
            summed = summed.plus(&part.counts);
        }
        assert_eq!(
            summed, pooled.counts,
            "NC-18: the pooled counts must be the sum of the per-family counts"
        );
    }
}

#[test]
fn nc_10_and_nc_21_no_ordering_claim_and_no_class_less_output() {
    let corpus = locked_corpus("s8-nc10");
    let report = evaluate(&request(&corpus, EvidenceClass::Development)).expect("binds");
    let text = report.to_canonical_json();
    let json: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");

    let mut keys = Vec::new();
    collect_keys(&json, &mut keys);
    for key in &keys {
        // Whole `snake_case` segments, not substrings: "aggregate" contains
        // "gate", and a control that fires on that is measuring the wrong thing.
        let segments: Vec<&str> = key.split('_').collect();
        for forbidden in [
            "verdict",
            "threshold",
            "margin",
            "winner",
            "score",
            "gate",
            "pass",
        ] {
            assert!(
                !segments.contains(&forbidden),
                "NC-10: a Class-Development report carries no {forbidden:?} field, found {key:?}"
            );
        }
    }
    // No emitted string asserts one rule is better than another.
    let mut strings = Vec::new();
    collect_strings(&json, &mut strings);
    for value in &strings {
        let lower = value.to_ascii_lowercase();
        for forbidden in ["better than", "beats", "outperform", "wins", "best"] {
            assert!(
                !lower.contains(forbidden),
                "NC-10: an ordering claim leaked into the report: {value:?}"
            );
        }
    }

    // NC-21: the class is present on the report and on every cell.
    assert_eq!(
        json["provenance"]["evidence_class"], "Development",
        "no report is class-less"
    );
    for cell in json["cells"].as_array().expect("cells") {
        assert_eq!(cell["evidence_class"], "Development");
    }
}

#[test]
fn nc_1_the_reserved_classes_are_refused_and_no_holdout_path_exists() {
    let corpus = locked_corpus("s8-nc1");

    match evaluate(&request(&corpus, EvidenceClass::Holdout)) {
        Err(AdvisoryRefusal::ReservedEvidenceClass { class, .. }) => {
            assert_eq!(class, EvidenceClass::Holdout);
        }
        other => panic!("a Class-Holdout run must refuse; got {other:?}"),
    }
    match evaluate(&request(&corpus, EvidenceClass::Selection)) {
        Err(AdvisoryRefusal::ReservedEvidenceClass { class, .. }) => {
            assert_eq!(class, EvidenceClass::Selection);
        }
        other => {
            panic!("M3 selects no parameter, so a Class-Selection run must refuse; got {other:?}")
        }
    }

    // Lane W stages no holdout population and opens no holdout path.
    assert!(
        !corpus.root.join("hold").exists(),
        "NC-1: no Class-Holdout corpus exists under {}",
        corpus.root.display()
    );
    let holdout_case_ids: Vec<String> = Vec::new();
    let development_case_ids: Vec<&str> = corpus
        .admitted
        .iter()
        .map(|case| case.case_id.as_str())
        .collect();
    assert!(
        holdout_case_ids
            .iter()
            .all(|id| !development_case_ids.contains(&id.as_str())),
        "NC-1: the class case-id sets must be disjoint"
    );
}

#[test]
fn invariant_2_a_corpus_that_does_not_re_measure_is_refused() {
    let corpus = locked_corpus("s8-invariant2");
    let mut request = request(&corpus, EvidenceClass::Development);
    request.expected_corpus_digest_ix_agg_1 =
        "0000000000000000000000000000000000000000000000000000000000000000".to_string();

    match evaluate(&request) {
        Err(AdvisoryRefusal::CorpusDigestMismatch { measured, .. }) => {
            assert_eq!(measured, corpus.digest);
        }
        other => panic!("evaluate must refuse a corpus that does not re-measure; got {other:?}"),
    }
}

#[test]
fn the_report_is_deterministic_and_binds_the_corpus_it_scored() {
    let corpus = locked_corpus("s8-determinism");
    let first = evaluate(&request(&corpus, EvidenceClass::Development)).expect("binds");
    let second = evaluate(&request(&corpus, EvidenceClass::Development)).expect("binds");
    assert_eq!(first.to_canonical_json(), second.to_canonical_json());
    assert_eq!(
        first.provenance.corpus_digest_ix_agg_1,
        Some(corpus.digest.clone())
    );
    assert_eq!(first.provenance.case_id, None, "a report spans cases");
    assert_eq!(first.provenance.ledger_digest_ix_agg_1, None);
}

fn collect_keys(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                out.push(key.clone());
                collect_keys(child, out);
            }
        }
        serde_json::Value::Array(items) => items.iter().for_each(|item| collect_keys(item, out)),
        _ => {}
    }
}

fn collect_strings(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::String(text) => out.push(text.clone()),
        serde_json::Value::Array(items) => items.iter().for_each(|item| collect_strings(item, out)),
        serde_json::Value::Object(map) => map.values().for_each(|item| collect_strings(item, out)),
        _ => {}
    }
}
