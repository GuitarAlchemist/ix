//! Slice S2 — units, explicit null semantics, and the typed out-of-sample field.
//!
//! Red-first discriminators, all through the public seam:
//!
//! * a `Ratio` whose `denominator == 0` serializes `"value":null` — never
//!   `0.0`, never `1.0`, never an omitted key;
//! * `measured_bytes` of a declared file the root does not hold serializes
//!   `null`, **not** `0`. A zero there would read as "an empty file was
//!   measured", which is a different fact;
//! * `out_of_sample` serializes `reason:"NoEligibleUnseenPopulation"` with a
//!   three-element `also_applicable` array, in declaration order.
//!
//! Controls closed here: NC-7, NC-11, NC-11a, NC-17, NC-20, NC-21, NC-25, and
//! the pinned key set carried with `schema_version`.

mod support;

use ix_gaia_advisory::{
    advise, AdvisoryRequest, CountVector, OosUnknownReason, OutOfSampleStatus, Ratio, RuleId,
    SCHEMA_VERSION,
};

use support::{
    approved_evidence_root, crate_src_root, fresh_scratch, stage_pristine, MANIFEST_FILE_NAME,
};

fn request(root: std::path::PathBuf) -> AdvisoryRequest {
    AdvisoryRequest {
        evidence_root: root,
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        rule: RuleId::FullDigest,
        window_bytes: 0,
        window_reference_root: None,
        expected_provenance: None,
    }
}

// ---------------------------------------------------------------- null semantics

#[test]
fn measured_bytes_of_a_declared_file_the_root_lacks_is_null_not_zero() {
    let dir = fresh_scratch("s2-missing-row");
    let staged = dir.join("root");
    stage_pristine(&staged);
    let absent = "gaia-uncertainty-grammar-v0.1.ebnf";
    std::fs::remove_file(staged.join(absent)).expect("the fixture is removable");

    let artifact =
        advise(&request(staged)).expect("a missing declared row is reported, not refused");
    let json: serde_json::Value =
        serde_json::from_str(&artifact.to_canonical_json()).expect("the artifact is valid JSON");

    let row = json["rows"]
        .as_array()
        .expect("rows is an array")
        .iter()
        .find(|row| row["name"] == absent)
        .expect("the declared row is still reported");

    assert!(
        row["measured_bytes"].is_null(),
        "measured_bytes must be null when the file is not measurable, got {}",
        row["measured_bytes"]
    );
    assert_ne!(
        row["measured_bytes"],
        serde_json::json!(0),
        "a null must never be defaulted to 0 — an empty file is a different fact"
    );
    assert!(
        row.as_object()
            .expect("row is an object")
            .contains_key("measured_bytes"),
        "the key must be present and null, never omitted"
    );
    assert_eq!(
        json["root"]["missing_names"],
        serde_json::json!([absent]),
        "the root findings must name what the root is missing"
    );
    // The manifest never lists itself, so it declares thirteen of the fourteen
    // bundle files; one of those thirteen is now absent.
    assert_eq!(json["root"]["declared_files"], serde_json::json!(13));
    assert_eq!(json["root"]["present_files"], serde_json::json!(12));
}

#[test]
fn a_zero_denominator_ratio_serializes_value_null() {
    let empty = Ratio::new(0, 0);
    assert_eq!(empty.value, None);
    assert_eq!(
        serde_json::to_string(&empty).expect("a ratio serializes"),
        r#"{"numerator":0,"denominator":0,"value":null}"#
    );

    let bound = Ratio::new(0, 7);
    assert_eq!(
        bound.value,
        Some(0.0),
        "0.0 on a non-zero denominator is legal and must be emitted as such"
    );
    assert_eq!(
        serde_json::to_string(&bound).expect("a ratio serializes"),
        r#"{"numerator":0,"denominator":7,"value":0.0}"#
    );
}

// ---------------------------------------------------------------- NC-17

#[test]
fn nc_17_rate_algebra_identities_hold_over_every_count_vector() {
    // A deterministic sweep over the count space, not a random one: the
    // identities are arithmetic and must hold everywhere, and a seeded random
    // sample would only test a subset while looking like it tested more.
    for n_tt in 0..4u64 {
        for n_tf in 0..4u64 {
            for n_tu in 0..4u64 {
                for n_ft in 0..4u64 {
                    for n_ff in 0..4u64 {
                        for n_fu in 0..4u64 {
                            let counts = CountVector {
                                n_tt,
                                n_tf,
                                n_tu,
                                n_ft,
                                n_ff,
                                n_fu,
                            };
                            let total = counts.total();
                            assert_eq!(total, n_tt + n_tf + n_tu + n_ft + n_ff + n_fu);

                            let coverage = counts.coverage();
                            let unknown_rate = counts.unknown_rate();
                            assert_eq!(coverage.numerator + unknown_rate.numerator, total);
                            assert_eq!(coverage.denominator, total);
                            assert_eq!(unknown_rate.denominator, total);

                            let false_agreement = counts.false_agreement();
                            let detection = counts.detection();
                            assert_eq!(
                                false_agreement.numerator + detection.numerator,
                                false_agreement.denominator
                            );
                            assert_eq!(false_agreement.denominator, detection.denominator);

                            for ratio in [
                                coverage,
                                unknown_rate,
                                false_agreement,
                                detection,
                                counts.false_drift(),
                            ] {
                                assert_eq!(
                                    ratio.value.is_none(),
                                    ratio.denominator == 0,
                                    "value.is_none() must hold in both directions for {ratio:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- NC-7

#[test]
fn nc_7_a_rule_returning_unknown_everywhere_produces_null_not_zero_or_one() {
    // Truth-F cases the rule never bound: 3 of them, all Unknown.
    let all_unknown = CountVector {
        n_tt: 0,
        n_tf: 0,
        n_tu: 2,
        n_ft: 0,
        n_ff: 0,
        n_fu: 3,
    };
    assert_eq!(
        all_unknown.false_agreement().value,
        None,
        "an empty denominator is null, never a flattering 0.0"
    );
    assert_eq!(all_unknown.unknown_rate().value, Some(1.0));
    assert_eq!(all_unknown.coverage().value, Some(0.0));
}

#[test]
fn nc_3_an_all_true_population_cannot_report_a_false_agreement_rate() {
    let all_true_labels = CountVector {
        n_tt: 5,
        n_tf: 1,
        n_tu: 2,
        n_ft: 0,
        n_ff: 0,
        n_fu: 0,
    };
    assert_eq!(all_true_labels.false_agreement().denominator, 0);
    assert_eq!(
        all_true_labels.false_agreement().value,
        None,
        "a uniformly positive population must never yield a 100% claim"
    );

    let empty = CountVector::default();
    for ratio in [
        empty.coverage(),
        empty.unknown_rate(),
        empty.false_agreement(),
        empty.detection(),
        empty.false_drift(),
    ] {
        assert_eq!(
            ratio.value, None,
            "a zero-case corpus reports no rate at all"
        );
    }
}

// ---------------------------------------------------------------- NC-25 / §7.2

#[test]
fn nc_25_the_emitted_oos_reason_does_not_rest_on_a_contested_reading() {
    let artifact = advise(&request(approved_evidence_root())).expect("the approved evidence binds");

    match &artifact.out_of_sample {
        OutOfSampleStatus::Unknown {
            reason,
            also_applicable,
        } => {
            assert_eq!(
                *reason,
                OosUnknownReason::NoEligibleUnseenPopulation,
                "the primary reason must be a measured fact about the population, not a reading of L358"
            );
            assert_eq!(
                *also_applicable,
                vec![
                    OosUnknownReason::SyntheticOrExposedPopulationOnly,
                    OosUnknownReason::NoRealLabelPopulation,
                    OosUnknownReason::NoModelFitted,
                ],
                "every later applicable reason is listed, in declaration order"
            );
            assert!(
                !also_applicable.contains(reason),
                "the primary reason is not repeated in also_applicable"
            );
        }
        other => panic!("M3-as-selected constructs no Measured variant, got {other:?}"),
    }

    let json: serde_json::Value =
        serde_json::from_str(&artifact.to_canonical_json()).expect("the artifact is valid JSON");
    assert_eq!(json["out_of_sample"]["status"], "Unknown");
    assert_eq!(
        json["out_of_sample"]["reason"],
        "NoEligibleUnseenPopulation"
    );
    assert_eq!(
        json["out_of_sample"]["also_applicable"]
            .as_array()
            .expect("also_applicable is an array")
            .len(),
        3
    );
    assert_ne!(
        json["out_of_sample"]["reason"], "NoModelFitted",
        "NC-25: NoModelFitted is never the emitted primary reason"
    );
}

// ---------------------------------------------------------------- NC-11 / NC-21 / key set

#[test]
fn nc_11_and_nc_21_and_the_pinned_key_set() {
    let artifact = advise(&request(approved_evidence_root())).expect("the approved evidence binds");
    let text = artifact.to_canonical_json();
    let json: serde_json::Value = serde_json::from_str(&text).expect("the artifact is valid JSON");

    assert_eq!(artifact.schema_version, SCHEMA_VERSION);
    assert_eq!(
        SCHEMA_VERSION, 1,
        "the key set below is pinned to version 1"
    );

    // NC-11: no gradient value anywhere in the emitted artifact.
    let mut strings = Vec::new();
    collect_strings(&json, &mut strings);
    for value in &strings {
        assert!(
            value != "P" && value != "D",
            "NC-11: a forbidden lattice value {value:?} was emitted"
        );
    }

    // NC-21: every artifact carries the key, present even when null.
    let provenance = json["provenance"].as_object().expect("provenance object");
    assert!(
        provenance.contains_key("evidence_class"),
        "NC-21: evidence_class must be present as a key"
    );

    let mut keys = Vec::new();
    collect_keys(&json, String::new(), &mut keys);
    keys.sort();
    keys.dedup();

    let mut expected: Vec<String> = EXPECTED_KEYS.iter().map(|k| (*k).to_string()).collect();
    expected.sort();
    assert_eq!(
        keys, expected,
        "the emitted key set is pinned together with schema_version; bump SCHEMA_VERSION to change it"
    );
}

const EXPECTED_KEYS: &[&str] = &[
    "schema_version",
    "rule",
    "window_bytes",
    "provenance",
    "provenance.manifest_file_name",
    "provenance.manifest_sha256",
    "provenance.evidence_aggregate_ix_agg_1",
    "provenance.declared_bundle_aggregate_gaia_agg_1",
    "provenance.subject_aggregate_ix_agg_1",
    "provenance.window_reference_digest",
    "provenance.rule_source_digest",
    "provenance.corpus_digest_ix_agg_1",
    "provenance.ledger_digest_ix_agg_1",
    "provenance.case_id",
    "provenance.evidence_class",
    "rows",
    "rows[].name",
    "rows[].declared_bytes",
    "rows[].corroborated_bytes",
    "rows[].measured_bytes",
    "rows[].head_window_matches",
    "rows[].tail_window_matches",
    "rows[].state",
    "root",
    "root.declared_files",
    "root.present_files",
    "root.unlisted_names",
    "root.missing_names",
    "root.non_file_names",
    "cost",
    "cost.budget_bytes",
    "cost.bytes_read",
    "cost.files_opened",
    "cost.reference_bytes_read",
    "cost.wrt_construction_bytes",
    "cost.provenance_bytes_read",
    "cost.cost_ratio",
    "cost.cost_ratio.numerator",
    "cost.cost_ratio.denominator",
    "cost.cost_ratio.value",
    "reconciles",
    "out_of_sample",
    "out_of_sample.status",
    "out_of_sample.reason",
    "out_of_sample.also_applicable",
];

// ---------------------------------------------------------------- source-text controls

/// The one named private function inside which `OutOfSampleStatus::Measured`
/// may be constructed.
const MEASURED_CONSTRUCTION_SITE: &str = "fn measured_from_earned_holdout";

#[test]
fn nc_20_exactly_one_named_private_measured_construction_site() {
    // NC-20 restated (preflight §10.3), not relaxed. The former implementation
    // asserted **zero** occurrences of the token, which forbade the earned
    // construction along with the unearned one and so made `Measured`
    // unreachable on every input — blocker B2. The purpose NC-20 always stated
    // was "no *unearned* `Measured`", and that purpose is enforced exactly
    // here: exactly one construction site, inside one named private function.
    // Any second site, and any site outside that function, fails.
    //
    // What removing the zero-occurrence rule gave up is repaid with interest by
    // the behavioural controls NC-20a/b/c, which the source-text grep never
    // covered: the grep could only see *where* the variant is written, never
    // *under what conditions* it is reached.
    let mut sites: Vec<String> = Vec::new();
    let mut named_body: Option<(String, std::ops::Range<usize>)> = None;

    for (path, text) in crate_source_files() {
        // The scanner strips line comments only, so a block comment could hide
        // a construction from it. There are none, and this keeps it that way.
        assert!(
            !text.contains("/*"),
            "NC-20: {path} carries a block comment; the scanner strips only `//` line comments \
             and would not see a construction hidden inside one"
        );

        let lines: Vec<&str> = text.lines().collect();

        if let Some(start) = lines
            .iter()
            .position(|line| line.trim_start().starts_with(MEASURED_CONSTRUCTION_SITE))
        {
            // A free function at module level closes with `}` in column zero.
            let end = lines
                .iter()
                .skip(start)
                .position(|line| *line == "}")
                .map(|offset| start + offset)
                .unwrap_or_else(|| {
                    panic!("NC-20: {MEASURED_CONSTRUCTION_SITE} in {path} has no closing brace")
                });
            assert!(
                named_body.is_none(),
                "NC-20: {MEASURED_CONSTRUCTION_SITE} is declared more than once"
            );
            named_body = Some((path.clone(), start..end));
        }

        for (index, line) in lines.iter().enumerate() {
            if strip_line_comment(line).contains("OutOfSampleStatus::Measured") {
                sites.push(format!("{path}:{}", index + 1));
            }
        }
    }

    let (site_path, site_line) = {
        assert_eq!(
            sites.len(),
            1,
            "NC-20: `OutOfSampleStatus::Measured` must be constructed at exactly one site; got {sites:?}"
        );
        let site = &sites[0];
        let (path, line) = site
            .rsplit_once(':')
            .expect("a site is rendered as path:line");
        (
            path.to_string(),
            line.parse::<usize>().expect("a line number"),
        )
    };

    let (named_path, body) =
        named_body.unwrap_or_else(|| panic!("NC-20: no `{MEASURED_CONSTRUCTION_SITE}` declared"));
    assert_eq!(
        site_path, named_path,
        "NC-20: the construction site must live in the file that declares {MEASURED_CONSTRUCTION_SITE}"
    );
    assert!(
        body.contains(&(site_line - 1)),
        "NC-20: the construction at {site_path}:{site_line} is outside the body of \
         {MEASURED_CONSTRUCTION_SITE} (lines {}..{})",
        body.start + 1,
        body.end + 1
    );
}

/// Everything before a `//`. Conservative: it can only hide text from the
/// scanner, and a variant construction is never written inside a string.
fn strip_line_comment(line: &str) -> &str {
    match line.find("//") {
        Some(at) => &line[..at],
        None => line,
    }
}

#[test]
fn nc_11a_the_derived_hexavalent_ord_is_never_used() {
    // §7.3 states this as the honest weaker guarantee it is: a rule-level
    // scan, not a construction-level impossibility. `ix_types::Hexavalent`
    // derives `Ord` in declaration order, which is not a truth ordering, so
    // `a < b` compiles and would be silently wrong.
    for (path, text) in crate_source_files() {
        for (index, line) in text.lines().enumerate() {
            if !line.contains("Hexavalent") {
                continue;
            }
            // Generic positions carry angle brackets that are not comparisons.
            let stripped = line
                .replace("<Hexavalent", "")
                .replace("Hexavalent>", "")
                .replace("-> ", "");
            for forbidden in [
                " < ", " > ", " <= ", " >= ", ".cmp(", ".min(", ".max(", ".sort",
            ] {
                assert!(
                    !stripped.contains(forbidden),
                    "NC-11a: {forbidden:?} applied on a Hexavalent-bearing line at {}:{}: {line}",
                    path,
                    index + 1
                );
            }
        }
    }
}

// ---------------------------------------------------------------- helpers

fn crate_source_files() -> Vec<(String, String)> {
    let mut out = Vec::new();
    collect_sources(&crate_src_root(), &mut out);
    assert!(!out.is_empty(), "the crate has source files to scan");
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn collect_sources(dir: &std::path::Path, out: &mut Vec<(String, String)>) {
    for entry in std::fs::read_dir(dir).expect("src directory is readable") {
        let entry = entry.expect("directory entry is readable");
        let path = entry.path();
        if path.is_dir() {
            collect_sources(&path, out);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push((
                path.display().to_string(),
                std::fs::read_to_string(&path).expect("source file is UTF-8"),
            ));
        }
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

fn collect_keys(value: &serde_json::Value, prefix: String, out: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                out.push(path.clone());
                collect_keys(child, path, out);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_keys(item, format!("{prefix}[]"), out);
            }
        }
        _ => {}
    }
}
