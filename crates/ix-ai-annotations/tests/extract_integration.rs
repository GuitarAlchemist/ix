use ix_ai_annotations::{extract_workspace, AnnotationKind, TruthValue};
use std::path::PathBuf;

#[test]
fn fixture_dir_yields_expected_annotations() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut annotations = extract_workspace(&fixtures).expect("extract ok");

    // Deterministic order for assertions
    annotations.sort_by(|a, b| {
        a.location
            .path
            .cmp(&b.location.path)
            .then_with(|| a.location.line_start.cmp(&b.location.line_start))
    });

    // 5 in sample.rs (invariant, assumption, contract, hypothesis, smell)
    // 1 in sample.py (hint)
    assert_eq!(
        annotations.len(),
        6,
        "expected 6 annotations, got {} -> {:#?}",
        annotations.len(),
        annotations.iter().map(|a| &a.claim).collect::<Vec<_>>()
    );

    let kinds: Vec<_> = annotations.iter().map(|a| a.kind).collect();
    assert!(kinds.contains(&AnnotationKind::Invariant));
    assert!(kinds.contains(&AnnotationKind::Assumption));
    assert!(kinds.contains(&AnnotationKind::Contract));
    assert!(kinds.contains(&AnnotationKind::Hypothesis));
    assert!(kinds.contains(&AnnotationKind::Smell));
    assert!(kinds.contains(&AnnotationKind::Hint));

    // Truth values cover the spectrum we wrote in the fixture
    let tvs: Vec<_> = annotations.iter().map(|a| a.truth_value).collect();
    assert!(tvs.contains(&TruthValue::T));
    assert!(tvs.contains(&TruthValue::P));
    assert!(tvs.contains(&TruthValue::U));
    assert!(tvs.contains(&TruthValue::D));

    // Every annotation gets a deterministic sha256 id
    for a in &annotations {
        assert!(
            a.id.starts_with("sha256:") && a.id.len() == "sha256:".len() + 64,
            "bad id: {}",
            a.id
        );
        assert_eq!(a.schema_version, 1);
    }
}

#[test]
fn id_is_stable_across_runs() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let first = extract_workspace(&fixtures).unwrap();
    let second = extract_workspace(&fixtures).unwrap();
    let ids1: Vec<_> = first.iter().map(|a| a.id.clone()).collect();
    let ids2: Vec<_> = second.iter().map(|a| a.id.clone()).collect();
    assert_eq!(ids1, ids2);
}

#[test]
fn confidence_defaults_to_half_when_omitted() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let annotations = extract_workspace(&fixtures).unwrap();
    let hint = annotations
        .iter()
        .find(|a| a.kind == AnnotationKind::Hint)
        .expect("hint present in sample.py");
    assert_eq!(hint.confidence, 0.5);
}
