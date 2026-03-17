---
title: "ML toolkit expansion: gradient boosting, cross-validation, SMOTE, TF-IDF, BatchNorm, bilingual docs"
category: feature-implementation
date: 2026-03-16
tags: [ml-toolkit, gradient-boosting, cross-validation, smote, tfidf, batch-norm, bilingual-docs, psap, first-responder]
components:
  - crates/ix-supervised   # cross-validation, SMOTE, TF-IDF, expanded metrics
  - crates/ix-ensemble     # gradient boosted classifier
  - crates/ix-nn           # BatchNorm
  - crates/ix-agent        # new MCP tool handlers
  - docs/supervised-learning  # EN algorithm tutorials
  - docs/use-cases            # EN end-to-end project tutorials
  - docs/fr                   # full French documentation suite
related_prs: [16]
related_docs:
  - docs/solutions/integration-issues/cross-pollination-4-repo-ecosystem.md
---

# ML Toolkit Expansion + French Documentation + PSAP Use Cases

## What Was Built

Six new modules filling gaps identified from the dataquest.io "14 ML Projects for Beginners to Advanced (2026)" article:

| Module | Crate | Key Types | Tests |
|--------|-------|-----------|-------|
| Gradient Boosted Trees | ix-ensemble | `GradientBoostedClassifier` (multiclass, softmax, regression stumps) | 10 |
| Cross-Validation | ix-supervised | `KFold`, `StratifiedKFold`, `cross_val_score()` | 11 |
| Enhanced Metrics | ix-supervised | `ConfusionMatrix`, ROC/AUC, log loss, MAE, macro/weighted F1 | 26 |
| SMOTE Resampling | ix-supervised | `Smote`, `random_undersample`, `class_distribution` | 13 |
| TF-IDF Text Vectorization | ix-supervised | `CountVectorizer`, `TfidfVectorizer` | 15 |
| BatchNorm | ix-nn | `BatchNorm` (train/inference, running stats, backward) | 7 |

Plus bilingual documentation (EN + FR) and PSAP/first responder GIS scenarios.

## Key Design Decisions

1. **Gradient boosting uses depth-1 stumps only.** Originally accepted a `max_depth` parameter but never honored it. Caught by all 3 reviewers, fixed by removing the parameter entirely. Lesson: don't accept parameters you don't use.

2. **SMOTE placed in ix-supervised, not a separate crate.** Resampling is tightly coupled to classification labels. Correct boundary for current scale.

3. **TF-IDF placed in ix-supervised.** Text vectorization is preprocessing for supervised classifiers. If image/audio features are added later, extract to `ix-features`.

4. **French docs live in `docs/fr/`, English stays at root.** Standard convention (Rust docs, MDN, React). Visibility added via links in README.md and docs/INDEX.md.

## Issues Caught During Review

Three parallel reviewers (architecture, performance, simplicity) found these issues:

### Fixed

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `max_depth` accepted but unused | Speculative parameter | Removed from API (all 9 files) |
| TF-IDF tokenized 3x, built matrix 2x | `fit_transform` called `fit` then `transform`, each re-tokenizing | Extracted `apply_tfidf` helper; `fit_transform` reuses count matrix |
| Per-row Vec allocation in softmax | `softmax_row` returned `Vec<f64>` in hot training loop | Inlined computation directly into proba matrix |
| `total_sum` inside feature loop | Copy-paste from inner context | Hoisted outside the `for feat in 0..p` loop |
| SMOTE used sqrt for KNN distances | Unnecessary computation | Squared distance preserves neighbor ordering |
| SMOTE element-wise matrix copy | Naive loop | `slice_mut().assign()` bulk copy |
| French docs missing accents | ASCII-only authoring | Rewrote all files with proper diacritics |

### Noted for Future

| Issue | Recommendation |
|-------|---------------|
| Dead GPU/CPU branch in transformer.rs | Both branches do identical work; remove in follow-up PR |
| Dead TransformerClassifierState/RegressorState structs | Remove until serialization is implemented |
| BatchNorm/LayerNorm API asymmetry | Align naming (`forward_train`/`forward_inference` vs `forward`/`forward_cache`) |
| `cross_val_score` hardcoded to accuracy | Accept optional scoring function parameter |

## Key API Examples

```rust
// Gradient Boosting
let mut gbc = GradientBoostedClassifier::new(50, 0.1);
gbc.fit(&x, &y);
let pred = gbc.predict(&x_test);

// Cross-Validation (one-liner)
let scores = cross_val_score(&x, &y, || DecisionTree::new(5), 4, 42);

// SMOTE
let smote = Smote::new(5, 42);
let (x_balanced, y_balanced) = smote.fit_resample(&x, &y);

// TF-IDF
let mut tfidf = TfidfVectorizer::new();
let matrix = tfidf.fit_transform(&docs);

// Confusion Matrix + ROC/AUC
let cm = ConfusionMatrix::from_labels(&y_true, &y_pred, 3);
let (prec, rec, f1, support) = cm.classification_report();
let auc = auc_score(&y_true, &y_scores);

// BatchNorm
let mut bn = BatchNorm::new(features);
let out = bn.forward_train(&x);  // uses batch stats
let out = bn.forward_inference(&x);  // uses running stats
```

## Prevention Strategies

### API Design
- **Every builder field must map to behavior.** Write a test that varies the parameter and asserts different outcomes. If you can't, the parameter shouldn't exist.
- **Audit with grep:** search for every struct field in method bodies. If it only appears in the constructor, remove it.

### ML Performance Review Checklist
- [ ] Count how many times each input is traversed (tokenize passes, matrix builds)
- [ ] Hoist loop-invariant computation outside inner loops
- [ ] Ban heap allocations inside training iteration loops
- [ ] Pre-allocate output matrices when shape is known
- [ ] Profile with 10k+ samples to surface quadratic algorithms

### Translation Checklist
- [ ] French text must contain accented characters (e with accents, c-cedilla, etc.)
- [ ] CI lint: flag files in `docs/fr/` with zero non-ASCII characters
- [ ] Keep Rust code blocks identical between EN and FR
- [ ] Maintain FR/EN glossary in `docs/fr/INDEX.md`
- [ ] Update both indexes when adding new docs

## Documentation Delivered

| Type | EN | FR |
|------|----|----|
| Algorithm tutorials | 12 | 11 |
| End-to-end use cases | 4 (churn, fraud, spam, GIS+PSAP) | 4 |
| French glossary | -- | 25 ML terms |
| GIS/PSAP scenarios | 5 (caller location, dispatch, hotspots, MCI, response time) | 5 |

## Cross-References

- [Cross-pollination ecosystem doc](../integration-issues/cross-pollination-4-repo-ecosystem.md)
- [PR #16: ML toolkit expansion + French docs](https://github.com/GuitarAlchemist/ix/pull/16)
- [PR #13: Trainable transformers](https://github.com/GuitarAlchemist/ix/pull/13)
- Inspiration: [14 ML Projects for Beginners to Advanced (2026)](https://www.dataquest.io/blog/machine-learning-projects-for-beginners-to-advanced/)
