---
name: machin-adversarial
description: Test model robustness with adversarial attacks and defenses
---

# Adversarial ML

Test and improve ML model robustness against adversarial manipulation. Defensive security context only.

## When to Use
When the user wants to evaluate model robustness, generate adversarial examples for training, detect poisoned data, or add privacy protections.

## Evasion Attacks (test robustness)
- **FGSM** — Fast single-step gradient attack
- **PGD** — Iterative projected gradient descent (stronger)
- **C&W** — Optimization-based, defeats many defenses
- **JSMA** — Minimal feature perturbation via saliency maps
- **UAP** — Universal perturbation that fools many inputs

## Defenses
- **Adversarial training** — Augment training set with adversarial examples
- **Feature squeezing** — Reduce input precision
- **Statistical detection** — Detect adversarial inputs via randomized smoothing
- **Gradient regularization** — Penalize large input gradients

## Poisoning Detection
- **KNN label consistency** — Flag training points whose labels disagree with neighbors
- **Spectral signatures** — Detect backdoor triggers via eigenvalue analysis
- **Influence functions** — Estimate impact of each training point

## Privacy
- **Differential privacy noise** — Gaussian mechanism for gradient privatization
- **Confidence masking** — Temperature scaling to reduce information leakage
- **Prediction purification** — Zero out low-confidence outputs

## Programmatic Usage
```rust
use machin_adversarial::evasion::{fgsm, pgd};
use machin_adversarial::defense::detect_adversarial;
use machin_adversarial::robustness::empirical_robustness;
use machin_adversarial::poisoning::detect_label_flips;
use machin_adversarial::privacy::differential_privacy_noise;
```
