# ix — Parcours d'apprentissage (FR)

> Tutoriels en français pour les algorithmes de ML implémentés dans ix.

Cette section contient des traductions françaises des tutoriels ix. Le code Rust est identique à la version anglaise — seules les explications sont traduites. Pour la documentation complète, consultez l'[index anglais](../INDEX.md).

---

## Apprentissage supervisé

| # | Sujet | Ce que vous apprendrez |
|---|-------|----------------------|
| 1 | [Régression linéaire](apprentissage-supervise/regression-lineaire.md) | Ajuster une droite aux données — prédiction de prix immobiliers |
| 2 | [Régression logistique](apprentissage-supervise/regression-logistique.md) | Classification binaire — spam vs. non-spam |
| 3 | [Arbres de décision](apprentissage-supervise/arbres-de-decision.md) | Règles si/alors apprises des données — approbation de prêts |
| 4 | [Forêt aléatoire](apprentissage-supervise/foret-aleatoire.md) | Plusieurs arbres votent ensemble — détection de fraude |
| 5 | [Gradient Boosting](apprentissage-supervise/gradient-boosting.md) | Correction séquentielle d'erreurs — champion des données tabulaires |
| 6 | [K plus proches voisins](apprentissage-supervise/knn.md) | Classifier par voisinage — systèmes de recommandation |
| 7 | [Bayes naïf](apprentissage-supervise/naive-bayes.md) | Classification probabiliste rapide — analyse de sentiments |
| 8 | [SVM](apprentissage-supervise/svm.md) | Classification à marge maximale — frontières d'images |
| 9 | [Métriques d'évaluation](apprentissage-supervise/metriques-evaluation.md) | Matrice de confusion, précision, rappel, F1, courbe ROC, AUC, log loss |
| 10 | [Validation croisée](apprentissage-supervise/validation-croisee.md) | K-Fold, K-Fold stratifié — évaluation fiable de la généralisation |
| 11 | [Rééchantillonnage & SMOTE](apprentissage-supervise/reechantillonnage-smote.md) | Gérer les données déséquilibrées — suréchantillonnage synthétique, sous-échantillonnage |

---

## Terminologie FR/EN

| Français | English | Notes |
|----------|---------|-------|
| Exactitude | Accuracy | Fraction de prédictions correctes |
| Précision | Precision | VP / (VP + FP) — ne pas confondre avec « exactitude » |
| Rappel | Recall | VP / (VP + FN), aussi appelé « sensibilité » |
| Matrice de confusion | Confusion matrix | Tableau VP/FP/FN/VN |
| Validation croisée | Cross-validation | Évaluation sur k plis |
| Surajustement | Overfitting | Le modèle mémorise au lieu de généraliser |
| Sous-ajustement | Underfitting | Le modèle est trop simple pour les données |
| Taux d'apprentissage | Learning rate | Pas de descente, $\eta$ |
| Apprenant faible | Weak learner | Arbre peu profond dans le boosting |
| Pli | Fold | Un sous-ensemble dans la validation croisée |
| Faux positif | False positive | Fausse alarme |
| Faux négatif | False negative | Cas raté |
| Vrai positif | True positive | Détection correcte |
| Forêt aléatoire | Random forest | Ensemble d'arbres par bagging |
| Arbre de décision | Decision tree | Classificateur à base de règles |
| Données déséquilibrées | Imbalanced data | Classes à fréquences très différentes |
| Régression linéaire | Linear regression | Prédiction de valeurs continues |
| Régression logistique | Logistic regression | Classification via sigmoïde |
| Descente de gradient | Gradient descent | Optimisation itérative |
| Impureté de Gini | Gini impurity | Mesure de pureté des nœuds |
| Perte charnière | Hinge loss | Fonction de perte du SVM |
| Hyperplan | Hyperplane | Surface de décision linéaire |
| Marge | Margin | Distance entre frontière et points les plus proches |
| Vraisemblance | Likelihood | Probabilité des données sachant le modèle |
| Caractéristique | Feature | Variable d'entrée du modèle |

---

## Cas pratiques

Projets de bout en bout combinant plusieurs algorithmes.

| # | Sujet | Algorithmes combinés |
|---|-------|---------------------|
| 1 | [Détection de fraude (SMOTE + Boosting)](cas-pratiques/detection-fraude.md) | SMOTE + Gradient Boosting + Matrice de confusion + ROC/AUC |
| 2 | [Prédiction d'attrition client](cas-pratiques/prediction-attrition.md) | Régression logistique + Validation croisée + ROC/AUC |
| 3 | [Classifieur de spam (TF-IDF)](cas-pratiques/classifieur-spam.md) | TF-IDF + Bayes naïf + Validation croisée + Matrice de confusion |
| 4 | [SIG & Analyse spatiale](cas-pratiques/sig-analyse-spatiale.md) | Kalman + DBSCAN + A* + FFT + HMM/Viterbi + PSAP/premiers intervenants |

---

## Comment utiliser ces tutoriels

**En parallèle avec la version anglaise ?** Les exemples de code sont identiques. Ouvrez la version française pour les explications et la version anglaise pour référence.

**Exécuter le code :** Tous les exemples utilisent les crates ix. Exécutez-les avec :
```bash
cargo run --example <nom>
```

**Contribuer une traduction :** Les tutoriels non encore traduits sont dans [`docs/`](../). Les conventions de nommage utilisent des noms français pour les fichiers (par ex. `validation-croisee.md` pour `cross-validation.md`).
