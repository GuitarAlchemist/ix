# `ix_code_health` — une carte de santé du code de l'espace de travail IX, par IX sur IX

Une démo exécutable qui transforme l'arbre source d'IX lui-même en un jeu de données de
santé du code interrogeable sur le banc DuckDB — **pur dogfood**, sans données externes,
exécutable sur n'importe quel dépôt cloné.

> _English version: [`docs/walkthroughs/code-health.md`](../../walkthroughs/code-health.md)._

```bash
cargo run -p ix-duck --example ix_code_health --features duck
```

## La question

> **Quelles crates sont des aberrations de santé — et la santé du code prédit-elle la
> propension au changement (churn git) ?**

`read_text('crates/**/*.rs')` diffuse chaque fichier Rust dans le banc ; les UDF
`ix_code_*` notent chacun (complexité cyclomatique + cognitive, SLOC, smells lexicaux) ;
un `GROUP BY` agrège par crate. 79 crates, 632 fichiers, ~154 k SLOC — le tout analysé
en une seule requête.

## Trois étapes

**1. Table de santé.** Complexité et densité de smells par crate. Les têtes de densité
de smells (`ix-net`, `ix-bracelet`, `ix-manifold`, `ix-dynamics`, `ix-fractal`) sont
toutes des **crates mathématiques** — un indice que la densité de smells lexicaux mesure
en partie la *densité de littéraux numériques*, pas la « mauvaise santé ». C'est
précisément ce biais que l'étape 3 teste.

**2. Carte.** `ix_pca_project` réduit les vecteurs de caractéristiques standardisés par
crate `[mean_cc, max_cc, mean_cog, smell_density, ln SLOC]` en 2-D ; les crates les plus
éloignées du centroïde sont les aberrations multivariées :

```text
ix-duck-ext  dist 9,74  (mean_cc 65,9 — l'extension C-API riche en macros)
ix-agent     dist 5,64
ix-net       dist 4,26  (smells/KLOC 468)
```

**3. Validation — la santé prédit-elle le churn ?** Le churn git par crate (commits
touchant `crates/<name>/`, normalisé en commits/KLOC pour que le test ne soit pas
simplement « les grandes crates changent plus ») est corrélé à chaque métrique de santé,
avec **2 000 permutations nulles** pour la significativité (l'analogue du modèle nul du
maillage — un contrôle de validité externe adapté à une affirmation *prédictive*) :

| Prédicteur | r vs churn/KLOC | p (permutation) | Verdict |
|---|---|---|---|
| cyclomatique moyen | −0,15 | 0,18 | **non prédictif** |
| densité de smells | −0,12 | 0,30 | **non prédictif** |

## Le résultat (un négatif honnête)

**Ni la complexité ni la densité de smells lexicaux ne prédisent le churn dans cet espace
de travail** — les deux corrélations sont dans la plage du modèle nul, et même légèrement
*négatives*. Le signe négatif colle à l'indice de l'étape 1 : la densité de smells suit
surtout la lourdeur mathématique, et ces crates de noyaux numériques (`ix-bracelet`,
`ix-manifold`, …) sont *matures et stables*, pas churny. Le résultat actionnable est donc :
**n'utilisez pas ces smells lexicaux comme proxy de risque de changement ici** — ils n'ont
aucune validité externe mesurée pour le churn.

Un négatif qui survit à un modèle nul vaut plus qu'un positif qui n'en a jamais affronté.

## Portée et réserves

- **Consultatif uniquement** — une *lentille* de santé, pas une barrière.
- `ix_code_smells` est **lexical** (heuristique : TODO, nombres magiques, lignes
  longues, …), donc la densité est bruitée et biaisée par le domaine (le code
  mathématique en déclenche davantage). Le palier B (`code-semantic`, tree-sitter)
  l'affinerait.
- Le churn est aussi déterminé par l'**âge de la crate**, non contrôlé ici — une crate
  récente a peu de commits quelle que soit sa santé. Avec n = 79 crates, le test est peu
  puissant ; lire le négatif comme « aucun signal *détectable* », pas « indépendance
  prouvée ».
- Les données sont le dépôt lui-même : pas de repli sur échantillon — la démo tourne
  partout où `git` et l'arbre source sont présents.
