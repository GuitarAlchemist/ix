# `ix_dev_rhythm` — un modèle de Markov / HMM du développement d'IX lui-même

Une démo dogfood à saveur séquentielle : le jeu de données est l'**historique git d'IX**,
et les outils sont la chaîne de Markov + le modèle de Markov caché (HMM) d'IX. Elle lit le
*type* de chaque commit (convention : feat / fix / docs / chore / refactor / test / other)
dans l'ordre chronologique et demande quelle structure porte cette séquence de 539 symboles.

> _English version: [`docs/walkthroughs/dev-rhythm.md`](../../walkthroughs/dev-rhythm.md)._

```bash
cargo run -p ix-duck --example ix_dev_rhythm --features duck
```

## Trois étapes

**1. Chaîne de Markov.** Une matrice de transition de premier ordre sur les types de
commit alimente `ix_graph::markov::MarkovChain` ; sa **distribution stationnaire** est le
mélange à long terme :

```text
feat 45%   chore 17%   docs 13%   fix 10%   other 6%   refactor 5%   test 4%
P(fix | feat) = 6,8%   vs base P(fix) = 9,8%
```

Un fix est *moins* probable juste après un feat qu'au niveau de base — parce que les feat
**se regroupent** (pendant une poussée de fonctionnalité, le commit suivant est
généralement un autre feat), évinçant tout le reste du créneau immédiatement suivant.

**2. Validation — l'ordre porte-t-il une structure ?** Si les types étaient i.i.d., le type
suivant ne serait pas plus prévisible que la marginale. La statistique est l'exactitude de
prédiction « prédire `argmax P(suivant | courant)` » ; le modèle nul **permute la séquence**
(détruit l'ordre, conserve la marginale) et recalcule, 2 000 fois :

```text
exactitude Markov du type suivant = 51,3%   (base toujours-feat = 47,7%)
p de permutation = 0,0005  → vraie structure séquentielle
```

Le gain sur la base est **modeste (~3,6 pts)** mais bien hors du nul — le flux de commits
d'IX a réellement une structure de premier ordre, ce n'est pas un sac de types
indépendants.

**3. Phases cachées — apprises, pas supposées.** Un HMM à 2 états est **entraîné par
Baum-Welch** (`ix_graph::hmm`, sans émissions fixées à la main), puis le chemin d'états
caché le plus probable est décodé par l'UDF **`ix_viterbi`** sur le banc :

```text
phase A (69% de l'histoire) émet : feat 64%, docs 12%, fix 7%     ← construction de fonctionnalités
phase B (31% de l'histoire) émet : chore 40%, docs 16%, fix 15%   ← maintenance
rythme récent (40 derniers commits) : AAAAAAAAAA BBBBBBBBBBBBBBBBBBBB AAAAAAAAAA
```

Le modèle retrouve deux modes de développement interprétables à partir du flux brut — une
phase de **construction de fonctionnalités** (dominée par feat) et une phase de
**maintenance** (dominée par chore : commits-bot de fleet-status / snapshot + correctifs) —
et la chronologie Viterbi les montre en alternance, avec une récente rafale de maintenance
encadrée par du travail de fonctionnalité.

## Ce qui est natif IX ici

| Étape | Primitive IX |
|---|---|
| transition + stationnaire | `ix_graph::markov::MarkovChain` |
| entraînement des phases cachées | `ix_graph::hmm::HiddenMarkovModel::baum_welch` |
| décodage des phases (sur le banc) | UDF `ix_viterbi` |
| significativité | modèle nul par permutation (même motif que les démos maillage / santé-code) |

## Portée et réserves

- **Consultatif, illustratif.** Une description de la façon dont IX a été développé, pas une
  prescription.
- L'analyse du type de commit est une **heuristique** sur la ligne de sujet ; les commits de
  fusion / non conventionnels tombent dans `other`.
- La validation est **en échantillon** (le modèle nul l'est aussi, la comparaison est donc
  équitable) et l'effet, quoique significatif, est faible.
- **2 états** est un choix, pas une découverte ; davantage d'états affineraient les phases.
  Baum-Welch trouve un optimum local depuis le départ initialisé.
- S'exécute partout où `git` + le dépôt sont présents — les données sont l'historique
  lui-même.
