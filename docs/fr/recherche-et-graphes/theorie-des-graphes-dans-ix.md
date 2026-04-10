# Théorie des graphes dans IX — où tout se trouve

> **État :** traduction partielle. Version anglaise complète :
> [`docs/guides/graph-theory-in-ix.md`](../../guides/graph-theory-in-ix.md).
> Cette version française couvre l'essentiel ; le texte complet suivra.

**Objectif :** IX fournit des primitives de théorie des graphes réparties
sur dix crates et modules. Cette page est le point d'entrée unique pour
les retrouver. Avant d'ajouter un nouvel algorithme, d'importer `petgraph`
ou de réécrire un BFS à la main, lisez d'abord cette page.

**Audience :** les humains qui rejoignent le workspace, et les sessions
Claude Code qui choisissent une primitive de graphe pour une tâche.

---

## 1. Inventaire

IX couvre trois préoccupations que la plupart des projets séparent en
écosystèmes distincts :

1. **Algorithmes de graphes classiques** — parcours, plus courts chemins,
   DAG, chaînes de Markov
2. **Invariants topologiques et algébriques** — homologie persistante,
   K-théorie, analyse spectrale
3. **Analyse de code appliquée** — graphes d'appels statiques, graphes
   de trajectoire git, walkers orientés agents

### 1.1 Algorithmes de graphes classiques

- **`crates/ix-graph`** — graphes génériques, chaînes de Markov, HMM avec
  Viterbi, espaces d'états, routage d'agents
- **`crates/ix-pipeline::dag::Dag<N>`** — DAG générique avec détection de
  cycles et tri topologique
- **`crates/ix-search`** — A*, Q*, MCTS, minimax, alpha-bêta, BFS/DFS,
  recherche adverse

### 1.2 Invariants topologiques et algébriques

- **`crates/ix-topo`** — homologie persistante, complexes simpliciaux,
  nombres de Betti
- **`crates/ix-ktheory`** — K-théorie algébrique sur les graphes, suites
  de Mayer-Vietoris
- **`crates/ix-code::physics`** — spectre du Laplacien appliqué aux
  graphes d'appels
- **`crates/ix-code::advanced`** — plongements hyperboliques, BSP,
  méthodes spectrales

### 1.3 Analyse de code appliquée

- **`crates/ix-code::semantic`** — extraction de graphes d'appels via
  tree-sitter, avec variantes `CalleeHint` riches (`Bare`, `Scoped`,
  `MethodCall`). **Portée par fichier uniquement** — la résolution
  inter-fichiers est confiée à `ix-context`.
- **`crates/ix-code::topology`** — homologie persistante appliquée aux
  graphes d'appels, retourne les nombres de Betti β₀ (composantes
  connexes) et β₁ (cycles indépendants)
- **`crates/ix-context`** *(en cours)* — walker sur AST + graphe d'appels +
  graphe d'imports + trajectoire git. Cadré comme un système de
  récupération déterministe pour les agents. Voir le brainstorm dans
  [`docs/brainstorms/2026-04-10-context-dag.md`](../../brainstorms/2026-04-10-context-dag.md).

---

## 2. Règles

1. **Ne jamais ajouter `petgraph`, `daggy`, `graph-rs`** ou toute autre
   crate générique de graphes comme dépendance. Les primitives d'IX
   couvrent toutes les formes de graphes courantes. Si vous pensez
   avoir besoin de l'une d'elles, documentez la lacune précise dans
   `docs/brainstorms/` avant le PR.
2. **Ne pas réécrire BFS/DFS à la main.** `ix-search::uninformed` existe
   et est testé.
3. **Ne pas confondre `ix-pipeline::dag` et `ix-graph::graph`.** Le
   premier garantit l'acyclicité à chaque insertion d'arête ; le second
   est un graphe général où les cycles sont permis.
4. **Ne pas ajouter la résolution inter-fichiers à `ix-code::semantic`.**
   Cette crate est conçue pour un seul fichier à la fois. L'inter-
   fichiers est le travail de `ix-context`.
5. **Ne pas chercher les embeddings (similarité vectorielle) quand des
   parcours structurels suffisent.** Voir
   [`docs/brainstorms/2026-04-10-context-dag.md`](../../brainstorms/2026-04-10-context-dag.md)
   qui cadre explicitement la récupération structurelle comme la réponse
   à « RAG est un piège pour le code ».

---

## 3. Ajouter quelque chose de vraiment nouveau

Si après avoir lu cette page vous pensez toujours qu'IX a besoin d'une
primitive de théorie des graphes absente :

1. **Écrire un brainstorm** dans `docs/brainstorms/YYYY-MM-DD-<sujet>.md`
   expliquant la lacune et la primitive existante la plus proche.
2. **Relier la lacune à un cas d'usage concret.** « Complétude en
   théorie des graphes » n'est pas un cas d'usage.
3. **Vérifier si la primitive appartient à une crate existante** (en
   étendant `ix-graph` ou `ix-topo`) ou mérite une nouvelle crate. Les
   nouvelles crates sont coûteuses — le workspace compte déjà plus de
   33 crates.
4. **Escalader via `/octo:brainstorm`** si l'espace de conception est
   ouvert.
5. **Mettre à jour ce guide** dans le même PR que celui qui livre la
   nouvelle primitive.

---

## Voir aussi

- Version anglaise complète avec matrice de sélection, exemples de code
  et références bibliographiques :
  [`docs/guides/graph-theory-in-ix.md`](../../guides/graph-theory-in-ix.md)
- Stratégie « IX comme oracle structurel pour harnesses d'agents » :
  [`docs/brainstorms/2026-04-10-ix-harness-primitives.md`](../../brainstorms/2026-04-10-ix-harness-primitives.md)
- Conception du walker `ix-context` :
  [`docs/brainstorms/2026-04-10-context-dag.md`](../../brainstorms/2026-04-10-context-dag.md)
