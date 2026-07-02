# `mesh_correlate` — le skill de réduction (fan-in) maillage-corrélation dans le pipeline

Un skill `ix-agent` enregistré (`#[ix_skill] name = "mesh_correlate"`) qui transforme N
séries en un graphe de corrélation **à l'intérieur d'un pipeline exécutable**, calculant la
centralité là où un spec ne le pourrait pas autrement.

> _English version: [`docs/walkthroughs/mesh-correlate-skill.md`](../../walkthroughs/mesh-correlate-skill.md)._

## Pourquoi il existe

La démo de maillage exécutable ([`ix_pipeline_mesh`](maillage-pipeline-executable.md), plan
`docs/plans/2026-06-23-executable-pipeline-mesh.md`) a révélé une vraie limite : un
`PipelineSpec` **ne peut pas seuiller les arêtes au moment de la construction** (les poids
sont des valeurs d'exécution), donc le regroupement/la centralité du maillage devaient être
calculés dans le harnais. `mesh_correlate` comble cet écart — il fait le seuillage
`|Pearson| ≥ τ` **et** la centralité d'intermédiarité à l'exécution, dans un seul nœud de
réduction (Option A du plan).

```
mesh_correlate({ series: [[…],[…],…], threshold: τ })
  → { correlation: N×N, centrality: [betweenness…], components: […], hub, n_streams }
```

Il enveloppe `ix_math::inference::pearson` + `ix_graph::graph::Graph`
(`betweenness_centrality`, `connected_components`) — pas de nouvel algorithme, juste la
forme de réduction en tant que skill.

## Dans un pipeline

`crates/ix-agent/examples/ix_pipeline_mesh_hub.rs` :

```text
mesh    = mesh_correlate(streams, τ=0.4)   → {centrality, hub, …}
summary = stats({from: mesh.centrality})   → dispersion des scores
✓ executed 2 nodes through ix_pipeline::executor
```

Validation — un **moyeu-et-rayons** (hub-and-spoke) planté (un moyeu = moyenne de 3 rayons
orthogonaux, plus des distracteurs de bruit pur). Le maillage exécuté retourne le moyeu :

```text
stream 0: 3.00  ← planted hub        stream 1: 0.00 (spoke)   …distractors: 0.00
hub = 0; planted hub = 0  →  ✓ RECOVERED (centralité calculée dans le pipeline exécuté)
```

L'étape `stats` en aval consomme `mesh.centrality`, donc le résultat du maillage circule
dans le pipeline comme toute autre sortie d'étape.

## Portée et réserves

- C'est la réduction **Option A** (maillage dans un nœud), complémentaire de l'Option C de
  `ix_pipeline_mesh` (le maillage par paires explicite à 100+ nœuds). Ensemble, elles
  montrent le compromis : un DAG de maillage explicite vs. la centralité dans le pipeline.
- L'exemple passe les séries directement à `mesh_correlate` plutôt que par N étapes de
  conditionnement par flux, car le cond évident (`autocorrelation`) déforme la corrélation
  (chaque ACF partage le pic au lag 0 = 1). Un skill de conditionnement *identité*
  préservant la corrélation restaurerait la forme N-pipelines-par-flux — un suivi.
- Un moyeu de betweenness nécessite une structure **moyeu-et-rayons** ; un ensemble de flux
  mutuellement corrélés est une clique sans moyeu (la même leçon que le maillage de
  voicings).
- Consultatif/illustratif, comme les autres démos de maillage.
