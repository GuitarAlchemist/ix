# `ix_pipeline_mesh` — un maillage de pipelines *exécutable* à 100+ nœuds

Là où [`ix_voicing_mesh`](maillage-voicings.md) **matérialisait** un DAG d'opérateurs pour
l'*affichage* (le vrai calcul tournait en SQL DuckDB), celui-ci construit un véritable
`ix_pipeline::PipelineSpec`, le `lower()`-ise en `Dag`, et l'**`execute()`** via l'exécuteur
ix-pipeline — chaque nœud est une invocation de *skill* `ix-agent` réelle. Voir le plan :
`docs/plans/2026-06-23-executable-pipeline-mesh.md`.

> _English version: [`docs/walkthroughs/executable-pipeline-mesh.md`](../../walkthroughs/executable-pipeline-mesh.md)._

```bash
cargo run -p ix-agent --example ix_pipeline_mesh
```

## Ce que « exécutable » signifie ici

La pile de pipelines IX est `PipelineSpec` (YAML ou construit en code) → `lower()` →
`Dag<PipelineNode>` → `executor::execute`, où chaque étape invoque un **skill enregistré**
(`ix-registry`). (IXQL — le DSL de gouvernance de Demerzel — est *spec-only* ; son exécuteur
n'existe pas, cf. ADR-0001. `ix-pipeline` est la couche de pipelines IX exécutable.)

```text
executable pipeline mesh — 136 stages (N=16 streams)
lowered → 136 nodes, 240 edges, 2 levels (widest 120 — the pairwise tier)
✓ executed 136 real skill nodes through ix_pipeline::executor (not reified)
```

La forme (Option C du plan) :

```text
cond_i   = autocorrelation(series_i)          [16 nœuds, niveau 0]
dist_ij  = distance(cond_i, cond_j, cosine)   [120 nœuds, niveau 1]   ← l'étage maillage
```

Chaque étape `dist_ij` est câblée à deux étapes `cond` par des références
`{from: "cond_i.autocorrelation"}`, que `lower()` résout en dépendances et que l'exécuteur
substitue à l'exécution. Aucun code de liaison — le câblage *est* le spec.

## Validation (vérité terrain)

Les 16 flux sont **15 tons mutuellement similaires + 1 aberration plantée** (une fréquence
très différente). Le pivot est relu depuis les sorties par paires exécutées (le flux ayant
la plus grande distance cosinus moyenne au reste). Le maillage exécuté doit retrouver
l'aberration plantée :

```text
mean cosine distance to the rest (top 5):
   stream 15: 0.8899  ← planted outlier
   stream 12: 0.0597
   stream  4: 0.0595
validation — hub = stream 15; planted outlier = 15  →  ✓ RECOVERED
```

## Une vraie contrainte, révélée par la construction

La première tentative agrégeait avec un nœud de réduction `graph` **pagerank** (poids
d'arêtes en références `{from: "dist_ij.distance"}` — qui *se* résolvent bien dans les
tableaux). Cela s'exécutait, mais renvoyait un rang **uniforme** : le pagerank sur un graphe
*complet* est uniforme quel que soit le poids, et un spec **ne peut pas seuiller les arêtes
au moment de la construction** car les poids sont des valeurs d'exécution.

Le maillage exécutable est donc l'**étage par paires** (les 100+ nœuds qui tournent
réellement), et l'agrégation seuillée/de centralité est calculée localement à partir des
sorties exécutées. Une centralité pondérée/seuillée *dans* le pipeline nécessiterait un
skill de réduction dédié (Option A du plan) — reporté, car cela figerait un contrat de skill
public.

## Portée et réserves

- Les flux ici sont synthétiques-mais-structurés (aberration plantée) pour fournir une
  vérité terrain vérifiable ; brancher les vrais profils de position de voicings est l'étape
  suivante évidente (cela nécessite le banc DuckDB comme source de données — une dépendance
  plus lourde pour un exemple `ix-agent`).
- Le nombre de nœuds est en O(N²) dans l'étage par paires, par conception — c'est *cela*, le
  maillage ; N = 16 → 136.
- Consultatif/illustratif, comme les autres démos `ix_duck` / maillage.
