# `ix_voicing_mesh` — un maillage de pipelines à 100+ nœuds sur le corpus de voicings

Une démonstration exécutable et concrète du **substrat de maillage de pipelines**
([ADR-0004](../../adr/0004-duckdb-sql-pipeline-mesh.md)) : composer ~120 « pipelines »
IX sur le corpus de voicings de guitare de GA, sur le banc d'analyse DuckDB, puis
corréler leurs sorties pour en dégager la structure.

> _English version: [`docs/walkthroughs/voicing-mesh.md`](../../walkthroughs/voicing-mesh.md)._

Pour l'exécuter :

```bash
cargo run -p ix-duck --example ix_voicing_mesh --features duck
```

## La question

> **Quelles classes d'ensembles (set-classes) d'accords se comportent de la même façon
> le long du manche, et quelle set-class est le pivot structurel de cette géométrie ?**

La démo répond à cette question pour **deux axes**, un maillage chacun :

- **Position** (`minFret`) — *où* sur le manche se situent les voicings d'une set-class.
- **Écart** (`fretSpan`) — *à quel point* ses formes s'étirent (une lentille de
  difficulté ergonomique).

Le corpus (`state/voicings/raw/guitar.jsonl`, ~667 k doigtés, dont 558 k sont
classables en 136 set-classes de Forte) est lu directement avec `read_json_auto` —
sans étape d'ingestion ni base de données séparée.

> **Disponibilité du corpus.** Le dump brut complet est exclu de git (110 Mo) : il
> n'existe que sur une machine ayant exécuté `ix-voicings`. Sur un dépôt fraîchement
> cloné, la démo bascule automatiquement sur l'échantillon suivi de 500 voicings
> (`state/voicings/guitar-corpus.json`, même schéma) et réduit ses seuils en
> conséquence — elle s'exécute donc toujours, mais le résultat-phare du pivot 5-29
> ci-dessous nécessite le corpus complet. Pour le produire : `cargo run -p ix-voicings`.

## Un maillage « à deux couches »

C'est la forme la plus complète d'ADR-0004 : un graphe de **~460 opérateurs** dont les
**~120 flux** (streams) en sortie alimentent un maillage de corrélation N×N.

- Un **flux** correspond à une set-class de Forte. L'axe d'alignement est la **position
  sur le manche** (`minFret`) : chaque flux est donc le *profil de position sur le
  manche* de cette set-class — la répartition de ses voicings le long du manche.
- Le **graphe d'opérateurs** est matérialisé sous forme d'`ix_pipeline::dag::Dag` et
  affiché :

  ```text
  read_json_auto ─▶ ix_forte_number (annoter) ─▶ GROUP BY <axe> (regrouper)    [tête partagée]
    ─▶ profile:k ─▶ normalize:k ─▶ residual:k ─▶ smooth:k                        [×120 flux]
                         └────────▶ mode-commun ────────┘                        [barrière]
    ─▶ ix_pearson (N×N) ─▶ |r| ≥ τ ─▶ ix_connected_components ─▶ ix_centrality   [queue partagée]
  ```

  Une exécution typique rapporte **464 opérateurs, 803 arêtes, profondeur 12,
  éventail de 114** — un véritable graphe de pipeline à plus de 100 nœuds, et non un
  graphe conceptuel.

Chaque étape est une UDF IX sur le banc DuckDB : `ix_forte_number`,
`ix_wavelet_denoise`, `ix_pearson`, `ix_connected_components`, `ix_centrality`. Aucune
statistique n'est réimplémentée dans la démo.

## L'idée porteuse : la suppression du mode commun

La première exécution (balle traçante) à 8 flux a révélé un biais que l'exécution
complète aurait masqué : **toutes** les set-classes courantes s'entassent dans les
frettes basses, si bien que les profils bruts se corrèlent tous à ~1,0 — une seule
grande clique, une centralité d'intermédiarité (betweenness) nulle pour chaque nœud,
aucun pivot. La tendance partagée « la plupart des voicings vivent en bas du manche »
noie le signal.

Le remède est une suppression du mode commun classique (comme les résidus du modèle de
marché ou la régression du signal global en IRMf), et il réside dans le **pipeline de
construction des flux**, pas dans le maillage générique :

1. normaliser chaque profil en une **distribution** positionnelle (afin qu'une
   set-class courante ne pèse pas plus qu'une rare — Pearson est déjà invariant
   d'échelle, mais cela rend l'étape 2 propre) ;
2. soustraire la **distribution moyenne inter-set-classes** frette par frette, ce qui
   laisse l'*anomalie* de chaque set-class — là où elle sur- ou sous-représente par
   rapport à la set-class typique.

Pearson sur ces résidus corrèle la co-localisation *distinctive*, ce qui est
précisément ce que demande la question.

> C'est la discipline de la balle traçante telle qu'elle doit fonctionner : construire
> d'abord la tranche de bout en bout la plus fine, la laisser révéler l'inconnu, puis
> passer à l'échelle.

## La sortie brute du maillage

Sur les set-classes les mieux supportées (≥ 1000 voicings chacune), les deux axes
rapportent des **set-classes de tête (betweenness) différentes** — ce qui suggère que la
géométrie de *où* se situent les accords diffère de celle de *à quel point* ils
s'étirent :

| Axe | Flux | Tête betweenness (τ = 0,8) | Betweenness |
|---|---|---|---|
| **Position** (`minFret`) | 114 | **5-29** | ≈ 112 (3,5× le deuxième) |
| **Écart** (`fretSpan`) | 120 | **2-3** | ≈ 185 (2,2× le deuxième) |

Ce sont les sorties *brutes* du maillage, pas encore des conclusions. Les deux têtes sont
bien supportées (5-29 : 8 568 voicings ; 2-3 : 2 024) : aucune n'est un artefact de
*profil clairsemé* — mais « bien supporté » n'est pas « significatif ». La section
**Validation** ci-dessous les soumet à un modèle nul, et une seule survit.

### Le balayage de τ : une toile qui se fracture en régions nommées

Le seuil `|r|` noté `τ` est le levier de réglage (ADR-0004). À `τ = 0,8`, chaque axe
est une seule toile connexe ; augmenter `τ` la fracture, et la démo nomme chaque région
survivante par la bande de frettes que ses membres *surreprésentent de façon distinctive*
(l'argmax de leur résidu moyen — ce que le maillage a réellement regroupé, et non le
pic brut, qui vaut « open » pour presque toutes les set-classes) :

```text
POSITION :  τ=0,80 → 1 région   τ=0,90 → 2   τ=0,95 → 3   τ=0,98 → 3
ÉCART :     τ=0,80 → 1 région   τ=0,90 → 3   τ=0,95 → 3   τ=0,98 → 5
```

Sur l'axe de l'écart, la fracture est la plus parlante : une région **wide** (large)
dominante détache de petites régions **moderate** (modérée) — p. ex.
`{3-10, 4-9, 6-33, 6-32}` —, c.-à-d. des grappes de set-classes qui préfèrent un
écart modéré là où la masse préfère un écart large.

## Validation (modèle nul)

Un « pivot » de betweenness n'a de sens que si la concentration observée dans les vraies
données dépasse ce que le pipeline fabrique à partir d'une entrée sans structure. On teste
chaque axe contre **1 000 maillages nuls**, chacun construit en **permutant les comptes de
chaque case (bin) entre les set-classes** — une permutation qui *conserve* la tendance de
mode commun (les frettes basses / les grands écarts sont surpeuplés) mais *détruit* toute
co-variation propre à chaque set-class. Le même pipeline (normaliser → suppression du mode
commun → |Pearson| → τ = 0,8 → betweenness) tourne sur le réel et le nul ; la statistique
est le betweenness de tête, et `p` est la fraction de nuls qui égalent ou dépassent la
valeur réelle.

| Axe | Betweenness réel de tête | Nul (moyenne / 95e / max) | `p`(réel ≤ nul) | Verdict |
|---|---|---|---|---|
| **Position** | 119,9 | 0,1 / 0 / 5,0 | **0,001** | ✅ **signal** |
| **Écart** | 122,1 | 258 / 455 / 788 | **0,98** | ❌ **artefact** |

Trois conclusions honnêtes :

1. **La structure de position est réelle.** La permutation effondre le betweenness à ≈ 0
   (le graphe nul n'a aucun pont), tandis que les résidus de position réels forment un
   véritable squelette de grappes-et-ponts (betweenness ≈ 120, `p` = 0,001). Les set-classes
   de voicings de guitare ont bel et bien une **co-variation de position non aléatoire**.
2. **L'*identité* du pivot unique est fragile.** Une exécution de validation sans l'étape de
   lissage par ondelettes (et avec le betweenness de networkx) désigne **4-17**, et non 5-29,
   les deux premiers étant quasi à égalité. Donc *« il existe une vraie structure de position »*
   est étayé ; *« 5-29 est **le** pivot »* est **surévalué** — la tête désignée bouge selon de
   petits changements de pipeline.
3. **Le résultat de l'écart est un artefact.** Le betweenness réel de l'écart est *en dessous*
   de la médiane nulle (`p` = 0,98) : avec seulement 5 cases de `fretSpan`, l'espace des
   résidus est trop grossier pour porter un signal, et les données aléatoires produisent *plus*
   de structure de pivot que le corpus réel. Le « pivot » `2-3` de l'écart ne survit **pas** —
   à traiter comme du bruit.

La leçon se généralise : **ce maillage, comme toute méthode de corrélation, exige un modèle
nul avant que ses pivots ne deviennent des affirmations.** Le résultat de position passe ce
seuil ; celui de l'écart, non.

## Portée et réserves

- **Analyse consultative uniquement.** Selon ADR-0004, le maillage n'est jamais une
  barrière contraignante ni une source de vérité. Les verdicts contraignants passent
  par le `maintain-gate` gouverné (ADR-0002).
- Le langage de composition ici est le **SQL DuckDB**, pas IXQL — ils restent
  complémentaires (ADR-0001 + ADR-0004).
- Les voicings avec `minFret` ≥ 18 sont écartés (une petite queue) ; la démo utilise
  `FRETS = 18`.
- Les paramètres de réglage sont des constantes en tête de l'exemple : `FRETS`,
  `MIN_SUPPORT`, `TOP_K`, `THRESHOLD`.
