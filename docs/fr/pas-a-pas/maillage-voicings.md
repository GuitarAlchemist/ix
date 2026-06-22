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

> **Quelles classes d'ensembles (set-classes) d'accords occupent les mêmes régions du
> manche, et quelle set-class est le pivot structurel de la géométrie des voicings de
> la guitare ?**

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
  read_json_auto ─▶ ix_forte_number (annoter) ─▶ GROUP BY minFret (regrouper)  [tête partagée]
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

## Le résultat

À `τ = 0,8` sur les 114 set-classes les mieux supportées (≥ 1000 voicings chacune,
afin que chaque case de frette soit peuplée et qu'un profil clairsemé ne puisse pas
simuler un pont) :

- Les set-classes forment **une seule toile connexe de géométrie du manche** — aucune
  famille d'accords isolée ; elles sont toutes interreliées en position.
- **5-29 est le pivot structurel** — betweenness **≈ 112**, soit environ 3,5× le
  deuxième, et bien supportée (8 568 voicings) : ce n'est donc pas un artefact de
  profil clairsemé. C'est la set-class dont le résidu de position sur le manche fait le
  plus le pont entre les autres.

Le levier de réglage est le seuil `|r|` noté `τ` (ADR-0004) : à `τ = 0,4`, le maillage
détendancé reste une clique quasi complète avec des égalités de betweenness
dégénérées ; augmenter `τ` affine le pivot.

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
