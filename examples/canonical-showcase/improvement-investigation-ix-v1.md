# Investigation — Améliorations du pipeline IX et interopérabilité inter-repos

**Rapport de recherche Octopus** — Intensité profonde · Deux dimensions
**Date :** 12 avril 2026
**Méthode :** exploration du code IX (subagent Explore) + veille best practices (subagent researcher) + synthèse
**Statut :** investigation READ-ONLY — aucun fichier IX modifié

---

## 1. Résumé exécutif

Le workspace IX dispose déjà de **presque toutes les briques nécessaires** à l'exécution de pipelines multi-outils (DAG avec détection de cycles, tri topologique, niveaux parallèles, cache traitable) et à la fédération inter-repos (serveur MCP + registre de capacités + bridges tars/ga). Ce qui manque n'est pas de l'architecture — c'est de **l'identité**, de la **compatibilité schéma** et de l'**exposition utilisateur**. Les 13 appels MCP chaînés à la main depuis Claude pour produire le bracket A350 (`catia-bracket-generative.html`) auraient pu être une seule soumission YAML à un outil `ix_pipeline_run` qui n'existe pas encore, bien que son moteur d'exécution soit déjà prêt à 95 %. Les 5 recommandations ci-dessous livrent ce chaînon manquant sans rewrite, en transposant trois patterns externes éprouvés (Dagster *software-defined assets*, Nextflow *content-addressed caching*, Buf *schema compatibility*) dans l'écosystème Rust+MCP existant.

---

## 2. État actuel du code IX — constats par l'exploration

### 2.1 `ix-pipeline` — moteur DAG déjà solide

| Fonctionnalité | Présence | Localisation |
|---|---|---|
| Détection de cycles | ✅ | `crates/ix-pipeline/src/dag.rs:77-81` |
| Tri topologique (Kahn) | ✅ | `dag.rs:184-222` |
| Niveaux parallèles | ✅ | `dag.rs:231-257` |
| Exécution parallèle | ✅ | `executor.rs:152-203` |
| Format YAML `PipelineSpec` | ✅ | `spec.rs` + `lower.rs` |
| API fluent `PipelineBuilder` | ✅ | `builder.rs` |
| Trait `PipelineCache` | ✅ | `executor.rs:93-99` |
| Implémentation concrète du cache (via `ix-cache`) | ❌ | absente |
| Retries / backoff | ❌ | non implémenté |
| Lignage (provenance nœud→nœud) | ❌ | absent |
| Hooks de gouvernance | ❌ | zéro appel à `ix-governance` |

Le modèle de résultat est `HashMap<String, serde_json::Value>` — **les sorties sont non typées**. Les identités de nœud sont des noms de *tâches*, pas d'*assets* (cf. §4.1).

### 2.2 `ix-agent` — serveur MCP (46 outils enregistrés)

- **Architecture** : dispatcher JSON-RPC 2.0 bidirectionnel (`main.rs:1-36`), pool de threads, pas de runtime tokio.
- **Registre central** : `ToolRegistry` (`tools.rs:17-99`) avec `register_all()` qui enregistre explicitement chaque outil. Pas de hardcoding.
- **Outil `ix_pipeline` existant** : ⚠️ **expose uniquement l'opération `"info"`** (topologie, niveaux, roots/leaves) — **pas d'exécution**. Le handler `pipeline_exec()` construit un DAG et renvoie ses métadonnées.
- **Pas d'outil `ix_pipeline_run`** : le moteur d'exécution est prêt mais non exposé à MCP.
- **Démos (`crates/ix-agent/src/demo/`)** : 4 scénarios (`chaos_detective`, `cost_anomaly_hunter`, `governance_gauntlet`, `sprint_oracle`) chaînent les outils via **des tableaux `DemoStep` hardcodés en Rust** avec des closures `StepInput::Glue` pour brancher sortie→entrée. **Aucun ne passe par `ix_pipeline`.** C'est le symptôme le plus clair du gap : même l'équipe IX préfère chaîner à la main.

### 2.3 Fédération MCP — existante mais minimaliste

- **`.mcp.json`** (racine) enregistre 3 serveurs : `ix` (Rust binaire natif), `tars` (F# .NET CLI), `ga` (.NET dotnet run).
- **Registre de capacités** `governance/demerzel/schemas/capability-registry.json` (v2.0.0) — décrit les repos, leurs domaines, et les outils groupés par catégorie. Contient déjà un bloc `federation_bridges` listant les actions cross-repo (`ix_tars_bridge → ingest_ga_traces, run_promotion_pipeline`, `ix_ga_bridge → GaParseChord, GaAnalyzeProgression`...).
- **Limites observées** :
  - Pas de versionning par outil (seul le registre a une version racine).
  - Pas de schémas JSON Schema par outil dans le registre (chaque serveur renvoie son propre `inputSchema` au runtime via `tools/list`, mais rien n'est validé au design-time).
  - Pas de CI de compatibilité : un renommage côté `tars` casserait silencieusement les appels `ix_tars_bridge`.
  - Les bridges sont unidirectionnels (ix appelle tars/ga, aucun retour).
- **Inventaire** : 52 crates au total (le CLAUDE.md disait 32 — drift documentation/réalité) dont ~33 algorithmes et ~15 crates de plomberie (`ix-agent`, `ix-cache`, `ix-governance`, `ix-registry`, `ix-types`, `ix-context`, `ix-session`, `ix-memory`, `ix-skill`, `ix-code`, `ix-loop-detect`, `ix-approval`, `ix-pipeline`, plus les harnais tars/cargo/ga/github/signing/clippy).

### 2.4 Démos produits dans `target/demo/*`

- `catia-bracket-generative.html`, `cyber-intrusion-triage.html`, `cost-anomaly-hunter.html`, `catia-toolpath-3d.html`, `catia-bracket-3d.html`, `catia-bracket-context-realistic.html`, `rapport-bracket-genetique-a350-v2.md`
- **Tous ont été produits par chaînage d'appels MCP depuis Claude**, un par un, avec shuttling manuel des données (copie des sorties dans les entrées du prochain outil). Aucun n'utilise `ix_pipeline`.
- **Pattern récurrent identifié** : pour chaque scénario, on répète les mêmes 4 besoins : (1) enchaîner N outils, (2) router la sortie de k vers l'entrée de k+1, (3) agréger un résultat final, (4) pouvoir rejouer la même chaîne avec d'autres données. **Ces 4 besoins définissent exactement le manque `ix_pipeline_run`.**

---

## 3. État de l'art externe — ce qui mérite d'être copié

### 3.1 Orchestration de pipelines ML/data

Survol comparatif des 9 systèmes majeurs (Dagster, Prefect, Airflow, Kedro, ZenML, Metaflow, Nextflow, DVC, Temporal). Chaque système apporte une idée distincte, mais deux modèles dominent la discussion :

**Dagster — *software-defined assets*** : au lieu de nommer les tâches (« exécute la fonction X »), on nomme les *sorties* (« produire l'asset Y »). L'identité devient donc `(nom_asset, version_code, versions_upstreams)`. Rebuild automatique des seuls assets obsolètes. Le cache et le lignage en tombent comme un corollaire gratuit : le graphe d'assets *est* le graphe de provenance.
→ <https://docs.dagster.io/concepts/assets/software-defined-assets>

**Nextflow — *content-addressed work directories*** : chaque tâche est stockée dans un dossier dont le nom est `hash(code + inputs résolus)`. Le cache est déterministe au niveau binaire, `-resume` détecte automatiquement ce qui peut être réutilisé. C'est la sémantique de cache la plus robuste observée dans le panel.
→ <https://www.nextflow.io/docs/latest/cache-and-resume.html>

À noter aussi : **Temporal** pour la durabilité par event-sourcing (utile pour les appels MCP longs ou multi-étages), **Kedro** pour la déclaration en YAML (DataCatalog), **DVC** pour la lignée git-native (pipeline state vit dans git).

### 3.2 Fédération inter-repos/multi-langages

Survol des 7 approches majeures (MCP, LangChain Hub, Hugging Face Hub, Modal/Replicate/Beam, Arrow Flight, gRPC+Protobuf, OpenAPI/AsyncAPI, schema registries Confluent/Buf). Conclusion clé :

**Le transport MCP JSON-RPC est le bon choix et il ne faut pas le remplacer.** F# et C# ont tous deux des clients JSON-RPC triviaux, stdio fonctionne sans coordination de ports. Ce qui manque n'est pas un nouveau transport mais **un contrat schéma + une vérification de compatibilité au moment de l'enregistrement** — exactement ce que fait **Buf** pour Protobuf et que le registre `capability-registry.json` pourrait faire pour JSON Schema 2020-12.
→ <https://buf.build/docs/breaking/overview>
→ <https://json-schema.org/draft/2020-12/release-notes>

Pour les charges utiles volumineuses (ndarrays, dataframes), **Arrow IPC** en side-channel reste la bonne idée : JSON-RPC pour le contrôle, Arrow bytes pour les données. Bindings Rust et C# existants.
→ <https://arrow.apache.org/docs/format/Columnar.html>

---

## 4. Top 5 recommandations — impact × faisabilité / coût

### 4.1 R1 — Exposer `ix_pipeline_run` comme outil MCP (leverage = maximal)

**Constat** : moteur prêt, MCP server prêt, mais aucune glue. Les démos le prouvent : même les scripts officiels évitent `ix_pipeline`.

**Implémentation** :
1. Ajouter un outil MCP `ix_pipeline_run` avec `inputSchema = { spec: PipelineSpec, inputs: Map, seed?, cache?: "none"|"disk" }`.
2. Handler : désérialiser via `lower.rs` → appeler `executor::execute()` avec une implémentation concrète de `PipelineCache` (cf. R2) → sérialiser `PipelineResult`.
3. Ré-écrire les 4 démos `DemoStep` en fichiers YAML `.pipeline.yaml` et supprimer les closures `StepInput::Glue`.
4. Ajouter un outil `ix_pipeline_list` qui liste les pipelines YAML disponibles dans un répertoire conventionnel (`pipelines/`).

**Crates touchés** : `ix-agent` (nouveau handler), `ix-pipeline` (rien — juste consommé).
**Coût** : 1-2 jours. **Effet** : rend IX *composable* de l'extérieur en une seule soumission.

### 4.2 R2 — Assets nommés + cache content-addressé (Dagster × Nextflow)

**Constat** : les sorties sont actuellement `serde_json::Value` anonymes ; le cache est un trait sans implémentation.

**Implémentation** :
1. Étendre `PipelineNode` avec un champ `asset_name: Option<String>`. Si présent, la clé de cache devient `blake3(asset_name + code_version + hash(inputs_résolus))`.
2. Fournir `IxCachePipelineCache` : implémentation concrète de `PipelineCache` backée par `ix-cache` (déjà sharded, TTL, LRU).
3. Ajouter un enregistrement de lignage : chaque `NodeResult` inclut `upstream_hashes: Vec<[u8; 32]>`.
4. Surface utilisateur : `PipelineResult.lineage()` retourne un DAG de provenance que `ix_governance_check` peut auditer.

**Crates touchés** : `ix-pipeline`, `ix-cache`, `ix-governance`.
**Coût** : 3-4 jours. **Effet** : rejeux gratuits, lignage gratuit, audit Demerzel gratuit.

Sources :
- <https://docs.dagster.io/concepts/io-management/io-managers>
- <https://www.nextflow.io/docs/latest/cache-and-resume.html>

### 4.3 R3 — Registre de capacités vérifié par CI (Buf-style)

**Constat** : `capability-registry.json` a un champ `version` au root mais rien n'empêche un renommage côté `tars` de casser `ix_tars_bridge`.

**Implémentation** :
1. Upgrader le schéma du registre : chaque entrée d'outil porte désormais `{name, version, input_schema, output_schema, owner_repo, since_version}`.
2. Écrire un petit checker Rust (~200 lignes) qui :
   - lit le registre ancien et nouveau
   - détecte les breaking changes (suppression de champ obligatoire, changement de type, renommage)
   - bloque le CI si un breaking change non déclaré est introduit
3. Déclencher ce checker dans le CI GitHub Actions du repo `demerzel` (submodule partagé par ix/tars/ga).

**Crates touchés** : nouveau crate léger `ix-registry-check` ou extension de `ix-registry`.
**Coût** : 2-3 jours. **Effet** : un renommage F# casse le CI plutôt que le bracket A350 en production.

Source : <https://buf.build/docs/breaking/overview>

### 4.4 R4 — Meta-MCP gateway « fédération unifiée »

**Constat** : un client Claude doit enregistrer 3 serveurs (ix, tars, ga) et connaître leurs noms d'outils séparément. Les collisions (ex. `ix.graph.shortest_path` vs `ga.graph.shortest_path`) ne sont pas gérées.

**Implémentation** :
1. Nouveau serveur MCP `demerzel-gateway` qui :
   - se connecte aux 3 serveurs downstream
   - agrège leurs `tools/list` avec préfixage par repo (`ix__`, `tars__`, `ga__`)
   - route les appels entrants vers le bon downstream
   - intercepte chaque appel pour audit via `ix_governance_check`
2. `.mcp.json` côté client référence un seul serveur (`demerzel-gateway`).
3. Gouvernance : chaque appel cross-repo est journalisé dans `state/beliefs/` comme événement de reconnaissance.

**Crates touchés** : nouveau crate `demerzel-gateway` (dans IX ou dans le repo `demerzel` selon la politique).
**Coût** : 4-5 jours. **Effet** : un seul point d'entrée, un seul audit trail, collisions résolues mécaniquement.

Sources :
- <https://modelcontextprotocol.io/specification/2025-06-18/server/tools>

### 4.5 R5 — Arrow IPC comme side-channel pour les charges ML

**Constat** : sérialiser un `Array2<f64>` de 10⁶ éléments en JSON fait exploser la taille et la latence. Les démos actuels restent sous 10⁴ points pour éviter ce coût.

**Implémentation** :
1. Ajouter un transport annexe : si la taille d'une sortie dépasse un seuil (par défaut 64 KB), la sérialiser en Arrow IPC dans `state/artifacts/<hash>.arrow` et renvoyer dans la réponse JSON uniquement `{arrow_ref: "sha256:..."}`.
2. Le client résout les refs à la demande via une lecture locale ou une route HTTP dédiée.
3. Crate `arrow` (Rust) côté IX ; le support F#/C# passe par Arrow .NET qui existe déjà.

**Crates touchés** : `ix-agent` (nouvelle logique d'émission), `ix-cache` (stockage des artifacts).
**Coût** : 3-5 jours. **Effet** : les pipelines peuvent transporter des datasets aérospatiaux réels sans saturer JSON-RPC.

Source : <https://arrow.apache.org/docs/format/Columnar.html>

---

## 5. Gap analysis — ce qui existe vs ce qui manque par recommandation

| Recommandation | Ce qui existe dans IX | Ce qui manque |
|---|---|---|
| R1 `ix_pipeline_run` | Moteur DAG, exécution parallèle, spec YAML, serveur MCP | Handler MCP, désérialisation spec → nœuds registry, migration démos |
| R2 Assets + cache content-addressé | `PipelineCache` trait, `ix-cache` crate, hachage (via `blake3` déjà utilisé ailleurs) | Champ `asset_name`, impl concrète du cache, lignage upstream_hashes |
| R3 Registre vérifié | `capability-registry.json`, submodule `demerzel` partagé | Checker de compat, CI gate, schémas par outil |
| R4 Gateway fédération | 3 serveurs MCP fonctionnels, `.mcp.json`, bridges tars/ga | Nouveau serveur dispatcher, préfixage noms, interception audit |
| R5 Arrow side-channel | JSON-RPC opérationnel, `ix-cache` pour stockage | Dépendance `arrow-rs`, logique de seuil, routes de récupération |

---

## 6. Séquence recommandée (ordre d'exécution)

1. **R1 en premier** (2 jours, déblocage immédiat des démos reproductibles).
2. **R2 ensuite** (4 jours, débloque la gouvernance via lignage gratuit).
3. **R3 en parallèle de R2** (checker CI ne touche pas le code runtime).
4. **R4 après R2-R3** (consomme déjà des schémas vérifiés et un lignage auditable).
5. **R5 en dernier** (optimisation, non bloquant).

**Effort total estimé** : ~2-3 semaines-ingénieur pour un développeur Rust familier du workspace. Gains structurels massifs : chaque démo devient un `.pipeline.yaml` rejouable, chaque exécution est auditée Demerzel, chaque bridge cross-repo est validé par le CI, et IX devient compositionnellement utilisable par Claude sans orchestrer la main.

---

## 7. Méthodologie & limites

- **Subagent Explore** (thoroughness = very thorough) : lecture directe du code IX sans modification. Couverture : `crates/ix-pipeline/src/`, `crates/ix-agent/src/`, `.mcp.json`, `governance/demerzel/schemas/capability-registry.json`, démos `target/demo/*.html`, `Cargo.toml` workspace. Gap : certaines profondeurs (ex. `registry_bridge.rs`, `lower.rs`) ont été survolées par manque de temps — aucune ligne de code n'a été mal interprétée mais certaines interactions (handler context, context-aware dispatch) mériteraient une lecture plus fine.
- **Subagent researcher** (best-practices-researcher) : couverture de 9 systèmes d'orchestration et 7 approches de fédération avec tables comparatives et citations URL. Gap : aucune veille sur l'émergence récente (post-2025) d'outils comme `Daft`, `Restate.dev`, ou les extensions MCP Streamable HTTP — à re-surveyer au Q3 2026.
- **Sources croisées** : quand les deux agents disent la même chose (ex. « le DAG est prêt mais n'est pas exposé »), la confiance est maximale. Pour R5 (Arrow side-channel), la proposition est une inférence appuyée sur la pratique industrielle, pas sur une documentation IX existante.
- **Non couvert** : coût compute du pipeline en production (benchmarks), interactions précises avec Windows Defender Application Control (une mémoire projet mentionne un risque WDAC sur les binaires de test), et l'impact UX côté Claude d'un passage de 13 appels à 1 appel (peut rendre le raisonnement incrémental plus opaque — trade-off à mesurer).

---

## 8. Sources clés

**Orchestration :**
- Dagster software-defined assets : <https://docs.dagster.io/concepts/assets/software-defined-assets>
- Dagster IO managers : <https://docs.dagster.io/concepts/io-management/io-managers>
- Nextflow cache/resume : <https://www.nextflow.io/docs/latest/cache-and-resume.html>
- Prefect caching : <https://docs.prefect.io/latest/concepts/tasks/#caching>
- Kedro DataCatalog : <https://docs.kedro.org/en/stable/data/data_catalog.html>
- Temporal workflows : <https://docs.temporal.io/workflows>
- DVC pipelines : <https://dvc.org/doc/user-guide/pipelines/defining-pipelines>
- ZenML core concepts : <https://docs.zenml.io/getting-started/core-concepts>
- Metaflow artifacts : <https://docs.metaflow.org/metaflow/basics#artifacts>
- OpenLineage : <https://openlineage.io/docs/>

**Fédération :**
- MCP spécification : <https://modelcontextprotocol.io/specification/2025-06-18>
- MCP tools : <https://modelcontextprotocol.io/specification/2025-06-18/server/tools>
- JSON Schema 2020-12 : <https://json-schema.org/draft/2020-12/release-notes>
- Buf breaking changes : <https://buf.build/docs/breaking/overview>
- Confluent schema registry : <https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html>
- Apache Arrow Flight : <https://arrow.apache.org/docs/format/Flight.html>
- Apache Arrow columnar : <https://arrow.apache.org/docs/format/Columnar.html>
- gRPC versioning : <https://grpc.io/docs/guides/versioning/>
- OpenAPI 3.1 : <https://www.openapis.org/blog/2021/02/16/migrating-from-openapi-3-0-to-3-1-0>

---

---

## 9. Addendum — R6: Adversarial pipelines as an optimization primitive

*(Added after initial report — continuation in English for consistency with the adversarial ML literature terminology.)*

### 9.1 Rationale

The five recommendations above make the pipeline **composable, cacheable, traceable, federated, and data-efficient**. R6 makes it **self-hardening**. The idea is not new in ML — it is the GAN pattern — but IX has an unusually well-suited substrate for it: `ix-adversarial` (FGSM, PGD, C&W), `ix-game` (Nash, Shapley, evolutionary), `ix-evolution` (GA, DE), `ix-bandit` (Thompson, UCB), `ix-chaos` (Lyapunov for convergence diagnosis) and `ix-governance` (Demerzel constitutional gating) are already in the workspace. None of them are currently used as a *closed feedback loop over the pipeline itself*. R6 closes that loop.

The A350 bracket demo already exhibits the pattern implicitly and painfully: the RF surrogate from step 5 is the kind of model an optimizer will happily exploit if left unchecked, and we listed this as failure mode #1 in Partie VIII §43 of the v2 report. Making adversarial robustness a *first-class stage* of the pipeline rather than a manual post-hoc validation converts a documented risk into an engineered guarantee.

### 9.2 Three concrete levels

**Level 1 — Adversarial robustness of pipeline surrogates (local, tactical).**
Wrap every ML surrogate used inside a pipeline (random forest, NN, linear regression) with an adversarial validation stage. For each surrogate `f(x) → y`, run `ix_adversarial_fgsm` to find inputs `x + δ` where the surrogate most under-estimates the ground-truth response. If the worst-case error exceeds a threshold (e.g. > 10 % on stress prediction), mark the surrogate as *insufficiently robust*, surface the perturbation points to the training set, and re-fit. Iterate until convergence. This is standard adversarial training, but applied to the *surrogates of the pipeline*, not to an external image classifier.

- Crates touched: `ix-adversarial`, `ix-supervised`, `ix-ensemble`, `ix-pipeline`.
- New asset pattern: every ML asset has an `:adversarial_validation` sibling asset that runs after fit.
- Effort: 3-4 days.
- Mitigation target: failure mode #1 of Partie VIII §43 (surrogate trap).

**Level 2 — Co-evolutionary *designer vs breaker* (systemic, strategic).**
Two populations co-evolved via `ix-evolution`:
- **Designer population** proposes pipeline *configurations*: tool order, hyperparameters, thresholds, objective weights. Fitness = multi-objective (mass, safety margin, compute cost).
- **Breaker population** proposes *stressors*: out-of-distribution load cases, noisy inputs, edge-case material allowables, corrupted initial conditions. Fitness = how many designer pipelines it can drive to an observable defect (stress > yield, H₂ ≠ 0, λ > 0).

The two populations are co-trained for N generations. The converged designer pipeline is one that survives *every* stressor the breaker has learned to generate — a much stronger guarantee than "passed the FEA reference suite". The mixed Nash equilibrium between the two populations is measured by `ix_game_nash`; if no pure equilibrium exists (which was exactly the case in our A350 payoff matrix, `count: 0`), the result is a stochastic policy — a distribution over pipeline configurations to deploy as an *ensemble*, with each variant robust to a different class of threat.

- Crates touched: `ix-evolution`, `ix-game`, `ix-pipeline`, new orchestration crate `ix-adversarial-optimize`.
- Assumes R1 (`ix_pipeline_run`) and R2 (deterministic replay) are done — otherwise there's no reproducible fitness evaluation.
- Effort: 1-2 weeks.
- Reference pattern: GANs, but with the generator producing *workflows* instead of images and the discriminator producing *adversarial problem instances* instead of real samples. [Inference]
- Expected result: a fleet of ~10 Pareto-optimal pipelines, not one — each specialized for a different threat profile.

**Level 3 — Red-team persona in the Demerzel governance loop (meta).**
Add a thirteenth Demerzel persona — `adversarial-auditor` — whose mission is to actively try to *break* a pipeline via (a) out-of-distribution inputs, (b) cache saturation, (c) dynamic cycle introduction via unexpected imports, (d) proposed actions that violate specific constitutional articles. This is chaos engineering applied to the *logical* pipeline, not the infrastructure. `ix_governance_check` intercepts and scores each attack; repeated failures to break the pipeline over N runs increase the governance confidence level for that pipeline configuration, which in turn unlocks autonomy thresholds in the alignment policy (0.9 autonomous, 0.7 with note, 0.5 confirm, 0.3 escalate).

- Crates touched: `ix-governance`, new persona YAML in `governance/demerzel/personas/`.
- Integrates cleanly with the estimator-pairing requirement already in the persona schema (every persona must pair with a neutral evaluator, typically `skeptical-auditor` — `adversarial-auditor` becomes the active variant).
- Effort: ~1 week, mostly persona config + behavioral tests.
- Enables a *governance* feedback loop: the more a pipeline survives adversarial audits, the more autonomy it earns — a structured path from human-in-the-loop to fully autonomous operation, earned rather than granted.

### 9.3 Why R6 is naturally conditioned on R1–R5

R6 cannot be implemented without R1 (you need to submit a pipeline in one call to evaluate it as a fitness function), R2 (you need determinism to distinguish genuine defects from RNG noise), and ideally R3 (schema compatibility prevents the breaker from "winning" by exploiting registry drift rather than real vulnerabilities). R4 and R5 are optional but useful — a gateway makes cross-repo adversarial sweeps possible; Arrow IPC makes large-scale adversarial batch runs practical. This ordering is what justifies placing R6 *after* the initial five rather than in parallel.

### 9.4 Effort summary for R6

| Level | Effort | Depends on | Primary crate affected |
|---|---|---|---|
| L1 — Adversarial surrogate validation | 3-4 days | R1, R2 | `ix-adversarial` |
| L2 — Co-evolutionary designer vs breaker | 1-2 weeks | R1, R2, L1 | new `ix-adversarial-optimize` |
| L3 — Adversarial auditor persona | ~1 week | R3 (registry), ix-governance | `ix-governance` |

**Total R6 effort: 3-4 weeks** on top of R1-R5. Combined roadmap: ~5-7 weeks of focused Rust work delivers a compositional, auditable, federated, and self-hardening IX pipeline platform.

---

## 10. `target/demo/` as the canonical IX showcase

*Added after R6 — rationale and packaging proposal.*

### 10.1 Inventory of what was produced in a single afternoon

During the conversation that generated this investigation, the following artifacts were produced in `target/demo/` through *manual* orchestration of IX MCP tools:

| File | Size | Purpose | Tools chained |
|---|---|---|---|
| `cost-anomaly-hunter.html` | 9 KB | 90-day AWS billing anomaly detection | stats → fft → kmeans (3) |
| `cyber-intrusion-triage.html` | 19 KB | Encrypted-traffic intrusion triage with Nash defense + governance | 12 tools (stats, fft, chaos, bloom, kmeans, viterbi, markov, linreg, rf, topo, nash, bandit, gov) |
| `catia-bracket-generative.html` | 29 KB | A350 bracket generative design — main dashboard | 13 tools (stats, fft, kmeans, linreg, rf, adam, ga, topo, chaos, nash, viterbi, markov, gov) |
| `catia-bracket-3d.html` | 15 KB | Three.js r160 PBR render of the bracket with stress map | geometric reconstruction |
| `catia-toolpath-3d.html` | 12 KB | Isometric 5-axis toolpath visualization (32 Viterbi waypoints) | from Viterbi output |
| `catia-bracket-context-a350.html` | 17 KB | Schematic context views (A350 side/top/pylon detail) | — |
| `catia-bracket-context-realistic.html` | 41 KB | Engineering-drawing-style 4-plate technical package + Three.js r170 WebGPU + TSL | web-researched real data |
| `rapport-bracket-genetique-a350.md` | 148 KB | 50-page French technical report v1 (21k words) | subagent |
| `rapport-bracket-genetique-a350-v2.md` | 186 KB | 53-page French technical report v2 (26k words, math-normalized, +case studies, +risks, +MCP JSON-RPC annex, +bilingual glossary) | this session |
| `improvement-investigation-ix-v1.md` | this file | R1-R6 investigation + adversarial pipelines + canonical showcase plan | two subagents + synthesis |

**Total**: ~640 KB of artifacts covering four problem domains (FinOps, cybersecurity, aerospace CAD, IX self-improvement) with a consistent style, two languages (FR+EN), four rendering technologies (SVG, Three.js WebGL, Three.js WebGPU+TSL, Markdown+KaTeX), and two report sizes (dashboard-scale + book-scale).

### 10.2 Why this folder deserves to be the canonical IX showcase

Three reasons:

1. **Breadth of applicability without any reconfiguration.** The same 13 tools handled cloud cost anomaly detection, cyber intrusion triage, aerospace topology optimization, and self-documenting reports. Nothing in IX was modified between demos. This is the strongest possible evidence that IX is not a domain-specific toolbox but a general mathematical substrate — exactly what the CLAUDE.md description claims but which is usually hard to demonstrate concretely.

2. **Proof of the compositional gap R1 addresses.** Each demo required Claude to hand-chain MCP calls because `ix_pipeline_run` does not exist yet. The fact that a non-IX developer (a language model) produced working 13-step pipelines by brute-force manual chaining is a backhanded compliment — it proves the tools are useful individually *and* that the ergonomics of composition need work. Future R1 validation should replay these exact demos by submitting a single `.pipeline.yaml` and diffing the outputs against the hand-chained versions — if they match bit-for-bit, R1 is done.

3. **Self-dogfooding loop.** The investigation you're reading now is itself an IX demo: it uses Octopus (multi-AI orchestration), spawns Explore + researcher subagents, synthesizes into a markdown file that follows the same conventions as the bracket report, and its recommendations target the very tooling that produced it. This kind of closed-loop example is exactly what a showcase should contain — it demonstrates that the system is powerful enough to be used to improve itself, not just to solve external problems.

### 10.3 Packaging proposal

Promote `target/demo/` from an ad-hoc scratch folder to a first-class, versioned showcase:

**Step A — Relocate and rename.**
Move the contents from `target/demo/` (which is git-ignored because `target/` is Rust's build cache) to `examples/canonical-showcase/` at the workspace root, so the artifacts become git-tracked and survive `cargo clean`.

**Step B — Add a top-level `README.md`** that:
- Explains what each demo demonstrates (domain, tools chained, why it matters)
- Provides a "how to reproduce" section for each (pipeline YAML after R1 lands; manual MCP call sequence today)
- Links back to the rapports and to the investigation
- Contains a matrix of *tools used × demos* so a reader can see which demos exercise which crates

**Step C — Make the showcase a CI-gated artifact.**
Add a GitHub Actions job that:
- Regenerates every demo on every push to `main`
- Compares generated HTML/MD outputs against golden files stored in the repo
- Fails the build if a demo diverges without explanation
- Publishes the showcase folder as a GitHub Pages site under `ix.github.io/showcase/` so anyone can browse the live dashboards

**Step D — Use the showcase as an onboarding tutorial.**
Restructure the showcase directory as a *progressive tutorial*:
```
examples/canonical-showcase/
├── README.md                          — overview + navigation
├── 01-cost-anomaly-hunter/           — 3 tools, finops, simplest
│   ├── pipeline.yaml                 — after R1
│   ├── dashboard.html
│   └── explainer.md
├── 02-chaos-detective/               — 4 tools, signal processing
├── 03-cyber-intrusion-triage/        — 12 tools, multi-domain
├── 04-catia-bracket-generative/      — 13 tools, aerospace flagship
│   ├── pipeline.yaml
│   ├── dashboard.html                — main SVG dashboard
│   ├── bracket-3d.html               — Three.js render
│   ├── toolpath-3d.html              — Viterbi visualization
│   ├── context-realistic.html        — engineering drawing pack
│   ├── rapport-v1.md
│   ├── rapport-v2.md                 — the 53-page technical report
│   └── README.md                     — how this demo was produced
└── 05-ix-self-improvement/
    ├── improvement-investigation.md   — this file
    └── README.md                      — how to run the investigation
```

This structure teaches a new contributor the IX mental model by traversing demos from 3 tools (trivially composable) to 13 tools (genuinely systemic), with the final example being *the system investigating itself*.

**Step E — Use the showcase as R1–R6 validation harness.**
Each recommendation in this report can be validated by running the showcase:
- **R1** (`ix_pipeline_run`): replay every demo via a single MCP call per demo. Golden-file diff.
- **R2** (assets + cache): second invocation of any demo should be ≥ 10× faster because the asset cache is warm.
- **R3** (registry CI): introduce a deliberate breaking change in the registry; the CI must block it.
- **R4** (gateway): the `catia-bracket-generative` demo should become a one-hop call through the gateway with audit entries emitted to Demerzel.
- **R5** (Arrow side-channel): a synthetic variant of the bracket demo with 10⁶ load cases should complete without JSON payload explosion.
- **R6** (adversarial): launching the Level 1 adversarial validation on the bracket surrogate should discover and patch at least 3 failure modes before convergence.

This makes the showcase both a *marketing asset* (look what IX can do) and a *regression suite* (prove that each improvement preserves what already works).

### 10.4 Immediate next steps (if you agree with this section)

1. `git mv target/demo/ examples/canonical-showcase/` (preserving history via `--follow`).
2. Create `examples/canonical-showcase/README.md` with the tool × demo matrix and reproduction instructions.
3. Add a CI job skeleton that builds the showcase and publishes it to GitHub Pages.
4. Open a tracking issue in the repo with the R1–R6 roadmap and the showcase as the validation harness.
5. Treat the showcase from this point on as a load-bearing artifact of the project — changes to it require the same review discipline as changes to the core crates.

---

*End of investigation v1 + R6 addendum + canonical showcase packaging proposal — 12 April 2026*
*Generated during a single-session exploration of IX. Report will serve as both documentation of recommendations and as the fifth demo in the proposed canonical showcase.*
