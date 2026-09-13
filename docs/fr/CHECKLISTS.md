# Listes de contrôle par type de changement

> **Avant d'ouvrir ou de fusionner une PR, exécutez :**
>
> ```bash
> cargo run -p ix-skill --bin ix -- doctor
> ```
>
> Ajoutez `--full` pour lancer aussi l'invocation clippy et test utilisée par la
> CI. `ix doctor` sort avec `0` si tout est vert, `1` en cas d'avertissements
> seulement, `4` en cas d'échec, et indique quoi faire pour chaque cas.
> Version anglaise : [`docs/CHECKLISTS.md`](../CHECKLISTS.md).

IX comporte de nombreuses coutures : un registre de capacités résolu à l'édition
de liens, une surface d'outils MCP, des UDF DuckDB, des empreintes
stable-surface, des annotations `@ai:`, une documentation EN/FR, des instantanés
de catalogue. Chacune est peu coûteuse isolément et coûteuse à *mémoriser*. Ces
listes encodent les coutures pour que vous n'ayez pas à les redécouvrir en
cassant la CI (ix#185).

Chaque liste ci-dessous est volontairement courte. Si une étape peut être
mécanisée, sa place est dans `ix doctor`, pas ici.

---

## Ce que vérifie `ix doctor`

| Vérification | Ce qui la fait échouer | Correction |
| --- | --- | --- |
| `registry-snapshot` | Les skills ou outils MCP du binaire lié diffèrent de `state/registry/skills.snapshot.json` | `ix doctor --write`, puis relire le diff de **noms** |
| `orphan-traits` | Un `pub trait` n'a ni implémenteur, ni borne générique, ni entrée d'allowlist | Le supprimer, l'implémenter, ou l'inscrire dans `state/registry/orphan-traits.allow.json` avec un `reason` |
| `dark-features` | Une feature cargo qu'aucun membre de l'espace de travail n'active protège 100 lignes ou plus, ou au moins un test, sans entrée d'allowlist | L'activer depuis un membre de l'espace de travail, supprimer le code, ou l'inscrire dans `state/registry/dark-features.allow.json` avec un `reason` |
| `demerzel-governance` | `governance/demerzel` est absent | `git submodule update --init` |
| `default-constitution` | Le sous-module est présent mais incomplet | Resynchroniser le sous-module |
| `state-directory` | `state/` manque à la racine du dépôt | Vous n'êtes pas à la racine de l'espace de travail |

`ix check doctor` reste un alias de la même exécution.

### Ce qu'il ne vérifie délibérément pas

`ix doctor` est un raccourci pratique, pas une nouvelle couche d'autorité. Il
n'exécute ni les démos pipeline-mesh, ni Epistemic SQL, ni l'assumption-graph :
conformément à ix#185, ceux-ci restent optionnels et ne doivent jamais devenir
des barrières obligatoires pour une PR ordinaire.

---

## Ajouter un skill / outil MCP

1. Écrivez l'algorithme dans son crate de domaine. Annotez l'enveloppe avec
   `#[ix_skill(...)]` pour que le registre résolu à l'édition de liens le capte.
2. Si le crate est nouveau, ajoutez-le à `ix_skill::force_link` dans
   `crates/ix-skill/src/lib.rs` — sinon la LTO supprime l'entrée de la
   distributed-slice et le registre se retrouve silencieusement incomplet.
3. Ajoutez le nom de l'outil à `EXPECTED` dans
   `crates/ix-agent/tests/parity.rs`. Cette liste reste maintenue à la main
   volontairement : c'est le limiteur de débit qui force la relecture de chaque
   changement de surface.
4. Lancez `ix doctor --write` et validez le diff de `skills.snapshot.json`. Le
   diff doit montrer exactement les noms voulus, et rien d'autre.
5. Si l'outil est derrière une feature cargo non par défaut, ajoutez son nom à
   `feature_gated_tools` dans l'instantané pour que les deux configurations de
   compilation restent vertes.
6. Lancez `cargo test -p ix-agent --test parity`.
7. Documentez-le : `docs/MANUAL.md`, ainsi que la traduction française sous
   `docs/fr/`.

## Ajouter une UDF DuckDB

1. Implémentez-la dans `ix-duck` / `ix-duck-ext` et enregistrez-la dans la
   fonction d'enregistrement du crate — une UDF écrite mais non enregistrée est
   invisible.
2. Ajoutez un test qui exécute l'UDF via une vraie connexion DuckDB, pas
   seulement la fonction Rust sous-jacente. C'est l'enregistrement qui casse.
3. Conservez le style compact propre à `ix-duck` ; ne lancez **pas** `cargo fmt`
   dessus (voir `CLAUDE.md` — l'écart de formatage à l'échelle du dépôt est
   consultatif, pas bloquant).
4. Si l'UDF est aussi exposée comme outil MCP, suivez également la liste MCP
   ci-dessus.
5. Notez le drapeau de feature. `maintain-gate` embarque DuckDB et est désactivé
   dans la compilation par défaut / CI ; tout ce qui en dépend ne doit pas
   modifier la surface par défaut.
6. Documentez la signature dans `docs/DUCKDB.md` et sa contrepartie française.

## Ajouter ou modifier une API publique de crate

1. Vérifiez le palier du crate dans `crate-maturity.toml`. Les crates **Stable**
   sont sous contrôle.
2. Lancez `cargo run -p ix-skill --bin ix -- stable-surface`. La barrière hache
   les lignes préfixées par `pub `, donc ajouter un `pub fn` la déclenche alors
   que modifier un corps de fonction ne la déclenche pas.
3. Si l'empreinte d'un crate Stable change, la PR exige une montée de version
   explicite ou une rétrogradation de palier — pas une mise à jour silencieuse
   de l'empreinte.
4. Un nouveau `pub trait` ? Il lui faut un implémenteur ou une borne générique
   dans l'arbre, sinon `ix doctor` fera échouer la vérification
   `orphan-traits`. C'est bien l'objectif : un contrat déclaré que rien ne
   satisfait est soit du code mort, soit une promesse que le code ne tient pas.
   `DataSource` / `DataSink` de `ix-io` en étaient le cas fondateur (ix#299) :
   la doc du module affirmait que tous les backends les implémentaient alors
   qu'aucun ne le faisait. Le correctif mérite d'être imité — des
   implémenteurs *et* un consommateur générique (`protocol::pump`), plus une
   raison écrite dans chaque module qui n'implémente toujours pas le trait.

## Ajouter un module derrière une feature

1. Demandez-vous d'abord si la barrière est nécessaire. C'est le bon outil pour
   une dépendance lourde optionnelle — DuckDB, arrow, ONNX Runtime — et le
   `Cargo.toml` racine le dit. C'est le mauvais outil pour « ce n'est pas encore
   prêt » : un `#[cfg]` n'est pas un marqueur de brouillon.
2. Si vous posez la barrière, quelque chose doit tout de même compiler le code.
   Une feature qu'**aucun membre de l'espace de travail n'active** n'est jamais
   construite par `cargo build --workspace`, `cargo clippy --workspace` ni
   `cargo test --workspace` — les trois invocations de la CI. Ses tests ne sont
   ni réussis ni échoués : ils sont absents, et un crate ignoré ajoute zéro aux
   deux compteurs, donc rien dans la sortie du dépôt ne vous le dit (ix#315).
3. `ix doctor` fait échouer la vérification `dark-features` pour toute feature de
   ce type protégeant 100 lignes ou plus, ou au moins un test. Elle indique le
   crate, la feature, les modules, et combien de lignes et de tests se trouvent
   derrière — une barrière qui cache deux lignes est du bruit, une qui en cache
   95 tests ne l'est pas.
4. Le correctif, par ordre de préférence : activer la feature depuis un membre de
   l'espace de travail qui en a besoin (une dev-dependency compte) ; supprimer le
   code ; ou ajouter une entrée à `state/registry/dark-features.allow.json`. Une
   entrée exige un `reason` non vide et un `kind` :
   - `environment` — l'activer ici est bloqué ou déraisonnablement coûteux pour
     des raisons extérieures au code (chaîne d'outils native, binaire externe,
     GPU, chaîne d'outils plus récente que la MSRV de l'espace de travail).
     Destinée à rester.
   - `tracked` — rien d'environnemental ne l'empêche, ce n'est simplement pas
     encore branché. **Exige un `issue`**, et est comptée comme dette dans le
     résumé pour ne pas devenir un parking.
5. La feature `topology` de `ix-code` est le cas motivant : 296 lignes et cinq
   tests, fusionnés et relus, jamais compilés une seule fois. Le code était sain
   — `cargo test -p ix-code --features full` passe — et c'est précisément pour
   cela que personne ne l'a remarqué.

## Ajouter un invariant `@ai:`

1. Suivez `docs/contracts/2026-05-24-ai-annotation.contract.md`. Format :
   `// @ai:invariant <affirmation> [T:test conf:0.95 src:chemin::vers::test]`.
2. `certainty := force du lien vivant`. N'écrivez pas `[T]` sans lien vivant
   (un test, le compilateur, ou sentrux) ; plafonnez les affirmations purement
   humaines à `P:assumed` ; exposez les hypothèses non vérifiées avec
   `@ai:assumption [U:uncertain]`.
3. Gardez `src:` sous forme d'un jeton unique et propre — la barrière de dérive
   ne lie `[T:test]` que si `src:` est un chemin `::` ou préfixé par `test_`.
4. Annotez un module à la fois, et faites en sorte que chaque passe produise au
   moins une correction réelle. Ne générez jamais d'annotations en masse.

## Ajouter une vitrine ou une démo

1. Enregistrez le scénario pour que `ix demo list` le trouve — un scénario non
   enregistré est invisible, quelle que soit sa qualité d'exécution.
2. Fixez la graine de chaque RNG. Les démos sont des surfaces de
   reproductibilité ; une démo sans graine produit une transcription différente
   à chaque exécution et ne peut pas être comparée.
3. Gardez-la hors du chemin de test par défaut si elle est lente. Conformément à
   ix#185, l'itération Rust normale ne doit pas être bloquée derrière des démos
   lourdes.

## Modifier une documentation qui alimente Streeling ou un catalogue

1. La terminologie va dans `CONTEXT.md`. Les décisions d'architecture vont dans
   `docs/adr/`. Les enseignements de session vont dans `docs/solutions/` via
   `/learnings`. Ne laissez pas de markdown non indexé à la racine du dépôt.
2. Relancez l'indexeur `ix-streeling` pour que `state/streeling/catalog.jsonl`
   reste à jour.
3. Maintenez la traduction française à côté de la version anglaise, sous
   `docs/fr/`.
