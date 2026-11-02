# `ix_console` — Poste de navigation IX ASCII à deux panneaux (Mandat `IX-CONSOLE-ASCII` v1 / Revue `IX-CONSOLE-REVIEW` v1)

Prototype local de navigation hiérarchique ASCII inspiré du [Zoomable Icicle](https://observablehq.com/@d3/zoomable-icicle), construit sur `ratatui` et `crossterm` au sein de `crates/ix-dashboard/examples/ix_console.rs`.

## Périmètre et garde-fous
- **Données synthétiques explicitement étiquetées :** Le flux initial (`feed.rs`) est 100 % déterministe et synthétique (`[SYNTHETIC DEMO FEED - NO LIVE AGENT / NO PROMPT DELIVERY / READ-ONLY PROTOTYPE]`).
- **Aucun terminal Claude interactif hébergé, aucun envoi de prompt réel :** Ce prototype implémente uniquement la navigation UI et l'inspection d'état/preuves/logs bornés.
- **Aucun second serveur WebSocket ni moteur vocal additionnel :** L'adaptateur `action_from_json` est un parseur JSON pur en mémoire destiné aux tests d'équivalence d'actions (Clavier / Intention vocale / JSON), et non un endpoint réseau.

## Garanties de fiabilité (Phase 1 & Phase 2 — `IX-CONSOLE-REVIEW` v1)
1. **Navigation sans plafond à 64 nœuds & fenêtre icicle glissante (`P2-1`) :** Navigation pré-ordre sans troncature à 64 nœuds (`step_visible_selection`) et fenêtre icicle glissante incluant toujours l'enfant sélectionné ou l'ancêtre de la sélection (`hidden_siblings_before` / `hidden_siblings_after`).
2. **Ancrage de lecture par séquence (`P2-2`) :** Lorsque `Follow` est `OFF`, la position de lecture est ancrée sur `log_anchor_seq` ; l'éviction d'anciennes lignes ne décale jamais la première séquence affichée tant qu'elle existe, et son éviction éventuelle est signalée explicitement (`[ANCHOR EVICTED: seq #XXXX was evicted from ring]`).
3. **Poids hiérarchiques et profondeurs exacts (`P2-3`) :** Recalcul topologique/post-ordre exact lors du rattachement d'orphelins (enfants insérés avant leur parent) ou du déplacement d'une branche peuplée.
4. **Validation d'arbre sans cycles ni troncature à 256 niveaux (`P2-4`) :** Rejet explicite des cycles directs/indirects (`FeedRejectReason::CycleDetected`) et des profondeurs excédant `max_tree_depth` (`FeedRejectReason::MaxDepthExceeded`) sans altérer l'arbre valide ; support vérifié des chaînes de 300+ niveaux.
5. **Isolation des incarnations face aux événements tardifs (`P2-5`) :** Identification de chaque log par son incarnation (`[inc]`) et rejet des événements tardifs (`RemoveNodeForIncarnation`, `RecordGapForIncarnation`) issus d'une incarnation remplacée (`FeedRejectReason::StaleIncarnation`).
6. **Restauration garantie du terminal (`P2-6`) :** Garde RAII `TerminalSessionGuard` restaurant systématiquement l'écran alternatif et le mode brut en cas d'échec partiel d'initialisation ou de panic (`unwind`).
7. **Filtrage clavier Windows & bornage en octets (`Phase 2`) :** Ignorance de `KeyEventKind::Release` (et `Repeat` sur les bascules `f`/`q`/zoom) et bornage strict en octets après expansion d'échappement (`max_log_line_bytes`, `max_node_log_bytes`, `max_total_log_bytes`).

## Architecture des modules
- [`ix_console.rs`](./ix_console.rs) : Point d'entrée CLI (`TUI` interactif, `--snapshot`, `--bench-10k`) et suite de tests comportementaux TDD.
- [`state.rs`](./state.rs) : État de navigation (`ConsoleState`), contrat d'actions unifié (`ConsoleAction`), adaptateurs d'entrée (`action_from_key`, `action_from_voice`, `action_from_json`), calcul de santé (`Silence != Idle` : `Stale` et `Unknown`), résumé de couverture (`CoverageSummary`), et assainissement ASCII des logs (`sanitize_log_line`).
- [`layout.rs`](./layout.rs) : Panneaux gauche et droit à parts égales (50/50) ; partitionnement récursif *Zoomable Icicle* à gauche, ses colonnes partagées à égalité entre les niveaux affichés (invariant en profondeur avec fil d'Ariane) et vue de détail focalisée sur un seul nœud à droite (4 dimensions orthogonales : `Depth`, `VisualSpan`, `Progress`, `Quality`).
- [`feed.rs`](./feed.rs) : Fixture synthétique nominale, générateur de rafales de logs et générateur déterministe de **10 000 nœuds** (`build_synthetic_10k_state`).

## Commandes de lancement

### 1. Mode TUI interactif (clavier)
```powershell
cargo run -p ix-dashboard --example ix_console
```
Raccourcis clavier :
- `j` / `k` ou `Down` / `Up` : Sélectionner le nœud visible suivant / précédent
- `Right` / `l` : Sélectionner le premier enfant (`SelectChild`) ; le zoom descend seulement si l'enfant sortirait des quatre niveaux affichés
- `Left` / `h` : Sélectionner le parent (`SelectParent`) ; quitter la racine du zoom par le haut dézoome d'un niveau
- `Enter` / `z` : Zoomer dans la branche sélectionnée (`ZoomIn`)
- `Backspace` / `b` : Remonter le zoom au parent (`Back`)
- `Home` / `0` : Réinitialiser le zoom à la racine (`ResetZoom`)
- `f` : Activer/désactiver le suivi automatique des logs (`ToggleFollowLogs`)
- `PageUp` / `PageDown` : Faire défiler les logs bornés du nœud sélectionné
- `q` / `Esc` / `Ctrl+C` : Quitter

### 2. Aperçu ASCII déterministe dans le terminal (`--snapshot`)
```powershell
cargo run -p ix-dashboard --example ix_console -- --snapshot --width 104 --height 24
cargo run -p ix-dashboard --example ix_console -- --snapshot --width 104 --height 24 --zoom plane.ixctl --select svc.desk.loop
```

### 3. Benchmark de rendu `Buffer` en mémoire sur fixture 10 000 nœuds (`--bench-10k`)
```powershell
cargo run -p ix-dashboard --example ix_console -- --bench-10k --iters 200
```

### 4. Tests, lint et formatage
```powershell
cargo test -p ix-dashboard --example ix_console
cargo clippy -p ix-dashboard --all-targets -- -D warnings
cargo fmt -p ix-dashboard -- --check
```
