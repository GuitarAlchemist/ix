# `ix_console` — Poste de navigation IX ASCII à deux panneaux (Mandat `IX-CONSOLE-ASCII` v1)

Prototype local de navigation hiérarchique ASCII inspiré du [Zoomable Icicle](https://observablehq.com/@d3/zoomable-icicle), construit sur `ratatui` et `crossterm` au sein de `crates/ix-dashboard/examples/ix_console.rs`.

## Périmètre et garde-fous (Lot v1)
- **Données synthétiques explicitement étiquetées :** Le flux initial (`feed.rs`) est 100 % déterministe et synthétique (`[SYNTHETIC DEMO FEED — NO LIVE AGENT / NO PROMPT DELIVERY / READ-ONLY PROTOTYPE]`).
- **Aucun terminal Claude interactif hébergé, aucun envoi de prompt réel :** Ce prototype implémente uniquement la navigation UI et l'inspection d'état/preuves/logs bornés.
- **Aucun second serveur WebSocket ni moteur vocal additionnel :** L'adaptateur `action_from_json` est un parseur JSON pur en mémoire destiné aux tests d'équivalence d'actions (Clavier / Intention vocale / JSON), et non un endpoint réseau.

## Architecture des modules
- [`ix_console.rs`](./ix_console.rs) : Point d'entrée CLI (`TUI` interactif, `--snapshot`, `--bench-10k`) et suite de tests comportementaux TDD.
- [`state.rs`](./state.rs) : État de navigation (`ConsoleState`), contrat d'actions unifié (`ConsoleAction`), adaptateurs d'entrée (`action_from_key`, `action_from_voice`, `action_from_json`), calcul de santé (`Silence != Idle` : `Stale` et `Unknown`), résumé de couverture (`CoverageSummary`), et assainissement ASCII des logs (`sanitize_log_line`).
- [`layout.rs`](./layout.rs) : Partitionnement récursif *Zoomable Icicle* à gauche (invariant en profondeur avec fil d'Ariane) et vue de détail focalisée sur un seul nœud à droite (4 dimensions orthogonales : `Depth`, `VisualSpan`, `Progress`, `Quality`).
- [`feed.rs`](./feed.rs) : Fixture synthétique nominale, générateur de rafales de logs et générateur déterministe de **10 000 nœuds** (`build_synthetic_10k_state`).

## Commandes de lancement

### 1. Mode TUI interactif (clavier)
```powershell
cargo run -p ix-dashboard --example ix_console
```
Raccourcis clavier :
- `j` / `k` ou `Down` / `Up` : Sélectionner le nœud visible suivant / précédent
- `Enter` / `Right` / `l` / `z` : Zoomer dans la branche sélectionnée (`ZoomIn`)
- `Backspace` / `Left` / `h` / `b` : Remonter au parent (`Back`)
- `Home` / `0` : Réinitialiser le zoom à la racine (`ResetZoom`)
- `f` : Activer/désactiver le suivi automatique des logs (`ToggleFollowLogs`)
- `PageUp` / `PageDown` : Faire défiler les logs bornés du nœud sélectionné
- `q` / `Esc` / `Ctrl+C` : Quitter

### 2. Aperçu ASCII déterministe dans le terminal (`--snapshot`)
```powershell
cargo run -p ix-dashboard --example ix_console -- --snapshot --width 108 --height 26
cargo run -p ix-dashboard --example ix_console -- --snapshot --width 108 --height 26 --zoom plane.ixctl --select svc.desk.loop
```

### 3. Benchmark de performance sur fixture 10 000 nœuds (`--bench-10k`)
```powershell
cargo run --release -p ix-dashboard --example ix_console -- --bench-10k --iters 200
```

### 4. Tests, lint et formatage
```powershell
cargo test -p ix-dashboard --example ix_console
cargo clippy -p ix-dashboard --all-targets -- -D warnings
cargo fmt -p ix-dashboard -- --check
```
