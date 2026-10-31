# Preuves fournisseur pour #484 : le mode programmatique de Claude Code, épinglé

Note de recherche pour [#484](https://github.com/GuitarAlchemist/ix/issues/484) (carte
[#483](https://github.com/GuitarAlchemist/ix/issues/483)). Version anglaise :
[`docs/research/agent-composition/2026-10-30-484-claude-code-programmatic-evidence.md`](../../../research/agent-composition/2026-10-30-484-claude-code-programmatic-evidence.md).
Elle consigne des preuves et ne décide rien (l'opérateur décide #484). Aucun appel au modèle n'a été
fait pour l'écrire.

Statuts : **lu** (un document ou du code a été lu), **testé** (une commande a tourné ici),
**indéterminé** (ni l'un ni l'autre ne tranche).

## Version épinglée

- Installée : `2.1.314 (Claude Code)` dans `~/.local/bin/claude.exe` (testé : `claude --version`).
- Aide de cette version : `evidence/claude-2.1.314-help.txt` (testé : `claude --help`, 313 lignes).
- Documentation : <https://code.claude.com/docs/en/headless>, lue le 2026-10-30. La page ne s'épingle
  à aucune version ; elle cite des minimums (v2.1.259, v2.1.286, v2.1.302, ...), tous sous 2.1.314.

## Ce que la version offre à un appelant (aide, testé)

`-p/--print` avec `--output-format text|json|stream-json` ; `--input-format text|stream-json`
(« entrée en flux temps réel », mode print seulement) ; `--include-partial-messages` ;
`--replay-user-messages` (stream-json en entrée et en sortie) ; `--permission-prompts host|none` ;
`--permission-mode` (`acceptEdits`, `auto`, `bypassPermissions`, `manual`, `dontAsk`, `plan`) ;
`--json-schema` ; `--max-budget-usd` ; `--no-session-persistence` ; `--session-id <uuid>` ; `--resume`.

## Ce que dit la documentation de chaque classe d'événements de #484 (lu)

| Classe #484 | Signal documenté | Statut |
|---|---|---|
| acceptation de la requête | aucun : `system/init` dit que la session a démarré, pas qu'une requête est acceptée | indéterminé |
| delta de sortie | `stream_event` avec `event.delta.type == "text_delta"` (demande `--include-partial-messages`, `--verbose`) | lu |
| activité d'outil | messages `assistant` / `user` avec `tool_use` / `tool_result` ; `parent_tool_use_id` rattache les messages d'un sous-agent à son appel | lu |
| attente de permission | avec un hôte, l'exécution l'attend ; `--permission-prompts none` refuse, émet des messages système `permission_denied`, liste `permission_denials` dans le résultat | lu |
| résultat final | la dernière ligne du flux est un message `result` ; `structured_output` est absent si aucune sortie valide n'a été produite | lu |
| fin d'entrée | non spécifiée pour l'entrée stream-json, au-delà de « le travail de fond attend après la fermeture de stdin » | indéterminé |
| terminaison | code 0 succès, non nul échec ; un échec pendant l'exécution est imprimé comme résultat sur stdout ; SIGTERM → code 143, le tour en cours reste inachevé, sans résultat | lu |
| reprises | `system/api_retry` avec `attempt`, `max_retries`, catégorie `error` | lu |

Également documenté : `system/init` porte un tableau `capabilities` (détection de fonctionnalités,
par exemple `interrupt_receipt_v1`) ; stdin redirigé est plafonné à 10 Mo ; depuis v2.1.302 une
exécution `-p` démarre en mode Manual si rien ne fixe de mode ; les coûts du résultat sont des
estimations côté client.

**Absent de la page :** le schéma d'une ligne stream-json *en entrée* et la liste des sous-types de
`result`. Les deux renvoient à la référence TypeScript de l'Agent SDK, pas encore lue. D'ici là, un
consommateur doit traiter un `type`/`subtype` inconnu, une ligne malformée ou tronquée, et un flux
sans ligne `result` comme un non-succès, ce qui est la règle de #484 elle-même.

## Points d'appui qu'IX a déjà (lu, origin/main 99f2f869)

- `crates/ix-loop/src/node.rs`, `ClaudeCode` (un `SemanticNode`) : un processus par demande ;
  `-p --safe-mode --tools "" --no-session-persistence --output-format json --model <m>`, avec
  `--json-schema` et `--system-prompt` optionnels ; prompt sur stdin ; une échéance. Les huit options
  existent dans 2.1.314 (testé contre l'aide enregistrée).
- `crates/ix-baml/src/claude_cli.rs`, la `BamlOperation` d'IXQL : `--print --output-format json
  --json-schema ... --permission-mode manual --strict-mcp-config` ; prompt sur stdin. La doc du
  module consigne une mesure du 2026-07-29 (CLI 2.1.220) : un appel trivial a coûté 0,28 $, soit
  45 840 jetons d'entrée en création de cache pour le prompt système par défaut, les définitions
  d'outils et CLAUDE.md.
- Aucun code de `crates/` ou `go/` n'utilise `--input-format stream-json` ni `--output-format
  stream-json` (testé : `git grep stream-json -- crates go`).

## Ce que cela dit des options de #484

- **Un processus par tâche** existe déjà deux fois (ix-loop, ix-baml), toutes deux en sortie `json` :
  une enveloppe, aucun événement. Cela prouve la fin du processus et la forme de l'enveloppe, pas
  l'achèvement métier.
- **Session réutilisée** (stream-json en entrée et en sortie, plusieurs tours dans un processus) :
  aucune implémentation dans IX. La mesure d'ix-baml est l'argument en sa faveur : le coût fixe par
  appel est le contexte par défaut, payé à nouveau par chaque processus. Non mesuré sur 2.1.314 ;
  l'argument contre est l'état qui fuit d'une tâche à l'autre dans une même session.
- **Adaptateur fondé sur le SDK** : non évalué ici (aucun paquet SDK dans le workspace).
- `--bare` réduirait le contexte fixe mais ne lit pas la connexion OAuth et exige une clé d'API :
  exclu (pas de nouvelle clé, règle de l'opérateur).

## Prochaine étape bornée (hors ligne, sans appel au modèle)

Un test sur fixtures d'un classifieur de lignes stream-json : exécution valide, ligne malformée,
dernière ligne tronquée, `type` inconnu, aucune ligne `result`, `permission_denied`, code 143 sans
résultat. Chacun doit aboutir à une classe de #484 ou à un non-succès. Les fixtures sont écrites à la
main d'après les formes documentées ci-dessus : elles restent étiquetées synthétiques jusqu'à ce
qu'un flux enregistré les remplace, et un flux enregistré demande une exécution live admise à part.

**Fait (testé), toujours synthétique :** `crates/ix-router-spike/src/bin/claude_stream.rs` (binaire
`claude-stream`, 12 tests, chaque règle vue rouge sous un mutant). Ce que l'appelant peut prouver à
partir du stdout d'un processus : lesquels des tours qu'il a envoyés se sont terminés, dans l'ordre,
sur une ligne `result` de sa propre session. Seul `subtype: "success"` avec `is_error: false` est un
succès, et il porte le nombre de `permission_denials`. Tout le reste est un non-succès typé :
`result_error`, `malformed` (toute ligne illisible annule le tour ouvert, même avant un résultat de
succès), `truncated` (une dernière ligne sans saut de ligne, même si elle se lit), `limit_exceeded`,
`terminated` (sortie non nulle avant le résultat, 143 compris), `no_result`. Les lignes d'une autre
session sont comptées `stale` et ignorées ; un résultat qui ne porte pas le `session_id` attendu sous
forme de chaîne est `unattributed_result` et ne clôt aucun tour ; les résultats au-delà des tours
envoyés sont `unexpected_result`. Limites : 1 Mio par ligne, 100 000 lignes, 8 diagnostics d'au plus 200 octets, le
reste compté. L'acceptation de la requête et la fin d'entrée restent non classées : aucun signal
documenté. La corrélation se fait par l'ordre seulement, car aucun champ documenté ne porte l'id de
tâche de l'appelant.
