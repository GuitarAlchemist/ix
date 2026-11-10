# ixmux, fiabilité et préparation du pilote Gaia → MCP → ixmux → Claude (R1)

- **Mandat** : GAIA-20261110-IXMUX-FIABILITE-R1, relayé par Stef le 2026-11-10. Le feu vert de Stef, dans ses propres mots, date du même jour : « Oui, tout le mandat ».
- **Périmètre** : `go/ixmux`, sur la pull request #515 (HEAD de départ `b47f48c3`).
- **Branche** : `feat/ixmux-fiabilite-r1`, commits locaux, ni push ni merge.
- **Verdict** : **READY-FOR-GAIA, avec un blocage**. Le côté local est prêt et testé. Le chemin depuis Gaïa n'est pas prouvé, parce qu'il n'existe pas encore de transport : voir « Blocage ».

Chaque point porte sa source : [opérateur] pour une décision de Stef, [Gaïa] pour une exigence du mandat, [Claude] pour un constat ou un choix de l'exécutant.

## Tranches

| # | Hypothèse ou test rouge | Correction | Vert | SHA |
|---|---|---|---|---|
| 1a | [Gaïa] Risque lu dans le code : l'ouverture refuse plus de 64 Mio, les ajouts ne sont pas bornés. **Confirmé** par le test : 200 mandats dans une borne de 4 Kio ont écrit 100 709 octets, et la réouverture a été refusée `log-too-large`. | Aucun ajout ne dépasse la borne lue à l'ouverture. Un nouveau mandat est refusé `log-full` s'il laisserait moins que le dernier huitième, réservé aux enregistrements restants des livraisons acceptées. | `TestALogFilledByItsOwnAppendsStillOpens` | `1e15c56c` |
| 1b | [Claude] Avec 1a seul, un journal plein reste plein : tout nouveau mandat est refusé pour toujours. Rouge : aucun mandat ne passe après le redémarrage. | Compaction à l'ouverture, quand le journal dépasse les 3/4 de la borne et que la compaction le réduit : un enregistrement par mandat (état, empreinte, incarnation, Ack, raison, cause, état précédent, date de création), écrit dans `.compact`, synchronisé, puis renommé, et le dossier synchronisé. Le journal n'est jamais supprimé. | `TestARestartCompactsAFullLogAndForgetsNothing`, qui relit le journal compacté avec un second moteur. 8 mutations attrapées. | `5f33f5a0` |
| 2a | [Gaïa] Timeout, erreur et saturation distingués. Rouge : à travers MCP, deadline, erreur et redémarrage donnent tous `uncertain`, sans autre information. | `cause` typée, dans un ensemble fermé : `deadline`, `adapter-error`, `restart`, `log-write-failed`. Elle est écrite dans le journal et conservée par la compaction ; MCP ne la transmet que si sa valeur est connue. La saturation était déjà un refus typé (`target-busy`, `busy`). | `TestMCPSaysWhyAMandateIsUncertain`, `TestMCPIxmuxKilledDuringATurn` (cause `restart`), `TestNoFreeTextFromTheNetwork` (une cause hostile est écartée ; la mutation est attrapée) | `e4c5df5b` |
| 2b | [Gaïa] Le Job Object Windows ne prouve rien sous Linux. Constat [Claude] : aucun test hors Windows, et l'hypothèse est marquée `[U:uncertain]` dans `job_unix.go`. | Un test mesure la garantie unix, plus faible : l'enfant survit à un ixmux tué tant que son tour dure, et meurt quand le tour finit. Le redémarrage ne rejoue rien. Le nettoyage du test tue un éventuel orphelin. | `TestClaudeKillIxmuxOrphansTheChildUntilItsTurnEnds`, vert sous Linux avec `-race` | `e4c5df5b` |
| 2c | [Claude] `log-full` était absent des codes connus de l'adaptateur MCP : il serait remonté comme `ixmux-error`. | Code ajouté. Doc mise à jour en anglais et en français. | `TestOnlyKnownCodesPass/log-full` | `dca7d33d` |
| 3a | [Claude] Le `Recovery` de l'ouverture était ignoré : l'opérateur ne voyait ni enregistrement tronqué, ni livraison rendue incertaine, ni compaction. | Une ligne sur stderr au démarrage. | `TestMCPIxmuxKilledDuringATurn` | `aafd9f72` |
| R1 | [Claude] Revue de `5f33f5a0`, P1-a : un enregistrement final qui ne tient pas était refusé `log-full`, c'est-à-dire « rien n'a été envoyé », alors que la cible avait déjà reçu le texte. | Seul un nouveau mandat peut être refusé `log-full`. Un enregistrement final qui ne tient pas laisse la livraison `uncertain`, cause `log-write-failed`, sans refus. | `TestAFinalRecordThatDoesNotFitIsNoRefusal` | `6ba876c1` |
| R2 | [Claude] Revue, P1-b : si les transitions de redémarrage ne tenaient pas dans un journal plein, l'ouverture échouait et ixmux ne démarrait plus. | Ces transitions sont appliquées en mémoire seulement : le journal dit déjà ce qui les produit, puisqu'un `delivering` redevient incertain à chaque ouverture. La ligne de démarrage le signale. | `TestARestartWhoseTransitionsDoNotFitStillStarts` | `6ba876c1` |
| R3 | [Claude] Revue, P2 : une compaction qui ne réduirait pas le journal est sautée, et un `.compact` laissé par un crash est supprimé à l'ouverture. Côté HTTP, une erreur qui n'est pas un refus donne 500 `internal`, que l'adaptateur MCP traite comme une indisponibilité. | Corrections | `TestACompactionThatWouldNotShrinkIsSkipped`, `TestALeftoverCompactFileIsRemoved`, `TestReceiptAfterACutPOST` | `6ba876c1` |
| R4 | [Claude] La contre-vérification a trouvé une annotation `@ai:invariant` devenue fausse sur `Close` : une transition refusée faute de place ne casse pas le journal, donc `Close` renvoie nil. | Annotation corrigée et liée au test qui le prouve. Le code est inchangé. | `TestAFinalRecordThatDoesNotFitIsNoRefusal` | `d884ea09` |

Les exigences déjà couvertes avant ce mandat ont été vérifiées et non recréées [Claude] :
- dépôt et résultat corrélés, doublon sans second tour : `TestMCPMandates` ;
- déconnexion et reprise : `TestMCPAdapterKilledDuringATurn`, `TestReceiptAfterACutPOST`, `TestMCPWithoutIxmux` ;
- crash pendant un tour, état explicite et pas de rejeu : `TestMCPIxmuxKilledDuringATurn`.

## Validation

| Où | SHA | Commande | Résultat |
|---|---|---|---|
| Windows 11, Go 1.27.1 | `6ba876c1`, puis `d884ea09` | `gofmt -l .`, `go vet ./...`, `go test -count=1 ./...` (dans `go/ixmux`) | 5 paquets ok |
| Ubuntu (WSL2, noyau 6.18), Go 1.27.1 | `e4c5df5b`, `6ba876c1`, puis `d884ea09` | `go vet ./... && go test -race -count=1 ./...` | 5 paquets ok, tests unix compris |
| CI de la pull request | — | non lancée : pas de push sans autorisation | — |

- `-race` ne tourne pas sous Windows ici (pas de cgo) ; il a tourné sous Linux.
- Après chaque suite, aucun processus de test n'est resté : contrôle par chemin d'exécutable, le faux Claude s'appelant `claude.exe`.

**Revue indépendante** (persistance), par un agent séparé qui avait le code et non mes conclusions [Claude] :
- Sur `5f33f5a0`, 0 P0, 3 P1 et 6 P2. P1-a et P1-b sont corrigés (lignes R1 et R2), les P2 utiles aussi (R3). P1-c est laissé en limite : voir « Limites ».
- La contre-vérification de `6ba876c1` confirme que P1-b est corrigé et que P1-a l'est en partie. Elle trouve l'annotation fausse (corrigée en R4) et quatre P2 restants, listés dans « Limites ». Aucun P0.

## Mode opératoire de l'essai Gaïa

### Binaire

```
cd go/ixmux
go build -trimpath -ldflags "-X main.version=$(git describe --always --dirty --abbrev=8)" -o ixmux.exe ./cmd/ixmux
```

Le build est reproductible : deux constructions au SHA de code final `d884ea09` donnent le même `sha256 b64c2c1ea259f11d1ca684ca86b2d3ac516027ee6be9590e5a4b54b1758995e2` pour `v0.5.0-772-gd884ea09` (Go 1.27.1, windows/amd64). Le commit de ce rapport ne touche que la doc ; construit après lui, le binaire porte une autre version, donc un autre hash. `ixmux version` affiche la version, et `ixmux.json` contient le SHA-256 de l'exécutable en cours.

### Commandes

```
ixmux.exe serve --addr 127.0.0.1:0 --state <état> --claude-target c
ixmux.exe mcp --state <état>
```

- [Gaïa] **Claude sans outils** : c'est imposé par le code (`claudestream.go:53-58`) avec `--tools ""`, `--strict-mcp-config` sans `--mcp-config` et le mode de permission par défaut. Un init qui liste des outils fait refuser toute livraison.
- **Outils MCP exposés** : `ixmux_status`, `ixmux_submit`, `ixmux_result`. Aucun outil n'arrête ixmux ni ne donne accès à un shell.

### Scénario

Exécuté le 2026-11-10 avec le faux Claude du dépôt, sans aucun modèle et en loopback uniquement. Le scénario complet a tourné sur `aafd9f72`. Sur le binaire final `d884ea09`, les étapes 1 à 4 et 6 ont été rejouées : dépôt livré, rejeu `replayed: true`, kill, ligne `3 records replayed, 0 made uncertain`, résultat relu avec son Ack, et jeton absent des sorties. L'étape 5 est couverte sur ce SHA par `TestMCPIxmuxKilledDuringATurn`.

1. **Hello** : `initialize` répond le protocole `2025-06-18`, et `serverInfo` donne la version. `ixmux_submit` sur `pilot-hello-1` répond `accepted: true, state: delivered`.
2. **Résultat** : `ixmux_result` renvoie `state: delivered, ack_seen: true` et le texte de la réponse.
3. **Rejeu** : le même dépôt répond `replayed: true`, sans nouveau tour.
4. **Reconnexion** : après le kill d'ixmux et son redémarrage sur le même état, stderr affiche `command log: 3 records replayed, 0 made uncertain`, et un nouvel adaptateur relit le résultat.
5. **Crash pendant un tour** : un tour retenu, puis le kill d'ixmux. Au redémarrage, stderr affiche `1 made uncertain`, et le mandat est `uncertain` avec la cause `restart`. Le renvoyer ne déclenche pas de second tour.
6. Le jeton n'apparaît dans aucune sortie (vérifié avec `grep`).

C'est une preuve **locale**. [Gaïa] Ce n'est pas une preuve depuis Gaïa.

### Diagnostic

| Ce que voit Gaïa | Sens | Action |
|---|---|---|
| `ixmux-unavailable` | ixmux est arrêté, ou `ixmux.json` ou le jeton manquent | Relancer `serve`, puis lire sa ligne `command log:` |
| `ixmux-unverified` | Le port est tenu par un autre processus que ixmux | Ne rien envoyer ; trouver le processus |
| `uncertain` et `cause: deadline` | Pas de réponse dans le délai de livraison ; `still_running` dit si le tour court encore | Attendre, puis lire `ixmux_result` ; ne jamais renvoyer |
| `uncertain` et `cause: adapter-error` | La cible a répondu une erreur (`is_error`) après avoir peut-être pris le mandat | **Ne rien relancer**, ni sous le même id ni sous un autre. Lire la sortie de la session ; une personne décide |
| `uncertain` et `cause: restart` | ixmux a redémarré pendant la livraison | **Ne rien relancer**. Réconcilier hors MCP (souvent impossible avec claude : nouvelle incarnation) ; une personne décide |

Correction R2 (mandat GAIA-20261110-IXMUX-P1-R2) : cette table conseillait « nouveau mandat sous un nouvel id » pour `adapter-error` et `restart`. C'était une relance déguisée : `uncertain` veut dire que la cible a peut-être pris le mandat, et rien ne le relance automatiquement.
| `target-busy` ou `busy` | Saturation : rien n'a été envoyé | Réessayer plus tard, avec le même id |
| `submits-in-flight` | 4 dépôts attendent encore une réponse | Réessayer quand l'un d'eux a fini |
| `log-full` | Journal de commandes plein ; depuis R2, ixmux l'a déjà compacté sans succès : les mandats terminés dans la fenêtre de déduplication le remplissent | Rien n'a été envoyé. Réessayer plus tard, quand d'anciens mandats sortent de la fenêtre, ou relancer `serve` avec un `--dedup-window` plus court ; **ne jamais supprimer le journal** |
| `mandate-id-reuse` | Même id avec un autre texte ou une autre incarnation | Après un redémarrage, redonner l'incarnation d'origine |

### Arrêt et retour arrière

- **Arrêt propre** : Ctrl+C dans la console de `serve`, ou `POST /v1/shutdown`. Code de sortie 0, ou 4 si des appels sont encore en cours ; dans ce cas, le journal et son verrou sont gardés.
- **Retour arrière** vers le binaire `b47f48c3`, vérifié le 2026-11-10. L'ancien binaire relit un journal écrit par le nouveau, qui contient `cause`. Le mandat livré garde son Ack, le mandat incertain reste incertain, et un renvoi ne fait pas de second tour. Les champs ajoutés (`cause`, `prev`, `created`) sont ignorés par l'ancien binaire.
- **Avant tout essai**, copier le dossier d'état pendant que ixmux est arrêté.

## Blocage (READY-FOR-GAIA)

[Claude] `ixmux mcp` parle MCP sur **stdio**, et `serve` n'écoute qu'en **loopback**. Une conversation Gaïa hébergée n'atteint un MCP que par HTTPS public (streamable HTTP ou SSE). C'est la doc OpenAI, lue le 2026-10-17 et non revérifiée depuis. Le chemin d'entrée cloud n'est pas établi : la dernière case de l'issue #480 n'est pas cochée. [Gaïa] Ce mandat interdit de créer un tunnel concurrent de celui de Brutal.

Deux branchements possibles, à trancher par l'opérateur :
1. **Le tunnel de Brutal** expose un pont streamable-HTTP vers `ixmux mcp --state <état>`. C'est un travail du chantier tunnel.
2. **Un fil d'exécution local de Gaïa** (app de bureau ChatGPT, qui lit `~/.codex/config.toml`) avec une entrée `[mcp_servers.ixmux]` en stdio. Il faut modifier la config et redémarrer l'app, ce qui coupe l'appel. Le fil parent de Gaïa est hébergé, ce qui n'est pas prouvé suffisant.

## Limites

- La compaction n'a lieu **qu'au démarrage**. Un serveur qui tourne longtemps refuse `log-full` jusqu'à son prochain redémarrage.
- **P1-c (rétention)** : si les mandats distincts remplissent à eux seuls la borne, la compaction ne libère rien, car la mémoire anti-doublon n'est jamais effacée. Avec 64 Mio, cela arrive vers 13 000 mandats ; ixmux refuse alors tout nouveau mandat `log-full`. Il faudrait relever `MaxLogBytes`, qui n'est pas une option de la ligne de commande aujourd'hui, ou décider d'une politique de rétention. C'est une décision de l'opérateur, et c'est sans effet pour un pilote.
- **P1-a, résidu** : la réserve du dernier huitième est partagée entre toutes les livraisons acceptées. Beaucoup de livraisons simultanées sur un journal presque plein peuvent l'épuiser : l'Ack est alors perdu et la livraison reste `uncertain`, cause `log-write-failed`. Rien n'est renvoyé, et l'état reste honnête. De plus, un refus `log-broken` qui survient après l'appel de l'adaptateur existait avant ce mandat ; il n'a pas été traité.
- Un `Reconcile` sur un journal plein répond 500 et perd la réponse de la cible : le mandat reste `uncertain`. Un enregistrement `delivering` qui ne tient pas répond aussi 500, alors que rien n'a été envoyé ; l'adaptateur MCP le lit comme une indisponibilité.
- Une compaction sautée parce qu'elle ne réduirait pas le journal n'apparaît pas dans la ligne de démarrage.
- Sous Windows, la durabilité du renommage (`MoveFileEx` avec `WRITE_THROUGH`) reste une hypothèse, annotée `[U:uncertain]` : elle n'est pas testable sans couper le courant.
- Si `MaxLogBytes` est abaissé entre deux exécutions sous la taille du journal existant, l'ouverture le refuse (P2-3, non traité ; ce n'est pas une option de la ligne de commande).
- Le journal compacté ne reçoit pas de droits 0600 explicites ; il hérite des droits du dossier d'état (P2-6).
- Sous unix, un enfant Claude survit à un ixmux tué jusqu'à la fin de son tour. Sous Windows, le Job Object le tue.
- Réconcilier un mandat incertain n'est pas un outil MCP (`POST /v1/mandates/{id}/reconcile`).
- La cause n'existe que pour l'état `uncertain`. Les causes de `refused` restent du texte libre, non transmis.
- Aucun appel à un vrai modèle dans ce mandat, donc aucun budget consommé.

## Décisions réservées à l'opérateur

1. Pousser `feat/ixmux-fiabilite-r1`, le fusionner dans `feat/ixmux` (#515), ou en faire une pull request séparée.
2. Le branchement de l'essai : tunnel de Brutal, ou config locale de Codex.
3. Le premier essai réel avec Claude, qui consomme des tokens : le budget est à confirmer.
4. La politique de rétention du journal (P1-c), si ixmux doit vivre au-delà du pilote.

Porte : à double sens. Le format du journal reste `ixmux.cmdlog.v1`, l'ancien binaire relit le nouveau journal et les champs sont additifs. À revoir si une compaction échoue en essai, ou au premier `log-full` réel.
