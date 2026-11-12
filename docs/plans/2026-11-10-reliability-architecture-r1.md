# Écosystème d'agents récupérable et observable, R1 : carte, tranche d'inventaire et limites

- **Mandat** : GAIA-INDIGO-RELIABILITY-ARCHITECTURE-R1, relayé par Stef. Stef a donné son feu vert le 2026-11-10, dans ses propres mots, par AskUserQuestion : « Oui, tout le mandat ».
- **Base** : `5911f37c` (rapport R2 d'ixmux). Branche locale `feat/ixmux-inventory-r1`. Ni push ni merge.
- **Commits** :
  - `682e446f` : un arrêt propre en plein tour donne la cause `restart`, plus `adapter-error` ;
  - `ecbddbde` : l'inventaire en lecture seule ;
  - `94de6b8e` : les corrections de la revue contradictoire (section 7) ;
  - `03d916b0` : le contrôle de fuite du pid, rouge sous Linux seulement (section 7) ;
  - le commit de ce rapport.
- **Propriétaires respectés** [Claude] :
  - le raccord tunnel→WMUX est à Brutal ;
  - les services et les hooks sont à IX Service Windows (`ix-winsvc-7d`), qui porte aussi `ixctl resume check` et la sonde `claude-sessions@1` d'ixctl ;
  - la passerelle Delta Remote est à `brickyard-remote-91` ;
  - la PWA est à Scribe ;
  - la migration .NET 11 est à GA ;
  - le mandat de qualification cloud d'ixmux est à `ix-ixmux-38` (session `1b129d7c`). Elle a confirmé qu'elle n'écrit aucun code produit d'ixmux.

  Je n'ai écrit que dans `go/ixmux`, où je tiens déjà R1 et R2.
- **Verdict** : voir la fin du document.

Chaque point porte sa source :
- [opérateur] : une décision de Stef ;
- [Gaïa] : une exigence du mandat ;
- [Claude] : un constat ou un choix de l'exécutant.

Les preuves sont marquées ainsi :
- **PROUVÉ** : par un test, du code lu ou une commande lancée le 2026-11-10 ;
- **SUPPOSÉ** : par la documentation seulement ;
- **MANQUE** : rien ne le fournit.

## 1. Carte des contrats

### GoMux et ixmux

GoMux est l'ancien nom d'ixmux. L'issue #511 le dit : le 2026-11-05, l'opérateur a renommé « ixmux » le prototype GoMux V2. Ce prototype était un backend Go HTTP/WebSocket qui multiplexe des sessions et lit des journaux JSONL par offset.

GoMux V2 n'a vécu que dans un scratchpad, sans révision publiée. ixmux en est la reprise, dans `go/ixmux` : branche `feat/ixmux` (pull request #515, brouillon), puis R1, R2 et cette R1 d'architecture. Ce ne sont donc pas deux composants à départager : il n'y en a qu'un. ixmux n'est ni WMUX ni un relais.

Le blocage de l'issue #511, une reprise qui s'arrête sur une ligne trop longue, est corrigé dans le code et couvert par un test :
- une ligne au-delà de 1 Mio devient un événement typé `line-too-long` ;
- tests : `TestIssue511ResumePastOversizedLine` et `TestWitnessV2Stalls`, dans `journal/issue511_test.go`.

L'issue reste ouverte sur GitHub.

### Ce que chaque composant offre pour une reprise

| Contrat | Propriétaire | Identité stable | Incarnation | Dernier événement | Vivant ou seulement connu ? | Preuve |
|---|---|---|---|---|---|---|
| ixmux : `ixmux.json`, `commands.jsonl` | Claude (R1, R2, cette R1) | nom de la cible, id du mandat | `epoch` par processus ; session_id du `claude -p` enfant | par livraison : `since`, `age` ; par session : MANQUE avant cette tranche | enfant vivant ou non ; aucune sonde de réactivité | PROUVÉ (tests R1 et R2) |
| Sessions Claude Code : `~/.claude/sessions/<pid>.json` | Claude Code, non documenté | `sessionId` | `pid` + `procStart` (FILETIME exact sous Windows) | `updatedAt`, `statusUpdatedAt`, date de modification du transcript | rien ne le dit : un fichier survit au plantage de son processus | PROUVÉ (lu, et `procStart` = heure de création du processus, vérifié) |
| wmux : `tree`, `agent-state`, `%APPDATA%\wmux\sessions\session.json` | application wmux | `claudeSessionId` par volet | MANQUE | `agent-state.updatedAt`, `blockedSince` | seulement connu ; l'état d'agent dit `blocked` | PROUVÉ (commandes lancées) ; outil MCP `ix_wmux_snapshot` fusionné |
| Service Windows ixctl | IX Service Windows | nom du service | `epoch` ixctl | journal de service (`t`) | état SCM ; démarrage manuel, pas de relance | PROUVÉ (`sc.exe qc`) |
| `ixctl resume check` (branche `feat/ixctl-resume-check`, non fusionnée) | IX Service Windows | nom du composant | MANQUE | MANQUE | ok, degraded, down, unknown par service et par port | PROUVÉ (tests de sa branche) ; aucun fichier d'inventaire n'existe |
| Sonde `claude-sessions@1` (branche `feat/ixctl-claude-sessions-probe`, pull request #501) | IX Service Windows | `session_id` | `pid:creation_ms` | `read_at_ms` | `identity: verified` ou `unknown`, `gone` | PROUVÉ (tests de sa branche) ; état en mémoire seulement ; inopérante depuis un service en session 0 |
| Registre `~/.agents/claims.jsonl` | flotte | `lane` | MANQUE | `ts` | seulement déclaré | PROUVÉ (format) ; statuts libres, sans schéma |
| Reçus `ix-recover` | ix (expérimental) | `run_id` | `attempt_id` | — | ne s'applique qu'aux pipelines purs | PROUVÉ (tests) |
| Sessions cloud | claude.ai | — | — | — | aucun fichier local ne les liste | MANQUE en local |
| Brickyard (cerveau de Delta, passerelle) | `brickyard-remote-91` | — | `brain.pid` | — | aucune route `/status` ni `/health` | PROUVÉ (code) ; statut : MANQUE |

## 2. La tranche verticale minimale

[Gaïa] La demande : un inventaire local et cloud, une identité stable et une incarnation, le dernier événement et sa fraîcheur, l'état (en cours, bloqué, terminé, incertain), les preuves de reprise et l'accès à la session. Une session non joignable ne doit jamais être présentée comme active.

[Claude] Le choix est un inventaire **en lecture seule, hors ligne**, qui ne repose que sur des fichiers et des heures de création de processus. Il répond donc même après un plantage, quand rien ne tourne. Il se trouve dans `go/ixmux/inventory`, avec la sous-commande `ixmux inventory` et l'outil MCP `ixmux_inventory`.

- **Preuve d'activité, séparée des métadonnées.**
  - Une entrée n'est `running` ou `idle` que si un processus vivant porte le pid de l'enregistrement **et** a été créé à l'heure que l'enregistrement déclare. Sinon elle est `unreachable` (`absent` ou `pid-reused`) ou `unknown` (`unverified`).
  - Ce que disent les fichiers reste à part, dans `declared`.
- **Identité et incarnation.**
  - Session : son `sessionId`, avec `pid:<pid>@<création>`.
  - ixmux : son dossier, avec `epoch:<…>`.
  - Mandat : son id, avec son incarnation.
- **Fraîcheur.** `last_activity` et `age_s`. Pour une session, c'est la plus récente de ses heures et de la date de modification de son transcript, dont le contenu n'est jamais lu.
- **États.** `running`, `idle`, `stopped`, `done`, `uncertain`, `unreachable`, `unknown`, `unreadable`. `blocked` ne vient que de wmux, qui n'est pas lu dans cette tranche (voir `gaps`).
- **Reprise.** Chaque entrée dit dans `resume` ce qu'une personne peut faire ; l'inventaire ne fait rien.
  - Une session restée `unreachable` : `claude --resume <id>`.
  - Un doublon dont une incarnation vit : ne pas reprendre.
  - Un mandat incertain : une personne décide, et il n'est jamais renvoyé, sous aucun id.
  - Un ixmux planté : le relancer. Son démarrage note `uncertain` ce qui était en livraison et `refused` ce qui n'était qu'accepté. Il ne renvoie rien.
- **Lacunes dites.** `gaps` nomme ce qui n'est pas lu : les sessions cloud (aucune n'est montrée, aucune n'est présumée active), les volets wmux et les services Windows.

### Pourquoi ce sous-ensemble, et pas le reste

[Claude]
- **Les services et les ports** sont déjà couverts par `ixctl resume check` (IX Service Windows). Les dupliquer ferait un second rédacteur.
- **La sonde de sessions d'ixctl** lit `claude agents --json`, garde son état en mémoire et ne voit rien depuis la session 0. L'inventaire lit les enregistrements que cette commande résume, hors ligne, et ne la remplace pas. Converger est une décision réservée (section 6).
- **wmux** donne `blocked`, mais par sa CLI. Celle-ci, sur ce poste, passe par `wmux.cmd`, ce qui sort du chemin Go. Lire wmux est un pas suivant possible, en lecture seule.
- **Les sessions cloud** n'ont aucune source locale. Les lister demanderait un accès à l'API de claude.ai : c'est un accès élargi, hors mandat.

## 3. Ce qui est livré, avec les tests

### `682e446f` : la cause d'un arrêt propre

Le défaut a été trouvé par la sonde de `ix-ixmux-38`. Un `POST /v1/shutdown` en plein tour donnait, après le redémarrage, `uncertain` avec la cause `adapter-error`. Or cette cause dit que la cible a répondu une erreur.

- **Rouge** : `TestACleanStopDuringATurnIsCauseRestart`, la sonde adoptée, à laquelle j'ai ajouté les assertions sur la cause, le renvoi refusé et un seul tour. Il échouait avec `cause after a clean stop mid-turn: adapter-error, want restart`.
- **Vert** : le serveur appelle `Engine.Stopping()` avant de fermer les enfants `claude`. Une erreur d'adaptateur reçue ensuite est notée avec la cause `restart`.
- **Mutation** : `TestAnAdapterErrorDuringAStopIsCauseRestart` tue le mutant où `Stopping` ne fait rien.
- **Retour arrière** : la valeur `restart` existait déjà, et les binaires R1 et #515 la relisent.

### `ecbddbde` : l'inventaire

| Exigence [Gaïa] | Test |
|---|---|
| Redémarrage : un enregistrement laissé par un processus disparu | `TestASessionWhoseProcessIsGoneIsUnreachable` |
| Redémarrage : un pid réutilisé, créé avant ou après l'heure déclarée | `TestAReusedPIDIsNotTheSession` |
| Preuve d'activité | `TestALiveSessionIsProven`, `TestCreatedProvesALiveProcessOnly` (vrais processus) |
| Doublon | `TestADuplicateSessionIsNeverResumedTwice` |
| Concurrence : un enregistrement réécrit pendant 8 lecteurs (`-race`) | `TestReadsDuringRewritesSeeAWholeRecordOrNone` |
| Saturation : bornes comptées, mandat incertain jamais omis, journal illisible | `TestUnreadableAndTooManyRecordsAreSaid`, `TestAnUncertainMandateIsNeverLeftOut`, `TestAnUnreadableLogIsSaid` |
| Déconnexion : un ixmux planté, avec ses mandats ; aucun fichier modifié | `TestACrashedIxmuxShowsItsMandatesForAPerson` |
| Bout en bout : vrai ixmux, faux claude, tué en plein tour, redémarré ; l'outil MCP pendant la panne ; un seul tour | `TestInventoryAcrossAKillDuringATurn` |
| Le lecteur du journal ne casse jamais un ixmux en marche | `TestReadLogGivesEachMandateItsLastState`, `TestReadLogNeverBreaksARunningEngine` |

**Mutations** : 9 sur 9 tuées. M9, une tolérance d'une heure au lieu d'une milliseconde, a d'abord survécu. Le test ne décalait le processus que dans un sens. Je l'ai renforcé, et M9 est maintenant tuée.

**Rouge trouvé en route.** Sous Windows, un lecteur ordinaire de `commands.jsonl` fait échouer la compaction d'un ixmux en marche, avec « Access is denied ». Le partage `FILE_SHARE_DELETE` n'y change rien, car `MoveFileEx` ne remplace pas un fichier ouvert.

Ce n'est pas une casse :
- le journal reste tel quel, et la compaction est retentée une minute plus tard ;
- au pire, un nouveau mandat est refusé `log-full`, avant tout envoi.

Le comportement est documenté et testé. L'option d'un renommage POSIX est réservée à l'opérateur (section 6).

### Démonstration réelle (2026-11-11 vers 02:34 UTC, binaire local)

- `ixmux inventory` sur ce poste :
  - 12 enregistrements de session Claude Code, **tous prouvés vivants** (pid et heure de création exacte) ;
  - **un doublon réel** : la session `1ca11168` (celle de ce mandat) vit dans deux processus prouvés, « IX » et « ix-da ». L'inventaire dit de n'en reprendre aucun. Aucun processus n'a été touché.
- `ixmux inventory --no-claude --state <ixmux-s4/measure/state>` sur un vrai dossier d'état du 6 novembre :
  - l'ixmux est `stopped`, avec son processus `absent` ;
  - 3 mandats `done` ;
  - le sha256 de chaque fichier du dossier est identique avant et après.

## 4. Brickyard : une intégration future, en lecture puis en contrôle

[Claude] Ce n'est pas implémenté. C'est une étude, qui ne bloque pas la tranche.

1. **Lecture.** Aujourd'hui, Brickyard n'expose aucun état :
   - le cerveau (`127.0.0.1:8807`) n'a aucune route `/status` ;
   - `/hear` et `/said` exigent les secrets `X-Delta-Hook` et `X-Delta-Remote`.

   Le pas le moins invasif consisterait, pour son propriétaire, à écrire un `brain.json` à la manière d'`ixmux.json` : pid, heure de démarrage, epoch, état. L'inventaire le lirait avec la même preuve (pid et heure de création), sans secret ni réseau.
2. **Contrôle.** Il passerait par les transports existants seulement : un mandat ixmux vers une cible Delta, avec l'at-most-once déjà prouvé. Pas de nouveau relais. Cela demande une cible ixmux pour Delta, que seul le propriétaire de Brickyard peut ouvrir.

## 5. Compromis, coûts, dépendances, Windows et Linux

| Sujet | Windows | Linux |
|---|---|---|
| Preuve d'un processus | `GetProcessTimes` ; un processus sorti mais retenu par un handle compte comme absent (`WaitForSingleObject`, depuis la revue) | `/proc/<pid>/stat` (starttime) et `btime` ; USER_HZ supposé à 100, donc 10 ms de précision ; un zombie compte comme absent |
| Tolérance sur l'heure de création | 1 ms pour un FILETIME exact ; 1 min, dans les deux sens, pour une heure lue sur l'horloge murale (un saut d'horloge ne tue pas un processus vivant) | `btime` est une horloge murale : toujours 1 min |
| Sessions Claude Code | `procStart` est un FILETIME : correspondance exacte ; `pidDomain` comparé au nom de la machine | format de `procStart` non vérifié sous Linux : repli sur `startedAt`, à une minute près, plus faible |
| Fichier spécial à la place d'un enregistrement | ouverture ordinaire, puis refus de tout ce qui n'est pas un fichier régulier | ouverture `O_NONBLOCK` : une FIFO ne bloque jamais, puis refus (`unreadable`) |
| Lecture du journal pendant une compaction | la compaction échoue, sans casse, et est retentée une minute plus tard | aucun effet : un renommage n'attend pas les lecteurs |
| Enfants `claude` d'ixmux | Job Object : ils meurent avec ixmux | ils survivent à un ixmux tué jusqu'à la fin de leur tour (R2) |

- **Dépendances** : aucune nouvelle. `golang.org/x/sys` était déjà là.
- **Coût d'un passage** : quelques ms. Il ne lit aucun transcript, seulement leur date de modification.
- **Ce qui reste fragile** : le format de `~/.claude/sessions/*.json` n'est pas un contrat. S'il change, les entrées deviennent `unreadable` et ne sont jamais devinées.

## 6. Décisions réservées à l'opérateur

1. **Pousser ou fusionner** `feat/ixmux-fiabilite-r1`, `feat/ixmux-p1-r2` et `feat/ixmux-inventory-r1` dans `feat/ixmux` (#515). Rien n'est poussé. État vérifié le 2026-11-11, à `14532df1` :
   - **`origin/feat/ixmux` a avancé** de deux commits d'une autre session : `c93f1502` (écritures bloquées bornées, reçus de dépôt liés, `fingerprint` renommé `Fingerprint`) et `0da99ee0`.
   - **Une fusion d'essai** a été faite dans un worktree jetable, en HEAD détaché, puis supprimé. Elle donne un conflit dans `deliver/engine.go`, mécanique : garder mon côté, nommer la fonction `Fingerprint` et sa doc, et porter les deux champs `Fingerprint: r.fp` et `Cause: r.cause` dans `entry()`.
   - **Un piège que git ne signale pas** : mes tests `logbound_review_test.go`, `logbound_test.go` et `r2_test.go` appellent encore `fingerprint(`. Ils fusionnent sans conflit mais ne compilent plus. Il faut les renommer.
   - Ainsi résolue, la fusion passe `gofmt`, `go vet` et les 6 paquets sous Windows.
   - **La CI** : seul `main` (`5c99d6ab`) découvre `go/ixmux` (job `go-modules`, Linux avec `-race` et Windows). Ma branche, partie d'un `main` plus ancien, ne lance que `go/ixctl`. Pour obtenir la preuve Linux, il faut intégrer `main`, ce qui se fait sans conflit en simulation, ou passer par la pull request vers `main`.
   - Les 8 commits R1 (`1e15c56c` à `96f20e90`) n'ont pas de lien `Claude-Session` ; le résumé de ma session précédente me les attribue.
2. **Converger avec ixctl.** Faut-il que l'inventaire devienne la source hors ligne de la sonde `claude-sessions@1` et de `ixctl resume check` ? C'est l'affaire du propriétaire IX Service Windows. Les options :
   - (a) les garder séparés ;
   - (b) ixctl lit le rapport `ixmux.inventory.v1` ;
   - (c) déplacer le paquet dans un module partagé.
3. **La compaction en lecture concurrente sous Windows.** Les options :
   - (a) garder la nouvelle tentative une minute plus tard (l'état actuel) ;
   - (b) un renommage POSIX (`FILE_RENAME_FLAG_POSIX_SEMANTICS`), qui touche la durabilité validée en R2 et demande sa propre revue.
4. **Lire wmux** (`agent-state`, pour l'état bloqué), et par quel chemin : sa CLI, `session.json`, ou l'outil MCP `ix_wmux_snapshot`.
5. **Le doublon réel de la session `1ca11168`**, deux processus vivants. Lequel garder ? Seul l'opérateur peut fermer un terminal.
6. **Les sessions cloud.** Faut-il un accès à l'API de claude.ai pour les lister ? Ce serait un accès élargi.
7. **WSL et NO-BASH-R1.** `ix-ixmux-38` cite une consigne NO-BASH-R1 qui interdirait WSL aux sessions « Branche » et « Feuille ». Ma passe Linux `-race` passe par `wsl.exe … bash -lc` : une commande ponctuelle, permise par GAIA-IX-HOOKS-SCOPE-R2 pour cette session. Si l'opérateur veut l'interdire aussi ici, la preuve Linux devra venir de la CI (trois OS) après un push qu'il autorise.

## 7. Revue contradictoire

[Claude] Un relecteur indépendant, sans mon raisonnement, a reçu le diff `5911f37c..ecbddbde` et la consigne de chercher un chemin où une session injoignable passe pour active, où un mandat pourrait être renvoyé, ou où l'outil MCP en dit trop. Résultat : 0 P0, 2 P1, 4 P2, 4 P3. Les correctifs sont dans `94de6b8e`, chacun rouge avant, vert après.

| Constat | Scénario | Correctif | Preuve |
|---|---|---|---|
| P1-1 | un saut d'horloge de plus d'1 ms rendait `pid-reused` un ixmux vivant, donc `unreachable`, et pouvait conseiller `claude --resume` sur une session vivante | `ExactTol` (1 ms) pour un FILETIME, `ClockTol` (1 min, dans les deux sens) sinon | `TestAClockStepDoesNotKillALiveProcess`, rouge avant |
| P1-2 | `ixmux_inventory` portait chemins, pids, adresse, tous les ids terminés et toute cause, plus qu'`ixmux_status` | vue réduite `ixmux.inventory.mcp.v1` | contrôle sur le texte brut (chemin, pids, adresse, reason, session plantée) ; le mutant « rapport complet » fuit et est attrapé |
| P2-3 | un mandat seulement `accepted` sous un ixmux mort pouvait être omis par la borne | seuls `delivered` et `refused` sont bornés | `TestAnUnfinishedMandateIsNeverLeftOut`, rouge avant |
| P2-4 | un ixmux qui démarre entre la lecture d'`ixmux.json` et celle du journal faisait croire qu'un mandat juste accepté n'avait jamais été envoyé | relecture d'`ixmux.json` après le journal ; tout changement rend `unknown` les mandats non terminés | `TestAnIxmuxStartingDuringTheReadIsReadAgain`, 3 échecs sans la relecture |
| P2-5 | une FIFO nommée comme un enregistrement bloquait `open()` pour toujours sous Linux | `deliver.OpenRegular` : ouverture non bloquante, puis refus d'un fichier non régulier | `TestAFIFOInPlaceOfAFileNeverHangs` (Linux), bloqué 10 s avec un open bloquant |
| P2-6 | les assertions MCP ne pouvaient pas échouer | contrôles sur le texte brut, session plantée dans `CLAUDE_CONFIG_DIR` | voir P1-2 |
| P3-8 | le README disait qu'un journal trop gros était compté dans `omitted` | texte corrigé (EN et FR) | relecture |
| P3-9 | la vivacité Windows reposait sur le code de sortie 259 | `WaitForSingleObject(h, 0)` | tests de vivacité, vrais processus |
| P3-10 | un enregistrement venu d'une autre machine (`pidDomain`) pouvait être « prouvé » ici par un pid homonyme | `unknown` / `unverified` si `pidDomain` nomme une autre machine | `TestARecordFromAnotherMachineIsNotProvenHere` |

**Validations.**
- Windows : `go vet` et la suite complète (`claudestream`, `cmd/ixmux`, `deliver`, `inventory`, `journal`, `server`) passent sur `03d916b0`.
- Linux (WSL2, Ubuntu, `go vet` puis `go test -race -count=1 ./...`) :
  - sur `94de6b8e` : **rouge**, mais par un défaut du test seul. Le pid de l'ixmux valait 2026, et le contrôle de fuite le trouvait dans les dates (« 2026-11-11 »). La réponse MCP ne portait aucun pid ;
  - `03d916b0` cherche le pid comme un nombre isolé. Le mutant qui recopie le pid dans l'entrée de l'ixmux est tué ;
  - sur `03d916b0` : **vert** sur les 6 paquets (2026-11-10, de 21:54:05 à 21:55:22 −05:00).

## 8. Limites restantes

- **P3-7** : pendant un arrêt, une vraie erreur d'une autre cible qu'un `claude` (s'il en existe un jour) serait notée avec la cause `restart`. Son `reason` garde le texte d'origine, et la conséquence reste « ne pas renvoyer ». Aucune autre cible n'existe aujourd'hui.
- ~~Le glob des transcripts n'est pas borné~~ et ~~`LogMandate.Answered` est lu mais pas exposé~~ : traités en R2 (section 9).
- **P2-5 n'a qu'une preuve Linux** (section 9).
- **Les sessions cloud, les volets wmux et les services Windows** ne sont pas lus ; `gaps` le dit à chaque rapport.
- **Le format de `~/.claude/sessions/*.json`** n'est pas un contrat (section 5).
- **Gaïa n'est pas raccordée** : aucun test réel depuis elle.

## Verdict

**READY-FOR-PILOT, côté ixmux, en lecture seule.** L'inventaire ne présente jamais une session injoignable comme active : la preuve d'activité exige un pid vivant créé à l'heure déclarée, sur cette machine, et tout le reste est `unreachable` ou `unknown`. Il ne renvoie ni ne modifie rien ; un mandat incertain n'est jamais omis et le conseil est de ne jamais le renvoyer. Le retour arrière est simple : la sous-commande et l'outil MCP sont nouveaux, et le format du journal n'a pas changé.

Ce que le verdict ne couvre pas : Gaïa (non raccordée), les sessions cloud, wmux, et tout push ou merge, qui attendent l'opérateur (section 6).

## 9. R2 : bornes de la recherche des transcripts (GAIA-IX-INVENTORY-BOUNDS-R2)

[Gaïa] La demande : borner la recherche des transcripts (fichiers, octets, temps, configurables), avec un statut partiel explicite et jamais de faux « complet » ; ne plus lire `LogMandate.Answered` ; tester sur de grosses fixtures, avec erreurs et épuisement, sans fuite de contenu privé ; examiner la preuve de P2-5 ; Windows seulement, sans WSL. [opérateur] Feu vert le 2026-11-10, dans ses propres mots, par AskUserQuestion : « Oui, toute la tranche ». Commit `a6cc36e1`.

### La recherche bornée

- **Chemin direct d'abord.** Un seul `stat`, à `projects/<cwd, tout caractère autre qu'une lettre ou un chiffre ASCII changé en '-'>/<id>.jsonl`. Sur ce poste, cette règle trouve le transcript de 12 sessions sur 12 : PROUVÉ, en ne lisant que les métadonnées.
- **Repli.** Les autres dossiers de `projects/`, listés une seule fois par lecture, avec un `stat` par dossier.
- **Budget, un seul par lecture.**
  - Le budget compte les fichiers : 4096 chemins examinés et entrées listées.
  - Il compte aussi le temps : 2 s, mesurées par `Options.Now`, ce qui rend le test déterministe.
  - Les deux se règlent par `--transcript-files` et `--transcript-time`. Une valeur nulle est refusée (code 2).
- **Octets** : aucun octet de transcript n'est lu, seulement les dates de modification. Le budget en octets vaut donc zéro par construction. Un test place un texte privé dans chaque transcript et vérifie qu'aucun rapport ne le porte. Les enregistrements de session restent bornés à 64 Kio chacun et à 256 au total.
- **Statut explicite.** `declared.transcript` vaut `found`, `none` (recherche menée jusqu'au bout), `search-cut` ou `unreadable`. Une recherche coupée ou en échec :
  - pose un `problem` (la dernière activité est celle de l'enregistrement, peut-être plus ancienne) ;
  - est comptée dans `omitted` ;
  - et rend `partial: true`. Le rapport ne se dit complet que si rien n'a été laissé de côté.

| Test | Rouge sur l'ancien code |
|---|---|
| `TestTheTranscriptAtItsProjectPathIsOneStat` (1000 projets leurres, budget 1) | oui : pas de `declared.transcript` |
| `TestATranscriptElsewhereIsFoundWithinTheBudget` | oui |
| `TestASpentFileBudgetIsPartialNeverComplete` (2 sessions, 1000 leurres, budget 100, partagé) | oui : le glob trouvait ce que le budget ne doit pas atteindre |
| `TestASpentTimeBudgetIsPartial` (horloge qui avance d'1 s par lecture, budget 5 s) | oui |
| `TestNoTranscriptIsNotPartial` | oui |
| `TestAnUnreadableProjectsDirectoryIsPartial` (`projects` est un fichier) | oui |
| `TestACutListingIsNeverNone` (300 fichiers, aucun dossier, budget 50) | oui ; seul à tuer le mutant « coupé pris pour none » |
| `TestInventoryTranscriptBudgetFlags` (commande, JSON et texte) | oui : option inconnue |

**Mutants** : 6 sur 6 tués (pas de budget, liste coupée prise pour `none`, jamais `partial`, pas d'échéance, coupe non comptée, pas de chemin direct).

**Mesure réelle** : `ixmux inventory` sur ce poste : 12 sessions `found`, `partial: false`, environ 50 ms (266 ms au premier lancement). Avec `--transcript-files 1` : 11 recherches coupées, `partial: true`.

**`LogMandate.Answered`** est retiré : le rapport ne s'en servait pas.

**Validations** : sous Windows, `go vet` et les 6 paquets passent sur `a6cc36e1`. **Linux n'a pas été lancé pour cette tranche** (pas de WSL, par le mandat). La preuve Linux viendra de la CI après un push que l'opérateur autorisera.

### Examen de la preuve P2-5 (le test FIFO)

- **Nom exact** : `TestAFIFOInPlaceOfAFileNeverHangs`, dans `go/ixmux/inventory/fifo_linux_test.go`.
- **Sous Windows, il n'existe pas.**
  - Le suffixe `_linux_test.go` l'exclut de la compilation : `go list -f '{{.TestGoFiles}}' ./inventory/` donne `[inventory_test.go transcript_test.go]` sous Windows, `[fifo_linux_test.go …]` avec `GOOS=linux`.
  - `go test -count=1 -run TestAFIFOInPlaceOfAFileNeverHangs -v ./inventory/` répond `ok … [no tests to run]`. Un résultat Windows « ok » à cet endroit ne prouve donc rien.
  - `GOOS=linux go vet ./inventory/` passe : cela prouve seulement que le test compile pour Linux.
- **Sous Linux**, deux preuves, toutes deux par WSL2 le 2026-11-10 :
  1. **Le rouge puis le vert, avant le commit `94de6b8e`.** La commande, retrouvée mot pour mot dans la transcription de la session : `wsl.exe -d Ubuntu -- bash -lc "… go test -count=1 -timeout 60s -v -run 'TestAFIFO' ./inventory"`, lancée deux fois.
     - La première fois, `openshared_unix.go` était remplacé à la main par un `os.Open` bloquant. Résultat : `fifo_linux_test.go:36: the inventory hung on a FIFO`, `--- FAIL: TestAFIFOInPlaceOfAFileNeverHangs (10.01s)`, paquet `FAIL` en 10,013 s.
     - Le fichier a ensuite été restauré (`git diff` vide). Résultat : `--- PASS: TestAFIFOInPlaceOfAFileNeverHangs (0.00s)` et `ok … inventory 0.003s`.

     **Le chiffre de 0,003 s vient de là.** C'est le temps du paquet avec ce seul test sélectionné, sans `-race`. Il est court parce qu'une FIFO ouverte en `O_NONBLOCK` rend la main tout de suite, puis est refusée. La ligne `--- PASS` le nomme : il a vraiment tourné. Ce n'est ni un cache (`-count=1`) ni un « no tests to run ».
  2. **La suite complète** : `go test -race -count=1 ./...`. Le paquet `inventory` y passe en 2,067 s sur `03d916b0`. Le test n'a pas de `t.Skip`.
- **Assertions réelles** :
  - quatre FIFO (`mkfifo`) remplacent un enregistrement de session, `ixmux.json` et `commands.jsonl`, plus un lien symbolique vers une FIFO ;
  - chacune doit donner une seule entrée `unreadable` ;
  - le tout doit rendre la main en moins de 10 s, sinon `t.Fatal`.

  Avec une ouverture bloquante, il restait bloqué 10 s, puis échouait.
- **Unitaire ou intégration** : c'est un test d'intégration du paquet `inventory`. Il passe par `Collect`, avec de vraies FIFO du système de fichiers et une fausse table de processus. Il ne lance ni ixmux ni `claude`. Aucun test unitaire de `deliver.OpenRegular` n'existe seul.
- **Limite** : sous Windows, la branche « pas un fichier ordinaire » d'`OpenRegular` n'a aucun test, faute de FIFO nommée dans un dossier. Seule la CI Linux, ou une passe WSL autorisée, la vérifie.
