# ixmux

[English](README.md)

`ixmux` est un processus Go durable, unique, qui sert les paquets `journal` et `deliver` en HTTP et en WebSocket, en loopback uniquement. Un client qui perd sa connexion la retrouve et reprend sans exécuter un mandat deux fois. À l'arrêt propre, le processus ferme tout correctement ; au redémarrage, il reprend son état depuis le journal de commandes.

```
ixmux serve --addr 127.0.0.1:0 --state <dossier> [--journal nom=chemin ...] [--target nom=chemin ...]
            [--claude-target nom ...] [--claude-bin chemin] [--claude-env NOM ...]
            [--timeout 30s] [--max-conns 16] [--max-http-conns 64] [--read-timeout 10s] [--queue 256] [--poll 200ms]
ixmux mcp --state <dossier>
ixmux version
```

## Construction

```
go -C %USERPROFILE%\.cache\ix-ixmux\go\ixmux build -ldflags "-X main.version=<git describe>" -o %USERPROFILE%\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe ./cmd/ixmux
```

`bin/` est ignoré par git. `ixmux.json` donne le SHA-256 de l'exécutable en cours.

## API (jeton Bearer, `Host` en loopback, pas d'`Origin`)

| Route | Rôle |
|---|---|
| `POST /v1/hello` `{nonce}` | La seule route sans jeton (le `Host` en loopback et l'absence d'`Origin` restent exigés, le corps est borné). Le corps est une seule valeur JSON, `{nonce}`, où `nonce` fait 32 octets en hexadécimal minuscule ; tout le reste (une deuxième valeur comprise) donne 400 `bad-nonce`. La réponse est `{mac}` : l'hexadécimal de HMAC-SHA-256(HMAC-SHA-256(jeton, `ixmux/hello/v1`), les 32 octets du nonce). Un client le vérifie avant d'envoyer le jeton, sur la même connexion : le jeton ne part alors que vers un serveur qui a prouvé le détenir. `ixmux mcp` le fait toujours ; le runbook ci-dessous et les autres clients envoient le Bearer seul, et peuvent faire le même défi. La clé dérivée ne signe rien d'autre, et les curseurs de journal sont signés par une autre clé, aléatoire à chaque processus. |
| `POST /v1/mandates` `{mandate_id, target, incarnation, text}` | Livre au plus une fois. La réponse donne `state` et `replayed`. Un refus revient sous la forme `{error, detail}`, avec 400, 404, 409, 413 ou 503. |
| `GET /v1/mandates/{id}` | L'état d'une livraison et, une fois livrée, son résultat : `ack` (`receipt`, `text`, `ack_seen`, `truncated`), lu dans le journal de commandes, donc aussi après un redémarrage. |
| `POST /v1/mandates/{id}/reconcile` | Demande à la cible si une livraison incertaine est arrivée. |
| `GET /v1/status` | Les cibles et leurs incarnations, les livraisons (âge, `still_running`), les connexions et les bornes. |
| `GET /v1/journal/{nom}?from=&incarnation=&max_bytes=&cursor=` | Une page de `journal.ReadCursor`. Renvoie le `cursor` qu'elle rend : ixmux le signe en HMAC. Un curseur non signé ou mal signé est ignoré (`cursor_ignored`), et la lecture repart de `from`. |
| `GET /v1/stream[?since=<seq>&epoch=<époque>]` | WebSocket. Le premier message est `hello`. Viennent ensuite soit les événements manqués, soit un `gap` suivi de l'état : des messages `state-page`, puis un `state-end`. Puis les événements en direct : `mandate`, `journal`, `journal-reset`. |
| `POST /v1/shutdown` | Arrêt propre. |

Fonctionnement du flux :
- Les événements sont numérotés par `seq` au sein d'une `epoch`. L'époque change à chaque démarrage d'ixmux.
- Pour reprendre, passe `since=<seq du dernier événement>&epoch=<époque>`.
- Un `gap` a l'une de quatre raisons :
  - `epoch` : ixmux a redémarré ;
  - `ring` : les événements ne sont plus gardés ;
  - `ahead` : `since` est au-delà du dernier événement ;
  - `slow` : ta file a débordé. La connexion se ferme alors avec le code 1008 ; reprends avec `since=last_seq`.
- Un message qui ne peut pas être écrit en 10 s (le client ne lit plus) ferme aussitôt la connexion, sans `gap` : plus rien ne peut atteindre ce client. Reprends avec `since`, comme après toute coupure.
- Après un `gap`, chaque `state-page` contient `deliveries` (la dernière transition de chacune, sa raison coupée à 1 Kio), et `state-end` donne `pages`, `deliveries`, `targets` et `journals`. Leur `seq` est le dernier événement qu'ils couvrent : l'événement suivant est `seq+1`, et aucune transition qu'ils contiennent ne revient comme événement.
- Un client doit relever sa limite de lecture WebSocket à `max_stream_message` (1 Mio). Tout message y tient, l'état compris.

Bornes HTTP : les en-têtes et le corps d'une requête sont lus en `--read-timeout` au plus, et sa réponse écrite en 10 s, ou dans le délai de livraison plus 10 s pour une livraison ou une réconciliation. ixmux tient au plus `--max-http-conns` connexions TCP, celles du flux comprises ; une de plus reçoit aussitôt 503 `too-many-connections`. `--max-conns` ne borne que les connexions du flux.

## Fichiers de `<state>`

- `commands.jsonl` : le journal de commandes des livraisons. Tant qu'ixmux tourne, il tient `commands.jsonl.lock`. Il est borné à 64 Mio, la taille qu'un démarrage relit : aucun ajout ne le fait dépasser cette borne. **Chaque livraison réserve dans le journal la place des enregistrements qu'elle peut encore écrire** : une fois acceptée, son enregistrement `delivering`, son enregistrement final (avec la réponse : un enregistrement fait au plus 32 Kio, et le texte d'une réponse plus longue est coupé et marqué `truncated`) et une réponse tardive ; une fois `uncertain`, l'enregistrement qui la tranche (une réponse tardive, ou celui d'une réconciliation). Aucun autre enregistrement ne prend cette place : une livraison acceptée enregistre toujours son issue, quel que soit le nombre de livraisons simultanées. Un nouveau mandat qui ne peut pas obtenir sa place est refusé `log-full`, avant tout envoi. Un mandat dont l'id, la cible et l'incarnation prennent plus de la moitié d'un enregistrement de 1 Kio est refusé `bad-mandate`. Un enregistrement qui ne tient toujours pas (dans un journal écrit par un ixmux plus ancien, ou sous une borne abaissée) n'est pas un refus : la livraison est `uncertain`, cause `log-write-failed`, et la requête échoue `internal` ; de même pour une livraison dont l'enregistrement final trouve le journal cassé par l'écriture ratée d'une autre. **Rétention** : un mandat terminé (`delivered` ou `refused`) est gardé en mémoire, donc dédoublonné (rejoué, ou refusé `mandate-id-reuse`), au moins `--dedup-window` après sa dernière transition (720 h, 30 jours, par défaut). La première compaction au-delà l'oublie : son id redevient un nouveau mandat, et `GET /v1/mandates/{id}` répond 404. Un mandat `uncertain`, `accepted` ou `delivering` n'est jamais oublié. Le journal est compacté quand un nouveau mandat ne trouve pas de place, et au démarrage s'il dépasse les trois quarts de sa borne ou n'a plus de place pour un nouveau mandat, si cela le rend plus petit : un enregistrement par mandat, dans son état courant, avec son empreinte, son incarnation, son reçu et sa cause. En cours de route, ixmux compacte au plus une fois par minute. Le journal compacté est écrit dans `commands.jsonl.compact`, synchronisé, puis renommé par-dessus le journal (sous Windows, en écriture directe). Si les transitions du redémarrage ne tiennent pas, elles sont appliquées en mémoire : le journal les refait à chaque démarrage. Le journal se remplit encore si les mandats terminés dans la fenêtre le remplissent (environ 14 000 avec des réponses de 4 Kio) : un nouveau mandat est alors refusé `log-full` jusqu'à ce que les plus anciens sortent de la fenêtre ; raccourcir la fenêtre le libère. Ne jamais supprimer le journal. La fenêtre lit l'horloge du système : une horloge avancée fait oublier plus tôt. À chaque démarrage, stderr dit ce qui a été trouvé et fait : `ixmux: command log: N records replayed[, a torn last record of B bytes cut], U made uncertain, R made refused[ (K of them in memory only: the log is full)][, compacted from X to Y bytes[ (F finished mandates past the deduplication window forgotten)]][, compaction not done (the log is kept as read): raison]; a finished mandate is remembered at least 720h0m0s` (la ligne est en anglais).
- `ixmux.json` : adresse, PID, heure de démarrage, version, SHA-256 de l'exécutable, époque, et `state` (`running` ou `stopped`, puis `exit_code`).
- `ixmux.token` : le jeton Bearer. Il est réécrit à chaque démarrage et supprimé à l'arrêt propre. Sous Windows, sa DACL est protégée et n'accorde l'accès qu'à l'utilisateur courant ; ailleurs, son mode est 0600. Si ixmux ne peut pas le créer ainsi, il refuse de démarrer.

## Codes de sortie

| Code | Sens |
|---|---|
| 0 | Arrêt propre : plus aucun appel d'adaptateur, journal fermé, verrou libéré. |
| 1 | Échec du service. |
| 2 | Ligne de commande invalide, y compris un nom de `--claude-target` donné deux fois (refusé avant tout lancement d'enfant). |
| 3 | Démarrage refusé : l'adresse n'est pas une IP de loopback, l'état est tenu par un autre ixmux, ou le fichier jeton a échoué. |
| 4 | Arrêt alors que des livraisons, des réconciliations ou des appels d'adaptateur étaient encore en cours : le `Close` de l'engine a refusé avec `calls-in-flight`. |
| 5 | Arrêt, mais le fichier jeton n'a pas pu être supprimé ou `ixmux.json` n'a pas pu être marqué arrêté. L'erreur est sur stderr ; elle nomme le fichier, jamais le jeton. Le démarrage suivant réécrit le jeton. |

Un arrêt sort avec 1 si une transition n'a pas pu être écrite dans le journal de commandes (`log-broken`).

Un arrêt (Ctrl+C, Ctrl+Break ou `POST /v1/shutdown`) refuse aussitôt toute nouvelle livraison ou réconciliation (`engine-closed`), et attend celles en cours jusqu'à l'écriture de leur état final. Il prend au plus le délai de livraison (`--timeout`) plus 10 s, plus jusqu'à 20 s par cible claude (10 s pour que l'enfant sorte, 10 s de plus après un kill).

## Runbook (cmd.exe, chemins absolus)

La session de ce runbook est fictive : c'est le fichier JSONL `%USERPROFILE%\.cache\ixmux-fake\session.jsonl`, créé au démarrage s'il manque. Son incarnation nomme le chemin et le fichier qui s'y trouve : elle est la même d'un redémarrage à l'autre, et change si un autre fichier est mis à ce chemin.

1. Démarre ixmux dans sa propre fenêtre :

   ```
   start "ixmux" %USERPROFILE%\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe serve --addr 127.0.0.1:47011 --state %USERPROFILE%\.cache\ixmux-fake\state --target fake=%USERPROFILE%\.cache\ixmux-fake\session.jsonl
   ```

2. Vérifie l'état. Note l'`incarnation` de la cible `fake`.

   ```
   set /p IXMUX_TOKEN=<%USERPROFILE%\.cache\ixmux-fake\state\ixmux.token
   C:\Windows\System32\curl.exe -s -H "Authorization: Bearer %IXMUX_TOKEN%" http://127.0.0.1:47011/v1/status
   ```

3. Envoie Hello. Remplace `<INC>` par cette incarnation. Le fichier `session.jsonl` doit alors contenir une ligne, et la réponse doit indiquer `"state":"delivered"`. Renvoie-le : la réponse indique `"replayed":true`, et le fichier contient toujours une seule ligne.

   ```
   C:\Windows\System32\curl.exe -s -H "Authorization: Bearer %IXMUX_TOKEN%" -H "Content-Type: application/json" -d "{\"mandate_id\":\"hello-1\",\"target\":\"fake\",\"incarnation\":\"<INC>\",\"text\":\"hello\"}" http://127.0.0.1:47011/v1/mandates
   ```

4. Arrête-le, par Ctrl+C dans sa fenêtre, ou avec :

   ```
   C:\Windows\System32\curl.exe -s -X POST -H "Authorization: Bearer %IXMUX_TOKEN%" http://127.0.0.1:47011/v1/shutdown
   ```

   `%USERPROFILE%\.cache\ixmux-fake\state\ixmux.json` indique alors `"state": "stopped"` et `"exit_code": 0`.

**Retour arrière :** arrête ixmux (étape 4), puis supprime l'état et le fichier de la session fictive :

```
rmdir /s /q %USERPROFILE%\.cache\ixmux-fake
```

Rien d'autre n'est écrit : aucun service, aucune clé de registre, aucun fichier hors de `--state` et des fichiers `--target`.

## Vraie session (`--claude-target`)

`--claude-target c` garde ouvert un enfant `claude -p` en stream-json, sans outils :

```
claude -p --input-format stream-json --output-format stream-json --verbose --tools "" --strict-mcp-config --session-id <uuid>
```

- Jamais `--bare`, `--dangerously-skip-permissions` ni `--permission-mode bypassPermissions`. `--setting-sources` n'est pas touché : tes hooks s'appliquent tels quels. ixmux ne fixe aucun mode de permission : l'enfant tourne dans le mode de ta configuration (`auto` ici). Ce qui protège, c'est qu'il n'a aucun outil.
- L'incarnation de la cible est le `session_id` du `system/init` de l'enfant. Le vrai claude n'envoie `init` qu'après un premier message utilisateur : ixmux envoie donc un message d'amorce (« ixmux : ouverture de la session. Réponds seulement : OK. ») à chaque démarrage. Cela coûte un tour de modèle par démarrage, et la cible est prête environ 28 s après le démarrage.
- Si `init` liste un outil, n'a pas de tableau `tools` (absent ou `null`), ou nomme une autre session, ixmux refuse de livrer à cette cible. Le contrôle et l'écriture sur l'entrée standard de l'enfant se font sous une même prise : un `init` qui arrive entre les deux fait refuser le mandat, et rien n'est écrit.
- Chaque mandat est un message utilisateur : `ixmux mandate_id: <id>`, puis la consigne de commencer par `ACK <id>`, puis le texte. La réponse porte `ack.receipt` (`<session>:<id>`), `ack.text` (4 Kio au plus ; `ack.truncated` vaut vrai quand la réponse était plus longue et a été coupée) et `ack.ack_seen`.
- Un `result` ne compte que si son `session_id` est l'incarnation de la cible et qu'un tour est en attente. Tout autre `result` est journalisé (`{"ixmux":"ignored-result"}`) et ignoré.
- L'ACK est exact : la réponse commence par `ACK <id>`, suivi d'un espace ou de la fin. `ACK m2-extra` n'acquitte pas `m2`.
  - `ack_seen: true` : la réponse acquitte ce mandat.
  - `ack_seen: false` : la réponse n'a pas d'ACK. Le mandat reste `delivered`, puisque le résultat est de cette session et de ce tour, mais le modèle n'a pas confirmé à quel mandat il répond.
  - Un ACK qui nomme un **autre** mandat rend la livraison `uncertain`, jamais `delivered`, et l'écart est journalisé (`{"ixmux":"ack-mismatch"}`).
- Une réconciliation lit `stream.jsonl` : un mandat envoyé et suivi d'un `result` de cette incarnation passe à `delivered`, sans ack (aucune réponse n'est relue). Un `result` avec `is_error: true` ne prouve rien : le mandat reste `uncertain`.
- L'environnement de l'enfant est une liste blanche, pas tout l'environnement d'ixmux. Il contient ce que Windows, un système Unix et claude exigent pour démarrer (`SystemRoot`, `ComSpec`, `PATH`, `PATHEXT`, `USERPROFILE`, `HOMEDRIVE`, `HOMEPATH`, `APPDATA`, `LOCALAPPDATA`, `TEMP`, `TMP`, `HOME` et quelques autres, voir `claudestream.BaseEnv`), puis chaque variable nommée avec `--claude-env NOM` (répétable). Aucune variable n'est ajoutée par motif : un `*_TOKEN` ou un `*_KEY` n'atteint l'enfant que s'il est nommé. Ta connexion à Claude vient de ton profil.
- Un tour sans résultat à l'échéance de la livraison rend le mandat `uncertain`, mais le tour reste réservé : aucun autre mandat ne part tant qu'il peut encore répondre (ils sont refusés `target-busy`, avant toute écriture). L'appel de la livraison continue d'attendre (`still_running: true` dans `/v1/status`), et une réponse venue plus tard fait passer le mandat à `delivered` avec son ack, sauf si c'est un résultat en erreur ou qu'elle acquitte un autre mandat (il reste alors `uncertain`). La réservation dure au plus `TurnMax`, 10 minutes, plus la relance. Passé `TurnMax`, ixmux tue l'enfant par son Job et attend jusqu'à `StopWait` (10 s) qu'il sorte ; puis il en lance un nouveau, sous une nouvelle incarnation, et la cible est libérée. Une relance prend au plus `InitWait` (90 s), et une relance qui échoue est arrêtée en 2×`StopWait` de plus : la réservation dure au plus `TurnMax` + `StopWait` + `InitWait` + 2×`StopWait`. Un arrêt termine la réservation plus tôt : il arrête l'enfant (tué au-delà de 10 s), et un remplaçant lancé à l'instant de l'arrêt est tué par cet arrêt, en `StopWait` (3×`StopWait` si ses kills échouent). Si un kill échoue, ou si l'enfant n'est pas sorti `StopWait` après, l'incarnation est perdue (`kill-failed` dans `stream.jsonl`) : son tour est `uncertain`, sauf si son résultat était déjà arrivé ; la réservation est libérée ; et rien ne remplace l'enfant. `/v1/status` montre alors la cible avec `"state": "lost"` et son `pid` (le processus tourne peut-être encore), et un mandat pour elle est refusé 409 `stale-incarnation`. Le mandat ne peut pas être renvoyé sous la nouvelle incarnation (409 `stale-incarnation`). Avec le vrai claude, cette relance coûte un tour d'amorce.
- Chaque incarnation peut ajouter au plus 256 Mio de sortie standard à `stream.jsonl` et 16 Mio à `stderr.log`. Au-delà de l'un ou l'autre plafond, l'enfant est tué et le tour en attente devient `uncertain` avec la raison `stream-too-large` (ou `stderr-too-large`). Une ligne de sortie standard de plus de 8 Mio est coupée dans le journal, et le tour en attente devient `uncertain`. Ces plafonds terminent l'incarnation : l'enfant n'est pas relancé.
- ixmux attend au plus 90 s (`InitWait`) l'`init` de l'enfant et la réponse à l'amorce ; au-delà, le démarrage échoue.
- Le répertoire de travail de l'enfant est `<state>\claude-<nom>\`, qu'ixmux crée. Il contient `stream.jsonl` (la sortie standard brute de chacun des enfants de la cible tour à tour, plus des lignes `{"ixmux":...}` pour le démarrage, l'envoi, le résultat, la sortie, l'arrêt et les cas ci-dessus) et `stderr.log`. `stream.jsonl` est servi comme journal `claude-<nom>`.
- L'enfant tourne dans un Job Object kill-on-close : si ixmux meurt, tué ou non, l'enfant meurt avec lui. Un redémarrage lance un nouvel enfant, donc une nouvelle incarnation : un mandat qui était en livraison devient `uncertain`, sa réconciliation répond 409 `stale-incarnation`, et son renvoi répond 409.
- Un vrai tour dépasse souvent le `--timeout 30s` par défaut. Passe `--timeout 120s`.
- Un arrêt attend aussi jusqu'à 10 s que l'enfant sorte une fois son entrée standard fermée, puis tue le job et attend jusqu'à 10 s de plus. `stream.jsonl` en garde la trace : `{"ixmux":"stop","step":"stdin-closed"}`, puis `exited`, `killed` ou `kill-failed` avec `after_ms`. Un enfant qui exécute encore ses hooks SessionEnd à 10 s est tué avec eux.

Runbook (cmd.exe), avec un état jetable :

```
start "ixmux" %USERPROFILE%\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe serve --addr 127.0.0.1:47012 --state %USERPROFILE%\.cache\ixmux-claude\state --claude-target c --timeout 120s
```

ixmux n'écrit `ixmux.json` et ne répond qu'une fois l'enfant prêt (environ 30 s). `/v1/status` montre alors la cible `c` avec son `incarnation` et `"alive":true`. Envoie Hello comme à l'étape 3, avec `"target":"c"`. Arrête-le comme à l'étape 4.

**Piège :** un `start` lancé dans un `cmd /c` dont la sortie est capturée hérite du pipe. L'appelant attend alors la sortie d'ixmux, et de son enfant. Lance `start` depuis une vraie fenêtre de console.

**Retour arrière :** arrête ixmux, vérifie qu'il ne reste aucun enfant `claude` (`tasklist /fi "imagename eq claude.exe"`), puis supprime l'état jetable :

```
rmdir /s /q %USERPROFILE%\.cache\ixmux-claude
```

La session Claude elle-même est gardée par claude sous ton profil, comme toute session `claude -p`. ixmux n'écrit rien d'autre.

## Adaptateur MCP (`ixmux mcp`)

`ixmux mcp --state <dossier>` est un serveur MCP local sur stdio, branché sur l'ixmux serve de ce
dossier d'état. Il y lit l'adresse et le jeton à chaque appel, et suit donc un ixmux redémarré
entre-temps. Il a quatre outils : `ixmux_status`, `ixmux_submit` (un reçu en 2 s, même quand le
tour dure plus longtemps), `ixmux_result` (lu dans le journal de commandes) et `ixmux_inventory`
(l'inventaire ci-dessous, de ce seul dossier d'état, lu dans ses fichiers). Aucun outil n'arrête
ixmux. Il n'envoie le jeton qu'à un serveur qui a répondu à son défi `/v1/hello`, sur la même
connexion ; un processus qui aurait pris le port sans le jeton n'obtient rien (`ixmux-unverified`).
Un processus qui peut lire le fichier du jeton est hors de ce modèle. Le contrat, avec la commande
exacte, les schémas et les erreurs : [MCP.fr.md](MCP.fr.md).

## Inventaire après un redémarrage (`ixmux inventory`)

`ixmux inventory [--state <dossier> ...] [--claude-home <dossier>] [--no-claude] [--json]` ne lit
que des fichiers et les heures de création des processus. Il n'envoie, ne lance, ne rejoue, ne
verrouille et n'écrit rien. Après un plantage, une coupure de courant ou un redémarrage de Windows,
il liste :

- **les sessions Claude Code locales** : Claude Code écrit un enregistrement pour chaque processus
  en marche dans `<claude-home>/sessions/<pid>.json` (`--claude-home` vaut par défaut
  `%CLAUDE_CONFIG_DIR%`, sinon `~/.claude`). Ce fichier n'est pas documenté : un enregistrement
  illisible ou incomplet est signalé `unreadable`, jamais deviné. L'`id` d'une session est son id
  de session. Son `incarnation` est `pid:<pid>@<heure de création>`. Elle n'est `running` ou
  `idle` que si c'est prouvé : un processus vivant a ce pid et a été créé à cette heure, exactement
  sous Windows (`procStart`), à une minute près dans les deux sens de `startedAt` ailleurs (un saut
  d'horloge la déplace). Un enregistrement dont le `pidDomain` nomme une autre machine est
  `unknown`. Un enregistrement laissé par un plantage est `unreachable`, avec le
  `claude --resume <id>` à lancer. Un id de session que portent deux enregistrements figure dans
  `duplicates`. Tant que l'un d'eux est vivant, les autres disent « ne pas reprendre » ;
- **chaque ixmux donné par `--state`** et ses mandats. L'ixmux est `running`, `stopped`, ou
  `unreachable` (`ixmux.json` dit qu'il tourne et son processus a disparu). Chaque mandat est
  `running`, `done` ou `uncertain`, avec son `state`, sa `cause` et sa `reason` enregistrés. Leurs
  champs sont décrits dans [MCP.fr.md](MCP.fr.md), où `ixmux_inventory` en donne une vue réduite.

`last_activity` est la plus récente des heures de l'enregistrement et de la date de modification
du fichier de transcription, dont le contenu n'est jamais lu. `declared` garde ce que dit un
fichier, à part de ce qui est prouvé. Une borne n'est jamais silencieuse : au-delà de 256
enregistrements de session ou des 50 mandats terminés les plus récents par ixmux, `omitted` compte
ce qui a été laissé de côté (jamais un mandat non terminé). Un enregistrement de plus de 64 Kio, un
journal de plus de 64 Mio, ou tout ce qui n'est pas un fichier ordinaire (une FIFO n'est jamais
attendue) donne une entrée `unreadable`. `gaps` nomme ce qui n'est pas
lu : les sessions cloud, qu'aucun fichier local ne liste et dont aucune n'est montrée active, l'état
des volets wmux, et les services Windows (`ixctl resume check`). Code de sortie 0 dès que le
rapport est imprimé.

## Limites de cette tranche

- La session fictive (un fichier JSONL) et une vraie session Claude en stream-json, sans outils, sont les seuls adaptateurs.
- La vraie session a été déroulée à la main, en 20 tours de modèle au plus. La suite de tests ne lance jamais de vrai claude : elle utilise un faux (`claudestream/internal/fakeclaude`).
- Les journaux suivis repartent de l'offset 0 à chaque démarrage, sous une nouvelle époque, et le flux transporte leurs pages avec leurs lignes.
- L'incarnation d'un journal nomme son fichier (périphérique et inode sous Unix, volume et index de fichier sous Windows) et son premier enregistrement. Un système de fichiers peut donner à un nouveau fichier l'identité d'un fichier supprimé : Linux réutilise les inodes. Deux fichiers posés l'un après l'autre au même chemin ne se distinguent alors que par leur premier enregistrement, et un point de reprise pris sur un premier enregistrement incomplet peut être lu depuis 0 sur le nouveau fichier au lieu d'être refusé. Un journal vide ne nomme aucun fichier : tous les journaux vides ont la même incarnation, et un point de reprise pris sur l'un d'eux est lu depuis 0 sur le fichier présent au chemin, ce qui ne perd ni ne répète rien (`promoted_from` le signale).
- Une connexion n'est déclarée `slow` que lorsque sa file déborde, ce qui suppose une contre-pression TCP. Un client qui lit lentement mais suit le rythme n'est jamais coupé.
- Non testé ici :
  - `-race`, faute de gcc sur la machine : il ne tourne qu'en CI (`go-modules`, ubuntu) ;
  - les chemins Unix (mode 0600 du jeton, `flock`), qui ne tournent qu'en CI ;
  - Ctrl+C et Ctrl+Break, couverts seulement par le runbook ; les tests arrêtent ixmux par `/v1/shutdown`.
- Le runbook a été déroulé étape par étape depuis PowerShell, avec les mêmes chemins absolus. `start` et `set /p` sont des commandes internes de cmd.exe ; elles n'ont pas été exécutées ici.
