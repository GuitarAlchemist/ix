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

- `commands.jsonl` : le journal de commandes des livraisons. Tant qu'ixmux tourne, il tient `commands.jsonl.lock`.
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

Un arrêt (Ctrl+C, Ctrl+Break ou `POST /v1/shutdown`) refuse aussitôt toute nouvelle livraison ou réconciliation (`engine-closed`), et attend celles en cours jusqu'à l'écriture de leur état final. Il prend au plus le délai de livraison (`--timeout`) plus 10 s, plus jusqu'à 10 s par cible claude.

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
- Un tour sans résultat à l'échéance de la livraison rend le mandat `uncertain`, mais le tour reste réservé : aucun autre mandat ne part tant qu'il peut encore répondre (ils sont refusés `target-busy`, avant toute écriture). L'appel de la livraison continue d'attendre (`still_running: true` dans `/v1/status`), et une réponse venue plus tard fait passer le mandat à `delivered` avec son ack. La réservation dure au plus `TurnMax`, 10 minutes. Au-delà, ixmux tue l'enfant par son Job et en lance un nouveau, sous une nouvelle incarnation, et la cible est libérée. Le mandat ne peut pas être renvoyé sous la nouvelle incarnation (409 `stale-incarnation`). Avec le vrai claude, cette relance coûte un tour d'amorce.
- Chaque incarnation peut ajouter au plus 256 Mio de sortie standard à `stream.jsonl` et 16 Mio à `stderr.log`. Au-delà de l'un ou l'autre plafond, l'enfant est tué et le tour en attente devient `uncertain` avec la raison `stream-too-large` (ou `stderr-too-large`). Une ligne de sortie standard de plus de 8 Mio est coupée dans le journal, et le tour en attente devient `uncertain`. Ces plafonds terminent l'incarnation : l'enfant n'est pas relancé.
- ixmux attend au plus 90 s (`InitWait`) l'`init` de l'enfant et la réponse à l'amorce ; au-delà, le démarrage échoue.
- Le répertoire de travail de l'enfant est `<state>\claude-<nom>\`, qu'ixmux crée. Il contient `stream.jsonl` (la sortie standard brute de chacun des enfants de la cible tour à tour, plus des lignes `{"ixmux":...}` pour le démarrage, l'envoi, le résultat, la sortie, l'arrêt et les cas ci-dessus) et `stderr.log`. `stream.jsonl` est servi comme journal `claude-<nom>`.
- L'enfant tourne dans un Job Object kill-on-close : si ixmux meurt, tué ou non, l'enfant meurt avec lui. Un redémarrage lance un nouvel enfant, donc une nouvelle incarnation : un mandat qui était en livraison devient `uncertain`, sa réconciliation répond 409 `stale-incarnation`, et son renvoi répond 409.
- Un vrai tour dépasse souvent le `--timeout 30s` par défaut. Passe `--timeout 120s`.
- Un arrêt attend aussi jusqu'à 10 s que l'enfant sorte une fois son entrée standard fermée, puis tue le job. `stream.jsonl` en garde la trace : `{"ixmux":"stop","step":"stdin-closed"}`, puis `exited` ou `killed` avec `after_ms`. Un enfant qui exécute encore ses hooks SessionEnd à 10 s est tué avec eux.

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
entre-temps. Il a trois outils : `ixmux_status`, `ixmux_submit` (un reçu en 2 s, même quand le tour
dure plus longtemps) et `ixmux_result` (lu dans le journal de commandes). Aucun outil n'arrête
ixmux. Le contrat, avec la commande exacte, les schémas et les erreurs : [MCP.fr.md](MCP.fr.md).

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
