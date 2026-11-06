# ixmux

[English](README.md)

`ixmux` est un processus Go durable, unique, qui sert les paquets `journal` et `deliver` en HTTP et en WebSocket, en loopback uniquement. Un client qui perd sa connexion la retrouve et reprend sans exécuter un mandat deux fois. À l'arrêt propre, le processus ferme tout correctement ; au redémarrage, il reprend son état depuis le journal de commandes.

```
ixmux serve --addr 127.0.0.1:0 --state <dossier> [--journal nom=chemin ...] [--target nom=chemin ...]
            [--claude-target nom ...] [--claude-bin chemin]
            [--timeout 30s] [--max-conns 16] [--queue 256] [--poll 200ms]
ixmux version
```

## Construction

```
go -C C:\Users\spare\.cache\ix-ixmux\go\ixmux build -ldflags "-X main.version=<git describe>" -o C:\Users\spare\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe ./cmd/ixmux
```

`bin/` est ignoré par git. `ixmux.json` donne le SHA-256 de l'exécutable en cours.

## API (jeton Bearer, `Host` en loopback, pas d'`Origin`)

| Route | Rôle |
|---|---|
| `POST /v1/mandates` `{mandate_id, target, incarnation, text}` | Livre au plus une fois. La réponse donne `state` et `replayed`. Un refus revient sous la forme `{error, detail}`, avec 400, 404, 409, 413 ou 503. |
| `GET /v1/mandates/{id}` | L'état d'une livraison. |
| `POST /v1/mandates/{id}/reconcile` | Demande à la cible si une livraison incertaine est arrivée. |
| `GET /v1/status` | Les cibles et leurs incarnations, les livraisons (âge, `still_running`), les connexions et les bornes. |
| `GET /v1/journal/{nom}?from=&incarnation=&max_bytes=&cursor=` | Une page de `journal.ReadCursor`. Renvoie le `cursor` qu'elle rend : ixmux le signe en HMAC. Un curseur non signé ou mal signé est ignoré (`cursor_ignored`), et la lecture repart de `from`. |
| `GET /v1/stream[?since=<seq>&epoch=<époque>]` | WebSocket. Le premier message est `hello`. Viennent ensuite soit les événements manqués, soit un `gap` suivi de l'état complet (`state`). Puis les événements en direct : `mandate`, `journal`, `journal-reset`. |
| `POST /v1/shutdown` | Arrêt propre. |

Fonctionnement du flux :
- Les événements sont numérotés par `seq` au sein d'une `epoch`. L'époque change à chaque démarrage d'ixmux.
- Pour reprendre, passe `since=<seq du dernier événement>&epoch=<époque>`.
- Un `gap` a l'une de quatre raisons :
  - `epoch` : ixmux a redémarré ;
  - `ring` : les événements ne sont plus gardés ;
  - `ahead` : `since` est au-delà du dernier événement ;
  - `slow` : ta file a débordé. La connexion se ferme alors avec le code 1008 ; reprends avec `since=last_seq`.
- Un client doit relever sa limite de lecture WebSocket à `max_stream_message` (1 Mio).

## Fichiers de `<state>`

- `commands.jsonl` : le journal de commandes des livraisons. Tant qu'ixmux tourne, il tient `commands.jsonl.lock`.
- `ixmux.json` : adresse, PID, heure de démarrage, version, SHA-256 de l'exécutable, époque, et `state` (`running` ou `stopped`, puis `exit_code`).
- `ixmux.token` : le jeton Bearer. Il est réécrit à chaque démarrage et supprimé à l'arrêt propre. Sous Windows, sa DACL est protégée et n'accorde l'accès qu'à l'utilisateur courant ; ailleurs, son mode est 0600. Si ixmux ne peut pas le créer ainsi, il refuse de démarrer.

## Codes de sortie

| Code | Sens |
|---|---|
| 0 | Arrêt propre : plus aucun appel d'adaptateur, journal fermé, verrou libéré. |
| 1 | Échec du service. |
| 2 | Ligne de commande invalide. |
| 3 | Démarrage refusé : l'adresse n'est pas une IP de loopback, l'état est tenu par un autre ixmux, ou le fichier jeton a échoué. |
| 4 | Arrêt alors que des appels d'adaptateur tournaient encore : le `Close` de l'engine a refusé avec `calls-in-flight`. |

Un arrêt (Ctrl+C, Ctrl+Break ou `POST /v1/shutdown`) prend au plus le délai de livraison (`--timeout`) plus 10 s.

## Runbook (cmd.exe, chemins absolus)

La session de ce runbook est fictive : c'est le fichier JSONL `C:\Users\spare\.cache\ixmux-fake\session.jsonl`.

1. Démarre ixmux dans sa propre fenêtre :

   ```
   start "ixmux" C:\Users\spare\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe serve --addr 127.0.0.1:47011 --state C:\Users\spare\.cache\ixmux-fake\state --target fake=C:\Users\spare\.cache\ixmux-fake\session.jsonl
   ```

2. Vérifie l'état. Note l'`incarnation` de la cible `fake`.

   ```
   set /p IXMUX_TOKEN=<C:\Users\spare\.cache\ixmux-fake\state\ixmux.token
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

   `C:\Users\spare\.cache\ixmux-fake\state\ixmux.json` indique alors `"state": "stopped"` et `"exit_code": 0`.

**Retour arrière :** arrête ixmux (étape 4), puis supprime l'état et le fichier de la session fictive :

```
rmdir /s /q C:\Users\spare\.cache\ixmux-fake
```

Rien d'autre n'est écrit : aucun service, aucune clé de registre, aucun fichier hors de `--state` et des fichiers `--target`.

## Vraie session (`--claude-target`)

`--claude-target c` garde ouvert un enfant `claude -p` en stream-json, sans outils :

```
claude -p --input-format stream-json --output-format stream-json --verbose --tools "" --strict-mcp-config --session-id <uuid>
```

- Jamais `--bare`, `--dangerously-skip-permissions` ni `--permission-mode bypassPermissions`. `--setting-sources` n'est pas touché : tes hooks s'appliquent tels quels. `--permission-mode` n'est pas passé du tout : `default` n'est pas un choix valide de claude 2.1.318, donc l'enfant tourne dans ton mode configuré (`auto` ici), sans aucun outil sur lequel l'exercer.
- L'incarnation de la cible est le `session_id` du `system/init` de l'enfant. Le vrai claude n'envoie `init` qu'après un premier message utilisateur : ixmux envoie donc un message d'amorce (« ixmux : ouverture de la session. Réponds seulement : OK. ») à chaque démarrage. Cela coûte un tour de modèle par démarrage, et la cible est prête environ 28 s après le démarrage.
- Si `init` liste un outil, ou nomme une autre session, ixmux refuse de livrer à cette cible.
- Chaque mandat est un message utilisateur : `ixmux mandate_id: <id>`, puis la consigne de commencer par `ACK <id>`, puis le texte. La réponse porte `ack.receipt` (`<session>:<id>`), `ack.text` (4 Kio au plus) et `ack.ack_seen`.
- Le répertoire de travail de l'enfant est `<state>\claude-<nom>\`, qu'ixmux crée. Il contient `stream.jsonl` (la sortie standard brute de l'enfant, plus des lignes `{"ixmux":...}` pour le démarrage, l'envoi, le résultat et la sortie) et `stderr.log`. `stream.jsonl` est servi comme journal `claude-<nom>`.
- L'enfant tourne dans un Job Object kill-on-close : si ixmux meurt, tué ou non, l'enfant meurt avec lui. Un redémarrage lance un nouvel enfant, donc une nouvelle incarnation : un mandat qui était en livraison devient `uncertain`, sa réconciliation répond 409 `stale-incarnation`, et son renvoi répond 409.
- Un vrai tour dépasse souvent le `--timeout 30s` par défaut. Passe `--timeout 120s`.
- Un arrêt attend aussi jusqu'à 10 s que l'enfant sorte une fois son entrée standard fermée, puis tue le job.

Runbook (cmd.exe), avec un état jetable :

```
start "ixmux" C:\Users\spare\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe serve --addr 127.0.0.1:47012 --state C:\Users\spare\.cache\ixmux-claude\state --claude-target c --timeout 120s
```

ixmux n'écrit `ixmux.json` et ne répond qu'une fois l'enfant prêt (environ 30 s). `/v1/status` montre alors la cible `c` avec son `incarnation` et `"alive":true`. Envoie Hello comme à l'étape 3, avec `"target":"c"`. Arrête-le comme à l'étape 4.

**Piège :** un `start` lancé dans un `cmd /c` dont la sortie est capturée hérite du pipe. L'appelant attend alors la sortie d'ixmux, et de son enfant. Lance `start` depuis une vraie fenêtre de console.

**Retour arrière :** arrête ixmux, vérifie qu'il ne reste aucun enfant `claude` (`tasklist /fi "imagename eq claude.exe"`), puis supprime l'état jetable :

```
rmdir /s /q C:\Users\spare\.cache\ixmux-claude
```

La session Claude elle-même est gardée par claude sous ton profil, comme toute session `claude -p`. ixmux n'écrit rien d'autre.

## Limites de cette tranche

- La session fictive (un fichier JSONL) et une vraie session Claude en stream-json, sans outils, sont les seuls adaptateurs.
- La vraie session a été déroulée à la main, en 20 tours de modèle au plus. La suite de tests ne lance jamais de vrai claude : elle utilise un faux (`claudestream/internal/fakeclaude`).
- Les journaux suivis repartent de l'offset 0 à chaque démarrage, sous une nouvelle époque, et le flux transporte leurs pages avec leurs lignes.
- Une connexion n'est déclarée `slow` que lorsque sa file déborde, ce qui suppose une contre-pression TCP. Un client qui lit lentement mais suit le rythme n'est jamais coupé.
- Non testé ici :
  - `-race`, faute de gcc sur la machine ;
  - les chemins Unix (mode 0600 du jeton, `flock`), compilés mais pas exécutés ;
  - Ctrl+C et Ctrl+Break, couverts seulement par le runbook ; les tests arrêtent ixmux par `/v1/shutdown`.
- Le runbook a été déroulé étape par étape depuis PowerShell, avec les mêmes chemins absolus. `start` et `set /p` sont des commandes internes de cmd.exe ; elles n'ont pas été exécutées ici.
