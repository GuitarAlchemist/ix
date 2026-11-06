# ixmux

[English](README.md)

`ixmux` est un processus Go durable, unique, qui sert les paquets `journal` et `deliver` en HTTP et en WebSocket, en loopback uniquement. Un client qui perd sa connexion la retrouve et reprend sans exécuter un mandat deux fois. À l'arrêt propre, le processus ferme tout correctement ; au redémarrage, il reprend son état depuis le journal de commandes.

```
ixmux serve --addr 127.0.0.1:0 --state <dossier> [--journal nom=chemin ...] [--target nom=chemin ...]
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

## Limites de cette tranche

- Le seul adaptateur est la session fictive (un fichier JSONL). Aucune vraie session Claude n'est encore atteinte.
- Les journaux suivis repartent de l'offset 0 à chaque démarrage, sous une nouvelle époque, et le flux transporte leurs pages avec leurs lignes.
- Une connexion n'est déclarée `slow` que lorsque sa file déborde, ce qui suppose une contre-pression TCP. Un client qui lit lentement mais suit le rythme n'est jamais coupé.
- Non testé ici :
  - `-race`, faute de gcc sur la machine ;
  - les chemins Unix (mode 0600 du jeton, `flock`), compilés mais pas exécutés ;
  - Ctrl+C et Ctrl+Break, couverts seulement par le runbook ; les tests arrêtent ixmux par `/v1/shutdown`.
- Le runbook a été déroulé étape par étape depuis PowerShell, avec les mêmes chemins absolus. `start` et `set /p` sont des commandes internes de cmd.exe ; elles n'ont pas été exécutées ici.
