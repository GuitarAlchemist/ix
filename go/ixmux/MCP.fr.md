# ixmux mcp : l'adaptateur MCP

[English](MCP.md)

`ixmux mcp` est un serveur MCP local sur stdio. Il remet des mandats aux sessions que gère un
`ixmux serve`, et en lit les résultats. C'est le même binaire qu'`ixmux serve`, et il lui parle
par son API HTTP, en loopback seulement.

## Transport et commande

```
ixmux mcp --state <dossier>
```

- `<dossier>` est le dossier `--state` de l'`ixmux serve` à joindre. À chaque appel d'outil,
  l'adaptateur y lit l'adresse dans `ixmux.json` et le jeton dans `ixmux.token`. Il ne prend le
  jeton ni en argument ni en variable, et ne l'écrit nulle part : ni sur stdout, ni sur stderr, ni
  dans une réponse d'outil.
- JSON-RPC 2.0 sur stdin et stdout, un message par ligne (`\n` ; un `\r` avant est accepté).
  stdout ne porte que du JSON-RPC ; stderr ne reçoit que le message d'usage d'une ligne de commande
  fausse.
- Codes de sortie : 0 une fois stdin fermé ; 2 pour une ligne de commande fausse.

Pour l'enregistrer dans Claude Code (cmd.exe, avec l'état du runbook « vraie session » du README) :

```
claude mcp add ixmux -- %USERPROFILE%\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe mcp --state %USERPROFILE%\.cache\ixmux-claude\state
```

Démonstration sans client, depuis PowerShell, une fois `ixmux serve` lancé sur cet état :

```
'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"demo","version":"0"}}}',
'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ixmux_submit","arguments":{"target":"c","mandate_id":"demo-1","text":"Dis bonjour."}}}',
'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ixmux_result","arguments":{"mandate_id":"demo-1"}}}' |
  & "$env:USERPROFILE\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe" mcp --state "$env:USERPROFILE\.cache\ixmux-claude\state"
```

## Méthodes

| Méthode | Réponse |
|---|---|
| `initialize` | `protocolVersion` : celle demandée si l'adaptateur la parle (`2025-11-25`, `2025-06-18`, `2025-03-26`, `2024-11-05`), sinon `2025-11-25`. `capabilities.tools.listChanged` vaut faux. `serverInfo` est `ixmux` et sa version. Elle marche sans ixmux. |
| `notifications/initialized` | Une notification : pas de réponse. Toute autre notification est ignorée. |
| `ping` | `{}`. |
| `tools/list` | Les trois outils ci-dessous, avec leur `inputSchema`. |
| `tools/call` | `{content: [{type: "text", text}], structuredContent, isError}`. `text` est le JSON de `structuredContent`. |

Les erreurs de protocole sont des erreurs JSON-RPC, et l'adaptateur continue de lire :

| Code | Quand | `data.reason` |
|---|---|---|
| -32700 | une ligne qui n'est pas du JSON | |
| -32600 | une ligne de plus de 1 Mio (lue jusqu'au bout puis écartée), un lot, pas de `jsonrpc: "2.0"` ou pas de `method` | `line-too-long` pour la première |
| -32601 | une méthode inconnue | |
| -32602 | un outil inconnu, ou des `params` de `tools/call` illisibles | `unknown-tool`, `bad-arguments` |

## Outils

Un outil qui échoue répond `isError: true`, avec un `structuredContent` `{error, detail}`. `error`
est une raison typée : une de celles de l'adaptateur (ci-dessous), ou le code de refus d'ixmux tel
quel (`mandate-id-reuse`, `stale-incarnation`, `unknown-target`, `uncertain-needs-reconcile`,
`in-flight`, `target-busy`, `busy`, `unknown-mandate`, `text-too-long`, `engine-closed`…).
`detail` s'adresse à une personne : ne jamais l'analyser.

| Raison de l'adaptateur | Quand |
|---|---|
| `ixmux-unavailable` | pas d'`ixmux.json` ou pas de jeton dans `<dossier>`, `ixmux.json` dit `stopped`, son adresse n'est pas en loopback, ixmux ne répond pas, ou refuse le jeton |
| `bad-arguments` | un argument inconnu, un mauvais type, `target` absent |
| `bad-mandate-id` | un `mandate_id` vide, de plus de 128 octets, fait d'autre chose que des lettres, chiffres, `.`, `_` et `-`, ou qui commence par `.` |
| `ixmux-error` | une réponse d'ixmux que l'adaptateur ne sait pas lire |

### `ixmux_status`

Entrée : `{}`. Sortie :

```
{"version": "...", "epoch": "...",
 "targets": [{"name": "c", "kind": "claude", "incarnation": "<uuid>", "alive": true,
              "in_flight": [{"mandate_id": "m1", "state": "delivering", "still_running": true}]}]}
```

Métadonnées seulement : ni texte, ni réponse, ni chemin, ni jeton. `in_flight` liste les livraisons
de la cible qui sont `accepted` ou `delivering`, ou dont l'appel à l'adaptateur n'est pas revenu.

### `ixmux_submit`

Entrée : `{target, mandate_id, text, incarnation?}`, avec `target`, `mandate_id` et `text`
obligatoires (`text` : 64 Kio au plus, sinon `text-too-long`). Sans `incarnation`, l'adaptateur
prend l'incarnation courante de la cible dans `/v1/status`. **Après un redémarrage, redonne
l'incarnation d'origine pour un nouveau dépôt** : le même mandat sous une autre incarnation est un
autre mandat, refusé `mandate-id-reuse`.

Sortie, le reçu, en 2 s au plus, même quand le tour dure 10 minutes :

```
{"mandate_id": "m1", "target": "c", "incarnation": "<uuid>",
 "accepted": true, "state": "delivering", "replayed": false, "receipt": "<session>:m1"}
```

- `accepted: true` seulement une fois la ligne `accepted` du mandat écrite et synchronisée dans le
  journal de commandes d'ixmux : soit ixmux a répondu au POST, soit `GET /v1/mandates/{id}` le
  montre `accepted` ou `delivering`, ce que le moteur ne montre qu'après la synchronisation.
- `state` vaut `accepted` ou `delivering` (le tour continue), `delivered` (il a été rapide, ou
  c'est un rejeu : alors `replayed: true` et aucun nouveau tour), ou `refused` (la cible ne l'a
  certainement pas pris).
- `accepted: false, state: "submitted"` : ixmux n'a rien confirmé en 1,5 s. Rien n'est promis ;
  `ixmux_result` le dira, et refaire le même dépôt est sans risque (au plus une fois par
  `mandate_id`).
- `receipt` est le reçu de la cible, une fois livré.

Pourquoi ce choix : `POST /v1/mandates` attend la fin du tour. L'adaptateur l'envoie et le laisse
courir, lit `GET /v1/mandates/{id}` toutes les 25 ms, et répond à la première preuve, ou au bout
de 1,5 s. ixmux livre que l'adaptateur attende encore ou non : un adaptateur tué n'annule pas le
tour. C'est le plus petit changement qui tient la promesse : aucun mode nouveau côté serveur.

### `ixmux_result`

Entrée : `{mandate_id}`. Sortie :

```
{"mandate_id": "m1", "target": "c", "state": "delivered", "last": "delivering->delivered",
 "still_running": false, "reason": "...",
 "receipt": "<session>:m1", "text": "ACK m1\n...", "ack_seen": true, "truncated": false}
```

- `state` : `accepted`, `delivering`, `delivered`, `refused` ou `uncertain`. Un mandat
  `uncertain` ne se rejoue jamais : un nouveau dépôt est refusé `uncertain-needs-reconcile`.
- `receipt`, `text`, `ack_seen` et `truncated` apparaissent une fois livré. `text` fait 4 Kio au
  plus ; `truncated: true` dit que la réponse était plus longue et que sa fin manque.
- Le résultat est lu dans le journal de commandes d'ixmux (`GET /v1/mandates/{id}`) : il survit à
  un redémarrage de l'adaptateur et à un redémarrage d'ixmux.
- Un id inconnu donne `unknown-mandate`.

## Arrêt

stdin fermé : l'adaptateur finit l'appel en cours (2 s au plus pour `ixmux_submit`) et sort avec 0.
Un POST encore en cours est abandonné par l'adaptateur ; ixmux continue et finit la livraison.
ixmux continue de tourner dans tous les cas.

## Ce que l'adaptateur ne fait pas

- Il n'arrête pas ixmux : `/v1/shutdown` n'est pas un outil. Il ne réconcilie pas non plus un
  mandat incertain (`/v1/mandates/{id}/reconcile` n'est pas un outil).
- Aucun shell, aucun accès fichier hors d'`ixmux.json` et `ixmux.token`, aucun contrôle de
  sessions WMUX, aucun tunnel, aucun service, aucun réseau au-delà de l'adresse loopback
  d'`ixmux.json`.
- Il ne garde aucun état propre : tout ce qu'il répond vient d'ixmux.
- Ni ressources, ni prompts, ni sampling, ni notifications à lui.

## Limites

- Un `ixmux_submit` long retient les appels suivants : l'adaptateur répond à un appel à la fois.
- Les 2 s du reçu sont 1,5 s d'attente plus le temps d'une lecture HTTP en loopback ; une machine
  bloquée au-delà répond plus tard.
- Testé contre le faux claude seulement (`claudestream/internal/fakeclaude`) ; aucun appel de
  modèle réel.
