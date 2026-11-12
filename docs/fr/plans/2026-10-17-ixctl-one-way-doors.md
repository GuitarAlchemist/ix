---
title: "ixctl, tranche 1 : les trois portes à sens unique"
type: decision
status: signées le 2026-10-25 (porte 1 en option C, porte 2 figée sous ixctl-desired/0, porte 3 en option B) ; les décisions sont l'ADR-0014
date: 2026-10-17
branch: exp/ixctl à f77b31e (poussée, sans PR)
reversibility: réversibles aujourd'hui toutes les trois (rien n'est fusionné, aucun consommateur hors de cette branche) ; chacune devient irréversible au déclencheur nommé dans sa section
revisit-trigger: un programme hors de go/ixctl et crates/ix-ctl lit le protocole, envoie le document ou analyse le journal ; ou exp/ixctl est proposée à la fusion
---

# ixctl, tranche 1 : les trois portes à sens unique

Version anglaise : [`docs/plans/2026-10-17-ixctl-one-way-doors.md`](../../plans/2026-10-17-ixctl-one-way-doors.md).

La tranche 1 du plan de contrôle (commit `f77b31e` : un plan en Go dans `go/ixctl`, un client
Rust dans `crates/ix-ctl`) définit trois formats sur lesquels d'autres programmes vont
s'appuyer. Son message de commit les dit provisoires tant qu'ils ne sont pas signés. Cette page
soumet chacun d'eux à la décision de l'utilisateur :

1. l'enveloppe et le protocole `ixctl.v1` ;
2. le document d'état désiré ;
3. le format du journal.

Pour chaque porte, elle donne la forme exacte d'aujourd'hui, lue dans le code (numéros de ligne
à `f77b31e`), qui l'écrit et qui la lit, ce que la figer verrouille, les options, une
recommandation et la ligne à signer. Une dernière section pèse la prochaine tranche,
provisoire : une identité durable des exécutions de pipeline, indexée par un `run_id` choisi
par le client.

Rien ici ne modifie le code. Les constats marqués **F** viennent de la lecture du code. Aucun
n'a été reproduit par un test, et aucun n'est corrigé sur cette branche.

## Comment signer

Copiez la ligne de signature d'une porte, gardez ou changez son option, puis ajoutez un nom et
une date. Une porte non signée reste provisoire : n'importe quel commit peut la changer, et rien
hors de `exp/ixctl` ne devrait en dépendre.

## Résumé

| Porte | Lue aujourd'hui par (hors `go/ixctl`) | Recommandation | À régler avant tout gel |
|---|---|---|---|
| 1. `ixctl.v1` | `crates/ix-ctl`, réponses HTTP seulement | **B** : provisoire ; le nom du sous-protocole sert déjà de champ de version | F1.1 : un `since` d'un démarrage précédent est mal lu, sans un mot. F1.2 : `ev.t` n'a pas d'unité |
| 2. État désiré | écrit par `crates/ix-ctl` et par `ix.ctl.plan` / `ix.ctl.apply` en IXQL | **B** : provisoire derrière un champ `schema` ; le garder partiel ; abandonner la règle « une génération ultérieure remplace » | F2.1 : une génération ultérieure peut masquer un échec. F2.2 : strict en HTTP, laxiste en WebSocket |
| 3. Journal | aucun programme (trois tests le parcourent avec grep) | **B** : provisoire derrière un en-tête par démarrage ; journal d'audit seulement | F3.1 : générations et numéros de séquence se répètent d'un démarrage à l'autre dans un même fichier. F3.2 : il contient la porte 1 telle quelle |

Les portes sont liées. Le journal recopie octet pour octet les enveloppes de la porte 1 : la
porte 3 ne peut donc pas être figée avant la porte 1. Les générations de la porte 2 voyagent
dans la réponse de statut et dans les événements `ctl` de la porte 1. Un seul ajout sert les
trois : une **époque d'instance**, un identifiant choisi à chaque démarrage (F1.1, F2.4, F3.1).

---

## Porte 1 : l'enveloppe et le protocole `ixctl.v1`

### Forme actuelle

**Transports.** Un même plan sert trois faces, et chaque requête passe d'abord les mêmes
contrôles (`go/ixctl/http.go:44-59`, `:64-86`) :

- JSON sur HTTP : `GET /health` (sans jeton), `GET /v1/status` (`?peer=` lit le statut de ce
  pair, depuis la revue Codex du 2026-10-20 : une génération appliquée via un pair s'attend
  chez lui), `POST /v1/plan` et `POST /v1/apply` (le corps est le document ; `?peer=` nomme un
  pair, correctif F2.3), `POST /v1/stop` avec `{service, peer?}`, et `POST /v1/shutdown`. Un
  pair nommé mais vide est refusé sur tous les transports (400 `peer must not be empty`) : ce
  n'est pas « local ». Un refus
  est `{"error": "..."}` avec le statut 400, 401, 403, 500, 502 ou 503 (`http.go:134-136`,
  `plane.go:295`, `peers.go:54-62`).
- SSE : `GET /v1/events?ch=PREFIXE&since=SEQ[&epoch=EPOQUE]`, ou `Last-Event-ID`. Chaque
  événement est `id: <epoch>:<seq>` (un `<seq>` nu était la forme avant le correctif F1.1
  ci-dessous), `event: <ev.kind>`, `data: <enveloppe>`. Un trou est `event: gap` avec l'objet
  du trou, et `: ping` arrive après 15 s de silence (`http.go:138-207`).
- WebSocket : `GET /v1/ws`. Le sous-protocole `ixctl.v1` est exigé ; sans lui, la connexion est
  fermée pour violation de politique (`ws.go:14-29`, `client.go:16`). La limite de lecture du
  serveur est de 1 Mio, celle du client Go de 16 Mio (`ws.go:29`, `client.go:77`).

**Enveloppe.** Chaque événement, sur chaque transport, a cette forme (`hub.go:17-33` ;
l'encodage est construit à la main à `hub.go:131-146`) :

```json
{"ch":"svc/pod-a","seq":42,"epoch":"3f9a1c0b7d2e","origin":"local","t":1760700000000,
 "ev":{"kind":"stats","t":1760699999980,"data":{"nodes":[{"name":"src","emitted":20}],"final":false}}}
```

(Les données `stats` du pod sont abrégées ici ; chaque nœud porte dix champs,
`crates/ix-stream/src/lib.rs:125-134`. `ev.t` est en millisecondes Unix sur l'horloge du pod
lui-même, correctif F1.2.)

`seq` appartient à une instance et repart de 1 à chaque démarrage (`hub.go:100`, `:134`) ;
`epoch` nomme ce démarrage (12 caractères hexadécimaux tirés au lancement, correctif F1.1). `t`
est en millisecondes Unix au moment de la publication, ou le `t` du pair pour une enveloppe
miroir (`hub.go:109-130`). L'anneau de rejeu garde 4096 enveloppes (`hub.go:15`).

**Trames du client** (WebSocket). Une seule forme, `Request`, sert toutes les opérations
(`hub.go:50-61`) :

```json
{"id":7,"op":"subscribe","ch":"ctl","since":40}
{"id":8,"op":"apply","desired":{"services":{"pod-a":{"state":"running"}}}}
{"id":9,"op":"stop","service":"pod-a","peer":"b"}
{"id":10,"op":"unsubscribe","sub":7}
```

Les opérations sont `subscribe` et `unsubscribe` (`session.go:59-63`), puis `status`, `ping`,
`plan`, `apply`, `stop` et `shutdown` (`plane.go:109-133`). `status`, `ping`, `plan`, `apply` et
`stop` peuvent porter `peer` ; elles sont alors relayées d'un saut (`plane.go:136-155`). Un
`subscribe` avec `peer` est refusé : les canaux d'un pair se lisent par leurs miroirs
(`session.go:102-104`).

**Trames du serveur** (`session.go:72-77`, `:198-214` ; le client Go les lit comme une seule
union, `client.go:28-39`) :

```json
{"re":7,"ok":true,"data":{"sub":7,"head":41,"epoch":"3f9a1c0b7d2e"}}
{"re":8,"ok":false,"error":"unknown service \"pod-z\""}
{"sub":7,"env":{"ch":"ctl","seq":42,"epoch":"3f9a1c0b7d2e","origin":"local","t":1760700000000,"ev":{}}}
{"sub":7,"gap":{"reason":"since-too-old","from":12,"oldest":310,"epoch":"3f9a1c0b7d2e"}}
{"sub":7,"gap":{"reason":"epoch-changed","from":1,"oldest":1,"epoch":"3f9a1c0b7d2e"}}
```

Un trou est `since-too-old` (un rejeu demandait plus que ce que l'anneau garde ; l'abonnement
continue), `overflow` (un lecteur a pris du retard ; l'abonnement prend fin), ou, depuis le
correctif F1.1, `epoch-changed` (le `since` nommait l'époque d'un autre démarrage) ou
`since-ahead` (un `since` sans époque dépasse la tête de ce démarrage) ; pour ces deux-là
l'abonnement continue avec tout ce que l'anneau garde encore (`hub.go:43-52`,
`session.go:151-196`). Tout trou porte l'`epoch` courante (les deux premiers depuis la revue
Codex du 2026-10-20 : un lecteur SSE n'a pas de réponse d'abonnement, un trou était donc sa
seule façon de l'apprendre). Le rejeu attend de la place dans la file de la
connexion ; les événements en direct n'attendent jamais (`session.go:151-153`).

**Canaux et types d'événements.**

- `ctl` porte les événements du plan lui-même, chacun avec `service` quand il s'applique :
  `service.starting {restart}`, `service.started {pid, restarts, generation}`,
  `service.exited {pid, code, requested, generation}`, `service.stopped {pid?, code?, ms?}`
  (`generation` est l'application qui a lancé l'exécution, depuis la tranche des jobs),
  `service.restarting {attempt, in_ms}`, `service.failed {reason, restarts_in_window?, code?}`
  (`supervisor.go:121-124`, `:168-198`, `:300-348`) ; puis `converged {generation}` et
  `diverged {generation, service, reason}` (`plane.go:379`, `:387`) ; puis
  `peer.up {peer, addr, origin}` et `peer.down {peer, reason}` (`peers.go:78`, `:154`).
- `svc/<nom>` porte un événement par ligne de sortie du service. Une ligne de stdout qui est un
  objet JSON avec un `kind` de type chaîne passe telle quelle, sous la forme `{kind, t?, data}`,
  `t` étant en millisecondes Unix sur l'horloge du service, arrondi à l'entier (correctif
  F1.2) ; toute autre ligne de stdout devient `log {line}`, et une ligne de stderr `stderr {line}`
  (`supervisor.go:351-414`). Un événement `stats {nodes: [...]}` fixe aussi les `nodes` du
  service dans le statut (`supervisor.go:375-384`).
- `peer/<pair>/<canal>` porte les enveloppes `ctl` et `svc/` d'un pair, recopiées avec leur
  origine et jamais relayées plus loin (`peers.go:121-131`, `:187-197`).

**Réponses.** Le statut est `{name, epoch, generation, observed_generation, settled_generation,
diverged, failed[{generation, service, reason}], services[{name, desired, state, pid, restarts,
started_at, generation?, last_exit{code, at, generation?}, nodes}], peers[{name, state,
services}]}` (`plane.go:157-181` ; `settled_generation` et `failed` depuis le correctif F2.1
ci-dessous, `epoch` depuis F2.4 ; la `generation` d'un service et celle de son `last_exit` sont
l'application qui a lancé l'exécution courante ou la dernière, depuis la tranche des jobs ;
`last_exit.code` est le code de sortie, ou sous Unix 128 plus le signal qui a tué le processus,
la convention du shell). Le
plan est `{generation, actions[{op, service}], noop}`, l'application
`{generation, epoch, actions}`, l'arrêt `{stopped, generation, exit}` (`plane.go:202-216`,
`:392-396`).

**Accès.** Le plan n'écoute que sur la boucle locale (`http.go:209-220`). Il compare `Host` et
`Origin` à des listes autorisées et exige un jeton porteur ; il n'envoie jamais d'en-tête CORS
(`http.go:64-86`). L'adresse et le jeton viennent du fichier d'état `{name, addr, pid, token}`
(`main.go:178`, `client.go:228-233`). `ixctl up` imprime une ligne de disponibilité,
`{event: "ready", name, addr, pid, state_file, ms_to_ready}` (`main.go:129-136`).

### Qui l'écrit, qui la lit

- **Écrivent :** le plan (hub, sessions, SSE), et chaque service supervisé par ses lignes de
  stdout. Aujourd'hui, le seul service est le pod `ix-stream`
  (`crates/ix-stream/examples/pod.rs:7-12`).
- **Lit le côté WebSocket :** seulement `go/ixctl` lui-même, par ses liens de pair
  (`peers.go:93-180`) et `ixctl bench` (`bench.go:169-379`).
- **Lit le côté SSE :** les tests (`http_test.go`) et, depuis le 2026-10-20, `crates/ix-ctl`
  (`Ctl::events`, IXQL `ix.ctl.events`) : il lit `id`, `event`, `data`, les champs `ch`, `seq`,
  `epoch`, `origin`, `ev.{kind, t, data}` de l'enveloppe, et `reason`, `from`, `epoch` d'un
  événement `gap` (un trou se distingue d'une enveloppe par ces champs, pas par `event: gap`,
  qu'un événement de service de type `gap` porterait aussi). Les attentes ne s'en servent pas
  encore : elles sondent `/v1/status` toutes les 25 ms, et la mesure M1 de la passerelle
  chiffre l'écart sur un vrai démarrage (Windows, 2026-10-20) : `converged` atteint un lecteur
  du flux en direct 3,4 ms après la réponse de l'apply, le sondage à 25 ms le voit 12 à 28 ms
  après, selon l'endroit de sa période où il tombe (Linux en conteneur : 1,5 ms et 26,2 ms).
  Au plus une période de sondage, voilà ce qu'une attente par flux gagnerait.
- **Lit le côté HTTP :** `crates/ix-ctl`. Il lit `settled_generation` et `failed[]` quand ils
  sont présents, sinon `observed_generation` et
  `diverged.{generation, service, reason}`, `services[].{name, state, nodes}`, `generation` dans
  la réponse d'application et `error` dans un refus (`crates/ix-ctl/src/lib.rs:215-317`), ainsi
  que `addr` et `token` dans le fichier d'état (`lib.rs:142-164`). Ses attentes interrogent
  `/v1/status` à intervalles ; elles ne s'abonnent pas encore (`lib.rs:30-31`). Une attente
  limitée à un démarrage, sur un client construit depuis le fichier d'état, relit ce fichier
  quand le plan cesse de répondre (pas de connexion, ou 401) : un plan qui redémarre sous
  l'attente est signalé redémarré plutôt qu'injoignable (revue Codex, 2026-10-20). Une attente
  limitée à un pair continue d'interroger à travers un 502 ou un 503 (le lien de ce plan vers le
  pair est coupé pendant que l'un des deux redémarre) jusqu'à son échéance ; un 400 (pair
  inconnu) est définitif.

Aucun programme hors d'un seul binaire Go ne lit donc le côté WebSocket, et un client Rust lit
le côté SSE. Le contrat entre langages, aujourd'hui, ce sont les réponses HTTP et l'enveloppe
SSE.

### Constats

- **Corrigé (F1.1), cette tranche :** le hub tire une **époque d'instance** au démarrage
  (12 caractères hexadécimaux) et la pose sur chaque enveloppe (`epoch`), la réponse
  d'abonnement (`data.epoch`), `/health`, chaque id SSE (`<epoch>:<seq>`) et chaque trou. Un
  `subscribe` peut dire à quel démarrage appartient son `since` (`epoch` ; `Last-Event-ID` le
  porte sous la forme `<epoch>:<seq>`, ou `?epoch=` à côté de `?since=`). Un `since` d'un autre
  démarrage reçoit un trou `epoch-changed`, un `since` nu au-delà de la tête un trou
  `since-ahead`, tous deux nommant l'époque courante, puis tout ce que l'anneau garde encore. Un
  `since` nu sous la tête reste cru : rien ne permet de le distinguer, donc un client qui veut
  la garantie envoie l'époque. Les liens de pair envoient l'époque de la dernière enveloppe
  reflétée. Les lignes du journal portent désormais `epoch`, ce qui distingue les `seq` d'un
  démarrage de ceux du suivant (F3.1). Tests : `TestHubSinceFromAnotherBoot`,
  `TestSSEReplayWithLastEventID` (cas de l'id périmé), `TestHubEnvelopeShape`.
- **Corrigé (F1.2), cette tranche :** `ev.t` est en **millisecondes Unix sur l'horloge de
  l'émetteur**, ou absent s'il n'en a pas donné ; le `t` de l'enveloppe reste l'horloge du plan
  à la publication. Le superviseur arrondit le `t` d'un service au lieu de le tronquer, et
  écarte un `t` qui n'est pas un nombre fini (`supervisor.go`, `parseEventLine`). Les événements
  `ctl` du plan portent désormais un `t` (`plane.go`, `event`). Le pod fixe une fois la
  naissance de son bus sur l'horloge murale et écrit des millisecondes Unix
  (`crates/ix-stream/examples/pod.rs`) ; `bus.rs` n'est pas touché et compte toujours les
  secondes depuis la création du bus. `ixctl bench` lisait déjà `ev.t` comme des millisecondes
  Unix pour sa latence de diffusion (`bench.go:251-252`) : il reçoit maintenant un vrai nombre du
  pod. Test : `TestParseEventLine` (arrondi ; un non-nombre est écarté).
- **F1.1 Un `since` d'un démarrage précédent est lu comme un `since` de celui-ci.**
  `Subscribe` ne rejoue que si `since < head`, et part en direct si `since >= head`
  (`hub.go:174-186`). Après un redémarrage, un client qui reprend avec son dernier numéro de
  séquence, comme le fait une reconnexion SSE standard avec `Last-Event-ID`, est mal lu dans les
  deux cas. Sous la nouvelle tête, il rejoue depuis le mauvais endroit ; à la tête ou au-delà,
  il part en direct et manque ce que le nouveau démarrage a publié avant sa reconnexion. Dans
  les deux cas, rien ne le lui dit. Un client WebSocket pourrait comparer `since` à la `head` de
  la réponse d'abonnement (`session.go:117`) ; un client SSE ne reçoit aucune réponse. Les liens
  de pair évitent le piège en traitant un nouveau jeton comme une nouvelle instance
  (`peers.go:116-119`). Rien dans l'enveloppe ne dit à quel démarrage appartient un numéro de
  séquence.
- **F1.2 `ev.t` n'a pas d'unité définie.** Les événements du plan n'en portent pas
  (`plane.go:435-438`). Le `t` d'un service est lu comme un flottant puis tronqué en entier
  (`supervisor.go:406-409`). Le pod écrit les secondes écoulées depuis la création de son bus,
  arrondies à la milliseconde (`crates/ix-stream/src/bus.rs:19-26`) : 0,2 s devient 0, que
  `omitempty` supprime ensuite, et 1,4 s devient 1. Le `t` de l'enveloppe elle-même est en
  millisecondes Unix.
- **F1.3 Le nom dit v1, la version dit 0.1.0.** Le sous-protocole est `ixctl.v1` et les chemins
  commencent par `/v1/`, alors que `/health` annonce `version: "0.1.0"` (`http.go:16`,
  `:88-91`) et que le contrat n'est pas signé. Ailleurs dans l'écosystème, un brouillon s'écrit
  `/0` (`ix-recover-intent/0`) ou v0.1.x (CLAUDE.md, contrats entre dépôts).

### Ce que la figer verrouille

Les cinq champs de l'enveloppe et leur sens. `seq` comme seule clé de reprise, rattachée à un
démarrage qui n'a pas de nom (F1.1). Les deux raisons de trou, et la règle selon laquelle un
débordement met fin à un abonnement. Chaque nom d'opération et chaque forme de réponse
ci-dessus. Les noms de canaux `ctl`, `svc/` et `peer/<pair>/`. Les types `ctl` et leurs champs,
sur lesquels s'appuient à la fois l'`await` du plan (`plane.go:350-371`) et les attentes Rust.
Le contrat de ligne de stdout `{kind, t?, data}` pour chaque futur service. Le relais limité à
un saut.

### Options

- **A. Figer `ixctl.v1` tel quel.** Le moins cher aujourd'hui. F1.1 et F1.2 restent alors pour
  toujours, ou chacun exige une `v2`.
- **B. Rester provisoire derrière le champ de version qu'il a déjà.** La chaîne du
  sous-protocole et le préfixe `/v1/` sont des champs de version explicites : un plan ultérieur
  peut accepter `ixctl.v1` et `ixctl.v2` côte à côte, puisque `websocket.Accept` prend une liste
  (`ws.go:18-20`). Tant qu'il n'est pas signé, `ixctl.v1` veut dire « brouillon », et n'importe
  quel commit peut le casser. Corriger F1.1 (une époque d'instance dans l'enveloppe et dans la
  réponse d'abonnement, et un trou quand `since` dépasse `head`) et F1.2 avant tout gel.
  Facultatif : renommer le brouillon `ixctl.v0`, pour que le fil dise ce que dit cette page.
- **C. Figer un noyau et laisser le vocabulaire ouvert.** Après les corrections de F1.1 et F1.2,
  figer l'enveloppe, la sémantique `seq` / `since` / trou et l'union des trames. Déclarer les
  noms d'opérations, les noms de canaux et les valeurs de `ev.kind` extensibles : de nouveaux
  peuvent apparaître, les existants gardent leur sens.

### Recommandation

**B maintenant, puis C au déclencheur.** Aucun programme hors d'un seul binaire Go ne lit le
côté WebSocket ou SSE : un gel n'apporte rien aujourd'hui et rendrait F1.1 permanent. Figer le
noyau (C) quand le premier consommateur hors de `go/ixctl` s'abonne, ce que prévoit le
commentaire du client Rust lui-même (`lib.rs:30-31`), ou avant que `exp/ixctl` soit proposée à
la fusion, selon ce qui arrive en premier. Garder le nom `ixctl.v1`.

> **Signature de la porte 1 :** `ixctl.v1` reste provisoire (option B) jusqu'à ce qu'un
> consommateur hors de go/ixctl s'abonne ou qu'exp/ixctl passe en revue ; avant d'être figée
> selon l'option C, l'enveloppe reçoit une époque d'instance et `ev.t` une unité définie.
> Signé : l'opérateur, en option C au lieu de B (ci-dessous). Date : 2026-10-25

Signée en option C une fois ses deux préalables livrés : l'époque d'instance (F1.1) et l'unité de
`ev.t` (F1.2). Le noyau figé, les extensions qu'il permet et la règle de migration sont dans
l'[ADR-0014](../adr/0014-ixctl-v1-and-desired-state-frozen.md).

---

## Porte 2 : le document d'état désiré

### Forme actuelle

```json
{"schema": "ixctl-desired/0",
 "services": {"pod-a": {"state": "running"}, "pod-b": {"state": "stopped"}}}
```

(`go/ixctl/spec.go:161-175`)

- `schema` est facultatif et nomme ce format (depuis le correctif F2.2 ci-dessous). Un document
  qui nomme un autre schéma reçoit un 400 ; un document qui n'en nomme aucun est lu comme
  `ixctl-desired/0`.
- `services` associe un nom de service déclaré à `{state}`, et `state` vaut `running` ou
  `stopped`. Une table vide, un nom non déclaré ou un autre état donnent un 400
  (`plane.go:234-249`).
- **Le document est partiel.** « Les services qu'il ne nomme pas restent comme ils sont »
  (`spec.go:161-162`). Les appelants internes du plan en dépendent : le démarrage automatique
  n'applique que les services marqués `autostart` (`plane.go:83-91`), et `stop` applique
  `{nom: stopped}` (`plane.go:403`).
- **`peer` relève du routage, pas de l'état.** Il se trouvait dans le document ; depuis le
  correctif F2.3 il voyage à côté, comme le `peer` de la trame en WebSocket et `?peer=` en HTTP,
  et toute l'opération part vers ce pair (`plane.go`, `Request` et `forward` ; `http.go`,
  `desiredOp`). Un document qui porte `peer` est refusé comme champ inconnu.
- **Générations.** `apply` incrémente un compteur de l'instance et le renvoie aussitôt
  (`plane.go:298-313`) ; `plan` est pur et renvoie le compteur courant (`plane.go:251-262`). Le
  compteur vit en mémoire et repart de 0 à chaque démarrage (`plane.go:29`, `:41-47`). L'issue de
  chaque application est publiée sur `ctl`, sous la forme `converged {generation}` ou
  `diverged {generation, service, reason}` (`plane.go:373-388`). Le statut garde
  `observed_generation`, la plus haute génération qui a convergé, et `diverged`, la plus haute
  qui a divergé (`plane.go:374-386`). La règle (`plane.go:218-219`,
  `crates/ix-ctl/src/lib.rs:203-214`) : la génération g est terminée dès que
  `observed_generation >= g`, et elle a échoué tant que `observed_generation < g` et
  `diverged.generation >= g`. Un test fixe « une génération ultérieure qui converge remplace
  celle qui a échoué » (`plane_test.go:196-199`).
- **Décodage.** HTTP refuse les champs inconnus (`http.go:119-126`). Le chemin WebSocket utilise
  un simple `json.Unmarshal` (`session.go:54-58`) : les champs inconnus, y compris dans
  `desired`, sont écartés sans bruit.
- **Corrigé (F2.2), commit `09193b7` :** les deux transports passent par un même décodeur strict
  (`decodeStrict`) ; un champ inconnu reçoit le même refus en HTTP et en WebSocket. Enveloppe et
  schéma inchangés.
- **Corrigé (F2.2, le champ de version), cette tranche :** le document peut porter
  `"schema": "ixctl-desired/0"` (`spec.go`, `desiredSchema`) ; `validate` refuse une autre valeur
  par le même 400 sur les deux transports, et accepte toujours un document qui n'en nomme aucun.
  Le programme d'exemple `crates/ix-ctl/examples/pods.ixql` l'envoie, de sorte que le client Rust
  et le gate E2E l'exercent. Test : `TestUnknownFieldRefusedOnHTTPAndWS` (`ixctl-desired/9`
  refusé, `ixctl-desired/0` et l'absence de schéma acceptés, en HTTP et en WebSocket).
- **Corrigé (F2.3), cette tranche :** `peer` est sorti du document. En WebSocket, c'était déjà un
  champ de la trame ; en HTTP, c'est désormais `?peer=` sur `/v1/plan` et `/v1/apply`, le corps
  étant le document seul. Le client Rust prend `peer: Option<&str>` sur `plan` et `apply`, et
  `ix.ctl.plan` / `ix.ctl.apply` en IXQL prennent `peer:` à côté du document. Un document
  journalisé ou stocké ne dit plus où il a été envoyé. Tests : `TestFederationForwardsAndMirrors`
  (HTTP `?peer=b`, refusé puis accepté selon les règles de b), `TestUnknownFieldRefusedOnHTTPAndWS`
  (`peer` dans le document est un champ inconnu), et `a_peer_is_named_beside_the_document` côté
  client.
- **Corrigé (F2.4), cette tranche :** les générations sont rapportées au démarrage. La réponse
  d'application et le statut portent `epoch` (l'époque d'instance de F1.1). Côté client Rust,
  `wait_generation_of(epoch, g)` se termine aussitôt par `CtlError::Restarted` quand le statut
  annonce une autre époque, au lieu d'attendre que le nouveau démarrage atteigne le même
  numéro ; `wait_generation(g)` garde l'ancien comportement, sans portée, et un plan qui
  n'annonce pas d'époque est attendu par numéro. En IXQL, `ix.ctl.wait` prend `epoch:` à côté
  de `generation:`, et `pods.ixql` passe `applied.epoch`. Tests : `TestApplyConverges` (l'époque
  de la réponse et du statut est celle du plan), et côté client
  `a_wait_scoped_to_a_boot_ends_at_once_after_a_restart`.

### Qui l'écrit, qui le lit

- **Écrivent :** `crates/ix-ctl` (`Ctl::plan` et `Ctl::apply`, `lib.rs:175-191`) ; tout
  programme IXQL, par `ix.ctl.plan` et `ix.ctl.apply` (`crates/ix-ctl/src/ixql.rs:30-38`,
  `:81-96`, qui vérifie seulement que `services` est un enregistrement) ; le plan lui-même
  (démarrage automatique et `stop`) ; un pair qui relaie une opération.
- **Lisent :** `validate`, `plan` et `Apply` du plan (`plane.go:234-314`), et le journal, qui
  enregistre chaque document appliqué (porte 3).

### Constats

- **Corrigé (F2.1), cette tranche :** le statut gagne `settled_generation` (toute génération
  inférieure ou égale est terminée, convergée ou divergée ; il n'avance qu'au-delà d'une suite
  ininterrompue de générations terminées) et `failed` (chaque application divergée depuis le
  démarrage, par génération croissante). Le `wait_generation(g)` de `crates/ix-ctl` échoue quand
  `failed` nomme g, est terminé dès que `settled_generation >= g`, et attend sinon ; face à un
  plan plus ancien, il retombe sur l'ancienne règle. `observed_generation` et `diverged` sont
  inchangés. Tests : `TestALaterGenerationDoesNotHideAFailedOne` (plan) et
  `a_later_generation_does_not_hide_a_failed_one` (client). Le constat ci-dessous est gardé tel
  qu'écrit ; le cas concurrent (B converge pendant que A démarre encore) relève de la même règle
  mais n'a pas de test, `service.started` suivant le lancement à quelques millisecondes.
- **F2.1 Une génération ultérieure peut masquer un échec.** La règle de remplacement vient d'un
  état désiré total (l'`observedGeneration` de Kubernetes), où la génération g+1 reformule tout
  ce que demandait g. Ici, le document est partiel. Supposons que g démarre A et que g+1 démarre
  B. Si B converge d'abord, `observed_generation` passe à g+1 et `wait_generation(g)` répond
  « terminé » alors que A démarre encore ; si A échoue ensuite, `diverged` nomme g, mais
  `observed_generation >= g` est déjà vrai, et aucune attente fondée sur le statut ne le
  signalera jamais. En séquence, c'est pareil : g échoue sur A, g+1 ne démarre que B et
  converge, et `wait_generation(g)` répond « terminé ». Les événements `ctl` sont exacts, puisque
  `converged` et `diverged` portent leur propre génération ; le résumé du statut ne l'est pas.
  Lu, non reproduit : aucun test n'applique deux documents qui nomment des services différents.
- **F2.2 Strict en HTTP, laxiste en WebSocket.** Le même document avec un champ en trop donne un
  400 sur `POST /v1/apply`, et il est accepté, champ écarté, comme `apply` WebSocket. Comme HTTP
  est strict, un champ de version ajouté plus tard casse un client récent face à un plan plus
  ancien.
- **F2.3 Le routage se trouve dans l'état.** `peer` fait partie du document : un document
  stocké ou journalisé dit où il a été envoyé autant que ce qu'il veut. `stop` porte déjà `peer`
  à côté de `service` (`http.go:49-56`).
- **F2.4 Une génération ne veut rien dire d'un démarrage à l'autre.** Un client qui détient g
  d'avant un redémarrage attend jusqu'à son délai (`lib.rs:248-253`), ou, une fois que le
  nouveau démarrage atteint g, prend l'issue d'un autre document pour la sienne.
- **Les jobs, cette tranche (pas de nouvelle porte) :** un service déclaré avec
  `restart: never` est un job. Il court jusqu'à sa fin et le plan le laisse `stopped`,
  `last_exit` disant comment, son `desired` toujours `running` (`supervisor.go`, `onExit`). Le
  document ne change pas : `running` veut dire « démarre-le », donc un plan sur un job terminé
  dit de nouveau `start`, et réappliquer le même document le relance ; le « réappliquer ne
  démarre rien » de E7 vaut pour les services, pas pour les jobs terminés. La convergence de la
  génération qui l'a démarré dit « lancé », pas « terminé » (`await` s'arrête sur
  `service.started`) : `crates/ix-ctl` gagne donc `wait_exit` / `wait_exit_at` et, en IXQL,
  `ix.ctl.wait(service:, exit: true)`, qui répondent l'entrée du service dans le statut dès
  qu'il est `stopped` ou `failed` avec un `last_exit`. La sortie que répond `wait_exit` est la
  dernière que le plan a vue, qui entre une application et son lancement est encore celle de
  l'exécution précédente ; le plan marque donc chaque exécution de la génération de
  l'application qui l'a lancée (la `generation` d'un service et celle de son `last_exit` dans le
  statut, `generation` sur `service.started` et `service.exited`, conservée à travers les
  redémarrages de l'exécution), et `wait_exit_of(service, generation, epoch)` /
  `ix.ctl.wait(service:, exit: true, generation:, epoch:)` ne répondent que la sortie de cette
  exécution, celle d'une exécution antérieure restant en attente et un autre démarrage donnant
  un `Restarted` ; seule une génération dont l'application a eu une action `start` a lancé une
  exécution : le superviseur marque un service `starting`, estampillé de la génération de
  l'application, dès que celle-ci le demande (`want`), de sorte qu'un plan qui suit le voit en
  route et dit noop, comme pour un service `running` ou dans le délai d'un redémarrage, et
  `await` ne règle un démarrage que sur le `service.started` portant sa propre génération (revue
  Codex 11, P1 : avant, deux applications à la suite pouvaient revendiquer le même démarrage, la
  seconde convergeant sur l'exécution de la première). La revue Codex 12 a fermé les trois
  écarts restants : un `running` que le plan a trouvé en route n'est plus transmis au
  superviseur (un job terminé avant la lecture du want aurait été relancé sous la génération
  noop) ; un stop mis en file sur un service `failed` avant qu'un start ne le pré-marque
  `starting` finit quand même `stopped` ; et `service.failed` porte la génération de
  l'exécution qui a échoué, prise sous le verrou avec la transition, de sorte qu'`await` ne
  règle un démarrage que sur son propre échec. La revue Codex 13 a étendu la même règle aux
  arrêts et aux files : chaque `service.stopped` qu'une commande d'arrêt produit porte la
  génération de cet arrêt (la fin propre d'une exécution porte celle de l'exécution ; un
  arrêt du plan, aucune) et `await` ne règle un arrêt que sur son propre événement ; un noop
  `stopped` n'est pas transmis au superviseur non plus (seul `desired` suit le document) ; un
  service dont le processus a disparu alors qu'un want `running` est encore en file derrière
  affiche `starting`, estampillé de la génération de ce want, pour qu'aucun plan entre-temps
  n'annonce un démarrage que celui en file fera ; `Apply` vérifie l'arrêt du plan sous
  `applyMu`, que `Shutdown` prend pour fermer `quit` ; et le repli sur perte d'événements fait
  diverger un démarrage sur un service `failed` estampillé de sa génération, avec la raison de
  l'échec ; `crates/ix-ctl/examples/job.ixql` passe `applied.generation` et
  `applied.epoch`. Revue Codex 14 (un P2) : le service garde ses échecs par génération, si bien
  qu'un démarrage dont le `service.failed` a été perdu de l'anneau diverge sur son propre échec
  même quand un démarrage en file possède à nouveau l'état (le repli lisait l'état visible, qui
  disait `starting` pour le suivant, et attendait 60 s) ; une application oublie les échecs de
  ses démarrages quand elle se règle. La vérification
  E10 de la passerelle lance le vrai `ix pipeline run --json` (une étape `stats`, à travers la
  porte constitutionnelle Demerzel) comme job sous le plan et relit son NDJSON depuis `svc/job` ;
  un job dont le fichier de pipeline manque sort avec le code 10, sa génération converge quand
  même, et il reste `stopped` avec ce code. Tests : `a_job_is_waited_to_its_exit`,
  `job_ixql_runs_a_job_to_its_end`, `a_peers_job_is_waited_to_its_exit` (client) ; E10
  (passerelle).
- **Limites, cette tranche (pas de nouvelle porte ; la spec d'instance n'est pas l'une des
  trois) :** un service peut déclarer `limits: {"wall_ms": N}`. Une exécution qui dure au-delà
  est tuée par le plan, son arbre avec elle, et la sortie nomme la limite : `last_exit.reason`
  et `service.exited.reason` valent `wall_limit` (avec `wall_ms`), absents quand le processus
  s'est terminé de lui-même ou sur un arrêt demandé. La politique de redémarrage s'applique
  ensuite comme à tout échec : un job (`restart: never`) reste `stopped` avec cette raison,
  `on-failure` et `always` redémarrent avec recul, et l'exécution compte dans `max_restarts`.
  Le client Rust transmet `last_exit` tel quel, donc `Ctl::wait_exit*` et
  `ix.ctl.wait(exit: true)` portent la raison. `limits.memory_mb` borne la mémoire que tient
  l'arbre entier de l'exécution : sous Windows le job object émet une notification que le
  système garantit (`JobObjectNotificationLimitInformation` ; les autres messages du job sont
  au mieux, selon Microsoft) quand la charge de commit de l'arbre la dépasse, et un plafond dur
  un huitième au-dessus (au moins un Mio) refuse les commits entre-temps ; sous Linux le plan
  somme l'ensemble résident du groupe depuis /proc toutes les 100 ms (rien ne borne un groupe
  sans cgroup, donc un pic plus court qu'une lecture passe inaperçu). Dans les deux cas
  l'exécution est tuée, raison `memory_limit` (avec `memory_mb`). La sortie n'est classée
  qu'une fois les processus de l'arbre disparus et son moniteur ayant lu chaque message
  (`drain`), si bien qu'un processus mort d'un commit refusé avant la mise à mort reçoit la
  raison aussi, pour autant que ce message au mieux arrive (revue Codex 16, P1 : avant, une
  seule lecture non bloquante faisait la course avec le moniteur). Chaque sortie porte `peak_memory_bytes` quand l'OS l'a
  compté (le pic de commit du job sous Windows ; le pic rusage du meneur, ou le pic sommé du
  groupe sous surveillance, sous Linux) et `cpu_ms`, le temps processeur consommé (le total du
  job sous Windows ; le rusage du meneur avec les enfants qu'il a attendus sous Linux) : ce
  qu'une exécution a coûté, à côté de la façon dont elle s'est terminée. Les P2 de la revue
  16 : un lancement raté libère le port et son écouteur ; le `service.exited` d'un arrêt
  demandé porte aussi le coût ; le code de sortie et l'usage du meneur voyagent ensemble
  (`exitStat`), donc un arbre tué sans que sa sortie soit lue ne rapporte ni l'un ni l'autre ;
  une limite mémoire au-delà de ce que la plateforme adresse est refusée ; la E11 de la
  passerelle chronomètre depuis avant la requête d'application. Le vrai job
  `ix pipeline run` de la passerelle déclare `wall_ms: 120000, memory_mb: 2048`, comme tout job
  de l'usine est censé le faire, et E10 relit son coût (le pic est exigé ; `cpu_ms` est
  enregistré, car Windows compte le temps processeur par pas de 15,6 ms et une exécution si
  courte peut n'en recevoir aucun : revue Codex 17, P2). Revue 17 (un P1, cinq P2) :
  l'écouteur Windows lisait le port dans le champ de l'arbre pendant que `close` le remettait à
  zéro sur une autre goroutine (une course de données que `-race` aurait vue sur une ligne
  Windows) ; il prend désormais le handle par valeur, et `drain` poste un marqueur sur le port
  derrière les messages du job (les paquets d'un port se lisent dans l'ordre de leur dépôt),
  si bien que l'écouteur revient dès qu'il les a tous lus, au lieu d'attendre
  ACTIVE_PROCESS_ZERO (au mieux lui aussi) ou la seconde entière ; `close` draine d'abord, donc
  aucune lecture du port ne survit à son handle. Un arrêt demandé garde sa raison vide quoi que
  tienne le moniteur (c'est le plan qui a terminé l'exécution, pas une limite). Pour un Windows
  32 bits (aucune cible du dépôt) : la structure de notification est complétée aux 48 octets
  du SDK et sa taille vérifiée à la compilation, le plafond dur sature à l'espace d'adressage,
  et le message d'erreur de la spec ne déborde plus `int`. `limits.cpu_ms`, la troisième
  limite, borne le temps processeur, utilisateur et noyau, que tout l'arbre consomme : le plan
  le lit toutes les 100 ms dans la comptabilité même que la sortie rapporte en `cpu_ms` (celle
  du job sous Windows, les temps de `/proc/<pid>/stat` du groupe sous Linux, enfants attendus
  compris, en pas `AT_CLKTCK`), si bien qu'une exécution est tuée dès que le chiffre qu'elle
  rapporterait dépasse la limite, au plus une lecture en retard, raison `cpu_limit` (la limite
  voyage en `cpu_limit_ms` sur l'événement, `cpu_ms` étant ce qui a été consommé). La limite de
  temps propre au job object Windows (temps utilisateur seul, et elle termine le job elle-même)
  n'est pas utilisée, de sorte que la limite et la preuve concordent par construction ; la
  veille Linux alimente aussi `cpu_ms` pour tout le groupe désormais. La E13 de la passerelle
  fait tourner un Python borné à 300 ms de temps processeur (35 vérifications) ; son job
  déclare aussi `cpu_ms: 60000`. Revue Codex 18 (un P1, trois P2) : une exécution au-delà de sa
  limite de temps processeur qui se terminait avant qu'une lecture puisse la tuer (ou entre
  deux) passait pour une sortie propre ; `reap` classe désormais la sortie d'après le chiffre
  qu'elle rapporte, donc `cpu_limit` est posé dès que le temps consommé dépasse la limite
  (test : 60 ms de calcul sous 20 ms, terminé avant la première lecture). Sous Linux la veille
  garde le plus grand temps sommé, la somme baissant quand un membre meurt avant que son parent
  l'attende ; le README dit qu'un nom d'hôte est refusé pour `--addr` (seule une IP de boucle
  locale est acceptée). `limits.processes`, la quatrième limite, borne les processus que l'arbre
  tient à la fois : la même veille à 100 ms les compte (les processus actifs du job sous
  Windows, les membres vivants du groupe sous Linux), rapporte le plus grand compte en
  `peak_processes`, et l'exécution est tuée dès que le compte dépasse la limite, raison
  `process_limit` (la limite en `processes` sur l'événement) ; la sortie est classée d'après le
  pic aussi, comme pour le temps processeur. Sous Windows le job object refuse aussi tout
  processus au-delà d'un plafond un huitième au-dessus (au moins un ;
  `JOB_OBJECT_LIMIT_ACTIVE_PROCESS`, dont le message est au mieux, comme celui du plafond
  mémoire, et atteint le port de complétion, que l'arbre crée désormais pour une limite de
  processus comme pour une limite mémoire), si bien qu'une tempête de forks est bornée entre
  deux lectures ; sous Linux rien ne
  borne un groupe sans cgroup, une tempête a donc une période de lecture. L'arbre est relu une
  dernière fois au moment où il est tué (`kill` relève avant de terminer), si bien que le pic
  que le message du plafond interrompt quelques millisecondes après le spawn est rapporté : sans
  cela `peak_processes` lisait 3 sous une limite de 3, ou manquait. La E14 de la
  passerelle tient quatre enfants endormis sous une limite de trois (36 vérifications) ; le job
  déclare `processes: 16`. Revue Codex 15 (un P1, deux P2) : une sortie déjà en
  attente quand une limite se déclenche est la fin propre de l'exécution (pas `wall_limit` ;
  toujours `memory_limit`, le noyau l'a dit) ; une limite au-delà de ce qu'une `time.Duration`
  ou un compte d'octets contient est refusée ; le job mural de la passerelle est borné à 1,5 s
  pour que son pid soit lu avant la mise à mort sur un runner chargé. Revue Codex 19 (aucun
  P0/P1, quatre P2) : sous Windows le message du plafond plancher `peak_processes` au plafond
  (un enfant disparu entre le refus et le relevé de la mise à mort ne donne plus un pic sous la
  limite à côté de `process_limit`) ; la doc de `wait_exit` nomme `peak_processes` et dit
  qu'une raison lue dans un chiffre marque aussi une exécution terminée avant la lecture.
  Tests : `TestARunPastItsWallLimitIsKilledAndSaysSo`, `TestARunPastItsWallLimitRestarts`,
  `TestARunPastItsMemoryLimitIsKilledAndSaysSo`, `TestAnExitReportsItsPeakMemory`,
  `TestARunPastItsCpuLimitIsKilledAndSaysSo`, `TestARunUnderItsCpuLimitEndsOnItsOwn`,
  `TestARunPastItsCpuLimitThatEndsBeforeThePollSaysSoToo`,
  `TestARunPastItsProcessLimitIsKilledAndSaysSo`, `TestARunUnderItsProcessLimitEndsOnItsOwn`,
  la spec refuse une limite négative ou inconnue (Go) ;
  `a_run_killed_at_its_wall_limit_says_so_in_its_exit` (client) ; E11 à E14 (passerelle : un
  sleep Python borné à 400 ms, tué, son processus disparu, `restarts` à 0 ; un Python tenant
  256 Mio borné à 64 Mio, tué, pic rapporté ; un Python qui tourne sous 300 ms de temps
  processeur, tué, `cpu_ms` rapporté ; un Python qui lance quatre dormeurs sous trois, tué,
  `peak_processes` rapporté).

### Ce que le figer verrouille

La sémantique partielle et les deux états. `peer` dans le document. L'absence de champ de
version, avec un décodage HTTP strict qui en rend l'ajout difficile plus tard. La génération
comme compteur qui repart à chaque démarrage. La règle de remplacement, dans chaque client qui
attend sur le statut, y compris `ix.ctl.wait(generation:)` en IXQL.

### Options

- **A. Le figer tel quel.** Documenter F2.1 comme une limite : une attente sur le statut n'est
  sûre que pour un rédacteur unique dont les documents nomment tous les mêmes services.
- **B. Rester provisoire derrière un champ de version explicite.** Ajouter dès maintenant un
  champ facultatif `"schema": "ixctl-desired/0"`, tant qu'aucun plan hors de cette branche ne le
  refuserait, et décoder de la même façon sur les deux transports. Garder le document partiel.
  Avant tout gel, remplacer la règle de remplacement par l'issue propre à chaque génération (le
  statut garde les issues des générations récentes, ou les attentes lisent `converged` et
  `diverged` sur `ctl`), et rattacher les générations à une époque d'instance (F2.4).
- **C. Rendre le document total.** Chaque application nomme chaque service, et un service absent
  veut dire `stopped`. Une génération ultérieure remplace alors vraiment la précédente, et F2.1
  disparaît. Le coût : `stop` et le démarrage automatique deviennent une lecture-modification-
  écriture, en concurrence entre rédacteurs, et une application relayée doit reformuler tout
  l'état du pair.

### Recommandation

**B.** Garder le document partiel : `stop`, le démarrage automatique et le relais sont partiels
par nature, et C imposerait une concurrence à chaque rédacteur. Abandonner plutôt la règle :
l'issue d'une génération est son propre `converged` ou `diverged`, pas le plus grand numéro vu
jusque-là. Sortir `peer` du document (F2.3) vaut la peine dans le même changement, mais ce n'est
pas un préalable.

> **Signature de la porte 2 :** le document d'état désiré reste provisoire (option B) derrière
> `"schema": "ixctl-desired/0"`, et reste partiel (un service qu'il ne nomme pas reste comme il
> est) ; avant tout gel, l'issue d'une génération est son propre `converged` ou `diverged` (la
> règle « une génération ultérieure remplace » est abandonnée), et les générations sont
> rattachées à une époque d'instance.
> Signé : l'opérateur, figé sous `ixctl-desired/0` plutôt que laissé provisoire (ci-dessous). Date : 2026-10-25

Signée au-delà de l'option B une fois ses préalables livrés : l'issue propre à chaque génération
(F2.1), le même décodage sur les deux transports (F2.2), `peer` à côté du document (F2.3) et des
générations rattachées à une époque d'instance (F2.4). `ixctl-desired/0` est figé, et un
changement du document sort sous `ixctl-desired/1`, accepté à côté. Le texte est dans
l'[ADR-0014](../adr/0014-ixctl-v1-and-desired-state-frozen.md).

---

## Porte 3 : le format du journal

### Forme actuelle

Il y a un fichier par nom d'instance, `<répertoire d'état>/<nom>.journal.jsonl`. Il est ouvert
en ajout seul, avec le mode 0600, et conservé d'un démarrage à l'autre (`main.go:170`,
`plane.go:446-452`). Depuis la revue Codex du 2026-10-20, l'instance tient aussi un verrou
système sur `<répertoire d'état>/<nom>.lock` pendant toute sa vie, pris avant la vérification
du fichier d'état, pour que deux démarrages d'un même nom ne puissent pas ouvrir le journal
tous les deux et entrelacer leurs enregistrements sous un seul en-tête de démarrage ; le verrou
meurt avec le processus. Il contient trois sortes de lignes (`plane.go:48-60`, `:465-472` ; la
ligne `boot` depuis le correctif F3.1) :

```json
{"epoch":"3f9a1c0b7d2e","kind":"boot","name":"local","schema":"ixctl-journal/0","t":1760699999000}
{"actions":[{"op":"start","service":"pod-a"}],"desired":{"services":{"pod-a":{"state":"running"}}},"generation":1,"kind":"apply","t":1760700000000}
{"ch":"ctl","seq":3,"epoch":"3f9a1c0b7d2e","origin":"local","t":1760700000412,"ev":{"kind":"service.started","t":1760700000412,"data":{"generation":1,"pid":4242,"restarts":0,"service":"pod-a"}}}
```

- Un en-tête `boot`, écrit à l'ouverture du journal, avant toute autre ligne de ce démarrage.
  Il porte le nom du schéma, le nom de l'instance, l'époque d'instance de F1.1 et l'heure. Chaque
  `generation` et chaque `seq` en dessous, jusqu'à la ligne `boot` suivante, appartient à cette
  époque.
- Un enregistrement `apply`, écrit une fois la génération incrémentée et avant qu'un service
  n'agisse. Ses clés sortent triées, parce qu'il est sérialisé depuis une table Go, comme celles
  de l'en-tête `boot`.
- Chaque enveloppe dont le canal est exactement `ctl`, octet pour octet comme la porte 1
  l'envoie. Les lignes `svc/` et les enveloppes miroirs `peer/<pair>/ctl` ne sont pas
  journalisées.
- Chaque ligne est un seul `Write` sur un fichier `O_APPEND` (`plane.go:454-463`). Il n'y a pas
  de fsync, et les erreurs d'écriture sont ignorées. Les lignes `ctl` sont écrites pendant que
  le verrou du hub est tenu (`hub.go:147-150`).
- Rien ne relit le fichier. Au démarrage, le plan part de la génération 0 et applique le
  démarrage automatique comme génération 1 (`plane.go:73-92`). Le journal est une trace d'audit,
  pas un journal de reprise.

### Qui l'écrit, qui le lit

- **Écrit :** le plan seulement.
- **Lisent :** aucun programme. Trois tests le parcourent avec grep : pour `"kind":"apply"`,
  `"generation":2` et trois types `ctl` (`plane_test.go:90-99`), et pour vérifier qu'aucun jeton
  n'y figure (`peers_test.go:65-72`, `http_test.go:211-213`). Un test l'analyse ligne par ligne
  (`TestJournalBootHeaderTellsBootsApart`, correctif F3.1).

### Constats

- **Corrigé (F3.1), cette tranche :** l'option B ci-dessous, comme recommandé. Chaque démarrage
  ouvre sa section du fichier par `{"kind":"boot","schema":"ixctl-journal/0","name","epoch","t"}`
  (`plane.go`, `journal.boot`, appelé depuis `NewPlane` juste après l'ouverture du fichier). Rien
  d'autre ne change : les enregistrements `apply` et les enveloppes `ctl` sont comme avant, et
  les enveloppes portent déjà `epoch` depuis F1.1. Un lecteur lit désormais (époque, génération)
  et (époque, seq). Test : deux démarrages d'une même instance dans un même répertoire d'état,
  chacun appliquant la génération 1 ; le fichier contient deux lignes `boot` d'époques
  distinctes, chaque enveloppe `ctl` porte l'époque de la ligne `boot` au-dessus d'elle, et
  aucune ligne ne précède le premier en-tête (`TestJournalBootHeaderTellsBootsApart`).
  `TestPlanIsPure` attend maintenant exactement la ligne `boot` dans un journal qui n'a vu que
  des plans.
- **F3.1 Les identités se répètent dans un même fichier.** `generation` et `seq` repartent à
  chaque démarrage tandis que le fichier continue de grandir : `generation: 1` et `seq: 3`
  apparaissent une fois par démarrage, et seul l'ordre des lignes les distingue. Aucune ligne ne
  marque un démarrage.
- **F3.2 Le journal est la porte 1 sous un autre nom.** Ses lignes `ctl` sont les enveloppes de
  la porte 1 telles quelles : figer le journal fige aussi l'enveloppe et le vocabulaire `ctl`.
- **F3.3 Les deux sortes de lignes partagent le nom `kind` à des profondeurs différentes.** Un
  enregistrement `apply` a un `kind` au premier niveau ; une enveloppe a `ch` et `ev.kind`. Un
  lecteur les distingue par les clés présentes.

### Ce que le figer verrouille

Un format d'audit sans version et sans frontière de démarrage. L'enveloppe de la porte 1 (F3.2).
La promesse que seul `ctl` est conservé. L'absence de toute promesse de durabilité, dont les
lecteurs finiraient par dépendre.

### Options

- **A. Le figer tel quel.**
- **B. Rester provisoire derrière un champ de version explicite.** Ouvrir chaque démarrage par
  une ligne d'en-tête,
  `{"kind":"boot","schema":"ixctl-journal/0","name":"...","epoch":"...","t":...}`, qui porte
  l'époque d'instance de F1.1 ; ne rien changer d'autre. Les lignes se lisent alors comme
  (époque, génération) et (époque, seq).
- **C. En faire le journal de reprise du plan.** Rejouer le dernier état désiré au démarrage,
  conserver les générations d'un démarrage à l'autre, et faire un fsync de chaque enregistrement
  `apply`. Cela ferait d'ixctl un troisième stockage durable, à côté de celui d'ix-recover et de
  celui de Gaia.

### Recommandation

**B.** C'est la porte la moins chère à garder ouverte : aucun programme ne la lit. Laisser C de
côté : l'identité durable revient aux pièces existantes pesées plus bas, pas au superviseur de
services. Ne figer le journal qu'après la porte 1, et seulement quand un programme commence à
l'analyser (un importateur de preuves pour Gaia, une requête DuckDB dessus).

> **Signature de la porte 3 :** le journal reste provisoire (option B) : un journal d'audit
> jamais rejoué, ouvert à chaque démarrage par un en-tête
> `{"kind":"boot","schema":"ixctl-journal/0","epoch":...}` ; il n'est figé qu'après la porte 1,
> quand un programme l'analyse pour la première fois.
> Signé : l'opérateur, tel quel (option B). Date : 2026-10-25

### Si les trois sont signées comme recommandé

Ce sont des commits ultérieurs, pas celui-ci :

1. Une époque d'instance, choisie au démarrage : dans le fichier d'état, dans l'enveloppe ou la
   réponse d'abonnement, dans l'en-tête de démarrage du journal, et comme portée des générations
   (F1.1, F2.4, F3.1).
2. Un trou quand `since` dépasse `head` (F1.1).
3. Une unité définie pour `ev.t`, ou le nombre du producteur transmis tel quel (F1.2).
4. L'issue propre à chaque génération à la place de la règle de remplacement, dans le plan et
   dans `crates/ix-ctl` (F2.1).
5. Le champ `schema` du document, et le même décodage sur les deux transports (F2.2).

---

## Prochaine tranche, provisoire : une identité durable des exécutions de pipeline, indexée par un `run_id` du client

La question : quand un appelant lance une exécution de pipeline sous un identifiant qu'il
choisit, comment retrouver cette exécution après un arrêt brutal, en gardant son résultat et
sans jamais le publier deux fois ? Les générations d'ixctl n'y répondent pas : elles numérotent
des applications d'état de services, et seulement au sein d'un démarrage (F2.4). Voici les
pièces existantes, pesées avant toute proposition :

| Pièce | Ce qu'elle fait aujourd'hui | Peut-elle porter un identifiant d'exécution du client ? |
|---|---|---|
| L'exécuteur d'`ix-pipeline` | Exécute un DAG abaissé dans le processus, niveau par niveau, avec des threads au sein d'un niveau (`crates/ix-pipeline/src/executor.rs:115-123`). Sa seule mémoire est `PipelineCache`, indexé par `pipeline:<nœud>:<FNV-64 du texte Debug des entrées>` (`executor.rs:97-113`, `:309-316`) : une mémoïsation par étape, pas une adresse de contenu durable. ix-lock/v2 a un `run_id`, mais il est nouveau à chaque exécution et volontairement non reproductible (`lock.rs:37-40`, `:155-167`). | Non. Et le nom `run_id` est déjà pris par un identifiant propre à chaque exécution. |
| `StageGate` | Contrôle les arguments résolus de chaque étape juste avant que son outil s'exécute (`crates/ix-pipeline/src/gate.rs:18-29`). Le `lower()` simple utilise `AllowAll`, qui laisse tout passer (`gate.rs:31-39`, `lower.rs:35-39`) ; `ix pipeline run` abaisse plutôt avec le `ConstitutionGate` de Demerzel (`crates/ix-cli/src/verbs/pipeline.rs:546-547`). | Non. Il voit `(stage_id, skill, args)` et aucune exécution, et il intervient trop tard pour refuser une exécution avant l'ouverture d'un stockage. |
| La compilation d'`ix-ixql` | Transforme un programme en un `PipelineSpec` par sa forme, mais « compiler n'est pas abaisser » : `lower()` échoue avec `UnknownSkill` sur les appels hôtes comme `ix.ctl.*`, tant que personne n'a défini de correspondance entre fonctions hôtes et compétences du registre (`crates/ix-ixql/src/compile.rs:18-24`). | Non. Un programme IXQL ne peut pas être une exécution récupérable aujourd'hui. |
| Les nœuds SQL d'`ix-stream` | `query` et `Window` lancent un processus `duckdb` par fenêtre (`crates/ix-stream/src/sql.rs:46-66`) ; `ParquetSink` écrase son fichier sur place à la fin du flux (`sql.rs:148-203`). Ce sont des flux, pas des exécutions. | Non. Les sous-processus et les écritures sur place sont exactement ce qu'ix-recover tient hors d'une tentative. |
| `ix-recover` (branche `exp/ix-recover` à `3f2e935`, absente d'`exp/ixctl`) | `Store::begin(run_id, bindings)` répond `Fresh`, `Done` ou `Repaired`. Un identifiant d'exécution a au plus une tentative validée, par `hard_link` ; le verrou de propriétaire est `File::try_lock` ; les schémas sont `ix-recover-{intent,manifest,receipt}/0` (`crates/ix-recover/src/lib.rs:63-65`, `:294-354`). `ix pipeline run --run-id` lie la spécification et le binaire `ix`, n'admet que `stats` et `distance`, et rejoue la sortie standard enregistrée (`crates/ix-cli/src/verbs/pipeline.rs:18`, `:28`, `:587`, `:697-726` sur cette branche). Astra a approuvé le source de `2b83590` pour cet usage étroit, à titre provisoire (ix#386, toujours ouvert). | **Oui.** Pour les étapes pures, c'est la tranche. |
| Gaia | Je n'ai trouvé aucun suivi d'état des exécutions dans `gaia/src`. Ce qui est livré, c'est `gaia-lineage-receipt/1`, avec des ensembles de champs fermés (`src/lineage-receipt.mjs:82`). La distinction entre une identité de travail durable (Work Identity) et des identités de génération remplaçables (Generation Identity) vient de la conception de continuité de gaia#76 (ticket fermé le 2026-09-21 ; le brouillon du reçu n'est pas commité). La consigne d'Astra sur ix#386 : transmettre à Gaia l'identifiant logique du stockage et l'identifiant de tentative, jamais le `run_id` d'ix.lock imbriqué dedans. | Pas encore. C'est l'enregistreur naturel, pas le stockage. |

**Proposition : câbler, ne pas construire.** L'identité durable existe déjà. L'identifiant
d'exécution du client est l'identité de travail, l'identifiant de la tentative validée est
l'identité de génération, et le stockage d'ix-recover garde les deux. La tranche est un traceur
à travers ce qui existe :

1. Un appelant (Gaia, ou un script qui en tient lieu) lance
   `ix pipeline run --run-id <id> --runs-dir <dir>` sur une spécification `stats` / `distance`.
2. Il relit l'identifiant d'exécution, l'identifiant de tentative et l'empreinte du reçu, et les
   enregistre. Rien d'autre, et jamais le `run_id` d'ix.lock.
3. Un arrêt brutal avant la validation, puis une relance, donnent une nouvelle tentative et un
   seul enregistrement. Un arrêt brutal après la validation, puis une relance, donnent la même
   tentative et n'exécutent rien. Les tests d'ix-recover prouvent déjà le côté stockage
   (`crates/ix-cli/tests/pipeline_kill.rs` et `pipeline_recover.rs` sur `exp/ix-recover`). La
   nouvelle vérification : l'enregistreur finit avec exactement une entrée par identifiant
   d'exécution.

**Laissé de côté exprès :** une notion d'exécution dans ixctl (les services vivent longtemps et
redémarrent selon une politique ; une exécution ponctuelle demanderait un nouvel état dans la
porte 2, une raison de plus de la garder provisoire) ; un nouvel exécuteur ; un abaissement IXQL
inventé ; des nœuds SQL d'ix-stream dans une exécution récupérable ; le `run_id` d'ix.lock comme
clé.

**Conditions :** le feu vert de l'utilisateur pour le consommateur Gaia, et ix-recover qui reste
provisoire (ses schémas `/0`), comme les trois portes ci-dessus. `exp/ixctl` et `exp/ix-recover`
ne partagent aucun commit au-delà de `main` : `exp/ix-recover` doit donc être intégrée, ou
rebasée, d'abord.

---

## Provisoire : la surface des sondes (2026-10-20)

L'arbre des sondes (ops `probe.*`, tuyaux `probe/<nœud>`, événements `sample`, les `limits` de
`probe.kinds`) repose sur la porte 1 et hérite de son statut : provisoire, derrière `ixctl.v1`,
lu par rien hors de `go/ixctl`. Il n'ajoute aucun champ à la porte 2 (les sondes ne sont pas
dans le document d'état désiré, hypothèse sous la décision D1, ix#399) et rien à la porte 3
au-delà des événements `probe.*` que `ctl` journalise déjà comme les autres. Ses propres
décisions et mesures sont sur [`2026-10-20-ixctl-probes.md`](2026-10-20-ixctl-probes.md) ; il
devient à sens unique au même déclencheur que la porte 1, ou quand un programme hors du plan
crée des nœuds ou lit un tuyau.

## Provisoire : signets et rejeu (2026-10-20)

Les signets (ops `bookmark.*`, `bookmark` sur `subscribe` et `/v1/events`, le fichier
`<nom>.bookmarks.json`, schéma `ixctl-bookmarks/0`) reposent de la même façon sur la porte 1. Ils
n'ajoutent rien à la porte 2 ni à la porte 3 : un rejeu ne lit que l'anneau, et le journal reste
un journal d'audit que rien ne rejoue. La décision qui rouvrirait la porte 3 (un journal par canal
que le plan écrirait à la demande, rejoué au-delà de l'anneau) est consignée sur D9, ix#414, comme
non prise ; le fichier des signets est un second magasin dans le dossier d'état, provisoire, lu
par rien d'autre que le plan qui l'a écrit. Conception et mesures :
[`2026-10-20-ixctl-bookmarks.md`](2026-10-20-ixctl-bookmarks.md).

Les trois décisions, signées le 2026-10-25, sont consignées dans
l'[ADR-0014](../adr/0014-ixctl-v1-and-desired-state-frozen.md).
