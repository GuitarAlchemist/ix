# ixctl : `ixctl.v1` fige un noyau et garde son vocabulaire ouvert ; le document d'état désiré est figé sous `ixctl-desired/0` ; le journal reste provisoire

Statut : acceptée (2026-10-25)

Version anglaise : [`docs/adr/0014-ixctl-v1-and-desired-state-frozen.md`](../../adr/0014-ixctl-v1-and-desired-state-frozen.md).

## Contexte

Le plan de contrôle (`go/ixctl`, avec son client Rust et IXQL `crates/ix-ctl`) définit trois
formats sur lesquels d'autres programmes s'appuient : le protocole `ixctl.v1`, le document d'état
désiré et le journal. La page des portes à sens unique,
[`docs/fr/plans/2026-10-17-ixctl-one-way-doors.md`](../plans/2026-10-17-ixctl-one-way-doors.md),
a pesé chacun d'eux et nommé ce qu'il fallait corriger avant de rien figer. Ces corrections sont
faites :

- une époque d'instance (F1.1, F2.4, F3.1) ;
- une unité pour `ev.t` (F1.2) ;
- l'issue propre à chaque génération (F2.1) ;
- le même décodage strict sur les deux transports (F2.2) ;
- le routage à côté du document (F2.3).

Le contrat consommateur run/1 (#443, #444) repose sur les portes 1 et 2 : il ne peut pas être
annoncé avant qu'elles soient signées. Son annonce est une décision à part, avec sa propre preuve.

## Décision

### Porte 1 : `ixctl.v1`, option C, un noyau figé et un vocabulaire ouvert

#### 1.1 Garanties figées

**Accès.**

- Le plan n'écoute que sur la boucle locale.
- Chaque requête passe les mêmes contrôles :
  - son `Host` doit être celui de l'écoute ;
  - un `Origin`, s'il est présent, doit être l'origine locale de ce port ;
  - toute route sauf `GET /health` exige `Authorization: Bearer <token>`.
- Le fichier d'état donne `addr` et `token`.

**Transports.** Trois faces :

- du JSON sur HTTP sous `/v1/` ;
- les événements serveur sur `GET /v1/events` ;
- le WebSocket `GET /v1/ws`, dont le sous-protocole `ixctl.v1` est obligatoire.

`GET /health` répond sans jeton : `{ok, name, epoch, …}`.

**Corps des requêtes.** Ils sont lus strictement : une seule valeur JSON, rien après, 1 Mio au
plus. Un champ inconnu est refusé (400), de la même façon sur HTTP et sur le WebSocket.

**Erreurs.** Une erreur que le plan répond est `{"error": TEXTE}`, avec l'un de ces statuts :

- 400 : requête refusée, ou nom non déclaré ;
- 401 : le jeton ;
- 403 : l'hôte ou l'origine ;
- 429 : les places d'attente sont pleines ;
- 500 : une panne interne, comme un arrêt qui n'a pas convergé ;
- 502 : un pair injoignable ;
- 503 : arrêt en cours.

Le TEXTE est fait pour un humain et ne fait pas partie du contrat. Une route ou une méthode que le
plan ne sert pas reçoit le 404 ou le 405 du serveur HTTP lui-même, qui n'en fait pas partie non
plus.

**Trames WebSocket.**

- Une requête est `{id, op, …les champs de l'op}`, et chaque requête reçoit exactement une
  réponse `{re: id, ok, data | error}`.
- Les autres trames du serveur sont :
  - `{sub, env}`, un événement ;
  - `{sub, gap}`, un trou ;
  - `{sub, live: true}`, envoyée seulement à un abonné qui l'a demandée.
- La réponse d'un abonnement précède son premier événement.

**Enveloppe.**

```json
{"ch":"svc/pod-a","seq":17,"epoch":"8c1f…","origin":"local",
 "run":{"origin":"local","epoch":"8c1f…","seq":12},"t":1760990000123,
 "ev":{"kind":"stats","t":1760990000120,"data":{}}}
```

- `seq` est propre à une instance et repart de 1 à chaque démarrage.
- `epoch` nomme ce démarrage.
- `origin` est l'instance qui a émis l'enveloppe.
- `t` est l'heure de publication, en millisecondes Unix.
- `ev.kind` est une chaîne. `ev.t`, s'il est présent, est en millisecondes Unix sur l'horloge de
  l'émetteur. `ev.data` est la charge.
- Une enveloppe recopiée d'un pair porte en plus `src_seq` et `src_epoch`, ceux de la source. Ce
  sont les canaux `peer/<pair>/…`.

**Reprise et trous.**

- Un abonnement peut nommer `since`, et `epoch` à côté.
- En SSE, l'`id` est `epoch:seq`, et `Last-Event-ID` vaut `since`.
- Les raisons de trou forment un **ensemble fermé**. Un trou porte toujours `reason` et `from` (la
  première seq non reçue), plus `oldest` et `epoch` quand ils s'appliquent :
  - `since-too-old` : la reprise continue avec la plus ancienne enveloppe gardée ;
  - `overflow` : l'abonnement se termine ;
  - `epoch-changed` : la reprise continue avec la plus ancienne enveloppe gardée ;
  - `since-ahead` : seulement pour un `since` sans `epoch`, au-delà de la tête ; la reprise
    continue avec la plus ancienne enveloppe gardée.
- Un `since` qui nomme l'époque de ce démarrage est cru, même au-delà de la tête : pas de trou, et
  l'abonnement part en direct.

**Fédération.**

- Une op qui porte `peer` est transmise à ce pair, sur un seul saut.
- `subscribe` est toujours local. Les canaux d'un pair se lisent comme des copies sous
  `peer/<pair>/`.
- Sur `peer/<pair>/ctl`, une enveloppe `link.gap` marque la continuité. Sa `reason` est
  `epoch-changed`, `jumped`, ou l'une des quatre raisons de trou que le pair a répondues (`prefix`
  nomme alors l'abonnement).

**Canaux et événements.** Ces canaux gardent leur sens :

- `ctl` ;
- `svc/<nom>` ;
- `peer/<pair>/<canal>`.

Ces sortes `ctl` gardent leur sens et leurs champs de données :

- `service.starting`, `service.started`, `service.exited`, `service.restarting`,
  `service.stopped`, `service.failed` ;
- `converged`, `diverged` ;
- `peer.up`, `peer.down`.

**Ligne de service.**

- Une ligne stdout qui est un objet JSON avec un `kind` chaîne devient cet événement. Seuls
  `kind`, `t` et `data` sont gardés (D17, #448).
- Toute autre ligne stdout est `log {line}`, et une ligne stderr est `stderr {line}`.
- Un `kind` qui commence par le préfixe réservé `ixctl.` est réécrit en `log`.

#### 1.2 Extensions permises, sous le même nom

- De nouvelles ops, de nouvelles routes, de nouveaux canaux et de nouvelles valeurs de `ev.kind`.
  Ceux qui existent gardent leur sens.
- De nouveaux champs **optionnels** dans une requête. Une requête qui ne les porte pas garde
  exactement son sens d'avant ; c'est ainsi que D16 (#447) pourra venir.
- De nouveaux champs dans une réponse, une enveloppe ou les données d'un événement. **Un client
  ignore les champs qu'il ne connaît pas.**
- Les bornes qui ne sont pas des valeurs du contrat (la taille de l'anneau, les plafonds) peuvent
  changer. Un client lit celles que le statut annonce, et ne suppose pas les autres.

#### 1.3 Ce qui exige un nouveau nom : la règle de migration

Exigent un nouveau nom :

- retirer ou renommer un élément figé ;
- en changer le sens ou le type ;
- ajouter une raison de trou ;
- rendre obligatoire un champ optionnel ;
- refuser une requête qui était valide.

La migration se fait ainsi :

- un tel changement sort sous le sous-protocole `ixctl.v2` et les chemins `/v2/`, **servis à
  côté** de `ixctl.v1` (`websocket.Accept` prend une liste) ;
- un plan ne cesse de servir `ixctl.v1` que dans une version qui l'annonce, une fois qu'aucun
  client connu ne l'utilise plus.

#### 1.4 Hors de la porte 1 : reste provisoire

- La surface des sondes : les ops `probe.*`, les routes `/v1/probe*`, les canaux `probe/<nœud>`,
  les sortes `probe.*` sur `ctl`, et les `link.gap` des tuyaux de sonde (`live-start`,
  `dropped`).
- Les signets et la relecture : les ops `bookmark.*` et `/v1/bookmark*`.
- `ixctl bench`, les options de la ligne de commande, et les champs de `/health` autres que `ok`,
  `name` et `epoch`.
- `status.peers[]`, sauf ce que run/1 en dit.

### Porte 2 : le document d'état désiré, figé sous `ixctl-desired/0`

L'option B recommandée à l'époque le laissait provisoire. Cette décision va plus loin : elle fige
`/0` et laisse la suite à une nouvelle valeur de `schema`.

#### 2.1 Garanties figées

**Forme.**

```json
{"schema": "ixctl-desired/0", "services": {"pod-a": {"state": "running"}, "job": {"state": "stopped"}}}
```

`schema` est facultatif, et son absence vaut `ixctl-desired/0`. Chacun de ces cas est un 400, sur
les deux transports :

- une autre valeur de `schema` ;
- un champ inconnu, dans le document ou dans un service ;
- un `services` vide ;
- un service non déclaré ;
- un autre état.

**Partiel.** Un service que le document ne nomme pas reste tel qu'il est.

**Routage à côté.** `peer` voyage à côté du document (`?peer=` sur HTTP, le champ `peer` de la
trame sur le WebSocket), jamais dedans.

**`plan`.**

- Il est pur.
- Il répond `{generation, actions: [{op: start|stop, service}], noop}`.
- Son `generation` est la génération courante.

**`apply`.**

- Il incrémente la génération.
- Il répond tout de suite : `{generation, epoch, actions, …}`.
- Il agit ensuite, en arrière-plan.

**Générations.**

- La génération est un compteur propre au démarrage, qui part de 0 ; `epoch` la situe.
- L'issue d'une génération est la sienne propre, `converged` ou `diverged` sur `ctl`. Une
  génération plus récente ne la masque jamais.
- Le statut porte `generation`, `observed_generation`, `settled_generation`, `diverged` et
  `failed`, avec le sens que leur donne [`go/ixctl/README.fr.md`](../../../go/ixctl/README.fr.md).

#### 2.2 Extensions permises, sous `ixctl-desired/0`

- De nouveaux champs dans les **réponses** de `plan` et d'`apply`. Un client ignore ceux qu'il ne
  connaît pas.
- Rien dans le document lui-même : le décodage strict refuserait un champ nouveau, donc tout ajout
  au document est un nouveau schéma.

#### 2.3 La règle de migration

- Un changement du document sort sous `"schema": "ixctl-desired/1"`.
- Le plan accepte `/0` et `/1` côte à côte.
- Il ne cesse d'accepter `/0` que dans une version qui l'annonce, une fois qu'aucun client connu
  ne l'écrit plus.

### Porte 3 : le journal reste provisoire

Inchangé : un journal d'audit, ouvert à chaque démarrage par un en-tête
`{"kind":"boot","schema":"ixctl-journal/0","epoch":…}`. Il n'est jamais rejoué (D14, #445), et il
n'est figé qu'une fois qu'un programme le lit.

## Conséquences

- Un programme hors de `go/ixctl` et de `crates/ix-ctl` peut s'appuyer sur le noyau figé
  d'`ixctl.v1` et sur `ixctl-desired/0`. Ce que liste 1.4, et le journal, restent des brouillons.
- Un changement qui casse une garantie figée est servi à côté de l'ancienne (`ixctl.v2`,
  `ixctl-desired/1`), jamais à sa place.
- L'annonce de run/1 (#444) est une décision à part. Les portes 1 et 2 en sont la première
  condition ; les autres sont une validation indépendante (#454) sur le commit annoncé, et la liste
  des cas qu'aucun validateur indépendant n'a atteints.

## Signature

Signée par l'opérateur le 2026-10-25, sur le brouillon français de ce texte. Une omission du
brouillon a été corrigée d'après le code avant ce commit : sa liste d'erreurs n'avait pas le
statut 500 (`go/ixctl/plane.go`, `Stop` ; `hub.go`, `statusOf` ; `http.go`, un flux SSE qui ne
peut pas être vidé), et elle ne disait pas qu'une route ou une méthode que le plan ne sert pas
reçoit le 404 ou le 405 du serveur HTTP lui-même.
