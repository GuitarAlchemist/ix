# ixctl — le plan de contrôle d'IX

English: [`README.md`](README.md).

`ixctl` est un petit processus Go qui supervise les services qu'une spec déclare, les réconcilie
vers un document d'état désiré à la manière d'un Terraform local (`plan` dit ce qui changerait,
`apply` incrémente une génération et agit), borne chaque exécution (temps mur, mémoire) et
rapporte ce que chaque exécution a coûté. Il expose une seule face HTTP/WebSocket en boucle locale
et se fédère avec des instances pairs. Le client Rust et IXQL est
[`crates/ix-ctl`](../../crates/ix-ctl) (`ix.ctl.status`, `ix.ctl.plan`, `ix.ctl.apply`,
`ix.ctl.stop`, `ix.ctl.wait`, `ix.ctl.events`, et une capacité `ix.ctl.probe.*` par opération
de sonde).

Les trois formats qu'il définit (l'enveloppe `ixctl.v1`, le document d'état désiré, le journal)
sont **provisoires** : voir la page des portes à sens unique,
[`docs/fr/plans/2026-10-17-ixctl-one-way-doors.md`](../../docs/fr/plans/2026-10-17-ixctl-one-way-doors.md),
pour leur forme exacte, ce que figer chacun verrouillerait, et les rondes de revue qui les ont
façonnés. Rien hors de `go/ixctl` et `crates/ix-ctl` ne doit en dépendre tant qu'une porte n'est
pas signée.

## Construire et lancer

```sh
cd go/ixctl
go build -o ixctl .            # Go 1.27 ; dépendances : coder/websocket, x/sys
go test ./...                  # la suite unitaire (faux services, sessions en mémoire)
./ixctl up spec.json [--addr 127.0.0.1:0] [--name NOM] [--state DOSSIER] [--bookmarks disk|memory] [--probe-rate N] [--probe-tree-rate N]
./ixctl bench spec.json        # distributions de latence : start, stop, restart, RTT status, fan-out
./ixctl probes                 # les kinds de sonde de ce binaire et les limites du système (JSON, hors ligne)
./ixctl version
```

`up` imprime une ligne JSON quand il écoute, puis tourne jusqu'à SIGINT/SIGTERM ou
`POST /v1/shutdown` :

```json
{"event":"ready","name":"local","addr":"127.0.0.1:52011","pid":4242,"state_file":".../ixctl/local.json","ms_to_ready":3.1}
```

L'écoute est en boucle locale seulement (`--addr` doit être une IP de boucle locale, 127.0.0.0/8
ou `::1` ; un nom d'hôte, `localhost` compris, est refusé). Une instance par nom et par dossier
d'état : un fichier verrou est tenu toute la vie
de l'instance, et un fichier d'état `<nom>.json` (`{name, addr, pid, token}`, mode 0600) dit aux
clients où se connecter et avec quel jeton porteur. Le dossier d'état par défaut est
`%LOCALAPPDATA%\ixctl` sous Windows, sinon `$XDG_STATE_HOME/ixctl` ou `~/.local/state/ixctl`. Le
journal `<nom>.journal.jsonl` est à côté du fichier d'état.

## La spec

La spec est le monde déclaré d'une instance : les seuls services qu'elle pourra jamais lancer, et
les pairs avec lesquels elle se fédère. Les champs inconnus sont refusés.

```json
{
  "name": "local",
  "services": [
    {
      "name": "pod-a",
      "command": "bin/pod",
      "args": ["--nodes", "4"],
      "dir": ".",
      "env": {"RUST_LOG": "info"},
      "restart": "on-failure",
      "max_restarts": 5,
      "autostart": false,
      "limits": {"wall_ms": 120000, "memory_mb": 2048, "cpu_ms": 60000, "processes": 16}
    },
    {
      "name": "job",
      "command": "bin/ix",
      "args": ["pipeline", "run", "--file", "job/ix.yaml", "--json"],
      "restart": "never"
    }
  ],
  "peers": [
    {"name": "b", "state_file": "state/b.json"}
  ]
}
```

| Champ | Sens |
|---|---|
| `name` | Nom d'instance ou de service : `^[a-z0-9][a-z0-9-]{0,62}$`, unique dans la spec. |
| `command`, `args`, `dir`, `env` | Quoi lancer. `command`, `dir` et `state_file` des pairs relatifs se résolvent contre le dossier du fichier de spec. Sous Windows, une commande sans extension qui n'existe pas telle quelle est essayée avec `.exe`. |
| `restart` | `never`, `on-failure` (défaut) ou `always`. Un **job** est un service avec `restart: never` : il court jusqu'à sa fin et le plan le laisse `stopped`, son `last_exit` disant comment. |
| `max_restarts` | Redémarrages permis dans une fenêtre de 60 s (défaut 5) ; au-delà, le service est `failed` (« too many restarts »). Le délai double de 200 ms à 10 s. |
| `autostart` | Démarrer le service au démarrage du plan, en génération 1. |
| `limits.wall_ms` | Durée maximale d'une exécution, depuis son lancement. 0 ou absent : pas de limite. |
| `limits.memory_mb` | Mémoire maximale que tout l'arbre de processus peut tenir, en MiB. 0 ou absent : pas de limite. |
| `limits.cpu_ms` | Temps processeur maximal, utilisateur et noyau, que tout l'arbre peut consommer, en millisecondes. 0 ou absent : pas de limite. |
| `limits.processes` | Nombre maximal de processus que l'arbre peut tenir à la fois. 0 ou absent : pas de limite. |
| `peers[].state_file` | Le fichier d'état d'une instance pair ; le lien y lit son adresse et son jeton et se reconnecte avec délai croissant (200 ms à 5 s). |
| `peers[].probes` | Les nœuds de sonde du pair dont cette instance veut les tuyaux en miroir sous `peer/<nom>/probe/<nœud>`, par nom exact (D4, ix#402). Un tuyau ne traverse que tant que le pair marque le nœud `export: true` ; le statut dit ce qu'est chaque nom demandé. Vide ou absent : aucun tuyau de sonde ne traverse. |

### Limites

Une exécution qui dépasse une limite est tuée, tout son arbre avec elle, et sa sortie nomme la
limite : `reason` vaut `wall_limit`, `memory_limit`, `cpu_limit` ou `process_limit` sur
`service.exited` et dans `last_exit`, à côté de la limite posée sur l'événement (`wall_ms`,
`memory_mb`, `processes`, ou `cpu_limit_ms`, puisque `cpu_ms` est ce que l'exécution a
consommé). La politique de redémarrage
s'applique ensuite comme à tout échec : un job ainsi terminé reste `stopped`, un service redémarre
avec délai.

- **Le temps mur** est une minuterie armée au lancement.
- **La mémoire**, sous Windows, est une *limite de notification* du job object à `memory_mb` (le
  message que le noyau garantit de livrer) plus un plafond dur un huitième au-dessus (au moins un
  MiB) : le kill du plan sur la notification applique la limite, le plafond borne de combien un
  seul commit peut la dépasser. Sous Linux le plan lit toutes les 100 ms l'ensemble résident du
  groupe de processus dans `/proc`. La sortie d'une exécution n'est classée qu'une fois son
  moniteur drainé (une seconde au plus), donc une sortie à l'heure n'est jamais lue comme une
  limite.
- **Le temps processeur** est lu toutes les 100 ms dans la même comptabilité que la sortie
  rapporte en `cpu_ms` (temps utilisateur + noyau du job sous Windows ; les temps de
  `/proc/<pid>/stat` du groupe sous Linux, enfants attendus compris) : l'exécution est tuée dès
  que le chiffre qu'elle rapporterait dépasse la limite, au plus une lecture en retard. La
  limite de temps propre au job object Windows ne compte que le temps utilisateur et termine le
  job elle-même ; le plan ne s'en sert pas.
- **Les processus** sont comptés par la même lecture toutes les 100 ms (les processus actifs du
  job sous Windows, les membres vivants du groupe sous Linux) et rapportés en `peak_processes` ;
  une exécution dont le compte dépasse la limite est tuée. Sous Windows le job object refuse
  aussi tout processus au-delà d'un plafond un huitième au-dessus de la limite (au moins un),
  si bien qu'une tempête de forks est bornée entre deux lectures, et le message du noyau sur ce
  refus (au mieux, comme celui du plafond mémoire) nomme la limite lui aussi. L'arbre est relu
  une dernière fois au moment où il est tué, si bien qu'un pic que la veille n'a pas eu le temps
  de lire (le message du plafond arrive quelques millisecondes après le spawn) est quand même
  rapporté.

Chaque sortie porte aussi ce que l'exécution a coûté : `peak_memory_bytes` (Windows : le pic de
commit du job ; Linux : le plus grand du pic sondé et du `ru_maxrss` du meneur), `cpu_ms`
(Windows : temps utilisateur + noyau du job ; Linux : le plus grand du temps sondé du groupe et
du `ru_utime + ru_stime` du meneur) et, sous une limite de temps processeur ou de processus,
`peak_processes`, quand la plateforme a pu les mesurer. Une limite lue dans un chiffre rapporté
(`cpu_ms`, `peak_processes`) s'applique aussi à ce chiffre à la sortie : une exécution qui l'a
dépassé et s'est terminée avant qu'une lecture puisse la tuer porte quand même la raison.

## État désiré, plan et apply

Un document d'état désiré nomme l'état où chaque service devrait être. Les services qu'il ne
nomme pas sont laissés tels quels. Il dit *ce qui* est voulu, jamais *où* : le routage (`peer`)
voyage à côté, si bien qu'un document journalisé ne contient aucune adresse.

```json
{"schema": "ixctl-desired/0", "services": {"pod-a": {"state": "running"}, "job": {"state": "stopped"}}}
```

- `plan` est pur : il renvoie `{generation, actions: [{op: start|stop, service}], noop: [...]}`
  contre l'état observé, avec la génération qu'un apply obtiendrait.
- `apply` incrémente la génération, journalise le document, renvoie aussitôt
  `{generation, epoch, actions}` et agit en arrière-plan. Le statut rapporte `generation`
  (dernière appliquée), `observed_generation` (dernière dont toutes les actions ont été vues
  jusqu'au bout) et `settled_generation` (dernière dont chaque service a atteint son état
  désiré) ; `diverged` nomme la génération que le plan a abandonnée, avec le service et la
  raison, et `failed` garde les échecs de lancement récents par génération, pour qu'un apply
  ultérieur ne cache pas un échec antérieur.
- `epoch` nomme le démarrage auquel ces générations appartiennent. Un client qui attend une
  génération devrait nommer l'epoch aussi, pour qu'un redémarrage du plan termine l'attente au
  lieu de laisser passer la génération de même numéro du nouveau démarrage pour l'ancienne.
- `stop` arrête un service hors bande et répond une fois son arbre disparu :
  `{stopped, generation, exit}`. Un arrêt demandé ferme stdin, attend deux secondes, puis tue.

Le `desired` d'un job reste `running` après sa fin : appliquer de nouveau le même document le
relance ; le client attend la fin de l'exécution *qu'une génération a lancée*
(`last_exit.generation`), pas la dernière sortie vue par le plan.

## HTTP et WebSocket

Chaque requête passe les mêmes contrôles avant toute route : le `Host` doit être celui de
l'écouteur (DNS rebinding), un `Origin`, s'il est présent, doit être l'origine locale de ce port,
et toute route sauf `GET /health` exige `Authorization: Bearer <jeton>` (le jeton du fichier
d'état). Les corps sont lus strictement : une valeur JSON, aucun champ inconnu, rien après, un MiB
au plus.

| Route | Fait |
|---|---|
| `GET /health` | `{ok, name, version, epoch, uptime_ms, ops}` ; sans jeton. |
| `GET /v1/status[?peer=NOM]` | La réponse de statut (ci-dessus), ou celle de ce pair. |
| `POST /v1/plan[?peer=]`, `POST /v1/apply[?peer=]` | Corps : un document d'état désiré. |
| `POST /v1/stop` | Corps `{"service": NOM, "peer"?: NOM}`. |
| `POST /v1/shutdown` | Arrête chaque arbre et chaque sonde, puis les transports ; le fichier d'état est retiré. |
| `GET /v1/probes[?peer=]` | Les kinds de sonde et les limites du système (`probe.kinds`, ci-dessous). |
| `GET /v1/probe[?peer=]` | L'arbre des sondes (`probe.list`). |
| `POST /v1/probe/create[?peer=]` | Corps : un nœud (`probe.create`). |
| `POST /v1/probe/start`, `stop`, `touch`, `remove` | Corps `{"node": NOM, "peer"?: NOM}`. |
| `GET /v1/bookmark[?peer=]` | Les signets (`bookmark.list`, ci-dessous). |
| `POST /v1/bookmark/set` | Corps `{"bookmark": NOM, "since"?: SEQ, "epoch"?: E, "note"?: TEXTE, "peer"?: NOM}`. |
| `POST /v1/bookmark/remove` | Corps `{"bookmark": NOM, "peer"?: NOM}`. |
| `GET /v1/events?ch=PREFIXE&since=SEQ&epoch=E`, ou `?ch=PREFIXE&bookmark=NOM` | Server-sent events : `id: epoch:seq`, `event: <kind>`, `data: <enveloppe>`. `Last-Event-ID` vaut `since` ; `bookmark` rejoue depuis un signet à la place (pas à côté de `since`). Un lecteur qui tombe de l'anneau reçoit un événement `gap` et le flux se termine. |
| `GET /v1/ws` | Le WebSocket `ixctl.v1` (ci-dessous). |

Les erreurs sont `{"error": "..."}` avec 400 (requête refusée, nom non déclaré), 401 (jeton),
403 (host ou origin), 502 (pair hors ligne).

### `ixctl.v1` sur le WebSocket

Les trames du client sont des requêtes : `{id, op, ...}` avec `op` parmi `status`, `ping`,
`plan`, `apply`, `stop`, `subscribe`, `unsubscribe`, les sept ops `probe.*` et les quatre ops
`bookmark.*` (ci-dessous), et les champs de l'op (`desired`, `service`, `peer`, `ch`, `since`,
`epoch`, `sub`, `probe`, `node`, `bookmark`, `note`). Chaque requête reçoit une trame de réponse
`{re: id, ok, data | error}`. Les abonnements sont posés en ligne (leur réponse précède leur
premier événement) ; les autres ops courent de leur côté, pour qu'un stop lent ne retienne jamais
la connexion.

Les événements arrivent en enveloppes :

```json
{"ch":"svc/pod-a","seq":17,"epoch":"8c1f…","origin":"local","t":1760990000123,
 "ev":{"kind":"stats","t":1760990000120,"data":{"nodes":[...]}}}
```

`seq` compte depuis 1 à chaque démarrage ; `epoch` nomme le démarrage ; `origin` est l'instance
émettrice. Un abonné avec `since` rejoue depuis l'anneau (4096 enveloppes par instance) et reçoit
`{sub, gap}` quand c'est impossible : `since-too-old`, `overflow`, `epoch-changed`, `since-ahead`.

### Canaux et événements

| Canal | Porte |
|---|---|
| `ctl` | Les événements du plan lui-même : `service.starting`, `service.started` (`pid`, `restarts`, `generation`), `service.exited` (`pid`, `code`, `requested`, `generation`, et quand posés `reason` + `wall_ms`/`memory_mb`/`cpu_limit_ms`/`processes`, `peak_memory_bytes`, `cpu_ms`, `peak_processes`), `service.restarting` (`attempt`, `in_ms`), `service.stopped`, `service.failed` (`reason`), `converged` / `diverged` (`generation`), `peer.up` / `peer.down`. |
| `svc/<nom>` | La sortie du service. Une ligne stdout qui est un objet JSON avec un `kind` chaîne est transmise comme cet événement (`t` en ms Unix sur l'horloge du service ; le tableau `nodes` d'un événement `stats` devient les `nodes` du service dans le statut). Toute autre ligne stdout est un événement `log` `{line}` ; les lignes stderr sont des événements `stderr`. |
| `probe/<nœud>` | Le tuyau d'un nœud de sonde : des événements `sample` (ci-dessous). En miroir chez un pair seulement si ce nœud est créé avec `export: true` et que la spec de ce pair le nomme (D4, ix#402). |
| `peer/<nom>/<canal>` | Les canaux `ctl` et `svc/` d'un pair, en miroir, plus les tuyaux `probe/<nœud>` que cette instance a demandés dans `peers[].probes` tant que le pair les exporte. `subscribe` est toujours local ; les ops portent `peer` pour être relayées. |

**Tuyaux de sondes à travers le lien (D4, ix#402, tranchée le 2026-10-21 ; S2 ix#408, tranche 1).**
Ce qui traverse est l'intersection de deux déclarations : l'enfant marque un nœud exportable à la
création (`export: true`, immuable, faux par défaut : une disponibilité, pas une demande) et la
spec du pair côté parent nomme exactement les nœuds voulus (`peers[].probes`). Le lien s'abonne au
tuyau de chaque nœud voulu par son nom de canal exact (`probe/cpu` ne couvre pas
`probe/cpu-extra`), depuis le `probe.list` du pair au raccordement puis depuis les
`probe.created` / `probe.removed` en miroir ; `status.peers[].probes` dit ce qu'est chaque nom
demandé : `mirrored`, `not-exportable`, `opening` (abonnement en cours) ou `absent` (pas de tel
nœud, ou lien tombé). Un retrait oublie la séquence de reprise : un nom ré-exporté après une
incarnation non exportée ne rejoue jamais les échantillons de celle-ci. L'agrégation
vit chez le lecteur : en V1 le lecteur est le consommateur de `peer/<nom>/probe/<nœud>` (IXQL
`ix.ctl.events`), aucun nœud local ne nomme une source distante. L'export est une règle de
fédération, pas une barrière de confidentialité : l'abonnement direct à un tuyau est inchangé.
Pas encore dans cette tranche : la provenance source sur l'enveloppe miroir (`src_seq`,
`src_epoch`), les trous distants relayés comme marqueur de continuité, les files bornées par
classe et le rattrapage borné, l'instantané lié à un curseur ; voir la résolution sur ix#402.

`ctl` porte aussi le cycle de vie des sondes : `probe.created` (`node`, `kind`, `parent`, `export` quand posé),
`probe.started` (`expires_in_ms` quand il y a un bail), `probe.stopped` et `probe.removed` avec
une `reason` (`requested`, `expired`, `removed`, `parent-removed`, `shutdown`).

## Sondes

Une sonde est un échantillonneur qui tourne **dans le processus du plan**, sur un minuteur, et
publie ce qu'il lit sur son propre canal ; un arbre de sondes filtre et multiplexe ces
échantillons vers une racine. Le plan porte ses **kinds** de sonde, versionnés et compilés dedans
(`nom@version`), et connaît ses propres limites : `probe.kinds` liste les deux, chaque refus nomme
la borne touchée, et `probe.list` montre l'usage. La conception, les décisions qu'elle suppose et
celles encore ouvertes sont sur la page des sondes,
[`docs/fr/plans/2026-10-20-ixctl-probes.md`](../../docs/fr/plans/2026-10-20-ixctl-probes.md), et
le travail est suivi sous l'epic ix#397. La surface est provisoire, comme les trois portes.

| Kind | Champs | Lit |
|---|---|---|
| `heartbeat@1` | `n` | Son propre compte de ticks, depuis 1. Le kind des tests et des démonstrations. |
| `cpu@1` | `busy_pct`, `cores` | La part d'occupation du processeur de l'hôte sur le dernier intervalle (`GetSystemTimes` ; `/proc/stat`). Le premier intervalle n'émet rien. |
| `mem@1` | `total_bytes`, `avail_bytes`, `used_pct` | La mémoire physique de l'hôte (`GlobalMemoryStatusEx` ; `/proc/meminfo`). |
| `mux@1` | (ceux de ses enfants) | Pas d'échantillonneur : il porte ce que ses enfants lui passent. |

Un **nœud** est une instance d'un kind dans l'arbre :

```json
{"name": "hb", "kind": "heartbeat@1", "every_ms": 20, "parent": "agg", "ttl_ms": 1500,
 "filter": {"where": [{"field": "n", "op": ">", "value": 2}], "keep": ["n"], "every": 2}}
```

`every_ms` est l'intervalle d'échantillonnage (un `mux@1` n'en a pas). `parent` nomme un nœud
`mux@1`, en marche ou arrêté ; un échantillon émis par un nœud va sur son propre tuyau
`probe/<nom>` et, quand son parent est en marche, remonte au parent, qui applique son propre
filtre et le publie sur `probe/<parent>` estampillé de `src` (le nom de l'enfant). La propagation
est synchrone : rien n'est mis en file et rien n'est perdu en montant ; la seule perte est celle
de l'anneau, et c'est un `gap` que le lecteur voit. Un `filter` s'applique avant qu'un
échantillon soit émis, dans cet ordre : `where` (chaque terme doit tenir ; `op` parmi `==`, `!=`,
`<`, `<=`, `>`, `>=` ; `value` est un nombre, une chaîne ou un booléen, un booléen avec `==` ou
`!=` seulement, tout le reste refusé à la création ; un champ absent ou un type qui ne correspond
pas vaut faux), `every` (un échantillon sur N de ceux qui sont passés), `keep` (les champs
gardés ; `node`/`kind`/`seq`, le `src` d'un mux et l'`error` d'un échantillon raté toujours). Un échantillon est `{node, kind, seq, src?, ...champs}` ; une erreur de lecture devient
`{node, kind, seq, error}`, compte dans les `errors` du nœud et atteint le tuyau quel que soit le
filtre (`where` et `every` ne s'y appliquent pas). Les enregistrements d'un nœud atteignent son
tuyau dans l'ordre de `seq`, et `probe.stopped` est le dernier mot de son tuyau jusqu'au prochain
start ; les événements de cycle de vie de `ctl` viennent dans l'ordre des changements d'état qu'ils
rapportent, quel que soit l'entrelacement des opérations.

`ttl_ms` est un **bail** : compté depuis `probe.start`, renouvelé par `probe.touch`, et quand il
s'épuise le nœud est retiré avec tout son sous-arbre (`probe.removed`, raison `expired`). Un nœud
sans `ttl_ms` n'a pas de bail et `probe.touch` le refuse. `probe.remove` emporte aussi le
sous-arbre (les descendants sont arrêtés et retirés avec la raison `parent-removed`) ;
`probe.stop` arrête un nœud et laisse ses enfants tourner (leurs échantillons restent sur leurs
propres tuyaux). Un nœud qu'un remove a réclamé refuse un start, un touch ou un nouvel enfant
jusqu'à sa disparition ; un nœud dont le stop attend encore son dernier tick refuse un start
jusqu'à ce que `probe.stopped` soit sorti. `POST /v1/shutdown` ferme l'admission (plus de create
ni de start) et arrête chaque nœud en marche, raison `shutdown`.

| Op | Champs | Réponse |
|---|---|---|
| `probe.kinds` | | `{kinds: [{kind, version, fields, doc}], limits, version}` |
| `probe.create` | `probe` (un nœud) | `{node, kind, state: "stopped", depth}` (`depth` : les mux au-dessus) |
| `probe.list` | | `{nodes: [{name, kind, state, every_ms, ttl_ms, seq, errors, bytes, children, rate_per_s, max_rate_per_s?, export?, parent?, filter?, expires_in_ms?}], count, max_nodes, rate_per_s, max_rate_per_s, tree_rate_per_s}` |
| `probe.start` | `node` | `{node, state: "running", expires_in_ms?}` (sans effet sur un nœud en marche) |
| `probe.stop` | `node` | `{node, state: "stopped"}` |
| `probe.touch` | `node` | `{node, state, ttl_ms, expires_in_ms?}` (le temps restant, sur un nœud en marche) |
| `probe.remove` | `node` | `{node, removed}` (nœuds retirés, sous-arbre compté) |

Les limites que `probe.kinds` rapporte (et que `ixctl probes` imprime hors ligne) : `max_nodes`
256 dans l'arbre d'un plan, `max_depth` 8 (une feuille sous au plus autant de mux),
`min_every_ms` 10, `max_every_ms` 3 600 000, `max_ttl_ms` (ce qu'une durée contient), `ring` 4096
(celui du hub, partagé par tous les canaux), `max_where` 16 termes, `max_keep` 32 champs. Une
création au-delà de l'une d'elles est refusée avec son nom (`every_ms 5 is under min_every_ms
10`). Chaque op `probe.*` peut être relayée à un pair (`peer`) ; le cycle de vie `probe.*` du pair
revient en miroir sur `peer/<nom>/ctl`, ses tuyaux non (D4). Depuis IXQL, chaque op est une
capacité qui porte son nom (`ix.ctl.probe.create(probe: {…})`, `ix.ctl.probe.start(node: "agg")`,
…) et un tuyau se lit avec `ix.ctl.events(channel: "probe/<nœud>")` ; il n'y a pas d'attente sur
un échantillon (D7, ix#405). Le programme est
[`crates/ix-ctl/examples/probe.ixql`](../../crates/ix-ctl/examples/probe.ixql) : une sonde CPU
sous un mux, dix enregistrements, l'arbre retiré.

Depuis D10 (ix#416), les limites portent aussi le **budget** du plan : `max_rate_per_s` 1 000, ce
qu'un sous-arbre (un mux et ses descendants, ou une feuille à la racine seule) peut déclarer
émettre en enregistrements par seconde, et `tree_rate_per_s` 2 500 pour l'arbre entier ;
`ixctl up --probe-rate N --probe-tree-rate N` les règle (les défauts font environ un quart de
cœur en extrapolant M2). Le débit déclaré est le pire cas, sommé au `create` quel que soit l'état
des nœuds : 1000 / `every_ms` par feuille. Le filtre ne l'abaisse pas : un échantillonneur en
échec émet un enregistrement d'erreur à chaque intervalle, et les erreurs contournent `where` et
`every`. La somme est exacte (en rationnels, ni flottants ni entiers tronqués) : douze feuilles à
12 ms remplissent 1 000 à l'unité et sont admises ; 32 feuilles à 31 999 ms dépassent un budget de 1
d'un cheveu et sont refusées. Le budget se règle avant tout nœud et gèle dès le premier. Le `max_rate_per_s` d'un mux resserre le budget auquel son
sous-arbre est tenu et n'élargit jamais celui de son parent ni du plan ; il est immuable (retirer
et recréer) ; une feuille n'en porte pas. Une création qui franchirait une borne est refusée à
l'avance avec le nom du sous-arbre et la borne (`subtree "agg" would declare 1100 records/s with
"cpu-10"; its max_rate_per_s is 1000`), comme `max_nodes` : rien n'est ralenti ni arrêté pour
budget, et aucun événement n'est publié. `probe.list` dit l'usage : le `rate_per_s` de chaque
nœud, le `max_rate_per_s` de chaque mux, le `rate_per_s` de l'arbre face à `tree_rate_per_s`, et
`bytes`, les enregistrements encodés que chaque nœud a émis, mesurés et non bornés. CPU et RSS
sont ceux du processus et ne sont pas attribués aux sondes : un seul processus ne sait pas les
séparer honnêtement.

## Signets et rejeu

Un **signet** nomme un point de la séquence de l'instance : `{name, epoch, seq, t, note}`. Le
`seq` du hub est partagé par tous les canaux, donc un seul signet couvre les tuyaux de tout un
arbre de sondes, ceux des services et le cycle de vie `ctl` au même instant : un rejeu depuis lui
redonne ce que chaque lecteur a vu après ce point, dans l'ordre où c'est arrivé. Poser `before`,
lancer l'arbre, poser `after` quand quelque chose semble anormal, puis rejouer `probe/` (les
tuyaux) ou `` (tout) depuis `before` : ce que le pipeline a vu, après coup.

Le plan possède les signets. `ixctl up --bookmarks disk` (par défaut) les garde dans
`<nom>.bookmarks.json` à côté du journal, réécrit en entier à chaque changement (un fichier
temporaire, puis un renommage ; mode 0600), donc ils survivent à un redémarrage ;
`--bookmarks memory` les garde pour ce démarrage seulement. Un fichier que le plan ne peut pas
lire comme des signets est refusé au démarrage, pas vidé. La source du rejeu est l'anneau (4096
enveloppes) : un signet plus vieux que lui répond d'abord `since-too-old`, puis ce que l'anneau
tient encore, jamais un rejeu silencieusement partiel ; un signet d'un démarrage antérieur répond
d'abord `epoch-changed`, puis l'anneau du nouveau démarrage. Rejouer depuis le journal est une
décision ultérieure (D9, ix#414 ; porte 3 intacte).

| Op | Champs | Réponse |
|---|---|---|
| `bookmark.set` | `bookmark`, `since?`, `epoch?`, `note?` | `{bookmark, epoch, seq, t, note, head}`. Sans `since` le point est la tête (« ici ») ; `since` doit être au plus la tête et `epoch`, s'il est donné, celui de ce démarrage. Un nom existant est déplacé ; une `note` donnée remplace la sienne (une note vide l'efface), pas de `note` la garde. |
| `bookmark.get` | `bookmark` | `{name, epoch, seq, t, note}` |
| `bookmark.list` | | `{bookmarks: [{name, epoch, seq, t, note, live, retained}], count, store, head, oldest, max_bookmarks, max_note_len, ring}` ; `live` dit que ce démarrage l'a posé ; `retained` dit que l'anneau tient encore tout ce qui suit son point (de ce démarrage, et `seq + 1 >= oldest`, le plus ancien seq tenu), donc qu'un rejeu depuis lui s'ouvre sans trou. Les deux sont lus au même instant ; l'anneau continue de tourner. |
| `bookmark.remove` | `bookmark` | `{bookmark, removed: true}` |

`subscribe` (`ixctl.v1`) et `GET /v1/events` prennent `bookmark` à la place de `since`/`epoch` ;
les deux ensemble sont refusés (« bookmark or since »), sauf qu'un en-tête `Last-Event-ID` à côté
d'un `bookmark` dans la requête est une reconnexion d'un flux déjà passé le signet (un
`EventSource` renvoie son URL d'origine avec l'en-tête) et reprend depuis l'en-tête. Précédence
sur `/v1/events` : l'en-tête l'emporte sur un `since` de la requête et sur un `bookmark` (un seq
nu dans l'en-tête prend l'`epoch` de la requête, `epoch:seq` porte le sien) ; sans l'en-tête,
`bookmark` et `since`/`epoch` sont deux points et refusés ensemble. Les en-têtes d'un flux ne
partent qu'une fois son curseur créé : un lecteur qui les a ne manque rien de ce qui est publié
ensuite. Le rejeu est
tout ce qui suit le point du signet sur le préfixe demandé, puis le direct. Les bornes, rapportées
par `bookmark.list` : `max_bookmarks` 64 par plan, un nom selon la règle des noms d'instance,
`max_note_len` 256 caractères ; un refus nomme ce qu'il a touché, et un fichier dont les
enregistrements les enfreignent est refusé au démarrage. Chaque op `bookmark.*` peut être relayée
à un pair (`peer`), dans la séquence duquel le signet pointe alors ; le rejeu reste local à ce
pair.

## Journal

`<nom>.journal.jsonl` ajoute, en lignes JSON, un en-tête `boot` par démarrage (`schema:
"ixctl-journal/0"`, `name`, `epoch`), chaque document appliqué avec sa génération et ses actions,
et l'enveloppe de chaque événement `ctl`. C'est un journal d'audit, pas encore un magasin qu'un
programme devrait lire (porte 3).

## Barrière

`e2e/e2e.py` est la barrière de bout en bout que la CI fait tourner sous Windows et Linux : le
vrai plan, le vrai client Rust via son exemple `ixql-ctl`, l'exemple `pod` d'ix-stream en deux
instances fédérées, le vrai `ix pipeline run` comme job, des exécutions tuées à leurs limites
de temps mur, de mémoire, de temps processeur et de processus, un arbre de sondes sous le plan
(E15 : kinds et limites, filtre et mux, expiration du bail, cpu et mem de l'hôte) et des signets
(E16 : un rejeu, seq pour seq, de ce que les tuyaux d'un arbre de sondes et le cycle de vie `ctl`
ont porté après un point nommé par l'opérateur ; bornes, refus, signet d'un pair ; survie à un
redémarrage) et la surface IXQL des sondes (E17 : `probe.ixql` via le client Rust, une sonde CPU
sous un mux, dix enregistrements, l'arbre retiré) et les budgets de sondes (E18 : le budget du
plan dans `limits`, une feuille qui dépasserait le `max_rate_per_s` de son mux refusée en nommant
les deux, une feuille sans budget, un mux qui ne peut pas élargir, débits et octets dans la liste).
45 vérifications, E1..E19 (E19 : seul un tuyau de sonde que le pair exporte et que le parent demande traverse le lien, par nom exact, le statut disant ce qu'est chaque nom demandé), une boucle de
redémarrage et deux mesures : M1 (ce qu'une attente sur
le flux gagnerait sur la sonde à 25 ms) et M2 (ce qu'un mux sur seize heartbeats à 10 ms livre,
et ce qu'il coûte au plan).

```sh
python go/ixctl/e2e/e2e.py --build            # construit ixctl, pod, ixql-ctl et ix, puis lance
python go/ixctl/e2e/e2e.py --bin <dossier>    # utilise des binaires déjà construits là
python go/ixctl/e2e/e2e.py --build --work D   # garde binaires, état, exécutions et results.json dans D
```

Le job lit la constitution Demerzel dans le sous-module `governance/demerzel`
(`IX_GOVERNANCE_DIR` la remplace). La barrière n'imprime jamais un jeton et scanne chaque fichier
écrit pour en trouver un. La CI lance aussi `go vet` et `go test -race` sous Linux.

## Limites connues

- Windows d'abord : le test de vie de la barrière et le code processus ont été écrits pour
  Windows ; la veille mémoire Linux est une sonde, et une mort brutale sous Linux n'est contenue
  que pour le meneur.
- Un seul commit refusé au-delà du plafond mémoire Windows dépend encore du message au mieux du
  noyau ; la notification à `memory_mb` est le chemin garanti.
- Les limites de temps processeur et de processus sont des sondes : une exécution peut consommer
  jusqu'à une période de lecture sur tous les cœurs de la machine au-delà de la première avant
  la mise à mort, et sous Linux (sans plafond) lancer librement des processus pendant une
  période de lecture au-delà de la seconde.
