# ixctl — le plan de contrôle d'IX

English: [`README.md`](README.md).

`ixctl` est un petit processus Go qui supervise les services qu'une spec déclare, les réconcilie
vers un document d'état désiré à la manière d'un Terraform local (`plan` dit ce qui changerait,
`apply` incrémente une génération et agit), borne chaque exécution (temps mur, mémoire) et
rapporte ce que chaque exécution a coûté. Il expose une seule face HTTP/WebSocket en boucle locale
et se fédère avec des instances pairs. Le client Rust et IXQL est
[`crates/ix-ctl`](../../crates/ix-ctl) (`ix.ctl.status`, `ix.ctl.plan`, `ix.ctl.apply`,
`ix.ctl.stop`, `ix.ctl.wait`, `ix.ctl.events`).

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
./ixctl up spec.json [--addr 127.0.0.1:0] [--name NOM] [--state DOSSIER]
./ixctl bench spec.json        # distributions de latence : start, stop, restart, RTT status, fan-out
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
| `POST /v1/shutdown` | Arrête chaque arbre, puis les transports ; le fichier d'état est retiré. |
| `GET /v1/events?ch=PREFIXE&since=SEQ&epoch=E` | Server-sent events : `id: epoch:seq`, `event: <kind>`, `data: <enveloppe>`. `Last-Event-ID` vaut `since`. Un lecteur qui tombe de l'anneau reçoit un événement `gap` et le flux se termine. |
| `GET /v1/ws` | Le WebSocket `ixctl.v1` (ci-dessous). |

Les erreurs sont `{"error": "..."}` avec 400 (requête refusée, nom non déclaré), 401 (jeton),
403 (host ou origin), 502 (pair hors ligne).

### `ixctl.v1` sur le WebSocket

Les trames du client sont des requêtes : `{id, op, ...}` avec `op` parmi `status`, `ping`,
`plan`, `apply`, `stop`, `subscribe`, `unsubscribe`, et les champs de l'op (`desired`, `service`,
`peer`, `ch`, `since`, `epoch`, `sub`). Chaque requête reçoit une trame de réponse
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
| `peer/<nom>/<canal>` | Les canaux `ctl` et `svc/` d'un pair, en miroir. `subscribe` est toujours local ; les ops portent `peer` pour être relayées. |

## Journal

`<nom>.journal.jsonl` ajoute, en lignes JSON, un en-tête `boot` par démarrage (`schema:
"ixctl-journal/0"`, `name`, `epoch`), chaque document appliqué avec sa génération et ses actions,
et l'enveloppe de chaque événement `ctl`. C'est un journal d'audit, pas encore un magasin qu'un
programme devrait lire (porte 3).

## Barrière

`e2e/e2e.py` est la barrière de bout en bout que la CI fait tourner sous Windows et Linux : le
vrai plan, le vrai client Rust via son exemple `ixql-ctl`, l'exemple `pod` d'ix-stream en deux
instances fédérées, le vrai `ix pipeline run` comme job, et des exécutions tuées à leurs limites
de temps mur, de mémoire, de temps processeur et de processus. 36 vérifications, E1..E14, une
boucle de redémarrage et la mesure M1 (ce qu'une attente sur le flux gagnerait sur la sonde à
25 ms).

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
