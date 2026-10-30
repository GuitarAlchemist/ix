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

Des trois formats qu'il définit, deux sont **figés** depuis le 2026-10-25
([ADR-0014](../../docs/fr/adr/0014-ixctl-v1-and-desired-state-frozen.md)) : le protocole
`ixctl.v1`, dont le noyau est fixé tandis que de nouvelles ops, de nouveaux canaux, de nouvelles
sortes et de nouveaux champs optionnels peuvent apparaître, et le document d'état désiré
`ixctl-desired/0`. Le journal reste **provisoire**, comme la surface des sondes et les signets. La
page des portes à sens unique,
[`docs/fr/plans/2026-10-17-ixctl-one-way-doors.md`](../../docs/fr/plans/2026-10-17-ixctl-one-way-doors.md),
donne les options pesées et les rondes de revue qui les ont façonnés.

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
  contre l'état observé. Sa `generation` est la génération courante (la dernière appliquée, comme
  dans le statut), pas celle qu'un apply obtiendrait : un apply lancé maintenant obtient la
  suivante, sauf si un autre apply passe avant.
- `apply` incrémente la génération, journalise le document, renvoie aussitôt
  `{generation, epoch, actions, results}` et agit en arrière-plan. `results` est un objet
  indexé exactement par les services nommés dans le document ; chaque valeur porte `result`
  pris dans l'ensemble fermé ci-dessous. Un service omis n'a aucun résultat et reste inchangé.
  Le statut rapporte `generation`
  (dernière appliquée), `observed_generation` (dernière dont toutes les actions ont été vues
  jusqu'au bout) et `settled_generation` (dernière dont chaque service a atteint son état
  désiré) ; `diverged` nomme la génération que le plan a abandonnée, avec le service et la
  raison, et `failed` garde les échecs de lancement récents par génération, pour qu'un apply
  ultérieur ne cache pas un échec antérieur.
- Chaque résultat vient d'un même instantané d'admission, après validation et pendant que le plan
  calcule les actions et incrémente la génération, avant d'en mettre une en file :

  | Le document dit | Service à l'admission | Résultat | Action |
  |---|---|---|---|
  | `running` | le desired courant est `running` et l'état vaut `starting`, `running` ou le délai de redémarrage (`restarting`) | `{"result":"not-launched","current_run": RUN_OU_NULL}` | aucune |
  | `running` | `stopped` ou `failed`, ou un desired non-running est encore en cours d'exécution | `{"result":"start-pending"}` | `start` |
  | `stopped` | `stopped` | `{"result":"noop"}` | aucune |
  | `stopped` | `starting`, `running`, `restarting` ou `failed` | `{"result":"stop-pending"}` | `stop` |

  Un démarrage en file est `starting` : `current_run` est l'exécution précédente, ou null
  avant l'annonce de la première. Pendant le délai de redémarrage, c'est l'exécution qui vient de
  finir. Le `{origin, epoch, seq}` opaque prouve seulement quelle exécution était courante à
  l'admission, pas qu'elle existe encore à l'arrivée de la réponse. Ce n'est jamais une
  exécution lancée par cet apply : `not-launched` prouve que cette génération n'annonce aucune
  nouvelle exécution. Ces résultats sont des valeurs que le plan répond ; un apply refusé (un
  document invalide, un service inconnu, un plan qui s'arrête, une erreur de pair) est une erreur
  et ne porte aucun `results`. Ils n'attendent pas ; l'opération `wait` côté plan ci-dessous
  consomme leurs preuves d'epoch, de génération, de service et d'exécution.
- `epoch` nomme le démarrage auquel ces générations appartiennent. Un client qui attend une
  génération devrait nommer l'epoch aussi, pour qu'un redémarrage du plan termine l'attente au
  lieu de laisser passer la génération de même numéro du nouveau démarrage pour l'ancienne.
- `stop` arrête un service hors bande et répond une fois son arbre disparu :
  `{stopped, generation, exit}`. Un arrêt demandé ferme stdin, attend deux secondes, puis tue.

Le statut porte `contract: "run/1"`, le contrat consommateur que ce plan tient. Il repose sur le
noyau figé d'`ixctl.v1` et sur `ixctl-desired/0`
([ADR-0014](../../docs/fr/adr/0014-ixctl-v1-and-desired-state-frozen.md)), et il comprend :

- l'identité d'exécution `run = {origin, epoch, seq}` et les champs d'exécution du statut
  ci-dessous ;
- les `results` d'`apply`, l'ensemble fermé ci-dessus ;
- les marqueurs `ixctl.run.begin` et `ixctl.run.end`, sur demande et accusés ;
- l'opération `wait` et son ensemble fermé d'issues, dont les raisons d'`unknown` sont `evicted` et
  `other-epoch` seulement, avec l'ordre dans lequel elle lit ses preuves ;
- la provenance fédérée : `run` inchangé dans une copie, et `status.peers[].run_markers`.

Un consommateur se règle sur cette valeur : un plan dont le statut ne la porte pas ne tient rien de
run/1. Les ajouts suivent la règle d'extension de l'ADR-0014 ; changer ce que run/1 promet, une
raison d'`unknown` comprise, exige `run/2`. Une génération peut porter plusieurs exécutions quand
un service redémarre automatiquement :

| Champ du statut | Sens |
|---|---|
| `services[].run` | La dernière exécution annoncée par `service.starting`. Elle est l'exécution active en `starting` (une fois annoncée) et `running` ; pendant le délai de redémarrage (`restarting`), c'est celle dont la sortie a causé le délai, jusqu'à l'annonce de sa remplaçante ; `stopped` et `failed` gardent la dernière exécution. Avant la première annonce, le champ est absent ; un démarrage en file peut déjà afficher `starting` tout en gardant l'exécution précédente, ou aucune avant la première. |
| `services[].recent_exits_capacity` | La borne par service annoncée pour `recent_exits`, actuellement 16. Ce nombre est provisoire ; la valeur du statut fait autorité. |
| `services[].recent_exits[]` | Les tentatives terminales gardées pendant ce démarrage, de la plus ancienne à la plus récente. Chaque enregistrement porte son `run` opaque et exactement les champs de résultat de son `ixctl.run.end`. Les échecs de lancement comptent ; l'exécution encore en cours n'y figure pas. En cas de débordement, l'enregistrement le plus ancien est évincé. |
| `services[].last_exit.run` | Vue de compatibilité de la dernière tentative ayant démarré un processus, avec son heure, sa génération, son usage et son compte de lignes tardives. Elle n'inclut pas les échecs de lancement et n'est pas une recherche par exécution ; après le début d'un redémarrage, elle peut différer de `services[].run`. `late_lines` compte, avec saturation, les lignes lues après son marqueur de fin et écartées, tant qu'elle reste la dernière sortie. |

### Attentes côté plan

Une seule opération possède les trois réducteurs. En HTTP, c'est
`POST /v1/wait[?peer=NOM]` ; sur `ixctl.v1`, c'est `{"op":"wait", ...}`. Sa requête commune
est :

```json
{"wait":"generation|launch|run","epoch":"...","generation":7,
 "service":"pod-a","run":{"origin":"local","epoch":"...","seq":12},"timeout_ms":30000}
```

L'attente de génération utilise `epoch,generation` ; celle de lancement ajoute `service` ;
celle d'une exécution exacte utilise seulement `run`. L'attente de lancement n'est valide que
si le résultat d'apply du service était `start-pending` ou `not-launched` ; la demander à
`stop-pending`, à `noop` ou pour un service que la génération ne nommait pas est une requête
mal formée. De même pour une attente de génération ou de lancement sur une génération que ce
démarrage n'a pas encore émise (au-delà de `status.generation`) : le plan répond 400 aussitôt
au lieu d'attendre qu'elle existe. `timeout_ms` est obligatoire, strictement
positif, et ne peut dépasser `status.limits.wait_timeout_ms` (60000 actuellement). Un plan
accepte au plus `status.limits.waiters` attentes simultanées (64 actuellement). Dépasser le
plafond de durée vaut 400 ; manquer de place vaut 429 (ou `ok:false` sur WebSocket).

Toute attente terminée répond HTTP 200 ou WebSocket `ok:true` avec une valeur de cet ensemble
fermé :

| Attente | Résultats |
|---|---|
| génération | `{"outcome":"converged"}` ; `{"outcome":"diverged","service":NOM,"reason":TEXTE}` ; `{"outcome":"timeout"}` ; `{"outcome":"unknown","reason":"other-epoch"}` |
| lancement | `{"outcome":"launched","run":RUN}` ; `{"outcome":"not-launched","current_run":RUN_OU_NULL}` ; `{"outcome":"timeout"}` ; `{"outcome":"unknown","reason":"other-epoch"|"evicted"}` |
| exécution | `{"outcome":"ended","run":RUN,"end":CHAMPS_RUN_END}` ; `{"outcome":"timeout"}` ; `{"outcome":"unknown","reason":"other-epoch"|"evicted"}` |

Un timeout est l'échéance d'attente du demandeur, pas une erreur. Une requête mal formée,
l'authentification, un service inconnu, la surcharge, une erreur de pair ou de transport et
l'arrêt du plan restent des erreurs. `unknown` n'a que `evicted` et `other-epoch` ; l'étendre
exige `run/2`.

Pour l'attente de lancement, le plan garde les 16 derniers résultats d'admission par service,
avec l'instantané `current_run` et la première exécution lancée par cette génération, telle
qu'`ixctl.run.begin` l'annonce ; un redémarrage sous la même génération est une exécution
ultérieure et ne la remplace jamais. `evicted` signifie que ce résultat génération/service
exact n'est plus gardé : il ne répond qu'à une génération pas plus récente que le plus récent
enregistrement écarté, si bien qu'une génération qui n'a jamais nommé le service reste une
requête mal formée. Un résultat `stop-pending` ou `noop` n'est une requête mal formée que tant
qu'il est gardé : une fois évincé, l'attente répond `unknown (evicted)`, puisque le plan ne
sait plus quel était ce résultat.
`not-launched.current_run` prouve seulement ce qui était courant à l'admission et ne nomme
jamais une exécution lancée par cette génération. Pour l'attente d'exécution exacte, `evicted`
signifie que l'exécution du même epoch n'est ni courante ni dans `recent_exits` ; la sortie
d'une autre exécution ne la remplace jamais. Une attente avec `?peer=` est transmise au plan
d'origine et décidée là-bas. Le client Rust enveloppe l'opération dans
`Ctl::wait_generation_outcome`, `Ctl::wait_launch` et `Ctl::wait_run`, un type de résultat fermé
par attente ; un handle créé par `Ctl::from_state` suit le fichier d'état, si bien qu'une
attente coupée par un arrêt brutal est reposée au nouveau démarrage et se termine en
`unknown (other-epoch)`. `wait_generation(g)`, par numéro seul, est déprécié au profit de
`wait_generation_of`. En IXQL, l'opération s'appelle
`ix.ctl.wait(wait: "generation"|"launch"|"run", ...)`.

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
| `POST /v1/wait[?peer=]` | Corps : la requête d'attente commune ci-dessus. |
| `POST /v1/stop` | Corps `{"service": NOM, "peer"?: NOM}`. |
| `POST /v1/shutdown` | Arrête chaque arbre et chaque sonde, puis les transports ; le fichier d'état est retiré. |
| `GET /v1/probes[?peer=]` | Les kinds de sonde et les limites du système (`probe.kinds`, ci-dessous). |
| `GET /v1/probe[?peer=][&with_cursor=1]` | L'arbre des sondes (`probe.list`) ; avec `with_cursor=1`, lié à un curseur du flux du plan. |
| `POST /v1/probe/create[?peer=]` | Corps : un nœud (`probe.create`). |
| `POST /v1/probe/start`, `stop`, `touch`, `remove` | Corps `{"node": NOM, "peer"?: NOM}`. |
| `GET /v1/bookmark[?peer=]` | Les signets (`bookmark.list`, ci-dessous). |
| `POST /v1/bookmark/set` | Corps `{"bookmark": NOM, "since"?: SEQ, "epoch"?: E, "note"?: TEXTE, "peer"?: NOM}`. |
| `POST /v1/bookmark/remove` | Corps `{"bookmark": NOM, "peer"?: NOM}`. |
| `GET /v1/events?ch=PREFIXE&since=SEQ&epoch=E`, ou `?ch=PREFIXE&bookmark=NOM` | Server-sent events : `id: epoch:seq`, `event: <kind>`, `data: <enveloppe>`. `Last-Event-ID` vaut `since` ; `bookmark` rejoue depuis un signet à la place (pas à côté de `since`). `markers=run` inclut les marqueurs et est acquitté par `Ixctl-Markers: run`. Un lecteur qui prend du retard sur l'anneau pendant la lecture reçoit un événement `gap` (`overflow`) et le flux se termine. Un point de reprise que l'anneau ne contient plus (`since-too-old`), ou qui vient d'un autre démarrage (`epoch-changed`, `since-ahead`), reçoit d'abord un événement `gap`, puis le flux continue depuis la plus ancienne enveloppe que l'anneau contient encore. `since-ahead` vaut pour un `since` au-delà de la tête qui ne nomme pas d'`epoch` ; un `since` qui nomme l'epoch de ce démarrage est cru même au-delà de la tête : ni gap ni rejeu, le flux part en direct depuis la tête. |
| `GET /v1/ws` | Le WebSocket `ixctl.v1` (ci-dessous). |

Les erreurs sont `{"error": "..."}` avec 400 (requête refusée, nom non déclaré), 401 (jeton),
403 (host ou origin), 429 (places d'attente pleines), 500 (panne interne, comme un arrêt qui n'a
pas convergé), 502 (pair hors ligne) ou 503 (arrêt). Le texte est fait pour un humain et ne fait
pas partie du contrat. Une route ou une méthode que le plan ne sert pas reçoit le 404 ou le 405 du
serveur HTTP lui-même, qui n'en fait pas partie non plus.

### `ixctl.v1` sur le WebSocket

Les trames du client sont des requêtes : `{id, op, ...}` avec `op` parmi `status`, `ping`,
`plan`, `apply`, `wait`, `stop`, `subscribe`, `unsubscribe`, les sept ops `probe.*` et les
quatre ops `bookmark.*` (ci-dessous), et les champs de l'op (`desired`, `service`, `peer`,
`wait`, `generation`, `run`, `timeout_ms`, `ch`, `since`, `epoch`, `sub`, `probe`, `node`,
`bookmark`, `note`, `markers`). Chaque requête reçoit une trame de réponse
`{re: id, ok, data | error}`. Les abonnements sont posés en ligne (leur réponse précède leur
premier événement) ; les autres ops courent de leur côté, pour qu'un stop lent ne retienne jamais
la connexion. Un abonnement avec `"markers":"run"` est acquitté par `"markers":"run"` dans les
données de sa réponse. Toute autre valeur de `markers` est une erreur.

Les événements arrivent en enveloppes :

```json
{"ch":"svc/pod-a","seq":17,"epoch":"8c1f…","origin":"local",
 "run":{"origin":"local","epoch":"8c1f…","seq":12},"t":1760990000123,
 "ev":{"kind":"stats","t":1760990000120,"data":{"nodes":[...]}}}
```

`seq` compte depuis 1 à chaque démarrage ; `epoch` nomme le démarrage ; `origin` est l'instance
émettrice ; une enveloppe miroir (canal `peer/`) porte aussi `src_seq` et `src_epoch`, la
séquence et le démarrage propres à la source, pour qu'un lecteur ici identifie et dédoublonne
comme un lecteur chez l'origine (S2 tranche 2). Un abonné avec `since` rejoue depuis l'anneau (4096 enveloppes par instance) et reçoit
`{sub, gap}` quand c'est impossible : `since-too-old`, `overflow`, `epoch-changed`, `since-ahead`.

`run = {origin, epoch, seq}` nomme une tentative de lancement ; son `seq` est la séquence du
`service.starting` de cette tentative sur son plan d'origine. Il est fixé avant toute sortie,
immuable et unique dans une origine et un epoch. Chaque enveloppe `svc/<nom>` de la tentative et
chaque événement de service `ctl` qui lui est attribuable portent le même `run`. Un redémarrage
automatique reçoit une autre exécution tout en gardant la génération qui a demandé le service. Une
enveloppe reflétée (`peer/<nom>/…`) garde le `run` source inchangé, séparément de ses `seq` /
`epoch` locaux et des `src_seq` / `src_epoch` de la source. Le lien demande les marqueurs sur ses
deux abonnements de contrôle ; `status.peers[].run_markers` n'est vrai que si la session courante
du pair les a acquittés tous les deux ; tant qu'il est faux (un pair plus ancien, un lien coupé ou
en réouverture), un reflet sans marqueurs ne dit rien de l'ouverture ni de la fin d'une exécution.
Les abonnés locaux ne les reçoivent que s'ils les demandent.

Le plan place deux enveloppes ordinaires dans le flux source `svc/<nom>`. `ixctl.run.begin`
porte `generation` et précède toute ligne du worker. `ixctl.run.end` suit toute ligne transmise
par le plan et porte :

- `outcome` : le résultat de la tentative, `started` ou `launch-failed` ;
- `output` : `complete` ou `truncated`, ainsi que `drain_ms` ;
- pour `started`, le résultat terminal : `code`, `requested` et l'éventuel `reason` d'une limite ;
- pour `launch-failed`, `error`. Même une tentative sans processus reçoit les deux marqueurs.

Pour prouver le résultat d'un `run` opaque donné, un consommateur suit cet ordre :

1. utiliser un `ixctl.run.end` reçu qui porte exactement ce `run` ;
2. sinon lire le statut du service d'origine dans le même epoch et utiliser l'enregistrement de
   `recent_exits` portant exactement ce `run` ;
3. sinon rapporter `unknown (other-epoch)` si l'epoch du statut diffère. Si l'epoch concorde et
   que l'enregistrement est absent, l'exécution n'est pas finie si elle est encore
   `services[].run`, et elle est `unknown (evicted)` sinon.

La sortie d'une autre exécution, y compris `last_exit`, ne la remplace jamais.

Après une fin tronquée, les lignes tardives sont écartées et comptées dans
`services[].last_exit.late_lines` ; le compteur sature au lieu de boucler. Les marqueurs restent
dans l'anneau, mais SSE et WebSocket les sautent, même en rejeu, sauf demande de `markers=run`
ou `"markers":"run"`. Un kind de worker commençant par le préfixe réservé `ixctl.` est réécrit
une seule fois à l'entrée en `log`, avec la ligne brute sous `data.line`.

### Canaux et événements

| Canal | Porte |
|---|---|
| `ctl` | Les événements du plan lui-même : `service.starting` (`restart`, `generation`), `service.started` (`pid`, `restarts`, `generation`), `service.exited` (`pid`, `code`, `requested`, `generation`, et quand posés `reason` + `wall_ms`/`memory_mb`/`cpu_limit_ms`/`processes`, `peak_memory_bytes`, `cpu_ms`, `peak_processes`), `service.restarting` (`attempt`, `in_ms`), `service.stopped`, `service.failed` (`reason`), `converged` / `diverged` (`generation`), `peer.up` / `peer.down`. Les événements de service attribuables à une exécution la portent dans l'enveloppe. |
| `svc/<nom>` | La sortie du service. Chaque enveloppe porte son exécution. Une ligne stdout qui est un objet JSON avec un `kind` chaîne devient cet événement, dont seuls `kind`, `t` et `data` sont gardés : tout autre champ de premier niveau est abandonné (`t` en ms Unix sur l'horloge du service, arrondi, et abandonné quand ce n'est pas un nombre fini ; un `data` à `null` vaut absence de données ; le tableau `nodes` d'un événement `stats` devient les `nodes` du service dans le statut). Toute autre ligne stdout est un événement `log` `{line}` ; les lignes stderr sont des événements `stderr`. |
| `probe/<nœud>` | Le tuyau d'un nœud de sonde : des événements `sample` (ci-dessous). En miroir chez un pair seulement si ce nœud est créé avec `export: true` et que la spec de ce pair le nomme (D4, ix#402). |
| `peer/<nom>/<canal>` | Les canaux `ctl` et `svc/` d'un pair, en miroir, plus les tuyaux `probe/<nœud>` que cette instance a demandés dans `peers[].probes` tant que le pair les exporte ; et les enveloppes `link.gap` propres au lien, les marqueurs de continuité (ci-dessous). `subscribe` est toujours local ; les ops portent `peer` pour être relayées. |

**Tuyaux de sondes à travers le lien (D4, ix#402, tranchée le 2026-10-21 ; S2 ix#408, tranche 1).**
Ce qui traverse est l'intersection de deux déclarations : l'enfant marque un nœud exportable à la
création (`export: true`, immuable, faux par défaut : une disponibilité, pas une demande) et la
spec du pair côté parent nomme exactement les nœuds voulus (`peers[].probes`). Le lien s'abonne au
tuyau de chaque nœud voulu par son nom de canal exact (`probe/cpu` ne couvre pas
`probe/cpu-extra`), depuis le `probe.list` du pair au raccordement puis depuis les
`probe.created` / `probe.removed` en miroir ; `status.peers[].probes` dit ce qu'est chaque nom
demandé : `mirrored`, `not-exportable`, `opening` (abonnement en cours) ou `absent` (pas de tel
nœud, ou lien tombé). Un tuyau démarre en direct à chaque session et après chaque retrait,
jamais depuis une séquence : le même nom de canal peut porter les enregistrements d'une autre
incarnation ; depuis la tranche 4 le lien sait quelle incarnation il reflète (la liste liée à un
curseur ci-dessous, `created_seq`, sur `status.peers[].probes[]`), si bien qu'une reprise depuis
le dernier seq reflété serait désormais fondée et est laissée pour plus tard ; la tranche 2 a
apporté l'identité, la tranche 3 le marqueur pour ce qu'un démarrage en direct saute (G8 du
contrat fermé) et le rattrapage borné. L'agrégation
vit chez le lecteur : en V1 le lecteur est le consommateur de `peer/<nom>/probe/<nœud>` (IXQL
`ix.ctl.events`), aucun nœud local ne nomme une source distante. L'export est une règle de
fédération, pas une barrière de confidentialité : l'abonnement direct à un tuyau est inchangé.
La tranche 2 ajoute les `src_seq` / `src_epoch` de la source sur chaque enveloppe miroir. La
tranche 3 marque ce que le miroir saute, dans le flux miroir lui-même : une enveloppe `link.gap`
publiée par le parent (son origine à lui, sans provenance, jamais un compte) sur le tuyau à
chaque démarrage en direct (`reason: live-start`, `src_epoch` le boot lu désormais,
`last_src_seq` / `last_src_epoch` la dernière trame reflétée avant, s'il y en eut) et sur
`peer/<nom>/ctl` quand le pair est revenu comme nouvelle instance (`epoch-changed`,
`last_src_epoch` le boot d'avant) ou a répondu un trou sur le `ctl` / `svc/` repris du lien
(`prefix`, `reason`, `from`, `oldest` tels qu'il les a dits : son historique a expiré) ; un trou
local reste une trame gap. Les files du lien sont bornées par classe (critère 6 de ix#408) :
`ctl` et `svc/` dans leur propre file, jamais écartés par le lien et lus d'abord (une trame de contrôle attend au plus l'application d'une trame de tuyau ; le débordement par le pair de ces abonnements les rouvre dans la session, repris ou en direct comme ci-dessous, et un rejeu que son anneau ne tient plus est un trou, marqué) ; les tuyaux demandés dans une
file bornée en enregistrements (1024) et en octets (4 Mio) que leurs relais n'attendent jamais
pour un enregistrement (la seule trame qu'un relais attend de déposer est le trou de débordement
de son abonnement, sa dernière), si bien qu'un parent plus lent que la charge offerte perd des
enregistrements de tuyau plutôt que de grossir ou de retenir le cycle de vie ; chaque série de pertes est comptée sur
`status.peers[].probes[].dropped` et marquée sur le tuyau (`link.gap`, `reason: dropped`,
`dropped` le compte exact du lien lui-même) avant l'enregistrement qui a suivi, et un abonnement
de tuyau que le pair a lui-même lâché (`overflow`) est marqué et rouvert en direct sans clore la
session. Un rattrapage de `ctl` / `svc/` (une reprise après une coupure ou après que le pair a
lâché l'abonnement) est borné par une provision que le lien possède (critère 7) : 8192 trames et
32 Mio par lien, remplie à nouveau en une minute, au plus 30 s par rattrapage, jamais remise à
zéro par une reconnexion. Un rattrapage ne commence qu'avec une trame et un octet de provision et
s'arrête à la première trame qu'elle ne couvre pas : le lien s'abonne alors en direct et marque le
saut sur `peer/<nom>/ctl` (`link.gap`, `reason: jumped`, `prefix`, `from` le dernier seq reflété
sur ce préfixe, `to` la tête du pair d'où part l'abonnement en direct, `src_epoch`, daté par `t`) :
un parent qui ne suit pas rejoue une fois, borné, puis saute ; il rejoue au plus une provision par
minute de remplissage, ne se reconnecte pour rien de cela (un abonnement de contrôle lâché est
rouvert dans la même session) et ne grossit pas. Un départ en direct qui ne manque rien (la tête du
pair est le dernier seq reflété sur le préfixe, dans le même démarrage) n'est pas un saut et ne
marque rien ; l'intervalle qu'un marqueur nomme est celui de la tête que tous les canaux partagent,
il peut donc ne contenir que des trames d'autres canaux (le marqueur sur-déclare, jamais l'inverse). Le
`status.peers[]` du parent distingue `live` (le lien est haut et ses deux abonnements de contrôle
ont dépassé leur rejeu : le lien demande au pair de le dire, `subscribe` avec `live: true` répondu
par une trame `{"sub", "live": true}` une fois le rejeu passé la tête d'où l'abonnement est parti,
tout de suite s'il n'y avait rien à rejouer ; aucun autre abonné ne la reçoit) de `reconciled` (le
lien est haut, la liste des sondes du pair a été appliquée dans cette session et aucun tuyau
demandé n'est encore en ouverture), ne lève `degraded` que lorsque les deux tiennent, et compte
dans `jumps` les rattrapages abandonnés. Un saut de `ctl` perd les événements de cycle de vie de
son intervalle : l'état n'est plus su réconcilié (`reconciled` faux, `degraded`, dès l'étape qui
abandonne le rattrapage) jusqu'à ce que le lien relise la liste, ce qu'il fait sur sa boucle de
session juste après que l'abonnement en direct tient, comme l'instantané ci-dessous ; voir la
résolution sur ix#402.

**La liste comme instantané (S2 tranche 4, critères 3 et 4 de ix#408).** `probe.list` pris avec
`with_cursor` (`GET /v1/probe?with_cursor=1`, IXQL `ix.ctl.probe.list(with_cursor: true)`) dit le
point du flux du plan où il se tient, `cursor: {seq, epoch}` (la tête et le démarrage), lu sous le
verrou de l'arbre avec les nœuds, là où chaque événement de cycle de vie est publié : quels nœuds
existent, leur kind, parent et export, lesquels tournent, sont exactement ce que `ctl` dit jusqu'à
ce seq et rien de ce qui suit, si bien qu'un lecteur peut replier le flux à partir du curseur sans
trou ni répétition (ce qu'aucun événement ne dit, `stopping`, `expires_in_ms`, les compteurs, est
la lecture propre de l'arbre au curseur). Chaque nœud porte `created_seq`, le
seq de son `probe.created` : l'identité de l'incarnation sous ce nom (une recréation du même nom
en est une nouvelle). Une feuille dont l'arrêt est en cours dit `state: "stopping"` (son
`probe.stopped` n'est pas encore sorti : `ctl` dit encore running, un démarrage est refusé
jusque-là ; l'événement et la fin de l'arrêt sont une seule étape, donc la liste ne dit jamais
`stopping` une fois que `ctl` a dit stopped). Le lien lit sa liste ainsi : un événement de cycle
de vie au curseur ou avant est déjà reflété et n'est pas suivi ; une liste qui n'est pas un
instantané (sans curseur, un nœud voulu sans `created_seq`) est refusée et met fin à la session ;
chaque tuyau voulu porte l'identité qu'il reflète (`status.peers[].probes[].created_seq` /
`src_epoch`, depuis la liste ou depuis le `probe.created` suivi), si bien qu'un consommateur qui
lit le statut, puis le tuyau, cle ce qu'il lit dès lors `origin:src_epoch:node:created_seq` sans
avoir lu cet événement (le statut est une lecture, pas un point du flux : une trame lue avant lui
n'est distinguée que par le `probe.created` miroir ou la liste avec son curseur) ; après un saut
sur `ctl` le lien relit la liste sur sa boucle de session, en retenant les trames de contrôle qui
arrivent entre-temps (bornées comme au début de session) pour les appliquer après elle, et s'y
réconcilie : un nœud voulu disparu de la liste est retiré, un nom dont l'identité a changé est
rouvert comme la nouvelle incarnation (son démarrage en direct marqué comme tout autre), et
`reconciled` tient de nouveau. Testé sur un arbre qui bouge (25 instantanés repliés depuis le
flux, des feuilles s'arrêtant sous la liste), sur les identités et les sauts du lien, sur les
trames retenues, sur `reconciled` qui tombe au saut, et sur un déluge de `ctl` qui fait sauter le
lien et relire la liste ; porte E23 sur les deux plans.

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
le travail est suivi sous l'epic ix#397. La surface est provisoire : l'ADR-0014 la laisse hors du
noyau figé d'`ixctl.v1`.

| Kind | Champs | Lit |
|---|---|---|
| `heartbeat@1` | `n` | Son propre compte de ticks, depuis 1. Le kind des tests et des démonstrations. |
| `cpu@1` | `busy_pct`, `cores` | La part d'occupation du processeur de l'hôte sur le dernier intervalle (`GetSystemTimes` ; `/proc/stat`). Le premier intervalle n'émet rien. |
| `mem@1` | `total_bytes`, `avail_bytes`, `used_pct` | La mémoire physique de l'hôte (`GlobalMemoryStatusEx` ; `/proc/meminfo`). |
| `delta@1` | `event`, `journal_seq`, `t_ms`, `t`, `pid`, `offset`, et ceux de l'événement (`utterance_id`, `text`, `asr_ms`, `route`, `triage`, `triage_ms`, `parrot`, `promise`, `say_id`, `by`, `request`, `cost`, `muted`, `version`, `replaced_by`) | Le journal JSONL de Delta, le robot vocal, nommé par `ixctl up --delta-journal FICHIER` : une ligne par tick, chaque champ de premier niveau de la ligne tel quel, plus `offset`, la position en octets de la ligne. Sans le flag, le kind est listé et refusé à la création. |
| `mux@1` | (ceux de ses enfants) | Pas d'échantillonneur : il porte ce que ses enfants lui passent. |

`delta@1`, c'est le plan qui *lit* un fichier qu'un autre programme écrit (le `brain.exe` de
brickyard, qui journalise ce que Delta a entendu et dit) ; rien n'est poussé dans le plan. Une
ligne par tick garde le débit déclaré du nœud (1000 / `every_ms`) comme borne quel que soit le
rythme du journal : une rafale s'écoule à ce débit et rien n'est perdu. Une ligne qui n'est pas
un objet JSON, ou qui dépasse 1 Mio (`max_line`), est un enregistrement d'erreur qui nomme son
offset, et les lignes suivantes passent ; un journal absent est dit une fois puis attendu ; un
journal plus court que ce qui a été lu est un nouveau journal sous le même nom, dit une fois puis
relu depuis l'offset 0. `node`, `kind` et `seq` sont les estampilles de l'arbre et l'emportent
sur un champ de ligne du même nom. Un consommateur dédoublonne sur la paire de l'écrivain
(`pid`, `journal_seq`), qui repart à chaque brain, et reprend sur `offset`, qui appartient au
fichier.

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
| `probe.list` | `with_cursor?` | `{nodes: [{name, kind, state, every_ms, ttl_ms, seq, errors, bytes, children, rate_per_s, created_seq, max_rate_per_s?, export?, parent?, filter?, expires_in_ms?}], count, max_nodes, rate_per_s, max_rate_per_s, tree_rate_per_s, cursor?: {seq, epoch}}` (`state` aussi `stopping` pour une feuille dont l'arrêt est en cours ; `cursor` avec `with_cursor`) |
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
d'abord `epoch-changed`, puis l'anneau du nouveau démarrage. Le journal n'est pas rejoué ; une
restauration durable serait un contrat distinct qui s'annoncerait lui-même (D9, ix#414 ;
porte 3 intacte).

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
et l'enveloppe de chaque événement `ctl`. Le plan ne le rejoue jamais au démarrage : l'état désiré
ne survit pas à un redémarrage ; le nouveau démarrage tire un nouvel `epoch`, commence à la
génération 0 avec tous les services déclarés arrêtés, puis applique les services marqués
`autostart` en génération 1. Les documents appliqués sont partiels, les ajouts ne sont pas
synchronisés par `fsync`, les erreurs d'écriture sont ignorées et un crash peut couper la dernière
ligne (l'en-tête du démarrage suivant commence alors sa propre ligne) ; aucun enregistrement n'est donc un point de validation, et une restauration durable serait
un contrat distinct qui s'annoncerait lui-même (porte 3).

## Service Windows

`ixctl service run <spec.json> --state DIR` héberge une instance pour un gestionnaire de services :
le SCM de Windows quand c'est lui qui a lancé le processus, sinon la console (une interruption vaut
un arrêt). C'est la même instance qu'`ixctl up`, avec son verrou, son jeton, son écoute en boucle
locale et ses Job Objects. Ce que l'hôte ajoute :

- **États.** Il rapporte `start-pending`, puis `running` dès que l'instance sert (arrêt et extinction
  acceptés), puis `stop-pending` avec un point de contrôle qui croît et le temps restant comme délai
  annoncé, puis `stopped`. Codes de sortie, propres au service : 0 un arrêt observé ; 1 l'instance
  n'a pas démarré ; 2 le nom est tenu par une instance en marche (rien n'est rapporté `running`) ;
  3 les arbres n'ont pas été vus disparus dans `--stop-timeout` (30 s par défaut) — le processus
  sort quand même et les jobs se ferment avec lui, ce qui tue ce qui reste, mais l'hôte ne prétend
  pas l'avoir vu ; de même si un arbre tenait encore un processus quand le superviseur a cessé de
  l'attendre. En console, le processus sort avec le même code. Sous le SCM, la spec, le répertoire
  d'état et le journal du service sont ouverts après la connexion au SCM : un échec à cette étape
  est le code 1, que le SCM voit.
- **Un arrêt en trois moments.** `host.stop` sur `ctl` avec `phase: requested` (la commande a atteint
  l'hôte ; `source` vaut `scm`, `signal` ou `api` pour `/v1/shutdown`), puis `phase: received` (le
  plan n'admet plus rien : un apply répond 503), tous deux avant l'arrêt de tout service ; puis
  `observed` (tous les arbres vus disparus dans la borne, avec `ms`) ou `unobserved` (avec
  `bound_ms` ou `trees_not_seen_gone`), dans le journal du service seulement, puisque le journal du
  plan est fermé à ce moment. Pour `api`, le plan a fermé ses admissions en prenant la requête, avant
  que l'hôte la voie : un apply peut alors recevoir 503 avant l'un ou l'autre événement.
- **Journal du service.** `<nom>.service.log` dans le répertoire d'état, un objet JSON par ligne
  (`host.ready`, `host.refused`, `host.failed`, `host.stop`), jamais le jeton ; borné à deux fichiers
  de 4 Mio (`.service.log` et `.service.log.1`).
- **Après un crash.** Les jobs meurent avec l'hôte. Le démarrage suivant tire un nouvel epoch, en
  génération 0, comme tout redémarrage (voir Journal) : ce que l'hôte mort devait faire tourner n'est
  pas relancé, et une exécution coupée par le crash reste l'effet que run/1 appelle `unknown`, à
  réconcilier par une personne ou un client.

`ixctl service plan --spec FICHIER --client-user DOMAINE\UTILISATEUR` affiche les commandes
d'installation, de démarrage, de statut, d'arrêt et de désinstallation, pour un PowerShell élevé ;
ixctl n'en exécute aucune. Le plan utilise le compte virtuel `NT SERVICE\ixctl` (aucun mot de passe
enregistré), un démarrage manuel, aucun redémarrage automatique (un plan redémarré part de sa spec,
pas de ce qu'il faisait tourner), la boucle locale seulement, et un répertoire d'état sous
`C:\ProgramData\ixctl` dont l'ACL donne la lecture au seul utilisateur client nommé, qui y lit le
jeton. Chaque arrêt du plan attend que le service soit arrêté (`sc.exe stop` ne fait que demander)
avant la suite. Le plan lit la spec pour le nom de l'instance ; un test en vérifie le contenu, mais
rien n'en a encore été exécuté sur un Windows installé. Le compte du service n'a aucun identifiant Claude : le travail qui en a besoin reste dans un
`ixctl up` de la session de l'utilisateur, pair du service.

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
52 vérifications, E1..E24 et M3 (E19 : seul un tuyau de sonde que le pair exporte et que le parent demande traverse le lien, par nom exact, le statut disant ce qu'est chaque nom demandé ; E20 : une enveloppe miroir porte les `src_seq` / `src_epoch` de la source, qui nomment l'enveloppe du pair portant le même enregistrement ; E21 : le lien marque ce que le miroir saute, un `link.gap` sur le tuyau au démarrage en direct et sur le `ctl` miroir au redémarrage du pair ; M3 : huit tuyaux à 10 ms à travers le lien, les enregistrements reflétés, les pertes comptées et marquées, et le temps qu'un `probe.created` met à traverser sous la charge ; E22 : autour du redémarrage de b, le statut du parent distingue un flux en direct d'un état réconcilié et ne lève `degraded` que lorsque les deux tiennent ; E23 : l'identité d'une sonde concorde entre la liste de b, la liste routée de a et le statut de a, et la liste de b repliée depuis son propre flux jusqu'au curseur est exactement la liste, sans trou ; E24 : les totaux qu'annoncent les deux README, la docstring du gate et le commentaire du workflow CI (`.github/workflows/ci.yml`) correspondent aux vérifications exécutées, chaque énoncé de chaque texte, si bien qu'une vérification ajoutée sans que les textes suivent fait échouer le gate), une boucle de
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
écrit pour en trouver un. La CI lance aussi `go vet` et `go test -race` sous Linux ; depuis Windows,
`pwsh scripts/ixctl-test-linux.ps1` lance la même chose dans une distro WSL avant un push (la suite
peut passer ici et échouer là : un test qui a besoin qu'un abonnement déborde pose le bouton
`writeDelay` du serveur, car les tampons loopback de Linux absorbent une rafale que ceux de Windows
ne tiennent pas).

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
