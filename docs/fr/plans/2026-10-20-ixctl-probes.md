---
title: "Sondes ixctl : un arbre d'échantillonneurs en processus sous le plan de contrôle (story S1)"
type: decision
status: traceur intégré à exp/ixctl ; surface provisoire ; décisions D1–D5 supposées, tickets ouverts
date: 2026-10-20
branch: exp/ixctl (PR #394)
reversibility: à double sens aujourd'hui (aucun consommateur hors go/ixctl ; la surface est derrière le protocole provisoire du plan, porte 1) ; devient à sens unique quand un programme hors du plan crée des nœuds ou lit des tuyaux `probe/`
revisit-trigger: un ticket de décision D1–D5 est tranché contre l'hypothèse notée ici ; ou Gaia / le client Rust lit un tuyau de sonde (S4, S7) ; ou le premier tuyau inter-hôtes est voulu (D4 / S2)
---

# Sondes ixctl : un arbre d'échantillonneurs en processus sous le plan de contrôle

English: [`docs/plans/2026-10-20-ixctl-probes.md`](../../plans/2026-10-20-ixctl-probes.md).
Suivi : epic [ix#397](https://github.com/GuitarAlchemist/ix/issues/397), carte Wayfinder
[ix#398](https://github.com/GuitarAlchemist/ix/issues/398), story S1
[ix#407](https://github.com/GuitarAlchemist/ix/issues/407). Référence utilisateur :
[`go/ixctl/README.fr.md`](../../../go/ixctl/README.fr.md#sondes).

## Le problème

L'opérateur veut **démarrer des sondes** depuis le plan de contrôle : des échantillonneurs
versionnés, écrits en Go pour l'instant, compilés vite, qui lisent quelque chose (l'hôte, un
service, plus tard un fichier ou un port) et mettent ce qu'ils lisent sur un tuyau ; les
**lister, créer, démarrer, arrêter** ; **filtrer et agréger** leur sortie ; **multiplexer** des
enfants dans un parent, le parent dans un grand-parent, en arbre ; donner à chacune une **durée
de vie** ; et connaître **les limites du système** en le faisant. La douleur aujourd'hui : le plan
rapporte ce qu'un *service* coûte (`last_exit`) et rien sur l'hôte ni sur l'intérieur d'un
service en marche, et les sondes prototypes (PR #390) sont un processus OS par sonde, hors de tout
plan, avec leur propre transport.

## Travaux antérieurs lus avant de concevoir

- `prototypes/probes` (PR #390) : un hôte Go qui lance des sondes modèles (WMI) en processus et
  écrit du JSONL pour DuckDB. Utile comme source de kinds futurs ; mauvaise forme pour le plan (un
  processus par sonde, pas d'arbre, pas de limites).
- ix#374 (frontière du runtime de sondes) : files bornées, **aucune perte silencieuse**,
  définitions versionnées séparées des observations, un hôte qui multiplexe des sondes logiques,
  mesure séparée de l'évaluation composite, pas d'actionneurs. Cette conception suit chaque point.
- ix#375 (langages) : Go d'abord, Rust ensuite.
- gaia#245 : Gaia consommateur d'observations, par une enveloppe (S7, D7).
- Le plan lui-même : le hub (anneau 4096, canaux, curseurs à gaps explicites), le point d'entrée
  unique `Request` sur HTTP et WebSocket, la fédération qui met en miroir `ctl` et `svc/` d'un
  pair, le jeton porteur et la face bouclage local seulement. Tout ce dont une sonde a besoin pour
  publier, être pilotée et être protégée existe déjà ; la tranche n'ajoute ni second hôte, ni
  transport, ni jeton.

## Conception (telle que construite)

**Les kinds** sont compilés dedans, versionnés, nommés `nom@version` : `heartbeat@1` (son propre
compte de ticks), `cpu@1` (part d'occupation depuis `GetSystemTimes` / `/proc/stat` ; le premier
intervalle n'émet rien), `mem@1` (`GlobalMemoryStatusEx` / `/proc/meminfo`), `mux@1` (pas
d'échantillonneur). Un kind est un port `sampler` (`sample() (champs, erreur)`) avec un adaptateur
OS par plateforme et un adaptateur heartbeat pour les tests : le même hexagone léger que le plan
utilise déjà pour son `Broker`.

**Les nœuds** sont des instances dans un arbre par plan : `{name, kind, every_ms, parent, filter,
ttl_ms}`. Un nœud publie des événements `sample` sur son propre canal du hub `probe/<nom>` ; quand
son parent tourne, l'échantillon remonte de façon synchrone, à travers le filtre du parent, sur
`probe/<parent>` estampillé `src`. Le cycle de vie (`probe.created/started/stopped/removed`, `stopped`
et `removed` avec une `reason`) est sur `ctl` comme tout événement du plan : journalisé et mis en miroir chez
les pairs sans rien ajouter.

**Les filtres** s'exécutent avant l'émission : `where` (termes typés, six opérateurs, champ absent
= faux), `every` (un sur N de ce qui est passé), `keep` (projection de champs ; `node/kind/seq/src`
toujours gardés). Ce sont des fonctions pures, testées par valeur.

**Les baux** : `ttl_ms` compte depuis `probe.start`, `probe.touch` renouvelle, l'expiration retire
le sous-arbre avec la raison `expired`. Un nœud sans `ttl_ms` n'a pas de bail ; `touch` le refuse
plutôt que d'en inventer un.

**Les ops** : `probe.kinds / list / create / start / stop / touch / remove`, un point d'entrée sur
les deux transports, routes HTTP `GET /v1/probes`, `GET /v1/probe`, `POST /v1/probe/<op>`, chaque
op relayable à un pair. `ixctl probes` imprime les kinds et les limites hors ligne : un binaire de
release se décrit lui-même. `POST /v1/shutdown` arrête d'abord chaque nœud en marche (raison
`shutdown`).

**Les limites font partie de la surface.** Chaque borne est une constante nommée que
`probe.kinds` rapporte sous `limits` (`max_nodes` 256, `max_depth` 8, `min_every_ms` 10,
`max_every_ms` 3 600 000, `max_ttl_ms`, `ring` 4096, `max_where` 16, `max_keep` 32) ; chaque refus
nomme la borne touchée ; `probe.list` montre l'usage (`count`, `max_nodes`). L'opérateur l'a
demandé explicitement.

## Décisions

### Prises dans cette tranche

- **Propagation enfant→parent synchrone, sans boîte de réception.** Le premier jet donnait à
  chaque mux une boîte bornée avec un compteur `dropped`. Remplacé avant que les tests passent au
  vert : une boîte ajoute une file dont le seul comportement honnête sous pression est de perdre,
  ce qu'ix#374 interdit, et l'anneau du hub est déjà l'unique tampon borné, avec un `gap`
  explicite. Un échantillon atteint maintenant le filtre du parent dans le tick de l'enfant, ou
  l'enfant attend le verrou du parent quelques microsecondes. Le coût : un filtre lent ralentit
  ses enfants ; les filtres sont purs et bornés par `max_where` / `max_keep`, c'est accepté. À
  revoir si un kind échantillonne un jour quelque chose qui peut bloquer (un socket, un fichier) :
  ce kind reçoit une échéance, pas le mux une file.
- **Un arbre par plan, dans le processus du plan.** Pas de second hôte. Un kind qui ne doit pas
  partager le processus du plan (un lecteur sujet aux plantages) sera un kind ultérieur qui lance
  un processus, pas un changement de l'arbre.
- **Kinds compilés, pas chargés.** Un nouveau kind est une release (D8 / S5), jamais un plugin.

### Supposées, avec leur ticket de décision (la tranche garde chacune réversible)

| Ticket | Hypothèse construite | Ce qui la changerait |
|---|---|---|
| D1 ix#399 | Les sondes ont leurs propres ops, pas une place dans le document d'état désiré (porte 2). | Si les sondes doivent être déclarées et réconciliées comme des services, `probe.create` devient une action de plan/apply. Le code de l'arbre ne change pas ; le point d'entrée se déplace. |
| D2 ix#400 | Le TTL est un bail renouvelé par `touch`, pas une inactivité sur les échantillons. | Une règle d'inactivité lirait l'heure du dernier échantillon au lieu de celle du touch : un champ. |
| D3 ix#401 | Deux nœuds d'un même kind échantillonnent indépendamment. | Un échantillonnage partagé mémoïserait par kind et intervalle ; le port sampler le permet déjà. |
| D4 ix#402 | Les tuyaux sont locaux ; le cycle de vie des sondes d'un pair est en miroir (`peer/<nom>/ctl`), ses tuyaux non. | Mettre les canaux `probe/` en miroir est un préfixe de plus dans `peers.go` une fois le volume et la sémantique des gaps entre hôtes tranchés (S2). |
| D5 ix#403 | Un filtre est `where` + `every` + `keep`, termes typés, pas d'expressions. | IXQL ou un langage d'expressions (D7) se placerait au-dessus, abaissé vers ceci ou le remplaçant. |

## Mesuré (Windows, 2026-10-20, barrière E15 / M2 sur un poste chargé)

| Mesure | Valeur |
|---|---|
| Échantillons filtrés arrivés au mux en 1,5 s d'un heartbeat à 20 ms (`n > 2`, `every` 2) | 36, valeurs 4, 6, 8, … |
| M2 : échantillons par seconde sur un mux nourri par 16 heartbeats à 10 ms, lus en SSE | 1447 sur un plafond de 1600, 16 sources, aucun gap de débordement |
| M2 : temps processeur du plan sur la fenêtre de 2,2 s | 297 ms (environ un huitième d'un cœur, 3 200 enveloppes publiées par seconde) |
| Bail de 1,5 s non renouvelé | retiré avec la raison `expired`, vu sur `ctl` et absent de `probe.list` |

M2 a aussi mis au jour une fragilité préexistante de la barrière : à 6 400 enveloppes par
seconde l'anneau ne tient plus la seq 1, donc une lecture SSE ultérieure depuis le début reçoit
d'abord un gap `since-too-old`. L'aide `sse()` de la barrière sépare maintenant les gaps des
enveloppes (`SSE_GAPS`) ; le comportement du plan était déjà celui documenté.

## Revue round 20 (Codex, lecture seule, de `2575167`)

Quatre P1 et huit P2, aucun P0 ; tous traités dans le commit suivant. Les P1 étaient des courses
entre une op qui relâche le verrou entre décider et agir et une autre qui s'intercale : `remove`
prenait l'instantané de son sous-arbre puis relâchait, donc un `create` dessous laissait un
orphelin et un `start` relançait un nœud sur le point d'être supprimé (son exécution fuyait) ;
`expire` vérifiait son échéance, relâchait, puis retirait, donc un `touch` arrivé avant perdait
quand même son nœud ; `shutdown` arrêtait les nœuds en marche sans fermer l'admission, donc un
`start` après lui tournait au-delà de la fermeture du journal ; et le `keep` d'un mux perdait
`src`, contre la promesse du protocole. Les correctifs gardent la forme : le sous-arbre est
réclamé (`removing`) sous le verrou même qui décide, et les nœuds réclamés refusent `start`,
`touch` et les nouveaux enfants ; `expire` est un `removeIf` dont la vérification d'échéance est
sous ce verrou ; `shutdown` pose `closed` avant son instantané et `create`/`start` refusent
ensuite ; `keep` porte toujours `src` et `error`. P2 : `emit` revérifie que le nœud est bien
celui de l'arbre et en marche (l'échantillon d'un enfant en vol n'atterrit plus après le
`probe.stopped` du parent) ; un nœud `stopping` refuse un start jusqu'au retour de son
exécution ; `max_depth` admet exactement `max_depth` mux au-dessus d'un nœud ; les valeurs des
filtres sont typées à la création ; le total de `/proc/stat` sous Linux s'arrête à `steal` (le
temps guest est dans user) ; un intervalle cpu dont les compteurs n'ont pas avancé n'émet rien
au lieu d'un repos fabriqué ; `probe.started` porte `expires_in_ms` quand il y a un bail. Codex a
aussi compté 38 vérifications dans la barrière ; le fichier en a 39 et l'exécution en imprime 39.

**Ordre sur le fil (2026-10-21, sur le feu vert de l'opérateur après que le contrat S7 a nommé les
lacunes G6 et G7).** L'enregistrement d'un nœud est publié sous le verrou de l'arbre avec son
`seq`, donc les enregistrements d'un mux atteignent son tuyau dans l'ordre sous des enfants
concurrents (vu d'abord comme un échec CI du test du mux sous Windows) et jamais après le
`probe.stopped` du nœud ; `probe.created`, `probe.started`, le `probe.stopped` d'un mux et
`probe.removed` sont publiés sous ce même verrou, dans l'ordre des changements d'état qu'ils
rapportent ; le `probe.stopped` d'une feuille attend sa boucle hors du verrou, et un remove qui
rencontre cet arrêt en cours l'attend (`sync.Cond` sur le verrou de l'arbre) avant de dire
`probe.removed` ; une lecture ratée atteint le tuyau quel que soit le filtre. Tests dans
`go/ixctl/probe_order_test.go` ; le contrat est en v0.1.1, G6 et G7 fermées. La ronde Codex 28
(de ce commit) a ajouté : `shutdown` attend un arrêt qu'une autre opération avait en cours (son
instantané ne liste que les nœuds en marche, donc le `probe.stopped` de cette feuille pouvait
suivre le retour du shutdown et la fermeture du journal) — corrigé avec la même variable de
condition ; les tests d'arrêt en cours pilotent désormais un vrai `probe.stop` à travers un kind
de test dont la lecture bloque jusqu'à libération (`block@1`, enregistré le temps d'un test), et le
test start/stop juge `ctl` après chaque tour. **Compromis gardé :** l'ajout au journal `ctl`
s'exécutait déjà sous le verrou du hub dans `Publish` ; avec les événements de cycle de vie publiés
sous le verrou de l'arbre, une écriture de journal bloquée retient désormais aussi les opérations
de sondes, pas seulement les publieurs. Déplacer le journal derrière un écrivain ordonné change le
moment où une ligne est sur disque, ce qui relève de la porte 3 : pas fait ici.

## Tests (les coutures convenues pour le TDD)

Unitaires, `go/ixctl/probe_test.go`, à travers `Plane.Request`, les filtres purs et les kinds OS
par plage de valeurs : création/arrêt/refus nommant les bornes ; le heartbeat échantillonne sur
son tuyau jusqu'à l'arrêt ; un filtre façonne ce qui est émis ; un mux estampille `src` ; un bail
expire sauf renouvellement et un remove emporte le sous-arbre ; cpu et mem dans les bornes ; les
kinds listés avec les limites ; les routes HTTP, la trame WebSocket et le shutdown qui termine
chaque exécution ; le relais à un pair avec son cycle de vie en miroir ; la CLI qui imprime les
kinds hors ligne. Barrière : E15 (deux vérifications) et M2 dans `go/ixctl/e2e/e2e.py`,
39 vérifications.

## Laissé de côté à dessein

Les tuyaux inter-hôtes (S2), l'échantillonnage partagé (S3), la surface Rust et IXQL (S4), les
releases GitHub avec un manifeste de sondes au-delà du squelette de workflow (S5), les kinds WMI du
prototype (S6), l'adaptateur de l'enveloppe Gaia (S7 : le côté IX est le brouillon de contrat
[`2026-10-21-ixctl-probe-observation.contract.md`](../../contracts/2026-10-21-ixctl-probe-observation.contract.md),
en anglais comme les autres contrats, avec trois fixtures consommateur écrites par le vrai plan,
reconnexion, recréation du même nom et redémarrage, le 2026-10-21 sur le feu vert de l'opérateur
après la revue de conception de Gaia ; l'adaptateur est à Gaia), le rejeu depuis un journal
au-delà de l'anneau (la seconde tranche de D9 / S8 ; les signets sur anneau sont livrés sur
[`2026-10-20-ixctl-bookmarks.md`](2026-10-20-ixctl-bookmarks.md)), les actionneurs (jamais :
ix#374).

Une fois D1–D5 tranchées, ces décisions ont leur place dans un ADR sous `docs/adr/`.
