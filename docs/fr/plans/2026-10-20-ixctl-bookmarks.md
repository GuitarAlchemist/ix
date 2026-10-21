---
title: "Signets ixctl : un point nommé de la séquence de l'instance, rejoué depuis l'anneau (story S8)"
type: decision
status: tranche sur anneau fusionnée dans exp/ixctl ; surface provisoire ; D9 résolue à titre provisoire (signature de l'opérateur en attente)
date: 2026-10-20
branch: exp/ixctl (PR #394)
reversibility: à double sens aujourd'hui (le fichier des signets n'est lu que par le plan qui l'a écrit ; les ops sont derrière le protocole provisoire de la porte 1) ; devient à sens unique quand un programme hors du plan garde des noms de signets, ou quand le journal devient une source de rejeu (porte 3)
revisit-trigger: D9 (ix#414) signée autrement ; un rejeu voulu au-delà de l'anneau (des secondes au rythme des sondes) ; Gaia ou le client Rust rejoue depuis un signet (S4, S7)
---

# Signets ixctl : un point nommé de la séquence de l'instance, rejoué depuis l'anneau

Anglais : [`docs/plans/2026-10-20-ixctl-bookmarks.md`](../../plans/2026-10-20-ixctl-bookmarks.md).
Suivi : epic [ix#397](https://github.com/GuitarAlchemist/ix/issues/397), décision D9
[ix#414](https://github.com/GuitarAlchemist/ix/issues/414), story S8
[ix#415](https://github.com/GuitarAlchemist/ix/issues/415). Référence utilisateur :
[`go/ixctl/README.fr.md`](../../../go/ixctl/README.fr.md#signets-et-rejeu). Les sondes, avec
lesquelles ceci se compose : [`2026-10-20-ixctl-probes.md`](2026-10-20-ixctl-probes.md).

## Le problème

L'opérateur veut **dépanner après coup** : « les états des pipelines devraient supporter les
bookmarks et replays ». Un arbre de sondes alimente un pipeline ; quelque chose semble anormal ;
l'opérateur veut marquer ce moment et rejouer plus tard ce que les tuyaux et le pipeline ont
porté autour, dans l'ordre. Aujourd'hui un lecteur reprend depuis `since` + `epoch` dans l'anneau
du hub, et un seq est un nombre que personne ne nomme ni ne garde ; le journal ne tient que `ctl`
et rien ne le rejoue (porte 3).

## Conception (telle que construite)

- **Un signet nomme un point de la séquence de l'instance** : `{name, epoch, seq, t, note}`. Le
  `seq` du hub est global à l'instance, partagé par tous les canaux, donc un seul signet suffit à
  couvrir les tuyaux de tout un arbre, ceux des services et le cycle de vie `ctl` au même
  instant. Les alternatives considérées : une position sur un canal (il en faudrait une par tuyau,
  et elles ne s'aligneraient pas), un ensemble de positions (la même chose, en plus lourd), un
  temps (`t` est gardé sur le signet pour l'humain ; un rejeu se fait par seq, qui est exact).
- **« Ici », c'est la tête.** `bookmark.set` sans `since` prend la tête du hub ; avec `since` le
  point doit être au plus la tête et, si `epoch` est donné, de ce démarrage. Un rejeu depuis un
  signet est un abonnement depuis `since = seq` (tout ce qui suit le point), par le même
  `Hub.Subscribe` que tout lecteur ; rien de nouveau dans le hub.
- **Le plan les possède.** `ixctl up --bookmarks disk` (par défaut) les garde dans
  `<nom>.bookmarks.json` à côté du journal (`{"schema": "ixctl-bookmarks/0", "bookmarks": [...]}`),
  réécrit en entier à chaque changement : un fichier temporaire, puis un renommage, mode 0600,
  pour qu'un plantage laisse l'ancien fichier ou le nouveau. `--bookmarks memory` les garde pour
  ce démarrage seulement (l'opérateur a demandé les deux magasins, 2026-10-20). Un fichier que le
  plan ne peut pas lire comme des signets fait échouer le démarrage avec le nom du fichier, au
  lieu d'être vidé. Le magasin est un choix au démarrage, pas dans la spec : il ne dit rien du
  monde déclaré, comme `--state`.
- **L'anneau est la seule source de rejeu** (tranche 1). Un signet plus vieux que l'anneau
  répond le gap que le hub produit déjà (`since-too-old`, avec le plus vieux seq gardé), puis ce
  que l'anneau tient : jamais un rejeu silencieusement partiel. Un signet d'un démarrage antérieur
  répond `epoch-changed`, puis l'anneau du nouveau démarrage. Le rejeu depuis un journal au-delà
  de l'anneau est la seconde tranche, suspendue à D9 ; la porte 3 n'est pas touchée par celle-ci.
- **Les transports restent minces.** `subscribe` (ixctl.v1) et `GET /v1/events` acceptent
  `bookmark` à la place de `since`/`epoch` ; les deux ensemble sont refusés (« bookmark or
  since »). Le transport résout le nom par `Broker.Request(bookmark.get)`, donc l'interface
  `Broker` ne change pas et un transport n'entre jamais dans le plan.
- **Des bornes que le système rapporte** : `max_bookmarks` 64 par plan, le nom selon la règle des
  noms d'instance, `max_note_len` 256, `ring` 4096 ; `bookmark.list` les rapporte avec `count` et
  `store` ; chaque refus nomme ce qu'il a touché. Poser un nom existant déplace son point (une
  note donnée remplace la sienne, une note vide l'efface, pas de note la garde) : un signet est
  un curseur que l'opérateur déplace, pas un journal. Un fichier chargé est tenu aux mêmes
  invariants (nombre, règle des noms, noms uniques, longueur de la note, un epoch sur chaque
  enregistrement) sinon le démarrage est refusé ; un changement que le magasin ne peut pas
  sauver n'est pas adopté, donc un lecteur ne voit jamais un signet que le disque a refusé.

## Revue 21 (Codex, lecture seule, de `1ca7ec6`)

Pas de P0. Deux P1 : un `save` raté laissait la mutation visible (`set`/`remove` construisent
maintenant une liste candidate, la sauvent, puis l'adoptent) ; un `EventSource` qui se
reconnectait à un flux `?bookmark=` envoyait `Last-Event-ID` et était refusé avec « bookmark or
since » (l'en-tête l'emporte désormais sur le signet ; un `since`/`epoch` explicite dans la
requête reste refusé à côté). Quatre P2, tous pris : une note vide ne pouvait pas effacer
l'ancienne (`note` est désormais un pointeur : absente garde, présente remplace) ;
`max_note_len` comptait des octets (désormais des caractères) ; les enregistrements chargés
contournaient les invariants (désormais vérifiés, le démarrage refusé en nommant
l'enregistrement) ; le « lecteur en direct » de la barrière utilisait `since=0` et aurait pu
passer par le seul rejeu (il s'abonne désormais en direct, après une poignée de main SSE, avant
que l'arbre existe).

## Revue 22 (Codex, lecture seule, de `e7cc088`)

Pas de P0. Un P1 : le gestionnaire SSE envoyait ses en-têtes avant de créer son curseur, donc un
lecteur qui prenait les en-têtes pour « sur le tuyau » (l'E16 de la barrière le fait) pouvait
manquer ce qui était publié dans cette fenêtre ; le curseur est désormais créé d'abord. Un P2 :
la règle de reconnexion était sous-documentée ; la précédence sur `/v1/events` est désormais
énoncée (l'en-tête l'emporte sur un `since` de la requête et sur un `bookmark` ; un seq nu dans
l'en-tête prend l'`epoch` de la requête ; sans l'en-tête, `bookmark` à côté de `since`/`epoch`
est refusé).
- **Relayable comme les ops des sondes** : `bookmark.*` avec `peer` s'exécute chez le pair, dans
  la séquence du pair, dans le magasin du pair ; un rejeu depuis lui se fait chez ce pair
  (subscribe est local).

## Décisions

### Prises dans cette tranche (D9, provisoire)

1. Forme : un point de la séquence de l'instance, couvrant tous les canaux. 2. Propriétaire : le
plan, fichier `<nom>.bookmarks.json` ou mémoire, au choix au démarrage. 3. Source du rejeu :
l'anneau ; le journal n'est pas rouvert. 4. Bornes : 64 signets, note 256, règle des noms
d'instance. 5. La porte 3 reste telle que signée (journal d'audit, jamais rejoué). Signature de
l'opérateur sur D9 en attente ; consigné sur ix#414.

### Laissées à la seconde tranche

Un journal par canal que le plan écrirait à la demande, rejoué au-delà de l'anneau, avec les
bornes de rotation qu'avait le JSONL du prototype ; et si un signet devrait publier son propre
événement `ctl` (pour que le journal montre que quelqu'un a marqué un moment).

## Mesuré (Windows, 2026-10-20, barrière E16 sur un poste chargé)

Un `mux@1` sur un `heartbeat@1` à 20 ms avec un filtre, lancé après `before` et démonté après
`after`, posé une seconde plus tard. Le lecteur en direct sur `probe/` a vu 48 enveloppes entre
les deux signets ; le rejeu depuis `before` a redonné les mêmes 48, seq pour seq, sans gap. Le
rejeu de tout depuis `before` a donné 56 enveloppes sur `ctl`, `probe/agg3` et `probe/hb3` en une
seule séquence strictement croissante, les deux premières `probe.created` pour le mux et la
feuille. Un signet au seq 1 après que l'anneau eut tourné a répondu `since-too-old` d'abord (plus
vieux gardé 3 600+), puis l'anneau. Un signet posé via le plan A sur le plan B était dans la liste
de B et pas dans celle de A, et après le redémarrage de B il était encore listé, `live: false`, et
son rejeu commençait par `epoch-changed` nommant le nouveau démarrage.

## Tests (les coutures convenues pour le TDD)

Unitaires, `go/ixctl/bookmark_test.go`, par `Plane.Request`, le WebSocket et SSE : un signet posé
« ici » rapporte la tête ; le rejeu du tuyau d'une feuille depuis lui est, seq pour seq, ce qu'un
lecteur en direct a vu ; le rejeu de tout depuis lui est le cycle de vie et les deux tuyaux après
le point, dans l'ordre ; la forme SSE ; la liste avec ses bornes et `live` ; chaque refus nommant
sa cause (nom inconnu, signet à côté de since, règle des noms, since au-delà de la tête, epoch
étranger, longueur de la note, `max_bookmarks`) ; poser un nom existant le déplace ; persistance
à travers un redémarrage du même dossier d'état avec le gap `epoch-changed` ; un signet plus
vieux que l'anneau avec `since-too-old` ; le fichier sans jeton ; le magasin mémoire n'écrivant
aucun fichier et ne gardant rien à travers un redémarrage ; un nom de magasin qui n'est ni l'un
ni l'autre refusé ; le relais vers un pair en HTTP. Barrière : E16, trois vérifications dans
`go/ixctl/e2e/e2e.py`, 42 vérifications en tout.

## Laissé de côté exprès

Le rejeu depuis un journal au-delà de l'anneau (D9, seconde tranche) ; un événement `bookmark`
sur `ctl` ; la surface Rust et IXQL (`ix.ctl.subscribe(bookmark:)`, S4) ; des signets sur un
canal de pair en miroir (S2) ; la CLI (`ixctl bookmark`, rien qu'un `curl` ne fasse déjà).

Une fois D9 signée, ces décisions ont leur place dans un ADR sous `docs/adr/`.
