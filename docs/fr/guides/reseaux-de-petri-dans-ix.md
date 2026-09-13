# Réseaux de Petri dans IX — quand y recourir, et quand s'en abstenir

**Objectif :** `ix-petri` modélise les systèmes qui combinent **cycles,
concurrence et contention de ressources**, et répond aux questions de
comportement les concernant : cela peut-il s'interbloquer, cette file peut-elle
croître sans borne, cette transition peut-elle seulement se déclencher un jour.
Cette page indique quand c'est le bon outil, et quand une crate IX existante
l'est davantage.

**Audience :** les humains et les sessions Claude Code qui choisissent entre
`ix-petri`, `ix-pipeline`, `ix-graph` et `ix-fuzzy`.

**English:** [docs/guides/petri-nets-in-ix.md](../../guides/petri-nets-in-ix.md)

---

## 1. Choisir d'abord la bonne crate

La plupart des tâches « machine à états » dans IX sont déjà couvertes. Ne
recourez à `ix-petri` que si la ligne ci-dessous est vraiment la vôtre.

| Votre question | Utiliser | Pourquoi |
|---|---|---|
| « Dans quel ordre ces tâches s'exécutent-elles ? Quel est le chemin critique ? » | `ix_pipeline::dag::Dag` | Un DAG, avec tri topologique, niveaux parallèles et chemin critique. Rejette les cycles par construction. |
| « Quelle est la probabilité que le système soit dans l'état X ? » | `ix_graph::markov`, `hmm`, `state_space` | Évolution *probabiliste* — distributions, Viterbi, comportement stationnaire. |
| « À quel point cette affirmation est-elle vraie ? » | `ix_fuzzy`, `ix_types::Hexavalent` | Degrés de vérité, pas flot de contrôle. |
| « Cette chaîne est-elle analysable / comment en générer une ? » | `ix_grammar` (`ebnf`, `abnf`, `weighted`, `constrained`) | Dérivation, pas concurrence. |
| **« Ces deux voies peuvent-elles se bloquer mutuellement ? Cette file peut-elle croître indéfiniment ? Cette étape peut-elle seulement s'exécuter ? »** | **`ix-petri`** | Cycles + concurrence + contention, et la réponse est une *possibilité*, pas une probabilité. |

La ligne de partage est nette et mérite d'être répétée : **un DAG ne peut pas
exprimer une ressource rendue à un pool puis reprise**, car c'est un cycle et
`Dag::add_edge` le refuse. Si votre modèle comporte un verrou, un pool de
travailleurs, un tampon à nombre fini d'emplacements, ou une voie qui revient à
`READY`, un DAG est la mauvaise forme et aucune précaution ne le rendra correct.

De même : si vous avez seulement besoin de *représenter* places et transitions
sans jamais poser de question de comportement, cette crate est inutile. Une
`HashMap` suffit. **Ce sont les propriétés qui constituent le produit.**

## 2. Ce qui est calculé

`ix_petri::analyze` énumère les marquages accessibles et en déduit cinq
propriétés. Chacune vaut `holds` (vérifiée), `fails` (fausse) **avec un
témoin**, ou `unknown` (indécise) — jamais une supposition.

| Propriété | Ce que `fails` fournit |
|---|---|
| **sans interblocage** | chaque marquage mort, avec la séquence de tir la *plus courte* qui y mène |
| **bornée** | un couple `m < m'` avec `m'` accessible depuis `m`, et la séquence intermédiaire — répétez-la et l'excédent s'accumule |
| **quasi-vivante** | les transitions activées dans aucun marquage accessible (des étapes écrites qui ne s'exécuteront jamais) |
| **vivante** (L4) | les transitions absentes d'une composante fortement connexe terminale |
| **réversible** | si le marquage initial est accessible depuis partout |

### La frontière d'honnêteté

L'énumération est bornée par `Limits::max_states` (50 000 par défaut). Cette
borne est précisément le point où l'analyse cesse d'affirmer :

- **Épuisée** — les résultats sont exacts.
- **Tronquée avec témoin de non-bornitude** — le témoin *prouve* que le réseau
  est non borné ; toute propriété portant sur l'espace d'états (désormais
  infini) devient `unknown`.
- **Tronquée sans témoin** — rien n'est affirmé du tout.

Un interblocage trouvé pendant une exécution tronquée reste rapporté : une
séquence témoin est une preuve d'existence positive que la troncature
n'invalide pas. En revanche, l'*absence* d'interblocage n'est jamais affirmée à
partir d'une exécution tronquée.

## 3. Déterminisme

L'ordre de tir est fixé par le type du réseau lui-même, et non choisi au site
d'appel. Places et transitions sont triées par `id` lors du `build()` et les
`id` dupliqués sont rejetés ; les identifiants sont donc uniques, et la
comparaison octet à octet de deux `id` distincts ne renvoie jamais `Equal` — un
ordre **total**, sans égalité résiduelle qu'une autre règle devrait départager.
C'est la même discipline, pour la même raison, que la règle de départage en bas
de [`crates/ix-duck/sql/pareto_frontier.sql`](../../../crates/ix-duck/sql/pareto_frontier.sql).

Conséquence : la numérotation des états, les séquences témoins et le point
d'arrêt d'une exécution tronquée sont identiques à chaque exécution et sur
chaque machine. L'ordre d'insertion dans le constructeur ne peut pas fuir dans
un résultat.

## 4. Interopérabilité : PNML, en lecture seule

`ix_petri::read_pnml` lit la sous-classe **Place/Transition** de PNML, le format
d'échange normalisé par **ISO/IEC 15909-2** (la partie 1 donne la sémantique, la
partie 3 le cadre d'extensibilité ; le site de référence est
<https://www.pnml.org/>, qui publie les grammaires RELAX NG).

Trois décisions, prises délibérément :

- **Lire, pas écrire.** La lecture permet à IX d'analyser des réseaux rédigés
  par des outils que personne ici n'a écrits — exactement la vérification
  indépendante dont ce dépôt a régulièrement besoin. L'écriture apporterait le
  bénéfice symétrique, mais constitue une tranche distincte : un émetteur doit
  satisfaire les lecteurs *d'autres* outils, et rien ici ne peut le vérifier.
  Voir `crates/ix-petri/src/pnml.rs` pour ce qu'un émetteur devrait d'abord
  franchir.
- **Réseaux P/T uniquement.** `ptnet.pntd` n'ajoute que deux étiquettes au
  modèle noyau — `initialMarking` sur une place, `inscription` sur un arc. Les
  réseaux symétriques et de haut niveau embarquent un système de sortes et une
  algèbre de termes : un marquage y est un multi-ensemble de jetons structurés,
  et rien de l'énumération faite ici ne s'y applique tel quel. Un document
  déclarant l'un d'eux est **rejeté nommément**, et non mal interprété.
- **Aucune dépendance XML.** Le workspace ne contient aucune crate XML ; en
  ajouter une pour analyser une grammaire dont le cœur sémantique tient en huit
  noms d'éléments ne valait pas cette famille de dépendances.
  `crates/ix-petri/src/xml.rs` est un analyseur strict d'environ 200 lignes qui
  rejette `DOCTYPE` d'emblée — sans déclaration d'entités, pas d'amplification
  « billion laughs » sur des fichiers reçus d'autres outils.

Pour l'utiliser sur un fichier :

```bash
cargo run -p ix-petri --example analyze_pnml -- chemin/vers/reseau.pnml
```

## 5. L'exemple concret à l'origine de la crate

`crates/ix-petri/tests/worktree_pump.rs` modélise le risque signalé par
`CLAUDE.md` dans le préambule chargé à chaque session : la pile `git stash` est
partagée entre le dépôt principal et tous les worktrees. Une voie de la pompe
détient deux choses à la fois — un arbre de travail et cette unique pile
partagée — et les voies bouclent.

Les tests établissent, par énumération et non par argumentation :

- deux voies acquérant les deux ressources dans des **ordres opposés peuvent se
  bloquer**, avec le témoin « L0 prend l'arbre → L1 prend la pile » et plus rien
  d'activable ensuite ;
- **une seule** voie divergente suffit, et ajouter des voies conformes ne répare rien ;
- un **ordre d'acquisition canonique** supprime l'interblocage pour tous les
  nombres de voies testés ;
- **donner un arbre à chaque voie** aussi — une ressource unique en contention
  ne peut pas provoquer d'interblocage.

C'est la forme de question à laquelle cette crate répond. Si vous ne pouvez pas
formuler votre problème ainsi, il vous faut probablement l'une des crates du §1.

## 6. Hors périmètre

Volontairement exclu de cette tranche, chaque point parce qu'il n'a pas encore
de consommateur, et non parce qu'il serait difficile :

- **Arcs inhibiteurs, capacités de places, priorités, temps.** Aucun n'appartient
  à la définition de type P/T de PNML ; les ajouter rendrait les réseaux d'IX
  illisibles par les outils conformes, ce qui annulerait tout l'intérêt du
  standard.
- **Écriture de PNML** (voir §4).
- **Réseaux colorés / de haut niveau** (voir §4).
- **Analyse structurelle sans énumération** — invariants P et T, siphons et
  trappes. Ils décident la bornitude et certaines questions de vivacité sur des
  réseaux bien trop grands pour être énumérés. À ajouter le jour où un réseau de
  ce dépôt dépassera `max_states` ; pas avant.
