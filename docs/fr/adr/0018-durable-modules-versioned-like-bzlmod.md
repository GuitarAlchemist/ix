# Modules durables : composés par `load` et des runs enfants, versionnés et verrouillés comme bzlmod, livrés comme pièces jointes de releases GitHub

Statut : proposée (2026-10-29)

Version anglaise : [`docs/adr/0018-durable-modules-versioned-like-bzlmod.md`](../../adr/0018-durable-modules-versioned-like-bzlmod.md).

S'appuie sur l'ADR-0016 (runs durables) et l'ADR-0017 (DSL d'agents).

## Contexte

L'opérateur a demandé des agents durables composables, versionnés et empaquetés, « un peu comme
Maven, Nexus et les poms », sans réinventer la roue (2026-10-28). Trois besoins :

- **Composition :** des bibliothèques Starlark partagées, et des agents qui exécutent d'autres
  agents (composites).
- **Versionnement :** un module a une version et déclare les versions dont il dépend, et leur
  résolution est reproductible.
- **Distribution :** une version est livrée sous la forme d'un fichier que n'importe qui peut
  récupérer et vérifier, depuis une release GitHub ou tout autre serveur.

Chaque besoin a une réponse éprouvée, et la plus proche est faite pour Starlark lui-même :

- **Le bzlmod de Bazel.** Un `MODULE.bazel` déclare `module(name, version)` et `bazel_dep(name,
  version)`, comme un pom. Les versions sont résolues par la sélection de version minimale (MVS),
  l'algorithme des modules Go : la plus haute version que demande un dépendant, et jamais une plus
  récente. Le résultat est reproductible et ne demande aucun solveur. Un fichier de verrouillage le
  fige. Un registre est un répertoire de métadonnées (le Bazel Central Registry est un dépôt Git),
  où le `source.json` de chaque version donne l'URL d'une archive et son empreinte d'intégrité. Des
  modules de `compatibility_level` différents ne peuvent pas être mélangés.
- **starlark-rust** fournit `load` par un `FileLoader` que l'hôte écrit
  (`eval/runtime/file_loader.rs:28`, `Evaluator::set_loader`). L'hôte décide de ce qu'un `load`
  peut atteindre.
- **Les workflows enfants de Temporal :** un workflow en démarre un autre avec son propre
  historique, et le parent enregistre le résultat de l'enfant.
- **Les releases GitHub** avec un fichier `SHA256SUMS` et une attestation d'artefact
  (`actions/attest-build-provenance`, vérifiée par `gh attestation verify`). IX le prévoit déjà
  pour `ixrelay` (#473, tranche S5) et le pack ComfyUI (ADR-0009), et les skills Claude viennent
  déjà de plugins versionnés (ADR-0008).

Pour les runs durables, le versionnement compte plus que partout ailleurs. Le rejeu réexécute le
script, si bien que chaque ligne qu'un run charge doit être la même ligne à chaque reprise
(ADR-0016 §2).

## Décision

**1. Un module est un répertoire avec un `MODULE.star`.** Ce fichier est évalué dans un dialecte
qui lui est propre, qui lui donne deux fonctions et rien d'autre :

```python
module(name = "fold-planner", version = "1.2.0", compatibility_level = 1)
dep(name = "origami-dsl", version = "0.3.1")
```

Un module contient des bibliothèques Starlark (`*.star`), des agents (des scripts qu'un run peut
démarrer) et des grammaires (`*.ebnf`, ADR-0017). Les noms font de 1 à 40 lettres, chiffres ou
`-`, et les versions suivent SemVer.

**2. Les versions sont résolues par MVS, avant tout run.** `ix-durable lock <module dir>` résout
le graphe des dépendances par MVS. À l'intérieur d'un même `compatibility_level`, il prend la plus
haute version que demande un dépendant. Deux niveaux d'un même module dans un graphe sont une
erreur. La commande écrit `ix-durable.lock`, qui enregistre pour chaque module :

- son nom, sa version et l'URL de sa source ;
- le SHA-256 de son archive ;
- le SHA-256 de chacun de ses fichiers.

La résolution a besoin du réseau, c'est donc une commande qu'une personne lance, comme `accept`.
Aucun run et aucune capacité ne résout quoi que ce soit.

**3. Un run porte toute sa clôture.** `start.json` (ADR-0016 §2) contient aussi le fichier de
verrouillage et les octets de chaque fichier que le run peut charger, dans une limite de taille
fixe (4 Mio). Un `resume` ultérieur ne lit rien d'autre. Un run démarré sur `fold-planner` 1.2.0
se termine donc sur 1.2.0, même après que le fichier de verrouillage passe à 1.3.0, et le rejeu
reste prouvé.

**4. `load` n'atteint que des fichiers verrouillés.** Le `FileLoader` de l'hôte sert deux sortes
d'étiquette :

- `load("@origami-dsl//parse.star", "plan")` charge un fichier d'une dépendance ;
- `load(":util.star", "x")` charge un fichier du même module.

Tout le reste est une erreur de chargement : un chemin, une URL, ou un fichier absent de la
clôture. Un fichier chargé s'exécute avec les mêmes globales que le script. Son code de premier
niveau peut définir des valeurs mais pas appeler une capacité. Un appel au moment du chargement
arrête le run avec `LOAD_EFFECT`, si bien que le chargement n'a aucun effet que le rejeu aurait à
réconcilier. Les grammaires que prend le `parse` de l'ADR-0017 sont les fichiers `*.ebnf` de la
clôture, nommés de la même façon.

**5. Un composite est un run parent avec des runs enfants.** Ceci amende la table fermée de
l'ADR-0016 §5 (elle dit que `run` ne revient que par un amendement) :

| Capacité | Classe | Rejeu | Coupure en pleine étape (intention, pas de résultat) |
|---|---|---|---|
| `child(agent, input, grant)` | effet, avec preuve | l'issue journalisée | l'identifiant de l'enfant est fixe, `<parent ID>.<seq>` : son `start.json` présent et égal signifie qu'il existe, donc le reprendre ; absent signifie le créer, puis le reprendre ; différent signifie `UNKNOWN` sur l'étape du parent |

- `agent` est une étiquette de la clôture, comme `@fold-planner//plan.star`. Le `start.json` de
  l'enfant prend les octets dans la clôture du parent.
- L'identifiant de l'enfant est le seul qui contienne un point, si bien qu'il ne peut pas entrer
  en collision avec un identifiant choisi par l'opérateur (ADR-0016 §2). Les runs s'imbriquent sur
  quatre niveaux au plus, et le `seq` d'un run reste sous 10^15 (un run qui l'atteint se termine en
  `failed(limit)`), si bien que chaque niveau ajoute au plus 16 caractères. Quatre niveaux sur une
  racine de 64 caractères en donnent au plus 128, la limite de segment d'`ix-recover`
  (`lib.rs:156`). Un `child` plus profond renvoie `{"ok": False, "err": {"code": "TOO_DEEP"}}`.
- **L'allocation de l'enfant doit tenir dans ce qui reste au parent.** Chaque champ cumulé
  (`calls`, `ticks`, `active_seconds`, `activations`) doit valoir au plus le reste du parent après
  ses réservations et après l'appel que l'`attempt` propre de l'étape `child` impute au parent. Un
  parent à qui il reste un appel ne peut donc pas donner d'appel à un enfant. `memory_mb` doit
  valoir au plus celui du parent, puisque c'est un plafond, pas une quantité dépensée. Ce test se
  calcule à partir de la réduction, avant que l'intention ne soit écrite. Une allocation qui ne
  tient pas donne une intention comme les autres, avec ses arguments canoniques, allocation
  comprise, et la marque `"refused": "GRANT_EXCEEDED"`. Le résultat de l'étape est
  `{"ok": False, "err": {"code": "GRANT_EXCEEDED"}}`. La réduction soustrait l'allocation de chaque
  intention `child` sauf d'une intention refusée et d'une intention résolue `--not-performed`, si
  bien qu'une intention refusée ne réserve rien. Un plantage entre cette intention et son résultat
  rejoue le refus que l'intention enregistre, sans refaire le test, puisque la réduction a avancé
  depuis. Le rejeu vérifie la même empreinte d'arguments, et écrit le même résultat. Aucun enfant n'est créé.
- **L'allocation est réservée dans l'intention.** L'enregistrement `intent` du parent pour une
  étape `child` contient l'allocation de l'enfant, et la réduction de l'ADR-0016 §8 soustrait
  l'allocation de chaque intention d'enfant de ce qui reste au parent, sauf d'une intention refusée
  et d'une intention résolue `--not-performed`. Cet enregistrement est écrit avant que l'enfant
  n'existe, si bien qu'un plantage laisse soit l'intention sans enfant, enfant que le rejeu crée
  alors sous la même réservation, soit les deux. Une intention refusée ne crée jamais
  d'enfant, au rejeu non plus.
  - **Une réservation n'est jamais rendue au parent,** sauf par une personne. Un parent qui se
    termine (`done`, ou `failed`, `failed(budget)` compris) avant de créer l'enfant garde la
    réservation dépensée, avec le reste de son allocation. L'erreur va vers la sous-dépense, jamais
    vers la surdépense.
  - **`--not-performed` sur une étape `child` libère sa réservation.** Par cette résolution, une
    personne dit que l'enfant prévu n'existe pas. Une nouvelle tentative sous un nouveau `seq`
    réserve alors une fois de plus, et aucune allocation n'est déduite deux fois.
  - #416 pose la même question pour les arbres de sondes d'ixctl ; pour les runs, cet ADR y répond
    en découpant l'allocation d'avance.
- **Un enfant présent est repris, pas attendu.** Le worker du parent exécute `ix-durable resume
  <child>` et lit ce qu'il répond. Un enfant créé juste avant la mort du parent est donc démarré
  lui aussi :
  - un enfant en cours répond `BUSY`, et le parent reste en `waiting(child <id>)` ;
  - un enfant terminé affiche son issue, que le parent journalise comme résultat de l'étape.
- L'état d'un enfant se traduit ainsi dans l'étape de son parent :
  - `done` : l'issue de l'enfant est le résultat de l'étape, sur le rail par lequel elle est
    arrivée ;
  - `failed(reason)`, quelle que soit la raison (`budget`, `limit`, `journal`) : le parent se
    termine lui aussi en `failed(child)`. Le script d'un enfant signale ses propres échecs par
    `done`, sur le rail d'échec, si bien que `failed` ne vient jamais que d'un arrêt d'urgence
    (ADR-0016 §4). Le transmettre comme une valeur laisserait un script redémarrer l'agent, et
    l'enfant a pu accomplir un effet qu'un nouvel enfant répéterait. Un `failed(journal)` qui n'est
    jamais devenu durable laisse l'enfant dans l'état de son dernier enregistrement valide, et le
    parent le reprend comme tout enfant présent (ADR-0016 §3) ;
  - `nondeterministic(seq)` : le parent s'arrête lui aussi, en `nondeterministic(<its seq>)`, où
    `<its seq>` est l'étape `child` du parent, et son statut nomme l'identifiant de l'enfant et le
    `seq` propre de l'enfant. C'est un arrêt d'urgence, pas une valeur, pour la même raison. Les
    deux runs restent arrêtés, puisque l'exécutable de l'enfant est lié dans son `start.json`
    (ADR-0016 §6), et la cause est un bogue dans IX ;
  - `waiting` ou `unknown(seq)` : le parent reste en `waiting(child <id>)`. Son statut nomme
    l'enfant et l'état de l'enfant. Un signal à l'enfant, ou la résolution par une personne de
    l'étape de l'enfant (ADR-0016 §7), laisse l'enfant continuer, et la reprise suivante du parent
    lit le résultat.
- Un enfant `different` est l'`UNKNOWN` propre du parent, résolu sur l'étape du parent
  (ADR-0016 §7).

**6. Une version est livrée comme une archive avec sa somme, et un registre est un répertoire.**

- `ix-durable module pack <dir>` écrit `<name>-<version>.tar.gz` et son `SHA256SUMS`, sans
  horodatages et dans un ordre de fichiers fixe, si bien que le même module s'empaquette en les
  mêmes octets.
- Une release porte le tag `<name>-v<version>` sur GitHub, avec l'archive, `SHA256SUMS` et une
  attestation d'artefact comme pièces jointes. C'est la convention qu'utilise `ixrelay` (#473,
  tranche S5), si bien que les deux utilisent un même workflow de release.
- Un registre est un répertoire, comme dans bzlmod : `modules/<name>/<version>/MODULE.star` plus
  `source.json`, qui contient `url` et `sha256`. Ce peut être un dépôt Git ou un chemin local.
  L'URL peut pointer partout où HTTPS arrive : une pièce jointe de release GitHub, un dépôt raw
  dans Nexus ou Artifactory, un serveur web interne. Une URL `file://` sert un registre local. IX
  n'a besoin d'aucun serveur de registre.
- `ix-durable lock` vérifie chaque archive contre son `sha256` et refuse une discordance. Avec
  `--verify-attestation`, il lance aussi `gh attestation verify`, et le fichier de verrouillage
  enregistre qu'il l'a fait.

**7. Le code extérieur entre par une personne.** Un module venu d'hors des dépôts de l'opérateur
n'entre dans un run que par un fichier de verrouillage qu'une personne a écrit. Un agent propose
des fichiers isolés par le magasin (`lua`, `pipeline`, `ebnf`), mais pas des modules : une
proposition au magasin est un seul corps et `accept` écrit un seul fichier (`store.rs:172`), alors
qu'un module est un répertoire. Une personne assemble, verrouille et publie un module. Poser le
tag d'une release reste un acte de personne.

Le fichier de verrouillage et la clôture font partie de `start.json`, si bien que la règle de
l'ADR-0016 §2 les couvre : `run` avec un identifiant existant et un autre fichier de verrouillage
donne `CONFLICT`, jamais une réutilisation silencieuse de l'ancienne clôture.

## Tranche traceuse (test d'abord, chaque test vu rouge)

1. **Deux modules locaux, un run durable.** Un module bibliothèque définit une fonction qui
   appelle `claude` et contient une grammaire ; un module agent charge les deux. Le fichier de
   verrouillage provient d'un registre en répertoire local avec des URL `file://`. La matrice
   d'arrêts forcés de l'ADR-0016 tient avec le code chargé.
2. **Un run garde sa version.** Après le démarrage d'un run, le fichier de verrouillage passe à
   une nouvelle version de la bibliothèque. Le `resume` suivant rejoue sur les anciens octets et se
   termine. Un nouveau run reçoit la nouvelle version.
3. **MVS.**
   - Le losange A → B 1.0 et C 1.1, B → D 1.0, C → D 1.1 résout D en 1.1, et jamais en un D 1.2
     publié.
   - Deux `compatibility_level` de D dans un même graphe sont une erreur.
4. **Barrières de chargement.** Chacun de ces cas arrête le run : un `load` d'un fichier hors de
   la clôture ; un chemin ; une URL ; une capacité appelée au moment du chargement
   (`LOAD_EFFECT`).
5. **Runs enfants.**
   - Un parent avec un enfant, sous la matrice d'arrêts forcés à travers la frontière entre les
     deux : exactement un enfant est créé, et le parent se termine avec l'issue de l'enfant.
   - L'allocation de l'enfant est imputée au parent une seule fois, y compris quand le parent
     meurt entre son intention et la création de l'enfant, et entre la création et la première
     reprise.
   - Un enfant créé juste avant la mort du parent est démarré par la reprise suivante du parent.
   - Un enfant arrêté en `UNKNOWN` laisse le parent en `waiting`, et résoudre l'enfant fait
     reprendre le parent. Un `start.json` d'enfant qui diffère de l'intention rend l'étape du
     parent `UNKNOWN`. `--not-performed` sur cette étape libère la réservation, et la nouvelle
     tentative réserve exactement une fois.
   - Une allocation d'enfant plus grande que le reste du parent dans un champ quelconque donne
     `GRANT_EXCEEDED`, et ne réserve rien, y compris quand le parent meurt entre l'intention
     refusée et son résultat : un enfant ultérieur qui tient est alors créé. Un cinquième niveau
     d'imbrication donne `TOO_DEEP`.
   - Un enfant qui se termine en `failed(journal)` après un `say` journalisé termine le parent en
     `failed(child)`, et aucun second enfant n'est créé. Un enfant qui s'arrête en
     `NONDETERMINISTIC` après un `say` journalisé arrête le parent en `nondeterministic` à sa
     propre étape `child`, son statut nomme l'identifiant de l'enfant et le `seq` de l'enfant, et
     aucun second enfant n'est créé. Un enfant dont le `failed(journal)` n'a pas pu être écrit est
     repris par le parent.
   - Un parent à qui il reste un appel et qui demande un enfant avec un appel reçoit
     `GRANT_EXCEEDED`.
   - Un `seq` de 10^15 − 1 donne un identifiant d'enfant d'au plus 128 caractères sur une racine de
     64 caractères, et le run qui atteint 10^15 se termine en `failed(limit)`.
6. **Intégrité.**
   - Empaqueter deux fois un module donne les mêmes octets.
   - Une archive dont les octets diffèrent de son `sha256` est refusée au moment du verrouillage.

## Rejeté

- **Un serveur de registre à nous** (à la Nexus) : bzlmod et Go montrent qu'un répertoire de
  métadonnées plus des URL d'archives suffit. Tout serveur qui sert des fichiers tient déjà
  derrière une URL.
- **Les poms Maven, npm ou Cargo comme gestionnaire de paquets :** ils versionnent le code de leur
  propre écosystème. Aucun ne connaît une clôture Starlark ni le besoin d'un run durable de garder
  ses octets pour chaque reprise.
- **Des intervalles de versions résolus à l'exécution :** un run ne se rejouerait pas si une
  version plus récente apparaissait entre deux reprises.
- **`load` depuis une URL ou un chemin :** un octet non verrouillé dans un run casse le rejeu et
  laisse entrer du code sans personne.
- **Les artefacts OCI d'abord** (ORAS vers `ghcr.io`) : un second outil et un second protocole.
  Les pièces jointes de releases GitHub sont ce avec quoi `ixrelay` est livré. OCI reste un
  déclencheur de révision.
- **Un cache global partagé de modules dans lequel lisent les runs :** un run ne lit que son propre
  `start.json`, si bien que supprimer un cache ne casse jamais un run en attente.

## Réversibilité

**Porte à double sens** tant que les modules restent dans les dépôts de l'opérateur. Elle devient
à sens unique au premier module publié pour que d'autres le verrouillent. Cela fige les deux
fonctions de `MODULE.star` sous `ix-module/0`, le fichier de verrouillage sous `ix-durable-lock/0`
et la convention de tag `<name>-v<version>`. La règle d'identifiant de la capacité enfant
(`<parent ID>.<seq>`) est figée avec le schéma du journal de l'ADR-0016.

## Déclencheurs de révision

- **Un second dépôt publie des modules :** déplacer le registre dans son propre dépôt (comme l'est
  le Bazel Central Registry).
- **Quelqu'un a besoin d'OCI** (un dépôt OCI Nexus ou Artifactory, `ghcr.io`) : ajouter un type de
  source `oci://` récupéré par ORAS, avec la même règle `sha256`.
- **Des modules de tiers :** exiger la vérification d'attestation, ou des signatures (sigstore), au
  moment du verrouillage.
- **Les agents doivent proposer des modules entiers :** donner au magasin une sorte paquet
  qu'`accept` déballe dans un répertoire par un seul renommage.
- **Une clôture dépasse 4 Mio :** stocker la clôture sous forme de fichiers à côté de
  `start.json`, publiés par un seul renommage de répertoire, et garder `start.json` comme son
  empreinte.
