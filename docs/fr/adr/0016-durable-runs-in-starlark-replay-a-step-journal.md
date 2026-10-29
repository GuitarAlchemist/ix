# Runs durables : un script Starlark, réexécuté et servi étape par étape depuis un journal à écriture anticipée

Statut : proposée (2026-10-29), révision 9 après les tours Codex 1 à 8 et le prototype Starlark

Version anglaise : [`docs/adr/0016-durable-runs-in-starlark-replay-a-step-journal.md`](../../adr/0016-durable-runs-in-starlark-replay-a-step-journal.md).

Numéro 0016 : l'ADR-0015 est pris par `docs/adr/0015-local-isolated-jobs.md` sur la branche non
fusionnée `feat/local-isolated-execution`. Liés : l'ADR-0017 (DSL d'agents) et l'ADR-0018
(composition et empaquetage) s'appuient sur celui-ci.

## Contexte

L'opérateur a demandé de vrais agents IX durables (2026-10-29). Un agent durable doit faire
quatre choses :

- survivre à un arrêt forcé et reprendre à la dernière étape terminée, sans redemander au modèle ;
- ne jamais répéter un effet dont il ne peut pas prouver qu'il n'a pas eu lieu ;
- attendre des heures un événement ou une approbation sans aucun processus vivant ;
- garder une seule identité et un seul budget à travers tout cela.

C'est le modèle d'exécution durable de Temporal, Restate et DBOS. AutoKitteh exécutait sur
Temporal des workflows durables écrits en Starlark ; le projet est archivé (2026), il compte donc
ici comme un précédent pour la conception, pas comme une preuve maintenue. IX a les pièces, mais
rien ne les compose :

- **`ix-loop` (ADR-0011) :** des boucles Lua dans un bac à sable, mais rien ne survit au
  processus. Son `Ledger` est tenu sans garantie, écrit après l'appel et jamais synchronisé.
- **`ix-recover` (#389) :** un verrou de propriétaire et une réconciliation, mais au niveau du
  run, et un effet extérieur n'est « ni défait ni réconcilié ».
- **`ixctl` run/1 (ADR-0014) :** de la supervision, mais aucune notion d'étape.
- **`ixrelay` (#473, pas encore construit) :** une file durable vers les sessions Claude locales,
  dont les états de requête (`QUEUED`, `ACCEPTED`, `REPLIED`, `UNKNOWN`) et le refus de renvoyer à
  l'aveugle sont les mêmes règles que celles de ce journal.

La réexécution exige un script qui calcule deux fois la même chose. On ne peut pas prendre
d'instantané de la VM, donc un run repris doit réexécuter son script et obtenir les mêmes appels.
Un premier brouillon gardait le Lua d'`ix-loop`. Le tour Codex 1 a inventorié ce qu'une boucle Lua
peut observer et qui change d'un processus à l'autre :

- la graine du hachage des chaînes, qui mélange `time(NULL)` et des adresses
  (`lua-5.4.7/lstate.c:71`), si bien que l'ordre de `pairs` change ;
- `math.random`, initialisé à partir de l'heure et d'une adresse (`lmathlib.c:623`) ;
- `tostring` d'une table, d'une fonction ou d'un thread, qui affiche une adresse ;
- les tables faibles, qui dépendent du ramasse-miettes ;
- la conversion de Lua vers JSON, qui itère sur les paires brutes (`sandbox.rs:86`) ;
- `budget()` et les résultats des enfants, qui exposent des compteurs courants et des secondes
  (`sandbox.rs:365`, `:425`).

Fixer la graine du hachage exige un fork de `lua-src`, qui n'offre à `mlua` aucun moyen de passer
une définition. Les adresses et le ramasse-miettes ne peuvent pas être fixés du tout. Une
vérification à chaque étape ne détecterait pas une branche prise après le dernier appel, si bien
qu'un mode durable en Lua reposerait sur une garde que Codex a montrée non fiable.

Un prototype jetable a ensuite fait tourner un hôte durable sur `starlark-rust` 0.14.2. Il est
gardé sur la branche `proto/starlark-durable`, avec ses résultats dans son `RESULTS.md` :

- trois processus ont donné le même ordre de dict, le même `hash`, le même `json.encode` et le
  même `repr` ;
- il n'existe ni horloge ni générateur de nombres aléatoires ;
- `while` ne passe pas l'analyse, et la récursion est permise ;
- une limite de ticks déterministe a arrêté une boucle de dix millions d'itérations ;
- des arrêts forcés à chacun des 15 passages entre étapes ont repris jusqu'à la même issue, avec
  une seule proposition et aucun envoi répété ;
- un envoi coupé s'est arrêté en `UNKNOWN`, et une personne l'a réglé ;
- un script modifié a été refusé ;
- Sonnet a écrit du premier coup un agent valide à partir du contrat des capacités, et cet agent a
  survécu à un arrêt forcé.

Le tour Codex 2 a lu le source de l'interpréteur et n'a trouvé aucune adresse ni aucune
disposition du tas observable dans l'environnement Standard plus JSON :

- le `repr` d'une fonction affiche son nom ou sa signature ;
- il n'y a pas d'`id()` ;
- `hash()` ne prend que des chaînes et c'est le hachage de chaînes de Java ;
- les dicts gardent l'ordre d'insertion, et `dir()` trie.

Starlark est le langage que Bazel et Buck2 ont choisi pour une évaluation hermétique et
reproductible. Il a deux interpréteurs maintenus, en Rust (Meta) et en Go (Google). Il n'a pas
d'exceptions, ce qui convient aux résultats sur deux rails ci-dessous.

## Décision

**1. Les runs durables sont des scripts Starlark, dans un nouveau crate expérimental
`crates/ix-durable`.** Il dépend de `starlark` (Rust pur), et d'`ix-loop` pour le nœud sémantique
et le magasin. Les boucles Lua d'`ix-loop` ne sont pas touchées. Leur passage éventuel à Starlark
est une décision ultérieure, sur les preuves de ce crate. Le dialecte est Standard plus les
instructions de premier niveau, et `load` est désactivé (l'ADR-0018 l'active pour les modules
verrouillés). Les limites de ticks et de tas de l'interpréteur sont toujours posées, mais
`starlark-rust` les documente toutes deux comme sans garantie : elles sont le budget déterministe
et bon marché. Les bornes dures viennent du système d'exploitation (décision 10).

**2. Un run est créé par une seule publication atomique, et repris par son seul identifiant.** `ix-durable run
--run-id ID <script> [--input file.json] [--grant grant.json]` écrit un seul fichier, `start.json`.
Il contient :

- le texte source et l'entrée ;
- leurs empreintes ;
- l'empreinte de l'exécutable `ix-durable` et la version de `starlark` ;
- l'allocation (décision 9).

Il est écrit dans un fichier temporaire de `runs/<ID>/`, synchronisé, puis lié en dur vers
`runs/<ID>/start.json`. Un lien est atomique, si bien qu'un run existe tout entier ou n'existe
pas. `start.json` est du JSON canonique sans horodatage, si bien que le même run a toujours les
mêmes octets.

- Un identifiant de run choisi par l'opérateur fait de 1 à 64 caractères parmi `A-Z a-z 0-9 _ -`.
  Il ne peut pas être un nom de périphérique Windows, que la vérification de segment
  d'`ix-recover` refuse déjà (`lib.rs:156`). Le point est réservé aux runs enfants (ADR-0018),
  dont l'hôte construit les identifiants.
- Un `runs/<ID>/` sans `start.json` n'a jamais été créé. `run` le complète ; `resume` répond
  `NOT_CREATED`. Les fichiers temporaires restants sont ignorés et supprimés par le `run` suivant.
- `run` avec un identifiant existant donne `CONFLICT` quand le nouveau `start.json` diffère de
  celui qui est publié, ne serait-ce que d'un octet. Des octets identiques ne sont pas une erreur :
  le run est déjà créé.
- `ix-durable resume ID` relit `start.json`. Un autre exécutable ou un autre interpréteur est
  refusé (`CONFLICT`), puisque le rejeu n'est prouvé que sur le code qui a écrit le journal.
- Un seul lanceur possède un run, par `File::try_lock` sur `owner.lock` ; un second reçoit
  `BUSY`. La décision 10 ajoute le `writer.lock` du worker.
- Le crate fixe `rust-version = "1.89"` pour `try_lock`, comme le fait `ix-recover`.

**3. Le journal est à écriture anticipée, et il arrête le run quand il ne peut pas être écrit.**
Il se trouve dans `runs/<ID>/journal.jsonl`, un enregistrement JSON terminé par LF par ligne :

- `intent` pour chaque étape : `seq`, `cap`, les arguments canoniques et leur empreinte ;
- `attempt`, avant chaque exécution en direct d'une étape ;
- `result`, qui contient la valeur entière, sur l'un ou l'autre rail, et le nombre cumulé de
  ticks à ce point ;
- `activation`, qui ouvre le travail de chaque processus ;
- `lease`, qui impute, avant que le worker ne les dépense, la seconde de temps actif suivante et,
  en travail en direct, les 50 000 ticks suivants. Il contient les deux montants, `ticks` et
  `seconds`, et un bail de rejeu a `ticks: 0` (décision 9). Un mutex sérialise tous les rédacteurs
  du journal ;
- `resolution`, écrit par une personne ;
- `done` ou `failed(reason)`.

Chaque enregistrement est vidé puis synchronisé par `sync_data` avant que sa valeur n'atteigne le
script, ou avant que son effet ne commence. Un run a un seul enregistrement terminal. Une
écriture ou une synchronisation qui échoue termine le run en `failed(journal)` (hors `resolution`,
décision 7), à une exception
près : un enregistrement terminal (`done` ou tout `failed(reason)`) dont les octets ont atteint le
fichier avec leur LF est l'état, même si sa synchronisation a échoué. Le worker s'arrête, et le
lanceur, après la mort du worker (aussitôt quand l'écriture qui a échoué est la sienne), ouvre le
journal comme le fait `resume`. Si le dernier enregistrement valide est terminal, il n'ajoute
rien ; sinon il ajoute `failed(journal)`. Si cette écriture échoue avant son LF, rien n'est
enregistré, et le run reste dans l'état de
son dernier enregistrement valide. Le `resume` suivant reprend à partir de là, et ne répète rien,
puisque chaque enregistrement antérieur a été écrit d'avance : une étape coupée après son effet
est `UNKNOWN` (décision 7). À l'ouverture, chaque enregistrement terminé par LF doit s'analyser et
respecter l'ordre des `seq` et des phases : pas de trou, pas de doublon, pas de `result` sans son
`intent`, et aucun enregistrement après un enregistrement terminal. Seul un fragment final sans LF peut être abandonné, et il est tronqué et synchronisé
avant tout ajout. La durabilité n'est revendiquée que pour les arrêts forcés de processus. Rien
n'est revendiqué pour une coupure de courant, comme dans `ix-recover`.

**4. Les capacités répondent sur deux rails ; les arrêts d'urgence ne sont pas des valeurs.**
Chaque capacité renvoie
`{"ok": True, "value": …}` ou `{"ok": False, "err": {"code": …, "message": …}}`. Les scripts
branchent sur `err.code`, un identifiant stable. Ils peuvent aussi lire `err.message`. C'est sans
danger pour le rejeu, puisque le résultat entier est journalisé, mais le message n'est pas un
contrat. Un échec est journalisé et rejoué comme un succès. Les arrêts d'urgence sont des erreurs
Rust, et Starlark n'a pas de `try`, si bien qu'aucun script ne peut en attraper un :

- `UNKNOWN(seq)`, `NONDETERMINISTIC(seq)`, `CONFLICT` ;
- le budget épuisé, la limite de ticks ou de tas, une borne du système d'exploitation,
  `failed(journal)`.

**5. La table des capacités est fermée ; ce qu'elle ne liste pas n'existe pas dans un run
durable.** Première tranche :

| Capacité | Classe | Rejeu | Coupure en pleine étape (intention, pas de résultat) |
|---|---|---|---|
| `claude(prompt, opts)` | enregistrée | la valeur journalisée | la relancer, après un nouvel `attempt` |
| `propose(kind, name, body)` | effet, avec preuve | la valeur journalisée | l'hôte relit la proposition `d<run>-<seq>` et compare son empreinte de contenu à l'intention : présente et égale signifie faite ; absente signifie l'exécuter ; différente signifie `UNKNOWN` |
| `say(text)` | effet, sans preuve | la valeur journalisée | `UNKNOWN` : le run s'arrête jusqu'à ce qu'une personne le résolve |
| `wait(name)` | attente | le signal journalisé | chercher à nouveau le signal ; aucun signifie `waiting(name)` et une sortie avec le code 0 |
| `print(…)` | sortie de l'hôte | supprimée | sans objet |

`budget()`, les mesures de temps, `run` (scripts enfants), `tasks.*`, `gh.*`, `wasm.*` et
`loops.list/load` sont absents des runs durables. Chacun ne revient que par un amendement qui le
classe et dit comment le rejeu reconstruit l'état qu'il touche.

`propose` passe par le `Store` existant, si bien que l'opérateur accepte une proposition durable
avec le même `accept` que celle d'une boucle. Deux changements au magasin d'`ix-loop` le rendent
possible :

- **Une publication sous un identifiant fourni par l'appelant.** Un nouveau
  `Store::publish_as(id, draft, by)` écrit de façon atomique (fichier temporaire, synchronisation,
  lien en dur) et refuse un identifiant existant au contenu différent. Le « contenu » désigne les
  champs stables `kind`, `name`, `body`, `note` et `proposed_by`, jamais `ts_ms`, si bien qu'une
  publication rejouée avec un nouvel horodatage correspond encore. Aujourd'hui,
  `Store::propose` choisit son propre identifiant et écrit directement le fichier final
  (`store.rs:97`, `:120`).
- **Une seconde forme d'identifiant.** `valid_id` (`store.rs:51`) admet aussi
  `d<12 hex>-<digits>`. Les chiffres hexadécimaux sont les 12 premiers du SHA-256 de l'identifiant
  du run, si bien que tout identifiant de run donne un nom de fichier sûr. `kind` garde les
  valeurs du magasin (`lua`, `pipeline`).

**6. Le rejeu se vérifie lui-même sur trois choses.** À chaque étape journalisée, le rejeu doit
demander :

- le même `cap` ;
- la même empreinte d'arguments ;
- le même nombre cumulé de ticks que l'enregistrement `result`.

Le nombre de ticks est déterministe pour le même exécutable, le même source, la même entrée et
les mêmes réponses enregistrées. Le tour Codex 2 l'a confirmé dans le source : les ticks comptent
les appels et les arcs retour des boucles, et la période de vérification de 1 000 ticks ne fait
que déplacer le compteur. C'est un compteur de flot de contrôle, pas une preuve d'état :

- Deux branches avec le même nombre d'appels et d'arcs retour le passent.
- Une divergence après le dernier résultat journalisé passe inaperçue si le processus meurt avant
  `done`, puisqu'aucune étape ultérieure ne la compare.

Les deux résidus sont acceptés. Avec les globales fermées ci-dessus, on n'a trouvé rien de ce qui
varie d'un processus à l'autre qui alimente une telle branche. Une discordance provoque l'arrêt
`NONDETERMINISTIC(seq)` et nomme les deux côtés. Comme le source et l'exécutable sont liés
(décision 2), une discordance signifie un bogue dans IX, pas un script modifié.

**7. Une personne résout `UNKNOWN` ; le script décide de la suite.** `ix-durable resolve ID seq
--performed <value.json> | --not-performed` ajoute un enregistrement `resolution`. Il tient les
deux verrous (décision 10), ouvre le journal, et n'ajoute l'enregistrement que tant que cette
étape est encore `UNKNOWN` : un second `resolve` de la même étape est refusé, et n'écrit rien. Une `resolution` dont l'écriture
échoue ne termine pas le run : `resolve` signale l'erreur et n'ajoute rien de plus, et le
`resolve` suivant trouve l'étape encore `UNKNOWN`, ou résolue si l'enregistrement a atteint le
fichier avec son LF. Aucun script
ne peut atteindre cette commande (comme pour `accept`, ADR-0011 §6). Après `--performed`, l'étape
renvoie cette valeur et l'effet n'est jamais répété. Après `--not-performed`, l'étape renvoie
`{"ok": False, "err": {"code": "NOT_PERFORMED"}}`, et le script peut appeler à nouveau la
capacité sous un nouveau `seq`.

**8. L'état de l'hôte est une réduction du journal, jamais de la mémoire.** À chaque activation,
l'hôte reconstruit à partir du journal avant d'exécuter une ligne :

- les appels dépensés, c'est-à-dire le nombre d'enregistrements `attempt` ;
- les ticks et le temps actif dépensés (décision 9) ;
- les propositions et leurs octets ;
- les énoncés de `say` ;
- le coût théorique, sommé à partir des seuls enregistrements `result`. Une tentative interrompue
  n'a pas de coût connu.

La suite en direct voit alors le run qu'elle aurait vu sans l'arrêt forcé.

**9. Un seul budget, fixé à la création et cumulé sur tout le run.** L'allocation vit dans
`start.json`. Un `resume` ultérieur ne peut pas la changer, quoi que dise l'environnement.
L'augmenter exige un nouveau run. Une allocation a cinq champs finis, chacun avec une valeur par
défaut :

| Champ | Compte | Par défaut |
|---|---|---|
| `calls` | les enregistrements `attempt` | 50 |
| `ticks` | les ticks Starlark, travail en direct seulement | 10 000 000 |
| `active_seconds` | le temps pendant lequel un worker tourne | 600 |
| `activations` | les enregistrements `activation` | 20 |
| `memory_mb` | le job du worker, à tout instant : un plafond dur (décision 10) | 1024 |

Le `Grant` d'`ix-loop` (`calls`, `fuel`, `seconds`) n'est pas réutilisé, puisqu'il n'a ni
activations ni mémoire.

- **Appels :** un par `attempt`, si bien qu'un appel tué à répétition est imputé à chaque fois.
- **Activations :** le lanceur écrit et synchronise l'enregistrement `activation` avant de créer le
  worker, pendant qu'il tient les deux verrous, et cet enregistrement est l'imputation. Il impute
  aussi une fenêtre d'amorçage de 5 secondes de temps actif. Un run sans activation restante se
  termine en `failed(budget)` sans démarrer de worker, si bien que le nombre d'activations est
  fini, quoi qui les tue.
- **Les ticks et le temps actif se paient d'avance.** Avant que le worker n'exécute une ligne, et à
  nouveau avant l'expiration de chaque bail, il écrit et synchronise un enregistrement `lease`
  (décision 3). L'imputation est l'enregistrement. Deux gardes arrêtent le travail au-delà du
  bail :
  - **Ticks :** la limite propre de l'interpréteur ne peut être posée qu'une fois
    (`set_max_tick_count` refuse un second appel, `evaluator.rs:1058`) ; elle est donc posée une
    seule fois, au nombre de ticks de la frontière plus les ticks en direct restants plus 1 000.
    Les baux sont tenus par le crochet que Starlark appelle dans sa vérification tous les
    1 000 ticks (`set_check_cancelled`, appelé à `evaluator.rs:1115`, et aussi après le retour
    d'une évaluation, `eval.rs:159` et `:191`). Le crochet ne voit aucun nombre de ticks ; il
    compte donc chacun de ses appels pour 1 000 ticks ; un appel de plus, ou le premier appel après un bail
    commencé entre deux vérifications, le fait compter en avance, jamais en retard. Quand le
    compte atteint la fin du bail, le crochet écrit et synchronise lui-même le bail suivant et
    envoie son échéance, puis laisse l'évaluation continuer. Si l'écriture échoue, il renvoie
    `true`, ce qui arrête l'évaluation, et le run se termine en `failed(journal)`, comme pour toute
    écriture qui échoue. Aucun tick ne
    s'exécute à plus de 1 000 ticks au-delà d'un bail synchronisé.
  - **Temps :** le worker envoie au lanceur l'échéance de chaque bail synchronisé par un tube. Le
    chien de garde du lanceur arme l'échéance d'amorçage, 5 secondes plus tard, quand il laisse le
    worker s'exécuter, et termine le job quand une échéance passe sans bail plus récent. Un worker qui se
    bloque avant son premier bail, dans `sync_data` ou sur le tube, est donc tué à l'échéance
    d'amorçage, comme un thread affamé ou une longue opération native plus tard. Un lanceur tué
    ferme le job, ce qui tue le worker.

  Un arrêt forcé gaspille au plus ce qui est déjà payé. La dépense ne dépasse jamais les
  imputations, plus 1 000 ticks par bail et le temps de réaction du chien de garde, quel que
  soit le nombre d'arrêts forcés du run.
- **Le rejeu se paie en temps, pas en ticks.** `ticks` ne compte que le travail en direct. Le rejeu
  jusqu'à la frontière réexécute exactement le nombre de ticks de la frontière, que contient le
  dernier enregistrement `result`, et la décision 6 le vérifie à chaque étape. Ses baux ont `ticks: 0` et
  imputent la seconde suivante. La frontière est le retour de l'appel de capacité dont le
  `result` est le dernier journalisé. Cet appel est du code de l'hôte qui a l'évaluateur en main ;
  il lit donc le nombre exact de ticks, et avant de revenir il écrit et synchronise le premier bail
  en direct, qui impute des ticks. Chaque bail à partir de là impute aussi des ticks. Le travail
  qu'une activation tuée a fait au-delà de son dernier `result` est donc imputé à nouveau dans la
  suivante : l'erreur va vers la surimputation, jamais vers la sous-imputation. Un run qui n'a
  encore aucun `result` est en direct dès son premier tick.
- Le temps passé à attendre n'est pas compté.

**10. Les bornes dures et la propriété des processus viennent du système d'exploitation, sous
Windows d'abord.** `ix-durable run` et `resume` sont un lanceur, et le lanceur démarre un
processus worker. Le worker évalue le script et écrit le journal. Deux verrous garantissent un
seul rédacteur à la fois :

- Le lanceur tient `owner.lock` pendant toute l'activation, mort du worker comprise.
- Le worker tient `writer.lock` tant qu'il vit.
- Un nouveau lanceur prend `owner.lock` (sinon `BUSY`), puis attend `writer.lock`. Un worker qui
  survit à son lanceur empêche donc tout nouveau worker d'écrire jusqu'à sa mort. Tenant les deux,
  le lanceur ajoute l'enregistrement `activation` (décision 9), puis libère `writer.lock` et
  démarre le worker, qui le prend. Aucun autre lanceur ne peut s'intercaler, puisque `owner.lock`
  reste tenu.
- Quand une borne tue le worker, le lanceur, qui tient encore `owner.lock`, attend `writer.lock`,
  ouvre le journal comme dans la décision 3, et ajoute `failed(limit)` sauf si le dernier
  enregistrement valide est déjà terminal : un worker tué après son `done` reste `done`. Il ne
  libère `owner.lock` qu'ensuite. Aucun `resume` ne peut démarrer entre les deux.
- `resolve` prend les deux verrous de la même façon (sinon `BUSY`), si bien qu'il n'écrit jamais à
  côté d'un worker, d'un lanceur ou d'un autre `resolve` (décision 7).

Les bornes valent pour tout l'arbre de processus, pas pour un seul processus. Le lanceur fait ce
que fait déjà `ixctl` (`go/ixctl/proc_windows.go:104`) :

- il démarre le worker suspendu (`CREATE_SUSPENDED`) ;
- il place le worker dans un Job Object neuf dont ses enfants ne peuvent pas s'échapper, avec
  l'arrêt forcé à la fermeture ;
- alors seulement il laisse le worker s'exécuter, si bien qu'aucun descendant ne naît jamais hors
  du job.

Un sous-processus `claude` (`node.rs:133`) et tout autre descendant vivent donc dans le job, et un
worker tué ne laisse aucun appel en cours à côté de la tentative suivante. Les limites du job
valent pour le job entier :

- **Mémoire.** Le plafond dur de mémoire validée du job est `memory_mb`, si bien que le job n'en
  détient jamais plus. Une limite de notification aux sept huitièmes de ce plafond envoie un
  message à un port de complétion ; le système garantit ce message. C'est la paire de limites
  d'`ixctl` (`proc_windows.go:108` et `:252`), avec l'allocation comme plafond dur plutôt que comme
  notification. Le lanceur écoute sur le port. Quand la notification arrive, il termine le job et
  ajoute `failed(limit)`, même si le processus qui a atteint la limite a traité l'échec de son
  allocation et a continué.
- **Temps CPU.** Une limite de temps CPU à l'échelle du job appuie le chien de garde des baux de la
  décision 9.

Une seule opération native coûteuse, comme `sorted(range(100_000_000))`, atteint la limite de
mémoire ou survit à son bail, et termine le run en `failed(limit)`. Elle ne peut pas se glisser
au-delà de la vérification des ticks.

Sous Linux, un groupe de processus, un signal de mort du parent et `setrlimit` bornent chaque
processus mais pas l'arbre : un descendant peut quitter le groupe, et le signal de mort du parent
n'atteint que le meneur (`go/ixctl/proc_other.go:35`). Un worker Linux a donc besoin d'un cgroup
v2 délégué (`memory.max`, `cpu.max`, `pids.max`, `cgroup.kill`). Sans lui, `ix-durable` refuse d'y
démarrer un run, plutôt que de revendiquer une borne qu'il ne tient pas. La prise en charge de
Linux est une tranche à part.

**11. Les états se lisent sans le propriétaire.** `ix-durable status ID` rapporte l'un de
ceux-ci :

- `not_created` ;
- `running`, quand le verrou de propriétaire est tenu ;
- `waiting(name)`, `unknown(seq)`, `nondeterministic(seq)` ;
- `done`, où l'enregistrement `done` contient l'issue ; un `resume` ultérieur l'affiche et
  n'exécute rien ;
- `failed(reason)`.

**12. Un signal est publié de façon atomique.** `ix-durable signal ID name value.json` utilise un
fichier temporaire, une synchronisation et un lien en dur vers `inbox/<name>`. Un second signal
différent donne `CONFLICT`, et les mêmes octets à nouveau ne sont pas une erreur. Pas de délai
d'expiration dans la première tranche. Qu'`ixctl` reprenne un run à l'arrivée d'un signal, ou
après un plantage, relève d'une tranche ultérieure ; le moteur reste une ligne de commande.

## Tranche traceuse (test d'abord, chaque test vu rouge)

Un seul script : `claude`, `propose`, `say`, puis `wait("approve")`. C'est fait quand chaque
vérification ci-dessous tient :

1. **Matrice d'arrêts forcés.** Un vrai arrêt forcé du processus avant et après chaque écriture
   d'`intent`, d'`attempt`, d'effet, de `result` et de `done`, et chaque mort reprise :
   - le modèle est interrogé une fois par enregistrement `attempt`, et jamais à nouveau pour un
     résultat journalisé ;
   - exactement une proposition, que l'`accept` existant déplace dans la bibliothèque ;
   - `say` a lieu au plus une fois ;
   - chaque run se termine avec la même issue.
2. **Création.** Un arrêt forcé avant et après le lien de `start.json` : `resume` répond
   `NOT_CREATED`, puis `run` complète le run. Un second `run` avec une autre allocation donne
   `CONFLICT`. Les identifiants de run avec `..`, un séparateur, un point ou un nom de périphérique
   sont refusés.
3. **Arrêts forcés répétés d'un même appel Claude.** Chaque arrêt forcé ajoute un `attempt` et une
   imputation ; l'allocation s'épuise et le run s'arrête ; aucun processus descendant ne survit au
   propriétaire.
4. **Arrêts forcés répétés pendant le calcul local.** Des arrêts forcés entre deux étapes, certains
   avant le premier bail. Les imputations d'activations et de baux croissent à chaque reprise, le
   run se termine en `failed(budget)`, et les ticks dépensés ne dépassent jamais les imputations de
   plus que la tolérance de la décision 9.
5. **Une opération native au-delà des bornes.**
   - `sorted(range(100_000_000))` se termine en `failed(limit)`, et la machine garde sa mémoire.
   - Un descendant du worker qui alloue au-delà de `memory_mb` et traite le refus termine quand
     même le run en `failed(limit)`.
   - Un worker dont la synchronisation de bail est bloquée est terminé à l'échéance du bail, et un
     worker bloqué avant son premier bail, à l'échéance d'amorçage.
   - Une boucle de 200 000 ticks écrit quatre baux par le crochet de vérification et se
     termine ; la même boucle, avec des écritures de bail rendues défaillantes, se termine en
     `failed(journal)` à moins de 1 000 ticks du bail.
   - Une frontière à un nombre de ticks qui n'est pas un multiple de 1 000, tuée avant la
     vérification suivante : le premier bail en direct a été synchronisé à la frontière, et les
     ticks qui le suivent sont imputés.
   - Un arrêt forcé autour de chaque étape de la passation : après l'enregistrement `activation`
     du lanceur, après qu'il libère `writer.lock`, et avant le premier bail du worker.
   - Le job ne valide jamais plus de `memory_mb`.
   - Un petit-enfant démarré par le worker dès sa première instruction est dans le job.
   - Un `resume` lancé pendant que le lanceur ajoute `failed(limit)` donne `BUSY`.
6. **Les deux résolutions.** Après un `say` en `UNKNOWN`, `--performed` ne reparle jamais, et
   `--not-performed` parle exactement une fois lors de la nouvelle tentative du script.
7. **Preuve de proposition.** Un arrêt forcé avant, pendant et après la publication, et une
   proposition dont le contenu diffère de l'intention, qui doit donner `UNKNOWN`.
8. **Divergence et déterminisme.**
   - Un hôte construit avec une capacité rendue délibérément non déterministe s'arrête en
     `NONDETERMINISTIC` à l'étape suivante.
   - Un source modifié, un autre exécutable et une entrée différente sont refusés.
   - Deux processus indépendants donnent le même vecteur de ticks par étape sur un script qui
     dépasse 1 000 ticks.
9. **Dommages au journal.**
   - Un fragment final déchiré est tronqué, et le run reprend.
   - Un enregistrement malformé qui se termine par un LF, intérieur ou dernier, est `Corrupt`, de
     même qu'un trou de séquence, un doublon et un `result` sans son `intent`.
   - Un échec d'écriture ou de synchronisation injecté sur un enregistrement qui n'est pas terminal
     donne `failed(journal)`.
   - Un échec de synchronisation sur un enregistrement terminal complet (`done`, le
     `failed(reason)` d'un worker, le `failed(limit)` ou le `failed(journal)` d'un lanceur) laisse
     cet enregistrement l'état, et aucun second enregistrement terminal ne le suit.
   - Un `failed(journal)` de lanceur coupé avant son LF laisse le run dans l'état de son dernier
     enregistrement valide, et `resume` reprend à partir de là.
   - Un enregistrement valide après un enregistrement terminal est `Corrupt`.
   - Une borne qui tue le worker après son `done` synchronisé laisse `done`, sans `failed(limit)`.
   - Une `resolution` coupée avant son LF laisse l'étape `UNKNOWN` et ouverte à un autre `resolve`.
   - Deux `resolve` concurrents d'une même étape : l'un ajoute, l'autre reçoit `BUSY` ou est
     refusé, et le journal s'ouvre proprement.
10. **Budget à travers les attentes.** Un script qui dépense l'essentiel de ses ticks avant
    l'attente finit quand même après elle : le rejeu jusqu'à la frontière impute du temps et aucun
    tick, et les ticks en direct au-delà ne dépassent jamais ce qui reste de l'allocation, plus
    1 000.
11. **État reconstruit.** Après chaque reprise, chaque champ de la décision 8 est égal à ce que
    contient un run non tué, octets des propositions et coût théorique compris.
12. **Propriété et signaux.** Un second processus sur un run en cours donne `BUSY` ; un second
    signal différent donne `CONFLICT`.
13. **Preuve du suffixe.** Après chaque reprise, l'hôte vérifie par assertion qu'il a rejoué
    exactement les étapes du journal avant de passer en direct, plutôt que de revenir trop tôt.

## Tranches suivantes (décidées ailleurs)

- **DSL d'agents** (ADR-0017) : une capacité `parse(dsl, text)`, enregistrée, sur une grammaire
  verrouillée par empreinte.
- **Composition et empaquetage** (ADR-0018) : `load` de modules verrouillés, runs enfants,
  modules versionnés publiés sur GitHub.
- **Un envoi avec preuve** (#473) : une fois qu'`ixrelay` existe, `send(target, text)` remplace
  `say` pour les messages vers une session. L'identifiant de requête est `d<run>-<seq>`, et
  ixrelay refuse le même identifiant avec un contenu différent. `ixrelay request <id>` donne la
  preuve qu'utilise la règle des propositions :
  - `ACCEPTED` ou `REPLIED` avec le même contenu signifie fait ;
  - absent signifie la soumettre ;
  - le `UNKNOWN` propre à ixrelay signifie qu'une personne la résout, comme ici.

## Rejeté

- **Lua pour les runs durables :** les sources de dérive listées plus haut, dont deux (les
  adresses, le ramasse-miettes) ne peuvent pas être fixées, et une graine de hachage qu'on ne peut
  fixer qu'en forkant `lua-src`.
- **Prendre un instantané d'une VM :** la persistance à la Pluto ou à la Eris est en C, non
  maintenue pour 5.4, et casse sur les userdata de l'hôte. Starlark n'offre pas d'instantané non
  plus.
- **Temporal, Restate ou DBOS comme moteur :** un serveur ou une base de données hors d'IX pour un
  seul opérateur local. AutoKitteh, archivé, a montré que Starlark sur Temporal peut fonctionner ;
  IX garde plutôt le journal en local.
- **`starlark-go` :** les capacités de l'agent (le nœud sémantique, le magasin, puis les oracles,
  `ix-wasm` et `ix-origami`) sont en Rust. Go garde le plan de contrôle (`ixctl`) et la livraison
  (`ixrelay`). Les scripts restent portables entre les deux interpréteurs s'ils s'en tiennent au
  dialecte commun. Une version wasm de `starlark-go` existe (`starlark-webasm` sur npm), si bien
  que l'éditeur de ga pourrait vérifier un script dans le navigateur sans l'exécuter de façon
  durable.
- **Faire confiance aux limites de l'interpréteur comme bornes dures :** sa propre documentation
  dit qu'elles sont sans garantie et manquent les allocations natives.
- **Des limites par processus comme borne dure** (`setrlimit`, une limite de mémoire de job par
  processus) : un arbre de processus, chacun sous sa limite, peut dépasser ensemble celle du run.
- **Trois fichiers d'identité publiés un par un :** un arrêt forcé entre deux liens laisse un run
  que ni `run` ni `resume` ne peuvent régler. Un fichier, un lien.
- **`ix-recover` comme magasin des issues :** il opère au niveau du run et a son propre verrou de
  propriétaire ; deux autorités sur un même run ouvriraient deux fenêtres de plantage.
  L'enregistrement `done` est la seule autorité (Codex, question ouverte 1).
- **Transformer `Ledger` en journal :** il est écrit après l'appel, non synchronisé, et ne doit
  jamais arrêter une boucle. Les deux restent.
- **Refaire un effet interrompu :** c'est le double envoi que ce modèle existe pour empêcher.
- **Un modèle déterministe** (température 0, une graine) : Claude Code sans interface ne promet
  ni l'un ni l'autre. Le journal enregistre plutôt les réponses.

## Réversibilité

**Porte à double sens** tant qu'`ix-durable` est expérimental et que ses journaux sont des
fichiers locaux que rien d'autre ne lit. Elle devient à sens unique quand un autre dépôt lit le
journal ou les états (Gaia, l'éditeur de ga), ou quand un run doit survivre à une version. L'un ou
l'autre fige le schéma des enregistrements sous `ix-durable-journal/0`. La décision 2 refuse déjà
de rejouer sous un autre exécutable ; lever ce refus est le même pas à sens unique. La seconde
forme d'identifiant du magasin (décision 5) est à double sens : elle ne fait qu'élargir ce
qu'`accept` prend.

## Déclencheurs de révision

- **Une divergence que la vérification en trois parties de la décision 6 manque :** journaliser
  alors une empreinte de l'état complet du module Starlark à chaque étape.
- **Les scripts ont besoin de `run`, `tasks.judge`, `gh` ou `wasm` :** les ajouter un à la fois,
  chacun avec sa règle de rejeu et l'état qu'il reconstruit.
- **Les boucles Lua sont nécessaires dans les runs durables :** décider alors s'il faut les migrer
  vers Starlark, avec les chiffres de ce crate comme preuves.
- **Les runs attendent plus longtemps que l'opérateur ne tolère un état périmé :** ajouter une
  échéance de run après laquelle `resume` le termine en `failed(expired)`.
- **Un run doit passer sur une autre machine :** le runtime mathématique du système
  d'exploitation n'est pas lié par l'empreinte de l'exécutable, si bien que le formatage des
  flottants n'est prouvé que sur une seule plateforme.
