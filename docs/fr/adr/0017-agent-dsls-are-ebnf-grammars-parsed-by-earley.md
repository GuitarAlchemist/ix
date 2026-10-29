# DSL d'agents : une grammaire `.ebnf`, analysée par un analyseur d'Earley général, parcourue par Starlark

Statut : proposée (2026-10-29)

Version anglaise : [`docs/adr/0017-agent-dsls-are-ebnf-grammars-parsed-by-earley.md`](../../adr/0017-agent-dsls-are-ebnf-grammars-parsed-by-earley.md).

S'appuie sur l'ADR-0016 (runs durables en Starlark). L'ADR-0018 dit comment un DSL est versionné
et livré.

## Contexte

L'opérateur veut que les agents parlent de petits langages faits pour leur tâche, au lieu de JSON
ou de texte libre (2026-10-29). Chaque agent aurait son propre DSL : un plan en trois lignes, une
séquence de plis, une fiche de nœuds. Un tel DSL est plus court que JSON, et une grammaire refuse
chaque écart à une position donnée. Un modèle l'écrit bien à partir de la grammaire et de deux
exemples. Il convient aussi aux runs durables : le texte va dans le journal, l'analyse est
déterministe, et le rejeu reconstruit le même arbre.

IX en a déjà une partie :

- **`ix-grammar`** lit l'EBNF ISO 14977 (un sous-ensemble, `src/ebnf.rs`) et l'ABNF
  (`src/abnf.rs`) dans un même `EbnfGrammar`. Il génère des phrases qui la respectent (MCTS
  contraint, `src/constrained.rs`). Il ne peut pas vérifier un texte contre une grammaire : il n'a
  pas de reconnaisseur.
- **Les tours de grammaires** (une grammaire par métier, à plusieurs niveaux, chacun abaissé dans
  le suivant de façon déterministe) et l'évolution de grammaires d'`ix-grammar` et de TARS.
- **ADR-0010 (origami) :** des fichiers qu'IX vérifie et que TARS ébauche, sous la constante
  `ix_origami::GRAMMAR`.
- **TARS** stocke des grammaires EBNF et y contraint la sortie d'un modèle local, par llama.cpp
  ou vLLM.

La grammaire d'origami montre pourquoi une grammaire gardée pour la documentation ne suffit pas
(`crates/ix-origami/src/sequence.rs:51`) :

- elle définit `line` deux fois, une fois comme ligne du fichier et une fois comme ligne
  géométrique ;
- elle laisse `letter`, `digit`, `digits`, `integer`, `newline` et `any character` non définis.

Un analyseur général la refuserait. Elle ne fonctionne aujourd'hui que parce qu'un analyseur écrit
à la main fait le vrai travail, et le seul test vérifie que des mots-clés apparaissent dans le
texte de la grammaire (`sequence.rs:1056`).

Les analyseurs disponibles :

- **Les bibliothèques d'analyse Rust maintenues** sont PEG (`pest`, qui peut interpréter une
  grammaire à l'exécution par `pest_vm`) ou à combinateurs (`chumsky`). Aucune ne lit telle quelle
  une grammaire EBNF hors contexte : le choix ordonné et la répétition gloutonne de PEG changent
  silencieusement ce que signifie une grammaire EBNF.
- **Les générateurs LR** (`lalrpop`) et tree-sitter génèrent du code à la compilation.
- **Earley** accepte toute grammaire hors contexte à l'exécution, et signale l'ambiguïté et les
  terminaux attendus à la première erreur. Les crates Earley en Rust sont petits et peu actifs :
  la dernière version d'`earlgrey`, 0.4.1, date de 2024-10, celle de `santiago`, 1.3.1, de 2022-05
  (crates.io, 2026-10-28).

## Décision

**1. Un DSL est une grammaire plus un parcoureur Starlark, et aucun Rust par DSL.** Un DSL est un
fichier `.ebnf`, dans le sous-ensemble que lit `ix-grammar::ebnf`. Un script durable appelle la
capacité `parse(dsl, text)`. Elle renvoie l'arbre sur le rail du succès, et le script le parcourt
et appelle d'autres capacités.

**2. Une grammaire doit être exécutable, sinon elle est refusée.** Les vérifications ont besoin
d'un lecteur qui garde ce que perd celui d'aujourd'hui. `ix-grammar::ebnf::parse` abaisse vers
`EbnfGrammar`, et cet abaissement perd trois choses :

- une règle en double écrase la première (`HashMap::insert`, `ebnf.rs:474`) ;
- un littéral entre guillemets et un nom de règle deviennent tous deux une simple `String`
  (`ebnf.rs:522`) ;
- un terminal est tout ce qui n'a pas de production (`constrained.rs:79`), si bien que `start = "digit";
  digit = "0";` lit le `"digit"` entre guillemets comme la règle.

`ix-grammar` gagne donc un arbre syntaxique typé, lu avant l'abaissement. Dans cet arbre, un
littéral est un littéral et un nom est un nom, et les doublons sont gardés pour pouvoir être
signalés. Les vérifications portent sur cet arbre :

- chaque nom utilisé est défini, par la grammaire ou par le prélude ci-dessous ;
- aucun nom n'est défini deux fois ;
- une seule règle de départ : la première règle, comme dans le fichier ;
- la grammaire n'est pas cyclique : aucune règle ne se dérive elle-même sans consommer de
  caractère. La vérification porte sur l'abaissement propre de l'arbre typé, règles synthétiques
  pour `[ ]` et `{ }` comprises, où un littéral reste un littéral. L'`EbnfGrammar` d'aujourd'hui ne
  peut pas la porter : là, `start = "start";` se lirait comme `start` se dérivant lui-même. La
  vérification construit un graphe avec un arc de A vers B partout où A a une alternative `α B β`,
  avec `B` un nom de règle, dont `α` et `β` peuvent tous deux reconnaître le vide, et refuse tout
  cycle dans ce graphe.
  - Cela couvre les cycles unitaires (`a = b; b = a | "x";`) et les répétitions annulables
    (`{ [ "x" ] }`, qui signifie la même chose que `{ "x" }`).
  - Cela admet encore la répétition ordinaire comme `{ item [ sep ] }`, puisque chaque cycle y
    consomme un `item`.
  - Une grammaire cyclique a une infinité d'analyses d'un même texte, si bien qu'aucun compte sur
    elles ne se terminerait.

Une grammaire qui échoue donne `GRAMMAR_INVALID`, avec les noms. Les termes lexicaux viennent d'un
prélude fixe et documenté, qu'une grammaire peut utiliser sans les définir. Chacun est un seul
identifiant, puisque le lecteur sépare les noms sur les espaces :

- `letter`, `digit` (ASCII) ;
- `space` (espace ou tabulation), `newline` (LF ou CRLF) ;
- `any_char` (tout scalaire Unicode sauf un saut de ligne).

Tout le reste, comme `digits` ou `integer`, la grammaire le définit elle-même. L'analyse se fait
sans analyseur lexical, sur les valeurs scalaires Unicode, si bien qu'aucun analyseur lexical
séparé n'existe pour s'écarter de la grammaire.

**3. L'analyse est Earley, déterministe et bornée.** Le résultat est l'un des deux rails de
l'ADR-0016 §4. Un échec est `{"ok": False, "err": {"code", "message", …}}`, avec les champs
ci-dessous dans `err` :

- **Une analyse :** `{"ok": True, "value": tree}`. Un nœud d'arbre est `{"rule": name, "span": [start,
  end], "children": [...]}` et un terminal est son texte. Les étendues sont des décalages en
  caractères. Seules les règles nommées de la grammaire donnent des nœuds : les règles
  synthétiques de `[ ]`, `{ }` et des groupes placent leurs enfants directement dans le nœud de la
  règle nommée qui les entoure, si bien que tout analyseur derrière l'interface donne le même
  arbre.
- **Aucune analyse :** code `SYNTAX`, avec `line`, `column` et `expected`, l'ensemble trié des
  terminaux que l'analyseur pouvait prendre à cet endroit.
- **Deux analyses :** code `AMBIGUOUS`, avec `line`, `column` et `rule`, où les premières
  dérivations se séparent. Un texte, un sens, un rejeu.
- **Au-delà des bornes :** code `TOO_COMPLEX`. Le texte fait au plus 64 Kio. La table
  d'analyse et la forêt d'analyse partagée contiennent chacune au plus un nombre fixe d'éléments,
  comptés à mesure qu'ils sont créés.

L'ambiguïté se décide sur la forêt. La grammaire est acyclique (décision 2), si bien que la forêt
est un graphe fini sans cycle. Le nombre d'arbres à chaque nœud est compté de bas en haut, en
saturant à deux, et le compte de la racine dit s'il y a une analyse ou `AMBIGUOUS`. Ce compte est
linéaire en le nombre de nœuds et d'arcs de la forêt. Les plafonds sont fixés à partir de la
mesure de la tranche traceuse : quatre fois ce dont a besoin un fichier origami non ambigu de
64 Kio. Cela prouve qu'un fichier origami de taille maximale tient. Une autre grammaire peut
demander davantage pour la même taille, si bien que la tranche traceuse de chaque DSL mesure sa
plus grande entrée attendue au regard des plafonds.

`earlgrey` est essayé en premier, derrière un module d'`ix-grammar`, sur le test de conformité de
la tranche traceuse. Un reconnaisseur d'Earley dans `ix-grammar` même ne le remplace que si
`earlgrey` ne peut pas signaler la position de l'ambiguïté ou l'ensemble attendu, ou échoue au
test. L'interface reste la même dans les deux cas.

**4. Une grammaire est identifiée par son empreinte.** L'empreinte est le SHA-256 des octets de la
grammaire. `parse` prend une grammaire qu'un run a verrouillée (ADR-0018), jamais un chemin, et
son intention porte l'empreinte. `parse` est une capacité enregistrée (ADR-0016 §5) : l'arbre est
journalisé, si bien que le rejeu ne dépend pas de la version de l'analyseur.

**5. Les nouveaux DSL passent par le magasin, comme toute proposition.** Un agent peut concevoir
une grammaire et la proposer avec `propose("ebnf", name, body)`. Le magasin ajoute la sorte
`ebnf`, dont le fichier de bibliothèque est `<name>.ebnf`. L'hôte refuse une grammaire qui échoue
à la décision 2 avant de la publier. Une personne l'accepte (ADR-0011 §6), et ce n'est qu'ensuite
qu'un run peut la verrouiller. L'opérateur importe de la même façon un fichier `.ebnf` venu de
l'extérieur.

**6. Les tours sont en Starlark, pas une nouvelle machinerie.** Un DSL de haut niveau est abaissé
par un parcoureur Starlark vers le texte d'un DSL de plus bas niveau. Ce texte est analysé par sa
propre grammaire, si bien que chaque niveau est vérifié par une grammaire, et le journal contient
le texte de chaque niveau.

**7. TARS ébauche, IX juge.** Le même fichier `.ebnf` peut contraindre un modèle local dans TARS,
par GBNF dans llama.cpp. C'est une tranche ultérieure, mais l'analyse d'IX reste juge dans tous
les cas : une ébauche qui ne s'analyse pas est refusée à sa première erreur, quel qu'en soit
l'auteur.

## Tranche traceuse (test d'abord, chaque test vu rouge)

0. **Le lecteur typé.** Il garde les littéraux à part des noms et signale une règle en double, un
   nom manquant et une grammaire cyclique, y compris le cycle unitaire `a = b; b = a | "x";`.
   `start = "digit"; digit = "0";` accepte le texte `digit`, pas `0`. `start = "start";` et
   `{ item [ sep ] }` sont acceptés, et non signalés comme cycliques.
1. **La grammaire d'origami devient exécutable.**
   - Sa ligne géométrique est renommée (`crease`).
   - Elle utilise le prélude, et elle définit elle-même `digits` et `integer`.
   - Son test de définition vérifie la décision 2 sur elle, pas des mots-clés dans son texte.
2. **`ix-grammar` analyse avec Earley :**
   - les quatre issues de la décision 3 sur de petites grammaires : une analyse, une erreur de
     syntaxe avec son ensemble attendu, une grammaire ambiguë, une entrée trop grande ;
   - ses arbres sur la grammaire d'origami ;
   - les tailles mesurées de la table d'analyse et de la forêt pour un fichier origami de 64 Kio,
     qui fixent les plafonds.
3. **Conformité avec l'analyseur écrit à la main**, sur chaque fichier `.origami` du dépôt plus des
   copies mutées (un jeton supprimé, un mot-clé échangé, un caractère parasite) :
   - chaque fichier que l'analyseur écrit à la main accepte, Earley l'accepte ;
   - chaque fichier qu'Earley accepte et que l'analyseur écrit à la main refuse enfreint une règle
     qu'une grammaire ne peut pas énoncer (une instruction qui vient en premier et une seule fois,
     deux coins différents), listée dans le test avec sa raison.
   L'item 3 ne détecte la dérive que dans ce corpus et ces classes de mutations ; l'item 4 teste
   au-delà.
4. **La génération s'accorde avec la reconnaissance.** Les phrases générées à partir de la
   grammaire par la recherche contrainte d'`ix-grammar` sont toutes acceptées par l'analyseur
   d'Earley. Elles sont aussi acceptées par l'analyseur écrit à la main, ou refusées seulement pour
   l'une des règles listées à l'item 3. Cette seconde vérification détecte une grammaire qui
   accepte plus que le langage : le générateur et l'analyseur partagent la grammaire, ils ne
   peuvent donc pas le détecter à eux deux. Le DSL Workflow-of-Thought de TARS garde son analyseur
   honnête de la même façon.
5. **Seulement alors, `parse` dans `ix-durable`.** Un script durable analyse une séquence de plis
   et la propose. Un arrêt forcé après l'analyse rejoue l'arbre journalisé.

## Rejeté

- **Un analyseur écrit à la main par DSL :** c'est la dérive que montre la grammaire d'origami.
- **PEG (`pest_vm`, une grammaire traduite) :** le choix ordonné cache l'ambiguïté et change ce
  qu'une grammaire EBNF accepte.
- **Les générateurs LR ou tree-sitter :** ils intègrent les grammaires au code à la compilation,
  mais les agents ont besoin de grammaires chargées à l'exécution. Les générateurs LR refusent
  aussi les grammaires ambiguës.
- **Un analyseur lexical séparé :** une seconde spécification à garder en phase avec la grammaire.
- **Un LLM qui juge si un texte suit une grammaire :** un juge au taux de fausses acceptations
  élevé, là où un analyseur n'en a aucune.

## Réversibilité

**Porte à double sens** tant qu'`ix-durable` est expérimental. La forme de l'arbre devient un
contrat dès qu'un module publié la parcourt (ADR-0018), et elle est alors figée sous
`ix-dsl-tree/0`. Le prélude fait partie de ce contrat.

## Déclencheurs de révision

- **Un DSL a besoin de règles sensibles au contexte** (l'indentation, un nom défini avant usage) :
  ajouter d'abord une passe de vérification en Starlark ; n'étendre le langage de grammaire que si
  beaucoup de DSL ont besoin de la même vérification.
- **Le temps d'analyse compte** (de grands textes, beaucoup d'appels) : mesurer, puis envisager un
  analyseur plus rapide pour les grammaires non ambiguës, derrière la même interface.
- **`earlgrey` cesse d'être maintenu d'une façon qui fait mal** (un rapport de sécurité, une casse
  du compilateur) : passer au reconnaisseur interne au crate, ce que l'interface permet déjà.
