# Flux du plan de contrôle IX — prototype

[English](README.md)

Les scripts derrière deux vues de la flotte IX, tels qu'ils tournaient le
2026-10-18 :

- l'**IX Control Room**, un tableau de bord Artifact privé : les pull requests
  d'IX, les exécutions sur main, les tickets, les niveaux de maturité des
  crates et le travail en cours d'Indigo ;
- l'**instantané de contrôle** que lit la scène 3D Observatory : un
  `snapshot.json` privé pour la vue de l'utilisateur, et un `public.json` que
  le flux public peut montrer.

Gardé ici comme source primaire de la conception, et non proposé pour main
tel quel : les chemins et les identifiants des ressources de l'Artifact sont
ceux de cette machine.

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `gather.py` | Lit GitHub avec `gh` et origin/main avec git ; écrit un fichier JSON d'objets-lignes par jeu de données. |
| `gather_indigo.py` | Une ligne par élément de travail d'Indigo. Les têtes et les états des PR sont lus ; `verdict`, `review` et `next` sont des notes tenues à la main. |
| `refresh.py` | `plan` lance les deux collecteurs dans `data-<horodatage>/`, affiche les jeux de données changés selon leur sha256, puis réécrit l'instantané. `record` note chaque nouvelle ressource dans `state.json`. Les envois eux-mêmes sont des appels à l'outil Artifact de Claude. |
| `make_docs.py` | Écrit les documents du magasin du tableau de bord, un par jeu de données, pour une seule écriture groupée. |
| `index.html` | La page du tableau de bord. Elle calcule les étapes, les comptes et les semaines à partir des lignes, de façon visible pour ses lecteurs. |
| `control_snapshot.py` | Écrit `snapshot.json` et `public.json` dans `%LOCALAPPDATA%\ix-control`, de façon atomique ; `--every 5` les réécrit toutes les 5 s. |
| `demo-plan.json` | Le chantier en cours et la prochaine démo, modifiés à la main au fil du travail. |

`IX_CLONE` désigne le clone d'ix (par défaut `~/source/repos/ix`).
`IX_DASH_SCRATCH` et `IX_DASH_TASK` désignent les dossiers des fichiers de
résultats et des lots de revue dont `gather_indigo.py` recalcule les comptes.

## Les deux fichiers d'instantané

`snapshot.json`, schéma `ix-control-snapshot/0`, est pour l'utilisateur seul.
Il nomme les sessions, cite les notes du registre des réservations et donne
les chemins locaux des bases DuckDB : il ne passe donc jamais sur le flux. Il
contient :

- les PR ouvertes avec leur CI et leurs commentaires Codex P0/P1 ouverts, et
  la dernière exécution de chaque workflow sur main ;
- les sessions vivantes, lues dans `~/.claude/sessions/*.json` (jamais les
  fichiers `*.key`), et les voies actives de `~/.agents/claims.jsonl` ;
- les décisions qui attendent l'utilisateur, et combien de temps les voies au
  travail disent pouvoir avancer sans lui ;
- chaque base DuckDB des dossiers `state/` et `code/` des dépôts, avec ses
  tables, lue avec `duckdb -readonly` toutes les 5 minutes. Une base qu'un
  autre processus tient ouverte apparaît `busy`.

`public.json`, schéma `ix-public/0`, est la seule partie que le flux public
peut montrer :

- les sessions occupées ou en attente, par nom et dépôt, et le nombre de
  sessions inactives ;
- le nombre de décisions qui attendent l'utilisateur ;
- les titres et l'avancement du chantier, et l'heure de la prochaine démo ;
- les jauges de la machine, venues de l'hôte de sondes (prototype/probes,
  PR #390). Une mesure périmée ou absente donne `null` : la scène n'affiche
  alors pas de jauge plutôt qu'une ancienne.

Les deux schémas sont provisoires : un lecteur se fie à `schema` et ignore les
champs qu'il ne connaît pas.

Chaque titre de `demo-plan.json` est `{en, fr, es}`, pour que la scène montre
tout le panneau dans sa propre langue. `public.json` les porte dans `titles`
et garde `title` en français pour les anciens lecteurs. Les jauges de
processus portent leurs `labels` de la même façon.

Une réécriture qui échoue est consignée dans `snapshot-loop.log` et la boucle
continue. Le 2026-10-18, la boucle était morte à 15 h 47 sur un dossier
`data-*` à moitié écrit, ce qui avait caché les jauges pendant une démo.

## Montrer une démo dans la scène

Une démo est un script ponctuel, lancé à l'heure de la démo, qui écrit les
phrases que le bot doit dire dans le canal de parole public de la scène, puis
avance `next_demo` dans le plan. Le canal contient une ligne
`"<langue>: <texte>"` d'au plus 400 caractères, écrite de façon atomique : le
bot la dit sur le flux. Seul ce canal atteint le flux, jamais les sous-titres
privés.

## Pas fait

- Les identifiants des ressources de l'Artifact sont écrits en dur dans
  `make_docs.py`.
- Rien ne vérifie que `public.json` reste sans noms ni chemins : cela repose
  sur `public()`, qui le construit champ par champ.
- Windows seulement.
