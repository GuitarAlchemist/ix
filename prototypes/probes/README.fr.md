# Probes — prototype

[English](README.md)

Un processus hôte qui exécute de petits programmes Go, les *probes*, chacune
recueillant une sorte de mesure sur la machine. Modifiez le source d'une probe
pendant que l'hôte tourne : l'hôte la recompile et remplace le processus en
cours, sans redémarrer lui-même. Tout ce que les probes impriment aboutit dans
un seul fichier JSONL que DuckDB lit.

C'est un prototype, sur une branche `prototype/probes` : il répond à la
question de savoir si la boucle *modifier une probe → elle tourne → DuckDB la
voit* fonctionne sous Windows, et ce qu'elle coûte. Il n'est pas branché à IX.

## Le contrat

Une probe est un répertoire sous `probes/` contenant un package Go `main`.
Elle imprime un objet JSON par ligne sur stdout :

```json
{"ts":"2026-10-07T22:47:41.123456Z","metric":"ram_free_mb","value":11133.4,"unit":"MB"}
```

`metric` (une chaîne) et `value` (un nombre) sont obligatoires ; `ts` et
`unit` sont facultatifs. L'hôte ajoute `probe` (le nom du répertoire) et
`build` (le hash de son source), ainsi qu'un `ts` si la probe n'en a pas
donné. Une ligne qui n'est pas un tel objet devient plutôt un événement
`bad_line`. Ce qu'une probe écrit sur stderr devient des événements `stderr`.
L'hôte transmet l'intervalle d'échantillonnage à la probe dans
`PROBE_INTERVAL_MS`.

Les probes présentes : `sysinfo` (RAM, CPU), `gpu` (nvidia-smi), `disk`, les
probes `proc-*` faites à partir du modèle de processus, et `stream`, qui
surveille la diffusion de la scène depuis cette machine seulement :
`stream_hls_up` et `stream_hls_ms` (la liste de lecture HLS sur la boucle
locale répond 200, et en combien de temps), `stream_segments`,
`stream_front_up` (la porte d'entrée de la page du lecteur répond) et
`stream_viewers`, lu dans le compte que la porte d'entrée écrit, et omis
quand il date de plus de 30 s. Elle ne consulte jamais le lien public : la
probe n'ajoute rien à ce que le flux montre.

L'hôte écrit :

- `out/probes.jsonl` — les enregistrements ;
- `out/host-events.jsonl` — ses propres événements : `host_started`, `built`,
  `build_failed` (avec `kept_running`, le build qui tourne encore), `started`,
  `exited`, `stopped`, `start_failed`, `bad_line`, `stderr`, `host_stopped` ;
- `out/bin/<probe>-<hash>.exe` — un exécutable par build.

## Comment fonctionne le rechargement à chaud

Le package `plugin` de Go n'existe pas sous Windows, et Windows verrouille un
exécutable en cours. Chaque build est donc un nouveau processus : toutes les
500 ms, l'hôte hache les fichiers `.go` de chaque probe (le contenu, pas les
dates de modification) ; quand un hash change, il lance `go build` vers un
exécutable nommé d'après le hash, le démarre, et n'arrête l'ancien qu'une fois
que le nouveau a imprimé sa première ligne (ou s'est arrêté, ou après 30 s) :
un remplacement ne laisse donc pas de trou. Si le nouveau source ne compile pas, l'ancien build continue de
tourner et l'hôte ne réessaie pas ce source tant qu'il ne change pas de
nouveau. Une probe qui plante est redémarrée, au plus toutes les 2 s. Une
probe dont le répertoire est supprimé est arrêtée.

## L'exécuter

Il faut Go 1.27 dans le PATH et, pour l'agrégation, la CLI DuckDB.

```powershell
cd prototypes/probes
go build -o out/bin/host.exe ./host
out/bin/host.exe -for 2m          # ou sans -for : jusqu'à Ctrl+C
duckdb -c ".read aggregate.sql"   # par métrique, puis par build
```

Options : `-probes` (par défaut `probes`), `-out` (`out`), `-poll` (`500ms`),
`-interval` (`1s`), `-for` (`0`, jusqu'à interruption), `-rotate-bytes`
(64 Mio : chaque fichier JSONL est renommé `<nom>-<heure UTC>.jsonl` à cette
taille, `0` jamais) et `-keep` (`4` fichiers renommés de chaque). Un fichier
nommé `stop` dans le répertoire de sortie arrête l'hôte et ses probes, comme
Ctrl+C.

Pour le faire tourner en arrière-plan comme l'attend le panneau de la scène 3D,
en écrivant dans `%LOCALAPPDATA%\ix-probes` toutes les 5 s, avec le veilleur de
demandes à côté :

```powershell
pwsh -File start.ps1   # compile l'hôte au besoin ; ne fait rien pour ce qui tourne déjà
pwsh -File stop.ps1    # par le fichier stop, puis le veilleur
```

`out/probes.jsonl` est du JSONL ordinaire avec des horodatages typés, donc
toute requête DuckDB s'y applique, par exemple la charge CPU par tranches de
10 secondes :

```sql
SELECT time_bucket(INTERVAL 10 SECOND, ts) AS t, build, round(avg(value), 1) AS cpu
FROM read_json_auto('out/probes.jsonl') WHERE metric = 'cpu_load_pct'
GROUP BY ALL ORDER BY t;
```

## La première probe : `sysinfo`

WMI, via `github.com/yusufpapurcu/wmi` (MIT) : `Win32_OperatingSystem` pour
`ram_total_mb` et `ram_free_mb`, `Win32_Processor` pour `cpu_logical` et
`cpu_load_pct` (la moyenne de `LoadPercentage` sur les processeurs, omise
tant que WMI n'a pas encore d'échantillon).

## Vérifié, et mesuré

`check_hot_reload.py` exécute toute la boucle et la vérifie. Il démarre un
hôte à lui sur une copie de `sysinfo` sous `out/check-hr`, compare un
échantillon avec `Get-CimInstance`, ajoute une métrique au source de la copie,
puis le casse, arrête l'hôte avec Ctrl+Break et exécute `aggregate.sql`. Un
hôte qui tourne déjà sur `probes/` n'est pas touché. Relancé le 2026-10-18 avec
le remplacement sans trou : PASS, le plus long intervalle sans échantillon
1,38 s pour un intervalle de 1 s.

```powershell
python -B check_hot_reload.py     # depuis prototypes/probes, après avoir compilé l'hôte
```

Le 2026-10-07, sur une machine Windows 11 à 24 threads dont la charge CPU
était de 90 à 100 % tout du long (d'autres travaux en cours), il a réussi :

| | Quoi | Mesuré |
| --- | --- | --- |
| S1 | premier échantillon après le démarrage de la probe, ≤ 5 s | 1,3 s ; plus le premier build, 12,8 s avec un cache Go chaud |
| S2 | RAM totale et CPU logiques égaux à `Get-CimInstance` ; RAM libre à 10 % près | égaux ; RAM libre à 1 % près |
| S3 | une modification du source est prise en compte, l'hôte continue de tourner | la nouvelle métrique 15 à 19 s après la modification ; un seul `host_started` |
| S4 | un source qui ne compile pas laisse l'ancien build tourner | `build_failed` avec `kept_running` renseigné, les échantillons continuent depuis l'ancien build |
| S5 | DuckDB agrège la sortie | les deux requêtes d'`aggregate.sql` s'exécutent ; `ts` est lu comme `timestamp` |

Observé aussi :

- **Une modification met 15 à 20 s à apparaître, presque tout en `go build`**
  (8 à 14 s sous cette charge). Démarrer un nouvel exécutable prend encore 3 à
  5 s, ce qui ressemble à l'antivirus qui l'analyse ; c'est pourquoi le
  nouveau build démarre avant que l'ancien ne s'arrête.
- **Le remplacement laissait un trou de 2 à 3,5 s** dans les échantillons :
  le temps entre le démarrage du nouveau processus et son premier échantillon
  WMI. Fermé le 2026-10-18 (voir plus bas).
- La période d'échantillonnage était de 1,2 à 1,3 s pour un intervalle de
  1 s, les requêtes WMI étant lentes sous la charge.

`check_host.py` vérifie ce qui est venu ensuite, sur une probe jetable à lui
sous `out/check-probes`, si bien qu'un hôte qui tourne déjà sur `probes/` n'est
pas touché :

```powershell
python -B check_host.py [host.exe]   # out/bin/host.exe par défaut
```

Le 2026-10-18, il a passé :

| | Quoi | Mesuré |
| --- | --- | --- |
| S6 | le JSONL est renommé à `-rotate-bytes`, seuls `-keep` fichiers renommés restent | 2 gardés pour keep 2, chacun de 20 150 octets pour une limite de 20 000 |
| S7 | un remplacement ne laisse aucun trou de plus de 1 s ; l'ancien build s'arrête après le premier échantillon du nouveau | plus long trou 51 ms pour un intervalle de 50 ms ; arrêté 50 ms après |
| S8 | le fichier stop arrête l'hôte et ses probes en moins de 5 s | 0,1 s, aucune probe restée en marche |

Un hôte qui arrête l'ancien build tout de suite fait passer S7 au rouge :
l'ancien build s'est arrêté avant le premier échantillon du nouveau. Le trou
seul ne le montrerait pas avec cette probe, qui échantillonne aussitôt ; le
trou de 2 à 3,5 s venait de la première requête WMI.

## Des probes à partir de modèles, et des demandes

`new_probe.py` écrit une probe à partir d'un modèle fixe, jamais de code libre.
Son seul modèle aujourd'hui est `process` : le CPU, la mémoire et le nombre des
processus dont le nom WMI correspond à un motif `LIKE`. Chaque paramètre doit
correspondre à un motif strict avant d'atteindre le source :

```powershell
python -B new_probe.py process scene "godot.windows.editor%"   # probes/proc-scene
```

`probe_requests.py --every 2` permet à la scène 3D, à une commande vocale ou à
un agent d'en demander une sans rien exécuter. Une demande est un fichier JSON
dans `%LOCALAPPDATA%\ix-probes\requests\`, écrit d'un seul coup (un `.part`
renommé en place) et nommé par son identifiant :

```json
{"template": "process", "name": "obs", "match": "obs64%"}
```

Le veilleur la passe à `new_probe.py` et écrit `<id>.result.json` à côté,
`{"ok": true, "probe": "proc-obs"}` ou `{"ok": false, "error": "..."}`. Une
probe qui existe est signalée, pas réécrite, et il y a au plus 8 probes de
modèle. Le 2026-10-18, il a refusé `obs; rm` comme nom, `x' OR 1=1` comme motif
et un modèle `shell`. Le premier échantillon de la demande `obs` est arrivé
environ 20 s plus tard.

## Limites, et suite

- Windows seulement tel quel : `sysinfo` lit WMI, et l'hôte n'a tourné que
  là. Un hôte WSL demanderait des probes qui lisent `/proc`.
- La sortie est bornée par la rotation, au plus 5 fichiers de 64 Mio pour les
  échantillons ; il n'y a pas de rétention par âge.
- Une probe est du code de confiance que l'hôte compile et exécute. Seules les
  probes de modèle peuvent être demandées de l'extérieur, et elles ne prennent
  pas de code.
- Rien ne relance l'hôte après un redémarrage ; il faut lancer `start.ps1`.

Ensuite, dans l'ordre qui semble utile : acheminer les enregistrements vers le
banc DuckDB d'IX (`ix-duck`) pour qu'ils rencontrent les UDF d'IX ; un
démarrage à l'ouverture de session ; un hôte WSL.
