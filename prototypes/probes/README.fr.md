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
exécutable nommé d'après le hash, le démarre, et seulement ensuite arrête
l'ancien. Si le nouveau source ne compile pas, l'ancien build continue de
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
`-interval` (`1s`), `-for` (`0`, jusqu'à interruption).

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

`check_hot_reload.py` exécute toute la boucle et la vérifie. Il démarre
l'hôte, compare un échantillon avec `Get-CimInstance`, ajoute une métrique au
source de la probe, puis casse le source, arrête l'hôte avec Ctrl+Break et
exécute `aggregate.sql`. Il restaure le source de la probe à la fin.

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
- **Le remplacement laisse encore un trou de 2 à 3,5 s** dans les
  échantillons : le temps entre le démarrage du nouveau processus et son
  premier échantillon WMI. N'arrêter l'ancien build que lorsque le nouveau a
  imprimé sa première ligne le fermerait.
- La période d'échantillonnage était de 1,2 à 1,3 s pour un intervalle de
  1 s, les requêtes WMI étant lentes sous la charge.

## Limites, et suite

- Windows seulement tel quel : `sysinfo` lit WMI, et l'hôte n'a tourné que
  là. Un hôte WSL demanderait des probes qui lisent `/proc`.
- La sortie grossit sans limite : ni rotation, ni rétention.
- Une probe est du code de confiance que l'hôte compile et exécute.

Ensuite, dans l'ordre qui semble utile : fermer le trou du remplacement ;
d'autres probes (disque, GPU, mémoire par processus) ; acheminer les
enregistrements vers le banc DuckDB d'IX (`ix-duck`) pour qu'ils rencontrent
les UDF d'IX ; un hôte WSL.
