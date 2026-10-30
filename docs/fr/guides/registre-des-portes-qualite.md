# Le registre des portes qualité dans IX

**Contrat :** [`docs/contracts/2026-05-24-quality-gate-ledger.contract.md`](../../contracts/2026-05-24-quality-gate-ledger.contract.md)
**Anglais :** [`docs/guides/quality-gate-ledger-in-ix.md`](../../guides/quality-gate-ledger-in-ix.md)

## De quoi il s'agit

Un fichier JSONL par dépôt, en ajout seul, situé à
`state/quality/gate-ledger.jsonl`. Chaque ligne est le verdict d'une porte
qualité à un instant donné : qui a tourné, ce qui a été mesuré, et si c'est
passé.

```
{"schema_version":1,"schema":"quality-gate-ledger-v1","id":"01a0b19d-…",
 "run_at":"2026-09-17T23:04:24Z","source":"ix-doctor","domain":"harness",
 "decision":"pass","metric":{"name":"doctor_checks_failing","value":0.0,"threshold":0.0},
 "extra":{"verdict":"T","checks":{"registry-snapshot":"ok", …}}}
```

## Qui l'écrit

**`ix doctor`** — la porte pré-PR. Chaque exécution ajoute exactement une ligne :

```bash
cargo run -p ix-skill --bin ix -- doctor
```

Une ligne par exécution, pas une par vérification. La réponse de la porte est
son verdict agrégé — c'est ce que rapporte le code de sortie — donc `metric`
porte le nombre de vérifications en échec face à un seuil de zéro, et le détail
par vérification voyage dans `extra`, aux côtés du mode d'exécution (`--write` /
`--full`). Une ligne produite sans `--full` a sauté les vérifications clippy et
tests : la comparer à une ligne `--full` sans lire `extra.mode` reviendrait à
comparer deux portes différentes.

La sortie humaine se termine par le chemin écrit. Si l'ajout échoue, il le dit
sur stderr et le verdict de la porte reste inchangé — un registre refusé par le
système de fichiers ne doit pas faire passer un dépôt vert au rouge.

`ix-sentrux-gate-writer` (`crates/ix-quality-trend/src/bin/sentrux_gate_writer.rs`)
est un second producteur, toujours dormant : il enveloppe `sentrux gate` et
exige un binaire `sentrux` dans le PATH, absent de cette machine.

## Qui le lit

L'outil MCP `ix_quality_gate_history` :

```jsonc
{ "source": "ix-doctor", "decision": "fail", "since": "2026-09-01T00:00:00Z", "limit": 20 }
```

Le chemin par défaut est résolu par rapport à la **racine de l'espace de travail
ix**, et non au répertoire courant du processus. Passez `ledger_path` pour lire
le registre d'un dépôt voisin (ga écrit le sien).

### Lire `ledger_status` avant `count`

```jsonc
{ "ledger_status": "absent", "count": 0, "note": "no ledger at this path — …" }
```

`count: 0` seul est ambigu, et l'une de ses deux lectures est dangereuse :

| `ledger_status` | Ce que signifie `count: 0` |
|---|---|
| `absent` | Aucune porte n'a jamais enregistré d'exécution ici. Ce n'est **pas** une preuve que les portes sont passées. |
| `empty` | Le fichier existe, sans lignes. Même poids qu'`absent`. |
| `present` | Des portes ont tourné ; votre filtre les a toutes exclues. Celui-là *est* une preuve. |

Avant ce câblage, l'outil répondait à toutes les requêtes depuis un fichier
absent, et `count: 0` ressemblait exactement à un historique sain. C'est le mode
de défaillance que le dépôt appelle *green-but-dead* : une réponse rassurante
avec rien derrière.

## Notes pratiques

- **Ignoré par git dans ix**, versionné dans ga. `ix doctor` tourne sur la
  machine de chaque contributeur : versionner le fichier mettrait un JSONL en
  ajout seul sur le chemin de fusion de chaque PR. ga versionne le sien parce
  qu'un script CI l'écrit, et non chaque contributeur. Conséquence : le registre
  d'ix est un historique propre à chaque copie de travail, et un clone frais
  répond `absent` jusqu'à la première exécution de la porte.
- **Borné.** À 4 Mio (~10 000 lignes), le fichier vivant est basculé vers
  `gate-ledger.1.jsonl` ; une seule génération est conservée et les
  consommateurs ne lisent que le fichier vivant. Tout ce qui doit survivre à la
  rotation appartient à un instantané daté sous `state/quality/`.
- **Résistant au crash, ligne par ligne.** Chaque ligne est un unique
  `write_all` sur un descripteur ouvert en `O_APPEND` : des producteurs
  concurrents ne peuvent pas entrelacer de demi-lignes. Les lecteurs ignorent
  les lignes vides et rétrogradent une ligne v1 malformée en v0 historique ; le
  pire cas est une ligne perdue, pas un fichier illisible.
- **Les lignes v0 historiques coexistent.** Les anciennes lignes de ga, en forme
  de PR chatbot, n'ont pas de `schema_version`. Elles sont exclues de `rows`
  mais comptent tout de même comme historique : un registre uniquement v0
  répond `present`, pas `empty`.
