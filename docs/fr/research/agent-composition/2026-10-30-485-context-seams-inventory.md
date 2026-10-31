# Inventaire pour #485 : les points d'appui de contexte qu'IX a déjà

Note de recherche pour [#485](https://github.com/GuitarAlchemist/ix/issues/485) (carte
[#483](https://github.com/GuitarAlchemist/ix/issues/483)). Version anglaise :
[`docs/research/agent-composition/2026-10-30-485-context-seams-inventory.md`](../../../research/agent-composition/2026-10-30-485-context-seams-inventory.md).
Lu seulement, sur origin/main 99f2f869 : rien ci-dessous n'a tourné sauf mention **testé**. Elle
consigne ce qui existe ; l'opérateur décide #485.

## Les points d'appui, face aux questions de #485

| Point d'appui | Ce que c'est | Portée / durabilité | Épinglage de version | Mise à jour concurrente | Entrée manquante |
|---|---|---|---|---|---|
| `ix-agent-core` `ReadContext` (`context.rs:51`) | projection immuable du journal d'événements d'une session à un `ordinal` ; les écritures passent par `WriteContext` comme nouveaux événements | privé à une session ; le crate fournit `VecEventSink` (mémoire, tests seulement) ; la durabilité relève d'`ix-session` | `(session_id, ordinal)`, plus un `replay_seed` | aucune : événements en ajout seul, pas de document partagé | n/a |
| `ix-session` `SessionLog` (`log.rs`) | puits d'événements JSONL | disque local ; `flush` d'un `BufWriter` à chaque émission, aucun `sync_all` trouvé | position dans le journal | non lu | non lu |
| `ix-context` `ContextBundle` + `ContextCache` (`cache.rs`) | parcours déterministe du contexte de code avec un `walk_trace` rejouable | en processus, `Mutex<HashMap>` seulement | clé `(HEAD git, hash du fichier, hash du manifeste, stratégie, racine)` ; SipHash, **non** cryptographique | `invalidate_path` manuel, aucun observateur | n/a |
| `ix-recover` `Store` (`lib.rs`) | résultat publié une seule fois par id d'exécution choisi par l'appelant : intention avant l'effet, fichiers de tentative + manifeste, validation par lien physique | disque local ; Unix synchronise les répertoires, Windows ne prétend **rien** en cas de coupure de courant | `sha256:` par fichier et des liaisons canoniques | un seul écrivain : verrou de propriétaire ; liaisons différentes pour un même id → `Error::Conflict` | vérifié à la réconciliation → `Corrupt` / `Unknown` |
| `ix-loop` `Store` (`store.rs`) | les boucles proposent (`proposals/<id>.json`), seul l'`accept` d'une personne en fait passer une dans `library/` | disque local, `IX_LOOP_STORE` ou `~/.cache/ix-loop` | aucun | **`accept` écrasait un fichier de bibliothèque du même nom** (`fs::write`) : aucun conflit | chaîne d'erreur |
| nœud `ClaudeCode` d'`ix-loop` (`node.rs`) | appel Claude enfant | un processus par demande, `--no-session-persistence` | n/a | n/a | n/a |
| `Host` IXQL (`host.rs:97`) | système de fichiers réel confiné à une racine (ancêtres canonicalisés) ; `ix.io.write` passe la barrière de schéma | la racine de l'exécution | **aucune empreinte** sur les lectures (aucun hash dans `host.rs`) | non lu | erreur de l'hôte |
| `ix-duck` `ArtifactLens` (`source.rs`) | répertoire de fichiers d'artefacts → une table DuckDB | une vue dérivée, reconstruite à la lecture | aucun | n/a | **répertoire absent ou vide → table vide** (« dégradation gracieuse ») |
| signets / répertoire d'état ixctl | signets par plan (tmp + renommage, 0600) et un verrou OS par nom | disque local | aucun | une instance par nom | — (le journal n'est jamais relu : voir la carte de grilling G1) |
| artefacts GitHub | — | — | aucun code ne récupère un chemin GitHub épinglé (`git grep` des URL raw/contents : rien) | — | — |

## Ce que cela dit de #485

- **Les deux modes de stockage existent déjà, avec des contrats différents.** Mémoire : `ReadContext`,
  une projection pure épinglée par ordinal. Disque local durable : `ix-recover`, avec un manifeste,
  un SHA-256 par fichier et un conflit à écrivain unique. Rien ne les relie, et la règle de #485 est
  de n'ajouter un magasin universel que si deux modes concrets l'exigent.
- **Privé ou partagé n'est une notion nulle part.** Chaque point d'appui ci-dessus est privé à un
  processus, un id d'exécution ou une session. Aucun n'offre de dossier partagé avec des mises à jour
  à version attendue (CAS) : le seul `Conflict` trouvé (`ix-recover`) protège les liaisons d'un id
  d'exécution, pas un document partagé.
- **Le seul chemin d'écriture à demi partagé perdait des mises à jour en silence.** Deux propositions
  du même nom : le second `accept` remplaçait le premier fichier de bibliothèque sans conflit. C'est
  exactement le contre-exemple que nomme #485 (« les écrivains concurrents entrent en conflit
  visiblement au lieu d'écraser »). **Testé** :
  `accepting_a_second_loop_under_a_taken_name_is_a_visible_conflict` a été vu rouge (« the second
  accept replaced the first loop ») ; la correction (`create_new`, la proposition reste en attente)
  est la pull request #491. Avec elle, remplacer volontairement une boucle demande de supprimer
  d'abord l'ancien fichier : pas de remplacement explicite.
- **Des propositions plutôt que des écritures existent déjà une fois.** La séparation proposition /
  acceptation d'`ix-loop` est la règle « les ajouts rendus sont des propositions », pour les boucles
  seulement.
- **Un artefact manquant se lit comme vide dans la lentille DuckDB.** Pour un tableau de bord, c'est la
  dégradation voulue ; pour le contexte d'une tâche, cela changerait « fichier manquant » en « aucune
  donnée » sans bruit. Le cas du fichier manquant de #485 ne doit pas passer tel quel par ce chemin.
- **Les références reproductibles ne sont adressées par contenu qu'à un seul endroit.** `ix-recover`
  hache en SHA-256 ; `ix-context` utilise SipHash (une clé de cache, pas une identité) ; les lectures
  IXQL ne portent aucune empreinte.

## Proposition pour l'opérateur (pas une décision)

D'abord des listes d'artefacts explicites et immuables, pas un ContextStore : une tâche reçoit une
liste d'éléments, chacun `{id, source (mémoire | chemin local | github dépôt/chemin/commit), sha256,
schéma, classification, octets}`, avec le format d'empreinte d'`ix-recover` réutilisé. Une exécution
en mémoire seulement dit que la reprise après redémarrage n'est pas disponible. Les mises à jour
partagées restent hors de la première tranche : aucun point d'appui n'a de CAS, donc toute écriture
partagée devrait en ajouter un.

Plus petite vérification discriminante (hors ligne, comme le demande #485), avec les points d'appui
ci-dessus :

1. Instantané privé en mémoire : un `ReadContext` à l'ordinal *n* est inchangé après un événement à
   *n+1*.
2. Instantané local exporté puis restauré : une validation `ix-recover` relue ; un octet changé →
   refusé.
3. Deux propositions partagées en conflit : deux propositions `ix-loop` du même nom, toutes deux
   acceptées. Attendu par #485 : un conflit visible. C'est le premier test rouge, fait (voir #491).
4. Fichier manquant, empreinte changée, schéma incompatible, export incomplet : via la liste
   d'artefacts, chacun une issue explicite, jamais la table vide de la lentille. Lecture révoquée :
   indéterminé (aucun point d'appui ne modélise les droits d'accès).
