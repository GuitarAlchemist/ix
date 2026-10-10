# ADR-0009 — proposition de périmètre borné via IXQL

Statut : proposé, draft uniquement. [Contrat détaillé anglais](../../adr/0009-mission-scope-refinement.md).

La tranche réutilise parser, évaluateur, registre de capacités et horloge IXQL.
Après un refus racine lié par l'hôte, `mission.refine_scope` peut proposer
`git add -- Blue/` si la propriété déclarée complète le permet. Refus inconnu,
propriété ambiguë, doublon, chemin non canonique, observation absente/périmée
ou corrélation incorrecte restent des refus explicites.

Contrat draft 0.1.0, versions/SHA exacts et identités mission/tentative/révision/refus.
Observation et échantillon expirent sous 30 secondes ; heartbeat sans preuve
ne prolonge rien. La sortie conserve `requires_runner_admission: true`.
Aucun lancement, autorité, effet, reprise ou reçu d'idempotence n'est fourni.

Priorité : boucle de reprise réelle testée par le propriétaire Gaia246/247.
Le raccord Git/gh247 dérive les fichiers de son changeSetIdentity existant ;
ce n'est pas un consommateur de cette proposition ni une API générique de reprise.
IX ne duplique pas Ed25519, admission, permissions ou reprise.

La CI RED démontre l'adaptateur manquant. La suite GREEN couvre uniquement
parser → registre → validation → proposition avec un hôte simulé.
Preuve réelle consommation → première action → résultat/timeout encore manquante.

Backlog ordonné : raccord Gaia exact, observation Go réelle, intégration au schéma
Incubateur existant, génération d'interfaces, catalogue/versionnement des artefacts
Claude/Codex/MCP, puis blue/green sans migration ni effets doubles.
Ces sujets ne sont pas implémentés dans cette tranche.
