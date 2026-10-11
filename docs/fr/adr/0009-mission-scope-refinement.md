# ADR-0009 — proposition de périmètre borné via IXQL

Statut : proposé, draft uniquement. [Contrat détaillé anglais](../../adr/0009-mission-scope-refinement.md).

La tranche réutilise parser, évaluateur, registre de capacités et horloge IXQL.
Après un refus racine lié par l'hôte, `mission.refine_scope` peut proposer
`git add -- Blue/mission.txt` si la l'inventaire déclaré nomme ce fichier régulier exact. Refus inconnu,
propriété ambiguë, dossier, symlink, type inconnu, doublon, chemin non canonique, observation absente/périmée
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

Les fixtures déclarent la propriété et le type ; elles ne prouvent aucune inspection
réelle du checkout. Celle-ci reste nécessaire à l'admission Gaia.


## Tranche WMUX : candidat sans effet (2026-10-11)

`request → mission.dispatch_candidate` réutilise le parser, Executor et registre
existants. Toujours `live_dispatch_available: false` et
`requires_runner_admission: true`. Aucun appel Python/WMUX, lecture/écriture du
ledger, boucle d'attente, annulation native ou nouvel observateur de permission.

L'archive locale `31d75c00409ff6227a199c4921be893e369c8b4e` contient les prototypes
Python, pas un commit upstream WMUX. Les trois hashes de sources ont été relus
localement ; le SHA d'archive et les 54 tests Python restent déclarés par l'auteur.
Voir la section anglaise pour les pins et les détails du contrat.

Le request typé lie mission/nonce, session/workspace/surface, brief/digest,
registry/ledger et pins exacts à la déclaration injectée par l'hôte. Entrée non
vide/inconnue ou admission non ready refuse un candidat préparé. Identité changée,
état inconnu/incohérent, observation ancienne/future, alias ou chemin invalide
refuse. Aucun producteur live n'inspecte l'entrée ou les fichiers.

Après intention ambiguë ou tout état ultérieur connu, la proposition devient
`observe_receipt` avec les mêmes identités, jamais un renvoi. Le rejeu est une
projection pure, pas un reçu d'idempotence. Un ACK, working ou consumed n'est pas
une preuve de Read réussi : `first_action_confirmed` reste false et le résultat
d'effet reste unknown, sans verdict de succès.

Budget : JSON sérialisé 4096 octets, IDs 128 caractères, chemins relatifs
canoniques distincts de 512 caractères. Cela ne borne pas les buffers Python.
Fraîcheur de déclaration inférieure à 30s ; deadline locale de 1 à 3 600 000 ms.
À expiration, `stop_waiting` ne signifie ni TTL transport ni annulation native.
La capacité ne fait aucune attente réelle.

Le cas Blue réel du 2026-10-11 à 01:08:18 UTC utilisait `relay_send.py`, sans
verrou/fsync/ledger/token de `cycle_dispatch`. Il prouve ce tour unique observé,
pas les garanties des prototypes. L'entrée ultérieure a été préservée, origine
inconnue. Aucun second envoi Blue.

Tests parser/adaptateur sur snapshots : rejeu, mauvais IDs/nonce/artifacts,
entrée humaine non vide/inconnue, expiration/overflow, pins/budget, doublons et
distinction des preuves. Aucun test de race GUI ou de reprise après crash n'est
prétendu. Intégration proposée dans la vue de proposition de mission et le gate
existants de l'Incubateur après accord de leur schéma ; aucun fichier UI/CP modifié.
