# Preuves pour #489 : ce que le spike de routage Jev garantit déjà, hors ligne

Note de recherche pour [#489](https://github.com/GuitarAlchemist/ix/issues/489) (carte
[#483](https://github.com/GuitarAlchemist/ix/issues/483)). Version anglaise :
[`docs/research/agent-composition/2026-10-30-489-jev-decision-node-evidence.md`](../../../research/agent-composition/2026-10-30-489-jev-decision-node-evidence.md).
Elle consigne des preuves et une proposition ; l'opérateur décide #489. Aucun appel réseau, aucune
clé lue, `run-live.ps1` non exécuté.

Statuts : **lu**, **testé** (une commande a tourné ici), **indéterminé**.

## Sources épinglées (testé)

- `crates/ix-router-spike/src/bin/jev_router.rs` au blob `0d8a6adcf1a13ee7182f53647eee85c3c2aabef8`
  et `state/router-spike/jev/run-live.ps1` au blob `a8a3702ef20a4bc4aac46d95215fcdba8ed703da`, les
  deux blobs que nomme #489 (`git rev-parse HEAD:<chemin>` sur origin/main 99f2f869).
- Le crate est autonome (table `[workspace]` vide) : la compilation du workspace en CI ne le compile
  jamais. Depuis #490, une étape CI dédiée (ubuntu, stable) lance ses tests.
- Commande : `cargo test --offline --manifest-path crates/ix-router-spike/Cargo.toml --bin jev-router`.
  Avant cette branche : 17 réussis. Avec les sept épingles ci-dessous : 24 réussis.

## Les cas de contrat de #489 face au code actuel

Les sept nouveaux tests épinglent le comportement **actuel**, manques compris. Ils ont été écrits verts
contre un code inchangé : aucun n'a été vu rouge, chacun protège donc contre une régression, pas
contre un bogue.

| Cas #489 | Comportement actuel | Preuve | Manque |
|---|---|---|---|
| mauvais modèle | `Invalid::WrongModel`, verdict KILL | existants `test_validate_rejects_contract_breaks`, `test_verdict_bands_follow_preregistration` | — |
| options non conformes | choix inconnu ou ensemble de clés ≠ options → `Contract` | existant `test_validate_rejects_contract_breaks` | — |
| scores malformés / non finis | null, négatif, > 1, chaîne → `Contract` ; le JSON ne peut pas porter NaN (`json!(f64::NAN)` vaut null) | nouveau `test_non_finite_or_out_of_range_scores_are_invalid` | — |
| preuve ambiguë | un libellé de référence `ambiguous` est routé, jamais compté (test existant) ; une **égalité** au maximum est acceptée quelle que soit l'option choisie, sans signalement | nouveau `test_argmax_tie_is_accepted_and_not_flagged` | oui |
| délai dépassé | `run-live.ps1` écrit `error`/`http_status` ; `load_receipt` ne garde que `response`, donc un délai dépassé devient `Contract("model missing")` | nouveau `test_timeout_and_provider_failure_collapse_into_one_contract_failure` | oui |
| panne du fournisseur | même `Contract("model missing")` qu'un délai dépassé : les deux sont indiscernables après chargement | même test | oui |
| usage manquant | par réponse : invalide, jamais un coût nul (existant) ; dans le total rapporté : compté pour 0 jeton | nouveau `test_usage_total_counts_a_missing_usage_as_zero` | oui |
| empreinte non conforme | `Invalid::DigestMismatch` ; une invite sans réponse est `Invalid::Missing` | nouveau `test_digest_mismatch_and_missing_answer_are_invalid` | — |
| retour en double | tout le calcul s'arrête (panic), jamais « le dernier gagne » ; un id imprévu l'arrête aussi | nouveaux `test_duplicate_receipt_id_aborts_scoring`, `test_unplanned_receipt_id_aborts_scoring` | — (fermé en cas d'échec, mais tout ou rien) |

Chaque échec est fermé : aucun ne devient un refus correct (existant
`test_invalid_response_is_never_a_correct_decline`). Ce qui manque, c'est la **nature** de l'échec :
#489 demande des issues explicites pour le délai dépassé et le fournisseur indisponible, et le spike
n'a qu'un seul panier `Contract` pour les deux.

## Ce qui quitte la machine (lu)

`request()` envoie exactement : `model` (`jev-1.13.0`), `state.user_message` (le texte de l'invite),
et une seule `questions.intent` avec `type: "choice"`, les `instructions` du fichier d'options et ses
`criteria` (id d'option → description). Rien d'autre : ni historique, ni fichier, ni identité.
L'empreinte de ces octets est épinglée dans `plan.json` et revérifiée par `run-live.ps1` avant envoi.

## L'étape live (lue, non exécutée)

`run-live.ps1` : clé lue dans l'environnement, une seconde variable d'approbation, une tentative par
requête, aucune redirection, délai de 30 s, refus d'ajouter à un ancien reçu, arrêt sur 401/403/429 et
sur un 200 sans usage. L'arrêt budgétaire est vérifié **après** chaque réponse (comme le dit #489, ce
n'est pas un plafond de dépense). Il ne s'arrête pas sur des 5xx ou des délais dépassés répétés :
chacun est consigné et la boucle continue.

## Résultats enregistrés sur les corpus figés (lu, fichiers `jev-eval.json`)

| bras | verdict | dans le périmètre | hors périmètre refusés | invalides |
|---|---|---|---|---|
| base | COMPETITIVE_WITH_HEAD | 106/110 | 16/16 | 0 |
| reversed | ROBUST | 106/110 | 16/16 | 1 |
| fr | **KILL** | 100/110 | 15/16 | 4 |
| es | ROBUST | 103/110 | 16/16 | 1 |
| fresh | ROBUST | 110/112 | 16/16 | 1 |
| full | ROBUST | 178/190 | 31/31 | 1 |
| musicse | ROBUST | 1/1 | 81/86 | 0 |

Les comptes du bras `fr` dépassent sa bande ROBUST (99, 13) ; il est KILL uniquement à cause de quatre
réponses invalides (limite 2). Ce sont les échecs de format, pas les erreurs de routage, qui ont tué
un bras : le chemin « invalide » du contrat compte autant que la précision. Les sept bras ont tourné
contre `jev-1.13.0` ; rien ici ne dit quoi que ce soit de 1.14 (indéterminé : sa documentation n'a
pas été lue).

## Points d'appui IXQL existants (lu)

- `ix-ixql` `Capability::call(CallArgs) -> Result<Produced, String>` (`capability.rs:77`), enregistrée
  par `Executor::capabilities()`. `Produced` peut porter un `Verdict`, seule voie d'entrée d'un
  verdict dans une exécution, lu par `→ when T >= 0.7:`.
- `ix-baml` `BamlOperation::invoke(&Value) -> Result<Value, BamlError>` (`lib.rs:75`), atteinte par
  une étape `baml.<nom>` (`eval.rs:708`) ; `BamlError` a trois variantes, aucune pour le délai dépassé
  ou l'abstention.
- Aucun des deux ne connaît le spike : aucun crate ne dépend d'`ix-router-spike`.

## Proposition pour l'opérateur (pas une décision)

Un nœud de décision Jev comme `Capability` IXQL (le point d'appui que #489 préfère), qui réutilise
`request`/`encode`/`validate` du spike, et dont `Produced.value` est une issue typée :
`choice | abstain | invalid | timeout | provider_unavailable | missing`, chacune avec sa provenance
(modèle, version des options, SHA-256 de la requête) et l'usage quand il est rapporté. `abstain` vaut
`__none__` seulement ; `Err(String)` reste pour les bogues de l'adaptateur. Il n'attache **aucun**
`Verdict` tiré des probabilités : un score numérique n'est pas une valeur de vérité calibrée, et un
verdict laisserait `→ when` agir dessus.

Plus petite tranche suivante, hors ligne : un adaptateur hors ligne à réponses enregistrées, sur les
reçus existants, qui rend ces six issues, avec les trois manques ci-dessus comme premiers tests rouges
(délai dépassé ≠ panne du fournisseur, égalité signalée, usage manquant compté comme coût inconnu dans
les totaux). Qu'une ligne de reçu en double ou illisible (lu : `load_receipt` panique dans les deux
cas) annule un élément ou toute l'exécution est un choix de l'opérateur. Un adaptateur live, la prise
en charge de 1.14 et toute dépense restent admis à part.
