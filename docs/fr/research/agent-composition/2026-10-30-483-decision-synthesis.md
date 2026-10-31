# Synthèse #483 : les décisions que l'opérateur peut prendre maintenant, et celles qui doivent attendre

Une page sur la carte de composition d'agents [#483](https://github.com/GuitarAlchemist/ix/issues/483).
Version anglaise :
[`docs/research/agent-composition/2026-10-30-483-decision-synthesis.md`](../../../research/agent-composition/2026-10-30-483-decision-synthesis.md).
Elle résume les trois notes de recherche voisines ; les preuves sont dans ces notes, pas ici. Elle ne
décide rien : chaque ligne est une recommandation avec sa porte et sa condition de réexamen.

## Où en est chaque feuille (2026-10-30)

| Feuille | Preuves à ce jour | Ce qui manque avant une décision |
|---|---|---|
| [#484](https://github.com/GuitarAlchemist/ix/issues/484) session ou tâche | Claude Code 2.1.314 épinglé ; signal documenté par classe d'événements ; deux classes (acceptation de la requête, fin d'entrée) n'en ont aucun ; un processus par demande existe déjà deux fois (ix-loop, ix-baml) | la plus petite vérification (deux sessions, résultat tardif d'une ancienne incarnation, JSONL malformé, attente de permission, sortie) sur fixtures synthétiques ; une mesure de coût sur 2.1.314 |
| [#485](https://github.com/GuitarAlchemist/ix/issues/485) instantanés de contexte | inventaire : mémoire (`ReadContext`), disque durable (`ix-recover`, SHA-256) ; aucun document partagé avec mises à jour à version attendue ; un écrasement silencieux trouvé et corrigé (#491) | les vérifications 1, 2 et 4 de la note (instantané privé, instantané restauré, entrées manquantes/changées/incompatibles) |
| [#489](https://github.com/GuitarAlchemist/ix/issues/489) nœud de décision Jev | 24 tests hors ligne ; trois manques épinglés (délai dépassé = panne du fournisseur, égalité non signalée, usage manquant compté zéro) ; points d'appui nommés | le choix de l'opérateur sur l'ensemble des issues ; tout ce qui concerne Jev 1.14 |
| [#486](https://github.com/GuitarAlchemist/ix/issues/486) besoins, [#487](https://github.com/GuitarAlchemist/ix/issues/487) classes de trafic, [#488](https://github.com/GuitarAlchemist/ix/issues/488) parcours de falsification | pas commencées | tout ; #486 et #488 attendent aussi [#420](https://github.com/GuitarAlchemist/ix/issues/420) |

## Décisions prêtes maintenant

| # | Décision | Recommandation | Porte | Réexaminer quand |
|---|---|---|---|---|
| D1 | #484 : premier profil d'agent | **Un processus par tâche**, la forme qu'ix-loop et ix-baml utilisent déjà, avec une enveloppe JSON validée ; sessions réutilisées reportées | réversible : une session réutilisée peut s'ajouter derrière le même contrat de tâche | un coût fixe de contexte mesuré sur 2.1.314 rend les processus par tâche trop chers, ou un classifieur stream-json sur fixtures passe la vérification de #484 |
| D2 | #485 : comment une tâche reçoit son contexte | **Listes d'artefacts explicites et immuables** (`id, source, sha256, schéma, classification, octets`, format d'empreinte d'ix-recover) ; pas de ContextStore ; pas d'écriture partagée dans la première tranche | réversible : un magasin pourra servir ces listes plus tard | un second mode de stockage concret exige des mises à jour partagées, qui exigeront alors des vérifications de version attendue (aucune n'existe) |
| D3 | #489 : issue du nœud de décision | Une Capability qui rend une **issue typée** (`choice, abstain, invalid, timeout, provider_unavailable, missing`) avec provenance ; **aucun Verdict** tiré des probabilités | irréversible dès qu'un pipeline s'appuie sur les noms d'issues : la versionner (`v0`) d'ici là | un consommateur a besoin d'une valeur hors des six, ou des preuves de calibration existent pour un verdict tiré des probabilités |
| D4 | prochaine tranche de code | le classifieur hors ligne à réponses enregistrées pour #489 (les trois manques en tests rouges), dans le crate du spike, avant tout branchement IXQL | réversible (jetable) | D3 change |

## Décisions qui doivent attendre

- **#420 d'abord.** Les besoins (#486), le parcours de falsification (#488) et toute admission live
  dépendent du contrat d'autorité des actionneurs ; rien ci-dessus n'accorde ni ne suppose d'autorité.
- **#487** demande une fixture de saturation avant tout choix de transport ; aucune preuve n'a été
  réunie.
- **Les exécutions live** (un flux Claude enregistré, un appel Jev 1.14) demandent leur propre
  admission bornée : coût, clé, données qui quittent la machine.

## Vérifications de cohérence entre les feuilles

- `abstain` (D3) est une réponse, pas un Besoin : une branche qui ne peut pas décider lève un Besoin
  #486, elle ne s'abstient pas. Les deux doivent rester des sortes d'issues distinctes.
- L'enveloppe de D1 et l'issue de D3 partagent la règle que pose #484 : un enregistrement inconnu,
  malformé ou manquant n'est jamais un succès, et un usage manquant est un coût inconnu, jamais nul.
- Les listes de D2 portent ce que reçoit la tâche de D1 ; aucune ne copie la conversation d'un parent
  dans un enfant.
