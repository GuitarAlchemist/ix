---
title: "Service Windows IX : modules rechargeables plus tard, en admissions versionnées avec un drainage borné (pas de hot reload)"
type: decision
status: proposé ; rien n'est construit ; le premier lot du service (feat/ix-windows-service) n'en dépend pas
date: 2026-10-30
branch: feat/ix-windows-service
reversibility: réversible (aucun code, aucun champ de contrat) ; devient irréversible quand un champ de version entre dans l'identité d'exécution sur le fil ou dans le schéma de la spec
revisit-trigger: la première charge hébergée sous le service qui doit changer de version sans arrêter le plan (ix-loop comme charge supervisée) ; ou la signature des ADR-0016 à 0018 (pull request #481), qui fixent la version d'un module durable
---

# Service Windows IX : modules rechargeables plus tard

English: [`docs/plans/2026-10-30-ixctl-service-reloadable-modules.md`](../../plans/2026-10-30-ixctl-service-reloadable-modules.md).
Référence utilisateur : [`go/ixctl/README.fr.md`](../../../go/ixctl/README.fr.md#service-windows).

## Le problème

Le brief du service Windows IX (étape 6) demande comment le service prendra plus tard une
nouvelle version d'un module sans être arrêté : le travail en cours reste sur sa version, les
nouvelles admissions vont à la nouvelle, l'ancienne se draine, et un retour arrière reste
possible. Il demande aussi de ne pas promettre un hot reload complet. Cette note fixe la forme
pour que le premier lot ne la ferme pas ; rien ici n'est construit.

## Ce qui existe aujourd'hui (go/ixctl, main à 99f2f869)

- Un plan lit sa spec une fois, au démarrage (`LoadSpec`). La commande, les arguments et le
  dossier d'un service sont fixés pour toute la vie du plan. L'état souhaité dit seulement
  `running` ou `stopped` par service.
- Chaque exécution porte déjà l'apply qui l'a lancée : `service.started` nomme sa `generation`,
  et un redémarrage la garde. Les générations forment un ordre total (`applyMu`).
- Les admissions se ferment à l'arrêt : une fois l'arrêt reçu, un apply répond 503.
- Un arrêt est borné et distingue demandé, reçu, puis observé ou non observé ; un arbre de
  processus appartient au plan par son Job Object et meurt avec l'hôte.
- Remplacer ixctl lui-même, c'est arrêter puis démarrer le service : une nouvelle époque à la
  génération 0, et l'état souhaité n'est pas rejoué (ADR-0014).
- Le magasin d'ix-loop accepte une proposition comme un fichier ; il n'a pas de version de script.

## Forme proposée

1. **La version d'un module fait partie de la spec, pas du processus.** Un service supervisé
   reçoit une `version` et l'empreinte de ce qu'il exécute, écrites `nom@version` comme les types
   de sonde le sont déjà. Un module durable garde l'identité de l'ADR-0018 (SemVer, un verrou,
   une clôture) ; le service ne résout pas de versions, il supervise ce qu'un verrou nomme.
2. **Une exécution garde sa version.** L'identité d'exécution reçoit la version avec laquelle
   elle a été lancée, comme elle porte déjà sa génération. L'ADR-0018 pose la même règle pour une
   exécution durable : la reprise suivante rejoue sur les anciens octets.
3. **Charger une version, c'est basculer les admissions.** Un `load` nomme la nouvelle version ;
   dès cet instant, tout nouveau démarrage et toute nouvelle exécution vont vers elle. Les
   exécutions de l'ancienne version ne sont pas touchées.
4. **L'ancienne version se draine, dans une borne.** Aucune admission ne l'atteint ; ses
   exécutions finissent d'elles-mêmes ou à la borne de drainage, puis s'arrêtent avec les mêmes
   trois moments et la même issue non observée qu'un arrêt de service. Un effet qu'une exécution
   tuée a pu produire n'est pas rejoué : dans une exécution durable, il reste `UNKNOWN` jusqu'à sa
   réconciliation (ADR-0016).
5. **Le retour arrière est un autre chargement.** Charger la version précédente ramène les
   admissions vers elle. Il faut ses octets : le plan garde donc une version tant qu'une
   exécution la nomme, plus celle qui précède la version courante. Le retour arrière n'annule
   aucun effet produit par une exécution de la version plus récente.

## Non promis

- Remplacer du code dans un processus qui tourne. Un changement de version termine les anciennes
  exécutions ; il ne les modifie jamais.
- Recharger ixctl lui-même : c'est un redémarrage du service, avec la perte de l'état souhaité de
  l'ADR-0014.
- Migrer l'état d'une exécution d'une version à l'autre, ou lire un ancien journal avec du code
  nouveau.

## Questions ouvertes pour Stef

- Où vivent les versions sur le disque : un fichier de spec par version, ou une spec qui en nomme
  plusieurs.
- Combien d'anciennes versions le plan garde au-delà de la règle du point 5.
