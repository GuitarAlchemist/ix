# `ix_voicing_similarity` — plus proches voisins harmoniques sur le corpus de voicings

Une démo exécutable de **recherche par similarité** sur le banc d'analyse DuckDB, dotée
d'un **oracle de correction** intégré. Compagnon de
[`maillage-voicings.md`](maillage-voicings.md) (qui fait de la *corrélation/du
regroupement*) ; celle-ci fait de la *recherche du plus proche voisin + détection
d'aberrants*.

> _English version: [`docs/walkthroughs/voicing-similarity.md`](../../walkthroughs/voicing-similarity.md)._

```bash
cargo run -p ix-duck --example ix_voicing_similarity --features duck
```

## La question

> **Étant donné la set-class d'un accord, quels sont ses plus proches voisins
> harmoniques par contenu intervallique — et quelles set-classes du répertoire réel sont
> harmoniquement isolées (sans voisin proche) ?**

Chaque set-class de Forte présente dans le corpus est un point. La distance est
**`ix_icv_l1(a, b)`** — la distance L1 entre les **vecteurs de classes d'intervalles**
(ICV) des deux PC-sets (le coût harmonique de Grothendieck). Une seule auto-jointure
calcule la table N×N complète des distances harmoniques sur le banc ; `ix_forte_number`
annote et `any_value(midiNotes)` choisit un représentant (l'ICV est un invariant de la
set-class : n'importe quel voicing d'une set-class donne le même vecteur).

## Plus proches voisins

```text
 3-11 → 2-3 (d=2), 2-4 (d=2), 2-5 (d=2), 3-3 (d=2), 3-4 (d=2)
 4-27 → 4-12 (d=2), 4-13 (d=2), 4-18 (d=2), 4-26 (d=2), 4-Z15 (d=2)
4-Z15 → 4-Z29 (d=0), 4-11 (d=2), 4-12 (d=2), 4-13 (d=2), 4-14 (d=2)
```

Le point fort est `4-Z15` : son **plus proche voisin est `4-Z29` à distance 0**. Ce sont
les deux **tétracordes tous-intervalles** — des accords distincts au contenu
intervallique *identique*. La recherche les fait remonter automatiquement, ce que la
validation formalise précisément.

## Validation — l'oracle de la relation Z

C'est l'analogue du modèle nul du maillage, mais adapté à une démo de *recherche* : au
lieu d'un test de significativité, un **oracle de correction fondé sur une vérité
terrain**.

Deux set-classes *distinctes* ont un ICV identique **si et seulement si** elles sont
**Z-reliées** (même cardinalité + même vecteur d'intervalles, forme première différente).
Donc **toute paire à distance harmonique 0 doit vérifier `ix_z_related`** — sinon
`ix_icv_l1` serait infidèle. La démo le vérifie contre le corpus :

```text
validation — 19 paires de set-classes distinctes à distance harmonique 0 :
   4-Z15 ≡ 4-Z29,  5-Z12 ≡ 5-Z36,  6-Z29 ≡ 6-Z50,  … (toutes les paires Z hexacordales)
   ✅ ORACLE PASS : toute paire à distance 0 est ix_z_related
```

Le compte est lui-même une **correspondance avec la théorie** : le 12-TET compte
exactement **19 paires Z-reliées** (1 tétracorde + 3 pentacordes + 15 hexacordes). La
démo récupère l'ensemble *complet* depuis le corpus de guitare — donc chaque set-class
Z-reliée apparaît bel et bien dans de vrais doigtés — et confirme que `ix_icv_l1` et
`ix_z_related` concordent sur toutes. C'est à la fois un contrôle externe (les 19, et la
fameuse `4-Z15/4-Z29`) et un contrôle de cohérence inter-UDF.

## Isolation harmonique

Parmi les set-classes assez courantes pour être « vraiment utilisées » (≥ 500 voicings),
les plus harmoniquement **isolées** — plus grande distance au plus proche voisin — sont :

```text
 5-7 : nn = 4  (10 546 voicings)      5-33 : nn = 4  (4 241 voicings)
5-31 : nn = 4  ( 7 994 voicings)      4-28 : nn = 3  (1 749 voicings)
5-15 : nn = 4  ( 4 952 voicings)
```

Les pentacordes `5-7`, `5-31`, `5-15`, `5-33` sont les plus éloignés de tout le reste par
contenu intervallique tout en apparaissant des milliers de fois — des accords
harmoniquement distinctifs que les guitaristes emploient malgré tout.

## Portée et réserves

- **Consultatif uniquement** (comme toutes les lentilles `ix_duck` ; cf. ADR-0002 /
  ADR-0004) — pas une barrière.
- L'espace ICV est **dense et quantifié** : les distances sont de petits entiers (nn ∈
  {2, 3, 4} ici), donc « l'isolation » ne différencie que faiblement. L'oracle (distance
  0) est exact ; le classement par isolation est un signal doux, pas une partition nette.
- La métrique est purement harmonique (contenu intervallique) ; elle ignore le
  voicing/registre/jouabilité. À combiner avec les axes position/écart de
  `ix_voicing_mesh` pour la vue manche.
- Le corpus complet est exclu de git (110 Mo) ; sur un dépôt fraîchement cloné, la démo
  bascule sur l'échantillon suivi de 500 voicings (moins de set-classes, donc moins de
  paires Z — l'oracle tient toujours).
