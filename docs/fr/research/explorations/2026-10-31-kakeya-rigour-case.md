# Kakeya dans ℝ³, cas pédagogique et test de rigueur agentique

Note d'exploration, faible priorité (derrière le raccord WMUX). Elle ne lance aucun chantier moteur et
n'annonce aucun gain audio ou RTX. Version anglaise :
[`docs/research/explorations/2026-10-31-kakeya-rigour-case.md`](../../../research/explorations/2026-10-31-kakeya-rigour-case.md).

Statuts : **lu à la source** (vérifié ici sur la page citée), **rappel** (résultat classique, non
revérifié dans cette note), **ouvert**.

## Pourquoi IX s'y intéresse

Le résultat teste proprement trois distinctions qu'un agent brouille facilement : dimension et mesure,
simulation et preuve, résultat établi et extrapolation. L'exercice proposé : un agent explique, un
adversaire cherche les erreurs, puis chaque affirmation est vérifiée contre les sources ci-dessous.

## Questions, réponses, pièges

**Q1. Qu'est-ce qui a été démontré exactement ?** Tout ensemble de Kakeya dans ℝ³ a une dimension de
Hausdorff et de Minkowski égale à 3. *Lu à la source :* H. Wang, J. Zahl, « Volume estimates for
unions of convex sets, and the Kakeya set conjecture in three dimensions »,
[arXiv:2502.17655](https://arxiv.org/abs/2502.17655), v1, 24 février 2025, résumé. *Piège :* la page
arXiv n'affiche qu'une version et aucune référence de revue : dire « prépublication », pas « relu par
les pairs », sauf source ultérieure.

**Q2. Dimension 3 veut-il dire volume positif ?** Non. *Rappel :* Besicovitch a construit des
ensembles de Kakeya de mesure de Lebesgue nulle, dans le plan et dans tout ℝⁿ, n ≥ 2. Un tel ensemble
dans ℝ³ a un volume nul et, par Q1, une dimension 3. *Piège :* lire « dimension pleine » comme
« remplit l'espace ».

**Q3. Que montre l'image des segments épaissis ?** On épaissit chaque segment unité en un tube de
rayon δ. Pour un ensemble de Kakeya E de mesure nulle, le volume du δ-voisinage tend vers 0 quand
δ → 0 ; la dimension 3 dit qu'il y tend plus lentement que toute puissance δ^ε. *Piège :* une
animation parcourt une plage finie de δ, où δ^0,1 et une constante se distinguent mal : elle ne
sépare pas la dimension 3 de 2,9, elle illustre et ne démontre jamais.

**Q4. Et le plan, et les dimensions supérieures ?** *Rappel :* dans ℝ², tout ensemble de Kakeya est
de dimension 2 (Davies, 1971). Pour n ≥ 4, la conjecture est **ouverte** d'après les sources de cette
note. *Piège :* extrapoler « 3 dans ℝ³ » en « n dans ℝⁿ ».

**Q5. Quelle route suit le résumé lui-même ?** *Lu à la source :* une estimation du volume des
réunions de δ-tubes dans ℝ³ quand peu de tubes tiennent dans un même convexe ; le résultat de
dimension en est énoncé comme conséquence. *Piège :* résumer la preuve au-delà du résumé sans avoir
lu les 127 pages.

## La liste de l'adversaire

- Chaque nombre ou date porte une source ou le mot *rappel*.
- « Dimension » nomme toujours Hausdorff ou Minkowski ; « mesure » ou « volume » ne la remplace jamais.
- Une image ou une simulation s'appelle une illustration.
- « Démontré » s'applique à ℝ² (rappel) et à ℝ³ (prépublication) ; rien n'est dit démontré pour n ≥ 4.

## Décision que cette note prépare (non prise)

Savoir si une visualisation Streeling (segments épaissis, changements d'échelle) mérite du travail. À
réexaminer seulement si les questions ci-dessus passent un tour explicateur/adversaire sans affirmation
non sourcée.
