# Probabilites et statistiques

> Le langage mathematique de l'incertitude -- comment decrire, mesurer et raisonner sur les donnees.

## Le probleme

Vous construisez un filtre anti-spam. Sur 10 000 emails, 2 000 sont des spams. Un nouvel email arrive contenant le mot "loterie". Quelle est la probabilite qu'il soit un spam ?

Impossible de repondre sans probabilites et statistiques. Tout algorithme de ML soit *utilise* directement les probabilites (Naive Bayes, regression logistique, bandits), soit *s'appuie sur les statistiques* pour evaluer ses performances (erreur moyenne, variance, intervalles de confiance). Vous avez besoin des deux.

## L'intuition

### Statistiques : decrire ce que l'on a

Les statistiques repondent a la question : "A quoi ressemblent mes donnees ?"

Imaginez que vous mesuriez la taille de 100 personnes. Vous ne pouvez pas retenir les 100 valeurs, alors vous resumez :

- **Moyenne** : Le "centre" de vos donnees. Si les tailles sont [170, 175, 180], la moyenne est 175.
- **Variance** : A quel point les donnees sont dispersees. Si tout le monde mesure 175 cm, la variance est 0. Si les tailles vont de 150 a 200, la variance est grande.
- **Ecart-type** : La racine carree de la variance. Il est dans les memes unites que vos donnees, donc plus intuitif. "Les tailles varient d'environ 10 cm" est plus parlant que "la variance est 100."
- **Mediane** : La valeur du milieu une fois les donnees triees. Moins sensible aux valeurs aberrantes que la moyenne -- une personne de 300 cm ne fausse pas votre resume.

### Probabilites : raisonner sur ce qui pourrait arriver

Les probabilites repondent a la question : "Quelle est la vraisemblance de cet evenement ?"

- **P(spam) = 0.2** signifie que 20 % des emails sont des spams (d'apres votre jeu de donnees).
- **P(loterie | spam) = 0.8** signifie que 80 % des spams contiennent le mot "loterie".
- **P(loterie | non spam) = 0.01** signifie que seulement 1 % des emails legitimes contiennent "loterie".

### Le theoreme de Bayes : retourner la question

Vous connaissez P(loterie | spam) -- la probabilite de voir "loterie" *sachant que* l'email est un spam. Mais vous voulez P(spam | loterie) -- la probabilite que l'email soit un spam *sachant que* vous voyez "loterie".

Le theoreme de Bayes retourne la question :

```
P(spam | loterie) = P(loterie | spam) x P(spam) / P(loterie)
```

En clair : "Mettez a jour votre croyance initiale (P(spam) = 0.2) en utilisant la nouvelle evidence (le mot 'loterie')."

En deroulant les calculs :
- P(loterie) = P(loterie|spam) x P(spam) + P(loterie|non spam) x P(non spam)
- P(loterie) = 0.8 x 0.2 + 0.01 x 0.8 = 0.168
- P(spam | loterie) = 0.8 x 0.2 / 0.168 = 0.952

Cet email a 95.2 % de chances d'etre un spam. Le theoreme de Bayes est le fondement des classifieurs Naive Bayes, de l'inference bayesienne et du raisonnement probabiliste dans tout le ML.

## Fonctionnement detaille

### Statistiques descriptives

**Moyenne** (esperance) :

`mu = (1/n) x somme(xi)`

En clair : additionner toutes les valeurs et diviser par leur nombre.

**Variance** (population) :

`sigma^2 = (1/n) x somme((xi - mu)^2)`

En clair : a quel point chaque valeur s'ecarte de la moyenne, en moyenne. Elever au carre garantit que les ecarts positifs et negatifs ne s'annulent pas.

**Variance d'echantillon** (correction de Bessel) :

`s^2 = (1/(n-1)) x somme((xi - mu)^2)`

En clair : lorsque l'on estime la variance a partir d'un echantillon (et non de la population entiere), on divise par n-1 au lieu de n. Cela corrige un biais subtil -- les echantillons tendent a sous-estimer la dispersion reelle.

**Ecart-type** :

`sigma = sqrt(sigma^2)`

En clair : la distance "typique" par rapport a la moyenne, dans les unites d'origine.

**Covariance** (entre deux variables) :

`cov(X, Y) = (1/n) x somme((xi - mu_x)(yi - mu_y))`

En clair : est-ce que X et Y evoluent ensemble ? Une covariance positive signifie que quand X augmente, Y tend a augmenter aussi. Negative signifie qu'ils evoluent en sens oppose.

**Correlation** (covariance normalisee) :

`cor(X, Y) = cov(X, Y) / (sigma_x x sigma_y)`

En clair : la covariance ramenee a l'intervalle [-1, 1]. +1 signifie une relation positive parfaite, -1 une relation negative parfaite, 0 aucune relation lineaire.

### Distributions de probabilite

Une distribution decrit quelles valeurs une variable aleatoire peut prendre et avec quelle vraisemblance.

**Distribution normale (gaussienne)** : La courbe en cloche. Decrite par la moyenne mu et l'ecart-type sigma. Beaucoup de phenomenes naturels sont approximativement normaux (tailles, erreurs de mesure). Theoreme central limite : les moyennes de nombreuses variables aleatoires tendent vers la loi normale.

**Distribution uniforme** : Toutes les valeurs sont equiprobables. Lancer un de equilibre. Initialisation aleatoire des poids d'un modele.

**Distribution de Bernoulli** : Deux issues possibles (pile/face, spam/non spam). Decrite par la probabilite p de succes.

## En Rust

ix fournit tout cela dans `ix-math` :

```rust
use ndarray::array;
use ix_math::stats;

// Statistiques de base
let data = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

let avg = stats::mean(&data).unwrap();             // 5.0
let var = stats::variance(&data).unwrap();          // 4.0 (population)
let svar = stats::sample_variance(&data).unwrap();  // 4.571 (Bessel)
let std = stats::std_dev(&data).unwrap();           // 2.0
let med = stats::median(&data).unwrap();            // 4.5
let (lo, hi) = stats::min_max(&data).unwrap();     // (2.0, 9.0)

// Matrices de covariance et de correlation
// Lignes = observations, colonnes = variables
let dataset = array![
    [1.0, 2.0],
    [2.0, 4.0],
    [3.0, 6.0],
    [4.0, 8.0],
];

let cov = stats::covariance_matrix(&dataset).unwrap();
// Forte covariance positive -- les deux variables evoluent ensemble

let cor = stats::correlation_matrix(&dataset).unwrap();
// cor[0][1] ~ 1.0 -- correlation positive parfaite (y = 2x)
```

### Standardisation

Beaucoup d'algorithmes de ML fonctionnent mieux quand les caracteristiques sont sur la meme echelle. La standardisation transforme chaque caracteristique pour avoir une moyenne de 0 et un ecart-type de 1 :

```rust
use ix_math::linalg;

let data = array![
    [100.0, 0.1],   // Caract. 1 est grande, caract. 2 est minuscule
    [200.0, 0.2],
    [300.0, 0.3],
];

let (standardized, means, stds) = linalg::standardize(&data);
// Maintenant les deux caracteristiques ont une moyenne ~ 0, ecart-type ~ 1
// Des algorithmes comme KNN et SVM en ont besoin pour fonctionner correctement
```

## Quand les utiliser

| Situation | Ce dont vous avez besoin |
|-----------|--------------------------|
| Resumer un jeu de donnees | Moyenne, variance, ecart-type, mediane |
| Verifier si des caracteristiques sont liees | Matrice de covariance/correlation |
| Normaliser les caracteristiques avant l'entrainement | Standardiser (moyenne nulle, variance unitaire) |
| Construire un classifieur probabiliste | Theoreme de Bayes (Naive Bayes) |
| Evaluer l'incertitude du modele | Distributions de probabilite, intervalles de confiance |
| Comparer les performances de deux modeles | Tests statistiques (pas encore dans ix) |

## Parametres cles

| Statistique | Ce qu'elle vous dit | Attention |
|-------------|--------------------|-----------| 
| Moyenne | Centre des donnees | Sensible aux valeurs aberrantes |
| Mediane | Valeur du milieu | Robuste aux valeurs aberrantes, mais ignore la dispersion |
| Variance | Dispersion des donnees | Unites au carre (utilisez l'ecart-type pour interpreter) |
| Ecart-type | Distance typique a la moyenne | Suppose une dispersion symetrique |
| Correlation | Force de la relation lineaire | Ne signifie PAS causalite ; ne detecte pas les relations non lineaires |

## Pieges courants

- **Correlation != causalite.** Deux variables peuvent etre parfaitement correlees sans que l'une cause l'autre. Les ventes de glaces et les noyades augmentent toutes les deux en ete.
- **Moyenne vs mediane.** Si vos donnees contiennent des valeurs aberrantes (une maison a 50 M EUR dans un quartier a 300 K EUR), la mediane est un meilleur resume que la moyenne.
- **Variance de population vs d'echantillon.** Utilisez `sample_variance` (denominateur n-1) quand vous travaillez avec un echantillon d'une population plus grande. Utilisez `variance` (denominateur n) quand vous avez la population entiere.
- **Standardisez avant les algorithmes bases sur la distance.** KNN, SVM et K-Means mesurent tous des distances entre points. Si une caracteristique va de 0 a 1000 et une autre de 0 a 1, la premiere domine. Standardisez d'abord.

## Pour aller plus loin

- **Suivant** : [Intuition du calcul differentiel](intuition-calcul.md) -- comment les gradients guident l'optimisation
- **Utilise ceci** : [Naive Bayes](../apprentissage-supervise/naive-bayes.md) utilise directement le theoreme de Bayes
- **Utilise ceci** : [ACP](../apprentissage-non-supervise/acp.md) utilise les matrices de covariance pour trouver les composantes principales
