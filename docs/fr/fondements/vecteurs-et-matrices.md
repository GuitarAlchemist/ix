# Vecteurs et matrices

> Les structures de donnees fondamentales du machine learning -- ce qu'elles sont, pourquoi elles comptent, et comment les utiliser en Rust.

## Le probleme

Vous avez un jeu de donnees immobilier. Chaque maison possede une surface (m2), un nombre de chambres, un age (annees) et un prix. Vous voulez qu'un ordinateur apprenne la relation entre les caracteristiques d'une maison et son prix.

Mais un ordinateur ne comprend pas les "maisons". Il comprend les nombres. Plus precisement, il comprend des *listes de nombres* et des *tableaux de nombres*. C'est exactement ce que sont les vecteurs et les matrices.

## L'intuition

### Vecteurs : une liste de nombres

Un **vecteur** est simplement une liste ordonnee de nombres. Rien de plus.

Une maison avec 3 caracteristiques devient un vecteur :
```
maison = [1500, 3, 10]
           ^     ^   ^
         m2   chambres  age
```

Pensez a un vecteur comme une fleche dans l'espace. Un vecteur 2D `[3, 4]` est une fleche pointant 3 unites a droite et 4 unites vers le haut. Un vecteur 3D ajoute la profondeur. Un vecteur ML a 100 caracteristiques est une fleche dans un espace a 100 dimensions -- impossible a visualiser, mais les mathematiques fonctionnent exactement de la meme maniere.

**Operations cles que vous verrez partout :**

- **Addition** : `[1, 2] + [3, 4] = [4, 6]` -- combine deux vecteurs element par element
- **Mise a l'echelle** : `2 * [1, 2] = [2, 4]` -- etire ou contracte un vecteur
- **Produit scalaire** : `[1, 2] . [3, 4] = 1x3 + 2x4 = 11` -- mesure a quel point deux vecteurs pointent dans la meme direction. C'est *l'operation la plus importante* en ML.
- **Norme (longueur)** : `||[3, 4]|| = sqrt(9+16) = 5` -- la magnitude du vecteur

### Matrices : un tableau de nombres

Une **matrice** est une grille rectangulaire de nombres. Imaginez plusieurs vecteurs empiles les uns sur les autres.

Votre jeu de donnees de 4 maisons devient une matrice :
```
X = | 1500  3  10 |    <- maison 1
    | 2000  4   5 |    <- maison 2
    | 1200  2  20 |    <- maison 3
    | 1800  3   8 |    <- maison 4
```

C'est une matrice 4x3 (4 lignes, 3 colonnes). Chaque ligne est un point de donnees. Chaque colonne est une caracteristique.

**Pourquoi les matrices comptent :** Presque tous les algorithmes de ML se ramenent a des operations matricielles. Regression lineaire ? Multiplication matricielle. Reseaux de neurones ? Des chaines de multiplications matricielles avec des fonctions non lineaires entre chaque etape. ACP ? Trouver des vecteurs speciaux d'une matrice.

### Multiplication matricielle

L'operation la plus importante. Quand vous multipliez une matrice par un vecteur, vous obtenez un nouveau vecteur. En clair, la multiplication matricielle *transforme* les donnees.

```
| 2  0 |     | 3 |     | 6 |
| 0  3 |  x  | 2 |  =  | 6 |
```

En clair : la matrice `[[2,0],[0,3]]` met x a l'echelle par 2 et y par 3. La multiplication matricielle est la maniere dont les modeles font leurs predictions -- la matrice contient les poids appris, et le vecteur est votre donnee d'entree.

## Fonctionnement detaille

### Operations sur les vecteurs

**Produit scalaire** (aussi appele produit interieur) :

Etant donnes deux vecteurs a = [a1, a2, ..., an] et b = [b1, b2, ..., bn] :

`a . b = a1*b1 + a2*b2 + ... + an*bn`

En clair, on multiplie les elements correspondants et on les additionne. Le resultat est un seul nombre.

Pourquoi c'est important : le produit scalaire indique a quel point deux vecteurs sont similaires. S'ils pointent dans la meme direction, le produit scalaire est grand et positif. S'ils sont perpendiculaires, il vaut zero. S'ils sont opposes, il est grand et negatif.

**Norme euclidienne** (longueur d'un vecteur) :

`||a|| = sqrt(a1^2 + a2^2 + ... + an^2)`

En clair, c'est le theoreme de Pythagore generalise a un nombre quelconque de dimensions.

### Operations sur les matrices

**Multiplication matrice-vecteur** (transforme un vecteur) :

Pour une matrice 2x2 M et un vecteur v :

```
| m11  m12 |     | v1 |     | m11*v1 + m12*v2 |
| m21  m22 |  x  | v2 |  =  | m21*v1 + m22*v2 |
```

Chaque element du resultat est le produit scalaire d'une ligne de la matrice avec le vecteur.

**Multiplication matrice-matrice** :

Pour multiplier A (m x k) par B (k x n), les dimensions interieures doivent correspondre. Le resultat est m x n. Chaque element (i,j) du resultat est le produit scalaire de la ligne i de A avec la colonne j de B.

**Transposee** (inverser lignes et colonnes) :

```
| 1  2  3 |  transposee   | 1  4 |
| 4  5  6 |  ---------->  | 2  5 |
                           | 3  6 |
```

**Determinant** (scalaire decrivant une matrice carree) :

Pour une matrice 2x2 : `det([[a,b],[c,d]]) = ad - bc`

En clair : le determinant indique de combien la matrice dilate les aires. S'il est nul, la matrice ecrase tout sur une dimension inferieure (la matrice est "singuliere" et ne peut pas etre inversee).

**Inverse** (la matrice "annulation") :

Si M x M^-1 = I (matrice identite), alors M^-1 est l'inverse de M. Toute matrice n'a pas d'inverse -- seules les matrices carrees dont le determinant est non nul en possedent une.

## En Rust

ix utilise `ndarray` pour toutes les operations sur les vecteurs et les matrices. Le crate `ix-math` ajoute des fonctions de plus haut niveau.

```rust
use ndarray::{array, Array1, Array2};
use ix_math::linalg;

// Vecteurs
let a: Array1<f64> = array![1.0, 2.0, 3.0];
let b: Array1<f64> = array![4.0, 5.0, 6.0];

// Produit scalaire
let dot = a.dot(&b);  // 1*4 + 2*5 + 3*6 = 32.0

// Norme (longueur)
let norm = a.dot(&a).sqrt();  // sqrt(1+4+9) = sqrt(14)

// Operations element par element
let sum = &a + &b;     // [5.0, 7.0, 9.0]
let scaled = &a * 2.0; // [2.0, 4.0, 6.0]

// Matrices
let m = array![[1.0, 2.0], [3.0, 4.0]];
let v = array![1.0, 1.0];

// Multiplication matrice-vecteur
let result = linalg::matvec(&m, &v).unwrap();  // [3.0, 7.0]

// Multiplication matrice-matrice
let a_mat = array![[1.0, 2.0], [3.0, 4.0]];
let b_mat = array![[5.0, 6.0], [7.0, 8.0]];
let product = linalg::matmul(&a_mat, &b_mat).unwrap();

// Transposee, determinant, inverse
let t = linalg::transpose(&m);
let det = linalg::determinant(&m).unwrap();      // 1*4 - 2*3 = -2.0
let inv = linalg::inverse(&m).unwrap();

// Matrice identite
let eye = linalg::eye(3);  // identite 3x3

// Moyennes par colonne et standardisation
let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
let means = linalg::col_mean(&data);                          // [2.0, 20.0]
let (standardized, means, stds) = linalg::standardize(&data); // moyenne nulle, variance unitaire
```

## Quand les utiliser

Vous ne choisissez pas d'utiliser les vecteurs et les matrices -- c'est la representation par defaut de tout en ML :

| Type de donnees | Representation |
|----------------|----------------|
| Un point de donnees unique | Vecteur (`Array1`) |
| Un jeu de donnees | Matrice (`Array2`) -- lignes = echantillons, colonnes = caracteristiques |
| Poids du modele | Vecteur ou matrice |
| Predictions | Vecteur |
| Transformation (rotation, mise a l'echelle) | Matrice |

## Parametres cles

| Operation | Regle de dimension | Echoue quand |
|-----------|--------------------|--------------|
| `a + b` (vecteurs) | Meme longueur | Longueurs differentes |
| `a . b` (produit scalaire) | Meme longueur | Longueurs differentes |
| `M x v` (matrice-vecteur) | M est m x n, v a n elements | Nombre de colonnes != longueur du vecteur |
| `A x B` (matrice-matrice) | A est m x k, B est k x n | Colonnes de A != lignes de B |
| `det(M)` | M doit etre carree | Matrice non carree |
| `M^-1` | M doit etre carree | Matrice singuliere (det = 0) |

## Pieges courants

- **Les incompatibilites de dimensions** sont la source d'erreur numero 1. Verifiez toujours les formes de vos tableaux avec `.dim()` ou `.shape()`.
- **Conventions lignes vs colonnes** : dans ix, les jeux de donnees sont toujours lignes=echantillons, colonnes=caracteristiques. Certains manuels utilisent la convention transposee.
- **Matrices singulieres** : si vous obtenez une erreur "matrix is singular", vos donnees comportent probablement des caracteristiques lineairement dependantes (une caracteristique est un multiple d'une autre). Essayez de supprimer les caracteristiques redondantes ou d'utiliser la regularisation.
- **Precision numerique** : `f64` offre environ 15 chiffres significatifs. Pour la plupart des usages en ML, c'est largement suffisant. Mais si vous inversez de grandes matrices, les petites erreurs peuvent s'accumuler -- preferez les algorithmes qui evitent l'inversion explicite quand c'est possible.

## Pour aller plus loin

- **Suivant** : [Probabilites et statistiques](probabilites-et-statistiques.md) -- le langage mathematique de l'incertitude
- **Voir aussi** : [Rust pour le ML](rust-pour-ml.md) pour les patterns `ndarray`
- **Approfondissement** : [ACP](../apprentissage-non-supervise/acp.md) utilise la decomposition en valeurs propres de la matrice de covariance
