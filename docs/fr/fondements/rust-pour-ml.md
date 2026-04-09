# Rust pour le ML

> Les patterns Rust et les crates a connaitre avant de plonger dans les algorithmes de machine learning.

## Le probleme

Vous voulez implementer des algorithmes de ML, mais Rust n'est pas Python. Il n'y a pas de NumPy, pas de `import sklearn`. En revanche, Rust vous offre quelque chose de mieux pour la production : des abstractions a cout zero, la securite memoire sans ramasse-miettes, et des performances comparables au C++.

La contrepartie ? Il faut comprendre quelques patterns specifiques a Rust avant que le code ML ne prenne sens. Ce document couvre exactement ces patterns -- rien de plus.

## Le crate essentiel : ndarray

Tous les algorithmes de ML dans ix utilisent `ndarray` -- l'equivalent Rust de NumPy. Il fournit des tableaux n-dimensionnels avec des operations element par element performantes.

### Array1 : les vecteurs

Un tableau a 1 dimension. Voyez-le comme une liste de nombres -- un point de donnees, un ensemble de poids ou un gradient.

```rust
use ndarray::Array1;

// Creer un vecteur
let v: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0]);

// Operations element par element -- fonctionnent exactement comme NumPy
let doubled = &v * 2.0;           // [2.0, 4.0, 6.0]
let sum = &v + &doubled;          // [3.0, 6.0, 9.0]
let dot_product = v.dot(&doubled); // 1*2 + 2*4 + 3*6 = 28.0

// Methodes utiles
let total: f64 = v.sum();         // 6.0
let length = v.len();             // 3
let mapped = v.mapv(|x| x * x);  // [1.0, 4.0, 9.0]
```

**Important** : remarquez les references `&`. En Rust, `&v * 2.0` emprunte `v` sans le consommer, ce qui vous permet de reutiliser `v` ensuite. C'est le systeme d'ownership de Rust qui protege vos donnees.

### Array2 : les matrices

Un tableau a 2 dimensions. Voyez-le comme un jeu de donnees ou chaque ligne est un echantillon et chaque colonne est une caracteristique.

```rust
use ndarray::{Array2, array};

// Creer une matrice 2x3 (2 echantillons, 3 caracteristiques)
let data: Array2<f64> = array![
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
];

let (rows, cols) = data.dim();     // (2, 3)
let row_0 = data.row(0);          // vue de [1.0, 2.0, 3.0]
let col_1 = data.column(1);       // vue de [2.0, 5.0]

// Transposee de la matrice
let transposed = data.t();         // matrice 3x2
```

### La macro `array!`

Le moyen le plus rapide de creer de petits tableaux pour les tests :

```rust
use ndarray::array;

let vector = array![1.0, 2.0, 3.0];           // Array1
let matrix = array![[1.0, 2.0], [3.0, 4.0]];  // Array2
```

## Les traits : comment ix organise ses algorithmes

Les traits Rust sont l'equivalent des interfaces -- ils definissent ce qu'un algorithme *sait faire*. ix utilise quelques traits cles a travers tous les crates :

### Regressor : predit un nombre

```rust
pub trait Regressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>);
    fn predict(&self, x: &Array2<f64>) -> Array1<f64>;
}
```

Tout algorithme de regression (lineaire, polynomiale, etc.) implemente ce trait. On appelle toujours `.fit()` avec les donnees d'entrainement, puis `.predict()` avec les nouvelles donnees.

### Classifier : predit une categorie

```rust
pub trait Classifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>);
    fn predict(&self, x: &Array2<f64>) -> Array1<usize>;
    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64>;
}
```

Meme schema, mais les etiquettes sont des `usize` (entiers representant des classes : 0, 1, 2, ...) et il y a un `predict_proba` supplementaire pour les estimations de probabilite.

### Clusterer : decouvre des groupes

```rust
pub trait Clusterer {
    fn fit(&mut self, x: &Array2<f64>);
    fn predict(&self, x: &Array2<f64>) -> Array1<usize>;
    fn fit_predict(&mut self, x: &Array2<f64>) -> Array1<usize>;
}
```

Pas besoin d'etiquettes -- l'algorithme decouvre la structure tout seul.

### Optimizer : trouve les meilleurs parametres

```rust
pub trait Optimizer {
    fn step(&mut self, params: &Array1<f64>, gradient: &Array1<f64>) -> Array1<f64>;
    fn name(&self) -> &str;
}
```

Prend les parametres actuels et un gradient, retourne les parametres mis a jour.

## Le pattern Builder

Beaucoup d'algorithmes ont des hyperparametres (reglages que vous choisissez avant l'entrainement). ix utilise le pattern Builder pour les configurer de maniere fluide :

```rust
use ix_optimize::ParticleSwarm;

let optimizer = ParticleSwarm::new()
    .with_particles(50)
    .with_max_iterations(1000)
    .with_bounds(-10.0, 10.0)
    .with_seed(42);
```

Ce pattern enchaine les appels `.with_*()` pour definir les options. Chaque methode retourne `Self`, ce qui permet de continuer la chaine. Toute option non specifiee utilise une valeur par defaut raisonnable.

## Generateur aleatoire avec graine pour la reproductibilite

Les algorithmes de ML utilisent souvent de l'aleatoire (initialisation aleatoire, echantillonnage aleatoire). ix accepte un parametre `seed` pour obtenir les memes resultats a chaque execution :

```rust
use ix_unsupervised::KMeans;

let mut kmeans = KMeans::new(3).with_seed(42);
// Executer deux fois avec la graine 42 donne des clusters identiques
```

En interne, cela utilise `rand::rngs::StdRng::seed_from_u64(seed)`.

## Gestion des erreurs : Result et MathError

Les operations mathematiques peuvent echouer (dimensions incompatibles, matrices singulieres). ix retourne `Result<T, MathError>` :

```rust
use ix_math::linalg;

let a = array![[1.0, 2.0], [3.0, 4.0]];
let b = array![[5.0], [6.0]];

match linalg::matmul(&a, &b) {
    Ok(result) => println!("Produit : {:?}", result),
    Err(e) => println!("Erreur : {}", e),
}
```

Dans les exemples et les experimentations rapides, vous verrez souvent `.unwrap()` qui panique en cas d'erreur -- acceptable pour l'apprentissage, mais gerez les erreurs correctement en production.

## f64 partout

ix utilise `f64` (virgule flottante 64 bits) pour tous les calculs CPU. Cela offre environ 15 chiffres significatifs de precision, ce qui est largement suffisant pour le ML. Le code GPU utilise `f32` pour les performances (les GPU sont beaucoup plus rapides en 32 bits).

## Les iterateurs : la maniere Rust de traiter les donnees

Vous verrez des chaines d'iterateurs dans tout le code. Voici un aide-memoire rapide :

```rust
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

// Map : transformer chaque element
let squared: Vec<f64> = data.iter().map(|x| x * x).collect();

// Filter : garder les elements correspondant a une condition
let big: Vec<&f64> = data.iter().filter(|&&x| x > 3.0).collect();

// Fold : reduire a une seule valeur (comme sum, mais generalise)
let sum: f64 = data.iter().fold(0.0, |acc, x| acc + x);

// Zip : apparier deux iterateurs
let a = vec![1.0, 2.0];
let b = vec![3.0, 4.0];
let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
// 1*3 + 2*4 = 11.0
```

Cout zero garanti -- le compilateur les optimise en code machine identique a une boucle ecrite a la main.

## Tout assembler

Voici un exemple complet utilisant tous ces patterns -- entrainer un modele de regression lineaire :

```rust
use ndarray::array;
use ix_supervised::{LinearRegression, Regressor};

fn main() {
    // Donnees d'entrainement : 2 caracteristiques par echantillon
    let x_train = array![
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ];
    let y_train = array![2.0, 4.0, 6.0, 8.0];

    // Creer et entrainer
    let mut model = LinearRegression::new();
    model.fit(&x_train, &y_train);

    // Predire
    let x_test = array![[5.0, 5.0], [6.0, 6.0]];
    let predictions = model.predict(&x_test);

    println!("Predictions : {:?}", predictions);
}
```

## Pour aller plus loin

Maintenant que vous connaissez les patterns Rust, commencez par les fondements mathematiques :
- **Suivant** : [Vecteurs et matrices](vecteurs-et-matrices.md) -- les objets mathematiques derriere `Array1` et `Array2`
- Ou passez directement a n'importe quel document d'algorithme -- ils utilisent tous les patterns de cette page.
