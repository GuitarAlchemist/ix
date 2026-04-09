# De zero a hero : Rust pour le machine learning

> Tout ce qu'il faut savoir sur Rust pour travailler avec ix -- du debutant absolu au developpeur ML productif.

Ce guide ne suppose aucune experience prealable en Rust. Il couvre exactement le Rust dont vous avez besoin pour le developpement ML, en laissant de cote les fonctionnalites du langage que vous n'utiliserez pas.

---

## 1. Variables et types

Les variables Rust sont immuables par defaut. Ajoutez `mut` pour les rendre mutables.

```rust
let x = 5;           // Immuable -- impossible a modifier
let mut y = 10;      // Mutable -- on peut reassigner
y = 20;              // OK

// Annotations de type (generalement optionnelles -- Rust infere les types)
let count: usize = 42;     // Entier non signe (utilise pour les indices, les compteurs)
let price: f64 = 19.99;    // Flottant 64 bits (tout le calcul ML dans ix)
let flag: bool = true;      // Booleen
let name: &str = "hello";   // Tranche de chaine (texte emprunte)
```

### Les types que vous verrez en ML

| Type | Ce que c'est | Ou vous le verrez |
|------|-------------|-------------------|
| `f64` | Flottant 64 bits | Tous les calculs CPU, poids, predictions |
| `f32` | Flottant 32 bits | Calculs GPU (plus rapide sur GPU) |
| `usize` | Entier non signe (taille de pointeur) | Indices de tableau, etiquettes de classe, compteurs |
| `bool` | Vrai/faux | Drapeaux, verifications de convergence |
| `Vec<f64>` | Liste extensible de flottants | Donnees brutes avant conversion en ndarray |
| `Array1<f64>` | ndarray 1D | Vecteurs, predictions, etiquettes |
| `Array2<f64>` | ndarray 2D | Jeux de donnees, matrices de poids |
| `Option<T>` | Valeur qui pourrait ne pas exister | Poids du modele avant l'entrainement |
| `Result<T, E>` | Valeur qui pourrait etre une erreur | Operations mathematiques faillibles |

## 2. Ownership et emprunt

C'est la fonctionnalite la plus distinctive de Rust. Elle previent les bugs memoire a la compilation.

**Les regles :**
1. Chaque valeur a exactement un **proprietaire**
2. Quand le proprietaire sort de la portee, la valeur est liberee
3. On peut **emprunter** une valeur sans en prendre la propriete

```rust
let data = vec![1.0, 2.0, 3.0];

// Emprunt immuable (&) -- lecture seule
let sum: f64 = data.iter().sum();  // emprunte data
println!("{:?}", data);             // data toujours utilisable

// Emprunt mutable (&mut) -- modification possible
let mut data = vec![1.0, 2.0, 3.0];
data.push(4.0);                     // modifie data

// Transfert d'ownership -- la variable originale disparait
let data = vec![1.0, 2.0, 3.0];
let data2 = data;                   // data transfere a data2
// println!("{:?}", data);          // ERREUR : data a ete deplace
```

### Pourquoi c'est important en ML

Quand vous voyez `&` dans les signatures de fonctions, cela signifie "j'emprunte ceci, je ne le consomme pas" :

```rust
// Cette fonction emprunte les tableaux -- vous pouvez toujours les utiliser apres
fn dot_product(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.dot(b)
}

let weights = array![1.0, 2.0, 3.0];
let input = array![4.0, 5.0, 6.0];
let result = dot_product(&weights, &input);  // emprunt avec &
// weights et input sont toujours utilisables ici
```

**Dans ix** : la plupart des fonctions prennent des references `&`. Le pattern `.fit(&mut self, x, y)` emprunte les donnees et mute le modele.

## 3. Structures et methodes

Les structures sont la maniere de Rust de regrouper des donnees. Les methodes sont des fonctions attachees a une structure.

```rust
struct LinearModel {
    weights: Vec<f64>,
    bias: f64,
}

impl LinearModel {
    // Constructeur (par convention appele `new`)
    fn new(n_features: usize) -> Self {
        LinearModel {
            weights: vec![0.0; n_features],
            bias: 0.0,
        }
    }

    // Methode qui emprunte self (lecture seule)
    fn predict(&self, x: &[f64]) -> f64 {
        let dot: f64 = self.weights.iter()
            .zip(x.iter())
            .map(|(w, xi)| w * xi)
            .sum();
        dot + self.bias
    }

    // Methode qui emprunte self de maniere mutable (peut modifier)
    fn update_bias(&mut self, new_bias: f64) {
        self.bias = new_bias;
    }
}

let mut model = LinearModel::new(3);
let prediction = model.predict(&[1.0, 2.0, 3.0]);
model.update_bias(0.5);
```

### Le pattern Builder

Beaucoup d'algorithmes ix l'utilisent pour la configuration :

```rust
let optimizer = ParticleSwarm::new()     // Commencer avec les valeurs par defaut
    .with_particles(50)                   // Enchainer la configuration
    .with_max_iterations(1000)
    .with_bounds(-10.0, 10.0)
    .with_seed(42);                       // Chacun retourne Self
```

Chaque methode `.with_*()` prend `mut self` et retourne `Self`, ce qui permet l'enchainement.

## 4. Les traits (interfaces)

Les traits definissent un comportement partage. Si vous connaissez les interfaces (Java) ou les protocoles (Swift), les traits sont similaires.

```rust
// ix definit des traits comme :
trait Classifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>);
    fn predict(&self, x: &Array2<f64>) -> Array1<usize>;
    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64>;
}

// Plusieurs algorithmes implementent le meme trait :
// KNN, DecisionTree, LogisticRegression, LinearSVM, GaussianNaiveBayes
// Tous ont .fit() et .predict() -- meme interface, algorithmes differents
```

**Pourquoi c'est important** : vous pouvez ecrire du code qui fonctionne avec *n'importe quel* classifieur :

```rust
fn evaluate<C: Classifier>(model: &C, test_x: &Array2<f64>, test_y: &Array1<usize>) -> f64 {
    let predictions = model.predict(test_x);
    metrics::accuracy(test_y, &predictions)
}

// Fonctionne avec n'importe quel classifieur
evaluate(&knn, &test_x, &test_y);
evaluate(&tree, &test_x, &test_y);
evaluate(&svm, &test_x, &test_y);
```

## 5. Option et Result (gestion des erreurs)

### Option : quelque chose pourrait ne pas exister

```rust
let mut model = LinearRegression::new();
// Avant l'entrainement, les poids n'existent pas
assert!(model.weights.is_none());

model.fit(&x, &y);
// Apres l'entrainement, les poids existent
if let Some(w) = &model.weights {
    println!("Poids : {:?}", w);
}
```

`Option<T>` est soit `Some(valeur)` soit `None`. Utilisez-le pour les valeurs qui pourraient ne pas encore etre definies (comme les poids d'un modele avant l'entrainement).

### Result : quelque chose pourrait echouer

```rust
use ix_math::linalg;

// La multiplication matricielle peut echouer (dimensions incompatibles)
match linalg::matmul(&a, &b) {
    Ok(product) => println!("Resultat : {:?}", product),
    Err(e) => println!("Erreur : {}", e),
}

// Raccourci : .unwrap() panique en cas d'erreur (acceptable pour les exemples, pas en production)
let product = linalg::matmul(&a, &b).unwrap();

// Raccourci : ? propage l'erreur a l'appelant
fn compute(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, MathError> {
    let product = linalg::matmul(a, b)?;  // Retourne Err immediatement en cas d'echec
    Ok(product)
}
```

**Dans ix** : les fonctions mathematiques retournent `Result`. Dans les exemples, vous verrez `.unwrap()` partout -- en production, gerez les erreurs correctement.

## 6. Les iterateurs

Les iterateurs Rust sont des abstractions a cout zero -- ils se compilent en code identique a une boucle ecrite a la main.

```rust
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

// Map : transformer chaque element
let squared: Vec<f64> = data.iter().map(|x| x * x).collect();
// [1.0, 4.0, 9.0, 16.0, 25.0]

// Filter : garder les elements correspondants
let big: Vec<&f64> = data.iter().filter(|&&x| x > 3.0).collect();
// [4.0, 5.0]

// Sum
let total: f64 = data.iter().sum();
// 15.0

// Enumerate : obtenir l'indice + la valeur
for (i, val) in data.iter().enumerate() {
    println!("Indice {} : {}", i, val);
}

// Zip : apparier deux iterateurs
let a = vec![1.0, 2.0, 3.0];
let b = vec![4.0, 5.0, 6.0];
let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
// 1*4 + 2*5 + 3*6 = 32.0

// Enchainer les operations
let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
let variance: f64 = data.iter()
    .map(|x| (x - mean).powi(2))
    .sum::<f64>() / data.len() as f64;
```

### Patterns d'iterateurs courants en ML

```rust
// Trouver l'argmax (indice de la valeur maximale)
let scores = vec![0.1, 0.7, 0.2];
let (best_idx, best_val) = scores.iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .unwrap();
// best_idx = 1, best_val = 0.7

// Normaliser en probabilites
let sum: f64 = scores.iter().sum();
let probs: Vec<f64> = scores.iter().map(|x| x / sum).collect();

// Calculer les entrees de la matrice de confusion
let true_positives = y_true.iter().zip(y_pred.iter())
    .filter(|(&t, &p)| t == 1 && p == 1)
    .count();
```

## 7. Les closures

Les closures sont des fonctions anonymes. Vous les utiliserez constamment avec les iterateurs et comme fonctions objectif pour les optimiseurs.

```rust
// Closure simple
let square = |x: f64| x * x;
println!("{}", square(3.0));  // 9.0

// Closure capturant une variable
let threshold = 0.5;
let is_positive = |x: f64| x > threshold;  // capture `threshold`

// Closures comme arguments de fonction (tres courant en ML)
let objective = |x: &Array1<f64>| -> f64 {
    // Fonction de Rosenbrock -- test classique d'optimisation
    let mut sum = 0.0;
    for i in 0..x.len()-1 {
        sum += 100.0 * (x[i+1] - x[i]*x[i]).powi(2) + (1.0 - x[i]).powi(2);
    }
    sum
};
```

**Dans ix** : `ClosureObjective` encapsule une closure dans un `ObjectiveFunction` :

```rust
use ix_optimize::ClosureObjective;

let objective = ClosureObjective {
    f: |x: &Array1<f64>| (x[0] - 3.0).powi(2) + (x[1] - 7.0).powi(2),
    dimensions: 2,
};
```

## 8. Generiques et contraintes de traits

Les generiques permettent d'ecrire du code fonctionnant avec plusieurs types. Les contraintes de traits limitent les types autorises.

```rust
// Cette fonction fonctionne avec tout type implementant Classifier
fn cross_validate<C: Classifier>(model: &mut C, x: &Array2<f64>, y: &Array1<usize>) -> f64 {
    model.fit(x, y);
    let preds = model.predict(x);
    metrics::accuracy(y, &preds)
}

// Contraintes multiples
fn search<S: SearchState + Clone + std::fmt::Debug>(start: S) {
    // S doit implementer SearchState ET Clone ET Debug
}
```

Vous lirez les generiques plus souvent que vous ne les ecrirez. Quand vous voyez `<S: Trait>`, lisez simplement "S est n'importe quel type qui sait faire les choses definies par Trait."

## 9. Modules et imports

```rust
// Importer des elements specifiques
use ndarray::{Array1, Array2, array};
use ix_supervised::{LinearRegression, Regressor};
use ix_math::distance;

// Tout importer d'un module (a utiliser avec parcimonie)
use ix_math::stats::*;

// Convention de nommage des crates ix :
// Nom du crate : ix-supervised  (tiret)
// Nom a l'import : ix_supervised (underscore)
```

## 10. Executer du code

```bash
# Compiler tout le workspace
cargo build --workspace

# Lancer les tests
cargo test --workspace

# Lancer un exemple specifique
cargo run --example pso_rosenbrock

# Lancer avec les optimisations (beaucoup plus rapide pour le calcul numerique)
cargo run --release --example pso_rosenbrock
```

**Utilisez toujours `--release` pour les benchmarks ou les charges de travail reelles.** Les builds en mode debug sont 10 a 50 fois plus lents pour le calcul numerique.

## 11. Pieges courants pour les nouveaux Rustaceans

### Le borrow checker

```rust
let mut data = vec![1.0, 2.0, 3.0];
let first = &data[0];       // Emprunt immuable
// data.push(4.0);          // ERREUR : impossible de muter pendant un emprunt
println!("{}", first);       // L'emprunt se termine ici
data.push(4.0);             // Maintenant OK
```

**Solution** : restructurez pour que les emprunts ne chevauchent pas les mutations.

### Conversions de types

```rust
let n: usize = 10;
let mean: f64 = sum / n as f64;  // Il faut convertir explicitement usize en f64

let x: f32 = 3.14;
let y: f64 = x as f64;           // f32 vers f64 (elargissement, toujours sur)
let z: f32 = y as f32;           // f64 vers f32 (retrecissement, peut perdre en precision)
```

### Ordre partiel (flottants)

Les flottants peuvent etre NaN, donc ils n'implementent pas l'ordre total. Vous verrez `.partial_cmp()` et `.unwrap()` :

```rust
// Ceci ne compile pas :
// vec.sort_by(|a, b| a.cmp(b));

// Utilisez partial_cmp a la place :
vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
```

### Le turbofish `::<>`

Parfois Rust n'arrive pas a inferer un type dans une chaine. La syntaxe "turbofish" le lui indique :

```rust
let sum = data.iter().sum::<f64>();
//                        ^^^^^^^^ turbofish : "sum produit un f64"

let collected: Vec<f64> = data.iter().copied().collect();
// OU de maniere equivalente :
let collected = data.iter().copied().collect::<Vec<f64>>();
```

## Aide-memoire rapide

```rust
// Creer des tableaux
let v = array![1.0, 2.0, 3.0];                    // Array1
let m = array![[1.0, 2.0], [3.0, 4.0]];           // Array2

// Entrainer et predire (tous les algorithmes suivent ce schema)
let mut model = Algorithm::new(/* params */);
model.fit(&train_x, &train_y);
let predictions = model.predict(&test_x);

// Evaluer
let acc = metrics::accuracy(&test_y, &predictions);

// Optimiser
let result = optimizer.minimize(&objective);
println!("Meilleur : {:?} a {}", result.best_params, result.best_value);

// Gestion des erreurs
let safe_result = risky_function()?;  // Propager l'erreur
let unsafe_result = risky_function().unwrap();  // Paniquer en cas d'erreur
```

## Pour aller plus loin

- **Rust Book** (gratuit) : https://doc.rust-lang.org/book/ -- le guide de reference
- **Rust by Example** : https://doc.rust-lang.org/rust-by-example/ -- apprendre par la pratique
- **Documentation ndarray** : https://docs.rs/ndarray -- la bibliotheque de tableaux sur laquelle ix s'appuie
- **Suivant** : [Rust pour le ML](rust-pour-ml.md) -- les patterns Rust specifiques au ML dans ix
- **Commencer a apprendre** : [INDEX.md](../INDEX.md) -- le parcours d'apprentissage complet
