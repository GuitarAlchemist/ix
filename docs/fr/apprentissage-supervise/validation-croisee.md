# Validation croisée

## Le problème

Vous avez construit un classifieur qui obtient 95 % de précision sur votre jeu de test. On déploie ? Pas si vite. Et si votre découpage aléatoire train/test avait été chanceux — les exemples faciles dans le test, les difficiles dans l'entraînement ? Ou si le jeu de test ne contenait aucun exemple de la classe minoritaire ?

Un seul découpage train/test donne un seul chiffre. Ce chiffre peut être élevé ou bas selon le découpage aléatoire. Vous ne savez pas si 95 % est stable ou si ça chuterait à 80 % avec un autre découpage. Il faut plusieurs évaluations pour obtenir une estimation fiable.

La validation croisée résout ce problème en alternant systématiquement quelles données servent à l'entraînement et au test, donnant non pas un score de précision mais *k* scores. La moyenne indique la qualité de généralisation du modèle. La variance indique sa stabilité.

## L'intuition

Imaginez un professeur qui veut évaluer équitablement un étudiant. Au lieu d'un seul examen, il donne cinq examens, chacun couvrant différents sujets. La note moyenne sur les cinq examens est bien meilleure que n'importe quel examen pris isolément.

La validation croisée k-fold fait la même chose. Elle découpe les données en k parts égales (les « plis »), entraîne le modèle k fois, et à chaque fois utilise un pli différent comme jeu de test :

```
Pli 1 : [TEST]         [entraînement] [entraînement] [entraînement] [entraînement]
Pli 2 : [entraînement] [TEST]         [entraînement] [entraînement] [entraînement]
Pli 3 : [entraînement] [entraînement] [TEST]         [entraînement] [entraînement]
Pli 4 : [entraînement] [entraînement] [entraînement] [TEST]         [entraînement]
Pli 5 : [entraînement] [entraînement] [entraînement] [entraînement] [TEST]
```

Chaque échantillon est testé exactement une fois. Aucun échantillon n'est jamais dans l'ensemble d'entraînement et de test en même temps.

## Comment ça marche

### K-Fold

1. Mélanger le jeu de données (optionnel mais recommandé)
2. Découper en k plis de taille égale
3. Pour chaque pli i :
   - Entraîner sur tous les plis sauf le pli i
   - Évaluer sur le pli i
   - Enregistrer le score
4. Rapporter les k scores, leur moyenne et leur écart-type

### K-Fold Stratifié

Le k-fold standard peut créer des problèmes avec des données déséquilibrées. Si vous avez 100 échantillons (90 classe A, 10 classe B) et découpez en 5 plis de 20, certains plis pourraient se retrouver avec 0 échantillon de classe B par hasard.

Le k-fold stratifié garantit que chaque pli a la même distribution de classes que le jeu de données complet. Chaque pli contiendrait environ 18 classe A et 2 classe B.

$$
\text{Ratio de classes dans chaque pli} \approx \text{Ratio de classes dans le jeu complet}
$$

C'est essentiel pour :
- Les jeux de données déséquilibrés (détection de fraude, diagnostic médical)
- Les petits jeux de données où la variation aléatoire compte davantage
- Tout problème de classification où l'on veut des scores fiables par pli

## En Rust

### K-Fold basique

```rust
use ix_supervised::validation::KFold;

fn main() {
    // Validation croisée 5-fold sur 100 échantillons
    let kf = KFold::new(5).with_seed(42);
    let folds = kf.split(100);

    for (i, (train, test)) in folds.iter().enumerate() {
        println!("Pli {} : {} entraînement, {} test", i + 1, train.len(), test.len());
    }
    // Pli 1 : 80 entraînement, 20 test
    // Pli 2 : 80 entraînement, 20 test
    // ...
}
```

### K-Fold Stratifié

```rust
use ndarray::array;
use ix_supervised::validation::StratifiedKFold;

fn main() {
    // Données déséquilibrées : 80 % classe 0, 20 % classe 1
    let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

    let skf = StratifiedKFold::new(5).with_seed(42);
    let folds = skf.split(&y);

    for (i, (train, test)) in folds.iter().enumerate() {
        let test_classe_1 = test.iter().filter(|&&idx| y[idx] == 1).count();
        println!("Pli {} : {} entraînement, {} test ({} classe-1 dans le test)",
            i + 1, train.len(), test.len(), test_classe_1);
    }
}
```

### Score de validation croisée en une ligne

```rust
use ndarray::{array, Array2};
use ix_supervised::validation::cross_val_score;
use ix_supervised::decision_tree::DecisionTree;

fn main() {
    let x = Array2::from_shape_vec((12, 2), vec![
        0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,  0.8, 0.1,  0.2, 0.7,
        5.0, 5.0,  5.5, 5.5,  6.0, 5.0,  5.3, 5.2,  5.8, 5.1,  5.2, 5.7,
    ]).unwrap();
    let y = array![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

    // Validation croisée stratifiée 4-fold avec un arbre de décision
    let scores = cross_val_score(&x, &y, || DecisionTree::new(5), 4, 42);

    let moyenne = scores.iter().sum::<f64>() / scores.len() as f64;
    let écart_type = (scores.iter().map(|s| (s - moyenne).powi(2)).sum::<f64>() / scores.len() as f64).sqrt();

    println!("Scores par pli : {:?}", scores);
    println!("Précision moyenne : {:.4} (+/- {:.4})", moyenne, écart_type);
}
```

## Quand l'utiliser

| Situation | Utiliser | Pourquoi |
|-----------|----------|----------|
| Sélection de modèle (comparer des algorithmes) | Toujours | Une comparaison équitable nécessite une moyenne sur plusieurs découpages |
| Petit jeu de données (< 1 000 échantillons) | Toujours | Un seul découpage est trop bruité |
| Classes déséquilibrées | K-Fold Stratifié | Garantit que la classe minoritaire apparaît dans chaque pli |
| Grand jeu de données (> 100 000 échantillons) | Optionnel | Un découpage unique 80/20 est généralement stable |
| Réglage d'hyperparamètres | Oui | Évite le surajustement à un jeu de test particulier |
| Évaluation finale du modèle | Oui, puis réentraîner sur toutes les données | Estimation fiable, puis utiliser toutes les données pour le modèle final |

### Choisir k

| k | Comportement | Quand |
|---|-------------|-------|
| 2 | Découpage 50/50, forte variance | Jamais recommandé |
| 5 | Bon équilibre biais/variance | Par défaut pour la plupart des problèmes |
| 10 | Faible biais, plus de calcul | Standard dans les articles académiques |
| n (LOO) | Leave-one-out, biais le plus faible | Très petits jeux de données uniquement |

## Pièges

**Fuite de données entre les plis.** Le prétraitement (normalisation, ACP, sélection de variables) doit être fait *à l'intérieur* de chaque pli, pas avant le découpage. Si vous normalisez tout le jeu de données d'abord, l'information du pli de test fuit dans l'entraînement.

**Les séries temporelles violent l'hypothèse i.i.d.** Si les données ont un ordre temporel (cours boursiers, capteurs), le mélange aléatoire crée des fuites — le modèle voit des données futures pendant l'entraînement. Utilisez des découpages temporels à la place.

**La stratification compte plus qu'on ne le croit.** Avec 5 % de classe positive et un CV 5-fold, certains plis pourraient avoir 0 % ou 10 % de positifs par hasard. Utilisez toujours StratifiedKFold pour la classification.

**k modèles, pas un seul.** La validation croisée entraîne k modèles séparés. Les scores estiment la généralisation, mais on ne garde pas les modèles. Après la validation croisée, réentraînez une seule fois sur *toutes* les données pour le déploiement.

## Pour aller plus loin

- **Validation croisée imbriquée :** Utilisez une boucle CV externe pour l'évaluation et une boucle CV interne pour le réglage des hyperparamètres. Empêche le biais optimiste du réglage sur les mêmes données que l'évaluation.
- **K-fold répété :** Exécutez le CV k-fold plusieurs fois avec différentes graines aléatoires et moyennez tous les scores. Réduit la variance de l'estimation au prix de k × n_répétitions entraînements de modèles.
- **K-fold par groupes :** Quand les échantillons sont groupés (par ex., plusieurs mesures par patient), assurez-vous que tous les échantillons d'un groupe restent dans le même pli.
