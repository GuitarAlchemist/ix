# Algorithme de Viterbi

## Le problème

Vous construisez un système de navigation GPS. La puce GPS vous donne une position bruitée chaque seconde, mais les lectures sautent (parfois de 50 mètres). Vous connaissez le réseau routier et la vitesse approximative du véhicule. Étant donné une séquence de lectures GPS bruitées, quelle est la séquence la plus probable de *segments routiers réels* sur lesquels le véhicule a circulé ?

C'est une application classique de l'algorithme de Viterbi : vous avez une séquence d'observations bruitées et un modèle de l'évolution de l'état caché, et vous voulez trouver la meilleure séquence d'états cachés.

Scénarios concrets :
- **Correction de trajectoire GPS (map matching) :** Accrocher les points GPS bruités aux segments routiers les plus probables.
- **Décodage en reconnaissance vocale :** À partir de caractéristiques acoustiques, trouver la séquence de mots la plus probable.
- **Correction d'erreurs en communications :** À partir d'une séquence de bits reçue (potentiellement corrompue), trouver la séquence transmise la plus probable.
- **Annotation génomique :** À partir d'une séquence ADN, déterminer quelles régions sont des exons, des introns ou des régions intergéniques.
- **Détection d'intrusion réseau :** À partir d'une séquence de caractéristiques de paquets, trouver la séquence d'états système la plus probable (normal, sondage, compromis).

## L'intuition

Imaginez que vous regardez un ami traverser un labyrinthe embrumé. Vous ne voyez pas les murs, mais vous entendez l'écho des pas. Certaines salles produisent des échos forts (grandes salles), d'autres des échos faibles (petites salles). Vous connaissez le plan du labyrinthe (quelles salles sont connectées) et le profil acoustique de chaque salle.

Votre ami fait 5 pas. Vous entendez : fort, faible, faible, fort, faible.

L'approche par force brute : énumérer tous les chemins possibles de 5 salles, calculer leur probabilité et choisir le meilleur. Avec 10 salles et 5 pas, cela fait 10^5 = 100 000 chemins. Avec 100 salles et 100 pas : 100^100. Impossible.

**L'idée de Viterbi :** Utiliser la programmation dynamique. À chaque étape, pour chaque salle, ne garder que le *meilleur chemin qui se termine dans cette salle*. Éliminer tous les autres. Pourquoi ? Parce que si le meilleur chemin global passe par la salle X à l'étape 3, le sous-chemin vers la salle X à l'étape 3 doit aussi être le meilleur moyen d'atteindre la salle X à l'étape 3. (C'est le **principe d'optimalité**.)

Cela réduit le travail de N^T (exponentiel) à N² × T (polynomial).

## Comment ça fonctionne

### Prérequis

Soit un HMM avec :
- N états cachés
- Distribution initiale pi[i] = P(démarrer dans l'état i)
- Matrice de transition A[i][j] = P(état j à t+1 | état i à t)
- Matrice d'émission B[i][k] = P(observation k | état i)
- Séquence d'observations O = [o_0, o_1, ..., o_{T-1}]

### L'algorithme

**Étape 1 : Initialisation (t = 0)**

```
delta[0][i] = ln(pi[i]) + ln(B[i][o_0])
psi[0][i] = 0  (pas de prédécesseur)
```

**En clair :** Pour chaque état, calculer la log-probabilité de démarrer là et d'émettre la première observation. On travaille en espace logarithmique pour éviter de multiplier de nombreuses probabilités minuscules (ce qui causerait un sous-dépassement).

**Étape 2 : Récursion (t = 1, ..., T-1)**

```
delta[t][j] = max_i(delta[t-1][i] + ln(A[i][j])) + ln(B[j][o_t])
psi[t][j]   = argmax_i(delta[t-1][i] + ln(A[i][j]))
```

**En clair :** Pour chaque état j au temps t, trouver l'état prédécesseur i qui maximise la probabilité du chemin. Enregistrer le meilleur prédécesseur dans psi (le « pointeur arrière »). delta[t][j] stocke la log-probabilité du meilleur chemin se terminant dans l'état j au temps t.

**Étape 3 : Terminaison**

```
meilleur_état_final = argmax_i(delta[T-1][i])
meilleure_log_prob  = max_i(delta[T-1][i])
```

**En clair :** Le meilleur chemin se termine à l'état qui a le delta le plus élevé au dernier pas de temps.

**Étape 4 : Retour arrière**

```
chemin[T-1] = meilleur_état_final
Pour t = T-2 jusqu'à 0 :
    chemin[t] = psi[t+1][chemin[t+1]]
```

**En clair :** Suivre les pointeurs arrière de la fin vers le début pour reconstruire le chemin complet.

### Complexité

| Métrique | Valeur |
|---|---|
| Temps | O(N² × T) |
| Espace | O(N × T) |

Où N = nombre d'états cachés, T = longueur de la séquence d'observations.

À comparer avec la force brute : O(N^T) — exponentiel en longueur de séquence.

## En Rust

### Utilisation de base

```rust
use ix_graph::hmm::HiddenMarkovModel;
use ndarray::array;

// États : Ensoleillé=0, Pluvieux=1
// Observations : Promenade=0, Shopping=1, Ménage=2
let hmm = HiddenMarkovModel::new(
    array![0.6, 0.4],                  // initial
    array![[0.7, 0.3], [0.4, 0.6]],    // transition
    array![[0.1, 0.4, 0.5],            // émission (Ensoleillé)
           [0.6, 0.3, 0.1]],           // émission (Pluvieux)
).unwrap();

let observations = [0, 1, 2, 0];  // Promenade, Shopping, Ménage, Promenade

let (path, log_prob) = hmm.viterbi(&observations);
println!("Observations : Promenade, Shopping, Ménage, Promenade");
println!("Meilleurs états : {:?}", path);       // ex. [1, 0, 0, 1]
println!("Log-prob :        {:.4}", log_prob);   // ex. -5.2832
```

### Pourquoi Promenade -> Ensoleillé (et non Pluvieux) ?

```rust
// À t=0, l'observation est Promenade (indice 0) :
// P(Ensoleillé) * P(Promenade|Ensoleillé) = 0.4 * 0.6 = 0.24
// P(Pluvieux) * P(Promenade|Pluvieux) = 0.6 * 0.1 = 0.06
// Ensoleillé l'emporte malgré un a priori plus faible, car Ensoleillé émet Promenade bien plus souvent.

let (path, _) = hmm.viterbi(&[0]);  // juste "Promenade"
assert_eq!(path, vec![1]);  // Ensoleillé
```

### HMM déterministe (vérification de correction)

```rust
use ix_graph::hmm::HiddenMarkovModel;
use ndarray::array;

// Correspondance quasi-déterministe : état 0 -> obs 0, état 1 -> obs 1
let hmm = HiddenMarkovModel::new(
    array![1.0, 0.0],                    // toujours démarrer dans l'état 0
    array![[0.1, 0.9], [0.9, 0.1]],      // transitions fortement alternantes
    array![[0.99, 0.01], [0.01, 0.99]],   // état i émet obs i avec 99% de prob
).unwrap();

let (path, log_prob) = hmm.viterbi(&[0, 1, 0, 1]);
assert_eq!(path, vec![0, 1, 0, 1]);  // décodage parfait
println!("Log-probabilité : {:.4}", log_prob);
```

### Viterbi vs. Forward-Backward (MAP)

Ces deux méthodes de décodage peuvent donner des résultats différents :

```rust
let observations = [0, 1, 2, 0, 1];

// Viterbi : meilleur chemin unique (globalement cohérent)
let (viterbi_path, _) = hmm.viterbi(&observations);

// MAP : état le plus probable à chaque pas de temps individuel
let map_path = hmm.map_estimate(&observations);

println!("Viterbi : {:?}", viterbi_path);
println!("MAP :     {:?}", map_path);
// Ceux-ci peuvent différer ! Viterbi assure des transitions valides entre états consécutifs.
// MAP choisit le meilleur état indépendamment à chaque pas de temps.
```

### Map matching GPS (conceptuel)

```rust
use ix_graph::hmm::HiddenMarkovModel;
use ndarray::{array, Array1, Array2};

// Map matching GPS simplifié :
// États cachés : 4 segments routiers (0, 1, 2, 3)
// Observations : 3 zones GPS (0=Nord, 1=Centre, 2=Sud)

// Le segment routier sur lequel vous êtes détermine la zone GPS probable
let hmm = HiddenMarkovModel::new(
    array![0.5, 0.3, 0.15, 0.05],  // départ le plus probable sur les routes nord
    Array2::from_shape_vec((4, 4), vec![
        0.6, 0.3, 0.1, 0.0,  // route 0 : reste probablement ou va vers route 1
        0.1, 0.5, 0.3, 0.1,  // route 1 : centre, connecte 0 et 2
        0.0, 0.2, 0.5, 0.3,  // route 2 : connecte vers le sud
        0.0, 0.0, 0.3, 0.7,  // route 3 : sud, tend à rester
    ]).unwrap(),
    Array2::from_shape_vec((4, 3), vec![
        0.8, 0.15, 0.05,  // route 0 -> généralement GPS Nord
        0.2, 0.6, 0.2,    // route 1 -> généralement GPS Centre
        0.05, 0.3, 0.65,  // route 2 -> généralement GPS Sud
        0.02, 0.08, 0.9,  // route 3 -> généralement GPS Sud
    ]).unwrap(),
).unwrap();

// Lectures GPS bruitées sur 6 secondes
let gps_readings = [0, 0, 1, 1, 2, 2];  // Nord, Nord, Centre, Centre, Sud, Sud
let (road_segments, log_prob) = hmm.viterbi(&gps_readings);
println!("Zones GPS :         {:?}", gps_readings);
println!("Segments routiers : {:?}", road_segments);
// Attendu : [0, 0, 1, 1, 2, 2] ou [0, 0, 1, 1, 2, 3] — trajet fluide vers le sud
```

Voir aussi : [`examples/sequence/viterbi_hmm.rs`](../../examples/sequence/viterbi_hmm.rs)

## Quand l'utiliser

| Méthode | Trouve | Complexité | Garanties |
|---|---|---|---|
| **Viterbi** | Le meilleur chemin unique | O(N² × T) | Chemin globalement optimal |
| **Forward-Backward (MAP)** | Le meilleur état à chaque pas | O(N² × T) | Localement optimal par pas |
| **Recherche en faisceau** | Les k meilleurs chemins | O(k × N × T) | Approximatif (peut manquer l'optimal) |
| **Force brute** | Le meilleur chemin | O(N^T) | Optimal mais impraticable |

**Utilisez Viterbi quand :**
- Vous avez besoin du chemin *complet* le plus probable (pas juste des marginales à chaque pas).
- La cohérence des transitions compte (le chemin doit suivre des transitions valides).
- L'espace d'états est assez petit pour un calcul exact (N < ~1 000).

**Utilisez plutôt Forward-Backward quand :**
- Vous voulez des distributions de probabilité sur les états à chaque pas de temps.
- Vous voulez quantifier l'incertitude (pas seulement choisir le meilleur).
- Vous prévoyez d'utiliser les postérieures pour des tâches en aval (ex. entraînement Baum-Welch).

## Paramètres clés

Viterbi en lui-même n'a pas de paramètres de réglage. Il prend un HMM et une séquence d'observations et renvoie le chemin optimal. La qualité du résultat dépend entièrement des paramètres du HMM.

| Entrée | Type | Description |
|---|---|---|
| `observations` | `&[usize]` | Séquence d'indices d'observations (0..M-1) |

| Sortie | Type | Description |
|---|---|---|
| `path` | `Vec<usize>` | État caché le plus probable à chaque pas de temps |
| `log_prob` | `f64` | Log-probabilité du meilleur chemin |

## Pièges courants

1. **Log-probabilité, pas probabilité.** La `log_prob` renvoyée est un grand nombre négatif (ex. -15,3). La probabilité réelle est exp(-15,3) ce qui est minuscule. N'exponentiez jamais pour les longues séquences — le résultat s'effondre à zéro par sous-dépassement.

2. **Les transitions nulles bloquent les chemins.** Si `A[i][j] = 0`, l'algorithme de Viterbi ne transitera jamais de l'état i à j (log(0) = -infini). Assurez-vous que votre matrice de transition autorise toutes les transitions nécessaires, même avec de faibles probabilités.

3. **Séquences d'observations vides.** `hmm.viterbi(&[])` renvoie `(vec![], 0.0)`. C'est un cas limite valide, pas une erreur.

4. **Viterbi donne un chemin, mais il peut y avoir de nombreux chemins quasi-optimaux.** Si les deux meilleurs chemins ont des log-probabilités de -15,30 et -15,31, ils sont essentiellement aussi probables. Viterbi ne renvoie que le premier. Pour les applications où les quasi-égalités comptent, envisagez le calcul des N meilleurs chemins (pas encore implémenté).

5. **Viterbi et MAP peuvent être en désaccord.** Viterbi : « la meilleure *séquence* d'états. » MAP : « le meilleur *état* à chaque pas de temps. » Exemple : Viterbi choisit l'état A au temps t car il mène à un excellent chemin global, même si l'état B est marginalement plus probable au temps t isolément. Les deux sont des réponses correctes à des questions différentes.

6. **Précision numérique.** L'implémentation travaille en espace logarithmique, ce qui gère la plupart des problèmes de sous-dépassement. Cependant, avec des probabilités extrêmement petites (ex. `emission[i][k] = 1e-300`), même l'espace logarithmique peut perdre en précision. Gardez les probabilités au-dessus de 1e-100 environ.

## Pour aller plus loin

- **Modèles de Markov cachés :** Traitement complet des HMM, incluant Forward, Forward-Backward et Baum-Welch. Voir [modeles-de-markov-caches.md](./modeles-de-markov-caches.md).
- **Chaînes de Markov :** Le fondement à états observables. Voir [chaines-de-markov.md](./chaines-de-markov.md).
- **Champs aléatoires conditionnels (CRF) :** Une alternative discriminative aux HMM qui donne souvent de meilleurs résultats pour l'étiquetage de séquences quand on dispose de données d'entraînement étiquetées.
- **Viterbi pour les codes convolutifs :** En communications, le même algorithme décode les codes correcteurs d'erreurs. Les « états » sont les états de l'encodeur et les « observations » sont les bits reçus.
- **Viterbi paresseux :** Pour de très grands espaces d'états, on ne développe que les états prometteurs. Lié à la recherche A* — voir Q* dans `ix-search` pour la recherche avec heuristiques apprises.
- **Viterbi en ligne :** Traiter les observations une par une, en émettant les états décodés avec un délai fixe. Utile pour les applications temps réel comme la reconnaissance vocale en direct.
