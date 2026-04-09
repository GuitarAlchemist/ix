# Modèles de Markov cachés

## Le problème

Vous êtes médecin et diagnostiquez un patient. L'état de santé réel du patient (Sain, Infection en développement, Infecté) est *caché* — vous ne pouvez pas l'observer directement. Ce que vous *pouvez* observer, ce sont les symptômes : température, numération des globules blancs, niveau d'énergie. Chaque état de santé produit des symptômes avec des probabilités différentes. Étant donné une séquence d'observations quotidiennes de symptômes, vous voulez inférer la séquence la plus probable d'états de santé cachés.

Scénarios concrets :
- **Reconnaissance vocale :** Les états cachés sont les phonèmes, les observations sont des caractéristiques acoustiques. Le même phonème sonne différemment selon le locuteur et le contexte.
- **Recherche de gènes :** Les états cachés sont les régions codantes/non-codantes. Les observations sont les bases nucléotidiques (A, T, C, G). Les régions codantes ont des fréquences de bases différentes.
- **Détection de régime financier :** Les états cachés sont les marchés Haussier/Baissier/Latéral. Les observations sont les rendements journaliers. Le même rendement peut provenir de n'importe quel régime.
- **Étiquetage morphosyntaxique :** Les états cachés sont les catégories grammaticales (nom, verbe, adjectif). Les observations sont les mots. « Avocat » peut être un nom (fruit ou juriste).
- **Reconnaissance d'activité :** Les états cachés sont les activités (marche, course, assis). Les observations sont les relevés d'accéléromètre.

## L'intuition

Imaginez que vous êtes enfermé dans une pièce sans fenêtre. Un ami se trouve dans une autre pièce et peut voir la météo (Ensoleillé ou Pluvieux). Chaque heure, votre ami fait une activité — Promenade, Shopping ou Ménage — et vous pouvez entendre ce qu'il fait. La météo influence son choix d'activité : par temps ensoleillé il se promène davantage, par temps pluvieux il fait le ménage.

Vous observez une séquence : Promenade, Shopping, Ménage, Promenade. Vous voulez deviner la séquence météo (Ensoleillé/Pluvieux pour chaque heure) qui explique le mieux ces activités.

Un HMM a trois composants :
1. **Distribution initiale (pi) :** Quelle est la probabilité de chaque état météo à l'heure zéro ?
2. **Matrice de transition (A) :** Connaissant la météo cette heure, quelle est la probabilité de chaque météo l'heure suivante ?
3. **Matrice d'émission (B) :** Connaissant la météo, quelle est la probabilité de chaque activité ?

Le « caché » dans HMM signifie que vous ne voyez jamais la météo directement — seulement les activités qu'elle produit.

## Comment ça fonctionne

### Les trois problèmes des HMM

**Problème 1 : Évaluation** — Quelle est la probabilité d'observer cette séquence ?

L'**algorithme Forward** calcule P(observations | modèle) efficacement en sommant sur toutes les séquences possibles d'états cachés :

```
alpha[t][i] = P(o_1, ..., o_t, q_t = i | modèle)
            = sum_j(alpha[t-1][j] * A[j][i]) * B[i][o_t]
```

**En clair :** À chaque pas de temps, la variable forward alpha[t][i] répond : « Quelle est la probabilité conjointe d'avoir vu les observations jusqu'ici ET d'être dans l'état i en ce moment ? » Elle calcule cela récursivement : prendre la probabilité d'être dans chaque état précédent, multiplier par la probabilité de transition pour arriver ici, puis multiplier par la probabilité d'émettre l'observation courante.

P(observations) = somme de toutes les valeurs alpha au dernier pas de temps.

**Problème 2 : Décodage** — Quelle est la séquence d'états cachés la plus probable ?

L'**algorithme de Viterbi** trouve le meilleur chemin unique à travers les états cachés (voir [algorithme-de-viterbi.md](./algorithme-de-viterbi.md)).

L'**algorithme Forward-Backward** calcule la probabilité a posteriori de chaque état à chaque pas de temps :

```
gamma[t][i] = P(q_t = i | observations, modèle)
            = alpha[t][i] * beta[t][i] / P(observations)
```

**En clair :** gamma[t][i] est la probabilité que le système était dans l'état i au temps t, connaissant *toutes* les observations (passées et futures). C'est plus lisse que Viterbi — au lieu de s'engager sur un chemin unique, on obtient une distribution de probabilité sur les états à chaque pas de temps.

**Problème 3 : Apprentissage** — Quels paramètres (A, B, pi) expliquent le mieux les données ?

L'**algorithme de Baum-Welch** (Espérance-Maximisation pour les HMM) ré-estime itérativement les paramètres pour augmenter la vraisemblance des observations.

```
Étape E : Calculer gamma et xi avec les paramètres courants
Étape M : Ré-estimer A, B, pi à partir de gamma et xi
Répéter jusqu'à convergence
```

**En clair :** Deviner les paramètres, calculer ce qu'étaient probablement les états cachés, puis mettre à jour les paramètres pour mieux correspondre à ces estimations. Répéter. Chaque itération est garantie d'augmenter (ou maintenir) la vraisemblance.

## En Rust

### Créer un HMM

```rust
use ix_graph::hmm::HiddenMarkovModel;
use ndarray::{array, Array2};

// HMM Météo :
//   États cachés : Pluvieux=0, Ensoleillé=1
//   Observations :  Promenade=0, Shopping=1, Ménage=2

let hmm = HiddenMarkovModel::new(
    array![0.6, 0.4],              // initial : 60% Pluvieux, 40% Ensoleillé
    array![[0.7, 0.3],             // transition : Pluvieux->Pluvieux=0.7, Pluvieux->Ensoleillé=0.3
           [0.4, 0.6]],            //              Ensoleillé->Pluvieux=0.4, Ensoleillé->Ensoleillé=0.6
    array![[0.1, 0.4, 0.5],        // émission : Pluvieux->Promenade=0.1, Shopping=0.4, Ménage=0.5
           [0.6, 0.3, 0.1]],       //            Ensoleillé->Promenade=0.6, Shopping=0.3, Ménage=0.1
).unwrap();

assert_eq!(hmm.n_states(), 2);
assert_eq!(hmm.n_observations(), 3);
```

### Problème 1 : Évaluation (algorithme Forward)

```rust
let observations = [0, 1, 2];  // Promenade, Shopping, Ménage

// Log-probabilité de cette séquence d'observations
let log_prob = hmm.forward(&observations);
println!("Log P(Promenade,Shopping,Ménage) = {:.4}", log_prob);
// Un nombre négatif fini (log d'une probabilité < 1)
```

### Problème 2a : Décodage (Viterbi)

```rust
let observations = [0, 1, 2, 0];  // Promenade, Shopping, Ménage, Promenade

let (path, log_prob) = hmm.viterbi(&observations);
println!("États les plus probables : {:?}", path);
println!("Log-probabilité :          {:.4}", log_prob);
// path[0] = 1 (Ensoleillé) car P(Ensoleillé)*P(Promenade|Ensoleillé) > P(Pluvieux)*P(Promenade|Pluvieux)
// 0.4*0.6 = 0.24 > 0.6*0.1 = 0.06
```

### Problème 2b : Lissage (Forward-Backward)

```rust
let observations = [0, 1, 2, 0, 1];

let gamma = hmm.forward_backward(&observations);
// gamma est une matrice 5x2 : gamma[t][i] = P(état i au temps t | toutes les observations)
println!("A posteriori à t=0 : Pluvieux={:.3}, Ensoleillé={:.3}", gamma[[0, 0]], gamma[[0, 1]]);
println!("A posteriori à t=2 : Pluvieux={:.3}, Ensoleillé={:.3}", gamma[[2, 0]], gamma[[2, 1]]);

// Chaque ligne somme à 1
for t in 0..5 {
    let sum: f64 = gamma.row(t).sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

// Estimation MAP : état le plus probable à chaque pas de temps
let map_states = hmm.map_estimate(&observations);
println!("États MAP : {:?}", map_states);
```

### Problème 3 : Apprentissage (Baum-Welch)

```rust
// Séquence observée (des séquences plus longues donnent de meilleures estimations)
let observations = vec![0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 1, 0];

let log_prob_before = hmm.forward(&observations);

// Exécuter Baum-Welch EM pour au plus 50 itérations, tolérance de convergence 1e-8
let trained = hmm.baum_welch(&observations, 50, 1e-8).unwrap();

let log_prob_after = trained.forward(&observations);

println!("Log-vraisemblance avant : {:.4}", log_prob_before);
println!("Log-vraisemblance après : {:.4}", log_prob_after);
// Après l'entraînement, la vraisemblance devrait augmenter (ou rester constante)

println!("Transition apprise :\n{:.4}", trained.transition);
println!("Émission apprise :\n{:.4}", trained.emission);
```

Voir l'exemple complet fonctionnel : [`examples/sequence/viterbi_hmm.rs`](../../examples/sequence/viterbi_hmm.rs)

## Quand l'utiliser

| Modèle | États cachés | Observations | Transitions | Idéal pour |
|---|---|---|---|---|
| **Chaîne de Markov** | Non (états observés) | N/A | État à état | Météo, PageRank, files d'attente |
| **HMM** | Oui (discrets) | Discrètes | État à état | Parole, gènes, étiquetage morphosyntaxique |
| **Filtre de Kalman** | Oui (continus) | Continues | Linéaires + gaussiennes | Suivi, navigation, finance |
| **CRF** | Non (mais discriminatif) | Caractéristiques | État à état | Étiquetage NLP (supérieur au HMM) |
| **RNN/LSTM** | Appris | Quelconques | Apprises | Grandes données, dépendances complexes |

**Utilisez les HMM quand :**
- Vous avez une séquence d'observations discrètes.
- Le processus sous-jacent a un petit nombre d'états cachés discrets.
- La propriété de Markov est raisonnable (l'état au temps t ne dépend que de l'état au temps t-1).
- Vous voulez des paramètres interprétables (matrices de transition et d'émission).

**N'utilisez pas quand :**
- Les observations sont continues et de haute dimension (utilisez des filtres de Kalman ou des RNN).
- Les dépendances à longue portée comptent (les HMM sont sans mémoire ; les LSTM gèrent cela).
- Vous avez des données d'entraînement étiquetées (utilisez des CRF ou des modèles supervisés au lieu de l'EM).

## Paramètres clés

### Constructeur

| Paramètre | Type | Contraintes |
|---|---|---|
| `initial` | `Array1<f64>` | Longueur N, somme à 1, non négatif |
| `transition` | `Array2<f64>` | N × N, chaque ligne somme à 1, non négatif |
| `emission` | `Array2<f64>` | N × M, chaque ligne somme à 1, non négatif |

N = nombre d'états cachés, M = nombre de symboles d'observation.

### Méthodes

| Méthode | Renvoie | Description |
|---|---|---|
| `forward(&[usize])` | `f64` | Log-probabilité de la séquence d'observations |
| `forward_backward(&[usize])` | `Array2<f64>` | Probabilités a posteriori des états T × N (gamma) |
| `viterbi(&[usize])` | `(Vec<usize>, f64)` | Chemin d'états le plus probable + sa log-probabilité |
| `map_estimate(&[usize])` | `Vec<usize>` | État le plus probable à chaque pas de temps (d'après gamma) |
| `baum_welch(&[usize], max_iter, tol)` | `Result<Self, String>` | HMM entraîné par EM avec paramètres mis à jour |

### Paramètres de Baum-Welch

| Paramètre | Valeur typique | Description |
|---|---|---|
| `max_iter` | 50-200 | Nombre maximum d'itérations EM |
| `tol` | 1e-6 à 1e-10 | Arrêt quand le changement de log-vraisemblance est inférieur à ce seuil |

## Pièges courants

1. **Les observations doivent être des indices `usize`.** La matrice d'émission associe les états à des *symboles* d'observation (entiers 0..M-1). Si vos observations sont continues, vous devez les discrétiser d'abord (ex. regrouper les températures en Basse=0, Moyenne=1, Haute=2).

2. **Baum-Welch trouve des optima locaux.** L'EM est garanti d'améliorer la vraisemblance à chaque étape mais peut converger vers un maximum local. Exécutez-le plusieurs fois avec différentes initialisations et gardez le meilleur résultat.

3. **Permutation des labels.** Après l'entraînement Baum-Welch, « état 0 » et « état 1 » peuvent avoir échangé de signification par rapport à votre modèle initial. L'algorithme ne sait pas quel état est « Pluvieux » — il trouve juste les meilleurs paramètres. Inspectez la matrice d'émission pour interpréter les états.

4. **Sous-dépassement avec les longues séquences.** L'algorithme Forward multiplie de nombreuses petites probabilités entre elles. ix utilise une mise à l'échelle (normalisation à chaque étape) pour prévenir le sous-dépassement, mais des séquences extrêmement longues (>10 000 observations) peuvent encore perdre en précision.

5. **Les probabilités nulles bloquent l'apprentissage.** Si `emission[i][k] = 0`, l'état i ne peut jamais émettre le symbole k. Baum-Welch ne peut pas récupérer de cette situation. Initialisez avec de petites valeurs positives partout (ex. ajoutez 0,01 et re-normalisez).

6. **Viterbi vs. estimation MAP.** Viterbi trouve la *séquence* d'états la plus probable dans son ensemble. L'estimation MAP trouve l'état le plus probable à chaque pas de temps *individuellement*. Ceux-ci peuvent différer ! Viterbi assure la cohérence des transitions ; MAP non.

## Pour aller plus loin

- **Chaînes de Markov :** Le fondement observable. Voir [chaines-de-markov.md](./chaines-de-markov.md).
- **Algorithme de Viterbi :** Plongée dans l'approche par programmation dynamique. Voir [algorithme-de-viterbi.md](./algorithme-de-viterbi.md).
- **HMM gaussiens :** Remplacer les émissions discrètes par des distributions gaussiennes continues. Nécessite de modifier le modèle d'émission (pas encore dans ix).
- **HMM d'ordre supérieur :** L'état caché dépend des k derniers états, pas seulement du dernier. Augmente la capacité du modèle au prix d'un espace d'états en O(N^k).
- **HMM entrée-sortie :** Les probabilités de transition et d'émission dépendent d'un signal d'entrée externe. Utile pour les applications de contrôle.
- **HMM hiérarchiques :** Les états peuvent eux-mêmes être des HMM, permettant une modélisation temporelle multi-échelle.
