# Bandits multi-bras

## Le problème

Vous gérez un site e-commerce avec trois designs de bannière. Chaque visiteur voit une bannière et clique (récompense = 1) ou l'ignore (récompense = 0). Le test A/B traditionnel vous verrouille dans une répartition fixe -- disons 33/33/33 -- pendant des semaines avant d'avoir assez de données pour désigner un gagnant. Pendant ce temps, chaque impression gaspillée sur la pire bannière est du revenu perdu.

Les bandits multi-bras résolvent ce problème : ils *apprennent tout en gagnant*, en dirigeant progressivement le trafic vers la variante la plus performante sans attendre la « fin » de l'expérience.

Autres scénarios concrets :
- **Placement publicitaire :** Quel visuel publicitaire génère le plus de clics ?
- **Essais cliniques :** Quel dosage de médicament est le plus sûr tout en recueillant des données sur les alternatives ?
- **Recommandation de contenu :** Quel titre d'article génère le plus d'engagement ?
- **Tarification dynamique :** Quel prix maximise la conversion ?

## L'intuition

Imaginez que vous entrez dans un casino avec trois machines à sous. Chaque machine paie à un taux différent (inconnu). Vous avez 1 000 jetons.

La stratégie naïve : jouer chaque machine 333 fois, puis choisir la meilleure. Mais cela gaspille des jetons sur des machines dont vous soupçonnez déjà qu'elles sont mauvaises.

La stratégie intelligente : continuer à jouer la machine qui *semble* la meilleure, mais essayer occasionnellement les autres au cas où vos premières impressions seraient fausses. Cette tension -- « s'en tenir à ce qui marche » vs. « peut-être qu'il y a mieux ailleurs » -- est le **compromis exploration vs. exploitation**, et chaque algorithme de bandit le résout différemment.

## Comment ça fonctionne

Les trois algorithmes maintiennent une **valeur estimée** Q(a) pour chaque bras a, mise à jour de manière incrémentale après chaque tirage.

### Mise à jour incrémentale de la moyenne

Chaque fois que le bras a est tiré et retourne la récompense r :

```
Q(a) <- Q(a) + (1/n) * (r - Q(a))
```

**En clair :** On ajuste l'estimation vers la nouvelle observation. L'ajustement diminue à mesure que l'on collecte plus d'échantillons (1/n rétrécit), donc les premiers tirages ont un effet plus important que les suivants.

### Epsilon-greedy

```
With probability epsilon: pick a random arm        (explore)
With probability 1 - epsilon: pick argmax Q(a)     (exploit)
```

**En clair :** La plupart du temps, choisir la meilleure option connue. Mais epsilon pour cent du temps, choisir aléatoirement pour s'assurer de ne pas manquer une meilleure alternative.

### UCB1 (Upper Confidence Bound)

```
score(a) = Q(a) + sqrt(2 * ln(t) / N(a))
Pick: argmax score(a)
```

Où t est le nombre total de tirages jusqu'ici et N(a) le nombre de fois que le bras a a été tiré.

**En clair :** Choisir le bras avec l'estimation *optimiste* la plus élevée. Le terme bonus est grand pour les bras peu testés (N(a) faible), donc les bras sous-explorés reçoivent un coup de pouce. Plus on tire un bras, plus le bonus diminue et la moyenne brute domine.

### Thompson Sampling

```
For each arm a:
    sample theta(a) ~ Normal(mean=Q(a), variance=1/N(a))
Pick: argmax theta(a)
```

**En clair :** Chaque bras a une distribution de croyance (votre degré de confiance sur sa valeur). On échantillonne un nombre de chaque distribution et on choisit le plus élevé. Les bras mal connus ont des distributions larges, donc ils produisent parfois des échantillons élevés -- c'est ainsi que Thompson explore. Les bras bien connus ont des distributions serrées centrées sur leur vraie valeur.

## En Rust

```rust
use ix_rl::bandit::{EpsilonGreedy, UCB1, ThompsonSampling};

// === Epsilon-Greedy : simple et prévisible ===
let mut eg = EpsilonGreedy::new(
    3,    // 3 bras (designs de bannière A, B, C)
    0.1,  // explorer 10% du temps
    42,   // graine RNG pour la reproductibilité
);

for _ in 0..1000 {
    let arm = eg.select_arm();          // retourne 0, 1, ou 2
    let reward = simulate_click(arm);   // votre signal de récompense
    eg.update(arm, reward);             // mettre à jour la Q-valeur de ce bras
}
// Inspecter les valeurs apprises
println!("CTR estimés : {:?}", eg.q_values);   // Vec<f64>
println!("Nombre de tirages : {:?}", eg.counts);      // Vec<usize>


// === UCB1 : pas de paramètre à régler, exploration automatique ===
let mut ucb = UCB1::new(3);

for _ in 0..1000 {
    let arm = ucb.select_arm();         // déterministe étant donné l'historique
    let reward = simulate_click(arm);
    ucb.update(arm, reward);
}
println!("Tirages totaux : {}", ucb.total_count);


// === Thompson Sampling : bayésien, souvent le meilleur en pratique ===
let mut ts = ThompsonSampling::new(3, 42);

for _ in 0..1000 {
    let arm = ts.select_arm();
    let reward = simulate_click(arm);
    ts.update(arm, reward);
}
// Thompson suit les moyennes et variances a posteriori
println!("Moyennes a posteriori :  {:?}", ts.means);
println!("Variances a posteriori : {:?}", ts.variances);
```

Voir l'exemple complet : [`examples/reinforcement-learning/bandits.rs`](../../examples/reinforcement-learning/bandits.rs)

## Quand l'utiliser

| Algorithme | Idéal quand | Réglage nécessaire | Style d'exploration |
|---|---|---|---|
| **Epsilon-Greedy** | Vous voulez la simplicité ; les distributions de récompenses sont stables | Oui (epsilon) | Aléatoire uniforme |
| **UCB1** | Vous voulez des garanties théoriques ; pas d'hyperparamètres à régler | Aucun | Guidée par l'optimisme |
| **Thompson Sampling** | Vous voulez la meilleure performance empirique ; le cadre bayésien est acceptable | Aucun (graine seulement) | Echantillonnage a posteriori |

**Utilisez les bandits plutôt que les tests A/B quand :**
- Vous ne pouvez pas vous permettre de gaspiller du trafic sur de mauvaises variantes pendant un long test.
- De nouvelles variantes sont ajoutées ou retirées au fil du temps.
- Vous souhaitez vous adapter à des distributions de récompenses non stationnaires.

**Utilisez les tests A/B plutôt que les bandits quand :**
- Vous avez besoin d'une significativité statistique stricte (p-valeurs, intervalles de confiance).
- Les exigences réglementaires imposent un protocole fixe.

## Paramètres clés

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `n_arms` | `usize` | -- | Nombre d'options parmi lesquelles choisir |
| `epsilon` (EpsilonGreedy) | `f64` | -- | Taux d'exploration : 0.0 = glouton pur, 1.0 = aléatoire pur. Typique : 0.05--0.15 |
| `seed` (EpsilonGreedy, Thompson) | `u64` | -- | Graine RNG pour la reproductibilité. UCB1 est déterministe, pas besoin de graine |

**Champs consultables après exécution :**

| Champ | Type | Disponible sur | Signification |
|---|---|---|---|
| `q_values` | `Vec<f64>` | EpsilonGreedy, UCB1 | Récompense moyenne courante par bras |
| `counts` | `Vec<usize>` | Les trois | Nombre de fois que chaque bras a été tiré |
| `total_count` | `usize` | UCB1 | Total des tirages sur tous les bras |
| `means` | `Vec<f64>` | ThompsonSampling | Moyenne a posteriori par bras |
| `variances` | `Vec<f64>` | ThompsonSampling | Variance a posteriori par bras (diminue avec l'accumulation d'observations) |

## Pièges

1. **Epsilon trop élevé = trop d'exploration.** Avec epsilon = 0.5, la moitié de votre trafic va vers des bras aléatoires indéfiniment. Commencez autour de 0.1 et envisagez de le décroître au fil du temps (pas intégré ; vous ajusteriez manuellement `bandit.epsilon` entre les tours).

2. **UCB1 explore agressivement au début.** Il doit jouer chaque bras au moins une fois avant d'utiliser la formule de confiance. Avec 100 bras, les 100 premiers tirages sont purement exploratoires.

3. **Thompson Sampling suppose des récompenses gaussiennes.** Cette implémentation utilise un a posteriori Normal, qui fonctionne bien pour les récompenses continues (taux de clics, revenus). Pour les récompenses binaires (clic/pas de clic), un modèle Beta-Bernoulli serait plus correct théoriquement, mais l'approximation gaussienne fonctionne bien en pratique avec suffisamment de données.

4. **Environnements non stationnaires.** Les trois algorithmes calculent une moyenne courante, qui pondère les observations anciennes aussi fortement que les récentes. Si les vrais taux de récompense changent au fil du temps (ex. effets saisonniers), les algorithmes s'adaptent lentement. Envisagez une fenêtre glissante ou une décroissance exponentielle (modifiez l'étape de mise à jour manuellement).

5. **Les bras à égalité perturbent la sélection gloutonne.** Si deux bras ont des Q-valeurs identiques, `select_arm()` choisit toujours celui avec l'indice le plus bas. C'est déterministe mais peut ne pas être ce que vous souhaitez.

## Pour aller plus loin

- **Bandits contextuels :** Choisir les bras en fonction des caractéristiques de l'utilisateur (âge, localisation). Pas encore implémenté, mais vous pourriez combiner la sélection de bandit avec un vecteur de caractéristiques d'`ix-supervised`.
- **Approfondissement exploration vs. exploitation :** Voir [exploration-vs-exploitation.md](./exploration-vs-exploitation.md) pour une comparaison conceptuelle des trois stratégies.
- **Epsilon décroissant :** Enveloppez le bandit dans une boucle qui réduit `epsilon` au fil du temps : `bandit.epsilon = 1.0 / (round as f64).sqrt()`.
- **Analyse du regret :** UCB1 a une borne de regret prouvable en O(sqrt(T * K * ln(T))). Thompson Sampling l'égale ou le bat souvent empiriquement.
- **Bandits non stationnaires :** Utilisez un taux d'apprentissage fixe au lieu de 1/n : remplacez la règle de mise à jour par `Q(a) <- Q(a) + alpha * (r - Q(a))` pour un alpha constant (ex. 0.1).
