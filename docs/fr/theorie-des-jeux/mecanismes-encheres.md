# Mécanismes d'enchères

## Le problème

Vous gérez une plateforme publicitaire en ligne. Chaque fois qu'un utilisateur charge une page, vous avez une fraction de seconde pour décider quelle publicité afficher. Des dizaines d'annonceurs veulent cet emplacement, et chacun lui accorde une valeur différente selon le profil de l'utilisateur. Vous avez besoin d'un mécanisme qui :

1. Sélectionne la meilleure publicité efficacement.
2. Facture un prix juste.
3. Incite les annonceurs à enchérir honnêtement (pour éviter les jeux stratégiques inutiles).

Ou bien : un gouvernement attribue des licences de spectre radio valant des milliards. Les opérateurs télécoms vont enchérir, mais la conception de l'enchère détermine si le résultat est efficace, si les entreprises surenchérissent (malédiction du vainqueur) et combien de revenus le gouvernement collecte.

Ce sont tous des problèmes de **conception d'enchères** -- le choix des règles qui régissent la transformation des enchères en allocations et paiements.

## L'intuition

Une enchère est un jeu. Chaque enchérisseur a une valeur privée pour l'objet. Les règles de l'enchère déterminent qui gagne et ce qu'il paie. Différentes règles créent différentes incitations :

**Enchère scellée au premier prix :** chacun soumet une offre sous enveloppe. La plus haute enchère gagne et paie le montant de son offre. Le problème : vous devriez enchérir *en dessous* de votre vraie valeur (réduire votre enchère), car payer votre vraie valeur signifie un profit nul. De combien réduire ? Cela dépend de ce que vous pensez que les autres vont enchérir -- c'est un jeu de devinettes.

**Enchère scellée au second prix (Vickrey) :** mêmes enveloppes, mais le gagnant paie la *deuxième plus haute* enchère. Plus aucune raison de réduire : si vous enchérissez votre vraie valeur, soit vous gagnez et payez moins que votre valeur (profit !), soit vous perdez face à quelqu'un qui valorise davantage l'objet (pas de perte). L'enchère sincère est une *stratégie dominante*. C'est le mécanisme utilisé par Google pour les enchères publicitaires.

**Enchère anglaise (ascendante) :** le commissaire-priseur augmente le prix ; les enchérisseurs se retirent quand il dépasse leur valeur. Le dernier restant gagne et paie juste au-dessus de la deuxième plus haute valeur. Stratégiquement équivalente à l'enchère au second prix.

**Enchère tous-paient :** tout le monde paie son enchère, mais seul le plus offrant gagne. Cela semble injuste, mais modélise le lobbying, les campagnes politiques et les courses aux brevets où chacun dépense des ressources quel que soit le résultat.

Le **théorème d'équivalence des revenus** dit que sous des hypothèses standard (valeurs privées indépendantes, enchérisseurs neutres au risque), tous les formats d'enchères standard génèrent le même revenu espéré pour le vendeur. Les différences résident dans le risque, la complexité et les incitations stratégiques.

## Comment ça fonctionne

### Enchère au premier prix

```
winner = bidder with highest bid
payment = winner's bid
```

**En clair :** vous payez ce que vous avez enchéri. Stratégie optimale : réduire votre enchère en dessous de votre vraie valeur d'un facteur `(n-1)/n` où `n` est le nombre d'enchérisseurs (pour des valeurs uniformément distribuées).

### Enchère au second prix (Vickrey)

```
winner = bidder with highest bid
payment = second-highest bid
```

**En clair :** vous payez ce que le deuxième enchérisseur a offert. Stratégie optimale : enchérir exactement votre vraie valeur. C'est le résultat élégant du mécanisme de Vickrey -- la révélation sincère est une stratégie dominante.

### Enchère anglaise (ascendante)

```
price starts low and increases by fixed increments
bidders drop out when price exceeds their value
last remaining bidder wins at current price
```

**En clair :** le prix monte comme un ascenseur. Les enchérisseurs descendent quand il dépasse leur étage. Le gagnant paie l'étage où l'avant-dernier est descendu.

### Enchère tous-paient

```
winner = bidder with highest bid
all bidders pay their own bid (win or lose)
```

**En clair :** tout le monde dépense le montant de son enchère. Seul le plus offrant obtient quelque chose. Modélise les dépenses compétitives.

## En Rust

Le crate `ix-game` fournit les mécanismes d'enchères :

```rust
use ix_game::auction::{
    Bid, AuctionResult,
    first_price_auction, second_price_auction, all_pay_auction,
    english_auction, dutch_auction, revenue_equivalence_test,
};

fn main() {
    // --- Créer des enchères ---
    let bids = vec![
        Bid { bidder: 0, amount: 10.0 },
        Bid { bidder: 1, amount: 25.0 },
        Bid { bidder: 2, amount: 18.0 },
        Bid { bidder: 3, amount: 30.0 },
    ];

    // --- Enchère scellée au premier prix ---
    let fp: AuctionResult = first_price_auction(&bids).unwrap();
    println!("Premier prix : gagnant={}, paiement={:.0}", fp.winner, fp.payment);
    // gagnant=3, paiement=30 (paie sa propre enchère)

    // --- Second prix (Vickrey) ---
    let sp: AuctionResult = second_price_auction(&bids).unwrap();
    println!("Second prix : gagnant={}, paiement={:.0}", sp.winner, sp.payment);
    // gagnant=3, paiement=25 (paie la deuxième plus haute enchère)

    // --- Tous-paient ---
    let (winner, payments) = all_pay_auction(&bids).unwrap();
    println!("Tous-paient : gagnant={}, paiements={:?}", winner, payments);
    // gagnant=3, tout le monde paie son enchère

    // --- Enchère anglaise (ascendante) ---
    // Passer les valeurs (pas les enchères stratégiques), le prix de départ et l'incrément.
    let values = vec![10.0, 25.0, 18.0, 30.0];
    let eng: AuctionResult = english_auction(&values, 0.0, 1.0);
    println!("Anglaise : gagnant={}, paiement={:.0}", eng.winner, eng.payment);
    // gagnant=3, paiement ~ 26 (juste au-dessus de la deuxième plus haute valeur)

    // --- Enchère hollandaise (descendante) ---
    let dut: AuctionResult = dutch_auction(&values, 50.0, 1.0);
    println!("Hollandaise : gagnant={}, paiement={:.0}", dut.winner, dut.payment);
    // Le premier enchérisseur dont la valeur >= prix gagne

    // --- Test d'équivalence des revenus ---
    // Comparer les revenus moyens des enchères au premier et au second prix
    // sur de nombreux essais aléatoires (avec enchères optimales).
    let (fp_rev, sp_rev) = revenue_equivalence_test(
        5,      // 5 enchérisseurs
        10_000, // 10 000 essais
        42,     // graine RNG
    );
    println!("Revenu moyen : premier prix={:.4}, second prix={:.4}", fp_rev, sp_rev);
    // Ces valeurs devraient être approximativement égales (théorème d'équivalence des revenus).
}
```

### Types clés

```rust
pub struct Bid {
    pub bidder: usize,  // Identifiant de l'enchérisseur
    pub amount: f64,    // Montant de l'enchère
}

pub struct AuctionResult {
    pub winner: usize,       // ID du gagnant
    pub payment: f64,        // Ce que le gagnant paie
    pub all_bids: Vec<Bid>,  // Toutes les enchères soumises
}
```

### Résumé de l'API

| Fonction | Détermination du gagnant | Règle de paiement | Propriété stratégique |
|----------|--------------------|--------------|--------------------|
| `first_price_auction(&bids)` | Plus haute enchère | Enchère du gagnant | Réduire en dessous de la vraie valeur |
| `second_price_auction(&bids)` | Plus haute enchère | Deuxième plus haute enchère | L'enchère sincère est dominante |
| `all_pay_auction(&bids)` | Plus haute enchère | Tout le monde paie son enchère | Equilibre complexe |
| `english_auction(&values, start, incr)` | Dernier restant | Prix au départ de l'avant-dernier | Equivalent au second prix |
| `dutch_auction(&values, start, decr)` | Premier à accepter | Prix d'acceptation | Equivalent au premier prix |
| `revenue_equivalence_test(n, trials, seed)` | -- | -- | Vérification empirique du théorème |

Voir l'exemple complet : [examples/game-theory/auctions.rs](../../examples/game-theory/auctions.rs)

## Quand l'utiliser

| Scénario | Mécanisme recommandé | Pourquoi |
|----------|----------------------|-----|
| Enchère sincère souhaitée (stratégie simple) | Second prix / Anglaise | Stratégie dominante = enchérir sa vraie valeur |
| Maximiser le revenu | L'un ou l'autre (équivalence des revenus) | Même revenu espéré sous hypothèses standard |
| Enchérisseurs averses au risque | Premier prix | Les enchérisseurs averses au risque réduisent moins, augmentant le revenu |
| Résolution rapide | Enchère scellée (premier ou second prix) | Un seul tour, pas d'itération |
| Découverte des prix | Anglaise | Les enchérisseurs observent la disposition à payer des autres |
| Modélisation de dépenses compétitives | Tous-paient | Capture le lobbying, les courses à la R&D |
| Objets multiples | Généraliser (pas encore dans ix-game) | Mécanisme VCG, enchères combinatoires |

## Paramètres clés

| Paramètre | Type | Ce qu'il contrôle |
|-----------|------|-----------------|
| `bids` | `&[Bid]` | Enchères soumises. Pour premier/second prix, ce sont des enchères stratégiques. |
| `values` | `&[f64]` | Vraies valeurs privées (utilisées dans les simulations anglaise/hollandaise). |
| `start_price` | `f64` | Prix de départ pour les enchères anglaise (bas) ou hollandaise (haut). |
| `increment` / `decrement` | `f64` | Pas de prix. Plus petit = plus précis, simulation plus lente. |
| `num_bidders` (test de revenus) | `usize` | Plus d'enchérisseurs = plus de concurrence = revenu plus élevé. |

## Pièges

1. **Les enchères au premier prix ne sont PAS les vraies valeurs.** Si vous passez les vraies valeurs à `first_price_auction`, vous simulez des enchérisseurs naïfs. Les enchérisseurs rationnels réduisent leurs offres. La fonction `revenue_equivalence_test` gère cela correctement en appliquant le facteur de réduction optimal `(n-1)/n`.

2. **La résolution de l'enchère anglaise dépend de l'incrément.** Avec `increment = 1.0` et des valeurs de 25 et 30, le prix saute au-dessus de 25 et se stabilise à 26. Avec `increment = 0.01`, il se stabiliserait à 25.01. Des incréments plus petits sont plus précis mais plus lents.

3. **L'ordre compte dans l'enchère hollandaise.** L'implémentation vérifie les enchérisseurs par ordre d'indice. Le premier enchérisseur (par indice) dont la valeur dépasse le prix descendant gagne. Dans une vraie enchère hollandaise, tous les enchérisseurs observeraient le prix simultanément.

4. **L'enchère tous-paient retourne les paiements différemment.** Elle retourne `(winner, Vec<f64>)` où le vecteur est indexé par l'ID de l'enchérisseur, pas un `AuctionResult`. C'est parce que tout le monde paie, pas seulement le gagnant.

5. **L'équivalence des revenus repose sur des hypothèses.** Elle vaut pour des valeurs privées indépendantes, des enchérisseurs neutres au risque et sans contrainte de budget. En pratique, les enchères se comportent différemment : le premier prix génère plus de revenu lorsque les enchérisseurs sont averses au risque, et l'enchère anglaise en génère plus lorsque les valeurs sont corrélées.

## Pour aller plus loin

- **Mécanisme VCG (Vickrey-Clarke-Groves) :** Généralise les enchères au second prix aux objets multiples. Chaque gagnant paie l'externalité qu'il impose aux autres.
- **Enchères combinatoires :** Les enchérisseurs offrent sur des *lots* d'objets. Utilisées pour l'allocation du spectre et les créneaux d'atterrissage.
- **Equilibres de Nash dans les enchères :** Les stratégies d'enchères forment un jeu. Voir [Equilibres de Nash](./equilibres-de-nash.md) pour la théorie générale.
- **Conception de mécanismes :** Concevoir les *règles* pour que l'équilibre atteigne votre objectif (efficacité, revenu, équité). Les fonctions d'enchères d'ix-game sont des briques de base pour cela.
- Lecture : Milgrom, *Putting Auction Theory to Work* (2004) -- le guide pratique par un lauréat du Nobel.
