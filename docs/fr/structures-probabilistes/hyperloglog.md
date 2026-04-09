# HyperLogLog

## Le problème

Vous gérez une plateforme d'analytique et devez rapporter le nombre de visiteurs uniques par page, chaque jour. Votre site voit 200 millions de pages vues par jour. Stocker chaque identifiant de visiteur dans un `HashSet` consommerait des gigaoctets de RAM par page. Pire, en fin de journée vous devez fusionner les comptages par serveur en un total global — et l'union d'ensembles sur d'immenses hash sets est lente. Vous avez besoin d'une structure de données capable d'estimer le nombre d'éléments distincts dans un flux en utilisant une quantité de mémoire **fixe et minuscule**, et qui supporte la fusion entre noeuds.

Ce schéma se retrouve dans d'autres contextes :

- **Comptage de requêtes uniques.** Combien de requêtes de recherche distinctes votre système a-t-il traitées aujourd'hui ?
- **Surveillance réseau.** Combien d'IP sources uniques ont contacté ce serveur dans la dernière heure ?
- **Estimation de cardinalité en base de données.** Les planificateurs de requêtes utilisent HLL en interne pour estimer le nombre de valeurs distinctes dans une colonne, ce qui guide les décisions d'ordre de jointure.

---

## L'intuition

Lancez une pièce à répétition et notez la plus longue série de faces. Si vous voyez 10 faces d'affilée, vous pouvez estimer que vous avez probablement lancé la pièce environ 2^10 = 1 024 fois. Une seule expérience est bruitée, mais si vous exécutez beaucoup d'expériences en parallèle et moyennez les résultats, l'estimation se resserre.

HyperLogLog fait exactement cela, mais avec des hachages au lieu de pièces :

1. Hacher chaque élément en un nombre de 64 bits d'apparence aléatoire.
2. Utiliser les premiers bits pour choisir l'un des `m` « seaux » (expériences).
3. Compter les zéros en tête dans les bits restants — c'est la « plus longue série de faces ».
4. Chaque seau ne retient que le maximum de zéros en tête jamais vu.

Pour estimer la cardinalité, on calcule la **moyenne harmonique** de `2^(max zéros en tête)` sur tous les seaux, puis on applique un facteur de correction.

Point clé : **chaque seau stocke un seul octet** (le maximum de zéros en tête, qui ne dépasse jamais 64). La structure de données entière en précision standard (p=14) ne fait que 16 Ko, que vous comptiez 1 000 ou 1 milliard d'éléments distincts.

---

## Comment ça fonctionne

| Symbole | Signification |
|--------|---------|
| `p` | Paramètre de précision (4 à 18) |
| `m = 2^p` | Nombre de seaux (registres) |
| `alpha_m` | Constante de correction du biais |

### Ajout

```
hash = hash64(élément)
seau = hash & (m - 1)            // premiers p bits
restant = hash >> p
zéros_en_tête = clz(restant) + 1  // compter les zéros en tête
registres[seau] = max(registres[seau], zéros_en_tête)
```

### Comptage (estimation de cardinalité)

```
estimation_brute = alpha_m * m^2 / sum(2^(-registre[i]) pour tout i)
```

**En clair :** on prend la moyenne harmonique des estimations par seau, on multiplie par un facteur de correction, et on obtient le nombre approximatif d'éléments distincts. Pour les petites estimations (inférieures à 2,5 × m), une « correction pour petite plage » est appliquée en utilisant le nombre de registres à zéro.

### Taux d'erreur

```
erreur_standard = 1.04 / sqrt(m)
```

**En clair :** avec p=14 (16 384 seaux), l'erreur typique est d'environ 0,81 %. Doubler le nombre de seaux (ajouter un à `p`) réduit l'erreur d'environ 30 %, au prix d'un doublement de mémoire.

### Fusion

Pour fusionner deux instances HLL de même précision, on prend le maximum élément par élément de leurs tableaux de registres. Cela fonctionne car le maximum de zéros en tête pour un seau donné est le même quel que soit le noeud qui l'a observé.

---

## En Rust

### Compter les visiteurs uniques

```rust
use ix_probabilistic::hyperloglog::HyperLogLog;

// Précision standard : p=14, ~16 Ko de mémoire, ~0,81% d'erreur
let mut hll = HyperLogLog::standard();

// Simuler des identifiants de visiteurs arrivant d'un flux
for visitor_id in 0..100_000u64 {
    hll.add(&visitor_id);
}

// Ajouter le même visiteur à nouveau n'augmente pas le compteur
for visitor_id in 0..50_000u64 {
    hll.add(&visitor_id);  // les doublons sont absorbés
}

let estimate = hll.count();
println!("Visiteurs uniques estimés : {:.0}", estimate);  // ~100 000
println!("Taux d'erreur : {:.2}%", hll.error_rate() * 100.0); // ~0,81%
println!("Mémoire utilisée : {} octets", hll.memory_bytes());     // 16 384
```

### Précision personnalisée pour différentes charges

```rust
use ix_probabilistic::hyperloglog::HyperLogLog;

// Peu de mémoire (256 octets), ~6,5% d'erreur -- bon pour les estimations grossières
let mut hll_small = HyperLogLog::new(8);

// Haute précision (262 144 octets), ~0,41% d'erreur -- quand la précision compte
let mut hll_precise = HyperLogLog::new(18);
```

### Fusion entre noeuds distribués

```rust
use ix_probabilistic::hyperloglog::HyperLogLog;

let mut server_a = HyperLogLog::new(12);
let mut server_b = HyperLogLog::new(12);

// Chaque serveur voit des visiteurs différents (avec un chevauchement)
for i in 0..5_000u64    { server_a.add(&i); }
for i in 3_000..8_000u64 { server_b.add(&i); }

// Fusionner en un comptage global
server_a.merge(&server_b).expect("même précision requise");
let total_unique = server_a.count();
println!("Total uniques sur les deux serveurs : {:.0}", total_unique); // ~8 000
```

---

## Quand l'utiliser

| Situation | HyperLogLog | `HashSet` | Filtre de Bloom | Count-Min Sketch |
|-----------|:-----------:|:---------:|:------------:|:----------------:|
| Compter les éléments distincts | Oui | Oui (exact) | Non | Non |
| Mémoire fixe quelle que soit la cardinalité | Oui | Non | Oui | Oui |
| Vérifier l'appartenance d'un élément spécifique | Non | Oui | Oui | Non |
| Estimer la fréquence d'un élément spécifique | Non | Oui | Non | Oui |
| Fusion entre noeuds | Facile (max des registres) | Coûteuse (union d'ensembles) | Facile (OU bit à bit) | Facile (addition des tables) |
| Erreur < 1 % en 16 Ko | Oui | N/A | N/A | N/A |

**Règle pratique :** utilisez HyperLogLog quand la question est « combien de choses *différentes* ai-je vues ? » plutôt que « ai-je vu *cette chose précise* ? » ou « combien de fois ai-je vu *cette chose précise* ? »

---

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Recommandation |
|-----------|-----------------|----------|
| `precision` (p) | Compromis entre mémoire et précision | p=14 (`standard()`) est le point idéal pour la plupart des charges |
| Mémoire | `2^p` octets | p=10 donne ~1 Ko ; p=14 donne ~16 Ko ; p=18 donne ~256 Ko |
| Taux d'erreur | `1.04 / sqrt(2^p)` | p=10 : ~3,25 % ; p=14 : ~0,81 % ; p=18 : ~0,41 % |

### Référence rapide

| Précision (p) | Registres | Mémoire | Erreur typique |
|:-:|:-:|:-:|:-:|
| 4 | 16 | 16 o | 26 % |
| 8 | 256 | 256 o | 6,5 % |
| 10 | 1 024 | 1 Ko | 3,25 % |
| 12 | 4 096 | 4 Ko | 1,63 % |
| **14** | **16 384** | **16 Ko** | **0,81 %** |
| 16 | 65 536 | 64 Ko | 0,41 % |
| 18 | 262 144 | 256 Ko | 0,20 % |

---

## Pièges courants

1. **Les petites cardinalités sont bruitées.** Si le vrai comptage est inférieur à environ `2,5 × m`, l'estimation brute par moyenne harmonique est biaisée. L'implémentation applique une correction par « comptage linéaire » en utilisant le nombre de registres vides, mais les estimations en dessous de quelques centaines d'éléments auront une erreur relative plus élevée.

2. **La fusion nécessite la même précision.** `merge()` renvoie `Err` si les deux instances ont des valeurs de `p` différentes. Dans un système distribué, standardisez sur une précision unique pour tous les noeuds.

3. **Impossible d'interroger un élément spécifique.** HLL répond à « combien d'éléments distincts ? » mais pas à « l'élément X est-il dans l'ensemble ? ». Pour cela, utilisez un [filtre de Bloom](./filtres-de-bloom.md).

4. **Impossible de supprimer des éléments.** Une fois qu'un hachage a mis à jour un registre, il n'y a pas moyen de l'annuler. Si vous devez suivre des éléments qui apparaissent et disparaissent, envisagez de maintenir des HLL par fenêtre temporelle et d'expirer les anciens.

5. **Collisions de hachage sur de petites entrées.** Les éléments qui produisent le même hachage 64 bits sont indistinguables. Pour les charges typiques, c'est négligeable (probabilité de collision ~1 sur 2^64), mais soyez vigilant quand vous comptez des éléments tirés d'un très petit alphabet.

---

## Pour aller plus loin

- **HyperLogLog++ (amélioration de Google)** ajoute une correction de biais pour les petites et moyennes cardinalités et utilise une représentation creuse pour les registres majoritairement nuls. C'est le standard dans les systèmes analytiques en production.
- Le **HLL à fenêtre glissante** maintient plusieurs esquisses HLL pour des fenêtres temporelles chevauchantes, permettant de répondre à « combien de visiteurs uniques dans la dernière heure ? » sans stocker les horodatages individuels.
- **Opérations ensemblistes.** L'opération de fusion calcule l'union. La cardinalité de l'intersection peut être *estimée* via le principe d'inclusion-exclusion : `|A inter B| ~ |A| + |B| - |A union B|`, bien que cette estimation ait une erreur relative élevée quand l'intersection est beaucoup plus petite que chaque ensemble.
- Combinez avec un [Count-Min Sketch](./count-min-sketch.md) pour répondre à la fois « combien d'éléments distincts ? » et « combien de fois chaque élément apparaît-il ? » — deux vues complémentaires du même flux de données.
