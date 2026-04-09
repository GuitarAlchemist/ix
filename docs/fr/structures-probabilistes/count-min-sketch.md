# Count-Min Sketch

## Le problème

Vous exploitez un système de surveillance réseau qui voit 500 000 paquets par seconde. Vous devez répondre à des questions comme « quelles IP sources envoient le plus de trafic ? » et « cette IP est-elle en train de monter en flèche ? ». Stocker un compteur exact pour chaque adresse IP jamais vue nécessiterait une mémoire illimitée — et la plupart des IP n'apparaissent qu'une ou deux fois. Vous avez besoin d'une structure de taille fixe capable d'estimer les fréquences à la volée, avec une erreur bornée et prévisible.

Ce schéma se retrouve dans d'autres contextes :

- **Détection de heavy hitters.** Trouver les N endpoints d'API les plus fréquemment appelés sans journaliser chaque requête.
- **Politiques d'admission en cache.** Ne promouvoir un élément dans le cache qu'après qu'il a été demandé plus de *k* fois. Un Count-Min Sketch suit les comptages approximatifs en espace constant.
- **Fréquence de termes en NLP.** Estimer la fréquence de chaque mot dans un corpus trop volumineux pour tenir dans une table de hachage.

---

## L'intuition

Imaginez que vous avez cinq rangées de seaux numérotés. Quand un paquet arrive de l'IP `10.0.0.1`, vous exécutez cinq fonctions de hachage différentes — une par rangée — et déposez une bille dans le seau indiqué par chaque hachage. Pour estimer combien de paquets proviennent de cette IP, vous regardez les cinq seaux et prenez le **minimum** des comptages de billes.

Pourquoi le minimum ? Parce que les seaux peuvent partager des billes avec *d'autres* IP (collisions de hachage). Le minimum est celui le moins gonflé par les collisions, donc il donne l'estimation la plus proche du vrai comptage.

Point clé : **l'esquisse ne fait que sur-compter, jamais sous-compter.** La fréquence réelle est toujours inférieure ou égale à l'estimation.

---

## Comment ça fonctionne

Un Count-Min Sketch est un tableau 2D avec `depth` lignes et `width` colonnes.

| Symbole | Signification |
|--------|---------|
| `w` | Largeur (colonnes par ligne) |
| `d` | Profondeur (nombre de lignes / fonctions de hachage) |

### Ajout

Pour l'élément `x`, incrémenter `table[ligne][hash_ligne(x) % w]` pour chacune des `d` lignes.

### Estimation

Pour l'élément `x`, renvoyer `min sur toutes les lignes de table[ligne][hash_ligne(x) % w]`.

### Dimensionnement à partir des bornes d'erreur

Étant donné un facteur d'erreur souhaité `epsilon` et une probabilité d'échec `delta` :

```
w = ceil(e / epsilon)
d = ceil(ln(1 / delta))
```

**En clair :** la largeur contrôle la précision (plus large = moins de sur-comptage) et la profondeur contrôle la confiance (plus de lignes = probabilité plus élevée qu'au moins une ligne donne une estimation serrée). Une esquisse avec `epsilon = 0,001` et `delta = 0,01` utilise environ 2 718 colonnes et 5 lignes — environ 54 Ko pour des compteurs `u64`.

### Borne d'erreur

```
estimation(x) <= vrai_comptage(x) + epsilon * comptage_total
```

**En clair :** le sur-comptage sur un élément donné est au plus une petite fraction (`epsilon`) du nombre total d'éléments jamais insérés, et cette garantie tient avec une probabilité d'au moins `1 - delta`.

---

## En Rust

### Suivi de fréquence basique

```rust
use ix_probabilistic::count_min::CountMinSketch;

// 100 colonnes, 5 fonctions de hachage
let mut sketch = CountMinSketch::new(100, 5);

// Enregistrer les observations
for _ in 0..1000 { sketch.add(&"GET /api/users"); }
for _ in 0..50  { sketch.add(&"GET /api/admin"); }
for _ in 0..3   { sketch.add(&"DELETE /api/users/42"); }

// Estimer les fréquences (toujours >= au vrai comptage)
println!("/api/users :  ~{}", sketch.estimate(&"GET /api/users"));    // >= 1000
println!("/api/admin :  ~{}", sketch.estimate(&"GET /api/admin"));    // >= 50
println!("total :        {}", sketch.total_count());                   // 1053
```

### Dimensionnement à partir des exigences d'erreur

```rust
use ix_probabilistic::count_min::CountMinSketch;

// "Je veux des estimations à 1% du comptage total, 99% du temps"
let mut sketch = CountMinSketch::with_error(0.01, 0.01);

for _ in 0..10_000 { sketch.add(&"frequent"); }
for _ in 0..10     { sketch.add(&"rare"); }

let est = sketch.estimate(&"frequent");
// est >= 10 000 et est <= 10 000 + 0.01 * 10 010 (avec 99% de probabilité)
```

### Ajout de comptages spécifiques et fusion

```rust
use ix_probabilistic::count_min::CountMinSketch;

let mut sketch = CountMinSketch::new(200, 5);
sketch.add_count(&"batch-event", 500);  // Ajouter 500 d'un coup

// Fusionner des esquisses de deux noeuds de surveillance
let mut node_a = CountMinSketch::new(200, 5);
let mut node_b = CountMinSketch::new(200, 5);
node_a.add(&"error-503");
node_b.add(&"error-503");
node_b.add(&"error-503");

node_a.merge(&node_b).expect("mêmes dimensions requises");
assert!(node_a.estimate(&"error-503") >= 3);
```

---

## Quand l'utiliser

| Situation | Count-Min Sketch | `HashMap<K, u64>` | HyperLogLog |
|-----------|:----------------:|:------------------:|:-----------:|
| Estimer la fréquence d'éléments spécifiques | Oui | Oui (exact) | Non |
| Mémoire fixe quel que soit le nombre d'éléments | Oui | Non | Oui |
| Compter les éléments distincts | Non | Oui | Oui |
| Supporte la suppression / décrément | Non | Oui | Non |
| Fusion entre noeuds distribués | Oui | Coûteuse | Oui |
| L'erreur est toujours dans un sens (sur-comptage) | Oui | N/A | N/A |

**Règle pratique :** utilisez un Count-Min Sketch quand vous vous intéressez à *combien de fois* un élément spécifique est apparu, pas seulement *s'il* est apparu (filtre de Bloom) ou *combien d'éléments distincts* il y a (HyperLogLog).

---

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Recommandation |
|-----------|-----------------|----------|
| `width` | Colonnes par ligne ; contrôle la précision | Plus élevé = moins de sur-comptage. `ceil(e / epsilon)` pour la borne formelle |
| `depth` | Nombre de lignes de hachage ; contrôle la confiance | Plus élevé = estimation plus probablement serrée. `ceil(ln(1/delta))` |
| `epsilon` (via `with_error`) | Erreur maximale en fraction du comptage total | 0,01 (1 %) est un bon point de départ |
| `delta` (via `with_error`) | Probabilité que l'erreur dépasse epsilon | 0,01 (1 %) signifie 99 % de confiance |

---

## Pièges courants

1. **Le sur-comptage est proportionnel au total d'insertions.** Si vous avez inséré 10 millions d'éléments, même l'estimation d'un élément rare peut être gonflée jusqu'à `epsilon × 10 000 000`. Dimensionnez l'esquisse suffisamment large pour votre charge.

2. **Impossible de décrémenter ou supprimer.** L'esquisse ne supporte que les mises à jour additives. Si vous devez suivre des éléments qui apparaissent et disparaissent, associez-la à un mécanisme séparé.

3. **La fusion nécessite des dimensions identiques.** Les deux esquisses doivent avoir les mêmes `width` et `depth`. La méthode `merge()` renvoie `Err` si elles diffèrent.

4. **Les éléments rares sont noyés.** Si un élément représente 99 % du trafic, le sur-comptage des éléments rares peut approcher la borne d'erreur de l'esquisse. Envisagez de combiner avec une liste séparée de « heavy hitters » pour les éléments les plus fréquents.

5. **Ce n'est pas un test d'appartenance.** Un Count-Min Sketch renverra une estimation non nulle pour des éléments jamais insérés (à cause des collisions de hachage). Si vous avez besoin de « cet élément est-il présent ? » avec une garantie sur les faux négatifs, utilisez un [filtre de Bloom](./filtres-de-bloom.md).

---

## Pour aller plus loin

- **Détection de heavy hitters avec Count-Min + tas :** maintenir un tas minimum des top-k éléments. À chaque insertion, interroger l'esquisse et promouvoir l'élément dans le tas si son comptage estimé dépasse le k-ième plus grand actuel.
- Les **esquisses à fenêtre glissante** maintiennent plusieurs esquisses par fenêtre temporelle et expirent les anciennes, donnant des estimations « requêtes dans les 5 dernières minutes ».
- La **mise à jour conservative** n'incrémente que le(s) compteur(s) minimum(s) parmi les `d` lignes, réduisant le sur-comptage sans mémoire supplémentaire. Une amélioration future potentielle pour `ix-probabilistic`.
- Associez au cache embarqué [`ix-cache`](../../crates/ix-cache) : utilisez l'esquisse pour implémenter une politique d'admission TinyLFU — n'admettre dans le cache que les éléments dont la fréquence estimée dépasse un seuil.
