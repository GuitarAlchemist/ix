# Filtres de Bloom

## Le problème

Vous gérez un navigateur web qui doit avertir les utilisateurs avant qu'ils visitent une URL malveillante. Votre liste noire contient 10 millions d'URL connues comme dangereuses. Vous pourriez les stocker dans un `HashSet`, mais cela consomme des centaines de mégaoctets de RAM — et la liste doit être vérifiée à chaque navigation. Vous avez besoin d'une structure de données capable de répondre à « cette URL est-elle dans la liste noire ? » en temps constant, avec une fraction de la mémoire, et vous acceptez un taux minime de fausses alertes (signaler un site sûr) tant que vous ne manquez **jamais** un site véritablement dangereux.

Ce schéma se retrouve dans d'autres contextes :

- **Vérification de cache.** Avant d'interroger une base de données lente, vérifier si la clé *pourrait* exister. Si le filtre de Bloom dit « non », on évite la requête.
- **Déduplication d'événements.** Un pipeline en streaming reçoit des millions d'événements par seconde. Un filtre de Bloom indique instantanément si un identifiant d'événement a déjà été traité.
- **Correcteurs orthographiques et registres d'identifiants.** Un premier passage rapide qui dit « ce mot est *probablement* dans le dictionnaire » ou « cet identifiant est *probablement* pris ».

---

## L'intuition

Imaginez un mur de 1 000 interrupteurs, tous en position OFF au départ. Quand vous voulez *mémoriser* un élément, vous le passez à travers trois fonctions de hachage différentes. Chacune vous donne un numéro d'interrupteur, et vous basculez ces trois interrupteurs sur ON.

Plus tard, quand vous voulez vérifier si un élément est dans l'ensemble, vous le hachez de la même façon et regardez les trois interrupteurs. Si **l'un d'eux** est OFF, l'élément n'a définitivement jamais été ajouté. Si **les trois** sont ON, l'élément a *probablement* été ajouté — mais il y a une petite chance que d'autres éléments aient basculé ces mêmes interrupteurs.

Point clé : **les faux positifs sont possibles, les faux négatifs sont impossibles.** Si le filtre dit « non », c'est non. S'il dit « oui », c'est « probablement oui ».

Les mathématiques permettent d'échanger mémoire contre précision. Plus de bits et plus de fonctions de hachage font baisser le taux de faux positifs au niveau souhaité.

---

## Comment ça fonctionne

Un filtre de Bloom a deux paramètres :

| Symbole | Signification |
|--------|---------|
| `m` | Nombre de bits dans le tableau de bits |
| `k` | Nombre de fonctions de hachage indépendantes |

### Insertion

Pour chaque élément, calculer `k` valeurs de hachage, chacune dans l'intervalle `[0, m)`. Mettre ces bits à 1.

### Requête

Calculer les mêmes `k` hachages. Si chaque bit correspondant est à 1, renvoyer `true` (probablement présent). Si un bit est à 0, renvoyer `false` (définitivement absent).

### Dimensionnement optimal

Étant donné `n` éléments et un taux de faux positifs souhaité `p` :

```
m = -(n * ln(p)) / (ln2)^2
k = (m / n) * ln2
```

**En clair :** le nombre de bits croît linéairement avec le nombre d'éléments prévus, et logarithmiquement avec la rigueur souhaitée du taux de faux positifs. Un taux de 1 % pour 1 million d'éléments coûte environ 1,2 Mo.

### Taux de faux positifs estimé au remplissage courant

```
Taux_FP ~ (fraction_de_bits_activés) ^ k
```

**En clair :** plus il y a de bits activés, plus il est probable qu'une requête aléatoire tombe sur les `k` bits par accident.

### Union

Deux filtres de Bloom avec les **mêmes `m` et `k`** peuvent être fusionnés par un OU bit à bit. Le résultat contient tous les éléments des deux filtres. Cela rend les filtres de Bloom parfaits pour les systèmes distribués où chaque noeud maintient un filtre local qu'on fusionne périodiquement.

---

## En Rust

> Exemple complet exécutable : [`examples/probabilistic/bloom_filter.rs`](../../examples/probabilistic/bloom_filter.rs)

### Usage de base — liste noire d'URL

```rust
use ix_probabilistic::bloom::BloomFilter;

// Créer un filtre dimensionné pour 10 000 URL à 1% de faux positifs.
// La bibliothèque calcule la taille optimale du tableau de bits et le nombre de hachages.
let mut blocklist = BloomFilter::new(10_000, 0.01);

// Peupler avec les URL malveillantes connues
blocklist.insert(&"malicious-site.com");
blocklist.insert(&"phishing-page.net");
blocklist.insert(&"scam-offer.org");

// Vérifier les URL entrantes
assert!(blocklist.contains(&"malicious-site.com"));   // true  -- dans l'ensemble
assert!(!blocklist.contains(&"safe-site.org"));         // false -- définitivement absent

// Inspecter le filtre
println!("Éléments insérés :   {}", blocklist.len());          // 3
println!("Taille tableau bits : {}", blocklist.bit_size());     // ~95 851
println!("Taux FP estimé :     {:.6}", blocklist.estimated_fp_rate());
```

### Paramètres manuels

```rust
use ix_probabilistic::bloom::BloomFilter;

// Quand vous savez exactement combien de bits et de hachages vous voulez
let mut bf = BloomFilter::with_params(1024, 5);
bf.insert(&42u64);
assert!(bf.contains(&42u64));
```

### Fusion de filtres distribués

```rust
use ix_probabilistic::bloom::BloomFilter;

let mut node_a = BloomFilter::with_params(10_000, 7);
let mut node_b = BloomFilter::with_params(10_000, 7);

node_a.insert(&"event-001");
node_b.insert(&"event-002");

// Fusionner en un seul filtre connaissant les deux événements
let merged = node_a.union(&node_b).expect("mêmes paramètres requis");
assert!(merged.contains(&"event-001"));
assert!(merged.contains(&"event-002"));
```

---

## Quand l'utiliser

| Situation | Filtre de Bloom | `HashSet` | Filtre coucou |
|-----------|:-----------:|:---------:|:-------------:|
| Test d'appartenance avec contrainte mémoire | Optimal | Le pire | Bon |
| Zéro faux négatif nécessaire | Oui | Oui | Oui |
| Zéro faux positif nécessaire | Non | Oui | Non |
| Besoin de suppression | Non | Oui | Oui |
| Fusion distribuée (union) | Facile | Coûteuse | Non supportée |
| Comptage de fréquences | Non | Non | Non |

**Règle pratique :** utilisez un filtre de Bloom quand vous avez besoin d'un portail rapide et économe en mémoire « définitivement absent / probablement présent » devant une recherche plus coûteuse.

---

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Recommandation |
|-----------|-----------------|----------|
| `capacity` | Nombre d'éléments attendus | Surestimez de 20-50 % pour garder le taux de FP bas au fur et à mesure du remplissage |
| `fp_rate` | Probabilité de faux positif cible | 0,01 (1 %) est un bon défaut ; 0,001 pour les cas plus stricts |
| `size` (manuel) | Nombre exact de bits | Utilisez `new()` pour laisser la bibliothèque le calculer |
| `num_hashes` (manuel) | Nombre de fonctions de hachage | Plus de hachages = taux FP plus bas mais insertion/requête plus lente |

---

## Pièges courants

1. **Impossible de supprimer des éléments.** Une fois un bit activé, il le reste. Si vous avez besoin de suppression, utilisez un [`CuckooFilter`](./filtres-coucou.md) à la place.

2. **Le sur-remplissage détruit la précision.** Si vous insérez bien plus d'éléments que `capacity`, le taux de faux positifs grimpe fortement. La méthode `estimated_fp_rate()` permet de surveiller cela en temps réel.

3. **L'union nécessite des paramètres identiques.** Appeler `union()` sur des filtres avec des `size` ou `num_hashes` différents renvoie `None`. Concevez votre système distribué pour que chaque noeud crée des filtres avec les mêmes arguments de construction.

4. **Les éléments ne sont pas récupérables.** Un filtre de Bloom indique l'appartenance, pas le contenu. Vous ne pouvez pas itérer sur ce qui a été inséré.

5. **La qualité du hachage compte.** L'implémentation actuelle utilise `DefaultHasher` avec mélange de graines. Face à des adversaires cryptographiques capables de forger des collisions, vous auriez besoin d'un hachage à clé. Pour les charges de travail ML/pipeline de données typiques, ce n'est pas un problème.

---

## Pour aller plus loin

- Les **filtres de Bloom compteurs** remplacent chaque bit par un compteur, permettant la suppression au prix d'une mémoire accrue. ix n'en inclut pas encore, mais le `CuckooFilter` couvre le cas d'usage de la suppression.
- Les **filtres de Bloom extensibles** ajoutent automatiquement de nouveaux tableaux de bits quand le niveau de remplissage monte, maintenant un taux de FP cible sans connaître le nombre d'éléments à l'avance.
- Le **Count-Min Sketch** ([document suivant](./count-min-sketch.md)) résout un problème connexe mais différent : estimer *combien de fois* un élément apparaît, pas seulement s'il existe.
- Les filtres de Bloom se marient naturellement avec le cache embarqué [`ix-cache`](../../crates/ix-cache) : utilisez le filtre comme portail d'admission rapide avant l'écriture dans le cache, évitant de le polluer avec des éléments vus une seule fois.
