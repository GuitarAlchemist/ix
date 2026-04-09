# Filtres coucou

## Le problème

Vous gérez un magasin de sessions temps réel pour une application de messagerie. Quand un utilisateur se connecte, vous l'ajoutez ; quand il se déconnecte, vous le retirez. Avant de router un message, vous devez vérifier « cet utilisateur est-il actuellement en ligne ? ». Un filtre de Bloom gère parfaitement l'ajout et la vérification, mais il ne peut pas gérer la *suppression* — une fois un bit activé, il le reste pour toujours. Vous avez besoin d'une structure de test d'appartenance qui supporte **l'insertion, la recherche et la suppression**, le tout en temps constant avec une mémoire constante par élément.

Ce schéma se retrouve dans d'autres contextes :

- **Listes blanches/noires de pare-feu avec révocation.** Bloquer une IP, puis la débloquer plus tard sans reconstruire le filtre.
- **Déploiement de feature flags.** Ajouter des identifiants utilisateur à un filtre « bêta », puis les retirer quand le déploiement est terminé.
- **Déduplication distribuée avec corrections.** Marquer un événement comme traité, puis annuler le marquage si le traitement a échoué.

---

## L'intuition

Un filtre coucou est comme un immeuble d'appartements où chaque logement peut accueillir jusqu'à 4 locataires (empreintes). Quand un nouveau locataire arrive :

1. Calculer l'**empreinte** du locataire (un hachage court) et deux numéros d'**appartement** possibles (indices de seaux).
2. Si l'un des appartements a une place libre, s'y installer.
3. Si les deux sont pleins, frapper à la porte d'un appartement, expulser un locataire existant au hasard et prendre sa place. Le locataire expulsé essaie alors *son* appartement alternatif. Cette chaîne d'expulsions continue jusqu'à ce que quelqu'un trouve une place libre ou qu'on atteigne une limite de coups (l'immeuble est trop plein).

Pour supprimer, trouver l'empreinte du locataire dans l'un des deux appartements candidats et la retirer.

Cette stratégie de « hachage coucou » — nommée d'après l'oiseau coucou qui pond ses oeufs dans les nids d'autres oiseaux — donne son nom au filtre.

Point clé : **les suppressions sont possibles car on stocke des empreintes, pas de simples bits.** Le compromis est un peu plus de mémoire par élément qu'un filtre de Bloom, et la possibilité que les insertions échouent quand le filtre approche de sa capacité.

---

## Comment ça fonctionne

| Concept | Détail |
|---------|--------|
| Empreinte | Un hachage 16 bits de l'élément (non nul) |
| Seau | Un petit tableau (capacité 4) d'empreintes |
| Nombre de seaux | Prochaine puissance de deux >= `capacity / 4` |
| Index alternatif | `i XOR hash(empreinte)` (permet de calculer l'autre seau depuis l'un ou l'autre) |

### Insertion

```
fp = empreinte(élément)
i1 = hash(élément) % nb_seaux
i2 = i1 XOR hash(fp) % nb_seaux

si seau[i1] a de la place -> stocker fp là
sinon si seau[i2] a de la place -> stocker fp là
sinon -> expulser une entrée au hasard de seau[i1], la relocaliser, répéter jusqu'à max_kicks fois
```

**En clair :** essayer les deux seaux candidats. Si aucun n'a de place, jouer aux chaises musicales avec les empreintes existantes jusqu'à ce que quelqu'un trouve un emplacement. Si personne n'en trouve après 500 coups, le filtre est trop plein — renvoyer `false`.

### Recherche

```
fp = empreinte(élément)
i1 = hash(élément) % nb_seaux
i2 = i1 XOR hash(fp) % nb_seaux

renvoyer seau[i1].contient(fp) OU seau[i2].contient(fp)
```

### Suppression

```
fp = empreinte(élément)
i1 = hash(élément) % nb_seaux
i2 = i1 XOR hash(fp) % nb_seaux

si seau[i1] contient fp -> le retirer, renvoyer true
sinon si seau[i2] contient fp -> le retirer, renvoyer true
sinon -> renvoyer false
```

**En clair :** trouver l'empreinte dans l'un des deux seaux candidats et la retirer. Si elle n'est pas trouvée, l'élément n'a jamais été inséré (ou a déjà été supprimé).

---

## En Rust

### Suivi de sessions avec suppression

```rust
use ix_probabilistic::cuckoo::CuckooFilter;

// Créer un filtre dimensionné pour ~1 000 sessions
let mut sessions = CuckooFilter::new(1000);

// Un utilisateur se connecte
assert!(sessions.insert(&"user-alice"));
assert!(sessions.insert(&"user-bob"));

// Vérifier qui est en ligne
assert!(sessions.contains(&"user-alice"));  // true
assert!(sessions.contains(&"user-bob"));    // true
assert!(!sessions.contains(&"user-eve"));   // false (jamais ajoutée)

// Un utilisateur se déconnecte
assert!(sessions.remove(&"user-alice"));    // true (trouvée et retirée)
assert!(!sessions.contains(&"user-alice")); // false (plus présente)

println!("Sessions actives : {}", sessions.len());          // 1
println!("Facteur de charge : {:.2}", sessions.load_factor());   // bas
```

### Gérer un filtre plein

```rust
use ix_probabilistic::cuckoo::CuckooFilter;

let mut cf = CuckooFilter::new(100);

let mut inserted = 0;
for i in 0..200 {
    if cf.insert(&i) {
        inserted += 1;
    } else {
        println!("Filtre plein après {} insertions", inserted);
        break;
    }
}
// Accommode typiquement 85-95% de la capacité théorique
```

---

## Quand l'utiliser

| Situation | Filtre coucou | Filtre de Bloom | `HashSet` |
|-----------|:-------------:|:------------:|:---------:|
| Insertion + recherche | Oui | Oui | Oui |
| Suppression | Oui | Non | Oui |
| Efficacité mémoire | Bonne | Optimale | La pire |
| Faux positifs | Possibles | Possibles | Aucun |
| Faux négatifs | Aucun | Aucun | Aucun |
| Fusion / union | Non supportée | OU bit à bit | Union d'ensembles |
| Échoue quand plein | Oui (insert renvoie false) | Non (le taux FP augmente) | Non (grandit simplement) |

**Règle pratique :** si vous avez besoin de suppression, utilisez un filtre coucou. Si vous ne supprimez jamais et voulez la plus petite empreinte mémoire possible ou avez besoin d'union distribuée, utilisez un [filtre de Bloom](./filtres-de-bloom.md). Si vous avez besoin d'exactitude, utilisez un `HashSet`.

---

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Recommandation |
|-----------|-----------------|----------|
| `capacity` | Nombre approximatif d'éléments que le filtre peut contenir | Le constructeur arrondit à une puissance de deux pour le nombre de seaux. Attendez-vous à ~85-95 % de capacité utilisable |
| Taille de seau | Éléments par seau (codé en dur à 4) | 4 est la valeur empiriquement optimale du papier original |
| Max kicks | Tentatives de relocalisation avant de déclarer « plein » (500) | Des valeurs plus élevées permettent quelques éléments de plus mais ralentissent les insertions en pire cas |
| Taille d'empreinte | 16 bits (codé en dur) | Détermine le taux de faux positifs : ~1/(2 × taille_seau × 2^f) où f=16 |

---

## Pièges courants

1. **Supprimer un élément jamais inséré peut corrompre le filtre.** Si les éléments A et B partagent par hasard une empreinte dans le même seau, supprimer B retirera l'empreinte de A. Ne supprimez que des éléments dont vous êtes certain qu'ils ont été insérés.

2. **Les insertions en double posent problème.** Insérer le même élément deux fois stocke deux copies de son empreinte. Un seul appel à `remove()` ne retire qu'une copie. Si vous avez besoin d'insertions idempotentes, vérifiez `contains()` avant d'insérer.

3. **L'insertion peut échouer.** Contrairement à un filtre de Bloom (qui dégrade gracieusement en augmentant le taux de FP), un filtre coucou a une limite de capacité stricte. Quand `insert()` renvoie `false`, vous devez soit supprimer des éléments existants, soit créer un filtre plus grand.

4. **Pas d'opération d'union.** Vous ne pouvez pas fusionner deux filtres coucou comme vous pouvez faire un OU sur deux filtres de Bloom. Pour les cas distribués, chaque noeud doit maintenir son propre filtre, et les requêtes doivent être envoyées en éventail.

5. **Collisions d'empreintes.** Deux éléments différents avec la même empreinte et les mêmes seaux candidats sont indistinguables. C'est la source des faux positifs. Avec des empreintes de 16 bits et 4 entrées par seau, le taux de FP théorique est d'environ 0,0012 % (1 sur 83 000).

---

## Pour aller plus loin

- Les **filtres coucou semi-triés** trient les empreintes au sein de chaque seau, permettant une meilleure compression et des taux de FP plus bas.
- Les **filtres coucou vacuum** réduisent le nombre moyen de coups lors de l'insertion en maintenant des données auxiliaires sur l'occupation des seaux.
- Le papier original, « Cuckoo Filter: Practically Better Than Bloom » (Fan et al., 2014), fournit l'analyse théorique et les benchmarks comparant avec les filtres de Bloom et les filtres de Bloom compteurs.
- Pour l'estimation de fréquence plutôt que l'appartenance, voir le [Count-Min Sketch](./count-min-sketch.md). Pour l'estimation de cardinalité, voir [HyperLogLog](./hyperloglog.md).
