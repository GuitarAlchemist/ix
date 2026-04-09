# Dimensions fractales

## Le problème

Vous êtes géographe et mesurez la longueur d'un littoral. Vous commencez avec une règle
de 100 km et mesurez 500 km. Puis vous passez à une règle de 10 km, en suivant chaque
baie et anse, et mesurez 700 km. Une règle de 1 km donne 950 km. La longueur mesurée ne
cesse de croître à mesure que la règle rétrécit — elle ne converge jamais. C'est le
paradoxe de Richardson, et la raison est que les littoraux sont des fractales : leur
complexité existe à toutes les échelles.

Vous avez besoin d'un nombre unique qui capture « combien rugueux » ou « combien
remplissant l'espace » est un objet. Ce nombre est la dimension fractale. On la retrouve
en analyse de texture (imagerie satellite), en science des matériaux (rugosité de
surface), en analyse de la volatilité financière et en caractérisation d'attracteurs de
systèmes dynamiques.

## L'intuition

Une droite a la dimension 1. Un carré rempli a la dimension 2. Un littoral est *entre
les deux* : c'est plus qu'une ligne (il zigzague et occupe plus d'espace) mais moins
qu'une surface pleine. La dimension fractale D vous dit exactement à quel point il est
« entre les deux ».

- D ~ 1.0 : Presque une courbe lisse.
- D ~ 1.25 : Un littoral modérément découpé (Grande-Bretagne ~ 1.25).
- D ~ 1.5 : Très découpé, comme une marche aléatoire.
- D ~ 2.0 : Tellement contourné qu'il remplit presque le plan.

Les attracteurs étranges classiques ont aussi des dimensions non entières : l'attracteur
de Lorenz a D ~ 2.06 (légèrement plus qu'une feuille, à cause de sa stratification
fractale).

## Fonctionnement

### Dimension par comptage de boîtes (box-counting)

```
D_box = lim_{eps->0} log(N(eps)) / log(1/eps)
```

**En clair :** Couvrez l'ensemble avec des boîtes de côté eps. Comptez combien de boîtes
N(eps) contiennent au moins un point. Réduisez les boîtes et recommencez. Tracez
log(N) en fonction de log(1/eps) ; la pente de la droite de régression est la dimension
fractale.

### Dimension de corrélation

```
C(r) = (2 / N(N-1)) * #{paires telles que |x_i - x_j| < r}
D_corr = lim_{r->0} log(C(r)) / log(r)
```

**En clair :** Pour chaque rayon r, comptez quelle fraction de toutes les paires de
points sont distantes de moins de r. Quand r diminue, cette fraction décroît selon une
loi de puissance. L'exposant de cette loi de puissance est la dimension de corrélation.

La dimension de corrélation est liée à la dimension par comptage de boîtes, mais
légèrement différente. Pour la plupart des attracteurs pratiques : D_corr <= D_box.

### Exposant de Hurst (apparenté)

```
H = pente de log(R/S) en fonction de log(n)
```

**En clair :** Pour une série temporelle, l'exposant de Hurst mesure la dépendance à
long terme. H = 0.5 signifie marche aléatoire ; H > 0.5 signifie tendance (persistant) ;
H < 0.5 signifie retour à la moyenne (anti-persistant). La dimension fractale du graphe
de la série temporelle est D = 2 - H.

## En Rust

```rust
use ix_chaos::fractal::{
    box_counting_dimension_2d,
    correlation_dimension,
    hurst_exponent,
};

// --- Comptage de boîtes : dimension d'une droite ---
let line: Vec<(f64, f64)> = (0..1000)
    .map(|i| {
        let t = i as f64 / 999.0;
        (t, t)  // droite diagonale
    })
    .collect();
let dim = box_counting_dimension_2d(&line, 8);
println!("Dimension d'une droite : {:.2}", dim);  // ~1.0

// --- Comptage de boîtes : dimension d'un carré rempli ---
let mut square = Vec::new();
for i in 0..50 {
    for j in 0..50 {
        square.push((i as f64 / 49.0, j as f64 / 49.0));
    }
}
let dim = box_counting_dimension_2d(&square, 6);
println!("Dimension du carré : {:.2}", dim);  // ~2.0

// --- Comptage de boîtes : attracteur de Hénon ---
use ix_chaos::attractors::{henon, HenonParams};
let traj = henon(0.1, 0.1, &HenonParams::default(), 50_000);
let points: Vec<(f64, f64)> = traj[1000..].to_vec(); // ignorer le transitoire
let dim = box_counting_dimension_2d(&points, 10);
println!("Dimension de l'attracteur de Hénon : {:.2}", dim);  // ~1.26

// --- Dimension de corrélation ---
// Convertir les points 2D en format vecteur de vecteurs
let data: Vec<Vec<f64>> = points.iter()
    .take(2000)  // sous-échantillonner pour la vitesse
    .map(|&(x, y)| vec![x, y])
    .collect();
let d_corr = correlation_dimension(&data, 0.001, 1.0, 20);
println!("Dimension de corrélation : {:.2}", d_corr);

// --- Exposant de Hurst d'une série temporelle ---
let prices: Vec<f64> = (0..1024)
    .scan(100.0, |state, _| {
        *state += 0.1 * (*state * 0.01);  // série avec tendance
        Some(*state)
    })
    .collect();
let h = hurst_exponent(&prices);
println!("Exposant de Hurst : {:.2}", h);  // > 0.5 pour des données avec tendance
```

## Quand l'utiliser

| Technique | Idéal pour | Limites |
|-----------|-----------|---------|
| **Comptage de boîtes** | Ensembles de points 2D, formes d'attracteurs, textures d'images | Sensible au nombre de points et à la plage d'échelles |
| **Dimension de corrélation** | Attracteurs de haute dimension, estimation de la dimension de plongement | Calcul en O(N^2) ; nécessite beaucoup de données |
| **Exposant de Hurst** | Séries temporelles : détection de tendance, analyse de volatilité | Suppose l'auto-similarité ; biais d'échantillon fini |

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Règle empirique |
|-----------|-------------------|-----------------|
| `num_scales` | Nombre de tailles de boîtes / valeurs de rayon à tester | 6-12 pour le comptage de boîtes ; 15-25 pour la dimension de corrélation |
| `r_min, r_max` | Plage de rayons pour la dimension de corrélation | r_min ~ plus petite distance inter-points ; r_max ~ étendue du jeu de données |
| Nombre de points | Fiabilité statistique | Comptage de boîtes : 1000+ ; dimension de corrélation : 2000+ ; Hurst : 256+ |

## Pièges courants

1. **Trop peu de points.** Les estimations de dimension fractale sont biaisées avec de
   petits échantillons. Pour le comptage de boîtes, l'ensemble doit remplir densément la
   structure fractale. Des milliers de points sont le minimum pour des estimations fiables.

2. **Choix de la plage d'échelles.** La dimension est estimée à partir de la *pente* d'un
   graphique log-log. Si vous incluez des échelles trop grandes (effets de taille finie)
   ou trop petites (effets de discrétisation), la pente sera fausse. Inspectez visuellement
   le graphique log-log si possible.

3. **Lacunarité.** Deux fractales peuvent avoir la même dimension mais une apparence
   visuelle très différente (caractère lacunaire). La dimension fractale seule ne
   caractérise pas complètement une fractale.

4. **Biais de l'exposant de Hurst.** La méthode R/S est connue pour son biais sur les
   séries courtes. Les séries de moins de ~256 points produisent des estimations de H
   peu fiables.

5. **Dimension de corrélation en O(N^2).** Le calcul de toutes les distances par paires
   est coûteux. Pour les grands jeux de données, sous-échantillonnez à quelques milliers
   de points ou utilisez des méthodes approximatives.

## Pour aller plus loin

- Calculez la dimension de trajectoires de l'attracteur de Lorenz en projetant les points
  3D en 2D (par ex. plan x-z) et en appelant `box_counting_dimension_2d`.
- Utilisez `ix_chaos::lyapunov::lyapunov_spectrum` conjointement avec la dimension
  fractale pour valider la conjecture de Kaplan-Yorke :
  D_KY = j + sum(lambda_1..lambda_j) / |lambda_{j+1}|.
- Pour les séries temporelles, utilisez `ix_chaos::embedding` pour reconstruire un
  attracteur en espace des phases par plongement temporel, puis mesurez sa dimension
  de corrélation.
- Comparez l'exposant de Hurst de rendements financiers sur différentes fenêtres
  temporelles pour détecter les changements de régime.
