# Classifieur de spam avec TF-IDF et Naive Bayes

Un pipeline ML complet de bout en bout qui classe les messages SMS en **spam**
ou **ham** (message légitime), en utilisant uniquement les crates `ix_supervised`.

## Le problème

Votre boîte de réception déborde. Certains messages sont réels
(« Réunion à 15h »), d'autres sont indésirables (« GRATUIT ! Cliquez
MAINTENANT !!! »). Il vous faut un modèle qui lit du texte brut et décide :
spam ou ham ? Le défi : les modèles ML opèrent sur des nombres, pas sur des
mots. Il faut un pont entre le texte et les vecteurs -- et ce pont, c'est le
**TF-IDF**.

## Les données

Nous travaillons avec un petit corpus étiqueté. Chaque message est un `&str`,
et chaque étiquette vaut `0` (ham) ou `1` (spam).

```rust
// Corpus d'entraînement -- un mélange de messages légitimes et de spam
let ham_messages = vec![
    "Hey, are we still meeting for lunch tomorrow?",
    "Can you send me the project report by Friday?",
    "Great presentation today, well done!",
    "Reminder: dentist appointment at 3pm",
    "Thanks for helping with the move last weekend",
    "The quarterly budget review is scheduled for Monday",
    "Happy birthday! Hope you have a wonderful day",
    "Could you pick up some milk on the way home?",
];

let spam_messages = vec![
    "FREE entry to win a brand new car! Text WIN to 80085",
    "Congratulations! You have been selected for a cash prize!",
    "URGENT: your account has been compromised, click here now",
    "Buy cheap viagra and pharmacy products online!!!",
    "You have won a free iPhone! Claim your prize today",
    "Make money fast working from home, guaranteed income!!!",
    "Special discount offer just for you, act now, limited time",
    "Free ringtones, games, and screensavers, download now!",
];
```

## Étape 1 -- Vectoriser le texte avec TfidfVectorizer

`TfidfVectorizer` convertit chaque document en un vecteur numérique où chaque
dimension correspond à un mot du vocabulaire. Le poids reflète le caractère
**distinctif** d'un mot : **TF** mesure la fréquence du terme dans le document,
tandis qu'**IDF** pénalise les mots présents partout (comme « the ») et
favorise les mots rares et informatifs (comme « viagra »).

```rust
use ix_supervised::text::TfidfVectorizer;

// Combine all messages into a single corpus
let docs: Vec<&str> = ham_messages.iter().chain(spam_messages.iter())
    .map(|s| s.as_str()).collect();

// Build the vectorizer -- learn vocabulary and IDF weights, then transform
let mut tfidf = TfidfVectorizer::new()
    .with_min_df(1)       // keep all words (small dataset)
    .with_max_df(0.9);    // drop words in >90% of documents

let x = tfidf.fit_transform(&docs);

println!("Feature matrix: {} documents x {} features",
    x.nrows(), x.ncols());
println!("Vocabulary size: {}", tfidf.vocab_size());
```

Chaque ligne de `x` est désormais un vecteur TF-IDF normalisé en L2. Les mots
comme « free », « win » et « click » auront des poids élevés dans les documents
de spam, tandis que « meeting » et « birthday » se démarqueront dans les ham.

## Étape 2 -- Entraîner un classifieur Naive Bayes

Le Naive Bayes gaussien est naturellement adapté à la classification de texte.
Il suppose que les caractéristiques sont indépendantes au sein de chaque classe
-- une simplification, certes, mais qui fonctionne remarquablement bien pour
les fréquences de mots.

```rust
use ndarray::{Array1, array};
use ix_supervised::naive_bayes::GaussianNaiveBayes;
use ix_supervised::traits::Classifier;

// Labels: 0 = ham (first 8), 1 = spam (last 8)
let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];

let mut model = GaussianNaiveBayes::new();
model.fit(&x, &y);

// Predict on training data (sanity check)
let preds = model.predict(&x);
println!("Training predictions: {:?}", preds);
```

## Étape 3 -- Validation croisée avec StratifiedKFold

La précision sur les données d'entraînement est trompeuse -- il faut savoir
comment le modèle se comporte sur des données **inédites**. `cross_val_score`
utilise `StratifiedKFold` en interne, préservant le ratio ham/spam dans chaque
pli.

```rust
use ix_supervised::validation::cross_val_score;

let scores = cross_val_score(
    &x, &y,
    || GaussianNaiveBayes::new(),
    4,   // 4-fold cross-validation
    42,  // random seed for reproducibility
);

let mean_acc = scores.iter().sum::<f64>() / scores.len() as f64;
println!("Per-fold accuracy: {:?}", scores);
println!("Mean CV accuracy:  {:.1}%", mean_acc * 100.0);
```

Avec des vocabulaires bien séparés, attendez-vous à une précision supérieure
à 80 %, même sur ce petit jeu de données.

## Étape 4 -- Évaluer avec la matrice de confusion et les métriques

La précision globale masque des détails importants. Pour un filtre anti-spam,
deux erreurs ont des coûts très différents : le **faux positif** (ham classé
comme spam -- un vrai message perdu, mesuré par la **précision**) et le
**faux négatif** (spam classé comme ham -- du spam dans la boîte, mesuré par
le **rappel**).

```rust
use ix_supervised::metrics::{
    ConfusionMatrix, precision, recall, f1_score, accuracy,
};

let preds = model.predict(&x);

// Build confusion matrix: 2 classes (ham=0, spam=1)
let cm = ConfusionMatrix::from_labels(&y, &preds, 2);
println!("{}", cm.display());

// Per-class metrics for the spam class (class 1)
let spam_class = 1;
let p = precision(&y, &preds, spam_class);
let r = recall(&y, &preds, spam_class);
let f1 = f1_score(&y, &preds, spam_class);

println!("Spam precision: {:.2}  (of predicted spam, how many are real spam?)", p);
println!("Spam recall:    {:.2}  (of real spam, how many did we catch?)", r);
println!("Spam F1:        {:.2}  (harmonic mean of precision and recall)", f1);

// Full classification report
let (prec_vec, rec_vec, f1_vec, support) = cm.classification_report();
println!("\n  Class | Precision | Recall |   F1   | Support");
println!("  ------|-----------|--------|--------|--------");
for c in 0..2 {
    let label = if c == 0 { "  ham " } else { " spam " };
    println!("  {} |   {:.2}    |  {:.2}  |  {:.2}  |   {}",
        label, prec_vec[c], rec_vec[c], f1_vec[c], support[c]);
}
```

## Programme complet

Voici le programme complet, prêt à être exécuté :

```rust
use ndarray::array;
use ix_supervised::text::TfidfVectorizer;
use ix_supervised::naive_bayes::GaussianNaiveBayes;
use ix_supervised::traits::Classifier;
use ix_supervised::validation::cross_val_score;
use ix_supervised::metrics::{ConfusionMatrix, precision, recall, f1_score};

fn main() {
    // -- Data --
    let ham = vec![
        "Hey are we still meeting for lunch tomorrow",
        "Can you send me the project report by Friday",
        "Great presentation today well done",
        "Reminder dentist appointment at 3pm",
        "Thanks for helping with the move last weekend",
        "The quarterly budget review is scheduled for Monday",
        "Happy birthday hope you have a wonderful day",
        "Could you pick up some milk on the way home",
    ];
    let spam = vec![
        "FREE entry to win a brand new car text WIN to 80085",
        "Congratulations you have been selected for a cash prize",
        "URGENT your account has been compromised click here now",
        "Buy cheap viagra and pharmacy products online",
        "You have won a free iPhone claim your prize today",
        "Make money fast working from home guaranteed income",
        "Special discount offer just for you act now limited time",
        "Free ringtones games and screensavers download now",
    ];
    let docs: Vec<&str> = ham.iter().chain(spam.iter()).map(|s| s.as_str()).collect();
    let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];

    // -- Step 1: Vectorize --
    let mut tfidf = TfidfVectorizer::new().with_max_df(0.9);
    let x = tfidf.fit_transform(&docs);
    println!("{} docs, {} features", x.nrows(), x.ncols());

    // -- Step 2: Train --
    let mut model = GaussianNaiveBayes::new();
    model.fit(&x, &y);

    // -- Step 3: Cross-validate --
    let scores = cross_val_score(&x, &y, || GaussianNaiveBayes::new(), 4, 42);
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    println!("CV accuracy: {:.1}%", mean * 100.0);

    // -- Step 4: Evaluate --
    let preds = model.predict(&x);
    let cm = ConfusionMatrix::from_labels(&y, &preds, 2);
    println!("{}", cm.display());
    println!("Spam precision: {:.2}", precision(&y, &preds, 1));
    println!("Spam recall:    {:.2}", recall(&y, &preds, 1));
    println!("Spam F1:        {:.2}", f1_score(&y, &preds, 1));
}
```

## Points clés à retenir

1. **Le TF-IDF rend le texte exploitable par le ML.** Les chaînes brutes
   deviennent des vecteurs numériques où chaque dimension capture le caractère
   distinctif d'un mot dans un document. Sans TF-IDF (ou un vectoriseur
   similaire), aucun modèle en aval ne peut traiter du texte.

2. **Le Naive Bayes est naturellement adapté au texte.** L'hypothèse
   d'indépendance s'accorde bien avec les caractéristiques de type
   « sac de mots ». L'entraînement est rapide et le modèle gère
   élégamment les données creuses en haute dimension.

3. **Précision contre rappel -- le vrai choix.** Pour un filtre anti-spam,
   une haute précision signifie « ne pas bloquer les bons messages » -- chaque
   message signalé comme spam l'est véritablement. Un rappel élevé signifie
   « attraper tout le spam ». En pratique, la plupart des filtres anti-spam
   privilégient la **précision** : il vaut mieux laisser passer quelques spams
   que de cacher un message légitime. Ajustez ce compromis via le seuil de
   classification sur la sortie de `predict_proba`.

4. **La validation croisée empêche l'excès de confiance.** Un modèle qui
   obtient 100 % sur les données d'entraînement mémorise peut-être au lieu
   d'apprendre. La validation croisée stratifiée en k plis fournit une
   estimation honnête de la performance de généralisation.

*Construit avec `ix_supervised` -- du ML en pur Rust, sans Python.*
