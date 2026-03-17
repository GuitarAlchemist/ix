# Spam Classifier with TF-IDF and Naive Bayes

A complete end-to-end ML pipeline that classifies SMS messages as **spam** or
**ham** (legitimate), using only `ix_supervised` crates -- no Python, no external frameworks.

## The Problem

Your inbox is flooded. Some messages are real ("Meeting at 3pm"), others are
junk ("FREE cash prize! Click NOW!!!"). You need a model that reads raw text
and decides: spam or ham?

The challenge is that ML models operate on numbers, not words. We need a bridge
from text to vectors -- and that bridge is **TF-IDF**.

## The Data

We work with a small labelled corpus. Each message is a `&str`, and each label
is `0` (ham) or `1` (spam).

```rust
// Training corpus -- a mix of legitimate messages and spam
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

## Step 1 -- Vectorize Text with TfidfVectorizer

`TfidfVectorizer` converts each document into a numeric vector where each
dimension corresponds to a word in the vocabulary. The weight reflects how
**distinctive** a word is to that document:

- **TF** (Term Frequency): how often the word appears in the document.
- **IDF** (Inverse Document Frequency): penalizes words that appear everywhere
  (like "the") and boosts words that are rare and informative (like "viagra").

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

Each row of `x` is now an L2-normalized TF-IDF vector. Words like "free",
"win", and "click" will have high weights in spam documents, while words like
"meeting", "report", and "birthday" will stand out in ham documents.

## Step 2 -- Train a Naive Bayes Classifier

Gaussian Naive Bayes is a natural fit for text classification. It assumes
features are independently distributed per class -- a simplification, but one
that works remarkably well for word frequencies.

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

## Step 3 -- Cross-Validate with StratifiedKFold

Training accuracy is misleading -- we need to know how the model performs on
**unseen** data. `cross_val_score` uses `StratifiedKFold` internally, which
preserves the ham/spam ratio in every fold.

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

With well-separated spam and ham vocabularies, expect accuracy above 80% even
on this tiny dataset.

## Step 4 -- Evaluate with Confusion Matrix and Metrics

Accuracy alone hides important details. For a spam filter, two errors have very
different costs:

- **False positive** (ham classified as spam): a real email lands in junk.
  The user misses an important message. This is measured by **precision**.
- **False negative** (spam classified as ham): junk reaches the inbox.
  Annoying, but less dangerous. This is measured by **recall**.

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

## Putting It All Together

Here is the complete, runnable program:

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

## Key Takeaways

1. **TF-IDF makes text ML-ready.** Raw strings become numeric feature vectors
   where each dimension captures how distinctive a word is to a document.
   Without TF-IDF (or a similar vectorizer), no downstream model can operate
   on text.

2. **Naive Bayes is a natural fit for text.** The independence assumption
   aligns well with bag-of-words features. Training is fast, and the model
   handles high-dimensional sparse data gracefully.

3. **Precision vs. recall is the real decision.** For a spam filter, high
   precision means "don't block good email" -- every message flagged as spam
   truly is spam. High recall means "catch all spam" -- no junk reaches the
   inbox. In practice, most spam filters favor **precision**: it is better to
   let a few spam messages through than to accidentally hide a legitimate
   email. Tune this tradeoff by adjusting the classification threshold on
   `predict_proba` output.

4. **Cross-validation prevents overconfidence.** A model that scores 100% on
   training data may be memorizing, not learning. Stratified k-fold CV gives
   an honest estimate of generalization performance.

*Built with `ix_supervised` -- pure Rust ML, no Python required.*
