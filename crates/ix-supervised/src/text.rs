//! Text vectorization for ML: CountVectorizer and TF-IDF.
//!
//! Converts text documents into numerical feature matrices suitable for
//! classification with Naive Bayes, Logistic Regression, SVM, etc.
//!
//! # Example: TF-IDF for spam classification
//!
//! ```
//! use ix_supervised::text::TfidfVectorizer;
//!
//! let docs = vec![
//!     "buy cheap viagra now",
//!     "meeting tomorrow at noon",
//!     "free money click here",
//!     "project deadline friday",
//! ];
//!
//! let mut tfidf = TfidfVectorizer::new();
//! let matrix = tfidf.fit_transform(&docs);
//!
//! assert_eq!(matrix.nrows(), 4);  // 4 documents
//! assert!(matrix.ncols() > 0);     // vocabulary size
//!
//! // Transform new documents using the learned vocabulary
//! let new_docs = vec!["free viagra offer"];
//! let new_matrix = tfidf.transform(&new_docs);
//! assert_eq!(new_matrix.nrows(), 1);
//! assert_eq!(new_matrix.ncols(), matrix.ncols());
//! ```
//!
//! # Example: CountVectorizer (bag of words)
//!
//! ```
//! use ix_supervised::text::CountVectorizer;
//!
//! let docs = vec!["the cat sat", "the dog sat", "the cat and the dog"];
//! let mut cv = CountVectorizer::new();
//! let matrix = cv.fit_transform(&docs);
//!
//! // "the" appears in all 3 docs
//! let the_idx = cv.vocabulary().get("the").unwrap();
//! assert!(matrix[[0, *the_idx]] >= 1.0);
//! assert!(matrix[[2, *the_idx]] >= 2.0); // "the" appears twice in doc 3
//! ```

use ndarray::Array2;
use std::collections::HashMap;

/// Tokenize a document into lowercase words.
fn tokenize(doc: &str) -> Vec<String> {
    doc.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

/// Count Vectorizer — converts documents to a term-frequency matrix.
///
/// Each document becomes a row, each unique word a column.
/// Values are raw word counts.
///
/// # Example
///
/// ```
/// use ix_supervised::text::CountVectorizer;
///
/// let docs = vec!["hello world", "hello rust"];
/// let mut cv = CountVectorizer::new();
/// let matrix = cv.fit_transform(&docs);
///
/// assert_eq!(matrix.nrows(), 2);
/// assert_eq!(matrix.ncols(), 3); // hello, world, rust
/// ```
pub struct CountVectorizer {
    vocab: HashMap<String, usize>,
    min_df: usize,
    max_df_ratio: f64,
}

impl CountVectorizer {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            min_df: 1,
            max_df_ratio: 1.0,
        }
    }

    /// Set minimum document frequency (ignore rare words).
    pub fn with_min_df(mut self, min_df: usize) -> Self {
        self.min_df = min_df;
        self
    }

    /// Set maximum document frequency ratio (ignore very common words).
    /// E.g., 0.9 means ignore words appearing in >90% of documents.
    pub fn with_max_df(mut self, max_df_ratio: f64) -> Self {
        self.max_df_ratio = max_df_ratio;
        self
    }

    /// Access the learned vocabulary (word → column index).
    pub fn vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocab
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Learn vocabulary from documents and transform them.
    pub fn fit_transform(&mut self, docs: &[&str]) -> Array2<f64> {
        self.fit(docs);
        self.transform(docs)
    }

    /// Learn vocabulary from documents.
    pub fn fit(&mut self, docs: &[&str]) {
        let n_docs = docs.len();

        // Count document frequency for each word
        let mut df: HashMap<String, usize> = HashMap::new();
        for doc in docs {
            let tokens = tokenize(doc);
            let unique: std::collections::HashSet<String> = tokens.into_iter().collect();
            for word in unique {
                *df.entry(word).or_insert(0) += 1;
            }
        }

        // Build vocabulary with min_df and max_df filtering
        let max_df_count = (self.max_df_ratio * n_docs as f64).ceil() as usize;
        let mut words: Vec<String> = df.into_iter()
            .filter(|(_, count)| *count >= self.min_df && *count <= max_df_count)
            .map(|(word, _)| word)
            .collect();
        words.sort(); // deterministic ordering

        self.vocab.clear();
        for (i, word) in words.into_iter().enumerate() {
            self.vocab.insert(word, i);
        }
    }

    /// Transform documents using the learned vocabulary.
    pub fn transform(&self, docs: &[&str]) -> Array2<f64> {
        let n_docs = docs.len();
        let n_features = self.vocab.len();
        let mut matrix = Array2::zeros((n_docs, n_features));

        for (i, doc) in docs.iter().enumerate() {
            let tokens = tokenize(doc);
            for token in &tokens {
                if let Some(&j) = self.vocab.get(token) {
                    matrix[[i, j]] += 1.0;
                }
            }
        }

        matrix
    }
}

impl Default for CountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// TF-IDF Vectorizer — converts documents to a TF-IDF weighted matrix.
///
/// TF-IDF = Term Frequency × Inverse Document Frequency.
/// - TF(t, d) = count of term t in document d / total terms in d
/// - IDF(t) = log(N / df(t)) + 1
///
/// This weights words that are distinctive to a document higher than
/// words that appear everywhere.
///
/// # Example
///
/// ```
/// use ix_supervised::text::TfidfVectorizer;
///
/// let docs = vec![
///     "the cat sat on the mat",
///     "the dog sat on the log",
///     "cats and dogs are friends",
/// ];
///
/// let mut tfidf = TfidfVectorizer::new();
/// let matrix = tfidf.fit_transform(&docs);
///
/// // "the" appears in docs 0 and 1 → lower IDF
/// // "friends" appears only in doc 2 → higher IDF
/// let vocab = tfidf.vocabulary();
/// if let (Some(&the_idx), Some(&friends_idx)) = (vocab.get("the"), vocab.get("friends")) {
///     // "friends" should have higher TF-IDF weight in doc 2 than "the"
///     assert!(matrix[[2, friends_idx]] > matrix[[2, the_idx]],
///         "Distinctive words should have higher TF-IDF");
/// }
/// ```
pub struct TfidfVectorizer {
    count_vectorizer: CountVectorizer,
    idf: Vec<f64>,
    normalize: bool,
}

impl TfidfVectorizer {
    pub fn new() -> Self {
        Self {
            count_vectorizer: CountVectorizer::new(),
            idf: Vec::new(),
            normalize: true,
        }
    }

    /// Set minimum document frequency.
    pub fn with_min_df(mut self, min_df: usize) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_min_df(min_df);
        self
    }

    /// Set maximum document frequency ratio.
    pub fn with_max_df(mut self, max_df_ratio: f64) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_max_df(max_df_ratio);
        self
    }

    /// Disable L2 normalization of output rows.
    pub fn without_normalization(mut self) -> Self {
        self.normalize = false;
        self
    }

    /// Access the learned vocabulary.
    pub fn vocabulary(&self) -> &HashMap<String, usize> {
        self.count_vectorizer.vocabulary()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.count_vectorizer.vocab_size()
    }

    /// Learn vocabulary + IDF weights and transform documents.
    ///
    /// More efficient than calling `fit()` then `transform()` separately,
    /// as it reuses the count matrix from fitting to compute TF-IDF directly.
    pub fn fit_transform(&mut self, docs: &[&str]) -> Array2<f64> {
        let n_docs = docs.len() as f64;
        self.count_vectorizer.fit(docs);

        // Build count matrix once, reuse for both IDF and TF-IDF
        let count_matrix = self.count_vectorizer.transform(docs);
        let vocab_size = self.count_vectorizer.vocab_size();

        // Compute IDF from the count matrix
        self.idf = Vec::with_capacity(vocab_size);
        for j in 0..vocab_size {
            let df: f64 = (0..docs.len())
                .filter(|&i| count_matrix[[i, j]] > 0.0)
                .count() as f64;
            self.idf.push(((1.0 + n_docs) / (1.0 + df)).ln() + 1.0);
        }

        // Apply TF-IDF directly to the count matrix (no second transform call)
        self.apply_tfidf(&count_matrix)
    }

    /// Learn vocabulary and IDF weights from documents.
    pub fn fit(&mut self, docs: &[&str]) {
        let n_docs = docs.len() as f64;
        self.count_vectorizer.fit(docs);

        let count_matrix = self.count_vectorizer.transform(docs);
        let vocab_size = self.count_vectorizer.vocab_size();
        self.idf = Vec::with_capacity(vocab_size);

        for j in 0..vocab_size {
            let df: f64 = (0..docs.len())
                .filter(|&i| count_matrix[[i, j]] > 0.0)
                .count() as f64;
            self.idf.push(((1.0 + n_docs) / (1.0 + df)).ln() + 1.0);
        }
    }

    /// Transform documents using learned vocabulary and IDF weights.
    pub fn transform(&self, docs: &[&str]) -> Array2<f64> {
        let count_matrix = self.count_vectorizer.transform(docs);
        self.apply_tfidf(&count_matrix)
    }

    /// Apply TF-IDF weighting to a pre-computed count matrix.
    fn apply_tfidf(&self, count_matrix: &Array2<f64>) -> Array2<f64> {
        let n_docs = count_matrix.nrows();
        let vocab_size = count_matrix.ncols();
        let mut result = Array2::zeros((n_docs, vocab_size));

        for i in 0..n_docs {
            let doc_total: f64 = count_matrix.row(i).sum();
            if doc_total < 1e-12 {
                continue;
            }
            for j in 0..vocab_size {
                let tf = count_matrix[[i, j]] / doc_total;
                result[[i, j]] = tf * self.idf[j];
            }

            if self.normalize {
                let norm: f64 = result.row(i).mapv(|v| v * v).sum().sqrt();
                if norm > 1e-12 {
                    for j in 0..vocab_size {
                        result[[i, j]] /= norm;
                    }
                }
            }
        }

        result
    }
}

impl Default for TfidfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_count_vectorizer_basic() {
        let docs = vec!["hello world", "hello rust"];
        let mut cv = CountVectorizer::new();
        let matrix = cv.fit_transform(&docs);

        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 3); // hello, rust, world

        let hello_idx = *cv.vocabulary().get("hello").unwrap();
        assert_eq!(matrix[[0, hello_idx]], 1.0);
        assert_eq!(matrix[[1, hello_idx]], 1.0);
    }

    #[test]
    fn test_count_vectorizer_repeated_words() {
        let docs = vec!["the cat the cat the cat"];
        let mut cv = CountVectorizer::new();
        let matrix = cv.fit_transform(&docs);

        let the_idx = *cv.vocabulary().get("the").unwrap();
        let cat_idx = *cv.vocabulary().get("cat").unwrap();
        assert_eq!(matrix[[0, the_idx]], 3.0);
        assert_eq!(matrix[[0, cat_idx]], 3.0);
    }

    #[test]
    fn test_count_vectorizer_min_df() {
        let docs = vec!["hello world", "hello rust", "rare xyz"];
        let mut cv = CountVectorizer::new().with_min_df(2);
        cv.fit(&docs);

        // "hello" appears in 2 docs → included
        assert!(cv.vocabulary().contains_key("hello"));
        // "xyz" appears in 1 doc → excluded
        assert!(!cv.vocabulary().contains_key("xyz"));
    }

    #[test]
    fn test_count_vectorizer_max_df() {
        let docs = vec!["the cat", "the dog", "the bird"];
        let mut cv = CountVectorizer::new().with_max_df(0.5);
        cv.fit(&docs);

        // "the" appears in 100% > 50% → excluded
        assert!(!cv.vocabulary().contains_key("the"));
        // "cat" appears in 33% < 50% → included
        assert!(cv.vocabulary().contains_key("cat"));
    }

    #[test]
    fn test_count_vectorizer_transform_unseen() {
        let docs = vec!["hello world"];
        let mut cv = CountVectorizer::new();
        cv.fit(&docs);

        let new_docs = vec!["hello unknown"];
        let matrix = cv.transform(&new_docs);
        let hello_idx = *cv.vocabulary().get("hello").unwrap();
        assert_eq!(matrix[[0, hello_idx]], 1.0);
        // "unknown" not in vocab → ignored
        assert_eq!(matrix.row(0).sum(), 1.0);
    }

    #[test]
    fn test_tfidf_basic() {
        let docs = vec!["the cat", "the dog", "a bird"];
        let mut tfidf = TfidfVectorizer::new();
        let matrix = tfidf.fit_transform(&docs);

        assert_eq!(matrix.nrows(), 3);
        assert!(matrix.ncols() > 0);

        // All values should be non-negative
        assert!(matrix.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_tfidf_distinctive_words_higher() {
        let docs = vec![
            "the cat sat on the mat",
            "the dog sat on the log",
            "cats and dogs are friends",
        ];
        let mut tfidf = TfidfVectorizer::new().without_normalization();
        let matrix = tfidf.fit_transform(&docs);

        let vocab = tfidf.vocabulary();
        // "friends" appears only in doc 2 → high IDF
        // "the" appears in docs 0 and 1 → lower IDF
        if let (Some(&the_idx), Some(&friends_idx)) = (vocab.get("the"), vocab.get("friends")) {
            assert!(matrix[[2, friends_idx]] > 0.0);
            // "the" doesn't appear in doc 2 at all, so this is trivially true
            // Better test: in doc 0, "sat" (appears in 2 docs) vs "mat" (appears in 1)
            if let (Some(&sat_idx), Some(&mat_idx)) = (vocab.get("sat"), vocab.get("mat")) {
                assert!(matrix[[0, mat_idx]] > matrix[[0, sat_idx]],
                    "Rarer word 'mat' should have higher TF-IDF than common word 'sat'");
            }
        }
    }

    #[test]
    fn test_tfidf_l2_normalized() {
        let docs = vec!["hello world foo", "bar baz qux"];
        let mut tfidf = TfidfVectorizer::new();
        let matrix = tfidf.fit_transform(&docs);

        // Each row should have L2 norm ≈ 1
        for i in 0..matrix.nrows() {
            let norm: f64 = matrix.row(i).mapv(|v| v * v).sum().sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "Row {} L2 norm should be 1, got {}", i, norm);
        }
    }

    #[test]
    fn test_tfidf_transform_new_docs() {
        let docs = vec!["buy cheap viagra", "meeting at noon"];
        let mut tfidf = TfidfVectorizer::new();
        let _ = tfidf.fit_transform(&docs);

        let new_docs = vec!["cheap meeting"];
        let matrix = tfidf.transform(&new_docs);
        assert_eq!(matrix.nrows(), 1);
        assert_eq!(matrix.ncols(), tfidf.vocab_size());
    }

    #[test]
    fn test_tfidf_empty_doc() {
        let docs = vec!["hello world", ""];
        let mut tfidf = TfidfVectorizer::new();
        let matrix = tfidf.fit_transform(&docs);

        // Empty doc should be all zeros
        assert_eq!(matrix.row(1).sum(), 0.0);
    }
}
