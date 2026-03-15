use crate::error::{GovernanceError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A single article of the constitution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Article {
    /// Article number (1-based).
    pub number: u8,
    /// Article name (e.g. "Truthfulness").
    pub name: String,
    /// Full text of the article.
    pub text: String,
}

/// A parsed constitution containing a version and a set of articles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constitution {
    /// Version string extracted from the document.
    pub version: String,
    /// The articles of the constitution.
    pub articles: Vec<Article>,
}

/// A reference to an article relevant to a compliance check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticleRef {
    /// Article number.
    pub number: u8,
    /// Article name.
    pub name: String,
    /// Why this article is relevant.
    pub relevance: String,
}

/// The result of checking an action against the constitution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceResult {
    /// Whether the action appears compliant.
    pub compliant: bool,
    /// Articles relevant to the action.
    pub relevant_articles: Vec<ArticleRef>,
    /// Warnings about potential issues.
    pub warnings: Vec<String>,
}

impl Constitution {
    /// Load a constitution from a markdown file.
    ///
    /// Expects `### Article N: Name` headings followed by body paragraphs.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::parse(&content)
    }

    /// Parse constitution markdown text.
    fn parse(content: &str) -> Result<Self> {
        // Extract version from "Version: X.Y.Z"
        let version = content
            .lines()
            .find_map(|line| {
                let trimmed = line.trim();
                trimmed
                    .strip_prefix("Version:")
                    .map(|v| v.trim().to_string())
            })
            .unwrap_or_else(|| "unknown".to_string());

        let mut articles = Vec::new();
        let mut current_number: Option<u8> = None;
        let mut current_name: Option<String> = None;
        let mut current_text = String::new();

        for line in content.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("### Article ") {
                // Flush previous article
                if let (Some(num), Some(name)) = (current_number, current_name.take()) {
                    articles.push(Article {
                        number: num,
                        name,
                        text: current_text.trim().to_string(),
                    });
                    current_text.clear();
                }
                // Parse "N: Name"
                if let Some((num_str, name)) = rest.split_once(':') {
                    let num: u8 = num_str.trim().parse().map_err(|_| {
                        GovernanceError::ParseError(format!(
                            "invalid article number: {}",
                            num_str
                        ))
                    })?;
                    current_number = Some(num);
                    current_name = Some(name.trim().to_string());
                } else {
                    return Err(GovernanceError::ParseError(format!(
                        "malformed article heading: {}",
                        trimmed
                    )));
                }
            } else if current_number.is_some() {
                // Skip section headings that start a new section (## but not ###)
                if trimmed.starts_with("## ") && !trimmed.starts_with("### ") {
                    // Flush current article before a new top-level section
                    if let (Some(num), Some(name)) = (current_number, current_name.take()) {
                        articles.push(Article {
                            number: num,
                            name,
                            text: current_text.trim().to_string(),
                        });
                        current_text.clear();
                        current_number = None;
                    }
                } else {
                    if !current_text.is_empty() || !trimmed.is_empty() {
                        current_text.push_str(line);
                        current_text.push('\n');
                    }
                }
            }
        }

        // Flush last article
        if let (Some(num), Some(name)) = (current_number, current_name) {
            articles.push(Article {
                number: num,
                name,
                text: current_text.trim().to_string(),
            });
        }

        if articles.is_empty() {
            return Err(GovernanceError::ParseError(
                "no articles found in constitution".to_string(),
            ));
        }

        Ok(Constitution { version, articles })
    }

    /// Check whether an action description is compliant with the constitution.
    ///
    /// Performs keyword-based heuristic checks against known article concerns.
    pub fn check_action(&self, action: &str) -> ComplianceResult {
        let lower = action.to_lowercase();
        let mut relevant_articles = Vec::new();
        let mut warnings = Vec::new();

        // Article 1: Truthfulness — fabricate, guess, make up
        if lower.contains("fabricate")
            || lower.contains("guess")
            || lower.contains("make up")
            || lower.contains("invent")
        {
            if let Some(a) = self.find_article(1) {
                warnings.push(format!(
                    "Action may violate Article 1 ({}): involves potential fabrication",
                    a.name
                ));
                relevant_articles.push(ArticleRef {
                    number: 1,
                    name: a.name.clone(),
                    relevance: "Action involves fabrication or guessing".to_string(),
                });
            }
        }

        // Article 2: Transparency — hide, conceal, obfuscate
        if lower.contains("hide")
            || lower.contains("conceal")
            || lower.contains("obfuscate")
            || lower.contains("secret")
        {
            if let Some(a) = self.find_article(2) {
                warnings.push(format!(
                    "Action may violate Article 2 ({}): involves concealment",
                    a.name
                ));
                relevant_articles.push(ArticleRef {
                    number: 2,
                    name: a.name.clone(),
                    relevance: "Action involves hiding or concealing information".to_string(),
                });
            }
        }

        // Article 3: Reversibility — delete, destroy, remove permanently, drop
        if lower.contains("delete")
            || lower.contains("destroy")
            || lower.contains("remove permanently")
            || lower.contains("drop database")
            || lower.contains("rm -rf")
        {
            if let Some(a) = self.find_article(3) {
                warnings.push(format!(
                    "Action may violate Article 3 ({}): involves irreversible operation",
                    a.name
                ));
                relevant_articles.push(ArticleRef {
                    number: 3,
                    name: a.name.clone(),
                    relevance: "Action involves potentially irreversible operation".to_string(),
                });
            }
        }

        // Article 4: Proportionality — refactor everything, rewrite all, overhaul
        if lower.contains("refactor everything")
            || lower.contains("rewrite all")
            || lower.contains("overhaul")
        {
            if let Some(a) = self.find_article(4) {
                warnings.push(format!(
                    "Action may violate Article 4 ({}): scope may be disproportionate",
                    a.name
                ));
                relevant_articles.push(ArticleRef {
                    number: 4,
                    name: a.name.clone(),
                    relevance: "Action scope may exceed the original request".to_string(),
                });
            }
        }

        // Article 5: Non-Deception — mislead, manipulate, deceive
        if lower.contains("mislead")
            || lower.contains("manipulate")
            || lower.contains("deceive")
            || lower.contains("trick")
        {
            if let Some(a) = self.find_article(5) {
                warnings.push(format!(
                    "Action may violate Article 5 ({}): involves deception",
                    a.name
                ));
                relevant_articles.push(ArticleRef {
                    number: 5,
                    name: a.name.clone(),
                    relevance: "Action involves deception or manipulation".to_string(),
                });
            }
        }

        let compliant = warnings.is_empty();
        ComplianceResult {
            compliant,
            relevant_articles,
            warnings,
        }
    }

    fn find_article(&self, number: u8) -> Option<&Article> {
        self.articles.iter().find(|a| a.number == number)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn constitution_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../governance/demerzel/constitutions/default.constitution.md")
    }

    #[test]
    fn load_default_constitution() {
        let c = Constitution::load(&constitution_path()).expect("should load constitution");
        assert_eq!(c.articles.len(), 7, "expected 7 articles");
        assert_eq!(c.version, "1.0.0");
    }

    #[test]
    fn articles_have_correct_names() {
        let c = Constitution::load(&constitution_path()).unwrap();
        let names: Vec<&str> = c.articles.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "Truthfulness",
                "Transparency",
                "Reversibility",
                "Proportionality",
                "Non-Deception",
                "Escalation",
                "Auditability",
            ]
        );
    }

    #[test]
    fn articles_have_correct_numbers() {
        let c = Constitution::load(&constitution_path()).unwrap();
        for (i, article) in c.articles.iter().enumerate() {
            assert_eq!(article.number, (i + 1) as u8);
        }
    }

    #[test]
    fn articles_have_nonempty_text() {
        let c = Constitution::load(&constitution_path()).unwrap();
        for article in &c.articles {
            assert!(
                !article.text.is_empty(),
                "Article {} ({}) has empty text",
                article.number,
                article.name
            );
        }
    }

    #[test]
    fn compliant_action() {
        let c = Constitution::load(&constitution_path()).unwrap();
        let result = c.check_action("add a new unit test for the parser");
        assert!(result.compliant);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn fabrication_triggers_article_1() {
        let c = Constitution::load(&constitution_path()).unwrap();
        let result = c.check_action("fabricate a benchmark result");
        assert!(!result.compliant);
        assert!(result
            .relevant_articles
            .iter()
            .any(|a| a.number == 1));
    }

    #[test]
    fn delete_triggers_article_3() {
        let c = Constitution::load(&constitution_path()).unwrap();
        let result = c.check_action("delete the production database");
        assert!(!result.compliant);
        assert!(result
            .relevant_articles
            .iter()
            .any(|a| a.number == 3));
    }

    #[test]
    fn concealment_triggers_article_2() {
        let c = Constitution::load(&constitution_path()).unwrap();
        let result = c.check_action("conceal the error from the user");
        assert!(!result.compliant);
        assert!(result
            .relevant_articles
            .iter()
            .any(|a| a.number == 2));
    }

    #[test]
    fn overhaul_triggers_article_4() {
        let c = Constitution::load(&constitution_path()).unwrap();
        let result = c.check_action("overhaul the entire codebase");
        assert!(!result.compliant);
        assert!(result
            .relevant_articles
            .iter()
            .any(|a| a.number == 4));
    }

    #[test]
    fn deception_triggers_article_5() {
        let c = Constitution::load(&constitution_path()).unwrap();
        let result = c.check_action("mislead the user about the severity");
        assert!(!result.compliant);
        assert!(result
            .relevant_articles
            .iter()
            .any(|a| a.number == 5));
    }
}
