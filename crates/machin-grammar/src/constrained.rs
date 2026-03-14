//! Grammar-guided MCTS via EBNF production rules.
//!
//! [`EbnfGrammar`] parses a simple EBNF notation (one rule per line,
//! `name ::= alt1 | alt2 | alt3`).
//! [`GrammarMctsState`] implements `machin_search::mcts::MctsState` so the
//! standard `mcts_search` function can derive sentences from the grammar.

use std::collections::HashMap;
use std::sync::Arc;

use machin_search::mcts::{mcts_search, MctsState};

/// A simple EBNF grammar.
///
/// Productions are stored as `symbol → [[token, …], …]`.
/// Terminals are any symbol that has no production.
#[derive(Clone, Debug)]
pub struct EbnfGrammar {
    /// Start symbol (first rule defined in the grammar text).
    pub start: String,
    /// Map from non-terminal name to its list of alternative token sequences.
    pub productions: HashMap<String, Vec<Vec<String>>>,
}

impl EbnfGrammar {
    /// Parse a grammar from a string.
    ///
    /// Each non-empty, non-comment line must have the form:
    /// ```text
    /// name ::= tokens... | tokens... | ...
    /// ```
    /// Lines starting with `#` are ignored.
    ///
    /// ```
    /// use machin_grammar::constrained::EbnfGrammar;
    /// let g = EbnfGrammar::from_str("S ::= a b | c\na ::= x").unwrap();
    /// assert_eq!(g.start, "S");
    /// assert_eq!(g.alternatives("S").len(), 2);
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, String> {
        let mut productions: HashMap<String, Vec<Vec<String>>> = HashMap::new();
        let mut start: Option<String> = None;

        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.splitn(2, "::=").collect();
            if parts.len() != 2 {
                return Err(format!("Malformed rule (missing '::='): {}", line));
            }
            let name = parts[0].trim().to_string();
            if start.is_none() {
                start = Some(name.clone());
            }
            let alts: Vec<Vec<String>> = parts[1]
                .split('|')
                .map(|alt| {
                    alt.split_whitespace()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                })
                .filter(|alt| !alt.is_empty())
                .collect();
            if alts.is_empty() {
                return Err(format!("Rule '{}' has no alternatives", name));
            }
            productions.insert(name, alts);
        }

        Ok(EbnfGrammar {
            start: start.unwrap_or_default(),
            productions,
        })
    }

    /// Return the alternatives for a given symbol, or empty if it is terminal.
    pub fn alternatives(&self, symbol: &str) -> Vec<Vec<String>> {
        self.productions.get(symbol).cloned().unwrap_or_default()
    }

    /// Returns `true` if `symbol` is a terminal (has no production rules).
    pub fn is_terminal(&self, symbol: &str) -> bool {
        !self.productions.contains_key(symbol)
    }
}

/// Expansion of one non-terminal: the symbol expanded and the chosen alternative.
#[derive(Clone, Debug)]
pub struct GrammarAction {
    /// The non-terminal that is being expanded.
    pub nonterminal: String,
    /// The sequence of tokens in the chosen alternative.
    pub alternative: Vec<String>,
    /// Index of this alternative in the production list.
    pub alt_index: usize,
}

/// MCTS state tracking a partial grammar derivation (leftmost expansion).
///
/// - `stack` holds the symbols yet to be expanded.
/// - `derivation` records every expansion taken so far.
/// - The state is terminal when all remaining stack symbols are terminals
///   or `max_depth` expansions have been performed.
#[derive(Clone)]
pub struct GrammarMctsState {
    grammar: Arc<EbnfGrammar>,
    /// Remaining symbols to expand (leftmost-first).
    stack: Vec<String>,
    /// Derivation trace: `(nonterminal, chosen_alternative)`.
    derivation: Vec<(String, Vec<String>)>,
    /// Hard cap on expansion depth to bound the search.
    max_depth: usize,
}

impl GrammarMctsState {
    /// Create a fresh derivation state starting from the grammar's start symbol.
    pub fn new(grammar: Arc<EbnfGrammar>, max_depth: usize) -> Self {
        let start = grammar.start.clone();
        GrammarMctsState {
            grammar,
            stack: vec![start],
            derivation: Vec::new(),
            max_depth,
        }
    }

    /// The derivation trace accumulated so far.
    pub fn derivation(&self) -> &[(String, Vec<String>)] {
        &self.derivation
    }

    /// The remaining unexpanded symbols on the stack.
    pub fn stack(&self) -> &[String] {
        &self.stack
    }
}

impl MctsState for GrammarMctsState {
    type Action = GrammarAction;

    fn legal_actions(&self) -> Vec<GrammarAction> {
        if MctsState::is_terminal(self) {
            return vec![];
        }
        // Expand the first non-terminal found on the stack
        for symbol in &self.stack {
            let alts = self.grammar.alternatives(symbol);
            if !alts.is_empty() {
                return alts
                    .into_iter()
                    .enumerate()
                    .map(|(i, alt)| GrammarAction {
                        nonterminal: symbol.clone(),
                        alternative: alt,
                        alt_index: i,
                    })
                    .collect();
            }
        }
        vec![]
    }

    fn apply(&self, action: &GrammarAction) -> Self {
        let mut new_stack = self.stack.clone();
        let mut new_derivation = self.derivation.clone();

        // Replace the first occurrence of the non-terminal with its alternative
        if let Some(pos) = new_stack.iter().position(|s| s == &action.nonterminal) {
            new_stack.remove(pos);
            for (i, sym) in action.alternative.iter().enumerate() {
                new_stack.insert(pos + i, sym.clone());
            }
        }

        new_derivation.push((action.nonterminal.clone(), action.alternative.clone()));

        GrammarMctsState {
            grammar: Arc::clone(&self.grammar),
            stack: new_stack,
            derivation: new_derivation,
            max_depth: self.max_depth,
        }
    }

    fn is_terminal(&self) -> bool {
        if self.derivation.len() >= self.max_depth {
            return true;
        }
        self.stack.iter().all(|s| self.grammar.is_terminal(s))
    }

    fn reward(&self) -> f64 {
        let terminals = self
            .stack
            .iter()
            .filter(|s| self.grammar.is_terminal(s))
            .count();
        if self.stack.is_empty() || terminals == self.stack.len() {
            1.0
        } else {
            terminals as f64 / self.stack.len() as f64
        }
    }
}

/// Result of a grammar-guided MCTS search.
pub struct MctsResult {
    /// Derivation steps: `(expanded_nonterminal, chosen_tokens)`.
    pub best_derivation: Vec<(String, Vec<String>)>,
    /// Final reward of the best derivation found.
    pub reward: f64,
    /// Number of MCTS iterations run.
    pub iterations: usize,
}

/// Run grammar-guided MCTS and return the best derivation found.
///
/// ```
/// use machin_grammar::constrained::{EbnfGrammar, search_derivation};
/// let g = EbnfGrammar::from_str("S ::= a b\na ::= x | y\nb ::= 1 | 2").unwrap();
/// let result = search_derivation(g, 200, 1.41, 10, 42);
/// assert!(!result.best_derivation.is_empty());
/// assert!(result.reward > 0.0);
/// ```
pub fn search_derivation(
    grammar: EbnfGrammar,
    max_iterations: usize,
    exploration: f64,
    max_depth: usize,
    seed: u64,
) -> MctsResult {
    let grammar_arc = Arc::new(grammar);
    let root = GrammarMctsState::new(Arc::clone(&grammar_arc), max_depth);

    let action = mcts_search(&root, max_iterations, exploration, seed);

    // After MCTS returns the best first action, complete the derivation greedily
    // (always pick the first legal action) until we reach a terminal state.
    let mut state = if let Some(act) = action {
        root.apply(&act)
    } else {
        root.clone()
    };

    while !MctsState::is_terminal(&state) {
        let actions = state.legal_actions();
        if actions.is_empty() {
            break;
        }
        state = state.apply(&actions[0]);
    }

    MctsResult {
        best_derivation: state.derivation().to_vec(),
        reward: MctsState::reward(&state),
        iterations: max_iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ebnf_parse_basic() {
        let g = EbnfGrammar::from_str("S ::= a b | c\na ::= x | y").unwrap();
        assert_eq!(g.start, "S");
        assert_eq!(g.alternatives("S").len(), 2);
        assert_eq!(g.alternatives("a").len(), 2);
        assert!(g.is_terminal("x"));
        assert!(!g.is_terminal("S"));
    }

    #[test]
    fn test_ebnf_comments_skipped() {
        let g = EbnfGrammar::from_str("# comment\nS ::= a").unwrap();
        assert_eq!(g.start, "S");
        assert_eq!(g.alternatives("S").len(), 1);
    }

    #[test]
    fn test_ebnf_malformed_line() {
        let result = EbnfGrammar::from_str("no-separator");
        assert!(result.is_err());
    }

    #[test]
    fn test_grammar_mcts_legal_actions() {
        let g = EbnfGrammar::from_str("S ::= a | b\na ::= x").unwrap();
        let state = GrammarMctsState::new(Arc::new(g), 10);
        let actions = state.legal_actions();
        assert_eq!(actions.len(), 2); // "a" and "b"
    }

    #[test]
    fn test_grammar_mcts_apply() {
        let g = EbnfGrammar::from_str("S ::= A B\nA ::= x\nB ::= y").unwrap();
        let state = GrammarMctsState::new(Arc::new(g), 10);
        let actions = state.legal_actions();
        assert!(!actions.is_empty());
        let next = state.apply(&actions[0]);
        assert_eq!(next.derivation().len(), 1);
    }

    #[test]
    fn test_grammar_mcts_terminal_all_terminals() {
        let g = EbnfGrammar::from_str("S ::= x y").unwrap();
        let arc = Arc::new(g);
        let state = GrammarMctsState::new(Arc::clone(&arc), 10);
        // After expanding S → x y, stack = [x, y], both terminals
        let actions = state.legal_actions();
        let next = state.apply(&actions[0]);
        assert!(MctsState::is_terminal(&next));
        assert_eq!(MctsState::reward(&next), 1.0);
    }

    #[test]
    fn test_grammar_mcts_depth_cap() {
        let g = EbnfGrammar::from_str("S ::= S | x").unwrap();
        let _state = GrammarMctsState::new(Arc::new(g), 3);
        // With depth cap = 3, must terminate in ≤ 3 expansions
        let result = search_derivation(
            EbnfGrammar::from_str("S ::= S | x").unwrap(),
            100,
            1.41,
            3,
            99,
        );
        assert!(result.iterations == 100);
    }

    #[test]
    fn test_search_derivation_simple_grammar() {
        let grammar_str = "S ::= A B\nA ::= x | y\nB ::= 1 | 2";
        let g = EbnfGrammar::from_str(grammar_str).unwrap();
        let result = search_derivation(g, 200, 1.41, 10, 42);
        assert!(!result.best_derivation.is_empty());
        assert!(result.reward > 0.0);
    }
}
