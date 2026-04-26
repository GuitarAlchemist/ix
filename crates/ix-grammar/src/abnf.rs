//! # RFC 5234 ABNF parser (subset)
//!
//! Reads Augmented Backus-Naur Form as used by IETF protocol specs
//! (HTTP, SMTP, DNS, TLS, URI, OAuth, ...) and emits an
//! [`EbnfGrammar`](crate::constrained::EbnfGrammar) ix-grammar can
//! consume.
//!
//! ## Supported subset
//!
//! - **Rule definition**: `name = expr` on one line, or `name =/`
//!   to add alternatives to an existing rule
//! - **Alternation**: `/` between expressions
//! - **Concatenation**: whitespace between elements
//! - **Optional**: `[ expr ]` — desugared to `_opt_N = expr | ε`
//! - **Grouping**: `( expr )` — desugared to `_grp_N = expr`
//! - **Repetition**:
//!   - `*element` — zero or more → `_rep_N = element _rep_N | ε`
//!   - `1*element` — one or more → `_rep_N = element | element _rep_N`
//!   - `N*M element` — treated as "one or more" when N >= 1 and as
//!     "zero or more" when N = 0; explicit count bounds are not
//!     enforced in the emitted grammar (this is a conservative
//!     over-approximation)
//! - **Quoted terminals**: `"text"` — ASCII literal, case-insensitive
//!   per ABNF convention (we preserve the original case in the
//!   emitted token; callers implementing case-insensitive match
//!   should normalise both sides)
//! - **Identifiers**: letters, digits, dash
//! - **Comments**: `; comment to end of line`
//!
//! ## Out of scope
//!
//! - Numeric value ranges (`%x41-5A`, `%d48-57`)
//! - Length-exact repetition (`3*3 element` as a strict 3-count)
//! - Incremental rule definitions via `=/` merge their alternatives
//!   into the base rule, but the base must be defined first
//!
//! For full RFC 5234 fidelity, use a dedicated ABNF library. This
//! module covers what real-world IETF specs use day-to-day and is
//! enough to parse subsets of HTTP, URI, and JSON grammars.

use crate::constrained::EbnfGrammar;
use std::collections::HashMap;

/// Parser error with line + column context.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub col: usize,
    pub context: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ABNF parse error at line {}, col {}: {} (near: {:?})",
            self.line, self.col, self.message, self.context
        )
    }
}

impl std::error::Error for ParseError {}

/// Parse an ABNF grammar string into an [`EbnfGrammar`].
pub fn parse(input: &str) -> Result<EbnfGrammar, ParseError> {
    let preprocessed = strip_line_comments(input);
    let tokens = tokenize(&preprocessed)?;
    let mut parser = Parser::new(tokens);
    parser.parse_grammar()
}

// ──────────────────────────────────────────────────────────────────
// Preprocessing
// ──────────────────────────────────────────────────────────────────

/// Strip `; comment` tails from each line. A semicolon inside a
/// quoted literal is NOT a comment; track the quote state.
fn strip_line_comments(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for line in input.split_inclusive('\n') {
        let mut in_quote = false;
        for c in line.chars() {
            match c {
                '"' => {
                    in_quote = !in_quote;
                    out.push(c);
                }
                ';' if !in_quote => {
                    // Skip to end of line but keep the newline itself.
                    break;
                }
                _ => out.push(c),
            }
        }
        if !line.ends_with('\n') && !line.is_empty() && !out.ends_with('\n') {
            // The last line without a trailing newline still needs
            // to be preserved as-is.
            continue;
        }
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

// ──────────────────────────────────────────────────────────────────
// Tokeniser
// ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Literal(String),
    /// `=`
    Defines,
    /// `=/`  — incremental alternative
    DefinesIncremental,
    Slash,
    LBracket,
    RBracket,
    LParen,
    RParen,
    /// Repetition operator encountered as `*` in `*element` or as
    /// part of `N*M element`. The tokeniser emits it with the
    /// optional bounds pre-parsed.
    Repetition {
        min: u32,
        max: Option<u32>,
    },
    Newline,
}

struct Tokenised {
    tok: Token,
    line: usize,
    col: usize,
}

fn tokenize(input: &str) -> Result<Vec<Tokenised>, ParseError> {
    let mut toks = Vec::new();
    let mut line = 1usize;
    let mut col = 1usize;
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0usize;

    while i < chars.len() {
        let c = chars[i];
        let tok_line = line;
        let tok_col = col;

        if c == '\n' {
            toks.push(Tokenised {
                tok: Token::Newline,
                line: tok_line,
                col: tok_col,
            });
            line += 1;
            col = 1;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            col += 1;
            continue;
        }

        match c {
            '=' => {
                if i + 1 < chars.len() && chars[i + 1] == '/' {
                    toks.push(Tokenised {
                        tok: Token::DefinesIncremental,
                        line: tok_line,
                        col: tok_col,
                    });
                    i += 2;
                    col += 2;
                } else {
                    toks.push(Tokenised {
                        tok: Token::Defines,
                        line: tok_line,
                        col: tok_col,
                    });
                    i += 1;
                    col += 1;
                }
            }
            '/' => {
                toks.push(Tokenised {
                    tok: Token::Slash,
                    line: tok_line,
                    col: tok_col,
                });
                i += 1;
                col += 1;
            }
            '[' => {
                toks.push(Tokenised {
                    tok: Token::LBracket,
                    line: tok_line,
                    col: tok_col,
                });
                i += 1;
                col += 1;
            }
            ']' => {
                toks.push(Tokenised {
                    tok: Token::RBracket,
                    line: tok_line,
                    col: tok_col,
                });
                i += 1;
                col += 1;
            }
            '(' => {
                toks.push(Tokenised {
                    tok: Token::LParen,
                    line: tok_line,
                    col: tok_col,
                });
                i += 1;
                col += 1;
            }
            ')' => {
                toks.push(Tokenised {
                    tok: Token::RParen,
                    line: tok_line,
                    col: tok_col,
                });
                i += 1;
                col += 1;
            }
            '"' => {
                i += 1;
                col += 1;
                let mut lit = String::new();
                let mut closed = false;
                while i < chars.len() {
                    let d = chars[i];
                    i += 1;
                    col += 1;
                    if d == '"' {
                        closed = true;
                        break;
                    }
                    if d == '\n' {
                        line += 1;
                        col = 1;
                    }
                    lit.push(d);
                }
                if !closed {
                    return Err(ParseError {
                        message: "unterminated quoted literal".into(),
                        line: tok_line,
                        col: tok_col,
                        context: lit,
                    });
                }
                toks.push(Tokenised {
                    tok: Token::Literal(lit),
                    line: tok_line,
                    col: tok_col,
                });
            }
            _ if c.is_ascii_digit() || c == '*' => {
                // Possible repetition prefix: `*`, `N*`, `*M`, `N*M`.
                let (min, max, consumed) = parse_repetition_prefix(&chars[i..]);
                if let Some(cons) = consumed {
                    toks.push(Tokenised {
                        tok: Token::Repetition { min, max },
                        line: tok_line,
                        col: tok_col,
                    });
                    i += cons;
                    col += cons;
                } else {
                    // Not a repetition — treat as part of an ident
                    // only if alphanumeric, otherwise error.
                    if c == '*' {
                        return Err(ParseError {
                            message: "unexpected '*'".into(),
                            line,
                            col,
                            context: "*".into(),
                        });
                    }
                    // Fall through to ident branch.
                    let mut name = String::new();
                    while i < chars.len() {
                        let d = chars[i];
                        if d.is_ascii_alphanumeric() || d == '-' || d == '_' {
                            name.push(d);
                            i += 1;
                            col += 1;
                        } else {
                            break;
                        }
                    }
                    toks.push(Tokenised {
                        tok: Token::Ident(name),
                        line: tok_line,
                        col: tok_col,
                    });
                }
            }
            _ if c.is_ascii_alphabetic() || c == '_' => {
                let mut name = String::new();
                while i < chars.len() {
                    let d = chars[i];
                    if d.is_ascii_alphanumeric() || d == '-' || d == '_' {
                        name.push(d);
                        i += 1;
                        col += 1;
                    } else {
                        break;
                    }
                }
                toks.push(Tokenised {
                    tok: Token::Ident(name),
                    line: tok_line,
                    col: tok_col,
                });
            }
            _ => {
                return Err(ParseError {
                    message: format!("unexpected character '{c}'"),
                    line,
                    col,
                    context: c.to_string(),
                });
            }
        }
    }
    Ok(toks)
}

/// Parse a repetition prefix of the form `N*M`, `*M`, `N*`, or `*`
/// from the start of `chars`. Returns `(min, max, consumed_chars)`
/// if one is present. If no `*` is found, returns None.
fn parse_repetition_prefix(chars: &[char]) -> (u32, Option<u32>, Option<usize>) {
    // Pattern: [N][*][M] with at least one `*` to qualify as rep.
    let mut idx = 0;
    let mut min = 0u32;
    let mut min_present = false;
    while idx < chars.len() && chars[idx].is_ascii_digit() {
        min = min * 10 + (chars[idx] as u32 - '0' as u32);
        min_present = true;
        idx += 1;
    }
    if idx >= chars.len() || chars[idx] != '*' {
        return (0, None, None);
    }
    idx += 1;
    let mut max = 0u32;
    let mut max_present = false;
    while idx < chars.len() && chars[idx].is_ascii_digit() {
        max = max * 10 + (chars[idx] as u32 - '0' as u32);
        max_present = true;
        idx += 1;
    }
    let min_val = if min_present { min } else { 0 };
    let max_val = if max_present { Some(max) } else { None };
    (min_val, max_val, Some(idx))
}

// ──────────────────────────────────────────────────────────────────
// Parser
// ──────────────────────────────────────────────────────────────────

struct Parser {
    tokens: Vec<Tokenised>,
    pos: usize,
    productions: HashMap<String, Vec<Vec<String>>>,
    start: Option<String>,
    next_aux: usize,
}

impl Parser {
    fn new(tokens: Vec<Tokenised>) -> Self {
        Self {
            tokens,
            pos: 0,
            productions: HashMap::new(),
            start: None,
            next_aux: 0,
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|t| &t.tok)
    }
    fn peek_pos(&self) -> (usize, usize) {
        self.tokens
            .get(self.pos)
            .map(|t| (t.line, t.col))
            .unwrap_or((0, 0))
    }
    fn advance(&mut self) -> Option<&Tokenised> {
        let t = self.tokens.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Some(Token::Newline)) {
            self.advance();
        }
    }

    fn fresh_aux(&mut self, kind: &str) -> String {
        let name = format!("_{}_{}", kind, self.next_aux);
        self.next_aux += 1;
        name
    }

    fn parse_grammar(&mut self) -> Result<EbnfGrammar, ParseError> {
        self.skip_newlines();
        while self.peek().is_some() {
            self.parse_rule()?;
            self.skip_newlines();
        }
        if self.productions.is_empty() {
            let (l, c) = self.peek_pos();
            return Err(ParseError {
                message: "no rules in grammar".into(),
                line: l,
                col: c,
                context: String::new(),
            });
        }
        Ok(EbnfGrammar {
            start: self.start.clone().unwrap_or_default(),
            productions: std::mem::take(&mut self.productions),
        })
    }

    fn parse_rule(&mut self) -> Result<(), ParseError> {
        let (line, col) = self.peek_pos();
        let name = match self.advance() {
            Some(Tokenised {
                tok: Token::Ident(n),
                ..
            }) => n.clone(),
            _ => {
                return Err(ParseError {
                    message: "expected rule name".into(),
                    line,
                    col,
                    context: "<ident>".into(),
                })
            }
        };
        let incremental = match self.peek() {
            Some(Token::Defines) => {
                self.advance();
                false
            }
            Some(Token::DefinesIncremental) => {
                self.advance();
                true
            }
            _ => {
                return Err(ParseError {
                    message: format!("expected '=' or '=/' after rule name '{name}'"),
                    line,
                    col,
                    context: name.clone(),
                });
            }
        };
        let alts = self.parse_alternation()?;

        if self.start.is_none() {
            self.start = Some(name.clone());
        }
        if incremental {
            self.productions.entry(name).or_default().extend(alts);
        } else {
            self.productions.insert(name, alts);
        }
        Ok(())
    }

    fn parse_alternation(&mut self) -> Result<Vec<Vec<String>>, ParseError> {
        let mut alts = Vec::new();
        alts.push(self.parse_concat()?);
        while matches!(self.peek(), Some(Token::Slash)) {
            self.advance();
            // Allow alternation to continue across a newline.
            self.skip_newlines();
            alts.push(self.parse_concat()?);
        }
        Ok(alts)
    }

    fn parse_concat(&mut self) -> Result<Vec<String>, ParseError> {
        let mut seq = Vec::new();
        loop {
            match self.peek() {
                None | Some(Token::Slash) | Some(Token::RBracket) | Some(Token::RParen) => break,
                Some(Token::Newline) => {
                    // Newline ends a rule unless the next line
                    // starts with `/` (alternation continuation),
                    // which is handled by the loop above.
                    let save = self.pos;
                    self.skip_newlines();
                    if matches!(self.peek(), Some(Token::Slash)) {
                        // Don't break — the outer loop will eat
                        // the slash and continue alternation.
                        break;
                    } else {
                        self.pos = save;
                        break;
                    }
                }
                _ => {
                    seq.push(self.parse_factor()?);
                }
            }
        }
        Ok(seq)
    }

    fn parse_factor(&mut self) -> Result<String, ParseError> {
        let (line, col) = self.peek_pos();
        let tok = self.advance().map(|t| t.tok.clone());
        match tok {
            Some(Token::Ident(name)) => Ok(name),
            Some(Token::Literal(lit)) => Ok(lit),
            Some(Token::Repetition { min, .. }) => {
                // Need one more factor to repeat.
                let body = self.parse_factor()?;
                let name = self.fresh_aux("rep");
                // _rep_N = body _rep_N | ε       if min == 0
                // _rep_N = body | body _rep_N    if min >= 1
                let desugared = if min == 0 {
                    vec![vec![body.clone(), name.clone()], vec![]]
                } else {
                    vec![vec![body.clone()], vec![body, name.clone()]]
                };
                self.productions.insert(name.clone(), desugared);
                Ok(name)
            }
            Some(Token::LBracket) => {
                let alts = self.parse_alternation()?;
                if !matches!(self.peek(), Some(Token::RBracket)) {
                    return Err(ParseError {
                        message: "expected ']' to close optional group".into(),
                        line,
                        col,
                        context: "[".into(),
                    });
                }
                self.advance();
                let name = self.fresh_aux("opt");
                let mut desugared = alts;
                desugared.push(Vec::new());
                self.productions.insert(name.clone(), desugared);
                Ok(name)
            }
            Some(Token::LParen) => {
                let alts = self.parse_alternation()?;
                if !matches!(self.peek(), Some(Token::RParen)) {
                    return Err(ParseError {
                        message: "expected ')' to close group".into(),
                        line,
                        col,
                        context: "(".into(),
                    });
                }
                self.advance();
                let name = self.fresh_aux("grp");
                self.productions.insert(name.clone(), alts);
                Ok(name)
            }
            other => Err(ParseError {
                message: format!("unexpected token {:?} while parsing factor", other),
                line,
                col,
                context: format!("{:?}", other),
            }),
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_abnf_rule() {
        let src = "
            greeting = \"hello\"
            farewell = \"bye\"
        ";
        let g = parse(src).expect("parse");
        assert_eq!(g.start, "greeting");
        assert_eq!(g.alternatives("greeting"), vec![vec!["hello".to_string()]]);
        assert_eq!(g.alternatives("farewell"), vec![vec!["bye".to_string()]]);
    }

    #[test]
    fn alternation_uses_slash() {
        let src = "method = \"GET\" / \"POST\" / \"PUT\"";
        let g = parse(src).expect("parse");
        assert_eq!(g.alternatives("method").len(), 3);
    }

    #[test]
    fn optional_brackets_desugar() {
        let src = "url = \"http://\" host [ \":\" port ]";
        let g = parse(src).expect("parse");
        let alts = g.alternatives("url");
        assert_eq!(alts.len(), 1);
        let last = alts[0].last().unwrap();
        assert!(
            last.starts_with("_opt_"),
            "last token should be optional aux"
        );
        let opt = g.alternatives(last);
        assert!(
            opt.iter().any(|a| a.is_empty()),
            "optional must have ε alternative"
        );
    }

    #[test]
    fn zero_or_more_repetition_desugars() {
        let src = "numbers = *digit\ndigit = \"0\" / \"1\"";
        let g = parse(src).expect("parse");
        let alts = g.alternatives("numbers");
        assert_eq!(alts.len(), 1);
        let rep = &alts[0][0];
        assert!(rep.starts_with("_rep_"));
        let rep_alts = g.alternatives(rep);
        assert!(
            rep_alts.iter().any(|a| a.is_empty()),
            "zero-or-more must have ε"
        );
    }

    #[test]
    fn one_or_more_repetition_has_no_epsilon() {
        let src = "digits = 1*digit\ndigit = \"0\" / \"9\"";
        let g = parse(src).expect("parse");
        let alts = g.alternatives("digits");
        let rep = &alts[0][0];
        let rep_alts = g.alternatives(rep);
        assert!(
            !rep_alts.iter().any(|a| a.is_empty()),
            "one-or-more must NOT have ε"
        );
    }

    #[test]
    fn line_comments_are_stripped() {
        let src = "
            ; this is a comment
            A = \"x\"  ; trailing comment
            B = \"y\"
        ";
        let g = parse(src).expect("parse");
        assert!(g.productions.contains_key("A"));
        assert!(g.productions.contains_key("B"));
    }

    #[test]
    fn incremental_rule_extends_base() {
        let src = "
            scheme = \"http\" / \"https\"
            scheme =/ \"ftp\" / \"file\"
        ";
        let g = parse(src).expect("parse");
        let alts = g.alternatives("scheme");
        assert_eq!(alts.len(), 4);
    }

    #[test]
    fn semicolon_inside_literal_is_not_a_comment() {
        let src = "A = \"foo; bar\"";
        let g = parse(src).expect("parse");
        assert_eq!(g.alternatives("A"), vec![vec!["foo; bar".to_string()]]);
    }

    #[test]
    fn missing_defines_is_clear_error() {
        let src = "A \"x\"";
        let err = parse(src).expect_err("should fail");
        assert!(format!("{err}").contains("expected '=' or '=/'"));
    }
}
