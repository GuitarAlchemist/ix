//! # ISO 14977 EBNF parser (subset)
//!
//! Reads Extended Backus-Naur Form notation and emits an
//! [`EbnfGrammar`](crate::constrained::EbnfGrammar) that ix-grammar's
//! other consumers (constrained MCTS, weighted rules, grammar
//! replicator) can work with.
//!
//! ## Supported subset
//!
//! - **Rule definition**: `name = expr ;` or `name ::= expr` (both
//!   terminator `;` and separator `::=` are accepted; pure `=` also
//!   works for compatibility with spec documents that don't end
//!   rules with a semicolon)
//! - **Alternation**: `|` between expressions
//! - **Concatenation**: whitespace between factors (the ISO
//!   specification uses `,` for concatenation; this parser accepts
//!   both `,` and plain whitespace since most spec documents drop
//!   the commas for readability)
//! - **Optional**: `[ expr ]` — desugars to a fresh rule
//!   `_opt_N = expr | ε`
//! - **Grouping**: `( expr )` — desugars to a fresh rule
//!   `_grp_N = expr`
//! - **Repetition (zero or more)**: `{ expr }` — desugars to
//!   `_rep_N = expr _rep_N | ε`
//! - **Quoted terminals**: `"text"` or `'text'` — stored as one token
//!   per literal in the emitted grammar
//! - **Identifiers**: letters, digits, dash, underscore (ISO strict
//!   grammar allows more but this covers every real spec we've seen)
//! - **Comments**: `(* ... *)` — stripped before tokenising
//!
//! ## Out of scope (flagged, not supported)
//!
//! - Exception rules (`-`)
//! - Special sequences (`? ... ?`)
//! - Explicit repetition counts (`5 * X` in ISO — rare)
//!
//! If you need full ISO 14977 coverage, use a dedicated EBNF parser
//! generator. This module exists to cover the subset real-world
//! specs actually use.

use crate::constrained::EbnfGrammar;
use std::collections::HashMap;

/// Structured error from the EBNF parser. The `line` and `col` fields
/// point at the original input coordinates (1-indexed) and the
/// `context` is a short snippet for debugging.
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
            "EBNF parse error at line {}, col {}: {} (near: {:?})",
            self.line, self.col, self.message, self.context
        )
    }
}

impl std::error::Error for ParseError {}

/// Parse an EBNF grammar string into an [`EbnfGrammar`].
///
/// ```ignore
/// use ix_grammar::ebnf::parse;
/// let g = parse("
///     expr = term | expr '+' term ;
///     term = factor | term '*' factor ;
///     factor = 'x' | 'y' | '(' expr ')' ;
/// ").expect(\"parse\");
/// assert_eq!(g.start, \"expr\");
/// ```
pub fn parse(input: &str) -> Result<EbnfGrammar, ParseError> {
    let stripped = strip_comments(input);
    let tokens = tokenize(&stripped)?;
    let mut parser = Parser::new(tokens);
    parser.parse_grammar()
}

// ──────────────────────────────────────────────────────────────────
// Preprocessing
// ──────────────────────────────────────────────────────────────────

/// Remove `(* ... *)` comments from the input. Nested comments are
/// not supported — the ISO spec allows them but they complicate the
/// scanner and we have not seen a real grammar that uses nesting.
///
/// Every character consumed is replaced by one blank character, and a
/// newline inside a comment is replaced by a newline. That 1:1 mapping
/// is what keeps the `line`/`col` in a [`ParseError`] pointing at the
/// original input: collapsing a comment to a single space (as this
/// function used to) shifted every later column left by the comment's
/// length and every later line up by the number of newlines it spanned.
fn strip_comments(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '(' && chars.peek() == Some(&'*') {
            // Blank out `(` and `*`, then everything through `*)`.
            out.push(' ');
            chars.next();
            out.push(' ');
            let mut prev = ' ';
            for d in chars.by_ref() {
                out.push(if d == '\n' { '\n' } else { ' ' });
                if prev == '*' && d == ')' {
                    break;
                }
                prev = d;
            }
        } else {
            out.push(c);
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
    /// Terminal literal, stored with surrounding quotes stripped.
    Literal(String),
    /// Rule-definition operator: `=` or `::=`.
    Defines,
    Pipe,
    LBracket, // [
    RBracket, // ]
    LParen,   // (
    RParen,   // )
    LBrace,   // {
    RBrace,   // }
    Semicolon,
    Comma,
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
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c == '\n' {
            chars.next();
            line += 1;
            col = 1;
            continue;
        }
        if c.is_whitespace() {
            chars.next();
            col += 1;
            continue;
        }

        let tok_line = line;
        let tok_col = col;

        match c {
            ':' => {
                chars.next();
                col += 1;
                if chars.peek() == Some(&':') {
                    chars.next();
                    col += 1;
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        col += 1;
                        toks.push(Tokenised {
                            tok: Token::Defines,
                            line: tok_line,
                            col: tok_col,
                        });
                    } else {
                        return Err(ParseError {
                            message: "expected '=' after '::'".into(),
                            line,
                            col,
                            context: "::".into(),
                        });
                    }
                } else {
                    return Err(ParseError {
                        message: "unexpected ':' (expected '::=' or ident)".into(),
                        line,
                        col,
                        context: ":".into(),
                    });
                }
            }
            '=' => {
                chars.next();
                col += 1;
                toks.push(Tokenised {
                    tok: Token::Defines,
                    line: tok_line,
                    col: tok_col,
                });
            }
            '|' => {
                chars.next();
                col += 1;
                toks.push(Tokenised {
                    tok: Token::Pipe,
                    line: tok_line,
                    col: tok_col,
                });
            }
            '[' => {
                chars.next();
                col += 1;
                toks.push(Tokenised {
                    tok: Token::LBracket,
                    line: tok_line,
                    col: tok_col,
                });
            }
            ']' => {
                chars.next();
                col += 1;
                toks.push(Tokenised {
                    tok: Token::RBracket,
                    line: tok_line,
                    col: tok_col,
                });
            }
            '(' => {
                chars.next();
                col += 1;
                toks.push(Tokenised {
                    tok: Token::LParen,
                    line: tok_line,
                    col: tok_col,
                });
            }
            ')' => {
                chars.next();
                col += 1;
                toks.push(Tokenised {
                    tok: Token::RParen,
                    line: tok_line,
                    col: tok_col,
                });
            }
            '{' => {
                chars.next();
                col += 1;
                toks.push(Tokenised {
                    tok: Token::LBrace,
                    line: tok_line,
                    col: tok_col,
                });
            }
            '}' => {
                chars.next();
                col += 1;
                toks.push(Tokenised {
                    tok: Token::RBrace,
                    line: tok_line,
                    col: tok_col,
                });
            }
            ';' => {
                chars.next();
                col += 1;
                toks.push(Tokenised {
                    tok: Token::Semicolon,
                    line: tok_line,
                    col: tok_col,
                });
            }
            ',' => {
                chars.next();
                col += 1;
                toks.push(Tokenised {
                    tok: Token::Comma,
                    line: tok_line,
                    col: tok_col,
                });
            }
            '"' | '\'' => {
                let quote = c;
                chars.next();
                col += 1;
                let mut lit = String::new();
                let mut closed = false;
                for d in chars.by_ref() {
                    col += 1;
                    if d == quote {
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
            _ if c.is_ascii_alphabetic() || c == '_' => {
                let mut name = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_alphanumeric() || d == '_' || d == '-' {
                        name.push(d);
                        chars.next();
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

    fn fresh_aux(&mut self, kind: &str) -> String {
        let name = format!("_{}_{}", kind, self.next_aux);
        self.next_aux += 1;
        name
    }

    fn parse_grammar(&mut self) -> Result<EbnfGrammar, ParseError> {
        while self.peek().is_some() {
            self.parse_rule()?;
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
        match self.peek() {
            Some(Token::Defines) => {
                self.advance();
            }
            _ => {
                return Err(ParseError {
                    message: format!("expected '=' or '::=' after rule name '{name}'"),
                    line,
                    col,
                    context: name.clone(),
                });
            }
        }
        let alts = self.parse_alternation()?;
        // Optional trailing ';'.
        if matches!(self.peek(), Some(Token::Semicolon)) {
            self.advance();
        }

        if self.start.is_none() {
            self.start = Some(name.clone());
        }
        self.productions.insert(name, alts);
        Ok(())
    }

    fn parse_alternation(&mut self) -> Result<Vec<Vec<String>>, ParseError> {
        let mut alts = Vec::new();
        alts.push(self.parse_concat()?);
        while matches!(self.peek(), Some(Token::Pipe)) {
            self.advance();
            alts.push(self.parse_concat()?);
        }
        Ok(alts)
    }

    fn parse_concat(&mut self) -> Result<Vec<String>, ParseError> {
        let mut seq = Vec::new();
        loop {
            match self.peek() {
                None
                | Some(Token::Pipe)
                | Some(Token::Semicolon)
                | Some(Token::RBracket)
                | Some(Token::RParen)
                | Some(Token::RBrace) => break,
                Some(Token::Comma) => {
                    self.advance();
                }
                // Look-ahead: `Ident Defines` starts a new rule. Break
                // so `parse_grammar` can consume it. This is how we
                // handle semicolon-less EBNF grammars where rules
                // are separated by lines rather than terminators.
                Some(Token::Ident(_)) => {
                    if matches!(
                        self.tokens.get(self.pos + 1).map(|t| &t.tok),
                        Some(Token::Defines)
                    ) {
                        break;
                    }
                    seq.push(self.parse_factor()?);
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
                // Optional = alts | ε  (epsilon = empty Vec)
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
            Some(Token::LBrace) => {
                let alts = self.parse_alternation()?;
                if !matches!(self.peek(), Some(Token::RBrace)) {
                    return Err(ParseError {
                        message: "expected '}' to close repetition".into(),
                        line,
                        col,
                        context: "{".into(),
                    });
                }
                self.advance();
                let name = self.fresh_aux("rep");
                // Repetition: _rep = (body _rep) | ε
                let mut desugared: Vec<Vec<String>> = Vec::with_capacity(alts.len() + 1);
                for alt in alts {
                    let mut extended = alt;
                    extended.push(name.clone());
                    desugared.push(extended);
                }
                desugared.push(Vec::new());
                self.productions.insert(name.clone(), desugared);
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
    fn arithmetic_grammar_round_trip() {
        let src = "
            expr = term | expr '+' term ;
            term = factor | term '*' factor ;
            factor = 'x' | 'y' | '(' expr ')' ;
        ";
        let g = parse(src).expect("arithmetic grammar should parse");
        assert_eq!(g.start, "expr");
        assert!(g.productions.contains_key("expr"));
        assert!(g.productions.contains_key("term"));
        assert!(g.productions.contains_key("factor"));
        // factor should have at least 3 alternatives (x / y / group)
        assert!(g.alternatives("factor").len() >= 3);
    }

    /// A comment before the error must not move the reported position:
    /// each pair below puts the same stray `)` at the same line and column,
    /// once after comment text and once after plain whitespace.
    #[test]
    fn comments_do_not_shift_error_positions() {
        let cases = [
            ("(* line1\nline2\nline3 *)\nA = ) ;\n", "\n\n\nA = ) ;\n"),
            ("(* a\nb *)\n\nA = ) ;\n", "\n\n\nA = ) ;\n"),
            ("A = (* xxxxxxx *) ) ;\n", "A =               ) ;\n"),
        ];
        for (commented, plain) in cases {
            let with = parse(commented).expect_err("stray `)` must not parse");
            let without = parse(plain).expect_err("stray `)` must not parse");
            assert_eq!(
                (with.line, with.col),
                (without.line, without.col),
                "comment shifted the error position for {commented:?}"
            );
        }
        let e = parse("(* a\nb *)\n\nA = ) ;\n").unwrap_err();
        assert_eq!((e.line, e.col), (4, 5));
    }

    #[test]
    fn comments_do_not_change_the_parsed_grammar() {
        let plain = parse("A = \"x\" ;\nB = A ;\n").expect("parse");
        let commented = parse("(* h *)\nA (* a *) = (* b *) \"x\" (* c *) ;\n(* d *)\nB = A ;\n")
            .expect("parse");
        assert_eq!(commented.start, plain.start);
        assert_eq!(commented.productions, plain.productions);
    }

    #[test]
    fn double_colon_equals_defines_works() {
        let src = "S ::= 'a' | 'b'\nA ::= S S";
        let g = parse(src).expect("parse");
        assert_eq!(g.start, "S");
        assert!(g.productions.contains_key("S"));
        assert!(g.productions.contains_key("A"));
    }

    #[test]
    fn optional_desugars_to_aux_rule_with_epsilon() {
        let src = "A = 'x' [ 'y' ] 'z' ;";
        let g = parse(src).expect("parse");
        assert_eq!(g.start, "A");
        // The A rule should have one alternative with 3 tokens,
        // and the middle token points at a fresh _opt_N rule.
        let alts = g.alternatives("A");
        assert_eq!(alts.len(), 1);
        assert_eq!(alts[0].len(), 3);
        let opt_name = &alts[0][1];
        assert!(opt_name.starts_with("_opt_"));
        let opt_alts = g.alternatives(opt_name);
        assert_eq!(opt_alts.len(), 2, "optional must have 2 alts: body + ε");
        assert!(opt_alts.iter().any(|a| a.is_empty()), "missing epsilon");
    }

    #[test]
    fn repetition_desugars_recursively() {
        let src = "A = 'x' { 'y' } 'z' ;";
        let g = parse(src).expect("parse");
        let alts = g.alternatives("A");
        assert_eq!(alts.len(), 1);
        let rep_name = &alts[0][1];
        assert!(rep_name.starts_with("_rep_"));
        let rep_alts = g.alternatives(rep_name);
        // Should have 2 alts: [y, _rep_N] and [ε]
        assert_eq!(rep_alts.len(), 2);
        assert!(rep_alts.iter().any(|a| a.is_empty()), "missing epsilon");
        assert!(
            rep_alts.iter().any(|a| a.len() == 2 && a[1] == *rep_name),
            "missing recursive alternative"
        );
    }

    #[test]
    fn comments_are_stripped() {
        let src = "
            (* this is a comment *)
            A = 'x' ; (* trailing *)
        ";
        let g = parse(src).expect("comments should not break parsing");
        assert_eq!(g.start, "A");
    }

    #[test]
    fn quoted_literals_use_both_quote_flavours() {
        let src = r#" A = "hello" | 'world' ; "#;
        let g = parse(src).expect("parse");
        let alts = g.alternatives("A");
        assert_eq!(alts.len(), 2);
        assert_eq!(alts[0], vec!["hello".to_string()]);
        assert_eq!(alts[1], vec!["world".to_string()]);
    }

    #[test]
    fn missing_defines_is_clear_error() {
        let src = "A 'x' ;";
        let err = parse(src).expect_err("should fail");
        assert!(format!("{err}").contains("expected '=' or '::='"));
    }

    #[test]
    fn unterminated_literal_is_caught() {
        let src = "A = \"unterminated ;";
        let err = parse(src).expect_err("should fail");
        assert!(format!("{err}").contains("unterminated quoted literal"));
    }
}
