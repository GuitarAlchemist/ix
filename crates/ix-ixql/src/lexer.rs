//! IXQL tokenizer.
//!
//! Line-oriented only insofar as it needs to be: comments run to end of line,
//! and every token carries the line it came from. The parser uses that line
//! number both for errors and for one disambiguation — a `(` may only open a
//! call when it sits on the same line as the callee, which is what keeps a
//! statement from swallowing the next one (IXQL has no statement terminator).

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Str(String),
    Num(serde_json::Number),
    /// `<-`
    Assign,
    /// `→` or `->`
    Arrow,
    Dot,
    Comma,
    Colon,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Minus,
    Plus,
    /// `++` — sequence concatenation.
    PlusPlus,
    Star,
    Slash,
    Bang,
    EqEq,
    /// `=>` — lambda introducer.
    FatArrow,
    NotEq,
    Gt,
    Gte,
    Lt,
    Lte,
    AndAnd,
    OrOr,
    Eof,
}

impl fmt::Display for Tok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tok::Ident(s) => write!(f, "`{s}`"),
            Tok::Str(_) => write!(f, "a string"),
            Tok::Num(n) => write!(f, "`{n}`"),
            Tok::Assign => write!(f, "`<-`"),
            Tok::Arrow => write!(f, "`→`"),
            Tok::Dot => write!(f, "`.`"),
            Tok::Comma => write!(f, "`,`"),
            Tok::Colon => write!(f, "`:`"),
            Tok::LParen => write!(f, "`(`"),
            Tok::RParen => write!(f, "`)`"),
            Tok::LBrace => write!(f, "`{{`"),
            Tok::RBrace => write!(f, "`}}`"),
            Tok::LBracket => write!(f, "`[`"),
            Tok::RBracket => write!(f, "`]`"),
            Tok::Minus => write!(f, "`-`"),
            Tok::Plus => write!(f, "`+`"),
            Tok::PlusPlus => write!(f, "`++`"),
            Tok::Star => write!(f, "`*`"),
            Tok::Slash => write!(f, "`/`"),
            Tok::Bang => write!(f, "`!`"),
            Tok::EqEq => write!(f, "`==`"),
            Tok::FatArrow => write!(f, "`=>`"),
            Tok::NotEq => write!(f, "`!=`"),
            Tok::Gt => write!(f, "`>`"),
            Tok::Gte => write!(f, "`>=`"),
            Tok::Lt => write!(f, "`<`"),
            Tok::Lte => write!(f, "`<=`"),
            Tok::AndAnd => write!(f, "`&&`"),
            Tok::OrOr => write!(f, "`||`"),
            Tok::Eof => write!(f, "end of input"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("line {line}: {message}")]
pub struct LexError {
    pub line: usize,
    pub message: String,
}

pub fn tokenize(src: &str) -> Result<Vec<Token>, LexError> {
    let chars: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut line = 1usize;

    macro_rules! push {
        ($tok:expr, $len:expr) => {{
            out.push(Token { tok: $tok, line });
            i += $len;
        }};
    }

    while i < chars.len() {
        let c = chars[i];

        if c == '\n' {
            line += 1;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // `--` comment to end of line — SQL's spelling, which is what the
        // pipelines use; the `---` banners are just that rule applied twice.
        // Checked before `->` and `-` so a comment never lexes as an arrow.
        if c == '-' && chars.get(i + 1) == Some(&'-') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        match c {
            '"' => {
                let (text, consumed, newlines) = lex_string(&chars, i, line)?;
                out.push(Token {
                    tok: Tok::Str(text),
                    line,
                });
                i += consumed;
                line += newlines;
            }
            '→' => push!(Tok::Arrow, 1),
            '-' if chars.get(i + 1) == Some(&'>') => push!(Tok::Arrow, 2),
            '-' => push!(Tok::Minus, 1),
            '+' if chars.get(i + 1) == Some(&'+') => push!(Tok::PlusPlus, 2),
            '+' => push!(Tok::Plus, 1),
            '*' => push!(Tok::Star, 1),
            '/' => push!(Tok::Slash, 1),
            '<' if chars.get(i + 1) == Some(&'-') => push!(Tok::Assign, 2),
            '<' if chars.get(i + 1) == Some(&'=') => push!(Tok::Lte, 2),
            '<' => push!(Tok::Lt, 1),
            '>' if chars.get(i + 1) == Some(&'=') => push!(Tok::Gte, 2),
            '>' => push!(Tok::Gt, 1),
            '=' if chars.get(i + 1) == Some(&'=') => push!(Tok::EqEq, 2),
            // `=>` introduces a lambda. Must be tested after `==` so that
            // `a == b` is never mis-lexed, and there is deliberately no bare
            // `=` token: assignment is `<-`, and a lone `=` in this corpus is
            // a named argument, which is a separate feature.
            '=' if chars.get(i + 1) == Some(&'>') => push!(Tok::FatArrow, 2),
            '!' if chars.get(i + 1) == Some(&'=') => push!(Tok::NotEq, 2),
            '!' => push!(Tok::Bang, 1),
            '&' if chars.get(i + 1) == Some(&'&') => push!(Tok::AndAnd, 2),
            '|' if chars.get(i + 1) == Some(&'|') => push!(Tok::OrOr, 2),
            '.' => push!(Tok::Dot, 1),
            ',' => push!(Tok::Comma, 1),
            ':' => push!(Tok::Colon, 1),
            '(' => push!(Tok::LParen, 1),
            ')' => push!(Tok::RParen, 1),
            '{' => push!(Tok::LBrace, 1),
            '}' => push!(Tok::RBrace, 1),
            '[' => push!(Tok::LBracket, 1),
            ']' => push!(Tok::RBracket, 1),
            _ if c.is_ascii_digit() => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                if matches!(chars.get(i), Some('e') | Some('E')) {
                    i += 1;
                    if matches!(chars.get(i), Some('+') | Some('-')) {
                        i += 1;
                    }
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let text: String = chars[start..i].iter().collect();
                // Parsed as a `serde_json::Number`, not an `f64`: an integer
                // literal past 2^53 would otherwise be silently rounded on the
                // way in — `9007199254740993` emitting as `…992` — and these
                // literals end up in ids and counters inside written artifacts.
                let value =
                    serde_json::from_str::<serde_json::Number>(&text).map_err(|_| LexError {
                        line,
                        message: format!("`{text}` is not a number"),
                    })?;
                out.push(Token {
                    tok: Tok::Num(value),
                    line,
                });
            }
            _ if is_ident_start(c) => {
                let start = i;
                while i < chars.len() && is_ident_continue(chars[i]) {
                    i += 1;
                }
                out.push(Token {
                    tok: Tok::Ident(chars[start..i].iter().collect()),
                    line,
                });
            }
            _ => {
                return Err(LexError {
                    line,
                    message: format!("unexpected character `{c}`"),
                })
            }
        }
    }

    out.push(Token {
        tok: Tok::Eof,
        line,
    });
    Ok(out)
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Read a `"…"` literal. Returns the decoded text, how many chars were
/// consumed including both quotes, and how many newlines were crossed.
///
/// `{{` / `}}` are left intact: interpolation is the parser's job, and it needs
/// to see the markers.
fn lex_string(
    chars: &[char],
    start: usize,
    line: usize,
) -> Result<(String, usize, usize), LexError> {
    let mut text = String::new();
    let mut i = start + 1;
    let mut newlines = 0usize;

    while i < chars.len() {
        match chars[i] {
            '"' => return Ok((text, i - start + 1, newlines)),
            '\\' => {
                let escaped = chars.get(i + 1).copied().ok_or_else(|| LexError {
                    line,
                    message: "string ends with a dangling `\\`".to_string(),
                })?;
                // Unknown escapes are rejected, not silently unwrapped to the
                // following character. Swallowing the backslash turned
                // `"state\quality"` into `"statequality"` and `"state\trigger"`
                // into a tab — quietly retargeting a read or a write, and
                // dodging a path-prefixed schema rule on the way.
                text.push(match escaped {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '"' => '"',
                    '/' => '/',
                    other => {
                        return Err(LexError {
                            line,
                            message: format!(
                                "unknown escape `\\{other}`; write `\\\\` for a literal backslash"
                            ),
                        })
                    }
                });
                i += 2;
            }
            c => {
                if c == '\n' {
                    newlines += 1;
                }
                text.push(c);
                i += 1;
            }
        }
    }

    Err(LexError {
        line,
        message: "unterminated string".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<Tok> {
        tokenize(src).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn comment_banners_are_not_arrows() {
        // `--- ## Step 1` begins with three dashes; a naive `->` rule would see
        // a stray `-` then an arrow and derail the whole file.
        assert_eq!(
            kinds("--- ## Step 1: x\nx <- 1"),
            vec![
                Tok::Ident("x".into()),
                Tok::Assign,
                Tok::Num(1.into()),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn a_two_dash_trailing_comment_is_a_comment() {
        // `→ step  -- why` is the common spelling in Demerzel's pipelines.
        assert_eq!(
            kinds("x <- 1  -- max 3 recons/day"),
            vec![
                Tok::Ident("x".into()),
                Tok::Assign,
                Tok::Num(1.into()),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn negative_numbers_still_lex_after_the_comment_rule() {
        assert_eq!(
            kinds("x <- -1"),
            vec![
                Tok::Ident("x".into()),
                Tok::Assign,
                Tok::Minus,
                Tok::Num(1.into()),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn both_arrow_spellings_lex_the_same() {
        assert_eq!(kinds("a → b"), kinds("a -> b"));
    }

    #[test]
    fn assign_is_not_a_less_than() {
        assert_eq!(kinds("a <- b")[1], Tok::Assign);
        assert_eq!(kinds("a <= b")[1], Tok::Lte);
        assert_eq!(kinds("a < b")[1], Tok::Lt);
    }

    #[test]
    fn interpolation_markers_survive_the_lexer() {
        assert_eq!(
            kinds(r#""{{a.b}}-tail""#),
            vec![Tok::Str("{{a.b}}-tail".into()), Tok::Eof]
        );
    }

    #[test]
    fn lines_are_tracked_across_strings_and_comments() {
        let toks = tokenize("--- c\nx <- 1\n\ny <- 2").unwrap();
        let y = toks
            .iter()
            .find(|t| t.tok == Tok::Ident("y".into()))
            .unwrap();
        assert_eq!(4, y.line);
    }

    #[test]
    fn unterminated_string_is_an_error_not_a_panic() {
        assert!(tokenize("x <- \"oops").is_err());
    }

    #[test]
    fn integer_literals_beyond_the_f64_range_keep_their_exact_value() {
        // 2^53 + 1: representable as an i64, not as an f64. Going through f64
        // silently produced …992.
        let Tok::Num(n) = &tokenize("x <- 9007199254740993").unwrap()[2].tok else {
            panic!("expected a number");
        };
        assert_eq!("9007199254740993", n.to_string());
    }

    #[test]
    fn fractional_literals_still_lex_as_fractions() {
        let Tok::Num(n) = &tokenize("x <- 0.15").unwrap()[2].tok else {
            panic!("expected a number");
        };
        assert_eq!(Some(0.15), n.as_f64());
    }

    #[test]
    fn known_escapes_decode() {
        assert_eq!(
            kinds(r#""a\nb\t\\c\"d""#),
            vec![Tok::Str("a\nb\t\\c\"d".into()), Tok::Eof]
        );
    }

    #[test]
    fn an_unknown_escape_is_rejected_rather_than_swallowing_the_backslash() {
        // `"state\quality"` used to decode to `"statequality"` and
        // `"state\trigger.json"` gained a tab — quietly retargeting a write and
        // dodging a path-prefixed schema rule.
        let err = tokenize(r#"p <- "state\quality""#).unwrap_err();
        assert!(err.to_string().contains("unknown escape"), "{err}");
        // `\t` IS a known escape, so this path silently becomes
        // `state<TAB>rigger.json` — a different file. Nothing in the lexer can
        // catch that one; it is why authors must write `\\` and why the
        // *unknown* escapes above now fail loudly instead of joining it.
        let toks = tokenize(r#"p <- "state\trigger.json""#).unwrap();
        assert_eq!(Tok::Str("state\trigger.json".into()), toks[2].tok);
    }
}
