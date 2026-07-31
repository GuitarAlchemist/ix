//! Recursive-descent parser for the IXQL pipeline surface.
//!
//! # Which IXQL is this?
//!
//! Demerzel carries two IXQL surfaces. `tree-sitter-ixql/grammar.js` describes
//! the ML-pipeline dialect from `grammars/sci-ml-pipelines.ebnf` (`csv(…) →
//! train(…)`). The files under `Demerzel/pipelines/*.ixql` — the ones the
//! governance cycles are actually written in — use the binding/record dialect
//! the design spec's AST models: `name <- expr`, records, `→` steps, and a
//! `compound:` tail. This parser covers that second dialect, which is the one
//! the executor has to run.
//!
//! Statements have no terminator, so the two boundary rules are:
//!
//! - a postfix `(` opens a call only when it is on the same line as its callee;
//! - `.` and `(` are the only postfix continuations, so a record or string
//!   followed by a bare identifier ends the statement.

use std::collections::BTreeMap;

use crate::ast::{BinaryOp, Block, CompoundOp, Expr, Literal, PipeStep, Statement, UnaryOp};
use crate::lexer::{tokenize, LexError, Tok, Token};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    #[error("{0}")]
    Lex(#[from] LexError),
    #[error("line {line}: {message}")]
    Syntax { line: usize, message: String },
}

impl ParseError {
    fn at(line: usize, message: impl Into<String>) -> Self {
        ParseError::Syntax {
            line,
            message: message.into(),
        }
    }
}

/// Parse a whole `.ixql` document.
pub fn parse_program(src: &str) -> Result<Block, ParseError> {
    let tokens = tokenize(src)?;
    let mut p = Parser::new(tokens);
    let block = p.parse_block()?;
    p.expect_eof()?;
    Ok(block)
}

/// Parse a single expression — used for `{{…}}` interpolation and for the
/// `check:` predicate strings that `tars.validate` carries.
pub fn parse_expression(src: &str) -> Result<Expr, ParseError> {
    let tokens = tokenize(src)?;
    let mut p = Parser::new(tokens);
    let expr = p.parse_expr()?;
    p.expect_eof()?;
    Ok(expr)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    // ---- token helpers -------------------------------------------------

    fn peek(&self) -> &Tok {
        &self.tokens[self.pos].tok
    }

    fn peek_at(&self, offset: usize) -> &Tok {
        let index = (self.pos + offset).min(self.tokens.len() - 1);
        &self.tokens[index].tok
    }

    fn line(&self) -> usize {
        self.tokens[self.pos].line
    }

    fn prev_line(&self) -> usize {
        self.tokens[self.pos.saturating_sub(1)].line
    }

    fn bump(&mut self) -> Tok {
        let tok = self.tokens[self.pos].tok.clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn eat(&mut self, want: &Tok) -> bool {
        if self.peek() == want {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, want: &Tok) -> Result<(), ParseError> {
        if self.eat(want) {
            Ok(())
        } else {
            Err(ParseError::at(
                self.line(),
                format!("expected {want}, found {}", self.peek()),
            ))
        }
    }

    fn expect_eof(&self) -> Result<(), ParseError> {
        if matches!(self.peek(), Tok::Eof) {
            Ok(())
        } else {
            Err(ParseError::at(
                self.line(),
                format!("unexpected {} after the last statement", self.peek()),
            ))
        }
    }

    fn peek_keyword(&self, word: &str) -> bool {
        matches!(self.peek(), Tok::Ident(name) if name == word)
    }

    fn eat_keyword(&mut self, word: &str) -> bool {
        if self.peek_keyword(word) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.bump() {
            Tok::Ident(name) => Ok(name),
            other => Err(ParseError::at(
                self.prev_line(),
                format!("expected an identifier, found {other}"),
            )),
        }
    }

    // ---- statements ----------------------------------------------------

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let mut block = Block::new();
        while !matches!(self.peek(), Tok::Eof) {
            block.push(self.parse_statement()?);
        }
        Ok(block)
    }

    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        if self.peek_keyword("when") {
            return Err(ParseError::at(
                self.line(),
                "`when` blocks are not supported by this parser yet — no pipeline in \
                 Demerzel/pipelines uses one, so the block-delimiting rule is still undecided",
            ));
        }

        // `name <- …`
        if matches!(self.peek(), Tok::Ident(_)) && matches!(self.peek_at(1), Tok::Assign) {
            let name = self.expect_ident()?;
            self.bump(); // `<-`
            let value = self.parse_pipeline()?;
            return Ok(Statement::Assign(name, Box::new(value)));
        }

        Ok(Statement::Do(Box::new(self.parse_pipeline()?)))
    }

    // ---- pipelines -----------------------------------------------------

    fn parse_pipeline(&mut self) -> Result<Expr, ParseError> {
        let head = self.parse_expr()?;
        let mut steps = Vec::new();
        while self.eat(&Tok::Arrow) {
            steps.push(self.parse_pipe_step()?);
        }
        if steps.is_empty() {
            Ok(head)
        } else {
            Ok(Expr::Pipeline(Box::new(head), steps))
        }
    }

    fn parse_pipe_step(&mut self) -> Result<PipeStep, ParseError> {
        if self.peek_keyword("compound") {
            self.bump();
            self.expect(&Tok::Colon)?;
            return Ok(PipeStep::Compound(self.parse_compound_ops()?));
        }

        let line = self.line();
        match self.parse_expr()? {
            Expr::Call {
                target,
                positional,
                named,
            } => Ok(PipeStep::CallStep {
                target,
                positional,
                named,
            }),
            _ => Err(ParseError::at(
                line,
                "a `→` step must be a call — a bare value cannot consume the piped input",
            )),
        }
    }

    fn parse_compound_ops(&mut self) -> Result<Vec<CompoundOp>, ParseError> {
        let mut ops = Vec::new();
        loop {
            if self.eat_keyword("harvest") {
                ops.push(CompoundOp::Harvest(Box::new(self.parse_expr()?)));
            } else if self.eat_keyword("promote") {
                let id = self.expect_ident()?;
                let condition = if self.eat_keyword("when") {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                };
                ops.push(CompoundOp::Promote { id, condition });
            } else if self.eat_keyword("log") {
                let id = self.expect_ident()?;
                if !self.eat_keyword("to") {
                    return Err(ParseError::at(
                        self.line(),
                        "expected `to` after `log <id>`",
                    ));
                }
                ops.push(CompoundOp::Log {
                    id,
                    destination: Box::new(self.parse_expr()?),
                });
            } else if self.eat_keyword("teach") {
                let id = self.expect_ident()?;
                if !self.eat_keyword("to") {
                    return Err(ParseError::at(
                        self.line(),
                        "expected `to` after `teach <id>`",
                    ));
                }
                ops.push(CompoundOp::Teach {
                    id,
                    target: self.expect_ident()?,
                });
            } else {
                break;
            }
        }

        if ops.is_empty() {
            return Err(ParseError::at(
                self.line(),
                "a `compound:` block needs at least one of harvest/promote/log/teach",
            ));
        }
        Ok(ops)
    }

    // ---- expressions ---------------------------------------------------

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        // Lambdas bind loosest, and the body is parsed with `parse_expr` again
        // so `x => a && b` takes the whole conjunction as the body rather than
        // stopping at `a`.
        if let Some(params) = self.take_lambda_params()? {
            let body = Box::new(self.parse_expr()?);
            return Ok(Expr::Lambda { params, body });
        }
        self.parse_or()
    }

    /// Recognise and consume a lambda head, or consume nothing.
    ///
    /// `x =>` is decidable with one token of lookahead. `(a, b) =>` is not: a
    /// `(` also opens a grouped expression, and the two only diverge after the
    /// matching `)`. Rather than parse-and-backtrack — which would mean
    /// unwinding a partly built AST — this scans ahead for the matching paren
    /// and commits only once it has seen the `=>` behind it. Nothing is
    /// consumed on the `None` paths.
    fn take_lambda_params(&mut self) -> Result<Option<Vec<String>>, ParseError> {
        if matches!(self.peek(), Tok::Ident(_)) && matches!(self.peek_at(1), Tok::FatArrow) {
            let param = self.expect_ident()?;
            self.bump(); // `=>`
            return Ok(Some(vec![param]));
        }

        if !matches!(self.peek(), Tok::LParen) {
            return Ok(None);
        }

        let mut depth = 0usize;
        let mut offset = 0usize;
        loop {
            match self.peek_at(offset) {
                Tok::LParen => depth += 1,
                Tok::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                // `peek_at` clamps to `Eof`, so an unbalanced `(` terminates
                // here instead of scanning forever. Report it as not-a-lambda
                // and let the normal path produce the real error.
                Tok::Eof => return Ok(None),
                _ => {}
            }
            offset += 1;
        }
        if !matches!(self.peek_at(offset + 1), Tok::FatArrow) {
            return Ok(None);
        }

        self.bump(); // `(`
        let mut params = Vec::new();
        while !matches!(self.peek(), Tok::RParen) {
            params.push(self.expect_ident()?);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::RParen)?;
        self.bump(); // `=>`
        Ok(Some(params))
    }

    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_and()?;
        while self.eat(&Tok::OrOr) {
            let right = self.parse_and()?;
            left = Expr::BinOp(Box::new(left), BinaryOp::Or, Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_comparison()?;
        while self.eat(&Tok::AndAnd) {
            let right = self.parse_comparison()?;
            left = Expr::BinOp(Box::new(left), BinaryOp::And, Box::new(right));
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_term()?;

        let op = match self.peek() {
            Tok::EqEq => Some(BinaryOp::Eq),
            Tok::NotEq => Some(BinaryOp::Neq),
            Tok::Gt => Some(BinaryOp::Gt),
            Tok::Gte => Some(BinaryOp::Gte),
            Tok::Lt => Some(BinaryOp::Lt),
            Tok::Lte => Some(BinaryOp::Lte),
            Tok::Ident(name) if name == "in" => Some(BinaryOp::In),
            Tok::Ident(name)
                if name == "not" && matches!(self.peek_at(1), Tok::Ident(w) if w == "in") =>
            {
                Some(BinaryOp::NotIn)
            }
            _ => None,
        };

        if let Some(op) = op {
            self.bump();
            if op == BinaryOp::NotIn {
                self.bump(); // the `in` of `not in`
            }
            let right = self.parse_term()?;
            return Ok(Expr::BinOp(Box::new(left), op, Box::new(right)));
        }

        // `x is empty` / `x is not empty`
        if self.peek_keyword("is") {
            self.bump();
            let negated = self.eat_keyword("not");
            if !self.eat_keyword("empty") {
                return Err(ParseError::at(
                    self.line(),
                    "expected `empty` after `is` / `is not`",
                ));
            }
            let op = if negated {
                UnaryOp::IsNotEmpty
            } else {
                UnaryOp::IsEmpty
            };
            return Ok(Expr::Unary(op, Box::new(left)));
        }

        Ok(left)
    }

    /// `+`, `-` and `++`, left-associative and looser than `*` / `/`.
    ///
    /// `++` sits at this level rather than its own: the corpus only ever
    /// concatenates whole sequences, so there is no expression where its
    /// precedence against `+` is observable.
    fn parse_term(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_factor()?;
        loop {
            let op = match self.peek() {
                Tok::Plus => BinaryOp::Add,
                Tok::Minus => BinaryOp::Sub,
                Tok::PlusPlus => BinaryOp::Concat,
                _ => return Ok(left),
            };
            self.bump();
            let right = self.parse_factor()?;
            left = Expr::BinOp(Box::new(left), op, Box::new(right));
        }
    }

    /// `*` and `/`, binding tighter than `+` / `-`.
    fn parse_factor(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Tok::Star => BinaryOp::Mul,
                Tok::Slash => BinaryOp::Div,
                _ => return Ok(left),
            };
            self.bump();
            let right = self.parse_unary()?;
            left = Expr::BinOp(Box::new(left), op, Box::new(right));
        }
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        if self.eat(&Tok::Bang) {
            return Ok(Expr::Unary(UnaryOp::Not, Box::new(self.parse_unary()?)));
        }
        if self.eat(&Tok::Minus) {
            let line = self.prev_line();
            return match self.parse_unary()? {
                Expr::Lit(Literal::Number(n)) => Ok(Expr::Lit(Literal::Number(negate(&n, line)?))),
                _ => Err(ParseError::at(
                    line,
                    "unary `-` is only defined on numeric literals",
                )),
            };
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;
        loop {
            if self.eat(&Tok::Dot) {
                expr = Expr::Member(Box::new(expr), self.expect_ident()?);
                continue;
            }
            // Same-line rule: without it, `x <- foo` followed by a statement
            // that opens with `(` would silently parse as a call.
            if matches!(self.peek(), Tok::LParen) && self.line() == self.prev_line() {
                self.bump();
                let (positional, named) = self.parse_arguments()?;
                expr = Expr::Call {
                    target: Box::new(expr),
                    positional,
                    named,
                };
                continue;
            }
            return Ok(expr);
        }
    }

    fn parse_arguments(&mut self) -> Result<(Vec<Expr>, BTreeMap<String, Expr>), ParseError> {
        let mut positional = Vec::new();
        let mut named = BTreeMap::new();

        while !matches!(self.peek(), Tok::RParen) {
            // `name: value` — but only when the `:` really follows an
            // identifier, so `a.b` and bare values still read as positional.
            if matches!(self.peek(), Tok::Ident(_)) && matches!(self.peek_at(1), Tok::Colon) {
                let key = self.expect_ident()?;
                self.bump(); // `:`
                let value = self.parse_pipeline()?;
                if named.insert(key.clone(), value).is_some() {
                    return Err(ParseError::at(
                        self.prev_line(),
                        format!("argument `{key}` given twice"),
                    ));
                }
            } else {
                if !named.is_empty() {
                    return Err(ParseError::at(
                        self.line(),
                        "positional arguments cannot follow named ones",
                    ));
                }
                positional.push(self.parse_pipeline()?);
            }

            if !self.eat(&Tok::Comma) {
                break;
            }
        }

        self.expect(&Tok::RParen)?;
        Ok((positional, named))
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let line = self.line();
        match self.bump() {
            Tok::Num(n) => Ok(Expr::Lit(Literal::Number(n))),
            Tok::Str(raw) => interpolate(&raw, line),
            Tok::Ident(name) => Ok(match name.as_str() {
                "true" => Expr::Lit(Literal::Bool(true)),
                "false" => Expr::Lit(Literal::Bool(false)),
                "null" => Expr::Lit(Literal::Null),
                _ => Expr::Var(name),
            }),
            Tok::LBracket => {
                let mut items = Vec::new();
                while !matches!(self.peek(), Tok::RBracket) {
                    items.push(self.parse_expr()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RBracket)?;
                Ok(Expr::Array(items))
            }
            Tok::LBrace => {
                let mut fields = Vec::new();
                while !matches!(self.peek(), Tok::RBrace) {
                    let key = match self.bump() {
                        Tok::Ident(name) => name,
                        Tok::Str(text) => text,
                        other => {
                            return Err(ParseError::at(
                                self.prev_line(),
                                format!(
                                    "record keys must be identifiers or strings, found {other}"
                                ),
                            ))
                        }
                    };
                    self.expect(&Tok::Colon)?;
                    fields.push((key, self.parse_expr()?));
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RBrace)?;
                Ok(Expr::Record(fields))
            }
            Tok::LParen => {
                let inner = self.parse_expr()?;
                self.expect(&Tok::RParen)?;
                Ok(inner)
            }
            other => Err(ParseError::at(
                line,
                format!("expected a value, found {other}"),
            )),
        }
    }
}

/// Negate a numeric literal without going through `f64`, so `-9007199254740993`
/// stays exact the way the positive form does.
fn negate(n: &serde_json::Number, line: usize) -> Result<serde_json::Number, ParseError> {
    if let Some(i) = n.as_i64() {
        return Ok(serde_json::Number::from(-i));
    }
    if let Some(u) = n.as_u64() {
        // The one integer that fits in u64 but whose negation fits in i64 is
        // 2^63; anything larger has no exact negative representation.
        if u == (i64::MAX as u64) + 1 {
            return Ok(serde_json::Number::from(i64::MIN));
        }
    }
    n.as_f64()
        .and_then(|f| serde_json::Number::from_f64(-f))
        .ok_or_else(|| ParseError::at(line, format!("`-{n}` is not a representable number")))
}

/// Split a string literal into its literal and `{{…}}` pieces.
///
/// A string with no markers stays a plain literal, so the common case costs
/// nothing at evaluation time.
fn interpolate(raw: &str, line: usize) -> Result<Expr, ParseError> {
    if !raw.contains("{{") {
        return Ok(Expr::Lit(Literal::String(raw.to_string())));
    }

    let mut parts: Vec<Expr> = Vec::new();
    let mut rest = raw;

    while let Some(open) = rest.find("{{") {
        let close = rest[open..].find("}}").ok_or_else(|| {
            ParseError::at(line, "interpolation opened with `{{` but never closed")
        })? + open;

        if open > 0 {
            parts.push(Expr::Lit(Literal::String(rest[..open].to_string())));
        }
        let inner = rest[open + 2..close].trim();
        if inner.is_empty() {
            return Err(ParseError::at(line, "empty `{{}}` interpolation"));
        }
        parts.push(parse_expression(inner)?);
        rest = &rest[close + 2..];
    }

    if !rest.is_empty() {
        parts.push(Expr::Lit(Literal::String(rest.to_string())));
    }
    Ok(Expr::Interpolation(parts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_of_a_record_then_the_next_binding() {
        // The statement boundary here is implicit: `}` followed by an
        // identifier must end the first statement.
        let block = parse_program("a <- { k: 1 }\nb <- 2").unwrap();
        assert_eq!(2, block.len());
        assert!(matches!(&block[0], Statement::Assign(name, _) if name == "a"));
        assert!(matches!(&block[1], Statement::Assign(name, _) if name == "b"));
    }

    #[test]
    fn a_call_cannot_span_a_line_break() {
        // `(x)` on its own line is a parenthesised statement, not a call of `a`.
        let block = parse_program("a <- b\n(c)").unwrap();
        assert_eq!(2, block.len());
    }

    #[test]
    fn namespaced_call_with_named_arguments() {
        let block =
            parse_program(r#"tars.validate(check: "a == 1", reject_message: "no")"#).unwrap();
        let Statement::Do(expr) = &block[0] else {
            panic!("expected a bare call statement");
        };
        let Expr::Call { target, named, .. } = expr.as_ref() else {
            panic!("expected a call");
        };
        assert_eq!(2, named.len());
        assert!(matches!(target.as_ref(), Expr::Member(_, m) if m == "validate"));
    }

    #[test]
    fn pipeline_steps_chain() {
        let block =
            parse_program("x <- read(\"p\")\n  → default({a: 1})\n  → tars.validate(check: \"x\")")
                .unwrap();
        let Statement::Assign(_, expr) = &block[0] else {
            panic!("expected an assignment");
        };
        let Expr::Pipeline(_, steps) = expr.as_ref() else {
            panic!("expected a pipeline");
        };
        assert_eq!(2, steps.len());
    }

    #[test]
    fn compound_block_collects_every_op() {
        let block = parse_program(
            "write(v)\n → compound:\n   harvest v.followups\n   log cycle to \"state/\"",
        )
        .unwrap();
        let Statement::Do(expr) = &block[0] else {
            panic!("expected a bare statement");
        };
        let Expr::Pipeline(_, steps) = expr.as_ref() else {
            panic!("expected a pipeline");
        };
        let PipeStep::Compound(ops) = &steps[0] else {
            panic!("expected a compound step");
        };
        assert_eq!(2, ops.len());
        assert!(matches!(&ops[1], CompoundOp::Log { id, .. } if id == "cycle"));
    }

    #[test]
    fn interpolation_splits_into_literal_and_expression_pieces() {
        let expr = parse_expression(r#""{{a.b}}-tail""#).unwrap();
        let Expr::Interpolation(parts) = expr else {
            panic!("expected interpolation");
        };
        assert_eq!(2, parts.len());
        assert!(matches!(&parts[0], Expr::Member(_, m) if m == "b"));
        assert!(matches!(&parts[1], Expr::Lit(Literal::String(s)) if s == "-tail"));
    }

    #[test]
    fn plain_string_stays_a_literal() {
        assert!(matches!(
            parse_expression(r#""no markers""#).unwrap(),
            Expr::Lit(Literal::String(_))
        ));
    }

    #[test]
    fn comparison_and_is_empty_parse() {
        assert!(matches!(
            parse_expression("v.schema_version == 1").unwrap(),
            Expr::BinOp(_, BinaryOp::Eq, _)
        ));
        assert!(matches!(
            parse_expression("v.followups is not empty").unwrap(),
            Expr::Unary(UnaryOp::IsNotEmpty, _)
        ));
    }

    #[test]
    fn a_bare_value_is_rejected_as_a_pipeline_step() {
        let err = parse_program("x <- 1 → 2").unwrap_err();
        assert!(err.to_string().contains("must be a call"));
    }

    #[test]
    fn when_is_refused_explicitly_rather_than_misparsed() {
        let err = parse_program("when x: y <- 1").unwrap_err();
        assert!(err.to_string().contains("`when` blocks are not supported"));
    }
    // ---- lambdas ---------------------------------------------------------

    #[test]
    fn a_single_parameter_lambda_takes_the_whole_body() {
        // `=>` binds loosest: the body is the entire conjunction, not just `a`.
        let expr = parse_expression("x => x.a && x.b").unwrap();
        let Expr::Lambda { params, body } = expr else {
            panic!("expected a lambda, got {expr:?}")
        };
        assert_eq!(vec!["x".to_string()], params);
        assert!(matches!(*body, Expr::BinOp(_, BinaryOp::And, _)));
    }

    #[test]
    fn a_parenthesised_parameter_list_is_a_lambda_not_a_group() {
        // The fold-shaped callers in the corpus: `(sum, d) => …`.
        let expr = parse_expression("(sum, d) => sum").unwrap();
        let Expr::Lambda { params, .. } = expr else {
            panic!("expected a lambda, got {expr:?}")
        };
        assert_eq!(vec!["sum".to_string(), "d".to_string()], params);
    }

    #[test]
    fn a_parenthesised_expression_is_still_a_group() {
        // The lookahead must not claim `(a && b)` — there is no `=>` behind
        // the matching paren, so nothing may be consumed.
        let expr = parse_expression("(a && b)").unwrap();
        assert!(matches!(expr, Expr::BinOp(_, BinaryOp::And, _)));
    }

    #[test]
    fn a_lambda_is_an_argument_to_a_higher_order_call() {
        let expr = parse_expression("items.map(i => i.name)").unwrap();
        let Expr::Call { positional, .. } = expr else {
            panic!("expected a call, got {expr:?}")
        };
        assert!(matches!(positional.as_slice(), [Expr::Lambda { .. }]));
    }

    // ---- arithmetic ------------------------------------------------------

    #[test]
    fn multiplication_binds_tighter_than_addition() {
        // Shape, not value: `2 + (3 * 4)`, so the top node is the `+`.
        let expr = parse_expression("2 + 3 * 4").unwrap();
        let Expr::BinOp(_, BinaryOp::Add, right) = expr else {
            panic!("expected `+` at the top, got {expr:?}")
        };
        assert!(matches!(*right, Expr::BinOp(_, BinaryOp::Mul, _)));
    }

    #[test]
    fn arithmetic_binds_tighter_than_comparison() {
        // `(a * 0.7) >= b`, otherwise the comparison would swallow the `*`.
        let expr = parse_expression("a * 0.7 >= b").unwrap();
        let Expr::BinOp(left, BinaryOp::Gte, _) = expr else {
            panic!("expected `>=` at the top, got {expr:?}")
        };
        assert!(matches!(*left, Expr::BinOp(_, BinaryOp::Mul, _)));
    }

    #[test]
    fn double_plus_is_concatenation_not_two_additions() {
        let expr = parse_expression("a ++ b").unwrap();
        assert!(matches!(expr, Expr::BinOp(_, BinaryOp::Concat, _)));
    }

    // ---- pipelines as arguments -----------------------------------------

    #[test]
    fn an_argument_may_be_a_whole_pipeline() {
        // `fan_out` branches are pipelines; a `,` ends one, and the `→`
        // belongs to the argument because the outer pipeline cannot resume
        // until the `)`.
        let expr = parse_expression("fan_out(a -> f(), b -> g())").unwrap();
        let Expr::Call { positional, .. } = expr else {
            panic!("expected a call, got {expr:?}")
        };
        assert_eq!(2, positional.len());
        assert!(positional
            .iter()
            .all(|arg| matches!(arg, Expr::Pipeline(_, _))));
    }
}
