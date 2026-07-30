use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    And,
    Or,
    In,
    NotIn,
    IsEmpty,
    IsNotEmpty,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Lit(Literal),
    Var(String),
    Member(Box<Expr>, String),
    Array(Vec<Expr>),
    Record(HashMap<String, Expr>),
    Interpolation(Vec<Expr>),
    BinOp(Box<Expr>, BinaryOp, Box<Expr>),
    Call {
        target: Box<Expr>,
        positional: Vec<Expr>,
        named: HashMap<String, Expr>,
    },
    Lambda(Vec<String>, Box<Block>),
    Pipeline(Box<Expr>, Vec<PipeStep>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipeStep {
    CallStep {
        target: Box<Expr>,
        positional: Vec<Expr>,
        named: HashMap<String, Expr>,
    },
    FanOut(Vec<Block>),
    Parallel(Vec<Block>),
    Compound(Vec<CompoundOp>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompoundOp {
    Harvest(Box<Expr>),
    Promote {
        id: String,
        condition: Option<Box<Expr>>,
    },
    Log {
        id: String,
        destination: Box<Expr>,
    },
    Teach {
        id: String,
        target: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Assign(String, Box<Expr>),
    Do(Box<Expr>),
    When(Box<Expr>, Box<Block>),
}

pub type Block = Vec<Statement>;
