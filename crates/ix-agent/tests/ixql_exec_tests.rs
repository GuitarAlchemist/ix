use ix_agent::ast::{Expr, Literal, Statement, BinaryOp, PipeStep};
use ix_agent::ixql::{ExecutionContext, eval_expr, execute_block};
use serde_json::json;
use std::collections::HashMap;

#[tokio::test]
async fn test_ixql_ast_evaluation_basic() {
    let mut ctx = ExecutionContext::new();
    
    // Assign a variable: x <- 42.0
    let assign_stmt = Statement::Assign(
        "x".to_string(),
        Box::new(Expr::Lit(Literal::Number(42.0))),
    );
    
    let block = vec![assign_stmt];
    execute_block(&mut ctx, &block).await.unwrap();
    
    assert_eq!(ctx.env.get("x").unwrap(), &json!(42.0));
}

#[tokio::test]
async fn test_ixql_binops_and_conditionals() {
    let mut ctx = ExecutionContext::new();
    ctx.env.insert("x".to_string(), json!(10.0));
    ctx.env.insert("y".to_string(), json!(20.0));

    // Evaluate x < y
    let check_expr = Expr::BinOp(
        Box::new(Expr::Var("x".to_string())),
        BinaryOp::Lt,
        Box::new(Expr::Var("y".to_string())),
    );

    let res = eval_expr(&mut ctx, &check_expr).await.unwrap();
    assert_eq!(res, json!(true));

    // When x < y: z <- "Passed"
    let when_stmt = Statement::When(
        Box::new(check_expr),
        Box::new(vec![Statement::Assign(
            "z".to_string(),
            Box::new(Expr::Lit(Literal::String("Passed".to_string()))),
        )]),
    );

    execute_block(&mut ctx, &vec![when_stmt]).await.unwrap();
    assert_eq!(ctx.env.get("z").unwrap(), &json!("Passed"));
}

#[tokio::test]
async fn test_ixql_baml_mock_invocation() {
    std::env::set_var("IXQL_TEST_MOCK_BAML", "1");
    let mut ctx = ExecutionContext::new();

    // Setup input variables matching types
    ctx.env.insert("telemetry".to_string(), json!({
        "errors": [],
        "warnings": [],
        "metrics": {}
    }));
    ctx.env.insert("role".to_string(), json!("Architect"));
    ctx.env.insert("bounds".to_string(), json!({
        "max_token_spend": 1000,
        "allowed_libs": []
    }));

    // BAML function call expr: baml.EvaluateSignalSwarm(telemetry, role, bounds)
    let baml_call = Expr::Call {
        target: Box::new(Expr::Var("baml.EvaluateSignalSwarm".to_string())),
        positional: vec![
            Expr::Var("telemetry".to_string()),
            Expr::Var("role".to_string()),
            Expr::Var("bounds".to_string()),
        ],
        named: HashMap::new(),
    };

    let result = eval_expr(&mut ctx, &baml_call).await.unwrap();
    
    // Assert against mock shape
    assert_eq!(result.get("agent_id").unwrap().as_str().unwrap(), "test-agent");
    assert_eq!(result.get("role").unwrap().as_str().unwrap(), "Architect");
    assert_eq!(result.get("value").unwrap().as_str().unwrap(), "TrueVal");
}

#[tokio::test]
async fn test_ixql_pipeline_filter() {
    let mut ctx = ExecutionContext::new();

    // Data array: [ { "status": "pending" }, { "status": "processed" } ]
    let data = json!([
        { "status": "pending", "id": 1 },
        { "status": "processed", "id": 2 }
    ]);
    ctx.env.insert("data".to_string(), data);

    // pipeline: data -> filter(status == "pending")
    // Note: in ixql, filter(status == "pending") evaluates item fields.
    // For our simplified mock filter evaluator, it evaluates condition where 'r' is the item.
    // So filter(r.status == "pending") is the expression.
    let filter_expr = Expr::Pipeline(
        Box::new(Expr::Var("data".to_string())),
        vec![
            PipeStep::CallStep {
                target: Box::new(Expr::Var("filter".to_string())),
                positional: vec![
                    Expr::BinOp(
                        Box::new(Expr::Member(Box::new(Expr::Var("r".to_string())), "status".to_string())),
                        BinaryOp::Eq,
                        Box::new(Expr::Lit(Literal::String("pending".to_string()))),
                    )
                ],
                named: HashMap::new(),
            }
        ]
    );

    let result = eval_expr(&mut ctx, &filter_expr).await.unwrap();
    let arr = result.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0].get("id").unwrap().as_i64().unwrap(), 1);
}
