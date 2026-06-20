//! A tiny JSON-Schema builder DSL for MCP tool input/output schemas.
//!
//! Tool schemas were hand-written as `serde_json::json!({...})` blobs — the same
//! property shapes (`{"type":"array","items":{"type":"number"},…}`) re-typed in
//! `tools.rs` literals and again in every migrated `#[ix_skill]` `schema_fn`. This
//! collapses those shapes to composable helpers so a schema reads as intent, not
//! boilerplate, and the shapes stop diverging.
//!
//! Faithful by construction: constraint setters take `impl Into<Value>`, so
//! `.minimum(1)` emits an integer `1` and `.exclusive_max(0.5)` emits a float —
//! the advertised schema is byte-equivalent to the old literals (see tests).

use serde_json::{Map, Value};

/// A single JSON-Schema property under construction (the value of one
/// `properties[name]` entry). Build with a constructor, refine with chained
/// setters, e.g. `Prop::integer().minimum(1).default(100).desc("Max iterations")`.
pub(crate) struct Prop(Map<String, Value>);

impl Prop {
    fn typed(t: &str) -> Self {
        let mut m = Map::new();
        m.insert("type".into(), Value::String(t.into()));
        Prop(m)
    }

    pub(crate) fn number() -> Self {
        Self::typed("number")
    }
    pub(crate) fn integer() -> Self {
        Self::typed("integer")
    }
    pub(crate) fn string() -> Self {
        Self::typed("string")
    }
    pub(crate) fn boolean() -> Self {
        Self::typed("boolean")
    }

    /// `[number]` — an array of numbers.
    pub(crate) fn num_array() -> Self {
        let mut p = Self::typed("array");
        p.0.insert("items".into(), Self::number().build());
        p
    }
    /// `[integer]` — an array of integers.
    pub(crate) fn int_array() -> Self {
        let mut p = Self::typed("array");
        p.0.insert("items".into(), Self::integer().build());
        p
    }
    /// `[[number]]` — a row-major numeric matrix.
    pub(crate) fn num_matrix() -> Self {
        let mut p = Self::typed("array");
        p.0.insert("items".into(), Self::num_array().build());
        p
    }
    /// `[object]` — an array of free-form objects.
    pub(crate) fn obj_array() -> Self {
        let mut p = Self::typed("array");
        p.0.insert("items".into(), Self::typed("object").build());
        p
    }

    pub(crate) fn desc(mut self, d: &str) -> Self {
        self.0.insert("description".into(), Value::String(d.into()));
        self
    }
    pub(crate) fn minimum(mut self, v: impl Into<Value>) -> Self {
        self.0.insert("minimum".into(), v.into());
        self
    }
    pub(crate) fn maximum(mut self, v: impl Into<Value>) -> Self {
        self.0.insert("maximum".into(), v.into());
        self
    }
    pub(crate) fn exclusive_min(mut self, v: impl Into<Value>) -> Self {
        self.0.insert("exclusiveMinimum".into(), v.into());
        self
    }
    pub(crate) fn exclusive_max(mut self, v: impl Into<Value>) -> Self {
        self.0.insert("exclusiveMaximum".into(), v.into());
        self
    }
    pub(crate) fn default(mut self, v: impl Into<Value>) -> Self {
        self.0.insert("default".into(), v.into());
        self
    }
    pub(crate) fn enum_of(mut self, variants: &[&str]) -> Self {
        self.0.insert(
            "enum".into(),
            Value::Array(
                variants
                    .iter()
                    .map(|s| Value::String((*s).into()))
                    .collect(),
            ),
        );
        self
    }
    /// Set a `minimum` on the array's `items` (e.g. integer labels `>= 0`).
    pub(crate) fn item_minimum(mut self, v: impl Into<Value>) -> Self {
        if let Some(Value::Object(items)) = self.0.get_mut("items") {
            items.insert("minimum".into(), v.into());
        }
        self
    }

    fn build(self) -> Value {
        Value::Object(self.0)
    }
}

/// An object schema with `required` fields — the standard MCP tool input schema.
pub(crate) fn object(props: Vec<(&str, Prop)>, required: &[&str]) -> Value {
    let mut schema = build_object(props);
    if let Value::Object(m) = &mut schema {
        m.insert(
            "required".into(),
            Value::Array(
                required
                    .iter()
                    .map(|s| Value::String((*s).into()))
                    .collect(),
            ),
        );
    }
    schema
}

/// An object schema with no `required` block — used for output schemas.
pub(crate) fn output(props: Vec<(&str, Prop)>) -> Value {
    build_object(props)
}

fn build_object(props: Vec<(&str, Prop)>) -> Value {
    let mut properties = Map::new();
    for (name, prop) in props {
        properties.insert(name.into(), prop.build());
    }
    let mut m = Map::new();
    m.insert("type".into(), Value::String("object".into()));
    m.insert("properties".into(), Value::Object(properties));
    Value::Object(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Faithfulness: the DSL emits a Value EQUAL to the hand-written literal.
    // serde_json::Value compares object members structurally (order-independent),
    // so this proves the advertised schema does not drift.

    #[test]
    fn num_array_object_matches_literal() {
        let built = object(
            vec![(
                "data",
                Prop::num_array().desc("List of numbers to compute statistics on"),
            )],
            &["data"],
        );
        let literal = json!({
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": { "type": "number" },
                    "description": "List of numbers to compute statistics on"
                }
            },
            "required": ["data"]
        });
        assert_eq!(built, literal);
    }

    #[test]
    fn integer_constraints_preserve_int_vs_float() {
        // The tricky case: `minimum`/`default` must stay INTEGERS, while a
        // float bound stays a float. `impl Into<Value>` infers each.
        let built = object(
            vec![
                ("k", Prop::integer().minimum(1).desc("Number of clusters")),
                (
                    "max_iter",
                    Prop::integer().default(100).desc("Max iterations"),
                ),
                (
                    "cutoff",
                    Prop::number()
                        .exclusive_min(0)
                        .exclusive_max(0.5)
                        .desc("Normalized cutoff"),
                ),
            ],
            &["k"],
        );
        let literal = json!({
            "type": "object",
            "properties": {
                "k": { "type": "integer", "minimum": 1, "description": "Number of clusters" },
                "max_iter": { "type": "integer", "default": 100, "description": "Max iterations" },
                "cutoff": { "type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 0.5, "description": "Normalized cutoff" }
            },
            "required": ["k"]
        });
        assert_eq!(built, literal);
    }

    #[test]
    fn enum_matrix_item_min_and_output_shapes() {
        let built = object(
            vec![
                (
                    "metric",
                    Prop::string()
                        .enum_of(&["euclidean", "cosine", "manhattan"])
                        .desc("Distance metric"),
                ),
                ("data", Prop::num_matrix().desc("rows")),
                (
                    "labels",
                    Prop::int_array().item_minimum(0).desc("Cluster id per row"),
                ),
                ("flag", Prop::boolean().default(false).desc("toggle")),
            ],
            &["metric", "data"],
        );
        let literal = json!({
            "type": "object",
            "properties": {
                "metric": { "type": "string", "enum": ["euclidean", "cosine", "manhattan"], "description": "Distance metric" },
                "data": { "type": "array", "items": { "type": "array", "items": { "type": "number" } }, "description": "rows" },
                "labels": { "type": "array", "items": { "type": "integer", "minimum": 0 }, "description": "Cluster id per row" },
                "flag": { "type": "boolean", "default": false, "description": "toggle" }
            },
            "required": ["metric", "data"]
        });
        assert_eq!(built, literal);

        let out = output(vec![("n", Prop::integer().desc("count"))]);
        assert_eq!(
            out,
            json!({ "type": "object", "properties": { "n": { "type": "integer", "description": "count" } } })
        );
    }
}
