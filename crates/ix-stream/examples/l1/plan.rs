//! The parts of the L1 loop over tracer A (main.rs) that call nothing: the chatbot request
//! templates, the hash that freezes the judge, the graph of a run, and the choice. Shared with
//! tests/l1.rs.
#![allow(dead_code)]

use serde_json::{json, Value};

/// The chatbot request templates L1 tries, in order; the first is the baseline. `{prompt}` is
/// the question. They name no note, chord or interval, so no question's expected value.
pub const TEMPLATES: &[(&str, &str)] = &[
    ("baseline", "{prompt}"),
    ("notes", "{prompt}\n\nWrite every chord you name with its notes in parentheses."),
    ("first", "{prompt}\n\nPut your answer in the first sentence, then explain."),
    ("key", "{prompt}\n\nSay which key you are working in."),
    (
        "first+notes",
        "{prompt}\n\nPut your answer in the first sentence, then explain. Write every chord you name with its notes in parentheses.",
    ),
    (
        "first+notes+key",
        "{prompt}\n\nPut your answer in the first sentence, then explain. Write every chord you name with its notes in parentheses. Say which key you are working in.",
    ),
];

/// The nodes L1 may not change: the judge, its extraction, its report and the GaApi node.
pub const FROZEN: &[&str] = &["extract", "ga", "judge", "report"];

/// The words of music a template may not hold, besides note and chord names.
const MUSIC: &[&str] = &[
    "unison",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "octave",
    "semitone",
    "major",
    "minor",
    "diminished",
    "augmented",
    "perfect",
    "dim",
    "aug",
    "maj",
    "min",
];

pub fn node<'a>(graph: &'a Value, name: &str) -> Option<&'a Value> {
    graph["nodes"]
        .as_array()?
        .iter()
        .find(|n| n["name"] == name)
}

/// The blake3 hash of the frozen nodes, as JSON, in the order of [`FROZEN`].
pub fn frozen(graph: &Value) -> Result<String, String> {
    let mut h = blake3::Hasher::new();
    for name in FROZEN {
        let n = node(graph, name).ok_or_else(|| format!("the graph has no `{name}` node"))?;
        h.update(name.as_bytes());
        h.update(&[0]);
        h.update(n.to_string().as_bytes());
        h.update(&[0]);
    }
    Ok(h.finalize().to_hex().to_string())
}

/// Why `template` may not be tried: a placeholder other than one `{prompt}`, or a note, chord
/// or interval word in its own text.
pub fn refuse(template: &str) -> Option<String> {
    if template.matches("{prompt}").count() != 1 {
        return Some("a template holds `{prompt}` once".into());
    }
    let own = template.replace("{prompt}", " ");
    if own.contains(['{', '}']) {
        return Some(format!("{own:?}: no placeholder but `{{prompt}}`"));
    }
    for w in own.split(|c: char| !c.is_alphanumeric() && !"#♯♭".contains(c)) {
        if chord_name(w) || MUSIC.contains(&w.to_lowercase().as_str()) {
            return Some(format!("{own:?} holds `{w}`"));
        }
    }
    None
}

/// Whether a word names a note or a chord: A, Bb, F#m, C#m7b5, Gsus4, Ebmaj7, Bdim.
fn chord_name(w: &str) -> bool {
    let Some(rest) = w.strip_prefix(|c: char| ('A'..='G').contains(&c)) else {
        return false;
    };
    let rest = rest.trim_start_matches(['#', 'b', '♯', '♭']);
    let rest = ["maj", "min", "dim", "aug", "sus", "m"]
        .iter()
        .find_map(|q| rest.strip_prefix(q))
        .unwrap_or(rest);
    rest.chars()
        .all(|c| c.is_ascii_digit() || "b#+".contains(c))
}

/// The graph of a run: `graph` with the chatbot's message set to `template` and the questions
/// set to `values`. Refused when the frozen nodes do not hash to `want`, or the template may not
/// be tried.
pub fn candidate(
    graph: &Value,
    want: &str,
    template: &str,
    values: &[Value],
) -> Result<Value, String> {
    let got = frozen(graph)?;
    if got != want {
        return Err(format!(
            "refused: the judge changed (its nodes hash to {got}, frozen at {want})"
        ));
    }
    if let Some(why) = refuse(template) {
        return Err(format!("refused: {why}"));
    }
    let mut g = graph.clone();
    for n in g["nodes"].as_array_mut().ok_or("the graph has no nodes")? {
        if n["name"] == "chatbot" {
            n["http"]["body"]["message"] = json!(template);
        } else if n["name"] == "questions" {
            n["values"] = json!(values);
        }
    }
    Ok(g)
}

/// One run of a template over a set of questions: the counts of its report.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub template: String,
    pub run: u32,
    pub questions: i64,
    pub t: i64,
    pub f: i64,
    pub u: i64,
    pub claims: i64,
    pub verified: i64,
}

/// Claims verified over claims checked; 0 when none was checked.
pub fn precision(verified: i64, claims: i64) -> f64 {
    verified as f64 / claims.max(1) as f64
}

/// A template's runs summed: T minus F, claims checked per question, U, and the precision of its
/// claims, which is reported beside the score and does not choose.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    pub template: String,
    pub runs: usize,
    pub score: i64,
    pub claims_per_question: f64,
    pub u: i64,
    pub precision: f64,
    /// Neither fewer claims per question nor more U than the baseline.
    pub eligible: bool,
}

/// The scores of the templates that ran, and the one chosen: the eligible template whose score
/// is highest and above the baseline's, the first of [`TEMPLATES`] on a tie, or the baseline.
pub fn choose(runs: &[Run]) -> (Vec<Score>, String) {
    let sum = |name: &str| {
        let mine: Vec<&Run> = runs.iter().filter(|r| r.template == name).collect();
        let total = |f: fn(&Run) -> i64| mine.iter().map(|r| f(r)).sum::<i64>();
        let questions = total(|r| r.questions).max(1) as f64;
        (
            mine.len(),
            total(|r| r.t - r.f),
            total(|r| r.claims) as f64 / questions,
            total(|r| r.u),
            precision(total(|r| r.verified), total(|r| r.claims)),
        )
    };
    let base = TEMPLATES[0].0;
    let (_, base_score, base_claims, base_u, _) = sum(base);
    let scores: Vec<Score> = TEMPLATES
        .iter()
        .filter(|(name, _)| runs.iter().any(|r| r.template == *name))
        .map(|(name, _)| {
            let (n, score, claims, u, precision) = sum(name);
            Score {
                template: name.to_string(),
                runs: n,
                score,
                claims_per_question: claims,
                u,
                precision,
                eligible: claims >= base_claims && u <= base_u,
            }
        })
        .collect();
    let best = scores
        .iter()
        .filter(|s| s.eligible && s.score > base_score)
        .fold(None::<&Score>, |b, s| match b {
            Some(b) if b.score >= s.score => Some(b),
            _ => Some(s),
        });
    (
        scores.clone(),
        best.map_or(base.to_string(), |s| s.template.clone()),
    )
}
