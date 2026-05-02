use ix_bracelet::{bracelet_prime_form, forte_number, grothendieck_delta, icv, z_related_pairs, PcSet};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Deserialize)]
pub struct AlgebraRequest {
    #[serde(alias = "Query")]
    pub query: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AlgebraResponse {
    #[serde(rename = "naturalLanguageAnswer")]
    pub natural_language_answer: String,
    #[serde(rename = "queryType")]
    pub query_type: String,
    pub facts: BTreeMap<String, String>,
    pub source: String,
    pub revision: String,
}

pub fn answer_query(query: &str, source: &str, revision: &str) -> Option<AlgebraResponse> {
    let sets = extract_pc_sets(query);
    if sets.is_empty() {
        return None;
    }

    let normalized = query.to_lowercase();

    if normalized.contains("z-related") || normalized.contains("z relation") || normalized.contains("z-pair") {
        return Some(answer_z_relation(&sets, source, revision));
    }

    if normalized.contains("prime form") {
        return Some(answer_prime_form(sets[0], source, revision));
    }

    if normalized.contains("interval-class vector") || normalized.contains("interval class vector") || normalized.contains(" icv") || normalized.ends_with("icv") {
        return Some(answer_interval_class_vector(sets[0], source, revision));
    }

    if normalized.contains("forte") {
        return Some(answer_forte(sets[0], source, revision));
    }

    if normalized.contains("set class") {
        return Some(answer_set_class_summary(sets[0], source, revision));
    }

    if (normalized.contains("distance") || normalized.contains("delta")) && sets.len() >= 2 {
        return Some(answer_distance(sets[0], sets[1], source, revision));
    }

    None
}

fn answer_prime_form(set: PcSet, source: &str, revision: &str) -> AlgebraResponse {
    let prime = bracelet_prime_form(set);
    let facts = BTreeMap::from([
        ("input".to_string(), format_set(set)),
        ("primeForm".to_string(), format_set(prime)),
    ]);

    AlgebraResponse {
        natural_language_answer: format!("The prime form of {} is {}.", format_set(set), format_set(prime)),
        query_type: "prime-form".to_string(),
        facts,
        source: source.to_string(),
        revision: revision.to_string(),
    }
}

fn answer_interval_class_vector(set: PcSet, source: &str, revision: &str) -> AlgebraResponse {
    let vector = icv(set);
    let formatted = format_icv(vector.data);
    let facts = BTreeMap::from([
        ("input".to_string(), format_set(set)),
        ("intervalClassVector".to_string(), formatted.clone()),
    ]);

    AlgebraResponse {
        natural_language_answer: format!("The interval-class vector for {} is {}.", format_set(set), formatted),
        query_type: "interval-class-vector".to_string(),
        facts,
        source: source.to_string(),
        revision: revision.to_string(),
    }
}

fn answer_forte(set: PcSet, source: &str, revision: &str) -> AlgebraResponse {
    let prime = bracelet_prime_form(set);
    let forte = forte_number(set).map(|value| value.to_string()).unwrap_or_else(|| "unavailable".to_string());
    let facts = BTreeMap::from([
        ("input".to_string(), format_set(set)),
        ("primeForm".to_string(), format_set(prime)),
        ("forte".to_string(), forte.clone()),
    ]);

    let natural_language_answer = if forte == "unavailable" {
        format!("I could compute the prime form for {}, but no Forte label was available.", format_set(set))
    } else {
        format!("The Forte label for {} is {}.", format_set(set), forte)
    };

    AlgebraResponse {
        natural_language_answer,
        query_type: "forte".to_string(),
        facts,
        source: source.to_string(),
        revision: revision.to_string(),
    }
}

fn answer_set_class_summary(set: PcSet, source: &str, revision: &str) -> AlgebraResponse {
    let prime = bracelet_prime_form(set);
    let vector = icv(set);
    let forte = forte_number(set).map(|value| value.to_string()).unwrap_or_else(|| "unavailable".to_string());
    let formatted_icv = format_icv(vector.data);
    let facts = BTreeMap::from([
        ("input".to_string(), format_set(set)),
        ("primeForm".to_string(), format_set(prime)),
        ("intervalClassVector".to_string(), formatted_icv.clone()),
        ("forte".to_string(), forte.clone()),
    ]);

    AlgebraResponse {
        natural_language_answer: format!(
            "Set-class summary for {}: prime form {}, ICV {}, Forte {}.",
            format_set(set),
            format_set(prime),
            formatted_icv,
            forte),
        query_type: "set-class-summary".to_string(),
        facts,
        source: source.to_string(),
        revision: revision.to_string(),
    }
}

fn answer_z_relation(sets: &[PcSet], source: &str, revision: &str) -> AlgebraResponse {
    if sets.len() >= 2 {
        let left = bracelet_prime_form(sets[0]);
        let right = bracelet_prime_form(sets[1]);
        let left_icv = icv(left);
        let right_icv = icv(right);
        let is_z_related = left != right && left_icv == right_icv;
        let left_forte = forte_number(left).map(|value| value.to_string()).unwrap_or_else(|| "unavailable".to_string());
        let right_forte = forte_number(right).map(|value| value.to_string()).unwrap_or_else(|| "unavailable".to_string());
        let facts = BTreeMap::from([
            ("left".to_string(), format_set(left)),
            ("right".to_string(), format_set(right)),
            ("leftIcv".to_string(), format_icv(left_icv.data)),
            ("rightIcv".to_string(), format_icv(right_icv.data)),
            ("leftForte".to_string(), left_forte.clone()),
            ("rightForte".to_string(), right_forte.clone()),
            ("zRelated".to_string(), is_z_related.to_string()),
        ]);

        let natural_language_answer = if is_z_related {
            format!(
                "{} ({}) and {} ({}) are Z-related: they share ICV {} but have different prime forms.",
                format_set(left),
                left_forte,
                format_set(right),
                right_forte,
                format_icv(left_icv.data))
        } else {
            format!("{} and {} are not Z-related.", format_set(left), format_set(right))
        };

        return AlgebraResponse {
            natural_language_answer,
            query_type: "z-relation".to_string(),
            facts,
            source: source.to_string(),
            revision: revision.to_string(),
        };
    }

    let set = bracelet_prime_form(sets[0]);
    let partner = z_related_pairs().iter().find_map(|(left, right)| {
        if *left == set {
            Some(*right)
        } else if *right == set {
            Some(*left)
        } else {
            None
        }
    });
    let vector = icv(set);
    let forte = forte_number(set).map(|value| value.to_string()).unwrap_or_else(|| "unavailable".to_string());
    let mut facts = BTreeMap::from([
        ("input".to_string(), format_set(set)),
        ("forte".to_string(), forte),
        ("intervalClassVector".to_string(), format_icv(vector.data)),
        ("isZRelated".to_string(), partner.is_some().to_string()),
    ]);

    let natural_language_answer = if let Some(partner) = partner {
        let partner_forte = forte_number(partner).map(|value| value.to_string()).unwrap_or_else(|| "unavailable".to_string());
        facts.insert("partner".to_string(), format_set(partner));
        facts.insert("partnerForte".to_string(), partner_forte.clone());
        format!(
            "{} is Z-related. One partner is {} ({}), and both share ICV {}.",
            format_set(set),
            format_set(partner),
            partner_forte,
            format_icv(vector.data))
    } else {
        format!("{} is not Z-related.", format_set(set))
    };

    AlgebraResponse {
        natural_language_answer,
        query_type: "z-relation".to_string(),
        facts,
        source: source.to_string(),
        revision: revision.to_string(),
    }
}

fn answer_distance(left: PcSet, right: PcSet, source: &str, revision: &str) -> AlgebraResponse {
    let delta = grothendieck_delta(left, right);
    let left_icv = icv(left);
    let right_icv = icv(right);
    let facts = BTreeMap::from([
        ("left".to_string(), format_set(left)),
        ("right".to_string(), format_set(right)),
        ("leftIcv".to_string(), format_icv(left_icv.data)),
        ("rightIcv".to_string(), format_icv(right_icv.data)),
        ("delta".to_string(), format_delta(delta.data)),
        ("l1Distance".to_string(), delta.l1_norm().to_string()),
    ]);

    AlgebraResponse {
        natural_language_answer: format!(
            "The Grothendieck delta between {} and {} is {}, so the L1 harmonic distance is {}.",
            format_set(left),
            format_set(right),
            format_delta(delta.data),
            delta.l1_norm()),
        query_type: "distance".to_string(),
        facts,
        source: source.to_string(),
        revision: revision.to_string(),
    }
}

fn extract_pc_sets(query: &str) -> Vec<PcSet> {
    let mut results = Vec::new();

    for capture in bracketed_set_regex().find_iter(query) {
        let candidate = token_regex()
            .find_iter(capture.as_str())
            .map(|m| normalize_token(m.as_str()))
            .collect::<String>();
        if let Some(set) = parse_pc_set(&candidate) {
            results.push(set);
        }
    }

    if !results.is_empty() {
        return results;
    }

    for capture in compact_set_regex().find_iter(query) {
        if let Some(set) = parse_pc_set(capture.as_str()) {
            results.push(set);
        }
    }

    results
}

fn parse_pc_set(candidate: &str) -> Option<PcSet> {
    let trimmed = candidate.trim();
    if trimmed.len() < 2 {
        return None;
    }

    let mut pcs = Vec::new();
    let chars: Vec<char> = trimmed.chars().collect();
    let mut index = 0;
    while index < chars.len() {
        let current = chars[index].to_ascii_uppercase();
        if current == '1' && index + 1 < chars.len() {
            match chars[index + 1] {
                '0' => {
                    pcs.push(10);
                    index += 2;
                    continue;
                }
                '1' => {
                    pcs.push(11);
                    index += 2;
                    continue;
                }
                _ => {}
            }
        }

        let value = match current {
            '0'..='9' => current.to_digit(10)? as u8,
            'T' | 'A' => 10,
            'E' | 'B' => 11,
            _ => return None,
        };
        pcs.push(value);
        index += 1;
    }

    if pcs.len() < 2 {
        return None;
    }

    Some(PcSet::from_pcs(pcs))
}

fn normalize_token(token: &str) -> String {
    match token.trim().to_ascii_uppercase().as_str() {
        "10" => "T".to_string(),
        "11" => "E".to_string(),
        value => value.to_string(),
    }
}

fn format_set(set: PcSet) -> String {
    let pcs = set.iter_pcs().map(|pc| pc.to_string()).collect::<Vec<_>>().join(",");
    format!("[{pcs}]")
}

fn format_icv(data: [u32; 6]) -> String {
    format!("[{}, {}, {}, {}, {}, {}]", data[0], data[1], data[2], data[3], data[4], data[5])
}

fn format_delta(data: [i32; 6]) -> String {
    format!("[{}, {}, {}, {}, {}, {}]", data[0], data[1], data[2], data[3], data[4], data[5])
}

fn bracketed_set_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"[\[\{][^\]\}]+[\]\}]").unwrap())
}

fn compact_set_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\b(?:10|11|[0-9]|[TEAB]){2,}\b").unwrap())
}

fn token_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"10|11|[0-9]|[TEAB]").unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_prime_form_query() {
        let response = answer_query("What is the prime form of [0,1,4,6]?", "ix", "7b02a56").expect("answer");
        assert_eq!(response.query_type, "prime-form");
        assert_eq!(response.facts.get("primeForm").map(String::as_str), Some("[0,1,4,6]"));
        assert_eq!(response.source, "ix");
        assert_eq!(response.revision, "7b02a56");
    }

    #[test]
    fn answers_z_relation_query() {
        let response = answer_query("Are 0146 and 0137 z-related?", "ix", "7b02a56").expect("answer");
        assert_eq!(response.query_type, "z-relation");
        assert_eq!(response.facts.get("zRelated").map(String::as_str), Some("true"));
        assert!(response.natural_language_answer.contains("share ICV"));
    }

    #[test]
    fn answers_distance_query() {
        let response = answer_query(
            "What is the harmonic distance between [0,4,8] and [0,3,6]?",
            "ix",
            "7b02a56").expect("answer");
        assert_eq!(response.query_type, "distance");
        assert_eq!(response.facts.get("l1Distance").map(String::as_str), Some("6"));
    }
}
