//! Oracles: tasks whose answers the host checks, never a model.
//!
//! A task family has instances: `lipogram:<words>:<letter>` is one English sentence of exactly
//! that many words without that letter, and `acrostic:<WORD>:<letter>` is a sentence whose words'
//! first letters spell the word, again without the letter. The host can draw an instance from a
//! seed, so a loop can be judged on instances its writer never saw.

/// One instance of a task family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Task {
    Lipogram { words: usize, letter: char },
    Acrostic { word: String, letter: char },
}

/// The oracle's verdict on one answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub ok: bool,
    /// `ok`, or what is wrong.
    pub why: String,
}

/// The families [`Task::draw`] knows.
pub const FAMILIES: [&str; 2] = ["lipogram", "acrostic"];

/// The letters drawn instances ban: common ones, so the task is never trivial.
const LETTERS: &[u8] = b"eaiounrst";
const ACROSTIC_WORDS: [&str; 10] = [
    "LOOP", "MIND", "CRAFT", "GLYPH", "SOLID", "QUICK", "FJORD", "NYMPH", "BRAVO", "PLUMB",
];

/// splitmix64: a fixed, portable way from a seed to a well-mixed number.
pub(crate) fn mix(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn words_of(s: &str) -> Vec<&str> {
    s.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .collect()
}

fn check_lipogram(s: &str, words_wanted: usize, letter: char) -> Check {
    let banned = [letter.to_ascii_lowercase(), letter.to_ascii_uppercase()];
    let words = words_of(s);
    let mut problems = Vec::new();
    if words.len() != words_wanted {
        problems.push(format!("it has {} words, not {words_wanted}", words.len()));
    }
    let bad: Vec<&str> = words
        .iter()
        .copied()
        .filter(|w| w.contains(banned))
        .collect();
    if !bad.is_empty() {
        problems.push(format!(
            "these words contain the letter {letter}: {}",
            bad.join(", ")
        ));
    } else if s.contains(banned) {
        problems.push(format!("the text contains the letter {letter}"));
    }
    if problems.is_empty() {
        Check {
            ok: true,
            why: "ok".into(),
        }
    } else {
        Check {
            ok: false,
            why: problems.join("; "),
        }
    }
}

fn check_acrostic(s: &str, word: &str, letter: char) -> Check {
    let mut c = check_lipogram(s, word.chars().count(), letter);
    let initials: String = words_of(s)
        .iter()
        .filter_map(|w| w.chars().next())
        .collect::<String>()
        .to_uppercase();
    if initials != word.to_uppercase() {
        let note = format!(
            "the first letters spell {initials}, not {}",
            word.to_uppercase()
        );
        c.why = if c.ok {
            note
        } else {
            format!("{}; {note}", c.why)
        };
        c.ok = false;
    }
    c
}

impl Task {
    /// Reads `lipogram:<words>:<letter>` (1 to 40 words) or `acrostic:<WORD>:<letter>` (a word of
    /// 2 to 20 letters, without the banned one).
    pub fn parse(spec: &str) -> Option<Task> {
        let parts: Vec<&str> = spec.split(':').collect();
        let letter_of = |l: &str| {
            let mut cs = l.chars();
            match (cs.next(), cs.next()) {
                (Some(c), None) if c.is_ascii_alphabetic() => Some(c.to_ascii_lowercase()),
                _ => None,
            }
        };
        match parts.as_slice() {
            ["lipogram", n, l] => {
                let words = n.parse::<usize>().ok().filter(|n| (1..=40).contains(n))?;
                Some(Task::Lipogram {
                    words,
                    letter: letter_of(l)?,
                })
            }
            ["acrostic", word, l] => {
                let letter = letter_of(l)?;
                let ok = (2..=20).contains(&word.len())
                    && word.chars().all(|c| c.is_ascii_alphabetic())
                    && !word.to_ascii_lowercase().contains(letter);
                ok.then(|| Task::Acrostic {
                    word: word.to_ascii_uppercase(),
                    letter,
                })
            }
            _ => None,
        }
    }

    /// The instance's name, which [`Task::parse`] reads back.
    pub fn spec(&self) -> String {
        match self {
            Task::Lipogram { words, letter } => format!("lipogram:{words}:{letter}"),
            Task::Acrostic { word, letter } => format!("acrostic:{word}:{letter}"),
        }
    }

    pub fn family(&self) -> &'static str {
        match self {
            Task::Lipogram { .. } => "lipogram",
            Task::Acrostic { .. } => "acrostic",
        }
    }

    /// The instance, in words a writer can be shown.
    pub fn describe(&self) -> String {
        match self {
            Task::Lipogram { words, letter } => format!(
                "Write one English sentence of exactly {words} words that never uses the letter \
                 {letter} (neither lower nor upper case). Words are separated by spaces; \
                 punctuation is not a word."
            ),
            Task::Acrostic { word, letter } => format!(
                "Write one English sentence of exactly {} words whose first letters, in order, \
                 spell {word}, and that never uses the letter {letter} (neither lower nor upper \
                 case). Words are separated by spaces; punctuation is not a word.",
                word.chars().count()
            ),
        }
    }

    pub fn check(&self, answer: &str) -> Check {
        match self {
            Task::Lipogram { words, letter } => check_lipogram(answer, *words, *letter),
            Task::Acrostic { word, letter } => check_acrostic(answer, word, *letter),
        }
    }

    /// An instance of `family` drawn from `seed`; `None` for an unknown family.
    pub fn draw(family: &str, seed: u64) -> Option<Task> {
        let r = mix(seed);
        let pick = |i: u64| LETTERS[(i % LETTERS.len() as u64) as usize] as char;
        match family {
            "lipogram" => Some(Task::Lipogram {
                words: 6 + (r % 10) as usize,
                letter: pick(r >> 8),
            }),
            "acrostic" => {
                let word = ACROSTIC_WORDS[(r % ACROSTIC_WORDS.len() as u64) as usize];
                let lower = word.to_ascii_lowercase();
                let letter = (0..LETTERS.len() as u64)
                    .map(|k| pick((r >> 8) + k))
                    .find(|c| !lower.contains(*c))?;
                Some(Task::Acrostic {
                    word: word.to_string(),
                    letter,
                })
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_read_back() {
        for spec in ["lipogram:12:e", "acrostic:LOOP:e"] {
            assert_eq!(Task::parse(spec).unwrap().spec(), spec);
        }
        for bad in [
            "lipogram:0:e",
            "lipogram:12:ee",
            "acrostic:LOOP:o",
            "sonnet:14:e",
        ] {
            assert_eq!(Task::parse(bad), None, "{bad}");
        }
    }

    #[test]
    fn the_oracles_judge_the_text_not_a_claim() {
        let t = Task::parse("lipogram:4:e").unwrap();
        assert!(t.check("A big cat sat.").ok);
        assert!(!t.check("The big cat sat.").ok);
        assert!(!t.check("A big cat").ok);
        let a = Task::parse("acrostic:CAT:e").unwrap();
        assert!(a.check("Cats always talk").ok);
        assert!(!a.check("Dogs always talk").ok);
    }

    #[test]
    fn every_drawn_instance_is_valid() {
        for family in FAMILIES {
            for seed in 0..200 {
                let t = Task::draw(family, seed).unwrap();
                assert_eq!(Task::parse(&t.spec()), Some(t.clone()));
                assert_eq!(t.family(), family);
            }
        }
        assert_eq!(Task::draw("sonnet", 0), None);
    }
}
