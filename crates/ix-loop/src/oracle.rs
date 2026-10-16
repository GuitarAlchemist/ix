//! Oracles: tasks whose answers the host checks, never a model.
//!
//! A task family has instances: `lipogram:<words>:<letter>` is one English sentence of exactly
//! that many words without that letter, and `acrostic:<WORD>:<letter>` is a sentence whose words'
//! first letters spell the word, again without the letter. `origami:<faces>:<ply>` is a
//! `.origami` file whose folded result passes every check of `ix-origami` with that many faces
//! and that ply; `ix-origami` replays and judges it. The host can draw an instance from a seed,
//! so a loop can be judged on instances its writer never saw.

use ix_origami::{Limits, Sequence};

/// One instance of a task family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Task {
    Lipogram { words: usize, letter: char },
    Acrostic { word: String, letter: char },
    Origami { faces: usize, ply: usize },
}

/// The work one origami answer may ask of the checks: those of `ix_origami_check`.
const ORIGAMI_LIMITS: Limits = Limits {
    faces: 1_000,
    face_vertices: 4_000,
    cells: 20_000,
    cell_pairs: 1_000_000,
    tacos: 100_000,
    // The oracle runs no swap census.
    census_steps: 0,
};

/// The origami targets the host draws, each with a known solution in the tests.
///
/// Each is rare under blind search. A random walk of 1 to 7 folds between the reference points
/// and lines met each one in at most 2.5e-4 of its valid files, about one in 30 s; it met the
/// easy targets these replace (2:2, 4:4, 8:8, …) within a second. A judge that grants a child a
/// few seconds therefore tells a loop that reasons about folding from one that samples.
const ORIGAMI_TARGETS: [(usize, usize); 9] = [
    (5, 2),
    (9, 5),
    (9, 9),
    (10, 5),
    (11, 5),
    (11, 7),
    (11, 11),
    (12, 4),
    (12, 7),
];

fn check_origami(text: &str, faces: usize, ply: usize) -> Check {
    let fail = |why: String| Check { ok: false, why };
    let sequence: Sequence = match text.parse() {
        Ok(s) => s,
        Err(e) => return fail(format!("not a .origami file: {e}")),
    };
    // The file's own `expect` lines are its claims; the oracle reads only the folded result.
    let checked = match sequence.check_within(&ORIGAMI_LIMITS) {
        Ok(c) => c,
        Err(e) => return fail(format!("refused: {e}")),
    };
    let report = match &checked.report {
        Ok(r) => r,
        Err(problems) => return fail(format!("the replay is unchecked: {}", problems.join("; "))),
    };
    let got_faces = checked.fold.faces_vertices.len();
    let got_ply = report.layers.cells.max_layers;
    let mut problems = Vec::new();
    if !report.ok {
        problems.push(format!("the checks reject it: {:?}", report.rejected_by));
    }
    if got_faces != faces {
        problems.push(format!("it has {got_faces} faces, not {faces}"));
    }
    if got_ply != ply {
        problems.push(format!("its ply is {got_ply}, not {ply}"));
    }
    if problems.is_empty() {
        Check {
            ok: true,
            why: "ok".into(),
        }
    } else {
        fail(problems.join("; "))
    }
}

/// The oracle's verdict on one answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub ok: bool,
    /// `ok`, or what is wrong.
    pub why: String,
}

/// The families [`Task::draw`] knows.
pub const FAMILIES: [&str; 3] = ["lipogram", "acrostic", "origami"];

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
            ["origami", f, p] => {
                let faces = f.parse::<usize>().ok().filter(|f| (1..=64).contains(f))?;
                let ply = p
                    .parse::<usize>()
                    .ok()
                    .filter(|p| (1..=faces).contains(p))?;
                Some(Task::Origami { faces, ply })
            }
            _ => None,
        }
    }

    /// The instance's name, which [`Task::parse`] reads back.
    pub fn spec(&self) -> String {
        match self {
            Task::Lipogram { words, letter } => format!("lipogram:{words}:{letter}"),
            Task::Acrostic { word, letter } => format!("acrostic:{word}:{letter}"),
            Task::Origami { faces, ply } => format!("origami:{faces}:{ply}"),
        }
    }

    pub fn family(&self) -> &'static str {
        match self {
            Task::Lipogram { .. } => "lipogram",
            Task::Acrostic { .. } => "acrostic",
            Task::Origami { .. } => "origami",
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
            Task::Origami { faces, ply } => format!(
                "Write a .origami file: a folding sequence of simple folds on the unit square, in \
                 the grammar below. IX replays it and checks the folded result. The result must \
                 pass every check and have exactly {faces} faces, and its ply (the most faces \
                 stacked at one point) must be {ply}. The answer is the file's text alone, \
                 starting with `origami <name>`: no code fence, no prose. A step that cannot be \
                 folded is refused with its line number.\n\nGrammar (EBNF):\n{}\nExample, two \
                 faces and ply 2:\norigami half\nvalley B to A\n",
                ix_origami::GRAMMAR
            ),
        }
    }

    pub fn check(&self, answer: &str) -> Check {
        match self {
            Task::Lipogram { words, letter } => check_lipogram(answer, *words, *letter),
            Task::Acrostic { word, letter } => check_acrostic(answer, word, *letter),
            Task::Origami { faces, ply } => check_origami(answer, *faces, *ply),
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
            "origami" => {
                let (faces, ply) = ORIGAMI_TARGETS[(r % ORIGAMI_TARGETS.len() as u64) as usize];
                Some(Task::Origami { faces, ply })
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

    /// A solution for every target the host draws, so no hidden instance is impossible.
    #[test]
    fn every_origami_target_has_a_solution() {
        // 5:2 is the blintz; the others are the first file a random walk found for each.
        let witnesses = [
            ((5, 2), "valley A to O\nvalley B to O\nvalley C to O\nvalley D to O"),
            (
                (9, 5),
                "mountain A to mBC\nvalley AB to DA\nvalley A to mDA\nmountain mBC to D",
            ),
            (
                (9, 9),
                "valley mAB to B\nmountain BC to AC\nvalley D to mBC\nvalley A to C",
            ),
            (
                (10, 5),
                "mountain mAB to C\nmountain D to mBC\nmountain BC to AB\nvalley CD to AB\nvalley A to mDA",
            ),
            (
                (11, 5),
                "valley BC to AC\nmountain C to O\nmountain B to mAB\nmountain mDA to mAB\nmountain mAB to A",
            ),
            (
                (11, 7),
                "valley mBC to B\nmountain BC to AC\nmountain mBC to A\nvalley O to C\nvalley B to mCD",
            ),
            (
                (11, 11),
                "mountain mBC to D\nvalley A to mBC\nmountain mAB to C\nvalley mDA to O",
            ),
            (
                (12, 4),
                "mountain B to O\nvalley O to mAB\nmountain D to B\nvalley D to C\nvalley A to D\nvalley mBC to A",
            ),
            (
                (12, 7),
                "valley B to mAB\nvalley D to mBC\nvalley BD to DA\nvalley mBC to mCD",
            ),
        ];
        assert_eq!(witnesses.len(), ORIGAMI_TARGETS.len());
        for ((faces, ply), steps) in witnesses {
            assert!(ORIGAMI_TARGETS.contains(&(faces, ply)));
            let c = Task::Origami { faces, ply }.check(&format!("origami w\n{steps}\n"));
            assert!(c.ok, "{faces} faces, ply {ply}: {}", c.why);
        }
    }

    #[test]
    fn the_origami_oracle_reads_the_result_not_the_file_s_claims() {
        let target = Task::parse("origami:4:4").unwrap();
        // It claims the target, and folds only once.
        let claims = target.check("origami liar\nvalley B to A\nexpect faces 4\nexpect ply 4\n");
        assert!(!claims.ok);
        assert!(claims.why.contains("2 faces, not 4"), "{}", claims.why);
        // Its claims are wrong, and its result is the target.
        let modest = target.check("origami modest\nvalley B to A\nvalley D to A\nexpect faces 1\n");
        assert!(modest.ok, "{}", modest.why);
        // A refusal names its line, so a loop can repair the file.
        let refused = target.check("origami x\nvalley B to A\nvalley B to A\n");
        assert!(refused.why.contains("line 3"), "{}", refused.why);
        let fenced = target.check("```\norigami x\nvalley B to A\n```");
        assert!(fenced.why.contains("line 1"), "{}", fenced.why);
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
