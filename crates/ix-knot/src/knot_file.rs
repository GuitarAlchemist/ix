//! The `.knot` language: a knot written down so that IX can check it.
//!
//! A `.knot` file names a knot, gives it — drawn through control points, or
//! spelled by its Gauss code — and states what it must be. IX reads the file
//! and checks every `expect` line against what it computes: a file is right
//! when it parses and all its expectations hold. Whoever writes the file, a
//! person or a model drafting under [`GRAMMAR`], proposes; IX decides.
//!
//! ```
//! use ix_knot::knot_file::KnotFile;
//!
//! let text = "
//! ## The overhand knot, spelled by its crossings.
//! knot overhand
//! fr \"Nœud simple\"
//! gauss U1 O2 U3 O1 U2 O3
//! closure 3_1
//! expect crossings 3
//! expect jones 3_1
//! expect slips untied 3
//! ";
//! let file: KnotFile = text.parse().unwrap();
//! let checked = file.check().unwrap();
//! assert!(checked.holds());
//! assert_eq!(checked.diagram.jones().to_string(), "t + t^3 - t^4");
//! ```

use crate::catalog::{closure_jones, MAX_SUMMANDS};
use crate::diagram::{Rope, RopeDiagram};
use crate::gauss::{draw, GaussCode};
use crate::jones::Jones;
use crate::mechanics::Mechanics;
use crate::mistakes::Outcome;
use std::str::FromStr;

/// The most bytes a `.knot` file may have.
pub const MAX_KNOT_FILE: usize = 64 * 1024;

/// The grammar of a `.knot` file, in EBNF. One statement per line; blank
/// lines and comments anywhere. A comment starts at a `#` that begins the line
/// or follows a space, so `3_1#m3_1` is a connected sum, not a comment.
pub const GRAMMAR: &str = r##"file        = { line } ;
line        = [ statement ] , [ comment ] , newline ;
comment     = "#" , { any character } ;                (* at the line's start or after a space *)
statement   = "knot" , name                                (* first, once *)
            | "fr" , text | "en" , text
            | "family" , word
            | "abok" , integer                            (* Ashley's number *)
            | "rope" , ( "open" | "closed" )              (* starts a rope *)
            | number , number , [ number ]                (* a control point of the last rope: x y, or x y z *)
            | "over" , ( "height" | "alternating" | letters )
            | "gauss" , code                              (* instead of ropes: the rest of the line *)
            | "closure" , knot                            (* with gauss: the knot its drawing must close into *)
            | "radius" , number
            | "pull" , end , { end }                     (* the end each rope is pulled from, in rope order *)
            | "expect" , expectation ;
expectation = "crossings" , integer
            | "components" , integer
            | "writhe" , integer
            | "jones" , ( knot | text )                   (* the closure's Jones polynomial: that knot's, or this text *)
            | "clearance" , ">=" , number                 (* in rope diameters, at the file's radius *)
            | "slips" , ( "same" | "untied" | "apart" | "other" ) , integer
            | "twist" , number                          (* Patil et al.'s tau, each rope oriented toward its pull, to 0.01 *)
            | "circulation" , number ;                  (* their Gamma, to 0.01 *)
end         = "start" | "end" ;                         (* "end" for every rope without "pull" *)
name        = letter , { letter | digit | "-" } ;
word        = letter , { letter | "-" } ;
text        = '"' , { any character but '"' } , '"' ;
letters     = { "O" | "U" } ;
code        = { ( "O" | "U" ) , integer | "(" | ")" | "|" } ;
knot        = rolfsen , { "#" , rolfsen } ;             (* a connected sum: the product of the polynomials *)
rolfsen     = [ "m" ] , digit , "_" , digit , [ digit ] ;  (* 0_1 to 7_7, and 8_20 *)
"##;

/// Why a file was refused, and on which line (1-based; 0 for the file as a
/// whole).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("line {line}: {reason}")]
pub struct KnotFileError {
    pub line: usize,
    pub reason: String,
}

fn fail<T>(line: usize, reason: impl Into<String>) -> Result<T, KnotFileError> {
    Err(KnotFileError {
        line,
        reason: reason.into(),
    })
}

/// How the knot is given.
#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    /// Ropes through control points, and which passage is in front.
    Drawn { ropes: Vec<Rope>, over: String },
    /// A Gauss code, and the knot its drawing must close into.
    Gauss {
        code: GaussCode,
        closure: Option<String>,
    },
}

/// What the closure must be.
#[derive(Debug, Clone, PartialEq)]
pub enum Expect {
    Crossings(usize),
    Components(usize),
    Writhe(i64),
    /// The Jones polynomial of the knot with this Rolfsen name, or of a sum
    /// of them written with `#`.
    JonesOf(String),
    /// The Jones polynomial, written in t as IX writes it.
    JonesText(String),
    Clearance(f64),
    Slips(Outcome, usize),
    /// Patil et al.'s twist fluctuation τ, to 0.01.
    Twist(f64),
    /// Their circulation Γ, to 0.01.
    Circulation(f64),
}

/// One `expect` line.
#[derive(Debug, Clone, PartialEq)]
pub struct Expectation {
    pub line: usize,
    /// The statement as written, comment removed.
    pub text: String,
    pub expect: Expect,
}

/// A parsed `.knot` file.
#[derive(Debug, Clone, PartialEq)]
pub struct KnotFile {
    pub name: String,
    pub fr: Option<String>,
    pub en: Option<String>,
    pub family: Option<String>,
    pub abok: Option<u16>,
    pub source: Source,
    /// Line of the `rope` or `gauss` statement that gives the knot.
    pub source_line: usize,
    pub radius: Option<f64>,
    /// The `pull` line and, per rope, whether it is pulled from its start.
    pub pull: Option<(usize, Vec<bool>)>,
    pub expectations: Vec<Expectation>,
}

/// An expectation checked: what IX found, and whether it is what was expected.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub line: usize,
    pub text: String,
    pub got: String,
    pub holds: bool,
}

/// A file checked: the diagram it gives, the radius it is drawn at, and a
/// verdict per expectation.
#[derive(Debug, Clone)]
pub struct Checked {
    pub diagram: RopeDiagram,
    pub radius: f64,
    pub verdicts: Vec<Verdict>,
}

impl Checked {
    /// Every expectation holds.
    pub fn holds(&self) -> bool {
        self.verdicts.iter().all(|v| v.holds)
    }
}

impl FromStr for KnotFile {
    type Err = KnotFileError;

    fn from_str(text: &str) -> Result<Self, KnotFileError> {
        if text.len() > MAX_KNOT_FILE {
            return fail(0, format!("a .knot file has at most {MAX_KNOT_FILE} bytes"));
        }
        let mut name = None;
        let (mut fr, mut en, mut family, mut abok) = (None, None, None, None);
        let mut ropes: Vec<Rope> = Vec::new();
        let (mut over, mut gauss, mut closure, mut radius) = (None, None, None, None);
        let mut pull = None;
        let mut source_line = 0;
        let mut expectations = Vec::new();
        for (i, raw) in text.lines().enumerate() {
            let line = i + 1;
            let statement = strip_comment(raw).trim();
            let Some((head, rest)) = split_head(statement) else {
                continue;
            };
            if name.is_none() && head != "knot" {
                return fail(line, "a .knot file starts with `knot <name>`");
            }
            let once = |seen: bool, what: &str| {
                if seen {
                    fail(line, format!("`{what}` is given twice"))
                } else {
                    Ok(())
                }
            };
            match head {
                "knot" => {
                    once(name.is_some(), "knot")?;
                    name = Some(read_name(line, rest)?);
                }
                "fr" => {
                    once(fr.is_some(), "fr")?;
                    fr = Some(read_text(line, rest)?);
                }
                "en" => {
                    once(en.is_some(), "en")?;
                    en = Some(read_text(line, rest)?);
                }
                "family" => {
                    once(family.is_some(), "family")?;
                    family = Some(read_name(line, rest)?);
                }
                "abok" => {
                    once(abok.is_some(), "abok")?;
                    abok = Some(read_number::<u16>(line, rest, "Ashley's number")?);
                }
                "rope" => {
                    if gauss.is_some() {
                        return fail(line, "give the knot by `rope` or by `gauss`, not both");
                    }
                    let closed = match rest {
                        "open" => false,
                        "closed" => true,
                        _ => return fail(line, "`rope open` or `rope closed`"),
                    };
                    if ropes.is_empty() {
                        source_line = line;
                    }
                    ropes.push(Rope {
                        points: Vec::new(),
                        closed,
                    });
                }
                "over" => {
                    once(over.is_some(), "over")?;
                    over = Some(rest.to_string());
                }
                "gauss" => {
                    once(gauss.is_some(), "gauss")?;
                    if !ropes.is_empty() {
                        return fail(line, "give the knot by `rope` or by `gauss`, not both");
                    }
                    let code: GaussCode = rest.parse().map_err(|e| KnotFileError {
                        line,
                        reason: format!("{e}"),
                    })?;
                    source_line = line;
                    gauss = Some(code);
                }
                "closure" => {
                    once(closure.is_some(), "closure")?;
                    closure = Some(read_rolfsen(line, rest)?);
                }
                "radius" => {
                    once(radius.is_some(), "radius")?;
                    let r = read_number::<f64>(line, rest, "a radius")?;
                    if !(r.is_finite() && r > 0.0) {
                        return fail(line, "the radius is a positive number");
                    }
                    radius = Some(r);
                }
                "pull" => {
                    once(pull.is_some(), "pull")?;
                    let ends: Result<Vec<bool>, _> = rest
                        .split_whitespace()
                        .map(|w| match w {
                            "start" => Ok(true),
                            "end" => Ok(false),
                            _ => fail(
                                line,
                                format!("`pull` takes start or end per rope, got {w:?}"),
                            ),
                        })
                        .collect();
                    let ends = ends?;
                    if ends.is_empty() {
                        return fail(line, "`pull` takes start or end per rope");
                    }
                    pull = Some((line, ends));
                }
                "expect" => expectations.push(Expectation {
                    line,
                    text: statement.to_string(),
                    expect: read_expect(line, rest)?,
                }),
                _ if head.starts_with(|c: char| c == '-' || c == '.' || c.is_ascii_digit()) => {
                    let Some(rope) = ropes.last_mut() else {
                        return fail(line, "a point belongs to a rope: `rope open` first");
                    };
                    rope.points.push(read_point(line, statement)?);
                }
                other => return fail(line, format!("unknown statement `{other}`")),
            }
        }
        let Some(name) = name else {
            return fail(0, "a .knot file starts with `knot <name>`");
        };
        let source = match gauss {
            Some(code) => {
                if over.is_some() {
                    return fail(0, "`over` goes with ropes; a Gauss code says it already");
                }
                Source::Gauss { code, closure }
            }
            None if ropes.is_empty() => {
                return fail(0, "give the knot: `rope open` and its points, or `gauss`")
            }
            None => {
                if closure.is_some() {
                    return fail(0, "`closure` goes with `gauss`; for ropes, `expect jones`");
                }
                let Some(over) = over else {
                    return fail(0, "ropes need `over`: height, alternating, or O/U letters");
                };
                Source::Drawn { ropes, over }
            }
        };
        Ok(Self {
            name,
            fr,
            en,
            family,
            abok,
            source,
            source_line,
            radius,
            pull,
            expectations,
        })
    }
}

impl KnotFile {
    /// Draw the knot and check every expectation.
    // @ai:invariant a failed expectation is answered with what IX found, not refused: one verdict per expect line, in file order [T:test conf:0.85 src:knot_file::tests::a_wrong_expectation_is_reported_with_what_ix_found]
    pub fn check(&self) -> Result<Checked, KnotFileError> {
        let at_source = |reason: String| KnotFileError {
            line: self.source_line,
            reason,
        };
        let (diagram, drawn_radius) = match &self.source {
            Source::Drawn { ropes, over } => (
                RopeDiagram::new(ropes, over).map_err(|e| at_source(e.to_string()))?,
                None,
            ),
            Source::Gauss { code, closure } => {
                let want = match closure {
                    None => None,
                    Some(k) => Some(known(self.source_line, k)?),
                };
                let d = draw(code, want.as_ref()).map_err(|e| at_source(e.to_string()))?;
                (d.diagram, Some(d.radius))
            }
        };
        let radius = self.radius.or(drawn_radius);
        let mut slips = None;
        let mut mechanics = None;
        let mut verdicts = Vec::new();
        for e in &self.expectations {
            let (got, holds) = match &e.expect {
                Expect::Crossings(n) => count(diagram.drawn_crossings(), *n),
                Expect::Components(n) => count(diagram.components(), *n),
                Expect::Writhe(w) => (diagram.writhe().to_string(), diagram.writhe() == *w),
                Expect::JonesOf(k) => {
                    let v = diagram.jones();
                    (v.to_string(), *v == known(e.line, k)?)
                }
                Expect::JonesText(t) => {
                    let v = diagram.jones().to_string();
                    let holds = squeeze(&v) == squeeze(t);
                    (v, holds)
                }
                Expect::Clearance(at_least) => {
                    let Some(r) = radius else {
                        return fail(e.line, "`expect clearance` needs `radius`");
                    };
                    let g = diagram.geometry(r).map_err(|err| KnotFileError {
                        line: e.line,
                        reason: err.to_string(),
                    })?;
                    match g.min_clearance {
                        Some(c) => (format!("{c:.3}"), c >= *at_least),
                        None => ("none".to_string(), true),
                    }
                }
                Expect::Slips(outcome, n) => {
                    if slips.is_none() {
                        slips = Some(diagram.mistakes().map_err(|err| KnotFileError {
                            line: e.line,
                            reason: err.to_string(),
                        })?);
                    }
                    let all = slips.as_deref().unwrap_or_default();
                    count(all.iter().filter(|m| m.outcome == *outcome).count(), *n)
                }
                Expect::Twist(want) | Expect::Circulation(want) => {
                    if mechanics.is_none() {
                        mechanics = Some(self.mechanics(&diagram)?);
                    }
                    let m = mechanics.unwrap_or_else(|| unreachable!());
                    let got = match e.expect {
                        Expect::Twist(_) => m.twist,
                        _ => m.circulation,
                    };
                    (format!("{got:.2}"), (got - want).abs() < 0.005)
                }
            };
            verdicts.push(Verdict {
                line: e.line,
                text: e.text.clone(),
                got,
                holds,
            });
        }
        Ok(Checked {
            diagram,
            radius: radius.unwrap_or(1.0),
            verdicts,
        })
    }
}

/// The line up to its comment: a `#` at its start or after whitespace.
fn strip_comment(raw: &str) -> &str {
    let mut after_space = true;
    for (i, c) in raw.char_indices() {
        if c == '#' && after_space {
            return &raw[..i];
        }
        after_space = c.is_whitespace();
    }
    raw
}

impl KnotFile {
    /// Patil et al.'s counts for the file's knot, each rope pulled as `pull`
    /// says, or from its last point.
    pub fn mechanics(&self, diagram: &RopeDiagram) -> Result<Mechanics, KnotFileError> {
        let (line, pulls) = match &self.pull {
            Some((line, pulls)) => (*line, pulls.clone()),
            None => (0, vec![false; diagram.components()]),
        };
        diagram.mechanics(&pulls).map_err(|e| KnotFileError {
            line,
            reason: e.to_string(),
        })
    }
}

fn count(got: usize, want: usize) -> (String, bool) {
    (got.to_string(), got == want)
}

fn squeeze(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// The Jones polynomial of a knot named in the table, or of a sum of them,
/// or the line's error.
fn known(line: usize, name: &str) -> Result<Jones, KnotFileError> {
    closure_jones(name).ok_or_else(|| KnotFileError {
        line,
        reason: format!("no knot {name:?} in the table IX knows: \"0_1\" to \"7_7\" and \"8_20\", `m` in front for the mirror, at most {MAX_SUMMANDS} joined by `#`"),
    })
}

/// The first word and the rest, or `None` for an empty statement.
fn split_head(statement: &str) -> Option<(&str, &str)> {
    if statement.is_empty() {
        return None;
    }
    Some(match statement.split_once(char::is_whitespace) {
        Some((head, rest)) => (head, rest.trim()),
        None => (statement, ""),
    })
}

fn read_name(line: usize, rest: &str) -> Result<String, KnotFileError> {
    let ok = rest.starts_with(|c: char| c.is_ascii_alphabetic())
        && rest
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if ok {
        Ok(rest.to_string())
    } else {
        fail(
            line,
            format!("a name is letters, digits and dashes, got {rest:?}"),
        )
    }
}

fn read_text(line: usize, rest: &str) -> Result<String, KnotFileError> {
    match rest
        .strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .filter(|inner| !inner.contains('"'))
    {
        Some(inner) => Ok(inner.to_string()),
        None => fail(line, format!("text goes in double quotes, got {rest:?}")),
    }
}

fn read_number<T: FromStr>(line: usize, rest: &str, what: &str) -> Result<T, KnotFileError> {
    rest.parse()
        .or_else(|_| fail(line, format!("expected {what}, got {rest:?}")))
}

fn read_rolfsen(line: usize, rest: &str) -> Result<String, KnotFileError> {
    known(line, rest)?;
    Ok(rest.to_string())
}

fn read_point(line: usize, statement: &str) -> Result<[f64; 3], KnotFileError> {
    let numbers: Result<Vec<f64>, _> = statement.split_whitespace().map(str::parse).collect();
    match numbers.as_deref() {
        Ok([x, y]) => Ok([*x, *y, 0.0]),
        Ok([x, y, z]) => Ok([*x, *y, *z]),
        _ => fail(
            line,
            format!("a point is `x y` or `x y z`, got {statement:?}"),
        ),
    }
}

fn read_expect(line: usize, rest: &str) -> Result<Expect, KnotFileError> {
    let Some((what, arg)) = split_head(rest) else {
        return fail(
            line,
            "`expect` what? crossings, components, writhe, jones, clearance, slips, twist or circulation",
        );
    };
    let integer = |arg: &str| read_number::<usize>(line, arg, "a whole number");
    Ok(match what {
        "crossings" => Expect::Crossings(integer(arg)?),
        "components" => Expect::Components(integer(arg)?),
        "writhe" => Expect::Writhe(read_number(line, arg, "a whole number")?),
        "jones" if arg.starts_with('"') => Expect::JonesText(read_text(line, arg)?),
        "jones" => Expect::JonesOf(read_rolfsen(line, arg)?),
        "clearance" => match arg.strip_prefix(">=") {
            Some(n) => Expect::Clearance(read_number(line, n.trim(), "a number")?),
            None => return fail(line, "`expect clearance >= <diameters>`"),
        },
        "twist" => Expect::Twist(read_number(line, arg, "a number")?),
        "circulation" => Expect::Circulation(read_number(line, arg, "a number")?),
        "slips" => {
            let (kind, n) = split_head(arg).unwrap_or(("", ""));
            let outcome = match kind {
                "same" => Outcome::Same,
                "untied" => Outcome::Untied,
                "apart" => Outcome::Apart,
                "other" => Outcome::Other,
                _ => return fail(line, "`expect slips same|untied|apart|other <count>`"),
            };
            Expect::Slips(outcome, integer(n)?)
        }
        other => return fail(line, format!("cannot expect `{other}`")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOWLINE: &str = include_str!("../knots/bowline.knot");
    const OVERHAND: &str = include_str!("../knots/overhand.knot");

    #[test]
    fn the_bowline_file_holds_every_expectation() {
        let file: KnotFile = BOWLINE.parse().unwrap();
        assert_eq!(file.name, "bowline");
        assert_eq!(file.fr.as_deref(), Some("Nœud de chaise"));
        let checked = file.check().unwrap();
        for v in &checked.verdicts {
            assert!(v.holds, "line {}: {} — IX found {}", v.line, v.text, v.got);
        }
        assert_eq!(checked.verdicts.len(), 6);
        assert_eq!(checked.radius, 0.16);
    }

    #[test]
    fn the_overhand_spelled_by_its_code_holds() {
        let checked = OVERHAND.parse::<KnotFile>().unwrap().check().unwrap();
        assert!(checked.holds());
        assert_eq!(checked.diagram.drawn_crossings(), 3);
    }

    #[test]
    fn a_wrong_expectation_is_reported_with_what_ix_found() {
        let text = BOWLINE.replace("expect crossings 7", "expect crossings 8");
        let checked = text.parse::<KnotFile>().unwrap().check().unwrap();
        assert!(!checked.holds());
        let wrong: Vec<_> = checked.verdicts.iter().filter(|v| !v.holds).collect();
        assert_eq!(wrong.len(), 1);
        assert_eq!(
            (wrong[0].text.as_str(), wrong[0].got.as_str()),
            ("expect crossings 8", "7")
        );
    }

    /// The reef and the thief: one drawing, pulled from one tail or the other.
    #[test]
    fn pull_says_which_ends_load_the_knot() {
        let reef: String = crate::testing::bights()
            .iter()
            .map(|r| {
                let points: Vec<String> = r
                    .points
                    .iter()
                    .map(|p| format!("{} {}", p[0], p[1]))
                    .collect();
                format!("rope open\n{}\n", points.join("\n"))
            })
            .collect();
        let file = |pull: &str| {
            format!("knot reef\n{reef}over OUOOUO UOUUOU\n{pull}\nexpect twist 1\nexpect circulation 4\n")
        };
        let reef = file("pull end end")
            .parse::<KnotFile>()
            .unwrap()
            .check()
            .unwrap();
        assert!(reef.holds());
        let thief = file("pull end start")
            .parse::<KnotFile>()
            .unwrap()
            .check()
            .unwrap();
        let got: Vec<&str> = thief.verdicts.iter().map(|v| v.got.as_str()).collect();
        assert_eq!(got, ["1.00", "1.00"]);
        // Without `pull`, each rope is pulled from its last point.
        assert!(file("")
            .parse::<KnotFile>()
            .unwrap()
            .check()
            .unwrap()
            .holds());
        let err = file("pull end")
            .parse::<KnotFile>()
            .unwrap()
            .check()
            .unwrap_err();
        assert!(err.reason.contains("one pull per rope"), "{err}");
        assert!(file("pull sideways").parse::<KnotFile>().is_err());
    }

    /// Two overhand knots in a row, spelled by their code: the closure named as
    /// a sum picks the reef's drawing or the granny's.
    #[test]
    fn a_sum_names_which_drawing_of_two_overhands_to_keep() {
        let file = |closure: &str| {
            format!(
                "knot two-overhands\ngauss U1 O2 U3 O1 U2 O3 U4 O5 U6 O4 U5 O6\nclosure {closure}\n\
                 expect crossings 6\nexpect jones {closure}\n"
            )
        };
        let reef = file("3_1#m3_1")
            .parse::<KnotFile>()
            .unwrap()
            .check()
            .unwrap();
        assert!(reef.holds());
        assert!(reef.diagram.jones().is_symmetric());
        let granny = file("3_1#3_1")
            .parse::<KnotFile>()
            .unwrap()
            .check()
            .unwrap();
        assert!(granny.holds());
        assert_ne!(granny.diagram.jones(), reef.diagram.jones());
    }

    #[test]
    fn refuses_with_the_line_and_a_reason() {
        let err = |text: &str| text.parse::<KnotFile>().unwrap_err();
        assert_eq!(err("rope open").line, 1);
        assert!(err("knot a\n0 0\n").reason.contains("rope open"));
        assert_eq!(err("knot a\nrope open\n0 0\ngauss U1 O1\n").line, 4);
        assert!(err("knot a\ngauss U1 O2 U3 O1 U2 O3\nover height\n")
            .reason
            .contains("`over`"));
        assert!(
            err("knot a\nrope open\n0 0\n1 1\nover height\nclosure 3_1\n")
                .reason
                .contains("closure")
        );
        assert_eq!(err("knot a\nexpect jones 9_42\n").line, 2);
        assert_eq!(err("knot a\nexpect jones 3_1#9_42\n").line, 2);
        // A comment starts at a # after a space, not inside a sum.
        assert_eq!(err("knot a\nexpect jones 9_42 # 3_1#m3_1\n").line, 2);
        assert_eq!(err("knot a # 3_1#9_42\n").line, 0);
        assert_eq!(err("knot a\nfr unquoted\n").line, 2);
        assert_eq!(
            err("knot a\nwhat is this\n").reason,
            "unknown statement `what`"
        );
        assert_eq!(err("knot a\n").line, 0);
    }

    #[test]
    fn the_grammar_names_every_statement_and_expectation() {
        for word in [
            "knot",
            "fr",
            "en",
            "family",
            "abok",
            "rope",
            "over",
            "gauss",
            "closure",
            "radius",
            "pull",
            "expect",
            "crossings",
            "components",
            "writhe",
            "jones",
            "clearance",
            "slips",
            "twist",
            "circulation",
        ] {
            assert!(GRAMMAR.contains(&format!("\"{word}\"")), "{word}");
        }
    }
}
