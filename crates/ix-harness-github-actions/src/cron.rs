//! Minimal POSIX cron matcher for GitHub Actions `on.schedule` entries
//! (UTC, five fields: minute hour day-of-month month day-of-week).
//!
//! Supports `*`, numbers, `a-b`, `*/s`, `a-b/s`, comma lists and
//! three-letter month / weekday names. When both day-of-month and
//! day-of-week are restricted, a day matches if EITHER does (cron's
//! OR rule, which GitHub follows).

/// A parsed cron expression: one membership table per field.
#[derive(Debug, Clone)]
pub struct Cron {
    minutes: [bool; 60],
    hours: [bool; 24],
    dom: [bool; 32],
    months: [bool; 13],
    dow: [bool; 7],
    dom_any: bool,
    dow_any: bool,
}

impl Cron {
    /// `None` for anything outside the supported grammar.
    pub fn parse(expr: &str) -> Option<Cron> {
        let fields: Vec<&str> = expr.split_whitespace().collect();
        if fields.len() != 5 {
            return None;
        }
        let mut minutes = [false; 60];
        let mut hours = [false; 24];
        let mut dom = [false; 32];
        let mut months = [false; 13];
        let mut dow_raw = [false; 8];
        fill(fields[0], 0, 59, &[], &mut minutes)?;
        fill(fields[1], 0, 23, &[], &mut hours)?;
        fill(fields[2], 1, 31, &[], &mut dom)?;
        fill(fields[3], 1, 12, &MONTHS, &mut months)?;
        fill(fields[4], 0, 7, &DAYS, &mut dow_raw)?;
        let mut dow = [false; 7];
        dow.copy_from_slice(&dow_raw[..7]);
        dow[0] |= dow_raw[7];
        Some(Cron {
            minutes,
            hours,
            dom,
            months,
            dow,
            dom_any: fields[2] == "*",
            dow_any: fields[4] == "*",
        })
    }

    /// Does the cron fire at this Unix minute (seconds are ignored)?
    pub fn matches(&self, epoch_secs: i64) -> bool {
        let days = epoch_secs.div_euclid(86_400);
        let secs = epoch_secs.rem_euclid(86_400);
        let (_, month, day) = civil_from_days(days);
        // 1970-01-01 was a Thursday (4).
        let weekday = (days + 4).rem_euclid(7) as usize;
        let day_ok = match (self.dom_any, self.dow_any) {
            (true, true) => true,
            (false, true) => self.dom[day as usize],
            (true, false) => self.dow[weekday],
            (false, false) => self.dom[day as usize] || self.dow[weekday],
        };
        self.minutes[(secs / 60 % 60) as usize]
            && self.hours[(secs / 3600) as usize]
            && self.months[month as usize]
            && day_ok
    }

    /// Fire times in the half-open interval `(after, until]`, counted
    /// minute by minute. Capped at `cap` so a stale window stays cheap.
    pub fn fires_between(&self, after: i64, until: i64, cap: usize) -> usize {
        let mut t = (after.div_euclid(60) + 1) * 60;
        let mut n = 0;
        while t <= until && n < cap {
            if self.matches(t) {
                n += 1;
            }
            t += 60;
        }
        n
    }
}

const MONTHS: [&str; 12] = [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
];
const DAYS: [&str; 7] = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"];

/// `names[i]` stands for `lo + i`.
fn fill(field: &str, lo: u32, hi: u32, names: &[&str], table: &mut [bool]) -> Option<()> {
    let value = |s: &str| -> Option<u32> {
        let lower = s.to_ascii_lowercase();
        match names.iter().position(|n| *n == lower) {
            Some(i) => Some(lo + i as u32),
            None => s.parse().ok(),
        }
    };
    for part in field.split(',') {
        let (range, step) = match part.split_once('/') {
            Some((r, s)) => (r, s.parse::<u32>().ok().filter(|s| *s > 0)?),
            None => (part, 1),
        };
        let (start, end) = if range == "*" {
            (lo, hi)
        } else if let Some((a, b)) = range.split_once('-') {
            (value(a)?, value(b)?)
        } else {
            let v = value(range)?;
            (v, if part.contains('/') { hi } else { v })
        };
        if start < lo || end > hi || start > end {
            return None;
        }
        for v in (start..=end).step_by(step as usize) {
            table[v as usize] = true;
        }
    }
    Some(())
}

/// Hinnant's civil-from-days: days since 1970-01-01 → (year, month, day).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

/// Cron expressions from a workflow file's `- cron: '...'` lines.
/// A line scan, not a YAML parser: GitHub only reads `cron` under
/// `on.schedule`, and no workflow in the ecosystem uses the key elsewhere.
pub fn crons_in_workflow(yaml: &str) -> Vec<String> {
    yaml.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.trim_start_matches("- ").strip_prefix("cron:"))
        .map(|v| {
            let v = v.trim();
            let v = match v.chars().next() {
                Some(q @ ('\'' | '"')) => v[1..].split(q).next().unwrap_or(""),
                _ => v.split(" #").next().unwrap_or(v),
            };
            v.trim().to_string()
        })
        .filter(|v| !v.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_epoch;

    fn at(ts: &str) -> i64 {
        parse_epoch(ts).unwrap() as i64
    }

    #[test]
    fn daily_cron_matches_its_minute_only() {
        let c = Cron::parse("30 9 * * *").unwrap();
        assert!(c.matches(at("2026-09-14T09:30:00Z")));
        assert!(!c.matches(at("2026-09-14T09:31:00Z")));
        assert_eq!(
            c.fires_between(at("2026-09-10T09:30:00Z"), at("2026-09-14T12:00:00Z"), 100),
            4
        );
    }

    #[test]
    fn dom_and_dow_restricted_use_or_semantics() {
        // Demerzel Show & Tell: days 1-7 and 15-21 of any month, OR any Monday.
        let c = Cron::parse("7 14 1-7,15-21 * 1").unwrap();
        assert!(c.matches(at("2026-09-07T14:07:00Z"))); // Mon 7th
        assert!(!c.matches(at("2026-09-09T14:07:00Z"))); // Wed 9th
        assert!(c.matches(at("2026-09-14T14:07:00Z"))); // Mon 14th
        assert_eq!(
            c.fires_between(at("2026-09-07T18:00:00Z"), at("2026-09-14T13:00:00Z"), 100),
            0,
            "no fire due between the 7th and Monday the 14th at 14:07"
        );
    }

    #[test]
    fn steps_ranges_and_names() {
        let c = Cron::parse("*/15 0-6/3 * JAN,jul MON-FRI").unwrap();
        assert!(c.matches(at("2026-07-01T03:45:00Z"))); // Wednesday
        assert!(!c.matches(at("2026-07-01T04:45:00Z")));
        assert!(!c.matches(at("2026-07-04T03:45:00Z"))); // Saturday
        assert!(!c.matches(at("2026-08-03T03:45:00Z"))); // August
        let sunday7 = Cron::parse("0 0 * * 7").unwrap();
        assert!(sunday7.matches(at("2026-09-13T00:00:00Z")));
    }

    #[test]
    fn unsupported_expressions_are_none() {
        assert!(Cron::parse("@daily").is_none());
        assert!(Cron::parse("0 0 * *").is_none());
        assert!(Cron::parse("61 0 * * *").is_none());
        assert!(Cron::parse("0 0 L * *").is_none());
    }

    #[test]
    fn crons_are_extracted_from_workflow_yaml() {
        let yaml = "on:\n  schedule:\n    # - cron: '0 1 * * *'\n    - cron: '7 14 1-7,15-21 * 1'\n    - cron: \"0 5 * * 0\"  # weekly\n  workflow_dispatch:\n";
        assert_eq!(
            crons_in_workflow(yaml),
            vec!["7 14 1-7,15-21 * 1".to_string(), "0 5 * * 0".to_string()]
        );
        assert!(crons_in_workflow("on:\n  push:\n    branches: [main]\n").is_empty());
    }
}
