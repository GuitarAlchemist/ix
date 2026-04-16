//! Karnaugh maps for tetravalent logic minimization.
//!
//! Extends classical 2-valued Karnaugh maps to the 4-valued {T, F, U, C}
//! logic system used by Demerzel governance. Identifies prime implicants
//! and produces minimal policy expressions.
//!
//! # Example: Minimize a 2-variable governance policy
//!
//! ```
//! use ix_governance::karnaugh::KarnaughMap;
//! use ix_governance::tetravalent::TruthValue::*;
//!
//! // Policy: when should we Escalate?
//! // Variables: A = action_risky, B = evidence_contradictory
//! //
//! //        B=T     B=F     B=U     B=C
//! // A=T  [ Esc ]  [ OK ]  [ Esc ] [ Esc ]
//! // A=F  [ OK  ]  [ OK ]  [ OK  ] [ Esc ]
//! // A=U  [ Esc ]  [ OK ]  [ Unk ] [ Esc ]
//! // A=C  [ Esc ]  [ Esc ] [ Esc ] [ Esc ]
//!
//! let mut kmap = KarnaughMap::new(2, vec!["action_risky", "evidence_contradictory"]);
//! // Set cells where policy outputs True (= Escalate)
//! kmap.set(&[True, True], True);
//! kmap.set(&[True, False], False);
//! kmap.set(&[True, Unknown], True);
//! kmap.set(&[True, Contradictory], True);
//! kmap.set(&[False, True], False);
//! kmap.set(&[False, False], False);
//! kmap.set(&[False, Unknown], False);
//! kmap.set(&[False, Contradictory], True);
//! kmap.set(&[Unknown, True], True);
//! kmap.set(&[Unknown, False], False);
//! kmap.set(&[Unknown, Unknown], Unknown);
//! kmap.set(&[Unknown, Contradictory], True);
//! kmap.set(&[Contradictory, True], True);
//! kmap.set(&[Contradictory, False], True);
//! kmap.set(&[Contradictory, Unknown], True);
//! kmap.set(&[Contradictory, Contradictory], True);
//!
//! // Find which input combinations produce True (Escalate)
//! let escalate_cells = kmap.cells_matching(True);
//! assert!(escalate_cells.len() >= 8, "Multiple conditions trigger escalation");
//!
//! // Minimize: find the simplest description
//! let implicants = kmap.prime_implicants(True);
//! assert!(!implicants.is_empty(), "Should find prime implicants");
//!
//! // Display the map
//! let display = kmap.display();
//! assert!(display.contains("action_risky"));
//! ```

use crate::tetravalent::TruthValue;

/// A Karnaugh map for tetravalent logic with 1-4 variables.
///
/// Each variable can take 4 values {T, F, U, C}, so an n-variable
/// map has 4^n cells. The map stores one TruthValue per cell.
pub struct KarnaughMap {
    /// Number of variables (1-4).
    pub n_vars: usize,
    /// Variable names for display.
    pub var_names: Vec<String>,
    /// Flat cell storage: index = mixed-radix encoding of variable values.
    cells: Vec<TruthValue>,
}

/// A prime implicant: a set of variable constraints that covers
/// a group of cells all having the target value.
#[derive(Debug, Clone, PartialEq)]
pub struct Implicant {
    /// For each variable: Some(value) = must equal this, None = don't care.
    pub constraints: Vec<Option<TruthValue>>,
    /// How many cells this implicant covers.
    pub coverage: usize,
}

impl std::fmt::Display for Implicant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parts: Vec<String> = self.constraints.iter().enumerate()
            .filter_map(|(i, c)| c.map(|v| format!("x{}={}", i, v)))
            .collect();
        if parts.is_empty() {
            write!(f, "(always)")
        } else {
            write!(f, "{}", parts.join(" AND "))
        }
    }
}

impl Implicant {
    /// Format with variable names instead of x0, x1, ...
    pub fn display_named(&self, var_names: &[String]) -> String {
        let parts: Vec<String> = self.constraints.iter().enumerate()
            .filter_map(|(i, c)| {
                let name = var_names.get(i).map(|s| s.as_str()).unwrap_or("?");
                c.map(|v| format!("{}={}", name, v))
            })
            .collect();
        if parts.is_empty() {
            "(always)".to_string()
        } else {
            parts.join(" AND ")
        }
    }
}

impl KarnaughMap {
    /// Create a new Karnaugh map with `n_vars` variables.
    ///
    /// All cells default to `Unknown`.
    pub fn new(n_vars: usize, var_names: Vec<&str>) -> Self {
        assert!((1..=4).contains(&n_vars), "Karnaugh maps support 1-4 variables");
        assert_eq!(var_names.len(), n_vars, "Must provide one name per variable");

        let n_cells = 6usize.pow(n_vars as u32);
        Self {
            n_vars,
            var_names: var_names.into_iter().map(|s| s.to_string()).collect(),
            cells: vec![TruthValue::Unknown; n_cells],
        }
    }

    /// Set the cell value for a given variable assignment.
    pub fn set(&mut self, values: &[TruthValue], output: TruthValue) {
        let idx = self.index(values);
        self.cells[idx] = output;
    }

    /// Get the cell value for a given variable assignment.
    pub fn get(&self, values: &[TruthValue]) -> TruthValue {
        self.cells[self.index(values)]
    }

    /// Evaluate a tetravalent expression by looking up the map.
    pub fn evaluate(&self, values: &[TruthValue]) -> TruthValue {
        self.get(values)
    }

    /// Find all variable assignments that produce the target value.
    pub fn cells_matching(&self, target: TruthValue) -> Vec<Vec<TruthValue>> {

        let n_cells = 6usize.pow(self.n_vars as u32);
        let mut results = Vec::new();

        for idx in 0..n_cells {
            if self.cells[idx] == target {
                results.push(self.decode(idx));
            }
        }

        results
    }

    /// Find prime implicants for the target value.
    ///
    /// A prime implicant is a maximal group of cells with the target value
    /// that can be described by fixing some variables and leaving others
    /// as "don't care". Larger groups (fewer constraints) are preferred.
    ///
    /// Uses a greedy approach: start with all-constrained terms, then
    /// try relaxing each variable to find larger groups.
    pub fn prime_implicants(&self, target: TruthValue) -> Vec<Implicant> {
        let matching = self.cells_matching(target);
        if matching.is_empty() {
            return vec![];
        }


        let mut implicants: Vec<Implicant> = Vec::new();

        // Generate candidate implicants by trying all possible constraint patterns
        // For each subset of variables, check if fixing them covers only target cells
        self.find_implicants_recursive(
            &vec![None; self.n_vars],
            0,
            target,
            &mut implicants,
        );

        // Remove implicants that are subsumed by larger ones
        let mut prime = Vec::new();
        for imp in &implicants {
            let subsumed = implicants.iter().any(|other| {
                other.coverage > imp.coverage && self.subsumes(other, imp)
            });
            if !subsumed {
                prime.push(imp.clone());
            }
        }

        // Deduplicate
        prime.sort_by_key(|imp| std::cmp::Reverse(imp.coverage));
        prime.dedup();
        prime
    }

    /// Find a minimal cover: smallest set of implicants that covers all
    /// cells matching the target.
    pub fn minimal_cover(&self, target: TruthValue) -> Vec<Implicant> {
        let primes = self.prime_implicants(target);
        let matching = self.cells_matching(target);

        if matching.is_empty() || primes.is_empty() {
            return vec![];
        }

        // Greedy set cover: pick the largest implicant, remove covered cells, repeat
        let mut uncovered: Vec<Vec<TruthValue>> = matching;
        let mut cover = Vec::new();

        while !uncovered.is_empty() {
            // Find the implicant that covers the most uncovered cells
            let best = primes.iter()
                .max_by_key(|imp| {
                    uncovered.iter().filter(|cell| self.implicant_covers(imp, cell)).count()
                });

            match best {
                Some(imp) => {
                    uncovered.retain(|cell| !self.implicant_covers(imp, cell));
                    cover.push(imp.clone());
                }
                None => break,
            }
        }

        cover
    }

    /// Display the Karnaugh map as a formatted string.
    pub fn display(&self) -> String {
        if self.n_vars == 1 {
            self.display_1var()
        } else if self.n_vars == 2 {
            self.display_2var()
        } else {
            self.display_generic()
        }
    }

    /// Generate the full truth table as a string.
    pub fn truth_table(&self) -> String {

        let n_cells = 6usize.pow(self.n_vars as u32);
        let mut lines = Vec::new();

        // Header
        let header: Vec<&str> = self.var_names.iter().map(|s| s.as_str()).collect();
        lines.push(format!("{} | Output", header.join(" | ")));
        lines.push("-".repeat(lines[0].len()));

        for idx in 0..n_cells {
            let values = self.decode(idx);
            let vals: Vec<String> = values.iter().map(|v| format!("{:>1}", v)).collect();
            lines.push(format!("{}  |  {}", vals.join("  |  "), self.cells[idx]));
        }

        lines.join("\n")
    }

    // ── Internal helpers ──

    fn index(&self, values: &[TruthValue]) -> usize {
        assert_eq!(values.len(), self.n_vars);
        let mut idx = 0;
        for &v in values {
            idx = idx * 6 + Self::value_to_int(v);
        }
        idx
    }

    fn decode(&self, mut idx: usize) -> Vec<TruthValue> {
        let mut values = vec![TruthValue::True; self.n_vars];
        for i in (0..self.n_vars).rev() {
            values[i] = Self::int_to_value(idx % 6);
            idx /= 6;
        }
        values
    }

    fn value_to_int(v: TruthValue) -> usize {
        match v {
            TruthValue::True => 0,
            TruthValue::Probable => 1,
            TruthValue::Unknown => 2,
            TruthValue::Disputed => 3,
            TruthValue::False => 4,
            TruthValue::Contradictory => 5,
        }
    }

    fn int_to_value(i: usize) -> TruthValue {
        match i {
            0 => TruthValue::True,
            1 => TruthValue::Probable,
            2 => TruthValue::Unknown,
            3 => TruthValue::Disputed,
            4 => TruthValue::False,
            _ => TruthValue::Contradictory,
        }
    }

    fn find_implicants_recursive(
        &self,
        constraints: &[Option<TruthValue>],
        var_idx: usize,
        target: TruthValue,
        results: &mut Vec<Implicant>,
    ) {
        // Check if current constraints cover only target cells
        let (covers_all, coverage) = self.check_coverage(constraints, target);

        if covers_all && coverage > 0 {
            results.push(Implicant {
                constraints: constraints.to_vec(),
                coverage,
            });
        }

        if var_idx >= self.n_vars {
            return;
        }

        // Try fixing this variable to each value
        for &val in &TruthValue::all() {
            let mut new_constraints = constraints.to_vec();
            new_constraints[var_idx] = Some(val);
            self.find_implicants_recursive(&new_constraints, var_idx + 1, target, results);
        }

        // Try leaving this variable as don't-care
        let mut dc_constraints = constraints.to_vec();
        dc_constraints[var_idx] = None;
        self.find_implicants_recursive(&dc_constraints, var_idx + 1, target, results);
    }

    fn check_coverage(&self, constraints: &[Option<TruthValue>], target: TruthValue) -> (bool, usize) {

        let mut count = 0;
        let mut all_match = true;

        self.iterate_cells(constraints, 0, &mut vec![TruthValue::True; self.n_vars], &mut |values| {
            let cell = self.get(values);
            if cell == target {
                count += 1;
            } else {
                all_match = false;
            }
        });

        (all_match, count)
    }

    fn iterate_cells(
        &self,
        constraints: &[Option<TruthValue>],
        var_idx: usize,
        current: &mut Vec<TruthValue>,
        callback: &mut dyn FnMut(&[TruthValue]),
    ) {
        if var_idx >= self.n_vars {
            callback(current);
            return;
        }

        match constraints[var_idx] {
            Some(val) => {
                current[var_idx] = val;
                self.iterate_cells(constraints, var_idx + 1, current, callback);
            }
            None => {
                for &val in &TruthValue::all() {
                    current[var_idx] = val;
                    self.iterate_cells(constraints, var_idx + 1, current, callback);
                }
            }
        }
    }

    fn subsumes(&self, larger: &Implicant, smaller: &Implicant) -> bool {
        // larger subsumes smaller if every cell covered by smaller is also covered by larger
        for (l, s) in larger.constraints.iter().zip(smaller.constraints.iter()) {
            match (l, s) {
                (None, _) => {} // don't care covers everything
                (Some(lv), Some(sv)) if lv == sv => {} // same constraint
                (Some(_), None) => return false, // larger is more constrained here
                (Some(_), Some(_)) => return false, // different values
            }
        }
        true
    }

    fn implicant_covers(&self, imp: &Implicant, cell: &[TruthValue]) -> bool {
        imp.constraints.iter().zip(cell.iter()).all(|(c, v)| {
            match c {
                None => true,
                Some(cv) => cv == v,
            }
        })
    }

    fn display_1var(&self) -> String {
        let all = TruthValue::all();
        let mut s = format!("Karnaugh Map (1 var: {})\n", self.var_names[0]);
        s.push_str(&format!("{:>4} | Output\n", self.var_names[0]));
        s.push_str("-----|-------\n");
        for &v in &all {
            s.push_str(&format!("{:>4} | {}\n", v, self.get(&[v])));
        }
        s
    }

    fn display_2var(&self) -> String {
        let all = TruthValue::all();
        let mut s = format!("Karnaugh Map (2 vars: {}, {})\n", self.var_names[0], self.var_names[1]);
        s.push_str(&format!("{:>12} |", self.var_names[1]));
        for &v in &all {
            s.push_str(&format!("  {:>1} ", v));
        }
        s.push('\n');
        s.push_str(&format!("{:>12} |", ""));
        s.push_str("----".repeat(4).as_str());
        s.push('\n');
        for &row in &all {
            s.push_str(&format!("{:>8}={:>1}  |", self.var_names[0], row));
            for &col in &all {
                s.push_str(&format!("  {:>1} ", self.get(&[row, col])));
            }
            s.push('\n');
        }
        s
    }

    fn display_generic(&self) -> String {
        format!("Karnaugh Map ({} vars: {})\n{}",
            self.n_vars,
            self.var_names.join(", "),
            self.truth_table())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use TruthValue::*;

    #[test]
    fn test_set_get() {
        let mut kmap = KarnaughMap::new(2, vec!["A", "B"]);
        kmap.set(&[True, False], Contradictory);
        assert_eq!(kmap.get(&[True, False]), Contradictory);
        assert_eq!(kmap.get(&[True, True]), Unknown); // default
    }

    #[test]
    fn test_1var_map() {
        let mut kmap = KarnaughMap::new(1, vec!["risky"]);
        kmap.set(&[True], True);
        kmap.set(&[False], False);
        kmap.set(&[Unknown], Unknown);
        kmap.set(&[Contradictory], True);

        let matches = kmap.cells_matching(True);
        // True and Contradictory are set to True; Probable and Disputed default to Unknown
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_2var_cells_matching() {
        let mut kmap = KarnaughMap::new(2, vec!["A", "B"]);
        kmap.set(&[True, True], True);
        kmap.set(&[True, False], False);
        kmap.set(&[False, True], True);
        kmap.set(&[False, False], False);

        let true_cells = kmap.cells_matching(True);
        // Only (T,T) and (F,T) are set to True; all other cells default to Unknown
        assert_eq!(true_cells.len(), 2);
    }

    #[test]
    fn test_prime_implicants_simple() {
        // If B=T -> output T regardless of A (a "don't care" on A)
        let mut kmap = KarnaughMap::new(2, vec!["A", "B"]);
        for &a in &TruthValue::all() {
            kmap.set(&[a, True], True);
            kmap.set(&[a, Probable], False);
            kmap.set(&[a, False], False);
            kmap.set(&[a, Unknown], False);
            kmap.set(&[a, Disputed], False);
            kmap.set(&[a, Contradictory], False);
        }

        let implicants = kmap.prime_implicants(True);
        // Should find one implicant: B=T (A is don't-care)
        assert!(!implicants.is_empty());
        let best = &implicants[0];
        assert_eq!(best.constraints[0], None, "A should be don't-care");
        assert_eq!(best.constraints[1], Some(True), "B should be fixed to T");
        assert_eq!(best.coverage, 6); // covers all 6 values of A
    }

    #[test]
    fn test_prime_implicants_all_true() {
        // All cells are True -> one implicant with all don't-cares
        let mut kmap = KarnaughMap::new(2, vec!["A", "B"]);
        for &a in &TruthValue::all() {
            for &b in &TruthValue::all() {
                kmap.set(&[a, b], True);
            }
        }

        let implicants = kmap.prime_implicants(True);
        assert!(!implicants.is_empty());
        let best = &implicants[0];
        assert_eq!(best.coverage, 36); // 6 x 6 = 36 cells
        assert!(best.constraints.iter().all(|c| c.is_none()));
    }

    #[test]
    fn test_minimal_cover() {
        let mut kmap = KarnaughMap::new(2, vec!["A", "B"]);
        // True when A=T or B=C
        for &a in &TruthValue::all() {
            for &b in &TruthValue::all() {
                if a == True || b == Contradictory {
                    kmap.set(&[a, b], True);
                } else {
                    kmap.set(&[a, b], False);
                }
            }
        }

        let cover = kmap.minimal_cover(True);
        assert!(!cover.is_empty());
        // Should need at most 2 implicants: A=T and B=C
        assert!(cover.len() <= 2, "Minimal cover should have <= 2 terms, got {}", cover.len());
    }

    #[test]
    fn test_display_2var() {
        let mut kmap = KarnaughMap::new(2, vec!["risk", "evidence"]);
        kmap.set(&[True, True], True);
        let display = kmap.display();
        assert!(display.contains("risk"));
        assert!(display.contains("evidence"));
    }

    #[test]
    fn test_truth_table() {
        let mut kmap = KarnaughMap::new(1, vec!["X"]);
        kmap.set(&[True], True);
        kmap.set(&[False], False);
        let table = kmap.truth_table();
        assert!(table.contains("Output"));
    }

    #[test]
    fn test_implicant_display() {
        let imp = Implicant {
            constraints: vec![Some(True), None, Some(Contradictory)],
            coverage: 4,
        };
        assert_eq!(format!("{}", imp), "x0=T AND x2=C");

        let named = imp.display_named(&["risk".into(), "urgency".into(), "evidence".into()]);
        assert_eq!(named, "risk=T AND evidence=C");
    }

    #[test]
    fn test_implicant_display_all_dont_care() {
        let imp = Implicant {
            constraints: vec![None, None],
            coverage: 16,
        };
        assert_eq!(format!("{}", imp), "(always)");
    }

    #[test]
    fn test_governance_escalation_policy() {
        // Real-world governance: when to escalate?
        // A = action_risky, B = evidence_contradictory
        // Escalate (True) when: A=C (always), B=C (always), or both indefinite
        let mut kmap = KarnaughMap::new(2, vec!["action_risky", "evidence_contradictory"]);

        for &a in &TruthValue::all() {
            for &b in &TruthValue::all() {
                let output = if a == Contradictory || b == Contradictory {
                    True // Always escalate contradictions
                } else if a == True && b == True {
                    True // Risky + contradictory evidence = escalate
                } else if a.is_indefinite() && b.is_indefinite() {
                    True // Both uncertain = escalate
                } else {
                    False
                };
                kmap.set(&[a, b], output);
            }
        }

        let cover = kmap.minimal_cover(True);
        assert!(!cover.is_empty());

        // Display the policy
        for imp in &cover {
            let desc = imp.display_named(&kmap.var_names);
            assert!(!desc.is_empty());
        }
    }

    #[test]
    fn test_3var_map() {
        let mut kmap = KarnaughMap::new(3, vec!["A", "B", "C"]);
        kmap.set(&[True, True, True], True);
        kmap.set(&[True, True, False], True);
        assert_eq!(kmap.get(&[True, True, True]), True);
        assert_eq!(kmap.get(&[False, False, False]), Unknown); // default
    }

    #[test]
    fn test_implies() {
        // T→T = T, T→F = F, F→anything = T
        assert_eq!(True.implies(True), True);
        assert_eq!(True.implies(False), False);
        assert_eq!(False.implies(True), True);
        assert_eq!(False.implies(False), True);
        assert_eq!(Unknown.implies(True), True);
        assert_eq!(Unknown.implies(False), Unknown);
    }

    #[test]
    fn test_xor() {
        assert_eq!(True.xor(True), False);
        assert_eq!(True.xor(False), True);
        assert_eq!(False.xor(True), True);
        assert_eq!(False.xor(False), False);
    }

    #[test]
    fn test_equiv() {
        assert_eq!(True.equiv(True), True);
        assert_eq!(True.equiv(False), False);
        assert_eq!(False.equiv(False), True);
        assert_eq!(Unknown.equiv(Unknown), Unknown);
    }
}
