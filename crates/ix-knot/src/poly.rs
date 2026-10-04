//! Laurent polynomials in A with integer coefficients, the bracket's ring.

use std::collections::BTreeMap;

/// Exponent → coefficient, with no zero coefficients stored.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Laurent(BTreeMap<i32, i128>);

impl Laurent {
    pub(crate) fn one() -> Self {
        Self(BTreeMap::from([(0, 1)]))
    }

    /// The terms in ascending exponent order.
    pub(crate) fn terms(&self) -> impl Iterator<Item = (i32, i128)> + '_ {
        self.0.iter().map(|(e, c)| (*e, *c))
    }

    /// `self += coeff · A^shift · other`.
    pub(crate) fn add_scaled(&mut self, other: &Laurent, coeff: i128, shift: i32) {
        for (e, c) in other.terms() {
            let entry = self.0.entry(e + shift).or_insert(0);
            *entry += coeff * c;
            if *entry == 0 {
                self.0.remove(&(e + shift));
            }
        }
    }

    /// `(-A² - A⁻²) · self`: the factor a closed loop contributes.
    pub(crate) fn times_loop(&self) -> Self {
        let mut out = Self::default();
        out.add_scaled(self, -1, 2);
        out.add_scaled(self, -1, -2);
        out
    }

    /// [`Self::add_scaled`], or `None` when a coefficient overflows `i128`
    /// (`self` is then left part-way).
    pub(crate) fn checked_add_scaled(
        &mut self,
        other: &Laurent,
        coeff: i128,
        shift: i32,
    ) -> Option<()> {
        for (e, c) in other.terms() {
            let entry = self.0.entry(e + shift).or_insert(0);
            *entry = entry.checked_add(coeff.checked_mul(c)?)?;
            if *entry == 0 {
                self.0.remove(&(e + shift));
            }
        }
        Some(())
    }

    /// [`Self::times_loop`], or `None` on overflow.
    pub(crate) fn checked_times_loop(&self) -> Option<Self> {
        let mut out = Self::default();
        out.checked_add_scaled(self, -1, 2)?;
        out.checked_add_scaled(self, -1, -2)?;
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_loop_squared_is_a4_plus_2_plus_a_minus_4() {
        let d2 = Laurent::one().times_loop().times_loop();
        assert_eq!(
            d2.terms().collect::<Vec<_>>(),
            vec![(-4, 1), (0, 2), (4, 1)]
        );
    }

    #[test]
    fn cancelling_terms_are_dropped() {
        let mut p = Laurent::one();
        p.add_scaled(&Laurent::one(), -1, 0);
        assert_eq!(p, Laurent::default());
    }
}
