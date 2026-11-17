//! One call that runs the checks and the optional readings of a fold, and applies the verdict
//! rules of ADR-0007 itself: a stack is reported beside the checks and never changes their
//! verdict, and a stack is `rigid_ok` only when the checks pass and the stack is not refused.
//!
//! The census and the stack read one prepared [`Context`], built once.

use thiserror::Error;

use crate::fold::Fold;
use crate::layers::{analyse_within, swap_census_within, CensusRow, Context, Report, Unchecked};
use crate::limits::Limits;
use crate::thickness::{stack, BadThickness, Stack};

/// What to read beside the checks.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Options {
    /// The sheet's thickness: also read the stack it implies.
    pub thickness: Option<f64>,
    /// Also flip each stated pair alone and record which rules reject it.
    pub census: bool,
}

/// The checks, and what was asked for beside them.
#[derive(Debug, Clone, PartialEq)]
pub struct Assessment {
    pub report: Report,
    /// `Some` when a thickness was given.
    pub stack: Option<Stack>,
    /// `Some` when the census was asked for.
    pub census: Option<Vec<CensusRow>>,
}

impl Assessment {
    /// Whether the stated order can be realised by parallel rigid panels: the checks pass and
    /// the stack is not refused. `None` when no thickness was given.
    pub fn rigid_ok(&self) -> Option<bool> {
        self.stack
            .as_ref()
            .map(|s| self.report.ok && !s.rigid.refused)
    }
}

/// Why [`assess`] gave no assessment.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AssessError {
    /// The structure problems that stop the checks, or a count over the limits.
    #[error(transparent)]
    Unchecked(#[from] Unchecked),
    /// The thickness is not usable: see [`BadThickness`].
    #[error(transparent)]
    Thickness(#[from] BadThickness),
}

/// The checks of [`analyse_within`], then the census and the stack when asked for, in that order:
/// the first refusal is the one returned. A fold or census over `limits` is refused, never
/// checked in part.
pub fn assess(fold: &Fold, limits: &Limits, options: &Options) -> Result<Assessment, AssessError> {
    let report = analyse_within(fold, limits)?;
    let ctx = if options.census || options.thickness.is_some() {
        Some(Context::within(fold, limits)?)
    } else {
        None
    };
    let census = match (&ctx, options.census) {
        (Some(ctx), true) => {
            Some(swap_census_within(fold, ctx, limits).map_err(Unchecked::OverLimit)?)
        }
        _ => None,
    };
    let stack = match (&ctx, options.thickness) {
        (Some(ctx), Some(t)) => Some(stack(ctx, t)?),
        _ => None,
    };
    Ok(Assessment {
        report,
        stack,
        census,
    })
}
