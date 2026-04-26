//! `IxNode` — the node enum for v0 of the visual pipeline editor.
//!
//! Each variant maps conceptually to a registered skill in `ix-registry`, and
//! declares its input/output sockets statically so the viewer can render
//! typed pins without touching the registry on every repaint.

use super::sockets::SocketType;
use serde::{Deserialize, Serialize};

/// Starter v0 nodes: enough coverage to demo typed sockets + governance
/// gates + save/load, without requiring every registry skill to be wrapped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IxNode {
    /// Load a CSV file → Dataset.
    CsvRead { path: String },
    /// Write a Dataset to CSV.
    CsvWrite { path: String },
    /// Literal scalar constant.
    Constant { value: f64 },
    /// Normalize a dataset (z-score / min-max / L2).
    Normalize { method: NormMethod },
    /// K-Means clustering (Dataset in, Vector labels out).
    KMeans {
        k: usize,
        max_iter: usize,
        seed: u64,
    },
    /// Ordinary least-squares linear regression.
    LinearReg,
    /// Fast Fourier Transform on a Vector signal.
    Fft { inverse: bool },
    /// Governance policy gate — distinct red visual, Any-in/Belief-out.
    PolicyGate { policy: String, threshold: f64 },
    /// Display a value (sink — no outputs).
    Plot { title: String },
    /// Standalone hexavalent belief — no inputs, emits a Belief value.
    Belief {
        proposition: String,
        value: HexValue,
        confidence: f64,
    },
    /// Generic registry skill — socket metadata is runtime-determined.
    /// Used by `ix.yaml` import: each imported stage becomes a `Skill` node.
    Skill {
        /// Dotted skill name resolved against `ix-registry`.
        skill: String,
        /// JSON args blob from the YAML stage.
        args: serde_json::Value,
        /// Human-visible node title (defaults to the skill name).
        label: String,
        /// Input pins (name + socket type) — a single synthetic `in: Any`
        /// is used when no upstream deps are known.
        ui_inputs: Vec<Socket>,
        /// Output pins (name + socket type).
        ui_outputs: Vec<Socket>,
    },
}

/// Runtime socket descriptor used by the `Skill` variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Socket {
    pub name: String,
    pub ty: SocketType,
}

/// Thin local mirror of `ix_types::Hexavalent` — avoids forcing serde tags
/// through the upstream enum to keep the editor's JSON compact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HexValue {
    T,
    P,
    U,
    D,
    F,
    C,
}

impl HexValue {
    pub const fn all() -> [HexValue; 6] {
        [
            HexValue::T,
            HexValue::P,
            HexValue::U,
            HexValue::D,
            HexValue::F,
            HexValue::C,
        ]
    }

    pub const fn letter(self) -> &'static str {
        match self {
            HexValue::T => "T",
            HexValue::P => "P",
            HexValue::U => "U",
            HexValue::D => "D",
            HexValue::F => "F",
            HexValue::C => "C",
        }
    }

    pub const fn color(self) -> eframe::egui::Color32 {
        use eframe::egui::Color32;
        match self {
            HexValue::T => Color32::from_rgb(0x22, 0xc5, 0x5e), // green
            HexValue::P => Color32::from_rgb(0xa3, 0xe6, 0x35), // amber-green
            HexValue::U => Color32::from_rgb(0x6b, 0x72, 0x80), // gray
            HexValue::D => Color32::from_rgb(0xf9, 0x73, 0x16), // amber-red
            HexValue::F => Color32::from_rgb(0xef, 0x44, 0x44), // red
            HexValue::C => Color32::from_rgb(0xd9, 0x46, 0xef), // magenta
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum NormMethod {
    ZScore,
    MinMax,
    L2,
}

impl IxNode {
    pub fn title(&self) -> String {
        match self {
            IxNode::CsvRead { .. } => "CSV Read".into(),
            IxNode::CsvWrite { .. } => "CSV Write".into(),
            IxNode::Constant { .. } => "Constant".into(),
            IxNode::Normalize { .. } => "Normalize".into(),
            IxNode::KMeans { .. } => "K-Means".into(),
            IxNode::LinearReg => "Linear Regression".into(),
            IxNode::Fft { .. } => "FFT".into(),
            IxNode::PolicyGate { .. } => "Policy Gate".into(),
            IxNode::Plot { .. } => "Plot".into(),
            IxNode::Belief { .. } => "Belief".into(),
            IxNode::Skill { label, .. } => label.clone(),
        }
    }

    pub fn input_sockets(&self) -> Vec<Socket> {
        use SocketType::*;
        let pairs: &[(&str, SocketType)] = match self {
            IxNode::CsvRead { .. } => &[],
            IxNode::CsvWrite { .. } => &[("data", Dataset)],
            IxNode::Constant { .. } => &[],
            IxNode::Normalize { .. } => &[("in", Dataset)],
            IxNode::KMeans { .. } => &[("in", Dataset)],
            IxNode::LinearReg => &[("X", Matrix), ("y", Vector)],
            IxNode::Fft { .. } => &[("signal", Vector)],
            IxNode::PolicyGate { .. } => &[("value", Any)],
            IxNode::Plot { .. } => &[("data", Any)],
            IxNode::Belief { .. } => &[],
            IxNode::Skill { ui_inputs, .. } => return ui_inputs.clone(),
        };
        pairs
            .iter()
            .map(|(name, ty)| Socket {
                name: (*name).to_string(),
                ty: *ty,
            })
            .collect()
    }

    pub fn output_sockets(&self) -> Vec<Socket> {
        use SocketType::*;
        let pairs: &[(&str, SocketType)] = match self {
            IxNode::CsvRead { .. } => &[("data", Dataset)],
            IxNode::CsvWrite { .. } => &[],
            IxNode::Constant { .. } => &[("v", Scalar)],
            IxNode::Normalize { .. } => &[("out", Dataset)],
            IxNode::KMeans { .. } => &[("labels", Vector), ("centroids", Matrix)],
            IxNode::LinearReg => &[("model", Model), ("coefs", Vector)],
            IxNode::Fft { .. } => &[("spectrum", Vector)],
            IxNode::PolicyGate { .. } => &[("passed", Any), ("verdict", Belief)],
            IxNode::Plot { .. } => &[],
            IxNode::Belief { .. } => &[("belief", Belief)],
            IxNode::Skill { ui_outputs, .. } => return ui_outputs.clone(),
        };
        pairs
            .iter()
            .map(|(name, ty)| Socket {
                name: (*name).to_string(),
                ty: *ty,
            })
            .collect()
    }

    /// True for any node that performs ML training or prediction — used by
    /// static validation to enforce "every Learner has an upstream PolicyGate".
    pub const fn is_learner(&self) -> bool {
        matches!(self, IxNode::KMeans { .. } | IxNode::LinearReg)
    }

    /// True for the PolicyGate variant — the viewer draws its header in red.
    pub const fn is_gate(&self) -> bool {
        matches!(self, IxNode::PolicyGate { .. })
    }

    /// Hint at the registry skill this node maps to (for future execution).
    pub fn registry_skill(&self) -> Option<&str> {
        match self {
            IxNode::KMeans { .. } => Some("kmeans"),
            IxNode::LinearReg => Some("linear_regression"),
            IxNode::Fft { .. } => Some("fft"),
            IxNode::PolicyGate { .. } => Some("governance.check"),
            IxNode::Skill { skill, .. } => Some(skill.as_str()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_has_title() {
        let variants = [
            IxNode::CsvRead {
                path: String::new(),
            },
            IxNode::CsvWrite {
                path: String::new(),
            },
            IxNode::Constant { value: 0.0 },
            IxNode::Normalize {
                method: NormMethod::ZScore,
            },
            IxNode::KMeans {
                k: 3,
                max_iter: 100,
                seed: 42,
            },
            IxNode::LinearReg,
            IxNode::Fft { inverse: false },
            IxNode::PolicyGate {
                policy: "x".into(),
                threshold: 0.5,
            },
            IxNode::Plot { title: "p".into() },
        ];
        for v in &variants {
            assert!(!v.title().is_empty(), "variant {:?} has empty title", v);
        }
    }

    #[test]
    fn only_policy_gate_is_gate() {
        assert!(IxNode::PolicyGate {
            policy: "x".into(),
            threshold: 0.5
        }
        .is_gate());
        assert!(!IxNode::KMeans {
            k: 3,
            max_iter: 100,
            seed: 1
        }
        .is_gate());
        assert!(!IxNode::LinearReg.is_gate());
    }

    #[test]
    fn registry_skill_mapping_covers_ml_nodes() {
        assert_eq!(
            IxNode::KMeans {
                k: 3,
                max_iter: 100,
                seed: 1
            }
            .registry_skill(),
            Some("kmeans")
        );
        assert_eq!(
            IxNode::LinearReg.registry_skill(),
            Some("linear_regression")
        );
        assert_eq!(IxNode::Fft { inverse: false }.registry_skill(), Some("fft"));
        assert_eq!(
            IxNode::PolicyGate {
                policy: "x".into(),
                threshold: 0.5
            }
            .registry_skill(),
            Some("governance.check")
        );
        // IO / viz nodes return None (no direct skill mapping).
        assert_eq!(IxNode::Plot { title: "x".into() }.registry_skill(), None);
        assert_eq!(IxNode::CsvRead { path: "x".into() }.registry_skill(), None);
    }

    #[test]
    fn input_output_sockets_never_panic() {
        // Static arrays — just confirm len + first-pair types for a few.
        assert_eq!(
            IxNode::KMeans {
                k: 3,
                max_iter: 1,
                seed: 1
            }
            .input_sockets()
            .len(),
            1
        );
        assert_eq!(
            IxNode::KMeans {
                k: 3,
                max_iter: 1,
                seed: 1
            }
            .output_sockets()
            .len(),
            2
        );
        assert_eq!(IxNode::LinearReg.input_sockets().len(), 2);
        assert_eq!(IxNode::LinearReg.output_sockets().len(), 2);
        assert_eq!(
            IxNode::CsvRead { path: "x".into() }.input_sockets().len(),
            0
        );
        assert_eq!(IxNode::Plot { title: "x".into() }.output_sockets().len(), 0);
    }

    #[test]
    fn belief_variant_has_zero_inputs_one_output() {
        let b = IxNode::Belief {
            proposition: "test".into(),
            value: HexValue::P,
            confidence: 0.7,
        };
        assert_eq!(b.title(), "Belief");
        assert_eq!(b.input_sockets().len(), 0);
        assert_eq!(b.output_sockets().len(), 1);
        assert_eq!(
            b.output_sockets()[0].ty,
            super::super::sockets::SocketType::Belief
        );
        assert_eq!(b.registry_skill(), None);
        assert!(!b.is_gate());
        assert!(!b.is_learner());
    }

    #[test]
    fn is_learner_only_ml_nodes() {
        assert!(IxNode::KMeans {
            k: 3,
            max_iter: 1,
            seed: 1,
        }
        .is_learner());
        assert!(IxNode::LinearReg.is_learner());
        assert!(!IxNode::Fft { inverse: false }.is_learner());
        assert!(!IxNode::PolicyGate {
            policy: "x".into(),
            threshold: 0.5,
        }
        .is_learner());
        assert!(!IxNode::CsvRead { path: "x".into() }.is_learner());
    }

    #[test]
    fn hex_value_all_six_symbols() {
        let letters: Vec<&str> = HexValue::all().iter().map(|h| h.letter()).collect();
        assert_eq!(letters, vec!["T", "P", "U", "D", "F", "C"]);
    }

    #[test]
    fn hex_value_colors_distinct() {
        let colors: Vec<_> = HexValue::all().iter().map(|h| h.color()).collect();
        for (i, a) in colors.iter().enumerate() {
            for b in &colors[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn belief_serde_roundtrip() {
        let original = IxNode::Belief {
            proposition: "API stable".into(),
            value: HexValue::D,
            confidence: 0.3,
        };
        let json = serde_json::to_string(&original).unwrap();
        let back: IxNode = serde_json::from_str(&json).unwrap();
        match back {
            IxNode::Belief {
                proposition,
                value,
                confidence,
            } => {
                assert_eq!(proposition, "API stable");
                assert_eq!(value, HexValue::D);
                assert_eq!(confidence, 0.3);
            }
            other => panic!("expected Belief, got {:?}", other),
        }
    }

    #[test]
    fn ixnode_roundtrips_through_serde() {
        let original = IxNode::KMeans {
            k: 5,
            max_iter: 200,
            seed: 7,
        };
        let json = serde_json::to_string(&original).unwrap();
        let back: IxNode = serde_json::from_str(&json).unwrap();
        match back {
            IxNode::KMeans { k, max_iter, seed } => {
                assert_eq!(k, 5);
                assert_eq!(max_iter, 200);
                assert_eq!(seed, 7);
            }
            other => panic!("expected KMeans, got {:?}", other),
        }
    }
}
