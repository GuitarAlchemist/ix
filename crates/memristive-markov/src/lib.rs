pub mod conductance;
pub mod consolidator;
pub mod engine;
pub mod error;
pub mod sampler;
pub mod serde_state;
pub mod tensor;
pub mod vlmm;

#[cfg(feature = "reservoir")]
pub mod reservoir;

#[cfg(feature = "ffi")]
pub mod ffi;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use conductance::ConductanceMatrix;
pub use consolidator::MemoryConsolidator;
pub use error::{MemristiveError, Result};
pub use sampler::SamplingStrategy;
pub use tensor::MarkovTensor;
pub use vlmm::VariableOrderSelector;
pub mod governance;

pub use engine::MemristiveEngine;
pub use governance::{GovernanceMarkov, GovernanceState};
