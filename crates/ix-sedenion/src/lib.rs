pub mod bsp;
pub mod cayley_dickson;
pub mod octonion;
pub mod sedenion;

pub use bsp::BspNode;
pub use cayley_dickson::{double_conjugate, double_multiply, double_norm};
pub use octonion::Octonion;
pub use sedenion::Sedenion;
