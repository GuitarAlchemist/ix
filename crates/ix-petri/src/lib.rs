//! # ix-petri — Place/Transition Petri nets, and the properties you cannot read off one
//!
//! A [`PetriNet`] is places, transitions, weighted arcs and a marking. That
//! part is easy and is not why the crate exists. The crate exists for
//! [`analysis::analyze`]: **deadlock detection**, **boundedness**, **dead
//! transitions**, **liveness** and **reversibility**, each with a witness and
//! each honest about the limit of its own enumeration.
//!
//! ## Why a Petri net rather than something IX already has
//!
//! IX ships plenty of state machinery, and none of it answers these questions:
//!
//! | Existing | What it models | Why it does not cover this |
//! |---|---|---|
//! | `ix_pipeline::dag::Dag` | task dependencies | acyclic **by construction** — `add_edge` rejects cycles, so it cannot express a resource returned to a pool and taken again |
//! | `ix_graph::markov`, `hmm`, `state_space` | *probabilistic* state evolution | a distribution over states, not "can these two lanes wedge each other" |
//! | `ix_fuzzy`, `ix_grammar` | degrees of truth; string derivation | no notion of concurrent tokens contending for a resource |
//!
//! A Petri net is what you reach for when the thing you are modelling has
//! **cycles, concurrency and contention at once** — two workers entering one
//! working tree, a lane holding a lock nobody released, a queue that can
//! deadlock. A DAG cannot express any of them.
//!
//! ## Determinism
//!
//! Firing order is fixed by the net's own type: places and transitions are
//! sorted by id, ids are unique, so the ordering is **total** and no tie is
//! left for another rule to break. Every analysis therefore returns the same
//! state numbering and the same witness sequences on every run and machine.
//! See [`net`] for the full argument.
//!
//! ## Example — a lock that is taken but never released
//!
//! ```
//! use ix_petri::{analysis::{analyze, Limits, Verdict}, PetriNet};
//!
//! let net = PetriNet::builder()
//!     .name("leaked-lock")
//!     .place("lock", 1)
//!     .place("working", 0)
//!     .transition("acquire")
//!     .arc("lock", "acquire")
//!     .arc("acquire", "working")
//!     .build()?;
//!
//! let report = analyze(&net, Limits::default());
//! let Verdict::Fails(deadlocks) = &report.deadlock_free else {
//!     panic!("a lock nobody releases must wedge");
//! };
//! assert_eq!(deadlocks[0].witness, ["acquire"]);
//! assert_eq!(deadlocks[0].marking, "working=1");
//! # Ok::<(), ix_petri::PetriError>(())
//! ```
//!
//! ## Reading nets written elsewhere
//!
//! [`pnml::read_pnml`] reads the Place/Transition subclass of PNML
//! (ISO/IEC 15909-2), so a net authored in another tool can be analysed here.
//! IX does not *write* PNML; [`pnml`] documents that choice, the subset that is
//! read, and what an emitter would have to satisfy first.

pub mod analysis;
pub mod models;
pub mod net;
pub mod pnml;
pub mod xml;

pub use analysis::{analyze, Analysis, Limits, Verdict};
pub use net::{Marking, PetriError, PetriNet, PetriNetBuilder, Place, Transition};
pub use pnml::{read_pnml, PnmlError, PnmlNet};
