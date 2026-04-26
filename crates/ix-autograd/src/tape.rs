//! Wengert-style reverse-mode tape.
//!
//! Day 1 scaffold: types and skeleton API only. Day 2 fills in the
//! backward walk and the finite-difference verifier that consumes it.

use crate::mode::ExecutionMode;
use crate::tensor::Tensor;
use std::any::Any;
use std::collections::HashMap;

/// Opaque index into the tape. Returned by every op; consumed by
/// subsequent ops and by [`DiffContext::backward`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorHandle(
    /// Underlying tape index. Public so that external backends can
    /// construct handles; prefer using the opaque interface.
    pub usize,
);

/// A single node on the Wengert tape.
#[derive(Debug)]
pub struct TapeNode {
    /// Op name (`"input"`, `"add"`, `"mul"`, `"sum"`, ...). Used by
    /// the reverse walker to dispatch to the right backward function.
    pub op: &'static str,
    /// Handles of the upstream operands. Empty for input leaves.
    pub inputs: Vec<TensorHandle>,
    /// The forward-computed value of this node.
    pub value: Tensor,
    /// Gradient accumulated during the reverse walk. `None` until
    /// `backward` touches this node.
    pub grad: Option<Tensor>,
    /// Tool-specific saved state used by `backward`. JSON for now so
    /// tools can record whatever they need without coupling to this crate.
    pub saved: Option<serde_json::Value>,
}

/// The Wengert tape. Append-only during forward, walked in reverse
/// during backward.
#[derive(Debug, Default)]
pub struct Tape {
    nodes: Vec<TapeNode>,
}

impl Tape {
    /// Construct an empty tape.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a node onto the tape and return its handle.
    pub fn push(&mut self, node: TapeNode) -> TensorHandle {
        let id = self.nodes.len();
        self.nodes.push(node);
        TensorHandle(id)
    }

    /// Read-only access to a node by handle.
    pub fn get(&self, handle: TensorHandle) -> Option<&TapeNode> {
        self.nodes.get(handle.0)
    }

    /// Mutable access to a node by handle — used by the reverse
    /// walker to store accumulated gradients on the node's `grad`
    /// field.
    pub fn get_mut(&mut self, handle: TensorHandle) -> Option<&mut TapeNode> {
        self.nodes.get_mut(handle.0)
    }

    /// Number of nodes on the tape.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tape is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Runtime context threaded through every `DifferentiableTool::forward`
/// and `DifferentiableTool::backward` call. Carries the tape, the
/// execution mode, and a bag of tool-scoped state.
///
/// Day 3 refactor (per r7-day2-review.md §3.2): `tool_state` is now a
/// typed `Box<dyn Any>` map instead of `serde_json::Value`. Tools
/// serialize concrete state types without JSON round-tripping. The
/// type parameter on `set_tool_state` / `get_tool_state` ensures the
/// read side recovers the same type the write side stored.
pub struct DiffContext {
    /// The append-only tape built during the forward pass.
    pub tape: Tape,
    /// Current execution mode.
    pub mode: ExecutionMode,
    tool_state: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl std::fmt::Debug for DiffContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiffContext")
            .field("tape", &self.tape)
            .field("mode", &self.mode)
            .field(
                "tool_state_keys",
                &self.tool_state.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl DiffContext {
    /// Construct a fresh context with an empty tape in the given mode.
    pub fn new(mode: ExecutionMode) -> Self {
        Self {
            tape: Tape::new(),
            mode,
            tool_state: HashMap::new(),
        }
    }

    /// Store a typed value in the tool-state bag. Overwrites any
    /// previous value stored under the same key.
    pub fn set_tool_state<T>(&mut self, key: impl Into<String>, value: T)
    where
        T: Any + Send + Sync,
    {
        self.tool_state.insert(key.into(), Box::new(value));
    }

    /// Retrieve a typed reference from the tool-state bag. Returns
    /// `None` if the key is missing or if the stored type does not
    /// match `T`.
    pub fn get_tool_state<T>(&self, key: &str) -> Option<&T>
    where
        T: Any + Send + Sync,
    {
        self.tool_state.get(key).and_then(|b| b.downcast_ref::<T>())
    }

    /// Remove and return a typed value from the tool-state bag. Returns
    /// `None` if the key is missing or if the stored type does not
    /// match `T`.
    pub fn take_tool_state<T>(&mut self, key: &str) -> Option<T>
    where
        T: Any + Send + Sync,
    {
        let boxed = self.tool_state.remove(key)?;
        match boxed.downcast::<T>() {
            Ok(b) => Some(*b),
            Err(reinsert) => {
                // Type mismatch — put it back and return None.
                self.tool_state.insert(key.to_string(), reinsert);
                None
            }
        }
    }
}
