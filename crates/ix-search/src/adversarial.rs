//! Adversarial search: Minimax, Alpha-Beta pruning, Expectiminimax.

/// A game state for adversarial search.
pub trait GameState: Clone {
    type Move: Clone;

    /// Generate legal moves from this state.
    fn legal_moves(&self) -> Vec<Self::Move>;

    /// Apply a move and return the resulting state.
    fn apply_move(&self, m: &Self::Move) -> Self;

    /// Is this a terminal state?
    fn is_terminal(&self) -> bool;

    /// Whose turn is it? true = maximizer, false = minimizer.
    fn is_maximizer_turn(&self) -> bool;

    /// Utility/evaluation for terminal or leaf nodes.
    /// Positive favors maximizer, negative favors minimizer.
    fn evaluate(&self) -> f64;
}

/// Result of adversarial search.
#[derive(Debug, Clone)]
pub struct AdversarialResult<M> {
    pub best_move: Option<M>,
    pub value: f64,
    pub nodes_evaluated: usize,
}

/// Minimax search (full game tree).
pub fn minimax<S: GameState>(state: &S, depth: usize) -> AdversarialResult<S::Move> {
    let mut nodes = 0;
    let (value, best_move) = minimax_recursive(state, depth, &mut nodes);
    AdversarialResult {
        best_move,
        value,
        nodes_evaluated: nodes,
    }
}

fn minimax_recursive<S: GameState>(
    state: &S,
    depth: usize,
    nodes: &mut usize,
) -> (f64, Option<S::Move>) {
    *nodes += 1;

    if depth == 0 || state.is_terminal() {
        return (state.evaluate(), None);
    }

    let moves = state.legal_moves();
    if moves.is_empty() {
        return (state.evaluate(), None);
    }

    if state.is_maximizer_turn() {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_move = None;

        for m in moves {
            let child = state.apply_move(&m);
            let (val, _) = minimax_recursive(&child, depth - 1, nodes);
            if val > best_val {
                best_val = val;
                best_move = Some(m);
            }
        }

        (best_val, best_move)
    } else {
        let mut best_val = f64::INFINITY;
        let mut best_move = None;

        for m in moves {
            let child = state.apply_move(&m);
            let (val, _) = minimax_recursive(&child, depth - 1, nodes);
            if val < best_val {
                best_val = val;
                best_move = Some(m);
            }
        }

        (best_val, best_move)
    }
}

/// Alpha-Beta pruning search — minimax with branch elimination.
pub fn alpha_beta<S: GameState>(state: &S, depth: usize) -> AdversarialResult<S::Move> {
    let mut nodes = 0;
    let (value, best_move) =
        ab_recursive(state, depth, f64::NEG_INFINITY, f64::INFINITY, &mut nodes);
    AdversarialResult {
        best_move,
        value,
        nodes_evaluated: nodes,
    }
}

fn ab_recursive<S: GameState>(
    state: &S,
    depth: usize,
    mut alpha: f64,
    mut beta: f64,
    nodes: &mut usize,
) -> (f64, Option<S::Move>) {
    *nodes += 1;

    if depth == 0 || state.is_terminal() {
        return (state.evaluate(), None);
    }

    let moves = state.legal_moves();
    if moves.is_empty() {
        return (state.evaluate(), None);
    }

    if state.is_maximizer_turn() {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_move = None;

        for m in moves {
            let child = state.apply_move(&m);
            let (val, _) = ab_recursive(&child, depth - 1, alpha, beta, nodes);
            if val > best_val {
                best_val = val;
                best_move = Some(m);
            }
            alpha = alpha.max(val);
            if alpha >= beta {
                break; // Beta cutoff
            }
        }

        (best_val, best_move)
    } else {
        let mut best_val = f64::INFINITY;
        let mut best_move = None;

        for m in moves {
            let child = state.apply_move(&m);
            let (val, _) = ab_recursive(&child, depth - 1, alpha, beta, nodes);
            if val < best_val {
                best_val = val;
                best_move = Some(m);
            }
            beta = beta.min(val);
            if alpha >= beta {
                break; // Alpha cutoff
            }
        }

        (best_val, best_move)
    }
}

/// Negamax with alpha-beta — simplified implementation using negation.
pub fn negamax<S: GameState>(state: &S, depth: usize) -> AdversarialResult<S::Move> {
    let mut nodes = 0;
    let color = if state.is_maximizer_turn() { 1.0 } else { -1.0 };
    let (value, best_move) = negamax_recursive(
        state,
        depth,
        f64::NEG_INFINITY,
        f64::INFINITY,
        color,
        &mut nodes,
    );
    AdversarialResult {
        best_move,
        value: value * color,
        nodes_evaluated: nodes,
    }
}

fn negamax_recursive<S: GameState>(
    state: &S,
    depth: usize,
    mut alpha: f64,
    beta: f64,
    color: f64,
    nodes: &mut usize,
) -> (f64, Option<S::Move>) {
    *nodes += 1;

    if depth == 0 || state.is_terminal() {
        return (color * state.evaluate(), None);
    }

    let moves = state.legal_moves();
    if moves.is_empty() {
        return (color * state.evaluate(), None);
    }

    let mut best_val = f64::NEG_INFINITY;
    let mut best_move = None;

    for m in moves {
        let child = state.apply_move(&m);
        let (val, _) = negamax_recursive(&child, depth - 1, -beta, -alpha, -color, nodes);
        let val = -val;

        if val > best_val {
            best_val = val;
            best_move = Some(m);
        }
        alpha = alpha.max(val);
        if alpha >= beta {
            break;
        }
    }

    (best_val, best_move)
}

/// Expectiminimax — minimax for games with chance nodes.
///
/// Requires `chance_outcomes` returning (probability, resulting_state) pairs.
pub trait StochasticGameState: GameState {
    /// Is this a chance node?
    fn is_chance_node(&self) -> bool;

    /// Chance outcomes: (probability, resulting_state).
    fn chance_outcomes(&self) -> Vec<(f64, Self)>;
}

pub fn expectiminimax<S: StochasticGameState>(
    state: &S,
    depth: usize,
) -> AdversarialResult<S::Move> {
    let mut nodes = 0;
    let (value, best_move) = expecti_recursive(state, depth, &mut nodes);
    AdversarialResult {
        best_move,
        value,
        nodes_evaluated: nodes,
    }
}

fn expecti_recursive<S: StochasticGameState>(
    state: &S,
    depth: usize,
    nodes: &mut usize,
) -> (f64, Option<S::Move>) {
    *nodes += 1;

    if depth == 0 || state.is_terminal() {
        return (state.evaluate(), None);
    }

    if state.is_chance_node() {
        let outcomes = state.chance_outcomes();
        let expected: f64 = outcomes
            .iter()
            .map(|(prob, child)| {
                let (val, _) = expecti_recursive(child, depth - 1, nodes);
                prob * val
            })
            .sum();
        return (expected, None);
    }

    let moves = state.legal_moves();
    if moves.is_empty() {
        return (state.evaluate(), None);
    }

    if state.is_maximizer_turn() {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_move = None;
        for m in moves {
            let child = state.apply_move(&m);
            let (val, _) = expecti_recursive(&child, depth - 1, nodes);
            if val > best_val {
                best_val = val;
                best_move = Some(m);
            }
        }
        (best_val, best_move)
    } else {
        let mut best_val = f64::INFINITY;
        let mut best_move = None;
        for m in moves {
            let child = state.apply_move(&m);
            let (val, _) = expecti_recursive(&child, depth - 1, nodes);
            if val < best_val {
                best_val = val;
                best_move = Some(m);
            }
        }
        (best_val, best_move)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple number game: pick +1 or -1, goal is to reach 0 or 10.
    #[derive(Clone, Debug)]
    struct NumberGame {
        value: i32,
        max_turn: bool,
        turns_left: usize,
    }

    impl GameState for NumberGame {
        type Move = i32;

        fn legal_moves(&self) -> Vec<i32> {
            if self.is_terminal() {
                vec![]
            } else {
                vec![1, -1]
            }
        }

        fn apply_move(&self, m: &i32) -> Self {
            NumberGame {
                value: self.value + m,
                max_turn: !self.max_turn,
                turns_left: self.turns_left - 1,
            }
        }

        fn is_terminal(&self) -> bool {
            self.turns_left == 0 || self.value <= 0 || self.value >= 10
        }

        fn is_maximizer_turn(&self) -> bool {
            self.max_turn
        }

        fn evaluate(&self) -> f64 {
            self.value as f64
        }
    }

    #[test]
    fn test_minimax_basic() {
        let state = NumberGame {
            value: 5,
            max_turn: true,
            turns_left: 4,
        };
        let result = minimax(&state, 4);
        assert!(result.best_move.is_some());
        // Maximizer should try to increase value
        assert_eq!(result.best_move.unwrap(), 1);
    }

    #[test]
    fn test_alpha_beta_prunes() {
        let state = NumberGame {
            value: 5,
            max_turn: true,
            turns_left: 6,
        };

        let mm = minimax(&state, 6);
        let ab = alpha_beta(&state, 6);

        // Same result
        assert_eq!(mm.value, ab.value);
        // Alpha-beta should evaluate fewer nodes
        assert!(
            ab.nodes_evaluated <= mm.nodes_evaluated,
            "AB: {}, MM: {}",
            ab.nodes_evaluated,
            mm.nodes_evaluated
        );
    }
}
