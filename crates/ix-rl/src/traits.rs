//! Core RL traits.

/// An environment the agent interacts with.
pub trait Environment {
    type State: Clone;
    type Action: Clone;

    /// Reset to initial state.
    fn reset(&mut self) -> Self::State;
    /// Take an action, return (next_state, reward, done).
    fn step(&mut self, action: &Self::Action) -> (Self::State, f64, bool);
    /// Available actions from current state.
    fn actions(&self) -> Vec<Self::Action>;
}

/// An RL agent.
pub trait Agent<E: Environment> {
    fn select_action(&self, state: &E::State) -> E::Action;
    fn update(
        &mut self,
        state: &E::State,
        action: &E::Action,
        reward: f64,
        next_state: &E::State,
        done: bool,
    );
}
