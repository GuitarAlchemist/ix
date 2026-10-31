//! #485 check 1, in-memory private snapshot: a `ReadContext` projected at ordinal n is not
//! changed by an event emitted at n + 1; only a new projection sees it.
//!
//! A pin, green from the start: `ReadContext` is owned data with no interior mutability and
//! `WriteContext` holds it by shared reference, so the type system is the real guard. The test
//! records the behaviour #485 asks about, at the seam a task would receive.

use ix_agent_core::{
    project_beliefs, EventSink, ReadContext, SessionEvent, VecEventSink, WriteContext,
};
use ix_types::Hexavalent;
use serde_json::json;

fn belief(ordinal: u64, old: Option<Hexavalent>, new: Hexavalent) -> SessionEvent {
    SessionEvent::BeliefChanged {
        ordinal,
        proposition: "tool:a:will_succeed".into(),
        old,
        new,
        evidence: json!({}),
    }
}

#[test]
fn test_snapshot_at_n_is_unchanged_by_an_event_at_n_plus_one() {
    let mut log = VecEventSink::default();
    log.emit(belief(0, None, Hexavalent::True));

    let mut cx = ReadContext::synthetic_for_legacy();
    cx.ordinal = 0;
    cx.beliefs = project_beliefs(&log.events);
    let before = cx.clone();

    {
        let wc = WriteContext {
            read: &cx,
            sink: &mut log,
        };
        wc.sink
            .emit(belief(1, Some(Hexavalent::True), Hexavalent::False));
    }

    assert_eq!(cx, before, "the snapshot a task holds must not move");
    assert_eq!(cx.beliefs["tool:a:will_succeed"], Hexavalent::True);
    assert_eq!(
        project_beliefs(&log.events)["tool:a:will_succeed"],
        Hexavalent::False,
        "the event is in the log: a new projection at n + 1 sees it"
    );
}
