package server

import (
	"encoding/json"
	"sort"
	"sync"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// Message is one message of the stream. Events (mandate, journal, journal-reset) are numbered
// by Seq, 1, 2, … within an Epoch, which changes at every start of ixmux. hello, gap and state
// are not events: their Seq is the last event they cover, and a client resumes with
// since=<Seq of the last event it has>&epoch=<Epoch>. After a gap, the state comes as state-page
// messages, each within MaxStreamMessage, then one state-end, all with the Seq they cover: the
// next event is Seq+1, and none of the transitions they hold comes again as an event.
type Message struct {
	Type  string `json:"type"`
	Epoch string `json:"epoch"`
	Seq   uint64 `json:"seq"`
	Data  any    `json:"data,omitempty"`
}

// Gap says the client missed events: reason epoch (another start), ring (the events are no
// longer kept), ahead (since is past the last event), or slow (its queue overflowed: the
// connection is then closed; resume with since=LastSeq).
type Gap struct {
	Reason  string `json:"reason"`
	LastSeq uint64 `json:"last_seq,omitempty"`
}

type ringEntry struct {
	seq   uint64
	frame []byte
}

// hub numbers the events, keeps the last RingSize of them, and fans them out to every
// connection's bounded queue without ever blocking the publisher. It also keeps the last
// transition of every delivery, updated under the lock that numbers it: a snapshot taken under
// that lock covers exactly the events up to its seq.
type hub struct {
	mu    sync.Mutex
	epoch string
	seq   uint64
	ring  []ringEntry
	size  int
	subs  map[*sub]struct{}
	view  map[[2]string]deliver.Transition // by target and mandate id
}

// snapshot is the hub's deliveries as of event seq.
type snapshot struct {
	seq        uint64
	deliveries []deliver.Transition
}

type sub struct {
	q    chan []byte
	over chan struct{} // closed when q overflowed; the sub is then out of the hub
	last uint64        // the last event seq put in q (under hub.mu)
}

func newHub(epoch string, ringSize int) *hub {
	return &hub{epoch: epoch, size: ringSize, subs: map[*sub]struct{}{}, view: map[[2]string]deliver.Transition{}}
}

// seed records the deliveries the engine replayed from its log, before any event.
func (h *hub) seed(entries []deliver.Entry) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, e := range entries {
		k := [2]string{e.Target, e.MandateID}
		if _, ok := h.view[k]; !ok {
			h.view[k] = deliver.Transition{At: e.Since, Target: e.Target, MandateID: e.MandateID, State: e.State, Reason: e.Reason}
		}
	}
}

func (h *hub) frame(typ string, seq uint64, data any) []byte {
	b, _ := json.Marshal(Message{Type: typ, Epoch: h.epoch, Seq: seq, Data: data})
	return b
}

// publish numbers one event and offers it to every queue. A full queue is not waited for: its
// connection is marked slow and dropped from the hub. It returns false, publishing nothing, if
// the message would exceed MaxStreamMessage.
// @ai:invariant publishing never blocks on a connection: a full queue drops its connection [T:test conf:0.8 src:go/ixmux/cmd/ixmux/integration_test.go:TestBounds]
func (h *hub) publish(typ string, data any) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	b := h.frame(typ, h.seq+1, data)
	if len(b) > MaxStreamMessage {
		return false
	}
	h.seq++
	if tr, ok := data.(deliver.Transition); ok && typ == "mandate" {
		h.view[[2]string{tr.Target, tr.MandateID}] = tr
	}
	h.ring = append(h.ring, ringEntry{h.seq, b})
	if len(h.ring) > h.size {
		h.ring = h.ring[len(h.ring)-h.size:]
	}
	for s := range h.subs {
		select {
		case s.q <- b:
			s.last = h.seq
		default:
			close(s.over)
			delete(h.subs, s)
		}
	}
	return true
}

// subscribe registers a connection with a queue of queue messages and returns what it must be
// sent first: hello, then either the events after since kept in the ring, or a gap, and then the
// snapshot (non-nil) to send as state pages. Taken under the hub's lock, so no event falls
// between the replay or the snapshot and the queue; the snapshot is only copied here, and its
// pages are built outside the lock.
// @ai:invariant the state after a gap covers exactly the events up to its seq: none is lost or repeated by the queue [T:test conf:0.85 src:go/ixmux/server/stream_test.go:TestSnapshotAndStreamDoNotOverlap]
func (h *hub) subscribe(resume bool, since uint64, epoch string, queue int) (*sub, [][]byte, *snapshot) {
	h.mu.Lock()
	defer h.mu.Unlock()
	s := &sub{q: make(chan []byte, queue), over: make(chan struct{}), last: h.seq}
	first := [][]byte{h.frame("hello", h.seq, nil)}
	var snap *snapshot
	if resume {
		oldest := h.seq + 1
		if len(h.ring) > 0 {
			oldest = h.ring[0].seq
		}
		reason := ""
		switch {
		case epoch != h.epoch:
			reason = "epoch"
		case since > h.seq:
			reason = "ahead"
		case since+1 < oldest:
			reason = "ring"
		}
		if reason != "" {
			first = append(first, h.frame("gap", h.seq, Gap{Reason: reason}))
			snap = &snapshot{seq: h.seq, deliveries: make([]deliver.Transition, 0, len(h.view))}
			for _, tr := range h.view {
				snap.deliveries = append(snap.deliveries, tr)
			}
		} else {
			for _, e := range h.ring {
				if e.seq > since {
					first = append(first, e.frame)
				}
			}
		}
	}
	h.subs[s] = struct{}{}
	return s, first, snap
}

// MaxReason bounds the reason of one delivery in a state page, so one page always fits.
const MaxReason = 1 << 10

// pages turns a snapshot into state-page frames of at most StatePageBytes of deliveries each,
// then a state-end frame carrying end, all with the snapshot's seq.
func (h *hub) pages(snap *snapshot, end func(pages, deliveries int) any) [][]byte {
	sort.Slice(snap.deliveries, func(i, j int) bool {
		a, b := snap.deliveries[i], snap.deliveries[j]
		if a.Target != b.Target {
			return a.Target < b.Target
		}
		return a.MandateID < b.MandateID
	})
	var out [][]byte
	var page []json.RawMessage
	size := 0
	flush := func() {
		b, _ := json.Marshal(map[string][]json.RawMessage{"deliveries": page})
		out = append(out, h.frame("state-page", snap.seq, json.RawMessage(b)))
		page, size = nil, 0
	}
	for _, tr := range snap.deliveries {
		if len(tr.Reason) > MaxReason {
			tr.Reason = tr.Reason[:MaxReason] + "…"
		}
		b, _ := json.Marshal(tr)
		if size+len(b) > StatePageBytes && len(page) > 0 {
			flush()
		}
		page, size = append(page, b), size+len(b)+1
	}
	if len(page) > 0 {
		flush()
	}
	return append(out, h.frame("state-end", snap.seq, end(len(out), len(snap.deliveries))))
}

func (h *hub) unsubscribe(s *sub) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.subs, s)
}

func (h *hub) current() (string, uint64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.epoch, h.seq
}
