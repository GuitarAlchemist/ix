package server

import (
	"encoding/json"
	"sync"
)

// Message is one message of the stream. Events (mandate, journal, journal-reset) are numbered
// by Seq, 1, 2, … within an Epoch, which changes at every start of ixmux. hello, gap and state
// are not events: their Seq is the last event they cover, and a client resumes with
// since=<Seq of the last event it has>&epoch=<Epoch>.
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
// connection's bounded queue without ever blocking the publisher.
type hub struct {
	mu    sync.Mutex
	epoch string
	seq   uint64
	ring  []ringEntry
	size  int
	subs  map[*sub]struct{}
}

type sub struct {
	q    chan []byte
	over chan struct{} // closed when q overflowed; the sub is then out of the hub
	last uint64        // the last event seq put in q (under hub.mu)
}

func newHub(epoch string, ringSize int) *hub {
	return &hub{epoch: epoch, size: ringSize, subs: map[*sub]struct{}{}}
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
// sent first: hello, then either the events after since kept in the ring, or a gap and the
// state. Taken under the hub's lock, so no event falls between the replay and the queue.
func (h *hub) subscribe(resume bool, since uint64, epoch string, queue int, state func(epoch string, seq uint64) any) (*sub, [][]byte) {
	h.mu.Lock()
	defer h.mu.Unlock()
	s := &sub{q: make(chan []byte, queue), over: make(chan struct{}), last: h.seq}
	first := [][]byte{h.frame("hello", h.seq, nil)}
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
			first = append(first, h.frame("gap", h.seq, Gap{Reason: reason}), h.frame("state", h.seq, state(h.epoch, h.seq)))
		} else {
			for _, e := range h.ring {
				if e.seq > since {
					first = append(first, e.frame)
				}
			}
		}
	}
	h.subs[s] = struct{}{}
	return s, first
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
