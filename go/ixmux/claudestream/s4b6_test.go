package claudestream

import (
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// S4B-6 (Codex round 7, P2): an init whose tools field is null or missing proves nothing: every
// delivery is refused.
func TestToolsFieldIsRequired(t *testing.T) {
	for _, mode := range []string{"null", "absent"} {
		t.Run(mode, func(t *testing.T) {
			a, e, eff := setup(t, 5*time.Second, "FAKE_TOOLS_FIELD="+mode)
			if res, err := send(t, a, e, "m1", "x"); res.State != deliver.Refused {
				t.Errorf("init with tools %s: %+v %v, want refused", mode, res, err)
			}
			if n := effects(t, eff); n != 0 {
				t.Errorf("%d turns, want 0", n)
			}
		})
	}
}

// S4B-6 (Codex round 7, P2): the Ack keeps at most MaxReply bytes of a longer answer.
func TestAckTextIsBounded(t *testing.T) {
	a, e, _ := setup(t, 5*time.Second, "FAKE_REPLY=10000")
	res, err := send(t, a, e, "m1", "x")
	if err != nil || res.Ack == nil || len(res.Ack.Text) != MaxReply {
		t.Errorf("an answer of 10000 bytes: %+v %v, want an Ack text of exactly %d bytes", res, err, MaxReply)
	}
}
