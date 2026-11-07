package claudestream

import (
	"fmt"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// S4B-2 (Codex round 7, P1): a new system/init arrives between the check of the child and the
// write on its stdin, with tools or for another session: the mandate is refused, nothing written.
func TestInitBetweenCheckAndWrite(t *testing.T) {
	for _, tc := range []struct{ name, init string }{
		{"with tools", `{"type":"system","subtype":"init","session_id":"%s","tools":["Bash"]}`},
		{"another session", `{"type":"system","subtype":"init","session_id":"another-%s","tools":[]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, e, eff := setup(t, 5*time.Second)
			sid := a.sessionID()
			testHookAfterCheck = func(route func([]byte)) {
				testHookAfterCheck = nil
				route([]byte(fmt.Sprintf(tc.init, sid) + "\n"))
			}
			t.Cleanup(func() { testHookAfterCheck = nil })
			res, err := send(t, a, e, "m1", "x")
			if res.State != deliver.Refused {
				t.Errorf("an init %s between the check and the write: %+v %v, want refused", tc.name, res, err)
			}
			if n := effects(t, eff); n != 0 {
				t.Errorf("%d messages written, want 0", n)
			}
		})
	}
}
