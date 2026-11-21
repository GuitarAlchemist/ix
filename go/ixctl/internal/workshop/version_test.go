package workshop

import (
	"encoding/json"
	"testing"
)

// TestEveryDeviceLearnsTheDesktopVersion: the version the Desktop runs comes with /me, to every approved device (not
// only admin ones), so the app can say which Desktop it talks to. A Desktop given no version says none.
func TestEveryDeviceLearnsTheDesktopVersion(t *testing.T) {
	s, tablet, _ := fixture(t)
	for _, c := range []struct {
		opts []Option
		want any
	}{{[]Option{WithVersion("0.1.0-ixforge.18")}, "0.1.0-ixforge.18"}, {nil, nil}} {
		h, err := Handler(s, "https://workshop.invalid", c.opts...)
		if err != nil {
			t.Fatal(err)
		}
		w := request(t, h, "GET", "/workshop/api/me", tablet, "")
		var me map[string]any
		if err := json.Unmarshal(w.Body.Bytes(), &me); err != nil || w.Code != 200 || me["version"] != c.want {
			t.Errorf("version %v: %d %v", c.want, w.Code, me)
		}
	}
}
