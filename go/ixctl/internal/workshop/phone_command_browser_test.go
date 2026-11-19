package workshop

import (
	"encoding/json"
	"testing"
)

// TestPhoneKeyboardCommandsAreUnderstood: a phone keyboard rarely types straight double quotes. French iOS puts « »
// with no-break spaces, many keyboards put “ ”, and the keyboard's own voice typing (the PWA's local dictation is not
// offered by mobile Chrome) puts no quotes at all. Each of these must give the same intent as the documented phrase,
// in the real client, before any pairing.
func TestPhoneKeyboardCommandsAreUnderstood(t *testing.T) {
	f := newBrowserFixture(t)
	page := startChrome(t, f.chrome, f.pin)
	page.navigate(f.origin + "/workshop/")
	page.until("boot", clientReady)

	interpret := func(text, language string) map[string]string {
		t.Helper()
		var out map[string]string
		arg, _ := json.Marshal(text)
		page.evalInto(`import('/workshop/sync.js').then(m=>{try{return m.interpret(`+string(arg)+`,'`+language+`')}catch(e){return {error:e.message}}})`, &out)
		return out
	}
	list := map[string]string{"workspaceTitle": "Courses", "checklistTitle": "Épicerie"}
	for _, c := range []struct {
		text, language string
		want           map[string]string
	}{
		{`Créer un workspace "Courses" avec une liste "Épicerie"`, "fr", list},
		{"Créer un workspace « Courses » avec une liste « Épicerie »", "fr", list},
		{"Créer un workspace « Courses » avec une liste à cocher « Épicerie ».", "fr", list},
		{`Créer un workspace “Courses” avec une liste “Épicerie”`, "fr", list},
		{`créer un workspace Courses avec une liste Épicerie`, "fr", list},
		{`Créer un workspace Courses avec une liste à cocher Épicerie.`, "fr", list},
		{`Créer un workspace Maison avec un compteur Livres`, "fr", map[string]string{"workspaceTitle": "Maison", "counterTitle": "Livres"}},
		{`Create a workspace “Home” with a checklist “Groceries”`, "en", map[string]string{"workspaceTitle": "Home", "checklistTitle": "Groceries"}},
		{`create a workspace Home with a counter Books`, "en", map[string]string{"workspaceTitle": "Home", "counterTitle": "Books"}},
		{`run powershell Get-Content Scribe.db`, "en", map[string]string{"error": "command_unrecognized"}},
		{`Créer un workspace Courses avec une liste`, "fr", map[string]string{"error": "command_unrecognized"}},
		{`Créer un workspace « Courses" avec une liste "Épicerie »`, "fr", map[string]string{"error": "command_unrecognized"}},
	} {
		got := interpret(c.text, c.language)
		if len(got) != len(c.want) {
			t.Errorf("%q: got %v, want %v", c.text, got, c.want)
			continue
		}
		for k, v := range c.want {
			if got[k] != v {
				t.Errorf("%q: got %v, want %v", c.text, got, c.want)
				break
			}
		}
	}

	// The same through the page, as typed from a phone keyboard's voice typing: the sentence makes the tool it names.
	page.typeInto("new-name", "créer un workspace Courses avec une liste Épicerie")
	page.press("#new-counter")
	page.until("a checklist made from a phrase without quotes", `document.getElementById('tool-title').value==='Épicerie' && document.querySelector('#widget-host ix-checklist')!==null`)
}
