package workshop

import (
	"errors"
	"strings"
	"testing"
)

func checklistDraft() Project {
	return Project{ID: "project-list", Title: "Courses", Workspaces: []Workspace{{ID: "workspace-list", Title: "Courses",
		Widgets:          []Widget{{ID: "widget-list", Title: "Épicerie", Kind: "checklist"}},
		IXLETDefinitions: []IXLETDefinition{{ID: "ixlet-list", Title: "Épicerie", Kind: "checklist", Status: "draft"}}}}}
}

func listAct(t *testing.T, s *Store, token, op, action, text, item string) XletState {
	t.Helper()
	state, err := s.Act(token, XletAction{OpID: op, ProjectID: "project-list", DefinitionID: "ixlet-list", Action: action, Text: text, ItemID: item})
	if err != nil {
		t.Fatalf("%s %s: %v", op, action, err)
	}
	return state
}

func items(x XletState) string {
	var parts []string
	for _, i := range x.Items {
		mark := " "
		if i.Done {
			mark = "x"
		}
		parts = append(parts, "["+mark+"] "+i.Text)
	}
	return strings.Join(parts, ", ")
}

func TestChecklistIXLETAddsChecksAndRemovesItemsOnTheDesktop(t *testing.T) {
	s, token, _ := fixture(t)
	if _, err := s.Save(token, operation("save", 0, checklistDraft())); err != nil {
		t.Fatalf("a checklist widget and IXLET definition must be a valid document: %v", err)
	}
	if got := listAct(t, s, token, "on", "activate", "", ""); got.Kind != "checklist" || len(got.Items) != 0 || got.Version != 1 {
		t.Fatalf("activation creates an empty checklist: %+v", got)
	}
	milk := listAct(t, s, token, "a1", "add", "Lait", "").Items[0].ID
	listAct(t, s, token, "a2", "add", "Pain", "")
	eggs := listAct(t, s, token, "a3", "add", "Œufs", "")
	if items(eggs) != "[ ] Lait, [ ] Pain, [ ] Œufs" || eggs.Version != 4 {
		t.Fatalf("items keep their order: %q v%d", items(eggs), eggs.Version)
	}
	bread, egg := eggs.Items[1].ID, eggs.Items[2].ID
	listAct(t, s, token, "c1", "check", "", bread)
	if got := listAct(t, s, token, "r1", "remove", "", egg); items(got) != "[ ] Lait, [x] Pain" {
		t.Fatalf("check then remove: %q", items(got))
	}
	if got := listAct(t, s, token, "u1", "uncheck", "", bread); items(got) != "[ ] Lait, [ ] Pain" {
		t.Fatalf("uncheck: %q", items(got))
	}

	// Exactly once per op_id, as for the counter.
	if got := listAct(t, s, token, "a1", "add", "Lait", ""); len(got.Items) != 1 || got.Version != 2 {
		t.Fatalf("a repeated add must replay its first result, not add again: %+v", got)
	}
	if _, err := s.Act(token, XletAction{OpID: "a1", ProjectID: "project-list", DefinitionID: "ixlet-list", Action: "add", Text: "Beurre"}); !errors.Is(err, ErrOpReuse) {
		t.Fatalf("a reused op_id with another text must be refused: %v", err)
	}

	for name, bad := range map[string]XletAction{
		"empty text":           {OpID: "b1", Action: "add", Text: "   "},
		"too long":             {OpID: "b2", Action: "add", Text: strings.Repeat("x", 241)},
		"text on check":        {OpID: "b3", Action: "check", ItemID: milk, Text: "Lait"},
		"item on add":          {OpID: "b4", Action: "add", Text: "Thé", ItemID: milk},
		"counter on a list":    {OpID: "b5", Action: "increment"},
		"missing item id":      {OpID: "b6", Action: "check"},
		"malformed item id":    {OpID: "b7", Action: "remove", ItemID: "../x"},
		"text on a counter op": {OpID: "b8", Action: "activate", Text: "x"},
	} {
		bad.ProjectID, bad.DefinitionID = "project-list", "ixlet-list"
		if _, err := s.Act(token, bad); !errors.Is(err, ErrInvalid) {
			t.Errorf("%s must be refused as invalid: %v", name, err)
		}
	}
	if _, err := s.Act(token, XletAction{OpID: "gone", ProjectID: "project-list", DefinitionID: "ixlet-list", Action: "check", ItemID: egg}); !errors.Is(err, ErrNoItem) {
		t.Fatalf("a removed item cannot be checked: %v", err)
	}

	// Shared by a second device and durable across a Desktop restart.
	other, _ := paired(t, s)
	listAct(t, s, other, "o1", "check", "", milk)
	path := s.path
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	s2, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()
	states, err := s2.Xlets(token)
	if err != nil || len(states) != 1 || items(states[0]) != "[x] Lait, [ ] Pain" {
		t.Fatalf("after restart the first device must read the second device's check: %+v %v", states, err)
	}
}

func TestChecklistSizeIsBounded(t *testing.T) {
	s, token, _ := fixture(t)
	if _, err := s.Save(token, operation("save", 0, checklistDraft())); err != nil {
		t.Fatal(err)
	}
	listAct(t, s, token, "on", "activate", "", "")
	for i := 0; i < maxChecklistItems; i++ {
		listAct(t, s, token, "add-"+strings.Repeat("i", 1)+string(rune('a'+i%26))+strings.Repeat("z", i/26), "add", "élément", "")
	}
	if _, err := s.Act(token, XletAction{OpID: "one-more", ProjectID: "project-list", DefinitionID: "ixlet-list", Action: "add", Text: "de trop"}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("a full checklist must refuse another item: %v", err)
	}
}

func TestDocumentKindsStayClosed(t *testing.T) {
	s, token, _ := fixture(t)
	p := checklistDraft()
	p.Workspaces[0].IXLETDefinitions[0].Kind = "shell"
	if _, err := s.Save(token, operation("bad-kind", 0, p)); !errors.Is(err, ErrInvalid) {
		t.Fatalf("only counter and checklist kinds are accepted: %v", err)
	}
	p = checklistDraft()
	p.Workspaces[0].Widgets[0].InitialValue = 3
	if _, err := s.Save(token, operation("bad-initial", 0, p)); !errors.Is(err, ErrInvalid) {
		t.Fatalf("a checklist widget has no initial value: %v", err)
	}
}
