package main

import (
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// A durable entry for an earlier submission cannot validate a conflicting POST.
func TestReceiptWaitsForConflictingSubmission(t *testing.T) {
	for _, old := range []deliver.Mandate{
		{Target: "c", Incarnation: "inc-1", Text: "old text"},
		{Target: "other", Incarnation: "inc-1", Text: "new text"},
		{Target: "c", Incarnation: "old-incarnation", Text: "new text"},
	} {
		t.Run(old.Target+"/"+old.Incarnation+"/"+old.Text, func(t *testing.T) {
			state := fakeIxmux(t, func(w http.ResponseWriter, r *http.Request) {
				switch {
				case r.URL.Path == "/v1/status":
					w.Write([]byte(statusJSON))
				case r.Method == "POST":
					time.Sleep(150 * time.Millisecond)
					w.WriteHeader(http.StatusConflict)
					w.Write([]byte(`{"error":"mandate-id-reuse","detail":"different submission"}`))
				default:
					json.NewEncoder(w).Encode(deliver.Entry{Target: old.Target, MandateID: "m1", State: deliver.Delivered, Fingerprint: deliver.Fingerprint(old)})
				}
			})
			rc := newInproc(t, state).tool("ixmux_submit", map[string]any{"target": "c", "mandate_id": "m1", "text": "new text"})
			if !rc.IsError || rc.Structured["error"] != "mandate-id-reuse" {
				t.Fatalf("conflicting submission: %v, want mandate-id-reuse", rc.Structured)
			}
		})
	}
}
