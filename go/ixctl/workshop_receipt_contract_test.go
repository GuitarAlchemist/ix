package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// Contract tests for consumers of the IxForge updater: the rollback receipt (deploy/workshop/receipt.schema.json and
// RECEIPTS.md) and the branding.json fields. They pin the documents to what the code actually does.

func TestRollbackReceiptContract(t *testing.T) {
	if _, err := exec.LookPath("pwsh"); err != nil {
		t.Skip("pwsh 7.4+ not installed")
	}
	fixtures, _ := filepath.Glob("testdata/receipts/*.json")
	if len(fixtures) < 10 {
		t.Fatalf("receipt fixtures missing: %v", fixtures)
	}
	for _, f := range fixtures {
		name := strings.TrimSuffix(filepath.Base(f), ".json")
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			os.WriteFile(filepath.Join(dir, "ixforge-receipt-fixture"), nil, 0o600)
			out, err := exec.Command("pwsh", "-NoProfile", "-NonInteractive", "-File", "testdata/receipt-contract.ps1", "-Receipt", f, "-Dir", dir).CombinedOutput()
			if err != nil {
				t.Fatalf("harness: %v\n%s", err, out)
			}
			lines := strings.Split(strings.TrimSpace(string(out)), "\n")
			var got struct {
				Admitted        bool   `json:"admitted"`
				Error           string `json:"error"`
				NeighboursReset bool   `json:"neighbours_reset"`
			}
			if err := json.Unmarshal([]byte(strings.TrimSpace(lines[len(lines)-1])), &got); err != nil {
				t.Fatalf("harness output: %v\n%s", err, out)
			}
			switch {
			case strings.HasPrefix(name, "accepted-"):
				if !got.Admitted || got.Error != "" || !got.NeighboursReset {
					t.Fatalf("receipt not admitted, or its recorded neighbours were trusted: %+v", got)
				}
			case strings.HasPrefix(name, "refused-"):
				if got.Admitted || got.Error == "" {
					t.Fatalf("receipt admitted: %+v", got)
				}
			default:
				t.Fatalf("fixture must be named accepted-* or refused-*")
			}
		})
	}
}

func TestReceiptSchemaMatchesUpdater(t *testing.T) {
	core, err := os.ReadFile("deploy/workshop/Update-IxForge.Core.ps1")
	if err != nil {
		t.Fatal(err)
	}
	var schema struct {
		Required   []string `json:"required"`
		Properties struct {
			State struct {
				Enum []string `json:"enum"`
			} `json:"State"`
		} `json:"properties"`
	}
	raw, err := os.ReadFile("deploy/workshop/receipt.schema.json")
	if err != nil || json.Unmarshal(raw, &schema) != nil {
		t.Fatalf("schema unreadable: %v", err)
	}
	list := func(re string) []string {
		m := regexp.MustCompile(re).FindSubmatch(core)
		if m == nil {
			t.Fatalf("updater no longer contains %s", re)
		}
		var out []string
		for _, s := range strings.Split(string(m[1]), ",") {
			out = append(out, strings.Trim(s, "' "))
		}
		return out
	}
	// Undo-IxForgeUpdate: identity checks, then the per-field presence loop, then the state allow-list.
	required := append([]string{"Schema", "RunID", "BackupDirectory", "CandidateSHA256", "PreviousSHA256"}, list(`foreach\(\$field in @\(([^)]*)\)\)`)...)
	states := list(`\$plan\.State -cnotin @\(([^)]*)\)`)
	slices.Sort(required)
	slices.Sort(states)
	gotRequired, gotStates := slices.Clone(schema.Required), slices.Clone(schema.Properties.State.Enum)
	slices.Sort(gotRequired)
	slices.Sort(gotStates)
	if !slices.Equal(required, gotRequired) {
		t.Fatalf("schema required %v, updater checks %v", gotRequired, required)
	}
	if !slices.Equal(states, gotStates) {
		t.Fatalf("schema states %v, updater accepts %v", gotStates, states)
	}
	// Every state the updater writes must be admitted on replay.
	for _, m := range regexp.MustCompile(`\.State='([a-z_]+)'`).FindAllSubmatch(core, -1) {
		if !slices.Contains(states, string(m[1])) {
			t.Fatalf("updater writes state %q that a rollback refuses", m[1])
		}
	}
	accepted, _ := filepath.Glob("testdata/receipts/accepted-*.json")
	for _, f := range accepted {
		var r map[string]any
		raw, _ := os.ReadFile(f)
		if err := json.Unmarshal(raw, &r); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		for _, k := range required {
			if v, ok := r[k]; !ok || v == "" || v == nil {
				t.Fatalf("%s lacks required %s", f, k)
			}
		}
		if !slices.Contains(states, r["State"].(string)) {
			t.Fatalf("%s state %v", f, r["State"])
		}
	}
}

func TestBrandingCarriesNameAndVersion(t *testing.T) {
	raw, err := os.ReadFile("internal/workshop/web/branding.json")
	if err != nil {
		t.Fatal(err)
	}
	var brand map[string]any
	if err := json.Unmarshal(raw, &brand); err != nil {
		t.Fatal(err)
	}
	keys := make([]string, 0, len(brand))
	for k := range brand {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	if !slices.Equal(keys, []string{"name", "short_name", "version"}) {
		t.Fatalf("branding.json fields changed: %v (update RECEIPTS.md)", keys)
	}
	if brand["name"] != "IxForge" {
		t.Fatalf("branding name %v", brand["name"])
	}
	if v, _ := brand["version"].(string); !regexp.MustCompile(`^[0-9][A-Za-z0-9.+-]{0,80}$`).MatchString(v) {
		t.Fatalf("branding version %q does not match the updater's client-version pattern", v)
	}
	core, _ := os.ReadFile("deploy/workshop/Update-IxForge.Core.ps1")
	for _, want := range []string{
		`($activeBrand.Body|ConvertFrom-Json).version`,                            // plan: PreviousClientVersion
		`($brand.Body|ConvertFrom-Json).version -ceq $Plan.ExpectedClientVersion`, // health: version
		`'<title>IxForge</title>'`,                                                // health: name, from the private page
	} {
		if !strings.Contains(string(core), want) {
			t.Fatalf("updater no longer reads %s (update RECEIPTS.md)", want)
		}
	}
	if strings.Contains(string(core), ").name") {
		t.Fatalf("updater now reads branding name: update RECEIPTS.md")
	}
}
