package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// The ecosystem manifest: GuitarAlchemist's dev-data endpoint, rendered as Markdown into
// .claude/manifest-bootstrap.md at session start. This is scripts/manifest-sync.ps1, which the
// bash hook ran through pwsh; the rendering is the script's, line for line.
const (
	manifestURL      = "https://demos.guitaralchemist.com/dev-data/manifest"
	manifestLocalURL = "http://localhost:5176/dev-data/manifest"
)

// syncManifest fetches the live manifest (10 s), else the local Vite dev server's (5 s), and
// rewrites path. When both fail the file is left as it was, as the script left it.
func (e *env) syncManifest(path string) {
	raw, err := e.sys.fetch(manifestURL, 10*time.Second)
	if err != nil {
		raw, err = e.sys.fetch(manifestLocalURL, 5*time.Second)
		if err != nil {
			return
		}
	}
	md, ok := renderManifest(raw, e.sys.now())
	if !ok {
		return
	}
	os.MkdirAll(filepath.Dir(path), 0o755)
	os.WriteFile(path, []byte(md+"\n"), 0o644)
}

func renderManifest(raw []byte, now time.Time) (string, bool) {
	var m map[string]any
	if json.Unmarshal(raw, &m) != nil || len(m) == 0 {
		return "", false
	}
	quality, _ := m["quality"].(map[string]any)
	var b strings.Builder
	fmt.Fprintf(&b, "# Ecosystem Manifest Bootstrap\n\n**Ecosystem:** GuitarAlchemist\n**Source Repository:** %s\n**Fetched At:** %s\n**Manifest Generation Time:** %s\n\n---\n\n## 🚦 System Health & Quality Scorecard\n",
		ps(m["repo"]), now.Format("2006-01-02 15:04:05"), ps(m["generated_at"]))

	if domains, ok := quality["domains"].(map[string]any); ok && len(quality) > 0 {
		for _, name := range domainOrder(raw, domains) {
			domain, _ := domains[name].(map[string]any)
			data, _ := domain["data"].(map[string]any)
			status := "🔴 DEGRADED"
			if ps(data["oracle_status"]) == "ok" {
				status = "🟢 OK"
			}
			fmt.Fprintf(&b, "### %s (%s)\n- **Source:** %s\n", name, status, ps(domain["source"]))
			if v, ok := data["metric_value"]; ok && v != nil {
				fmt.Fprintf(&b, "- **Metric Value:** %s\n", ps(v))
			}
			f, w := data["totalFailures"], data["totalWarnings"]
			if f != nil || w != nil {
				fmt.Fprintf(&b, "- **Failures/Warnings:** %s fail(s), %s warn(s)\n", psOr0(f), psOr0(w))
			}
			if s := data["summary"]; truthy(s) {
				fmt.Fprintf(&b, "- **Summary:** %s\n", ps(s))
			}
			b.WriteString("\n")
		}
	}

	regressions := items(quality["regressions"])
	if len(regressions) > 0 {
		b.WriteString("### ⚠️ ACTIVE REGRESSIONS\n")
		for _, r := range regressions {
			fmt.Fprintf(&b, "- %s\n", ps(r))
		}
		b.WriteString("\n")
	} else {
		b.WriteString("### 🟢 Regressions: None detected\n\n")
	}

	b.WriteString("## 🌐 Active Services & Dev Ports\n\n| Service | Port | Public Path | Expected Behavior |\n|---|---|---|---|")
	if truthy(m["services"]) {
		for _, s := range items(m["services"]) {
			sv, _ := s.(map[string]any)
			fmt.Fprintf(&b, "\n| %s | %s | %s | %s |", ps(sv["name"]), ps(sv["port"]), ps(sv["public_path"]), ps(sv["expected"]))
		}
	}
	b.WriteString("\n\n")

	if backlog, ok := m["backlog"].(map[string]any); ok && truthy(m["backlog"]) {
		fmt.Fprintf(&b, "## 📋 Project Backlog Progress\n\n**Overall Progress:** %s%% Shipped (%s of %s items across %s epics)\n\n| Epic | Shipped | Active | Backlog | Progress |\n|---|---|---|---|---|",
			ps(backlog["overall_progress_pct"]), ps(backlog["total_shipped"]), ps(backlog["total_items"]), ps(backlog["total_epics"]))
		if truthy(backlog["epics"]) {
			for _, ep := range items(backlog["epics"]) {
				em, _ := ep.(map[string]any)
				fmt.Fprintf(&b, "\n| %s | %s | %s | %s | %s%% |", ps(em["title"]), ps(em["shipped"]), ps(em["active"]), ps(em["backlog"]), ps(em["progress_pct"]))
			}
		}
	}
	b.WriteString("\n\n")

	b.WriteString("## 🕒 Recent Commit Activity\n\n| Commit | Author | Date | Subject |\n|---|---|---|---|")
	if truthy(m["activity"]) {
		for _, a := range items(m["activity"]) {
			am, _ := a.(map[string]any)
			fmt.Fprintf(&b, "\n| %s | %s | %s | %s |", ps(am["short_sha"]), ps(am["author"]), ps(am["date"]), ps(am["subject"]))
		}
	}
	return b.String(), true
}

// domainOrder is quality.domains' keys in the JSON's order, as PowerShell enumerates them.
func domainOrder(raw []byte, domains map[string]any) []string {
	var doc struct {
		Quality struct {
			Domains json.RawMessage `json:"domains"`
		} `json:"quality"`
	}
	var keys []string
	if json.Unmarshal(raw, &doc) == nil {
		dec := json.NewDecoder(bytes.NewReader(doc.Quality.Domains))
		if t, err := dec.Token(); err == nil && t == json.Delim('{') {
			for dec.More() {
				t, err := dec.Token()
				if err != nil {
					break
				}
				k, _ := t.(string)
				keys = append(keys, k)
				var skip json.RawMessage
				if dec.Decode(&skip) != nil {
					break
				}
			}
		}
	}
	if len(keys) != len(domains) {
		keys = keys[:0]
		for k := range domains {
			keys = append(keys, k)
		}
		sort.Strings(keys)
	}
	return keys
}

// ps renders a JSON value the way PowerShell interpolates it into a string.
func ps(v any) string {
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return x
	case bool:
		if x {
			return "True"
		}
		return "False"
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	case []any:
		parts := make([]string, len(x))
		for i, it := range x {
			parts[i] = ps(it)
		}
		return strings.Join(parts, " ")
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		parts := make([]string, len(keys))
		for i, k := range keys {
			parts[i] = k + "=" + ps(x[k])
		}
		return "@{" + strings.Join(parts, "; ") + "}"
	}
	return fmt.Sprint(v)
}

func psOr0(v any) string {
	if v == nil {
		return "0"
	}
	return ps(v)
}

// truthy is PowerShell's `if ($x)`.
func truthy(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case string:
		return x != ""
	case bool:
		return x
	case float64:
		return x != 0
	case []any:
		return len(x) > 0
	}
	return true
}

// items is what `foreach ($x in $v)` visits: the elements of an array, a scalar once, nothing for null.
func items(v any) []any {
	switch x := v.(type) {
	case nil:
		return nil
	case []any:
		return x
	}
	return []any{v}
}
