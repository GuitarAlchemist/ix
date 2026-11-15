package workshop

import (
	"encoding/json"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

type terminalReply struct {
	Output    string `json:"output"`
	ExitCode  int    `json:"exit_code"`
	TimedOut  bool   `json:"timed_out"`
	Truncated bool   `json:"truncated"`
}

// TestTheTerminalRunsOnlyWhereTheOwnerAllowedIt: the terminal runs commands on the Desktop's PC for the PC the owner
// linked from the owner page, and for no other device unless the owner allows it there, device by device. Commands run
// in the terminal's own folder, as the service's account, one at a time, within a time limit and an output limit.
// Each command is journaled (who, what, how it ended), never its output.
func TestTheTerminalRunsOnlyWhereTheOwnerAllowedIt(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the terminal runs Windows PowerShell")
	}
	dir := t.TempDir()
	s, err := Open(filepath.Join(dir, "terminal.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	// Ordinary commands get the real limit: a cold Windows PowerShell start on a loaded CI runner took more than 4 s.
	// The time-limit case below lowers it to 4 s for itself.
	s.terminalTimeout = defaultTerminalLimit
	const app = "https://workshop.invalid"
	h, err := Handler(s, app)
	if err != nil {
		t.Fatal(err)
	}
	link, err := s.NewLinkInvite(app)
	if err != nil {
		t.Fatal(err)
	}
	_, pcToken, err := s.Claim(link.Code, "PC")
	if err != nil {
		t.Fatal(err)
	}
	tabletToken, tabletID := paired(t, s)
	run := func(token, command string) (int, terminalReply, string) {
		t.Helper()
		body, _ := json.Marshal(map[string]string{"command": command})
		w := request(t, h, "POST", "/workshop/api/terminal", token, string(body))
		var reply terminalReply
		_ = json.Unmarshal(w.Body.Bytes(), &reply)
		return w.Code, reply, w.Body.String()
	}

	var me struct{ Terminal bool }
	if w := request(t, h, "GET", "/workshop/api/me", pcToken, ""); json.Unmarshal(w.Body.Bytes(), &me) != nil || !me.Terminal {
		t.Fatalf("the linked PC must know it has the terminal: %s", w.Body)
	}
	if w := request(t, h, "GET", "/workshop/api/me", tabletToken, ""); json.Unmarshal(w.Body.Bytes(), &me) != nil || me.Terminal {
		t.Fatalf("another device has no terminal: %s", w.Body)
	}
	if code, _, body := run(tabletToken, "Write-Output 1"); code != 403 || !strings.Contains(body, "terminal_not_allowed") {
		t.Fatalf("another device must be refused: %d %s", code, body)
	}
	if code, _, _ := run("", "Write-Output 1"); code != 401 {
		t.Fatalf("an anonymous request must be refused: %d", code)
	}
	// Administering devices is not running commands.
	owner := newOwnerFace(t, s, app)
	session := owner.session()
	if code := owner.do("POST", "/workshop-owner/api/devices/"+tabletID+"/admin", "{}", session, nil); code != 200 {
		t.Fatalf("grant admin: %d", code)
	}
	if code, _, _ := run(tabletToken, "Write-Output 1"); code != 403 {
		t.Fatalf("an administering device must not get the terminal with it: %d", code)
	}

	if code, reply, body := run(pcToken, "Write-Output ('bon'+'jour é'); exit 3"); code != 200 || strings.TrimSpace(reply.Output) != "bonjour é" || reply.ExitCode != 3 || reply.TimedOut {
		t.Fatalf("the PC runs a command and sees its output and exit code: %d %s", code, body)
	}
	if code, reply, body := run(pcToken, "(Get-Location).Path"); code != 200 || !strings.EqualFold(strings.TrimSpace(reply.Output), filepath.Join(dir, "terminal")) {
		t.Fatalf("commands run in the terminal's own folder: %d %s", code, body)
	}
	s.terminalTimeout = 4 * time.Second
	started := time.Now()
	if code, reply, body := run(pcToken, "Start-Sleep -Seconds 60"); code != 200 || !reply.TimedOut || reply.ExitCode != -1 || time.Since(started) > 15*time.Second {
		t.Fatalf("a command that runs too long is stopped: %d %s after %s", code, body, time.Since(started))
	}
	if code, reply, body := run(pcToken, "'x' * 400000"); code != 200 || !reply.Truncated || len(reply.Output) > maxTerminalOutput {
		t.Fatalf("the output is capped: %d truncated=%v len=%d %.80s", code, reply.Truncated, len(reply.Output), body)
	}
	for _, bad := range []string{"", "   ", strings.Repeat("a", maxTerminalCommand+1), "Write-Output 1\x00"} {
		if code, _, _ := run(pcToken, bad); code != 400 {
			t.Fatalf("an invalid command must be refused (%d bytes): %d", len(bad), code)
		}
	}

	// The owner allows the terminal on another device, then withdraws it.
	if code := owner.do("POST", "/workshop-owner/api/devices/"+tabletID+"/terminal", "{}", session, nil); code != 200 {
		t.Fatalf("the owner must allow the terminal: %d", code)
	}
	if code, reply, _ := run(tabletToken, "Write-Output 7"); code != 200 || strings.TrimSpace(reply.Output) != "7" {
		t.Fatalf("allowed by the owner: %d", code)
	}
	if code := owner.do("POST", "/workshop-owner/api/devices/"+tabletID+"/terminal-off", "{}", session, nil); code != 200 {
		t.Fatalf("the owner must withdraw the terminal: %d", code)
	}
	if code, _, _ := run(tabletToken, "Write-Output 7"); code != 403 {
		t.Fatalf("withdrawn by the owner: %d", code)
	}
	var state struct {
		Devices []struct {
			ID       string
			Terminal bool
		}
	}
	if code := owner.do("GET", "/workshop-owner/api/state", "", session, &state); code != 200 {
		t.Fatalf("owner state: %d", code)
	}
	with := 0
	for _, d := range state.Devices {
		if d.Terminal {
			with++
		}
	}
	if with != 1 {
		t.Fatalf("the owner page must show which device has the terminal: %+v", state.Devices)
	}

	// The journal: one line per command run, with no output.
	log, err := s.TerminalLog(50)
	if err != nil {
		t.Fatal(err)
	}
	if len(log) != 5 {
		t.Fatalf("every command run is journaled: %d %+v", len(log), log)
	}
	if log[0].Command != "Write-Output 7" || log[len(log)-1].Command != "Write-Output ('bon'+'jour é'); exit 3" || log[len(log)-1].ExitCode != 3 || !log[2].TimedOut {
		t.Fatalf("the journal, newest first: %+v", log)
	}
	for _, entry := range log {
		if entry.DeviceID == "" || entry.At.IsZero() {
			t.Fatalf("a journal line names its device and time: %+v", entry)
		}
	}
}
