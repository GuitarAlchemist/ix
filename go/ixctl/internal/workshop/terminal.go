package workshop

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
	"unicode/utf16"
	"unicode/utf8"
)

// The terminal runs commands on the Desktop's PC, authorized by the owner on 2026-11-14 for the PC only: the PC the owner
// links from the owner page has it, and any other device only once the owner allows it there from the owner page,
// device by device. Administering devices does not include it. Commands run with Windows PowerShell as the service's
// own account (not the owner's Windows account), in the terminal's folder beside the database, one at a time, stopped
// after a time limit, their output capped. Each run is journaled — device, command, how it ended — never its output.

var ErrNoTerminal = errors.New("terminal_not_allowed")

const (
	maxTerminalCommand    = 4096
	maxTerminalOutput     = 256 << 10
	defaultTerminalLimit  = 30 * time.Second
	terminalJournalLength = 1000
)

var terminalSchema = []string{
	`CREATE TABLE IF NOT EXISTS workshop_terminal (device_id TEXT PRIMARY KEY)`,
	`CREATE TABLE IF NOT EXISTS workshop_terminal_log (at INTEGER NOT NULL, device_id TEXT NOT NULL, command TEXT NOT NULL, exit_code INTEGER NOT NULL, timed_out INTEGER NOT NULL, ms INTEGER NOT NULL, output_bytes INTEGER NOT NULL)`,
}
var terminalSchemaChecks = []string{
	`SELECT device_id FROM workshop_terminal LIMIT 0`,
	`SELECT at,device_id,command,exit_code,timed_out,ms,output_bytes FROM workshop_terminal_log LIMIT 0`,
}

// TerminalResult is what one command gave.
type TerminalResult struct {
	Output    string `json:"output"`
	ExitCode  int    `json:"exit_code"`
	TimedOut  bool   `json:"timed_out"`
	Truncated bool   `json:"truncated"`
	MS        int64  `json:"ms"`
}

// TerminalRun is one journal line.
type TerminalRun struct {
	At          time.Time `json:"at"`
	DeviceID    string    `json:"device_id"`
	Command     string    `json:"command"`
	ExitCode    int       `json:"exit_code"`
	TimedOut    bool      `json:"timed_out"`
	MS          int64     `json:"ms"`
	OutputBytes int       `json:"output_bytes"`
}

func (s *Store) hasTerminal(id string) (bool, error) {
	var n int
	err := s.db.QueryRow(`SELECT count(*) FROM workshop_terminal WHERE device_id=?`, id).Scan(&n)
	return n == 1, err
}

// AllowTerminal is owner-only (owner page): an approved device gets the terminal, or loses it.
func (s *Store) AllowTerminal(id string, allowed bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !allowed {
		_, err := s.db.Exec(`DELETE FROM workshop_terminal WHERE device_id=?`, id)
		return err
	}
	var status string
	if err := s.db.QueryRow(`SELECT status FROM workshop_devices WHERE id=?`, id).Scan(&status); err != nil || status != "approved" {
		return ErrInvalid
	}
	_, err := s.db.Exec(`INSERT OR IGNORE INTO workshop_terminal VALUES(?)`, id)
	return err
}

func validCommand(command string) bool {
	return strings.TrimSpace(command) != "" && len(command) <= maxTerminalCommand && utf8.ValidString(command) && !strings.ContainsRune(command, 0)
}

// cappedBuffer keeps the first max bytes written to it and counts the rest. The buffer is a field, not embedded: an
// embedded bytes.Buffer would lend its ReadFrom to io.Copy, which would then bypass Write and the cap.
type cappedBuffer struct {
	buf          bytes.Buffer
	max, dropped int
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	room := b.max - b.buf.Len()
	if room < len(p) {
		if room > 0 {
			b.buf.Write(p[:room])
		}
		b.dropped += len(p) - max(room, 0)
		return len(p), nil
	}
	return b.buf.Write(p)
}

// RunTerminal runs command for the device behind token, if it has the terminal.
func (s *Store) RunTerminal(ctx context.Context, token, command string) (TerminalResult, error) {
	if !validCommand(command) {
		return TerminalResult{}, ErrInvalid
	}
	s.mu.Lock()
	d, err := authenticate(s.db, token)
	allowed := false
	if err == nil {
		allowed, err = s.hasTerminal(d.ID)
	}
	s.mu.Unlock()
	if err != nil {
		return TerminalResult{}, err
	}
	if !allowed {
		return TerminalResult{}, ErrNoTerminal
	}
	if runtime.GOOS != "windows" {
		return TerminalResult{}, ErrNoTerminal
	}

	s.terminalRun.Lock()
	defer s.terminalRun.Unlock()
	dir := filepath.Join(filepath.Dir(s.path), "terminal")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return TerminalResult{}, err
	}
	limit := s.terminalTimeout
	if limit <= 0 {
		limit = defaultTerminalLimit
	}
	runCtx, cancel := context.WithTimeout(ctx, limit)
	defer cancel()
	// UTF-8 output, and the command passed encoded so that no quoting can change it.
	script := "[Console]::OutputEncoding=[Text.Encoding]::UTF8;$OutputEncoding=[Text.Encoding]::UTF8;$ProgressPreference='SilentlyContinue'\n" + command
	units := utf16.Encode([]rune(script))
	raw := make([]byte, 2*len(units))
	for i, u := range units {
		raw[2*i], raw[2*i+1] = byte(u), byte(u>>8)
	}
	shell := filepath.Join(os.Getenv("SystemRoot"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
	cmd := exec.CommandContext(runCtx, shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", base64.StdEncoding.EncodeToString(raw))
	cmd.Dir = dir
	out := &cappedBuffer{max: maxTerminalOutput}
	cmd.Stdout, cmd.Stderr = out, out
	cmd.WaitDelay = 2 * time.Second
	started := time.Now()
	runErr := cmd.Run()
	result := TerminalResult{Output: strings.ToValidUTF8(out.buf.String(), "�"), Truncated: out.dropped > 0, MS: time.Since(started).Milliseconds()}
	var exit *exec.ExitError
	switch {
	case runCtx.Err() != nil:
		result.TimedOut, result.ExitCode = true, -1
	case runErr == nil:
	case errors.As(runErr, &exit):
		result.ExitCode = exit.ExitCode()
	default:
		return TerminalResult{}, runErr
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.db.Exec(`INSERT INTO workshop_terminal_log VALUES(?,?,?,?,?,?,?)`, s.now().UnixMilli(), d.ID, command, result.ExitCode, result.TimedOut, result.MS, out.buf.Len()+out.dropped); err != nil {
		return result, err
	}
	_, err = s.db.Exec(`DELETE FROM workshop_terminal_log WHERE rowid NOT IN (SELECT rowid FROM workshop_terminal_log ORDER BY rowid DESC LIMIT ?)`, terminalJournalLength)
	return result, err
}

// TerminalLog is the journal, newest first.
func (s *Store) TerminalLog(limit int) ([]TerminalRun, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rows, err := s.db.Query(`SELECT at,device_id,command,exit_code,timed_out,ms,output_bytes FROM workshop_terminal_log ORDER BY rowid DESC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	runs := []TerminalRun{}
	for rows.Next() {
		var r TerminalRun
		var at int64
		if err := rows.Scan(&at, &r.DeviceID, &r.Command, &r.ExitCode, &r.TimedOut, &r.MS, &r.OutputBytes); err != nil {
			return nil, err
		}
		r.At = time.UnixMilli(at)
		runs = append(runs, r)
	}
	return runs, rows.Err()
}
