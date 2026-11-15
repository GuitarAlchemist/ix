package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ready_journal: the control plane listens on a port chosen at each boot (--addr 127.0.0.1:0),
// so the inventory names the service's host log and the check reads the address from the current
// host's host.ready line (the log, else its rotation, and no final stop after it; hostjournal_test.go),
// accepted only when that line's PID is the process the service runs now.
// Fake SCM, fake journal, no socket.

const journalPath = `C:\ProgramData\ixctl\ixctl-trial.service.log`

const controlInventory = `{"components":[{"name":"control","service":"ixctl","ready_journal":"` +
	`C:\\ProgramData\\ixctl\\ixctl-trial.service.log","health":"/health","health_expect":"\"ok\":true"}]}`

func journalReady(pid, addr string) string {
	return `{"t":1,"event":"host.ready","name":"ixctl-trial","addr":"` + addr + `","pid":` + pid + `,"epoch":"e","version":"v"}`
}

type journalRun struct {
	pid       uint32
	pidErr    error
	journal   string
	readErr   error
	listened  []string
	healthURL []string
	tailPath  string
	tailMax   int64
	// healthBody is what /health answers; the default carries the epoch journalReady writes.
	healthBody string
}

func (j *journalRun) probes() resumeProbes {
	return resumeProbes{
		service: func(string) (string, string, string, error) {
			return "running", "automatic (delayed)", configuredImage, nil
		},
		servicePID: func(string) (uint32, error) { return j.pid, j.pidErr },
		readTail: func(path string, max int64) ([]byte, error) {
			j.tailPath, j.tailMax = path, max
			return []byte(j.journal), j.readErr
		},
		listening: func(addr string) (bool, error) {
			j.listened = append(j.listened, addr)
			return true, nil
		},
		health: func(u, _ string) ([]byte, error) {
			j.healthURL = append(j.healthURL, u)
			if j.healthBody != "" {
				return []byte(j.healthBody), nil
			}
			return []byte(`{"ok":true,"epoch":"e"}`), nil
		},
	}
}

func TestResumeReadyJournalGivesTheAddress(t *testing.T) {
	inv := mustInventory(t, controlInventory)
	j := &journalRun{pid: 4242, journal: strings.Join([]string{
		`artial line cut by the tail read"}`,
		journalReady("1111", "127.0.0.1:50001"),
		`{"t":2,"event":"host.stop","pid":1111}`,
		journalReady("4242", "127.0.0.1:59762"),
		`{"t":3,"event":"host.re`, // a line being written
	}, "\n")}
	r := checkResume(inv, j.probes())[0]
	if r.Verdict != verdictOK {
		t.Fatalf("verdict %s, want ok: %q", r.Verdict, r.Checks)
	}
	if j.tailPath != journalPath || j.tailMax != resumeJournalTail {
		t.Errorf("read %q (last %d bytes), want %q (last %d)", j.tailPath, j.tailMax, journalPath, resumeJournalTail)
	}
	if len(j.listened) != 1 || j.listened[0] != "127.0.0.1:59762" {
		t.Errorf("listened %q, want only 127.0.0.1:59762", j.listened)
	}
	if len(j.healthURL) != 1 || j.healthURL[0] != "http://127.0.0.1:59762/health" {
		t.Errorf("health %q, want only http://127.0.0.1:59762/health", j.healthURL)
	}
	if want := "journal " + journalPath + ": host.ready of PID 4242, the service's process, says 127.0.0.1:59762"; !strings.Contains(resumeChecks(r), want) {
		t.Errorf("checks %q lack %q", r.Checks, want)
	}
}

// The probed address is the parsed one, not the journal's text: a long IPv6 loopback form reaches
// the listener and the health URL as [::1].
func TestResumeReadyJournalProbesTheParsedAddress(t *testing.T) {
	inv := mustInventory(t, controlInventory)
	j := &journalRun{pid: 4242, journal: journalReady("4242", "[0:0:0:0:0:0:0:1]:59762")}
	r := checkResume(inv, j.probes())[0]
	if r.Verdict != verdictOK {
		t.Fatalf("verdict %s, want ok: %q", r.Verdict, r.Checks)
	}
	if len(j.listened) != 1 || j.listened[0] != "[::1]:59762" {
		t.Errorf("listened %q, want only [::1]:59762", j.listened)
	}
	if len(j.healthURL) != 1 || j.healthURL[0] != "http://[::1]:59762/health" {
		t.Errorf("health %q, want only http://[::1]:59762/health", j.healthURL)
	}
}

func TestResumeReadyJournalRefusals(t *testing.T) {
	inv := mustInventory(t, controlInventory)
	for _, c := range []struct {
		name string
		run  journalRun
		line string
	}{
		{"stale line: the service restarted and its new host has not said ready",
			journalRun{pid: 4242, journal: journalReady("1111", "127.0.0.1:50001")},
			"the last host.ready line in " + journalPath + " is PID 1111's, the service runs PID 4242: this host has not said ready, or the journal is another host's"},
		{"no ready line", journalRun{pid: 4242, journal: `{"t":1,"event":"host.stop","pid":1111}`},
			"no host.ready line in the last 65536 bytes of " + journalPath},
		{"journal unreadable", journalRun{pid: 4242, readErr: errors.New("Access is denied.")},
			"journal " + journalPath + ": Access is denied."},
		{"service PID unreadable", journalRun{pidErr: errors.New("Access is denied.")},
			"service ixctl's process: Access is denied."},
		{"service runs no process", journalRun{pid: 0, journal: journalReady("1111", "127.0.0.1:50001")},
			"service ixctl runs no process: no address to read from " + journalPath},
		{"address off loopback", journalRun{pid: 4242, journal: journalReady("4242", "10.0.0.5:80")},
			"host.ready of PID 4242 in " + journalPath + " says 10.0.0.5:80, which is not a loopback address"},
		{"address without a port", journalRun{pid: 4242, journal: journalReady("4242", "127.0.0.1")},
			"host.ready of PID 4242 in " + journalPath + " says 127.0.0.1, which is not a loopback address"},
		// The address is pasted into the health URL: anything but IP:port could move its host.
		{"userinfo after the port", journalRun{pid: 4242, journal: journalReady("4242", "127.0.0.1:80@example.com")},
			"host.ready of PID 4242 in " + journalPath + " says 127.0.0.1:80@example.com, which is not a loopback address"},
		{"empty port", journalRun{pid: 4242, journal: journalReady("4242", "127.0.0.1:")},
			"host.ready of PID 4242 in " + journalPath + " says 127.0.0.1:, which is not a loopback address"},
		{"port zero", journalRun{pid: 4242, journal: journalReady("4242", "127.0.0.1:0")},
			"host.ready of PID 4242 in " + journalPath + " says 127.0.0.1:0, which is not a loopback address"},
		{"named port", journalRun{pid: 4242, journal: journalReady("4242", "127.0.0.1:http")},
			"host.ready of PID 4242 in " + journalPath + " says 127.0.0.1:http, which is not a loopback address"},
		{"no epoch", journalRun{pid: 4242, journal: `{"t":1,"event":"host.ready","addr":"127.0.0.1:59762","pid":4242}`},
			"host.ready of PID 4242 in " + journalPath + " names no epoch: nothing ties it to the process that answers"},
	} {
		t.Run(c.name, func(t *testing.T) {
			j := c.run
			r := checkResume(inv, j.probes())[0]
			if r.Verdict != verdictUnknown {
				t.Errorf("verdict %s, want unknown: %q", r.Verdict, r.Checks)
			}
			if !strings.Contains(resumeChecks(r), c.line) {
				t.Errorf("checks %q lack %q", r.Checks, c.line)
			}
			if len(j.listened)+len(j.healthURL) != 0 {
				t.Errorf("probed listen %q health %q with no trusted address", j.listened, j.healthURL)
			}
		})
	}
}

// PIDs come back after a reboot: a line an earlier run of the same PID wrote (that run died
// without a stop line, this one has not said ready) passes the PID check. Its epoch, drawn at
// each boot, does not match what /health answers.
func TestResumeReadyJournalEpochBindsTheAnsweringHost(t *testing.T) {
	inv := mustInventory(t, controlInventory)
	j := &journalRun{pid: 4242, journal: journalReady("4242", "127.0.0.1:59762"), healthBody: `{"ok":true,"epoch":"other"}`}
	r := checkResume(inv, j.probes())[0]
	if r.Verdict != verdictUnknown {
		t.Errorf("verdict %s, want unknown: %q", r.Verdict, r.Checks)
	}
	want := `health http://127.0.0.1:59762/health answers epoch "other", the host.ready line of PID 4242 in ` + journalPath + ` is epoch "e"'s: an earlier run wrote it, this one has not said ready`
	if !strings.Contains(resumeChecks(r), want) {
		t.Errorf("checks %q lack %q", r.Checks, want)
	}
	j = &journalRun{pid: 4242, journal: journalReady("4242", "127.0.0.1:59762"), healthBody: `{"ok":true}`}
	if r := checkResume(inv, j.probes())[0]; r.Verdict != verdictUnknown {
		t.Errorf("health without an epoch: verdict %s, want unknown: %q", r.Verdict, r.Checks)
	}
}

func TestResumeReadyJournalParse(t *testing.T) {
	for _, c := range []struct{ name, js, want string }{
		{"without service", `{"components":[{"name":"c","ready_journal":"j","health":"/health"}]}`,
			"inventory: c ready_journal needs service, whose process wrote the host.ready line"},
		{"with listen", `{"components":[{"name":"c","service":"s","ready_journal":"j","listen":"127.0.0.1:1"}]}`,
			"inventory: c has listen and ready_journal: the journal gives the address"},
		{"with a full health URL", `{"components":[{"name":"c","service":"s","ready_journal":"j","health":"http://127.0.0.1:1/health"}]}`,
			"inventory: c health with ready_journal is a path such as /health, the journal gives the address"},
		{"without health", `{"components":[{"name":"c","service":"s","ready_journal":"j"}]}`,
			"inventory: c ready_journal needs health, whose epoch must be the host.ready line's"},
		{"a health path without ready_journal", `{"components":[{"name":"c","health":"/health"}]}`,
			"inventory: c health"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := parseResumeInventory([]byte(c.js))
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("err %v, want %q", err, c.want)
			}
		})
	}
	if _, err := parseResumeInventory([]byte(controlInventory)); err != nil {
		t.Errorf("control inventory refused: %v", err)
	}
}

func TestReadTailReadsOnlyTheEnd(t *testing.T) {
	path := filepath.Join(t.TempDir(), "service.log")
	body := strings.Repeat("x", 100) + "\n" + journalReady("7", "127.0.0.1:1")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := realResumeProbes().readTail(path, 40)
	if err != nil || string(got) != body[len(body)-40:] {
		t.Errorf("tail %q, %v; want the last 40 bytes %q", got, err, body[len(body)-40:])
	}
	got, err = realResumeProbes().readTail(path, 1<<20)
	if err != nil || string(got) != body {
		t.Errorf("whole file %q, %v; want %q", got, err, body)
	}
}
