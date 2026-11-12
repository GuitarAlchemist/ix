package main

import (
	"errors"
	"strings"
	"testing"
)

// The gaps GAIA-IX-WINSVC-REVIEW-R4 found in the resume check's tests: each case below uses fake
// probes only (no socket, no service manager, no process) and names the faulty change it catches.

func resumeChecks(r resumeResult) string { return strings.Join(r.Checks, "\n") }

// Delta Remote's contract: a "-dirty-HHmmss" build is not the expected build, even though its
// name starts with the expected one. Catches a prefix comparison in place of the exact one.
func TestResumeListenImageRefusesADirtyBuild(t *testing.T) {
	const want = "delta-remote-v0.6.0-21-g2ca5e6c.exe"
	inv := mustInventory(t, `{"components":[{"name":"remote","listen":"192.168.86.250:8810","listen_image":"`+want+`"}]}`)
	p := resumeProbes{
		listening:   func(string) (bool, error) { return true, nil },
		listenOwner: func(string) (uint32, error) { return 4242, nil },
		imageOf: func(uint32) (string, func(), error) {
			return `C:\Temp\delta-remote\delta-remote-v0.6.0-21-g2ca5e6c-dirty-142233.exe`, func() {}, nil
		},
	}
	r := checkResume(inv, p)[0]
	if r.Verdict != verdictDown || !strings.Contains(resumeChecks(r), "held by delta-remote-v0.6.0-21-g2ca5e6c-dirty-142233.exe, not "+want) {
		t.Fatalf("dirty build: %s %q, want down naming both builds", r.Verdict, r.Checks)
	}
}

// A service that is not installed (the SCM answers ERROR_SERVICE_DOES_NOT_EXIST, read as
// "not installed") is down, not degraded. Catches a check for "stopped" in place of "not running".
func TestResumeServiceNotInstalledIsDown(t *testing.T) {
	inv := mustInventory(t, `{"components":[{"name":"plane","service":"ixctl"}]}`)
	p := fakeProbes(map[string][2]string{"ixctl": {"not installed", "none"}}, nil, nil)
	r := checkResume(inv, p)[0]
	if r.Verdict != verdictDown || !strings.Contains(resumeChecks(r), "service ixctl is not installed (start type none)") {
		t.Fatalf("not installed: %s %q, want down", r.Verdict, r.Checks)
	}
}

// A TCP table that cannot be read is unknown, not "nothing listens", and a component after it is
// blocked. Catches an ignored probe error (read as down) and a block on down only.
func TestResumeListenerTableRefusedIsUnknown(t *testing.T) {
	inv := mustInventory(t, `{"components":[
		{"name":"voice","listen":"127.0.0.1:59126"},
		{"name":"hook","after":["voice"],"service":"hook-svc"}]}`)
	p := fakeProbes(map[string][2]string{"hook-svc": {"running", "automatic"}}, nil, nil)
	p.listening = func(string) (bool, error) { return false, errors.New("Access is denied.") }
	got := checkResume(inv, p)
	voice, hook := got[0], got[1]
	if voice.Verdict != verdictUnknown || !strings.Contains(resumeChecks(voice), "listener 127.0.0.1:59126: Access is denied.") {
		t.Fatalf("voice: %s %q, want unknown with the probe error", voice.Verdict, voice.Checks)
	}
	if strings.Contains(resumeChecks(voice), "nothing listens") {
		t.Fatalf("voice reported as refused: %q", voice.Checks)
	}
	if len(hook.BlockedBy) != 1 || hook.BlockedBy[0] != "voice" {
		t.Fatalf("hook blocked by %v, want voice", hook.BlockedBy)
	}
}

// After the image is read, the table is read again: an empty owner or a failed read means the
// name may belong to another process, so the verdict is unknown and the handle is still released.
// Catches a dropped re-read and a re-read whose error is ignored.
func TestResumeListenImageSecondReadEmptyOrFailed(t *testing.T) {
	const want = "delta-remote-v0.6.0-21-g2ca5e6c.exe"
	inv := mustInventory(t, `{"components":[{"name":"remote","listen":"192.168.86.250:8810","listen_image":"`+want+`"}]}`)
	for _, c := range []struct {
		name   string
		second func() (uint32, error)
		line   string
	}{
		{"empty", func() (uint32, error) { return 0, nil }, "changed from process 4242 to 0"},
		{"failed", func() (uint32, error) { return 0, errors.New("table read failed") }, "image: table read failed"},
	} {
		reads, released := 0, 0
		p := resumeProbes{
			listening: func(string) (bool, error) { return true, nil },
			listenOwner: func(string) (uint32, error) {
				reads++
				if reads == 1 {
					return 4242, nil
				}
				return c.second()
			},
			imageOf: func(uint32) (string, func(), error) {
				return `C:\Temp\` + want, func() { released++ }, nil
			},
		}
		r := checkResume(inv, p)[0]
		if r.Verdict != verdictUnknown || !strings.Contains(resumeChecks(r), c.line) {
			t.Errorf("%s: %s %q, want unknown with %q", c.name, r.Verdict, r.Checks, c.line)
		}
		if reads != 2 || released != 1 {
			t.Errorf("%s: %d table reads, handle released %d times, want 2 and 1", c.name, reads, released)
		}
	}
}
