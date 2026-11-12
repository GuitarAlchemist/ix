//go:build linux

package main

import (
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"
)

// A zombie (exited, not yet reaped) still has a /proc/<pid>/stat: it is not a live process, and
// its creation time verifies nothing.
func TestAZombieHasNoStartTime(t *testing.T) {
	cmd := exec.Command("true")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer cmd.Wait()
	pid := cmd.Process.Pid
	stat := "/proc/" + strconv.Itoa(pid) + "/stat"
	for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(5 * time.Millisecond) {
		b, _ := os.ReadFile(stat)
		if i := strings.LastIndexByte(string(b), ')'); i > 0 && strings.HasPrefix(string(b[i+1:]), " Z") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the child never became a zombie")
		}
	}
	if ms, err := procStartMs(pid); err == nil {
		t.Fatalf("zombie %d has a start time %d", pid, ms)
	}
}
