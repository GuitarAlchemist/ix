//go:build !windows && !unix

package claudestream

import (
	"errors"
	"os/exec"
)

type job struct{}

func (j *job) kill() error  { return nil }
func (j *job) Close() error { return nil }

// startInJob refuses: no way here to tie the child's life to ixmux's.
func startInJob(cmd *exec.Cmd) (*job, error) {
	return nil, errors.New("claude targets are not supported on this system")
}

func alive(pid int) bool { return false }
