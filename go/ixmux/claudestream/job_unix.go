//go:build unix

package claudestream

import (
	"os/exec"
	"syscall"
)

// job is the child's process group. Unlike the Windows Job Object, nothing kills it if ixmux
// itself is killed: Stop does.
// @ai:assumption outside Windows a killed ixmux leaves its claude child running until stdin closes it [U:uncertain conf:0.5]
type job struct{ pgid int }

func (j *job) kill() error { return syscall.Kill(-j.pgid, syscall.SIGKILL) }

func (j *job) Close() error { return nil }

func startInJob(cmd *exec.Cmd) (*job, error) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &job{cmd.Process.Pid}, nil
}

func alive(pid int) bool { return syscall.Kill(pid, 0) == nil }
