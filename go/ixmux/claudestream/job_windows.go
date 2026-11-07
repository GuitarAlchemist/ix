package claudestream

import (
	"fmt"
	"os/exec"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

// job is a Job Object with kill-on-close: when its last handle closes (Stop, or the death of
// ixmux, whatever kills it), every process in it is killed. The handle is not inheritable, so
// the child does not keep its own job alive.
type job struct{ h windows.Handle }

// kill ends every process of the job.
func (j *job) kill() error { return windows.TerminateJobObject(j.h, 1) }

func (j *job) Close() error { return windows.CloseHandle(j.h) }

// startInJob starts cmd suspended, assigns it to a new kill-on-close job, then resumes it: the
// child runs no instruction, and spawns no process, outside the job.
// @ai:invariant the claude child is in a kill-on-close Job Object before it runs, so it dies with ixmux even when ixmux is killed [T:test conf:0.85 src:go/ixmux/cmd/ixmux/claude_windows_test.go:TestClaudeKillIxmuxKillsTheChild]
func startInJob(cmd *exec.Cmd) (*job, error) {
	h, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return nil, fmt.Errorf("create the job object: %w", err)
	}
	j := &job{h}
	info := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
	info.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	if _, err := windows.SetInformationJobObject(h, windows.JobObjectExtendedLimitInformation, uintptr(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info))); err != nil {
		j.Close()
		return nil, fmt.Errorf("set kill-on-close: %w", err)
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= windows.CREATE_SUSPENDED
	if err := cmd.Start(); err != nil {
		j.Close()
		return nil, err
	}
	fail := func(err error) (*job, error) {
		cmd.Process.Kill()
		cmd.Wait()
		j.Close()
		return nil, err
	}
	pid := uint32(cmd.Process.Pid)
	ph, err := windows.OpenProcess(windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE, false, pid)
	if err != nil {
		return fail(fmt.Errorf("open the child: %w", err))
	}
	err = windows.AssignProcessToJobObject(h, ph)
	windows.CloseHandle(ph)
	if err != nil {
		return fail(fmt.Errorf("assign the child to the job: %w", err))
	}
	if err := resume(pid); err != nil {
		return fail(err)
	}
	return j, nil
}

// resume resumes the threads of the suspended process pid (its main thread).
func resume(pid uint32) error {
	snap, err := windows.CreateToolhelp32Snapshot(windows.TH32CS_SNAPTHREAD, 0)
	if err != nil {
		return fmt.Errorf("snapshot threads: %w", err)
	}
	defer windows.CloseHandle(snap)
	te := windows.ThreadEntry32{Size: uint32(unsafe.Sizeof(windows.ThreadEntry32{}))}
	resumed := 0
	for err = windows.Thread32First(snap, &te); err == nil; err = windows.Thread32Next(snap, &te) {
		if te.OwnerProcessID != pid {
			continue
		}
		th, err := windows.OpenThread(windows.THREAD_SUSPEND_RESUME, false, te.ThreadID)
		if err != nil {
			return fmt.Errorf("open thread %d: %w", te.ThreadID, err)
		}
		_, err = windows.ResumeThread(th)
		windows.CloseHandle(th)
		if err != nil {
			return fmt.Errorf("resume thread %d: %w", te.ThreadID, err)
		}
		resumed++
	}
	if resumed == 0 {
		return fmt.Errorf("no thread of process %d to resume", pid)
	}
	return nil
}

// alive: pid exists and has not exited.
func alive(pid int) bool {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return false
	}
	defer windows.CloseHandle(h)
	var code uint32
	return windows.GetExitCodeProcess(h, &code) == nil && code == 259 // STILL_ACTIVE
}
