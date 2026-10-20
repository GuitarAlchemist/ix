//go:build windows

package main

import (
	"fmt"
	"os"
	"os/exec"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	procNtResumeProcess = windows.NewLazySystemDLL("ntdll.dll").NewProc("NtResumeProcess")
	procQPC             = windows.NewLazySystemDLL("kernel32.dll").NewProc("QueryPerformanceCounter")
	procQPF             = windows.NewLazySystemDLL("kernel32.dll").NewProc("QueryPerformanceFrequency")
	qpcFreq             int64
)

// procTree is a service's process tree: a job object that holds the service
// and everything it spawns. Closing the job (or the plane dying) kills it all.
type procTree struct {
	job  windows.Handle
	port windows.Handle // the job's completion port, when a memory limit is watched
	over chan string    // the limit the tree went past, once
}

// Job completion messages (winnt.h JOB_OBJECT_MSG_*), as GetQueuedCompletionStatus reports
// them in its byte count.
const (
	jobMsgActiveProcessZero  = 4
	jobMsgProcessMemoryLimit = 9
	jobMsgJobMemoryLimit     = 10
)

// jobPort mirrors JOBOBJECT_ASSOCIATE_COMPLETION_PORT.
type jobPort struct {
	CompletionKey  uintptr
	CompletionPort windows.Handle
}

// startTree starts cmd suspended, puts it in a fresh job with
// KILL_ON_JOB_CLOSE, then lets it run, so no grandchild can be born outside
// the job. The child gets its own console process group: Ctrl+C reaches the
// plane only, and the plane decides how its services stop. With a memory
// limit, the job refuses commits past it and tells a completion port, which
// the tree reports on `exceeded`.
func startTree(cmd *exec.Cmd, limits *Limits) (*procTree, error) {
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return nil, fmt.Errorf("create job: %w", err)
	}
	t := &procTree{job: job}
	var info windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
	info.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	if limits != nil && limits.MemoryMB > 0 {
		info.BasicLimitInformation.LimitFlags |= windows.JOB_OBJECT_LIMIT_JOB_MEMORY
		info.JobMemoryLimit = uintptr(limits.MemoryMB) << 20
	}
	if _, err := windows.SetInformationJobObject(job, windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info))); err != nil {
		windows.CloseHandle(job)
		return nil, fmt.Errorf("set job limits: %w", err)
	}
	if limits != nil && limits.MemoryMB > 0 {
		port, err := windows.CreateIoCompletionPort(windows.InvalidHandle, 0, 0, 1)
		if err != nil {
			windows.CloseHandle(job)
			return nil, fmt.Errorf("create completion port: %w", err)
		}
		assoc := jobPort{CompletionKey: uintptr(job), CompletionPort: port}
		if _, err := windows.SetInformationJobObject(job, windows.JobObjectAssociateCompletionPortInformation,
			uintptr(unsafe.Pointer(&assoc)), uint32(unsafe.Sizeof(assoc))); err != nil {
			windows.CloseHandle(port)
			windows.CloseHandle(job)
			return nil, fmt.Errorf("associate completion port: %w", err)
		}
		t.port, t.over = port, make(chan string, 1)
		go t.listen()
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= windows.CREATE_SUSPENDED | windows.CREATE_NEW_PROCESS_GROUP
	if err := cmd.Start(); err != nil {
		windows.CloseHandle(job)
		return nil, err
	}
	var cerr error
	herr := cmd.Process.WithHandle(func(h uintptr) {
		if err := windows.AssignProcessToJobObject(job, windows.Handle(h)); err != nil {
			cerr = fmt.Errorf("assign to job: %w", err)
			return
		}
		if st, _, _ := procNtResumeProcess.Call(h); st != 0 {
			cerr = fmt.Errorf("resume: NTSTATUS 0x%x", st)
		}
	})
	if herr != nil && cerr == nil {
		cerr = herr
	}
	if cerr != nil {
		cmd.Process.Kill()
		cmd.Wait()
		t.close()
		return nil, cerr
	}
	return t, nil
}

// listen reads the job's completion messages until the port is closed or the job is empty,
// and reports a memory limit gone past on `over`, once.
func (t *procTree) listen() {
	for {
		var code uint32
		var key uintptr
		var ov *windows.Overlapped
		if err := windows.GetQueuedCompletionStatus(t.port, &code, &key, &ov, windows.INFINITE); err != nil {
			return
		}
		switch code {
		case jobMsgJobMemoryLimit, jobMsgProcessMemoryLimit:
			select {
			case t.over <- reasonMemory:
			default:
			}
		case jobMsgActiveProcessZero:
			return
		}
	}
}

// exceeded yields the limit the tree went past, once; nil when none is watched.
func (t *procTree) exceeded() <-chan string { return t.over }

// usage is what the job's processes, exited ones included, have used as the kernel counted
// it: the most memory held at once (peak commit charge) and the processor time, user and
// kernel, in milliseconds; 0 for what cannot be read. Read before close.
func (t *procTree) usage() (peakBytes, cpuMs int64) {
	var info windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
	if err := windows.QueryInformationJobObject(t.job, windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info)), nil); err == nil {
		peakBytes = int64(info.PeakJobMemoryUsed)
	}
	var acc jobAccounting
	if err := windows.QueryInformationJobObject(t.job, windows.JobObjectBasicAccountingInformation,
		uintptr(unsafe.Pointer(&acc)), uint32(unsafe.Sizeof(acc)), nil); err == nil {
		cpuMs = (acc.TotalUserTime + acc.TotalKernelTime) / 10_000 // 100 ns units
	}
	return peakBytes, cpuMs
}

func (t *procTree) kill() error {
	return windows.TerminateJobObject(t.job, 1)
}

// jobAccounting mirrors JOBOBJECT_BASIC_ACCOUNTING_INFORMATION.
type jobAccounting struct {
	TotalUserTime             int64
	TotalKernelTime           int64
	ThisPeriodTotalUserTime   int64
	ThisPeriodTotalKernelTime int64
	TotalPageFaultCount       uint32
	TotalProcesses            uint32
	ActiveProcesses           uint32
	TotalTerminatedProcesses  uint32
}

// waitGone waits until no process is left in the job.
func (t *procTree) waitGone(limit time.Duration) bool {
	deadline := time.Now().Add(limit)
	for {
		var acc jobAccounting
		err := windows.QueryInformationJobObject(t.job, windows.JobObjectBasicAccountingInformation,
			uintptr(unsafe.Pointer(&acc)), uint32(unsafe.Sizeof(acc)), nil)
		if err != nil || acc.ActiveProcesses == 0 {
			return err == nil
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(time.Millisecond)
	}
}

// exitCode is the child's exit code; a terminated process reports the code
// TerminateProcess gave it.
func exitCode(ee *exec.ExitError) int { return ee.ExitCode() }

func (t *procTree) close() {
	if t.port != 0 {
		windows.CloseHandle(t.port) // ends listen
		t.port = 0
	}
	windows.CloseHandle(t.job)
}

// leaderUsage is the leader's peak resident set and processor time as the OS accounted them
// at exit; on Windows the job's accounting covers the whole tree instead (usage).
func leaderUsage(*os.ProcessState) (rssBytes, cpuMs int64) { return 0, 0 }

// hrnow is a machine-wide clock in nanoseconds (QueryPerformanceCounter).
// Go's time.Now on Windows ticks in steps of about 0.5 ms, too coarse to
// measure a hop between two processes.
func hrnow() int64 {
	var c int64
	procQPC.Call(uintptr(unsafe.Pointer(&c)))
	return c/qpcFreq*1e9 + c%qpcFreq*1e9/qpcFreq
}

func init() {
	procQPF.Call(uintptr(unsafe.Pointer(&qpcFreq)))
	if qpcFreq == 0 {
		qpcFreq = 1
	}
}

// msSinceProcessStart measures from the OS's creation time of this process.
func msSinceProcessStart() float64 {
	var creation, exit, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(windows.CurrentProcess(), &creation, &exit, &kernel, &user); err != nil {
		return float64(time.Since(initTime).Microseconds()) / 1000
	}
	var now windows.Filetime
	windows.GetSystemTimePreciseAsFileTime(&now)
	return float64(now.Nanoseconds()-creation.Nanoseconds()) / 1e6
}
