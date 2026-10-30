package main

import (
	"runtime"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	kernel32              = windows.NewLazySystemDLL("kernel32.dll")
	procGetSystemTimes    = kernel32.NewProc("GetSystemTimes")
	procGlobalMemoryStatu = kernel32.NewProc("GlobalMemoryStatusEx")
)

// cpuProbe reads the host's processor time (GetSystemTimes: idle, kernel and user, over
// every logical processor, in 100 ns units) and reports the busy share of the last
// interval. Kernel time includes idle time, as the API defines it.
type cpuProbe struct {
	idle, total uint64
	primed      bool
}

func filetime(ft windows.Filetime) uint64 {
	return uint64(ft.HighDateTime)<<32 | uint64(ft.LowDateTime)
}

func (c *cpuProbe) sample() (map[string]any, error) {
	var idle, kernel, user windows.Filetime
	r, _, err := procGetSystemTimes.Call(uintptr(unsafe.Pointer(&idle)), uintptr(unsafe.Pointer(&kernel)), uintptr(unsafe.Pointer(&user)))
	if r == 0 {
		return nil, err
	}
	return c.delta(filetime(idle), filetime(kernel)+filetime(user))
}

// memoryStatusEx is MEMORYSTATUSEX (64 bytes).
type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

// memProbe reads physical memory (GlobalMemoryStatusEx).
type memProbe struct{}

func (memProbe) sample() (map[string]any, error) {
	var st memoryStatusEx
	st.Length = uint32(unsafe.Sizeof(st))
	r, _, err := procGlobalMemoryStatu.Call(uintptr(unsafe.Pointer(&st)))
	if r == 0 {
		return nil, err
	}
	return memFields(st.TotalPhys, st.AvailPhys), nil
}

func cpuCores() int { return runtime.NumCPU() }
