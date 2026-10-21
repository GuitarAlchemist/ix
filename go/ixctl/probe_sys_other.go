//go:build !windows

package main

import (
	"errors"
	"os"
	"runtime"
	"strconv"
	"strings"
)

// cpuProbe reads the host's processor time from the `cpu` line of /proc/stat (user, nice,
// system, idle, iowait, irq, softirq, steal, in clock ticks, summed over every processor)
// and reports the busy share of the last interval; idle and iowait count as idle.
type cpuProbe struct {
	idle, total uint64
	primed      bool
}

func (c *cpuProbe) sample() (map[string]any, error) {
	b, err := os.ReadFile("/proc/stat")
	if err != nil {
		return nil, err
	}
	line, _, _ := strings.Cut(string(b), "\n")
	f := strings.Fields(line)
	if len(f) < 5 || f[0] != "cpu" {
		return nil, errors.New("/proc/stat: no cpu line")
	}
	var vals []uint64
	for _, s := range f[1:] {
		n, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			return nil, err
		}
		vals = append(vals, n)
	}
	// user nice system idle iowait irq softirq steal [guest guest_nice]: the last two are
	// already inside user and nice, so the total stops at steal.
	if len(vals) > 8 {
		vals = vals[:8]
	}
	var total uint64
	for _, v := range vals {
		total += v
	}
	idle := vals[3]
	if len(vals) > 4 {
		idle += vals[4]
	}
	return c.delta(idle, total)
}

// memProbe reads physical memory from /proc/meminfo (MemTotal, MemAvailable, in kB).
type memProbe struct{}

func (memProbe) sample() (map[string]any, error) {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return nil, err
	}
	var total, avail uint64
	for _, line := range strings.Split(string(b), "\n") {
		key, rest, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		f := strings.Fields(rest)
		if len(f) == 0 {
			continue
		}
		n, err := strconv.ParseUint(f[0], 10, 64)
		if err != nil {
			continue
		}
		switch key {
		case "MemTotal":
			total = n * 1024
		case "MemAvailable":
			avail = n * 1024
		}
	}
	if total == 0 || avail == 0 {
		return nil, errors.New("/proc/meminfo: MemTotal or MemAvailable missing")
	}
	return memFields(total, avail), nil
}

func cpuCores() int { return runtime.NumCPU() }
