package inventory

import (
	"bytes"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"strconv"
	"strings"
	"time"
)

// clockTicks is USER_HZ, the unit of /proc/<pid>/stat's starttime: 100 on every Linux build Go
// supports (sysconf needs cgo). It bounds the precision of a start time to 10 ms.
const clockTicks = 100

// created gives the start time of process pid from /proc (its starttime after boot, plus the boot
// time btime of /proc/stat), or ErrNoProcess when there is none or it is a zombie.
func created(pid int) (time.Time, error) {
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if errors.Is(err, fs.ErrNotExist) {
		return time.Time{}, ErrNoProcess
	}
	if err != nil {
		return time.Time{}, err
	}
	// The command name, field 2, is in parentheses and may hold spaces: fields count after it.
	end := bytes.LastIndexByte(b, ')')
	if end < 0 {
		return time.Time{}, fmt.Errorf("/proc/%d/stat: no command name", pid)
	}
	f := strings.Fields(string(b[end+1:]))
	if len(f) < 20 { // state is field 3, starttime field 22: f[0] and f[19]
		return time.Time{}, fmt.Errorf("/proc/%d/stat: %d fields", pid, len(f))
	}
	if f[0] == "Z" || f[0] == "X" {
		return time.Time{}, ErrNoProcess
	}
	ticks, err := strconv.ParseInt(f[19], 10, 64)
	if err != nil {
		return time.Time{}, err
	}
	st, err := os.ReadFile("/proc/stat")
	if err != nil {
		return time.Time{}, err
	}
	for _, line := range strings.Split(string(st), "\n") {
		if v, ok := strings.CutPrefix(line, "btime "); ok {
			boot, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
			if err != nil {
				return time.Time{}, err
			}
			return time.Unix(boot, 0).Add(time.Duration(ticks) * time.Second / clockTicks).UTC(), nil
		}
	}
	return time.Time{}, errors.New("/proc/stat: no btime")
}
