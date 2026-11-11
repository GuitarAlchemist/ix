package inventory

import (
	"errors"
	"time"

	"golang.org/x/sys/windows"
)

// stillActive is the exit code GetExitCodeProcess gives a process that has not exited.
const stillActive = 259

// created gives the creation time of process pid (GetProcessTimes), or ErrNoProcess when there is
// none: no such pid, or a process that has exited and whose handle another process still holds.
func created(pid int) (time.Time, error) {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		if errors.Is(err, windows.ERROR_INVALID_PARAMETER) {
			return time.Time{}, ErrNoProcess
		}
		return time.Time{}, err
	}
	defer windows.CloseHandle(h)
	var code uint32
	if err := windows.GetExitCodeProcess(h, &code); err != nil {
		return time.Time{}, err
	}
	if code != stillActive {
		return time.Time{}, ErrNoProcess
	}
	var c, x, k, u windows.Filetime
	if err := windows.GetProcessTimes(h, &c, &x, &k, &u); err != nil {
		return time.Time{}, err
	}
	return time.Unix(0, c.Nanoseconds()).UTC(), nil
}
