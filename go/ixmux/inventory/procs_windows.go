package inventory

import (
	"errors"
	"time"

	"golang.org/x/sys/windows"
)

// created gives the creation time of process pid (GetProcessTimes), or ErrNoProcess when there is
// none: no such pid, or a process that has exited and whose handle another process still holds
// (its handle is signalled; an exit code of 259 would not tell, a process may exit with it).
func created(pid int) (time.Time, error) {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION|windows.SYNCHRONIZE, false, uint32(pid))
	if err != nil {
		if errors.Is(err, windows.ERROR_INVALID_PARAMETER) {
			return time.Time{}, ErrNoProcess
		}
		return time.Time{}, err
	}
	defer windows.CloseHandle(h)
	switch ev, err := windows.WaitForSingleObject(h, 0); {
	case err != nil:
		return time.Time{}, err
	case ev == windows.WAIT_OBJECT_0:
		return time.Time{}, ErrNoProcess
	case ev != uint32(windows.WAIT_TIMEOUT):
		return time.Time{}, errors.New("cannot tell whether the process has exited")
	}
	var c, x, k, u windows.Filetime
	if err := windows.GetProcessTimes(h, &c, &x, &k, &u); err != nil {
		return time.Time{}, err
	}
	return time.Unix(0, c.Nanoseconds()).UTC(), nil
}
