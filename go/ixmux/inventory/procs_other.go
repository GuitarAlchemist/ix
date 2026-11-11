//go:build !windows && !linux

package inventory

import (
	"errors"
	"time"
)

// created is not implemented on this system: every liveness is unverified.
func created(pid int) (time.Time, error) {
	return time.Time{}, errors.New("process start times are not read on this system")
}
