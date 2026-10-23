// Command disk reads the free and total space of the machine's local fixed
// drives from WMI and prints one JSON line per metric and drive, every
// PROBE_INTERVAL_MS milliseconds (1000 by default): disk_c_free_gb,
// disk_c_total_gb. It exits when it can no longer write to stdout.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/yusufpapurcu/wmi"
)

type logicalDisk struct {
	DeviceID  string  // "C:"
	FreeSpace *uint64 // bytes; null for a drive with no medium
	Size      *uint64
}

type observation struct {
	TS     string  `json:"ts"`
	Metric string  `json:"metric"`
	Value  float64 `json:"value"`
	Unit   string  `json:"unit"`
}

var errStdout = errors.New("stdout")

func main() {
	every := time.Second
	if ms, err := strconv.Atoi(os.Getenv("PROBE_INTERVAL_MS")); err == nil && ms > 0 {
		every = time.Duration(ms) * time.Millisecond
	}
	enc := json.NewEncoder(os.Stdout)
	tick := time.NewTicker(every)
	for ; ; <-tick.C {
		if err := sample(enc); err != nil {
			fmt.Fprintln(os.Stderr, err)
			if errors.Is(err, errStdout) {
				os.Exit(1)
			}
		}
	}
}

func sample(enc *json.Encoder) error {
	var disks []logicalDisk
	// DriveType 3: local fixed disks, not network shares or removable media.
	if err := wmi.Query("SELECT DeviceID, FreeSpace, Size FROM Win32_LogicalDisk WHERE DriveType = 3", &disks); err != nil {
		return fmt.Errorf("Win32_LogicalDisk: %w", err)
	}
	ts := time.Now().UTC().Format("2006-01-02T15:04:05.000000Z07:00")
	const gb = 1 << 30
	for _, d := range disks {
		if d.FreeSpace == nil || d.Size == nil {
			continue
		}
		name := "disk_" + strings.ToLower(strings.TrimSuffix(d.DeviceID, ":"))
		for _, o := range []observation{
			{ts, name + "_free_gb", float64(*d.FreeSpace) / gb, "GB"},
			{ts, name + "_total_gb", float64(*d.Size) / gb, "GB"},
		} {
			if err := enc.Encode(o); err != nil {
				return fmt.Errorf("%w: %v", errStdout, err)
			}
		}
	}
	return nil
}
