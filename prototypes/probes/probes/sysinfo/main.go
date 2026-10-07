// Command sysinfo reads the machine's memory and processor load from WMI and
// prints one JSON line per metric, every PROBE_INTERVAL_MS milliseconds
// (1000 by default). It exits when it can no longer write to stdout, that is
// when the host that started it is gone.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/yusufpapurcu/wmi"
)

type operatingSystem struct {
	TotalVisibleMemorySize uint64 // KB
	FreePhysicalMemory     uint64 // KB
}

type processor struct {
	LoadPercentage            *uint16 // null while WMI has no sample yet
	NumberOfLogicalProcessors uint32
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
	// A ticker rather than a sleep: a WMI query can take a second under load,
	// and the interval is between samples, not between a sample and the next.
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
	var system []operatingSystem
	q := "SELECT TotalVisibleMemorySize, FreePhysicalMemory FROM Win32_OperatingSystem"
	if err := wmi.Query(q, &system); err != nil {
		return fmt.Errorf("Win32_OperatingSystem: %w", err)
	}
	if len(system) == 0 {
		return errors.New("Win32_OperatingSystem: no instance")
	}
	var cpus []processor
	q = "SELECT LoadPercentage, NumberOfLogicalProcessors FROM Win32_Processor"
	if err := wmi.Query(q, &cpus); err != nil {
		return fmt.Errorf("Win32_Processor: %w", err)
	}
	// Fixed width, unlike RFC3339Nano, so the strings sort in time order and
	// DuckDB reads them as timestamps.
	ts := time.Now().UTC().Format("2006-01-02T15:04:05.000000Z07:00")
	var logical uint32
	var load float64
	var loads int
	for _, c := range cpus {
		logical += c.NumberOfLogicalProcessors
		if c.LoadPercentage != nil {
			load += float64(*c.LoadPercentage)
			loads++
		}
	}
	obs := []observation{
		{ts, "ram_total_mb", float64(system[0].TotalVisibleMemorySize) / 1024, "MB"},
		{ts, "ram_free_mb", float64(system[0].FreePhysicalMemory) / 1024, "MB"},
		{ts, "cpu_logical", float64(logical), "count"},
	}
	if loads > 0 {
		obs = append(obs, observation{ts, "cpu_load_pct", load / float64(loads), "%"})
	}
	for _, o := range obs {
		if err := enc.Encode(o); err != nil {
			return fmt.Errorf("%w: %v", errStdout, err)
		}
	}
	return nil
}
