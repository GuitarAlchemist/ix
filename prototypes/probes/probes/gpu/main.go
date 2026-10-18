// Command gpu reads the NVIDIA GPUs' load, memory, temperature and power from
// nvidia-smi and prints one JSON line per metric and GPU, every
// PROBE_INTERVAL_MS milliseconds (1000 by default). With more than one GPU the
// metric names carry the GPU's index, gpu1_util_pct; the first keeps the plain
// name. It exits when it can no longer write to stdout.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

type observation struct {
	TS     string  `json:"ts"`
	Metric string  `json:"metric"`
	Value  float64 `json:"value"`
	Unit   string  `json:"unit"`
}

// The nvidia-smi fields, in query order, with the metric and unit each becomes.
var fields = []struct{ query, metric, unit string }{
	{"utilization.gpu", "util_pct", "%"},
	{"memory.used", "mem_used_mb", "MB"},
	{"memory.total", "mem_total_mb", "MB"},
	{"temperature.gpu", "temp_c", "C"},
	{"power.draw", "power_w", "W"},
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
	queries := make([]string, len(fields))
	for i, f := range fields {
		queries[i] = f.query
	}
	out, err := exec.Command("nvidia-smi", "--query-gpu="+strings.Join(queries, ","),
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return fmt.Errorf("nvidia-smi: %w", err)
	}
	ts := time.Now().UTC().Format("2006-01-02T15:04:05.000000Z07:00")
	for gpu, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		prefix := "gpu_"
		if gpu > 0 {
			prefix = fmt.Sprintf("gpu%d_", gpu)
		}
		for i, cell := range strings.Split(line, ",") {
			if i >= len(fields) {
				break
			}
			// "[N/A]" or "[Not Supported]" when a board does not report a field.
			v, err := strconv.ParseFloat(strings.TrimSpace(cell), 64)
			if err != nil {
				continue
			}
			o := observation{ts, prefix + fields[i].metric, v, fields[i].unit}
			if err := enc.Encode(o); err != nil {
				return fmt.Errorf("%w: %v", errStdout, err)
			}
		}
	}
	return nil
}
