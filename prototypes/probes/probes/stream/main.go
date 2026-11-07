// Command stream watches the scene cast from this machine, every
// PROBE_INTERVAL_MS milliseconds (1000 by default), and prints one JSON line per
// metric. It only talks to loopback: the HLS server (stream_hls_up, 1 when the
// playlist answers 200; stream_hls_ms, the time that took; stream_segments, the
// playlist's segment count), the player page's front door (stream_front_up, 1 when
// anything answers on it) and the viewer count the front door writes to
// viewers.json (stream_viewers, skipped when the file is missing or older than
// 30 s, so a stale count is never reported). It never fetches the public link.
// It exits when it can no longer write to stdout.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
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

var errStdout = errors.New("stdout")

// Loopback only. STREAM_HLS and STREAM_FRONT override the two playlist URLs,
// STREAM_VIEWERS the count file (default: %LOCALAPPDATA%\scene-cast\viewers.json).
func setting(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}

var (
	hlsURL   = setting("STREAM_HLS", "http://127.0.0.1:8890/scene/index.m3u8")
	frontURL = setting("STREAM_FRONT", "http://127.0.0.1:8888/scene/index.m3u8")
	viewers  = setting("STREAM_VIEWERS", filepath.Join(os.Getenv("LOCALAPPDATA"), "scene-cast", "viewers.json"))
	// MediaMTX answers the first playlist request with a redirect to the same path plus
	// ?cookieCheck=1. Redirects are followed only while they stay on loopback, three at most.
	client = &http.Client{Timeout: 3 * time.Second, CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 3 {
			return errors.New("too many redirects")
		}
		if host := req.URL.Hostname(); host != "127.0.0.1" && host != "localhost" && host != "::1" {
			return fmt.Errorf("redirect off loopback to %s refused", host)
		}
		return nil
	}}
)

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
	ts := time.Now().UTC().Format("2006-01-02T15:04:05.000000Z07:00")
	emit := func(metric string, value float64, unit string) error {
		if err := enc.Encode(observation{ts, metric, value, unit}); err != nil {
			return fmt.Errorf("%w: %v", errStdout, err)
		}
		return nil
	}

	start := time.Now()
	code, body := get(hlsURL)
	up := 0.0
	if code == http.StatusOK {
		up = 1
	}
	if err := emit("stream_hls_up", up, ""); err != nil {
		return err
	}
	if code != 0 {
		if err := emit("stream_hls_ms", float64(time.Since(start).Microseconds())/1000, "ms"); err != nil {
			return err
		}
	}
	if up == 1 {
		if err := emit("stream_segments", float64(strings.Count(body, "#EXTINF")), ""); err != nil {
			return err
		}
	}

	// The front door is up when it answers at all: 404 there means no publisher, not no proxy.
	front := 0.0
	if code, _ := get(frontURL); code != 0 {
		front = 1
	}
	if err := emit("stream_front_up", front, ""); err != nil {
		return err
	}

	if n, ok := viewerCount(); ok {
		if err := emit("stream_viewers", n, ""); err != nil {
			return err
		}
	}
	return nil
}

// The last error per URL, so a change in why an endpoint fails is written once to stderr
// (the host keeps it as a `stderr` event) and a steady failure stays quiet.
var lastErr = map[string]string{}

// get returns the status code, 0 when nothing answered, and the body when it was a playlist.
func get(url string) (int, string) {
	resp, err := client.Get(url)
	if err != nil {
		if msg := err.Error(); msg != lastErr[url] {
			lastErr[url] = msg
			fmt.Fprintln(os.Stderr, msg)
		}
		return 0, ""
	}
	lastErr[url] = ""
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return resp.StatusCode, ""
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if err != nil {
		return resp.StatusCode, ""
	}
	return resp.StatusCode, string(b)
}

// viewerCount reads {"at": "<UTC ISO>", "viewers": N}; false when missing, malformed or older than 30 s.
func viewerCount() (float64, bool) {
	raw, err := os.ReadFile(viewers)
	if err != nil {
		return 0, false
	}
	var v struct {
		At      string   `json:"at"`
		Viewers *float64 `json:"viewers"`
	}
	if json.Unmarshal(raw, &v) != nil || v.Viewers == nil {
		return 0, false
	}
	at, err := time.Parse(time.RFC3339, v.At)
	if err != nil || time.Since(at) > 30*time.Second {
		return 0, false
	}
	return *v.Viewers, true
}
