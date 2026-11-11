package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"text/tabwriter"
	"time"
)

// `ixctl resume check` answers, after a reboot or a power cut, "what came back, what did not,
// whose is it and what does a human run": it reads an inventory of components in start order,
// observes each one and prints a verdict per component. It starts nothing, stops nothing and
// replays nothing: a component whose outcome is unknown stays reported as unknown, and its remedy
// is text for a human, never a command ixctl runs.

// resumeInventory is the components to observe, each after the ones it names in after.
type resumeInventory struct {
	Components []resumeComponent `json:"components"`
}

type resumeComponent struct {
	Name  string   `json:"name"`
	After []string `json:"after,omitempty"`
	// Service is a Windows service name: its state, and whether its start type brings it back
	// after a reboot.
	Service string `json:"service,omitempty"`
	// Health is a loopback http(s) URL that answers 2xx once the component is ready.
	Health string `json:"health,omitempty"`
	// HealthHost is the Host header the health request carries, for a server that gates on it
	// (IxForge's workshop answers 403 to any other host).
	HealthHost string `json:"health_host,omitempty"`
	// Listen is a host:port the component listens on, read from the system's TCP table: no
	// connection is made, so a port nobody may touch can still be observed.
	Listen string `json:"listen,omitempty"`
	// Durable is where the component's state lives (projects, mandates, receipts): each path is
	// reported present or missing, never opened or changed.
	Durable []string `json:"durable,omitempty"`
	Owner   string   `json:"owner,omitempty"`
	Remedy  string   `json:"remedy,omitempty"`
}

// resumeProbes is how a check observes the machine; tests replace it.
type resumeProbes struct {
	// service is a Windows service's state ("running", "stopped", …) and start type
	// ("automatic", "automatic (delayed)", "manual", "disabled").
	service   func(name string) (state, start string, err error)
	health    func(url, host string) error
	listening func(addr string) (bool, error)
	exists    func(path string) error
}

func realResumeProbes() resumeProbes {
	client := &http.Client{Timeout: 2 * time.Second}
	return resumeProbes{
		service: queryService,
		health: func(u, host string) error {
			req, err := http.NewRequest(http.MethodGet, u, nil)
			if err != nil {
				return err
			}
			if host != "" {
				req.Host = host
			}
			resp, err := client.Do(req)
			if err != nil {
				return err
			}
			defer resp.Body.Close()
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
			if resp.StatusCode/100 != 2 {
				return fmt.Errorf("HTTP %d", resp.StatusCode)
			}
			return nil
		},
		listening: tcpListening,
		exists: func(p string) error {
			_, err := os.Stat(p)
			return err
		},
	}
}

type resumeResult struct {
	Order     int      `json:"order"`
	Name      string   `json:"name"`
	Verdict   string   `json:"verdict"` // ok, degraded, down or unknown
	Checks    []string `json:"checks"`
	BlockedBy []string `json:"blocked_by,omitempty"`
	Owner     string   `json:"owner,omitempty"`
	Remedy    string   `json:"remedy,omitempty"`
}

const (
	verdictOK       = "ok"
	verdictDegraded = "degraded" // up now, but will not come back by itself
	verdictDown     = "down"
	verdictUnknown  = "unknown" // could not be observed
)

var verdictRank = map[string]int{verdictOK: 0, verdictDegraded: 1, verdictUnknown: 2, verdictDown: 3}

func parseResumeInventory(b []byte) (*resumeInventory, error) {
	var inv resumeInventory
	dec := json.NewDecoder(strings.NewReader(string(b)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&inv); err != nil {
		return nil, fmt.Errorf("inventory: %w", err)
	}
	if len(inv.Components) == 0 {
		return nil, errors.New("inventory: no components")
	}
	seen := map[string]bool{}
	for _, c := range inv.Components {
		switch {
		case c.Name == "":
			return nil, errors.New("inventory: a component has no name")
		case seen[c.Name]:
			return nil, fmt.Errorf("inventory: %s is declared twice", c.Name)
		case c.Service == "" && c.Health == "" && c.Listen == "":
			return nil, fmt.Errorf("inventory: %s has nothing to observe (service, health or listen)", c.Name)
		}
		seen[c.Name] = true
		if c.Health != "" {
			if err := loopbackURL(c.Health); err != nil {
				return nil, fmt.Errorf("inventory: %s health: %w", c.Name, err)
			}
		} else if c.HealthHost != "" {
			return nil, fmt.Errorf("inventory: %s has health_host without health", c.Name)
		}
		if c.Listen != "" {
			if _, _, err := net.SplitHostPort(c.Listen); err != nil {
				return nil, fmt.Errorf("inventory: %s listen: %w", c.Name, err)
			}
		}
	}
	for _, c := range inv.Components {
		for _, a := range c.After {
			if !seen[a] {
				return nil, fmt.Errorf("inventory: %s starts after %s, which is not declared", c.Name, a)
			}
		}
	}
	if _, err := resumeOrder(inv.Components); err != nil {
		return nil, err
	}
	return &inv, nil
}

// loopbackURL refuses a health URL that would leave the machine.
func loopbackURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return err
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("%q is not http(s)", raw)
	}
	host := u.Hostname()
	if ip := net.ParseIP(host); host != "localhost" && (ip == nil || !ip.IsLoopback()) {
		return fmt.Errorf("%q is not a loopback address", raw)
	}
	return nil
}

// resumeOrder is the start order: each component after the ones it names, ties kept in
// inventory order. A cycle is refused, since no order would start it.
func resumeOrder(cs []resumeComponent) ([]resumeComponent, error) {
	placed := map[string]bool{}
	var out []resumeComponent
	for len(out) < len(cs) {
		progress := false
		for _, c := range cs {
			if placed[c.Name] {
				continue
			}
			ready := true
			for _, a := range c.After {
				ready = ready && placed[a]
			}
			if ready {
				placed[c.Name] = true
				out = append(out, c)
				progress = true
			}
		}
		if !progress {
			var left []string
			for _, c := range cs {
				if !placed[c.Name] {
					left = append(left, c.Name)
				}
			}
			return nil, fmt.Errorf("inventory: %s depend on each other in a cycle", strings.Join(left, ", "))
		}
	}
	return out, nil
}

// checkResume observes every component in start order.
func checkResume(inv *resumeInventory, p resumeProbes) []resumeResult {
	order, _ := resumeOrder(inv.Components) // validated by parseResumeInventory
	verdicts := map[string]string{}
	var results []resumeResult
	for i, c := range order {
		r := resumeResult{Order: i + 1, Name: c.Name, Verdict: verdictOK, Owner: c.Owner, Remedy: c.Remedy}
		worse := func(v, line string) {
			if verdictRank[v] > verdictRank[r.Verdict] {
				r.Verdict = v
			}
			r.Checks = append(r.Checks, line)
		}
		if c.Service != "" {
			state, start, err := p.service(c.Service)
			switch {
			case err != nil:
				worse(verdictUnknown, fmt.Sprintf("service %s: %v", c.Service, err))
			case state != "running":
				worse(verdictDown, fmt.Sprintf("service %s is %s (start type %s)", c.Service, state, start))
			case !strings.HasPrefix(start, "automatic"):
				worse(verdictDegraded, fmt.Sprintf("service %s is running, start type %s: it will not start after a reboot", c.Service, start))
			default:
				worse(verdictOK, fmt.Sprintf("service %s is running, start type %s", c.Service, start))
			}
		}
		if c.Listen != "" {
			on, err := p.listening(c.Listen)
			switch {
			case err != nil:
				worse(verdictUnknown, fmt.Sprintf("listener %s: %v", c.Listen, err))
			case !on:
				worse(verdictDown, fmt.Sprintf("nothing listens on %s (a client gets connection refused)", c.Listen))
			default:
				worse(verdictOK, fmt.Sprintf("a listener holds %s", c.Listen))
			}
		}
		if c.Health != "" {
			if err := p.health(c.Health, c.HealthHost); err != nil {
				worse(verdictDown, fmt.Sprintf("health %s: %v", c.Health, err))
			} else {
				worse(verdictOK, fmt.Sprintf("health %s answers", c.Health))
			}
		}
		for _, d := range c.Durable {
			if err := p.exists(d); err != nil {
				r.Checks = append(r.Checks, fmt.Sprintf("durable state %s is missing: %v", d, err))
			} else {
				r.Checks = append(r.Checks, fmt.Sprintf("durable state %s is present", d))
			}
		}
		for _, a := range c.After {
			if verdicts[a] == verdictDown || verdicts[a] == verdictUnknown {
				r.BlockedBy = append(r.BlockedBy, a)
			}
		}
		verdicts[c.Name] = r.Verdict
		results = append(results, r)
	}
	return results
}

func cmdResume(args []string, stdout io.Writer, p resumeProbes) (ready bool, err error) {
	if len(args) == 0 || args[0] != "check" {
		return false, errors.New("usage: ixctl resume check <inventory.json> [--json]")
	}
	fs := flag.NewFlagSet("resume check", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "print the results as JSON")
	pos, err := parseArgs(fs, args[1:])
	if err != nil {
		return false, err
	}
	if len(pos) != 1 {
		return false, errors.New("usage: ixctl resume check <inventory.json> [--json]")
	}
	b, err := os.ReadFile(pos[0])
	if err != nil {
		return false, err
	}
	inv, err := parseResumeInventory(b)
	if err != nil {
		return false, err
	}
	results := checkResume(inv, p)
	ready = true
	for _, r := range results {
		ready = ready && r.Verdict == verdictOK
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return ready, enc.Encode(map[string]any{"ready": ready, "components": results})
	}
	writeResume(stdout, results, ready)
	return ready, nil
}

func writeResume(w io.Writer, results []resumeResult, ready bool) {
	notReady := 0
	for _, r := range results {
		if r.Verdict != verdictOK {
			notReady++
		}
	}
	fmt.Fprintf(w, "resume check: %d components, %d not ready\n", len(results), notReady)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	for _, r := range results {
		fmt.Fprintf(tw, "%d\t%s\t%s\t%s\n", r.Order, r.Name, r.Verdict, r.Checks[0])
		for _, c := range r.Checks[1:] {
			fmt.Fprintf(tw, "\t\t\t%s\n", c)
		}
		if len(r.BlockedBy) > 0 {
			fmt.Fprintf(tw, "\t\t\tstarts after %s, which is not up\n", strings.Join(r.BlockedBy, ", "))
		}
		if r.Verdict != verdictOK {
			if r.Owner != "" {
				fmt.Fprintf(tw, "\t\t\towner: %s\n", r.Owner)
			}
			if r.Remedy != "" {
				fmt.Fprintf(tw, "\t\t\tremedy (run by a human): %s\n", r.Remedy)
			}
		}
	}
	_ = tw.Flush()
	if ready {
		fmt.Fprintln(w, "ready")
	}
}
