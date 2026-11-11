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
	"regexp"
	"sort"
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
	// HealthExpect is text the health body must contain, naming the program and, when it
	// matters, its version: a 2xx from another program holding the port after a reboot is not
	// health.
	HealthExpect string `json:"health_expect,omitempty"`
	// HealthFields are the string fields the health body, a JSON object, must carry with exactly
	// these values. For IxForge's /workshop/branding.json: {"name":"IxForge","version":"0.1.1"}.
	// That version is the workshop web client's (the PWA asset), not the ixctl binary's
	// ("ixctl version"): pin it by hand from the deployed branding.json, as listen_image is.
	HealthFields map[string]string `json:"health_fields,omitempty"`
	// Receipt is an update receipt (IxForge's <DataDir>\update-<RunID>\receipt.json), named by
	// the operator. A receipt is a record, never proof of the current state: it can lower the
	// verdict (reconciliation_required is down), never raise it, and only once it is correlated
	// (see receiptUncorrelated): an uncorrelated receipt is unknown, whatever its state.
	Receipt string `json:"receipt,omitempty"`
	// ReceiptCandidateSHA256 is the SHA-256 of the candidate the operator expects installed,
	// required with receipt: a receipt whose CandidateSHA256 differs is about another candidate.
	ReceiptCandidateSHA256 string `json:"receipt_candidate_sha256,omitempty"`
	// Listen is a host:port the component listens on, read from the system's TCP table: no
	// connection is made, so a port nobody may touch can still be observed.
	Listen string `json:"listen,omitempty"`
	// ListenImage is the file name the listening process's executable must have (compared
	// without case, path ignored), such as "delta-remote-v0.6.0-21-g2ca5e6c.exe": which build
	// holds the port, read from the system with no connection. It is a name, not a proof of
	// content (see listenerImage). Windows only.
	ListenImage string `json:"listen_image,omitempty"`
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
	service func(name string) (state, start string, err error)
	// health GETs url (with Host host when set) and returns the first 64 KiB of a 2xx body.
	health    func(url, host string) ([]byte, error)
	listening func(addr string) (bool, error)
	// listenOwner is the PID holding addr in the TCP table, 0 when none.
	listenOwner func(addr string) (uint32, error)
	// imageOf opens a process, which pins its PID until release, and returns its image path.
	imageOf func(pid uint32) (image string, release func(), err error)
	exists  func(path string) error
	// readFile reads a small file (an update receipt), never writes it.
	readFile func(path string) ([]byte, error)
	// listDir names a directory's entries (the other update runs beside a receipt's).
	listDir func(dir string) ([]string, error)
}

// listenerImage is the executable file name of the process holding addr, "" when nobody holds
// it. The PID is read from the table, the process opened (Windows does not reuse a PID while a
// handle to it is open) and its image read, then the table read again: when another PID holds
// the address by then, the answer is an error, never a name that may belong to a process that
// exited and whose PID was reused.
//
// Trust limits: the name is what the file is called, not what it contains (no hash, no
// signature): a renamed program passes. A process that refuses query access (protected, or
// HTTP.sys's PID 4) is an error, so the verdict is unknown.
func listenerImage(p resumeProbes, addr string) (string, error) {
	pid, err := p.listenOwner(addr)
	if err != nil || pid == 0 {
		return "", err
	}
	image, release, err := p.imageOf(pid)
	if err != nil {
		return "", fmt.Errorf("process %d: %w", pid, err)
	}
	defer release()
	again, err := p.listenOwner(addr)
	if err != nil {
		return "", err
	}
	if again != pid {
		return "", fmt.Errorf("the listener changed from process %d to %d while it was read", pid, again)
	}
	return image[strings.LastIndexAny(image, `\/`)+1:], nil
}

func realResumeProbes() resumeProbes { return realResumeProbesOver(nil) }

// realResumeProbesOver is realResumeProbes with health requests sent over rt (nil: the default
// transport), so a test can stall them without a socket.
func realResumeProbesOver(rt http.RoundTripper) resumeProbes {
	return resumeProbes{
		service:     queryService,
		health:      healthProbe(rt, resumeHealthTimeout),
		listening:   tcpListening,
		listenOwner: tcpListenOwner,
		imageOf:     processImage,
		exists: func(p string) error {
			_, err := os.Stat(p)
			return err
		},
		readFile: func(p string) ([]byte, error) {
			f, err := os.Open(p)
			if err != nil {
				return nil, err
			}
			defer f.Close()
			return io.ReadAll(io.LimitReader(f, 1<<20))
		},
		listDir: func(dir string) ([]string, error) {
			entries, err := os.ReadDir(dir)
			names := make([]string, 0, len(entries))
			for _, e := range entries {
				names = append(names, e.Name())
			}
			return names, err
		},
	}
}

// healthFieldsMismatch is why body does not carry want's fields exactly, "" when it does.
func healthFieldsMismatch(body []byte, want map[string]string) string {
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil || got == nil {
		return "body is not a JSON object"
	}
	keys := make([]string, 0, len(want))
	for k := range want {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v, ok := got[k]
		s, isString := v.(string)
		switch {
		case !ok:
			return "body has no " + k
		case !isString:
			return k + " is not a string"
		case s != want[k]:
			return fmt.Sprintf("%s is %q, not %q", k, s, want[k])
		}
	}
	return ""
}

// describeFields prints fields in key order: name "IxForge", version "0.1.1".
func describeFields(fields map[string]string) string {
	parts := make([]string, 0, len(fields))
	for k, v := range fields {
		parts = append(parts, fmt.Sprintf("%s %q", k, v))
	}
	sort.Strings(parts)
	return strings.Join(parts, ", ")
}

// updateReceipt is the part of IxForge's receipt (schema 1) the check correlates and reads.
type updateReceipt struct {
	Schema          int
	RunID           string
	CandidateSHA256 string
	State           string
	UpdatedUTC      string
}

var (
	receiptRunID  = regexp.MustCompile(`^[0-9a-f]{32}$`)
	receiptSHA256 = regexp.MustCompile(`^[0-9a-fA-F]{64}$`)
)

func readReceipt(p resumeProbes, path string) (updateReceipt, error) {
	var rc updateReceipt
	b, err := p.readFile(path)
	if err != nil {
		return rc, err
	}
	if err := json.Unmarshal(b, &rc); err != nil {
		return rc, fmt.Errorf("is not JSON: %w", err)
	}
	return rc, nil
}

// receiptUncorrelated is why a receipt cannot speak for the current install, "" when it can: its
// schema is 1, its RunID names its directory (<DataDir>\update-<RunID>), its candidate is the one
// the inventory expects, and no other run beside it is as recent. Anything it cannot establish
// (an unreadable neighbour, a missing time) is a reason too.
func receiptUncorrelated(c resumeComponent, p resumeProbes, rc updateReceipt) string {
	sep := strings.LastIndexAny(c.Receipt, `\/`)
	if sep < 0 {
		return "it is not inside an update-<RunID> directory"
	}
	dir := c.Receipt[:sep]
	parentSep := strings.LastIndexAny(dir, `\/`)
	dirName := dir[parentSep+1:]
	switch {
	case rc.Schema != 1:
		return fmt.Sprintf("schema %d, not 1", rc.Schema)
	case !receiptRunID.MatchString(rc.RunID):
		return fmt.Sprintf("RunID %q is not 32 lowercase hex digits", rc.RunID)
	case !strings.EqualFold(dirName, "update-"+rc.RunID):
		return fmt.Sprintf("RunID %s does not name its directory %s", rc.RunID, dirName)
	case !strings.EqualFold(rc.CandidateSHA256, c.ReceiptCandidateSHA256):
		return "it is about another candidate than receipt_candidate_sha256"
	case parentSep < 0:
		return "no data directory to compare its run with the others"
	}
	mine, err := time.Parse(time.RFC3339Nano, rc.UpdatedUTC)
	if err != nil {
		return fmt.Sprintf("UpdatedUTC %q is not a time", rc.UpdatedUTC)
	}
	parent, s := dir[:parentSep], string(c.Receipt[sep])
	names, err := p.listDir(parent)
	if err != nil {
		return "the other runs cannot be listed: " + err.Error()
	}
	sort.Strings(names)
	for _, n := range names {
		if !strings.HasPrefix(strings.ToLower(n), "update-") || strings.EqualFold(n, dirName) {
			continue
		}
		other, err := readReceipt(p, parent+s+n+s+"receipt.json")
		if err != nil {
			return fmt.Sprintf("run %s cannot be compared: %v", n, err)
		}
		if t, err := time.Parse(time.RFC3339Nano, other.UpdatedUTC); err != nil || !t.Before(mine) {
			return fmt.Sprintf("run %s is as recent or newer", n)
		}
	}
	return ""
}

// receiptVerdict reads a correlated update receipt's State: reconciliation_required is down, a
// state short of completed or rolled_back is degraded, an unknown state is unknown. completed
// and rolled_back add a line only: a record never raises the verdict. An unreadable or
// uncorrelated receipt is unknown, whatever its state.
func receiptVerdict(c resumeComponent, p resumeProbes) (verdict, line string) {
	rc, err := readReceipt(p, c.Receipt)
	if err != nil {
		return verdictUnknown, err.Error()
	}
	if why := receiptUncorrelated(c, p, rc); why != "" {
		return verdictUnknown, fmt.Sprintf("not correlated with the current install (%s): its state %q is not applied", why, rc.State)
	}
	switch rc.State {
	case "completed", "rolled_back":
		return verdictOK, "state " + rc.State + " (a record, not proof of the current state)"
	case "reconciliation_required":
		return verdictDown, "state reconciliation_required: identity or health not proven, a human must reconcile"
	case "available", "backed_up", "installed", "failed_before_replacement":
		return verdictDegraded, "state " + rc.State + ": the update did not reach an end state"
	}
	return verdictUnknown, fmt.Sprintf("unknown state %q", rc.State)
}

// resumeHealthTimeout bounds one health probe: connection, redirects, headers and body. A server
// that accepts and never answers, or stalls inside its body, is down after it, never ok.
const resumeHealthTimeout = 2 * time.Second

// healthProbe GETs a health URL over rt (nil: the default transport) within the given time (the
// check passes resumeHealthTimeout; tests pass a shorter one). A redirect is followed only while
// it stays on loopback (at most 5): the inventory's loopback promise holds for every hop, not
// only the first.
func healthProbe(rt http.RoundTripper, within time.Duration) func(u, host string) ([]byte, error) {
	client := &http.Client{Timeout: within, Transport: rt,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if err := loopbackURL(req.URL.String()); err != nil {
				return fmt.Errorf("redirect to %s leaves the loopback", req.URL.Redacted())
			}
			if len(via) >= 5 {
				return errors.New("more than 5 redirects")
			}
			return nil
		}}
	return func(u, host string) ([]byte, error) {
		req, err := http.NewRequest(http.MethodGet, u, nil)
		if err != nil {
			return nil, err
		}
		if host != "" {
			req.Host = host
		}
		resp, err := client.Do(req)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		if resp.StatusCode/100 != 2 {
			return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
		}
		return body, err
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
		} else if c.HealthHost != "" || c.HealthExpect != "" || c.HealthFields != nil {
			return nil, fmt.Errorf("inventory: %s has health_host, health_expect or health_fields without health", c.Name)
		}
		if c.HealthFields != nil && len(c.HealthFields) == 0 {
			return nil, fmt.Errorf("inventory: %s health_fields is empty", c.Name)
		}
		for k, v := range c.HealthFields {
			if k == "" || v == "" {
				return nil, fmt.Errorf("inventory: %s health_fields %s is empty", c.Name, k)
			}
		}
		if c.Listen != "" {
			if _, _, err := net.SplitHostPort(c.Listen); err != nil {
				return nil, fmt.Errorf("inventory: %s listen: %w", c.Name, err)
			}
		}
		switch {
		case c.ListenImage != "" && c.Listen == "":
			return nil, fmt.Errorf("inventory: %s has listen_image without listen", c.Name)
		case strings.ContainsAny(c.ListenImage, `\/`):
			return nil, fmt.Errorf("inventory: %s listen_image is a file name, not a path", c.Name)
		case c.Receipt != "" && !receiptSHA256.MatchString(c.ReceiptCandidateSHA256):
			return nil, fmt.Errorf("inventory: %s receipt needs receipt_candidate_sha256, 64 hex digits", c.Name)
		case c.Receipt == "" && c.ReceiptCandidateSHA256 != "":
			return nil, fmt.Errorf("inventory: %s has receipt_candidate_sha256 without receipt", c.Name)
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
				if c.ListenImage != "" {
					image, err := listenerImage(p, c.Listen)
					switch {
					case err != nil:
						worse(verdictUnknown, fmt.Sprintf("listener %s image: %v", c.Listen, err))
					case image == "":
						worse(verdictUnknown, fmt.Sprintf("the listener on %s was gone when its image was read", c.Listen))
					case !strings.EqualFold(image, c.ListenImage):
						worse(verdictDown, fmt.Sprintf("%s is held by %s, not %s", c.Listen, image, c.ListenImage))
					default:
						worse(verdictOK, fmt.Sprintf("%s is held by %s", c.Listen, image))
					}
				}
			}
		}
		if c.Health != "" {
			if body, err := p.health(c.Health, c.HealthHost); err != nil {
				worse(verdictDown, fmt.Sprintf("health %s: %v", c.Health, err))
			} else if c.HealthExpect != "" && !strings.Contains(string(body), c.HealthExpect) {
				worse(verdictDown, fmt.Sprintf("health %s answers, but its body lacks %q: another program or version holds it", c.Health, c.HealthExpect))
			} else if c.HealthFields != nil {
				if why := healthFieldsMismatch(body, c.HealthFields); why != "" {
					worse(verdictDown, fmt.Sprintf("health %s answers, but its %s: another program or version holds it", c.Health, why))
				} else {
					worse(verdictOK, fmt.Sprintf("health %s answers with %s", c.Health, describeFields(c.HealthFields)))
				}
			} else {
				worse(verdictOK, fmt.Sprintf("health %s answers", c.Health))
			}
		}
		if c.Receipt != "" {
			v, line := receiptVerdict(c, p)
			worse(v, fmt.Sprintf("update receipt %s: %s", c.Receipt, line))
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
