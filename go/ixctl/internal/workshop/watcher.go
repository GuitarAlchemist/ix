package workshop

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os/exec"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// A queries widget's source is a repo the Desktop's service cannot read (the user's ix checkout). The watcher runs
// under the user's account instead, as a device of its own: paired once with an invite code and approved by the
// owner, it sends the widget whenever the query files or their commit change. The devices get it at their next
// automatic sync, so an open page shows it without being reloaded.

// DesktopClient talks to the Desktop's API as a paired device, from outside a browser. Origin is the Desktop's
// origin (its API refuses any other host); Token and StoreID come from Pair.
type DesktopClient struct {
	Origin, Token, StoreID string
	HTTP                   *http.Client
}

// APIError is the Desktop's refusal: its HTTP status and code (device_pending, revision_conflict, …).
type APIError struct {
	Status int
	Code   string
}

func (e *APIError) Error() string { return fmt.Sprintf("desktop: %s (%d)", e.Code, e.Status) }

func (c *DesktopClient) call(ctx context.Context, method, path string, body, out any) error {
	var payload []byte
	if body != nil {
		var err error
		if payload, err = json.Marshal(body); err != nil {
			return err
		}
	}
	req, err := http.NewRequestWithContext(ctx, method, strings.TrimRight(c.Origin, "/")+"/workshop/api/"+path, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Origin", c.Origin)
	req.Header.Set("Content-Type", "application/json")
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	client := c.HTTP
	if client == nil {
		client = &http.Client{Timeout: 15 * time.Second}
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		var refusal struct{ Code string }
		_ = json.NewDecoder(resp.Body).Decode(&refusal)
		return &APIError{Status: resp.StatusCode, Code: refusal.Code}
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// Pair claims an invite code under name. The device is pending until the owner approves it.
func (c *DesktopClient) Pair(ctx context.Context, code, name string) error {
	var reply struct {
		Token   string `json:"token"`
		StoreID string `json:"store_id"`
	}
	if err := c.call(ctx, http.MethodPost, "pair", map[string]string{"code": code, "name": name, "store_id": c.StoreID}, &reply); err != nil {
		return err
	}
	if reply.Token == "" || reply.StoreID == "" {
		return errors.New("desktop: pairing answered no access")
	}
	c.Token, c.StoreID = reply.Token, reply.StoreID
	return nil
}

// projects reads every project the Desktop keeps, at its latest revision.
func (c *DesktopClient) projects(ctx context.Context) (map[string]Event, error) {
	latest := map[string]Event{}
	for since := int64(0); ; {
		var changes ChangeSet
		if err := c.call(ctx, http.MethodGet, "changes?since="+strconv.FormatInt(since, 10), nil, &changes); err != nil {
			return nil, err
		}
		if changes.StoreID != c.StoreID {
			return nil, &APIError{Status: http.StatusConflict, Code: "store_changed"}
		}
		for _, e := range changes.Events {
			if e.Revision >= latest[e.Project.ID].Revision {
				latest[e.Project.ID] = e
			}
		}
		if len(changes.Events) < 100 || changes.Cursor <= since {
			return latest, nil
		}
		since = changes.Cursor
	}
}

// QueriesWatch sends the queries widget Title, read from Files of the git repo Repo, to the Desktop.
type QueriesWatch struct {
	Desktop *DesktopClient
	Repo    string
	Files   []string
	Title   string
	Now     func() time.Time
}

// WatchStep is what one step did: Pushed when it sent the widget; otherwise Waiting says why it did not
// ("uncommitted": the query files differ from their commit; empty: the Desktop already has these queries).
type WatchStep struct {
	Pushed  bool
	Waiting string
	Commit  string
	Queries int
}

// errUncommitted: the query files differ from the commit the widget would name.
var errUncommitted = errors.New("the query files differ from their commit; commit them first")

// readRepo reads the query files of a git repo with the commit they come from and its subject. It refuses while the
// files differ from that commit, so the commit a widget names is always true.
func readRepo(repo string, files []string) (entries []QueryEntry, commit, subject string, err error) {
	git := func(args ...string) (string, error) {
		out, err := exec.Command("git", append([]string{"-C", repo}, args...)...).Output()
		return strings.TrimSpace(string(out)), err
	}
	if commit, err = git("rev-parse", "HEAD"); err != nil {
		return nil, "", "", fmt.Errorf("reading the repo's commit: %w", err)
	}
	changed, err := git(append([]string{"status", "--porcelain", "--"}, files...)...)
	if err != nil {
		return nil, commit, "", fmt.Errorf("reading the repo's status: %w", err)
	}
	if changed != "" {
		return nil, commit, "", errUncommitted
	}
	if subject, err = git("log", "-1", "--format=%s"); err != nil {
		return nil, commit, "", fmt.Errorf("reading the commit's subject: %w", err)
	}
	if len(subject) > 240 {
		subject = strings.ToValidUTF8(subject[:237], "") + "…"
	}
	entries, err = ReadQueries(repo, files)
	return entries, commit, subject, err
}

// QueriesWidgetFromRepo is the widget file for the query files of a git repo, made at, to import by hand.
func QueriesWidgetFromRepo(repo string, files []string, title string, at time.Time) (widget []byte, commit string, err error) {
	entries, commit, subject, err := readRepo(repo, files)
	if err != nil {
		return nil, commit, err
	}
	widget, err = QueriesWidget(title, entries, Generation{At: at, Commit: commit, Subject: subject})
	return widget, commit, err
}

// Step reads the repo once and sends the widget if the Desktop does not have these queries from this commit.
func (q *QueriesWatch) Step(ctx context.Context) (WatchStep, error) {
	entries, commit, subject, err := readRepo(q.Repo, q.Files)
	if errors.Is(err, errUncommitted) {
		return WatchStep{Waiting: "uncommitted", Commit: commit}, nil
	}
	if err != nil {
		return WatchStep{}, err
	}
	step := WatchStep{Commit: commit, Queries: len(entries)}

	projects, err := q.Desktop.projects(ctx)
	if err != nil {
		return step, err
	}
	var found *Event
	var at [2]int // workspace, widget
	for id := range projects {
		e := projects[id]
		for i, ws := range e.Project.Workspaces {
			for j, w := range ws.Widgets {
				if w.Kind == "queries" && w.Title == q.Title && (found == nil || e.Project.ID < found.Project.ID) {
					found, at = &e, [2]int{i, j}
				}
			}
		}
	}
	if found != nil {
		w := found.Project.Workspaces[at[0]].Widgets[at[1]]
		if w.SourceCommit == commit && w.SourceSubject == subject && reflect.DeepEqual(w.Entries, entries) {
			return step, nil
		}
	}

	made := Generation{At: q.Now(), Commit: commit, Subject: subject}
	widget := Widget{Title: q.Title, Kind: "queries", Entries: entries, GeneratedAt: made.At.UTC().Format(time.RFC3339), SourceCommit: commit, SourceSubject: subject}
	if !titleOK(q.Title) || !queriesOK(widget) {
		return step, errors.New("queries_invalid: 1 to 100 queries, SQL up to 8 KiB each")
	}
	op := Operation{OpID: newID()}
	if found != nil {
		op.BaseRevision, op.Project = found.Revision, found.Project
		widget.ID = op.Project.Workspaces[at[0]].Widgets[at[1]].ID
		op.Project.Workspaces[at[0]].Widgets[at[1]] = widget
	} else {
		widget.ID = newID()
		op.Project = Project{ID: newID(), Title: q.Title, Workspaces: []Workspace{{ID: newID(), Title: q.Title, Widgets: []Widget{widget}, IXLETDefinitions: []IXLETDefinition{}}}}
	}
	var receipt Receipt
	if err := q.Desktop.call(ctx, http.MethodPost, "operations", op, &receipt); err != nil {
		return step, err
	}
	if receipt.OpID != op.OpID || receipt.Revision != op.BaseRevision+1 {
		return step, errors.New("desktop: unexpected receipt")
	}
	step.Pushed = true
	return step, nil
}

func newID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
