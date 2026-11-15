package workshop

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// testQueriesRepo is a git repo holding the ix query files of testdata/ix, committed once as testSubject.
func testQueriesRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is needed to watch a repo")
	}
	repo := t.TempDir()
	for _, name := range DefaultQueryFiles {
		raw, err := os.ReadFile(filepath.Join("testdata", "ix", filepath.FromSlash(name)))
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, raw, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	gitIn(t, repo, "init", "-q")
	gitIn(t, repo, "add", ".")
	gitIn(t, repo, "commit", "-q", "-m", testSubject)
	return repo
}

func gitIn(t *testing.T, repo string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false"}, args...)...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

// addQuery appends a numbered query to the streeling file of repo.
func addQuery(t *testing.T, repo, heading, sql string) {
	t.Helper()
	path := filepath.Join(repo, "docs", "streeling", "queries.sql")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.WriteString("\n-- " + heading + "\n" + sql + "\n"); err != nil {
		t.Fatal(err)
	}
}

// testDesktopHTTP is an HTTP client that reaches server under any host name, trusting its certificate: the Desktop
// answers only on its own origin's host.
func testDesktopHTTP(server *httptest.Server) *http.Client {
	pool := x509.NewCertPool()
	pool.AddCert(server.Certificate())
	addr := server.Listener.Addr().String()
	return &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{
		TLSClientConfig: &tls.Config{RootCAs: pool, ServerName: "example.com"},
		DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, addr)
		},
	}}
}

// newTestDesktop is a Desktop on a loopback TLS listener, and a client for it that has no access yet.
func newTestDesktop(t *testing.T) (*Store, *DesktopClient) {
	t.Helper()
	store, err := Open(filepath.Join(t.TempDir(), "watch.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	var handler http.Handler
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { handler.ServeHTTP(w, r) }))
	t.Cleanup(server.Close)
	u, _ := url.Parse(server.URL)
	origin := "https://workshop.invalid:" + u.Port()
	if handler, err = Handler(store, origin); err != nil {
		t.Fatal(err)
	}
	return store, &DesktopClient{Origin: origin, HTTP: testDesktopHTTP(server)}
}

// approve approves the pending device named name, as the owner does on the owner page.
func approve(t *testing.T, store *Store, name string) {
	t.Helper()
	devices, err := store.Devices()
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range devices {
		if d.Name == name && d.Status == "pending" {
			if err := store.Decide(d.ID, "approved"); err != nil {
				t.Fatal(err)
			}
			return
		}
	}
	t.Fatalf("no pending device %q", name)
}

// desktopQueries are the queries widgets the Desktop keeps, with the revision of their project.
func desktopQueries(t *testing.T, store *Store) (widgets []Widget, revisions []int64) {
	t.Helper()
	rows, err := store.db.Query(`SELECT document, revision FROM workshop_projects`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	for rows.Next() {
		var raw string
		var revision int64
		var p Project
		if err := rows.Scan(&raw, &revision); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal([]byte(raw), &p); err != nil {
			t.Fatal(err)
		}
		for _, ws := range p.Workspaces {
			for _, w := range ws.Widgets {
				if w.Kind == "queries" {
					widgets, revisions = append(widgets, w), append(revisions, revision)
				}
			}
		}
	}
	return widgets, revisions
}

// TestTheWatcherKeepsOurQueriesUpToDate: ixforge-queries -watch is a device of its own, paired once and approved by the
// owner. Each step reads the query files and the commit of the ix repo; when they changed, it sends the queries widget
// to the Desktop (creating it the first time, updating it after), and the devices get it at their next sync. It sends
// nothing when nothing changed, and it waits while the query files differ from their commit, so the commit it names is
// always true.
func TestTheWatcherKeepsOurQueriesUpToDate(t *testing.T) {
	ctx := context.Background()
	repo := testQueriesRepo(t)
	store, desktop := newTestDesktop(t)
	invite, err := store.NewInvite(desktop.Origin)
	if err != nil {
		t.Fatal(err)
	}
	if err := desktop.Pair(ctx, invite.Code, "Veilleur requêtes"); err != nil {
		t.Fatal(err)
	}
	if desktop.Token == "" || desktop.StoreID == "" {
		t.Fatalf("pairing must give the watcher its access: %+v", desktop)
	}
	watch := &QueriesWatch{Desktop: desktop, Repo: repo, Files: DefaultQueryFiles, Title: "Requêtes DuckDB", Now: func() time.Time { return testMadeAt }}
	if _, err := watch.Step(ctx); err == nil {
		t.Fatal("a watcher the owner has not approved must not save anything")
	}
	if widgets, _ := desktopQueries(t, store); len(widgets) != 0 {
		t.Fatalf("nothing saved before approval: %+v", widgets)
	}
	approve(t, store, "Veilleur requêtes")

	first := gitIn(t, repo, "rev-parse", "HEAD")
	step, err := watch.Step(ctx)
	if err != nil || !step.Pushed || step.Queries != 15 || step.Commit != first {
		t.Fatalf("the first step sends the 15 queries of %s: %+v %v", first, step, err)
	}
	widgets, revisions := desktopQueries(t, store)
	if len(widgets) != 1 || len(widgets[0].Entries) != 15 || widgets[0].SourceCommit != first || widgets[0].SourceSubject != testSubject || widgets[0].GeneratedAt != "2026-11-15T01:02:03Z" || widgets[0].Title != "Requêtes DuckDB" {
		t.Fatalf("the Desktop keeps the queries widget: %+v", widgets)
	}
	kept := revisions[0]

	if step, err = watch.Step(ctx); err != nil || step.Pushed {
		t.Fatalf("nothing changed, nothing sent: %+v %v", step, err)
	}
	if _, revisions = desktopQueries(t, store); revisions[0] != kept {
		t.Fatalf("an unchanged repo must not make a new revision: %d, was %d", revisions[0], kept)
	}

	addQuery(t, repo, "6) Newest learnings first.", "SELECT * FROM learnings ORDER BY date DESC;")
	gitIn(t, repo, "commit", "-q", "-am", "docs: add a query")
	second := gitIn(t, repo, "rev-parse", "HEAD")
	watch.Now = func() time.Time { return testMadeAt.Add(time.Hour) }
	if step, err = watch.Step(ctx); err != nil || !step.Pushed || step.Queries != 16 || step.Commit != second {
		t.Fatalf("a committed query is sent: %+v %v", step, err)
	}
	widgets, revisions = desktopQueries(t, store)
	if len(widgets) != 1 || len(widgets[0].Entries) != 16 || widgets[0].SourceCommit != second || widgets[0].SourceSubject != "docs: add a query" || widgets[0].GeneratedAt != "2026-11-15T02:02:03Z" || revisions[0] != kept+1 {
		t.Fatalf("the same widget takes the new queries, in one new revision: %+v revision %d", widgets, revisions[0])
	}

	addQuery(t, repo, "7) Not committed yet.", "SELECT 1;")
	if step, err = watch.Step(ctx); err != nil || step.Pushed || step.Waiting != "uncommitted" {
		t.Fatalf("the watcher waits for the commit: %+v %v", step, err)
	}
	if widgets, _ = desktopQueries(t, store); len(widgets[0].Entries) != 16 || widgets[0].SourceCommit != second {
		t.Fatalf("nothing sent while the files differ from their commit: %+v", widgets)
	}
}
