package repograph

import (
	"encoding/csv"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// gitRepo makes a repo whose commits change the given files, oldest first.
func gitRepo(t *testing.T, commits ...[]string) string {
	t.Helper()
	dir := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-c", "user.name=t", "-c", "user.email=t@t.invalid", "-c", "commit.gpgsign=false"}, args...)...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	git("init", "-q")
	for i, files := range commits {
		for _, f := range files {
			p := filepath.Join(dir, filepath.FromSlash(f))
			os.MkdirAll(filepath.Dir(p), 0o755)
			old, _ := os.ReadFile(p)
			os.WriteFile(p, append(old, byte('a'+i)), 0o644)
		}
		git("add", "-A")
		git("commit", "-q", "-m", "commit "+string(rune('A'+i)))
	}
	return dir
}

func rows(t *testing.T, path string) [][]string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	all, err := csv.NewReader(f).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	return all[1:] // header
}

func pairs(t *testing.T, path string) []string {
	t.Helper()
	var out []string
	for _, r := range rows(t, path) {
		out = append(out, r[0]+">"+r[1])
	}
	sort.Strings(out)
	return out
}

// The pull requests and issues of a repo as `gh pr list` and `gh issue list` write them (--json).
const ghPRs = `[
 {"number":7,"title":"fix a","state":"OPEN","headRefName":"fix/a","updatedAt":"2026-11-16T10:00:00Z","files":[{"path":"go/a.go"},{"path":"go/new.go"}],
  "closingIssuesReferences":[{"number":3,"repository":{"name":"ix"}},{"number":9,"repository":{"name":"tars"}}]},
 {"number":8,"title":"also a","state":"OPEN","headRefName":"feat/b","updatedAt":"2026-11-16T11:00:00Z","files":[{"path":"go/a.go"}],"closingIssuesReferences":[]}
]`
const ghIssues = `[{"number":3,"title":"a.go is slow","state":"OPEN","updatedAt":"2026-11-15T09:00:00Z","body":"See go/a.go and web/c.js; nothing.md"}]`

// TestTheGraphLinksLearningsPullRequestsAndIssuesToTheFilesOfTheirRepo: the graph holds each learning with its repo and
// tags; the commits of every repo with the files each changed (a commit changing more files than the bound is left
// out: a rename or a vendored drop says nothing about what changes together); a MENTIONS edge from a learning to each
// file of its own repo its text names, by its path or by a base name no other file of that repo has; each pull request
// with the files it touches and the issues of the graph it closes; each issue with the files its text names. A file is
// known by its repo and its path, so ix's go/a.go and ga's go/a.go are two files.
func TestTheGraphLinksLearningsPullRequestsAndIssuesToTheFilesOfTheirRepo(t *testing.T) {
	ix := gitRepo(t,
		[]string{"go/a.go", "go/b.go"},
		[]string{"go/a.go", "web/c.js"},
		[]string{"x/1.go", "x/2.go", "x/3.go", "x/4.go"}, // over the bound of 3
		[]string{"docs/solutions/l1.md", "docs/solutions/l2.md"},
	)
	ga := gitRepo(t, []string{"go/a.go", "src/Chord.cs"})
	os.WriteFile(filepath.Join(ix, "docs", "solutions", "l1.md"), []byte("The fix is in a.go; web/c.js reads it. Not util.go.\n"), 0o644)
	os.WriteFile(filepath.Join(ix, "docs", "solutions", "l2.md"), []byte("Nothing named here.\n"), 0o644)
	catalog := filepath.Join(t.TempDir(), "catalog.jsonl")
	os.WriteFile(catalog, []byte(strings.Join([]string{
		`{"id":"ix:docs/solutions/l1.md","repo":"ix","kind":"solution","category":"ci","title":"One, \"quoted\"","date":"2026-11-01","tags":["windows","fsync"],"path":"docs/solutions/l1.md"}`,
		`{"id":"ix:docs/solutions/l2.md","repo":"ix","kind":"plan","category":"uncategorized","title":"Two","tags":["fsync","go"],"path":"docs/solutions/l2.md"}`,
		`{"id":"ga:docs/solutions/g.md","repo":"ga","kind":"solution","category":"ci","title":"Three","path":"docs/solutions/g.md"}`,
	}, "\n")+"\n"), 0o644)

	learnings, err := ReadCatalog(catalog)
	if err != nil || len(learnings) != 3 {
		t.Fatalf("catalog: %d learnings, %v", len(learnings), err)
	}
	commits, err := ReadCommits(ix, "ix", 100, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(commits) != 3 {
		t.Fatalf("%d commits kept, want 3 (the one over the bound left out)", len(commits))
	}
	// Commit times are UTC to the second, so that "the last 30 commits" sorts the same in every engine.
	if at, err := time.Parse(time.RFC3339, commits[0].Date); err != nil || !strings.HasSuffix(commits[0].Date, "Z") || at.IsZero() {
		t.Errorf("commit date %q: want a UTC RFC 3339 time", commits[0].Date)
	}
	gaCommits, err := ReadCommits(ga, "ga", 100, 3)
	if err != nil || len(gaCommits) != 1 || gaCommits[0].Repo != "ga" {
		t.Fatalf("ga commits %+v, %v", gaCommits, err)
	}
	prs, err := ParsePRs("ix", []byte(ghPRs))
	if err != nil {
		t.Fatal(err)
	}
	issues, err := ParseIssues("ix", []byte(ghIssues))
	if err != nil {
		t.Fatal(err)
	}
	out := t.TempDir()
	if err := Export(out, Data{
		Roots:     map[string]string{"ix": ix, "ga": ga},
		Learnings: learnings,
		Commits:   append(commits, gaCommits...),
		PRs:       prs,
		Issues:    issues,
	}); err != nil {
		t.Fatal(err)
	}

	if got := pairs(t, filepath.Join(out, "tagged.csv")); strings.Join(got, " ") != "ix:docs/solutions/l1.md>fsync ix:docs/solutions/l1.md>windows ix:docs/solutions/l2.md>fsync ix:docs/solutions/l2.md>go" {
		t.Errorf("tagged %v", got)
	}
	if got := pairs(t, filepath.Join(out, "in_repo.csv")); len(got) != 3 || got[0] != "ga:docs/solutions/g.md>ga" {
		t.Errorf("in_repo %v", got)
	}
	if got := pairs(t, filepath.Join(out, "mentions.csv")); strings.Join(got, " ") != "ix:docs/solutions/l1.md>ix:go/a.go ix:docs/solutions/l1.md>ix:web/c.js" {
		t.Errorf("mentions %v, want ix's a.go by its unique base name in ix (ga has one too) and web/c.js by its path", got)
	}
	if changed := pairs(t, filepath.Join(out, "changed.csv")); len(changed) != 8 {
		t.Errorf("changed %v, want the 6 file changes of ix's 3 kept commits and the 2 of ga's", changed)
	}
	files := map[string]string{}
	for _, r := range rows(t, filepath.Join(out, "file.csv")) {
		files[r[0]] = r[1] + "|" + r[2]
	}
	if files["ga:go/a.go"] != "ga|go/a.go" || files["ix:go/a.go"] != "ix|go/a.go" || files["ix:go/new.go"] != "ix|go/new.go" {
		t.Errorf("files %v: one per repo and path, with the files a pull request adds", files)
	}
	if got := pairs(t, filepath.Join(out, "touches.csv")); strings.Join(got, " ") != "ix#7>ix:go/a.go ix#7>ix:go/new.go ix#8>ix:go/a.go" {
		t.Errorf("touches %v", got)
	}
	if got := pairs(t, filepath.Join(out, "closes.csv")); strings.Join(got, " ") != "ix#7>ix#3" {
		t.Errorf("closes %v, want only the issue of the graph (tars#9 is not in it)", got)
	}
	if got := pairs(t, filepath.Join(out, "names.csv")); strings.Join(got, " ") != "ix#3>ix:go/a.go ix#3>ix:web/c.js" {
		t.Errorf("names %v", got)
	}
	for _, r := range rows(t, filepath.Join(out, "pr.csv")) {
		if r[0] == "ix#7" && strings.Join(r[1:], "|") != "ix|7|fix a|OPEN|fix/a|2026-11-16T10:00:00Z" {
			t.Errorf("pr %v", r)
		}
	}
	for _, r := range rows(t, filepath.Join(out, "learning.csv")) {
		if r[0] == "ix:docs/solutions/l1.md" && r[3] != `One, "quoted"` {
			t.Errorf("a title with a comma and quotes comes back whole: %q", r[3])
		}
	}
	build, err := os.ReadFile(filepath.Join(out, "build.cypher"))
	if err != nil || !strings.Contains(string(build), "CREATE REL TABLE TOUCHES(FROM PR TO File)") {
		t.Fatalf("build.cypher: %v\n%s", err, build)
	}
}

// TestTheNewestFinishedBuildIsTheGraph: builds go into stamped folders of one directory, and a build counts once it is
// finished; pruning keeps the newest finished builds and drops older ones, finished or not, and nothing else.
func TestTheNewestFinishedBuildIsTheGraph(t *testing.T) {
	into := t.TempDir()
	if _, err := Latest(into); err == nil {
		t.Fatal("an empty folder has no graph")
	}
	at := time.Date(2026, 11, 16, 20, 0, 0, 0, time.UTC)
	var dirs []string
	for i := 0; i < 4; i++ {
		dir, err := NewBuild(into, at.Add(time.Duration(i)*time.Hour))
		if err != nil {
			t.Fatal(err)
		}
		os.WriteFile(filepath.Join(dir, "db"), []byte("x"), 0o644)
		dirs = append(dirs, dir)
	}
	for _, i := range []int{0, 1, 2} { // the newest, 3, is still being built
		if err := Finish(dirs[i]); err != nil {
			t.Fatal(err)
		}
	}
	os.WriteFile(filepath.Join(into, "notes.txt"), []byte("mine"), 0o644)
	if db, err := Latest(into); err != nil || db != filepath.Join(dirs[2], "db") {
		t.Fatalf("latest %q, %v; want the newest finished build", db, err)
	}
	if err := Prune(into, 1); err != nil {
		t.Fatal(err)
	}
	for i, want := range []bool{false, false, true, true} {
		if _, err := os.Stat(dirs[i]); (err == nil) != want {
			t.Errorf("build %d kept=%v, want %v", i, err == nil, want)
		}
	}
	if _, err := os.Stat(filepath.Join(into, "notes.txt")); err != nil {
		t.Error("pruning removed a file that is not a build")
	}
}

// TestLadybugAnswersMultiHopQuestions builds the graph with the LadybugDB CLI named by IXFORGE_LBUG and asks which
// learnings name a file that changes together with web/c.js (l1 names go/a.go, changed with web/c.js in one commit), and
// which open pull requests touch a file another open one touches.
func TestLadybugAnswersMultiHopQuestions(t *testing.T) {
	lbug := os.Getenv("IXFORGE_LBUG")
	if lbug == "" {
		t.Skip("IXFORGE_LBUG names no lbug.exe")
	}
	repo := gitRepo(t, []string{"go/a.go", "go/b.go"}, []string{"go/a.go", "web/c.js"}, []string{"docs/solutions/l1.md"})
	os.WriteFile(filepath.Join(repo, "docs", "solutions", "l1.md"), []byte("see go/a.go\n"), 0o644)
	catalog := filepath.Join(t.TempDir(), "catalog.jsonl")
	os.WriteFile(catalog, []byte(`{"id":"ix:l1","repo":"ix","kind":"solution","category":"ci","title":"L1","tags":["t"],"path":"docs/solutions/l1.md"}`+"\n"), 0o644)
	learnings, _ := ReadCatalog(catalog)
	commits, _ := ReadCommits(repo, "ix", 100, 40)
	prs, _ := ParsePRs("ix", []byte(ghPRs))
	issues, _ := ParseIssues("ix", []byte(ghIssues))
	into := t.TempDir()
	dir, err := NewBuild(into, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if err := Export(dir, Data{Roots: map[string]string{"ix": repo}, Learnings: learnings, Commits: commits, PRs: prs, Issues: issues}); err != nil {
		t.Fatal(err)
	}
	if err := Build(lbug, filepath.Join(dir, "db"), filepath.Join(dir, "build.cypher")); err != nil {
		t.Fatal(err)
	}
	if err := Finish(dir); err != nil {
		t.Fatal(err)
	}
	db, err := Latest(into)
	if err != nil {
		t.Fatal(err)
	}
	ask := func(q, want string) {
		t.Helper()
		got, err := Query(lbug, db, q)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Join(strings.Fields(string(got)), "") != want {
			t.Fatalf("%s\ngot  %s\nwant %s", q, got, want)
		}
	}
	ask("MATCH (l:Learning)-[:MENTIONS]->(f:File)<-[:CHANGED]-(:Commit)-[:CHANGED]->(:File {id:'ix:web/c.js'}) RETURN DISTINCT l.id AS id, f.path AS file;",
		`[{"id":"ix:l1","file":"go/a.go"}]`)
	ask("MATCH (a:PR {state:'OPEN'})-[:TOUCHES]->(f:File)<-[:TOUCHES]-(b:PR {state:'OPEN'}) WHERE a.number < b.number RETURN a.number AS a, b.number AS b, f.path AS file;",
		`[{"a":7,"b":8,"file":"go/a.go"}]`)
	if _, err := Query(lbug, db, "MATCH (n:Nope) RETURN n;"); err == nil || !strings.Contains(err.Error(), "Nope") {
		t.Fatalf("an error from lbug comes back as an error, not as rows: %v", err)
	}
}
