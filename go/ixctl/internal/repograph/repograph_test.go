package repograph

import (
	"encoding/csv"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
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

// TestTheGraphLinksLearningsToTheFilesTheyNameAndTheFilesThatChangeTogether: the graph holds each learning with its
// repo and tags, each commit with the files it changed (a commit changing more files than the bound is left out: a
// rename or a vendored drop says nothing about what changes together), and a MENTIONS edge from a learning to each
// file of the repo its text names, by its path or by a base name no other file has.
func TestTheGraphLinksLearningsToTheFilesTheyNameAndTheFilesThatChangeTogether(t *testing.T) {
	repo := gitRepo(t,
		[]string{"go/a.go", "go/b.go"},
		[]string{"go/a.go", "web/c.js"},
		[]string{"x/1.go", "x/2.go", "x/3.go", "x/4.go"}, // over the bound of 3
		[]string{"docs/solutions/l1.md", "docs/solutions/l2.md"},
	)
	os.WriteFile(filepath.Join(repo, "docs", "solutions", "l1.md"), []byte("The fix is in a.go; web/c.js reads it. Not util.go.\n"), 0o644)
	os.WriteFile(filepath.Join(repo, "docs", "solutions", "l2.md"), []byte("Nothing named here.\n"), 0o644)
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
	commits, err := ReadCommits(repo, 100, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(commits) != 3 {
		t.Fatalf("%d commits kept, want 3 (the one over the bound left out)", len(commits))
	}
	mentions, err := Mentions(map[string]string{"ix": repo}, learnings, commits)
	if err != nil {
		t.Fatal(err)
	}
	out := t.TempDir()
	if err := Export(out, learnings, commits, mentions); err != nil {
		t.Fatal(err)
	}

	if got := pairs(t, filepath.Join(out, "tagged.csv")); strings.Join(got, " ") != "ix:docs/solutions/l1.md>fsync ix:docs/solutions/l1.md>windows ix:docs/solutions/l2.md>fsync ix:docs/solutions/l2.md>go" {
		t.Errorf("tagged %v", got)
	}
	if got := pairs(t, filepath.Join(out, "in_repo.csv")); len(got) != 3 || got[0] != "ga:docs/solutions/g.md>ga" {
		t.Errorf("in_repo %v", got)
	}
	if got := pairs(t, filepath.Join(out, "mentions.csv")); strings.Join(got, " ") != "ix:docs/solutions/l1.md>go/a.go ix:docs/solutions/l1.md>web/c.js" {
		t.Errorf("mentions %v, want a.go by its unique base name and web/c.js by its path", got)
	}
	if changed := pairs(t, filepath.Join(out, "changed.csv")); len(changed) != 6 {
		t.Errorf("changed %v, want the 6 file changes of the 3 kept commits", changed)
	}
	for _, r := range rows(t, filepath.Join(out, "learning.csv")) {
		if r[0] == "ix:docs/solutions/l1.md" && r[3] != `One, "quoted"` {
			t.Errorf("a title with a comma and quotes comes back whole: %q", r[3])
		}
	}
	build, err := os.ReadFile(filepath.Join(out, "build.cypher"))
	if err != nil || !strings.Contains(string(build), "CREATE REL TABLE MENTIONS(FROM Learning TO File)") {
		t.Fatalf("build.cypher: %v\n%s", err, build)
	}
}

// TestLadybugAnswersATwoHopQuestion builds the graph with the LadybugDB CLI named by IXFORGE_LBUG and asks which
// learnings name a file that changes together with web/c.js: l1 names go/a.go, changed with web/c.js in one commit.
func TestLadybugAnswersATwoHopQuestion(t *testing.T) {
	lbug := os.Getenv("IXFORGE_LBUG")
	if lbug == "" {
		t.Skip("IXFORGE_LBUG names no lbug.exe")
	}
	repo := gitRepo(t, []string{"go/a.go", "go/b.go"}, []string{"go/a.go", "web/c.js"}, []string{"docs/solutions/l1.md"})
	os.WriteFile(filepath.Join(repo, "docs", "solutions", "l1.md"), []byte("see go/a.go\n"), 0o644)
	catalog := filepath.Join(t.TempDir(), "catalog.jsonl")
	os.WriteFile(catalog, []byte(`{"id":"ix:l1","repo":"ix","kind":"solution","category":"ci","title":"L1","tags":["t"],"path":"docs/solutions/l1.md"}`+"\n"), 0o644)
	learnings, _ := ReadCatalog(catalog)
	commits, _ := ReadCommits(repo, 100, 40)
	mentions, err := Mentions(map[string]string{"ix": repo}, learnings, commits)
	if err != nil {
		t.Fatal(err)
	}
	out := t.TempDir()
	if err := Export(out, learnings, commits, mentions); err != nil {
		t.Fatal(err)
	}
	db := filepath.Join(out, "db")
	if err := Build(lbug, db, filepath.Join(out, "build.cypher")); err != nil {
		t.Fatal(err)
	}
	got, err := Query(lbug, db, "MATCH (l:Learning)-[:MENTIONS]->(f:File)<-[:CHANGED]-(:Commit)-[:CHANGED]->(:File {path:'web/c.js'}) RETURN DISTINCT l.id AS id, f.path AS file;")
	if err != nil {
		t.Fatal(err)
	}
	if strings.ReplaceAll(string(got), "\n", "") != `[{"id":"ix:l1","file":"go/a.go"}]` {
		t.Fatalf("two hops: %s", got)
	}
	if _, err := Query(lbug, db, "MATCH (n:Nope) RETURN n;"); err == nil || !strings.Contains(err.Error(), "Nope") {
		t.Fatalf("an error from lbug comes back as an error, not as rows: %v", err)
	}
}
