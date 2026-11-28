// Package repograph builds a LadybugDB graph of what the repos know about themselves: the Streeling learnings
// (state/streeling/catalog.jsonl) with their repo and tags, each repo's recent commits with the files each changed, its
// pull requests with the files they touch and the issues they close, its issues, and which files each learning or issue
// names. It answers multi-hop questions edge-list SQL makes awkward, such as "which learnings name a file that usually
// changes together with this one?" or "which open pull requests touch the same files?".
//
// Export writes the graph as CSV files beside a build.cypher script; Build runs that script with the lbug CLI into a
// new database; Query runs one read-only Cypher query and returns its rows as JSON. Builds go into stamped folders of
// one directory (NewBuild, Finish): the newest finished one is the graph (Latest). The inputs are repo files, git
// history and, when asked, the pull requests and issues gh lists; nothing secret is read.
package repograph

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Learning is one entry of the Streeling catalog.
type Learning struct {
	ID       string   `json:"id"`
	Repo     string   `json:"repo"`
	Kind     string   `json:"kind"`
	Category string   `json:"category"`
	Title    string   `json:"title"`
	Date     string   `json:"date"`
	Path     string   `json:"path"`
	Tags     []string `json:"tags"`
}

// Commit is a commit of a repo and the repo-relative paths (with /) it changed.
type Commit struct {
	Repo, SHA, Date, Subject string
	Files                    []string
}

// PR is a pull request of a repo: the files it touches and the issues it closes, as "repo#number".
type PR struct {
	Repo                          string
	Number                        int
	Title, State, Branch, Updated string
	Files                         []string
	Closes                        []string
}

// Issue is an issue of a repo, with its text.
type Issue struct {
	Repo                  string
	Number                int
	Title, State, Updated string
	Body                  string
}

// Data is what a graph is made of. Roots holds each repo's folder, where the learnings' texts are read.
type Data struct {
	Roots     map[string]string
	Learnings []Learning
	Commits   []Commit
	PRs       []PR
	Issues    []Issue
}

// ReadCatalog reads the Streeling catalog, one JSON learning per line.
func ReadCatalog(file string) ([]Learning, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []Learning
	scan := bufio.NewScanner(f)
	scan.Buffer(make([]byte, 64<<10), 4<<20)
	for n := 1; scan.Scan(); n++ {
		line := bytes.TrimSpace(scan.Bytes())
		if len(line) == 0 {
			continue
		}
		var l Learning
		if err := json.Unmarshal(line, &l); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", file, n, err)
		}
		if l.ID == "" || l.Repo == "" {
			return nil, fmt.Errorf("%s:%d: a learning needs an id and a repo", file, n)
		}
		out = append(out, l)
	}
	return out, scan.Err()
}

// ReadCommits reads the last max non-merge commits of the repo in dir, named name in the graph, newest first, keeping
// those that changed at least one and at most maxFiles files: a mass rename or a vendored drop says nothing about which
// files change together.
func ReadCommits(dir, name string, max, maxFiles int) ([]Commit, error) {
	cmd := exec.Command("git", "-C", dir, "-c", "core.quotepath=false", "log", "--no-merges", "--no-renames",
		fmt.Sprintf("-n%d", max), "--name-only", "--date=format-local:%Y-%m-%dT%H:%M:%SZ", "--format=%x00%H%x1f%cd%x1f%s")
	cmd.Env = append(os.Environ(), "TZ=UTC") // commit times in UTC, so they sort as text
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	raw, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git log: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	var out []Commit
	for _, chunk := range strings.Split(string(raw), "\x00") {
		lines := strings.Split(strings.ReplaceAll(chunk, "\r", ""), "\n")
		head := strings.SplitN(lines[0], "\x1f", 3)
		if len(head) != 3 {
			continue
		}
		c := Commit{Repo: name, SHA: head[0], Date: head[1], Subject: head[2]}
		for _, l := range lines[1:] {
			if l = strings.TrimSpace(l); l != "" {
				c.Files = append(c.Files, l)
			}
		}
		if len(c.Files) > 0 && len(c.Files) <= maxFiles {
			out = append(out, c)
		}
	}
	return out, nil
}

// ParsePRs reads the pull requests of repo as `gh pr list --json
// number,title,state,headRefName,updatedAt,files,closingIssuesReferences` writes them.
func ParsePRs(repo string, raw []byte) ([]PR, error) {
	var list []struct {
		Number      int    `json:"number"`
		Title       string `json:"title"`
		State       string `json:"state"`
		HeadRefName string `json:"headRefName"`
		UpdatedAt   string `json:"updatedAt"`
		Files       []struct {
			Path string `json:"path"`
		} `json:"files"`
		Closing []struct {
			Number     int `json:"number"`
			Repository struct {
				Name string `json:"name"`
			} `json:"repository"`
		} `json:"closingIssuesReferences"`
	}
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, fmt.Errorf("pull requests of %s: %w", repo, err)
	}
	out := make([]PR, 0, len(list))
	for _, p := range list {
		pr := PR{Repo: repo, Number: p.Number, Title: p.Title, State: p.State, Branch: p.HeadRefName, Updated: p.UpdatedAt}
		for _, f := range p.Files {
			if f.Path != "" {
				pr.Files = append(pr.Files, f.Path)
			}
		}
		for _, c := range p.Closing {
			if c.Repository.Name != "" {
				pr.Closes = append(pr.Closes, fmt.Sprintf("%s#%d", c.Repository.Name, c.Number))
			}
		}
		out = append(out, pr)
	}
	return out, nil
}

// ParseIssues reads the issues of repo as `gh issue list --json number,title,state,updatedAt,body` writes them.
func ParseIssues(repo string, raw []byte) ([]Issue, error) {
	var list []struct {
		Number    int    `json:"number"`
		Title     string `json:"title"`
		State     string `json:"state"`
		UpdatedAt string `json:"updatedAt"`
		Body      string `json:"body"`
	}
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, fmt.Errorf("issues of %s: %w", repo, err)
	}
	out := make([]Issue, 0, len(list))
	for _, i := range list {
		out = append(out, Issue{Repo: repo, Number: i.Number, Title: i.Title, State: i.State, Updated: i.UpdatedAt, Body: i.Body})
	}
	return out, nil
}

// ReadGitHub lists the last prs pull requests and issues issues of the GitHub repo slug (owner/name), open or closed,
// with the gh CLI and its own sign-in, and names them repo in the graph.
func ReadGitHub(repo, slug string, prs, issues int) ([]PR, []Issue, error) {
	gh := func(args ...string) ([]byte, error) {
		cmd := exec.Command("gh", args...)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		if err != nil {
			return nil, fmt.Errorf("gh %s: %w: %s", args[0], err, strings.TrimSpace(stderr.String()))
		}
		return out, nil
	}
	raw, err := gh("pr", "list", "--repo", slug, "--state", "all", "--limit", strconv.Itoa(prs),
		"--json", "number,title,state,headRefName,updatedAt,files,closingIssuesReferences")
	if err != nil {
		return nil, nil, err
	}
	pulls, err := ParsePRs(repo, raw)
	if err != nil {
		return nil, nil, err
	}
	raw, err = gh("issue", "list", "--repo", slug, "--state", "all", "--limit", strconv.Itoa(issues),
		"--json", "number,title,state,updatedAt,body")
	if err != nil {
		return nil, nil, err
	}
	list, err := ParseIssues(repo, raw)
	return pulls, list, err
}

// A file name as a text names it: a path or base name ending in an extension.
var fileToken = regexp.MustCompile(`[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,6}`)

// namer finds the files of one repo a text names: by its path, by a trailing part of its path, or by a base name no
// other file of the repo has.
type namer struct {
	files  map[string]bool
	byBase map[string][]string
	all    []string
}

func newNamer(files map[string]bool) *namer {
	n := &namer{files: files, byBase: map[string][]string{}}
	for f := range files {
		n.all = append(n.all, f)
		n.byBase[path.Base(f)] = append(n.byBase[path.Base(f)], f)
	}
	sort.Strings(n.all)
	return n
}

func (n *namer) names(text string) map[string]bool {
	named := map[string]bool{}
	for _, tok := range fileToken.FindAllString(text, -1) {
		tok = strings.TrimPrefix(strings.ReplaceAll(tok, `\`, "/"), "./")
		switch {
		case n.files[tok]:
			named[tok] = true
		case !strings.Contains(tok, "/"):
			if same := n.byBase[tok]; len(same) == 1 {
				named[same[0]] = true
			}
		default:
			var hit []string
			for _, f := range n.all {
				if strings.HasSuffix(f, "/"+tok) {
					hit = append(hit, f)
				}
			}
			if len(hit) == 1 {
				named[hit[0]] = true
			}
		}
	}
	return named
}

// fileID is how the graph knows a file: by its repo and its path.
func fileID(repo, p string) string { return repo + ":" + p }

// Stats counts what a graph holds.
type Stats struct{ Files, Mentions, Touches, Names, Closes int }

func writeCSV(dir, name string, header []string, rows [][]string) error {
	f, err := os.Create(filepath.Join(dir, name))
	if err != nil {
		return err
	}
	w := csv.NewWriter(f)
	w.Write(header)
	for _, r := range rows {
		for i := range r {
			r[i] = strings.Join(strings.Fields(r[i]), " ") // one line per row: lbug reads CSV in parallel
		}
		w.Write(r)
	}
	w.Flush()
	if err := w.Error(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// The graph's tables, created and loaded by build.cypher in this order. A file is known by its id, repo:path; a pull
// request or an issue by repo#number.
const schema = `CREATE NODE TABLE Repo(name STRING PRIMARY KEY);
CREATE NODE TABLE Learning(id STRING PRIMARY KEY, kind STRING, category STRING, title STRING, date STRING, path STRING);
CREATE NODE TABLE Tag(name STRING PRIMARY KEY);
CREATE NODE TABLE File(id STRING PRIMARY KEY, repo STRING, path STRING);
CREATE NODE TABLE Commit(sha STRING PRIMARY KEY, repo STRING, date STRING, subject STRING);
CREATE NODE TABLE PR(id STRING PRIMARY KEY, repo STRING, number INT64, title STRING, state STRING, branch STRING, updated STRING);
CREATE NODE TABLE Issue(id STRING PRIMARY KEY, repo STRING, number INT64, title STRING, state STRING, updated STRING);
CREATE REL TABLE IN_REPO(FROM Learning TO Repo);
CREATE REL TABLE TAGGED(FROM Learning TO Tag);
CREATE REL TABLE CHANGED(FROM Commit TO File);
CREATE REL TABLE MENTIONS(FROM Learning TO File);
CREATE REL TABLE TOUCHES(FROM PR TO File);
CREATE REL TABLE CLOSES(FROM PR TO Issue);
CREATE REL TABLE NAMES(FROM Issue TO File);
`

var loads = []struct{ table, file string }{
	{"Repo", "repo.csv"}, {"Learning", "learning.csv"}, {"Tag", "tag.csv"}, {"File", "file.csv"}, {"Commit", "commit.csv"},
	{"PR", "pr.csv"}, {"Issue", "issue.csv"},
	{"IN_REPO", "in_repo.csv"}, {"TAGGED", "tagged.csv"}, {"CHANGED", "changed.csv"}, {"MENTIONS", "mentions.csv"},
	{"TOUCHES", "touches.csv"}, {"CLOSES", "closes.csv"}, {"NAMES", "names.csv"},
}

// Export writes the graph's CSV files and build.cypher, which creates the tables and loads them, into dir.
func Export(dir string, d Data) error {
	_, err := ExportStats(dir, d)
	return err
}

// ExportStats is Export, and says what the graph holds.
func ExportStats(dir string, d Data) (Stats, error) {
	var stats Stats
	seen := map[string]bool{}
	once := func(kind, v string) bool {
		if seen[kind+"\x00"+v] {
			return false
		}
		seen[kind+"\x00"+v] = true
		return true
	}
	var repoRows, learningRows, tagRows, fileRows, commitRows, prRows, issueRows [][]string
	var inRepo, tagged, changed, mentions, touches, closes, names [][]string
	repoFiles := map[string]map[string]bool{}
	addRepo := func(r string) {
		if once("r", r) {
			repoRows = append(repoRows, []string{r})
		}
	}
	addFile := func(repo, p string) string {
		id := fileID(repo, p)
		if once("f", id) {
			fileRows = append(fileRows, []string{id, repo, p})
			if repoFiles[repo] == nil {
				repoFiles[repo] = map[string]bool{}
			}
			repoFiles[repo][p] = true
		}
		return id
	}
	for _, l := range d.Learnings {
		if !once("l", l.ID) {
			return stats, fmt.Errorf("learning %s twice in the catalog", l.ID)
		}
		addRepo(l.Repo)
		learningRows = append(learningRows, []string{l.ID, l.Kind, l.Category, l.Title, l.Date, l.Path})
		inRepo = append(inRepo, []string{l.ID, l.Repo})
		mine := map[string]bool{}
		for _, t := range l.Tags {
			if t = strings.TrimSpace(t); t != "" && !mine[t] {
				mine[t] = true
				if once("t", t) {
					tagRows = append(tagRows, []string{t})
				}
				tagged = append(tagged, []string{l.ID, t})
			}
		}
	}
	for _, c := range d.Commits {
		if !once("c", c.SHA) {
			continue // the same commit read twice
		}
		addRepo(c.Repo)
		commitRows = append(commitRows, []string{c.SHA, c.Repo, c.Date, c.Subject})
		for _, f := range c.Files {
			changed = append(changed, []string{c.SHA, addFile(c.Repo, f)})
		}
	}
	for _, i := range d.Issues {
		id := fmt.Sprintf("%s#%d", i.Repo, i.Number)
		if once("i", id) {
			addRepo(i.Repo)
			issueRows = append(issueRows, []string{id, i.Repo, strconv.Itoa(i.Number), i.Title, i.State, i.Updated})
		}
	}
	for _, p := range d.PRs {
		id := fmt.Sprintf("%s#%d", p.Repo, p.Number)
		if !once("p", id) {
			continue
		}
		addRepo(p.Repo)
		prRows = append(prRows, []string{id, p.Repo, strconv.Itoa(p.Number), p.Title, p.State, p.Branch, p.Updated})
		for _, f := range p.Files {
			touches = append(touches, []string{id, addFile(p.Repo, f)})
		}
		for _, issue := range p.Closes {
			if seen["i\x00"+issue] {
				closes = append(closes, []string{id, issue})
			}
		}
	}
	namers := map[string]*namer{}
	for repo, files := range repoFiles {
		namers[repo] = newNamer(files)
	}
	for _, l := range d.Learnings {
		root, ok := d.Roots[l.Repo]
		if !ok || l.Path == "" || namers[l.Repo] == nil {
			continue
		}
		text, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(l.Path)))
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return stats, err
		}
		named := namers[l.Repo].names(string(text))
		delete(named, l.Path) // a learning does not name itself
		for f := range named {
			mentions = append(mentions, []string{l.ID, fileID(l.Repo, f)})
		}
	}
	for _, i := range d.Issues {
		if namers[i.Repo] == nil {
			continue
		}
		id := fmt.Sprintf("%s#%d", i.Repo, i.Number)
		for f := range namers[i.Repo].names(i.Title + "\n" + i.Body) {
			names = append(names, []string{id, fileID(i.Repo, f)})
		}
	}
	stats = Stats{Files: len(fileRows), Mentions: len(mentions), Touches: len(touches), Names: len(names), Closes: len(closes)}
	sorted := func(rows [][]string) [][]string {
		sort.Slice(rows, func(i, j int) bool { return strings.Join(rows[i], "\x00") < strings.Join(rows[j], "\x00") })
		return rows
	}
	for _, t := range []struct {
		name   string
		header []string
		rows   [][]string
	}{
		{"repo.csv", []string{"name"}, repoRows},
		{"learning.csv", []string{"id", "kind", "category", "title", "date", "path"}, learningRows},
		{"tag.csv", []string{"name"}, tagRows},
		{"file.csv", []string{"id", "repo", "path"}, fileRows},
		{"commit.csv", []string{"sha", "repo", "date", "subject"}, commitRows},
		{"pr.csv", []string{"id", "repo", "number", "title", "state", "branch", "updated"}, prRows},
		{"issue.csv", []string{"id", "repo", "number", "title", "state", "updated"}, issueRows},
		{"in_repo.csv", []string{"from", "to"}, inRepo},
		{"tagged.csv", []string{"from", "to"}, tagged},
		{"changed.csv", []string{"from", "to"}, changed},
		{"mentions.csv", []string{"from", "to"}, sorted(mentions)},
		{"touches.csv", []string{"from", "to"}, touches},
		{"closes.csv", []string{"from", "to"}, closes},
		{"names.csv", []string{"from", "to"}, sorted(names)},
	} {
		if err := writeCSV(dir, t.name, t.header, t.rows); err != nil {
			return stats, err
		}
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return stats, err
	}
	var script strings.Builder
	script.WriteString(schema)
	for _, l := range loads {
		fmt.Fprintf(&script, "COPY %s FROM '%s' (header=true);\n", l.table, filepath.ToSlash(filepath.Join(abs, l.file)))
	}
	return stats, os.WriteFile(filepath.Join(dir, "build.cypher"), []byte(script.String()), 0o644)
}

// A build folder is named by the UTC time it started; a finished build holds a done file beside its database.
const buildStamp = "20060102T150405Z"

// NewBuild makes the folder of a build started at now, inside into.
func NewBuild(into string, now time.Time) (string, error) {
	dir := filepath.Join(into, now.UTC().Format(buildStamp))
	if err := os.MkdirAll(into, 0o755); err != nil {
		return "", err
	}
	return dir, os.Mkdir(dir, 0o755)
}

// Finish marks a build finished: from then on Latest may answer it.
func Finish(dir string) error { return os.WriteFile(filepath.Join(dir, "done"), nil, 0o644) }

// builds lists the build folders of into, newest first, and whether each is finished.
func builds(into string) ([]string, map[string]bool, error) {
	entries, err := os.ReadDir(into)
	if err != nil {
		return nil, nil, err
	}
	var dirs []string
	done := map[string]bool{}
	for _, e := range entries {
		if _, err := time.Parse(buildStamp, e.Name()); err != nil || !e.IsDir() {
			continue
		}
		dirs = append(dirs, e.Name())
		if _, err := os.Stat(filepath.Join(into, e.Name(), "done")); err == nil {
			done[e.Name()] = true
		}
	}
	sort.Sort(sort.Reverse(sort.StringSlice(dirs)))
	return dirs, done, nil
}

// ErrNoGraph says no build of the folder is finished.
var ErrNoGraph = errors.New("no finished graph build")

// Latest is the database of the newest finished build in into.
func Latest(into string) (string, error) {
	dirs, done, err := builds(into)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", ErrNoGraph
		}
		return "", err
	}
	for _, d := range dirs {
		if done[d] {
			return filepath.Join(into, d, "db"), nil
		}
	}
	return "", ErrNoGraph
}

// Prune keeps the newest keep finished builds of into, and any build newer than the oldest of them (one may be under
// way); it removes the other build folders, and nothing that is not a build.
func Prune(into string, keep int) error {
	dirs, done, err := builds(into)
	if err != nil {
		return err
	}
	kept := 0
	for _, d := range dirs {
		if kept >= keep {
			if err := os.RemoveAll(filepath.Join(into, d)); err != nil {
				return err
			}
			continue
		}
		if done[d] {
			kept++
		}
	}
	return nil
}

// ErrorLine is the first line lbug printed as an error, or "": lbug reports errors on its output and still exits 0.
func ErrorLine(out []byte) string {
	for _, line := range strings.Split(strings.ReplaceAll(string(out), "\r", ""), "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "Error") {
			return strings.TrimSpace(line)
		}
	}
	return ""
}

func lbugError(out []byte) error {
	if line := ErrorLine(out); line != "" {
		return errors.New(line)
	}
	return nil
}

func run(lbug string, args []string, stdin string) ([]byte, error) {
	cmd := exec.Command(lbug, args...)
	cmd.Stdin = strings.NewReader(stdin)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("lbug: %w: %s", err, bytes.TrimSpace(out))
	}
	if err := lbugError(out); err != nil {
		return nil, fmt.Errorf("lbug: %w", err)
	}
	return out, nil
}

// Build runs script with the lbug CLI into db, which must not exist yet (a build never adds to an older graph).
func Build(lbug, db, script string) error {
	if _, err := os.Stat(db); err == nil {
		return fmt.Errorf("%s exists: remove it to rebuild the graph", db)
	}
	text, err := os.ReadFile(script)
	if err != nil {
		return err
	}
	_, err = run(lbug, []string{db, "-b", "-s"}, string(text))
	return err
}

// Query runs one Cypher query on db, read-only, and returns its rows as a JSON array.
func Query(lbug, db, query string) ([]byte, error) {
	out, err := run(lbug, []string{db, "-r", "-b", "-s", "-m", "json"}, query)
	if err != nil {
		return nil, err
	}
	return bytes.TrimSpace(bytes.ReplaceAll(out, []byte("\r"), nil)), nil
}
