// Package repograph builds a LadybugDB graph of what the repos know about themselves: the Streeling learnings
// (state/streeling/catalog.jsonl) with their repo and tags, a repo's recent commits with the files each changed, and
// which files each learning names. It answers multi-hop questions edge-list SQL makes awkward, such as "which learnings
// name a file that usually changes together with this one?".
//
// Export writes the graph as CSV files beside a build.cypher script; Build runs that script with the lbug CLI into a
// new database; Query runs one read-only Cypher query and returns its rows as JSON. Every input is a repo file or git
// history: nothing is fetched and nothing secret is read.
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
	"strings"
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

// Commit is a commit and the repo-relative paths (with /) it changed.
type Commit struct {
	SHA, Date, Subject string
	Files              []string
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

// ReadCommits reads the last max non-merge commits of repo, newest first, keeping those that changed at least one and
// at most maxFiles files: a mass rename or a vendored drop says nothing about which files change together.
func ReadCommits(repo string, max, maxFiles int) ([]Commit, error) {
	cmd := exec.Command("git", "-C", repo, "-c", "core.quotepath=false", "log", "--no-merges", "--no-renames",
		fmt.Sprintf("-n%d", max), "--name-only", "--format=%x00%H%x1f%cs%x1f%s")
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
		c := Commit{SHA: head[0], Date: head[1], Subject: head[2]}
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

// A file name as a text names it: a path or base name ending in an extension.
var fileToken = regexp.MustCompile(`[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,6}`)

// Mentions finds, for each learning whose repo has a root in roots (the repo the commits were read from), the changed
// files its text names: by its path, by a trailing part of its path, or by a base name no other file has. Learnings
// of other repos, and texts that are not there, name nothing.
func Mentions(roots map[string]string, learnings []Learning, commits []Commit) (map[string][]string, error) {
	files := map[string]bool{}
	byBase := map[string][]string{}
	for _, c := range commits {
		for _, f := range c.Files {
			if !files[f] {
				files[f] = true
				byBase[path.Base(f)] = append(byBase[path.Base(f)], f)
			}
		}
	}
	all := make([]string, 0, len(files))
	for f := range files {
		all = append(all, f)
	}
	out := map[string][]string{}
	for _, l := range learnings {
		root, ok := roots[l.Repo]
		if !ok || l.Path == "" {
			continue
		}
		text, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(l.Path)))
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		named := map[string]bool{}
		for _, tok := range fileToken.FindAllString(string(text), -1) {
			tok = strings.TrimPrefix(strings.ReplaceAll(tok, `\`, "/"), "./")
			switch {
			case files[tok]:
				named[tok] = true
			case !strings.Contains(tok, "/"):
				if same := byBase[tok]; len(same) == 1 {
					named[same[0]] = true
				}
			default:
				var hit []string
				for _, f := range all {
					if strings.HasSuffix(f, "/"+tok) {
						hit = append(hit, f)
					}
				}
				if len(hit) == 1 {
					named[hit[0]] = true
				}
			}
		}
		delete(named, l.Path) // a learning does not name itself
		for f := range named {
			out[l.ID] = append(out[l.ID], f)
		}
		sort.Strings(out[l.ID])
	}
	return out, nil
}

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

// The graph's tables, created and loaded by build.cypher in this order.
const schema = `CREATE NODE TABLE Repo(name STRING PRIMARY KEY);
CREATE NODE TABLE Learning(id STRING PRIMARY KEY, kind STRING, category STRING, title STRING, date STRING, path STRING);
CREATE NODE TABLE Tag(name STRING PRIMARY KEY);
CREATE NODE TABLE File(path STRING PRIMARY KEY);
CREATE NODE TABLE Commit(sha STRING PRIMARY KEY, date STRING, subject STRING);
CREATE REL TABLE IN_REPO(FROM Learning TO Repo);
CREATE REL TABLE TAGGED(FROM Learning TO Tag);
CREATE REL TABLE CHANGED(FROM Commit TO File);
CREATE REL TABLE MENTIONS(FROM Learning TO File);
`

var loads = []struct{ table, file string }{
	{"Repo", "repo.csv"}, {"Learning", "learning.csv"}, {"Tag", "tag.csv"}, {"File", "file.csv"}, {"Commit", "commit.csv"},
	{"IN_REPO", "in_repo.csv"}, {"TAGGED", "tagged.csv"}, {"CHANGED", "changed.csv"}, {"MENTIONS", "mentions.csv"},
}

// Export writes the graph's CSV files and build.cypher, which creates the tables and loads them, into dir.
func Export(dir string, learnings []Learning, commits []Commit, mentions map[string][]string) error {
	var repos, tags, files []string
	seen := map[string]bool{}
	add := func(list *[]string, kind, v string) {
		if !seen[kind+"\x00"+v] {
			seen[kind+"\x00"+v] = true
			*list = append(*list, v)
		}
	}
	var learningRows, inRepo, tagged, commitRows, changed, mentionRows [][]string
	for _, l := range learnings {
		if seen["l\x00"+l.ID] {
			return fmt.Errorf("learning %s twice in the catalog", l.ID)
		}
		seen["l\x00"+l.ID] = true
		add(&repos, "r", l.Repo)
		learningRows = append(learningRows, []string{l.ID, l.Kind, l.Category, l.Title, l.Date, l.Path})
		inRepo = append(inRepo, []string{l.ID, l.Repo})
		mine := map[string]bool{}
		for _, t := range l.Tags {
			if t = strings.TrimSpace(t); t != "" && !mine[t] {
				mine[t] = true
				add(&tags, "t", t)
				tagged = append(tagged, []string{l.ID, t})
			}
		}
	}
	for _, c := range commits {
		commitRows = append(commitRows, []string{c.SHA, c.Date, c.Subject})
		for _, f := range c.Files {
			add(&files, "f", f)
			changed = append(changed, []string{c.SHA, f})
		}
	}
	for id, named := range mentions {
		if !seen["l\x00"+id] {
			return fmt.Errorf("mentions name an unknown learning %s", id)
		}
		for _, f := range named {
			if !seen["f\x00"+f] {
				return fmt.Errorf("learning %s names %s, which no commit changed", id, f)
			}
			mentionRows = append(mentionRows, []string{id, f})
		}
	}
	sort.Slice(mentionRows, func(i, j int) bool {
		return strings.Join(mentionRows[i], "\x00") < strings.Join(mentionRows[j], "\x00")
	})
	one := func(vs []string) [][]string {
		rows := make([][]string, len(vs))
		for i, v := range vs {
			rows[i] = []string{v}
		}
		return rows
	}
	for _, t := range []struct {
		name   string
		header []string
		rows   [][]string
	}{
		{"repo.csv", []string{"name"}, one(repos)},
		{"learning.csv", []string{"id", "kind", "category", "title", "date", "path"}, learningRows},
		{"tag.csv", []string{"name"}, one(tags)},
		{"file.csv", []string{"path"}, one(files)},
		{"commit.csv", []string{"sha", "date", "subject"}, commitRows},
		{"in_repo.csv", []string{"from", "to"}, inRepo},
		{"tagged.csv", []string{"from", "to"}, tagged},
		{"changed.csv", []string{"from", "to"}, changed},
		{"mentions.csv", []string{"from", "to"}, mentionRows},
	} {
		if err := writeCSV(dir, t.name, t.header, t.rows); err != nil {
			return err
		}
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	var script strings.Builder
	script.WriteString(schema)
	for _, l := range loads {
		fmt.Fprintf(&script, "COPY %s FROM '%s' (header=true);\n", l.table, filepath.ToSlash(filepath.Join(abs, l.file)))
	}
	return os.WriteFile(filepath.Join(dir, "build.cypher"), []byte(script.String()), 0o644)
}

// lbugError is the first line lbug printed as an error: it reports errors on its output and still exits 0.
func lbugError(out []byte) error {
	for _, line := range strings.Split(strings.ReplaceAll(string(out), "\r", ""), "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "Error") {
			return errors.New(strings.TrimSpace(line))
		}
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
