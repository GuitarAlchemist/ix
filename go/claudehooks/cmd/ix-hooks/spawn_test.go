package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestMain lets this test binary stand in for the programs a hook may start and for ix-hooks
// itself. Named git, gh or cargo, it appends its command line to $IX_HOOKS_SPAWN_LOG and fails,
// as a missing tool would. With IX_HOOKS_RUN_MAIN=1 it is ix-hooks: main(), with realSystem.
func TestMain(m *testing.M) {
	switch name := strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe"); name {
	case "git", "gh", "cargo":
		f, err := os.OpenFile(os.Getenv("IX_HOOKS_SPAWN_LOG"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err == nil {
			_, _ = fmt.Fprintln(f, name+" "+strings.Join(os.Args[1:], " "))
			_ = f.Close()
		}
		os.Exit(1)
	}
	if os.Getenv("IX_HOOKS_RUN_MAIN") == "1" {
		main()
	}
	os.Exit(m.Run())
}

// The hooks that fire on every prompt and every tool call, run as the real program (realSystem,
// the real repository lookup) with only recording git, gh and cargo on PATH, start nothing. The
// three spawns the scripts had, and only they, start their tools; they are also the proof that
// the recorder sees a spawn. TestHotHooksStartNoProcess cannot see realSystem: its fake replaces
// toplevel, which was a `git rev-parse` on every prompt before PR #512's review.
func TestHotHooksStartNoProcessAsTheRealProgram(t *testing.T) {
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	exe, _ := os.ReadFile(self)
	tools := t.TempDir()
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	for _, name := range []string{"git", "gh", "cargo"} {
		if err := os.WriteFile(filepath.Join(tools, name+suffix), exe, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	root := t.TempDir()
	os.Mkdir(filepath.Join(root, ".git"), 0o755)
	writeDigest(t, root, "digest-skill", 5*time.Minute, time.Now()) // fresh: no stub by age
	pipeline := filepath.Join(root, "my_pipeline.rs")
	os.WriteFile(pipeline, []byte("b.input(\"x\", \"loop\").node(\"loop\", f)\n"), 0o644)
	spawns := filepath.Join(t.TempDir(), "spawns.log")

	run := func(hook, stdin string, env ...string) []string {
		t.Helper()
		_ = os.Remove(spawns)
		cmd := exec.Command(self, hook)
		cmd.Dir = root
		cmd.Stdin = strings.NewReader(stdin)
		cmd.Env = append([]string{"IX_HOOKS_RUN_MAIN=1", "IX_HOOKS_SPAWN_LOG=" + spawns, "PATH=" + tools,
			"PATHEXT=.EXE", "SystemRoot=" + os.Getenv("SystemRoot")}, env...)
		if out, err := cmd.CombinedOutput(); err != nil {
			if _, exited := err.(*exec.ExitError); !exited {
				t.Fatalf("%s: %v %s", hook, err, out)
			}
		}
		data, _ := os.ReadFile(spawns)
		var programs []string
		for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
			if f := strings.Fields(line); len(f) > 0 {
				programs = append(programs, f[0])
			}
		}
		return programs
	}
	tool := func(name string, input map[string]any, response any) string {
		b, _ := json.Marshal(map[string]any{"tool_name": name, "tool_input": input, "tool_response": response})
		return string(b)
	}
	bash := func(command string) map[string]any { return map[string]any{"command": command} }
	file := func(path string) map[string]any { return map[string]any{"file_path": path} }

	for _, c := range []struct {
		hook, stdin string
		env         []string
		want        string
	}{
		{"correction-detector", `{"prompt":"non, pas ce fichier"}`, nil, ""},
		{"digest-staleness-nudge", `{"prompt":"continue"}`, nil, ""},
		{"governance-check", tool("Bash", bash("git status"), nil), nil, ""},
		{"governance-check", tool("Write", file("/app/.env"), nil), nil, ""},
		{"pipeline-validate", tool("Edit", file(pipeline), nil), nil, ""},
		{"digest-activity-tracker", "", []string{"IX_DIGEST_MID_COUNT=1000"}, ""},
		{"rust-check", tool("Edit", file(`C:\r\crates\ix-math\src\lib.rs`), nil), nil, ""},
		{"pr-rationale-capture", tool("Bash", bash("gh pr view 1"), map[string]any{"output": "x"}), nil, ""},
		// The spawns kept from the scripts: a stub's branch, sha and subject every
		// IX_DIGEST_MID_COUNT mutations; cargo check after a forward-slash crate edit; branch and
		// shortstat after a `gh pr create`.
		{"digest-activity-tracker", "", []string{"IX_DIGEST_MID_COUNT=1"}, "git git git"},
		{"rust-check", tool("Edit", file("/r/crates/ix-math/src/lib.rs"), nil), nil, "cargo"},
		{"pr-rationale-capture", tool("Bash", bash(`gh pr create --title "T" --body 'B'`),
			map[string]any{"output": "https://github.com/o/r/pull/7\n"}), nil, "git git"},
	} {
		if got := strings.Join(run(c.hook, c.stdin, c.env...), " "); got != c.want {
			t.Errorf("%s %s %v started %q, want %q", c.hook, c.stdin, c.env, got, c.want)
		}
	}
}

// Which hooks can start a process (system.output or .combined) or reach the network
// (system.fetch), over every branch, not only the cases run above: the call graph of this package
// from each hook. os/exec, net, net/http and syscall appear only in realSystem's output, combined
// and fetch, so these three fields are the only way out.
func TestWhichHooksCanStartAProcessOrFetch(t *testing.T) {
	pkg, files, info := checkedPackage(t)
	sysStruct := pkg.Scope().Lookup("system").Type().Underlying().(*types.Struct)
	effect := map[*types.Var]string{}
	for i := 0; i < sysStruct.NumFields(); i++ {
		switch f := sysStruct.Field(i); f.Name() {
		case "output", "combined":
			effect[f] = "process"
		case "fetch":
			effect[f] = "network"
		}
	}
	decls := map[*types.Func]*ast.FuncDecl{}
	var entries map[string]*types.Func
	var exits []ast.Node // realSystem's output, combined and fetch values
	for _, f := range files {
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok {
				decls[info.Defs[fd.Name].(*types.Func)] = fd
				if fd.Name.Name == "realSystem" {
					ast.Inspect(fd.Body, func(n ast.Node) bool {
						if kv, ok := n.(*ast.KeyValueExpr); ok {
							if k, ok := kv.Key.(*ast.Ident); ok && (k.Name == "output" || k.Name == "combined" || k.Name == "fetch") {
								exits = append(exits, kv.Value)
							}
						}
						return true
					})
				}
			}
			if gd, ok := d.(*ast.GenDecl); ok && gd.Tok == token.VAR {
				for _, spec := range gd.Specs {
					vs := spec.(*ast.ValueSpec)
					if len(vs.Names) == 1 && vs.Names[0].Name == "hooks" {
						entries = map[string]*types.Func{}
						for _, elt := range vs.Values[0].(*ast.CompositeLit).Elts {
							kv := elt.(*ast.KeyValueExpr)
							name, _ := strconv.Unquote(kv.Key.(*ast.BasicLit).Value)
							entries[name] = info.Uses[kv.Value.(*ast.Ident)].(*types.Func)
						}
					}
				}
			}
		}
	}

	reach := func(entry *types.Func) string {
		got := map[string]bool{}
		seen := map[*types.Func]bool{}
		var walk func(*types.Func)
		walk = func(fn *types.Func) {
			if seen[fn] || decls[fn] == nil {
				return
			}
			seen[fn] = true
			ast.Inspect(decls[fn].Body, func(n ast.Node) bool {
				switch x := n.(type) {
				case *ast.SelectorExpr:
					if s := info.Selections[x]; s != nil {
						if v, ok := s.Obj().(*types.Var); ok && effect[v] != "" {
							got[effect[v]] = true
						}
					}
				case *ast.Ident:
					if fn, ok := info.Uses[x].(*types.Func); ok && fn.Pkg() == pkg {
						walk(fn)
					}
				}
				return true
			})
		}
		walk(entry)
		var out []string
		for k := range got {
			out = append(out, k)
		}
		sort.Strings(out)
		return strings.Join(out, "+")
	}

	hot := "" // every prompt or every tool call: nothing, ever
	want := map[string]string{
		"correction-detector":    hot,
		"digest-staleness-nudge": hot,
		"governance-check":       hot,
		"pipeline-validate":      hot,
		// Hot too, with the spawns the scripts had (TestHotHooksStartNoProcessAsTheRealProgram
		// pins when): never the network.
		"digest-activity-tracker": "process",
		"rust-check":              "process",
		"pr-rationale-capture":    "process",
		// Once per session, compaction or turn.
		"session-start":        "network",
		"sessionstart-digest":  "network",
		"digest-stop-finalize": "process",
		"precompact-digest":    "process",
		"digest-validate":      "",
	}
	if len(entries) != len(want) {
		t.Errorf("hooks %d, table %d: give every hook a row", len(entries), len(want))
	}
	for name, fn := range entries {
		if w, ok := want[name]; !ok {
			t.Errorf("%s has no row", name)
		} else if got := reach(fn); got != w {
			t.Errorf("%s can reach %q, want %q", name, got, w)
		}
	}

	inExit := func(p token.Pos) bool {
		for _, n := range exits {
			if n.Pos() <= p && p < n.End() {
				return true
			}
		}
		return false
	}
	for id, obj := range info.Uses {
		if _, isPkg := obj.(*types.PkgName); isPkg || obj.Pkg() == nil {
			continue
		}
		path := obj.Pkg().Path()
		outward := path == "os/exec" || path == "net" || strings.HasPrefix(path, "net/http") || path == "syscall" ||
			(path == "os" && obj.Name() == "StartProcess")
		if outward && !inExit(id.Pos()) && obj.Name() != "StatusText" {
			t.Errorf("%s: %s.%s outside realSystem's output, combined and fetch", checkedFset.Position(id.Pos()), path, obj.Name())
		}
	}
}

var (
	checkOnce    sync.Once
	checkedFset  = token.NewFileSet()
	checkedPkg   *types.Package
	checkedFiles []*ast.File
	checkedInfo  *types.Info
	checkedErr   error
)

// checkedPackage type-checks this package's non-test files once for the tests that read its code.
// "source" type-checks the standard library from GOROOT: about 15 s, offline. "gc" was slower
// (77 s, one `go list -export` per import).
func checkedPackage(t *testing.T) (*types.Package, []*ast.File, *types.Info) {
	t.Helper()
	checkOnce.Do(func() {
		names, _ := filepath.Glob("*.go")
		for _, name := range names {
			if strings.HasSuffix(name, "_test.go") {
				continue
			}
			f, err := parser.ParseFile(checkedFset, name, nil, 0)
			if err != nil {
				checkedErr = err
				return
			}
			checkedFiles = append(checkedFiles, f)
		}
		checkedInfo = &types.Info{
			Types:      map[ast.Expr]types.TypeAndValue{},
			Defs:       map[*ast.Ident]types.Object{},
			Uses:       map[*ast.Ident]types.Object{},
			Selections: map[*ast.SelectorExpr]*types.Selection{},
		}
		conf := types.Config{Importer: importer.ForCompiler(checkedFset, "source", nil)}
		checkedPkg, checkedErr = conf.Check("main", checkedFset, checkedFiles, checkedInfo)
	})
	if checkedErr != nil {
		t.Fatal(checkedErr)
	}
	return checkedPkg, checkedFiles, checkedInfo
}
