package main

import (
	"bytes"
	"reflect"
	"strings"
	"testing"
)

// Three pull requests, given out of order: #12 and #10 both change a.go, #11 is built on
// #10's branch, #12 has a red check, a running one and a commit status in error.
const prsJSON = `[
 {"number":12,"title":"twelve","headRefName":"feat/c","baseRefName":"main","isDraft":false,"mergeable":"MERGEABLE","changedFiles":2,
  "files":[{"path":"a.go"},{"path":"c.go"}],
  "statusCheckRollup":[
   {"name":"build","status":"COMPLETED","conclusion":"FAILURE"},
   {"name":"build","status":"COMPLETED","conclusion":"FAILURE"},
   {"name":"lint","status":"COMPLETED","conclusion":"SUCCESS"},
   {"name":"skipped","status":"COMPLETED","conclusion":"SKIPPED"},
   {"name":"slow","status":"IN_PROGRESS","conclusion":""},
   {"context":"deploy/preview","state":"ERROR"}]},
 {"number":10,"title":"ten","headRefName":"feat/a","baseRefName":"main","isDraft":true,"mergeable":"MERGEABLE","changedFiles":2,
  "files":[{"path":"b.go"},{"path":"a.go"}],"statusCheckRollup":[]},
 {"number":11,"title":"eleven","headRefName":"feat/b","baseRefName":"feat/a","isDraft":true,"mergeable":"CONFLICTING","changedFiles":1,
  "files":[{"path":"d.go"}],"statusCheckRollup":[]}
]`

const claimsJSONL = `{"ts":"2026-11-01T00:00:00Z","session":"aaaaaaaa-1111","branch":"feat/a","status":"in-progress"}
{"ts":"2026-11-02T00:00:00Z","session":"bbbbbbbb-2222","branch":"feat/a","status":"pr-open"}
{"ts":"2026-11-03T00:00:00Z","session":"cccccccc-3333","lane":"no-branch-here","status":"done"}
`

func build(t *testing.T, prs, claims string) Report {
	t.Helper()
	var c *strings.Reader
	if claims != "" {
		c = strings.NewReader(claims)
	}
	var r Report
	var err error
	if c == nil {
		r, err = Build(strings.NewReader(prs), nil, "main")
	} else {
		r, err = Build(strings.NewReader(prs), c, "main")
	}
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	return r
}

func TestPullRequestsComeOutInNumberOrderWhateverTheInputOrder(t *testing.T) {
	r := build(t, prsJSON, claimsJSONL)
	var got []int
	for _, p := range r.PRs {
		got = append(got, p.Number)
	}
	if want := []int{10, 11, 12}; !reflect.DeepEqual(got, want) {
		t.Errorf("order %v, want %v", got, want)
	}
}

func TestAFileChangedByTwoPullRequestsIsAnOverlapAndAFileChangedByOneIsNot(t *testing.T) {
	r := build(t, prsJSON, claimsJSONL)
	want := []Overlap{{Path: "a.go", PRs: []int{10, 12}}}
	if !reflect.DeepEqual(r.Overlaps, want) {
		t.Errorf("overlaps %+v, want %+v", r.Overlaps, want)
	}
}

func TestAPullRequestBuiltOnAnothersBranchNamesThatPullRequest(t *testing.T) {
	r := build(t, prsJSON, claimsJSONL)
	want := []Dependency{{PR: 11, Base: "feat/a", On: 10}}
	if !reflect.DeepEqual(r.Dependencies, want) {
		t.Errorf("dependencies %+v, want %+v", r.Dependencies, want)
	}
}

func TestABaseNoOpenPullRequestOwnsIsSaidSo(t *testing.T) {
	r := build(t, `[{"number":1,"headRefName":"x","baseRefName":"release/1","changedFiles":0,"files":[]}]`, "")
	want := []Dependency{{PR: 1, Base: "release/1", On: 0}}
	if !reflect.DeepEqual(r.Dependencies, want) {
		t.Errorf("dependencies %+v, want %+v", r.Dependencies, want)
	}
	var out bytes.Buffer
	Text(&out, r)
	if !strings.Contains(out.String(), "#1 is built on release/1, which no open pull request owns") {
		t.Errorf("text does not say the base has no pull request:\n%s", out.String())
	}
}

func TestRedChecksAreNamedOnceAndRunningOnesApart(t *testing.T) {
	r := build(t, prsJSON, claimsJSONL)
	p := r.PRs[2]
	if want := []string{"build", "deploy/preview"}; !reflect.DeepEqual(p.Red, want) {
		t.Errorf("red %v, want %v", p.Red, want)
	}
	if want := []string{"slow"}; !reflect.DeepEqual(p.Running, want) {
		t.Errorf("running %v, want %v", p.Running, want)
	}
}

func TestTheOwnerIsTheLastClaimOnTheBranchAndUnknownWithoutOne(t *testing.T) {
	r := build(t, prsJSON, claimsJSONL)
	o := r.PRs[0].Owner
	if o == nil || o.Session != "bbbbbbbb-2222" || o.Status != "pr-open" {
		t.Errorf("#10 owner %+v, want the later claim (bbbbbbbb-2222, pr-open)", o)
	}
	if r.PRs[1].Owner != nil {
		t.Errorf("#11 owner %+v, want none: no claim names feat/b", r.PRs[1].Owner)
	}
	var out bytes.Buffer
	Text(&out, r)
	if !strings.Contains(out.String(), "#11 feat/b -> feat/a  draft  CONFLICTING  owner unknown") {
		t.Errorf("text does not say #11's owner is unknown:\n%s", out.String())
	}
}

func TestWhatTheReportCouldNotKnowIsAWarningNotASilence(t *testing.T) {
	// No ledger at all.
	r := build(t, prsJSON, "")
	if len(r.Warnings) != 1 || !strings.Contains(r.Warnings[0], "no claims ledger") {
		t.Errorf("warnings %v, want the missing ledger named", r.Warnings)
	}
	// A ledger line that is not JSON.
	r = build(t, prsJSON, claimsJSONL+"not json\n")
	if len(r.Warnings) != 1 || !strings.Contains(r.Warnings[0], "1 line(s) are not JSON") {
		t.Errorf("warnings %v, want the unread line counted", r.Warnings)
	}
	// gh lists fewer files than the pull request changes.
	r = build(t, `[{"number":7,"headRefName":"big","baseRefName":"main","changedFiles":250,"files":[{"path":"a"}]}]`, claimsJSONL)
	if len(r.Warnings) != 1 || !strings.Contains(r.Warnings[0], "#7: 1 of 250 changed files listed") {
		t.Errorf("warnings %v, want the truncated file list named", r.Warnings)
	}
}

func TestInputThatIsNotJSONIsAnErrorNotAnEmptyMap(t *testing.T) {
	if _, err := Build(strings.NewReader("gh: not logged in"), nil, "main"); err == nil {
		t.Error("Build accepted input that is not JSON")
	}
	var out, errOut bytes.Buffer
	if code := run([]string{"-in", "-"}, strings.NewReader("{"), &out, &errOut); code != 2 {
		t.Errorf("exit %d, want 2", code)
	}
	if out.Len() != 0 {
		t.Errorf("a report was written on bad input: %q", out.String())
	}
}

func TestTheSameInputGivesTheSameReportByteForByte(t *testing.T) {
	var first string
	for i := 0; i < 20; i++ {
		var out, errOut bytes.Buffer
		if code := run([]string{"-in", "-", "-json"}, strings.NewReader(prsJSON), &out, &errOut); code != 0 {
			t.Fatalf("exit %d: %s", code, errOut.String())
		}
		if i == 0 {
			first = out.String()
		} else if out.String() != first {
			t.Fatalf("run %d differs from run 0", i)
		}
	}
	if !strings.Contains(first, `"overlaps": [`) || !strings.Contains(first, `"owner": null`) {
		t.Errorf("JSON report lacks its fields:\n%s", first)
	}
}
