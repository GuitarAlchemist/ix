package main

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// sseFrames reads /v1/events with the given query for d, or until n data frames, and returns
// the envelopes and the gaps, in order.
func sseFrames(t *testing.T, in *instance, query string, n int, d time.Duration, header ...string) ([]Envelope, []Gap) {
	t.Helper()
	req, _ := http.NewRequest("GET", "http://"+in.addr+"/v1/events?"+query, nil)
	req.Header.Set("Authorization", "Bearer "+in.token)
	for i := 0; i+1 < len(header); i += 2 {
		req.Header.Set(header[i], header[i+1])
	}
	ctx, cancel := context.WithTimeout(context.Background(), d)
	defer cancel()
	resp, err := loopbackHTTP.Do(req.WithContext(ctx))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b := make([]byte, 300)
		k, _ := resp.Body.Read(b)
		t.Fatalf("GET /v1/events?%s: %d %s", query, resp.StatusCode, b[:k])
	}
	var envs []Envelope
	var gaps []Gap
	sc := bufio.NewScanner(resp.Body)
	kind := ""
	for len(envs) < n && sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			kind = line[7:]
		case strings.HasPrefix(line, "data: ") && kind == "gap":
			var g Gap
			json.Unmarshal([]byte(line[6:]), &g)
			gaps = append(gaps, g)
		case strings.HasPrefix(line, "data: "):
			var e Envelope
			if err := json.Unmarshal([]byte(line[6:]), &e); err != nil {
				t.Fatal(err)
			}
			envs = append(envs, e)
		}
	}
	return envs, gaps
}

func strp(s string) *string { return &s }

func sampleSeqs(envs []Envelope) []float64 {
	var out []float64
	for _, e := range envs {
		if e.Ev.Kind != "sample" {
			continue
		}
		var m map[string]any
		json.Unmarshal(e.Ev.Data, &m)
		out = append(out, m["seq"].(float64))
	}
	return out
}

// A bookmark names one point of the instance's sequence, which every channel shares: set
// before a probe tree starts, it replays the tree's pipes and the ctl lifecycle in the order
// they happened, seq for seq with what a live reader saw; an unknown or ill-formed bookmark,
// a bookmark beside since, and the bound on their number are refused by name.
func TestABookmarkReplaysEveryChannelFromOnePoint(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "a", nil), "")
	c := dial(t, in)
	var before struct {
		Bookmark string `json:"bookmark"`
		Epoch    string `json:"epoch"`
		Seq      uint64 `json:"seq"`
	}
	json.Unmarshal(do(t, c, Request{Op: "bookmark.set", Bookmark: "before", Note: strp("before the tree")}), &before)
	if before.Bookmark != "before" || before.Epoch != in.plane.Epoch() || before.Seq != in.plane.hub.Head() {
		t.Fatalf("bookmark.set here: %+v (head %d)", before, in.plane.hub.Head())
	}
	live := subscribe(t, c, "probe/hb", nil)
	do(t, c, Request{Op: "probe.create", Probe: node("agg", "mux@1", 0, nil)})
	do(t, c, Request{Op: "probe.create", Probe: node("hb", "heartbeat@1", 10, map[string]any{"parent": "agg"})})
	do(t, c, Request{Op: "probe.start", Node: "agg"})
	do(t, c, Request{Op: "probe.start", Node: "hb"})
	var seen []uint64
	for len(seen) < 5 {
		e := waitEnv(t, live, 2*time.Second, "a live sample", func(e *Envelope) bool { return e.Ev.Kind == "sample" })
		seen = append(seen, e.Seq)
	}
	do(t, c, Request{Op: "bookmark.set", Bookmark: "after"})
	// The replay of the leaf's pipe from `before`: the same envelopes, seq for seq.
	_, frames, err := c.SubscribeBookmark(context.Background(), "probe/hb", "before", 256)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		f := <-frames
		if f.Gap != nil || f.Env == nil || f.Env.Seq != seen[i] {
			t.Fatalf("replayed frame %d: %+v, want seq %d", i, f, seen[i])
		}
	}
	// The replay of everything from `before`: the lifecycle on ctl and both pipes in one
	// stream, in sequence order, nothing before the bookmark — what the pipeline saw.
	_, all, err := c.SubscribeBookmark(context.Background(), "", "before", 256)
	if err != nil {
		t.Fatal(err)
	}
	var kinds []string
	last := before.Seq
	for len(kinds) < 10 {
		f := <-all
		if f.Gap != nil || f.Env.Seq <= last {
			t.Fatalf("replay of everything: %+v after seq %d", f, last)
		}
		last = f.Env.Seq
		kinds = append(kinds, f.Env.Ch+":"+f.Env.Ev.Kind)
	}
	joined := strings.Join(kinds, " ")
	if kinds[0] != "ctl:probe.created" || strings.Count(joined, "ctl:probe.started") != 2 ||
		!strings.Contains(joined, "probe/hb:sample") || !strings.Contains(joined, "probe/agg:sample") {
		t.Errorf("replay of everything from before: %v", kinds)
	}
	// Over HTTP: `bookmark=` in place of since/epoch gives the same replay; a reconnect of
	// that stream (an EventSource sends the same URL back with Last-Event-ID) resumes where
	// it was, the header winning over the bookmark.
	envs, gaps := sseFrames(t, in, "ch=probe/hb&bookmark=before", 5, 3*time.Second)
	if len(gaps) != 0 || len(envs) != 5 || envs[0].Seq != seen[0] || envs[4].Seq != seen[4] {
		t.Errorf("SSE from the bookmark: %d envelopes, gaps %+v", len(envs), gaps)
	}
	envs, gaps = sseFrames(t, in, "ch=probe/hb&bookmark=before", 2, 3*time.Second,
		"Last-Event-ID", in.plane.Epoch()+":"+strconv.FormatUint(seen[2], 10))
	if len(gaps) != 0 || len(envs) != 2 || envs[0].Seq != seen[3] || envs[1].Seq != seen[4] {
		t.Errorf("SSE reconnect with Last-Event-ID beside the bookmark: %d envelopes, gaps %+v", len(envs), gaps)
	}
	// The list: both, in creation order, with their point and whether this boot made them.
	var list struct {
		Bookmarks []struct {
			Name  string `json:"name"`
			Epoch string `json:"epoch"`
			Seq   uint64 `json:"seq"`
			Note  string `json:"note"`
			Live  bool   `json:"live"`
		} `json:"bookmarks"`
		Count        int `json:"count"`
		MaxBookmarks int `json:"max_bookmarks"`
		Ring         int `json:"ring"`
	}
	json.Unmarshal(do(t, c, Request{Op: "bookmark.list"}), &list)
	if list.Count != 2 || list.MaxBookmarks != maxBookmarks || list.Ring != ringSize ||
		list.Bookmarks[0].Name != "before" || list.Bookmarks[0].Note != "before the tree" || !list.Bookmarks[0].Live ||
		list.Bookmarks[1].Name != "after" || list.Bookmarks[1].Seq <= list.Bookmarks[0].Seq {
		t.Errorf("bookmark.list %+v", list)
	}
	// Refusals name what they refuse.
	for _, r := range []struct {
		req  Request
		word string
	}{
		{Request{Op: "subscribe", Ch: "probe/hb", Bookmark: "nope"}, "nope"},
		{Request{Op: "subscribe", Ch: "probe/hb", Bookmark: "before", Since: uint64p(1)}, "bookmark or since"},
		{Request{Op: "bookmark.set", Bookmark: "bad name!"}, "name"},
		{Request{Op: "bookmark.set", Bookmark: "ahead", Since: uint64p(1 << 40)}, "head"},
		{Request{Op: "bookmark.set", Bookmark: "other", Epoch: "0ddb00700000", Since: uint64p(1)}, "epoch"},
		{Request{Op: "bookmark.set", Bookmark: "long", Note: strp(strings.Repeat("é", maxNoteLen+1))}, "max_note"},
		{Request{Op: "bookmark.remove", Bookmark: "nope"}, "nope"},
		{Request{Op: "bookmark.set"}, "bookmark"},
	} {
		if _, err := c.Do(context.Background(), r.req); err == nil || !strings.Contains(err.Error(), r.word) {
			t.Errorf("%s %+v: err %v, want a refusal naming %q", r.req.Op, r.req.Bookmark, err, r.word)
		}
	}
	for i := list.Count; i < maxBookmarks; i++ {
		do(t, c, Request{Op: "bookmark.set", Bookmark: "b" + itoa(i)})
	}
	if _, err := c.Do(context.Background(), Request{Op: "bookmark.set", Bookmark: "one-too-many"}); err == nil || !strings.Contains(err.Error(), "max_bookmarks") {
		t.Errorf("past the bound: %v", err)
	}
	// Setting an existing name moves it: a bookmark is a cursor the operator drags. Its
	// note stays unless a new one is given; an empty one clears it. A note of exactly the
	// bound, in characters, is accepted.
	do(t, c, Request{Op: "bookmark.set", Bookmark: "before"})
	json.Unmarshal(do(t, c, Request{Op: "bookmark.list"}), &list)
	if list.Count != maxBookmarks || list.Bookmarks[0].Name != "before" || list.Bookmarks[0].Seq < list.Bookmarks[1].Seq ||
		list.Bookmarks[0].Note != "before the tree" {
		t.Errorf("after moving before: %+v", list.Bookmarks[:2])
	}
	do(t, c, Request{Op: "bookmark.set", Bookmark: "before", Note: strp("")})
	do(t, c, Request{Op: "bookmark.set", Bookmark: "after", Note: strp(strings.Repeat("é", maxNoteLen))})
	json.Unmarshal(do(t, c, Request{Op: "bookmark.list"}), &list)
	if list.Bookmarks[0].Note != "" || len([]rune(list.Bookmarks[1].Note)) != maxNoteLen {
		t.Errorf("notes after clearing and a full-length one: %q, %d runes", list.Bookmarks[0].Note, len([]rune(list.Bookmarks[1].Note)))
	}
	if raw := do(t, c, Request{Op: "bookmark.remove", Bookmark: "before"}); !strings.Contains(string(raw), `"removed":true`) {
		t.Errorf("remove: %s", raw)
	}
}

// Bookmarks are the plane's, kept in its state dir: a restart keeps them, and a replay from a
// bookmark of the earlier boot says so (epoch-changed) before giving what the new boot holds.
// A bookmark older than the ring says since-too-old. A bookmark set on a peer lives there.
func TestABookmarkSurvivesARestartAndSaysWhatItCannotReplay(t *testing.T) {
	dir := t.TempDir()
	first := newTestInstance(t, fakeSpec(t, "a", nil), dir)
	publishN(first.plane, "svc/x", 3)
	c := dial(t, first)
	do(t, c, Request{Op: "bookmark.set", Bookmark: "mark", Note: strp("boot one")})
	oldEpoch := first.plane.Epoch()
	if _, err := os.Stat(filepath.Join(dir, "a.bookmarks.json")); err != nil {
		t.Fatalf("bookmarks file: %v", err)
	}
	c.Close()
	first.close()

	second := newTestInstance(t, fakeSpec(t, "a", nil), dir)
	publishN(second.plane, "svc/x", 2)
	c = dial(t, second)
	raw := do(t, c, Request{Op: "bookmark.list"})
	if !strings.Contains(string(raw), `"name":"mark"`) || !strings.Contains(string(raw), `"epoch":"`+oldEpoch+`"`) || !strings.Contains(string(raw), `"live":false`) {
		t.Fatalf("after the restart: %s", raw)
	}
	_, frames, err := c.SubscribeBookmark(context.Background(), "svc/", "mark", 16)
	if err != nil {
		t.Fatal(err)
	}
	f := <-frames
	if f.Gap == nil || f.Gap.Reason != "epoch-changed" || f.Gap.Epoch != second.plane.Epoch() {
		t.Fatalf("first frame after a restart: %+v, want an epoch-changed gap", f)
	}
	if f = <-frames; f.Env == nil || f.Env.Epoch != second.plane.Epoch() || f.Env.Ch != "svc/x" || f.Env.Seq > 3 {
		t.Fatalf("after the gap: %+v, want this boot's first svc/x envelope", f)
	}
	// Older than the ring: a bookmark at this boot's seq 1 once the ring has turned over.
	// (A fresh connection: the subscription above would otherwise hold its reader back
	// while the ring turns over.)
	do(t, c, Request{Op: "bookmark.set", Bookmark: "old", Since: uint64p(1)})
	c.Close()
	c = dial(t, second)
	publishN(second.plane, "svc/x", ringSize+10)
	_, frames, err = c.SubscribeBookmark(context.Background(), "svc/", "old", 16)
	if err != nil {
		t.Fatal(err)
	}
	if f = <-frames; f.Gap == nil || f.Gap.Reason != "since-too-old" || f.Gap.Oldest == 0 {
		t.Fatalf("older than the ring: %+v, want since-too-old with the oldest kept", f)
	}
	if f = <-frames; f.Env == nil || f.Env.Seq != second.plane.hub.Head()-ringSize+1 {
		t.Fatalf("after since-too-old: %+v", f)
	}
	// The file is rewritten on every change and holds nothing but bookmarks.
	b, _ := os.ReadFile(filepath.Join(dir, "a.bookmarks.json"))
	if !strings.Contains(string(b), `"old"`) || strings.Contains(string(b), second.token) {
		t.Errorf("bookmarks file: %s", b)
	}

	// A file whose records break what set enforces (here a name twice, and a record
	// without an epoch) refuses the startup, naming the file and the record, rather than
	// being loaded or emptied; a store that cannot save keeps what a reader saw unchanged.
	badDir := t.TempDir()
	os.WriteFile(filepath.Join(badDir, "x.bookmarks.json"), []byte(`{"schema":"ixctl-bookmarks/0","bookmarks":[{"name":"dup","epoch":"e","seq":1,"t":1},{"name":"dup","epoch":"e","seq":2,"t":2}]}`), 0o600)
	if _, err := startInstance(fakeSpec(t, "x", nil), "", "127.0.0.1:0", badDir, "disk"); err == nil || !strings.Contains(err.Error(), "x.bookmarks.json") || !strings.Contains(err.Error(), "dup") {
		t.Fatalf("a duplicate name in the file: %v", err)
	}
	os.WriteFile(filepath.Join(badDir, "x.bookmarks.json"), []byte(`{"schema":"ixctl-bookmarks/0","bookmarks":[{"name":"old","seq":1,"t":1}]}`), 0o600)
	if _, err := startInstance(fakeSpec(t, "x", nil), "", "127.0.0.1:0", badDir, "disk"); err == nil || !strings.Contains(err.Error(), "epoch") {
		t.Fatalf("a record without an epoch: %v", err)
	}
	unsavable := &bookmarkStore{hub: second.plane.hub, path: filepath.Join(badDir, "no-such-dir", "x.bookmarks.json")}
	if _, err := unsavable.set(Request{Bookmark: "lost"}); err == nil {
		t.Fatal("a set the store could not save succeeded")
	}
	if _, err := unsavable.get("lost"); err == nil || len(unsavable.marks) != 0 {
		t.Fatalf("a set that failed to save is visible: %v, %d marks", err, len(unsavable.marks))
	}

	// The store is the operator's choice at startup: `--bookmarks memory` keeps them for
	// this boot only and writes no file; anything else than disk or memory is refused.
	if _, err := startInstance(fakeSpec(t, "m", nil), "", "127.0.0.1:0", t.TempDir(), "cloud"); err == nil || !strings.Contains(err.Error(), "memory") {
		t.Fatalf("store cloud: %v", err)
	}
	memDir := t.TempDir()
	for boot := 1; boot <= 2; boot++ {
		mem, err := startInstance(fakeSpec(t, "m", nil), "", "127.0.0.1:0", memDir, "memory")
		if err != nil {
			t.Fatal(err)
		}
		mem.plane.Start()
		cm := dial(t, mem)
		if boot == 1 {
			do(t, cm, Request{Op: "bookmark.set", Bookmark: "volatile"})
		}
		raw := do(t, cm, Request{Op: "bookmark.list"})
		if !strings.Contains(string(raw), `"store":"memory"`) || strings.Contains(string(raw), `"volatile"`) != (boot == 1) {
			t.Fatalf("memory store, boot %d: %s", boot, raw)
		}
		if _, err := os.Stat(filepath.Join(memDir, "m.bookmarks.json")); !os.IsNotExist(err) {
			t.Fatalf("memory store wrote a file: %v", err)
		}
		cm.Close()
		mem.close()
	}

	// On a peer: set from a, listed on b, b's point.
	dirA, dirB := t.TempDir(), t.TempDir()
	specB := fakeSpec(t, "b", []PeerSpec{{Name: "a", StateFile: filepath.Join(dirA, "a.json")}})
	specA := fakeSpec(t, "a", []PeerSpec{{Name: "b", StateFile: filepath.Join(dirB, "b.json")}})
	pb := newTestInstance(t, specB, dirB)
	pa := newTestInstance(t, specA, dirA)
	waitStatus(t, pa, 10*time.Second, "peer b up", func(st StatusReply) bool { return st.Peers[0].State == "up" })
	code, r := httpJSON(t, pa, "POST", "/v1/bookmark/set", map[string]any{"bookmark": "onb", "peer": "b"})
	if code != 200 || r["epoch"] != pb.plane.Epoch() {
		t.Fatalf("forwarded bookmark.set: %d %v", code, r)
	}
	code, r = httpJSON(t, pa, "GET", "/v1/bookmark?peer=b", nil)
	if code != 200 || r["count"] != 1.0 {
		t.Fatalf("forwarded bookmark.list: %d %v", code, r)
	}
	if code, r = httpJSON(t, pa, "GET", "/v1/bookmark", nil); code != 200 || r["count"] != 0.0 {
		t.Fatalf("a's own list: %d %v", code, r)
	}
	if code, r = httpJSON(t, pa, "POST", "/v1/bookmark/remove", map[string]any{"bookmark": "onb", "peer": "b"}); code != 200 || r["removed"] != true {
		t.Fatalf("forwarded bookmark.remove: %d %v", code, r)
	}
}
