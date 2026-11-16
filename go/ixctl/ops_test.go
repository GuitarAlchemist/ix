package main

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"
)

// recordingBroker is a peer plane that keeps the last request it was asked, as it arrived.
type recordingBroker struct {
	*Plane
	got chan Request
}

func (r *recordingBroker) Request(ctx context.Context, req Request) (any, error) {
	if req.Op == "status" && req.Service != "" { // the test's request; the link asks status with no service
		select {
		case r.got <- req:
		default:
		}
	}
	return r.Plane.Request(ctx, req)
}

// fullRequest sets every field of a Request to a non-zero value, by reflection: a field added
// later of a kind it covers is set with no change here; one of another kind (a slice, a map, a
// struct) fails the test until it is covered. A pointer field gets a zero value behind it.
func fullRequest(t *testing.T, op string) Request {
	t.Helper()
	var req Request
	v := reflect.ValueOf(&req).Elem()
	for i := 0; i < v.NumField(); i++ {
		f := v.Field(i)
		switch f.Kind() {
		case reflect.String:
			f.SetString("x-" + v.Type().Field(i).Name)
		case reflect.Bool:
			f.SetBool(true)
		case reflect.Int64:
			f.SetInt(7)
		case reflect.Uint64:
			f.SetUint(7)
		case reflect.Pointer:
			f.Set(reflect.New(f.Type().Elem()))
			if e := f.Elem(); e.Kind() == reflect.String {
				e.SetString("x")
			} else if e.Kind() == reflect.Uint64 {
				e.SetUint(7)
			}
		default:
			t.Fatalf("Request.%s: kind %s not covered by fullRequest", v.Type().Field(i).Name, f.Kind())
		}
	}
	req.Op = op
	return req
}

// A relayed op reaches the peer whole: every field the client set, but the peer (an op crosses
// one hop) and the frame id (the link's own).
func TestForwardCarriesTheWholeRequest(t *testing.T) {
	b := memPlane(t, fakeSpec(t, "b", nil), "b")
	rb := &recordingBroker{Plane: b, got: make(chan Request, 1)}
	a := memPlane(t, fakeSpec(t, "a", []PeerSpec{{Name: "b", StateFile: filepath.Join(t.TempDir(), "never-read.json")}}), "a")
	a.peers["b"].connect = func(context.Context) (*Client, stateFile, error) {
		return dialMem(b.ctx, rb), stateFile{Name: "b", Addr: "mem:b", Token: "mem:" + b.Epoch()}, nil
	}
	b.Start()
	a.Start()
	waitPeerState(t, a, "up")

	sent := fullRequest(t, "status")
	sent.Peer = "b"
	if _, err := a.Request(t.Context(), sent); err != nil {
		t.Fatal(err)
	}
	var got Request
	select {
	case got = <-rb.got:
	case <-time.After(5 * time.Second):
		t.Fatal("the forwarded status never reached the peer")
	}
	want := sent
	want.Peer, want.ID = "", got.ID
	if !reflect.DeepEqual(got, want) {
		gv, wv := reflect.ValueOf(got), reflect.ValueOf(want)
		var lost []string
		for i := 0; i < gv.NumField(); i++ {
			if !reflect.DeepEqual(gv.Field(i).Interface(), wv.Field(i).Interface()) {
				lost = append(lost, gv.Type().Field(i).Name)
			}
		}
		t.Fatalf("the peer got the request without %s", strings.Join(lost, ", "))
	}
}

// The relay column of the op table is what forward refuses: every op that is not relayed is
// refused by name before any peer is asked, every relayed one is not.
func TestEveryOpOfTheTableIsRelayedOrRefusedByName(t *testing.T) {
	a := memPlane(t, fakeSpec(t, "a", []PeerSpec{{Name: "b", StateFile: filepath.Join(t.TempDir(), "never-read.json")}}), "a")
	a.peers["b"].connect = func(context.Context) (*Client, stateFile, error) { return nil, stateFile{}, errors.New("down") }
	if len(planeOps) == 0 {
		t.Fatal("empty op table")
	}
	// The ops a peer is never asked: a shutdown is the local host's, a need lives in one
	// plane's store (README: "never forwarded to a peer").
	var local []string
	for name, op := range planeOps {
		if !op.relay {
			local = append(local, name)
		}
	}
	sort.Strings(local)
	if got := strings.Join(local, " "); got != "need.claim need.refuse need.reserve need.resolve need.settle need.status need.submit need.wait shutdown" {
		t.Errorf("ops not relayed: %s", got)
	}
	// The whole table: an op dropped from it would read as "unknown op".
	var all []string
	for name := range planeOps {
		all = append(all, name)
	}
	sort.Strings(all)
	if got := strings.Join(all, " "); got != "apply bookmark.get bookmark.list bookmark.remove bookmark.set "+
		"need.claim need.refuse need.reserve need.resolve need.settle need.status need.submit need.wait "+
		"ping plan probe.create probe.kinds probe.list probe.remove probe.start probe.stop probe.touch "+
		"shutdown status stop wait" {
		t.Errorf("op table: %s", got)
	}
	// The ops that hand a request on themselves, once their own checks pass: the wait alone.
	var own []string
	for name, op := range planeOps {
		if op.ownsPeer {
			own = append(own, name)
		}
	}
	if got := strings.Join(own, " "); got != "wait" {
		t.Errorf("ops that own their peer: %q, want wait", got)
	}
	// The link was never started, so a relayed op reaches forward's peer and finds it down.
	for name, op := range planeOps {
		if op.ownsPeer {
			continue // its run checks the request first, then relays (Plane.Wait)
		}
		_, err := a.Request(t.Context(), Request{Op: name, Peer: "b"})
		want := "op \"" + name + "\" cannot be forwarded"
		if op.relay {
			want = "peer b is down"
		}
		if err == nil || err.Error() != want {
			t.Errorf("%s: forward said %v, want %q", name, err, want)
		}
	}
	// An op that owns its peer checks the request before it hands it on: a malformed wait naming
	// the down peer is refused for its own fields here, never relayed.
	if _, err := a.Request(t.Context(), Request{Op: "wait", Peer: "b"}); err == nil || !strings.Contains(err.Error(), "timeout_ms") {
		t.Errorf("a malformed wait naming a peer: %v, want its own refusal (timeout_ms)", err)
	}
}
