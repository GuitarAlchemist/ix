package main

import (
	"encoding/json"
	"strings"
	"time"
)

// hostReady is the record a host writes to its log once it serves, and the one a resume check
// reads back (a component's ready_journal): which process said ready, where it listens, under
// which epoch. ADR 0014 does not cover this line; this type is its one definition. Fields are in
// key order, as the line was when it was written from a map.
type hostReady struct {
	Addr      string `json:"addr"`
	Epoch     string `json:"epoch"`
	Event     string `json:"event"`
	Name      string `json:"name"`
	Pid       uint32 `json:"pid"`
	StateFile string `json:"state_file"`
	T         int64  `json:"t"`
	Version   string `json:"version"`
}

const hostReadyEvent = "host.ready"

// hostStopEvent is the line a host writes at each phase of its stop: requested and received as
// the stop begins, then one final phase, observed (every tree seen gone within the bound) or
// unobserved (not). A host writes nothing after its final phase: the reader of the ready line
// (lastReady) takes a final stop as that host's end.
const (
	hostStopEvent  = "host.stop"
	stopRequested  = "requested"
	stopReceived   = "received"
	stopObserved   = "observed"
	stopUnobserved = "unobserved"
)

// finalStop says whether a stop phase is a host's last line.
func finalStop(phase string) bool { return phase == stopObserved || phase == stopUnobserved }

// stop writes a host.stop line: its phase, the stop's source, and fields (which it may add to).
func (l *hostLog) stop(phase, source string, fields map[string]any) {
	if fields == nil {
		fields = map[string]any{}
	}
	fields["phase"], fields["source"] = phase, source
	l.event(hostStopEvent, fields)
}

// ready writes r as the host's host.ready line, stamped with the time.
func (l *hostLog) ready(r hostReady) {
	r.Event, r.T = hostReadyEvent, time.Now().UnixMilli()
	b, _ := json.Marshal(r)
	l.mu.Lock()
	l.w.Write(append(b, '\n'))
	l.mu.Unlock()
}

// lastReady finds the current host's host.ready record in the host log at path, through read
// (the last max bytes of a file, never written). Read from the newest line back, the first that
// decides is either a host.ready line (the record) or a host's final host.stop (observed or
// unobserved: that host ended after it said ready, so the host running now, maybe under the same
// PID, has not said ready yet, and nothing is found). A line that does not parse is skipped: the
// first may be cut by the tail read, the last still being written. When path's tail holds
// neither, path.1's is read: the rotation (rotatingLog) moves the file there once it passes its
// size, and a host that said ready before has its line there, while it is within that file's
// last max bytes. A path.1 that cannot be read is as if it held none. The two reads are not one
// snapshot: a rotation between them can hide the line (nothing found, as before the rotation
// was read at all). from is the file the deciding line was found in: the record's, or the
// final stop's when found is false; "" when neither file holds one.
//
// @ai:invariant the record is the newest host.ready of path, else of path.1, unless a final host.stop follows it [P:assumed conf:0.8] (pinned by TestTheReadyLineIsFoundAcrossTheRotation, TestTheCurrentLogsReadyLineWinsOverTheRotatedOne, TestAReadyLineFollowedByTheHostsEndIsNotTheCurrentHosts)
func lastReady(read func(path string, max int64) ([]byte, error), path string, max int64) (r hostReady, from string, found bool, err error) {
	tail, err := read(path, max)
	if err != nil {
		return hostReady{}, "", false, err
	}
	if r, decided, ok := lastReadyIn(tail); decided {
		return r, path, ok, nil
	}
	if old, err := read(path+".1", max); err == nil {
		if r, decided, ok := lastReadyIn(old); decided {
			return r, path + ".1", ok, nil
		}
	}
	return hostReady{}, "", false, nil
}

// lastReadyIn reads tail from its newest line back to the first that decides: a host.ready
// record (ok), or a host's final host.stop (decided, not ok).
func lastReadyIn(tail []byte) (r hostReady, decided, ok bool) {
	lines := strings.Split(string(tail), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		var l struct {
			hostReady
			Phase string `json:"phase"`
		}
		if json.Unmarshal([]byte(lines[i]), &l) != nil {
			continue
		}
		switch {
		case l.Event == hostReadyEvent:
			return l.hostReady, true, true
		case l.Event == hostStopEvent && finalStop(l.Phase):
			return hostReady{}, true, false
		}
	}
	return hostReady{}, false, false
}
