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

// ready writes r as the host's host.ready line, stamped with the time.
func (l *hostLog) ready(r hostReady) {
	r.Event, r.T = hostReadyEvent, time.Now().UnixMilli()
	b, _ := json.Marshal(r)
	l.mu.Lock()
	l.w.Write(append(b, '\n'))
	l.mu.Unlock()
}

// lastReady finds the last host.ready record of the host log at path, through read (the last
// max bytes of a file, never written). A line that does not parse is skipped: the first may be
// cut by the tail read, the last still being written. When path's tail holds none, path.1's is
// read: the rotation (rotatingLog) moves the file there once it passes its size, and a host that
// said ready long ago has its line there; a newer line in path always wins. A path.1 that cannot
// be read is as if it held none. from is the file the record was found in.
//
// @ai:invariant the ready record is the last one of path, else of path.1, never an older one when path holds one [P:assumed conf:0.8] (pinned by TestTheReadyLineIsFoundAcrossTheRotation, TestTheCurrentLogsReadyLineWinsOverTheRotatedOne)
func lastReady(read func(path string, max int64) ([]byte, error), path string, max int64) (r hostReady, from string, found bool, err error) {
	tail, err := read(path, max)
	if err != nil {
		return hostReady{}, "", false, err
	}
	if r, ok := lastReadyIn(tail); ok {
		return r, path, true, nil
	}
	if old, err := read(path+".1", max); err == nil {
		if r, ok := lastReadyIn(old); ok {
			return r, path + ".1", true, nil
		}
	}
	return hostReady{}, "", false, nil
}

func lastReadyIn(tail []byte) (hostReady, bool) {
	lines := strings.Split(string(tail), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		var r hostReady
		if json.Unmarshal([]byte(lines[i]), &r) == nil && r.Event == hostReadyEvent {
			return r, true
		}
	}
	return hostReady{}, false
}
