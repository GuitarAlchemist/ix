//go:build !windows

package main

import (
	"net"
	"strings"
	"testing"
)

// The /proc/net/tcp twin of TestListenerRowOwnerWildcard: a fake table, no socket.
func TestProcNetTCPWildcard(t *testing.T) {
	const header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
	// Addresses are printed little-endian: 0100007F is 127.0.0.1, 0500000A is 10.0.0.5. 2239 is 8761.
	table := header +
		"   0: 0500000A:2239 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 1\n" +
		"   1: 0100007F:2328 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 2\n" +
		"   2: 00000000:2239 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 3\n" +
		"   3: 0100007F:223A 0100007F:9C40 01 00000000:00000000 00:00000000 00000000  1000        0 4\n"
	loopback := net.ParseIP("127.0.0.1").To4()
	for _, c := range []struct {
		name  string
		table string
		port  uint64
		want  bool
	}{
		{"wildcard holds loopback", table, 8761, true},
		{"exact address", table, 9000, true},
		{"an established row is no listener", table, 8762, false},
		{"another address only", header + "   0: 0500000A:2239 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 1\n", 8761, false},
	} {
		got, err := procNetTCPHolds(strings.NewReader(c.table), loopback, c.port)
		if err != nil || got != c.want {
			t.Errorf("%s: %v %v, want %v", c.name, got, err, c.want)
		}
	}
}
