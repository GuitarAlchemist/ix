package main

import (
	"encoding/binary"
	"net"
	"testing"
)

// A wildcard bind (0.0.0.0) holds every address, so it holds the inventory's 127.0.0.1:port. The
// rule is tested on a fake MIB_TCPTABLE_OWNER_PID: a real 0.0.0.0 bind would open the port on
// every interface (and may raise the firewall prompt), which no test may do.

// tcpTable builds a MIB_TCPTABLE_OWNER_PID listener table from rows of (address, port, pid).
func tcpTable(rows ...struct {
	ip   string
	port int
	pid  uint32
}) []byte {
	buf := make([]byte, 4+24*len(rows))
	binary.LittleEndian.PutUint32(buf, uint32(len(rows)))
	for i, r := range rows {
		row := buf[4+24*i : 4+24*(i+1)]
		binary.LittleEndian.PutUint32(row[0:4], 2) // MIB_TCP_STATE_LISTEN
		copy(row[4:8], net.ParseIP(r.ip).To4())
		row[8], row[9] = byte(r.port>>8), byte(r.port)
		binary.LittleEndian.PutUint32(row[20:24], r.pid)
	}
	return buf
}

type tcpRow = struct {
	ip   string
	port int
	pid  uint32
}

func TestListenerRowOwnerWildcard(t *testing.T) {
	loopback := net.ParseIP("127.0.0.1").To4()
	table := tcpTable(
		tcpRow{"10.0.0.5", 8761, 7},  // another address, same port: not ours
		tcpRow{"127.0.0.1", 9000, 9}, // ours, another port
		tcpRow{"0.0.0.0", 8761, 42},  // the wildcard bind
	)
	for _, c := range []struct {
		name string
		buf  []byte
		port int
		want uint32
	}{
		{"wildcard holds loopback", table, 8761, 42},
		{"exact address", table, 9000, 9},
		{"nobody on the port", table, 8762, 0},
		{"another address only", tcpTable(tcpRow{"10.0.0.5", 8761, 7}), 8761, 0},
		{"empty table", tcpTable(), 8761, 0},
		{"short buffer", []byte{1, 0}, 8761, 0},
		// A count past the rows the buffer holds: the rows held are read, the rest never.
		{"count larger than the rows", append([]byte{9, 0, 0, 0}, tcpTable(tcpRow{"10.0.0.5", 8761, 7})[4:]...), 8761, 0},
	} {
		if got := listenerRowOwner(c.buf, loopback, c.port); got != c.want {
			t.Errorf("%s: pid %d, want %d", c.name, got, c.want)
		}
	}
}
