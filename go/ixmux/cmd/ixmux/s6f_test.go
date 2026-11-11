package main

import (
	"bufio"
	"encoding/hex"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/server"
)

// tcpIxmux is a fake ixmux at the TCP level: serve gets each accepted connection, with a small
// send buffer, so what it manages to write is about what the adapter read. The state directory
// is written as for rawIxmux.
func tcpIxmux(t *testing.T, serve func(c net.Conn, br *bufio.Reader)) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var conns []net.Conn
	var wg sync.WaitGroup
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.(*net.TCPConn).SetWriteBuffer(4 << 10)
			mu.Lock()
			conns = append(conns, c)
			mu.Unlock()
			wg.Add(1)
			go func() { defer wg.Done(); defer c.Close(); serve(c, bufio.NewReader(c)) }()
		}
	}()
	t.Cleanup(func() {
		ln.Close()
		mu.Lock()
		for _, c := range conns {
			c.Close()
		}
		mu.Unlock()
		wg.Wait()
	})
	state := t.TempDir()
	info, _ := json.Marshal(server.Info{Addr: ln.Addr().String(), State: "running"})
	os.WriteFile(filepath.Join(state, server.InfoFile), info, 0o600)
	os.WriteFile(filepath.Join(state, server.TokenFile), []byte(fakeToken), 0o600)
	return state
}

// readReq reads one request and its body; it reports the nonce of a hello.
func readReq(br *bufio.Reader) (nonce []byte, ok bool) {
	r, err := http.ReadRequest(br)
	if err != nil {
		return nil, false
	}
	b, _ := io.ReadAll(r.Body)
	var h struct {
		Nonce string `json:"nonce"`
	}
	json.Unmarshal(b, &h)
	nonce, _ = hex.DecodeString(h.Nonce)
	return nonce, true
}

// endless writes head, then unit again and again until the connection fails; it counts the bytes.
func endless(c net.Conn, head, unit string, n *atomic.Int64) {
	if k, err := io.WriteString(c, head); err != nil {
		return
	} else {
		n.Add(int64(k))
	}
	for {
		k, err := io.WriteString(c, unit)
		n.Add(int64(k))
		if err != nil {
			return
		}
	}
}

// What the adapter may read before it fails, kernel buffers included: far below what an
// unbounded reader takes in its 5 s.
const s6fMaxWritten = 4 << 20

// S6F oracle 1 (Codex round 12, P1) and oracle 2 (P2): a process on the port, without the token,
// answers the hello with an endless header, endless header lines, a body that never ends
// (Content-Length or chunked). The adapter refuses within a second, having read a bounded amount.
func TestHelloAnswerIsBounded(t *testing.T) {
	big := strings.Repeat("a", 1<<10)
	for name, c := range map[string]struct{ head, unit string }{
		"endless header line":  {"HTTP/1.1 200 OK\r\nX-Pad: ", big},
		"endless header lines": {"HTTP/1.1 200 OK\r\n", "X-Pad: " + big + "\r\n"},
		"endless body":         {"HTTP/1.1 200 OK\r\nContent-Length: 1073741824\r\n\r\n", big},
		"endless chunked body": {"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n", "400\r\n" + big + "\r\n"},
	} {
		t.Run(name, func(t *testing.T) {
			var written atomic.Int64
			state := tcpIxmux(t, func(conn net.Conn, br *bufio.Reader) {
				if _, ok := readReq(br); ok {
					endless(conn, c.head, c.unit, &written)
				}
			})
			began := time.Now()
			rc := newInproc(t, state).tool("ixmux_status", map[string]any{})
			took := time.Since(began)
			if !rc.IsError || rc.Structured["error"] != reasonUnverified {
				t.Errorf("got %v, want isError %s", rc.Structured, reasonUnverified)
			}
			if took > time.Second {
				t.Errorf("refused after %v, want within 1s", took)
			}
			if w := written.Load(); w > s6fMaxWritten {
				t.Errorf("the fake wrote %d bytes before the adapter stopped, want at most %d", w, s6fMaxWritten)
			} else {
				t.Logf("refused after %v; the fake wrote %d bytes", took, w)
			}
		})
	}
}

// S6F oracle 1, after the hello: a server that holds the token answers the authenticated request
// with an endless header. Refused ixmux-answer-too-large within a second, a bounded read.
func TestAuthenticatedAnswerHeadersAreBounded(t *testing.T) {
	big := strings.Repeat("a", 1<<10)
	var written atomic.Int64
	state := tcpIxmux(t, func(conn net.Conn, br *bufio.Reader) {
		nonce, ok := readReq(br)
		if !ok {
			return
		}
		mac, _ := json.Marshal(map[string]string{"mac": hex.EncodeToString(server.HelloMAC([]byte(fakeToken), nonce))})
		io.WriteString(conn, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "+
			strconv.Itoa(len(mac))+"\r\n\r\n"+string(mac))
		if _, ok := readReq(br); ok {
			endless(conn, "HTTP/1.1 200 OK\r\nX-Pad: ", big, &written)
		}
	})
	began := time.Now()
	rc := newInproc(t, state).tool("ixmux_status", map[string]any{})
	took := time.Since(began)
	if !rc.IsError || rc.Structured["error"] != reasonAnswerTooLarge {
		t.Errorf("got %v, want isError %s", rc.Structured, reasonAnswerTooLarge)
	}
	if took > time.Second {
		t.Errorf("refused after %v, want within 1s", took)
	}
	if w := written.Load(); w > s6fMaxWritten {
		t.Errorf("the fake wrote %d bytes before the adapter stopped, want at most %d", w, s6fMaxWritten)
	} else {
		t.Logf("refused after %v; the fake wrote %d bytes", took, w)
	}
}
