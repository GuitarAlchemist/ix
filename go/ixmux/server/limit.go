package server

import (
	"fmt"
	"io"
	"net"
	"sync"
	"time"
)

// limitListener holds at most cap(slots) accepted connections at a time. One more is answered
// 503 at once and closed, without waiting for a slot.
// @ai:invariant ixmux holds at most MaxHTTP TCP connections; one more is answered 503 and closed [T:test conf:0.85 src:go/ixmux/cmd/ixmux/s3b_test.go:TestHTTPConnectionCap]
type limitListener struct {
	net.Listener
	slots chan struct{}
}

func (l *limitListener) Accept() (net.Conn, error) {
	for {
		c, err := l.Listener.Accept()
		if err != nil {
			return nil, err
		}
		select {
		case l.slots <- struct{}{}:
			return &limitConn{Conn: c, release: sync.OnceFunc(func() { <-l.slots })}, nil
		default:
			go refuseConn(c, cap(l.slots))
		}
	}
}

// refuseConn answers 503 too-many-connections, bounded in time, then closes c.
func refuseConn(c net.Conn, bound int) {
	defer c.Close()
	body := fmt.Sprintf(`{"error":"too-many-connections","detail":"bound %d"}`+"\n", bound)
	c.SetWriteDeadline(time.Now().Add(time.Second))
	io.WriteString(c, fmt.Sprintf("HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: %d\r\n\r\n%s", len(body), body))
}

// limitConn frees its slot once closed, by net/http or, for a stream, by the WebSocket.
type limitConn struct {
	net.Conn
	release func()
}

func (c *limitConn) Close() error {
	err := c.Conn.Close()
	c.release()
	return err
}
