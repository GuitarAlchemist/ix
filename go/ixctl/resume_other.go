//go:build !windows

package main

import (
	"bufio"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
)

func queryService(string) (string, string, error) {
	return "", "", errors.New("service checks read the Windows service manager; this system has none")
}

// tcpListening reads /proc/net/tcp (Linux): addr is held when a LISTEN row (state 0A) has its
// port on its address or on 0.0.0.0. Nothing connects to the port.
func tcpListening(addr string) (bool, error) {
	host, portText, err := net.SplitHostPort(addr)
	if err != nil {
		return false, err
	}
	port, err := strconv.ParseUint(portText, 10, 16)
	if err != nil {
		return false, err
	}
	ip := net.ParseIP(host).To4()
	if ip == nil {
		return false, fmt.Errorf("%s: only IPv4 addresses are read", addr)
	}
	f, err := os.Open("/proc/net/tcp")
	if err != nil {
		return false, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 4 || fields[3] != "0A" {
			continue
		}
		hostHex, portHex, ok := strings.Cut(fields[1], ":")
		if !ok {
			continue
		}
		p, err1 := strconv.ParseUint(portHex, 16, 16)
		raw, err2 := hex.DecodeString(hostHex)
		if err1 != nil || err2 != nil || len(raw) != 4 || p != port {
			continue
		}
		local := net.IPv4(raw[3], raw[2], raw[1], raw[0]) // the kernel prints the address little-endian
		if local.Equal(ip) || local.Equal(net.IPv4zero) {
			return true, nil
		}
	}
	return false, sc.Err()
}
