package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"net"
	"strconv"
	"unsafe"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/svc"
	"golang.org/x/sys/windows/svc/mgr"
)

// queryService reads a service's state and start type with query rights only
// (SC_MANAGER_CONNECT, SERVICE_QUERY_STATUS|SERVICE_QUERY_CONFIG): no elevation, no control.
func queryService(name string) (state, start string, err error) {
	m, err := windows.OpenSCManager(nil, nil, windows.SC_MANAGER_CONNECT)
	if err != nil {
		return "", "", err
	}
	defer windows.CloseServiceHandle(m)
	n, err := windows.UTF16PtrFromString(name)
	if err != nil {
		return "", "", err
	}
	h, err := windows.OpenService(m, n, windows.SERVICE_QUERY_STATUS|windows.SERVICE_QUERY_CONFIG)
	if errors.Is(err, windows.ERROR_SERVICE_DOES_NOT_EXIST) {
		return "not installed", "none", nil
	}
	if err != nil {
		return "", "", err
	}
	s := &mgr.Service{Name: name, Handle: h}
	defer s.Close()
	st, err := s.Query()
	if err != nil {
		return "", "", err
	}
	cfg, err := s.Config()
	if err != nil {
		return "", "", err
	}
	state = map[svc.State]string{
		svc.Stopped: "stopped", svc.StartPending: "start pending", svc.StopPending: "stop pending",
		svc.Running: "running", svc.ContinuePending: "continue pending", svc.PausePending: "pause pending",
		svc.Paused: "paused",
	}[st.State]
	switch cfg.StartType {
	case mgr.StartAutomatic:
		start = "automatic"
		if cfg.DelayedAutoStart {
			start = "automatic (delayed)"
		}
	case mgr.StartManual:
		start = "manual"
	case mgr.StartDisabled:
		start = "disabled"
	default:
		start = "type " + strconv.Itoa(int(cfg.StartType))
	}
	return state, start, nil
}

var procGetExtendedTcpTable = windows.NewLazySystemDLL("iphlpapi.dll").NewProc("GetExtendedTcpTable")

// tcpListening reads the IPv4 listener table (GetExtendedTcpTable,
// TCP_TABLE_OWNER_PID_LISTENER): addr is held when a listener has its port on its address or on
// 0.0.0.0. Nothing connects to the port.
func tcpListening(addr string) (bool, error) {
	host, portText, err := net.SplitHostPort(addr)
	if err != nil {
		return false, err
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		return false, err
	}
	ip := net.ParseIP(host).To4()
	if ip == nil {
		return false, fmt.Errorf("%s: only IPv4 addresses are read", addr)
	}
	const afInet, tcpTableOwnerPidListener = 2, 3
	size := uint32(0)
	var buf []byte
	for i := 0; i < 4; i++ {
		var p uintptr
		if len(buf) > 0 {
			p = uintptr(unsafe.Pointer(&buf[0]))
		}
		r, _, _ := procGetExtendedTcpTable.Call(p, uintptr(unsafe.Pointer(&size)), 0, afInet, tcpTableOwnerPidListener, 0)
		if r == 0 {
			break
		}
		if windows.Errno(r) != windows.ERROR_INSUFFICIENT_BUFFER {
			return false, fmt.Errorf("GetExtendedTcpTable: %w", windows.Errno(r))
		}
		buf = make([]byte, size)
		if i == 3 {
			return false, errors.New("GetExtendedTcpTable: the table kept growing")
		}
	}
	if len(buf) < 4 {
		return false, nil
	}
	// MIB_TCPTABLE_OWNER_PID: dwNumEntries, then rows of six DWORDs: state, local address,
	// local port (network order in the low 16 bits), remote address, remote port, owning pid.
	n := binary.LittleEndian.Uint32(buf)
	for i := uint32(0); i < n; i++ {
		row := buf[4+24*i : 4+24*(i+1)]
		local := net.IP(row[4:8])
		p := int(row[8])<<8 | int(row[9])
		if p == port && (local.Equal(ip) || local.Equal(net.IPv4zero)) {
			return true, nil
		}
	}
	return false, nil
}
