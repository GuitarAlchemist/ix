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
// image is the configured command line (BinaryPathName), read from the same configuration.
func queryService(name string) (state, start, image string, err error) {
	m, err := windows.OpenSCManager(nil, nil, windows.SC_MANAGER_CONNECT)
	if err != nil {
		return "", "", "", err
	}
	defer windows.CloseServiceHandle(m)
	n, err := windows.UTF16PtrFromString(name)
	if err != nil {
		return "", "", "", err
	}
	h, err := windows.OpenService(m, n, windows.SERVICE_QUERY_STATUS|windows.SERVICE_QUERY_CONFIG)
	if errors.Is(err, windows.ERROR_SERVICE_DOES_NOT_EXIST) {
		return "not installed", "none", "", nil
	}
	if err != nil {
		return "", "", "", err
	}
	s := &mgr.Service{Name: name, Handle: h}
	defer s.Close()
	st, err := s.Query()
	if err != nil {
		return "", "", "", err
	}
	cfg, err := s.Config()
	if err != nil {
		return "", "", "", err
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
	return state, start, cfg.BinaryPathName, nil
}

var procGetExtendedTcpTable = windows.NewLazySystemDLL("iphlpapi.dll").NewProc("GetExtendedTcpTable")

// tcpListening reports whether a listener holds addr (see tcpListenOwner). Nothing connects to
// the port.
func tcpListening(addr string) (bool, error) {
	pid, err := tcpListenOwner(addr)
	return pid != 0, err
}

// tcpListenOwner reads the IPv4 listener table (GetExtendedTcpTable,
// TCP_TABLE_OWNER_PID_LISTENER) and returns the PID of a listener with addr's port on addr's
// address or on 0.0.0.0, 0 when there is none.
func tcpListenOwner(addr string) (uint32, error) {
	host, portText, err := net.SplitHostPort(addr)
	if err != nil {
		return 0, err
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		return 0, err
	}
	ip := net.ParseIP(host).To4()
	if ip == nil {
		return 0, fmt.Errorf("%s: only IPv4 addresses are read", addr)
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
			return 0, fmt.Errorf("GetExtendedTcpTable: %w", windows.Errno(r))
		}
		buf = make([]byte, size)
		if i == 3 {
			return 0, errors.New("GetExtendedTcpTable: the table kept growing")
		}
	}
	return listenerRowOwner(buf, ip, port), nil
}

// listenerRowOwner reads a MIB_TCPTABLE_OWNER_PID listener table and returns the PID of the first
// row with port on ip or on 0.0.0.0 (a wildcard bind holds every address), 0 when there is none.
// The table is dwNumEntries, then rows of six DWORDs: state, local address, local port (network
// order in the low 16 bits), remote address, remote port, owning pid. Rows the buffer does not
// hold are not read.
func listenerRowOwner(buf []byte, ip net.IP, port int) uint32 {
	if len(buf) < 4 {
		return 0
	}
	n := binary.LittleEndian.Uint32(buf)
	for i := uint32(0); i < n && 4+24*(int(i)+1) <= len(buf); i++ {
		row := buf[4+24*i : 4+24*(i+1)]
		local := net.IP(row[4:8])
		p := int(row[8])<<8 | int(row[9])
		if p == port && (local.Equal(ip) || local.Equal(net.IPv4zero)) {
			return binary.LittleEndian.Uint32(row[20:24])
		}
	}
	return 0
}

// processImage opens pid with PROCESS_QUERY_LIMITED_INFORMATION only and returns its image path.
// The handle stays open until release, so the PID cannot be reused meanwhile.
func processImage(pid uint32) (string, func(), error) {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, pid)
	if err != nil {
		return "", nil, err
	}
	buf := make([]uint16, windows.MAX_LONG_PATH)
	n := uint32(len(buf))
	if err := windows.QueryFullProcessImageName(h, 0, &buf[0], &n); err != nil {
		_ = windows.CloseHandle(h)
		return "", nil, err
	}
	return windows.UTF16ToString(buf[:n]), func() { _ = windows.CloseHandle(h) }, nil
}
