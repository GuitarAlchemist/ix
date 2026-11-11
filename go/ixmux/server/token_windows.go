package server

import (
	"errors"
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

// advapi32 is a KnownDLL: LoadLibrary takes it from the system directory, never from the
// executable's directory.
var procSDDLToSD = syscall.NewLazyDLL("advapi32.dll").NewProc("ConvertStringSecurityDescriptorToSecurityDescriptorW")

// writeSecret creates path holding b, readable and writable by the current user only: its DACL is
// protected (no inherited entry) and grants full access to the user's SID alone. Any failure
// leaves no file and refuses the start.
// @ai:invariant the token file's DACL grants the current user alone [T:test conf:0.8 src:go/ixmux/cmd/ixmux/integration_test.go:TestTokenFileIsTheUsersOnly]
func writeSecret(path string, b []byte) (err error) {
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	tok, err := syscall.OpenCurrentProcessToken()
	if err != nil {
		return err
	}
	defer tok.Close()
	u, err := tok.GetTokenUser()
	if err != nil {
		return err
	}
	sid, err := u.User.Sid.String()
	if err != nil {
		return err
	}
	sddl, err := syscall.UTF16PtrFromString("D:P(A;;FA;;;" + sid + ")")
	if err != nil {
		return err
	}
	var sd uintptr
	if r, _, e := procSDDLToSD.Call(uintptr(unsafe.Pointer(sddl)), 1, uintptr(unsafe.Pointer(&sd)), 0); r == 0 {
		return fmt.Errorf("build the security descriptor: %w", e)
	}
	defer syscall.LocalFree(syscall.Handle(sd))
	sa := syscall.SecurityAttributes{SecurityDescriptor: sd}
	sa.Length = uint32(unsafe.Sizeof(sa))
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return err
	}
	h, err := syscall.CreateFile(p, syscall.GENERIC_WRITE, 0, &sa, syscall.CREATE_NEW, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return err
	}
	f := os.NewFile(uintptr(h), path)
	defer func() {
		if cerr := f.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			os.Remove(path)
		}
	}()
	if _, err := f.Write(b); err != nil {
		return err
	}
	return f.Sync()
}
