package server

import (
	"fmt"
	"os"
	"syscall"
)

// fileIdentity names the file behind f (volume serial and file index): another file later put
// under the same path is another identity.
func fileIdentity(f *os.File) (string, error) {
	var fi syscall.ByHandleFileInformation
	if err := syscall.GetFileInformationByHandle(syscall.Handle(f.Fd()), &fi); err != nil {
		return "", err
	}
	return fmt.Sprintf("%08x:%08x%08x", fi.VolumeSerialNumber, fi.FileIndexHigh, fi.FileIndexLow), nil
}
