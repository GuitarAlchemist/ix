package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// service_image_sha256: which build the SCM starts, by content, read from the executable file the
// service's configured command line names. Fake SCM and temp files only.

const (
	ixctlExe  = `C:\ProgramData\ixctl\ixctl.exe`
	ixctlHash = "D5BF23A7D5BF23A7D5BF23A7D5BF23A7D5BF23A7D5BF23A7D5BF23A7D5BF23A7"
	otherHash = "E09A95BAE09A95BAE09A95BAE09A95BAE09A95BAE09A95BAE09A95BAE09A95BA"
)

func TestResumeServiceImageSHA256(t *testing.T) {
	inv := mustInventory(t, `{"components":[{"name":"plane","service":"ixctl","service_image_sha256":"`+strings.ToLower(ixctlHash)+`"}]}`)
	run := func(image string, svcErr error, sum string, hashErr error) (resumeResult, []string) {
		var hashed []string
		p := resumeProbes{
			service: func(string) (string, string, string, error) {
				if svcErr != nil {
					return "", "", "", svcErr
				}
				if image == "" {
					return "not installed", "", "", nil
				}
				return "running", "automatic (delayed)", image, nil
			},
			hashFile: func(path string) (string, error) {
				hashed = append(hashed, path)
				return sum, hashErr
			},
		}
		return checkResume(inv, p)[0], hashed
	}
	// The pinned build: ok, hash compared without case, the file read is the configured executable.
	r, hashed := run(configuredImage, nil, ixctlHash, nil)
	if r.Verdict != verdictOK || !strings.Contains(resumeChecks(r), "service ixctl starts "+ixctlExe+", SHA-256 "+ixctlHash) {
		t.Errorf("pinned build: %s %q, want ok", r.Verdict, r.Checks)
	}
	if len(hashed) != 1 || hashed[0] != ixctlExe {
		t.Errorf("hashed %q, want only %s", hashed, ixctlExe)
	}
	for _, c := range []struct {
		name    string
		image   string
		svcErr  error
		sum     string
		hashErr error
		verdict string
		line    string
	}{
		{"another build", configuredImage, nil, otherHash, nil, verdictDown, "service ixctl starts " + ixctlExe + ", whose SHA-256 is " + otherHash + ", not " + strings.ToLower(ixctlHash)},
		{"unreadable executable", configuredImage, nil, "", errors.New("Access is denied."), verdictUnknown, "service ixctl's executable " + ixctlExe + " cannot be hashed: Access is denied."},
		{"ambiguous command line", `C:\Program Files\ixctl\ixctl.exe service run`, nil, ixctlHash, nil, verdictUnknown, "service ixctl's configured command line cannot be read: unquoted executable"},
		{"SCM refused", "", errors.New("Access is denied."), ixctlHash, nil, verdictUnknown, "service ixctl's configured command line cannot be read: Access is denied."},
		{"not installed", "", nil, ixctlHash, nil, verdictDown, "service ixctl has no configured command line to hash"},
	} {
		r, hashed := run(c.image, c.svcErr, c.sum, c.hashErr)
		if r.Verdict != c.verdict || !strings.Contains(resumeChecks(r), c.line) {
			t.Errorf("%s: %s %q, want %s with %q", c.name, r.Verdict, r.Checks, c.verdict, c.line)
		}
		if c.svcErr != nil || c.image == "" || strings.HasPrefix(c.image, `C:\Program Files`) {
			if len(hashed) != 0 {
				t.Errorf("%s: hashed %q, want nothing read", c.name, hashed)
			}
		}
	}
	for _, c := range []struct{ js, want string }{
		{`{"components":[{"name":"a","listen":"127.0.0.1:1","service_image_sha256":"` + ixctlHash + `"}]}`, "service_image_sha256 without service"},
		{`{"components":[{"name":"a","service":"ixctl","service_image_sha256":"D5BF23A7"}]}`, "service_image_sha256 is not 64 hex digits"},
	} {
		if _, err := parseResumeInventory([]byte(c.js)); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want %q", c.js, err, c.want)
		}
	}
}

// The real hashFile reads the whole file: SHA-256 of "abc" is the FIPS 180-2 test vector.
func TestResumeHashFileIsSHA256(t *testing.T) {
	path := filepath.Join(t.TempDir(), "abc.exe")
	if err := os.WriteFile(path, []byte("abc"), 0o600); err != nil {
		t.Fatal(err)
	}
	sum, err := realResumeProbesOver(nil).hashFile(path)
	if err != nil || !strings.EqualFold(sum, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
		t.Errorf("hashFile(abc) = %q, %v", sum, err)
	}
	if _, err := realResumeProbesOver(nil).hashFile(filepath.Join(t.TempDir(), "absent.exe")); err == nil {
		t.Error("hashFile of an absent file: no error")
	}
}
