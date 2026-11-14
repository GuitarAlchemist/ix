package workshop

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestInviteQRCodeRoundTrip(t *testing.T) {
	if os.Getenv("WORKSHOP_QR_DECODER") == "" {
		t.Skip("set WORKSHOP_QR_DECODER to jsqr dist/jsQR.js or install pinned devDependencies")
	}
	s, _, _ := fixture(t)
	invite, err := s.NewInvite("https://workshop.invalid:54321")
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("node", "e2e/decode-qr.mjs", invite.URL)
	cmd.Stdin = strings.NewReader(invite.QRSVG)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %s", err, output)
	}
	t.Log(string(output))
}
