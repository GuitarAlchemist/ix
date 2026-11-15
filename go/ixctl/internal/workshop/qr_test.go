// Copied unchanged from Delta (github.com/GuitarAlchemist/brickyard, brain/qr_test.go at 3a15210), MIT License.

package workshop

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"testing"
)

// The data codewords of "HELLO WORLD" at version 1, level M, and their error correction, as thonky.com's QR code
// tutorial works them out.
func TestReedSolomonGivesThePublishedErrorCorrection(t *testing.T) {
	data := []byte{32, 91, 11, 120, 209, 114, 220, 77, 67, 64, 236, 17, 236, 17, 236, 17}
	want := []byte{196, 35, 39, 119, 235, 215, 231, 226, 93, 23}
	if got := qrCorrection(data, 10); !bytes.Equal(got, want) {
		t.Fatalf("%v, want %v", got, want)
	}
}

// The format strings of level M, from the standard's table C.1.
func TestTheFormatBitsAreThePublishedOnes(t *testing.T) {
	for mask, want := range []string{"101010000010010", "101000100100101", "101111001111100", "101101101001011",
		"100010111111001", "100000011001110", "100111110010111", "100101010100000"} {
		if got := fmt.Sprintf("%015b", qrFormat(mask)); got != want {
			t.Errorf("mask %d: %s, want %s", mask, got, want)
		}
	}
}

func TestTheSymbolIsTheSmallestThatHoldsTheText(t *testing.T) {
	for n, size := range map[int]int{1: 21, 14: 21, 15: 25, 26: 25, 27: 29, 42: 29, 43: 33, 62: 33, 63: 37, 84: 37,
		85: 41, 106: 41} {
		rows, err := qrEncode(strings.Repeat("a", n))
		if err != nil || len(rows) != size {
			t.Fatalf("%d bytes: %d rows, %v; want %d", n, len(rows), err, size)
		}
		for _, row := range rows {
			if len(row) != size || strings.Trim(row, "01") != "" {
				t.Fatalf("%d bytes: row %q", n, row)
			}
		}
	}
	if _, err := qrEncode(strings.Repeat("a", 107)); err == nil {
		t.Fatal("107 bytes do not fit version 6")
	}
}

// The finders, their separators, the timing lines, the alignment pattern, the dark module, and both copies of the
// format bits, naming the mask the data was drawn with.
func TestTheSymbolHasItsFixedPatterns(t *testing.T) {
	for _, text := range []string{"a", "https://192.168.1.20:8810/#code=ABCD-EFGH", strings.Repeat("b", 106)} {
		rows, err := qrEncode(text)
		if err != nil {
			t.Fatal(err)
		}
		n := len(rows)
		dark := func(x, y int) bool { return rows[y][x] == '1' }
		for _, c := range [][2]int{{3, 3}, {n - 4, 3}, {3, n - 4}} {
			for dy := -4; dy <= 4; dy++ {
				for dx := -4; dx <= 4; dx++ {
					x, y := c[0]+dx, c[1]+dy
					if x < 0 || x >= n || y < 0 || y >= n {
						continue
					}
					if d := max(dx, -dx, dy, -dy); dark(x, y) != (d != 2 && d != 4) {
						t.Fatalf("%d: the finder around %v is wrong at %d,%d", n, c, x, y)
					}
				}
			}
		}
		for i := 8; i < n-8; i++ {
			if dark(i, 6) != (i%2 == 0) || dark(6, i) != (i%2 == 0) {
				t.Fatalf("%d: the timing lines are wrong at %d", n, i)
			}
		}
		if a := n - 7; n > 21 {
			for dy := -2; dy <= 2; dy++ {
				for dx := -2; dx <= 2; dx++ {
					if dark(a+dx, a+dy) != (max(dx, -dx, dy, -dy) != 1) {
						t.Fatalf("%d: the alignment pattern is wrong at %d,%d", n, a+dx, a+dy)
					}
				}
			}
		}
		if !dark(8, n-8) {
			t.Fatalf("%d: the dark module is light", n)
		}
		first, second := 0, 0
		for i, at := range [][2]int{{8, 0}, {8, 1}, {8, 2}, {8, 3}, {8, 4}, {8, 5}, {8, 7}, {8, 8}, {7, 8}, {5, 8},
			{4, 8}, {3, 8}, {2, 8}, {1, 8}, {0, 8}} {
			if dark(at[0], at[1]) {
				first |= 1 << i
			}
		}
		for i := range 15 {
			x, y := n-1-i, 8
			if i >= 8 {
				x, y = 8, n-15+i
			}
			if dark(x, y) {
				second |= 1 << i
			}
		}
		mask := -1
		for m := range 8 {
			if qrFormat(m) == first {
				mask = m
			}
		}
		if mask < 0 || second != first {
			t.Fatalf("%d: format bits %015b and %015b", n, first, second)
		}
	}
}

// testdata/qr-pairing.txt is the symbol of a pairing link as OpenCV's QR detector read it back, so a change to the
// encoder that still draws a valid symbol but another one shows here, to be read again before the file is renewed.
func TestThePairingSymbolIsTheOneOpenCVRead(t *testing.T) {
	rows, err := qrEncode("https://192.168.1.20:8810/#code=ABCD-EFGH")
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile("testdata/qr-pairing.txt")
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(rows, "\n") + "\n"; got != strings.ReplaceAll(string(want), "\r\n", "\n") {
		t.Fatalf("the symbol changed:\n%s", got)
	}
}
