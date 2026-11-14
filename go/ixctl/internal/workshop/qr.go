// Copied unchanged from Delta (github.com/GuitarAlchemist/brickyard, brain/qr.go at 3a15210), MIT License,
// Copyright (c) 2026 GuitarAlchemist. See THIRD_PARTY.md.

package workshop

import (
	"fmt"
	"slices"
)

// A QR code (ISO/IEC 18004) of the link a phone opens to pair with Delta's remote gateway, for Delta's window to show.
// Byte mode at error correction level M, versions 1 to 6: up to 106 bytes, which a link of an https address, a port
// and a pairing code fits. QR Code is a registered trademark of DENSO WAVE INCORPORATED.

// qrVersion is what the standard's table 9 gives for a version at level M, where all its blocks have one length.
type qrVersion struct{ codewords, ec, blocks int } // all codewords, error correction codewords per block, blocks

func (v qrVersion) data() int { return v.codewords - v.ec*v.blocks }

// qrMaxBytes is what version 6 holds at level M.
const qrMaxBytes = 106

// qrVersions are versions 1 to 6 at level M.
var qrVersions = []qrVersion{{26, 10, 1}, {44, 16, 1}, {70, 26, 1}, {100, 18, 2}, {134, 24, 2}, {172, 16, 4}}

// qrEncode draws text as the smallest symbol that holds it, one string per row from the top, '1' for a dark module.
// The quiet zone of four light modules around it is the drawer's.
func qrEncode(text string) ([]string, error) {
	version, codewords, err := qrCodewords(text)
	if err != nil {
		return nil, err
	}
	s := newQRSymbol(version)
	s.place(codewords)
	best, lowest := 0, -1
	for mask := range 8 {
		s.mask(mask)
		s.format(mask)
		if p := s.penalty(); lowest < 0 || p < lowest {
			best, lowest = mask, p
		}
		s.mask(mask)
	}
	s.mask(best)
	s.format(best)
	return s.rows(), nil
}

// qrCodewords are text's data codewords in byte mode and their error correction, interleaved block by block, and the
// version of the smallest symbol that holds them.
func qrCodewords(text string) (int, []byte, error) {
	version := 0
	for version < len(qrVersions) && 4+8+8*len(text) > 8*qrVersions[version].data() {
		version++
	}
	if version == len(qrVersions) {
		return 0, nil, fmt.Errorf("%d bytes do not fit a QR code of version 6 at level M", len(text))
	}
	v := qrVersions[version]

	var bits []bool
	add := func(value, n int) {
		for i := n - 1; i >= 0; i-- {
			bits = append(bits, value>>i&1 == 1)
		}
	}
	add(4, 4) // byte mode
	add(len(text), 8)
	for i := range len(text) {
		add(int(text[i]), 8)
	}
	add(0, min(4, 8*v.data()-len(bits))) // the terminator
	add(0, -len(bits)&7)
	data := make([]byte, 0, v.data())
	for i := 0; i < len(bits); i += 8 {
		var b byte
		for _, bit := range bits[i : i+8] {
			b <<= 1
			if bit {
				b |= 1
			}
		}
		data = append(data, b)
	}
	for pad := byte(0xEC); len(data) < v.data(); pad ^= 0xEC ^ 0x11 {
		data = append(data, pad)
	}

	per := v.data() / v.blocks
	var codewords []byte
	for i := range per {
		for b := range v.blocks {
			codewords = append(codewords, data[b*per+i])
		}
	}
	var ec [][]byte
	for b := range v.blocks {
		ec = append(ec, qrCorrection(data[b*per:(b+1)*per], v.ec))
	}
	for i := range v.ec {
		for b := range v.blocks {
			codewords = append(codewords, ec[b][i])
		}
	}

	return version + 1, codewords, nil
}

// qrSymbol is a symbol being drawn, indexed [y][x]: which modules are dark, and which belong to a function pattern or
// the format bits rather than to the data.
type qrSymbol struct {
	size        int
	dark, fixed [][]bool
}

// newQRSymbol draws a version's function patterns: the timing lines, the three finders with their separators, the
// alignment pattern of versions 2 to 6, and the format bits' place.
func newQRSymbol(version int) *qrSymbol {
	n := 17 + 4*version
	s := &qrSymbol{size: n}
	for range n {
		s.dark, s.fixed = append(s.dark, make([]bool, n)), append(s.fixed, make([]bool, n))
	}
	for i := range n {
		s.set(6, i, i%2 == 0)
		s.set(i, 6, i%2 == 0)
	}
	for _, c := range [][2]int{{3, 3}, {n - 4, 3}, {3, n - 4}} {
		for dy := -4; dy <= 4; dy++ {
			for dx := -4; dx <= 4; dx++ {
				if x, y := c[0]+dx, c[1]+dy; x >= 0 && x < n && y >= 0 && y < n {
					d := max(dx, -dx, dy, -dy)
					s.set(x, y, d != 2 && d != 4)
				}
			}
		}
	}
	if version > 1 {
		a := n - 7
		for dy := -2; dy <= 2; dy++ {
			for dx := -2; dx <= 2; dx++ {
				s.set(a+dx, a+dy, max(dx, -dx, dy, -dy) != 1)
			}
		}
	}
	s.format(0)
	return s
}

func (s *qrSymbol) rows() []string {
	rows := make([]string, s.size)
	for y, line := range s.dark {
		row := make([]byte, s.size)
		for x, dark := range line {
			row[x] = '0'
			if dark {
				row[x] = '1'
			}
		}
		rows[y] = string(row)
	}
	return rows
}

func (s *qrSymbol) set(x, y int, dark bool) {
	s.dark[y][x] = dark
	s.fixed[y][x] = true
}

// format draws the format bits of a mask, twice, and the dark module beside the bottom-left finder.
func (s *qrSymbol) format(mask int) {
	bits, n := qrFormat(mask), s.size
	bit := func(i int) bool { return bits>>i&1 == 1 }
	for i := range 6 {
		s.set(8, i, bit(i))
	}
	s.set(8, 7, bit(6))
	s.set(8, 8, bit(7))
	s.set(7, 8, bit(8))
	for i := 9; i < 15; i++ {
		s.set(14-i, 8, bit(i))
	}
	for i := range 8 {
		s.set(n-1-i, 8, bit(i))
	}
	for i := 8; i < 15; i++ {
		s.set(8, n-15+i, bit(i))
	}
	s.set(8, n-8, true)
}

// place draws the codewords' bits in the data modules, two columns at a time from the right, up then down, stepping
// over the vertical timing line. The remainder bits left over stay light.
func (s *qrSymbol) place(codewords []byte) {
	n, i := s.size, 0
	for right := n - 1; right >= 1; right -= 2 {
		if right == 6 {
			right = 5
		}
		upward := (right+1)&2 == 0
		for vert := range n {
			y := vert
			if upward {
				y = n - 1 - vert
			}
			for j := range 2 {
				if x := right - j; !s.fixed[y][x] && i < len(codewords)*8 {
					s.dark[y][x] = codewords[i>>3]>>(7-i&7)&1 == 1
					i++
				}
			}
		}
	}
}

// mask inverts the data modules that a mask pattern of the standard's table 10 selects; masking twice unmasks.
func (s *qrSymbol) mask(mask int) {
	for y := range s.size {
		for x := range s.size {
			var selected bool
			switch mask {
			case 0:
				selected = (x+y)%2 == 0
			case 1:
				selected = y%2 == 0
			case 2:
				selected = x%3 == 0
			case 3:
				selected = (x+y)%3 == 0
			case 4:
				selected = (x/3+y/2)%2 == 0
			case 5:
				selected = x*y%2+x*y%3 == 0
			case 6:
				selected = (x*y%2+x*y%3)%2 == 0
			default:
				selected = ((x+y)%2+x*y%3)%2 == 0
			}
			if selected && !s.fixed[y][x] {
				s.dark[y][x] = !s.dark[y][x]
			}
		}
	}
}

// penalty scores a masked symbol by the standard's four rules; the encoder keeps the mask that scores lowest.
func (s *qrSymbol) penalty() int {
	n, score, dark := s.size, 0, 0
	line := make([]bool, n)
	for i := range n {
		for j := range n {
			line[j] = s.dark[i][j]
		}
		score += qrLinePenalty(line)
		for j := range n {
			line[j] = s.dark[j][i]
		}
		score += qrLinePenalty(line)
	}
	for y := range n {
		for x := range n {
			if s.dark[y][x] {
				dark++
			}
			if c := s.dark[y][x]; x+1 < n && y+1 < n && c == s.dark[y][x+1] && c == s.dark[y+1][x] && c == s.dark[y+1][x+1] {
				score += 3
			}
		}
	}
	total := n * n
	return score + 10*((max(dark*20-total*10, total*10-dark*20)+total-1)/total-1)
}

// qrLinePenalty scores a row or a column: each run of five modules or more of one colour, and each 1:1:3:1:1 pattern
// like a finder's with four light modules on a side, where the quiet zone outside counts as light.
func qrLinePenalty(line []bool) int {
	score, run := 0, 1
	for i := 1; i <= len(line); i++ {
		if i < len(line) && line[i] == line[i-1] {
			run++
			continue
		}
		if run >= 5 {
			score += run - 2
		}
		run = 1
	}
	light := func(from, to int) bool {
		for i := from; i < to; i++ {
			if i >= 0 && i < len(line) && line[i] {
				return false
			}
		}
		return true
	}
	finder := []bool{true, false, true, true, true, false, true}
	for i := 0; i+7 <= len(line); i++ {
		if slices.Equal(line[i:i+7], finder) {
			if light(i-4, i) {
				score += 40
			}
			if light(i+7, i+11) {
				score += 40
			}
		}
	}
	return score
}

// qrCorrection is the n error correction codewords of a block: the remainder of its polynomial divided by the
// generator (x - α^0)…(x - α^(n-1)), over GF(2^8) modulo x^8 + x^4 + x^3 + x^2 + 1.
func qrCorrection(data []byte, n int) []byte {
	generator := make([]byte, n) // its coefficients below the leading 1, highest first
	generator[n-1] = 1
	root := byte(1)
	for range n {
		for j := range generator {
			generator[j] = qrMultiply(generator[j], root)
			if j+1 < n {
				generator[j] ^= generator[j+1]
			}
		}
		root = qrMultiply(root, 2)
	}
	remainder := make([]byte, n)
	for _, b := range data {
		factor := b ^ remainder[0]
		copy(remainder, remainder[1:])
		remainder[n-1] = 0
		for i, c := range generator {
			remainder[i] ^= qrMultiply(c, factor)
		}
	}
	return remainder
}

func qrMultiply(x, y byte) byte {
	z := 0
	for i := 7; i >= 0; i-- {
		z = z<<1 ^ z>>7*0x11D
		z ^= int(y>>i&1) * int(x)
	}
	return byte(z)
}

// qrFormat is the 15 format bits of level M, whose indicator is 00, with a mask: the mask's 3 bits and their 10 bits of
// BCH code, XORed with 101010000010010 so that they are never all light.
func qrFormat(mask int) int {
	remainder := mask
	for range 10 {
		remainder = remainder<<1 ^ remainder>>9*0x537
	}
	return (mask<<10 | remainder) ^ 0x5412
}
