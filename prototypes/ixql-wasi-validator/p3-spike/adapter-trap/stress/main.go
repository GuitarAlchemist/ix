// Allocation stress: grows the heap many times, with WASI calls (clock, env, stdout) in between.
package main

import (
	"fmt"
	"os"
	"time"
)

func main() {
	var keep [][]byte
	sum := 0
	for i := 0; i < 300; i++ {
		b := make([]byte, 256<<10)
		for j := range b {
			b[j] = byte(i + j)
		}
		sum += int(b[len(b)-1])
		if i%3 == 0 {
			keep = append(keep, b)
		}
		_ = time.Now()
		_ = os.Getenv("X")
	}
	fmt.Println("kept", len(keep), "sum", sum)
}
