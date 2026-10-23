"""Check the host's rotation, gap-free swap and stop file, end to end.

    python -B check_host.py [host.exe]    # from prototypes/probes; out/bin/host.exe by default

It runs its own host on a throwaway probe under out/check-probes, writing to out/check, so the
probes in probes/ and a host already running on them are left alone.

  S6  probes.jsonl is renamed at -rotate-bytes and only -keep renamed files remain;
  S7  a source edit swaps the build with no gap in the samples longer than 1 s, and the old
      build stops only after the new one has sampled;
  S8  a file named stop in the output directory stops the host and its probes within 5 s.
"""
import glob, json, os, shutil, subprocess, sys, time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
HOST = os.path.abspath(sys.argv[1]) if sys.argv[1:] else os.path.join(ROOT, "out", "bin", "host.exe")
PROBES = os.path.join(ROOT, "out", "check-probes")
OUT = os.path.join(ROOT, "out", "check")
TICK = """package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

const generation = %d

func main() {
	ms, _ := strconv.Atoi(os.Getenv("PROBE_INTERVAL_MS"))
	pad := strings.Repeat("x", 200)
	for {
		fmt.Printf("{\\"metric\\":\\"tick\\",\\"value\\":%%d,\\"pad\\":%%q}\\n", generation, pad)
		time.Sleep(time.Duration(ms) * time.Millisecond)
	}
}
"""


def write_tick(generation):
    os.makedirs(os.path.join(PROBES, "tick"), exist_ok=True)
    with open(os.path.join(PROBES, "tick", "main.go"), "w", encoding="utf-8", newline="\n") as f:
        f.write(TICK % generation)


def records(pattern):
    out = []
    for path in sorted(glob.glob(os.path.join(OUT, pattern))):
        with open(path, encoding="utf-8") as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


def ts(rec):
    return datetime.fromisoformat(rec["ts"].replace("Z", "+00:00")).timestamp()


def wait_for(pred, timeout, what):
    end = time.time() + timeout
    while time.time() < end:
        got = pred()
        if got:
            return got
        time.sleep(0.2)
    sys.exit(f"timed out waiting for {what}")


def run_host(*flags):
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    return subprocess.Popen([HOST, "-probes", os.path.relpath(PROBES, ROOT), "-out", OUT, "-interval", "50ms",
                             "-poll", "200ms", "-for", "120s", *flags], cwd=ROOT)


def stop(host):
    t = time.time()
    open(os.path.join(OUT, "stop"), "w").close()
    host.wait(timeout=10)
    return time.time() - t


def main():
    shutil.rmtree(PROBES, ignore_errors=True)
    write_tick(1)

    # S6 and S8: rotation at 20 kB, keep 2.
    host = run_host("-rotate-bytes", "20000", "-keep", "2")
    wait_for(lambda: len(records("probes.jsonl")) and len(records("host-events*.jsonl")) and
             len(glob.glob(os.path.join(OUT, "probes-*.jsonl"))) >= 2, 90, "two rotations")
    time.sleep(8)  # several more rotations
    rotated = glob.glob(os.path.join(OUT, "probes-*.jsonl"))
    current = os.path.getsize(os.path.join(OUT, "probes.jsonl"))
    s6 = len(rotated) == 2 and current < 20000 + 1024 and all(20000 <= os.path.getsize(p) < 21024 for p in rotated)
    print(f"S6 {len(rotated)} renamed files kept (keep 2), sizes {[os.path.getsize(p) for p in rotated]}, "
          f"current {current} bytes")
    took = stop(host)
    leftovers = subprocess.run(["tasklist", "/FI", "IMAGENAME eq tick-*"], capture_output=True, text=True).stdout
    s8 = took < 5 and "tick-" not in leftovers and not os.path.exists(os.path.join(OUT, "stop"))
    print(f"S8 host stopped {took:.1f}s after the stop file; probes left running: {'tick-' in leftovers}")

    # S7: a swap, rotation off.
    host = run_host("-rotate-bytes", "0")
    wait_for(lambda: any(r["metric"] == "tick" for r in records("probes.jsonl")), 90, "the first build's samples")
    time.sleep(2)
    write_tick(2)
    wait_for(lambda: any(r["value"] == 2 for r in records("probes.jsonl")), 90, "the second build's samples")
    time.sleep(2)
    stop(host)
    ticks = sorted(ts(r) for r in records("probes.jsonl") if r["metric"] == "tick")
    gap = max(b - a for a, b in zip(ticks, ticks[1:]))
    new_first = min(ts(r) for r in records("probes.jsonl") if r["value"] == 2)
    stopped = [e for e in records("host-events.jsonl") if e["event"] == "stopped" and e.get("why") == "source changed"]
    s7 = gap < 1 and len(stopped) == 1 and ts(stopped[0]) >= new_first
    print(f"S7 longest gap {gap * 1000:.0f} ms across the swap; old build stopped "
          f"{(ts(stopped[0]) - new_first) * 1000:.0f} ms after the new one's first sample" if stopped else
          f"S7 longest gap {gap * 1000:.0f} ms; no 'source changed' stop")

    shutil.rmtree(PROBES, ignore_errors=True)
    ok = s6 and s7 and s8
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
