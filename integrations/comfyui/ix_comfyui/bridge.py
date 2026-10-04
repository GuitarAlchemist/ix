"""Bounded stdio JSON-RPC bridge from ComfyUI to IX tools.

The executable is the pack's own `bin/ix-mcp` (`ix-mcp.exe` on Windows), put there by `install.py` with
its SHA-256 beside it; neither is a node input. Each call checks the hash, starts the process with a
minimal environment (no API keys) in the pack's run directory, sends initialize + tools/call over
stdin, and kills it if it has not answered in time. Only the tools in TOOLS can be called.

The hash guards against a binary swapped by accident (a rebuild without the tools a node needs); it is
not a defence against someone who can write to the pack, who could edit this file as well.
"""
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

PACK = Path(__file__).resolve().parent
BIN_DIR = PACK / "bin"
BINARY = BIN_DIR / ("ix-mcp.exe" if os.name == "nt" else "ix-mcp")
HASH_FILE = BIN_DIR / "ix-mcp.sha256"
RUN_DIR = PACK / "run"
SPECTROGRAM_TOOL = "ix_spectrogram"
BRAID_TOOL = "ix_braid"
KNOT_TOOL = "ix_knot"
TOOLS = frozenset({SPECTROGRAM_TOOL, BRAID_TOOL, KNOT_TOOL})

TIMEOUT_S = 30.0
ENV_KEYS = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP")

MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 48_000
MAX_SECONDS = 5.0
MAX_SAMPLES = 240_000  # 5 s at 48 kHz: about 2.6 MB of JSON in, about 5 MB of magnitudes out
WINDOW_SIZES = (256, 512, 1024, 2048)  # ix_spectrogram needs a power of two

BRAID_WORD_CHARS = frozenset("s0123456789^-, σ")
MAX_WORD_CHARS = 200
MAX_REPEAT = 32
BRAID_SAMPLES = 32  # points per crossing in each strand path

KNOT_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")
MAX_KNOT_ID = 64
MAX_ROPES = 8
MAX_KNOT_POINTS = 20_000


class IxBridgeError(RuntimeError):
    pass


def installed_hash(hash_file=HASH_FILE):
    """The SHA-256 install.py recorded, or IxBridgeError when there is none."""
    try:
        recorded = Path(hash_file).read_text(encoding="ascii").split()
    except OSError as e:
        raise IxBridgeError(f"ix-mcp is not installed in the pack ({e}); run integrations/comfyui/install.py") from e
    if not recorded or len(recorded[0]) != 64:
        raise IxBridgeError(f"{hash_file} does not hold a SHA-256")
    return recorded[0].lower()


def verify_binary(path=BINARY, sha256=None):
    """The pack's executable, or IxBridgeError when it is missing or its bytes changed."""
    sha256 = sha256 or installed_hash()
    try:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as e:
        raise IxBridgeError(f"ix-mcp binary unreadable: {e}") from e
    if digest != sha256:
        raise IxBridgeError(f"ix-mcp binary hash {digest} does not match the installed {sha256}")
    return Path(path)


def _minimal_env():
    return {k: os.environ[k] for k in ENV_KEYS if k in os.environ}


def call_tool(tool, arguments, timeout=TIMEOUT_S, _argv=None):
    """tools/call of `tool` (one of TOOLS) with `arguments`; returns the tool's JSON result.

    `_argv` replaces the pack's binary in tests only (a fake server or a hung process).
    """
    if tool not in TOOLS:
        raise IxBridgeError(f"tool {tool!r} is not one the pack calls")
    argv = _argv or [str(verify_binary())]
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "ix-comfyui", "version": "0.1.0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool, "arguments": arguments}},
    ]
    payload = "".join(json.dumps(m, allow_nan=False) + "\n" for m in messages).encode()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=RUN_DIR, env=_minimal_env(),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        stdout, stderr = proc.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise IxBridgeError(f"ix-mcp did not answer within {timeout:g} s; killed pid {proc.pid}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    for line in stdout.decode("utf-8", errors="replace").splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("id") != 2:
            continue
        if "error" in message:
            raise IxBridgeError(f"ix-mcp error: {message['error'].get('message')}")
        result = message.get("result") or {}
        text = "".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        if result.get("isError"):
            raise IxBridgeError(f"{tool} failed: {text[:300]}")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise IxBridgeError(f"{tool} returned non-JSON text: {text[:120]!r}") from e
    raise IxBridgeError(f"ix-mcp exited {proc.returncode} without answering tools/call; "
                        f"stderr: {stderr.decode('utf-8', errors='replace')[-300:]}")


def check_signal(n_samples, sample_rate, window_size):
    """Raise ValueError unless the signal fits the bounds. Cheap: call it before converting audio."""
    if not isinstance(sample_rate, int) or not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
        raise ValueError(f"sample rate {sample_rate!r} outside {MIN_SAMPLE_RATE}..{MAX_SAMPLE_RATE} Hz")
    if window_size not in WINDOW_SIZES:
        raise ValueError(f"window_size {window_size!r} not one of {WINDOW_SIZES}")
    if n_samples < window_size:
        raise ValueError(f"{n_samples} samples is shorter than the {window_size}-sample window")
    if n_samples > MAX_SAMPLES or n_samples / sample_rate > MAX_SECONDS:
        raise ValueError(f"{n_samples} samples at {sample_rate} Hz exceeds {MAX_SECONDS:g} s / {MAX_SAMPLES} samples")


def spectrogram_matrix(samples, sample_rate, window_size, db=False):
    """Mono `samples` through ix_spectrogram; returns the checked [frames][bins] magnitude matrix
    (window/2 hop, so frames overlap by half)."""
    check_signal(len(samples), sample_rate, window_size)
    signal = []
    for x in samples:
        x = float(x)
        if not math.isfinite(x):
            raise ValueError("signal contains NaN or infinity")
        signal.append(round(x, 6))
    hop = window_size // 2
    out = call_tool(SPECTROGRAM_TOOL, {"signal": signal, "window_size": window_size, "hop_size": hop, "db": bool(db)})

    matrix = out.get("spectrogram")
    n_bins = window_size // 2 + 1
    if (not isinstance(matrix, list) or not matrix or out.get("n_frames") != len(matrix)
            or out.get("n_bins") != n_bins or out.get("hop_size") != hop
            or any(len(row) != n_bins for row in matrix)):
        raise IxBridgeError(f"unexpected {SPECTROGRAM_TOOL} shape: n_frames={out.get('n_frames')} n_bins={out.get('n_bins')}")
    return matrix


def spectrogram(samples, sample_rate, window_size, db=False):
    """Mono `samples` through ix_spectrogram; returns a compact summary, not the whole matrix."""
    matrix = spectrogram_matrix(samples, sample_rate, window_size, db)
    n_bins = window_size // 2 + 1
    bin_hz = sample_rate / window_size
    mean = [sum(row[k] for row in matrix) / len(matrix) for k in range(n_bins)]
    peak_bin = max(range(n_bins), key=mean.__getitem__)
    return {
        "tool": SPECTROGRAM_TOOL,
        "sample_rate": sample_rate,
        "n_samples": len(samples),
        "window_size": window_size,
        "hop_size": window_size // 2,
        "db": bool(db),
        "n_frames": len(matrix),
        "n_bins": n_bins,
        "bin_hz": bin_hz,
        "peak_hz": peak_bin * bin_hz,
        "frame_peak_hz": [max(range(n_bins), key=row.__getitem__) * bin_hz for row in matrix],
    }


def braid_layout(word, repeat):
    """ix_braid on `word` written `repeat` times, with its strand layout; returns the checked result."""
    if (not isinstance(word, str) or not word.strip() or len(word) > MAX_WORD_CHARS
            or set(word) - BRAID_WORD_CHARS):
        raise ValueError(f"braid word must be 1..{MAX_WORD_CHARS} characters of s, digits, ^, -, comma, space")
    if not isinstance(repeat, int) or not 1 <= repeat <= MAX_REPEAT:
        raise ValueError(f"repeat {repeat!r} outside 1..{MAX_REPEAT}")
    out = call_tool(BRAID_TOOL, {"word": word, "repeat": repeat, "geometry": True,
                                 "samples_per_crossing": BRAID_SAMPLES})

    n, crossings = out.get("strands"), out.get("crossings")
    strands = (out.get("geometry") or {}).get("strands")
    shaped = (isinstance(n, int) and isinstance(crossings, int) and crossings >= 1
              and isinstance(strands, list) and len(strands) == n
              and sorted(s.get("start") for s in strands) == list(range(n))
              and sorted(out.get("permutation") or []) == list(range(n))
              and all(len(s.get("points") or []) == crossings * BRAID_SAMPLES + 1 for s in strands))
    if not shaped:
        raise IxBridgeError(f"unexpected {BRAID_TOOL} shape: strands={n} crossings={crossings}")
    for s in strands:
        _check_points(s["points"], BRAID_TOOL)
    return out


def knot_catalog():
    """The catalogue ix_knot carries: a list of {id, en, fr, family, abok, closure}."""
    out = call_tool(KNOT_TOOL, {"list": True})
    entries = out.get("entries")
    if not isinstance(entries, list) or not all(isinstance(e, dict) and _is_knot_id(e.get("id")) for e in entries):
        raise IxBridgeError(f"unexpected {KNOT_TOOL} catalogue shape")
    return entries


def knot_layout(knot_id):
    """ix_knot on a catalogue knot, with each rope's 3D path; returns the checked result."""
    if not _is_knot_id(knot_id):
        raise ValueError(f"knot id must be 1..{MAX_KNOT_ID} characters of a-z, 0-9 and -")
    out = call_tool(KNOT_TOOL, {"name": knot_id, "geometry": True})
    geometry = out.get("geometry") or {}
    ropes, radius = geometry.get("ropes"), geometry.get("radius")
    shaped = (out.get("id") == knot_id and isinstance(radius, (int, float)) and math.isfinite(radius)
              and radius > 0 and isinstance(ropes, list) and 1 <= len(ropes) <= MAX_ROPES
              and out.get("components") == len(ropes)
              and all(isinstance(r, dict) and isinstance(r.get("closed"), bool)
                      and isinstance(r.get("points"), list) and len(r["points"]) >= 2 for r in ropes)
              and sum(len(r["points"]) for r in ropes) <= MAX_KNOT_POINTS)
    if not shaped:
        raise IxBridgeError(f"unexpected {KNOT_TOOL} shape for {knot_id!r}")
    for r in ropes:
        _check_points(r["points"], KNOT_TOOL)
    return out


def _is_knot_id(value):
    return isinstance(value, str) and 1 <= len(value) <= MAX_KNOT_ID and not set(value) - KNOT_ID_CHARS


def _check_points(points, tool):
    for p in points:
        if (not isinstance(p, list) or len(p) != 3
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)):
            raise IxBridgeError(f"{tool} returned a bad point {p!r}")
