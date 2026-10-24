"""Bridge tests: the installed ix-mcp on synthetic tones, plus bounds, the hash check, the environment
and the timeout. Run `install.py --binary <ix-mcp build>` first."""
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ix_comfyui import bridge  # noqa: E402
from ix_comfyui.bridge import IxBridgeError, call_tool, check_signal, spectrogram, verify_binary  # noqa: E402

# Python sets LC_CTYPE itself when it starts under the C locale (PEP 538), as it does with the empty
# environment the bridge hands a child on Linux and macOS; on macOS, CoreFoundation also sets
# __CF_USER_TEXT_ENCODING in every process it starts in. Neither is inherited from the caller.
PYTHON_OWN_KEYS = {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"}


def tone(freqs_and_seconds, sample_rate):
    out = []
    for freq, seconds in freqs_and_seconds:
        n = int(seconds * sample_rate)
        out += [0.5 * math.sin(2 * math.pi * freq * i / sample_rate) for i in range(n)]
    return out


def pid_alive(pid):
    if os.name == "nt":
        listing = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in listing
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class InstalledIxMcp(unittest.TestCase):
    def test_installed_binary_hash_matches(self):
        self.assertEqual(verify_binary(), bridge.BINARY)

    def test_single_tone_peaks_at_its_bin(self):
        summary = spectrogram(tone([(440.0, 1.0)], 16_000), 16_000, 1024)
        self.assertEqual(summary["n_bins"], 513)
        self.assertEqual(summary["bin_hz"], 15.625)
        self.assertLessEqual(abs(summary["peak_hz"] - 440.0), summary["bin_hz"])

    def test_frames_follow_a_pitch_change(self):
        summary = spectrogram(tone([(440.0, 0.5), (1000.0, 0.5)], 16_000), 16_000, 512)
        frames = summary["frame_peak_hz"]
        bin_hz = summary["bin_hz"]
        self.assertLessEqual(abs(frames[2] - 440.0), bin_hz)
        self.assertLessEqual(abs(frames[-3] - 1000.0), bin_hz)
        self.assertEqual(summary["hop_size"], 256)

    def test_largest_allowed_clip_answers_in_time(self):
        started = time.monotonic()
        summary = spectrogram(tone([(1000.0, 5.0)], 48_000), 48_000, 2048)
        self.assertLess(time.monotonic() - started, bridge.TIMEOUT_S)
        self.assertLessEqual(abs(summary["peak_hz"] - 1000.0), summary["bin_hz"])


class Bounds(unittest.TestCase):
    def test_rejects_out_of_range_inputs(self):
        cases = [
            (16_000, 7_999, 1024),            # sample rate too low
            (16_000, 48_001, 1024),           # sample rate too high
            (16_000, 16_000, 1000),           # not an allowed window
            (512, 16_000, 1024),              # shorter than the window
            (240_001, 48_000, 1024),          # over the sample cap
            (16_000 * 5 + 1, 16_000, 1024),   # over 5 s
        ]
        for n, sr, w in cases:
            with self.subTest(n=n, sr=sr, w=w), self.assertRaises(ValueError):
                check_signal(n, sr, w)

    def test_rejects_nan_before_starting_ix(self):
        samples = tone([(440.0, 0.1)], 16_000)
        samples[10] = float("nan")
        with self.assertRaises(ValueError):
            spectrogram(samples, 16_000, 256)


class Process(unittest.TestCase):
    def test_changed_binary_is_refused(self):
        with self.assertRaises(IxBridgeError):
            verify_binary(sha256="0" * 64)

    def test_a_missing_install_says_how_to_install(self):
        with self.assertRaises(IxBridgeError) as caught:
            bridge.installed_hash(HERE / "no-such-file.sha256")
        self.assertIn("install.py", str(caught.exception))

    def test_only_the_packs_tools_can_be_called(self):
        with self.assertRaises(IxBridgeError):
            call_tool("ix_sentrux_annotate", {})

    def test_environment_carries_no_secrets(self):
        os.environ["IX_COMFYUI_FAKE_SECRET"] = "x"
        try:
            out = call_tool(bridge.KNOT_TOOL, {}, _argv=[sys.executable, str(HERE / "fakes" / "env_server.py")])
        finally:
            del os.environ["IX_COMFYUI_FAKE_SECRET"]
        self.assertTrue(set(out["env_keys"]) - PYTHON_OWN_KEYS <= set(bridge.ENV_KEYS), out["env_keys"])

    def test_hung_server_is_killed_at_the_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "pid"
            started = time.monotonic()
            with self.assertRaises(IxBridgeError):
                call_tool(bridge.KNOT_TOOL, {}, timeout=2,
                          _argv=[sys.executable, str(HERE / "fakes" / "hung_server.py"), str(pid_file)])
            self.assertLess(time.monotonic() - started, 10)
            pid = int(pid_file.read_text())
            self.assertFalse(pid_alive(pid), f"pid {pid} still running")


if __name__ == "__main__":
    unittest.main(verbosity=2)
