"""The torch side of the nodes: ComfyUI's AUDIO in and IMAGE out. These need torch, which ComfyUI
ships and CI does not install; they are skipped, by name, where torch is absent."""
import json
import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

try:
    import torch
except ImportError:  # CI: the pack's torch-free tests cover the rest
    torch = None

from ix_comfyui.nodes import IXKnotControl, IXSpectrogram, audio_to_mono  # noqa: E402


def audio(freq, seconds, sample_rate, channels=1, batch=1):
    t = torch.arange(int(seconds * sample_rate)) / sample_rate
    wave = 0.5 * torch.sin(2 * math.pi * freq * t)
    return {"waveform": wave.repeat(batch, channels, 1), "sample_rate": sample_rate}


@unittest.skipIf(torch is None, "torch is not installed: run these in ComfyUI's Python")
class ComfyTypes(unittest.TestCase):
    def test_stereo_is_downmixed(self):
        clip = audio(440.0, 0.1, 16_000, channels=2)
        clip["waveform"][0, 1] = 0.0
        mono, sr = audio_to_mono(clip, 256)
        self.assertEqual(sr, 16_000)
        self.assertAlmostEqual(mono[100], float(clip["waveform"][0, 0, 100]) / 2, places=6)

    def test_rejects_batches_multichannel_and_long_clips_before_converting(self):
        for clip in (audio(440.0, 0.1, 16_000, batch=2),
                     audio(440.0, 0.1, 16_000, channels=3),
                     audio(440.0, 6.0, 16_000)):
            with self.subTest(shape=tuple(clip["waveform"].shape)), self.assertRaises(ValueError):
                audio_to_mono(clip, 1024)

    def test_spectrogram_node_returns_ui_text_and_the_same_string(self):
        out = IXSpectrogram().analyze(audio(440.0, 1.0, 22_050, channels=2), "1024", False)
        text = out["result"][0]
        self.assertEqual(out["ui"]["text"], [text])
        summary = json.loads(text)
        self.assertLessEqual(abs(summary["peak_hz"] - 440.0), summary["bin_hz"])

    def test_knot_node_returns_comfy_images(self):
        out = IXKnotControl().draw("figure-eight", 512, 768)
        line, deep, text = out["result"]
        self.assertEqual(tuple(line.shape), (1, 768, 512, 3))
        self.assertEqual(tuple(deep.shape), (1, 768, 512, 3))
        self.assertTrue(0.0 <= float(line.min()) and float(line.max()) <= 1.0)
        self.assertEqual(json.loads(text)["fr"], "Nœud en huit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
