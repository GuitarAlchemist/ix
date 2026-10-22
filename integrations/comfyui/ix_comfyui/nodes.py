"""ComfyUI nodes over IX tools: an audio spectrogram summary, and braid and rope knots drawn as
ControlNet lineart and depth.

Each node checks its inputs, asks IX through the bridge, and only rasterizes what IX returns: the
geometry and the invariants are IX's.
"""
import json

from . import bridge
from .bridge import MAX_REPEAT, WINDOW_SIZES, IxBridgeError, braid_layout, check_signal, knot_layout, spectrogram

MIN_SIDE, MAX_WIDTH, MAX_HEIGHT = 256, 1024, 2048


def audio_to_mono(audio, window_size):
    """ComfyUI AUDIO ({"waveform": [batch, channels, samples] tensor, "sample_rate": int}) to a mono list.

    Shape and bounds are checked before any conversion, so an oversized clip is never copied.
    """
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if len(waveform.shape) != 3:
        raise ValueError(f"AUDIO waveform must be [batch, channels, samples], got shape {tuple(waveform.shape)}")
    batch, channels, n_samples = (int(d) for d in waveform.shape)
    if batch != 1:
        raise ValueError(f"one clip at a time: batch is {batch}")
    if channels not in (1, 2):
        raise ValueError(f"mono or stereo only: {channels} channels")
    check_signal(n_samples, sample_rate, window_size)
    return waveform[0].float().mean(dim=0).cpu().tolist(), sample_rate


def to_image(gray):
    """A (height, width) uint8 array as a ComfyUI IMAGE: float [1, height, width, 3] in 0..1."""
    import numpy as np
    import torch
    return torch.from_numpy(np.repeat(gray[None, :, :, None], 3, axis=3).astype(np.float32) / 255.0)


def check_size(width, height):
    width, height = int(width), int(height)
    if not (MIN_SIDE <= width <= MAX_WIDTH and MIN_SIDE <= height <= MAX_HEIGHT):
        raise ValueError(f"image size {width}x{height} outside {MIN_SIDE}..{MAX_WIDTH} x {MIN_SIDE}..{MAX_HEIGHT}")
    return width, height


def _knot_ids():
    """The catalogue's ids for the node's menu; a single explanatory entry when IX cannot be asked."""
    try:
        return [e["id"] for e in bridge.knot_catalog()]
    except IxBridgeError as e:
        return [f"(unavailable: {e})"[:120]]


class IXSpectrogram:
    """Runs IX's pure ix_spectrogram on one short clip through the pack's ix-mcp."""

    CATEGORY = "IX/analysis"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary_json",)
    FUNCTION = "analyze"
    OUTPUT_NODE = True
    DESCRIPTION = "STFT magnitude summary from IX ix_spectrogram. Mono or stereo (downmixed), 8-48 kHz, at most 5 s."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "window_size": ([str(w) for w in WINDOW_SIZES], {"default": "1024"}),
            "db": ("BOOLEAN", {"default": False}),
        }}

    def analyze(self, audio, window_size, db):
        window = int(window_size)
        samples, sample_rate = audio_to_mono(audio, window)
        summary = spectrogram(samples, sample_rate, window, db)
        summary["ix_mcp_sha256"] = bridge.installed_hash()
        text = json.dumps(summary)
        return {"ui": {"text": [text]}, "result": (text,)}


class IXBraidControl:
    """A braid word through IX's ix_braid, drawn as tube lineart and depth for ControlNet.

    IX checks and lays out the braid (strand paths, closure components, Jones polynomial); this node
    only rasterizes what IX returns. The images repeat vertically.
    """

    CATEGORY = "IX/knots"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("lineart", "depth", "summary_json")
    FUNCTION = "draw"
    DESCRIPTION = ("Braid word (s1 s2^-1: a sailor's plait) through IX ix_braid, drawn as ControlNet lineart "
                   "and depth that tile vertically.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "word": ("STRING", {"default": "s1 s2^-1", "multiline": False}),
            "repeat": ("INT", {"default": 6, "min": 1, "max": MAX_REPEAT}),
            "width": ("INT", {"default": 640, "min": MIN_SIDE, "max": MAX_WIDTH, "step": 64}),
            "height": ("INT", {"default": 1536, "min": MIN_SIDE, "max": MAX_HEIGHT, "step": 64}),
        }}

    def images(self, word, repeat, width, height):
        """(lineart, depth, summary) as uint8 arrays and a dict, before any torch conversion."""
        from .braid_render import control_images
        width, height = check_size(width, height)
        out = braid_layout(word, int(repeat))
        line, deep, _ = control_images(out, width, height)
        summary = {k: out[k] for k in ("word", "strands", "crossings", "writhe", "components",
                                       "permutation", "jones", "jones_symmetric")}
        summary.update(tool=bridge.BRAID_TOOL, ix_mcp_sha256=bridge.installed_hash(), width=width, height=height)
        return line, deep, summary

    def draw(self, word, repeat, width, height):
        line, deep, summary = self.images(word, repeat, width, height)
        text = json.dumps(summary)
        return {"ui": {"text": [text]}, "result": (to_image(line), to_image(deep), text)}


class IXKnotControl:
    """A knot from IX's catalogue through ix_knot, drawn as tube lineart and depth for ControlNet.

    IX holds the knot's drawing, checks it against the knot its closure must be, and lifts each rope
    where it passes in front; this node only rasterizes what IX returns.
    """

    CATEGORY = "IX/knots"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("lineart", "depth", "summary_json")
    FUNCTION = "draw"
    DESCRIPTION = "A rope knot from IX's catalogue (ix_knot), drawn as ControlNet lineart and depth."

    @classmethod
    def INPUT_TYPES(cls):
        ids = _knot_ids()
        default = "figure-eight" if "figure-eight" in ids else ids[0]
        return {"required": {
            "knot": (ids, {"default": default}),
            "width": ("INT", {"default": 768, "min": MIN_SIDE, "max": MAX_WIDTH, "step": 64}),
            "height": ("INT", {"default": 1024, "min": MIN_SIDE, "max": MAX_HEIGHT, "step": 64}),
        }}

    def images(self, knot, width, height):
        """(lineart, depth, summary) as uint8 arrays and a dict, before any torch conversion."""
        from .rope_render import control_images
        width, height = check_size(width, height)
        out = knot_layout(knot)
        line, deep, _ = control_images(out, width, height)
        summary = {k: out.get(k) for k in ("id", "en", "fr", "family", "abok", "closure", "crossings",
                                           "writhe", "components", "jones", "jones_symmetric")}
        summary.update(min_clearance=out["geometry"].get("min_clearance"), tool=bridge.KNOT_TOOL,
                       ix_mcp_sha256=bridge.installed_hash(), width=width, height=height)
        return line, deep, summary

    def draw(self, knot, width, height):
        line, deep, summary = self.images(knot, width, height)
        text = json.dumps(summary, ensure_ascii=False)
        return {"ui": {"text": [text]}, "result": (to_image(line), to_image(deep), text)}
