"""IX for ComfyUI: nodes that ask IX tools (through the pack's own ix-mcp) and draw what they return.
See integrations/comfyui/README.md and docs/adr/0007-comfyui-extension-lives-in-ix.md."""
from .nodes import IXBraidControl, IXKnotControl, IXSpectrogram

NODE_CLASS_MAPPINGS = {
    "IXSpectrogram": IXSpectrogram,
    "IXBraidControl": IXBraidControl,
    "IXKnotControl": IXKnotControl,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IXSpectrogram": "IX Spectrogram",
    "IXBraidControl": "IX Braid Control Image",
    "IXKnotControl": "IX Knot Control Image",
}
