# IX nodes for ComfyUI

[Français](README.fr.md)

ComfyUI custom nodes that ask IX for the maths and only draw what comes back.
Why the pack lives here and talks to IX this way: [ADR-0007](../../docs/adr/0007-comfyui-extension-lives-in-ix.md).

| Node | Category | IX tool | Gives |
| --- | --- | --- | --- |
| `IXKnotControl` | IX/knots | `ix_knot` | a catalogue rope knot as ControlNet lineart + depth, and its summary |
| `IXBraidControl` | IX/knots | `ix_braid` | a braid word (`s1 s2^-1` is a sailor's plait) as lineart + depth that tile vertically |
| `IXSpectrogram` | IX/analysis | `ix_spectrogram` | an STFT summary of a short clip (mono or stereo, 8–48 kHz, at most 5 s) |

The knot summary carries the knot's names (English and French), its family,
its Ashley Book of Knots number, the knot its ends close into (`3_1`, `4_1`),
its crossings, writhe and Jones polynomial, and the rope's clearance in rope
diameters. Every summary carries the SHA-256 of the `ix-mcp` that produced it.

## Install

```sh
cargo build --release -p ix-agent --bin ix-mcp
python integrations/comfyui/install.py --binary target/release/ix-mcp      # ix-mcp.exe on Windows
python integrations/comfyui/install.py --binary target/release/ix-mcp \
    --custom-nodes /path/to/ComfyUI/custom_nodes
```

The first command puts the build in `ix_comfyui/bin/` with its hash beside
it; the second does the same and copies the pack into ComfyUI. It refuses to
replace an `ix_comfyui` folder that is already there — remove it first. After
rebuilding `ix-mcp`, install again: the pack refuses a binary whose bytes no
longer match the recorded hash.

The nodes need numpy and Pillow, which ComfyUI already has.

## What the pack will and will not do

- It runs only `ix_comfyui/bin/ix-mcp`, only while its hash matches, and only
  the tools `ix_spectrogram`, `ix_braid` and `ix_knot`. No node input is a
  path, an executable or a tool name.
- Each node execution starts one `ix-mcp`, with an environment stripped to
  `SYSTEMROOT`, `WINDIR`, `TEMP` and `TMP`, in `ix_comfyui/run/`, and kills it
  after 30 s.
- Inputs are checked before IX starts: image sizes 256–1024 × 256–2048, knot
  ids of `a-z`, `0-9` and `-`, braid words of at most 200 characters, clips of
  at most 5 s.
- The hash catches a binary replaced by accident. It is not a defence against
  someone who can write to the pack.

## Tests

```sh
python -m pip install numpy pillow
python integrations/comfyui/install.py --binary target/debug/ix-mcp
python -m unittest discover -s integrations/comfyui/tests -v
```

CI runs exactly this (the `comfyui-pack` job). The tests of ComfyUI's own
types — `AUDIO` in, `IMAGE` out — need torch and skip without it; run them
with ComfyUI's Python to cover those too.
