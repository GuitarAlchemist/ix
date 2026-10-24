"""Put an ix-mcp build into the ComfyUI pack with its SHA-256 beside it, and optionally copy the pack
into a ComfyUI install.

    cargo build --release -p ix-agent --bin ix-mcp
    python integrations/comfyui/install.py --binary target/release/ix-mcp.exe
    python integrations/comfyui/install.py --binary target/release/ix-mcp.exe \
        --custom-nodes C:/ComfyUI/custom_nodes

The pack runs only the binary in its own bin/ folder, and only while its bytes match the recorded hash.
Copying into custom_nodes refuses to replace a folder that is already there: remove it first.
"""
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE / "ix_comfyui"
NAME = "ix-mcp.exe" if os.name == "nt" else "ix-mcp"


def install(binary, custom_nodes=None):
    binary = Path(binary)
    if not binary.is_file():
        raise SystemExit(f"no ix-mcp build at {binary}; cargo build -p ix-agent --bin ix-mcp first")
    bin_dir = PACK / "bin"
    bin_dir.mkdir(exist_ok=True)
    target = bin_dir / NAME
    shutil.copy2(binary, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (bin_dir / "ix-mcp.sha256").write_text(f"{digest}  {NAME}\n", encoding="ascii")
    print(f"installed {target} sha256 {digest}")
    if custom_nodes:
        dest = Path(custom_nodes) / "ix_comfyui"
        if dest.exists():
            raise SystemExit(f"{dest} already exists; remove it, then run this again")
        shutil.copytree(PACK, dest, ignore=shutil.ignore_patterns("__pycache__", "run"))
        print(f"copied the pack to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, help="the ix-mcp build to install")
    parser.add_argument("--custom-nodes", help="a ComfyUI custom_nodes folder to copy the pack into")
    args = parser.parse_args()
    install(args.binary, args.custom_nodes)
    sys.exit(0)
