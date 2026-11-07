"""Give the 3D scene's bot one line to say on the public stream.

  say_public.py --live DIR --en TEXT [--fr TEXT] [--es TEXT]

DIR is the scene's live folder (artpass/foundation/live in the Observatory project). The scene reads
say-public.txt there as one line of JSON {"en", "fr", "es"} and says it in its own language, falling
back to English, then to any language present. Only public-safe lines go here: no paths, no names of
people. Each text is at most 400 characters, the scene's limit. Written atomically (.part, then rename)."""
import argparse, json, os, sys

MAX_CHARS = 400

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--en", required=True)
ap.add_argument("--fr")
ap.add_argument("--es")
a = ap.parse_args()
line = {code: " ".join(text.split()) for code, text in (("en", a.en), ("fr", a.fr), ("es", a.es)) if text}
for code, text in line.items():
    if len(text) > MAX_CHARS:
        sys.exit(f"{code}: {len(text)} characters, the scene says at most {MAX_CHARS}")
path = os.path.join(a.live, "say-public.txt")
with open(path + ".part", "w", encoding="utf-8", newline="") as f:
    f.write(json.dumps(line, ensure_ascii=False))
os.replace(path + ".part", path)
print("wrote", path)
