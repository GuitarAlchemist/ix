"""Create a probe from a template, never from free code.

    python -B new_probe.py process <name> <wmi-name-pattern>
    python -B new_probe.py process scene "godot.windows.editor%"

It writes probes/proc-<name>/main.go from templates/<template>.go.tmpl; the running host
notices the new directory, builds it and starts it. The template list is fixed, and every
parameter must match a strict pattern before it reaches the source, so a request from the
scene, a voice command or an agent can choose a probe but cannot inject code.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = {
    # template: (parameters, each with its pattern)
    "process": (("name", r"[a-z][a-z0-9]{0,23}"), ("match", r"[A-Za-z0-9._#%-]{1,64}")),
}


def main(argv):
    if not argv or argv[0] not in TEMPLATES:
        sys.exit(f"templates: {', '.join(TEMPLATES)}\n{__doc__}")
    template, args = argv[0], argv[1:]
    spec = TEMPLATES[template]
    if len(args) != len(spec):
        sys.exit(f"{template} takes {len(spec)} parameters: {', '.join(p for p, _ in spec)}")
    params = {}
    for (key, pattern), value in zip(spec, args):
        if not re.fullmatch(pattern, value):
            sys.exit(f"{key} {value!r} does not match {pattern}")
        params[key] = value
    with open(os.path.join(HERE, "templates", template + ".go.tmpl"), encoding="utf-8") as f:
        source = f.read()
    probe = f"proc-{params['name']}"
    # json.dumps of these ASCII values is a valid Go string literal.
    for key, value in {"probe": probe, "match": json.dumps(params["match"]),
                       "match_comment": params["match"], "metric": params["name"],
                       "metric_literal": json.dumps(params["name"])}.items():
        source = source.replace("{{" + key + "}}", value)
    out = os.path.join(HERE, "probes", probe)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "main.go"), "w", encoding="utf-8", newline="\n") as f:
        f.write(source)
    print(os.path.join(out, "main.go"))


if __name__ == "__main__":
    main(sys.argv[1:])
