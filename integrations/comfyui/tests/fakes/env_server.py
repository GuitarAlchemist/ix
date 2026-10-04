"""Fake MCP server for tests: answers tools/call with the names of the environment variables it got."""
import json
import os
import sys

for line in sys.stdin:
    message = json.loads(line)
    if message.get("id") == 2:
        text = json.dumps({"env_keys": sorted(os.environ)})
        print(json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": text}]}}),
              flush=True)
