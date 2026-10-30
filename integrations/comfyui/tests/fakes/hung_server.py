"""Fake MCP server for tests: records its pid, then never answers."""
import os
import sys
import time

with open(sys.argv[1], "w") as f:
    f.write(str(os.getpid()))
time.sleep(120)
