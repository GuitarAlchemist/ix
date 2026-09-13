Some tests in a repository pass without actually testing anything. A common
shape: the test needs an external command-line tool, and when that tool is not
installed the test prints a note and returns early instead of failing. It is
reported as passing, but it asserted nothing.

Search this repository for tests of that shape.

For each one you find, report:

- the file path,
- the name of the test function,
- the exact line or condition that causes it to return without asserting,
- which external tool it needs.

Answer concisely, as a list. Do not modify any file. If you find none, say so
explicitly rather than guessing.
