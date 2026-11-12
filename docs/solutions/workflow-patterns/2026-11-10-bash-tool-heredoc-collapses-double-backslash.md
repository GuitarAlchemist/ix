---
title: "A quoted heredoc sent through the agent's Bash tool on Windows lost a backslash: `C:\\w` arrived as `C:\w`"
category: workflow-patterns
date: 2026-11-10
tags: [bash-tool, heredoc, windows, git-bash, backslash, test-fixture, go]
symptom: "a Go test appended with `cat >> x_test.go <<'EOF'` held `\"cwd\":\"C:\\\\w\"` in the command, but the file got `C:\\w` (one backslash): the JSON fixture was invalid, the record unreadable, and the test red for the wrong reason"
root_cause: "the text of the command, not bash, lost one level of backslashes on the way from the tool call to Git Bash; a quoted heredoc (<<'EOF') does no escaping of its own, so what bash received was already collapsed"
---

# A quoted heredoc lost a backslash on its way through the Bash tool

## Problem

A test fixture written by an appending heredoc:

```bash
cat >> cmd/ixmux/inventory_test.go <<'EOF'
os.WriteFile(..., []byte(`{"pid":1,"sessionId":"s1","cwd":"C:\\w","startedAt":1}`), 0o600)
EOF
```

landed in the file as `"cwd":"C:\w"`. In a Go raw string that is the JSON
`"C:\w"`, an invalid escape, so the session record was `unreadable`. The test
failed, and its message ("partial false: want the search cut") pointed at the
code under test, not at the fixture.

`od -c` on a fixture written the same way showed one `\` where the command
had two.

## Fix

Write any text that holds a backslash with the Edit or Write tool, not with
a heredoc through Bash. Then check the bytes (`od -c`, or `grep -n` on the
line) before you believe a red.

## Rule

A red test whose fixture went through a shell is first a question about the
fixture. Read the fixture's bytes before you debug the code.

Source: `go/ixmux/cmd/ixmux/inventory_test.go`
(`TestInventoryTranscriptBudgetFlags`), commit `a6cc36e1`.
