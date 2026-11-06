// Command fakeclaude is a test double of `claude -p` in stream-json: no model is called. It
// checks its flags, emits system/init, and answers each user message with an assistant message
// and a result. Behaviour comes from the environment:
//
//	FAKE_TOOLS    comma-separated tools listed in init (default none)
//	FAKE_INIT     "start" (default): init at once; "first": init with the first message
//	FAKE_EFFECTS  a file that gets one line per user message received (the turns)
//	FAKE_HOLD     while this file exists, a message whose text holds "slow" is not answered
//	FAKE_NOACK    answer without the ACK line
//	FAKE_SESSION  report this session_id instead of the --session-id given
//	FAKE_LINGER   once stdin is closed, wait this long (a Go duration) before exiting
//
// A message whose text holds "die" makes it exit 3 without answering.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strings"
	"time"
)

var mandateRE = regexp.MustCompile(`mandate_id: (\S+)`)

func emit(v any) {
	b, _ := json.Marshal(v)
	os.Stdout.Write(append(b, '\n'))
}

func main() {
	args := os.Args[1:]
	for _, bad := range []string{"--bare", "--dangerously-skip-permissions", "--setting-sources", "--permission-mode"} {
		if slices.Contains(args, bad) {
			fmt.Fprintln(os.Stderr, "fakeclaude: forbidden flag", bad)
			os.Exit(2)
		}
	}
	need := [][]string{{"-p"}, {"--input-format", "stream-json"}, {"--output-format", "stream-json"}, {"--verbose"}, {"--tools", ""}, {"--strict-mcp-config"}}
	for _, n := range need {
		i := slices.Index(args, n[0])
		if i < 0 || (len(n) == 2 && (i+1 >= len(args) || args[i+1] != n[1])) {
			fmt.Fprintln(os.Stderr, "fakeclaude: missing", n)
			os.Exit(2)
		}
	}
	session := ""
	if i := slices.Index(args, "--session-id"); i >= 0 && i+1 < len(args) {
		session = args[i+1]
	}
	if s := os.Getenv("FAKE_SESSION"); s != "" {
		session = s
	}
	tools := []string{}
	if t := os.Getenv("FAKE_TOOLS"); t != "" {
		tools = strings.Split(t, ",")
	}
	init := map[string]any{"type": "system", "subtype": "init", "session_id": session, "tools": tools, "mcp_servers": []any{}, "model": "fake"}
	inited := false
	if os.Getenv("FAKE_INIT") != "first" {
		emit(init)
		inited = true
	}
	in := bufio.NewReader(os.Stdin)
	for {
		line, err := in.ReadBytes('\n')
		if err != nil {
			if d, perr := time.ParseDuration(os.Getenv("FAKE_LINGER")); perr == nil {
				time.Sleep(d) // as a real claude running its SessionEnd hooks
			}
			os.Exit(0) // stdin closed: a clean end
		}
		var m struct {
			Type    string `json:"type"`
			Message struct {
				Content []struct {
					Text string `json:"text"`
				} `json:"content"`
			} `json:"message"`
		}
		if json.Unmarshal(line, &m) != nil || m.Type != "user" || len(m.Message.Content) == 0 {
			continue
		}
		text := m.Message.Content[0].Text
		if !inited {
			emit(init)
			inited = true
		}
		if f := os.Getenv("FAKE_EFFECTS"); f != "" {
			fh, _ := os.OpenFile(f, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
			fmt.Fprintln(fh, strings.SplitN(text, "\n", 2)[0])
			fh.Close()
		}
		if strings.Contains(text, "die") {
			os.Exit(3)
		}
		if strings.Contains(text, "slow") {
			for h := os.Getenv("FAKE_HOLD"); h != ""; time.Sleep(20 * time.Millisecond) {
				if _, err := os.Stat(h); err != nil {
					break
				}
			}
		}
		id := "?"
		if g := mandateRE.FindStringSubmatch(text); g != nil {
			id = g[1]
		}
		reply := "ACK " + id + "\nBONJOUR"
		if os.Getenv("FAKE_NOACK") != "" {
			reply = "BONJOUR"
		}
		emit(map[string]any{"type": "assistant", "session_id": session, "message": map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text", "text": reply}}}})
		emit(map[string]any{"type": "result", "subtype": "success", "is_error": false, "result": reply, "session_id": session, "num_turns": 1})
	}
}
