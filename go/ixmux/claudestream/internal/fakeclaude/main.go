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
//	FAKE_ENVNAMES a file that gets the names (never the values) of its environment, one per line
//	FAKE_MUTE     "all": answer no message (an init still comes, per FAKE_INIT); otherwise a
//	              message whose text holds "mute" is consumed and never answered
//	FAKE_FLOOD    a message whose text holds "flood" first gets this many bytes of assistant lines
//	FAKE_REPLY    the answer is padded to this many bytes
//	FAKE_ISERROR  a message whose text holds "fail" gets a result with is_error true
//	FAKE_TOOLS_FIELD "null" or "absent": the init's tools field is null, or missing
//	FAKE_M2       before or instead of the answer to a mandate named m2: "othersession" (a result
//	              of another session first), "duplicate" (the previous result again first),
//	              "prefix" (the answer begins ACK m2-extra), "othermandate" (it begins ACK m1)
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
	"strconv"
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
	if f := os.Getenv("FAKE_ENVNAMES"); f != "" {
		var names []string
		for _, kv := range os.Environ() {
			if k, _, ok := strings.Cut(kv, "="); ok && k != "" {
				names = append(names, k)
			}
		}
		os.WriteFile(f, []byte(strings.Join(names, "\n")+"\n"), 0o600)
	}
	init := map[string]any{"type": "system", "subtype": "init", "session_id": session, "tools": tools, "mcp_servers": []any{}, "model": "fake"}
	switch os.Getenv("FAKE_TOOLS_FIELD") {
	case "null":
		init["tools"] = nil
	case "absent":
		delete(init, "tools")
	}
	var last map[string]any // the last result emitted
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
		if os.Getenv("FAKE_MUTE") == "all" || (os.Getenv("FAKE_MUTE") != "" && strings.Contains(text, "mute")) {
			continue // consumed, never answered
		}
		if n, _ := strconv.Atoi(os.Getenv("FAKE_FLOOD")); n > 0 && strings.Contains(text, "flood") {
			pad := strings.Repeat("f", 4000)
			for sent := 0; sent < n; sent += len(pad) + 100 {
				emit(map[string]any{"type": "assistant", "session_id": session, "message": map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text", "text": pad}}}})
			}
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
		if id == "m2" {
			switch os.Getenv("FAKE_M2") {
			case "othersession":
				emit(map[string]any{"type": "result", "subtype": "success", "is_error": false, "result": "ACK m2\nFROM ELSEWHERE", "session_id": "another-session", "num_turns": 1})
			case "duplicate":
				if last != nil {
					emit(last)
				}
			case "prefix":
				reply = "ACK m2-extra\nBONJOUR"
			case "othermandate":
				reply = "ACK m1\nBONJOUR"
			}
		}
		if n, _ := strconv.Atoi(os.Getenv("FAKE_REPLY")); n > len(reply) {
			reply += strings.Repeat("r", n-len(reply))
		}
		emit(map[string]any{"type": "assistant", "session_id": session, "message": map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text", "text": reply}}}})
		last = map[string]any{"type": "result", "subtype": "success", "is_error": false, "result": reply, "session_id": session, "num_turns": 1}
		if os.Getenv("FAKE_ISERROR") != "" && strings.Contains(text, "fail") {
			last["subtype"], last["is_error"] = "error_during_execution", true
		}
		emit(last)
	}
}
