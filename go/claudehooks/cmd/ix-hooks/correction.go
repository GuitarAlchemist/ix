package main

import (
	"regexp"
	"strings"
)

// correctionDetector is the UserPromptSubmit hook that, when the user's message reads as a
// correction, tells Claude to run /correct in the same turn and to end with a teaching line (was
// correction-detector.sh). French first: the user dictates in French ("non, pas comme ça",
// "plutôt", "on en a déjà parlé"); the English openers stay. Every correction fires.
func correctionDetector(e *env) int {
	p, raw := e.payload()
	prompt := field(p, "prompt")
	if prompt == "" {
		prompt = field(p, "user_prompt")
	}
	if prompt == "" {
		prompt = string(raw) // not JSON, or no prompt member: the raw stdin, as the script did
	}
	if prompt == "" {
		return 0
	}
	if !isCorrection(prompt) {
		return 0
	}
	writeContext(e.stdout, "[correction-nudge] The user's message reads as a correction. In this same turn: run /correct if the correction would apply to future work (typos and one-off nudges are not rules), and end your reply with one teaching line telling the user what you ran, why, and that they can type /correct themselves right after a correction.")
	return 0
}

// An opener at the start of a line of the first 300 bytes, after optional punctuation and a
// filler word (grep is line-based, so `^` was any line's start); or a reminder anywhere.
var (
	correctionOpener = regexp.MustCompile(`(?im)^[[:space:][:punct:]]*((ah|euh|bon|ok|mais|hmm|hum)[[:space:],]+)?(non|no|nope|pas (comme )?(ç|c)a|arr(ê|e)te|attends?|plut(ô|o)t|c'est faux|faux|mauvais|pas du tout|stop|wait|don'?t|dont|actually|incorrect|that'?s wrong|thats wrong)\b`)
	correctionPhrase = regexp.MustCompile(`(?i)on en a d(é|e)j(à|a) parl(é|e)|je (te l'|t.)ai d(é|e)j(à|a) dit|on avait dit|we (already )?discussed this|je te l.avais dit`)
)

func isCorrection(prompt string) bool {
	head := prompt
	if len(head) > 300 {
		head = head[:300]
	}
	head = strings.ToValidUTF8(head, "")
	return correctionOpener.MatchString(head) || correctionPhrase.MatchString(prompt)
}
