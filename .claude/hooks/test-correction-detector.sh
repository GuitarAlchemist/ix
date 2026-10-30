#!/bin/bash
# Behavioral fixture for correction-detector.sh.
# Contract under test: UserPromptSubmit hook reads {"prompt": ...} on stdin and always exits 0.
# A correction, in French or English, prints JSON whose additionalContext tells Claude to run
# /correct in the same turn and to end with a teaching line for the user; any other prompt
# prints nothing. Every correction fires: no throttle, no counter.
# Run: bash .claude/hooks/test-correction-detector.sh

HOOK="$(dirname "$0")/correction-detector.sh"
fails=0

fires() {
    local desc="$1" prompt="$2"
    local out
    out="$(printf '{"prompt": "%s"}' "$prompt" | bash "$HOOK" 2>/dev/null)"
    local got=$?
    if [[ "$got" -eq 0 ]] && printf '%s' "$out" | grep -q '/correct' && printf '%s' "$out" | grep -q 'teaching line'; then
        echo "PASS  fires: $desc"
    else
        echo "FAIL  fires: $desc (exit $got, out: ${out:-<empty>})"
        fails=$((fails + 1))
    fi
}

silent() {
    local desc="$1" prompt="$2"
    local out
    out="$(printf '{"prompt": "%s"}' "$prompt" | bash "$HOOK" 2>/dev/null)"
    local got=$?
    if [[ "$got" -eq 0 && -z "$out" ]]; then
        echo "PASS  silent: $desc"
    else
        echo "FAIL  silent: $desc (exit $got, out: ${out:-<empty>})"
        fails=$((fails + 1))
    fi
}

# French openers, as the user dictates them.
fires "non, pas comme ça"            "Non, pas comme ça, on avait dit PowerShell."
fires "pas ça"                       "pas ça, je veux le worktree"
fires "arrête"                       "Arrête de faire ça à chaque fois."
fires "arrete (no accent)"           "arrete, on relit le fichier d'abord"
fires "plutôt"                       "Plutôt la commande slash rétro, pas celle-là."
fires "attends"                      "Attends, ce n'est pas ce que je t'ai demandé."
fires "c'est faux"                   "C'est faux, le port est 5176."
fires "on en a déjà parlé (mid)"     "Le fichier est bon mais on en a déjà parlé : jamais de stash."
fires "je t'ai déjà dit (mid)"       "Bon, je t'ai déjà dit de ne pas toucher au main tree."
fires "on avait dit (mid)"           "Pour le commit, on avait dit des chemins explicites."

# English openers kept from the first version.
fires "no, don't"                    "No, don't amend after CI passed."
fires "stop"                         "Stop rewriting the README."
fires "actually"                     "Actually we discussed this before."

# Every correction fires: the second one in a row too (the old hook spoke once, then 1 in 5).
fires "second correction in a row"   "Non, encore raté."
fires "third correction in a row"    "Non, toujours pas."

# Ordinary prompts stay silent.
silent "a request"                   "Peux-tu lancer les tests sur Windows ?"
silent "a status question"           "On en est où du développement ?"
silent "a word that contains non"    "Le nonce est bon, continue."
silent "no as part of a word"        "Nothing to change, go on."
silent "empty prompt"                ""

echo
if [[ "$fails" -eq 0 ]]; then
    echo "ALL PASS"
else
    echo "$fails FAILED"
    exit 1
fi
