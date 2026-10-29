# Agent DSLs: an `.ebnf` grammar, parsed by a general Earley parser, walked by Starlark

Status: proposed (2026-10-29)

Builds on ADR-0016 (durable runs in Starlark). ADR-0018 says how a DSL is versioned and shipped.

## Context

The operator wants agents to speak small languages made for their task, instead of JSON or free
text (2026-10-29). Each agent would get its own DSL: a plan in three lines, a fold sequence, a
knot sheet. Such a DSL is shorter than JSON, and a grammar refuses every deviation at a given
position. A model writes it well from the grammar and two examples. It also suits durable runs:
the text goes into the journal, parsing is deterministic, and replay rebuilds the same tree.

IX already has part of this:

- **`ix-grammar`** reads ISO 14977 EBNF (a subset, `src/ebnf.rs`) and ABNF (`src/abnf.rs`) into
  one `EbnfGrammar`. It generates sentences that follow it (constrained MCTS,
  `src/constrained.rs`). It cannot check a text against a grammar: it has no recognizer.
- **The grammar towers** (one grammar per craft, at several levels, each lowered into the next
  deterministically) and the grammar evolution of `ix-grammar` and TARS.
- **ADR-0010 (origami):** files that IX checks and TARS drafts, under the constant
  `ix_origami::GRAMMAR`.
- **TARS** stores EBNF grammars and constrains a local model's output to them, through llama.cpp
  or vLLM.

The origami grammar shows why a grammar kept for documentation is not enough
(`crates/ix-origami/src/sequence.rs:51`):

- it defines `line` twice, once as a line of the file and once as a geometric line;
- it leaves `letter`, `digit`, `digits`, `integer`, `newline` and `any character` undefined.

A general parser would refuse it. It works today only because a handwritten parser does the
real work, and the only test checks that keywords appear in the grammar's text
(`sequence.rs:1056`).

The available parsers:

- **Maintained Rust parser libraries** are PEG (`pest`, which can interpret a grammar at run time
  through `pest_vm`) or combinators (`chumsky`). Neither reads a context-free EBNF grammar as it
  is: PEG's ordered choice and greedy repetition silently change what an EBNF grammar means.
- **LR generators** (`lalrpop`) and tree-sitter generate code at build time.
- **Earley** accepts any context-free grammar at run time, and reports ambiguity and the expected
  terminals at the first error. The Rust Earley crates are small and quiet: `earlgrey` 0.4.1 was
  last released in 2024-10, `santiago` 1.3.1 in 2022-05 (crates.io, 2026-10-28).

## Decision

**1. A DSL is a grammar plus a Starlark walker, and no Rust per DSL.** A DSL is an `.ebnf` file,
in the subset `ix-grammar::ebnf` reads. A durable script calls the capability
`parse(dsl, text)`. It returns the tree on the success rail, and the script walks it and calls
other capabilities.

**2. A grammar must be executable, or it is refused.** The checks need a reader that keeps what
today's loses. `ix-grammar::ebnf::parse` lowers into `EbnfGrammar`, and that lowering loses three
things:

- a duplicate rule overwrites the first (`HashMap::insert`, `ebnf.rs:474`);
- a quoted literal and a rule name both become a plain `String` (`ebnf.rs:522`);
- a terminal is whatever has no production (`constrained.rs:79`), so `start = "digit";
  digit = "0";` reads the quoted `"digit"` as the rule.

`ix-grammar` therefore gains a typed syntax tree, read before the lowering. In that tree a literal
is a literal and a name is a name, and duplicates are kept so they can be reported. The checks
run on that tree:

- every name used is defined, by the grammar or by the prelude below;
- no name is defined twice;
- one start rule: the first rule, as in the file;
- the grammar is not cyclic: no rule derives itself without consuming a character. The check runs
  on the typed tree's own lowering, synthetic rules for `[ ]` and `{ }` included, where a literal
  stays a literal. Today's `EbnfGrammar` cannot carry it: there, `start = "start";` would read as
  `start` deriving itself. The check builds a graph with an edge from A to B wherever A has an
  alternative `α B β`, with `B` a rule name, whose `α` and `β` can both match nothing, and
  refuses any cycle in that graph.
  - That covers unit cycles (`a = b; b = a | "x";`) and nullable repetitions (`{ [ "x" ] }`,
    which means the same as `{ "x" }`).
  - It still allows ordinary repetition such as `{ item [ sep ] }`, since every cycle there
    consumes an `item`.
  - A cyclic grammar has infinitely many parses of one text, so no count over them would end.

A grammar that fails is `GRAMMAR_INVALID`, with the names. The lexical terms come from a fixed,
documented prelude, which a grammar may use without defining. Each is one identifier, since the
reader splits names on spaces:

- `letter`, `digit` (ASCII);
- `space` (space or tab), `newline` (LF or CRLF);
- `any_char` (any Unicode scalar but a newline).

Anything else, such as `digits` or `integer`, the grammar defines itself. Parsing is
scannerless, over Unicode scalar values, so no separate lexer exists to drift from the grammar.

**3. Parsing is Earley, deterministic and bounded.** The result is one of the two rails of
ADR-0016 §4. A failure is `{"ok": False, "err": {"code", "message", …}}`, with the fields below
inside `err`:

- **One parse:** `{"ok": True, "value": tree}`. A tree node is `{"rule": name, "span": [start,
  end], "children": [...]}` and a terminal is its text. Spans are character offsets. Only the
  grammar's named rules give nodes: the synthetic rules of `[ ]`, `{ }` and groups put their
  children straight into the node of the named rule around them, so any parser behind the
  interface gives the same tree.
- **No parse:** code `SYNTAX`, with `line`, `column` and `expected`, the sorted set of terminals
  the parser could take there.
- **Two parses:** code `AMBIGUOUS`, with `line`, `column` and `rule`, where the first derivations
  part. One text, one meaning, one replay.
- **Over the bounds:** code `TOO_COMPLEX`. The text holds at most 64 KiB. The chart and the
  shared parse forest each hold at most a fixed number of items, counted as they are created.

Ambiguity is decided on the forest. The grammar is acyclic (decision 2), so the forest is a finite
graph with no cycle. The number of trees at each node is counted bottom-up, saturating at two, and
the root's count says one parse or `AMBIGUOUS`. That count is linear in the forest's nodes and
edges. The caps are set from the tracer's measurement: four times what an unambiguous 64 KiB
origami file needs. That proves a maximum-size origami file fits. Another grammar may need more
for the same size, so each DSL's own tracer measures its largest expected input against the caps.

`earlgrey` is tried first, behind a module of `ix-grammar`, on the conformance test of the
tracer slice. An Earley recognizer in `ix-grammar` itself replaces it only if `earlgrey` cannot
report the ambiguity position or the expected set, or fails the test. The interface stays the
same either way.

**4. A grammar is identified by its digest.** The digest is the SHA-256 of the grammar's bytes.
`parse` takes a grammar a run has locked (ADR-0018), never a path, and its intent carries the
digest. `parse` is a recorded capability (ADR-0016 §5): the tree is journaled, so replay does not
depend on the parser version.

**5. New DSLs go through the store, like any proposal.** An agent may design a grammar and
propose it with `propose("ebnf", name, body)`. The store adds the kind `ebnf`, whose library file
is `<name>.ebnf`. The host refuses a grammar that fails decision 2 before publishing it. A person
accepts it (ADR-0011 §6), and only then can a run lock it. The operator imports an outside
`.ebnf` file the same way.

**6. Towers are Starlark, not new machinery.** A high-level DSL is lowered by a Starlark walker
into the text of a lower DSL. That text is parsed by its own grammar, so every level is checked
by a grammar, and the journal holds each level's text.

**7. TARS drafts, IX judges.** The same `.ebnf` file can constrain a local model in TARS, through
GBNF in llama.cpp. That is a later slice, but IX's parse stays the judge either way: a draft that
does not parse is refused at its first error, whoever wrote it.

## Tracer slice (test first, each test seen red)

0. **The typed reader.** It keeps literals apart from names and reports a duplicate rule, a
   missing name and a cyclic grammar, the unit cycle `a = b; b = a | "x";` included.
   `start = "digit"; digit = "0";` accepts the text `digit`, not `0`. `start = "start";` and
   `{ item [ sep ] }` are accepted, not reported as cyclic.
1. **The origami grammar becomes executable.**
   - Its geometric line is renamed (`crease`).
   - It uses the prelude, and it defines `digits` and `integer` itself.
   - Its definition test checks decision 2 on it, not keywords in its text.
2. **`ix-grammar` parses with Earley:**
   - the four outcomes of decision 3 on small grammars: one parse, a syntax error with its
     expected set, an ambiguous grammar, an oversized input;
   - its trees on the origami grammar;
   - the measured chart and forest sizes of a 64 KiB origami file, which set the caps.
3. **Conformance against the handwritten parser**, on every `.origami` file in the repo plus
   mutated copies (a dropped token, a swapped keyword, a stray character):
   - every file the handwritten parser accepts, Earley accepts;
   - every file Earley accepts and the handwritten parser refuses breaks a rule a grammar cannot
     state (a statement that comes first and once, two different corners), listed in the test
     with its reason.
   Item 3 catches drift only in that corpus and those mutation classes; item 4 tests beyond it.
4. **Generation agrees with recognition.** Sentences generated from the grammar by `ix-grammar`'s
   constrained search are all accepted by the Earley parser. They are also accepted by the
   handwritten parser, or refused only for one of item 3's listed rules. That second check catches
   a grammar that accepts more than the language: the generator and the parser share the
   grammar, so they cannot catch it between them. TARS's Workflow-of-Thought DSL keeps its parser
   honest the same way.
5. **Only then, `parse` in `ix-durable`.** A durable script parses a fold sequence and proposes
   it. A kill after the parse replays the journaled tree.

## Rejected

- **A handwritten parser per DSL:** it is the drift that the origami grammar shows.
- **PEG (`pest_vm`, a translated grammar):** ordered choice hides ambiguity and changes what an
  EBNF grammar accepts.
- **LR or tree-sitter generators:** they build grammars into code at build time, but agents need
  grammars loaded at run time. LR generators also refuse ambiguous grammars.
- **A separate lexer:** a second specification to keep in step with the grammar.
- **An LLM judging whether a text follows a grammar:** a judge with a high false-accept rate,
  where a parser has none.

## Reversibility

**Two-way door** while `ix-durable` is experimental. The tree shape becomes a contract once a
released module walks it (ADR-0018), and is then frozen as `ix-dsl-tree/0`. The prelude is part
of that contract.

## Revisit triggers

- **A DSL needs context-sensitive rules** (indentation, a name defined before use): add a check
  pass in Starlark first; extend the grammar language only if many DSLs need the same check.
- **Parse time matters** (large texts, many calls): measure, then consider a faster parser for
  unambiguous grammars behind the same interface.
- **`earlgrey` goes unmaintained in a way that hurts** (a security report, a compiler break):
  switch to the in-crate recognizer, which the interface already allows.
