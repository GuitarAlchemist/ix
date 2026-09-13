#!/usr/bin/env python3
"""Gaia S1 R2 bundle validator, R15 repair revision.

Offline, standard-library only. No network access, no package installation,
no subprocess, no writes. It reads the bundle files, performs static checks,
re-runs each check against a deliberately broken fixture to prove the check
discriminates rather than merely passing, and then re-runs three of them against
mutations that must NOT trip them, to prove each stops where it claims to stop.

The R3 revision added the two check groups the independent R2 review found
missing, S1R2-B01 and S1R2-B02, and a second EBNF reader written from scratch
for the reachability proof.

The R4 revision repairs what the independent R3 review found in that work. Both
of R3's class-bound checks asked their question over a set of member *names*
while the grammar's real surface is a set of *derivations*: each reader took the
first terminal of a production and stopped, so a seventh bus verb hidden as a
top-level alternative inside `handoff` (S1R3-B01) and an entirely unbounded
ledger class (S1R3-B02) both passed the whole suite while the prose named checks
as detecting them. R4 adds one alternation-aware, per-class leading-terminal
bound, applied identically to all three classes, and states its own scope —
leading position only — as four positive controls rather than as prose.

R5 through R9 each repaired the same family one notch further out: which lines
count as a table row, and where the section 3.3 region begins and ends. The R11
revision closes the last proxy in it. The region's *ending* boundary has been
parsed out of the document since R8 and made fence-aware at R9, but its *start*
was still the literal prose sentence at `spec-v0.2:103`, so eight lines of
section 3.3 above that sentence were read by none of the three section 3.3
checks and a class row placed there declared a seventh bus verb with the whole
suite green (S1R10-B01, Standards axis). Both boundaries are now decided by the
one heading scan this reader already performed. The banner above said "R6
repair revision" from R7 to R9, which was itself untrue of the file printing it;
it is corrected here.

The R13 revision replaced the last proxy at the region's *opening* boundary: the
marker sentence now declares its own section designator and the reader resolves
the unique unfenced heading carrying it, so *which* section the region is stopped
being decided by proximity (S1R12-B01, Standards axis).

The R15 revision repairs the same family at the region's *closing* boundary, the
one end R13 deliberately did not touch. The forward scan ended at the first
unfenced ATX heading of the declared heading's level or shallower, and this
document nests by *numbering* rather than by level — `### 7.1.1` stands beneath
`### 7.1`. A same-level `### 3.3.1` after the marker therefore ended section 3.3
inside itself, and a rogue seventh-verb row behind it left all three section 3.3
checks green at `bus 6` while the document declared seven (S1R14-B01, Standards
axis). The closing scan now steps over a heading whose own designator is a proper
descendant of the declared one, and stops at every other heading exactly as
before; `NEG-R3B2-as` is the shape and `POS-HEAD-d` is the sibling that must stay
out of scope. This banner and the paragraph above it were themselves left at R11
while `main()` printed R13, which is the untrue-self-description class this bundle
repairs elsewhere; both are re-stamped here from the bytes that print them.

Usage:
    python -B gaia-s1-r2-validator.py                 # checks + fixtures + controls
    python -B gaia-s1-r2-validator.py --verify-manifest

Exit code 0 means every positive check passed, every negative fixture failed its
target check as intended, every fixture and control actually applied, and no
positive control tripped. Any other exit code means the bundle is not a valid
S1 R2 candidate. A fixture that no longer applies is reported as INAPP and is
distinct from a defect that was not caught: anchor drift is lost coverage, never
detection.

This validator grants no approval. It is a deterministic mechanical aid, and an
Advisory Artifact in the sense of the specification it checks.
"""

import hashlib
import os
import re
import sys

# --------------------------------------------------------------------------
# Bundle inventory
# --------------------------------------------------------------------------

SPEC = "gaia-consolidated-mission-room-factory-spec-v0.2.md"
EBNF = "gaia-mission-room-protocol-v0.2.ebnf"
TPN = "gaia-s1-r2-third-party-notices.md"
LEDGER = "gaia-s1-r2-change-ledger.md"
MANIFEST = "gaia-s1-r2-bundle-manifest.md"
OUTPUT = "gaia-s1-r2-validator-output.txt"
VALIDATOR = "gaia-s1-r2-validator.py"

DOCTRINE = "gaia-engineering-doctrine-v0.1.md"
DOMAIN = "gaia-mission-room-domain-context-v0.1.md"

# Copied trusted inputs, with the byte length and SHA-256 declared for them at
# the start of this repair. The validator re-verifies these independently.
TRUSTED_COPIES = {
    "gaia-artifact-workgraph-context-v0.1.md": (
        10956, "a5e3492c94feafd6eeb3a243aed1359ce121417c5069cb59be0e17f99aa0c57a"),
    "gaia-artifact-workgraph-staleness-design-v0.1.md": (
        53037, "7b9d0cd64ad29e897f6b294b2a2c91e1a5bcb708a89a1a4871a14eba926fd219"),
    "gaia-engineering-doctrine-v0.1.md": (
        10866, "f27c54b52c0a6a3b0479052f048223fe1a5c4894b0091897e94f8ee65cb858a0"),
    "gaia-mission-room-domain-context-v0.1.md": (
        4859, "4aaf104d8a80ed3ee8a8d312a8a40ecd401cc51a0e5a901f355a431c57262bad"),
    "gaia-multiaxis-uncertainty-analysis-design-v0.1.md": (
        10999, "65a32cb0ddd5a579efb17c2d5fd66c25ffa876faa9afe8dd3f192cc67ea8c6a7"),
    "gaia-multiaxis-uncertainty-math-primary-research.md": (
        11568, "06da04bd64d8e092d71f3900ccc87806602909b1839ce46855910da79c6d8c2c"),
    "gaia-uncertainty-grammar-v0.1.ebnf": (
        10322, "ee9f61b6be06aed76cdc0493c95c1540c6b148c31d75801ef9b67729b7394517"),
}

# The five candidate inputs admitted by specification section 9.6.
CANDIDATE_INPUTS = [
    "gaia-artifact-workgraph-context-v0.1.md",
    "gaia-artifact-workgraph-staleness-design-v0.1.md",
    "gaia-uncertainty-grammar-v0.1.ebnf",
    "gaia-multiaxis-uncertainty-math-primary-research.md",
    "gaia-multiaxis-uncertainty-analysis-design-v0.1.md",
]

REQUIRED_FILES = sorted(
    list(TRUSTED_COPIES) + [SPEC, EBNF, TPN, LEDGER, VALIDATOR])
ALLOWED_FILES = sorted(REQUIRED_FILES + [MANIFEST, OUTPUT])

BUS_VERBS = ["ack", "handoff", "heartbeat", "inbox", "register", "send"]

# The other two classes of the protocol partition (S1R2-B01). Every production
# reachable from `protocol` at statement position belongs to exactly one class.
LEDGER_TERMINALS = [
    "advisory", "artifact", "assign", "capsule", "claim", "evidence",
    "fence", "lane", "receipt", "refuse", "transition", "verdict",
]
FRAME_TERMINALS = ["close", "mission"]

CLASS_RULES = ["bus_statement", "ledger_statement", "frame_statement"]

# Attribution fields (S1R2-B02). Each MUST resolve to actor_ref, and every
# lattice-bearing production MUST bind at least one of them.
ATTRIBUTION_FIELDS = ["acceptor", "executor", "producer", "reviewer"]

MUTATION_CODES_V01 = [
    "ACCOUNTING_UNKNOWN", "DUPLICATE_INTENT", "EVIDENCE_DRIFT",
    "EXECUTOR_UNFENCED", "FENCE_MISMATCH", "LINEAGE_DIVERGENCE",
    "PRECONDITION_DRIFT", "PRIMARY_NOT_PROVEN_UNAVAILABLE",
    "STALE_EPOCH", "STORE_UNAVAILABLE",
]
CONTROL_CODES_NEW = [
    "AMBIGUOUS_RECIPIENT", "NON_DISCRIMINATING_GATE", "SCOPE_VIOLATION",
    "SELF_REVIEW", "UNREGISTERED_ACTOR",
]

LATTICES = [
    "evidence_state", "verdict_state", "freshness_state",
    "execution_result", "acceptance_state",
]

# Entity -> the grammar rule that carries its typed identity.
REQUIRED_ENTITIES = [
    ("Actor", "actor_ref"),
    ("Agent Profile", "profile_id"),
    ("Agent Incarnation", "incarnation_id"),
    ("Mission Room", "mission_id"),
    ("Role Assignment", "role_assignment_id"),
    ("Lane", "lane_id"),
    ("Task Capsule", "task_id"),
    ("Artifact Revision", "revision_id"),
    ("Transition Definition", "transition_definition_id"),
    ("Transition Receipt", "transition_receipt_id"),
    ("CoordinatorFence", "fence_id"),
    ("Evidence Item", "evidence_ref"),
    ("Message correlation", "correlation_id"),
]

# The single permitted occurrence of "oracle" inside a code span: a historical
# content-addressed evidence filename that must not be renamed.
ORACLE_FILENAME_EXEMPTION = "gaia-workgraph-projection-r1-oracle.md"

# --------------------------------------------------------------------------
# EBNF reader
# --------------------------------------------------------------------------

COMMENT_RE = re.compile(r"\(\*.*?\*\)", re.S)
QUOTE_RE = re.compile(r"\"[^\"]*\"|'[^']*'")
MARK_RE = re.compile("\x01\\d+\x02")
TOKEN_RE = re.compile("\x01\\d+\x02|[A-Za-z_][A-Za-z0-9_]*")
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class Ebnf:
    """Minimal reader: strips comments, masks string literals, splits rules."""

    def __init__(self, text):
        self.text = text
        body = COMMENT_RE.sub(" ", text)
        self.lits = {}

        def repl(match):
            key = "\x01%d\x02" % len(self.lits)
            self.lits[key] = match.group(0)
            return key

        body = QUOTE_RE.sub(repl, body)
        self.rules = {}
        self.duplicates = []
        for chunk in body.split(";"):
            if "=" not in chunk:
                continue
            head, rhs = chunk.split("=", 1)
            name = head.strip()
            if not NAME_RE.match(name):
                continue
            if name in self.rules:
                self.duplicates.append(name)
            self.rules[name] = rhs.strip()

    def has(self, name):
        return name in self.rules

    def rhs(self, name):
        return self.rules.get(name, "")

    def terminals(self, name):
        out = []
        for mark in MARK_RE.findall(self.rhs(name)):
            lit = self.lits[mark]
            if lit.startswith('"'):
                out.append(lit[1:-1])
        return out

    def references(self, name):
        """Nonterminal names referenced on the right-hand side of `name`."""
        return [t for t in TOKEN_RE.findall(self.rhs(name))
                if not t.startswith("\x01") and t in self.rules]

    def alternatives(self, name):
        # A stray closer must not drive `depth` below zero: a negative depth
        # makes every following top-level `|` invisible, which hides alternation
        # instead of reporting it (S1R4-B01). Underflow is floored here and
        # decided as an imbalance by the reachability reader below.
        out, depth, cur = [], 0, ""
        for ch in self.rhs(name):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            if ch == "|" and depth == 0:
                out.append(cur.strip())
                cur = ""
            else:
                cur += ch
        out.append(cur.strip())
        return [a for a in out if a]

    def first_terminal(self, name, depth=0):
        if depth > 8 or name not in self.rules:
            return None
        for tok in TOKEN_RE.findall(self.rhs(name)):
            if tok.startswith("\x01"):
                lit = self.lits[tok]
                return lit[1:-1] if lit.startswith('"') else None
            if tok in self.rules:
                return self.first_terminal(tok, depth + 1)
            return None
        return None

    def optional_groups(self, name):
        """Count EBNF optional groups `[ ... ]` after literal masking."""
        return self.rhs(name).count("[")

    def all_terminals(self):
        out = set()
        for name in self.rules:
            out.update(self.terminals(name))
        return out


# --------------------------------------------------------------------------
# Independent reachability reader  (S1R2-B01, S1R2-B02)
#
# Written from scratch. It shares no code, no regex and no helper with the
# `Ebnf` class above, and it is deliberately stricter: comment stripping is
# nesting-aware rather than a non-greedy regex, literal masking is a character
# scan rather than an alternation, and rule splitting rejects anything that is
# not a bare `name = rhs` pair instead of skipping it silently.
#
# The R2 candidate's mechanical claim failed not because its reader was wrong
# but because every check reused it and therefore inherited its blind spot:
# nothing ever asked what `protocol` reaches. This reader asks exactly that.
# --------------------------------------------------------------------------

IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

# Literal-mask delimiters. STX/ETX are chosen because neither is whitespace:
# Python's str.strip() treats \x1c-\x1f as whitespace and would silently eat a
# marker built from those, which is exactly the kind of quiet corruption this
# reader exists to avoid.
MASK_OPEN = "\x02"
MASK_CLOSE = "\x03"

# Sentinel for a leading position the reader cannot decide (S1R3-B01/B02). It
# is not a legal EBNF terminal, so it can never coincide with a declared class
# terminal, and any class multiset containing it fails its bound. Undecidable
# fails closed.
LEADING_UNRESOLVED = "<unresolved>"


class ReachabilityModel:
    """Second, independent EBNF reader used only by the R3 checks."""

    def __init__(self, text):
        self.literals = []
        body = self._mask_literals(self._strip_comments(text))
        self.rules = {}
        self.malformed = []
        self.duplicated = []
        for chunk in body.split(";"):
            if "=" not in chunk:
                if chunk.strip():
                    self.malformed.append(chunk.strip()[:40])
                continue
            head, rhs = chunk.split("=", 1)
            name = head.strip()
            if not name or any(c not in IDENT_CHARS for c in name):
                self.malformed.append(name[:40])
                continue
            if name in self.rules:
                self.duplicated.append(name)
            self.rules[name] = rhs.strip()

    # -- lexing ----------------------------------------------------------
    @staticmethod
    def _strip_comments(text):
        """Nesting-aware `(* ... *)` removal."""
        out, depth, i, end = [], 0, 0, len(text)
        while i < end:
            if text.startswith("(*", i):
                depth += 1
                i += 2
                continue
            if text.startswith("*)", i):
                if depth:
                    depth -= 1
                i += 2
                continue
            if depth == 0:
                out.append(text[i])
            i += 1
        return "".join(out)

    def _mask_literals(self, text):
        """Replace every quoted literal with an opaque marker."""
        out, i, end = [], 0, len(text)
        while i < end:
            ch = text[i]
            if ch in "\"'":
                close = text.find(ch, i + 1)
                if close == -1:
                    out.append(ch)
                    i += 1
                    continue
                self.literals.append(text[i + 1:close])
                out.append("%s%d%s" % (MASK_OPEN, len(self.literals) - 1, MASK_CLOSE))
                i = close + 1
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    # -- structure -------------------------------------------------------
    def _tokens(self, rhs):
        """Ordered (kind, value) stream: ('lit', text) or ('ref', name)."""
        out, i, end = [], 0, len(rhs)
        while i < end:
            ch = rhs[i]
            if ch == MASK_OPEN:
                close = rhs.find(MASK_CLOSE, i + 1)
                if close == -1:
                    break
                out.append(("lit", self.literals[int(rhs[i + 1:close])]))
                i = close + 1
                continue
            if ch in IDENT_CHARS:
                j = i
                while j < end and rhs[j] in IDENT_CHARS:
                    j += 1
                out.append(("ref", rhs[i:j]))
                i = j
                continue
            i += 1
        return out

    def unmask(self, text):
        """Restore literals, so a failure message is readable by a human."""
        out, i, end = [], 0, len(text)
        while i < end:
            if text[i] == MASK_OPEN:
                close = text.find(MASK_CLOSE, i + 1)
                if close == -1:
                    break
                out.append('"%s"' % self.literals[int(text[i + 1:close])])
                i = close + 1
                continue
            out.append(text[i])
            i += 1
        return "".join(out)

    def references(self, name):
        return [v for kind, v in self._tokens(self.rules.get(name, ""))
                if kind == "ref" and v in self.rules]

    def alternatives(self, name):
        rhs, depth, cur, out = self.rules.get(name, ""), 0, "", []
        for ch in rhs:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)   # see `_bracket_fault` (S1R4-B01)
            if ch == "|" and depth == 0:
                out.append(cur.strip())
                cur = ""
            else:
                cur += ch
        out.append(cur.strip())
        return [a for a in out if a]

    def alternative_names(self, name):
        """Alternatives that are a single bare production name.

        An alternative that inlines a terminal instead of naming a production
        is reported as an anomaly rather than silently ignored, because that is
        one way to smuggle an unclassified verb into a class.
        """
        names, inline = [], []
        for alt in self.alternatives(name):
            if self._bracket_fault(alt) is not None:
                # A stray closer left inside a class alternative would otherwise
                # tokenise away and leave the alternative reading as a bare name.
                inline.append(self.unmask(alt).strip()[:48])
                continue
            toks = self._tokens(alt)
            if len(toks) == 1 and toks[0][0] == "ref" and toks[0][1] in self.rules:
                names.append(toks[0][1])
            else:
                inline.append(self.unmask(alt).strip()[:48])
        return names, inline

    def head_terminal(self, name, seen=None):
        """Leading terminal of a production, resolved through references."""
        seen = seen or set()
        if name in seen or name not in self.rules:
            return None
        seen.add(name)
        for kind, value in self._tokens(self.rules[name]):
            if kind == "lit":
                return value
            return self.head_terminal(value, seen)
        return None

    # -- alternation-aware leading terminals  (S1R3-B01, S1R3-B02) --------
    #
    # `head_terminal` above answers "what does this production start with?" by
    # taking the first token and stopping. That question is too weak to bound a
    # class: it sees `handoff` and stops, so a second top-level alternative
    # inside `handoff` — a whole statement, utterable on its own, inside the bus
    # class — is invisible. The R3 independent review demonstrated exactly that
    # (`S1R3-B01`), and the same blindness left `ledger_statement` unbounded
    # (`S1R3-B02`).
    #
    # The methods below answer the stronger question: which terminals can stand
    # at the leading position of an utterance of this production? That is a FIRST
    # set, so it must descend into groups, union across every top-level
    # alternative at every level, step past an omittable element to the element
    # behind it, and resolve through a bare reference to another production.
    # Anything it cannot decide — an absent rule, a recursive cycle, an
    # unbalanced group, an alternative that can be uttered as nothing — yields
    # the sentinel LEADING_UNRESOLVED and a note. A bound that cannot be decided
    # must fail, not pass.

    @staticmethod
    def _bracket_fault(fragment):
        """Residual bracket imbalance in `fragment`, or None if it is balanced.

        Both halves of the concept, decided the same way (S1R4-B01). An opening
        bracket with no closer was already reported by `_elements`, but only when
        the element scan reached it: behind a leading terminal the scan returns
        first and the imbalance was discarded. A *closing* bracket with no opener
        was never reported at all — it is not in `pairs` and not in IDENT_CHARS,
        so `_elements` stepped over it, while the depth counters above went
        negative and hid every following top-level `|`. Either shape now yields
        LEADING_UNRESOLVED and a note. An undecidable bound fails closed; it is
        never read as satisfied.
        """
        pairs = {"(": ")", "[": "]", "{": "}"}
        stack = []
        for ch in fragment:
            if ch in pairs:
                stack.append(ch)
            elif ch in ")]}":
                if not stack:
                    return "closing %r has no opening group" % ch
                if pairs[stack.pop()] != ch:
                    return "group closed by %r does not match its opener" % ch
        if stack:
            return "opening %r has no closing group" % stack[-1]
        return None

    def _split_top(self, fragment):
        """Top-level `|` split of an arbitrary right-hand-side fragment."""
        depth, cur, out = 0, "", []
        for ch in fragment:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)   # see `_bracket_fault` (S1R4-B01)
            if ch == "|" and depth == 0:
                out.append(cur.strip())
                cur = ""
            else:
                cur += ch
        out.append(cur.strip())
        return [a for a in out if a]

    def _elements(self, fragment):
        """Ordered element stream of one alternative.

        ('lit', text)   a terminal
        ('ref', name)   a production reference
        ('group', body) `( ... )`, mandatory
        ('opt', body)   `[ ... ]` or `{ ... }`, omittable
        ('bad', text)   unbalanced or mismatched grouping
        """
        pairs = {"(": ")", "[": "]", "{": "}"}
        out, i, end = [], 0, len(fragment)
        while i < end:
            ch = fragment[i]
            if ch == MASK_OPEN:
                close = fragment.find(MASK_CLOSE, i + 1)
                if close == -1:
                    out.append(("bad", fragment[i:i + 24]))
                    break
                out.append(("lit", self.literals[int(fragment[i + 1:close])]))
                i = close + 1
                continue
            if ch in pairs:
                depth, j = 0, i
                while j < end:
                    if fragment[j] in pairs:
                        depth += 1
                    elif fragment[j] in ")]}":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                if j >= end or fragment[j] != pairs[ch]:
                    out.append(("bad", self.unmask(fragment[i:i + 24])))
                    break
                out.append(("group" if ch == "(" else "opt", fragment[i + 1:j]))
                i = j + 1
                continue
            if ch in IDENT_CHARS:
                k = i
                while k < end and fragment[k] in IDENT_CHARS:
                    k += 1
                out.append(("ref", fragment[i:k]))
                i = k
                continue
            i += 1
        return out

    def _fragment_heads(self, fragment, seen, owner, notes):
        heads = set()
        for alt in self._split_top(fragment):
            heads |= self._alternative_heads(alt, seen, owner, notes)
        if not heads:
            notes.append("%s: empty alternation" % owner)
            heads.add(LEADING_UNRESOLVED)
        return heads

    def _alternative_heads(self, alt, seen, owner, notes):
        fault = self._bracket_fault(alt)
        if fault is not None:
            notes.append("%s: unbalanced group in %r: %s"
                         % (owner, self.unmask(alt).strip()[:32], fault))
            return {LEADING_UNRESOLVED}
        heads, elements = set(), self._elements(alt)
        if not elements:
            notes.append("%s: alternative %r has no element" % (owner, alt[:24]))
            return {LEADING_UNRESOLVED}
        for kind, payload in elements:
            if kind == "lit":
                heads.add(payload)
                return heads
            if kind == "bad":
                notes.append("%s: unbalanced group near %r" % (owner, payload))
                heads.add(LEADING_UNRESOLVED)
                return heads
            if kind == "group":
                # Mandatory: the utterance cannot step past it.
                heads |= self._fragment_heads(payload, seen, owner, notes)
                return heads
            if kind == "opt":
                # Omittable: it contributes a head AND the next element may lead.
                heads |= self._fragment_heads(payload, seen, owner, notes)
                continue
            if kind == "ref":
                if payload not in self.rules:
                    notes.append("%s: leading reference %r resolves to no production"
                                 % (owner, payload))
                    heads.add(LEADING_UNRESOLVED)
                    return heads
                if payload in seen:
                    notes.append("%s: leading position recurses through %r"
                                 % (owner, payload))
                    heads.add(LEADING_UNRESOLVED)
                    return heads
                heads |= self._fragment_heads(self.rules[payload],
                                              seen | {payload}, owner, notes)
                return heads
        # Every element was omittable, so the alternative can be uttered as
        # nothing and has no leading terminal of its own.
        notes.append("%s: alternative %r is entirely omittable"
                     % (owner, self.unmask(alt).strip()[:32]))
        heads.add(LEADING_UNRESOLVED)
        return heads

    def leading_terminals(self, name):
        """Distinct terminals that may stand at the leading position of `name`."""
        notes = []
        if name not in self.rules:
            return [LEADING_UNRESOLVED], ["%s: production absent" % name]
        heads = self._fragment_heads(self.rules[name], {name}, name, notes)
        return sorted(heads), notes

    def class_leading_terminals(self, cls):
        """Leading terminals of every top-level alternative of every member.

        Returned as a sorted multiset: one entry per distinct leading terminal
        per member, so a member carrying two alternatives contributes two
        entries and two members sharing a terminal contribute two entries. Both
        are cardinality faults against a declared class set and both must fail.
        """
        if cls not in self.rules:
            return [LEADING_UNRESOLVED], ["%s: class production absent" % cls]
        names, inline = self.alternative_names(cls)
        heads, notes = [], ["%s: alternative does not name a production: %s"
                            % (cls, bad) for bad in inline]
        if inline:
            heads.append(LEADING_UNRESOLVED)
        for member in names:
            member_heads, member_notes = self.leading_terminals(member)
            heads.extend(member_heads)
            notes.extend(member_notes)
        return sorted(heads), notes

    def reachable_from(self, root):
        """Transitive closure of production references from `root`."""
        if root not in self.rules:
            return set()
        seen, stack = set(), [root]
        while stack:
            current = stack.pop()
            for ref in self.references(current):
                if ref not in seen:
                    seen.add(ref)
                    stack.append(ref)
        return seen

    # -- the partition ---------------------------------------------------
    def statement_position(self):
        """Productions a conforming utterance may present at statement position.

        `protocol` names the frame productions directly and reaches the rest
        through `statement`; expanding the class containers yields the full set.
        A production introduced anywhere in that frontier — including at frame
        level, which is where the R2 candidate was blind — lands here.
        """
        frontier = set(self.references("protocol")) | set(self.references("statement"))
        out, inline = set(), []
        for name in sorted(frontier):
            if name == "statement":
                continue
            if name in CLASS_RULES:
                names, bad = self.alternative_names(name)
                out.update(names)
                inline.extend("%s -> %s" % (name, b) for b in bad)
            else:
                out.add(name)
        return out, inline

    def classification(self):
        """production -> the classes claiming it. Exactly one is required."""
        claims, inline = {}, []
        for cls in CLASS_RULES:
            if cls not in self.rules:
                continue
            names, bad = self.alternative_names(cls)
            inline.extend("%s -> %s" % (cls, b) for b in bad)
            for name in names:
                claims.setdefault(name, []).append(cls)
        return claims, inline

    def attribution_fields(self):
        """Rules whose entire right-hand side is `actor_ref`."""
        return sorted(n for n, rhs in self.rules.items()
                      if [t for t in self._tokens(rhs)] == [("ref", "actor_ref")])

    def lattice_bearing(self):
        """Productions that bind a value from one of the five lattices."""
        out = {}
        for name in sorted(self.rules):
            bound = [x for x in LATTICES if x in self.references(name)]
            if bound and name not in LATTICES:
                out[name] = bound
        return out


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------

class Ctx:
    def __init__(self, spec, ebnf_text, tpn, ledger, root):
        self.spec = spec
        self.ebnf_text = ebnf_text
        self.ebnf = Ebnf(ebnf_text)
        self.tpn = tpn
        self.ledger = ledger
        self.root = root

    def copy_with(self, spec=None, ebnf_text=None, tpn=None, ledger=None):
        return Ctx(
            spec if spec is not None else self.spec,
            ebnf_text if ebnf_text is not None else self.ebnf_text,
            tpn if tpn is not None else self.tpn,
            ledger if ledger is not None else self.ledger,
            self.root,
        )


def code_spans(text):
    return re.findall(r"`([^`\n]+)`", text)


_MD_LINK_RE = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")
_MD_EDGE_MARKUP = "`*_~ \t"


def md_plain_text(text):
    """One rendering-independent reading of a Markdown fragment.

    Resolves a Markdown link to its text and strips inline code and emphasis
    markup from the edges. Only the edges are stripped, so a respelled name
    stays a distinct name. This is the single normalizer both section 3.3 cells
    are read through: the `Leading terminals` cell name by name, and the `Class`
    cell that decides which rows the table has. Reading one of them through a
    normalizer and the other through a format-specific pattern is what made
    S1R6-B01 reachable.
    """
    return _MD_LINK_RE.sub(r"\1", text).strip().strip(_MD_EDGE_MARKUP).strip()


def cell_terminal_names(cell):
    """Every name a section 3.3 `Leading terminals` cell lists, in cell order.

    The cell used to be read with `code_spans`, which sees backticked text and
    nothing else, so a terminal added to the cell bare, bolded or as a Markdown
    link was invisible to an equality that section 3.3 states without
    qualification (S1R5-B01, Spec axis). Patching the three renderings one at a
    time would leave the fourth open; one normalizer over the whole cell decides
    all of them: resolve a Markdown link to its text, split the cell on its
    commas, and strip inline code and emphasis markup from each token's edges.
    Only the edges are stripped, so a respelled name stays a distinct name.

    A list is returned, not a set, because its length is the cell's own
    cardinality — the number `SPEC_CLASS_TERMINALS_MATCH_GRAMMAR` binds to the
    measured LEAD multiset and to the Count cell beside it (S1R5-B02).
    """
    text = _MD_LINK_RE.sub(r"\1", cell)
    names = []
    for token in text.split(","):
        token = md_plain_text(token)
        if token:
            names.append(token)
    return names


_MD_ROW_INDENT_RE = re.compile(r"^ {0,3}[^ ]")
_MD_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")
_MD_SEPARATOR_CELL_RE = re.compile(r"^:?-+:?$")
_BARE_INTEGER_RE = re.compile(r"[0-9]+")

CLASS_TABLE_COLUMNS = ("Class", "Leading terminals", "Count")


def table_row_cells(line):
    """The cells of a Markdown table row, or None if the line is not one.

    GFM admits up to three spaces of indentation before a table row and renders
    it identically; a fourth makes the line an indented code block, which is not
    a row and correctly ends the table. A pipe escaped as `\\|` is cell content,
    not a cell boundary.

    GFM also makes a row's **leading and trailing pipes optional**: `a | b` and
    `| a | b |` are the same row and render identically. R7 required the leading
    pipe (`^ {0,3}\\|`), which was S1R6-B01 one notch further out in both
    directions at once — a body row written without it declared a seventh bus
    leading terminal while both section 3.3 checks passed (S1R7-B01), and the
    truthful table rewritten in pipe-less GFM was rejected as "no class table"
    (S1R7-F01). One predicate decides both signs, so both are closed together.

    What a row still requires is **at least one unescaped pipe**. That is the
    cell separator itself, not a spelling of it, and it is what keeps ordinary
    prose out of the row set — which the header scan and the section-wide
    outside-the-table scan both depend on, since each reads every line of
    section 3.3. A line carrying no pipe at all is not read as a row here; GFM
    would render one directly beneath a body row as a degenerate single-cell
    row, and that shape is disclosed as open in the R8 handoff. It carries one
    cell, so it cannot state a Class, a Leading terminals cell and a Count, and
    cannot declare a verb; every shape that can is read.
    """
    if not _MD_ROW_INDENT_RE.match(line):
        return None
    if not _MD_CELL_SPLIT_RE.search(line):
        return None
    cells = _MD_CELL_SPLIT_RE.split(line.strip())
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return cells


def is_separator_row(cells):
    return bool(cells) and all(_MD_SEPARATOR_CELL_RE.match(c.strip())
                               for c in cells)


_ATX_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?: |\t|$)")

_MD_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def fenced_line_flags(text):
    """One flag per line of `text`: is this line part of a fenced code block?

    The region reader, then named `section_after_marker`, decided an ATX heading
    by matching a pattern against a line in isolation. A `#` line is only a
    heading where Markdown is reading
    *text*; inside a fenced code block it is content, and GFM renders no section
    boundary there at all. Both signs of S1R8-B01 came from that one omission,
    so the block context is established once, here, and both the enclosing-level
    scan and the terminating-heading scan consult it.

    Fence state is a property of the whole document, not of the slice being
    read, so the pass always starts at line 1 of `text`: a fence opened before
    the marker governs the lines after it.

    What is decided here, and only this:

    * An opening fence is a run of three or more backticks or three or more
      tildes, indented zero to three spaces. Four spaces make an indented code
      block instead, which cannot carry a heading either — `_ATX_HEADING_RE`
      already bounds indentation at three — so the two agree.
    * A backtick fence may not carry a backtick in its info string; a tilde
      fence may carry anything. This is the one place the two characters differ.
    * A fence is closed only by a run of the *same* character, at least as long
      as the opening run, carrying nothing but whitespace after it. A shorter
      run, or the other character, is content (`NEG-R3B2-ah`).
    * An unclosed fence runs to the end of the document. CommonMark closes it at
      the end of the containing block, and the document is the outermost block.
      The consequence is that no heading behind an unclosed fence ends a
      section, so the region over-reads rather than under-reads: `NEG-R3B2-ak`
      is caught, and `POS-FENCE-d` shows the same behaviour raises no false red
      on these bytes.

    This is **not** a CommonMark implementation and does not claim to be. It
    decides fenced code blocks at document level. A fence inside a block quote
    or a list item, an HTML block, and a setext heading are all outside what it
    reads; the setext consequence is pinned by `NEG-R3B2-aj` and carried as a
    disclosed residual. Nothing here changes what a line *means* once it is
    known not to be a heading: `class_rows` still reads every line of the region
    whatever it renders as, which is what S1R6-B01 established.
    """
    flags = []
    char, length = None, 0
    for line in text.split("\n"):
        match = _MD_FENCE_RE.match(line)
        if char is None:
            opening = bool(match) and not (match.group(1)[0] == "`"
                                           and "`" in match.group(2))
            flags.append(opening)
            if opening:
                char, length = match.group(1)[0], len(match.group(1))
        else:
            flags.append(True)
            if (match and match.group(1)[0] == char
                    and len(match.group(1)) >= length
                    and not match.group(2).strip()):
                char, length = None, 0
    return flags


_SECTION_DESIGNATOR_RE = re.compile("§(\\d+(?:\\.\\d+)*)")


def declared_section_region(text, marker):
    """The section the marker sentence *declares*, as `(region, problem)`.

    The three section 3.3 checks used to slice `text.split(marker, 1)[1][:2500]`
    and call the result "section 3.3". Measured over the R7 bytes section 3.3
    ran 9,548 characters past that marker, so 7,048 of them — 74% of the section
    — were read by nothing: neither the table parse nor the outside-the-table
    scan. `spec-v0.2:115` says "a row naming a class **anywhere in section 3.3**
    outside this table is a defect", and a second class table placed past the
    window declared a seventh bus leading terminal with the whole suite green
    (S1R7-B02).

    Raising the constant would replace one proxy with a larger one: wrong again
    as soon as the section grows, and reading into section 4 as soon as it
    shrinks. The section boundary is a property of the document, so it is parsed
    rather than approximated. The heading enclosing the marker fixes the level;
    the region ends at the next ATX heading of that level or shallower, which is
    where the section ends.

    R7 through R9 parsed only the *closing* boundary that way. The region still
    *began* at the marker string, so the eight lines of section 3.3 above that
    sentence — `spec-v0.2:96`-`103` on the R9 bytes, 403 of the section's 9,950
    characters, ordinary normative prose including the six-verb enumeration
    itself — were read by none of the three section 3.3 checks. One class-table
    row placed there declared a seventh `bus_statement` leading terminal beside
    a Count of 7 with all fifty-seven positive checks, all eighty-six fixtures
    and all nine controls green at `EXIT=0`; the identical row two lines later,
    across the marker, was caught immediately, so placement alone decided
    detection (S1R10-B01, Standards axis). A prose sentence standing in for a
    document property is the same class of proxy the 2500-character constant
    was, and the ledger asserted the opposite in normative voice: "the section
    boundary is a property of the document, so it is parsed rather than
    approximated" was true of one end only.

    R10 answered that by keeping the *position* of the nearest ATX heading above
    the marker, which the backward scan already computed in order to read its
    *level*. The region opened there and closed where it closed before, and the
    docstring claimed "the marker still says *which* section this is, and the
    document says where it starts and stops". Measured over the R11 bytes the
    first half of that sentence was not true of anything: the marker sentence
    named no section, so "which section this is" was decided by *proximity* —
    whichever heading happened to be nearest above the marker — and proximity is
    a property an ordinary edit changes. One legitimate subsection heading
    between the §3.3 heading and the marker moves the region's start down to it.
    A rogue `bus_statement` row declaring a seventh leading terminal `promote`
    beside a **Count** of 7, placed on the first line of §3.3 above that
    subsection, then left all fifty-seven positive checks green at 579 unshifted
    lines, and `NEG-R3B2-an` and `NEG-R3B2-ao` — two of the four fixtures the R10
    repair added to pin this very boundary — stopped being caught at the same
    time (S1R12-B01, Standards axis). Three heading forms did it: a deeper
    `#### 3.3.1`, a same-level `### 3.3.1` in this document's own house style
    (compare `### 7.1.1` beneath `### 7.1`, where the numbering and not the level
    carries the nesting), and a second `### 3.3` heading.

    The proxy under all three is the same one this family has replaced four
    times: a *position* standing in for an *identity*. So the identity is read
    rather than inferred. The marker sentence at `spec-v0.2:103` now declares its
    own section — "the scope of that witness is §3.3, the section whose heading
    stands above this sentence" — and this reader takes the designator from the
    marker's own line, then resolves the unique unfenced ATX heading whose text
    opens with that designator as a whole number. That is the minimum explicit
    anchoring the repair needs and the whole of it: no heading text, no section
    number and no sentence is hardcoded *here*; the specification says which
    section its own control is scoped to, and the document still says where that
    section starts and stops. A renumbered heading and an unrenumbered marker
    disagree in one document and fail closed together, which a validator-side
    literal `"### 3.3 "` could not do.

    `3.3` matches the heading `### 3.3 Exactly six non-privileged verbs` and does
    not match `### 3.3.1` or `### 3.30`, because the designator must be followed
    by neither a digit nor a dot-digit; a single trailing dot is admitted, so
    `### 3.3. Title` still resolves. A deeper subsection is therefore *inside* the
    region rather than the start of it, which is what a reader of `### 7.1.1`
    already assumes.

    The marker sentence stays load-bearing for two further reasons: the three
    checks fail closed when it is absent, which is what makes "this control is
    mechanically witnessable over the whole protocol" a claim the suite reads
    rather than one it assumes; and it is the one line whose text a reviewer can
    compare against the heading it names.

    R13 changed only the opening boundary, deliberately, and said so: "the closing
    boundary is untouched". That is what left `POS-FENCE-c`, `POS-FENCE-d`,
    `NEG-R3B2-aj` and the whole closing family reading exactly the bytes they read
    before, and `POS-HEAD-c` was the obligation from the other side — a genuine
    sibling `### 3.4` carrying a class table must stay out of scope, so a repair
    that widened the start by reading more of the document fails there as design
    (C) already failed at `POS-HEAD-a`.

    Leaving that end alone is what R15 repairs. The opening scan had just been
    taught that `### 3.3.1` is a *descendant* of `### 3.3` and not a rival for the
    same identity — the paragraph above says a deeper subsection is "inside the
    region rather than the start of it, which is what a reader of `### 7.1.1`
    already assumes" — while the closing scan still asked one question, "is this
    heading at the declared level or shallower", and answered it for `### 3.3.1`
    with yes. So one reader held two incompatible readings of the same document at
    its two ends. A same-level `### 3.3.1` placed after the marker ended §3.3
    inside itself, and the identical seventh-verb row this family has used since
    R10, placed behind it, left all three §3.3 checks green and both class checks
    printing `bus 6` while the section declared seven (S1R14-B01, Standards axis).
    Measured over the R14 bytes the region ran 8,260 characters and 39 lines where
    §3.3 ran 10,277 and 44 — the last five lines of the section, the rogue
    declaration among them, read by nothing — and the identical row with the
    subsection heading removed was caught immediately by both class checks, so
    placement alone decided detection for the eighth consecutive round.

    The repair is one predicate on the scan that already runs: a heading at the
    declared level or shallower still ends the region **unless its own designator
    is a proper descendant of the declared one**, which is decided by the same
    `number` the marker sentence declares and the same `re.escape` of it that the
    opening anchor uses. `### 3.3.1` and `### 3.3.10.2` are stepped over;
    `### 3.4`, `### 3.30` and `## 4.` are not, because a proper descendant must
    carry the declared designator followed by a dot **and a digit** — a prefix
    alone is not a descendant, which is the hazard this predicate introduces and
    the reason it is spelled `\\.\\d` rather than as a `startswith`. A heading that
    carries no designator at all is unaffected and is still decided by level
    alone, so a legitimate `#### Notes` inside §3.3 stays inside it and a decoy
    `#### ` heading still cannot truncate the section (`NEG-R3B2-ai`).

    `NEG-R3B2-as` is the shape, and `POS-HEAD-d` is the control it creates: the
    same two edits at the same two seams, differing in exactly one thing — whether
    the heading is `### 3.3.1` or `### 3.4`. A repair that answered the fixture by
    comparing levels more loosely, or by reading to the end of the document, is
    caught there and at `POS-HEAD-c` and `POS-FENCE-c`. The scan still starts on
    the line *after* the marker, so a heading between the section's own heading
    and the marker never terminates the region, and the level it compares against
    is still the declared heading's — the same value, 3, on these bytes.

    The over-read direction is preserved and is the one this reader has always
    chosen when it is unsure: a document that renumbered §4 as `### 3.3.9` would
    have its §4 read as part of §3.3, which raises a false red rather than hiding
    a declaration. That is disclosed, not defended as free.

    R8 parsed that boundary out of a line in isolation, with no Markdown block
    context, so `^ {0,3}#{1,6}` matched a `#` line inside a fenced code block —
    where GFM renders code, not a section boundary. One predicate, both signs
    (S1R8-B01). A fenced example carrying a `## ` line ended section 3.3 four
    lines into it, and a second class table behind the fence declared a seventh
    `bus_statement` leading terminal with all fifty-seven positive checks, all
    eighty fixtures and all five controls green at EXIT=0. On a truthful
    document the same omission reversed: a legitimate fenced example placed
    between the marker and the genuine table left an empty region, and the table
    a reader sees was reported as "section 3.3 carries no class table". Both are
    closed together by `fenced_line_flags`, because one reader decides both.

    Suppressing fenced *content* would close the fixtures and reopen S1R6-B01:
    `class_rows` deliberately reads a line that renders as code, since a row
    naming a class inside section 3.3 is a defect whatever it renders as. So the
    fence state is consulted only where a heading is *recognized* — the two
    heading scans — and the region's bytes are untouched.

    Both heading scans are fence-aware, which R8 established and this reader
    keeps: a `#` line inside a fenced code block is content, not a boundary. R8
    defaulted an undecidable enclosing level to 6, which ends the region at the
    first heading of any level after the marker — the nearest excuse to stop, and
    a decoy `#### ` heading is then enough to hide the rest of the section
    (`NEG-R3B2-ai`). R10 defaulted it to 0 instead, so an undecidable level read
    to the end of the document: undecidable failed *open* on the region and
    therefore closed on the check.

    That direction is now reversed for the opening boundary, deliberately, and
    the reversal is the second half of the identity repair. "Undecidable" no
    longer means "no heading was found near the marker"; it means **the document
    does not carry exactly one unfenced heading for the section this control
    declares itself scoped to**, which is not a region the reader may guess at —
    it is a defect in the pairing of `spec-v0.2:103` against the heading it
    names. So the region is empty and the reason is *reported*, rather than the
    reader silently over-reading and the caller printing "section 3.3 states no
    Count for bus_statement" — a true failure with a misleading cause, which this
    bundle treats as its own defect class (S1R3-N07). Zero headings is the fenced
    form (`NEG-R3B2-ai`, `NEG-R3B2-ao`, both still caught, now by name); two is
    the duplicated-designator form (`NEG-R3B2-ar`), a hazard this mechanism
    introduces and therefore pins itself.

    The *declared* scope is still what stops the region from becoming unbounded:
    `spec-v0.2:115` says "anywhere in §3.3", not anywhere in the document, so
    `POS-FENCE-c` on the closing side and `POS-HEAD-a` and `POS-HEAD-c` on the
    opening side all require a class table one section away to be out of scope. A
    repair that answered this family by reading the whole file catches every
    fixture and fails all three.
    """
    head, found, _ = text.partition(marker)
    if not found:
        return "", "the specification does not carry the marker sentence"
    lines = text.split("\n")
    marker_line = head.count("\n")
    declared = _SECTION_DESIGNATOR_RE.search(lines[marker_line])
    if not declared:
        return "", ("the marker sentence at spec line %d declares no section, "
                    "so the scope of these bounds is undecidable"
                    % (marker_line + 1))
    number = declared.group(1)
    heading_re = re.compile("^ {0,3}(#{1,6})[ \t]+"
                            + re.escape(number) + "\\.?(?![\\d.])")
    fenced = fenced_line_flags(text)
    anchors = [i for i in range(len(lines))
               if not fenced[i] and heading_re.match(lines[i])]
    if len(anchors) != 1:
        return "", ("the specification carries %d unfenced headings for the "
                    "declared section %s, and exactly one is required"
                    % (len(anchors), number))
    start = anchors[0]
    if start > marker_line:
        return "", ("the declared section %s opens at spec line %d, below the "
                    "marker sentence at line %d"
                    % (number, start + 1, marker_line + 1))
    level = len(_ATX_HEADING_RE.match(lines[start]).group(1))
    descendant_re = re.compile("^ {0,3}#{1,6}[ \t]+"
                               + re.escape(number) + "\\.\\d")
    end = len(lines)
    for i in range(marker_line + 1, len(lines)):
        if fenced[i]:
            continue
        match = _ATX_HEADING_RE.match(lines[i])
        if (match and len(match.group(1)) <= level
                and not descendant_re.match(lines[i])):
            end = i
            break
    return "\n".join(lines[start:end]), None


def class_rows(region):
    """Every row of the section 3.3 class table as (rows, problems).

    Three revisions in a row read this table through a pattern and called the
    result "the table".

    * R5 read it with `dict(re.findall(...))`, which keeps the *last* pair for a
      repeated key, so a bogus row above the genuine one shadowed it and
      position alone decided detection (S1R5-B01, Standards axis).
    * R6 collected every match instead of the last, which closed that route and
      left the pattern itself deciding what a row is. `CLASS_ROW_RE` required
      the Class cell to be a backtick code span, the Count cell to be bare
      digits and the line to begin at column 0, so a second `bus_statement` row
      spelled bare, bolded, linked, carrying a Count of `seven` or `7<sup>a</sup>`,
      or indented by one to three spaces, was not a duplicate, not a
      declaration, and not there. Thirteen such rows declared a seventh bus
      leading terminal while all fifty-seven positive checks, all sixty-two
      fixtures and all four controls stayed green (S1R6-B01, both axes). A
      substitution form was worse: the only `bus_statement` row a reader sees
      declared seven verbs while a decoy row carrying the truthful six absorbed
      both checks.

    Widening the pattern would have repeated the mistake one column to the left:
    the pattern would still decide which rows exist, and the next rendering
    nobody enumerated would be the next blind spot. So duplication is decided
    over the *table* rather than over a pattern's matches. The physical table is
    parsed — header, separator, and the contiguous body rows beneath it — its
    columns are resolved by name from its own header, and every body row is
    read. A row inside a normative table that names no known class, states a
    Count that is not a bare integer, or carries a cell count its header does
    not declare is a defect and is reported; it is never silently absent. This
    is what makes `spec-v0.2:115` and `spec-v0.2:127` true as written, and it
    subsumes the non-numeric Count cell disclosed as open in the R6 handoff.

    More than one row for a class is itself a defect in a table whose rows are
    normative, so it is reported rather than resolved: the caller fails closed
    instead of choosing a row. `rows` keeps the first occurrence purely so the
    caller has something to name in the surviving per-class messages. A Count
    cell that is not a bare integer is stored as None, so no caller converts a
    number the parse did not establish.
    """
    lines = region.splitlines()
    problems = []

    header_at, columns = None, None
    for i, line in enumerate(lines):
        cells = table_row_cells(line)
        if cells is None or is_separator_row(cells):
            continue
        names = [md_plain_text(c) for c in cells]
        if all(col in names for col in CLASS_TABLE_COLUMNS):
            header_at, columns = i, names
            break
    if header_at is None:
        return {}, ["section 3.3 carries no class table: no header row naming "
                    "%s" % ", ".join(CLASS_TABLE_COLUMNS)]
    index = {col: columns.index(col) for col in CLASS_TABLE_COLUMNS}

    body, separator_at, end = [], None, header_at + 1
    while end < len(lines):
        cells = table_row_cells(lines[end])
        if cells is None:
            break
        if is_separator_row(cells) and separator_at is None:
            separator_at = end
        else:
            body.append(cells)
        end += 1
    if separator_at != header_at + 1:
        problems.append("section 3.3's class table carries no separator row "
                        "beneath its header, so its rows are not table rows")

    rows, seen = {}, {}
    for cells in body:
        if len(cells) != len(columns):
            problems.append("section 3.3's class table carries a row of %d cells "
                            "where its header declares %d"
                            % (len(cells), len(columns)))
            continue
        cls = md_plain_text(cells[index["Class"]])
        if cls not in CLASS_RULES:
            problems.append("section 3.3's class table carries a row whose Class "
                            "cell names no class: %r" % cls)
            continue
        seen[cls] = seen.get(cls, 0) + 1
        count = cells[index["Count"]].strip()
        if not _BARE_INTEGER_RE.fullmatch(count):
            problems.append("section 3.3 states a Count of %r for %s, which is "
                            "not a bare integer" % (count, cls))
            count = None
        rows.setdefault(cls, (cells[index["Leading terminals"]], count))
    for cls in sorted(seen):
        if seen[cls] > 1:
            problems.append("section 3.3 carries %d rows for %s; the class table "
                            "declares exactly one row per class"
                            % (seen[cls], cls))

    # A normative class declaration belongs in the class table and nowhere else,
    # so a class-naming row found anywhere else in the region is reported. GFM's
    # three-space bound decides what a row of *this* table is; outside it the
    # indentation is stripped first, because a line that renders as a code block
    # rather than as a table row still names a class inside section 3.3, and
    # deciding it by its rendering is what S1R6-B01 was.
    for i, line in enumerate(lines):
        if header_at <= i < end:
            continue
        cells = table_row_cells(line.lstrip())
        if cells and md_plain_text(cells[0]) in CLASS_RULES:
            problems.append("section 3.3 carries a row for %s outside the class "
                            "table" % md_plain_text(cells[0]))
    return rows, problems


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Checks. Each returns (ok, detail).
# --------------------------------------------------------------------------

def c_capsule_fields_present(ctx):
    missing = [f for f in ("non_goals:", "dependencies:") if f not in ctx.spec]
    return (not missing,
            "capsule schema carries non_goals and dependencies"
            if not missing else "missing from capsule schema: %s" % missing)


def c_capsule_cardinality_table(ctx):
    rows = {}
    for line in ctx.spec.splitlines():
        if line.startswith("| **`non_goals`**"):
            rows["non_goals"] = line
        if line.startswith("| **`dependencies`**"):
            rows["dependencies"] = line
    if len(rows) != 2:
        return False, "cardinality rows absent for non_goals/dependencies"
    ok = ("1..n" in rows["non_goals"]
          and "empty list is refused" in rows["non_goals"]
          and "0..n" in rows["dependencies"]
          and "absent is refused" in rows["dependencies"])
    return ok, ("non_goals 1..n empty-refused; dependencies 0..n absent-refused"
                if ok else "cardinality rows do not state the refusals")


def c_ebnf_capsule_non_goals_nonempty(ctx):
    e = ctx.ebnf
    if not e.has("non_goal_set"):
        return False, "non_goal_set rule absent"
    groups = e.optional_groups("non_goal_set")
    return (groups == 0,
            "non_goal_set admits no optional group, so cardinality is 1..n"
            if groups == 0
            else "non_goal_set has %d optional group(s): empty list parses" % groups)


def c_ebnf_capsule_dependencies_typed(ctx):
    e = ctx.ebnf
    if not (e.has("dependency_set") and e.has("dependency_ref")):
        return False, "dependency_set or dependency_ref absent"
    if e.optional_groups("dependency_set") != 1:
        return False, "dependency_set is not 0..n"
    alts = [a.strip() for a in e.alternatives("dependency_ref")]
    ok = sorted(alts) == ["contract_ref", "task_id"]
    return ok, ("dependency_set 0..n; dependency_ref is task_id | contract_ref"
                if ok else "dependency_ref alternatives are %s" % alts)


def c_capsule_statement_binds_both(ctx):
    refs = ctx.ebnf.references("capsule")
    need = ["non_goal_set", "dependency_set", "task_id", "mission_id"]
    missing = [n for n in need if n not in refs]
    return (not missing,
            "capsule statement binds %s" % ", ".join(need)
            if not missing else "capsule statement missing %s" % missing)


def c_no_advisory_oracle(ctx):
    hits = []
    if "AdvisoryOracle" in ctx.spec:
        hits.append(SPEC)
    if "AdvisoryOracle" in ctx.ebnf_text:
        hits.append(EBNF)
    return (not hits,
            "AdvisoryOracle absent from specification and grammar"
            if not hits else "AdvisoryOracle present in %s" % hits)


def c_no_oracle_identifier(ctx):
    bad = [s for s in code_spans(ctx.spec)
           if "oracle" in s.lower() and s != ORACLE_FILENAME_EXEMPTION]
    exempt_used = ORACLE_FILENAME_EXEMPTION in code_spans(ctx.spec)
    if bad:
        return False, "declared name containing 'oracle': %s" % bad
    return True, ("no declared Gaia name contains 'oracle'; the single exemption "
                  "is the historical evidence filename, %s"
                  % ("present" if exempt_used else "absent"))


def c_ebnf_role_has_no_oracle(ctx):
    terms = ctx.ebnf.terminals("role")
    if not terms:
        return False, "role production absent"
    ok = "Oracle" not in terms and "AdvisorySpecialist" in terms
    return ok, ("role terminals carry AdvisorySpecialist and no Oracle"
                if ok else "role terminals are %s" % terms)


def c_per_system_adapters_declared(ctx):
    need = ["IxEvidenceAdapter", "TarsGrammarAdapter", "HariReplayAdapter",
            "No shared advisory seam is declared"]
    missing = [n for n in need if n not in ctx.spec]
    return (not missing,
            "three narrow adapters declared; no shared seam"
            if not missing else "missing %s" % missing)


def c_no_result_state_union(ctx):
    present = ctx.ebnf.has("result_state")
    return (not present,
            "the merged result_state union is gone"
            if not present else "result_state union is still defined")


def c_lattices_pairwise_disjoint(ctx):
    e = ctx.ebnf
    sets = {}
    for name in LATTICES:
        if not e.has(name):
            return False, "lattice %s absent" % name
        terms = set(e.terminals(name))
        if not terms:
            return False, "lattice %s is empty" % name
        sets[name] = terms
    for i, a in enumerate(LATTICES):
        for b in LATTICES[i + 1:]:
            overlap = sets[a] & sets[b]
            if overlap:
                return False, "%s and %s share %s" % (a, b, sorted(overlap))
    total = sum(len(v) for v in sets.values())
    return True, ("%d lattices, %d tokens, all pairwise disjoint"
                  % (len(LATTICES), total))


def _binds_only(ctx, statement, lattice):
    refs = ctx.ebnf.references(statement)
    if lattice not in refs:
        return False, "%s does not bind %s" % (statement, lattice)
    others = [x for x in LATTICES if x != lattice and x in refs]
    if others:
        return False, "%s also binds %s" % (statement, others)
    return True, "%s binds %s and no other lattice" % (statement, lattice)


def c_advisory_binds_evidence_state_only(ctx):
    return _binds_only(ctx, "advisory", "evidence_state")


def c_evidence_binds_evidence_state_only(ctx):
    return _binds_only(ctx, "evidence", "evidence_state")


def c_verdict_binds_verdict_state_only(ctx):
    return _binds_only(ctx, "verdict", "verdict_state")


def c_spec_declares_lattice_separation(ctx):
    need = [
        "pairwise disjoint token sets",
        "Advisory Artifact carries an evidence state only",
        "approval, acceptance, safety, authority, or freshness",
    ]
    missing = [n for n in need if n not in ctx.spec]
    return (not missing,
            "specification states lattice separation and the advisory boundary"
            if not missing else "missing %s" % missing)


def c_identifier_body_excludes_colon(ctx):
    e = ctx.ebnf
    if not e.has("identifier_body"):
        return False, "identifier_body absent"
    terms = e.terminals("identifier_body")
    return (":" not in terms,
            "identifier_body admits no ':', so a type prefix is unforgeable"
            if ":" not in terms else "identifier_body still admits ':'")


def _typed_prefixes(ctx):
    out = {}
    for name in sorted(ctx.ebnf.rules):
        terms = ctx.ebnf.terminals(name)
        if terms and terms[0].endswith(":") and "identifier_body" in ctx.ebnf.rhs(name):
            out[name] = terms[0]
    return out


def c_typed_id_prefixes_disjoint(ctx):
    prefixes = _typed_prefixes(ctx)
    if len(prefixes) < 20:
        return False, "only %d typed identifier rules found" % len(prefixes)
    seen = {}
    for name, pfx in sorted(prefixes.items()):
        if pfx in seen:
            return False, "prefix %r shared by %s and %s" % (pfx, seen[pfx], name)
        seen[pfx] = name
    return True, ("%d typed identifier prefixes, all distinct and colon-terminated"
                  % len(prefixes))


def c_recipient_unambiguous(ctx):
    e = ctx.ebnf
    if not e.has("actor_ref"):
        return False, "actor_ref absent"
    alts = e.alternatives("actor_ref")
    if len(alts) != 2:
        return False, "actor_ref has %d alternatives" % len(alts)
    tags = []
    for alt in alts:
        marks = MARK_RE.findall(alt)
        if not marks:
            return False, "actor_ref alternative %r carries no keyword tag" % alt
        tags.append(e.lits[marks[0]].strip('"'))
    if sorted(tags) != ["incarnation", "profile"]:
        return False, "actor_ref tags are %s" % tags
    if "actor_ref" not in e.references("recipient"):
        return False, "recipient is not an actor_ref"
    return True, "recipient is a keyword-tagged, prefix-disjoint actor_ref"


def c_required_entities_typed_and_referenced(ctx):
    e = ctx.ebnf
    problems = []
    for entity, rule in REQUIRED_ENTITIES:
        if not e.has(rule):
            problems.append("%s: rule %s absent" % (entity, rule))
            continue
        users = [n for n in e.rules if n != rule and rule in e.references(n)]
        if not users:
            problems.append("%s: rule %s is never referenced" % (entity, rule))
    return (not problems,
            "all %d required entities carry a typed identity and are referenced"
            % len(REQUIRED_ENTITIES)
            if not problems else "; ".join(problems))


def _bus_heads(ctx):
    return sorted(
        (ctx.ebnf.first_terminal(a) or "?")
        for a in ctx.ebnf.alternatives("bus_statement"))


def c_exactly_six_bus_verbs(ctx):
    e = ctx.ebnf
    if not e.has("bus_statement"):
        return False, "bus_statement absent"
    alts = e.alternatives("bus_statement")
    heads = _bus_heads(ctx)
    if len(alts) != 6:
        return False, "bus_statement has %d alternatives: %s" % (len(alts), heads)
    if heads != BUS_VERBS:
        return False, "bus terminals are %s, expected %s" % (heads, BUS_VERBS)
    return True, "exactly six bus alternatives: %s" % " ".join(heads)


def c_inbox_present_in_grammar(ctx):
    present = "inbox" in ctx.ebnf.all_terminals()
    return (present,
            "the inbox terminal is present in the grammar"
            if present else "inbox does not occur in the grammar")


def c_ledger_terminals_disjoint_from_bus(ctx):
    e = ctx.ebnf
    if not e.has("ledger_statement"):
        return False, "ledger_statement absent"
    heads = [(a, e.first_terminal(a)) for a in e.alternatives("ledger_statement")]
    clash = sorted(n for n, t in heads if t in BUS_VERBS)
    if clash:
        return False, "ledger statements collide with a bus verb: %s" % clash
    return True, ("%d ledger statements, none is a bus verb" % len(heads))


def c_statement_partition_exhaustive(ctx):
    alts = [a.strip() for a in ctx.ebnf.alternatives("statement")]
    ok = alts == ["bus_statement", "ledger_statement"]
    return ok, ("statement partitions into bus_statement | ledger_statement"
                if ok else "statement alternatives are %s" % alts)


def c_no_seventh_verb_production(ctx):
    """No rule other than the six bus alternatives leads with a bus verb.

    `statement` and `bus_statement` are the containers that reach those six by
    construction, so they are excluded by name rather than by accident.
    """
    e = ctx.ebnf
    allowed = set(e.alternatives("bus_statement")) | {"statement", "bus_statement"}
    strays = sorted(n for n in e.rules
                    if n not in allowed and e.first_terminal(n) in BUS_VERBS)
    return (not strays,
            "no production outside the six bus alternatives leads with a bus verb"
            if not strays else "productions leading with a bus verb: %s" % strays)


def c_bus_terminals_match_spec(ctx):
    marker = "The only bus verbs are:"
    if marker not in ctx.spec:
        return False, "specification does not declare the bus verb set"
    tail = ctx.spec.split(marker, 1)[1]
    line = next(l for l in tail.splitlines() if l.strip())
    declared = sorted(code_spans(line))
    heads = _bus_heads(ctx)
    ok = declared == BUS_VERBS == heads
    return ok, ("specification and grammar declare the same six verbs"
                if ok else "specification says %s, grammar says %s" % (declared, heads))


# -- S1R2-B01: the protocol partition is exhaustive ------------------------

def c_frame_statement_bounded(ctx):
    """frame_statement exists, has exactly two alternatives, and is inert."""
    model = ReachabilityModel(ctx.ebnf_text)
    if "frame_statement" not in model.rules:
        return False, "frame_statement absent: mission and close are unclassified"
    names, inline = model.alternative_names("frame_statement")
    if inline:
        return False, "frame alternative does not name a production: %s" % inline
    heads = sorted(model.head_terminal(n) or "?" for n in names)
    if len(names) != 2 or heads != FRAME_TERMINALS:
        return False, ("frame_statement has %d alternatives with heads %s, expected %s"
                       % (len(names), heads, FRAME_TERMINALS))
    intruder = [h for h in heads if h in BUS_VERBS or h in LEDGER_TERMINALS]
    if intruder:
        return False, "frame terminal collides with another class: %s" % intruder
    return True, "frame_statement has exactly two inert alternatives: %s" % " ".join(heads)


def c_protocol_partition_exhaustive(ctx):
    """Every production reachable from `protocol` is classified exactly once.

    This is the check the R2 candidate did not have. It enumerates reachability
    from `protocol` with an independent reader rather than asserting that no
    stray production leads with an already-known bus verb.
    """
    model = ReachabilityModel(ctx.ebnf_text)
    if "protocol" not in model.rules:
        return False, "protocol production absent"
    missing = [c for c in CLASS_RULES if c not in model.rules]
    if missing:
        return False, "class production absent: %s" % missing

    positions, inline_pos = model.statement_position()
    claims, inline_cls = model.classification()
    problems = []
    if inline_pos or inline_cls:
        problems.append("inline alternative(s): %s" % sorted(set(inline_pos + inline_cls)))

    unclassified = sorted(n for n in positions if n not in claims)
    if unclassified:
        problems.append("reachable from protocol but in no class: %s" % unclassified)

    twice = sorted(n for n, c in claims.items() if len(c) > 1)
    if twice:
        problems.append("classified more than once: %s"
                        % [(n, claims[n]) for n in twice])

    orphans = sorted(n for n in claims if n not in positions)
    if orphans:
        problems.append("classified but not reachable at statement position: %s" % orphans)

    if problems:
        return False, "; ".join(problems)
    counts = {c: len(model.alternative_names(c)[0]) for c in CLASS_RULES}
    return True, ("%d productions reachable from protocol, each classified exactly "
                  "once (bus %d, ledger %d, frame %d)"
                  % (len(positions), counts["bus_statement"],
                     counts["ledger_statement"], counts["frame_statement"]))


# -- S1R3-B01 and S1R3-B02: one alternation-aware, per-class bound ---------
#
# `EXACTLY_SIX_BUS_VERBS` counts the alternatives of `bus_statement` and maps
# each to its first token. `FRAME_STATEMENT_BOUNDED` does the same for the two
# frame members. `ledger_statement` had no cardinality or membership bound at
# all. All three questions are asked over a set of member *names*; the grammar's
# real surface is a set of *derivations*. The four checks below ask the question
# over derivations instead, per class, and require exact membership and exact
# cardinality against the declared class set.
#
# Scope. These checks read the leading position only. A verb-like production at
# field or continuation position is outside them by construction, which keeps
# the field-position boundary the specification discloses exactly where it was
# declared. That boundary is exercised as a positive control, not merely
# asserted: see PRESERVED below.

def _class_bound(ctx, cls, declared, label):
    model = ReachabilityModel(ctx.ebnf_text)
    heads, notes = model.class_leading_terminals(cls)
    want = sorted(declared)
    if heads != want:
        extra = sorted(set(heads) - set(want))
        absent = sorted(set(want) - set(heads))
        detail = ("%s leads with %d terminal(s), expected exactly %d: %s"
                  % (cls, len(heads), len(want), heads))
        if extra:
            detail += "; not declared: %s" % extra
        if absent:
            detail += "; declared but absent: %s" % absent
        if notes:
            detail += "; %s" % "; ".join(sorted(set(notes))[:3])
        return False, detail
    members = len(model.alternative_names(cls)[0])
    return True, ("every top-level alternative of all %d %s members leads with "
                  "exactly the %d declared terminals: %s"
                  % (members, label, len(want), " ".join(want)))


def c_bus_class_leading_terminals_bound(ctx):
    """No seventh bus verb is derivable at statement position (S1R3-B01)."""
    return _class_bound(ctx, "bus_statement", BUS_VERBS, "bus")


def c_ledger_class_leading_terminals_bound(ctx):
    """The ledger class is bounded at exactly twelve terminals (S1R3-B02)."""
    return _class_bound(ctx, "ledger_statement", LEDGER_TERMINALS, "ledger")


def c_frame_class_leading_terminals_bound(ctx):
    """The frame class is bounded at exactly two terminals, alternation-aware."""
    return _class_bound(ctx, "frame_statement", FRAME_TERMINALS, "frame")


def c_class_leading_terminals_resolve(ctx):
    """Every leading position in all three classes is decidable.

    Separates "the bound is wrong" from "the bound cannot be computed". A
    recursive, absent, unbalanced or entirely omittable leading position lands
    here with its reason, instead of silently contributing nothing.
    """
    model = ReachabilityModel(ctx.ebnf_text)
    problems = []
    for cls in CLASS_RULES:
        problems.extend(model.class_leading_terminals(cls)[1])
    if problems:
        return False, "undecidable leading position: %s" % sorted(set(problems))
    total = sum(len(model.class_leading_terminals(c)[0]) for c in CLASS_RULES)
    return True, ("all %d class leading terminals resolve to a literal; no "
                  "recursion, absent rule, unbalanced group or omittable head"
                  % total)


def c_spec_class_counts_match_grammar(ctx):
    """Section 3.3's Count column equals the measured class cardinality.

    `SPEC_ENUMERATES_ALL_CLASS_TERMINALS` asserts that the known terminals are
    named in the prose. It cannot see a terminal the prose does not know about,
    so on its own it does not decide prose/grammar drift in either direction.
    """
    marker = "mechanically witnessable over the whole protocol"
    if marker not in ctx.spec:
        return False, "specification does not claim witnessability over the whole protocol"
    region, undecidable = declared_section_region(ctx.spec, marker)
    if undecidable:
        return False, undecidable
    model = ReachabilityModel(ctx.ebnf_text)
    rows, problems = class_rows(region)
    for cls in CLASS_RULES:
        if cls not in rows:
            problems.append("section 3.3 states no Count for %s" % cls)
            continue
        measured = len(model.class_leading_terminals(cls)[0])
        declared = rows[cls][1]
        if declared is None:
            continue  # not a bare integer; `class_rows` already reported it
        if int(declared) != measured:
            problems.append("section 3.3 counts %s at %s, grammar leads with %d"
                            % (cls, declared, measured))
    return (not problems,
            "section 3.3 Count column matches the grammar: bus %s, ledger %s, frame %s"
            % (rows["bus_statement"][1], rows["ledger_statement"][1],
               rows["frame_statement"][1])
            if not problems else "; ".join(problems))


def c_spec_class_terminals_match_grammar(ctx):
    """Section 3.3's Leading terminals cell equals the measured class LEAD set.

    The Count column is decided by `SPEC_CLASS_COUNTS_MATCH_GRAMMAR`; the cell
    beside it was read by nothing (S1R4-B02). `SPEC_ENUMERATES_ALL_CLASS_TERMINALS`
    is a subset test, so it sees a terminal the prose dropped and never one the
    prose added; `BUS_TERMINALS_MATCH_SPEC` reads the sentence at spec-v0.2:99,
    not the table row. A name added to a cell while the grammar and the Count
    stay put therefore passed the whole suite, and three artifacts said it could
    not. This is the smallest general repair: parse each row's cell and require
    set equality with the measured leading terminals of that class, in both
    directions, for all three classes.

    The R5 form of that repair was one input-validation step short in three ways,
    and the R6 form in a fourth. Each is closed here without narrowing the claim.

    * It read its rows with `dict(re.findall(...))`, so a duplicate class row
      shadowed the genuine one (S1R5-B01, Standards axis). `class_rows` collects
      every row and fails closed on a repeated class.
    * It then decided duplication over a pattern's matches rather than over the
      table, so a duplicate row whose Class cell was not backticked, whose Count
      cell was not bare digits, or which carried one to three spaces of GFM-legal
      indentation was not a row at all (S1R6-B01, both axes). `class_rows` parses
      the physical table and reads every row it contains.
    * It reduced the cell with `code_spans`, so a name spelled bare, bold or as a
      Markdown link was not a name (S1R5-B01, Spec axis). `cell_terminal_names`
      normalizes the whole cell.
    * It compared sets where section 3.3 claims multiset equality "in membership
      and in cardinality", so a name listed twice collapsed and the check printed
      a cell cardinality it had not validated — `bus 7` on a PASS line four lines
      from its sibling's `bus 6` (S1R5-B02, both axes). Membership, the cell's own
      cardinality against the measured multiset, and the cell against the adjacent
      numeric Count are now each decided, so the number this check prints is one
      the run has proved equal on all three.
    """
    marker = "mechanically witnessable over the whole protocol"
    if marker not in ctx.spec:
        return False, "specification does not claim witnessability over the whole protocol"
    region, undecidable = declared_section_region(ctx.spec, marker)
    if undecidable:
        return False, undecidable
    model = ReachabilityModel(ctx.ebnf_text)
    rows, problems = class_rows(region)
    sizes = {}
    for cls in CLASS_RULES:
        if cls not in rows:
            problems.append("section 3.3 states no Leading terminals cell for %s" % cls)
            continue
        cell, count_text = rows[cls]
        prose = cell_terminal_names(cell)
        measured = list(model.class_leading_terminals(cls)[0])
        sizes[cls] = len(prose)
        added = sorted(set(prose) - set(measured))
        dropped = sorted(set(measured) - set(prose))
        if added:
            problems.append("section 3.3 names %s for %s; the grammar does not lead with %s"
                            % (added, cls, added))
        if dropped:
            problems.append("the grammar leads %s with %s; section 3.3 does not name %s"
                            % (cls, dropped, dropped))
        over = sorted({n for n in prose if prose.count(n) > measured.count(n)})
        under = sorted({n for n in measured if measured.count(n) > prose.count(n)})
        if not added and over:
            problems.append("section 3.3 lists %s for %s more often than the grammar "
                            "leads with it: cell %d names, grammar %d"
                            % (over, cls, len(prose), len(measured)))
        if not dropped and under:
            problems.append("the grammar leads %s with %s more often than section 3.3 "
                            "lists it: grammar %d, cell %d names"
                            % (cls, under, len(measured), len(prose)))
        if count_text is not None and len(prose) != int(count_text):
            problems.append("section 3.3 lists %d names in the Leading terminals cell "
                            "for %s, beside a Count cell of %s"
                            % (len(prose), cls, count_text))
    return (not problems,
            "section 3.3 Leading terminals cells equal the measured class LEAD multisets "
            "in both directions and their own Count cells: bus %d, ledger %d, frame %d"
            % (sizes["bus_statement"], sizes["ledger_statement"],
               sizes["frame_statement"])
            if not problems else "; ".join(problems))


def c_spec_enumerates_all_class_terminals(ctx):
    """Specification 3.3 must inventory all twenty terminals, not just six."""
    marker = "mechanically witnessable over the whole protocol"
    if marker not in ctx.spec:
        return False, "specification does not claim witnessability over the whole protocol"
    region, undecidable = declared_section_region(ctx.spec, marker)
    if undecidable:
        return False, undecidable
    spans = set(code_spans(region))
    for cls in CLASS_RULES:
        if cls not in spans:
            return False, "section 3.3 does not name the class %s" % cls
    missing = [t for t in BUS_VERBS + LEDGER_TERMINALS + FRAME_TERMINALS
               if t not in spans]
    if missing:
        return False, "section 3.3 does not enumerate %s" % missing
    return True, ("section 3.3 enumerates all %d class terminals (6 bus, 12 ledger, "
                  "2 frame)" % (len(BUS_VERBS) + len(LEDGER_TERMINALS)
                                + len(FRAME_TERMINALS)))


def c_frame_carries_no_authority(ctx):
    """The inertness of a frame statement is stated normatively, not assumed."""
    need_ebnf = ["carries no transport effect, opens no channel, grants no",
                 "authority, and is not a bus verb"]
    missing = [n for n in need_ebnf if n not in ctx.ebnf_text]
    if missing:
        return False, "grammar does not declare the frame statement inert"
    if "close records that decision; it does not make it" not in ctx.ebnf_text:
        return False, "grammar does not separate recording an acceptance from making one"
    return True, ("frame statements are declared inert, and recording an acceptance "
                  "is separated from producing one")


# -- S1R2-B02: lattice values are attributable -----------------------------

def c_lattice_productions_name_an_actor(ctx):
    """Every lattice-bearing production names an Actor.

    The attribution set is derived from the grammar (any rule whose whole
    right-hand side is `actor_ref`), so a newly introduced attribution name is
    honoured and a renamed one does not silently disable the check.
    """
    model = ReachabilityModel(ctx.ebnf_text)
    fields = set(model.attribution_fields())
    if not fields:
        return False, "no rule resolves to actor_ref, so nothing can be attributed"
    bearing = model.lattice_bearing()
    if not bearing:
        return False, "no lattice-bearing production found"
    bad = sorted(n for n in bearing if not set(model.references(n)) & fields)
    if bad:
        return False, ("lattice-bearing production names no Actor: %s"
                       % [(n, bearing[n]) for n in bad])
    return True, ("all %d lattice-bearing productions name an Actor drawn from %s"
                  % (len(bearing), sorted(fields & set(ATTRIBUTION_FIELDS))))


def c_actor_fields_resolve_to_actor_ref(ctx):
    model = ReachabilityModel(ctx.ebnf_text)
    bad = []
    for field in ATTRIBUTION_FIELDS:
        if field not in model.rules:
            bad.append("%s absent" % field)
            continue
        if model._tokens(model.rules[field]) != [("ref", "actor_ref")]:
            bad.append("%s is %r, not actor_ref" % (field, model.rules[field][:32]))
    return (not bad,
            "producer, reviewer, executor and acceptor are all actor_ref, so every "
            "attribution is tagged and resolvable"
            if not bad else "; ".join(bad))


def c_acceptance_binds_subject_digest(ctx):
    """An acceptance names the exact subject it accepts."""
    model = ReachabilityModel(ctx.ebnf_text)
    holders = [n for n in sorted(model.rules)
               if "acceptance_state" in model.references(n) and n != "acceptance_state"]
    if not holders:
        return False, "no production binds acceptance_state"
    bad = [n for n in holders if "subject_digest" not in model.references(n)]
    return (not bad,
            "every acceptance-bearing production (%s) binds subject_digest"
            % ", ".join(holders)
            if not bad else "acceptance recorded against no subject: %s" % bad)


def c_spec_declares_actor_attribution(ctx):
    need = [
        "The \"who may produce it\" column is attributed, not merely asserted",
        "every lattice-bearing production",
        "`acceptor`",
        "`executor`",
    ]
    missing = [n for n in need if n not in ctx.spec]
    if missing:
        return False, "specification does not declare actor attribution: %s" % missing
    if "byte-exact and case-sensitively" not in ctx.spec:
        return False, "specification does not declare case-sensitive lattice comparison"
    return True, ("specification binds the who-column to named attribution fields "
                  "and declares case-sensitive comparison")


def c_refusal_codes_typed_and_open(ctx):
    e = ctx.ebnf
    if not e.has("refusal_code"):
        return False, "refusal_code absent"
    alts = [a.strip() for a in e.alternatives("refusal_code")]
    ok = alts == ["registered_refusal_code", "extension_refusal_code"]
    return ok, ("refusal_code is registered | extension, so the set is typed and open"
                if ok else "refusal_code alternatives are %s" % alts)


def c_scope_violation_registered(ctx):
    terms = ctx.ebnf.terminals("registered_refusal_code")
    present = "SCOPE_VIOLATION" in terms
    return (present,
            "SCOPE_VIOLATION is a registered refusal code"
            if present else "SCOPE_VIOLATION is not registered")


def c_five_control_codes_present(ctx):
    terms = sorted(ctx.ebnf.terminals("registered_refusal_code"))
    missing = [c for c in MUTATION_CODES_V01 + CONTROL_CODES_NEW if c not in terms]
    if missing:
        return False, "registered set is missing %s" % missing
    if len(terms) != 15:
        return False, "registered set has %d codes, expected 15" % len(terms)
    return True, "15 registered codes: 10 carried forward, 5 closing controls 10/12/16/17/18"


def c_extension_form_disjoint(ctx):
    e = ctx.ebnf
    if not e.has("extension_refusal_code"):
        return False, "extension_refusal_code absent"
    terms = e.terminals("extension_refusal_code")
    if "X-" not in terms:
        return False, "extension form does not begin with X-"
    clash = [c for c in e.terminals("registered_refusal_code") if c.startswith("X-")]
    return (not clash,
            "extension form X-<UPPER-WORD> is lexically disjoint from every registered code"
            if not clash else "registered codes collide with the extension form: %s" % clash)


def c_refusal_carries_exact_scope(ctx):
    e = ctx.ebnf
    if "scope_ref" not in e.references("refusal"):
        return False, "refusal does not carry scope_ref"
    terms = e.terminals("scope_ref")
    ok = "s:" in terms and "NO_SCOPE" in terms
    return ok, ("refusal names an exact write-scope entry or NO_SCOPE"
                if ok else "scope_ref terminals are %s" % terms)


def c_unknown_code_fails_closed(ctx):
    need = [
        "fail closed",
        "MUST NOT convert an unknown code into a pass",
        "preserve the code verbatim",
        "MUST NOT emit an extension code where a registered code applies",
    ]
    missing = [n for n in need if n not in ctx.spec]
    return (not missing,
            "unknown codes fail closed, never become a pass, and are preserved verbatim"
            if not missing else "missing forward-compatibility rule: %s" % missing)


def c_milestone_marker(ctx):
    docs = {SPEC: ctx.spec, EBNF: ctx.ebnf_text, TPN: ctx.tpn, LEDGER: ctx.ledger}
    bad = sorted(n for n, t in docs.items()
                 if "M1 established; M2-M5 absent" not in t or "NOT INTEGRATED" not in t)
    return (not bad,
            "all four authored files state M1 established; M2-M5 absent; NOT INTEGRATED"
            if not bad else "milestone marker wrong or absent in %s" % bad)


def c_gate_table_m2_m5_absent(ctx):
    bad = []
    for gate in ("M2", "M3", "M4", "M5"):
        # The final cell only. `\s` is avoided at the line end because it would
        # cross a newline and capture the following row's cell instead.
        row = re.search(r"^\|[ \t]*%s[ \t]*\|[^\n]*\|[ \t]*([^|\n]+?)[ \t]*\|[ \t]*$"
                        % gate, ctx.spec, re.M)
        if not row or row.group(1).strip() != "Absent":
            bad.append(gate)
    if "**Pending" not in ctx.spec:
        bad.append("S1-not-pending")
    return (not bad,
            "gate table shows S1 Pending and M2-M5 Absent"
            if not bad else "gate rows wrong: %s" % bad)


def c_no_self_approval(ctx):
    if "does not self-approve" not in ctx.spec:
        return False, "specification does not disclaim self-approval"
    if "grants no approval" not in ctx.spec:
        return False, "specification does not state that it grants no approval"
    forbidden = [p for p in ("S1 is established", "S1 APPROVED", "Standards `APPROVE`")
                 if p in ctx.spec]
    return (not forbidden,
            "the author grants no approval and claims no S1 establishment"
            if not forbidden else "self-approving text present: %s" % forbidden)


def c_r1_verdict_not_superseded(ctx):
    need = ["84bc1e0be2cb59837e1e0a78219a7cb4c0f150eae226db0f5a135ec2c4f6f6b2",
            "f44ecac23f0c9d180f1bcdcbacb920163a89dc0d04cebbbe6825dbc556c32832",
            "is NOT superseded"]
    missing = [n for n in need if n not in ctx.spec]
    return (not missing,
            "the R1 verdict is cited by digest and explicitly not superseded"
            if not missing else "missing %s" % missing)


def c_copyright_gate_preserved(ctx):
    need = [
        "Matt Pocock",
        "84fdeffd12f2ee307994d1eb6feb48173b6e0502",
        "MIT",
        "none claimed or implied",
        "uninstalled and unpublished",
    ]
    missing = [n for n in need if n not in ctx.tpn]
    if missing:
        return False, "third-party ledger missing %s" % missing
    rows = re.findall(r"^\|\s*(\d+)\s*\|", ctx.tpn, re.M)
    if rows != [str(i) for i in range(1, 19)]:
        return False, "upstream source table does not enumerate exactly 18 rows"
    return True, "author, MIT, pinned commit, 18 sources, no endorsement, unpublished"


def c_candidate_inputs_declared(ctx):
    missing = [f for f in CANDIDATE_INPUTS if f not in ctx.spec]
    if missing:
        return False, "section 9.6 does not name %s" % missing
    if "candidate inputs, not accepted design" not in ctx.spec:
        return False, "section 9.6 does not state that the inputs are unaccepted"
    return True, "all five candidate inputs declared and marked unaccepted"


def c_trusted_copies_byte_exact(ctx):
    bad = []
    for name in sorted(TRUSTED_COPIES):
        size, digest = TRUSTED_COPIES[name]
        path = os.path.join(ctx.root, name)
        if not os.path.isfile(path):
            bad.append("%s absent" % name)
            continue
        if os.path.getsize(path) != size or sha256_of(path) != digest:
            bad.append("%s drifted" % name)
    return (not bad,
            "all %d copied trusted inputs reproduce byte-exact" % len(TRUSTED_COPIES)
            if not bad else "; ".join(bad))


def c_spec_hashes_match_copies(ctx):
    bad = []
    for name in CANDIDATE_INPUTS:
        digest = TRUSTED_COPIES[name][1]
        if digest not in ctx.spec:
            bad.append(name)
    return (not bad,
            "section 14.3 declares the correct digest for all five candidate inputs"
            if not bad else "declared digest wrong or absent for %s" % bad)


CHECKNAME_RE = re.compile(r"`([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)`")
FIXTUREID_RE = re.compile(r"`(NEG-[A-Za-z0-9-]+)`")
CONTROLID_RE = re.compile(r"`(POS-[A-Za-z0-9-]+)`")
LINEREF_RE = re.compile(r"`(spec|ebnf)-v0\.2:([0-9]+)(?:-([0-9]+))?`")

# S1R16-B01. `LINEREF_RE` above is the *only* admissible way to write a line
# citation. Until R17 it was also the only thing that decided what a citation
# *was*, and those are not the same job: the ledger carried five comma-list
# spans (`ebnf-v0.2:207,178,188,194,197,200` and four more) that read as
# citations to a person and as nothing at all to the machine, so twenty line
# numbers were never range-checked, sixteen of them were wrong, and an
# out-of-range member passed the whole suite at exit 0 while
# LEDGER_LINE_REFS_IN_RANGE reported that all citations fell inside the cited
# files.
#
# The repair is not to teach the reader a second dialect — that closes the
# instance and leaves the class open, and a comma list can never be bound by
# LEDGER_ANCHORS, which keys on one citation string and one line. The five spans
# were normalised into the canonical syntax instead, and this pattern makes that
# normalisation an invariant rather than a state: anything that opens like a
# citation and is not wholly matched by LINEREF_RE now fails closed with its own
# reason. A pattern no longer decides what a citation is; the canonical syntax
# does, and the pattern only has to recognise it.
#
# The digit lookahead is what keeps this off the prose. The ledger describes its
# own citation syntax at ledger:27 and elsewhere using `spec-v0.2:<line>` and a
# bare `spec-v0.2:`; neither carries a digit after the colon, so neither is a
# citation and neither needs an exemption.
CITATION_SPAN_RE = re.compile(r"`((?:spec|ebnf)-v0\.2:[0-9][^`]*)`")


def c_ledger_citations_canonical(ctx):
    """Every citation-shaped ledger span must be in the one canonical syntax."""
    spans = CITATION_SPAN_RE.findall(ctx.ledger)
    bad = sorted({s for s in spans if not LINEREF_RE.fullmatch("`%s`" % s)})
    return (not bad,
            "all %d line citations are written in the canonical "
            "`<doc>-v0.2:<line>` or `<doc>-v0.2:<from>-<to>` form, so every one "
            "of them is read by LEDGER_LINE_REFS_IN_RANGE" % len(spans)
            if not bad else
            "citation spans no check reads, because they are not in the "
            "canonical form: %s" % ", ".join("`%s`" % s for s in bad))


# One `**Verification.**` paragraph per repaired blocker group. This is still a
# hardcoded count and is still disclosed as one (`S1R4-N07` family): it is a
# claim about the ledger's shape that the ledger cannot make about itself. What
# R11 changes is that it is now a named constant edited in the same commit as the
# ledger sections it counts, rather than a literal buried in a message — the
# coupling `S1R10-B02` identified as the reason the R9 round skipped the ledger
# entry for `S1R8-B01` entirely.
LEDGER_VERIFICATION_PARAGRAPHS = 27


def _verification_paragraphs(ledger):
    return [p for p in ledger.split("\n") if p.startswith("**Verification.**")]


def c_ledger_names_resolve(ctx):
    """Every check and fixture the change ledger names must actually exist."""
    paras = _verification_paragraphs(ctx.ledger)
    if len(paras) != LEDGER_VERIFICATION_PARAGRAPHS:
        return False, ("expected one Verification paragraph per repaired blocker "
                       "group (6 from R1 + 2 from R2 + 2 from R3 + 2 from R4 + 1 "
                       "covering the R5 pair, whose two axes number three "
                       "mechanisms in the same two checks and which is repaired "
                       "as one group, + 1 for the R6 blocker `S1R6-B01`, which "
                       "both R6 axes reported as one identifier, + 1 covering the "
                       "R7 pair `S1R7-B01` and `S1R7-B02`, which both R7 axes "
                       "reported and the coordinator independently reproduced, "
                       "and which are two halves of one reader repaired as one "
                       "group, + 1 for the R8 blocker `S1R8-B01`, repaired at R9 "
                       "and recorded late at R11, + 3 for the R10 groups: the "
                       "Standards `S1R10-B01`, the Spec `S1R10-B01`, which the "
                       "two axes number alike and which are distinct defects in "
                       "different files, and `S1R10-B02`, + 2 for the R12 groups: "
                       "the Standards `S1R12-B01` and the Spec `S1R12-B01`, which "
                       "the two axes again number alike and which are again "
                       "distinct defects in different files, + 4 for the R14 "
                       "groups: the Standards `S1R14-B01`, and the Spec "
                       "`S1R14-B01`, `S1R14-B02` and `S1R14-B03`, the first pair "
                       "numbered alike across the two axes for the third time "
                       "and again distinct defects in different files, + 2 for "
                       "the R17 groups: the Spec `S1R16-B01` and the Spec "
                       "`S1R16-B02`, which are the first pair this bundle has "
                       "had to repair from a review round whose other axis "
                       "returned `APPROVE`), found %d"
                       % len(paras))
    known_fixtures = {f[1] for f in FIXTURES}
    known_controls = {p[1] for p in PRESERVED}
    bad = []
    for para in paras:
        for name in CHECKNAME_RE.findall(para):
            if name not in CHECK_BY_NAME:
                bad.append("check %s" % name)
    for fid in FIXTUREID_RE.findall(ctx.ledger):
        if fid not in known_fixtures:
            bad.append("fixture %s" % fid)
    for cid in CONTROLID_RE.findall(ctx.ledger):
        if cid not in known_controls:
            bad.append("control %s" % cid)
    return (not bad,
            "every check, fixture and control named in the change ledger exists "
            "in this validator"
            if not bad else "ledger names that do not resolve: %s" % sorted(set(bad)))


def c_ledger_line_refs_in_range(ctx):
    """Every `spec-v0.2:N` / `ebnf-v0.2:N` citation must fall inside the file."""
    bounds = {"spec": len(ctx.spec.splitlines()),
              "ebnf": len(ctx.ebnf_text.splitlines())}
    bad = []
    for doc, start, end in LINEREF_RE.findall(ctx.ledger):
        limit = bounds[doc]
        for value in (start, end):
            if value and not (1 <= int(value) <= limit):
                bad.append("%s-v0.2:%s exceeds %d lines" % (doc, value, limit))
    refs = LINEREF_RE.findall(ctx.ledger)
    return (not bad,
            "all %d line citations fall inside the cited files (spec %d lines, ebnf %d lines)"
            % (len(refs), bounds["spec"], bounds["ebnf"])
            if not bad else "; ".join(sorted(set(bad))))


# Load-bearing ledger citations, each with text the cited line MUST contain.
# LEDGER_LINE_REFS_IN_RANGE proves only that a number is inside the file, which
# is why the R2 review's S1R2-N01 and S1R2-N02 passed it. This table is the
# content half. It is author-selected and therefore partial: it closes the
# defect class for the citations that carry the argument, not for all of them.
# (citation exactly as written in the ledger, the line it claims, expected text)
LEDGER_ANCHORS = [
    ("spec-v0.2:103", "spec", 103, "mechanically witnessable over the whole protocol"),
    # The declared-scope clause of the same sentence (S1R12-B01, Standards axis).
    # `declared_section_region` resolves the §3.3 region from this text, so it is
    # load-bearing in the strongest sense the bundle has: delete it and all three
    # §3.3 checks fail closed. It is anchored here for the same reason the rest
    # of this table exists — a citation that resolves to a line number proves
    # nothing about what that line says.
    ("spec-v0.2:103", "spec", 103, "the scope of that witness is §3.3"),
    ("spec-v0.2:232", "spec", 232, "revokes role assignments"),
    ("spec-v0.2:287", "spec", 287, "All result fields declare type"),
    ("spec-v0.2:293-299", "spec", 293, "Who may produce it"),
    ("spec-v0.2:293-299", "spec", 299, "only a separate Acceptance Decision"),
    ("spec-v0.2:322", "spec", 322, "entitlement"),
    ("spec-v0.2:335", "spec", 335, "exact subject and fixed point"),
    ("spec-v0.2:463", "spec", 463, "(S1R2-B01)"),
    ("spec-v0.2:464", "spec", 464, "(S1R2-B02)"),
    ("ebnf-v0.2:122", "ebnf", 122, "statement       = bus_statement | ledger_statement"),
    ("ebnf-v0.2:168", "ebnf", 168, "frame_statement = mission | close"),
    ("ebnf-v0.2:261-262", "ebnf", 261, 'advisory        = "advisory"'),
    ("ebnf-v0.2:261-262", "ebnf", 262, "producer , evidence_state"),
    ("ebnf-v0.2:277-278", "ebnf", 277, 'close           = "close"'),
    ("ebnf-v0.2:434-435", "ebnf", 434, "executor                = actor_ref"),
    ("ebnf-v0.2:434-435", "ebnf", 435, "acceptor                = actor_ref"),
    # S1R3-B01 / S1R3-B02. The claims this revision rests on are anchored to
    # content on the same terms as its predecessors' claims.
    ("ebnf-v0.2:74-99", "ebnf", 74, "NORMATIVE (S1R3-B01, S1R3-B02)"),
    ("ebnf-v0.2:74-99", "ebnf", 99, "never read as satisfied"),
    ("ebnf-v0.2:101-106", "ebnf", 101, "these bounds read the leading position"),
    ("spec-v0.2:108", "spec", 108, "`ledger_statement` | `assign`"),
    ("spec-v0.2:115", "spec", 115, "bounds over *derivations*"),
    ("spec-v0.2:117-127", "spec", 117, "Seven defects are detected mechanically"),
    ("spec-v0.2:117-127", "spec", 127, "SPEC_CLASS_COUNTS_MATCH_GRAMMAR"),
    ("spec-v0.2:465", "spec", 465, "(S1R3-B01)"),
    ("spec-v0.2:466", "spec", 466, "(S1R3-B02)"),
    # S1R16-B01, the content half of the R17 repair. The five normalised spans
    # were exactly the rows the R16 Spec review identified as "the citations that
    # carry the argument" — evidence rows stating *why* a change was made — and
    # they were exempt from this table only because a comma list cannot be keyed
    # by it. Now that each number is its own citation string, the criterion
    # `spec-v0.2:571` already states for this table applies to them, so they are
    # anchored. Nineteen rows: the twentieth corrected number is
    # `ebnf-v0.2:168`, already anchored above and not duplicated here.
    #
    # This is what makes the correction of twenty-one wrong number instances a
    # measurement rather than an assertion. Every row below fails closed if the
    # author guessed, if the cited production later moves, or if the cited line
    # stops saying what the ledger claims about it.
    ("ebnf-v0.2:207", "ebnf", 207, 'lane            = "lane"'),
    ("ebnf-v0.2:224", "ebnf", 224, 'capsule         = "capsule"'),
    ("ebnf-v0.2:234", "ebnf", 234, 'fence           = "fence"'),
    ("ebnf-v0.2:240", "ebnf", 240, 'artifact_revision     = "artifact"'),
    ("ebnf-v0.2:243", "ebnf", 243, 'transition_definition = "transition"'),
    ("ebnf-v0.2:246", "ebnf", 246, 'transition_receipt    = "receipt"'),
    ("ebnf-v0.2:312", "ebnf", 312, 'freshness_state = "FRESH"'),
    ("ebnf-v0.2:313", "ebnf", 313, 'execution_result= "RECEIPT_WRITTEN"'),
    ("ebnf-v0.2:319", "ebnf", 319, 'replay_class    = "BYTE_EXACT"'),
    # The three added fields of S1R2-B01/S1R2-B02, each anchored on the line that
    # carries the added field rather than on the production's opening line.
    ("ebnf-v0.2:262", "ebnf", 262, "producer , evidence_state ;"),
    ("ebnf-v0.2:247", "ebnf", 247, "evidence_refs , executor ,"),
    ("ebnf-v0.2:277", "ebnf", 277, 'close           = "close" , mission_id , acceptor'),
    ("ebnf-v0.2:434", "ebnf", 434, "executor                = actor_ref ;"),
    ("ebnf-v0.2:435", "ebnf", 435, "acceptor                = actor_ref ;"),
    # The five annotated negative controls of section 11.
    ("spec-v0.2:448", "spec", 448, "10. ambiguous recipient"),
    ("spec-v0.2:450", "spec", 450, "12. writer reviews itself"),
    ("spec-v0.2:454", "spec", 454, "16. unregistered actor"),
    ("spec-v0.2:455", "spec", 455, "17. mutation outside"),
    ("spec-v0.2:456", "spec", 456, "18. phase exits zero"),
]


def c_ledger_anchors_resolve(ctx):
    """Load-bearing citations must resolve to a line that says what is claimed."""
    lines = {"spec": ctx.spec.split("\n"), "ebnf": ctx.ebnf_text.split("\n")}
    bad = []
    for citation, doc, index, expected in LEDGER_ANCHORS:
        if "`%s`" % citation not in ctx.ledger:
            bad.append("%s is no longer cited by the ledger" % citation)
            continue
        if not (1 <= index <= len(lines[doc])):
            bad.append("%s:%d is out of range" % (doc, index))
            continue
        if expected not in lines[doc][index - 1]:
            bad.append("%s line %d does not contain %r"
                       % (citation, index, expected[:34]))
    return (not bad,
            "all %d load-bearing citations resolve to a line containing the "
            "claimed text" % len(LEDGER_ANCHORS)
            if not bad else "; ".join(bad))


def c_no_debris(ctx):
    entries = sorted(os.listdir(ctx.root))
    dirs = [e for e in entries if os.path.isdir(os.path.join(ctx.root, e))]
    hidden = [e for e in entries if e.startswith(".")]
    junk = [e for e in entries
            if e.endswith((".pyc", ".pyo", ".stackdump", ".tmp", ".bak"))
            or e == "__pycache__"]
    unexpected = [e for e in entries if e not in ALLOWED_FILES]
    missing = [e for e in REQUIRED_FILES if e not in entries]
    problems = []
    if dirs:
        problems.append("subdirectories %s" % dirs)
    if hidden:
        problems.append("hidden entries %s" % hidden)
    if junk:
        problems.append("cache or crash debris %s" % junk)
    if unexpected:
        problems.append("unexpected files %s" % unexpected)
    if missing:
        problems.append("missing required files %s" % missing)
    return (not problems,
            "%d required files present, no subdirectory, no hidden entry, no debris"
            % len(REQUIRED_FILES)
            if not problems else "; ".join(problems))


CHECKS = [
    # S1-B1
    ("S1-B1", "CAPSULE_FIELDS_PRESENT", c_capsule_fields_present),
    ("S1-B1", "CAPSULE_CARDINALITY_TABLE", c_capsule_cardinality_table),
    ("S1-B1", "EBNF_CAPSULE_NON_GOALS_NONEMPTY", c_ebnf_capsule_non_goals_nonempty),
    ("S1-B1", "EBNF_CAPSULE_DEPENDENCIES_TYPED", c_ebnf_capsule_dependencies_typed),
    ("S1-B1", "CAPSULE_STATEMENT_BINDS_BOTH", c_capsule_statement_binds_both),
    # S1-B2
    ("S1-B2", "NO_ADVISORY_ORACLE", c_no_advisory_oracle),
    ("S1-B2", "NO_ORACLE_IDENTIFIER", c_no_oracle_identifier),
    ("S1-B2", "EBNF_ROLE_HAS_NO_ORACLE", c_ebnf_role_has_no_oracle),
    ("S1-B2", "PER_SYSTEM_ADAPTERS_DECLARED", c_per_system_adapters_declared),
    # S1-B3
    ("S1-B3", "NO_RESULT_STATE_UNION", c_no_result_state_union),
    ("S1-B3", "LATTICES_PAIRWISE_DISJOINT", c_lattices_pairwise_disjoint),
    ("S1-B3", "ADVISORY_BINDS_EVIDENCE_STATE_ONLY", c_advisory_binds_evidence_state_only),
    ("S1-B3", "EVIDENCE_BINDS_EVIDENCE_STATE_ONLY", c_evidence_binds_evidence_state_only),
    ("S1-B3", "VERDICT_BINDS_VERDICT_STATE_ONLY", c_verdict_binds_verdict_state_only),
    ("S1-B3", "SPEC_DECLARES_LATTICE_SEPARATION", c_spec_declares_lattice_separation),
    # S1-B4
    ("S1-B4", "IDENTIFIER_BODY_EXCLUDES_COLON", c_identifier_body_excludes_colon),
    ("S1-B4", "TYPED_ID_PREFIXES_DISJOINT", c_typed_id_prefixes_disjoint),
    ("S1-B4", "RECIPIENT_UNAMBIGUOUS", c_recipient_unambiguous),
    ("S1-B4", "REQUIRED_ENTITIES_TYPED_AND_REFERENCED", c_required_entities_typed_and_referenced),
    # S1-B5
    ("S1-B5", "EXACTLY_SIX_BUS_VERBS", c_exactly_six_bus_verbs),
    ("S1-B5", "INBOX_PRESENT_IN_GRAMMAR", c_inbox_present_in_grammar),
    ("S1-B5", "LEDGER_TERMINALS_DISJOINT_FROM_BUS", c_ledger_terminals_disjoint_from_bus),
    ("S1-B5", "STATEMENT_PARTITION_EXHAUSTIVE", c_statement_partition_exhaustive),
    ("S1-B5", "NO_SEVENTH_VERB_PRODUCTION", c_no_seventh_verb_production),
    ("S1-B5", "BUS_TERMINALS_MATCH_SPEC", c_bus_terminals_match_spec),
    # S1-B6
    ("S1-B6", "REFUSAL_CODES_TYPED_AND_OPEN", c_refusal_codes_typed_and_open),
    ("S1-B6", "SCOPE_VIOLATION_REGISTERED", c_scope_violation_registered),
    ("S1-B6", "FIVE_CONTROL_CODES_PRESENT", c_five_control_codes_present),
    ("S1-B6", "EXTENSION_FORM_DISJOINT", c_extension_form_disjoint),
    ("S1-B6", "REFUSAL_CARRIES_EXACT_SCOPE", c_refusal_carries_exact_scope),
    ("S1-B6", "UNKNOWN_CODE_FAILS_CLOSED", c_unknown_code_fails_closed),
    # S1R2-B01
    ("R2-B1", "FRAME_STATEMENT_BOUNDED", c_frame_statement_bounded),
    ("R2-B1", "PROTOCOL_PARTITION_EXHAUSTIVE", c_protocol_partition_exhaustive),
    ("R2-B1", "SPEC_ENUMERATES_ALL_CLASS_TERMINALS", c_spec_enumerates_all_class_terminals),
    ("R2-B1", "FRAME_CARRIES_NO_AUTHORITY", c_frame_carries_no_authority),
    # S1R3-B01 and S1R3-B02
    ("R3-B1", "BUS_CLASS_LEADING_TERMINALS_BOUND", c_bus_class_leading_terminals_bound),
    ("R3-B1", "FRAME_CLASS_LEADING_TERMINALS_BOUND", c_frame_class_leading_terminals_bound),
    ("R3-B1", "CLASS_LEADING_TERMINALS_RESOLVE", c_class_leading_terminals_resolve),
    ("R3-B2", "LEDGER_CLASS_LEADING_TERMINALS_BOUND", c_ledger_class_leading_terminals_bound),
    ("R3-B2", "SPEC_CLASS_COUNTS_MATCH_GRAMMAR", c_spec_class_counts_match_grammar),
    ("R3-B2", "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR", c_spec_class_terminals_match_grammar),
    # S1R2-B02
    ("R2-B2", "LATTICE_PRODUCTIONS_NAME_AN_ACTOR", c_lattice_productions_name_an_actor),
    ("R2-B2", "ACTOR_FIELDS_RESOLVE_TO_ACTOR_REF", c_actor_fields_resolve_to_actor_ref),
    ("R2-B2", "ACCEPTANCE_BINDS_SUBJECT_DIGEST", c_acceptance_binds_subject_digest),
    ("R2-B2", "SPEC_DECLARES_ACTOR_ATTRIBUTION", c_spec_declares_actor_attribution),
    # Bundle gates
    ("BUNDLE", "MILESTONE_MARKER", c_milestone_marker),
    ("BUNDLE", "GATE_TABLE_M2_M5_ABSENT", c_gate_table_m2_m5_absent),
    ("BUNDLE", "NO_SELF_APPROVAL", c_no_self_approval),
    ("BUNDLE", "R1_VERDICT_NOT_SUPERSEDED", c_r1_verdict_not_superseded),
    ("BUNDLE", "COPYRIGHT_GATE_PRESERVED", c_copyright_gate_preserved),
    ("BUNDLE", "CANDIDATE_INPUTS_DECLARED", c_candidate_inputs_declared),
    ("BUNDLE", "TRUSTED_COPIES_BYTE_EXACT", c_trusted_copies_byte_exact),
    ("BUNDLE", "SPEC_HASHES_MATCH_COPIES", c_spec_hashes_match_copies),
    ("BUNDLE", "LEDGER_NAMES_RESOLVE", c_ledger_names_resolve),
    ("BUNDLE", "LEDGER_CITATIONS_CANONICAL", c_ledger_citations_canonical),
    ("BUNDLE", "LEDGER_LINE_REFS_IN_RANGE", c_ledger_line_refs_in_range),
    ("BUNDLE", "LEDGER_ANCHORS_RESOLVE", c_ledger_anchors_resolve),
    ("BUNDLE", "NO_DEBRIS", c_no_debris),
]

CHECK_BY_NAME = {name: fn for _, name, fn in CHECKS}

# --------------------------------------------------------------------------
# Negative fixtures. Each mutation must flip its target check to FAIL.
# --------------------------------------------------------------------------

def m_spec(old, new):
    def apply(ctx):
        return ctx.copy_with(spec=ctx.spec.replace(old, new, 1))
    return apply


def m_spec_many(*pairs):
    """Several exact one-shot specification edits applied as one mutation."""
    def apply(ctx):
        text = ctx.spec
        for old, new in pairs:
            text = text.replace(old, new, 1)
        return ctx.copy_with(spec=text)
    return apply


# The section 3.3 frame row and the blank line behind it. A duplicate-row
# fixture consumes that blank line, so the specification stays at 579 lines and
# no `LEDGER_ANCHORS` entry and no line citation drifts. Anchoring on the frame
# row rather than on the separator also leaves the separator in place, so the
# mutated table is still a table and no earlier fixture becomes inapplicable.
FRAME_ROW_TAIL = ("| open or end the Mission Room frame; no transport effect, "
                  "no channel, no authority |\n\n")


def m_dup_class_row(row):
    """Insert `row` as a fourth body row of the section 3.3 class table.

    `row` carries the Class, Leading terminals and Count cells under test; the
    description cell is supplied here so every fixture in the family differs
    only in the cells that decide whether the row is read at all.
    """
    full = row + " utter on the non-privileged collaboration bus |"

    def apply(ctx):
        return ctx.copy_with(
            spec=ctx.spec.replace(FRAME_ROW_TAIL,
                                  FRAME_ROW_TAIL[:-1] + full + "\n", 1))
    return apply


CLASS_TABLE_HEADER = "| Class | Leading terminals | Count | What it may do |"


def m_class_table_pipeless(ctx):
    """Rewrite the section 3.3 class table in valid pipe-less GFM.

    GFM makes a row's leading and trailing pipes optional, and the two spellings
    render to byte-identical HTML, so this edit changes nothing a reader sees and
    the check must still pass. Under R7 it did not: the row predicate required a
    leading pipe, so the truthful table was rejected as "section 3.3 carries no
    class table" — a false red on a legitimate edit, and the same missing half of
    the GFM row grammar as S1R7-B01 with the sign reversed (S1R7-F01).
    """
    lines = ctx.spec.split("\n")
    if CLASS_TABLE_HEADER not in lines:
        return ctx
    index = lines.index(CLASS_TABLE_HEADER)
    while index < len(lines) and lines[index].startswith("|"):
        lines[index] = lines[index].strip()[1:-1].strip()
        index += 1
    return ctx.copy_with(spec="\n".join(lines))


# -- S1R8-B01 fixture material ---------------------------------------------
#
# Every member of the fence family inserts or moves the *same* two things, so
# the only variable across the family is the Markdown block context the reader
# has to get right. `SEVENTH_VERB_TABLE` is a second, rendered class table
# declaring a seventh `bus_statement` leading terminal; `FENCED_EXAMPLE` is a
# legitimate fenced example whose body happens to start with `#`.

SEVENTH_VERB_TABLE = (
    "| Class | Leading terminals | Count | What it may do |\n"
    "|---|---|---:|---|\n"
    "| `bus_statement` | `register`, `send`, `inbox`, `ack`, `heartbeat`, "
    "`handoff`, `promote` | 7 | utter on the non-privileged collaboration bus |")

# The last sentence of section 3.3 before the declared-scope paragraph. Text
# appended here is inside section 3.3 and past the class table.
SECTION_33_TAIL = "An undecidable bound fails closed and is never read as satisfied."

# The paragraph that ends section 3.3 and the heading that opens section 4.
SECTION_4_HEAD = ("## 4. Vendor-neutral task capsule\n\n"
                  "Every lane SHALL consume a capsule equivalent to:")

# The seam between the marker paragraph and the genuine class table.
BEFORE_CLASS_TABLE = "disjoint classes.\n\n| Class |"

FENCED_EXAMPLE = ("```markdown\n"
                  "# Example heading rendered as code, not as a section\n"
                  "```")

TILDE_EXAMPLE = ("~~~markdown\n"
                 "# Example heading rendered as code, not as a section\n"
                 "~~~")

# -- S1R10-B01 fixture material, Standards axis -----------------------------
#
# The R9 reader parsed the section's *end* out of the document and took its
# *start* from a prose sentence. Measured over the R9 bytes, section 3.3 ran
# 9,950 characters from its heading while the region ran 9,547 from the marker,
# so 403 characters and eight lines — `spec-v0.2:96`-`103`, ordinary normative
# prose including the six-verb enumeration itself — were read by none of the
# three section 3.3 checks. One class-table row placed in that prefix declared a
# seventh `bus_statement` leading terminal beside a Count of 7 with all
# fifty-seven positive checks, all eighty-six fixtures and all nine controls
# green at `EXIT=0`. The identical row two lines later, across the marker, was
# caught immediately, so placement alone decided detection.
#
# `SEVENTH_VERB_ROW` is that row: one line, so every fixture in this family is
# line-count preserving and no `LEDGER_ANCHORS` entry and no line citation
# drifts. `SEVENTH_VERB_TABLE` above is the same declaration as a whole table.

SEVENTH_VERB_ROW = (
    "| `bus_statement` | `promote`, `register`, `send`, `inbox`, `ack`, "
    "`heartbeat`, `handoff` | 7 | utter on the non-privileged collaboration bus |")

# The section 3.3 heading, the two seams inside its prose prefix, and the
# section 3.2 seam one heading earlier — the outside of the opening boundary.
SECTION_33_HEAD = "### 3.3 Exactly six non-privileged verbs\n"
SECTION_33_PREFIX_TOP = SECTION_33_HEAD + "\nThe only bus verbs are:"
SECTION_33_PREFIX_END = "explicit policy.\n\nThis control is"
SECTION_32_HEAD = "### 3.2 Collaboration/data plane\n\nAgents MAY"

# -- S1R12-B01 fixture material, Standards axis -----------------------------
#
# The R11 reader opened the region at the ATX heading *nearest* above the
# marker, of any level. A heading placed between §3.3's own heading and the
# marker therefore moved the region's start down to it, and everything above it
# — still §3.3 to every reader of the document — was read by none of the three
# §3.3 checks. Each heading below is legitimate: `#### 3.3.1` is a deeper
# subsection, `### 3.3.1` is the same-level form this specification already uses
# at `### 7.1.1` beneath `### 7.1`, and `### 3.3` restated is the duplicate the
# designator lookup must refuse rather than resolve by position.
#
# All three fixtures place `SEVENTH_VERB_ROW` on the first line of §3.3 and the
# heading on the blank line before the marker, so each consumes two blank lines,
# adds none, and leaves the specification at 579 lines: no `LEDGER_ANCHORS`
# entry and no line citation drifts, and the whole suite can be replayed over
# the mutated bytes with the exit code still meaning something.

SUBSECTION_DEEPER = "#### 3.3.1 Why the bound is stated over derivations"
SUBSECTION_SAME_LEVEL = "### 3.3.1 Why the bound is stated over derivations"
SUBSECTION_DUPLICATE = "### 3.3 Restated in full for the reader in a hurry"

# A genuine sibling section between §3.3 and §4, carrying its own class table.
# `POS-HEAD-c` requires it to stay outside the declared scope.
SECTION_34_HEAD = "### 3.4 Declared scope of the collaboration bus"
SIBLING_SECTION_34 = (SECTION_34_HEAD + "\n\n" + SEVENTH_VERB_TABLE + "\n\n")


def m_prefix_row_then_heading(heading):
    """Rogue class row at the top of §3.3, `heading` just above the marker.

    Two exact one-shot edits, each replacing a blank line, so the mutation is
    line-count preserving and differs from `NEG-R3B2-am` in exactly one thing:
    a heading now stands between the section's heading and the marker sentence.
    """
    return m_spec_many(
        (SECTION_33_PREFIX_TOP,
         SECTION_33_HEAD + SEVENTH_VERB_ROW + "\nThe only bus verbs are:"),
        (SECTION_33_PREFIX_END,
         "explicit policy.\n" + heading + "\nThis control is"))


# -- S1R14-B01 fixture material, Standards axis -----------------------------
#
# The R13 reader parsed the region's *start* from a declared designator and its
# *end* from a level comparison alone, so a heading that is a descendant of §3.3
# by numbering but a sibling by level — `### 3.3.1`, the form this specification
# already uses at `### 7.1.1` beneath `### 7.1` — ended §3.3 inside itself, and
# everything behind it was read by none of the three §3.3 checks.
#
# The two seams below are the blank line above the R2-disclosure paragraph and
# the blank line below it, both inside §3.3 and both past the genuine class
# table. A heading at the first and the rogue row at the second consume one
# blank line each and add none, so the mutation is line-count preserving, the
# specification stays at 579 lines, and no `LEDGER_ANCHORS` entry and no line
# citation drifts. `NEG-R3B2-as` and `POS-HEAD-d` differ in exactly one thing:
# whether the heading's designator is a proper descendant of the declared one.

SECTION_33_TAIL_HEADING_SEAM = "already-known bus verb.\n\n**The R3 candidate"
SECTION_33_TAIL_ROW_SEAM = "and neither did.\n\nBoth are the same failure"


def m_tail_heading_then_row(heading):
    """`heading` inside §3.3's closing prose, with a rogue class row behind it."""
    return m_spec_many(
        (SECTION_33_TAIL_HEADING_SEAM,
         "already-known bus verb.\n" + heading + "\n**The R3 candidate"),
        (SECTION_33_TAIL_ROW_SEAM,
         "and neither did.\n" + SEVENTH_VERB_ROW + "\nBoth are the same failure"))


def m_ebnf(old, new):
    def apply(ctx):
        return ctx.copy_with(ebnf_text=ctx.ebnf_text.replace(old, new, 1))
    return apply


def m_ledger(old, new):
    def apply(ctx):
        return ctx.copy_with(ledger=ctx.ledger.replace(old, new, 1))
    return apply


def m_ledger_all(old, new):
    """Replace every occurrence, so a repeated citation cannot mask the mutation.

    The R4 renumbering made one anchored citation occur twice in the ledger,
    which silently defeated the single-occurrence form of `NEG-L-c`: the check
    still found a surviving copy and reported success. Same failure shape as the
    one `m_tpn_all` exists to avoid.
    """
    def apply(ctx):
        return ctx.copy_with(ledger=ctx.ledger.replace(old, new))
    return apply


def m_ebnf_many(*pairs):
    """Several exact one-shot grammar edits applied as one mutation.

    Used where a mutation must add a production and reach it from an existing
    one while preserving the line count, so that the mutated grammar can also be
    replayed against the full suite without incidental line-number drift.
    """
    def apply(ctx):
        text = ctx.ebnf_text
        for old, new in pairs:
            text = text.replace(old, new, 1)
        return ctx.copy_with(ebnf_text=text)
    return apply


def m_tpn_all(old, new):
    """Replace every occurrence, so a repeated claim cannot mask the mutation."""
    def apply(ctx):
        return ctx.copy_with(tpn=ctx.tpn.replace(old, new))
    return apply


FIXTURES = [
    ("S1-B1", "NEG-B1-a", "capsule schema drops non_goals",
     "CAPSULE_FIELDS_PRESENT",
     m_spec("non_goals: [<explicit excluded outcome or change>]\n", "")),
    ("S1-B1", "NEG-B1-b", "non_goal_set made optional, so an empty list parses",
     "EBNF_CAPSULE_NON_GOALS_NONEMPTY",
     m_ebnf('non_goal_set    = "[" , quoted_text , { "," , quoted_text } , "]" ;',
            'non_goal_set    = "[" , [ quoted_text , { "," , quoted_text } ] , "]" ;')),
    ("S1-B1", "NEG-B1-c", "capsule statement drops the dependency set",
     "CAPSULE_STATEMENT_BINDS_BOTH",
     m_ebnf('                  "dependencies" , dependency_set ;',
            '                  "dependencies" ;')),
    ("S1-B2", "NEG-B2-a", "the AdvisoryOracle seam is reinstated",
     "NO_ADVISORY_ORACLE",
     m_spec("6. `IxEvidenceAdapter`",
            "6. `AdvisoryOracle`: adapt IX, TARS, HARI, metrics, or model analysis. "
            "Also `IxEvidenceAdapter`")),
    ("S1-B2", "NEG-B2-b", "the Oracle role returns to the grammar",
     "EBNF_ROLE_HAS_NO_ORACLE",
     m_ebnf('"Researcher" | "AdvisorySpecialist" ;', '"Researcher" | "Oracle" ;')),
    ("S1-B2", "NEG-B2-c", "a component is named with the ambiguous term",
     "NO_ORACLE_IDENTIFIER",
     m_spec("`AcceptanceGate`, deterministic checks", "`AcceptanceOracle`, deterministic checks")),
    ("S1-B3", "NEG-B3-a", "an advisory statement is allowed to carry a verdict",
     "ADVISORY_BINDS_EVIDENCE_STATE_ONLY",
     m_ebnf('                  producer , evidence_state ;',
            '                  producer , verdict_state ;')),
    ("S1-B3", "NEG-B3-b", "the freshness lattice reuses the evidence token UNKNOWN",
     "LATTICES_PAIRWISE_DISJOINT",
     m_ebnf('"FRESH" | "SUSPECT" | "STALE" | "FRESHNESS_UNKNOWN" ;',
            '"FRESH" | "SUSPECT" | "STALE" | "UNKNOWN" ;')),
    ("S1-B3", "NEG-B3-c", "the merged v0.1 result_state union is restored",
     "NO_RESULT_STATE_UNION",
     m_ebnf('evidence_state  = "PRESENT"',
            'result_state    = "PRESENT" | "ABSENT" | "UNKNOWN" | "CONTRADICTORY"\n'
            '                | "APPROVED" | "REJECTED" | "REQUEST_CHANGES" ;\n'
            'evidence_state  = "PRESENT"')),
    ("S1-B4", "NEG-B4-a", "recipient reverts to the ambiguous untagged union",
     "RECIPIENT_UNAMBIGUOUS",
     m_ebnf('actor_ref       = ( "profile" , profile_id ) | ( "incarnation" , incarnation_id ) ;',
            'actor_ref       = profile_id | incarnation_id ;')),
    ("S1-B4", "NEG-B4-b", "the identifier body readmits ':' so prefixes are forgeable",
     "IDENTIFIER_BODY_EXCLUDES_COLON",
     m_ebnf('identifier_body = letter , { letter | digit | "-" | "_" | "." | "/" } ;',
            'identifier_body = letter , { letter | digit | "-" | "_" | "." | "/" | ":" } ;')),
    ("S1-B4", "NEG-B4-c", "two domain identities share one type prefix",
     "TYPED_ID_PREFIXES_DISJOINT",
     m_ebnf('lane_id                 = "ln:"   , identifier_body ;',
            'lane_id                 = "t:"    , identifier_body ;')),
    ("S1-B5", "NEG-B5-a", "a seventh bus alternative is introduced",
     "EXACTLY_SIX_BUS_VERBS",
     m_ebnf("                | handoff ;\n\nledger_statement",
            "                | handoff\n                | escalation ;\n\n"
            'escalation      = "escalate" , task_id ;\n\nledger_statement')),
    ("S1-B5", "NEG-B5-b", "the inbox verb is deleted, restoring the v0.1 omission",
     "INBOX_PRESENT_IN_GRAMMAR",
     m_ebnf('poll            = "inbox" , incarnation_id , epoch ;',
            'poll            = "readbox" , incarnation_id , epoch ;')),
    ("S1-B5", "NEG-B5-c", "a ledger statement takes a bus verb as its head terminal",
     "LEDGER_TERMINALS_DISJOINT_FROM_BUS",
     m_ebnf('lane            = "lane" , lane_id', 'lane            = "send" , lane_id')),
    ("S1-B6", "NEG-B6-a", "the refusal set is closed again",
     "REFUSAL_CODES_TYPED_AND_OPEN",
     m_ebnf("refusal_code            = registered_refusal_code | extension_refusal_code ;",
            "refusal_code            = registered_refusal_code ;")),
    ("S1-B6", "NEG-B6-b", "the exact write-scope refusal code is removed",
     "SCOPE_VIOLATION_REGISTERED",
     m_ebnf('| "SCOPE_VIOLATION" | "SELF_REVIEW"', '| "SELF_REVIEW"')),
    ("S1-B6", "NEG-B6-c", "the refusal stops naming the exact violated scope",
     "REFUSAL_CARRIES_EXACT_SCOPE",
     m_ebnf('refusal         = "refuse" , refusal_code , subject_digest , scope_ref ;',
            'refusal         = "refuse" , refusal_code , subject_digest ;')),
    ("S1-B6", "NEG-B6-d", "an unknown refusal code is allowed to become a pass",
     "UNKNOWN_CODE_FAILS_CLOSED",
     m_spec("It MUST NOT convert an unknown code into a pass",
            "It MAY convert an unknown code into a pass")),
    # S1R2-B01. The first fixture is the independent reviewer's IM-2 mutation
    # reproduced exactly: it passed the R2 candidate's entire suite.
    ("R2-B1", "NEG-R2B1-a", "a seventh verb is reachable from protocol outside every class",
     "PROTOCOL_PARTITION_EXHAUSTIVE",
     m_ebnf("protocol        = mission , { statement } , close ;",
            "protocol        = mission , { statement } , escalation , close ;\n"
            'escalation      = "escalate" , task_id , profile_id ;')),
    ("R2-B1", "NEG-R2B1-b", "a seventh verb-like production is added at frame level",
     "FRAME_STATEMENT_BOUNDED",
     m_ebnf("frame_statement = mission | close ;",
            "frame_statement = mission | close | escalation ;\n\n"
            'escalation      = "escalate" , task_id , profile_id ;')),
    ("R2-B1", "NEG-R2B1-c", "the frame class is deleted, unclassifying mission and close",
     "PROTOCOL_PARTITION_EXHAUSTIVE",
     m_ebnf("frame_statement = mission | close ;", "")),
    ("R2-B1", "NEG-R2B1-d", "close is classified by two classes at once",
     "PROTOCOL_PARTITION_EXHAUSTIVE",
     m_ebnf("                | verdict\n                | refusal ;",
            "                | verdict\n                | refusal\n                | close ;")),
    ("R2-B1", "NEG-R2B1-e", "the specification stops enumerating the frame class",
     "SPEC_ENUMERATES_ALL_CLASS_TERMINALS",
     m_spec("| `frame_statement` | `mission`, `close` | 2 |",
            "| (unclassified) | (not enumerated) | 2 |")),
    # S1R2-B02. The first fixture is the reviewer's IM-3 mutation reproduced
    # exactly: deleting `producer` from `evidence` left the R2 suite passing.
    ("R2-B2", "NEG-R2B2-a", "producer is deleted from the evidence statement",
     "LATTICE_PRODUCTIONS_NAME_AN_ACTOR",
     m_ebnf('evidence        = "evidence" , evidence_ref , digest , producer , subject_digest , evidence_state ;',
            'evidence        = "evidence" , evidence_ref , digest , subject_digest , evidence_state ;')),
    ("R2-B2", "NEG-R2B2-b", "the advisory artifact records no producer",
     "LATTICE_PRODUCTIONS_NAME_AN_ACTOR",
     m_ebnf("                  producer , evidence_state ;",
            "                  evidence_state ;")),
    ("R2-B2", "NEG-R2B2-c", "the transition receipt records no executor",
     "LATTICE_PRODUCTIONS_NAME_AN_ACTOR",
     m_ebnf("evidence_refs , executor ,\n                        execution_result ;",
            "evidence_refs ,\n                        execution_result ;")),
    ("R2-B2", "NEG-R2B2-d", "the acceptance records no acceptor",
     "LATTICE_PRODUCTIONS_NAME_AN_ACTOR",
     m_ebnf('close           = "close" , mission_id , acceptor , subject_digest ,',
            'close           = "close" , mission_id , subject_digest ,')),
    ("R2-B2", "NEG-R2B2-e", "the acceptance is recorded against no exact subject",
     "ACCEPTANCE_BINDS_SUBJECT_DIGEST",
     m_ebnf('close           = "close" , mission_id , acceptor , subject_digest ,',
            'close           = "close" , mission_id , acceptor ,')),
    ("R2-B2", "NEG-R2B2-f", "an attribution field reverts to an untagged identity",
     "ACTOR_FIELDS_RESOLVE_TO_ACTOR_REF",
     m_ebnf("acceptor                = actor_ref ;",
            "acceptor                = profile_id ;")),
    # S1R3-B01. The first fixture is the R3 independent reviewer's M2b mutation
    # reproduced exactly: it passed the R3 candidate's entire suite, 51 positive
    # checks and 40 negative fixtures. The five that follow it move the hiding
    # place — one group level deeper, behind an omittable element, behind a
    # reference, behind a cycle, behind an entirely omittable alternative — so
    # that the repair is shown to decide the property rather than to relocate
    # the blind spot. Every one of the six is line-count preserving and passes
    # the pre-repair suite in full.
    ("R3-B1", "NEG-R3B1-a", "a seventh bus verb is a top-level alternative inside handoff",
     "BUS_CLASS_LEADING_TERMINALS_BOUND",
     m_ebnf('handoff         = "handoff" , task_id , "from" , incarnation_id , "to" , profile_id , evidence_ref ;',
            'handoff         = ( "handoff" , task_id , "from" , incarnation_id , "to" , profile_id , evidence_ref ) | ( "broadcast" , task_id , profile_id , evidence_ref ) ;')),
    ("R3-B1", "NEG-R3B1-b", "a seventh bus verb is nested two group levels deep in message",
     "BUS_CLASS_LEADING_TERMINALS_BOUND",
     m_ebnf('message         = "send" , envelope ;',
            'message         = ( ( "send" , envelope ) | ( ( "relay" , envelope ) ) ) ;')),
    ("R3-B1", "NEG-R3B1-c", "a seventh bus verb sits behind an omittable leading element",
     "BUS_CLASS_LEADING_TERMINALS_BOUND",
     m_ebnf('acknowledgement = "ack" , message_id , "by" , incarnation_id ;',
            'acknowledgement = [ "ack" , message_id ] , "nack" , message_id , "by" , incarnation_id ;')),
    ("R3-B1", "NEG-R3B1-d", "a seventh bus verb is reached through a referenced production",
     "BUS_CLASS_LEADING_TERMINALS_BOUND",
     m_ebnf('handoff         = "handoff" , task_id , "from" , incarnation_id , "to" , profile_id , evidence_ref ;',
            'handoff         = ( "handoff" , task_id , "from" , incarnation_id , "to" , profile_id , evidence_ref ) | broadcast_form ; broadcast_form = "broadcast" , task_id , profile_id ;')),
    ("R3-B1", "NEG-R3B1-e", "a third frame terminal is a top-level alternative inside mission",
     "FRAME_CLASS_LEADING_TERMINALS_BOUND",
     m_ebnf('mission         = "mission" , mission_id , lineage , epoch , objective ;',
            'mission         = ( "mission" , mission_id , lineage , epoch , objective ) | ( "reopen" , mission_id , epoch ) ;')),
    ("R3-B1", "NEG-R3B1-f", "a bus member's leading position recurses and cannot be decided",
     "CLASS_LEADING_TERMINALS_RESOLVE",
     m_ebnf('handoff         = "handoff" , task_id , "from" , incarnation_id , "to" , profile_id , evidence_ref ;',
            'handoff         = ( "handoff" , task_id , "from" , incarnation_id , "to" , profile_id , evidence_ref ) | handoff_more ; handoff_more = handoff_more , task_id ;')),
    ("R3-B1", "NEG-R3B1-g", "a frame member becomes entirely omittable, so it has no leading terminal",
     "CLASS_LEADING_TERMINALS_RESOLVE",
     m_ebnf('mission         = "mission" , mission_id , lineage , epoch , objective ;',
            'mission         = [ "mission" , mission_id , lineage , epoch , objective ] ;')),
    # The two shapes of "unbalanced group" the grammar names at ebnf-v0.2:95-99,
    # each of which the R4 candidate read as a satisfied bound. `-h` is a stray
    # closer, which also hid the top-level `|` behind it and with it a seventh
    # bus leading terminal; `-i` is an unmatched opener placed *behind* the
    # leading terminal, where the element scan returned before reaching it. Both
    # must fail closed, and neither may be answered by narrowing the claim.
    ("R3-B1", "NEG-R3B1-h", "a stray closing bracket at a bus member's leading position",
     "CLASS_LEADING_TERMINALS_RESOLVE",
     m_ebnf('heartbeat       = "heartbeat" , incarnation_id , epoch , timestamp ;',
            'heartbeat       = "heartbeat" , incarnation_id , epoch , timestamp ) | ( "sweep" , incarnation_id ) ;')),
    ("R3-B1", "NEG-R3B1-i", "an unmatched opening group behind a bus member's leading terminal",
     "CLASS_LEADING_TERMINALS_RESOLVE",
     m_ebnf('registration    = "register" , profile_id , incarnation_id , capability_set ;',
            'registration    = "register" , ( profile_id , incarnation_id , capability_set ;')),
    # S1R3-B02. The first fixture is the R3 reviewer's B02-P2b mutation
    # reproduced exactly: a thirteenth ledger statement binding the
    # authority-bearing acceptance lattice, which passed the entire suite.
    ("R3-B2", "NEG-R3B2-a", "a thirteenth ledger statement is added to the class",
     "LEDGER_CLASS_LEADING_TERMINALS_BOUND",
     m_ebnf_many(
         ("ledger_statement= role_assignment", "ledger_statement= ratify | role_assignment"),
         ('fence           = "fence" , fence_id , lineage , epoch , subject_digest ;',
          'fence           = "fence" , fence_id , lineage , epoch , subject_digest ; ratify = "ratify" , mission_id , witness , subject_digest , acceptance_state ; witness = actor_ref ;'))),
    ("R3-B2", "NEG-R3B2-b", "a thirteenth ledger terminal is a top-level alternative inside fence",
     "LEDGER_CLASS_LEADING_TERMINALS_BOUND",
     m_ebnf('fence           = "fence" , fence_id , lineage , epoch , subject_digest ;',
            'fence           = ( "fence" , fence_id , lineage , epoch , subject_digest ) | ( "deploy" , fence_id , epoch ) ;')),
    ("R3-B2", "NEG-R3B2-c", "a ledger terminal is respelled, so the count holds but membership does not",
     "LEDGER_CLASS_LEADING_TERMINALS_BOUND",
     m_ebnf('claim           = "claim" , task_id , "by" , incarnation_id , "in" , lane_id ,',
            'claim           = "claym" , task_id , "by" , incarnation_id , "in" , lane_id ,')),
    ("R3-B2", "NEG-R3B2-d", "section 3.3 states a ledger count the grammar does not carry",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_spec("`verdict`, `refuse` | 12 |", "`verdict`, `refuse` | 13 |")),
    # Additive prose-superset drift, one per class. Each adds a name to the
    # Leading terminals cell of section 3.3 and touches nothing else: the
    # grammar is byte-identical and the numeric Count cell is byte-identical, so
    # neither `SPEC_CLASS_COUNTS_MATCH_GRAMMAR` nor the three class bounds can
    # see it. All three passed the R4 suite in full.
    ("R3-B2", "NEG-R3B2-e", "section 3.3 names a seventh bus leading terminal the grammar does not carry",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec("`ack`, `heartbeat`, `handoff` | 6 |", "`ack`, `heartbeat`, `handoff`, `broadcast` | 6 |")),
    ("R3-B2", "NEG-R3B2-f", "section 3.3 names a thirteenth ledger leading terminal the grammar does not carry",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec("`advisory`, `verdict`, `refuse` | 12 |", "`advisory`, `verdict`, `ratify`, `refuse` | 12 |")),
    ("R3-B2", "NEG-R3B2-g", "section 3.3 names a third frame leading terminal the grammar does not carry",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec("`mission`, `close` | 2 |", "`mission`, `close`, `convene` | 2 |")),
    # S1R5-B01 and S1R5-B02, one family per group. Every one of the six passed
    # the R5 suite in full at EXIT=0, leaves the grammar byte-identical, and
    # preserves the specification's 579 lines, so all 25 `LEDGER_ANCHORS` and
    # every line citation still resolve and the mutation cannot be answered by
    # anchor drift.
    #
    # `-h` and `-i` are the duplicate-row family: the section 3.3 separator line
    # is replaced by a bogus `bus_statement` row placed *above* the genuine one,
    # declaring a seventh bus leading terminal and a Count of seven. Under
    # `dict(re.findall(...))` the genuine row overwrote it and both checks read
    # only the survivor, so the identical row placed *below* was caught and this
    # one was not — position, not content, decided detection. Both checks built
    # the table that way, so both carry a fixture.
    ("R3-B2", "NEG-R3B2-h", "a duplicate bus row above the genuine one hides a seventh terminal",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec("|---|---|---:|---|",
            "| `bus_statement` | `register`, `send`, `inbox`, `ack`, `heartbeat`, "
            "`handoff`, `broadcast` | 7 | utter on the non-privileged collaboration bus |")),
    ("R3-B2", "NEG-R3B2-i", "a duplicate bus row above the genuine one hides a Count of seven",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_spec("|---|---|---:|---|",
            "| `bus_statement` | `register`, `send`, `inbox`, `ack`, `heartbeat`, "
            "`handoff`, `broadcast` | 7 | utter on the non-privileged collaboration bus |")),
    # `-j`, `-k`, `-l` are the non-code-span family: the same seventh bus
    # terminal in the three renderings a Markdown table admits beside a code
    # span. `code_spans` saw none of them. One fixture per rendering, so a
    # future normalizer cannot regress one spelling while keeping another.
    ("R3-B2", "NEG-R3B2-j", "a seventh bus terminal is added to the cell as a bare word",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec("| `bus_statement` | `register`, `send`,",
            "| `bus_statement` | `register`, broadcast, `send`,")),
    ("R3-B2", "NEG-R3B2-k", "a seventh bus terminal is added to the cell in bold",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec("| `bus_statement` | `register`, `send`,",
            "| `bus_statement` | `register`, **broadcast**, `send`,")),
    ("R3-B2", "NEG-R3B2-l", "a seventh bus terminal is added to the cell as a Markdown link",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec("| `bus_statement` | `register`, `send`,",
            "| `bus_statement` | `register`, [broadcast](#bus), `send`,")),
    # `-m` is the cardinality family: a name the cell lists twice, against a
    # byte-identical Count cell and a byte-identical grammar. Set comparison
    # collapsed it, and the check then printed the cell's seven names on a PASS
    # line whose sibling printed six.
    ("R3-B2", "NEG-R3B2-m", "section 3.3 lists a bus terminal twice beside an unchanged Count",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec("| `bus_statement` | `register`, `send`,",
            "| `bus_statement` | `register`, `register`, `send`,")),
    # S1R6-B01. Every fixture below adds a second `bus_statement` row to the
    # section 3.3 table as a fourth body row, declaring a seventh bus leading
    # terminal. Each consumes the blank line after the frame row, so the
    # specification stays at 579 lines and all 25 `LEDGER_ANCHORS` and every line
    # citation still resolve; the grammar is byte-identical; the separator line
    # and the genuine bus cell are untouched, so no earlier fixture is rendered
    # inapplicable. Every one of them passed the R6 suite in full at EXIT=0 with
    # 57/0 positive, 62/0/0 fixtures and 4/0/0 controls.
    #
    # `-n`, `-o` and `-p` are the Class-cell rendering family: the three
    # spellings a Markdown table admits beside a code span, which is exactly the
    # family `-j`/`-k`/`-l` pin for the Leading terminals cell one column to the
    # right. One fixture per rendering, so a future revision cannot regress one
    # spelling while keeping another.
    ("R3-B2", "NEG-R3B2-n", "a duplicate bus row spells its Class cell bare",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_dup_class_row("| bus_statement | `register`, `send`, `inbox`, `ack`, "
                     "`heartbeat`, `handoff`, `broadcast` | 7 |")),
    ("R3-B2", "NEG-R3B2-o", "a duplicate bus row spells its Class cell in bold",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_dup_class_row("| **bus_statement** | `register`, `send`, `inbox`, `ack`, "
                     "`heartbeat`, `handoff`, `broadcast` | 7 |")),
    ("R3-B2", "NEG-R3B2-p", "a duplicate bus row spells its Class cell as a Markdown link",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_dup_class_row("| [bus_statement](#bus) | `register`, `send`, `inbox`, "
                     "`ack`, `heartbeat`, `handoff`, `broadcast` | 7 |")),
    # `-q` to `-t` are the Count-cell family. The R6 handoff disclosed the
    # non-numeric Count cell as open and both R6 reviews reproduced it; a Count
    # cell inside a normative table that is not a bare integer is a defect, not
    # an absence, so each of these is reported rather than skipped.
    ("R3-B2", "NEG-R3B2-q", "a duplicate bus row states its Count as a word",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_dup_class_row("| `bus_statement` | `register`, `send`, `inbox`, `ack`, "
                     "`heartbeat`, `handoff`, `broadcast` | seven |")),
    ("R3-B2", "NEG-R3B2-r", "a duplicate bus row decorates its Count with markup",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_dup_class_row("| `bus_statement` | `register`, `send`, `inbox`, `ack`, "
                     "`heartbeat`, `handoff`, `broadcast` | 7<sup>a</sup> |")),
    ("R3-B2", "NEG-R3B2-s", "a duplicate bus row bolds a worded Count",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_dup_class_row("| `bus_statement` | `register`, `send`, `inbox`, `ack`, "
                     "`heartbeat`, `handoff`, `broadcast` | **seven** |")),
    ("R3-B2", "NEG-R3B2-t", "a duplicate bus row annotates its Count with a note",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_dup_class_row("| `bus_statement` | `register`, `send`, `inbox`, `ack`, "
                     "`heartbeat`, `handoff`, `broadcast` | 7 (see note) |")),
    # `-u` to `-w` are the indentation family. GFM admits up to three spaces
    # before a table row and renders it identically, so one space decided
    # detection under `re.M` anchoring at column 0. All three indents are pinned
    # because the boundary, not one value inside it, is the property.
    ("R3-B2", "NEG-R3B2-u", "a duplicate bus row carries one space of indentation",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_dup_class_row(" | `bus_statement` | `register`, `send`, `inbox`, `ack`, "
                     "`heartbeat`, `handoff`, `broadcast` | 7 |")),
    ("R3-B2", "NEG-R3B2-v", "a duplicate bus row carries two spaces of indentation",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_dup_class_row("  | `bus_statement` | `register`, `send`, `inbox`, `ack`, "
                     "`heartbeat`, `handoff`, `broadcast` | 7 |")),
    ("R3-B2", "NEG-R3B2-w", "a duplicate bus row carries three spaces of indentation",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_dup_class_row("   | `bus_statement` | `register`, `send`, `inbox`, `ack`, "
                     "`heartbeat`, `handoff`, `broadcast` | 7 |")),
    # `-x` and `-y` extend the family to the other two classes, so the repair is
    # shown to decide the property for the table rather than for the bus row.
    ("R3-B2", "NEG-R3B2-x", "a duplicate ledger row with a bare Class cell hides a thirteenth terminal",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_dup_class_row("| ledger_statement | `assign`, `lane`, `claim`, `capsule`, "
                     "`fence`, `artifact`, `transition`, `receipt`, `evidence`, "
                     "`advisory`, `verdict`, `refuse`, `ratify` | 13 |")),
    ("R3-B2", "NEG-R3B2-y", "a duplicate frame row with a bare Class cell hides a third terminal",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_dup_class_row("| frame_statement | `mission`, `close`, `reopen` | 3 |")),
    # `-z` is the substitution-plus-decoy form the R6 Standards review recorded
    # as `X-2`, which removes the "it is only an added row" reading: the genuine
    # bus row loses its backticks and gains `broadcast` with a Count of seven, so
    # it is the only `bus_statement` row a reader sees in the normative table,
    # while a decoy row carrying the truthful six is parked behind the frame row
    # and absorbs both checks. Under R6 all six of the then-new fixtures reported
    # `caught` on this mutation while the table declared seven bus verbs.
    ("R3-B2", "NEG-R3B2-z", "the only visible bus row declares seven while a decoy carries the six",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec_many(
         ("| `bus_statement` | `register`, `send`, `inbox`, `ack`, `heartbeat`, "
          "`handoff` | 6 | utter on the non-privileged collaboration bus |",
          "| bus_statement | `register`, `send`, `inbox`, `ack`, `heartbeat`, "
          "`handoff`, `broadcast` | 7 | utter on the non-privileged collaboration bus |"),
         (FRAME_ROW_TAIL,
          FRAME_ROW_TAIL[:-1]
          + "| `bus_statement` | `register`, `send`, `inbox`, `ack`, `heartbeat`, "
            "`handoff` | 6 | utter on the non-privileged collaboration bus |\n"))),
    # `-aa` closes the route the physical-table reader opens by construction: a
    # fourth row inside the normative table whose Class cell names no class at
    # all. The pattern-scoped reader could not see it either, and a reader of the
    # rendered table sees a row declaring `broadcast`.
    ("R3-B2", "NEG-R3B2-aa", "a fourth class-table row names no known class",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_dup_class_row("| bus statement | `register`, `send`, `inbox`, `ack`, "
                     "`heartbeat`, `handoff`, `broadcast` | 7 |")),
    # `-ab` is the boundary one step past the three-space bound: four spaces make
    # the line an indented code block rather than a table row, so it is outside
    # the table by construction and is reported as a class row outside it. The
    # bound is pinned on both sides — `-u`/`-v`/`-w` inside it, `-ab` past it —
    # so neither side can be regressed by relaxing the other.
    ("R3-B2", "NEG-R3B2-ab", "a duplicate bus row carries four spaces of indentation",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_dup_class_row("    | `bus_statement` | `register`, `send`, `inbox`, "
                     "`ack`, `heartbeat`, `handoff`, `broadcast` | 7 |")),
    # `-ac` and `-ad` are the edge-pipe family. GFM makes a row's leading and
    # trailing pipes optional and renders both spellings identically, so R7's
    # `^ {0,3}\|` was S1R6-B01 one notch further out: each of these declared a
    # seventh bus leading terminal inside the rendered normative table while all
    # fifty-seven positive checks stayed green and both section 3.3 checks
    # printed `bus 6` (S1R7-B01). Both edges are pinned, so neither can be
    # regressed by relaxing the other, and `POS-PIPELESS-a` pins the same
    # predicate from the legitimate side.
    ("R3-B2", "NEG-R3B2-ac", "a duplicate bus row omits its leading pipe, which GFM makes optional",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_spec(FRAME_ROW_TAIL,
            FRAME_ROW_TAIL[:-1]
            + "`bus_statement` | `register`, `send`, `inbox`, `ack`, "
              "`heartbeat`, `handoff`, `broadcast` | 7 | utter on the "
              "non-privileged collaboration bus |\n")),
    ("R3-B2", "NEG-R3B2-ad", "a duplicate bus row omits both of its edge pipes",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec(FRAME_ROW_TAIL,
            FRAME_ROW_TAIL[:-1]
            + "`bus_statement` | `register`, `send`, `inbox`, `ack`, "
              "`heartbeat`, `handoff`, `broadcast` | 7 | utter on the "
              "non-privileged collaboration bus\n")),
    # `-ae` pins the section boundary rather than a row spelling. R7 read
    # `spec.split(marker, 1)[1][:2500]` and called it section 3.3; the section
    # runs 9,548 characters past that marker, so a whole second class table
    # placed in the last 74% of it declared a seventh bus leading terminal at
    # EXIT=0 with 77 of 77 fixtures caught and 4 of 4 controls clean
    # (S1R7-B02). The anchor sits at spec-v0.2:129, past the old window, so this
    # fixture is uncatchable by any constant that does not reach the real end of
    # the section. It is the one fixture in the family that is not line-count
    # preserving: a table needs three lines and the property under test is the
    # offset, not the count.
    ("R3-B2", "NEG-R3B2-ae", "a second class table sits in section 3.3 past the old 2500-character window",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec("An undecidable bound fails closed and is never read as satisfied.",
            "| Class | Leading terminals | Count | What it may do |\n"
            "|---|---|---:|---|\n"
            "| `bus_statement` | `register`, `send`, `inbox`, `ack`, "
            "`heartbeat`, `handoff`, `broadcast` | 7 | utter on the "
            "non-privileged collaboration bus |")),
    # `-af` .. `-ak` are the fenced-block family. The region reader, then named
    # `section_after_marker`, decided
    # an ATX heading by matching `^ {0,3}#{1,6}` against a line with no Markdown
    # block context, so a `#` line *inside a fenced code block* ended section 3.3
    # where GFM renders no section boundary at all. Everything past that fence
    # was read by nothing, and a second class table placed there declared a
    # seventh bus leading terminal with all fifty-seven positive checks, all
    # eighty fixtures and all five controls green at EXIT=0 (S1R8-B01). The same
    # missing block context produced the opposite sign on a truthful document —
    # see `POS-FENCE-a`/`-b` — so both signs are pinned, and a repair that
    # satisfies one by breaking the other is impossible.
    ("R3-B2", "NEG-R3B2-af", "a backtick-fenced example carrying a `#` line hides a second class table",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec(SECTION_33_TAIL,
            "```text\n"
            "## not a heading: this line is inside a fenced code block\n"
            "```\n\n" + SEVENTH_VERB_TABLE)),
    ("R3-B2", "NEG-R3B2-ag", "a tilde-fenced example carrying a `#` line hides a second class table",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_spec(SECTION_33_TAIL,
            "~~~text\n"
            "## not a heading: this line is inside a tilde-fenced block\n"
            "~~~\n\n" + SEVENTH_VERB_TABLE)),
    # `-ah` pins fence *length* and the three-space indentation bound together. A
    # scanner that closed on any run of the same character would close on the
    # inner three-backtick line, read the `#` line behind it as a heading, and be
    # S1R8-B01 again one notch in. CommonMark closes a fence only on a run of the
    # same character at least as long as the opening one, carrying no info string.
    ("R3-B2", "NEG-R3B2-ah", "a shorter inner fence does not close the outer one, and a `#` line behind it hides a table",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec(SECTION_33_TAIL,
            "   ````text\n"
            "```\n"
            "## not a heading: the inner fence is shorter than the opening one\n"
            "   ````\n\n" + SEVENTH_VERB_TABLE)),
    # `-ai` pins the *enclosing* heading scan, which the same repair makes
    # fence-sensitive for the first time. If every heading above the marker is
    # inside a fence, no heading level is established. Defaulting that to level 6
    # ends the region at the first heading of any level after the marker, so a
    # decoy `#### ` heading truncates section 3.3 and hides everything behind it.
    # The default is therefore level 0, which no ATX heading satisfies: an
    # undecidable enclosing level reads to the end of the document rather than to
    # the nearest excuse to stop.
    ("R3-B2", "NEG-R3B2-ai", "every heading above the marker is fenced and a decoy sub-heading truncates the section",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_spec_many(
         ("# Gaia Consolidated Mission-Room Software Factory Specification v0.2\n",
          "```text\n"
          "# Gaia Consolidated Mission-Room Software Factory Specification v0.2\n"),
         ("This control is **mechanically witnessable over the whole protocol**,",
          "```\n\nThis control is **mechanically witnessable over the whole protocol**,"),
         (SECTION_33_TAIL,
          "#### a decoy sub-heading\n\n" + SEVENTH_VERB_TABLE))),
    ("R3-B2", "NEG-R3B2-ak", "an unclosed fence carrying a `#` line hides a second class table",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec(SECTION_33_TAIL,
            "```text\n"
            "## not a heading: this fence is never closed\n\n" + SEVENTH_VERB_TABLE)),
    # `-aj` is the setext residual, pinned rather than asserted. This reader
    # decides section boundaries by ATX heading only; a setext heading
    # (`Title` over `===` or `---`) is not read as one. That is disclosed, and
    # the *direction* of the resulting error is what this fixture fixes in
    # place: a setext-headed section 4 does not end section 3.3, so the region
    # over-reads into section 4 and a class table placed there is caught. The
    # fail direction is over-read, never under-read, so setext blindness cannot
    # hide a declaration. Its converse — a legitimate class-naming row in a
    # setext-headed section 4 read as a section 3.3 defect, and a class terminal
    # named only there read as enumerated — is a false red and a false green
    # respectively, and is carried as an open residual, not as a claim.
    ("R3-B2", "NEG-R3B2-aj", "a setext-headed section 4 does not end section 3.3, so a table placed there is still read",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_spec(SECTION_4_HEAD,
            "4. Vendor-neutral task capsule\n"
            "---\n\n" + SEVENTH_VERB_TABLE
            + "\n\nEvery lane SHALL consume a capsule equivalent to:")),
    # `-al` .. `-ao` are the opening-boundary family (S1R10-B01, Standards
    # axis). `-af` .. `-ak` above pin where the region *ends*; these pin where it
    # *begins*. `-al` is the R10 reviewer's own witness replayed verbatim: one
    # row, one line changed, none shifted, in the eight-line prose prefix the
    # marker-anchored region could not see. `-am` puts the same row on the first
    # line of the section, the furthest point from the marker inside it. `-an` is
    # the whole-table form, so a repair that reads displaced *rows* but not a
    # displaced *header* cannot satisfy the family. `-ao` pins the direction of
    # the new backward scan when the section's own heading is fenced: the
    # enclosing heading is then section 3.2's, the region opens there, and it
    # over-reads rather than under-reads — the same direction `NEG-R3B2-ai` fixes
    # for an undecidable level. `POS-HEAD-a` is the matching obligation from the
    # other side and is what stops a repair from answering this family by simply
    # reading more of the document.
    ("R3-B2", "NEG-R3B2-al", "a rogue class row sits between the section 3.3 heading and the marker sentence",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_spec(SECTION_33_PREFIX_END,
            "explicit policy.\n" + SEVENTH_VERB_ROW + "\nThis control is")),
    ("R3-B2", "NEG-R3B2-am", "a rogue class row sits on the first line of section 3.3, directly beneath its heading",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec(SECTION_33_PREFIX_TOP,
            SECTION_33_HEAD + SEVENTH_VERB_ROW + "\nThe only bus verbs are:")),
    ("R3-B2", "NEG-R3B2-an", "a complete second class table sits in the prose prefix of section 3.3",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec("The only bus verbs are:\n\n",
            "The only bus verbs are:\n\n" + SEVENTH_VERB_TABLE + "\n\n")),
    ("R3-B2", "NEG-R3B2-ao", "the section 3.3 heading is fenced, so the region opens at section 3.2, where a rogue row sits",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_spec_many(
         (SECTION_33_HEAD, "```text\n" + SECTION_33_HEAD + "```\n"),
         (SECTION_32_HEAD,
          "### 3.2 Collaboration/data plane\n\n" + SEVENTH_VERB_ROW
          + "\n\nAgents MAY"))),
    # `-ap` .. `-ar` are the declared-identity family (S1R12-B01, Standards
    # axis). `-al` .. `-ao` above pin where the region begins *relative to the
    # marker*; these pin *which section* the region is, which proximity cannot
    # decide. Each places the identical rogue row on the first line of §3.3 —
    # the `NEG-R3B2-am` position, still caught there — and adds one legitimate
    # heading between §3.3's heading and the marker. Against the R11 mechanism
    # carrying this fixture set all three report "not caught" while all
    # fifty-seven positive checks stay green, and `NEG-R3B2-an` and
    # `NEG-R3B2-ao` stop being caught alongside them. `-ar` is the hazard the
    # designator lookup introduces rather than one it inherits: two headings for
    # one declared section is refused, never resolved by position, which is the
    # rule `class_rows` already applies to two rows for one class.
    ("R3-B2", "NEG-R3B2-ap", "a deeper subsection heading between the section 3.3 heading and the marker shrinks the region",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_prefix_row_then_heading(SUBSECTION_DEEPER)),
    ("R3-B2", "NEG-R3B2-aq", "a same-level subsection heading in the document's own house style shrinks the region",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_prefix_row_then_heading(SUBSECTION_SAME_LEVEL)),
    ("R3-B2", "NEG-R3B2-ar", "a second heading for the declared section is refused rather than resolved by position",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_prefix_row_then_heading(SUBSECTION_DUPLICATE)),
    # `-as` is the closing-boundary form of the same family (S1R14-B01, Standards
    # axis). `-ap` .. `-ar` pin which section the region *is* and where it
    # *begins*; this one pins where it *ends*. The R13 opening scan already read
    # `### 3.3.1` as a descendant of the declared section rather than as a rival
    # for its identity, while the closing scan still read it as a same-level
    # heading and stopped there — one reader, two incompatible readings of the
    # same document. Against the R14 mechanism this fixture reports "not caught"
    # while all fifty-seven positive checks stay green and both class checks print
    # `bus 6`. `POS-HEAD-d` is the obligation it creates and is the same mutation
    # with the designator changed.
    ("R3-B2", "NEG-R3B2-as", "a same-level descendant subsection heading past the marker ends section 3.3 inside itself",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_tail_heading_then_row(SUBSECTION_SAME_LEVEL)),
    ("BUNDLE", "NEG-M-a", "the milestone marker is advanced",
     "MILESTONE_MARKER",
     m_spec("M1 established; M2-M5 absent", "M2 established; M3-M5 absent")),
    ("BUNDLE", "NEG-M-b", "a gate row is marked complete",
     "GATE_TABLE_M2_M5_ABSENT",
     m_spec("out-of-sample baseline | Absent |", "out-of-sample baseline | Complete |")),
    ("BUNDLE", "NEG-S-a", "the author self-approves",
     "NO_SELF_APPROVAL",
     m_spec("does not self-approve", "self-approves")),
    ("BUNDLE", "NEG-L-a", "the ledger cites a validator check that does not exist",
     "LEDGER_NAMES_RESOLVE",
     m_ledger("`CAPSULE_STATEMENT_BINDS_BOTH`", "`CAPSULE_BINDS_EVERYTHING`")),
    ("BUNDLE", "NEG-L-b", "the ledger cites a line beyond the end of the grammar",
     "LEDGER_LINE_REFS_IN_RANGE",
     m_ledger("`ebnf-v0.2:176`", "`ebnf-v0.2:99999`")),
    ("BUNDLE", "NEG-L-c", "a citation is renumbered to another in-range line",
     "LEDGER_ANCHORS_RESOLVE",
     m_ledger_all("`ebnf-v0.2:122`", "`ebnf-v0.2:121`")),
    ("BUNDLE", "NEG-L-d", "a cited line stops saying what the ledger claims",
     "LEDGER_ANCHORS_RESOLVE",
     m_spec("This control is **mechanically witnessable over the whole protocol**, and the scope",
            "This control is asserted, and the scope")),
    # S1R16-B01. This is the exact evasion the R16 Spec review ran as MUT-1,
    # shipped as a fixture. On the R16 bytes it is a false green at exit 0 with
    # 0 FAILs; here it is refused on the *form* of the span, before the range
    # reader is even reached, which is why the citation carrying line 99,999 of
    # a 503-line grammar can no longer pass unread.
    ("BUNDLE", "NEG-L-e", "a comma-list citation dialect no range check reads is reintroduced",
     "LEDGER_CITATIONS_CANONICAL",
     m_ledger("`ebnf-v0.2:207`", "`ebnf-v0.2:207,178,188,194,197,99999`")),
    ("BUNDLE", "NEG-C-a", "the upstream author attribution is altered",
     "COPYRIGHT_GATE_PRESERVED",
     m_tpn_all("Matt Pocock", "M. Pocok")),
    ("BUNDLE", "NEG-C-b", "the pinned upstream commit is altered",
     "COPYRIGHT_GATE_PRESERVED",
     m_tpn_all("84fdeffd12f2ee307994d1eb6feb48173b6e0502",
               "0000000000000000000000000000000000000000")),
    ("BUNDLE", "NEG-C-c", "one of the eighteen upstream sources is dropped",
     "COPYRIGHT_GATE_PRESERVED",
     m_tpn_all("| 18 | `wizard` | `gaia-wizard` |", "")),
]

# --------------------------------------------------------------------------
# Disclosed-boundary positive controls.
#
# A negative fixture proves a check catches what it claims to catch. It cannot
# prove the check stops where it claims to stop. The specification discloses one
# boundary — the class bounds read the *leading* position, so a verb-like
# production at field or continuation position is outside them, deliberately and
# with the argument stated on both sides. `S1R3-B01` was raised precisely
# because a repair could over-reach into that disclosed gap and thereby change
# the declared scope without saying so.
#
# Each control below is a mutation that MUST NOT flip its named check. A control
# that trips is a false positive and fails the run exactly as an uncaught
# fixture does.
# --------------------------------------------------------------------------

PRESERVED = [
    ("R3-B1", "POS-FIELD-a", "a bus-verb-like production at field position inside envelope",
     "BUS_CLASS_LEADING_TERMINALS_BOUND",
     m_ebnf("                  epoch , deadline , hop_limit , authority_effect , evidence_refs , payload_ref ;",
            '                  epoch , deadline , hop_limit , authority_effect , evidence_refs , payload_ref , broadcast_field ; broadcast_field = "broadcast" , task_id ;')),
    ("R3-B2", "POS-FIELD-b", "a verb-like production at field position inside the fence ledger statement",
     "LEDGER_CLASS_LEADING_TERMINALS_BOUND",
     m_ebnf('fence           = "fence" , fence_id , lineage , epoch , subject_digest ;',
            'fence           = "fence" , fence_id , lineage , epoch , subject_digest , deployment ; deployment = "deploy" , task_id ;')),
    ("R3-B1", "POS-FIELD-c", "a verb-like production at continuation position inside mission",
     "FRAME_CLASS_LEADING_TERMINALS_BOUND",
     m_ebnf('mission         = "mission" , mission_id , lineage , epoch , objective ;',
            'mission         = "mission" , mission_id , lineage , epoch , objective , grant_note ; grant_note = "grant" , profile_id ;')),
    ("R3-B2", "POS-FIELD-d", "the author's own disclosed field-position probe, replayed verbatim",
     "LEDGER_CLASS_LEADING_TERMINALS_BOUND",
     m_ebnf('fence           = "fence" , fence_id , lineage , epoch , subject_digest ;',
            'fence           = "fence" , fence_id , lineage , epoch , subject_digest , escalation ; escalation = "escalate" , task_id ;')),
    # This last control guards a rendering equivalence rather than a disclosed
    # scope boundary, but it is the same obligation — a mutation the named check
    # MUST NOT flip — so it is pinned here rather than as a fixture. It is the
    # legitimate-edit side of the `NEG-R3B2-ac`/`-ad` predicate: a check that
    # fails closed on a pipe-less duplicate row by refusing to read pipe-less
    # rows at all would satisfy both fixtures and still be wrong, and this
    # control is what makes that impossible.
    ("R3-B2", "POS-PIPELESS-a", "the truthful class table rewritten in valid pipe-less GFM",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_class_table_pipeless),
    # The fenced-block controls. `-a` and `-b` are the legitimate-edit side of
    # the `NEG-R3B2-af`/`-ag` predicate and the second half of S1R8-B01: a
    # truthful class table placed behind a legitimate fenced example whose body
    # starts with `#` was rejected as "section 3.3 carries no class table", a
    # false red on a document a reader sees as correct. A repair that suppressed
    # fenced content instead of ignoring headings inside it would satisfy the
    # fixtures and still fail here, and so would one that stopped reading `#`
    # lines as headings at all.
    ("R3-B2", "POS-FENCE-a", "a legitimate backtick-fenced example carrying a `#` line precedes the truthful table",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec(BEFORE_CLASS_TABLE,
            "disjoint classes.\n\n" + FENCED_EXAMPLE + "\n\n| Class |")),
    ("R3-B2", "POS-FENCE-b", "a legitimate tilde-fenced example carrying a `#` line precedes the truthful table",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_spec(BEFORE_CLASS_TABLE,
            "disjoint classes.\n\n" + TILDE_EXAMPLE + "\n\n| Class |")),
    # `-c` is the boundary from the other side. Section 3.3's checks are scoped
    # to section 3.3; `spec-v0.2:115` says "anywhere in section 3.3", not
    # anywhere in the document. A repair that answered the fence family by
    # reading to the end of the file would catch every fixture above and be
    # wrong: an ATX heading outside a fence still ends the section, and content
    # in section 4 is outside the declared scope of these three checks.
    ("R3-B2", "POS-FENCE-c", "a class table in section 4 is outside the declared scope of the section 3.3 checks",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec(SECTION_4_HEAD,
            "## 4. Vendor-neutral task capsule\n\n" + SEVENTH_VERB_TABLE
            + "\n\nEvery lane SHALL consume a capsule equivalent to:")),
    # `-d` pins the declared behaviour of an unclosed fence: CommonMark closes it
    # at the end of the containing block, so this reader carries it to the end of
    # the document and no heading behind it ends the section. That is the
    # over-read direction, which is why `NEG-R3B2-ak` catches rather than misses;
    # this control is the matching obligation that the same behaviour raises no
    # false red on a truthful document. It holds because no line outside section
    # 3.3 names a class in a table row — a measured property of these bytes, not
    # a general one, and recorded as such.
    ("R3-B2", "POS-FENCE-d", "a legitimate unclosed fence at the end of section 3.3 raises no false red",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec(SECTION_33_TAIL,
            "```text\n"
            "# not a heading: this legitimate example fence is never closed")),
    # The opening-boundary controls, and the reason the `NEG-R3B2-al`..`-ao`
    # family cannot be satisfied by reading more of the document.
    #
    # `-a` is the exact mirror of the `-al` witness: the same seventh-verb
    # declaration one heading earlier, in section 3.2, where `spec-v0.2:115`
    # does not reach. It MUST NOT trip. A repair that opened the region at the
    # document start, or at section 3's heading, or at any fixed offset above
    # the marker, catches every fixture in the family and fails here. Together
    # with `POS-FENCE-c` on the closing side, the declared scope is pinned at
    # both ends by controls rather than asserted in prose.
    ("R3-B2", "POS-HEAD-a", "a class table in section 3.2 sits above the section 3.3 heading, outside the declared scope",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec(SECTION_32_HEAD,
            "### 3.2 Collaboration/data plane\n\n" + SEVENTH_VERB_TABLE
            + "\n\nAgents MAY")),
    # `-b` is a false-red guard, not a scope guard, and is recorded as such: the
    # backward scan is fence-aware for the first time here, and a legitimate
    # fenced example carrying a `#` line inside the prose prefix must not change
    # what the region is. If the scan read that line as a heading the region
    # would open there and over-read to the end of the document, which on these
    # bytes still raises no false red, so this control pins the behaviour rather
    # than discriminating the repair.
    ("R3-B2", "POS-HEAD-b", "a legitimate fenced example carrying a `#` line sits between the section 3.3 heading and the marker",
     "SPEC_CLASS_COUNTS_MATCH_GRAMMAR",
     m_spec(SECTION_33_PREFIX_TOP,
            SECTION_33_HEAD + "\n" + FENCED_EXAMPLE
            + "\n\nThe only bus verbs are:")),
    # `-c` is the obligation the `NEG-R3B2-ap`..`-ar` family creates. Those three
    # widen what the region's *start* may be; this one requires that widening not
    # to have moved its *end*. A genuine sibling `### 3.4`, at §3.3's own level
    # and carrying a complete class table declaring a seventh bus leading
    # terminal, is a different section: `spec-v0.2:103` scopes the witness to
    # §3.3 and `spec-v0.2:115` says "anywhere in §3.3", so this MUST NOT trip. It
    # is green on the R11 mechanism and green here, which is what makes it a
    # control rather than a fixture: it holds the closing boundary still while
    # the opening boundary is repaired. A repair that resolved the identity by
    # reading from §3.3's heading to the end of the document catches all three
    # fixtures and fails here, exactly as design (C) fails at `POS-HEAD-a`.
    ("R3-B2", "POS-HEAD-c", "a genuine sibling section 3.4 carrying a class table stays outside the declared scope",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_spec(SECTION_4_HEAD, SIBLING_SECTION_34 + SECTION_4_HEAD)),
    # `-d` is the obligation `NEG-R3B2-as` creates, and it is that fixture with
    # one thing changed: the heading's designator. `### 3.3.1` is a proper
    # descendant of the declared section and must not end it; `### 3.4` is a
    # different section at the same ATX level and must. A repair that answered
    # `-as` by comparing levels more loosely — stopping only at a strictly
    # shallower heading — catches the fixture and fails here, because §3.4 would
    # then be read as part of §3.3 all the way to `## 4.`. `POS-HEAD-c` pins the
    # same sibling one seam later, at §3.3's end; this one pins it at the seam the
    # fixture uses, so the pair differs in the designator alone.
    ("R3-B2", "POS-HEAD-d", "a genuine sibling section 3.4 at the fixture's own seam still ends the declared section",
     "SPEC_CLASS_TERMINALS_MATCH_GRAMMAR",
     m_tail_heading_then_row(SECTION_34_HEAD)),
]

# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def read(root, name):
    with open(os.path.join(root, name), "r", encoding="utf-8") as handle:
        return handle.read().replace("\r\n", "\n")


def rule(title):
    print("")
    print(title)
    print("-" * len(title))


def verify_manifest(root):
    path = os.path.join(root, MANIFEST)
    if not os.path.isfile(path):
        print("MANIFEST ABSENT: %s" % MANIFEST)
        return 2
    text = read(root, MANIFEST)
    rows = re.findall(r"^\|\s*`([^`]+)`\s*\|\s*([\d,]+)\s*\|\s*`([0-9a-f]{64})`\s*\|$",
                      text, re.M)
    if not rows:
        print("MANIFEST UNREADABLE: no `path|bytes|sha256` rows found")
        return 2
    print("manifest rows: %d" % len(rows))
    bad = 0
    for name, size, digest in rows:
        want = int(size.replace(",", ""))
        full = os.path.join(root, name)
        if not os.path.isfile(full):
            print("  MISSING  %s" % name)
            bad += 1
            continue
        got_size = os.path.getsize(full)
        got_hash = sha256_of(full)
        ok = (got_size == want and got_hash == digest)
        print("  %s %s|%d|%s" % ("OK      " if ok else "MISMATCH",
                                 name, got_size, got_hash))
        if not ok:
            bad += 1
    listed = {r[0] for r in rows}
    unlisted = sorted(e for e in os.listdir(root)
                      if e != MANIFEST and e not in listed)
    if unlisted:
        print("  UNLISTED FILES: %s" % unlisted)
        bad += 1
    print("")
    print("MANIFEST VERIFY: %s" % ("PASS" if bad == 0 else "FAIL (%d)" % bad))
    return 0 if bad == 0 else 1


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    if "--verify-manifest" in sys.argv:
        return verify_manifest(root)

    ctx = Ctx(read(root, SPEC), read(root, EBNF),
              read(root, TPN), read(root, LEDGER), root)

    print("Gaia S1 R2 bundle validator, R15 repair revision")
    print("================================================")
    print("mode: offline static analysis; no network, no install, no subprocess, no writes")
    print("subject: %s + %s + %s + %s + %d copied inputs"
          % (SPEC, EBNF, TPN, LEDGER, len(TRUSTED_COPIES)))

    rule("1. Positive checks")
    failed = []
    for blocker, name, fn in CHECKS:
        ok, detail = fn(ctx)
        print("%-5s %-4s %-42s %s" % ("PASS" if ok else "FAIL", blocker, name, detail))
        if not ok:
            failed.append(name)

    rule("2. Discriminating negative fixtures")
    print("Each fixture breaks the bundle in one exact way. The named check MUST fail.")
    print("A fixture whose text anchor no longer matches is INAPP: it is reported as")
    print("lost coverage, not as a defect the suite detected. The two are never merged,")
    print("because anchor drift that reads as detection is a false red (S1R3-N07).")
    print("")
    broken, inapplicable = [], []
    for blocker, fid, description, target, mutate in FIXTURES:
        mutated = mutate(ctx)
        changed = (mutated.spec != ctx.spec or mutated.ebnf_text != ctx.ebnf_text
                   or mutated.tpn != ctx.tpn or mutated.ledger != ctx.ledger)
        if not changed:
            print("%-5s %-5s %-11s %-38s FIXTURE INAPPLICABLE: anchor did not match" %
                  ("INAPP", blocker, fid, target))
            inapplicable.append(fid)
            continue
        ok, _ = CHECK_BY_NAME[target](mutated)
        caught = not ok
        print("%-5s %-5s %-11s %-38s %s" %
              ("PASS" if caught else "FAIL", blocker, fid, target, description))
        if not caught:
            broken.append(fid)

    rule("3. Disclosed-boundary positive controls")
    print("Each control mutates outside the declared scope of its check. The named check")
    print("MUST still pass. A control that trips is a false positive: it would mean the")
    print("repair silently widened the boundary the specification discloses as open.")
    print("")
    tripped, control_inapplicable = [], []
    for blocker, cid, description, target, mutate in PRESERVED:
        mutated = mutate(ctx)
        changed = (mutated.spec != ctx.spec or mutated.ebnf_text != ctx.ebnf_text
                   or mutated.tpn != ctx.tpn or mutated.ledger != ctx.ledger)
        if not changed:
            print("%-5s %-5s %-11s %-38s CONTROL INAPPLICABLE: anchor did not match" %
                  ("INAPP", blocker, cid, target))
            control_inapplicable.append(cid)
            continue
        ok, _ = CHECK_BY_NAME[target](mutated)
        print("%-5s %-5s %-11s %-38s %s" %
              ("PASS" if ok else "FAIL", blocker, cid, target, description))
        if not ok:
            tripped.append(cid)

    rule("4. Blocker coverage")
    for blocker in ("S1-B1", "S1-B2", "S1-B3", "S1-B4", "S1-B5", "S1-B6",
                    "R2-B1", "R2-B2", "R3-B1", "R3-B2", "BUNDLE"):
        pos = sum(1 for b, _, _ in CHECKS if b == blocker)
        neg = sum(1 for b, _, _, _, _ in FIXTURES if b == blocker)
        con = sum(1 for b, _, _, _, _ in PRESERVED if b == blocker)
        print("%-7s positive checks: %2d   negative fixtures: %2d   boundary controls: %2d   %s"
              % (blocker, pos, neg, con, "covered" if pos and neg else "NOT COVERED"))

    rule("5. Result")
    print("positive checks:   %d run, %d failed" % (len(CHECKS), len(failed)))
    print("negative fixtures: %d run, %d not caught, %d inapplicable"
          % (len(FIXTURES), len(broken), len(inapplicable)))
    print("boundary controls: %d run, %d falsely tripped, %d inapplicable"
          % (len(PRESERVED), len(tripped), len(control_inapplicable)))
    if failed:
        print("failing checks: %s" % failed)
    if broken:
        print("undetected fixtures: %s" % broken)
    if inapplicable:
        print("inapplicable fixtures (lost coverage, not detection): %s" % inapplicable)
    if tripped:
        print("falsely tripped controls: %s" % tripped)
    if control_inapplicable:
        print("inapplicable controls (lost coverage, not detection): %s"
              % control_inapplicable)
    verdict = ("VALIDATOR PASS"
               if not (failed or broken or inapplicable or tripped
                       or control_inapplicable)
               else "VALIDATOR FAIL")
    print("")
    print(verdict)
    print("")
    print("This result is mechanical evidence, not an approval. It establishes no")
    print("verdict, no acceptance, no freshness and no authority. S1 remains Pending")
    print("until an independent reviewer binds these exact bytes.")
    return 0 if verdict == "VALIDATOR PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
