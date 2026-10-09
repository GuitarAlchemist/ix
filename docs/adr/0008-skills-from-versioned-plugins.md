# Claude Code skills come from versioned plugins, not copies vendored into `.claude/skills/`

Status: accepted (2026-10-09)

## Context

On 2026-10-09 `.claude/skills/` held 103 skills. 54 of them were copies of other projects' skills:

- 47 from compound-engineering, committed with the repo's first commit (2026-03-12): its
  `workflows-*` and `ce-*` skills, and Rails, Ruby, gem, Xcode and image-generation skills that IX
  never uses;
- 7 from mattpocock/skills, installed with `npx skills add --copy` on 2026-06-14 (#100).

The same skills also reached every session through plugins enabled at user level, at other
versions: compound-engineering 2.35.2, from a marketplace not updated since February, and
mattpocock-skills 1.3.1, where `to-prd` and `to-issues` have since become `to-spec` and
`to-tickets`. Nothing updated the copies, so a session saw two versions of one skill under two
names.

## Decision

1. **The vendored copies are removed.** The skills come from the plugins, each at one version:
   compound-engineering 3.30.4 from its maintained marketplace (`compound-engineering-plugin`, which
   provides the `ce-brainstorm`, `ce-plan`, `ce-work` and `ce-compound` that CLAUDE.md and
   `docs/agent-blackbox/install.md` name), and mattpocock-skills 1.3.1. Both are enabled at user
   level; the stale `every-marketplace` was removed on the same day.
2. **IX's own algorithm skills, the 31 `ix-*`, become the `ix` plugin** (0.1.0) at
   `.claude/skills/ix/`. A plugin folder in `.claude/skills/` loads by itself in this repo, so ix
   sessions keep them, now invoked as `ix:<name>`; the GuitarAlchemist marketplace
   (`GuitarAlchemist/claude-plugins`, its ADR 0001) can list it so sessions in other repos install
   it too.
3. **IX's workflow skills stay project skills**, under their own names: `digest`, `correct`,
   `teach`, `chatbot`, `federation-*` and the rest. CLAUDE.md, the session hooks and CI invoke
   several of them by name.

## Consequences

- `.claude/skills/` holds 18 skills and the `ix` plugin, all IX's own.
- compound-engineering's `workflows:*` commands and its Rails and Ruby skills are gone from every
  session, not only ix's: 3.30.4 no longer ships them.
- A clone without these two user-level plugins has none of the aihero or compound-engineering
  skills. Declaring them in `.claude/settings.json` (`extraKnownMarketplaces`, `enabledPlugins`)
  would fix that; it is left for a change of its own, as that file also carries local edits in the
  shared checkout.
- The CE agents vendored under `.claude/agents/` are untouched; whether they follow is a separate
  decision.

## Reversibility

A two-way door: reverting this change restores the copies, and re-enabling compound-engineering
2.35.2 restores its commands. Revisit if a plugin update breaks a skill this repo depends on by
name (`ce-work`, `ce-compound`, `grill-with-docs`, `to-spec`, `to-tickets`): pin that plugin's
version rather than vendoring it again.
