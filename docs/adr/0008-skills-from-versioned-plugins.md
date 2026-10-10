# Claude Code skills come from versioned plugins, not copies vendored into `.claude/skills/`

Status: accepted (2026-10-09)

## Context

On 2026-10-09 `.claude/skills/` held 103 skills. 54 of them were copies of other projects' skills:

- 47 from compound-engineering, committed with the repo's first commit (2026-03-12): its
  `workflows-*` and `ce-*` skills, and Rails, Ruby, gem, Xcode and image-generation skills that IX
  never uses;
- 7 from mattpocock/skills, installed with `npx skills add --copy` on 2026-06-14 (#100), and
  recorded in `skills-lock.json`.

The same skills also reached every session through plugins enabled at user level, at other
versions: compound-engineering 2.35.2, from a marketplace not updated since February, and
mattpocock-skills 1.3.1, where `to-prd` and `to-issues` have since become `to-spec` and
`to-tickets`. Nothing updated the copies, so a session saw two versions of one skill under two
names.

## Decision

1. **The vendored copies are removed, and `skills-lock.json` with them** (`skills install` would
   otherwise restore the copies). The skills come from the plugins, each at one version:
   compound-engineering 3.30.4 from its maintained marketplace (`compound-engineering-plugin`, which
   provides the `ce-brainstorm`, `ce-plan`, `ce-work` and `ce-compound` that CLAUDE.md and
   `docs/agent-blackbox/install.md` name), and mattpocock-skills 1.3.1. Both are enabled at user
   level; the stale `every-marketplace` was removed on the same day.
2. **Plugin skills are namespaced** (`/mattpocock-skills:grill-with-docs`). The bare name also
   works for every skill this repo's docs cite, since each sets `name:` in its frontmatter, unless
   another command takes it.
3. **IX's own skills stay project skills**: the 31 `ix-*` algorithm skills and the workflow skills
   (`digest`, `correct`, `teach`, `chatbot`, `federation-*`, ...). Packaging the `ix-*` as an `ix`
   plugin was tried and dropped: a plugin folder in `.claude/skills/` loads only when Claude Code
   starts in the repo root, whereas project skills are found from any subdirectory up to the root
   (verified 2026-10-09: started in `crates/ix-search`, the plugin was not listed). Installing the
   plugin from the GuitarAlchemist marketplace instead would load a pinned copy, so an edit to an
   `ix-*` skill would not take effect until a version bump. Packaging waits for a way that keeps
   both.

## Consequences

- `.claude/skills/` holds 49 skills, all IX's own.
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
