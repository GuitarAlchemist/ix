# Ecosystem Manifest Bootstrap

**Ecosystem:** GuitarAlchemist
**Source Repository:** GuitarAlchemist/ga
**Fetched At:** 2026-07-30 00:12:32
**Manifest Generation Time:** 07/30/2026 04:12:25

---

## 🚦 System Health & Quality Scorecard
### chatbot-qa (🔴 DEGRADED)
- **Source:** 2026-07-19.json

### chatbot-qa-sessions (🔴 DEGRADED)
- **Source:** 2026-06-16.json

### embeddings (🟢 OK)
- **Source:** 2026-07-23.json
- **Metric Value:** 0.831083333333333
- **Summary:** Real measurement carried forward from 2026-06-25 — deterministic index unchanged (producer can't run on hosted CI).

### ga-harness (🟢 OK)
- **Source:** last.json
- **Metric Value:** 1
- **Summary:** Supervised-loop kit artifacts present and parseable.

### invariants (🟢 OK)
- **Source:** 2026-07-23.json
- **Metric Value:** 1
- **Summary:** 10 of 10 invariants passing

### maintain-gate (🔴 DEGRADED)
- **Source:** last.json
- **Metric Value:** 0
- **Summary:** metric evidence missing — cannot decide

### readme-drift (🔴 DEGRADED)
- **Source:** 2026-07-20.json
- **Metric Value:** 0.2222
- **Summary:** @{total=9; ok=2; borderline=2; stale=0; very_stale=5; absent=0}

### voicing-analysis (🟢 OK)
- **Source:** 2026-07-23.json
- **Metric Value:** 1
- **Summary:** All 4 voicing metrics at 100% across 313,047 voicings.

### ⚠️ ACTIVE REGRESSIONS
- maintain-gate: oracle_status=warn
- readme-drift: 0.3333 → 0.2222
- readme-drift: oracle_status=error

## 🌐 Active Services & Dev Ports

| Service | Port | Public Path | Expected Behavior |
|---|---|---|---|
| ga-react-components (Vite SPA) | 5176 | / | serves React SPA + dev-data middleware |
| GaApi | 5232 | /api/*, /hubs/* | /health → "Healthy" |
| GaChatbot.Api | 5252 | /chatbot/*, /api/chatbot/* | /api/chatbot/status → JSON |
| cloudflared (ga-demos) | 0 | demos.guitaralchemist.com | reverse tunnel to local services |

## 📋 Project Backlog Progress

**Overall Progress:** 20% Shipped (42 of 213 items across 13 epics)

| Epic | Shipped | Active | Backlog | Progress |
|---|---|---|---|---|
| Guitarist Problems to Solve | 0 | 0 | 16 | 0% |
| Prime Radiant / Living Cosmos Ideas | 42 | 9 | 6 | 74% |
| Infrastructure Ideas | 0 | 0 | 18 | 0% |
| Pro-Guitarist Usability Gaps (audit 2026-05-05) | 0 | 0 | 10 | 0% |
| Chatbot Track (curated 2026-05-10) | 0 | 0 | 21 | 0% |
| Jarvis Track — the auditable butler (epics captured 2026-07-02) | 0 | 0 | 29 | 0% |
| Spectral Music Intelligence Track (epics captured 2026-07-04) | 0 | 0 | 23 | 0% |
| Continuous AI Dev Team + Seldon Track (epics captured 2026-07-04) | 0 | 0 | 10 | 0% |
| Giskard Track — mentalics + psychohistoire (epics captured 2026-07-04) | 0 | 0 | 16 | 0% |
| Idea — Moteur de découverte mathématique (capturé 2026-07-04 → **plan approuvé le jour même** : [docs/plans/2026-07-04-research-math-discovery-engine-plan.md](docs/plans/2026-07-04-research-math-discovery-engine-plan.md)) | 0 | 0 | 4 | 0% |
| Compounding KB — retrieval + anti-rot (capturé 2026-07-05 → **plan** : [docs/plans/2026-07-05-arch-compounding-kb-retrieval-curation.md](docs/plans/2026-07-05-arch-compounding-kb-retrieval-curation.md)) | 0 | 0 | 6 | 0% |
| Core Domain Hardening — deferred structural findings (review 2026-07-24) | 0 | 0 | 3 | 0% |
| How to Start a Feature | 0 | 0 | 0 | 0% |

## 🕒 Recent Commit Activity

| Commit | Author | Date | Subject |
|---|---|---|---|
| 5d9b44a8 | Stephane Pareilleux | 07/29/2026 09:08:10 | docs(backlog): JEPA/OPTICK cluster triage + sequencing |
| 10e29d29 | Stephane Pareilleux | 07/24/2026 21:55:42 | fix(core): correct parsing, ordering and thread-safety defects in core domain |
| 428729b5 | Stephane Pareilleux | 07/23/2026 09:37:58 | feat(chatbot): arpeggio PerformanceIntent structured-output tracer (#589) |
| 22736771 | github-actions[bot] | 07/23/2026 10:26:37 | chore(context): daily decay report [skip ci] |
| a0b4f462 | github-actions[bot] | 07/23/2026 10:13:03 | chore(quality): invariants snapshot 2026-07-23 [skip ci] |
| fc81ffc8 | github-actions[bot] | 07/23/2026 09:29:41 | chore(quality): embeddings snapshot 2026-07-23 [skip ci] |
| e7bfdd13 | github-actions[bot] | 07/23/2026 08:10:24 | chore(fleet): fleet-status snapshot 2026-07-23 [skip ci] |
| ce7c7e0c | github-actions[bot] | 07/23/2026 04:43:03 | chore(presence): snapshot 2026-07-23T04:43Z [skip ci] |
| 692c46d9 | github-actions[bot] | 07/23/2026 00:13:01 | chore(quality): snapshot 2026-07-23 [skip ci] |
| 8e60ff14 | github-actions[bot] | 07/23/2026 00:11:39 | chore(fleet): fleet-status snapshot 2026-07-23 [skip ci] |
