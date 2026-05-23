# Ecosystem Manifest Bootstrap

**Ecosystem:** GuitarAlchemist
**Source Repository:** GuitarAlchemist/ga
**Fetched At:** 2026-05-23 12:42:10
**Manifest Generation Time:** 05/23/2026 16:42:10

---

## 🚦 System Health & Quality Scorecard
### chatbot-qa (🟢 OK)
- **Source:** last.json
- **Metric Value:** 1
- **Failures/Warnings:** 0 fail(s), 1 warn(s)

### embeddings (🔴 DEGRADED)
- **Source:** 2026-04-17.json

### ga-harness (🟢 OK)
- **Source:** last.json
- **Metric Value:** 1
- **Summary:** Supervised-loop kit artifacts present and parseable.

### invariants (🔴 DEGRADED)
- **Source:** 2026-04-17.json

### voicing-analysis (🔴 DEGRADED)
- **Source:** 2026-05-17.json

### 🟢 Regressions: None detected

## 🌐 Active Services & Dev Ports

| Service | Port | Public Path | Expected Behavior |
|---|---|---|---|
| ga-react-components (Vite SPA) | 5176 | / | serves React SPA + dev-data middleware |
| GaApi | 5232 | /api/*, /hubs/* | /health → "Healthy" |
| GaChatbot.Api | 5252 | /chatbot/*, /api/chatbot/* | /api/chatbot/status → JSON |
| cloudflared (ga-demos) | 0 | demos.guitaralchemist.com | reverse tunnel to local services |

## 📋 Project Backlog Progress

**Overall Progress:** 37% Shipped (42 of 115 items across 7 epics)

| Epic | Shipped | Active | Backlog | Progress |
|---|---|---|---|---|
| Guitarist Problems to Solve | 0 | 0 | 16 | 0% |
| Prime Radiant / Living Cosmos Ideas | 42 | 9 | 6 | 74% |
| Infrastructure Ideas | 0 | 0 | 8 | 0% |
| Pro-Guitarist Usability Gaps (audit 2026-05-05) | 0 | 0 | 10 | 0% |
| Chatbot Track (curated 2026-05-10) | 0 | 0 | 17 | 0% |
| How to Start a Feature | 0 | 0 | 0 | 0% |
| Modal Meadow roadmap (2026-05-18 session) | 0 | 0 | 7 | 0% |

## 🕒 Recent Commit Activity

| Commit | Author | Date | Subject |
|---|---|---|---|
| feaa7ebe | Stephane Pareilleux | 05/23/2026 12:41:18 | feat(test-page): human-friendly Overview with epic progress + 30d activity |
| 7abcf7ca | Stephane Pareilleux | 05/23/2026 12:19:28 | feat(test-page): ManifestViewer UI at /test/manifest |
| 820d4cc2 | Stephane Pareilleux | 05/23/2026 12:12:26 | feat(test-page): dev manifest endpoint + Architecture/Activity/TODO cards |
| 9091355e | Stephane Pareilleux | 05/23/2026 12:06:55 | feat(test-page): split /test into Development + Demos tabs |
| 51184208 | Stephane Pareilleux | 05/17/2026 16:23:31 | feat(install-audit): close ga's review-independence deduction (96 → 100) |
| 56e2ef78 | github-actions[bot] | 05/17/2026 20:03:21 | chore(quality): snapshot 2026-05-17 [skip ci] |
| 290a29c2 | Stephane Pareilleux | 05/17/2026 16:00:33 | feat(loop-kit): propagate supervised-loop kit to GA (Phase 5.5 adjacent-repo rollout) (#266) |
| 5452dcf5 | github-actions[bot] | 05/17/2026 19:12:26 | chore(quality): snapshot 2026-05-17 [skip ci] |
| 35efdec4 | Stephane Pareilleux | 05/17/2026 15:07:23 | feat(bsp-doom): v2 rebuild — usable scene + GA palette + key/mode rooms (#264) |
| 2abf5a08 | Stephane Pareilleux | 05/17/2026 15:07:00 | fix(prime-radiant): probe /api/health instead of /api/chatbot/status (#265) |
