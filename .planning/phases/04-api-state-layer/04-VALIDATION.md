---
phase: 04
slug: api-state-layer
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-03
---

# Phase 04 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Visual inspection + Vite dev server |
| **Config file** | frontend-vue/vite.config.ts (already configured) |
| **Quick run command** | `cd frontend-vue && npm run build` |
| **Full suite command** | Visual check of API responses in browser |
| **Estimated runtime** | ~30 seconds (dev server startup) |

---

## Sampling Rate

- **After every task commit:** Run `cd frontend-vue && npm run build` to verify no build errors
- **After every plan wave:** Visual inspection of API responses in browser
- **Before `/gsd:verify-work`:** Full visual check must pass
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | API-01, STATE-01 | build | `cd frontend-vue && npm run build` | ✅ W0 | ⬜ pending |
| 04-02-01 | 02 | 1 | API-02, STATE-02 | build | `cd frontend-vue && npm run build` | ✅ W0 | ⬜ pending |
| 04-02-02 | 02 | 1 | API-02, STATE-02 | manual | Visual browser check | ✅ W0 | ⬜ pending |
| 04-03-01 | 03 | 1 | API-03, STATE-03 | build | `cd frontend-vue && npm run build` | ✅ W0 | ⬜ pending |
| 04-03-02 | 03 | 1 | API-03, STATE-03 | manual | Visual browser check | ✅ W0 | ⬜ pending |
| 04-04-01 | 04 | 1 | API-04, STATE-04 | build | `cd frontend-vue && npm run build` | ✅ W0 | ⬜ pending |
| 04-04-02 | 04 | 1 | API-04, STATE-04 | manual | Visual browser check | ✅ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `frontend-vue/` — Vite project exists from Phase 1
- [x] `frontend-vue/src/api/client.ts` — Axios instances configured from Phase 1
- [x] `frontend-vue/src/api/types.ts` — Response types defined from Phase 1
- [x] `frontend-vue/src/components/panels/` — Panel components exist from Phase 3
- [x] `frontend-vue/src/App.vue` — Reactive state refs pre-declared from Phase 3

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| API response handling | API-01 | Requires Flask backend running | Start Flask server, verify API calls return correct data |
| Video stream display | API-02 | Requires camera/video input | Start camera, verify MJPEG stream displays |
| File upload detection | API-03, API-04 | Requires test files | Upload image/video, verify detection results appear |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
