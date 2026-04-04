---
phase: 06
slug: build-deployment
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-04
---

# Phase 06 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Manual E2E Testing (Phase 5 pattern) |
| **Config file** | none — manual verification documented in test cases |
| **Quick run command** | `python src/run/app.py` (Flask backend) + verify build output |
| **Full suite command** | Execute Phase 5 test cases after deployment |
| **Estimated runtime** | ~5 minutes (build + verify) |

---

## Sampling Rate

- **After every task commit:** Manual verification of file existence and content
- **After every plan wave:** Build verification (`npm run build` + Flask serving test)
- **Before `/gsd:verify-work`:** Full E2E test execution from Phase 5 test cases
- **Max feedback latency:** ~60 seconds per task

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | BUILD-01 | build | `npm run build && ls frontend-vue/dist/index.html` | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | BUILD-02 | integration | `curl http://localhost:5000` | ❌ W0 | ⬜ pending |
| 06-01-03 | 01 | 1 | BUILD-03 | env | `grep VITE_API_BASE_URL frontend-vue/.env.production` | ❌ W0 | ⬜ pending |
| 06-01-04 | 01 | 1 | BUILD-04 | build | `npm run build` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] No automated test framework needed — Phase 5 established manual E2E verification pattern
- [ ] Build verification commands — `ls frontend-vue/dist/` to check output
- [ ] Environment file verification — check `.env.production` exists and contains `VITE_API_BASE_URL`

*Rationale: Phase 6 is infrastructure configuration. Verification focuses on build output and Flask integration, following Phase 5's manual testing approach.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Production build optimization | BUILD-01 | Visual inspection of minified assets | 1. Run `npm run build` 2. Check `dist/assets/*.js` are minified 3. Verify file sizes are reasonable |
| Flask static file serving | BUILD-02 | Requires running Flask server | 1. Start Flask with new routes 2. Visit http://localhost:5000 3. Verify app loads and API calls work |
| Environment variable configuration | BUILD-03 | Requires build and runtime test | 1. Build with `npm run build` 2. Check Axios uses correct base URL in production |
| Dev/prod build scripts | BUILD-04 | Script execution verification | 1. Run `npm run dev` — verify dev mode works 2. Run `npm run build` — verify production build |

*All phase behaviors have manual verification following Phase 5's established E2E testing pattern.*

---

## Validation Sign-Off

- [ ] All tasks have manual verification steps defined
- [ ] Build output verification after each task
- [ ] Flask serving test after routes added
- [ ] Environment configuration tested for both dev and production
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

---

*Phase: 06-build-deployment*
*Validation strategy created: 2026-04-04*
