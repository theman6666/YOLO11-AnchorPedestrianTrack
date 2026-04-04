---
phase: 1
slug: project-setup
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-03
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Vite Built-in / Vitest (Wave 0 setup) |
| **Config file** | vite.config.ts (existing) |
| **Quick run command** | `npm run dev` |
| **Full suite command** | `npm run build` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `npm run dev` (verify no build/start errors)
- **After every plan wave:** Run `npm run build` (verify production build)
- **Before `/gsd:verify-work`:** Full build must succeed, dev server must start
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | SETUP-01 | build | `npm run dev` | package.json ✅ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | SETUP-02 | build | `grep -q "tailwindcss" package.json` | tailwind.config.js ✅ W0 | ⬜ pending |
| 01-01-03 | 01 | 1 | SETUP-03 | import | `grep -q "lucide-vue-next" package.json` | package.json ✅ W0 | ⬜ pending |
| 01-01-04 | 01 | 1 | SETUP-04 | import | `grep -q "axios" package.json` | src/api/index.ts ✅ W0 | ⬜ pending |
| 01-01-05 | 01 | 1 | SETUP-05 | build | `curl -f http://localhost:5173/video_feed` | vite.config.ts ✅ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `package.json` — root for Vue 3 + Vite project
- [ ] `vite.config.ts` — Vite configuration with proxy setup
- [ ] `tailwind.config.js` — Tailwind CSS dark mode configuration
- [ ] `tsconfig.json` — TypeScript configuration for Vue 3
- [ ] `src/api/index.ts` — Axios instance stub (Wave 0 creates file)
- [ ] `src/main.ts` — Vue app entry point
- [ ] `src/App.vue` — Root component stub
- [ ] `index.html` — HTML entry point

*Wave 0 creates the full project skeleton with all required config files.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Dev server HMR | SETUP-01 | Visual confirmation needed | 1. Run `npm run dev` 2. Edit `App.vue` 3. Verify browser updates without refresh |
| Dark mode toggle | SETUP-02 | Visual confirmation needed | 1. Add `class="dark"` to html 2. Verify Tailwind dark styles apply |
| Icon rendering | SETUP-03 | Visual confirmation needed | 1. Import `<CameraIcon />` in App.vue 2. Verify icon displays correctly |
| API proxy | SETUP-05 | Backend dependency | 1. Ensure Flask backend running on port 5000 2. Test `/video_feed` endpoint through proxy |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
