---
phase: 06-build-deployment
verified: 2026-04-04T09:38:00Z
status: gaps_found
score: 6/8 must-haves verified
gaps:
  - truth: "Flask serves index.html from frontend-vue/dist/ at root route"
    status: failed
    reason: "Duplicate root route defined at line 106 shadows the new SPA route at line 197. Flask matches routes in definition order, so the old render_template('index.html') route takes precedence over send_from_directory(str(DIST_DIR), 'index.html'). The Vue.js SPA will never be served."
    artifacts:
      - path: "src/run/app.py"
        issue: "Two conflicting @app.route('/') handlers exist (lines 106 and 197)"
    missing:
      - "Remove or comment out the old root route at lines 106-108 that returns render_template('index.html')"
      - "Verify only one @app.route('/') handler exists (the SPA serving route)"
  - truth: "All non-API routes return index.html (SPA support)"
    status: failed
    reason: "The catch-all route exists but cannot function correctly because the root route conflict prevents the SPA from being served. Users accessing http://localhost:5000 will see the old HTML template instead of the Vue.js application."
    artifacts:
      - path: "src/run/app.py"
        issue: "Catch-all route cannot work when root route is shadowed by old template route"
    missing:
      - "After fixing duplicate root route, test catch-all route with non-API paths"
    related_to_gap: 1
---

# Phase 06: Build & Deployment Verification Report

**Phase Goal:** Production-optimized build configured to serve alongside Flask backend
**Verified:** 2026-04-04T09:38:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running `npm run build` generates optimized dist/ directory | VERIFIED | dist/ exists with index.html (428 bytes), assets/index-*.js (112K), assets/index-*.css (16K) |
| 2 | dist/index.html contains references to hashed asset files | VERIFIED | index.html contains `<script src="/assets/index-hxKpNlEh.js">` and `<link href="/assets/index-CW_ZtgX1.css">` with content hashes |
| 3 | Flask serves index.html from frontend-vue/dist/ at root route | FAILED | Duplicate root route at line 106 (`render_template("index.html")`) shadows new SPA route at line 197 (`send_from_directory(str(DIST_DIR), "index.html")`). Flask matches routes in definition order, so old HTML template is served instead of Vue.js SPA |
| 4 | Flask serves static assets (JS, CSS, favicon) from dist/ directory | VERIFIED | Routes exist: `/assets/<path:filename>` (line 204), `/favicon.ico` (line 211), both use `send_from_directory(str(DIST_DIR), ...)` |
| 5 | API routes (/video_feed, /detect, /results) continue working | VERIFIED | All API routes preserved at lines 111, 125, 159, 189. Route ordering is correct: API routes (lines 111-191) defined before SPA routes (lines 197-230) |
| 6 | All non-API routes return index.html (SPA support) | FAILED | Catch-all route exists at line 218 but cannot function correctly due to root route conflict. SPA is never served, so catch-all returns wrong index.html |
| 7 | Build assets are minified and hashed for cache busting | VERIFIED | JS bundle (112K → 43.68K gzipped), CSS bundle (16K → 3.72K gzipped). Filenames use content hashes: `index-hxKpNlEh.js`, `index-CW_ZtgX1.css` |
| 8 | Development build uses empty VITE_API_BASE_URL (triggers Vite proxy) | VERIFIED | .env.development exists with `VITE_API_BASE_URL=` (empty string). src/api/client.ts uses `import.meta.env.VITE_API_BASE_URL` at lines 7 and 16 |

**Score:** 6/8 truths verified (75%)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `frontend-vue/.env.development` | Development environment configuration with empty API base URL | VERIFIED | File exists with `VITE_API_BASE_URL=` (empty string). Triggers Vite proxy per plan. |
| `frontend-vue/.env.production` | Production environment configuration with Flask URL | VERIFIED | File exists with `VITE_API_BASE_URL=http://localhost:5000`. URL confirmed embedded in built JS (grep found `baseURL:\`http://localhost:5000\`` in dist/assets/index-*.js). |
| `frontend-vue/.gitignore` | Git ignore rules for environment files | VERIFIED | Lines 42-43 contain `.env.development` and `.env.production` exclusions. |
| `frontend-vue/dist/index.html` | SPA entry point with asset references | VERIFIED | 428 bytes. Contains hashed asset references: `/assets/index-hxKpNlEh.js` and `/assets/index-CW_ZtgX1.css`. |
| `frontend-vue/dist/assets/` | Hashed JS and CSS bundles | VERIFIED | Directory exists with `index-hxKpNlEh.js` (112K) and `index-CW_ZtgX1.css` (16K). Content hashes change when source changes. |
| `src/run/app.py` | Flask routes for serving SPA and API | VERIFIED | DIST_DIR constant at line 30. Routes: `serve_spa_index()` (197), `serve_assets()` (204), `serve_favicon()` (211), `catch_all()` (218). All API routes preserved. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `npm run build` command | `frontend-vue/dist/` | Vite build process | VERIFIED | Build succeeds in 888ms. dist/ contains optimized assets. |
| Flask app.py routes | `frontend-vue/dist/` files | `send_from_directory()` calls | VERIFIED | Lines 200, 207, 214, 230 all use `send_from_directory(str(DIST_DIR), ...)` correctly. |
| Browser request to `/` | `dist/index.html` | Flask `serve_spa_index()` route | FAILED | Duplicate root route at line 106 shadows line 197. Old `render_template("index.html")` takes precedence. SPA never served. |
| Browser request to `/assets/*` | `dist/assets/*` | Flask `serve_assets()` route | VERIFIED | Route at line 204 correctly serves from `DIST_DIR / "assets"`. |
| `vite build command` | `.env.development` or `.env.production` | Vite mode selection (`--mode` flag) | VERIFIED | Default `npm run build` loads .env.production. `npm run build --mode development` loads .env.development. |
| `src/api/client.ts` | `import.meta.env.VITE_API_BASE_URL` | Build-time environment variable injection | VERIFIED | Lines 7 and 16 use `import.meta.env.VITE_API_BASE_URL`. Confirmed embedded in built JS. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `frontend-vue/dist/assets/index-*.js` | `baseURL` | `.env.production` | YES (FLOWING) | Embedded URL confirmed: `http://localhost:5000` found in minified JS. |
| `src/run/app.py` serve_spa_index() | Returns | `send_from_directory(str(DIST_DIR), "index.html")` | YES (FLOWING) | Route correctly points to dist/index.html. |
| `src/run/app.py` old index() | Returns | `render_template("index.html")` | YES (FLOWING) | Old template route returns HTML but wrong file (old Flask template, not Vue SPA). |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Step 7b: SKIPPED | Server not running | Cannot test runtime behavior without starting Flask server | SKIP |

**Reason for skipping:** Behavioral spot-checks require running Flask server and browser testing. Automated executor cannot start servers. Manual testing required per plan Task 4.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| BUILD-01 | 06-02 | Production build configured for static asset serving | VERIFIED | dist/ contains minified, hashed assets (index-*.js: 112K, index-*.css: 16K). Assets optimized with code splitting, tree-shaking per Vite defaults. |
| BUILD-02 | 06-02 | Build output compatible with Flask static file serving | FAILED | Flask routes configured correctly, but duplicate root route prevents SPA from being served. Old `render_template("index.html")` at line 106 shadows new `send_from_directory(str(DIST_DIR), "index.html")` at line 197. |
| BUILD-03 | 06-01 | Environment variables configuration for API base URL | VERIFIED | .env.development (empty URL) and .env.production (http://localhost:5000) exist. API client uses `import.meta.env.VITE_API_BASE_URL` correctly. |
| BUILD-04 | 06-01 | Build scripts defined for development and production | VERIFIED | package.json has `"build": "run-p type-check \"build-only {@}\" --"` and `"dev": "vite"`. Vite handles mode selection automatically. |

**Orphaned requirements:** None. All 4 requirements (BUILD-01, BUILD-02, BUILD-03, BUILD-04) are claimed in plan frontmatters.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/run/app.py` | 106-108 | Duplicate root route handler | BLOCKER | Old `render_template("index.html")` shadows new SPA serving route. Vue.js application never served at root path. |
| `src/run/app.py` | 197-200 | Shadowed route handler | BLOCKER | New SPA route is unreachable due to earlier duplicate route definition. |

**Anti-pattern classification:** A duplicate route definition is not a stub (both routes have implementations). However, the FIRST route (old template) renders the OLD Flask HTML template instead of the NEW Vue.js SPA, breaking the phase goal.

### Human Verification Required

### 1. Test Flask + SPA Integration After Fixing Duplicate Root Route

**Test:** Remove or comment out the old root route at lines 106-108 in `src/run/app.py`, then start Flask server and browse to `http://localhost:5000`

**Expected:**
- Vue.js application loads (not old HTML template)
- Browser DevTools Network tab shows:
  - `GET /` → 200 → Returns dist/index.html
  - `GET /assets/index-hxKpNlEh.js` → 200 → Returns JS bundle
  - `GET /assets/index-CW_ZtgX1.css` → 200 → Returns CSS bundle
- Application displays correctly with dark mode UI

**Why human:** Cannot start Flask server or browser in automated verification. Requires manual testing to confirm SPA loads after fixing route conflict.

### 2. Test API Endpoint Continuity

**Test:** After fixing root route, test that API endpoints still work correctly

**Expected:**
- `POST /detect/image` with file upload returns JSON (not HTML)
- `POST /detect/video` with file upload returns JSON (not HTML)
- `GET /video_feed?camera_id=0` returns MJPEG stream (not HTML)
- `GET /results/images/det_xxx.jpg` returns image file (not HTML)

**Why human:** Requires running Flask server and making actual API requests to verify no route conflicts exist.

### 3. Test Catch-All Route Behavior

**Test:** Navigate to a non-existent route like `http://localhost:5000/some/random/path` after fixing root route

**Expected:**
- Returns index.html (Vue.js app loads)
- No 404 error from Flask
- Vue Router (if present) handles route client-side

**Why human:** Cannot test browser behavior or client-side routing programmatically.

### Gaps Summary

**Critical blocker:** Duplicate root route in `src/run/app.py`

The old Flask template route at lines 106-108:
```python
@app.route("/")
def index():
    return render_template("index.html")
```

This route is defined BEFORE the new SPA serving route at lines 197-200:
```python
@app.route("/")
def serve_spa_index():
    return send_from_directory(str(DIST_DIR), "index.html")
```

**Impact:**
- Users accessing `http://localhost:5000` see the old HTML template instead of the Vue.js SPA
- The new SPA route is completely unreachable
- BUILD-02 requirement ("Build output compatible with Flask static file serving") is NOT satisfied
- Phase 6 goal ("Production-optimized build configured to serve alongside Flask backend") is BLOCKED

**Root cause:** Flask matches routes in definition order. The first matching route handler wins. Two routes with the same path (`"/"`) cannot coexist.

**Required fix:** Remove or comment out the old root route at lines 106-108. Verify only one `@app.route("/")` handler exists (the SPA serving route).

**Secondary gap:** Catch-all route verification blocked

The catch-all route at line 218 (`@app.route("/<path:path>")`) exists and appears correctly implemented, but cannot be verified as working until the root route conflict is resolved.

**Summary:**
- 6 out of 8 observable truths verified (75%)
- 1 critical blocker (duplicate root route)
- 1 secondary blocker (catch-all route untested due to primary blocker)
- All environment configuration artifacts verified and working
- All build artifacts verified and optimized
- All API routes preserved and correctly ordered

---

**Verified:** 2026-04-04T09:38:00Z
**Verifier:** Claude (gsd-verifier)
