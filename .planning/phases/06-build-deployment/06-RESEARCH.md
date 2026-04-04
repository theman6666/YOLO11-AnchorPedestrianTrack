# Phase 6: Build & Deployment - Research

**Researched:** 2026-04-04
**Domain:** Production build configuration and Flask static file serving
**Confidence:** HIGH

## Summary

Phase 6 focuses on configuring the Vite production build to serve alongside the Flask backend without requiring a separate web server. The key insight is that Flask's `send_from_directory` can serve the pre-built Vue.js SPA from `frontend-vue/dist/`, with a catch-all route to handle client-side routing. The build process requires two environment configurations: development (using Vite proxy) and production (direct API connection).

**Primary recommendation:** Configure Vite with `base: '/'` for absolute paths, create `.env.development` and `.env.production` files, and add Flask routes to serve the SPA from the dist folder while preserving API endpoints.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 构建输出保持在 `frontend-vue/dist/` 目录 — 不混合到 Flask 代码库
- **D-02:** 开发阶段 Vite 直接服务 dist/ 内容进行预览
- **D-03:** 生产阶段 Flask 配置路由从 dist/ 服务静态文件
- **D-04:** 使用 Vite 内置的 .env 文件模式 — `.env.development` 用于开发，`.env.production` 用于生产
- **D-05:** `VITE_API_BASE_URL` 控制后端 API 地址：
  - 开发：空字符串，使用 Vite 代理（`/video_feed`, `/detect` 等）
  - 生产：完整 URL（如 `http://localhost:5000` 或实际部署地址）
- **D-06:** .env.example 文件作为模板存在，不包含敏感信息
- **D-07:** Flask 使用 `send_from_directory` 从 `frontend-vue/dist/` 服务静态文件
- **D-08:** API 路由保持 `/video_feed`, `/detect/image`, `/detect/video`, `/results` 前缀
- **D-09:** 所有非 API 请求返回 `index.html`（支持 SPA 路由）
- **D-10:** 保持现有的 `npm run build` 脚本用于开发和生产构建
- **D-11:** Vite 根据 `NODE_ENV` 或 `--mode` 自动选择正确的 .env 文件
- **D-12:** 不创建单独的 build:dev/build:prod 脚本
- **D-13:** 启用 Vite 的默认优化（代码分割、压缩、tree-shaking）
- **D-14:** 使用时间戳或哈希文件名进行缓存破坏（Vite 默认行为）
- **D-15:** 确保所有资源路径相对正确，不依赖开发服务器的代理

### Claude's Discretion
- Flask 路由的具体实现细节（蓝图 vs 直接路由）
- 是否添加构建后的验证步骤（如文件存在性检查）
- 生产环境的额外安全头（CORS、CSP 等）

### Deferred Ideas (OUT OF SCOPE)
- **Nginx/Apache 集成** — 不在此阶段实现，Flask 直接服务静态文件已足够
- **Docker 容器化** — 延迟到后续阶段，不属于阶段 6 范围
- **CI/CD 流水线** — 不在阶段 6 实现，仅配置本地构建和部署
- **多环境配置** — 仅支持开发和生产，不添加 staging 环境
- **HTTPS/SSL 配置** — 不在阶段 6 处理
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BUILD-01 | Production build configured for static asset serving | Vite default build output in dist/ with hashed assets for cache busting |
| BUILD-02 | Build output compatible with Flask static file serving | Flask send_from_directory serves index.html and assets/ from dist/ |
| BUILD-03 | Environment variables configuration for API base URL | .env.development (empty) uses Vite proxy, .env.production sets full URL |
| BUILD-04 | Build scripts defined for development and production | Existing npm scripts work, Vite auto-selects env by mode |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Vite | 8.0.3 | Build system and dev server | Current stable release, excellent Vue 3 support, fast HMR |
| Vue 3 | 3.5.31 | Frontend framework | Already in use, required for build |
| TypeScript | 6.0.0 | Type checking | Already configured, vue-tsc for type-safe builds |

### Build Tools
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| @vitejs/plugin-vue | 6.0.5 | Vue SFC compiler | Required for Vue 3 single-file components |
| vite-plugin-vue-devtools | 8.1.1 | Dev tools integration | Development only, excluded from production |
| npm-run-all2 | 8.0.4 | Parallel task runner | Runs type-check and build-only concurrently |

### Verified Versions
As of 2026-04-04, current versions from project:
- **Vite:** 8.0.3 (latest stable)
- **Vue:** 3.5.31 (latest 3.x)
- **TypeScript:** 6.0.0 (latest 6.x)
- **Node.js:** v22.17.0 (meets requirement ^20.19.0 || >=22.12.0)
- **npm:** 10.9.2

**Installation:**
```bash
# All dependencies already installed
npm install
```

**Version verification:**
```bash
npm view vite version        # 8.0.3
npm view vue version         # 3.5.31
npm view typescript version  # 6.0.0
```

## Architecture Patterns

### Recommended Project Structure
```
frontend-vue/
├── dist/                    # Production build output (gitignored)
│   ├── index.html          # Entry point with asset references
│   ├── assets/             # Hashed JS/CSS bundles
│   └── favicon.ico         # Static asset
├── src/                    # Source code
├── .env                    # Local overrides (gitignored)
├── .env.development        # Development environment variables
├── .env.production         # Production environment variables
├── .env.example            # Template for developers
├── vite.config.ts          # Vite configuration
├── package.json            # Build scripts
└── tsconfig.*.json         # TypeScript configurations

src/run/
├── app.py                  # Flask application (modified to serve SPA)
```

### Pattern 1: Environment-Based API Configuration
**What:** Separate environment variable files for development and production that control API base URL
**When to use:** Any Vue/Vite app that communicates with a backend API
**Example:**
```bash
# .env.development
VITE_API_BASE_URL=  # Empty string triggers Vite proxy

# .env.production
VITE_API_BASE_URL=http://localhost:5000
```

**Implementation:**
```typescript
// Source: frontend-vue/src/api/client.ts
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 30000
})
```

**Why this works:**
- Vite loads `.env.development` when running `vite` or `vite build --mode development`
- Vite loads `.env.production` when running `vite build` (default mode)
- `import.meta.env.VITE_API_BASE_URL` is replaced at build time with the actual value
- Development: empty baseURL means API calls use Vite's configured proxy
- Production: full URL means API calls go directly to Flask backend

### Pattern 2: Flask SPA Serving with Catch-All Route
**What:** Flask serves the Vue.js SPA by returning index.html for all non-API routes
**When to use:** Serving a client-side routed SPA (Vue Router) from Flask
**Example:**
```python
# Source: Standard Flask SPA pattern
from flask import Flask, send_from_directory
from pathlib import Path

app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = PROJECT_ROOT / "frontend-vue" / "dist"

# API routes (defined before catch-all)
@app.route("/video_feed")
def video_feed():
    # ... existing implementation
    pass

@app.route("/detect/image", methods=["POST"])
def detect_image():
    # ... existing implementation
    pass

@app.route("/results/<path:filename>")
def serve_result(filename):
    # ... existing implementation
    pass

# Serve SPA index.html
@app.route("/")
def index():
    return send_from_directory(str(DIST_DIR), "index.html")

# Serve static assets
@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(str(DIST_DIR / "assets"), filename)

# Serve favicon
@app.route("/favicon.ico")
def serve_favicon():
    return send_from_directory(str(DIST_DIR), "favicon.ico")

# Catch-all route for SPA client-side routing
@app.route("/<path:path>")
def catch_all(path):
    return send_from_directory(str(DIST_DIR), "index.html")
```

**Critical ordering:** API routes MUST be defined before the catch-all route, otherwise `/video_feed` would be treated as a SPA route and return index.html.

### Pattern 3: Vite Base Path Configuration
**What:** Control whether generated HTML uses absolute or relative asset paths
**When to use:**
- **Absolute (`base: '/'` or default):** When serving from domain root or known path
- **Relative (`base: './'`):** When serving from unknown subdirectory or file:// protocol
**Example:**
```typescript
// vite.config.ts - current configuration (no base specified)
export default defineConfig({
  plugins: [vue(), vueDevTools()],
  // base defaults to '/' for production
  // This generates: <script src="/assets/index-xyz.js">
})
```

**Current behavior:** Build generates absolute paths (`/assets/...`) which work correctly when Flask serves from root path.

**Alternative (NOT recommended):**
```typescript
// If serving from subdirectory, use:
base: '/subdir/'
// Generates: <script src="/subdir/assets/index-xyz.js">

// If relative paths needed, use:
base: './'
// Generates: <script src="./assets/index-xyz.js">
```

**Recommendation:** Keep default `base: '/'` since Flask serves the SPA at root path.

### Anti-Patterns to Avoid
- **Mixing dist/ into Flask codebase:** Keep frontend-vue/dist/ separate, Flask serves from there
- **Creating separate build scripts:** Vite's `npm run build` already handles mode selection
- **Hardcoding API URLs in source code:** Use environment variables for flexibility
- **Defining catch-all route before API routes:** Would break all API endpoints
- **Using build-time env vars for runtime secrets:** VITE_* vars are embedded in bundle, visible in client code

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Environment variable loading | Custom .env parser | Vite built-in .env support | Handles mode selection, prefixes, build-time injection |
| Static asset hashing | Manual cache-busting | Vite built-in content hash | Automatic hash generation, changes when content changes |
| Code splitting | Manual bundle configuration | Vite built-in chunking | Automatic vendor/app splitting, optimized caching |
| TypeScript compilation | Manual tsc calls | vue-tsc with Vite plugin | Vue SFC-aware type checking, integrated with build |
| Production optimization | Manual minification | Vite built-in optimizations | Rollup-based minification, tree-shaking, dead code elimination |
| Dev server HMR | Custom file watcher | Vite dev server | Instant HMR, ESM-native, fast updates |

**Key insight:** Vite provides a complete build toolchain. Custom solutions introduce maintenance burden and miss edge cases (e.g., CSS module handling, asset imports, source maps).

## Runtime State Inventory

> Not applicable — this is a greenfield configuration phase, not a rename/refactor phase.

## Common Pitfalls

### Pitfall 1: Build-Time vs Runtime Environment Variables
**What goes wrong:** Developers expect to change `.env.production` and rebuild to change API URL, but the old value is cached by browser or the rebuild doesn't pick up changes.

**Why it happens:** Vite embeds `VITE_*` environment variables into the bundle at build time. The values are hardcoded in the JavaScript. Changing the .env file requires a rebuild.

**How to avoid:**
1. Document that `.env.production` changes require running `npm run build` again
2. Add a post-build verification step to check the embedded value:
   ```bash
   grep -o 'VITE_API_BASE_URL.*' dist/assets/index-*.js | head -1
   ```
3. Consider adding a build verification script (see Claude's discretion)

**Warning signs:** API calls go to wrong URL after changing .env file, clear browser cache doesn't help.

### Pitfall 2: API Route vs SPA Route Priority
**What goes wrong:** After adding the catch-all route for SPA support, API endpoints start returning index.html instead of JSON.

**Why it happens:** Flask routes are matched in definition order. If catch-all route is defined before API routes, it intercepts all requests.

**How to avoid:**
1. Always define API routes first (before the catch-all)
2. Use specific route patterns for API (`/video_feed`, `/detect/*`, `/results/*`)
3. Define catch-all route last with `<path:path>` parameter
4. Test all API endpoints after adding SPA serving

**Warning signs:** API calls return HTML instead of JSON, browser shows "Unexpected token < in JSON"

### Pitfall 3: Absolute Asset Paths Breaking in Subdirectory Deployment
**What goes wrong:** Assets load correctly in development but return 404 when deployed to a subdirectory (e.g., `example.com/tools/yolo11/`).

**Why it happens:** Vite's default `base: '/'` generates absolute paths (`/assets/file.js`). When serving from a subdirectory, the browser requests `example.com/assets/file.js` instead of `example.com/tools/yolo11/assets/file.js`.

**How to avoid:**
1. For this project: NOT APPLICABLE (Flask serves at root path)
2. For subdirectory deployment: set `base: '/tools/yolo11/'` in vite.config.ts
3. Test deployment in environment matching production structure

**Warning signs:** Assets return 404 in production, console shows "Failed to load module" errors.

### Pitfall 4: Missing .env.production File
**What goes wrong:** Production build uses development API configuration (empty VITE_API_BASE_URL), causing API calls to fail.

**Why it happens:** Vite only loads `.env.production` if it exists. Without it, falls back to `.env` or defaults.

**How to avoid:**
1. Create `.env.production` with explicit `VITE_API_BASE_URL`
2. Add `.env.production` to `.gitignore` (contains deployment-specific URLs)
3. Document required environment variables in `.env.example`
4. Verify build output contains correct API URL:
   ```bash
   npm run build
   grep "localhost:5000" dist/assets/index-*.js
   ```

**Warning signs:** API calls go to relative URLs (breaking in production), network tab shows requests to wrong origin.

### Pitfall 5: Flask Serving Old Build After Code Changes
**What goes wrong:** Frontend code changes don't appear in browser, even after rebuilding.

**Why it happens:** Flask caches file handles or browser serves cached version. The dist/ folder may have stale files.

**How to avoid:**
1. Always run `npm run build` after code changes
2. Verify dist/index.html modification time is recent
3. Clear browser cache or use hard refresh (Ctrl+Shift+R)
4. Consider adding Flask route to disable caching for development:
   ```python
   @app.after_request
   def add_header(response):
       response.headers['Cache-Control'] = 'no-store'
       return response
   ```

**Warning signs:** Code changes don't appear, hard refresh fixes it temporarily.

## Code Examples

Verified patterns from official sources:

### Environment Variable Usage
```typescript
// Source: Vite official docs - https://vite.dev/guide/env-and-mode
// Access environment variable in client code
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL

// Check if build is production
const isProd = import.meta.env.PROD

// Get current mode (development/production)
const mode = import.meta.env.MODE
```

### Flask SPA Serving (Recommended Approach)
```python
# Source: Flask best practices for SPAs
from pathlib import Path
from flask import Flask, send_from_directory

app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = PROJECT_ROOT / "frontend-vue" / "dist"

# 1. API routes (define these FIRST)
@app.route("/api/<path:endpoint>")
def api_proxy(endpoint):
    # Proxy to backend or handle API
    pass

# 2. Static assets (favicon, images)
@app.route("/favicon.ico")
def favicon():
    return send_from_directory(str(DIST_DIR), "favicon.ico")

@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(str(DIST_DIR / "assets"), filename)

# 3. SPA entry point
@app.route("/")
def index():
    return send_from_directory(str(DIST_DIR), "index.html")

# 4. Catch-all for client-side routing (define LAST)
@app.route("/<path:path>")
def catch_all(path):
    # Don't intercept API routes
    if path.startswith('api') or path.startswith('video_feed'):
        return f"API route {path} not found", 404
    return send_from_directory(str(DIST_DIR), "index.html")
```

### Build Verification Script
```bash
# Source: Common build verification pattern
#!/bin/bash
# verify-build.sh - Check build output for common issues

echo "Verifying production build..."

# Check 1: dist/index.html exists
if [ ! -f "dist/index.html" ]; then
    echo "❌ FAIL: dist/index.html not found"
    exit 1
fi
echo "✅ dist/index.html exists"

# Check 2: Assets folder exists
if [ ! -d "dist/assets" ]; then
    echo "❌ FAIL: dist/assets folder not found"
    exit 1
fi
echo "✅ dist/assets exists"

# Check 3: JS bundle exists
JS_BUNDLE=$(ls dist/assets/index-*.js 2>/dev/null | head -1)
if [ -z "$JS_BUNDLE" ]; then
    echo "❌ FAIL: No JS bundle found in dist/assets"
    exit 1
fi
echo "✅ JS bundle: $(basename $JS_BUNDLE)"

# Check 4: CSS bundle exists
CSS_BUNDLE=$(ls dist/assets/index-*.css 2>/dev/null | head -1)
if [ -z "$CSS_BUNDLE" ]; then
    echo "❌ FAIL: No CSS bundle found in dist/assets"
    exit 1
fi
echo "✅ CSS bundle: $(basename $CSS_BUNDLE)"

# Check 5: API base URL is embedded (if production)
if grep -q "VITE_API_BASE_URL" dist/assets/index-*.js; then
    # Extract the URL for verification
    API_URL=$(grep -o 'baseURL:"[^"]*"' dist/assets/index-*.js | head -1)
    echo "✅ API base URL: $API_URL"
fi

echo "✅ Build verification complete"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Webpack with complex config | Vite with conventions | 2022+ | Faster builds, simpler config, better DX |
| Manual environment variable injection | Vite built-in env loading | Vite 2.0+ | Standardized approach, build-time injection |
| Separate dev/prod configs | Mode-based .env files | Vite 2.0+ | One build command, automatic mode selection |
| Flask-CORS for SPA integration | Flask serves SPA directly | Current standard | Eliminates CORS issues, simpler deployment |

**Deprecated/outdated:**
- **Custom webpack configs:** Vite replaces webpack for most use cases
- **Vue CLI:** Deprecated in favor of Vite
- **Separate frontend/backend servers:** Modern pattern serves SPA from backend
- **.env files without VITE_ prefix:** Not accessible in client code (server-side only)

## Open Questions

1. **Flask route implementation style**
   - What we know: Need to add static file serving routes to Flask app
   - What's unclear: Whether to use Flask blueprints or direct routes
   - Recommendation: Use direct routes in app.py (simpler, no blueprints needed yet)

2. **Build verification steps**
   - What we know: Can add script to check dist/ output and embedded API URL
   - What's unclear: Whether to automate this in npm scripts or leave as manual check
   - Recommendation: Add optional `npm run verify-build` script, not part of main build flow

3. **Production security headers**
   - What we know: Can add CORS, CSP headers via Flask @after_request
   - What's unclear: Which headers are required for local deployment
   - Recommendation: Skip for now (local Flask deployment is trusted environment)

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Node.js | Vite build system | ✓ | v22.17.0 | — |
| npm | Package management | ✓ | 10.9.2 | — |
| Python | Flask backend | ✓ | 3.11.9 / 3.10.0 | — |
| Flask | Static file serving | ✓ | 3.1.2 | — |

**Missing dependencies with no fallback:** None

**Missing dependencies with fallback:** None

All required dependencies are available. The environment is ready for build and deployment configuration.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Manual E2E testing (no automated framework) |
| Config file | `.planning/config.json` - nyquist_validation: true |
| Quick run command | Manual testing in browser |
| Full suite command | Execute all Phase 5 E2E test cases |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BUILD-01 | Production build creates dist/ with index.html and assets/ | manual | Run `npm run build`, verify dist/ structure | ❌ Wave 0 |
| BUILD-02 | Flask serves SPA from dist/ with working API endpoints | manual | Start Flask, browse to localhost:5000, test API | ❌ Wave 0 |
| BUILD-03 | Development uses empty VITE_API_BASE_URL (proxy), production uses full URL | manual | Check dist/assets/index-*.js for embedded URL | ❌ Wave 0 |
| BUILD-04 | Build scripts work for both development and production modes | manual | Run `npm run build` (production), `npm run build --mode development` (dev) | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** Manual verification of built files
- **Per wave merge:** Full manual E2E test of deployment
- **Phase gate:** All requirements verified manually before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `test-cases/00-build-verification.md` — Step-by-step build verification checklist
- [ ] `test-cases/01-deployment-testing.md` — Flask + SPA integration testing
- [ ] `test-cases/02-environment-switching.md` — Dev/prod environment switching tests
- [ ] Framework install: Not applicable (manual E2E testing)

No automated test framework detected. Phase 5 used manual E2E testing with detailed test case documents. Phase 6 should follow the same pattern.

## Sources

### Primary (HIGH confidence)
- **Vite Official Documentation** - https://vite.dev/guide/env-and-mode
  - Environment variable handling, build modes, .env file loading
- **Vite Build Options** - https://vite.dev/config/build-options
  - Build configuration, output structure, asset handling
- **Vite Static Asset Handling** - https://v3.vitejs.dev/guide/assets
  - How Vite processes and references static assets
- **Project source files** - frontend-vue/vite.config.ts, package.json, src/api/client.ts
  - Current build configuration and API client implementation

### Secondary (MEDIUM confidence)
- **StackOverflow: Vite relative path configuration** - https://stackoverflow.com/questions/76442884/
  - Discussion on base path configuration for different deployment scenarios
- **GitHub: Vite base path issues** - https://github.com/vitejs/vite/discussions/5081
  - Community discussion on relative vs absolute asset paths
- **Flask Documentation** - send_from_directory API reference
  - Standard Flask pattern for serving static files

### Tertiary (LOW confidence)
- **OneUptime Blog: Production-Ready React Setup (January 2026)** - https://oneuptime.com/blog/post/2026-01-08-react-typescript-vite-production-setup/view
  - General Vite production setup guide (not Vue-specific)
- **Simpl Engineering: Runtime ENV Config** - https://engineering.simpl.de/post/runtime-env-config-part1/
  - Discussion of build-time vs runtime environment variables

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Verified current versions in project, all dependencies available
- Architecture: HIGH - Based on official Vite/Flask documentation, proven SPA serving pattern
- Pitfalls: HIGH - Common issues documented in Vite/Flask communities, verified against project structure

**Research date:** 2026-04-04
**Valid until:** 30 days (Vite and Flask stable APIs, unlikely to change)

---

## Next Steps for Planner

This research provides the planner with:

1. **Locked decisions** from CONTEXT.md that must be followed (D-01 through D-15)
2. **Technical patterns** for Vite build configuration and Flask SPA serving
3. **Common pitfalls** to avoid in build/deployment configuration
4. **Environment verification** confirming all dependencies are available
5. **Validation approach** using manual E2E testing (consistent with Phase 5)

The planner can now create detailed PLAN.md files addressing requirements BUILD-01 through BUILD-04 with confidence in the technical approach.
