---
status: verifying
trigger: "Investigate issue: file-upload-cors-issue"
created: 2026-04-04T00:00:00Z
updated: 2026-04-04T10:26:53Z
---

## Current Focus
hypothesis: Frontend needs to be rebuilt with empty VITE_API_BASE_URL to remove hardcoded localhost:5000 from built JavaScript
test: Execute GSD phase to rebuild frontend with `npm run build` in frontend directory, verify built JS no longer contains localhost:5000
expecting: After rebuild, built JavaScript should have empty baseURL (or no baseURL), file upload should work on first attempt
next_action: User verification - start Flask server, access via IP address, test file upload

**Evidence gathered:**
- Flask app has CORS middleware implemented (lines 45-62)
- CORS headers are added via `@app.before_request` for OPTIONS and `@app.after_request` for all responses
- Headers include `Access-Control-Allow-Origin: *`
- Error shows request from 192.168.2.13:5000 to localhost:5000 (cross-origin)
- Vite proxy should handle this in dev, but Flask serves built frontend in production
- API client uses VITE_API_BASE_URL which is empty by default in dev but http://localhost:5000 in production
- .env.production has VITE_API_BASE_URL=http://localhost:5000
- When Flask serves frontend at IP address, frontend calls localhost → CORS error
- Git diff shows API client was added in YOLOBOT (not in main)
- Symptoms: Works in main (no API client with hardcoded localhost), fails in YOLOBOT
- User reports issue still exists after fix attempt
- Built JavaScript still contains baseURL: 'http://localhost:5000' (file built at 09:38 today)
- .env.production updated but frontend not rebuilt

## Symptoms
expected: 一次选择显示预览，然后可以开始检测
actual: 需要两次选择文件才能开始检测过程
errors: Access to XMLHttpRequest at 'http://localhost:5000/detect/video' from origin 'http://192.168.2.13:5000' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource. Network Error, Failed to load resource: net::ERR_FAILED
reproduction: 1. 启动Flask服务器 2. 访问前端界面 3. 选择图片或视频文件进行检测 4. 第一次选择部分有效但检测失败 5. 第二次选择才能成功开始检测
started: 在main分支没有出现过这个问题，在YOLOBOT分支出现了

## Eliminated
- hypothesis: .env.production has VITE_API_BASE_URL=http://localhost:5000 causing frontend to call localhost when served from IP address
  evidence: User reports issue still exists after fix attempt
  timestamp: 2026-04-04T00:00:00Z

## Evidence
- timestamp: 2026-04-04T00:00:00Z
  checked: Flask app.py CORS configuration
  found: CORS headers implemented via @app.before_request and @app.after_request decorators with Access-Control-Allow-Origin: *
  implication: CORS should be working on backend side
- timestamp: 2026-04-04T00:00:00Z
  checked: Frontend API client configuration
  found: Uses axios with baseURL from VITE_API_BASE_URL environment variable (empty by default)
  implication: In development, requests go through Vite proxy (localhost:5173 → localhost:5000)
- timestamp: 2026-04-04T00:00:00Z
  checked: Vite proxy configuration
  found: Proxy configured for /detect, /video_feed, /results endpoints to http://localhost:5000
  implication: Frontend should be making requests to same-origin (localhost:5173) which gets proxied
- timestamp: 2026-04-04T00:00:00Z
  checked: Environment configuration files
  found: .env.production has VITE_API_BASE_URL=http://localhost:5000
  implication: In production build, frontend will call localhost:5000 directly
- timestamp: 2026-04-04T00:00:00Z
  checked: Error message analysis
  found: Request from 192.168.2.13:5000 to localhost:5000 - cross-origin due to different hostnames
  implication: When Flask serves frontend at IP address, frontend calls localhost → CORS error
- timestamp: 2026-04-04T00:00:00Z
  checked: Built frontend JavaScript file
  found: Still contains baseURL: 'http://localhost:5000' in dist/assets/index-hxKpNlEh.js
  implication: Frontend wasn't rebuilt after .env.production update, or environment variable not being picked up
- timestamp: 2026-04-04T00:00:00Z
  checked: API client source code
  found: Uses import.meta.env.VITE_API_BASE_URL || '' with fallback to empty string
  implication: When VITE_API_BASE_URL is empty, baseURL should be empty string (relative paths)
- timestamp: 2026-04-04T00:00:00Z
  checked: All environment files
  found: .env.production now has empty VITE_API_BASE_URL, .env and .env.development also empty
  implication: Environment variable should be empty at build time
- timestamp: 2026-04-04T10:26:53Z
  checked: Frontend rebuild with empty VITE_API_BASE_URL
  found: npm run build completed successfully, built JavaScript no longer contains localhost:5000
  implication: CORS issue should be resolved - file upload should work on first attempt

## Resolution
root_cause: .env.production has VITE_API_BASE_URL=http://localhost:5000, causing frontend to make API calls to localhost:5000 when served from IP address (192.168.2.13:5000), resulting in cross-origin requests blocked by CORS policy
fix: Update .env.production to have empty VITE_API_BASE_URL so frontend uses relative paths (same origin)
verification: 
- ✅ Rebuild frontend with `npm run build` in frontend directory (completed)
- Restart Flask server
- Access via IP address (192.168.2.13:5000)
- Test file upload - should work on first attempt
- Verify no CORS errors in browser console
files_changed: [frontend/.env.production, frontend/dist/ (rebuilt)]