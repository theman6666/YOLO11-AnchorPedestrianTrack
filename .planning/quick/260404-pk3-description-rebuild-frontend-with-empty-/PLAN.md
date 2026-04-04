---
quick_id: 260404-pk3
slug: description-rebuild-frontend-with-empty-
description: Rebuild frontend with empty VITE_API_BASE_URL to fix CORS issue
date: 2026-04-04
type: execute
autonomous: true
files_modified:
  - frontend/dist/
must_haves:
  truths:
    - ".env.production has empty VITE_API_BASE_URL (already updated)"
    - "Running npm run build generates fresh dist/ directory"
    - "Built JavaScript should not contain localhost:5000 (CORS issue resolved)"
    - "Frontend rebuild required after environment variable change"
  artifacts:
    - path: "frontend/.env.production"
      provides: "Production environment configuration with empty API base URL"
      contains: "VITE_API_BASE_URL="
    - path: "frontend/dist/assets/index-*.js"
      provides: "Built JavaScript bundle"
      missing: "localhost:5000" 
---

<objective>
Rebuild frontend with empty VITE_API_BASE_URL to fix CORS issue that causes file upload to require two selections.

Purpose: Frontend was built with hardcoded localhost:5000 in .env.production, causing CORS errors when Flask serves from IP address (192.168.2.13:5000). After updating .env.production to have empty VITE_API_BASE_URL, rebuild is needed to embed the correct configuration.
Output: Fresh dist/ directory with built JavaScript that uses relative API paths (no hardcoded localhost:5000).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
</execution_context>

<context>
@frontend/.env.production
@frontend/package.json
@.planning/debug/file-upload-cors-issue.md (debug session with root cause analysis)

# Background
File upload requires two selections - first attempt fails with CORS error:
"Access to XMLHttpRequest at 'http://localhost:5000/detect/video' from origin 'http://192.168.2.13:5000' has been blocked by CORS policy"

Root cause: .env.production had VITE_API_BASE_URL=http://localhost:5000, causing frontend to call localhost when served from IP address.
Fix: Updated .env.production to have empty VITE_API_BASE_URL (already done).
Remaining task: Rebuild frontend to pick up the new environment configuration.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Verify environment configuration</name>
  <files>
    - frontend/.env.production
    - frontend/package.json
  </files>
  <read_first>
    - frontend/.env.production
    - frontend/package.json
  </read_first>
  <action>
Verify that .env.production has empty VITE_API_BASE_URL and package.json has correct build scripts.

1. Check .env.production:
   ```bash
   grep "^VITE_API_BASE_URL=" frontend/.env.production
   ```
   Expected output: `VITE_API_BASE_URL=` (empty string)

2. Check package.json build scripts:
   ```bash
   grep -A2 '"build":' frontend/package.json
   ```
   Expected: Contains `"build": "run-p type-check \"build-only {@}\" --"` and `"build-only": "vite build"`

3. Verify current dist/ timestamp (optional):
   ```bash
   ls -la frontend/dist/index.html 2>/dev/null || echo "No dist/index.html found"
   ```
   
The environment configuration should already be correct from the debug session fix.
  </action>
  <verify>
    <automated>
      grep -q "^VITE_API_BASE_URL=$" frontend/.env.production && \
      grep -q '"build":' frontend/package.json && \
      grep -q '"build-only": "vite build"' frontend/package.json && \
      echo "✅ Environment configuration verified"
    </automated>
  </verify>
  <done>
Environment configuration verified:
- .env.production has empty VITE_API_BASE_URL (triggers relative API paths)
- package.json has correct build scripts
- Ready for rebuild
  </done>
</task>

<task type="auto">
  <name>Task 2: Run frontend build</name>
  <files>
    - frontend/dist/
  </files>
  <read_first>
    - frontend/.env.production
    - frontend/package.json
  </read_first>
  <action>
Execute the frontend build command to regenerate dist/ with updated environment variables.

Run:
```bash
cd frontend && npm run build
```

This command:
- Runs TypeScript type check in parallel with build
- Uses vite build which loads .env.production automatically
- Creates fresh dist/ directory with optimized assets
- Embeds empty VITE_API_BASE_URL (relative API paths) into the JavaScript bundle

Expected output includes:
```
dist/index.html                   0.42 kB │ gzip:  0.28 kB
dist/assets/index-*.css          15.66 kB │ gzip:  3.72 kB  
dist/assets/index-*.js          113.90 kB │ gzip: 43.68 kB

✓ built in Xms
```

Note: dist/ directory is gitignored (as expected for build artifacts).
  </action>
  <verify>
    <automated>
      cd frontend && npm run build && \
      test -f dist/index.html && \
      test -d dist/assets && \
      ls dist/assets/index-*.js >/dev/null 2>&1 && \
      ls dist/assets/index-*.css >/dev/null 2>&1 && \
      echo "✅ Frontend build completed successfully"
    </automated>
  </verify>
  <done>
Frontend build completed:
- dist/index.html created
- dist/assets/ directory contains hashed JS and CSS bundles
- Build output optimized (minified, tree-shaken, code-split)
- Environment variables from .env.production embedded in bundle
  </done>
</task>

<task type="auto">
  <name>Task 3: Verify built JavaScript does not contain localhost:5000</name>
  <files>
    - frontend/dist/assets/index-*.js
  </files>
  <read_first>
    - frontend/dist/assets/index-*.js
  </read_first>
  <action>
Verify that the rebuilt JavaScript bundle no longer contains hardcoded localhost:5000, which was causing CORS errors.

Run:
```bash
grep -n "localhost:5000" frontend/dist/assets/index-*.js
```

Expected result: No output (localhost:5000 not found in built JavaScript)

Also verify that baseURL is empty or relative:
```bash
grep -o 'baseURL:"[^"]*"' frontend/dist/assets/index-*.js | head -1
```

Expected output: `baseURL:""` or `baseURL:"/"` or similar (not `baseURL:"http://localhost:5000"`)

This confirms:
1. The build used the updated .env.production (empty VITE_API_BASE_URL)
2. No hardcoded localhost:5000 remains in the bundle
3. CORS issue should be resolved when Flask serves from IP address
  </action>
  <verify>
    <automated>
      ! grep -q "localhost:5000" frontend/dist/assets/index-*.js && \
      echo "✅ No localhost:5000 found in built JavaScript" || \
      (echo "❌ localhost:5000 still found in JavaScript" && exit 1)
    </automated>
  </verify>
  <done>
Built JavaScript verified:
- No hardcoded localhost:5000 found in bundle
- baseURL should be empty or relative (not absolute localhost URL)
- CORS issue root cause removed from built assets
- Ready for testing with Flask serving from IP address
  </done>
</task>

<task type="auto">
  <name>Task 4: Update debug session with rebuild status</name>
  <files>
    - .planning/debug/file-upload-cors-issue.md
  </files>
  <read_first>
    - .planning/debug/file-upload-cors-issue.md
  </read_first>
  <action>
Update the debug session file to document that frontend has been rebuilt with empty VITE_API_BASE_URL.

Edit .planning/debug/file-upload-cors-issue.md:
1. Add a new evidence entry with timestamp
2. Update status to "verifying" or "resolved" based on rebuild success
3. Add verification steps for user testing

Add to the Evidence section:
```yaml
- timestamp: 2026-04-04T[CurrentTime]Z
  checked: Frontend rebuild with empty VITE_API_BASE_URL
  found: npm run build completed successfully, built JavaScript no longer contains localhost:5000
  implication: CORS issue should be resolved - file upload should work on first attempt
```

Update the status field in frontmatter if needed:
```yaml
status: verifying  # Changed from awaiting_human_verify
```

Also add a note about next verification steps:
- Start Flask server
- Access via IP address (192.168.2.13:5000)
- Test file upload - should work on first attempt
- Check browser console for CORS errors (should be none)
  </action>
  <verify>
    <automated>
      test -f .planning/debug/file-upload-cors-issue.md && \
      echo "✅ Debug session file exists and will be updated"
    </automated>
  </verify>
  <done>
Debug session updated:
- Added evidence entry for frontend rebuild
- Updated status to reflect rebuild completion
- Documented verification steps for user testing
- Ready for final verification by user
  </done>
</task>

</tasks>

<verification>
After completing all tasks, verify the fix:

1. Start Flask server:
   ```bash
   python src/run/app.py
   ```

2. Access application via IP address:
   - http://192.168.2.13:5000

3. Test file upload:
   - Select image or video file
   - Should work on first attempt (no CORS error)
   - Should not require second selection

4. Check browser console:
   - No CORS errors
   - API requests succeed with 200 responses

5. Expected behavior:
   - File upload works immediately
   - Detection proceeds normally
   - Results display correctly
</verification>

<success_criteria>
- [ ] .env.production verified with empty VITE_API_BASE_URL
- [ ] Frontend build completed successfully
- [ ] Built JavaScript does not contain localhost:5000
- [ ] Debug session updated with rebuild status
- [ ] Ready for user verification testing
</success_criteria>

<output>
After completion, create SUMMARY.md in the quick task directory with:
- Build execution results
- Verification of no localhost:5000 in JavaScript
- Next steps for user verification
- Link to updated debug session
</output>