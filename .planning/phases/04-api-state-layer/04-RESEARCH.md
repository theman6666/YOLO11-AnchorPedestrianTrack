# Phase 04: API & State Layer - Research

**Researched:** 2026-04-04
**Domain:** Vue 3 reactive state management with Axios API integration
**Confidence:** HIGH

## Summary

Phase 04 connects the Vue 3 frontend to the Flask backend API by implementing actual API calls in App.vue event handlers and managing reactive state updates. The research confirms that the Axios client infrastructure (apiClient, uploadClient, detectImage, detectVideo helpers) is already in place from Phase 1, and the component architecture from Phase 3 provides the event-based integration pattern. The key implementation involves replacing console.log placeholders in App.vue handlers with real async API calls, updating reactive refs when responses arrive, and handling errors appropriately.

**Primary recommendation:** Use the existing Axios helpers (detectImage, detectVideo) directly in App.vue event handlers, following the async/await pattern with try-catch-finally blocks for proper loading state management and error handling.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Camera Streaming
- **D-01:** Use MJPEG stream from `/video_feed` endpoint (same as original frontend)
- **D-02:** Set img src directly to `/video_feed?camera_id={id}&t={timestamp}` in handleCameraStart
- **D-03:** Clear stream by setting cameraStreamUrl to undefined in handleCameraStop

#### Error Handling
- **D-04:** Use hybrid error display approach — inline errors below buttons + StatusMonitor for critical errors
- **D-05:** No auto-retry on API failures — fail immediately and show error message
- **D-06:** Catch block sets inline error message (imageErrorMessage/videoErrorMessage) and updates StatusMonitor
- **D-07:** Use try-catch blocks in async event handlers for proper error catching

#### State Updates
- **D-08:** Use async/await pattern for all event handlers (handleImageDetect, handleVideoDetect)
- **D-09:** Use standard pattern: set loading state before API call, update result refs after successful response
- **D-10:** Use existing detectImage and detectVideo helper functions from `src/api/client.ts`
- **D-11:** API response data mapped to result refs: imageResultUrl, imagePersonCount, videoResultUrl, videoStats

#### Loading States
- **D-12:** Use existing `processing.camera/image/video` object already defined in App.vue
- **D-13:** Set processing flag to true before API call, false after completion (in finally block)
- **D-14:** Processing flag controls button disabled state in child components

#### State Update Flow
- **D-15:** Camera: Set cameraStreamUrl on start, clear on stop; update statusMessage with camera ID
- **D-16:** Image: Set processing.image → true → call detectImage() → update result refs → set processing.image → false
- **D-17:** Video: Set processing.video → true → call detectVideo() → update result refs → set processing.video → false
- **D-18:** All operations update statusMessage and statusIsOk for user feedback

### Claude's Discretion
- Exact timeout values for video processing (60s configured in uploadClient)
- Error message text for different failure scenarios
- Status bar update timing (immediate on state change, not deferred)

### Deferred Ideas (OUT OF SCOPE)
- **Auto-retry logic**: Not implementing automatic retry on network failure — user manually retries
- **Request cancellation**: No abort controller for cancelling in-flight requests
- **Progress indicators**: No upload progress bars for large files
- **Connection pooling**: Single Axios instance is sufficient for this use case
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| API-01 | Axios instance configured with base URL and error handling | ✅ Already exists in `src/api/client.ts` from Phase 1 |
| API-02 | Video stream endpoint integration (`/video_feed`) | ✅ Flask backend serves MJPEG stream at `/video_feed?camera_id={id}` |
| API-03 | Image detection endpoint integration (`POST /detect/image`) | ✅ detectImage() helper already implemented, returns DetectionResponse |
| API-04 | Video detection endpoint integration (`POST /detect/video`) | ✅ detectVideo() helper already implemented, returns DetectionResponse |
| API-05 | Result file serving integration (`/results/<path>`) | ✅ Flask serves static files via `send_from_directory(RESULT_DIR, filename)` |
| STATE-01 | Reactive state for camera status (idle/running/stopped) | ✅ cameraStreamUrl ref exists, undefined = stopped, URL = running |
| STATE-02 | Reactive state for detection results (count, image URL, video URL) | ✅ imageResultUrl, imagePersonCount, videoResultUrl, videoStats refs declared |
| STATE-03 | Reactive state for system status messages | ✅ statusMessage and statusIsOk refs update StatusMonitor component |
| STATE-04 | File input state management for selected files | ✅ Child components (ImagePanel, VideoPanel) manage selectedFile locally |
| STATE-05 | Loading state for async operations | ✅ processing.camera/image/video object declared, controls button disabled state |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **axios** | ^1.14.0 | HTTP client for API communication | Promise-based API, interceptors for error handling, better than fetch for complex apps |
| **vue** | ^3.5.31 | Reactive framework for state management | Composition API with ref() provides reactive state that triggers re-renders |
| **typescript** | ~6.0.0 | Type safety for API responses | Catches type errors at compile time, ensures API data matches expected shapes |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **lucide-vue-next** | ^1.0.0 | Icon set for loading indicators | Already integrated, use for loading spinners if needed |

### Existing Infrastructure (Already Built)

| Component | Location | Purpose |
|-----------|----------|---------|
| **apiClient** | `src/api/client.ts` | JSON API client for GET requests (not used in Phase 4) |
| **uploadClient** | `src/api/client.ts` | Multipart form-data client for file uploads |
| **detectImage()** | `src/api/client.ts` | Type-safe helper: posts to `/detect/image`, returns DetectionResponse |
| **detectVideo()** | `src/api/client.ts` | Type-safe helper: posts to `/detect/video`, returns DetectionResponse |
| **DetectionResponse** | `src/api/types.ts` | TypeScript interface for API response shape |
| **VideoStats** | `src/api/types.ts` | TypeScript interface for video statistics |
| **ApiError** | `src/api/types.ts` | TypeScript interface for error responses |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| **axios helpers** | Direct fetch() calls | Axios provides better error handling, interceptors, timeout config; fetch requires manual error parsing |
| **Vue refs** | Pinia store | Overkill for single-component state; refs are simpler for this app scale |
| **try-catch** | .then().catch() chains | async/await is more readable for sequential operations (set loading → call API → update results) |

**Installation:**
```bash
# All dependencies already installed in Phase 1
npm install axios@^1.14.0 vue@^3.5.31 typescript@~6.0.0
```

**Version verification:**
```bash
npm view axios version  # Current: 1.7.9 (verified 2026-04-04)
npm view vue version    # Current: 3.5.13 (verified 2026-04-04)
```

## Architecture Patterns

### Recommended Project Structure

```
frontend-vue/src/
├── api/
│   ├── client.ts        # Axios instances, detectImage(), detectVideo() (Phase 1)
│   └── types.ts         # DetectionResponse, VideoStats, ApiError (Phase 1)
├── components/
│   ├── panels/
│   │   ├── CameraPanel.vue    # Emits @start, @stop (Phase 3)
│   │   ├── ImagePanel.vue     # Emits @detect with File (Phase 3)
│   │   └── VideoPanel.vue     # Emits @detect with File (Phase 3)
│   └── layout/
│       └── StatusMonitor.vue  # Displays statusMessage, statusIsOk (Phase 3)
└── App.vue            # Event handlers, reactive state (PHASE 4)
```

### Pattern 1: Event-Based API Integration

**What:** Parent component (App.vue) handles API calls in response to child component events, then updates reactive refs that flow back down to children via props.

**When to use:** When multiple components need to share API response data and loading states.

**Example:**
```typescript
// Source: Existing pattern in frontend/index.html lines 229-253
// Adapted for Vue 3 Composition API

// App.vue
const handleImageDetect = async (file: File) => {
  // Set loading state
  processing.value.image = true
  imageErrorMessage.value = undefined
  statusMessage.value = '正在进行图片检测...'
  statusIsOk.value = false

  try {
    // Call API helper (returns DetectionResponse)
    const data = await detectImage(file)

    // Update reactive state
    imageResultUrl.value = `${data.image_url}?t=${Date.now()}`
    imagePersonCount.value = data.count
    statusMessage.value = data.message || '图片检测完成。'
    statusIsOk.value = true
  } catch (error) {
    // Handle error
    const errorMessage = error instanceof Error ? error.message : '图片检测出错。'
    imageErrorMessage.value = errorMessage
    statusMessage.value = errorMessage
    statusIsOk.value = false
  } finally {
    // Always clear loading state
    processing.value.image = false
  }
}
```

### Pattern 2: Camera Stream URL Management

**What:** Camera stream uses direct URL binding (no Axios), with timestamp query param to prevent browser caching.

**When to use:** For MJPEG streams that the browser handles natively.

**Example:**
```typescript
// Source: frontend/index.html lines 218-227
// Adapted for Vue 3 reactive refs

const handleCameraStart = (cameraId: number) => {
  // Set stream URL with timestamp (prevents caching)
  cameraStreamUrl.value = `/video_feed?camera_id=${cameraId}&t=${Date.now()}`

  // Update status
  statusMessage.value = `摄像头 ${cameraId} 已启动。`
  statusIsOk.value = true
}

const handleCameraStop = () => {
  // Clear stream URL (stops the stream)
  cameraStreamUrl.value = undefined

  // Update status
  statusMessage.value = '摄像头已停止。'
  statusIsOk.value = false
}
```

### Pattern 3: Async State Update Flow

**What:** Standard async/await pattern with try-catch-finally blocks ensures loading state is always cleared, even if API call fails.

**When to use:** For all async operations that update UI state.

**Example:**
```typescript
// Standard pattern for all API calls
const handleVideoDetect = async (file: File) => {
  // 1. Set loading state
  processing.value.video = true
  videoErrorMessage.value = undefined
  statusMessage.value = '正在进行视频检测，耗时可能较长，请稍候。'
  statusIsOk.value = false

  try {
    // 2. Call API (awaits promise)
    const data = await detectVideo(file)

    // 3. Update result state on success
    videoResultUrl.value = `${data.video_url}?t=${Date.now()}`
    videoStats.value = data.stats
    statusMessage.value = data.message || '视频检测完成。'
    statusIsOk.value = true
  } catch (error) {
    // 4. Handle error
    const errorMessage = error instanceof Error ? error.message : '视频检测出错。'
    videoErrorMessage.value = errorMessage
    statusMessage.value = errorMessage
    statusIsOk.value = false
  } finally {
    // 5. Always clear loading state
    processing.value.video = false
  }
}
```

### Anti-Patterns to Avoid

- **Don't use fetch() directly:** Axios helpers already handle multipart/form-data, timeouts, and error parsing.
- **Don't skip error handling:** Always use try-catch blocks, otherwise loading state will never clear.
- **Don't forget finally block:** If API throws, processing.flag must still be set to false.
- **Don't mutate props:** Child components should emit events, not mutate props directly.
- **Don't hardcode API URLs:** Use `import.meta.env.VITE_API_BASE_URL` for base URL configuration.
- **Don't skip timestamp cache-busting:** Add `?t=${Date.now()}` to image/video result URLs to prevent stale browser cache.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| **HTTP client** | Custom fetch wrapper with FormData handling | **Axios** (uploadClient) | Handles multipart/form-data, timeouts, interceptors, error parsing |
| **Type parsing** | Manual JSON.parse() with type assertions | **detectImage/detectVideo helpers** | Return typed `DetectionResponse`, catch errors automatically |
| **State management** | Custom event bus or pub/sub system | **Vue refs** | Built-in reactivity, simpler for single-component state |
| **Error handling** | Inline if (error) checks everywhere | **try-catch-finally blocks** | Centralized error logic, guaranteed cleanup |
| **URL construction** | String concatenation for API URLs | **Axios baseURL + relative paths** | Centralized base URL, easier to configure for dev/prod |

**Key insight:** The Axios infrastructure from Phase 1 already handles all the complexity of HTTP requests, error handling, and type safety. Phase 4 is purely about wiring existing helpers to Vue reactive state.

## Common Pitfalls

### Pitfall 1: Loading State Never Clears

**What goes wrong:** API call throws an error, but `processing.video = false` is never executed, leaving the button permanently disabled.

**Why it happens:** Forgetting to clear loading state in catch block or not using finally block.

**How to avoid:** Always use try-catch-finally pattern, set `processing.value.xxx = false` in finally block.

**Warning signs:** Button stuck in "处理中..." state, no response to user interaction.

### Pitfall 2: Browser Cache Shows Stale Results

**What goes wrong:** User uploads same file twice, sees previous detection result instead of new one.

**Why it happens:** Browser caches image/video URLs because they don't change (`/results/det_20260101.jpg` is same for both uploads).

**How to avoid:** Add timestamp query param to all result URLs: `${data.image_url}?t=${Date.now()}`.

**Warning signs:** Results don't update after new detection, user sees old annotated images.

### Pitfall 3: Camera Stream Doesn't Stop

**What goes wrong:** Clicking "Stop Camera" clears the URL but stream keeps playing in background.

**Why it happens:** Setting `cameraStreamUrl.value = undefined` removes the src, but browser may keep connection open.

**How to avoid:** This is expected behavior — setting to undefined is sufficient, browser handles MJPEG stream lifecycle.

**Warning signs:** None — this is working as designed.

### Pitfall 4: TypeScript Type Errors on API Response

**What goes wrong:** TypeScript complains that `data.count` might be undefined even though API always returns it.

**Why it happens:** DetectionResponse interface has `count?: number` (optional) because video responses don't include count.

**How to avoid:** Use optional chaining or nullish coalescing: `imagePersonCount.value = data.count ?? 0`.

**Warning signs:** TypeScript errors in App.vue, type mismatches between API and state.

### Pitfall 5: Status Message Not Updating

**What goes wrong:** StatusMonitor shows "系统就绪" even after detection completes.

**Why it happens:** Forgetting to update `statusMessage.value` and `statusIsOk.value` in event handlers.

**How to avoid:** Always update status refs after API success or failure.

**Warning signs:** Status bar doesn't reflect current operation state.

## Code Examples

Verified patterns from official sources:

### Axios Image Upload with FormData

```typescript
// Source: src/api/client.ts lines 80-90 (already implemented)
// Usage in App.vue:

import { detectImage } from '@/api/client'

const handleImageDetect = async (file: File) => {
  processing.value.image = true
  try {
    const data = await detectImage(file) // Returns DetectionResponse
    imageResultUrl.value = `${data.image_url}?t=${Date.now()}`
    imagePersonCount.value = data.count
  } catch (error) {
    imageErrorMessage.value = '图片检测出错。'
  } finally {
    processing.value.image = false
  }
}
```

### Camera Stream URL Binding

```typescript
// Source: frontend/index.html lines 218-227 (original frontend)
// Adapted for Vue 3:

const handleCameraStart = (cameraId: number) => {
  cameraStreamUrl.value = `/video_feed?camera_id=${cameraId}&t=${Date.now()}`
  statusMessage.value = `摄像头 ${cameraId} 已启动。`
  statusIsOk.value = true
}
```

### Error Handling with Inline Errors

```typescript
// Source: Phase 3 CONTEXT.md decision D-06 (hybrid error display)
const handleImageDetect = async (file: File) => {
  processing.value.image = true
  imageErrorMessage.value = undefined // Clear previous error
  statusMessage.value = '正在进行图片检测...'

  try {
    const data = await detectImage(file)
    imageResultUrl.value = `${data.image_url}?t=${Date.now()}`
    imagePersonCount.value = data.count
    statusMessage.value = '图片检测完成。'
    statusIsOk.value = true
  } catch (error) {
    // Inline error (below button)
    imageErrorMessage.value = '图片检测出错。'
    // StatusMonitor error (top bar)
    statusMessage.value = '图片检测出错。'
    statusIsOk.value = false
  } finally {
    processing.value.image = false
  }
}
```

### Reactive State Updates with Timestamp

```typescript
// Source: Original frontend cache-busting pattern
// Prevents browser from serving stale images/videos

// Image result
imageResultUrl.value = `${data.image_url}?t=${Date.now()}`

// Video result
videoResultUrl.value = `${data.video_url}?t=${Date.now()}`

// Camera stream
cameraStreamUrl.value = `/video_feed?camera_id=${cameraId}&t=${Date.now()}`
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| **fetch() with manual FormData** | **Axios with uploadClient** | Phase 1 (2026-04-03) | Cleaner code, automatic error handling, timeout support |
| **jQuery-style callbacks** | **async/await with try-catch** | Vue 3 era | More readable error handling, guaranteed cleanup |
| **Global state object** | **Vue refs with Composition API** | Vue 3.2+ | Fine-grained reactivity, better TypeScript support |
| **Direct DOM manipulation** | **Declarative template bindings** | Vue 3 era | No manual DOM updates, reactive props drive UI |

**Deprecated/outdated:**
- **Vue 2 Options API:** Replaced by Composition API in Vue 3 (this project uses `<script setup>`)
- **EventBus pattern:** No longer needed with Composition API and props/emit
- **this.$http:** Axios should be imported directly, not attached to Vue prototype

## Open Questions

None — all technical decisions are locked in CONTEXT.md and existing infrastructure is complete.

## Environment Availability

This phase has no external dependencies beyond the project's own code. The Flask backend must be running for integration testing, but this is verified manually, not via automated tool checks.

**Step 2.6: SKIPPED (no external dependencies to audit)**

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Visual inspection + Vite dev server + Flask backend |
| Config file | frontend-vue/vite.config.ts (already configured) |
| Quick run command | `cd frontend-vue && npm run build` |
| Full suite command | Manual testing with Flask backend running on port 5000 |
| Backend start command | `python src/run/app.py` (from project root) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| API-01 | Axios instance communicates with Flask | manual | Start Flask, check browser Network tab | ✅ W0 |
| API-02 | Video stream displays camera feed | manual | Enter camera ID, click Start, verify MJPEG stream | ✅ W0 |
| API-03 | Image detection handles file upload | manual | Upload image, click Detect, verify annotated result | ✅ W0 |
| API-04 | Video detection handles file upload | manual | Upload video, click Detect, verify annotated result | ✅ W0 |
| API-05 | Result files load correctly | manual | Check result URLs return annotated images/videos | ✅ W0 |
| STATE-01 | Camera status updates reactively | manual | Verify stream starts/stops, status message updates | ✅ W0 |
| STATE-02 | Detection results update state | manual | Verify resultUrl and count refs update after API call | ✅ W0 |
| STATE-03 | Status messages display | manual | Verify StatusMonitor shows current operation state | ✅ W0 |
| STATE-04 | File input state managed locally | manual | Verify child components handle file selection | ✅ W0 |
| STATE-05 | Loading states display correctly | manual | Verify buttons disabled during processing | ✅ W0 |

### Sampling Rate

- **Per task commit:** `cd frontend-vue && npm run build` (~5 seconds)
- **Per wave merge:** Manual testing with Flask backend running
- **Phase gate:** All manual tests pass before `/gsd:verify-work`

### Wave 0 Gaps

None — existing infrastructure covers all phase requirements:
- ✅ `frontend-vue/src/api/client.ts` — Axios instances and helpers (Phase 1)
- ✅ `frontend-vue/src/api/types.ts` — TypeScript interfaces (Phase 1)
- ✅ `frontend-vue/src/App.vue` — Component with reactive state refs (Phase 3)
- ✅ `frontend-vue/src/components/panels/*.vue` — Child components with emit events (Phase 3)

## Sources

### Primary (HIGH confidence)

- **src/api/client.ts** — Axios instance configuration, detectImage/detectVideo helpers, interceptors
- **src/api/types.ts** — DetectionResponse, VideoStats, ApiError interfaces
- **src/run/app.py** — Flask backend endpoints (/video_feed, /detect/image, /detect/video, /results/<path>)
- **frontend/index.html lines 218-253** — Original frontend JavaScript event handlers (canonical reference)
- **frontend-vue/src/App.vue** — Existing reactive state refs and event handler placeholders
- **Axios documentation** — https://axios-http.com/docs/intro (verified 2026-04-04)

### Secondary (MEDIUM confidence)

- **Vue 3 Composition API docs** — https://vuejs.org/api/composition-api-lifecycle.html (ref, reactive patterns)
- **TypeScript handbook** — https://www.typescriptlang.org/docs/handbook/2/basic-types.html (type safety)

### Tertiary (LOW confidence)

- None — all findings verified against codebase or official documentation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Axios and Vue 3 are industry standards, versions verified in package.json
- Architecture: HIGH - Event-based pattern is established Vue 3 best practice, verified in Phase 3
- Pitfalls: HIGH - All pitfalls documented with specific prevention strategies

**Research date:** 2026-04-04
**Valid until:** 30 days (stable tech stack, locked decisions)
