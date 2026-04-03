# Phase 4: API & State Layer - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

## Phase Boundary

Integrate Flask backend API endpoints with reactive state management in App.vue. Implement actual API calls in event handlers (replacing console.log placeholders), update state when responses arrive, and handle errors appropriately.

## Implementation Decisions

### Camera Streaming

- **D-01:** Use MJPEG stream from `/video_feed` endpoint (same as original frontend)
- **D-02:** Set img src directly to `/video_feed?camera_id={id}&t={timestamp}` in handleCameraStart
- **D-03:** Clear stream by setting cameraStreamUrl to undefined in handleCameraStop

### Error Handling

- **D-04:** Use hybrid error display approach — inline errors below buttons + StatusMonitor for critical errors
- **D-05:** No auto-retry on API failures — fail immediately and show error message
- **D-06:** Catch block sets inline error message (imageErrorMessage/videoErrorMessage) and updates StatusMonitor
- **D-07:** Use try-catch blocks in async event handlers for proper error catching

### State Updates

- **D-08:** Use async/await pattern for all event handlers (handleImageDetect, handleVideoDetect)
- **D-09:** Use standard pattern: set loading state before API call, update result refs after successful response
- **D-10:** Use existing detectImage and detectVideo helper functions from `src/api/client.ts`
- **D-11:** API response data mapped to result refs: imageResultUrl, imagePersonCount, videoResultUrl, videoStats

### Loading States

- **D-12:** Use existing `processing.camera/image/video` object already defined in App.vue
- **D-13:** Set processing flag to true before API call, false after completion (in finally block)
- **D-14:** Processing flag controls button disabled state in child components

### State Update Flow

- **D-15:** Camera: Set cameraStreamUrl on start, clear on stop; update statusMessage with camera ID
- **D-16:** Image: Set processing.image → true → call detectImage() → update result refs → set processing.image → false
- **D-17:** Video: Set processing.video → true → call detectVideo() → update result refs → set processing.video → false
- **D-18:** All operations update statusMessage and statusIsOk for user feedback

### Claude's Discretion

- Exact timeout values for video processing (60s configured in uploadClient)
- Error message text for different failure scenarios
- Status bar update timing (immediate on state change, not deferred)

## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### API Integration Patterns
- `frontend/index.html` lines 218-253 — Complete reference for JavaScript event handlers and API calls
- `frontend-vue/src/api/client.ts` — Axios instances with interceptors, helper functions (detectImage, detectVideo)
- `frontend-vue/src/api/types.ts` — Response type definitions (DetectionResponse, VideoStats, ApiError)

### Phase 3 Established Components
- `frontend-vue/src/components/panels/CameraPanel.vue` — Camera panel with @start/@stop events
- `frontend-vue/src/components/panels/ImagePanel.vue` — Image panel with @detect event
- `frontend-vue/src/components/panels/VideoPanel.vue` — Video panel with @detect event
- `frontend-vue/src/components/layout/StatusMonitor.vue` — Status bar component

### Phase 3 Design Decisions
- `.planning/phases/03-component-architecture/03-CONTEXT.md` — D-13 through D-18: State lifted to parent, emit-based architecture, error handling UI

### Requirements Mapping
- `.planning/REQUIREMENTS.md` — API-01 through API-05, STATE-01 through STATE-05 define success criteria

## Existing Code Insights

### Reusable Assets

- **apiClient/uploadClient**: Already configured with base URL, timeout, interceptors
- **detectImage(file)**: Type-safe helper that posts to `/detect/image` and returns DetectionResponse
- **detectVideo(file)**: Type-safe helper that posts to `/detect/video` and returns DetectionResponse
- **Processing flags**: `processing.camera/image/video` object already declared in App.vue
- **Result refs**: cameraStreamUrl, imageResultUrl, imagePersonCount, videoResultUrl, videoStats already declared

### Established Patterns

- **Event-based architecture**: Components emit typed events (@start, @stop, @detect), parent handles API calls
- **Reactive state**: All state is Vue refs that trigger reactivity when updated
- **Error state**: Inline error refs (imageErrorMessage, videoErrorMessage) for panel-specific errors
- **Status updates**: statusMessage and statusIsOk refs update StatusMonitor component

### Integration Points

- **App.vue event handlers**: Replace console.log placeholders with actual API calls
- **CameraPanel**: streamUrl prop bound to cameraStreamUrl ref
- **ImagePanel**: processing, resultUrl, personCount, errorMessage props bound to App.vue refs
- **VideoPanel**: processing, resultUrl, videoStats, errorMessage props bound to App.vue refs
- **Phase 5**: Will add real camera/device detection (out of scope for this phase)

## Specific Ideas

- Original frontend JavaScript shows exact API patterns to replicate
- Status messages in Chinese: "正在进行图片检测...", "正在进行视频检测，耗时可能较长，请稍候"
- Success messages: "摄像头 {N} 已启动", "图片检测完成", "视频检测完成"
- Error messages: "请先选择一张图片", "请先选择一个视频文件"
- Timestamp query param (`&t=${Date.now()}`) prevents browser caching for video feed

## Deferred Ideas

- **Auto-retry logic**: Not implementing automatic retry on network failure — user manually retries
- **Request cancellation**: No abort controller for cancelling in-flight requests
- **Progress indicators**: No upload progress bars for large files
- **Connection pooling**: Single Axios instance is sufficient for this use case

---

*Phase: 04-api-state-layer*
*Context gathered: 2026-04-03*
