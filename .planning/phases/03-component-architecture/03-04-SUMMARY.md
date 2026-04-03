---
phase: 03-component-architecture
plan: 04
subsystem: Video Panel Component
tags: [component, video, file-upload, emit-pattern]
completion_date: 2026-04-03
duration_seconds: 45
tasks_completed: 1
requirements_satisfied: [COMP-04, COMP-07]
---

# Phase 03 Plan 04: VideoPanel Component Summary

**One-liner:** Video detection panel component with FileInput composition, emit-based state management, and native video player with statistics display.

## What Was Built

Created `VideoPanel.vue` component implementing video file upload UI with detection triggering, video result preview, and statistics display. Component follows Phase 3 decisions (D-01 through D-18) and emits events to parent for Phase 4 API integration.

## Files Modified

| File | Lines | Purpose |
| ---- | ----- | ------- |
| `frontend-vue/src/components/panels/VideoPanel.vue` | 104 | Video panel component with upload, preview, and stats |

## Component Structure

**TypeScript Interfaces:**
- `VideoStats`: `{ frames: number; avg_fps: number }` - Video processing statistics
- `Props`: Component props with sensible defaults for all fields

**Props:**
- `title`: string (default: "离线视频分析")
- `description`: string (default: "上传视频并输出带跟踪轨迹与标注的结果视频。")
- `processing`: boolean (default: false) - Disables button during detection
- `resultUrl`: string (optional) - URL for processed video result
- `videoStats`: VideoStats (optional) - Statistics display
- `errorMessage`: string (optional) - Inline error message

**Events:**
- `@detect(file: File)`: Emitted when user clicks detect button with selected file

**Component Composition:**
- Card wrapper component for title/description layout
- FileInput component with `accept="video/*"` for file selection
- PreviewArea component with `minHeight="280px"` for video display
- Native HTML5 video element with controls attribute

## Implementation Details

**File Upload Flow:**
1. User selects video file via FileInput component
2. Local `selectedFile` ref tracks the chosen file
3. Detect button enables only when file is selected and not processing
4. Clicking detect emits `@detect` event with File payload to parent
5. FileInput resets via exposed `reset()` method after detection starts

**Loading States:**
- Button disabled when `processing=true` or no file selected
- Button text changes to "检测中..." during processing
- Button opacity reduced to 50% when disabled
- Cursor changes to `not-allowed` for disabled state

**Result Display:**
- Conditional rendering: video element when `resultUrl` provided, placeholder text otherwise
- Video element uses native browser controls (play/pause, volume, timeline)
- Object-contain ensures video fits within preview area
- Statistics display shows frame count and average FPS when provided

**Error Handling:**
- Inline error message below button when `errorMessage` prop provided
- Red color with dark mode variant (`text-red-500 dark:text-red-400`)
- Error message conditionally rendered via `v-if="errorMessage"`

**Styling:**
- All styling via Tailwind utility classes
- Primary button: `bg-accent-600 hover:bg-accent-700` (per D-09)
- Dark mode variants on all color classes
- Spacing via `space-y-3` for vertical rhythm
- Statistics in gray-50 container with border

## Decision Adherence

| Decision | Implementation |
|----------|----------------|
| D-01 | Card wrapper with title/description props passed through |
| D-03 | Emit `@detect` event with File payload |
| D-04 | No direct API calls — all logic emits to parent |
| D-08 | FileInput with `accept="video/*"` prop |
| D-09 | Primary button with bg-accent-600 hover:bg-accent-700 |
| D-11 | Loading state shows "检测中..." text |
| D-12 | Button auto-disabled when processing prop is true |
| D-13 | State lifted to App.vue parent via emit pattern |
| D-16 | Inline error messages with text-red-500 dark:text-red-400 |

## Deviations from Plan

**None — plan executed exactly as written.**

## Known Stubs

**None — component is complete and ready for Phase 4 integration.**

## Integration Points

**Phase 4 (API & State Layer):**
- Parent App.vue will handle `@detect` event to call `/detect/video` endpoint
- `resultUrl` prop will be populated with processed video URL from backend
- `videoStats` prop will receive statistics from API response
- `processing` prop will be set by parent during API call
- `errorMessage` prop will be populated from API error responses

**Component Dependencies:**
- FileInput component (Plan 03-01) — ✓ Exists
- Card component (Phase 2) — ✓ Exists
- PreviewArea component (Phase 2) — ✓ Exists

## Technical Context

**Video Element:**
- Uses `controls` attribute for native browser playback controls
- `:src` binding allows dynamic URL updates when resultUrl changes
- No auto-play — user controls playback via native controls
- Advanced playback controls deferred to v2 per CONTEXT.md

**File Input Reset:**
- Component ref (`fileInputRef`) provides access to FileInput methods
- `reset()` method called after detection starts to allow re-selecting same file
- FileInput exposes reset method via `defineExpose()` in its implementation

**TypeScript Type Contract:**
- `VideoStats` interface defines shape of statistics data
- Matches original frontend structure from `frontend/index.html` line 280
- Used by both VideoPanel and will be used by App.vue for type safety

## Verification

**Build Verification:**
```bash
cd frontend-vue && npm run build
```
Result: ✓ Build succeeded without errors

**Component Structure Verification:**
- ✓ TypeScript interfaces defined with proper types
- ✓ Card wrapper component composition
- ✓ FileInput component with accept="video/*"
- ✓ Detect button with proper disabled state logic
- ✓ Emit definition with File payload type
- ✓ Error message with dark mode styling
- ✓ PreviewArea with minHeight="280px"
- ✓ Video element with controls and :src binding
- ✓ Video statistics display with Chinese format
- ✓ All color classes have dark mode variants

## Performance Metrics

| Metric | Value |
| ------ | ----- |
| Task Duration | 45 seconds |
| Files Created | 1 |
| Lines Added | 104 |
| Build Time | 646ms |
| TypeScript Compilation | ✓ Passed |

## Commit Information

**Commit:** `ac5d051`
**Message:** `feat(03-04): create VideoPanel component with file upload and processing controls`

---

*Plan completed: 2026-04-03*
*Phase: 03-component-architecture*
*Auto-execution mode: Active*
