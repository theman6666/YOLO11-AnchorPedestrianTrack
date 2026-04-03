---
phase: 03-component-architecture
plan: 02
subsystem: Frontend Components
tags: [vue3, typescript, component-architecture, camera-panel]
dependency_graph:
  requires:
    - "03-01: UI components (Card, PreviewArea)"
  provides:
    - "CameraPanel component for camera streaming UI"
    - "Typed emit pattern for panel components"
  affects:
    - "03-03: ImagePanel (will use CameraPanel as pattern reference)"
    - "03-04: VideoPanel (will use CameraPanel as pattern reference)"
    - "03-05: App.vue integration (will import and wire CameraPanel)"
tech_stack:
  added: []
  patterns:
    - "Vue 3 Composition API with TypeScript"
    - "Typed props with interface declaration"
    - "Typed emits with generic syntax"
    - "Component composition (Card + PreviewArea)"
    - "Emit-based state management pattern"
key_files:
  created:
    - path: "frontend-vue/src/components/panels/CameraPanel.vue"
      exports: ["CameraPanel"]
      props: ["title", "description", "processing", "streamUrl", "cameraId"]
      emits: ["start(cameraId: number)", "stop()"]
  modified: []
decisions: []
metrics:
  duration: "15 seconds"
  completed_date: "2026-04-03"
  tasks_completed: 1
  files_created: 1
  lines_of_code: 100
---

# Phase 03 Plan 02: CameraPanel Component Summary

**One-liner:** Camera streaming panel component with camera ID input, start/stop controls, and video preview area using emit-based state management.

## Objective

Create CameraPanel component with camera ID input, start/stop buttons, and video stream preview area, following the emit-based state management pattern defined in CONTEXT.md.

**Purpose:** Implement the camera streaming UI panel that emits control events to parent (App.vue) for Phase 4 API integration, following D-02 and D-13 decisions for state lifting.

## Implementation Summary

### What Was Built

**Component:** `frontend-vue/src/components/panels/CameraPanel.vue`

Created the first panel component in the component architecture, establishing patterns for ImagePanel and VideoPanel:

1. **TypeScript Props Interface**
   - `title?: string` - Panel title (default: "摄像头实时检测")
   - `description?: string` - Panel description (default: "实时视频流分析，画面叠加跟踪 ID 与 FPS。")
   - `processing?: boolean` - Disable controls during API calls (default: false)
   - `streamUrl?: string` - Video stream URL for img src binding
   - `cameraId?: number` - Optional pre-selected camera ID (per D-15)

2. **Typed Emits**
   - `@start(cameraId: number)` - Emit when start button clicked with camera ID
   - `@stop()` - Emit when stop button clicked with no parameters

3. **UI Components**
   - Camera ID number input with v-model.number binding
   - Primary start button (bg-accent-600 hover:bg-accent-700 per D-09)
   - Secondary stop button (bg-white dark:bg-gray-800 per D-10)
   - PreviewArea with conditional img/span rendering
   - Loading state text "处理中..." (per D-11)

4. **Styling**
   - All text has dark mode variants (dark: prefix)
   - Input has focus ring with accent-500 color
   - Buttons auto-disable when processing=true (per D-12)
   - Object-contain on img ensures stream fits preview area

### Component Structure

```
frontend-vue/src/components/panels/
└── CameraPanel.vue (NEW - 100 lines)
```

## Deviations from Plan

### Auto-fixed Issues

None - plan executed exactly as written. All implementation details matched the PLAN.md specification:

- TypeScript Props interface with all required fields ✓
- Card wrapper component composition ✓
- Camera ID number input with v-model.number ✓
- Start/stop buttons with proper emit handlers ✓
- Button disable logic when processing=true ✓
- Primary/secondary button styling per D-09/D-10 ✓
- PreviewArea with conditional img/span rendering ✓
- Stream URL binding on img element ✓
- Dark mode styling on all text elements ✓
- Input focus ring styling ✓

## Technical Details

### Component Composition Pattern

CameraPanel follows the established pattern from Phase 2 UI components:

1. **Card Wrapper** - Provides title, description, and content container
2. **PreviewArea** - Displays video stream or placeholder
3. **Emit-based Events** - State lifted to parent per D-13/D-14

### Key Implementation Decisions

1. **Number Input Binding**
   - Used `v-model.number` for proper type coercion
   - Input type="number" with min="0" validation
   - Local ref for camera ID with fallback to 0

2. **Button Styling**
   - Primary: bg-accent-600 hover:bg-accent-700 (Phase 2 D-02)
   - Secondary: bg-white dark:bg-gray-800 with text-accent-600 border (D-10)
   - Disabled state: opacity-50 + cursor-not-allowed

3. **Emit Payloads**
   - Start emits: cameraId as number (per D-02)
   - Stop emits: no parameters (per D-02)
   - Used generic syntax for type safety: `defineEmits<{ start: [cameraId: number] }>`

### Build Verification

Build completed successfully without errors:
- TypeScript compilation: ✓ PASSED
- Vite build: ✓ PASSED (66.69 kB JS, 15.73 kB CSS)
- No type errors or warnings

## Integration Points

### Phase 4 Dependencies

CameraPanel is ready for API integration in Phase 4:

1. **Stream URL Binding**
   - Parent (App.vue) will provide `streamUrl` prop
   - Will bind to Flask `/video_feed?camera_id={N}` endpoint
   - img element has `:src="streamUrl"` ready for binding

2. **Event Handlers**
   - Parent will implement `@start` handler to call API
   - Parent will implement `@stop` handler to stop stream
   - Processing flag will be set during API calls

3. **State Management**
   - Status messages will update StatusMonitor via parent state
   - Processing flags will disable buttons during API calls

### Pattern Reuse

CameraPanel establishes the pattern for:

- **03-03 (ImagePanel):** Similar structure with FileInput instead of number input
- **03-04 (VideoPanel):** Similar structure with FileInput instead of number input
- **03-05 (App.vue):** Will import all three panels with emit handlers

## Verification

### Automated Verification

Build succeeded with no errors:
```bash
cd frontend-vue && npm run build
# ✓ built in 678ms
```

### Manual Verification (Optional)

Component renders visibly in browser:
```bash
cd frontend-vue && npm run dev
# Visit http://localhost:5173
# CameraPanel visible with title, description, input, buttons, preview area
```

## Known Stubs

None - component is fully functional with no placeholder values or TODOs. All features are implemented and ready for Phase 4 API integration.

## Files Modified

### Created
- `frontend-vue/src/components/panels/CameraPanel.vue` (100 lines)

## Commits

| Commit | Hash | Message |
|--------|------|---------|
| Task 1 | 3601df2 | feat(03-02): create CameraPanel component with camera controls and emits |

## Self-Check: PASSED

- [x] Created file exists: `frontend-vue/src/components/panels/CameraPanel.vue`
- [x] Commit exists: `3601df2`
- [x] Build succeeds: `npm run build` completed without errors
- [x] Component has all required props and emits
- [x] TypeScript types compile without errors
- [x] All styling uses Tailwind utility classes
- [x] Dark mode variants present on all text elements
- [x] Button disable logic implemented correctly

## Next Steps

**Immediate Next Plan:** 03-03 - Create ImagePanel component
- Will reuse CameraPanel structure with FileInput for image upload
- Will emit @detect with file payload instead of @start/@stop
- Will show result image in PreviewArea instead of stream

**Phase Completion:** After 03-05 (App.vue integration)
- All three panels will be wired in App.vue
- Emit handlers will be stubbed for Phase 4 API integration
- StatusMonitor will be connected to panel state
