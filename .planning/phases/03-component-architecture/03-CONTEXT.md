# Phase 3: Component Architecture - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

## Phase Boundary

Create Vue 3 components for all three panels (Camera, Image, Video) and a shared FileInput component. These components will have defined prop interfaces and emit events for parent handling. They render within the existing Card components from Phase 2.

## Implementation Decisions

### Component Interface Design

- **D-01:** All panels use `title` and `description` props (pass through to Card wrapper)
- **D-02:** CameraPanel emits `@start` with `{cameraId}` and `@stop` events (no parameters)
- **D-03:** ImagePanel and VideoPanel emit `@detect` with `{file}` event on file selection
- **D-04:** No direct API calls in components — emit events to parent (App.vue) for Phase 4 integration
- **D-05:** PreviewArea used within all panels via slot or prop composition

### File Input Implementation

- **D-06:** Use standard HTML `<input type="file">` wrapped in a styled component
- **D-07:** Add visual drag-drop zone overlay (using existing PreviewArea with dashed border)
- **D-08:** FileInput component accepts `accept` prop (e.g., "image/*", "video/*") and emits `@change` with selected file

### Button Interaction Patterns

- **D-09:** Primary buttons use `bg-accent-600 hover:bg-accent-700` (from Phase 2 D-02)
- **D-10:** Secondary buttons (stop camera) use `bg-white dark:bg-gray-800 text-accent-600 border border-gray-200`
- **D-11:** Loading state: button gets `disabled` attribute and `opacity-50` class, text changes to "处理中..." or "检测中..."
- **D-12:** Buttons auto-disable when `processing` prop is true

### Data Flow Architecture

- **D-13:** State lifted to App.vue parent (statusMessage, statusIsOk, processing flags)
- **D-14:** Components emit events; parent handles API calls and updates state (Phase 4)
- **D-15:** CameraPanel accepts optional `cameraId` prop for pre-selected camera

### Error Handling UI

- **D-16:** Inline error messages displayed below buttons using text-red-500 in dark mode
- **D-17:** Critical errors also update StatusMonitor via parent state
- **D-18:** FileInput shows validation message for invalid file types

### Claude's Discretion

- Exact button sizing and spacing (use existing patterns from Phase 2)
- Icon choices for file input drop zones
- Empty state messaging within preview areas
- Hover state transitions and animations

## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Original Frontend Structure
- `frontend/index.html` lines 156-192 — Complete reference for all three panels, input types, buttons, and JavaScript event handlers

### Phase 2 Established Components
- `frontend-vue/src/components/ui/Card.vue` — Card wrapper with title, description, content slot
- `frontend-vue/src/components/ui/PreviewArea.vue` — Dashed border container for image/video results
- `frontend-vue/src/components/layout/StatusMonitor.vue` — Status bar component

### Phase 2 Design Decisions
- `.planning/phases/02-styling-foundation/02-CONTEXT.md` — All D-01 through D-12 locked decisions (colors, typography, spacing)

### Requirements Mapping
- `.planning/REQUIREMENTS.md` — COMP-01 through COMP-08 define the success criteria for this phase

## Existing Code Insights

### Reusable Assets

- **Card component**: Accepts `title`, `description` props and default slot — perfect wrapper for panels
- **PreviewArea component**: Has `minHeight` prop and gradient background — use for all result displays
- **StatusMonitor component**: Accepts `message` and `isOk` props — already wired in App.vue
- **Lucide icons**: Camera, Image, Video icons available for panel headers or drop zones

### Established Patterns

- **Component composition**: Vue 3 with `<script setup lang="ts">` and TypeScript
- **Dark mode styling**: All classes use `dark:` prefix for dark mode variants
- **Emits pattern**: Components emit events; parent handles state and side effects
- **Slot-based content**: Card uses `<slot>` for flexible content injection

### Integration Points

- **App.vue**: Has three Card components with placeholder content — replace with CameraPanel, ImagePanel, VideoPanel
- **StatusMonitor**: Already connected to `statusMessage` and `statusIsOk` reactive state
- **Phase 4**: Will connect API client (`src/api/client.ts`) to handle emitted events

## Specific Ideas

- Original frontend shows exact Chinese labels: "摄像头编号", "启动摄像头", "停止摄像头", "开始图片检测", "开始视频检测"
- Meta displays person count ("检测到行人数：{count}") and video stats ("总帧数：{frames}，平均 FPS：{avg_fps}")
- Status messages in Chinese: "系统就绪", "摄像头 {N} 已启动", "正在进行图片检测...", "图片检测完成"

## Deferred Ideas

- **Video playback controls**: Original uses `<video controls>` — Phase 3 implements basic display, advanced controls can be added later
- **File size validation**: Not in original frontend — defer to Phase 4/5 if needed
- **Camera selection dropdown**: Original uses number input — keep for now, dropdown can be v2 enhancement

---

*Phase: 03-component-architecture*
*Context gathered: 2026-04-03 via auto-mode*
