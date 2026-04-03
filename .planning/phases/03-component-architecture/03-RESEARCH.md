# Phase 3: Component Architecture - Research

**Researched:** 2026-04-03
**Domain:** Vue 3 Component Architecture with Composition API and TypeScript
**Confidence:** HIGH

## Summary

Phase 3 focuses on creating the complete Vue 3 component library for the YOLO11 pedestrian detection frontend. This phase builds directly on the styling foundation established in Phase 2, implementing five new components (CameraPanel, ImagePanel, VideoPanel, FileInput, and updating App.vue) that will form the core UI for camera streaming, image detection, and video analysis features.

The research confirms that Vue 3's Composition API with `<script setup lang="ts">` provides the cleanest TypeScript integration for defining typed props and emits. The established pattern from Phase 2 components (Card, PreviewArea, StatusMonitor) should be strictly followed: interface-based prop definitions, `withDefaults` for default values, and emit type declarations using generic syntax. All components should be pure presentation components that emit events to parent (App.vue), following the "state lifting" pattern mandated in CONTEXT.md decision D-13/D-14.

**Primary recommendation:** Use the existing Phase 2 components as templates for consistency, implement FileInput as a reusable wrapper around HTML `<input type="file">`, and create panel components that compose Card, PreviewArea, and FileInput together with typed event emissions.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Component Interface Design**
- **D-01:** All panels use `title` and `description` props (pass through to Card wrapper)
- **D-02:** CameraPanel emits `@start` with `{cameraId}` and `@stop` events (no parameters)
- **D-03:** ImagePanel and VideoPanel emit `@detect` with `{file}` event on file selection
- **D-04:** No direct API calls in components — emit events to parent (App.vue) for Phase 4 integration
- **D-05:** PreviewArea used within all panels via slot or prop composition

**File Input Implementation**
- **D-06:** Use standard HTML `<input type="file">` wrapped in a styled component
- **D-07:** Add visual drag-drop zone overlay (using existing PreviewArea with dashed border)
- **D-08:** FileInput component accepts `accept` prop (e.g., "image/*", "video/*") and emits `@change` with selected file

**Button Interaction Patterns**
- **D-09:** Primary buttons use `bg-accent-600 hover:bg-accent-700` (from Phase 2 D-02)
- **D-10:** Secondary buttons (stop camera) use `bg-white dark:bg-gray-800 text-accent-600 border border-gray-200`
- **D-11:** Loading state: button gets `disabled` attribute and `opacity-50` class, text changes to "处理中..." or "检测中..."
- **D-12:** Buttons auto-disable when `processing` prop is true

**Data Flow Architecture**
- **D-13:** State lifted to App.vue parent (statusMessage, statusIsOk, processing flags)
- **D-14:** Components emit events; parent handles API calls and updates state (Phase 4)
- **D-15:** CameraPanel accepts optional `cameraId` prop for pre-selected camera

**Error Handling UI**
- **D-16:** Inline error messages displayed below buttons using text-red-500 in dark mode
- **D-17:** Critical errors also update StatusMonitor via parent state
- **D-18:** FileInput shows validation message for invalid file types

### Claude's Discretion

- Exact button sizing and spacing (use existing patterns from Phase 2)
- Icon choices for file input drop zones
- Empty state messaging within preview areas
- Hover state transitions and animations

### Deferred Ideas (OUT OF SCOPE)

- **Video playback controls:** Original uses `<video controls>` — Phase 3 implements basic display, advanced controls can be added later
- **File size validation:** Not in original frontend — defer to Phase 4/5 if needed
- **Camera selection dropdown:** Original uses number input — keep for now, dropdown can be v2 enhancement
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| COMP-01 | App.vue root component with layout structure | Already exists in Phase 2, needs component integration |
| COMP-02 | CameraPanel component for real-time video stream | Research confirms emit-based pattern with typed events |
| COMP-03 | ImagePanel component for image upload and detection | Research confirms file input wrapper pattern |
| COMP-04 | VideoPanel component for video upload and processing | Research confirms shared FileInput component pattern |
| COMP-05 | StatusMonitor component for system status messages | Already exists from Phase 2, no changes needed |
| COMP-06 | Hero section component with project title and badges | Already exists from Phase 2, no changes needed |
| COMP-07 | Preview container component for image/video results | Already exists from Phase 2, no changes needed |
| COMP-08 | File input component with drag-and-drop support | Research confirms HTML input wrapper with drag-drop visual overlay |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Vue 3 | ^3.5.31 | Component framework with Composition API | Current stable release, excellent TypeScript support, verified in package.json |
| TypeScript | ~6.0.0 | Type safety for props, emits, and component interfaces | Standard for Vue 3 projects, already configured |
| Vite | ^8.0.3 | Build tool with hot module replacement | Fast development, already configured from Phase 1 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| lucide-vue-next | ^1.0.0 | Icon library for UI elements | Use Camera, Image, Video, Upload icons in components |
| Tailwind CSS | ^3.4.17 | Utility-first styling | All component styling follows Phase 2 design system |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Native file input | vue-dropzone / v-file-drop | External libraries add dependency weight; native input sufficient for requirements |
| Emit-based state | provide/inject | Inject adds complexity; emit pattern is simpler for this 3-level hierarchy |

**Installation:** No new packages needed — all dependencies already installed in Phase 1.

**Version verification:**
```bash
# Current verified versions from package.json
npm view vue@3.5.31 version   # Latest stable as of Phase 1
npm view typescript@6.0.0 version  # Active LTS version
npm view vite@8.0.3 version   # Latest stable
```

## Architecture Patterns

### Recommended Project Structure
```
src/components/
├── panels/              # NEW in Phase 3
│   ├── CameraPanel.vue
│   ├── ImagePanel.vue
│   └── VideoPanel.vue
├── ui/                  # Existing from Phase 2
│   ├── Card.vue
│   ├── PreviewArea.vue
│   └── FileInput.vue    # NEW in Phase 3
└── layout/              # Existing from Phase 2
    ├── HeroSection.vue
    ├── PanelGrid.vue
    └── StatusMonitor.vue
```

### Pattern 1: Typed Props with Interface Declaration
**What:** Define component props using TypeScript interface with `defineProps<Props>()` and `withDefaults`
**When to use:** All components with props — this is the established pattern from Phase 2
**Example:**
```typescript
// Source: Phase 2 Card.vue component
<script setup lang="ts">
interface Props {
  title?: string
  description?: string
}

withDefaults(defineProps<Props>(), {
  title: undefined,
  description: undefined,
})
</script>
```

### Pattern 2: Typed Emits with Generic Syntax
**What:** Declare emitted events with type-safe payloads using `defineEmits<Type>()`
**When to use:** All components that emit events to parent
**Example:**
```typescript
// Source: Vue.js official documentation - Component Events
<script setup lang="ts">
const emit = defineEmits<{
  start: [cameraId: number]
  stop: []
  detect: [file: File]
}>()
</script>
```

### Pattern 3: Component Composition with Slots
**What:** Use existing UI components (Card, PreviewArea) within panels via slot composition
**When to use:** When combining multiple reusable components into a feature component
**Example:**
```vue
<template>
  <Card :title="title" :description="description">
    <PreviewArea :minHeight="previewHeight">
      <slot><!-- Panel-specific content --></slot>
    </PreviewArea>
  </Card>
</template>
```

### Pattern 4: File Input Change Handler
**What:** Handle file input changes with proper TypeScript typing
**When to use:** FileInput component and any component handling file uploads
**Example:**
```typescript
// Source: Stack Overflow - What is the correct type for file input events
const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    emit('change', file)
  }
}
```

### Pattern 5: Conditional Rendering with v-if
**What:** Show/hide UI elements based on component state (empty state, loading state, results)
**When to use:** Displaying different preview states (empty vs. loaded image/video)
**Example:**
```vue
<PreviewArea>
  <img v-if="imageUrl" :src="imageUrl" alt="检测结果" />
  <span v-else class="text-gray-400">预览区域</span>
</PreviewArea>
```

### Anti-Patterns to Avoid
- **Direct API calls in components:** Violates D-04 — all API calls must be emitted to parent for Phase 4
- **Prop mutation:** Never mutate props directly; use emits to communicate state changes up
- **Complex state in panels:** Keep panels stateless — state belongs in App.vue per D-13
- **Untyped events:** Always use typed emits — `emit('change', file)` not `emit('change')` with untyped payload

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| File input drag-drop | Custom drag-drop handlers | Native HTML input with visual overlay | Native input handles all edge cases (keyboard, accessibility, file selection dialogs) |
| Type-safe props | Manual prop validation | TypeScript interfaces with `defineProps<Props>()` | Compiler enforces types at build time, better DX |
| Component styling | Custom CSS | Tailwind utility classes from Phase 2 | Consistent with existing design system, faster development |
| Event typing | String-based event names | Generic `defineEmits<Type>()` | Autocomplete, type safety, refactor-friendly |

**Key insight:** Vue 3's Composition API with TypeScript eliminates the need for custom prop validation or runtime type checking — the compiler handles this. For file inputs, the browser's native implementation is battle-tested and accessible.

## Common Pitfalls

### Pitfall 1: Mutable Props
**What goes wrong:** Component mutates a prop value, causing Vue warnings and state synchronization issues
**Why it happens:** Direct assignment to prop (e.g., `props.cameraId = newValue`)
**How to avoid:** Treat props as read-only. Use emits to request changes from parent: `emit('update:cameraId', newValue)`
**Warning signs:** Vue devtools shows "Avoid mutating a prop directly" warning

### Pitfall 2: Untyped Event Payloads
**What goes wrong:** Emit events without type definitions, causing type errors in parent components
**Why it happens:** Using `emit('change', value)` without generic type annotation
**How to avoid:** Always use `defineEmits<{ eventName: [payload: Type] }>()` syntax
**Warning signs:** TypeScript errors in parent when accessing `event` or `file` from emitted event

### Pitfall 3: File Input State Not Reset
**What goes wrong:** After file upload fails or completes, selecting the same file doesn't trigger change event
**Why it happens:** Browser doesn't fire change event if same file selected again; input value not cleared
**How to avoid:** Reset input value to empty string after handling: `input.value = ''`
**Warning signs:** User reports "can't upload same file twice"

### Pitfall 4: Incorrect Event Target Typing
**What goes wrong:** TypeScript error accessing `.files` property on event target
**Why it happens:** Using `event.target.files` without casting to `HTMLInputElement`
**How to avoid:** Always cast: `const target = event.target as HTMLInputElement`
**Warning signs:** TypeScript error "Property 'files' does not exist on type 'EventTarget'"

### Pitfall 5: Dark Mode Classes Missing
**What goes wrong:** Components look broken in dark mode (wrong colors, invisible text)
**Why it happens:** Forgetting `dark:` prefix on Tailwind classes
**How to avoid:** Always provide both light and dark variants: `bg-white dark:bg-gray-800`
**Warning signs:** Visual regression when toggling dark mode

## Code Examples

Verified patterns from official sources:

### Typed Props and Emits (CameraPanel Example)
```typescript
// Source: Vue.js Composition API + TypeScript best practices
<script setup lang="ts">
interface Props {
  title?: string
  description?: string
  processing?: boolean
}

withDefaults(defineProps<Props>(), {
  title: '摄像头实时检测',
  description: '实时视频流分析，画面叠加跟踪 ID 与 FPS。',
  processing: false,
})

const emit = defineEmits<{
  start: [cameraId: number]
  stop: []
}>()

const cameraId = ref<number>(0)
const handleStart = () => {
  emit('start', cameraId.value)
}
</script>
```

### File Input Component
```typescript
// Source: Stack Overflow - correct type for file input events
<script setup lang="ts">
interface Props {
  accept?: string
  label?: string
}

withDefaults(defineProps<Props>(), {
  accept: '*',
  label: '选择文件',
})

const emit = defineEmits<{
  change: [file: File]
}>()

const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    emit('change', file)
  }
}

// Reset input after use (e.g., after successful upload)
const inputRef = ref<HTMLInputElement | null>(null)
const reset = () => {
  if (inputRef.value) {
    inputRef.value.value = ''
  }
}
</script>

<template>
  <div>
    <label class="block">{{ label }}</label>
    <input
      ref="inputRef"
      type="file"
      :accept="accept"
      @change="handleFileChange"
      class="block w-full text-sm text-gray-500
        file:mr-4 file:py-2 file:px-4
        file:rounded-full file:border-0
        file:text-sm file:font-semibold
        file:bg-accent-600 file:text-white
        hover:file:bg-accent-700"
    />
  </div>
</template>
```

### Panel Component Composition (ImagePanel Example)
```typescript
<script setup lang="ts">
import Card from '@/components/ui/Card.vue'
import PreviewArea from '@/components/ui/PreviewArea.vue'
import FileInput from '@/components/ui/FileInput.vue'

interface Props {
  processing?: boolean
  resultUrl?: string
  personCount?: number
}

withDefaults(defineProps<Props>(), {
  processing: false,
  resultUrl: undefined,
  personCount: undefined,
})

const emit = defineEmits<{
  detect: [file: File]
}>()

const selectedFile = ref<File | null>(null)
const handleFileChange = (file: File) => {
  selectedFile.value = file
}

const handleDetect = () => {
  if (selectedFile.value) {
    emit('detect', selectedFile.value)
  }
}
</script>

<template>
  <Card
    title="单张图片检测"
    description="上传单张图片并输出检测标注结果。"
  >
    <div class="space-y-3">
      <FileInput
        accept="image/*"
        label="选择图片文件"
        @change="handleFileChange"
      />

      <button
        @click="handleDetect"
        :disabled="processing || !selectedFile"
        class="w-full bg-accent-600 hover:bg-accent-700 disabled:opacity-50
          text-white font-medium py-2 px-4 rounded-lg
          transition-colors"
      >
        {{ processing ? '检测中...' : '开始图片检测' }}
      </button>

      <div v-if="errorMessage" class="text-red-500 text-sm">
        {{ errorMessage }}
      </div>

      <PreviewArea>
        <img v-if="resultUrl" :src="resultUrl" alt="检测结果" />
        <span v-else class="text-gray-400">预览区域</span>
      </PreviewArea>

      <div v-if="personCount !== undefined" class="meta">
        检测到行人数：{{ personCount }}
      </div>
    </div>
  </Card>
</template>
```

### App.vue Integration
```typescript
<script setup lang="ts">
import { ref } from 'vue'
import CameraPanel from '@/components/panels/CameraPanel.vue'
import ImagePanel from '@/components/panels/ImagePanel.vue'
import VideoPanel from '@/components/panels/VideoPanel.vue'

// State lifted to parent (per D-13)
const statusMessage = ref('系统就绪。')
const statusIsOk = ref(false)
const processing = ref({
  camera: false,
  image: false,
  video: false,
})

// Event handlers (API calls deferred to Phase 4)
const handleCameraStart = (cameraId: number) => {
  console.log('Start camera:', cameraId)
  // Phase 4: Implement API call
}

const handleCameraStop = () => {
  console.log('Stop camera')
  // Phase 4: Implement API call
}

const handleImageDetect = (file: File) => {
  console.log('Detect image:', file.name)
  // Phase 4: Implement API call
}

const handleVideoDetect = (file: File) => {
  console.log('Detect video:', file.name)
  // Phase 4: Implement API call
}
</script>

<template>
  <div class="max-w-[1180px] mx-auto p-[28px_20px_36px]">
    <HeroSection />

    <PanelGrid>
      <CameraPanel
        :processing="processing.camera"
        @start="handleCameraStart"
        @stop="handleCameraStop"
      />

      <ImagePanel
        :processing="processing.image"
        @detect="handleImageDetect"
      />

      <VideoPanel
        :processing="processing.video"
        @detect="handleVideoDetect"
      />
    </PanelGrid>

    <StatusMonitor :message="statusMessage" :isOk="statusIsOk" />
  </div>
</template>
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Options API with `this.$emit` | Composition API with `defineEmits<Type>()` | Vue 3.2+ | Better TypeScript support, more concise syntax |
| Untyped props (runtime only) | Typed props with TypeScript interfaces | Vue 3 + TS | Compile-time type checking, IDE autocomplete |
| Manual file input styling | Tailwind file input utilities | Tailwind 3.0+ | Consistent design system, no custom CSS needed |
| String-based event names | Type-safe event declarations | Vue 3.2+ | Refactor-safe, better DX |

**Deprecated/outdated:**
- **Vue 2 Options API:** Still works but Composition API is the recommended approach for new code
- **EventBus pattern:** Replaced by provide/inject or props/emits in Vue 3
- **`this.$emit` in `<script setup>`:** Not available — must use `defineEmits`

## Open Questions

None — all technical questions resolved through research and existing code analysis. The component architecture is straightforward given the locked decisions in CONTEXT.md and the established patterns from Phase 2.

## Environment Availability

This phase has no external dependencies beyond the existing Vue 3 + Vite + TypeScript stack already installed in Phase 1. All component development uses standard web APIs (File API, Event API) built into browsers.

**Step 2.6: SKIPPED** (no new external dependencies identified — only uses existing Vue ecosystem and browser APIs)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | None yet — Wave 0 will install Vitest |
| Config file | None — Wave 0 will create vitest.config.ts |
| Quick run command | `npm run test` (after Wave 0 setup) |
| Full suite command | `npm run test` (after Wave 0 setup) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| COMP-01 | App.vue renders with all three panels | visual/smoke | Manual verification in browser | ❌ Wave 0 |
| COMP-02 | CameraPanel emits start/stop events | unit | `vitest run src/components/panels/CameraPanel.test.ts` | ❌ Wave 0 |
| COMP-03 | ImagePanel emits detect event with file | unit | `vitest run src/components/panels/ImagePanel.test.ts` | ❌ Wave 0 |
| COMP-04 | VideoPanel emits detect event with file | unit | `vitest run src/components/panels/VideoPanel.test.ts` | ❌ Wave 0 |
| COMP-05 | StatusMonitor displays status messages | unit | `vitest run src/components/layout/StatusMonitor.test.ts` | ❌ Wave 0 |
| COMP-06 | Hero section renders correctly | visual/smoke | Manual verification in browser | ❌ Wave 0 |
| COMP-07 | PreviewArea renders with slot content | unit | `vitest run src/components/ui/PreviewArea.test.ts` | ❌ Wave 0 |
| COMP-08 | FileInput emits change event with file | unit | `vitest run src/components/ui/FileInput.test.ts` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** Manual smoke test in dev server (`npm run dev`) — verify component renders without errors
- **Per wave merge:** Full manual verification — check all panels emit correct events, props flow correctly
- **Phase gate:** Visual inspection of all 3 panels in browser + console verification of emitted events

### Wave 0 Gaps
- [ ] **vitest** — Test framework to install: `npm install -D vitest @vue/test-utils`
- [ ] **vitest.config.ts** — Config file for Vitest with Vue support
- [ ] **src/components/panels/CameraPanel.test.ts** — Unit tests for CameraPanel emits
- [ ] **src/components/panels/ImagePanel.test.ts** — Unit tests for ImagePanel emits
- [ ] **src/components/panels/VideoPanel.test.ts** — Unit tests for VideoPanel emits
- [ ] **src/components/ui/FileInput.test.ts** — Unit tests for FileInput component
- [ ] **src/components/layout/StatusMonitor.test.ts** — Unit tests for StatusMonitor component
- [ ] **src/components/ui/PreviewArea.test.ts** — Unit tests for PreviewArea component
- [ ] **npm run test** script — Add to package.json: `"test": "vitest"`

**Note:** Component testing with `@vue/test-utils` provides better coverage than pure unit tests for Vue components. Wave 0 should set up this testing foundation.

## Sources

### Primary (HIGH confidence)
- [Vue.js Official Documentation - TypeScript with Composition API](https://vuejs.org/guide/typescript/composition-api) - Verified typed props and emits patterns
- [Vue.js Official Documentation - Component Events](https://vuejs.org/guide/components/events.html) - Verified emit typing syntax with `<script setup>`
- [Vue.js Official Documentation - Slots](https://vuejs.org/guide/components/slots) - Verified slot composition patterns
- [Vue.js Official Documentation - Props](https://vuejs.org/guide/components/props) - Verified prop declaration patterns
- [Project CONTEXT.md - Phase 3](D:\YOLO11-AnchorPedestrianTrack\.planning\phases\03-component-architecture\03-CONTEXT.md) - Locked decisions D-01 through D-18
- [Project REQUIREMENTS.md](D:\YOLO11-AnchorPedestrianTrack\.planning\REQUIREMENTS.md) - Phase requirements COMP-01 through COMP-08
- [Existing Phase 2 Components](D:\YOLO11-AnchorPedestrianTrack\frontend-vue\src\components\ui) - Card.vue, PreviewArea.vue, StatusMonitor.vue patterns
- [Original Frontend](D:\YOLO11-AnchorPedestrianTrack\frontend\index.html) lines 156-292 - Complete UI reference

### Secondary (MEDIUM confidence)
- [Vue 3 Composition API: Advanced Patterns for Enterprise Applications](https://yeasirarafat.com/posts/vue-composition-api-advanced-patterns) - Verified enterprise patterns for typed props/emits
- [TypeScript, Vue 3, and Strongly Typed Props](https://madewithlove.com/blog/typescript-vue-3-and-strongly-typed-props) - Verified prop typing best practices
- [How To Make a Drag-and-Drop File Uploader With Vue.js 3](https://www.smashingmagazine.com/2022/03/drag-drop-file-uploader-vuejs-3/) - Verified drag-drop patterns (decided against implementing per D-06)
- [Stack Overflow: How to upload file in vue.js version 3](https://stackoverflow.com/questions/65703814/how-to-upload-file-in-vue-js-version-3) - Verified file input handling patterns
- [Stack Overflow: Correct type for file input events](https://stackoverflow.com/questions/74110651/what-is-the-correct-type-for-file-input-events) - Verified TypeScript typing for file change events

### Tertiary (LOW confidence)
- [VueScript Best File Upload Components (2026 Update)](https://www.vuescript.com/best-file-upload/) - Surveyed alternatives to native file input (decided against per D-06)
- [v-file-drop GitHub Repository](https://github.com/Suv4o/v-file-drop) - Considered but rejected to avoid unnecessary dependency

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All packages already verified in package.json from Phase 1
- Architecture: HIGH - Established patterns from Phase 2 components, official Vue.js docs
- Pitfalls: HIGH - Common Vue 3 + TypeScript issues well-documented in community

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (30 days — Vue 3 and TypeScript ecosystem stable)
