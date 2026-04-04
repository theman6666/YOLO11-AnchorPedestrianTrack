# Phase 02: Styling Foundation - Research

**Researched:** 2026-04-03
**Domain:** Vue 3 + Tailwind CSS dark mode design system
**Confidence:** HIGH

## Summary

Phase 02 establishes the visual foundation for the YOLO11 frontend refactoring by implementing an industrial dark mode design system with a responsive 3-panel grid layout. The research confirms that the existing Tailwind CSS configuration from Phase 1 provides a solid foundation, with the industrial color palette (gray-850/900/950 and accent colors) already configured. The original frontend's layout structure and styling patterns can be faithfully recreated using Tailwind's utility classes while maintaining component-based architecture.

**Primary recommendation:** Build on the existing Tailwind configuration by creating reusable layout components that mirror the original frontend's 3-panel grid structure, using the locked color decisions from CONTEXT.md and implementing responsive patterns that preserve visual fidelity across screen sizes.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

### Color Scheme Translation
- **D-01:** Dark mode is the default view when users first open the application (aligns with 'industrial aesthetic' goal)
- **D-02:** Keep blue accent colors from original frontend (accent-500/600/700 already configured in Tailwind)
- **D-03:** Use standard Tailwind colors for status indicators (green-500, yellow-500, red-500 for success/warning/error)
- **D-04:** Keep existing gray palette (gray-850/900/950) configured in Phase 1 for dark mode backgrounds

### Layout Fidelity
- **D-05:** Recreate exact grid behavior from original using `grid-cols-[repeat(auto-fit,minmax(340px,1fr))]` or Tailwind arbitrary values
- **D-06:** Use Tailwind's spacing scale approximating original values (gap-3.5, p-4, mb-2.5) instead of exact pixels
- **D-07:** Cards in dark mode use bordered style (bg-gray-800, border-gray-700, shadow) to preserve separation
- **D-08:** Hero section uses prominent card styling — similar to cards but more prominent with different background/gradient

### Typography System
- **D-09:** Keep Inter font for English with Microsoft YaHei for Chinese (consistent with base.css from Phase 1)
- **D-10:** Use Tailwind's default heading scale (text-3xl for h1, text-xl for h2) instead of custom pixel sizes
- **D-11:** Use font-semibold (600) for headings instead of custom 650 weight
- **D-12:** Use leading-relaxed (1.625) for body text line height

### Claude's Discretion
- Specific shadow values for cards and hero section
- Exact border radius values (can use Tailwind's rounded-xl or custom)
- Text color opacity variations for muted text
- Badge styling in the hero section
- Button gradient values (can adapt original or use solid Tailwind colors)
- Status bar positioning and styling

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| STYLE-01 | Tailwind dark mode theme configured (industrial aesthetic) | Existing config with gray-850/900/950 and accent colors; default dark mode per D-01 |
| STYLE-02 | Color palette defined (dark backgrounds, accent colors, status colors) | D-02 through D-04 lock blue accents and standard Tailwind status colors |
| STYLE-03 | Responsive grid layout matching existing 3-panel structure | D-05 specifies exact grid replication with auto-fit minmax(340px) pattern |
| STYLE-04 | Typography and spacing system established | D-09 through D-12 define font family, sizing, weight, and line height |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Tailwind CSS | 3.4.17 | Utility-first styling framework | Already configured in Phase 1; provides industrial palette and dark mode |
| Vue 3 | 3.5.31 | Component framework for styling foundation | Composition API with `<script setup>` established pattern |
| Lucide-Vue-Next | 1.0.0 | Icon library for visual elements | Already integrated; provides consistent iconography |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| PostCSS | 8.4.35 | CSS processing for Tailwind | Already configured via postcss.config.js |
| Autoprefixer | 10.4.17 | CSS vendor prefixing | Already configured with Tailwind plugin |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Tailwind 3.4.17 | Tailwind 4.x | Migration risk; PostCSS plugin incompatibility documented in Phase 1; Phase 1 validated 3.4.17 works |
| `dark:` prefix variants | CSS custom properties | Locked decision D-01 uses .dark class strategy; established pattern from Phase 1 |

**Installation:**
No additional packages needed — all required dependencies installed in Phase 1.

**Version verification:**
```bash
npm view tailwindcss version  # Expected: 3.4.17 (verified in Phase 1)
npm view vue version  # Expected: 3.5.31 (verified in Phase 1)
npm view lucide-vue-next version  # Expected: 1.0.0 (verified in Phase 1)
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── components/
│   ├── layout/
│   │   ├── HeroSection.vue      # Hero section with title and badges
│   │   ├── PanelGrid.vue        # 3-panel grid layout wrapper
│   │   └── StatusMonitor.vue    # Status bar component (Phase 3)
│   └── ui/
│       ├── Card.vue             # Reusable card component
│       └── PreviewArea.vue      # Preview container with dashed border
├── composables/
│   └── useDarkMode.ts           # Dark mode toggle logic (extract from App.vue)
└── App.vue                      # Root component updated to use layout components
```

### Pattern 1: Dark Mode with Class Strategy
**What:** Toggle `.dark` class on `<html>` element for theme switching
**When to use:** All dark mode styling must use `dark:` prefix for variants
**Example:**
```vue
<!-- Source: Phase 1 established pattern, tailwind.config.js -->
<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
    <div class="max-w-[1180px] mx-auto p-[28px_20px_36px]">
      <!-- Content -->
    </div>
  </div>
</template>

<script setup lang="ts">
// Dark mode toggle function from Phase 1 App.vue
const toggleDarkMode = () => {
  document.documentElement.classList.toggle('dark')
}
</script>
```

### Pattern 2: Responsive 3-Panel Grid
**What:** Auto-fit grid with minimum 340px panels, 14px gap
**When to use:** Main layout structure matching original frontend
**Example:**
```vue
<!-- Source: CONTEXT.md D-05, original frontend line 71 -->
<template>
  <div class="grid grid-cols-[repeat(auto-fit,minmax(340px,1fr))] gap-[14px]">
    <div class="card"><!-- Camera panel --></div>
    <div class="card"><!-- Image panel --></div>
    <div class="card"><!-- Video panel --></div>
  </div>
</template>

<style scoped>
/* Alternative using Tailwind arbitrary values */
.grid {
  @apply grid gap-3.5;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
}
</style>
```

### Pattern 3: Industrial Card Styling
**What:** Bordered cards with subtle shadows for content separation
**When to use:** All panel containers and hero section
**Example:**
```vue
<!-- Source: CONTEXT.md D-07, original frontend lines 73-79 -->
<template>
  <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl shadow-lg p-4">
    <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
      Panel Title
    </h2>
    <div class="sub text-sm text-gray-600 dark:text-gray-400 mt-1.5 mb-3.5">
      Panel description
    </div>
    <!-- Panel content -->
  </div>
</template>
```

### Pattern 4: Typography Hierarchy
**What:** Consistent heading sizes using Tailwind scale
**When to use:** All text content following D-10 through D-12
**Example:**
```vue
<!-- Source: CONTEXT.md D-10, D-11, D-12 -->
<template>
  <h1 class="text-3xl font-semibold text-accent-600 dark:text-accent-500 leading-relaxed">
    主标题
  </h1>
  <h2 class="text-xl font-semibold text-gray-900 dark:text-white leading-relaxed">
    副标题
  </h2>
  <p class="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
    正文内容
  </p>
</template>
```

### Anti-Patterns to Avoid
- **Hardcoded pixel values in class names:** Use Tailwind's spacing scale (p-4, gap-3) instead of custom values unless exact replication requires arbitrary values
- **Missing dark mode variants:** Always provide `dark:` prefix for color backgrounds, text colors, and borders
- **Inconsistent border radius:** Use rounded-xl (12px) or rounded-lg (8px) consistently rather than mixing arbitrary values
- **Skipping transition classes:** Add `transition-colors` to containers that change appearance in dark mode
- **Overriding base.css:** Tailwind handles Inter font import; avoid duplicating font-family declarations

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Grid system | Custom CSS grid with media queries | Tailwind's `grid-cols-[repeat(auto-fit,minmax(340px,1fr))]` | Responsive behavior built-in, matches original layout exactly |
| Dark mode toggle | localStorage + system detection | Existing `toggleDarkMode()` from Phase 1 | Already implemented and tested |
| Color palette | Custom CSS variables | Tailwind's gray-850/900/950 and accent colors | Configured in Phase 1, provides consistent industrial aesthetic |
| Spacing scale | Custom margin/padding utilities | Tailwind's spacing scale (4px base unit) | Consistent spacing system, responsive utilities available |
| Typography system | Custom font-size classes | Tailwind's text sizing (text-sm, text-xl, text-3xl) | Responsive typography, line-height utilities included |

**Key insight:** The original frontend's styling is intentionally simple and maps directly to Tailwind utilities. Building custom CSS would introduce maintenance burden without benefit.

## Common Pitfalls

### Pitfall 1: Arbitrary Value Overuse
**What goes wrong:** Excessive use of arbitrary values like `p-[28px_20px_36px]` makes code harder to maintain
**Why it happens:** Desire to exactly match original frontend's pixel values
**How to avoid:** Use Tailwind's spacing scale where possible; reserve arbitrary values for exact layout requirements (like the 340px minmax breakpoint)
**Warning signs:** More than 20% of classes use square bracket notation

### Pitfall 2: Missing Dark Mode Variants
**What goes wrong:** Light mode colors leak into dark mode, causing poor contrast
**Why it happens:** Forgetting to add `dark:` prefix for backgrounds, text, borders
**How to avoid:** Audit all color classes for dark mode variants; use automated testing or linting
**Warning signs:** Dark mode toggle shows no visual change or washed-out colors

### Pitfall 3: Responsive Breakpoint Assumptions
**What goes wrong:** 3-panel grid breaks on smaller screens, causing overflow or stacking issues
**Why it happens:** Assuming desktop-only usage from original frontend
**How to avoid:** Test layout at 1024px, 768px, and 640px breakpoints; ensure minmax(340px) allows stacking
**Warning signs:** Horizontal scrolling on tablet/mini-laptop screens

### Pitfall 4: Inconsistent Status Colors
**What goes wrong:** Status messages use inconsistent colors (yellow vs amber, green vs emerald)
**Why it happens:** Tailwind has multiple color variations for semantic colors
**How to avoid:** Lock to specific status colors per D-03 (green-500, yellow-500, red-500)
**Warning signs:** Status colors don't match between panels and status bar

### Pitfall 5: Font Loading Flash
**What goes wrong:** Text renders in fallback font before Inter loads
**Why it happens:** Inter font loaded via CDN instead of bundling
**How to avoid:** Phase 1 base.css already imports Inter correctly; no action needed
**Warning signs:** Visible font shift on page load

## Code Examples

Verified patterns from official sources:

### Hero Section with Badges
```vue
<!-- Source: CONTEXT.md D-08, original frontend lines 36-69 -->
<template>
  <section class="bg-gradient-to-br from-white to-gray-50 dark:from-gray-800 dark:to-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl shadow-lg p-[22px] mb-4">
    <h1 class="text-3xl font-semibold text-accent-600 dark:text-accent-500 leading-relaxed">
      行人检测与跟踪系统
    </h1>
    <p class="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mt-2.5">
      基于 YOLO11 + CBAM + ByteTrack，支持摄像头实时推理、图片检测和离线视频分析。
    </p>
    <div class="flex flex-wrap gap-2 mt-3.5">
      <span class="inline-flex items-center px-2.5 py-1.5 text-xs font-medium rounded-full border border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/30">
        目标类别：行人
      </span>
      <span class="inline-flex items-center px-2.5 py-1.5 text-xs font-medium rounded-full border border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/30">
        跟踪算法：ByteTrack
      </span>
      <span class="inline-flex items-center px-2.5 py-1.5 text-xs font-medium rounded-full border border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/30">
        输出：人数统计 + FPS + 标注结果
      </span>
    </div>
  </section>
</template>
```

### Status Bar Component
```vue
<!-- Source: original frontend lines 131-142, CONTEXT.md Claude's Discretion -->
<template>
  <div 
    class="mt-3.5 border rounded-xl px-3 py-2.5 text-sm transition-colors"
    :class="isOk 
      ? 'border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300' 
      : 'border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300'"
  >
    {{ statusMessage }}
  </div>
</template>

<script setup lang="ts">
defineProps<{
  statusMessage: string
  isOk?: boolean
}>()
</script>
```

### Preview Area with Dashed Border
```vue
<!-- Source: original frontend lines 113-125 -->
<template>
  <div class="min-h-[230px] border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-xl bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-850 flex items-center justify-center overflow-hidden text-gray-500 dark:text-gray-400 text-sm p-2">
    <slot>
      <span class="text-gray-400 dark:text-gray-500">预览区域</span>
    </slot>
  </div>
</template>
```

### Page Container with Grid Background
```vue
<!-- Source: original frontend lines 27-32, grid background pattern -->
<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900" 
       style="background-image: linear-gradient(90deg, rgba(31,78,121,0.03) 1px, transparent 1px), linear-gradient(rgba(31,78,121,0.03) 1px, transparent 1px); background-size: 36px 36px;">
    <div class="max-w-[1180px] mx-auto p-[28px_20px_36px]">
      <!-- Content -->
    </div>
  </div>
</template>
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CSS custom properties for theming | Tailwind dark mode with `dark:` prefix | Tailwind 3.0+ | Utility-first approach, better performance |
| Media query based dark mode | Class-based dark mode toggle | Tailwind 2.0+ | User-controlled theme switching |
| Fixed pixel layouts | Responsive grid with auto-fit | Modern CSS | Better mobile support, flexible layouts |

**Deprecated/outdated:**
- `prefers-color-scheme` media query for user-controlled dark mode (use class strategy per Phase 1 config)
- Custom CSS grid frameworks (Tailwind's grid system is sufficient)
- Font loading via `@font-face` in CSS (use base.css approach from Phase 1)

## Open Questions

1. **Grid background pattern implementation**
   - What we know: Original frontend uses subtle grid pattern with linear-gradient
   - What's unclear: Whether to implement as Tailwind arbitrary value or custom CSS class
   - Recommendation: Use inline style or scoped CSS for the specific background pattern (36px grid with accent color at 3% opacity)

2. **Shadow values for cards and hero**
   - What we know: Original uses `box-shadow: 0 10px 30px rgba(17, 31, 52, 0.08)`
   - What's unclear: Whether to use Tailwind's shadow-lg or custom shadow
   - Recommendation: Start with shadow-lg (closest equivalent), adjust if visual comparison shows significant difference

3. **Button gradient vs solid colors**
   - What we know: Original uses gradient `linear-gradient(180deg, #2f5f8f 0%, #224a72 100%)`
   - What's unclear: Whether to replicate gradient or use solid accent-600
   - Recommendation: Use solid accent-600 for simplicity, add gradient if visual comparison shows it's materially better

## Environment Availability

### External Dependencies
None — this phase is purely styling with no external service dependencies.

### Development Environment
| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Node.js | Vite dev server | ✓ (assumed) | 18+ | — |
| npm | Package management | ✓ (assumed) | 9+ | — |
| Vue devtools | Debugging (optional) | ? | — | Not required |

**Missing dependencies with no fallback:** None detected

**Missing dependencies with fallback:** None detected

**Step 2.6: COMPLETE** — All dependencies available from Phase 1 setup.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Visual testing + manual verification (no automated testing for styling) |
| Config file | None — visual validation |
| Quick run command | `cd frontend-vue && npm run dev` |
| Full inspection command | Browser DevTools element inspection |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STYLE-01 | Dark mode theme applies | Manual | Open dev server, toggle dark mode, inspect .dark class | ❌ Wave 0 |
| STYLE-02 | Color palette consistent | Manual | DevTools color picker on cards, text, borders | ❌ Wave 0 |
| STYLE-03 | 3-panel grid responsive | Manual | Resize browser to 1024px/768px/640px, verify panel behavior | ❌ Wave 0 |
| STYLE-04 | Typography hierarchy | Manual | Inspect font sizes, line heights, weights | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** Start dev server, verify dark mode toggle, inspect rendered elements
- **Per wave merge:** Visual comparison with original frontend at multiple screen sizes
- **Phase gate:** Manual verification of all 4 STYLE requirements against original frontend

### Wave 0 Gaps
- [ ] `src/components/layout/HeroSection.vue` — hero section with title and badges
- [ ] `src/components/layout/PanelGrid.vue` — 3-panel grid wrapper component
- [ ] `src/components/ui/Card.vue` — reusable card component with dark mode variants
- [ ] `src/components/ui/PreviewArea.vue` — preview container with dashed border
- [ ] Framework setup: None required — Vue 3 + Tailwind CSS installed in Phase 1

*(No automated test framework needed for styling phase — visual verification is appropriate)*

## Sources

### Primary (HIGH confidence)
- `frontend/index.html` — Complete original frontend reference (296 lines)
- `frontend-vue/tailwind.config.js` — Phase 1 established color palette and dark mode config
- `frontend-vue/src/assets/base.css` — Inter font family setup from Phase 1
- `.planning/phases/01-project-setup/01-VERIFICATION.md` — Phase 1 verification confirming dark mode toggle works
- `REQUIREMENTS.md` — STYLE-01 through STYLE-04 formal requirements
- `02-CONTEXT.md` — Locked decisions D-01 through D-12

### Secondary (MEDIUM confidence)
- [TailwindCSS v4 Tutorial 2026 - Dark Mode](https://www.youtube.com/watch?v=EnnrqkL0CKI) — Current dark mode best practices (2026)
- [25 Best Vue.js Admin Dashboard Templates 2026](https://colorlib.com/wp/free-vuejs-admin-template/) — Vue 3 + Tailwind layout patterns
- [Tailwind Dynamic Grid Issue with Vue3 - Stack Overflow](https://stackoverflow.com/questions/74574559/tailwind-dynamic-grid-issue-with-vue3) — Grid layout implementation patterns

### Tertiary (LOW confidence)
- Rate limiting prevented full verification of spacing scale and typography system searches — relying on official Tailwind documentation and established Phase 1 patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All dependencies verified in Phase 1, versions locked
- Architecture: HIGH - Original frontend provides complete reference, decisions locked in CONTEXT.md
- Pitfalls: MEDIUM - Based on common Tailwind CSS mistakes, some specific to this codebase

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (30 days — Tailwind CSS 3.4.x is stable, unlikely to change)

**Canonical references verified:**
- ✓ Original frontend structure analyzed
- ✓ Phase 1 established patterns documented
- ✓ All locked decisions from CONTEXT.md incorporated
- ✓ No external API dependencies identified
- ✓ Visual verification approach appropriate for styling phase
