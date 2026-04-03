# Phase 2: Styling Foundation - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

## Phase Boundary

Create an industrial dark mode design system with responsive layout matching the existing 3-panel structure from the original frontend. This phase establishes the visual foundation that all components will build upon.

## Implementation Decisions

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

## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Original Frontend Structure
- `frontend/index.html` — Complete reference for layout structure, 3-panel grid, card design, hero section, status bar styling

### Phase 1 Established Patterns
- `frontend-vue/tailwind.config.js` — Existing color palette (gray-850/900/950, accent colors), dark mode configuration
- `frontend-vue/src/assets/base.css` — Base CSS setup, Inter font family configuration
- `.planning/phases/01-project-setup/01-VERIFICATION.md` — Confirmed Phase 1 setup including dark mode toggle functionality

### Requirements Mapping
- `.planning/REQUIREMENTS.md` — STYLE-01 through STYLE-04 define the success criteria for this phase

No external specs exist beyond the original frontend — requirements are fully captured in decisions above and the original HTML reference.

## Existing Code Insights

### Reusable Assets

- **Tailwind config with dark mode**: `tailwind.config.js` already has darkMode: 'class' and industrial color palette defined
- **Dark mode toggle**: `App.vue` has `toggleDarkMode()` function that toggles .dark class on documentElement
- **Lucide icons**: Already integrated via `lucide-vue-next` — icons like Camera, Image, Video, Settings available
- **Axios client**: API client already configured in `src/api/client.ts` — no styling work needed

### Established Patterns

- **Dark mode class strategy**: Use `dark:` prefix for all dark mode variants (e.g., `dark:bg-gray-900`)
- **Accent color usage**: Use accent-600 for primary buttons, accent-700 for hover states
- **Component structure**: Vue 3 with Composition API (`<script setup>`)
- **CSS approach**: Tailwind utility classes with minimal scoped CSS

### Integration Points

- **App.vue root component**: Will be updated to use new 3-panel layout instead of current test structure
- **base.css**: Already imports Tailwind directives — no changes needed
- **main.ts**: Entry point that mounts App.vue and imports base.css

## Specific Ideas

- Original frontend has a grid background pattern (linear-gradient) — consider replicating for visual interest
- Hero section badges with rounded-full styling — creates pill-shaped category indicators
- Preview areas use dashed borders and gradient backgrounds for empty states
- Status bar has conditional styling (normal vs. ok class) for different message types

## Deferred Ideas

None — discussion stayed within phase scope.

---

*Phase: 02-styling-foundation*
*Context gathered: 2026-04-03*
