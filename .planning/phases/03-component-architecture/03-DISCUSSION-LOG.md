# Phase 3: Component Architecture - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-03
**Phase:** 03-component-architecture
**Mode:** Auto-selected decisions (no interactive discussion)

---

## Gray Areas and Auto-Selected Decisions

### 1. Component Interface Design

**Decision:** Props for title/description, emit events for actions (no direct API calls)

**Rationale:**
- Keeps components pure and reusable
- Parent (App.vue) handles API calls in Phase 4
- Follows Vue 3 best practices for component composition

**Alternatives considered:**
- Components call API directly (rejected: creates tight coupling to backend)
- Use Vuex/Pinia store (rejected: overkill for this phase, parent state is sufficient)

### 2. File Input Implementation

**Decision:** Standard HTML input with drag-drop zone overlay

**Rationale:**
- Original frontend uses standard `<input type="file">`
- PreviewArea component already provides dashed border styling
- Drag-drop is visual enhancement only, same behavior underneath

**Alternatives considered:**
- Custom drag-drop library (rejected: adds dependency, standard input is sufficient)
- Dropzone.js or similar (rejected: overkill for simple file selection)

### 3. Button Interaction Patterns

**Decision:** Primary (accent-600) and secondary (gray) buttons, loading state with disabled + opacity

**Rationale:**
- Matches Phase 2 color decisions (D-02: blue accent colors)
- Original frontend uses "处理中..." text during video processing
- Disabled state prevents duplicate submissions

**Alternatives considered:**
- Loading spinners (rejected: text change is clearer per original)
- Button dimming only (rejected: disabled attribute is more accessible)

### 4. Data Flow Architecture

**Decision:** State lifted to App.vue parent, components emit events

**Rationale:**
- Single source of truth for status messages
- Components remain presentational and testable
- Phase 4 will wire API calls to emitted events

**Alternatives considered:**
- Local component state (rejected: makes parent coordination difficult)
- Global state management (rejected: unnecessary complexity for 3 panels)

### 5. Error Handling UI

**Decision:** Inline errors below buttons + StatusMonitor updates

**Rationale:**
- Original shows status messages in bottom bar
- Inline errors provide immediate feedback within panel
- StatusMonitor gives system-wide visibility

**Alternatives considered:**
- Toast notifications (rejected: not in original design)
- Modal dialogs (rejected: too heavy for simple validation errors)

---

## Deferred Ideas

- Video playback controls: Use native `<video controls>`, enhanced controls can be v2
- File size validation: Not in original, defer to Phase 4/5 if needed
- Camera selection dropdown: Number input is sufficient, dropdown can be v2

---

*Phase: 03-component-architecture*
*Discussion logged: 2026-04-03 via auto-mode*
