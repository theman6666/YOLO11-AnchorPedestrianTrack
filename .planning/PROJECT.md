# YOLO11 Frontend Refactoring Project

## What This Is

Refactoring the existing single-file HTML frontend for the YOLO11 pedestrian detection and tracking system into a modern, component-based Vue 3 application with an industrial dark mode UI.

## Core Value

A maintainable, scalable frontend architecture that preserves all existing detection/tracking functionality while providing a professional user experience.

## Requirements

### Validated

- ✓ Camera streaming with real-time detection boxes and overlays — existing
- ✓ Single image upload and detection with annotated results — existing
- ✓ Video upload and processing with tracking visualization — existing
- ✓ Flask backend API serving detection results — existing
- ✓ Person count and FPS overlay on video streams — existing

### Active

- [ ] Modern Vue 3 application with Vite build system
- [ ] Component-based architecture (CameraPanel, ImagePanel, VideoPanel, StatusMonitor)
- [ ] Tailwind CSS styling with industrial dark mode theme
- [ ] Lucide-Vue-Next icon library integration
- [ ] Axios-based API communication layer
- [ ] Responsive layout preserving existing 3-panel grid structure
- [ ] State management for detection results and status messages

### Out of Scope

- Backend API modifications — backend remains as-is
- New detection/tracking features — focus on frontend architecture only
- Mobile-native apps — web-only refactoring
- Authentication/user management — single-user local system

## Context

**Existing System:**
- Backend: Flask (`src/run/app.py`) serving `/video_feed`, `/detect/image`, `/detect/video` endpoints
- Frontend: Single HTML file (`frontend/index.html`) with vanilla JavaScript
- Model: YOLO11 + CBAM hybrid for pedestrian detection
- Tracker: ByteTrack for multi-object tracking with ID persistence

**Technical Environment:**
- Python 3.10/3.11 with CUDA 12.1 support
- Existing Node.js/npm not required (new dependency)
- Windows development environment
- Model weights: `result/hybrid_weights/YOLO11m_CBAM_Hybrid_local6/weights/best.pt`

**Current Pain Points:**
- Monolithic HTML file is hard to maintain
- No code reuse between panels
- Styling is scattered in inline CSS
- Limited extensibility for future features

## Constraints

- **Backend compatibility**: Must work with existing Flask API endpoints unchanged
- **Visual parity**: Preserve all existing functionality (camera feed, image/video upload, results display)
- **Deployment**: Must be deployable alongside Flask backend
- **Performance**: Video stream performance must match existing implementation
- **Dark mode**: Industrial aesthetic with Tailwind CSS

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Vue 3 + Vite | Modern build tooling, fast HMR, excellent ecosystem | — Pending |
| Tailwind CSS | Rapid styling, dark mode support, industrial aesthetic | — Pending |
| Component architecture | Code reuse, maintainability, testability | — Pending |
| Axios for API | Clean promise-based HTTP, better than fetch for complex apps | — Pending |
| Lucide icons | Modern, consistent icon set for industrial feel | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-03 after initialization*
