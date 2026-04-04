# Interaction State Test Suite

**Purpose:** Validate UI state management during operations including button states, loading feedback, and error recovery

**Context Decisions:** D-23, D-24, D-25

**Last Updated:** 2026-04-04

---

## Overview

This test suite validates UI state management and interaction behaviors across image and video detection features. Tests focus on button disabled states during processing, immediate loading feedback, and error recovery without page refresh.

**Focus:** UI behavior during async operations, not just functional correctness.

**Test Execution Order:** Tests can be executed independently, but grouping by aspect is recommended.

**Estimated Time:** 20-25 minutes

---

## Quick Reference

| Test ID | Priority | Context Decision | Aspect | Feature | Status |
|---------|----------|------------------|--------|---------|--------|
| TC-INT-001 | P0 | D-23 | Button Disabled | Image | ⬜ Not Executed |
| TC-INT-002 | P0 | D-23 | Button Disabled | Video | ⬜ Not Executed |
| TC-INT-003 | P0 | D-24 | Loading Feedback | Image | ⬜ Not Executed |
| TC-INT-004 | P0 | D-24 | Loading Feedback | Video | ⬜ Not Executed |
| TC-INT-005 | P0 | D-25 | Error Recovery | Image | ⬜ Not Executed |
| TC-INT-006 | P0 | D-25 | Error Recovery | Video | ⬜ Not Executed |

**Priority Legend:**
- **P0 (Critical):** Must pass for system to be usable
- **P1 (Important):** Should pass for good user experience
- **P2 (Nice-to-have):** Optional enhancements

**Aspect Legend:**
- **Button Disabled:** Verify button prevents duplicate requests
- **Loading Feedback:** Verify user receives immediate feedback
- **Error Recovery:** Verify user can retry without page refresh

---

## Interaction State Testing Checklist

| Aspect | Image | Video | Camera |
|--------|-------|-------|---------|
| Button disabled during operation | TC-INT-001 | TC-INT-002 | TC-CAM-010 |
| Loading feedback (immediate) | TC-INT-003 | TC-INT-004 | TC-CAM-004 |
| Error recovery (no refresh) | TC-INT-005 | TC-INT-006 | TC-CAM-009 |

---

## Test Cases

---

### Test Case: TC-INT-001 - Button Disabled During Image Detection

**Priority:** P0 (Critical)
**Context Decision:** D-23
**Aspect:** Button Disabled
**Feature:** Image
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify button prevents duplicate image detection requests during processing.

#### Prerequisites
- Backend and frontend are running
- Image file is selected
- Browser DevTools Network tab is open

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Open DevTools Network tab | Network tab visible |
| 2 | Filter by "/detect/image" | Only image detection requests shown |
| 3 | Click "检测图像" button | First request sent |
| 4 | Immediately try to click again (within 0.5 seconds) | Button should not respond |
| 5 | Try to click 3-4 more times rapidly | Only one request should be sent |
| 6 | Observe Network tab | Only one POST /detect/image request |
| 7 | Wait for completion | Processing completes |
| 8 | Observe button state | Button re-enabled after completion |

#### Expected Results
- **Button visually disabled:** Grayed out, lower opacity
- **Button not clickable:** Cursor indicates not-allowed (optional)
- **Only one API request:** Single POST /detect/image in Network tab
- **Multiple clicks ignored:** No response to additional clicks
- **Button re-enabled:** After completion, button clickable again

#### Verification Checklist

- [ ] Button visually disabled (grayed out, lower opacity)
- [ ] Button not clickable (cursor indicates not-allowed)
- [ ] Only one API request in Network tab
- [ ] Button re-enabled after completion

#### Actual Results

**Button visually disabled:** ⬜ Yes | ❌ No

**Button cursor changes:** ⬜ Yes (not-allowed) | ❌ No (normal cursor)

**Number of POST /detect/image requests:** _____

**Only one request got response:** ⬜ Yes | ❌ No

**Button re-enabled after completion:** ⬜ Yes | ❌ No

#### Test Evidence
<!-- Screenshot of Network tab showing single request -->
________________________________________

#### Tester Notes
<!-- Is button disabling effective and obvious to user? -->
________________________________________

---

### Test Case: TC-INT-002 - Button Disabled During Video Detection

**Priority:** P0 (Critical)
**Context Decision:** D-23
**Aspect:** Button Disabled
**Feature:** Video
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify button prevents duplicate video detection requests during processing.

#### Prerequisites
- Backend and frontend are running
- Video file is selected
- Browser DevTools Network tab is open

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Open DevTools Network tab | Network tab visible |
| 2 | Filter by "/detect/video" | Only video detection requests shown |
| 3 | Click "检测视频" button | First request sent |
| 4 | Immediately try to click again (within 0.5 seconds) | Button should not respond |
| 5 | Try to click 3-4 more times rapidly | Only one request should be sent |
| 6 | Observe Network tab | Only one POST /detect/video request |
| 7 | Wait for completion | Processing completes |
| 8 | Observe button state | Button re-enabled after completion |

#### Expected Results
- **Button visually disabled:** Grayed out, lower opacity
- **Button not clickable:** Cursor indicates not-allowed (optional)
- **Only one API request:** Single POST /detect/video in Network tab
- **Multiple clicks ignored:** No response to additional clicks
- **Button re-enabled:** After completion, button clickable again

#### Verification Checklist

- [ ] Button visually disabled (grayed out, lower opacity)
- [ ] Button not clickable (cursor indicates not-allowed)
- [ ] Only one API request in Network tab
- [ ] Button re-enabled after completion

#### Actual Results

**Button visually disabled:** ⬜ Yes | ❌ No

**Button cursor changes:** ⬜ Yes (not-allowed) | ❌ No (normal cursor)

**Number of POST /detect/video requests:** _____

**Only one request got response:** ⬜ Yes | ❌ No

**Button re-enabled after completion:** ⬜ Yes | ❌ No

#### Test Evidence
<!-- Screenshot of Network tab showing single request -->
________________________________________

#### Tester Notes
<!-- Is button disabling effective for long-running operations? -->
________________________________________

---

### Test Case: TC-INT-003 - Loading Feedback - Image Detection

**Priority:** P0 (Critical)
**Context Decision:** D-24
**Aspect:** Loading Feedback
**Feature:** Image
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify user receives clear loading feedback during image detection (immediate UI updates).

#### Prerequisites
- Backend and frontend are running
- Image file is selected
- DevTools is open

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Click "检测图像" button | Detection starts |
| 2 | Observe button text immediately (within 0.1 seconds) | Text changes to "检测中..." |
| 3 | Observe button state immediately (within 0.1 seconds) | Button becomes disabled |
| 4 | Observe status message immediately (within 0.1 seconds) | Message changes to "正在进行图片检测..." |
| 5 | Check for UI freeze or lag | UI remains responsive |
| 6 | Wait for completion | Processing finishes |

#### Expected Results

**Immediate Feedback (within 0.1 seconds):**
- Button text: Changes from "检测图像" to "检测中..."
- Button disabled: Yes (visually grayed out)
- Status message: Changes to "正在进行图片检测..."
- No delay: All changes happen instantly
- No lag: UI remains responsive, no freezing

#### Verification Checklist

- [ ] Button text change is instant (< 0.1s)
- [ ] Button disabled state is instant (< 0.1s)
- [ ] Status message updates immediately (< 0.1s)
- [ ] No UI freeze or lag
- [ ] All changes happen before API response

#### Actual Results

**Button text change timing:** ⬜ Instant (< 0.1s) | ❌ Delayed: _____ seconds

**Button disabled timing:** ⬜ Instant (< 0.1s) | ❌ Delayed: _____ seconds

**Status message timing:** ⬜ Instant (< 0.1s) | ❌ Delayed: _____ seconds

**UI freeze or lag:** ⬜ None | ❌ Laggy: _________________________

**All feedback before API response:** ⬜ Yes | ❌ No (API response came first)

#### Test Evidence
<!-- Screen recording showing immediate feedback -->
________________________________________

#### Tester Notes
<!-- Is loading feedback immediate and clear? -->
________________________________________

---

### Test Case: TC-INT-004 - Loading Feedback - Video Detection

**Priority:** P0 (Critical)
**Context Decision:** D-24
**Aspect:** Loading Feedback
**Feature:** Video
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify user receives clear loading feedback during video detection (immediate UI updates for long-running operation).

#### Prerequisites
- Backend and frontend are running
- Video file is selected
- DevTools is open

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Click "检测视频" button | Detection starts |
| 2 | Observe button text immediately (within 0.1 seconds) | Text changes to "检测中..." |
| 3 | Observe button state immediately (within 0.1 seconds) | Button becomes disabled |
| 4 | Observe status message immediately (within 0.1 seconds) | Message changes to "正在进行视频检测，耗时可能较长，请稍候。" |
| 5 | Check for UI freeze or lag | UI remains responsive |
| 6 | Wait for completion | Processing finishes (may take 10-30 seconds) |

#### Expected Results

**Immediate Feedback (within 0.1 seconds):**
- Button text: Changes from "检测视频" to "检测中..."
- Button disabled: Yes (visually grayed out)
- Status message: Changes to "正在进行视频检测，耗时可能较长，请稍候。"
- No delay: All changes happen instantly
- No lag: UI remains responsive, no freezing

#### Verification Checklist

- [ ] Button text change is instant (< 0.1s)
- [ ] Button disabled state is instant (< 0.1s)
- [ ] Status message updates immediately (< 0.1s)
- [ ] No UI freeze or lag
- [ ] All changes happen before API response

#### Actual Results

**Button text change timing:** ⬜ Instant (< 0.1s) | ❌ Delayed: _____ seconds

**Button disabled timing:** ⬜ Instant (< 0.1s) | ❌ Delayed: _____ seconds

**Status message timing:** ⬜ Instant (< 0.1s) | ❌ Delayed: _____ seconds

**UI freeze or lag:** ⬜ None | ❌ Laggy: _________________________

**All feedback before API response:** ⬜ Yes | ❌ No (API response came first)

#### Test Evidence
<!-- Screen recording showing immediate feedback -->
________________________________________

#### Tester Notes
<!-- Is loading feedback immediate even for long-running operation? -->
________________________________________

---

### Test Case: TC-INT-005 - Error Recovery - Image Detection

**Priority:** P0 (Critical)
**Context Decision:** D-25
**Aspect:** Error Recovery
**Feature:** Image
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify user can retry after image detection error without page refresh.

#### Prerequisites
- Backend is running
- Image file is selected
- Console is open for monitoring

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Click "检测图像" button | Detection starts |
| 2 | Wait 1 second | Processing begins |
| 3 | Stop backend (Ctrl+C) | Backend terminates (simulates error) |
| 4 | Wait 2-3 seconds | Error appears |
| 5 | Observe error display | Error message displays |
| 6 | Observe button state | Button becomes enabled after error |
| 7 | Check for stale state | No leftover loading state |
| 8 | Start backend (`python src/run/app.py`) | Backend restarts |
| 9 | Click "检测图像" button (no refresh) | Retry attempt |
| 10 | Observe result | Detection succeeds |

#### Expected Results

**Error Display:**
- Error message displays after simulated failure
- Error is user-friendly (not technical stack trace)
- Inline error displays in ImagePanel
- Status bar shows error state

**Button State:**
- Button becomes enabled after error
- Button text returns to "检测图像"
- No disabled state persists

**No Stale State:**
- No loading indicator persists
- No "检测中..." text remains
- Error clears on retry
- No leftover state from failed attempt

**Retry Success:**
- Second detection attempt succeeds
- No page refresh required
- Application remains responsive
- Results display correctly

#### Recovery Verification Checklist

- [ ] Error message displays
- [ ] Button becomes enabled after error
- [ ] No stale state from failed attempt
- [ ] Retry operation succeeds
- [ ] No need to refresh page
- [ ] Results display correctly on retry

#### Actual Results

**Error message displays:** ⬜ Yes | ❌ No

**Button becomes enabled after error:** ⬜ Yes | ❌ No

**Stale state (loading persists):** ⬜ None | ❌ Loading persists: _________________________

**Retry without refresh:** ⬜ Success | ❌ Failed

**Page refresh required:** ⬜ No | ❌ Yes (problematic)

**Second detection succeeds:** ⬜ Yes | ❌ No

#### Test Evidence
<!-- Screen recording of error recovery flow -->
________________________________________

#### Tester Notes
<!-- Is error recovery smooth or does user need to refresh? -->
________________________________________

---

### Test Case: TC-INT-006 - Error Recovery - Video Detection

**Priority:** P0 (Critical)
**Context Decision:** D-25
**Aspect:** Error Recovery
**Feature:** Video
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify user can retry after video detection error without page refresh.

#### Prerequisites
- Backend is running
- Video file is selected
- Console is open for monitoring

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Click "检测视频" button | Detection starts |
| 2 | Wait 5 seconds | Processing begins |
| 3 | Stop backend (Ctrl+C) | Backend terminates (simulates error) |
| 4 | Wait 5-10 seconds | Error appears |
| 5 | Observe error display | Error message displays |
| 6 | Observe button state | Button becomes enabled after error |
| 7 | Check for stale state | No leftover loading state |
| 8 | Start backend (`python src/run/app.py`) | Backend restarts |
| 9 | Click "检测视频" button (no refresh) | Retry attempt |
| 10 | Observe result | Detection succeeds |

#### Expected Results

**Error Display:**
- Error message displays after simulated failure
- Error is user-friendly (not technical stack trace)
- Inline error displays in VideoPanel
- Status bar shows error state

**Button State:**
- Button becomes enabled after error
- Button text returns to "检测视频"
- No disabled state persists

**No Stale State:**
- No loading indicator persists
- No "检测中..." text remains
- Error clears on retry
- No leftover state from failed attempt

**Retry Success:**
- Second detection attempt succeeds
- No page refresh required
- Application remains responsive
- Results display correctly

#### Recovery Verification Checklist

- [ ] Error message displays
- [ ] Button becomes enabled after error
- [ ] No stale state from failed attempt
- [ ] Retry operation succeeds
- [ ] No need to refresh page
- [ ] Results display correctly on retry

#### Actual Results

**Error message displays:** ⬜ Yes | ❌ No

**Button becomes enabled after error:** ⬜ Yes | ❌ No

**Stale state (loading persists):** ⬜ None | ❌ Loading persists: _________________________

**Retry without refresh:** ⬜ Success | ❌ Failed

**Page refresh required:** ⬜ No | ❌ Yes (problematic)

**Second detection succeeds:** ⬜ Yes | ❌ No

#### Test Evidence
<!-- Screen recording of error recovery flow -->
________________________________________

#### Tester Notes
<!-- Is error recovery smooth for long-running operations? -->
________________________________________

---

## UI State Transition Diagram

**Image Detection State Flow:**

```
Initial State
  ├─ Button: "检测图像" (enabled)
  ├─ Status: "系统就绪。"
  └─ Preview: Empty

    ↓ Click

Loading State
  ├─ Button: "检测中..." (disabled)
  ├─ Status: "正在进行图片检测..."
  ├─ Preview: Empty
  └─ Duration: 2-5 seconds

    ↓ Success OR Error

Success State
  ├─ Button: "检测图像" (enabled)
  ├─ Status: "图片检测完成。" (OK)
  ├─ Preview: Annotated image
  └─ Count: "检测到 N 人"

Error State
  ├─ Button: "检测图像" (enabled)
  ├─ Status: Error message (Error)
  ├─ Preview: Empty
  └─ Inline error: Displayed

    ↓ Click (retry)

Loading State...
  (User can retry without page refresh)
```

---

## Test Execution Summary

- **Total tests:** 6
- **Priority breakdown:** 6 P0 (all critical for UX)
- **Estimated time:** 20-25 minutes
- **Special notes:** Tests require DevTools for detailed observation

## Notes

- **DevTools required:** All tests require browser DevTools for detailed state observation
- **Timing precision:** Use stopwatch or video recording for precise timing measurements
- **Screen recording:** Recommended for capturing immediate feedback tests
- **Network tab:** Required for button disabled state tests
- **Error simulation:** Recovery tests require stopping/starting backend

## Test Execution Tips

1. **Use screen recording:** Capture immediate feedback for timing verification
2. **Network tab filtering:** Filter by endpoint to see only relevant requests
3. **Timing measurements:** Use precise timing (0.1 seconds, not "immediate")
4. **Recovery focus:** Emphasize recovery tests - these are critical for UX
5. **State persistence:** Check for stale state after errors carefully

## Next Steps

After completing this test suite:
1. Update status checkboxes for all executed tests
2. Record actual results for failed tests
3. Attach screen recordings for failures as evidence
4. All test suites complete - proceed to 05-VALIDATION.md for summary

---

**Suite Status:** ⬜ Not Started | 🔄 In Progress | ✅ Complete | ❌ Failed

**Execution Date:** _________________________

**Tester:** _________________________

**Overall Result:** _____ / 6 tests passed (_____ %)
