# Camera Streaming Test Suite

**Purpose:** Validate camera streaming functionality including start/stop controls, status messages, and error handling

**Requirements:** CAM-01, CAM-02, CAM-03, CAM-04, CAM-05

**Last Updated:** 2026-04-04

---

## Overview

This test suite validates all camera streaming functionality in the YOLO11 frontend application. Tests cover camera ID input, stream start/stop controls, video feed display, status message updates, and error handling for invalid inputs and backend failures.

**Test Execution Order:** Execute tests in order to build on state from previous tests.

**Estimated Time:** 30-45 minutes

---

## Quick Reference

| Test ID | Priority | Requirement | Context Decision | Status |
|---------|----------|-------------|------------------|--------|
| TC-CAM-001 | P0 | CAM-02 | D-04 | ⬜ Not Executed |
| TC-CAM-002 | P0 | CAM-03 | D-05 | ⬜ Not Executed |
| TC-CAM-003 | P0 | CAM-04 | — | ⬜ Not Executed |
| TC-CAM-004 | P1 | CAM-05 | D-06 | ⬜ Not Executed |
| TC-CAM-005 | P0 | CAM-01 | D-07 | ⬜ Not Executed |
| TC-CAM-006 | P1 | CAM-01 | D-07 | ⬜ Not Executed |
| TC-CAM-007 | P1 | CAM-01 | D-07 | ⬜ Not Executed |
| TC-CAM-008 | P2 | CAM-01 | D-07 | ⬜ Not Executed |
| TC-CAM-009 | P1 | — | D-19 | ⬜ Not Executed |
| TC-CAM-010 | P2 | — | D-23, D-24 | ⬜ Not Executed |

**Priority Legend:**
- **P0 (Critical):** Must pass for system to be usable
- **P1 (Important):** Should pass for good user experience
- **P2 (Nice-to-have):** Optional enhancements

---

## Test Cases

---

### Test Case: TC-CAM-001 - Start Camera Stream with Valid ID

**Priority:** P0 (Critical)
**Requirement:** CAM-02
**Context Decision:** D-04
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that user can successfully start camera stream when entering a valid camera ID (0, 1, 2, etc.).

#### Prerequisites
- Flask backend is running (`python src/run/app.py`)
- Vite dev server is running (`npm run dev`)
- Browser is open to http://localhost:5173
- Camera device is available on system
- Backend terminal shows "Running on http://127.0.0.1:5000"

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to Camera panel | Camera panel is visible with ID input and buttons |
| 2 | Enter camera ID "0" in input field | Input field shows "0" |
| 3 | Click "启动摄像头" (Start Camera) button | Button click triggers API call |
| 4 | Observe status message (top of page) | Status shows "摄像头 0 已启动。" |
| 5 | Observe preview area | Video stream appears within 2-3 seconds |
| 6 | Observe button state | Button text changes to "停止摄像头" |
| 7 | Wait 10 seconds and observe stream | Stream continues smoothly without interruptions |

#### Test Data: Multiple Camera IDs

| Camera ID | Expected Result | Actual Result | Status |
|-----------|----------------|---------------|--------|
| 0 | Stream starts successfully | ⬜ Pass | ❌ Fail | ⬜ Not Tested |
| 1 (if available) | Stream starts successfully | ⬜ Pass | ❌ Fail | ⬜ Not Tested |
| 2 (if available) | Stream starts successfully | ⬜ Pass | ❌ Fail | ⬜ Not Tested |

#### Expected Results
- **Status message:** "摄像头 {N} 已启动。" (where N is entered camera ID)
- **Status indicator:** Green/OK state in status bar
- **Video stream:** Appears in preview area within 2-3 seconds
- **Stream quality:** Resolution at least 640x480, smooth motion (15+ FPS)
- **Button state:** Changes to "停止摄像头" (Stop Camera)
- **No errors:** Console shows no errors, Network tab shows successful stream request

#### Actual Results

**Status message displayed:** _________________________

**Video stream appears:** ⬜ Yes | ❌ No

**Stream quality assessment:**
- Resolution: ⬜ 640x480 | ⬜ 1280x720 | ⬜ Other: ______
- Frame rate: ⬜ Smooth (15+ FPS) | ⬜ Acceptable (10-15 FPS) | ⬜ Poor (< 10 FPS)
- Visual artifacts: ⬜ None | ⬜ Minor | ⬜ Severe

**Button state:** ⬜ Changed to "停止摄像头" | ❌ Remained "启动摄像头"

**Errors encountered:** _________________________

#### Test Evidence
<!-- Screenshot or description of video stream -->
________________________________________

#### Tester Notes
<!-- Any observations or issues -->
________________________________________

---

### Test Case: TC-CAM-002 - Stop Camera Stream

**Priority:** P0 (Critical)
**Requirement:** CAM-03
**Context Decision:** D-05
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that camera stream truly stops when user clicks "停止摄像头" button (not just hidden).

#### Prerequisites
- Camera stream is currently running (from TC-CAM-001 or manual start)
- Backend and frontend are running
- Video stream is visible in preview area

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Observe active camera stream | Stream is displaying video feed |
| 2 | Click "停止摄像头" (Stop Camera) button | Button click triggers stop action |
| 3 | Observe preview area immediately | Video stream disappears/clears |
| 4 | Observe status message | Status shows "摄像头已停止。" |
| 5 | Observe button state | Button returns to "启动摄像头" |
| 6 | Open browser DevTools Network tab | No active stream requests (verify stream truly stopped) |
| 7 | Wait 5 seconds and verify | Preview remains empty, no stream restarts |

#### Expected Results
- **Preview area:** Completely cleared (no video element or black screen)
- **Status message:** "摄像头已停止。"
- **Status indicator:** Non-OK state in status bar
- **Button state:** Returns to "启动摄像头" (Start Camera)
- **Network activity:** No active `/video_feed` requests in DevTools
- **Stream truly stopped:** Not just hidden, video feed request is terminated
- **No errors:** Console shows no errors

#### Actual Results

**Preview area cleared:** ⬜ Yes | ❌ No

**Status message displayed:** _________________________

**Button state:** ⬜ Returned to "启动摄像头" | ❌ Remained "停止摄像头"

**Network activity:** ⬜ No active stream requests | ❌ Stream continues in background

**Errors encountered:** _________________________

#### Test Evidence
<!-- Screenshot of cleared preview area -->
________________________________________

#### Tester Notes
<!-- Verify stream is truly stopped, not just hidden -->
________________________________________

---

### Test Case: TC-CAM-003 - Camera Feed Displays in Preview

**Priority:** P0 (Critical)
**Requirement:** CAM-04
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that camera video stream displays with acceptable quality and performance in the preview area.

#### Prerequisites
- Camera stream is currently running (camera ID 0 or other valid ID)
- Backend and frontend are running
- Network connection is stable

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Observe video stream in preview area | Stream is visible and playing |
| 2 | Wait 5 seconds for stream to stabilize | Stream quality improves/stabilizes |
| 3 | Assess video resolution | Resolution is at least 640x480 |
| 4 | Assess frame rate (subjective) | Motion is smooth (15+ FPS) |
| 5 | Check for visual artifacts | No blocking, pixelation, or distortion |
| 6 | Check color accuracy | Colors are natural, not washed out |
| 7 | Monitor for stuttering | Motion is smooth, not freezing/skipping |
| 8 | Check browser console for errors | No video-related errors |
| 9 | Monitor memory usage (DevTools) | Memory is stable, not climbing continuously |

#### Expected Results

**Visual Quality:**
- **Resolution:** Minimum 640x480 (VGA), preferably 1280x720 (HD)
- **Frame rate:** 15+ FPS (subjective assessment: smooth motion)
- **Artifacts:** No visual artifacts (blocking, pixelation, tearing)
- **Colors:** Accurate and natural, not washed out or oversaturated
- **Stability:** No stuttering, freezing, or frame drops

**Performance:**
- **Memory usage:** Stable (not climbing continuously)
- **CPU usage:** Reasonable (< 50% on modern systems)
- **Network:** Consistent stream data flow
- **Latency:** < 2 seconds between camera movement and display update

#### Quality Checklist

- [ ] Resolution is 640x480 or higher
- [ ] Frame rate is smooth (15+ FPS subjective)
- [ ] No visual artifacts (blocking, pixelation)
- [ ] Colors are accurate
- [ ] No stuttering or freezing
- [ ] Browser console shows no errors
- [ ] Memory usage is stable

#### Actual Results

**Resolution detected:** _________________________

**Frame rate assessment:**
- ⬜ Smooth (15+ FPS)
- ⬜ Acceptable (10-15 FPS)
- ⬜ Poor (< 10 FPS)

**Visual artifacts:**
- ⬜ None
- ⬜ Minor (acceptable)
- ⬜ Severe (unacceptable)

**Colors:** ⬜ Accurate | ⬜ Washed out | ⬜ Oversaturated

**Stuttering:** ⬜ None | ⬜ Occasional | ⬜ Frequent

**Browser console errors:** ⬜ None | ❌ Errors: _________________________

#### Performance Metrics (Optional)

**Memory usage (start):** _________________________
**Memory usage (after 1 min):** _________________________
**CPU usage:** _________________________

#### Test Evidence
<!-- Screenshot of video stream or description -->
________________________________________

#### Tester Notes
<!-- Describe overall video quality and any issues -->
________________________________________

---

### Test Case: TC-CAM-004 - Status Message Updates on Start/Stop

**Priority:** P1 (Important)
**Requirement:** CAM-05
**Context Decision:** D-06
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that status bar displays correct Chinese messages when camera starts and stops.

#### Prerequisites
- Backend and frontend are running
- Status monitor is visible at bottom of page
- Camera device is available

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Observe initial status message | Status shows "系统就绪。" or similar |
| 2 | Enter valid camera ID (e.g., "0") | Input field shows ID |
| 3 | Click "启动摄像头" button | Camera starts, status updates |
| 4 | Observe status message after start | Status shows "摄像头 0 已启动。" |
| 5 | Observe status bar color/state | Status bar shows OK/green state |
| 6 | Click "停止摄像头" button | Camera stops, status updates |
| 7 | Observe status message after stop | Status shows "摄像头已停止。" |
| 8 | Observe status bar color/state | Status bar shows non-OK/yellow state |

#### Expected Results

**On Camera Start:**
- **Message:** "摄像头 {N} 已启动。" (where N is entered camera ID)
- **Status state:** OK (green background, checkmark icon)
- **Timing:** Message updates immediately (within 0.5 seconds)

**On Camera Stop:**
- **Message:** "摄像头已停止。"
- **Status state:** Non-OK (yellow/warning background)
- **Timing:** Message updates immediately

**Message Format:**
- Chinese text is correct and readable
- Camera ID is correctly inserted in start message
- No typos or encoding issues
- Message is consistent with original frontend behavior

#### Actual Results

**Status message on start:** _________________________
**Matches expected "摄像头 {N} 已启动。":** ⬜ Yes | ❌ No

**Status bar state on start:** ⬜ OK/Green | ❌ Non-OK/Yellow

**Status message on stop:** _________________________
**Matches expected "摄像头已停止。":** ⬜ Yes | ❌ No

**Status bar state on stop:** ⬜ Non-OK/Yellow | ❌ OK/Green

**Message timing:** ⬜ Immediate | ❌ Delayed: _____ seconds

**Text encoding issues:** ⬜ None | ❌ Issues: _________________________

#### Test Evidence
<!-- Screenshots of status messages -->
________________________________________

#### Tester Notes
<!-- Verify Chinese text displays correctly -->
________________________________________

---

### Test Case: TC-CAM-005 - Invalid Camera ID - Negative Number

**Priority:** P0 (Critical)
**Requirement:** CAM-01
**Context Decision:** D-07
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify error handling when user enters a negative camera ID.

#### Prerequisites
- Backend and frontend are running
- Camera panel is visible

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Enter camera ID "-1" in input field | Input field shows "-1" |
| 2 | Click "启动摄像头" button | Button attempts to start stream |
| 3 | Observe status message | Error message displays |
| 4 | Observe preview area | No video stream appears |
| 5 | Observe button state | Button remains clickable/enabled |
| 6 | Check browser console | May show error or warning |
| 7 | Enter valid camera ID "0" | System allows retry |

#### Expected Results
- **Error message:** User-friendly error (e.g., "摄像头ID无效" or "请输入有效的摄像头ID")
- **Status state:** Error state (red background)
- **Preview area:** Remains empty or shows placeholder
- **Button state:** Remains "启动摄像头" (not changed to "停止")
- **Recovery possible:** User can enter valid ID and retry
- **No crash:** Application remains responsive
- **Backend behavior:** Backend may return error or invalid stream

#### Actual Results

**Error message displayed:** _________________________

**Status bar state:** ⬜ Error/Red | ❌ OK/Green | ❌ Non-OK/Yellow

**Preview area:** ⬜ Remains empty | ❌ Shows stream

**Button state:** ⬜ Remains "启动摄像头" | ❌ Changed to "停止摄像头"

**Application remains responsive:** ⬜ Yes | ❌ No (froze/crashed)

**Recovery test - Enter valid ID "0":**
- System allows retry: ⬜ Yes | ❌ No
- Camera starts with valid ID: ⬜ Yes | ❌ No

#### Test Evidence
<!-- Screenshot of error state -->
________________________________________

#### Tester Notes
<!-- Is error message user-friendly or technical? -->
________________________________________

---

### Test Case: TC-CAM-006 - Invalid Camera ID - Non-Numeric

**Priority:** P1 (Important)
**Requirement:** CAM-01
**Context Decision:** D-07
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify error handling when user enters non-numeric text in camera ID field.

#### Prerequisites
- Backend and frontend are running
- Camera panel is visible

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Enter camera ID "abc" in input field | Input field shows "abc" |
| 2 | Click "启动摄像头" button | Button attempts to start stream |
| 3 | Observe status message | Error message displays |
| 4 | Observe preview area | No video stream appears |
| 5 | Observe button state | Button remains clickable |
| 6 | Check browser Network tab | May show failed request |
| 7 | Enter valid camera ID "0" | System recovers and allows retry |

#### Expected Results
- **Error message:** "请输入有效的摄像头ID" or similar
- **Status state:** Error state
- **Preview area:** Remains empty
- **Button state:** Unchanged
- **Recovery:** System allows retry with valid ID
- **Validation:** Ideally, frontend validates input before API call (but not required)

#### Actual Results

**Error message displayed:** _________________________

**Status bar state:** ⬜ Error | ❌ OK | ❌ Non-OK

**Preview area:** ⬜ Empty | ❌ Shows stream

**Recovery test - Enter valid ID "0":** ⬜ Success | ❌ Failed

#### Test Evidence
________________________________________

#### Tester Notes
<!-- Does frontend validate input or does backend reject it? -->
________________________________________

---

### Test Case: TC-CAM-007 - Invalid Camera ID - Very Large Number

**Priority:** P1 (Important)
**Requirement:** CAM-01
**Context Decision:** D-07
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify error handling when user enters a very large camera ID that doesn't exist on the system.

#### Prerequisites
- Backend and frontend are running
- Camera panel is visible

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Enter camera ID "9999" in input field | Input field shows "9999" |
| 2 | Click "启动摄像头" button | Button attempts to start stream |
| 3 | Observe status message | Error message displays |
| 4 | Observe preview area | No video stream appears |
| 5 | Wait 5 seconds and observe | No stream loads |
| 6 | Enter valid camera ID "0" | System recovers |

#### Expected Results
- **Error message:** "摄像头不存在" or "请输入有效的摄像头ID"
- **Status state:** Error state
- **Preview area:** Remains empty (no timeout or endless loading)
- **Timeout:** Error appears within 5 seconds (not endless loading)
- **Recovery:** System allows retry with valid ID

#### Actual Results

**Error message displayed:** _________________________

**Time to error:** _____ seconds

**Preview area:** ⬜ Empty | ❌ Shows stream | ❌ Endless loading

**Recovery test:** ⬜ Success | ❌ Failed

#### Test Evidence
________________________________________

#### Tester Notes
<!-- How long does it take to detect invalid camera? -->
________________________________________

---

### Test Case: TC-CAM-008 - Invalid Camera ID - Empty

**Priority:** P2 (Nice-to-have)
**Requirement:** CAM-01
**Context Decision:** D-07
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify error handling when user leaves camera ID field empty and clicks start button.

#### Prerequisites
- Backend and frontend are running
- Camera panel is visible
- Camera ID field is empty

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Ensure camera ID field is empty | Input field is blank |
| 2 | Click "启动摄像头" button | Button attempts action |
| 3 | Observe status message | Error message displays |
| 4 | Observe preview area | No video stream appears |
| 5 | Enter valid camera ID "0" | System recovers |

#### Expected Results
- **Error message:** "请输入摄像头ID" or similar
- **Status state:** Error state
- **No API call:** Ideally, frontend validates before calling backend (optional)
- **Recovery:** System allows retry

#### Actual Results

**Error message displayed:** _________________________

**Frontend validation:** ⬜ Yes (validates before API call) | ❌ No (calls backend)

**Recovery test:** ⬜ Success | ❌ Failed

#### Test Evidence
________________________________________

#### Tester Notes
<!-- Is empty field caught by frontend validation or backend error? -->
________________________________________

---

### Test Case: TC-CAM-009 - Backend Unavailable

**Priority:** P1 (Important)
**Context Decision:** D-19
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify graceful handling when backend is not running.

#### Prerequisites
- Frontend is running
- Backend is STOPPED (Ctrl+C to stop if running)
- Camera panel is visible

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Verify backend is stopped | Terminal shows no Flask process |
| 2 | Enter valid camera ID "0" | Input field shows "0" |
| 3 | Click "启动摄像头" button | Button attempts API call |
| 4 | Wait 2-3 seconds | System detects connection failure |
| 5 | Observe status message | User-friendly error displays |
| 6 | Observe button state | Button remains enabled for retry |
| 7 | Start backend (`python src/run/app.py`) | Backend starts |
| 8 | Click "启动摄像头" again | Camera starts successfully |

#### Expected Results
- **Error message:** User-friendly (not technical stack trace)
  - Examples: "无法连接到后端服务" or "网络错误，请稍后重试"
  - NOT: "ERR_CONNECTION_REFUSED" or technical HTTP errors
- **Status state:** Error state
- **Preview area:** Remains empty
- **Button state:** Remains enabled (allows retry after backend starts)
- **No crash:** Application remains fully responsive
- **Recovery:** After starting backend, retry succeeds

#### Actual Results

**Error message displayed:** _________________________
**Is user-friendly:** ⬜ Yes | ❌ No (too technical)

**Status bar state:** ⬜ Error | ❌ OK | ❌ Non-OK

**Button state:** ⬜ Enabled (allows retry) | ❌ Disabled (blocks retry)

**Recovery test - Start backend and retry:** ⬜ Success | ❌ Failed

#### Test Evidence
<!-- Screenshot of error message -->
________________________________________

#### Tester Notes
<!-- Is error message suitable for non-technical users? -->
________________________________________

---

### Test Case: TC-CAM-010 - Multiple Start/Stop Cycles

**Priority:** P2 (Nice-to-have)
**Context Decision:** D-23, D-24
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify system handles repeated start/stop cycles without state pollution or memory leaks.

#### Prerequisites
- Backend and frontend are running
- Camera device is available
- Multiple camera IDs available (0, 1, 2)

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Start camera with ID "0" | Stream starts, status updates |
| 2 | Stop camera | Stream stops, status updates |
| 3 | Start camera with ID "1" | Stream starts, status updates |
| 4 | Stop camera | Stream stops, status updates |
| 5 | Start camera with ID "0" again | Stream starts, status updates |
| 6 | Stop camera | Stream stops, status updates |
| 7 | Repeat steps 1-6 two more times | Total 6 start/stop cycles |
| 8 | Check browser memory usage | Memory is stable (not climbing) |
| 9 | Check console for errors | No accumulated errors or warnings |

#### Expected Results
- **All cycles successful:** Each start/stop works correctly
- **No state pollution:** Previous camera ID doesn't affect new start
- **Status messages:** Correct for each cycle (camera ID updates correctly)
- **Memory stability:** Memory usage doesn't climb significantly
- **No errors:** Console shows no errors after multiple cycles
- **UI responsiveness:** Application remains responsive throughout
- **Resource cleanup:** Previous stream is properly terminated before new stream starts

#### Cycle Results Table

| Cycle | Camera ID | Start Result | Stop Result | Status Message |
|-------|-----------|--------------|-------------|----------------|
| 1 | 0 | ⬜ Pass | ❌ Fail | _________________________ |
| 2 | 0 | ⬜ Pass | ❌ Fail | _________________________ |
| 3 | 1 | ⬜ Pass | ❌ Fail | _________________________ |
| 4 | 1 | ⬜ Pass | ❌ Fail | _________________________ |
| 5 | 0 | ⬜ Pass | ❌ Fail | _________________________ |
| 6 | 0 | ⬜ Pass | ❌ Fail | _________________________ |

#### Actual Results

**Total cycles completed successfully:** _____ / 6

**Memory usage (start):** _________________________
**Memory usage (after 6 cycles):** _________________________
**Memory increase:** ⬜ Minimal (< 50MB) | ❌ Significant (> 50MB)

**Console errors after 6 cycles:** ⬜ None | ❌ Errors: _________________________

**UI responsiveness:** ⬜ Fully responsive | ❌ Laggy | ❌ Froze

#### Test Evidence
<!-- Memory usage screenshots from DevTools -->
________________________________________

#### Tester Notes
<!-- Any degradation noticed after multiple cycles? -->
________________________________________

---

## Test Execution Summary

- **Total tests:** 10
- **Priority breakdown:**
  - P0 (Critical): 4 tests (TC-CAM-001, TC-CAM-002, TC-CAM-003, TC-CAM-005)
  - P1 (Important): 4 tests (TC-CAM-004, TC-CAM-006, TC-CAM-007, TC-CAM-009)
  - P2 (Nice-to-have): 2 tests (TC-CAM-008, TC-CAM-010)
- **Estimated time:** 30-45 minutes

## Notes

- **Camera availability:** If camera is unavailable, mark tests as "Blocked" and document reason
- **Multiple cameras:** If only one camera available, test IDs 0, 1, 2 may fail — this is expected
- **Stream quality:** Subjective assessment is acceptable for frame rate and visual quality
- **Memory leaks:** Use browser DevTools Memory tab for detailed analysis if issues suspected

## Next Steps

After completing this test suite:
1. Update status checkboxes for all executed tests
2. Record actual results for failed tests
3. Attach screenshots for failures as evidence
4. Proceed to next test suite: 02-image-detection-tests.md

---

**Suite Status:** ⬜ Not Started | 🔄 In Progress | ✅ Complete | ❌ Failed

**Execution Date:** _________________________

**Tester:** _________________________

**Overall Result:** _____ / 10 tests passed (_____ %)
