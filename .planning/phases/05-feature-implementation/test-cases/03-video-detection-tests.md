# Video Detection Test Suite

**Purpose:** Validate video detection functionality including file selection, processing state, result display, video playback, statistics, and error handling

**Requirements:** VID-01, VID-02, VID-03, VID-04, VID-05, VID-06

**Last Updated:** 2026-04-04

---

## Overview

This test suite validates all video detection functionality in the YOLO11 frontend application. Tests cover file selection, detection trigger, long-running processing state, annotated result display, video playback, statistics display, button state management, and error handling for missing files, invalid formats, timeouts, and network failures.

**Important:** Video detection takes longer than image detection (10-30 seconds depending on video length). Patience is required during testing.

**Test Execution Order:** Execute tests in order to build on state from previous tests.

**Estimated Time:** 45-60 minutes

---

## Quick Reference

| Test ID | Priority | Requirement | Context Decision | Status |
|---------|----------|-------------|------------------|--------|
| TC-VID-001 | P0 | VID-01 | D-13 | ⬜ Not Executed |
| TC-VID-002 | P0 | VID-02 | D-13, D-15, D-16 | ⬜ Not Executed |
| TC-VID-003 | P0 | VID-03 | D-15, D-24 | ⬜ Not Executed |
| TC-VID-004 | P0 | VID-04 | D-17 | ⬜ Not Executed |
| TC-VID-005 | P0 | VID-05 | D-16 | ⬜ Not Executed |
| TC-VID-006 | P0 | VID-06 | D-15, D-23 | ⬜ Not Executed |
| TC-VID-007 | P0 | VID-02 | D-11 | ⬜ Not Executed |
| TC-VID-008 | P1 | VID-02 | D-14 | ⬜ Not Executed |
| TC-VID-009 | P1 | VID-02 | D-14 | ⬜ Not Executed |
| TC-VID-010 | P1 | — | D-20 | ⬜ Not Executed |
| TC-VID-011 | P1 | — | D-19 | ⬜ Not Executed |
| TC-VID-012 | P2 | — | D-21 | ⬜ Not Executed |

**Priority Legend:**
- **P0 (Critical):** Must pass for system to be usable
- **P1 (Important):** Should pass for good user experience
- **P2 (Nice-to-have):** Optional enhancements

---

## Test Cases

---

### Test Case: TC-VID-001 - Select Valid Video File

**Priority:** P0 (Critical)
**Requirement:** VID-01
**Context Decision:** D-13
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that user can select MP4/AVI video files through the file input.

#### Prerequisites
- Backend and frontend are running
- Test video files are available (MP4, AVI formats)
- Video panel is visible

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to Video panel | Video panel is visible with file input |
| 2 | Click file input button | File selection dialog opens |
| 3 | Select an MP4 file | Dialog closes, file is selected |
| 4 | Observe file input display | Selected filename is visible |
| 5 | Select an AVI file | File selection works for AVI format |
| 6 | Observe no errors | Console shows no file-related errors |

#### Test Data: Multiple Formats

| File Format | Test File | Duration | Expected Result | Actual Result | Status |
|-------------|-----------|----------|----------------|---------------|--------|
| MP4 | test-video-short.mp4 | 10-15s | File selected successfully | ⬜ Pass | ❌ Fail | ⬜ Not Tested |
| AVI | test-video-medium.avi | 20-30s | File selected successfully | ⬜ Pass | ❌ Fail | ⬜ Not Tested |

#### Expected Results
- **File dialog opens:** Native browser file picker appears
- **File accepted:** MP4 and AVI files are accepted
- **Filename visible:** Selected filename displays in file input area
- **No errors:** Console shows no file access errors
- **File accessible:** Selected file can be accessed for upload

#### Actual Results

**File dialog opens:** ⬜ Yes | ❌ No

**MP4 file selection:** ⬜ Success | ❌ Failed

**AVI file selection:** ⬜ Success | ❌ Failed

**Filename visible:** ⬜ Yes | ❌ No

**Console errors:** ⬜ None | ❌ Errors: _________________________

#### Test Evidence
<!-- Screenshot of file selection -->
________________________________________

#### Tester Notes
<!-- Is file input user-friendly? -->
________________________________________

---

### Test Case: TC-VID-002 - Trigger Video Detection - Success Case

**Priority:** P0 (Critical)
**Requirement:** VID-02
**Context Decision:** D-13, D-15, D-16
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that video detection completes successfully and displays annotated results with statistics.

#### Prerequisites
- Backend and frontend are running
- Video file is selected (10-30 seconds recommended for faster testing)
- Video panel is visible

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select test video file | Filename displays in file input |
| 2 | Click "检测视频" (Detect Video) button | Button triggers detection |
| 3 | Observe button state immediately | Button text changes to "检测中..." |
| 4 | Observe button disabled state | Button is disabled (not clickable) |
| 5 | Observe status message | Status shows long-running message |
| 6 | Start timer | Record processing time |
| 7 | Wait for detection to complete (10-30 seconds) | Processing completes |
| 8 | Stop timer and record duration | _____ seconds |
| 9 | Observe preview area | Annotated video displays |
| 10 | Observe video statistics | Stats show "总帧数：{N}，平均 FPS：{X.X}" |
| 11 | Observe button state after completion | Button returns to "检测视频" |
| 12 | Observe status message after completion | Status shows "视频检测完成。" |

#### Expected Results

**Immediate Feedback (0-2 seconds):**
- Button text: "检测中..." (Detecting...)
- Button state: Disabled (grayed out, not clickable)
- Status message: "正在进行视频检测，耗时可能较长，请稍候。"

**During Processing:**
- Button remains disabled
- Loading indication persists
- No timeout errors (within 60 seconds)
- Console shows pending request

**Upon Completion:**
- Annotated video displays in preview area
- Video shows bounding boxes and tracking IDs
- Statistics display: "总帧数：{N}，平均 FPS：{X.X}"
- Button text returns to "检测视频"
- Button becomes enabled again
- Status message: "视频检测完成。" or backend-provided message
- Status state: OK (green)
- Video auto-plays or shows play button

#### Actual Results

**Button state changes:**
- Text changes to "检测中...": ⬜ Yes | ❌ No
- Button disabled: ⬜ Yes | ❌ No

**Processing time:** _____ seconds

**File size:** _____ MB
**Video duration:** _____ seconds

**Annotated video displays:** ⬜ Yes | ❌ No

**Bounding boxes visible:** ⬜ Yes | ❌ No

**Video statistics displayed:** _________________________

**Status message on completion:** _________________________

**Video auto-plays:** ⬜ Yes | ❌ No (blocked by browser)

**Button re-enabled after completion:** ⬜ Yes | ❌ No

#### Test Evidence
<!-- Screenshot of detection result -->
________________________________________

#### Tester Notes
<!-- How long did detection take? Was result smooth? -->
________________________________________

---

### Test Case: TC-VID-003 - Processing State Indication

**Priority:** P0 (Critical)
**Requirement:** VID-03
**Context Decision:** D-15, D-24
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that user receives clear feedback during long-running video detection operation.

#### Prerequisites
- Backend and frontend are running
- Video file is selected (30+ seconds recommended for this test)
- Console is open for monitoring

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select video file (30+ seconds) | File selected |
| 2 | Click "检测视频" button | Detection starts |
| 3 | Start timer | Record feedback timing |
| 4 | Observe button state (0-2 seconds) | Immediate changes visible |
| 5 | Observe status message (0-2 seconds) | Message updates immediately |
| 6 | Observe loading indication | Loading indicator visible |
| 7 | Wait for 30 seconds | Processing continues |
| 8 | Observe state during waiting | State persists, no premature completion |
| 9 | Wait for completion | Processing finishes |
| 10 | Observe final state | State returns to normal |

#### Expected Results

**Immediate Feedback (0-2 seconds):**
- Button text: "检测中..." (instant change)
- Button disabled: Yes (instant change)
- Status message: "正在进行视频检测，耗时可能较长，请稍候。"
- Loading indicator: Visible (spinner or progress bar)
- No delay or lag: UI updates immediately

**During Processing (2+ seconds):**
- Button remains disabled
- Loading state persists
- No timeout errors (within 60 seconds)
- Status message remains consistent
- Console shows pending request

**Upon Completion:**
- Button text: "检测视频" (restored)
- Button enabled: Yes
- Loading indicator: Disappears
- Status message: "视频检测完成。"
- Results display: Annotated video + statistics

#### State Verification Checklist

**Immediate (0-2 seconds):**
- [ ] Button text change is instant (< 0.5s)
- [ ] Button disabled state is instant
- [ ] Status message updates immediately
- [ ] Loading indicator visible
- [ ] No UI freeze or lag

**During Processing:**
- [ ] Button remains disabled
- [ ] Loading persists
- [ ] No timeout errors (within 60s)
- [ ] Status message consistent

**Upon Completion:**
- [ ] Button returns to normal state
- [ ] Loading disappears
- [ ] Results display correctly

#### Actual Results

**Button text change timing:** ⬜ Instant (< 0.5s) | ❌ Delayed: _____ seconds

**Button disabled timing:** ⬜ Instant (< 0.5s) | ❌ Delayed: _____ seconds

**Status message timing:** ⬜ Instant (< 0.5s) | ❌ Delayed: _____ seconds

**Loading indicator:** ⬜ Visible | ❌ Not visible

**UI freeze or lag:** ⬜ None | ❌ Laggy: _________________________

**Processing duration:** _____ seconds

**Timeout errors:** ⬜ None | ❌ Timeout occurred at _____ seconds

#### Test Evidence
<!-- Screenshots of processing state -->
________________________________________

#### Tester Notes
<!-- Is user clearly informed about long processing time? -->
________________________________________

---

### Test Case: TC-VID-004 - Display Annotated Video Result

**Priority:** P0 (Critical)
**Requirement:** VID-04
**Context Decision:** D-17
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that detection result video displays correctly with bounding boxes and playback controls.

#### Prerequisites
- Video detection has completed successfully
- Annotated result video is available

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Examine result video in preview area | Video player is visible |
| 2 | Check for bounding boxes | Bounding boxes visible around detected people |
| 3 | Check for tracking IDs | Tracking IDs visible near boxes |
| 4 | Test play button | Video plays smoothly |
| 5 | Test pause button | Video pauses correctly |
| 6 | Test volume control | Volume slider works |
| 7 | Test fullscreen (if available) | Fullscreen mode works |
| 8 | Check video quality | No pixelation or artifacts |
| 9 | Verify video is not cached | Video URL has timestamp parameter |
| 10 | Right-click video and inspect | Video src includes `?t={timestamp}` |

#### Expected Results

**Video Display:**
- Video loads completely without errors
- Video player is visible in preview area
- Resolution matches or exceeds original video
- No pixelation or compression artifacts

**Annotations:**
- Bounding boxes visible around each detected person
- Tracking IDs visible near boxes (e.g., "ID: 1", "ID: 2")
- Boxes and IDs are clearly visible (good contrast)
- Boxes update frame-by-frame (tracking)

**Playback Controls:**
- Play/Pause button works
- Volume control works
- Timeline/scrubber works (if available)
- Fullscreen button works (if available)
- Video auto-plays or shows play button (if autoplay blocked)

**Cache Prevention:**
- Video URL includes timestamp parameter: `?t={timestamp}`
- Each detection generates new URL (prevents browser caching)

#### Quality Checklist

- [ ] Video loads completely
- [ ] Bounding boxes are visible
- [ ] Tracking IDs are visible
- [ ] Play/Pause works
- [ ] Volume control works
- [ ] No video artifacts or pixelation
- [ ] Video URL has timestamp parameter
- [ ] Video playback is smooth

#### Actual Results

**Video loads completely:** ⬜ Yes | ❌ No

**Bounding boxes visible:** ⬜ Yes | ❌ No

**Tracking IDs visible:** ⬜ Yes | ❌ No

**Play/Pause works:** ⬜ Yes | ❌ No

**Volume control works:** ⬜ Yes | ❌ No

**Video artifacts:** ⬜ None | ⬜ Minor | ⬜ Severe

**Video URL has timestamp:** ⬜ Yes | ❌ No
**URL:** _________________________

**Video auto-plays:** ⬜ Yes | ❌ No (autoplay blocked)

#### Test Evidence
<!-- Screenshot of video player with annotations -->
________________________________________

#### Tester Notes
<!-- Are tracking IDs stable across frames? Any tracking errors? -->
________________________________________

---

### Test Case: TC-VID-005 - Video Statistics Display

**Priority:** P0 (Critical)
**Requirement:** VID-05
**Context Decision:** D-16
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that video statistics display correctly after detection and values are reasonable.

#### Prerequisites
- Video detection has completed successfully
- Result video is displayed

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Observe statistics text below result video | Text displays video stats |
| 2 | Read statistics values | Format: "总帧数：{N}，平均 FPS：{X.X}" |
| 3 | Note frame count value | Record for validation |
| 4 | Note average FPS value | Record for validation |
| 5 | Calculate expected frame count | video_duration * fps ≈ frame_count |
| 6 | Compare expected vs actual | Values should be approximately equal |

#### Expected Results

**Format:**
- Chinese text: "总帧数：{N}，平均 FPS：{X.X}"
- Displays below result video
- Values are clearly readable

**Frame Count Validation:**
- Frame count should be reasonable for video duration
- Example: 30 FPS * 10 seconds = ~300 frames
- Allow ±20% tolerance (encoding variations)

**Average FPS Validation:**
- FPS should be between 15-60 (typical video range)
- FPS should match expected video frame rate
- Allow ±5 FPS tolerance

**Example Calculations:**

| Video Duration | Expected FPS | Expected Frames | Actual Frames | Match? |
|----------------|--------------|-----------------|---------------|--------|
| 10 seconds | 30 fps | ~300 frames | _____ | ⬜ | ❌ |
| 20 seconds | 30 fps | ~600 frames | _____ | ⬜ | ❌ |
| 30 seconds | 30 fps | ~900 frames | _____ | ⬜ | ❌ |

#### Actual Results

**Statistics format matches "总帧数：{N}，平均 FPS：{X.X}":** ⬜ Yes | ❌ No

**Displayed statistics:** _________________________

**Frame count:** _____ frames

**Average FPS:** _____ fps

**Video duration:** _____ seconds

**Expected frames (duration * fps):** _____ frames

**Values match (within tolerance):** ⬜ Yes | ❌ No (difference: _____ %)

**Chinese text displays correctly:** ⬜ Yes | ❌ No

#### Test Evidence
<!-- Screenshot of statistics display -->
________________________________________

#### Tester Notes
<!-- Are statistics accurate? Any calculation errors? -->
________________________________________

---

### Test Case: TC-VID-006 - Button Disabled State During Processing

**Priority:** P0 (Critical)
**Requirement:** VID-06
**Context Decision:** D-15, D-23
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that button prevents duplicate detection requests during processing.

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
| 4 | Immediately try to click again (within 1 second) | Button should not respond |
| 5 | Try to click 3-4 more times rapidly | Only one request should be sent |
| 6 | Observe Network tab | Only one POST /detect/video request |
| 7 | Wait for completion | Processing completes |
| 8 | Observe button state | Button re-enabled after completion |

#### Expected Results
- **Button disabled during processing:** Not clickable, grayed out
- **Multiple clicks ignored:** No response to additional clicks
- **Single API request:** Only one POST /detect/video in Network tab
- **No duplicate processing:** Backend processes video only once
- **Button re-enabled after completion:** Clickable again
- **Request timing:** First request sent immediately, no subsequent requests

#### Network Tab Verification

- **Number of POST /detect/video requests:** Should be 1
- **Request timing:** All requests (if any) should be at same time (duplicate attempts)
- **Response status:** Only one request should get response (others should be blocked by frontend)

#### Actual Results

**Button disabled during processing:** ⬜ Yes | ❌ No

**Multiple clicks ignored:** ⬜ Yes | ❌ No (multiple requests sent)

**Number of POST /detect/video requests:** _____

**Only one request got response:** ⬜ Yes | ❌ No

**Button re-enabled after completion:** ⬜ Yes | ❌ No

#### Test Evidence
<!-- Screenshot of Network tab showing single request -->
________________________________________

#### Tester Notes
<!-- Is button disabling effective? Can user trigger duplicate requests? -->
________________________________________

---

### Test Case: TC-VID-007 - Empty File Selection Error

**Priority:** P0 (Critical)
**Requirement:** VID-02
**Context Decision:** D-11
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify error handling when user clicks detection button without selecting a file.

#### Prerequisites
- Backend and frontend are running
- No file is selected in video panel

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Verify no file is selected | File input shows "No file chosen" or similar |
| 2 | Click "检测视频" button | Button attempts action |
| 3 | Observe error message | Error displays: "请先选择一个视频文件" |
| 4 | Observe status message | Status shows error |
| 5 | Observe button state | Button remains enabled for retry |
| 6 | Select valid video file | System allows file selection |
| 7 | Click "检测视频" again | Detection proceeds normally |

#### Expected Results
- **Error message:** "请先选择一个视频文件" (Please select a video file first)
- **Error location:** Below button or in status bar
- **Error style:** Error state (red background or styling)
- **No API call:** Detection API is not called (frontend validation)
- **Button state:** Remains enabled (allows retry after file selection)
- **Recovery:** User can select file and retry detection
- **No crash:** Application remains responsive

#### Actual Results

**Error message displayed:** _________________________

**Matches expected "请先选择一个视频文件":** ⬜ Yes | ❌ No

**Error message location:** ⬜ Inline (below button) | ⬜ Status bar | ⬜ Both

**API call made:** ⬜ No (good) | ❌ Yes (unnecessary)

**Button state:** ⬜ Enabled (allows retry) | ❌ Disabled (blocks retry)

**Recovery test - Select file and retry:** ⬜ Success | ❌ Failed

#### Test Evidence
<!-- Screenshot of error message -->
________________________________________

#### Tester Notes
<!-- Is error message clear and actionable? -->
________________________________________

---

### Test Case: TC-VID-008 - Invalid File Format - Text File

**Priority:** P1 (Important)
**Requirement:** VID-02
**Context Decision:** D-14
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify error handling when user selects a non-video file (.txt).

#### Prerequisites
- Backend and frontend are running
- Text file available for testing (test-file.txt)

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select .txt file via file input | File input shows selected file |
| 2 | Click "检测视频" button | System detects invalid format |
| 3 | Observe error message | Error about invalid format displays |
| 4 | Observe preview area | No result displays |
| 5 | Check browser Network tab | May show validation error or no request |
| 6 | Select valid video file | System recovers and allows retry |

#### Expected Results
- **Error message:** Clear message about invalid format
  - Examples: "不支持的文件格式" or "请选择视频文件 (MP4, AVI)"
  - NOT: Technical error or generic message
- **Error location:** Inline error below button or status bar
- **No processing:** Detection API is not called for invalid file
- **Recovery:** System allows selecting valid file and retrying
- **No crash:** Application remains responsive

#### Actual Results

**Error message displayed:** _________________________

**Is user-friendly:** ⬜ Yes | ❌ No (too technical)

**API call made:** ⬜ No (good) | ❌ Yes (unnecessary)

**Recovery test - Select valid video:** ⬜ Success | ❌ Failed

#### Test Evidence
________________________________________

#### Tester Notes
<!-- Is format validation on frontend or backend? -->
________________________________________

---

### Test Case: TC-VID-009 - Invalid File Format - Image File

**Priority:** P1 (Important)
**Requirement:** VID-02
**Context Decision:** D-14
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify error handling when user selects an image file for video detection.

#### Prerequisites
- Backend and frontend are running
- Image file available for testing (test-image.jpg)

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select .jpg file via file input | File input shows selected file |
| 2 | Click "检测视频" button | System detects invalid format |
| 3 | Observe error message | Error about video format required displays |
| 4 | Select valid video file | System recovers |

#### Expected Results
- **Error message:** Clear message that video format is required
- **No processing:** Detection API not called
- **Recovery:** System allows retry with valid video

#### Actual Results

**Error message displayed:** _________________________

**Recovery test:** ⬜ Success | ❌ Failed

#### Test Evidence
________________________________________

#### Tester Notes
<!-- Are error messages consistent across invalid formats? -->
________________________________________

---

### Test Case: TC-VID-010 - Processing Timeout (60 seconds)

**Priority:** P1 (Important)
**Context Decision:** D-20
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify system handles long processing without timeout errors (tests 60-second timeout configuration).

#### Prerequisites
- Backend and frontend are running
- Large or long video file available (60+ seconds or high resolution)

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select large/long video file | File selected |
| 2 | Note file size and duration | Record for reference |
| 3 | Click "检测视频" button | Detection starts |
| 4 | Start timer | Record processing time |
| 5 | Wait for processing (may take 60+ seconds) | Processing continues |
| 6 | Observe button state during waiting | Button remains disabled |
| 7 | Observe status message | Status indicates processing |
| 8 | Wait for completion or timeout | Monitor for timeout errors |
| 9 | Stop timer when complete | Record actual duration |

#### Duration Tracking

| Metric | Value |
|--------|-------|
| File size | _____ MB |
| Video duration | _____ seconds |
| Processing time | _____ seconds |
| Timeout occurred? | ⬜ Yes | ❌ No |

#### Expected Results
- **No timeout error:** Processing completes within 60 seconds OR shows appropriate progress
- **Continuous feedback:** Button remains disabled, status message persists
- **User informed:** Status message indicates long processing time
- **Completion:** Processing finishes successfully (may take 60+ seconds)
- **No premature errors:** No "timeout" or "network error" before completion

#### Actual Results

**File size:** _____ MB

**Video duration:** _____ seconds

**Processing time:** _____ seconds

**Timeout error occurred:** ⬜ Yes (at _____ seconds) | ❌ No

**Button remained disabled throughout:** ⬜ Yes | ❌ No

**Status message consistent:** ⬜ Yes | ❌ No

**Processing completed successfully:** ⬜ Yes | ❌ Failed

#### Test Evidence
________________________________________

#### Tester Notes
<!-- Did system handle long processing gracefully? -->
________________________________________

---

### Test Case: TC-VID-011 - Network Error During Detection

**Priority:** P1 (Important)
**Context Decision:** D-19
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify graceful handling when backend fails during video detection (simulated network failure).

#### Prerequisites
- Backend is running
- Video file is selected
- Console is open for monitoring

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select valid video file | File is selected |
| 2 | Click "检测视频" button | Detection starts |
| 3 | Wait 5 seconds | Processing begins |
| 4 | Stop backend (Ctrl+C) | Backend terminates |
| 5 | Wait 5-10 seconds | Network request fails |
| 6 | Observe error message | User-friendly error displays |
| 7 | Observe button state | Button re-enabled for retry |
| 8 | Start backend (`python src/run/app.py`) | Backend restarts |
| 9 | Click "检测视频" again | Detection succeeds |

#### Expected Results
- **Error message:** User-friendly (not technical stack trace)
  - Examples: "网络错误，请稍后重试" or "无法连接到后端"
  - NOT: "ERR_CONNECTION_RESET" or HTTP error codes
- **Error location:** Inline error in VideoPanel + status bar
- **Button state:** Re-enabled after error
- **No crash:** Application remains fully responsive
- **Recovery:** After backend restart, retry succeeds
- **Stale state:** No leftover loading state or error from failed attempt

#### Actual Results

**Error message displayed:** _________________________
**Is user-friendly:** ⬜ Yes | ❌ No (too technical)

**Error location:** ⬜ Inline | ⬜ Status bar | ⬜ Both

**Button re-enabled:** ⬜ Yes | ❌ No

**Recovery test - Restart backend and retry:** ⬜ Success | ❌ Failed

**Application responsive after error:** ⬜ Yes | ❌ No

#### Test Evidence
<!-- Screenshot of error state -->
________________________________________

#### Tester Notes
<!-- Is error recovery smooth or does user need to refresh page? -->
________________________________________

---

### Test Case: TC-VID-012 - Large File Upload

**Priority:** P2 (Nice-to-have)
**Context Decision:** D-21
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify system handles large video files (50MB+) without errors.

#### Prerequisites
- Backend and frontend are running
- Large video file available (50MB+)

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select large video file (50MB+) | File selected |
| 2 | Note file size | Record: _____ MB |
| 3 | Click "检测视频" button | Upload and processing starts |
| 4 | Start timer | Record upload time |
| 5 | Observe progress (if any) | System indicates processing |
| 6 | Wait for upload to complete | Upload finishes (may take time) |
| 7 | Wait for detection to complete | Detection finishes |
| 8 | Stop timers | Record total time |

#### Duration Tracking

| Metric | Value |
|--------|-------|
| File size | _____ MB |
| Upload time | _____ seconds |
| Processing time | _____ seconds |
| Total time | _____ seconds |
| Upload speed | _____ MB/s (calculated) |

#### Expected Results
- **Upload succeeds:** File uploads without errors
- **Processing completes:** Detection finishes successfully
- **No memory errors:** No "out of memory" or "file too large" errors
- **Reasonable time:** Upload and processing complete in acceptable time
- **User feedback:** System indicates processing during upload
- **Results display:** Annotated video displays at completion

#### Actual Results

**File size:** _____ MB

**Upload successful:** ⬜ Yes | ❌ No

**Processing completed:** ⬜ Yes | ❌ No

**Memory errors:** ⬜ None | ❌ Errors: _________________________

**Upload time:** _____ seconds

**Processing time:** _____ seconds

**Total time:** _____ seconds

**User feedback during upload:** ⬜ Yes | ❌ No (unresponsive)

#### Test Evidence
________________________________________

#### Tester Notes
<!-- Did system handle large file well? Any performance issues? -->
________________________________________

---

## Test Execution Summary

- **Total tests:** 12
- **Priority breakdown:**
  - P0 (Critical): 7 tests (TC-VID-001 through TC-VID-007)
  - P1 (Important): 4 tests (TC-VID-008 through TC-VID-011)
  - P2 (Nice-to-have): 1 test (TC-VID-012)
- **Estimated time:** 45-60 minutes (video processing is slow)

## Notes

- **Test videos:** Prepare videos of various lengths (10s, 20s, 30s, 60s+) for comprehensive testing
- **Patience required:** Video detection takes 10-30 seconds for 10-30 second videos
- **Network tab:** Use browser DevTools Network tab to verify single API requests
- **Timeout testing:** TC-VID-010 requires very long video (60+ seconds) - optional if unavailable
- **Large file testing:** TC-VID-012 is optional if 50MB+ file not available
- **Autoplay blocking:** Browsers may block video autoplay - this is expected behavior

## Next Steps

After completing this test suite:
1. Update status checkboxes for all executed tests
2. Record actual results for failed tests
3. Attach screenshots for failures as evidence
4. Proceed to next test suite: 04-edge-cases-tests.md

---

**Suite Status:** ⬜ Not Started | 🔄 In Progress | ✅ Complete | ❌ Failed

**Execution Date:** _________________________

**Tester:** _________________________

**Overall Result:** _____ / 12 tests passed (_____ %)
