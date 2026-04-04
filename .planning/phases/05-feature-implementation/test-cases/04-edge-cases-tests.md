# Edge Cases Test Suite

**Purpose:** Validate error handling for boundary conditions, network failures, timeout scenarios, and invalid inputs

**Context Decisions:** D-19, D-20, D-21, D-22

**Last Updated:** 2026-04-04

---

## Overview

This test suite validates edge case scenarios and error handling across camera streaming, image detection, and video detection features. Tests focus on failure modes such as backend unavailability, network timeouts, large file uploads, and unsupported file formats.

**Important:** These tests simulate failure scenarios. Some tests require manual intervention (stopping/starting backend).

**Test Execution Order:** Tests can be executed independently, but grouping by feature is recommended.

**Estimated Time:** 30-45 minutes

---

## Quick Reference

| Test ID | Priority | Context Decision | Category | Feature | Status |
|---------|----------|------------------|----------|---------|--------|
| TC-EDGE-001 | P0 | D-19 | Network Failure | Camera | ⬜ Not Executed |
| TC-EDGE-002 | P0 | D-19 | Network Failure | Image | ⬜ Not Executed |
| TC-EDGE-003 | P0 | D-19 | Network Failure | Video | ⬜ Not Executed |
| TC-EDGE-004 | P1 | D-20 | Timeout | Video | ⬜ Not Executed |
| TC-EDGE-005 | P2 | D-21 | Large File | Image | ⬜ Not Executed |
| TC-EDGE-006 | P2 | D-21 | Large File | Video | ⬜ Not Executed |
| TC-EDGE-007 | P1 | D-22 | Unsupported Format | Image | ⬜ Not Executed |
| TC-EDGE-008 | P1 | D-22 | Unsupported Format | Video | ⬜ Not Executed |

**Priority Legend:**
- **P0 (Critical):** Must pass for system to be usable
- **P1 (Important):** Should pass for good user experience
- **P2 (Nice-to-have):** Optional enhancements

**Category Legend:**
- **Network Failure:** Backend unavailable or connection lost
- **Timeout:** Long-running operations exceeding expected duration
- **Large File:** File size exceeding typical usage
- **Unsupported Format:** File format not supported by system

---

## Edge Case Testing Matrix

| Feature | Network Failure | Timeout | Large File | Unsupported Format |
|---------|----------------|---------|------------|-------------------|
| Camera | TC-EDGE-001 | N/A | N/A | N/A |
| Image | TC-EDGE-002 | N/A | TC-EDGE-005 | TC-EDGE-007 |
| Video | TC-EDGE-003 | TC-EDGE-004 | TC-EDGE-006 | TC-EDGE-008 |

---

## Test Cases

---

### Test Case: TC-EDGE-001 - Backend Offline - Camera Start

**Priority:** P0 (Critical)
**Context Decision:** D-19
**Category:** Network Failure
**Feature:** Camera
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify camera start fails gracefully when backend is offline.

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
| 5 | Observe error message | User-friendly error displays |
| 6 | Observe button state | Button remains enabled for retry |
| 7 | Start backend (`python src/run/app.py`) | Backend starts |
| 8 | Click "启动摄像头" again | Camera starts successfully |

#### Expected Results
- **Error message:** User-friendly (not technical stack trace)
  - Examples: "无法连接到后端服务" or "网络错误，请稍后重试"
  - NOT: "ERR_CONNECTION_REFUSED" or technical HTTP errors
- **Status state:** Error state (red background)
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

**Application responsive:** ⬜ Yes | ❌ No (froze/crashed)

#### Test Evidence
<!-- Screenshot of error message -->
________________________________________

#### Tester Notes
<!-- Is error message suitable for non-technical users? -->
________________________________________

---

### Test Case: TC-EDGE-002 - Backend Offline - Image Detection

**Priority:** P0 (Critical)
**Context Decision:** D-19
**Category:** Network Failure
**Feature:** Image
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify image detection fails gracefully when backend is offline.

#### Prerequisites
- Frontend is running
- Backend is STOPPED
- Image file is selected

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Verify backend is stopped | Terminal shows no Flask process |
| 2 | Select valid image file | File is selected |
| 3 | Click "检测图像" button | Detection attempts API call |
| 4 | Wait 2-3 seconds | Network request fails |
| 5 | Observe error message | User-friendly error displays |
| 6 | Observe inline error (in ImagePanel) | Error displays below button |
| 7 | Observe button state | Button re-enabled for retry |
| 8 | Start backend | Backend restarts |
| 9 | Click "检测图像" again | Detection succeeds |

#### Expected Results
- **Error message:** User-friendly
- **Inline error:** Displays in ImagePanel below button
- **Status bar:** Shows error state
- **Button state:** Re-enabled after error
- **No crash:** Application remains responsive
- **Recovery:** After backend restart, retry succeeds

#### Actual Results

**Error message (status bar):** _________________________
**Inline error (ImagePanel):** _________________________

**Both errors displayed:** ⬜ Yes | ❌ No

**Button re-enabled:** ⬜ Yes | ❌ No

**Recovery test:** ⬜ Success | ❌ Failed

#### Test Evidence
<!-- Screenshot showing both error locations -->
________________________________________

#### Tester Notes
<!-- Are errors displayed in multiple locations for visibility? -->
________________________________________

---

### Test Case: TC-EDGE-003 - Backend Offline - Video Detection

**Priority:** P0 (Critical)
**Context Decision:** D-19
**Category:** Network Failure
**Feature:** Video
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify video detection fails gracefully when backend is offline.

#### Prerequisites
- Frontend is running
- Backend is STOPPED
- Video file is selected

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Verify backend is stopped | Terminal shows no Flask process |
| 2 | Select valid video file | File is selected |
| 3 | Click "检测视频" button | Detection attempts API call |
| 4 | Wait 5 seconds (video takes longer) | Network request fails |
| 5 | Observe error message | User-friendly error displays |
| 6 | Observe inline error (in VideoPanel) | Error displays below button |
| 7 | Observe button state | Button re-enabled for retry |
| 8 | Start backend | Backend restarts |
| 9 | Click "检测视频" again | Detection succeeds |

#### Expected Results
- **Error message:** User-friendly
- **Inline error:** Displays in VideoPanel below button
- **Status bar:** Shows error state
- **Button state:** Re-enabled after error
- **No crash:** Application remains responsive
- **Recovery:** After backend restart, retry succeeds

#### Actual Results

**Error message (status bar):** _________________________
**Inline error (VideoPanel):** _________________________

**Both errors displayed:** ⬜ Yes | ❌ No

**Button re-enabled:** ⬜ Yes | ❌ No

**Recovery test:** ⬜ Success | ❌ Failed

#### Test Evidence
<!-- Screenshot showing both error locations -->
________________________________________

#### Tester Notes
<!-- Does video detection handle network failure gracefully? -->
________________________________________

---

### Test Case: TC-EDGE-004 - Network Timeout - Video Detection

**Priority:** P1 (Important)
**Context Decision:** D-20
**Category:** Timeout
**Feature:** Video
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify system handles long processing without premature timeout (tests 60-second timeout configuration).

#### Prerequisites
- Backend and frontend are running
- Large or long video file available (60+ seconds or high resolution)

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select large/long video file | File selected |
| 2 | Note file size and duration | Record: _____ MB, _____ seconds |
| 3 | Click "检测视频" button | Detection starts |
| 4 | Start timer | Record processing time |
| 5 | Wait for processing (may take 60+ seconds) | Processing continues |
| 6 | Observe for timeout errors | Monitor for premature timeout |
| 7 | Wait for completion | Processing finishes |
| 8 | Stop timer | Record actual duration |

#### Duration Tracking

| Metric | Value |
|--------|-------|
| File size | _____ MB |
| Video duration | _____ seconds |
| Processing time | _____ seconds |
| Timeout error? | ⬜ Yes (at _____ s) | ❌ No |

#### Expected Results
- **No premature timeout:** No timeout error before 60 seconds
- **Continuous feedback:** Button disabled, status message persists
- **Completion:** Processing finishes successfully (may take 60+ seconds)
- **User informed:** Status message indicates long processing

#### Actual Results

**Processing time:** _____ seconds

**Timeout error occurred:** ⬜ Yes | ❌ No

**Timeout error time:** _____ seconds

**Button remained disabled:** ⬜ Yes | ❌ No

**Processing completed:** ⬜ Yes | ❌ Failed

#### Test Evidence
________________________________________

#### Tester Notes
<!-- Did system respect 60-second timeout? -->
________________________________________

---

### Test Case: TC-EDGE-005 - Large Image File Upload

**Priority:** P2 (Nice-to-have)
**Context Decision:** D-21
**Category:** Large File
**Feature:** Image
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify system handles large image files (10MB+) without errors.

#### Prerequisites
- Backend and frontend are running
- Large image file available (10MB+)

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select large image file (10MB+) | File selected |
| 2 | Note file size | Record: _____ MB |
| 3 | Click "检测图像" button | Upload and detection starts |
| 4 | Start timer | Record upload time |
| 5 | Wait for completion | Detection finishes |
| 6 | Stop timer | Record total time |

#### Duration Tracking

| Metric | Value |
|--------|-------|
| File size | _____ MB |
| Upload time | _____ seconds |
| Processing time | _____ seconds |
| Total time | _____ seconds |

#### Expected Results
- **Upload succeeds:** File uploads without errors
- **Processing completes:** Detection finishes successfully
- **No memory errors:** No "out of memory" errors
- **Reasonable time:** Completes in acceptable time

#### Actual Results

**File size:** _____ MB

**Upload successful:** ⬜ Yes | ❌ No

**Processing completed:** ⬜ Yes | ❌ No

**Memory errors:** ⬜ None | ❌ Errors: _________________________

**Total time:** _____ seconds

#### Test Evidence
________________________________________

#### Tester Notes
<!-- Did large image cause any issues? -->
________________________________________

---

### Test Case: TC-EDGE-006 - Large Video File Upload

**Priority:** P2 (Nice-to-have)
**Context Decision:** D-21
**Category:** Large File
**Feature:** Video
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify system handles very large video files (100MB+) without errors.

#### Prerequisites
- Backend and frontend are running
- Very large video file available (100MB+)

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select very large video file (100MB+) | File selected |
| 2 | Note file size | Record: _____ MB |
| 3 | Click "检测视频" button | Upload and detection starts |
| 4 | Start timer | Record upload time |
| 5 | Wait for upload to complete | Upload finishes (may take time) |
| 6 | Wait for detection to complete | Detection finishes |
| 7 | Stop timer | Record total time |

#### Duration Tracking

| Metric | Value |
|--------|-------|
| File size | _____ MB |
| Upload time | _____ seconds |
| Processing time | _____ seconds |
| Total time | _____ seconds |
| Upload speed | _____ MB/s |

#### Expected Results
- **Upload succeeds:** File uploads without errors (may take time)
- **Processing completes:** Detection finishes or returns appropriate error if too large
- **No crash:** No application crash or browser freeze
- **User feedback:** System indicates processing during upload

#### Actual Results

**File size:** _____ MB

**Upload successful:** ⬜ Yes | ❌ No

**Processing completed:** ⬜ Yes | ❌ No

**Crash or freeze:** ⬜ None | ❌ Issue: _________________________

**Upload time:** _____ seconds

**Processing time:** _____ seconds

#### Test Evidence
________________________________________

#### Tester Notes
<!-- Did very large video cause any issues? -->
________________________________________

---

### Test Case: TC-EDGE-007 - Unsupported Image Format - TIFF

**Priority:** P1 (Important)
**Context Decision:** D-22
**Category:** Unsupported Format
**Feature:** Image
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify error handling for unsupported image format (TIFF).

#### Prerequisites
- Backend and frontend are running
- TIFF file available for testing (test-image.tif or .tiff)

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select .tif or .tiff file | File input shows selected file |
| 2 | Click "检测图像" button | System detects unsupported format |
| 3 | Observe error message | Clear error about unsupported format |
| 4 | Observe preview area | No result displays |
| 5 | Select valid JPG/PNG image | System recovers |
| 6 | Click "检测图像" again | Detection succeeds |

#### Expected Results
- **Error message:** Clear message about unsupported format
  - Examples: "不支持的图片格式。请使用 JPG 或 PNG。"
  - NOT: Technical error or cryptic message
- **No crash:** No application crash
- **Recovery:** System allows retry with valid format

#### Actual Results

**Error message displayed:** _________________________

**Is clear and actionable:** ⬜ Yes | ❌ No

**Application crashed:** ⬜ No | ❌ Yes

**Recovery test:** ⬜ Success | ❌ Failed

#### Test Evidence
________________________________________

#### Tester Notes
<!-- Is error message helpful for users? -->
________________________________________

---

### Test Case: TC-EDGE-008 - Unsupported Video Format - WMV

**Priority:** P1 (Important)
**Context Decision:** D-22
**Category:** Unsupported Format
**Feature:** Video
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify error handling for unsupported video format (WMV).

#### Prerequisites
- Backend and frontend are running
- WMV file available for testing (test-video.wmv)

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select .wmv file | File input shows selected file |
| 2 | Click "检测视频" button | System detects unsupported format |
| 3 | Observe error message | Clear error about supported formats |
| 4 | Observe preview area | No result displays |
| 5 | Select valid MP4/AVI video | System recovers |
| 6 | Click "检测视频" again | Detection succeeds |

#### Expected Results
- **Error message:** Clear message listing supported formats
  - Examples: "不支持的视频格式。请使用 MP4 或 AVI。"
  - NOT: Technical error or cryptic message
- **No crash:** No application crash
- **Recovery:** System allows retry with valid format

#### Actual Results

**Error message displayed:** _________________________

**Lists supported formats:** ⬜ Yes | ❌ No

**Application crashed:** ⬜ No | ❌ Yes

**Recovery test:** ⬜ Success | ❌ Failed

#### Test Evidence
________________________________________

#### Tester Notes
<!-- Does error message guide users to correct formats? -->
________________________________________

---

## Test Execution Summary

- **Total tests:** 8
- **Priority breakdown:**
  - P0 (Critical): 3 tests (TC-EDGE-001, TC-EDGE-002, TC-EDGE-003)
  - P1 (Important): 3 tests (TC-EDGE-004, TC-EDGE-007, TC-EDGE-008)
  - P2 (Nice-to-have): 2 tests (TC-EDGE-005, TC-EDGE-006)
- **Estimated time:** 30-45 minutes
- **Special notes:** Some tests require stopping/starting backend manually

## Notes

- **Backend manipulation:** Tests TC-EDGE-001 through TC-EDGE-003 require stopping backend
- **Recovery testing:** All error scenarios include recovery steps to verify system can recover
- **Large files:** Tests TC-EDGE-005 and TC-EDGE-006 are optional if large files unavailable
- **Timeout test:** TC-EDGE-004 requires 60+ second video - optional if unavailable
- **Unsupported formats:** Tests TC-EDGE-007 and TC-EDGE-008 require TIFF and WMV files - optional if unavailable

## Test Execution Tips

1. **Group by feature:** Execute all camera tests, then image tests, then video tests
2. **Backend management:** Keep backend terminal accessible for quick start/stop
3. **File preparation:** Gather all test files before starting (TIFF, WMV, large files)
4. **Patience:** Large file tests may take significant time
5. **Recovery focus:** Verify system can recover from all error scenarios

## Next Steps

After completing this test suite:
1. Update status checkboxes for all executed tests
2. Record actual results for failed tests
3. Attach screenshots for failures as evidence
4. Proceed to next test suite: 05-interaction-state-tests.md

---

**Suite Status:** ⬜ Not Started | 🔄 In Progress | ✅ Complete | ❌ Failed

**Execution Date:** _________________________

**Tester:** _________________________

**Overall Result:** _____ / 8 tests passed (_____ %)
