# Image Detection Test Suite

**Purpose:** Validate image detection functionality including file selection, detection trigger, result display, person count, and error handling

**Requirements:** IMG-01, IMG-02, IMG-03, IMG-04, IMG-05

**Last Updated:** 2026-04-04

---

## Overview

This test suite validates all image detection functionality in the YOLO11 frontend application. Tests cover file selection, detection trigger, annotated result display, person count display, button state management, and error handling for missing files, invalid formats, and network failures.

**Test Execution Order:** Execute tests in order to build on state from previous tests.

**Estimated Time:** 20-30 minutes

---

## Quick Reference

| Test ID | Priority | Requirement | Context Decision | Status |
|---------|----------|-------------|------------------|--------|
| TC-IMG-001 | P0 | IMG-01 | D-08 | ⬜ Not Executed |
| TC-IMG-002 | P0 | IMG-02 | D-08, D-10 | ⬜ Not Executed |
| TC-IMG-003 | P0 | IMG-03 | — | ⬜ Not Executed |
| TC-IMG-004 | P0 | IMG-04 | D-10 | ⬜ Not Executed |
| TC-IMG-005 | P1 | IMG-02 | D-12 | ⬜ Not Executed |
| TC-IMG-006 | P0 | IMG-05 | D-11 | ⬜ Not Executed |
| TC-IMG-007 | P1 | IMG-05 | D-09 | ⬜ Not Executed |
| TC-IMG-008 | P1 | IMG-05 | D-09 | ⬜ Not Executed |
| TC-IMG-009 | P1 | IMG-05 | D-19 | ⬜ Not Executed |
| TC-IMG-010 | P2 | — | D-23, D-25 | ⬜ Not Executed |

**Priority Legend:**
- **P0 (Critical):** Must pass for system to be usable
- **P1 (Important):** Should pass for good user experience
- **P2 (Nice-to-have):** Optional enhancements

---

## Test Cases

---

### Test Case: TC-IMG-001 - Select Valid Image File

**Priority:** P0 (Critical)
**Requirement:** IMG-01
**Context Decision:** D-08
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that user can select JPG/PNG image files through the file input.

#### Prerequisites
- Backend and frontend are running
- Test image files are available (JPG, PNG formats)
- Image panel is visible

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to Image panel | Image panel is visible with file input |
| 2 | Click file input button | File selection dialog opens |
| 3 | Select a JPG file | Dialog closes, file is selected |
| 4 | Observe file input display | Selected filename is visible |
| 5 | Select a PNG file | File selection works for PNG format |
| 6 | Observe no errors | Console shows no file-related errors |

#### Test Data: Multiple Formats

| File Format | Test File | Expected Result | Actual Result | Status |
|-------------|-----------|----------------|---------------|--------|
| JPG | test-image-1-person.jpg | File selected successfully | ⬜ Pass | ❌ Fail | ⬜ Not Tested |
| PNG | test-image-3-people.png | File selected successfully | ⬜ Pass | ❌ Fail | ⬜ Not Tested |

#### Expected Results
- **File dialog opens:** Native browser file picker appears
- **File accepted:** JPG and PNG files are accepted
- **Filename visible:** Selected filename displays in file input area
- **No errors:** Console shows no file access errors
- **File accessible:** Selected file can be accessed for upload

#### Actual Results

**File dialog opens:** ⬜ Yes | ❌ No

**JPG file selection:** ⬜ Success | ❌ Failed

**PNG file selection:** ⬜ Success | ❌ Failed

**Filename visible:** ⬜ Yes | ❌ No

**Console errors:** ⬜ None | ❌ Errors: _________________________

#### Test Evidence
<!-- Screenshot of file selection -->
________________________________________

#### Tester Notes
<!-- Is file input user-friendly? -->
________________________________________

---

### Test Case: TC-IMG-002 - Trigger Image Detection - Success Case

**Priority:** P0 (Critical)
**Requirement:** IMG-02
**Context Decision:** D-08, D-10
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that image detection completes successfully and displays annotated results.

#### Prerequisites
- Backend and frontend are running
- Image file is selected (JPG/PNG with people)
- Image panel is visible

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select test image file | Filename displays in file input |
| 2 | Click "检测图像" (Detect Image) button | Button triggers detection |
| 3 | Observe button state immediately | Button text changes to "检测中..." |
| 4 | Observe button disabled state | Button is disabled (not clickable) |
| 5 | Observe status message | Status shows "正在进行图片检测..." |
| 6 | Wait for detection to complete (2-5 seconds) | Processing completes |
| 7 | Observe preview area | Annotated image displays |
| 8 | Observe person count | Count shows "检测到 N 人" |
| 9 | Observe button state after completion | Button returns to "检测图像" |
| 10 | Observe status message after completion | Status shows "图片检测完成。" |

#### Expected Results

**Immediate Feedback (0-1 second):**
- Button text: "检测中..." (Detecting...)
- Button state: Disabled (grayed out, not clickable)
- Status message: "正在进行图片检测..."

**During Processing:**
- Button remains disabled
- Loading indication persists
- No premature completion

**Upon Completion (2-5 seconds):**
- Annotated image displays in preview area
- Image shows bounding boxes around detected people
- Person count displays below image: "检测到 {N} 人"
- Button text returns to "检测图像"
- Button becomes enabled again
- Status message: "图片检测完成。" or backend-provided message
- Status state: OK (green)

#### Actual Results

**Button state changes:**
- Text changes to "检测中...": ⬜ Yes | ❌ No
- Button disabled: ⬜ Yes | ❌ No

**Processing time:** _____ seconds

**Annotated image displays:** ⬜ Yes | ❌ No

**Bounding boxes visible:** ⬜ Yes | ❌ No

**Person count displayed:** _________________________

**Status message on completion:** _________________________

**Button re-enabled after completion:** ⬜ Yes | ❌ No

#### Test Evidence
<!-- Screenshot of detection result -->
________________________________________

#### Tester Notes
<!-- How long did detection take? Was result accurate? -->
________________________________________

---

### Test Case: TC-IMG-003 - Display Annotated Image Result

**Priority:** P0 (Critical)
**Requirement:** IMG-03
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that detection result image displays correctly with bounding boxes and proper quality.

#### Prerequisites
- Image detection has completed successfully
- Annotated result image is available

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Examine result image in preview area | Image is visible and loads completely |
| 2 | Check for bounding boxes | Bounding boxes visible around detected people |
| 3 | Check bounding box colors | Boxes are clearly visible (typically red or green) |
| 4 | Check image quality | No pixelation, artifacts, or corruption |
| 5 | Check image orientation | Image is not rotated or distorted |
| 6 | Verify image is not cached | Image URL has timestamp parameter |
| 7 | Right-click image and inspect | Image src includes `?t={timestamp}` |

#### Expected Results

**Image Display:**
- Image loads completely without errors
- Resolution matches or exceeds original image
- No pixelation or compression artifacts
- Colors are accurate

**Bounding Boxes:**
- Clearly visible around each detected person
- Consistent color (typically red or green)
- Appropriate thickness (not too thin/thick)
- Boxes accurately outline people

**Cache Prevention:**
- Image URL includes timestamp parameter: `?t={timestamp}`
- Each detection generates new URL (prevents browser caching)

#### Quality Checklist

- [ ] Image loads completely
- [ ] Bounding boxes are visible
- [ ] Bounding boxes are clearly colored
- [ ] No image artifacts or pixelation
- [ ] Image orientation is correct
- [ ] Image URL has timestamp parameter

#### Actual Results

**Image loads completely:** ⬜ Yes | ❌ No

**Bounding boxes visible:** ⬜ Yes | ❌ No

**Bounding box color:** _________________________

**Image artifacts:** ⬜ None | ⬜ Minor | ⬜ Severe

**Image URL has timestamp:** ⬜ Yes | ❌ No
**URL:** _________________________

#### Test Evidence
<!-- Screenshot of annotated image -->
________________________________________

#### Tester Notes
<!-- Are bounding boxes accurate? Any detection errors? -->
________________________________________

---

### Test Case: TC-IMG-004 - Person Count Display

**Priority:** P0 (Critical)
**Requirement:** IMG-04
**Context Decision:** D-10
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that person count displays correctly after detection and matches actual number of people in image.

#### Prerequisites
- Image detection has completed successfully
- Result image is displayed

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Observe count text below result image | Text displays person count |
| 2 | Read count value | Format: "检测到 {N} 人" |
| 3 | Manually count people in image | Count detected people in annotated image |
| 4 | Compare displayed count with manual count | Values should match |
| 5 | Test with multiple images | Count updates correctly for each image |

#### Test Data: Known Person Counts

| Test Image | Expected Count | Displayed Count | Match? | Status |
|------------|----------------|-----------------|--------|--------|
| test-image-1-person.jpg | 1 | _____ | ⬜ Yes | ❌ No | ⬜ Not Tested |
| test-image-3-people.jpg | 3 | _____ | ⬜ Yes | ❌ No | ⬜ Not Tested |
| test-image-5-people.jpg | 5+ | _____ | ⬜ Yes | ❌ No | ⬜ Not Tested |

#### Expected Results
- **Format:** "检测到 {N} 人" (Detected N people)
- **Accuracy:** Count matches number of bounding boxes in image
- **Language:** Chinese text displays correctly
- **Visibility:** Count is clearly visible below result image
- **Updates:** Count updates when new image is detected

#### Actual Results

**Count format matches "检测到 {N} 人":** ⬜ Yes | ❌ No

**Displayed count:** _________________________

**Manual count from image:** _________________________

**Counts match:** ⬜ Yes | ❌ No (difference: _____)

**Chinese text displays correctly:** ⬜ Yes | ❌ No

**Count updates for new image:** ⬜ Yes | ❌ No

#### Test Evidence
<!-- Screenshots showing count for each test image -->
________________________________________

#### Tester Notes
<!-- Is detection accurate? Any false positives/negatives? -->
________________________________________

---

### Test Case: TC-IMG-005 - Button State During Processing

**Priority:** P1 (Important)
**Requirement:** IMG-02
**Context Decision:** D-12
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify that button behavior is correct during detection processing (disabled state, text change).

#### Prerequisites
- Backend and frontend are running
- Image file is selected

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Observe button before clicking | Text: "检测图像", enabled |
| 2 | Click "检测图像" button | Button state changes immediately |
| 3 | Observe button text (0-0.5 seconds) | Text: "检测中..." |
| 4 | Try to click button again | Button is disabled (no response) |
| 5 | Wait for processing to complete | Button remains disabled |
| 6 | Observe button after completion | Text: "检测图像", enabled |

#### Expected Results

**Initial State:**
- Button text: "检测图像" (Detect Image)
- Button state: Enabled (clickable)
- Button appearance: Normal (not grayed out)

**During Processing:**
- Button text: "检测中..." (Detecting...)
- Button state: Disabled (not clickable)
- Button appearance: Grayed out or visually disabled
- Cursor: Not-allowed pointer (optional)
- Multiple clicks: No effect, only one API call

**After Completion:**
- Button text: "检测图像" (Detect Image)
- Button state: Enabled (clickable)
- Button appearance: Normal

#### State Verification Checklist

- [ ] Initial text is "检测图像"
- [ ] Initial state is enabled
- [ ] Text changes to "检测中..." immediately (< 0.5s)
- [ ] Button is disabled during processing
- [ ] Button is visually grayed out
- [ ] Multiple clicks do not trigger multiple requests
- [ ] Text returns to "检测图像" after completion
- [ ] Button is enabled after completion

#### Actual Results

**Button state transitions:**
- Initial: ⬜ "检测图像" + enabled | ❌ Other: _________________________
- During: ⬜ "检测中..." + disabled | ❌ Other: _________________________
- Final: ⬜ "检测图像" + enabled | ❌ Other: _________________________

**Text change timing:** ⬜ Immediate (< 0.5s) | ❌ Delayed: _____ seconds

**Button visually disabled:** ⬜ Yes | ❌ No

**Multiple clicks prevented:** ⬜ Yes | ❌ No

#### Test Evidence
<!-- Screenshots of button states -->
________________________________________

#### Tester Notes
<!-- Is button state change obvious to user? -->
________________________________________

---

### Test Case: TC-IMG-006 - Empty File Selection Error

**Priority:** P0 (Critical)
**Requirement:** IMG-05
**Context Decision:** D-11
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify error handling when user clicks detection button without selecting a file.

#### Prerequisites
- Backend and frontend are running
- No file is selected in image panel

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Verify no file is selected | File input shows "No file chosen" or similar |
| 2 | Click "检测图像" button | Button attempts action |
| 3 | Observe error message | Error displays: "请先选择一张图片" |
| 4 | Observe status message | Status shows error |
| 5 | Observe button state | Button remains enabled for retry |
| 6 | Select valid image file | System allows file selection |
| 7 | Click "检测图像" again | Detection proceeds normally |

#### Expected Results
- **Error message:** "请先选择一张图片" (Please select an image first)
- **Error location:** Below button or in status bar
- **Error style:** Error state (red background or styling)
- **No API call:** Detection API is not called (frontend validation)
- **Button state:** Remains enabled (allows retry after file selection)
- **Recovery:** User can select file and retry detection
- **No crash:** Application remains responsive

#### Actual Results

**Error message displayed:** _________________________

**Matches expected "请先选择一张图片":** ⬜ Yes | ❌ No

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

### Test Case: TC-IMG-007 - Invalid File Format - Text File

**Priority:** P1 (Important)
**Requirement:** IMG-05
**Context Decision:** D-09
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify error handling when user selects a non-image file (.txt).

#### Prerequisites
- Backend and frontend are running
- Text file available for testing (test-file.txt)

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select .txt file via file input | File input shows selected file |
| 2 | Click "检测图像" button | System detects invalid format |
| 3 | Observe error message | Error about invalid format displays |
| 4 | Observe preview area | No result displays |
| 5 | Check browser Network tab | May show validation error or no request |
| 6 | Select valid image file | System recovers and allows retry |

#### Expected Results
- **Error message:** Clear message about invalid format
  - Examples: "不支持的文件格式" or "请选择图片文件 (JPG, PNG)"
  - NOT: Technical error or generic message
- **Error location:** Inline error below button or status bar
- **No processing:** Detection API is not called for invalid file
- **Recovery:** System allows selecting valid file and retrying
- **No crash:** Application remains responsive

#### Actual Results

**Error message displayed:** _________________________

**Is user-friendly:** ⬜ Yes | ❌ No (too technical)

**API call made:** ⬜ No (good) | ❌ Yes (unnecessary)

**Recovery test - Select valid image:** ⬜ Success | ❌ Failed

#### Test Evidence
________________________________________

#### Tester Notes
<!-- Is format validation on frontend or backend? -->
________________________________________

---

### Test Case: TC-IMG-008 - Invalid File Format - PDF

**Priority:** P1 (Important)
**Requirement:** IMG-05
**Context Decision:** D-09
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify error handling when user selects a PDF file.

#### Prerequisites
- Backend and frontend are running
- PDF file available for testing (test-file.pdf)

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select .pdf file via file input | File input shows selected file |
| 2 | Click "检测图像" button | System detects invalid format |
| 3 | Observe error message | Error about unsupported format displays |
| 4 | Select valid image file | System recovers |

#### Expected Results
- **Error message:** Clear message that PDF is not supported
- **No processing:** Detection API not called
- **Recovery:** System allows retry with valid image

#### Actual Results

**Error message displayed:** _________________________

**Recovery test:** ⬜ Success | ❌ Failed

#### Test Evidence
________________________________________

#### Tester Notes
<!-- Are error messages consistent across invalid formats? -->
________________________________________

---

### Test Case: TC-IMG-009 - Network Error During Detection

**Priority:** P1 (Important)
**Requirement:** IMG-05
**Context Decision:** D-19
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify graceful handling when backend fails during image detection (simulated network failure).

#### Prerequisites
- Backend is running
- Image file is selected
- Console is open for monitoring

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select valid image file | File is selected |
| 2 | Click "检测图像" button | Detection starts |
| 3 | Wait 1 second | Processing begins |
| 4 | Stop backend (Ctrl+C) | Backend terminates |
| 5 | Wait 2-3 seconds | Network request fails |
| 6 | Observe error message | User-friendly error displays |
| 7 | Observe button state | Button re-enabled for retry |
| 8 | Start backend (`python src/run/app.py`) | Backend restarts |
| 9 | Click "检测图像" again | Detection succeeds |

#### Expected Results
- **Error message:** User-friendly (not technical stack trace)
  - Examples: "网络错误，请稍后重试" or "无法连接到后端"
  - NOT: "ERR_CONNECTION_RESET" or HTTP error codes
- **Error location:** Inline error in ImagePanel + status bar
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

### Test Case: TC-IMG-010 - Multiple Detection Cycles

**Priority:** P2 (Nice-to-have)
**Context Decision:** D-23, D-25
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

#### Test Scenario
Verify system handles repeated detection operations without state pollution or errors.

#### Prerequisites
- Backend and frontend are running
- Multiple test images available (at least 3 different images)

#### Test Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select test-image-1-person.jpg | File selected |
| 2 | Click "检测图像" | Detection completes, result displays |
| 3 | Verify result is correct | Annotated image and count display |
| 4 | Select test-image-3-people.jpg | File selected |
| 5 | Click "检测图像" | Detection completes, result displays |
| 6 | Verify result is correct | New annotated image and count display |
| 7 | Select test-image-5-people.jpg | File selected |
| 8 | Click "检测图像" | Detection completes, result displays |
| 9 | Verify result is correct | New annotated image and count display |
| 10 | Check console for errors | No accumulated errors |

#### Expected Results
- **All detections successful:** Each detection completes without errors
- **Results update correctly:** New image replaces old image
- **Count updates:** Person count updates for each image
- **No state pollution:** Previous detection doesn't affect new detection
- **No memory leaks:** Browser memory stable (optional check)
- **Console clean:** No errors or warnings after multiple cycles
- **UI responsive:** Application remains responsive

#### Cycle Results Table

| Cycle | Test Image | Detection Result | Count Displayed | Status |
|-------|------------|------------------|-----------------|--------|
| 1 | test-image-1-person.jpg | ⬜ Success | ❌ Fail | _____ people | ⬜ Correct | ❌ Incorrect |
| 2 | test-image-3-people.jpg | ⬜ Success | ❌ Fail | _____ people | ⬜ Correct | ❌ Incorrect |
| 3 | test-image-5-people.jpg | ⬜ Success | ❌ Fail | _____ people | ⬜ Correct | ❌ Incorrect |

#### Actual Results

**Total cycles completed successfully:** _____ / 3

**Console errors after 3 cycles:** ⬜ None | ❌ Errors: _________________________

**Results update correctly:** ⬜ Yes | ❌ No (issue: _________________________)

**Memory stability (optional):** ⬜ Stable | ❌ Increasing

#### Test Evidence
<!-- Screenshots of each detection result -->
________________________________________

#### Tester Notes
<!-- Any degradation noticed after multiple detections? -->
________________________________________

---

## Test Execution Summary

- **Total tests:** 10
- **Priority breakdown:**
  - P0 (Critical): 5 tests (TC-IMG-001, TC-IMG-002, TC-IMG-003, TC-IMG-004, TC-IMG-006)
  - P1 (Important): 4 tests (TC-IMG-005, TC-IMG-007, TC-IMG-008, TC-IMG-009)
  - P2 (Nice-to-have): 1 test (TC-IMG-010)
- **Estimated time:** 20-30 minutes

## Notes

- **Test images:** Prepare images with known person counts for accurate validation
- **Detection accuracy:** This suite tests UI functionality, not detection algorithm accuracy
- **Processing time:** Image detection typically takes 2-5 seconds depending on image size
- **File validation:** Note whether validation happens on frontend or backend
- **Network simulation:** Stopping backend simulates network failure for error handling tests

## Next Steps

After completing this test suite:
1. Update status checkboxes for all executed tests
2. Record actual results for failed tests
3. Attach screenshots for failures as evidence
4. Proceed to next test suite: 03-video-detection-tests.md

---

**Suite Status:** ⬜ Not Started | 🔄 In Progress | ✅ Complete | ❌ Failed

**Execution Date:** _________________________

**Tester:** _________________________

**Overall Result:** _____ / 10 tests passed (_____ %)
