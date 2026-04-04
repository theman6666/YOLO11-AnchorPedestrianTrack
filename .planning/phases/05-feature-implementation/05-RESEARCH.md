# Phase 5: Feature Implementation - Research

**Researched:** 2026-04-04
**Domain:** Manual E2E Test Case Documentation for Vue 3 + Flask Web Applications
**Confidence:** HIGH

## Summary

Phase 5 focuses on creating comprehensive manual test case documentation to validate the Phase 4 implementation. Research reveals that effective manual E2E test documentation follows a structured template approach with clear test scenarios, prerequisites, step-by-step instructions, and expected results. For video/streaming applications, special attention must be paid to visual validation, timing-based assertions, and state management verification.

**Primary recommendation:** Use a Markdown-based test case template organized by feature workflow (Camera, Image, Video) with priority levels (P0-critical, P1-important, P2-nice-to-have), including dedicated sections for edge cases and error handling validation.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**验证方法**

- **D-01:** Phase 5 使用测试用例文档形式，每个测试用例包含：测试场景、前置条件、操作步骤、预期结果、实际结果记录区域
- **D-02:** 验证需要实际的 Flask 后端运行，不是单元测试或集成测试，而是手动端到端验证
- **D-03:** 测试结果记录在文档中，用于验证 Phase 4 实现是否满足所有需求

**摄像头功能验证**

- **D-04:** 多 ID 测试 — 验证摄像头 ID 0、1、2 等不同值都能正常启动流
- **D-05:** 启停控制 — 验证"启动"按钮开始流，"停止"按钮真正停止流（不是隐藏）
- **D-06:** 状态消息 — 验证顶部状态栏显示正确的中文消息："摄像头 N 已启动。" / "摄像头已停止。"
- **D-07:** 错误处理 — 验证当摄像头 ID 不存在或后端不可用时显示适当的错误信息

**图片检测验证**

- **D-08:** 有效文件测试 — 上传 JPG/PNG 图片，验证检测完成，显示标注结果图
- **D-09:** 无效格式测试 — 尝试上传非图片文件，验证显示错误提示
- **D-10:** 人数统计 — 验证检测结果下方正确显示检测到的人数（例如："检测到 5 人"）
- **D-11:** 空文件测试 — 未选择文件时点击检测，验证显示"请先选择一张图片"错误
- **D-12:** 按钮状态 — 验证检测期间按钮显示"检测中..."并被禁用，完成后恢复

**视频检测验证**

- **D-13:** 有效文件测试 — 上传 MP4/AVI 视频，验证处理完成后显示标注视频
- **D-14:** 无效格式测试 — 尝试上传非视频文件，验证显示错误提示
- **D-15:** 处理状态 — 验证长时间处理时按钮显示"检测中..."并被禁用
- **D-16:** 统计信息 — 验证结果下方显示视频统计（例如："总帧数：300，平均 FPS：24.5"）
- **D-17:** 视频播放 — 验证结果视频可以正常播放，有播放控制
- **D-18:** 错误处理 — 验证网络错误或超时时显示适当的错误信息

**边界情况**

- **D-19:** 网络失败 — 模拟后端关闭，验证显示友好的错误消息而非技术错误
- **D-20:** 超时处理 — 验证长时间处理时（60秒超时）有适当的反馈
- **D-21:** 大文件 — 验证大图片/视频文件能正确上传和处理
- **D-22:** 格式不支持 — 尝试上传不支持的文件格式，验证显示清晰的错误消息

**交互状态**

- **D-23:** 按钮禁用 — 验证处理期间按钮正确禁用，防止重复提交
- **D-24:** 加载反馈 — 验证用户能清楚知道系统正在处理（按钮文字变化、禁用状态）
- **D-25:** 错误恢复 — 验证错误发生后用户可以重试操作，无需刷新页面

### Claude's Discretion

- 测试用例文档的具体格式和布局
- 测试环境的配置说明（Flask 后端启动方式）
- 实际结果记录区域的格式（文本输入、勾选框等）
- 测试通过/失败的判定标准

### Deferred Ideas (OUT OF SCOPE)

- **自动化测试** — 不创建单元测试或 E2E 测试，仅手动验证（Phase 5 范围）
- **CI/CD 集成** — 测试验证是手动的，不集成到自动化流程
- **性能测试** — 不测试大文件上传的性能或流媒体延迟
- **跨浏览器测试** — 仅验证现代浏览器（Chrome/Edge），不考虑旧版浏览器兼容性
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CAM-01 | User can input camera ID number | Test case template for input validation |
| CAM-02 | User can start camera stream | Video streaming UI testing patterns |
| CAM-03 | User can stop camera stream | State transition testing practices |
| CAM-04 | Camera feed displays in preview area | Visual validation for video streams |
| CAM-05 | Status message updates on camera start/stop | Message assertion testing |
| IMG-01 | User can select image file for upload | File input testing patterns |
| IMG-02 | User can trigger image detection | Async operation testing |
| IMG-03 | Detection results display annotated image | Visual validation for images |
| IMG-04 | Person count displays after detection | Data assertion testing |
| IMG-05 | Error handling for failed detection or invalid files | Edge case testing patterns |
| VID-01 | User can select video file for upload | File input testing patterns |
| VID-02 | User can trigger video detection | Long-running operation testing |
| VID-03 | Processing state indication during detection | Loading state testing |
| VID-04 | Video results display annotated video with playback | Video player validation |
| VID-05 | Video statistics display (frames, average FPS) | Data assertion testing |
| VID-06 | Button disabled state during processing | UI state testing |
</phase_requirements>

## Standard Stack

### Core
| Tool/Library | Version | Purpose | Why Standard |
|--------------|---------|---------|--------------|
| Markdown | — | Test case documentation format | Universal format, version control friendly, readable in plain text |
| Manual testing | — | Test execution method | Best for visual validation, streaming UI, exploratory testing |
| Browser DevTools | Chrome/Edge | Debugging and verification | Standard for web UI testing |

### Supporting
| Tool/Library | Version | Purpose | When to Use |
|--------------|---------|---------|-------------|
| Flask Backend | Existing | Running server for testing | Required for all E2E tests |
| Vite Dev Server | Existing | Frontend development server | Required for serving Vue app |
| Screen recording | — | Documenting test execution | Optional: for bug reporting |

### Test Organization Tools
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual Markdown files | TestRail/Jira | Better for team collaboration, but adds dependency |
| Plain text | Excel/Google Sheets | Better for non-technical stakeholders, harder to version control |
| Local files | Confluence/Notion | Better sharing, harder to track in git |

**Installation:**
No installation required — using existing development environment.

**Version verification:** N/A (documentation-only phase)

## Architecture Patterns

### Recommended Test Document Structure

```
.planning/phases/05-feature-implementation/
├── 05-RESEARCH.md                          # This file
├── 05-CONTEXT.md                           # Context from planning
├── test-cases/                             # Test case documents
│   ├── 00-test-environment-setup.md        # Environment setup instructions
│   ├── 01-camera-streaming-tests.md        # CAM-01 to CAM-05
│   ├── 02-image-detection-tests.md         # IMG-01 to IMG-05
│   ├── 03-video-detection-tests.md         # VID-01 to VID-06
│   ├── 04-edge-cases-tests.md              # D-19 to D-22
│   └── 05-interaction-state-tests.md       # D-23 to D-25
└── 05-VALIDATION.md                        # Overall validation summary
```

### Pattern 1: Test Case Template Structure
**What:** Standardized format for individual test cases ensuring consistency and completeness
**When to use:** Every test case should follow this structure
**Example:**
```markdown
## Test Case: TC-CAM-001 - Start Camera Stream with Valid ID

**Priority:** P0 (Critical)
**Requirement:** CAM-02
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail

### Test Scenario
Verify that user can successfully start camera stream when entering a valid camera ID.

### Prerequisites
- Flask backend is running (`python src/run/app.py`)
- Vite dev server is running (`cd frontend-vue && npm run dev`)
- Browser is opened to `http://localhost:5173`
- Camera device is available on system

### Test Steps
1. Navigate to Camera panel
2. Enter camera ID "0" in the input field
3. Click "启动摄像头" (Start Camera) button
4. Observe status message and preview area

### Expected Results
- Status message displays: "摄像头 0 已启动。"
- Video stream appears in preview area
- Button text changes to "停止摄像头" (Stop Camera)
- No error messages are displayed

### Actual Results
<!-- Fill in during testing -->
Status message: _________________________
Video stream: ⬜ Yes | ⬜ No
Button state: _________________________
Errors: _________________________

### Test Evidence
<!-- Screenshots or notes -->
________________________________________

### Tester Notes
<!-- Any observations or issues -->
________________________________________
```

### Pattern 2: Test Suite Organization
**What:** Grouping test cases by feature workflow with priority levels
**When to use:** Organizing comprehensive test coverage
**Example:**
```markdown
# Camera Streaming Test Suite

## Overview
This suite validates all camera streaming functionality including start/stop controls, status messages, and error handling.

## Test Execution Order
Execute tests in order to build on state:
1. Environment setup (TC-CAM-000)
2. Basic functionality (TC-CAM-001 to TC-CAM-004)
3. Error handling (TC-CAM-005 to TC-CAM-007)
4. Edge cases (TC-CAM-008 to TC-CAM-010)

## Quick Reference
| Test ID | Priority | Requirement | Status |
|---------|----------|-------------|--------|
| TC-CAM-001 | P0 | CAM-02 | ⬜ |
| TC-CAM-002 | P0 | CAM-03 | ⬜ |
| TC-CAM-003 | P0 | CAM-04 | ⬜ |
| TC-CAM-004 | P1 | CAM-05 | ⬜ |
| TC-CAM-005 | P0 | CAM-01 | ⬜ |
| TC-CAM-006 | P1 | CAM-01 | ⬜ |
| TC-CAM-007 | P1 | D-07 | ⬜ |
```

### Pattern 3: Edge Case Documentation
**What:** Structured approach to documenting boundary conditions and error scenarios
**When to use:** Testing unusual inputs, extreme conditions, and failure modes
**Example:**
```markdown
## Edge Case: Invalid Camera ID

**Category:** Input Validation
**Priority:** P1 (Important)
**Context Decision:** D-07

### Test Data
| Input Type | Value | Expected Behavior |
|------------|-------|-------------------|
| Negative number | -1 | Error: "摄像头ID无效" |
| Non-numeric | "abc" | Error: "请输入有效的摄像头ID" |
| Very large number | 9999 | Error: "摄像头不存在" |
| Empty | "" | Error: "请输入摄像头ID" |
| Zero | "0" | Success (if camera 0 exists) |

### Test Steps
For each input value in Test Data table:
1. Enter value in camera ID field
2. Click "启动摄像头" button
3. Verify error message displays
4. Verify camera does NOT start
5. Verify button remains clickable for retry

### Recovery Test
After each error case:
1. Clear input field
2. Enter valid camera ID
3. Click "启动摄像头"
4. Verify system recovers and camera starts successfully
```

### Anti-Patterns to Avoid

- **Vague test steps:** "Click button" → Better: "Click the '启动摄像头' button located in the Camera panel"
- **Missing prerequisites:** Not documenting that backend must be running leads to false failures
- **No cleanup instructions:** Tests leaving cameras running interfere with subsequent tests
- **Hardcoded values only:** Not testing multiple inputs (e.g., only camera ID 0, not 1, 2, etc.)
- **Missing visual validation:** Not describing what "video displays" actually looks like
- **No recovery testing:** Not verifying system can recover from errors
- **Ignoring timing:** Not accounting for async operations (video loading, detection processing)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Test case management | Custom spreadsheet or database | Markdown files in git | Version control, simplicity, no dependencies |
| Test execution tracking | Custom status tracking tool | Manual checkboxes in Markdown | Transparent, audit trail, no tool overhead |
| Screenshot capture | Custom screenshot script | Browser DevTools screenshots or OS tools | Standard, reliable, no code to maintain |
| Test reporting | Custom report generator | Summary document with pass/fail counts | Simple, sufficient for manual testing |

**Key insight:** For manual E2E testing in a small team, simplicity beats sophistication. Use tools that are already available (git, markdown, browsers) rather than building custom test infrastructure.

## Common Pitfalls

### Pitfall 1: Insufficient Prerequisite Documentation
**What goes wrong:** Test fails because tester didn't know Flask backend needed to be running
**Why it happens:** Assumption that "everyone knows" the setup steps
**How to avoid:** Create a dedicated test environment setup document with verification steps
**Warning signs:** Testers reporting environment issues before testing functionality

### Pitfall 2: Ambiguous Expected Results
**What goes wrong:** "Video displays" is subjective — tester doesn't know what qualifies as pass/fail
**Why it happens:** Focusing on "what" without defining "how to verify"
**How to avoid:** Include specific verification criteria:
  - "Video stream displays at minimum 15 FPS"
  - "Video resolution matches camera resolution"
  - "No visual artifacts or stuttering"
**Warning signs:** Testers asking "does this count as passing?"

### Pitfall 3: Missing Timing Considerations
**What goes wrong:** Test fails because tester didn't wait for async operation (video detection takes 10+ seconds)
**Why it happens:** Treating all operations as synchronous
**How to avoid:** Document expected duration and waiting strategies:
  - "Expected duration: 10-30 seconds depending on video length"
  - "Wait for button state to change from '检测中...' to '检测图像'"
  - "Verify loading spinner displays during processing"
**Warning signs:** Intermittent test failures, inconsistent results

### Pitfall 4: Inadequate Error Case Coverage
**What goes wrong:** Only testing happy path, missing validation of error handling
**Why it happens:** Focus on "what works" rather than "what breaks"
**How to avoid:** Systematic edge case identification:
  - Input boundary values (empty, negative, extreme values)
  - Network failure scenarios (backend offline, timeout)
  - Invalid file types and formats
  - Concurrent operations (clicking start multiple times)
**Warning signs:** Production errors from scenarios never tested

### Pitfall 5: No Recovery Testing
**What goes wrong:** System handles error but can't recover — requires page refresh
**Why it happens:** Testing error state in isolation, not recovery flow
**How to avoid:** Include "recovery step" after each error case:
  - "After error, enter valid input and verify system works"
  - "After error, verify button is re-enabled for retry"
  - "After error, verify no stale state affects subsequent operations"
**Warning signs:** Testers needing to refresh page between tests

## Code Examples

Verified patterns from official sources:

### Test Case Template (Markdown)
```markdown
# Test Case Template

## Test Case: TC-{CATEGORY}-{NUMBER} - {Title}

**Priority:** P0 | P1 | P2
**Requirement:** REQ-ID
**Context Decision:** D-ID (if applicable)
**Status:** ⬜ Not Executed | ✅ Pass | ❌ Fail | ⚠️ Blocked

### Test Scenario
{Clear description of what is being tested}

### Prerequisites
- {Environment condition 1}
- {Environment condition 2}
- {Specific data or setup required}

### Test Steps
| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | {Specific action} | {Immediate verification} |
| 2 | {Specific action} | {Immediate verification} |
| 3 | {Specific action} | {Immediate verification} |

### Expected Results
- {Specific observable outcome 1}
- {Specific observable outcome 2}
- {No unexpected behavior}

### Actual Results
**Status:** ⬜ Pass | ❌ Fail

**Observations:**
{Fill in during test execution}

**Deviations from expected:**
{Document any differences}

**Evidence:**
{Screenshots, console logs, or notes}

### Test Evidence
<!-- Attach screenshots or describe evidence -->

### Tester Notes
<!-- Any observations, issues, or suggestions -->
```

### Environment Setup Document
```markdown
# Test Environment Setup

## System Requirements
- Python 3.8+ installed
- Node.js 16+ installed
- At least one camera device (for camera tests)
- Test media files (images/videos) available

## Backend Setup

### Start Flask Backend
```bash
cd /path/to/YOLO11-AnchorPedestrianTrack
python src/run/app.py
```

**Verification:**
- Browser to `http://localhost:5000`
- Should see Flask response or API documentation
- Terminal shows "Running on http://127.0.0.1:5000"

### Stop Flask Backend
- Press `Ctrl+C` in terminal
- Verify terminal shows "Shutting down..."

## Frontend Setup

### Start Vite Dev Server
```bash
cd frontend-vue
npm run dev
```

**Verification:**
- Browser to `http://localhost:5173`
- Should see YOLO11 application
- Terminal shows "Local: http://localhost:5173/"

### Stop Vite Dev Server
- Press `Ctrl+C` in terminal
- Verify terminal shows "closing server..."

## Pre-Test Checklist
- [ ] Flask backend is running
- [ ] Vite dev server is running
- [ ] Browser DevTools are open (F12)
- [ ] Console tab shows no errors
- [ ] Network tab is enabled for API monitoring
- [ ] Test media files are accessible

## Post-Test Cleanup
- [ ] Stop any active camera streams
- [ ] Clear browser cache if needed
- [ ] Close browser tabs
- [ ] Note any environment issues for next test session
```

### Video Streaming Test Pattern
```markdown
## Test Case: TC-CAM-003 - Video Stream Quality Validation

**Priority:** P1 (Important)
**Aspect:** Visual validation for streaming UI

### Test Scenario
Verify that camera video stream displays with acceptable quality and performance.

### Prerequisites
- Camera ID 0 is available
- Backend and frontend are running
- Network connection is stable

### Test Steps
1. Start camera stream with ID "0"
2. Wait 5 seconds for stream to stabilize
3. Observe video quality in preview area
4. Monitor browser DevTools Performance tab
5. Check for stuttering, artifacts, or frame drops

### Expected Results
**Visual Quality:**
- Video resolution is at least 640x480
- Colors are accurate and not washed out
- No visual artifacts (blocking, pixelation)
- Motion is smooth, not stuttering

**Performance:**
- Frame rate is 15+ FPS (subjective assessment)
- No browser warnings in console
- Memory usage is stable (not climbing continuously)
- CPU usage is reasonable (< 50%)

### Quality Checklist
- [ ] Resolution: ⬜ 640x480 | ⬜ 1280x720 | ⬜ Other: ______
- [ ] Frame rate: ⬜ Smooth (15+ FPS) | ⬜ Acceptable (10-15 FPS) | ⬜ Poor (< 10 FPS)
- [ ] Artifacts: ⬜ None | ⬜ Minor | ⬜ Severe
- [ ] Colors: ⬜ Accurate | ⬜ Washed out | ⬜ Other: ______
- [ ] Stuttering: ⬜ None | ⬜ Occasional | ⬜ Frequent

### Actual Results
**Resolution:** _________________________
**Frame rate assessment:** _________________________
**Visual artifacts:** _________________________
**Browser console errors:** ⬜ None | ⬜ Errors: _________________________

### Performance Metrics (Optional)
**Memory usage (start):** _________________________
**Memory usage (after 1 min):** _________________________
**CPU usage:** _________________________
```

### Long-Running Operation Test Pattern
```markdown
## Test Case: TC-VID-004 - Video Detection Processing State

**Priority:** P0 (Critical)
**Aspect:** Loading state for async operations

### Test Scenario
Verify that user receives appropriate feedback during long-running video detection operation.

### Prerequisites
- Backend and frontend are running
- Test video file is available (30+ seconds duration)
- Console is open for monitoring

### Test Steps
1. Select a video file (30+ seconds)
2. Click "检测视频" (Detect Video) button
3. Start timer
4. Observe button state and status messages
5. Wait for processing to complete
6. Stop timer and record duration
7. Verify results display

### Expected Results

**Immediate Feedback (0-2 seconds):**
- Button text changes to "检测中..."
- Button becomes disabled (not clickable)
- Loading indicator displays (spinner or progress)
- Status message shows: "正在进行视频检测，耗时可能较长，请稍候。"

**During Processing (2+ seconds):**
- Button remains disabled
- Loading state persists
- No timeout errors (within 60 seconds)
- Console shows pending request

**Upon Completion:**
- Button text returns to "检测视频"
- Button becomes enabled
- Loading indicator disappears
- Annotated video displays in preview area
- Video statistics show: "总帧数：{N}，平均 FPS：{X.X}"
- Video auto-plays (or shows play button if autoplay blocked)

### Duration Tracking
**File size:** _________________________
**Video duration:** _________________________
**Processing time:** _________________________
**Acceptable:** ⬜ Yes | ⬜ No (took too long)

### State Verification
**Button state changes:**
- [ ] Initial: "检测视频" + enabled
- [ ] During: "检测中..." + disabled
- [ ] Final: "检测视频" + enabled

**Status messages:**
- [ ] Start message displays
- [ ] Completion message displays
- [ ] No conflicting messages

### Actual Results
**Button behavior:** _________________________
**Status messages:** _________________________
**Processing time:** _________________________
**Issues observed:** _________________________
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Word/Excel test cases | Markdown in version control | 2018-2020 | Better collaboration, audit trail |
| Separate test management | Docs-as-code with code | 2019-2021 | Tests stay in sync with features |
| Manual screenshots only | Screen recording + structured notes | 2020-2022 | Richer bug reports, better reproduction |
| Test cases after development | Test cases during/before development | 2021-present | Shift-left testing, earlier defect detection |

**Deprecated/outdated:**
- **Heavy test management tools** (TestRail, Jira Test for small teams): Overhead outweighs benefits for simple projects
- **Test case numbering by department**: Prefix-based numbering (QA-001, DEV-001) creates confusion
- **Separate test documentation**: Keeping tests in external systems leads to drift from code

## Open Questions

1. **Test result storage format**
   - What we know: Results need to be recorded in markdown files
   - What's unclear: Whether to use checkboxes, status badges, or summary tables
   - Recommendation: Use combination of inline checkboxes for quick status and summary table for overview

2. **Test execution session management**
   - What we know: Tests will be executed manually over multiple sessions
   - What's unclear: How to track which tests were run in which session
   - Recommendation: Add "Test Session" metadata to each test execution (date, tester, environment)

3. **Evidence attachment strategy**
   - What we know: Screenshots and notes may be needed for failed tests
   - What's unclear: Where to store screenshots and how to reference them
   - Recommendation: Store screenshots in `test-evidence/` folder with naming convention: `TC-ID-timestamp.png`

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Flask Backend | All E2E tests | ✓ | Existing | — |
| Vite Dev Server | All E2E tests | ✓ | Existing | — |
| Browser (Chrome/Edge) | UI testing | ✓ | Latest | — |
| Camera Device | Camera tests | ✓ | System hardware | Use mock if unavailable |
| Test media files | Image/Video tests | ⚠️ | To be acquired | Need sample files |

**Missing dependencies with no fallback:**
- Sample test media files (images and videos) — must be acquired or created before test execution

**Missing dependencies with fallback:**
- Camera device — if unavailable, camera tests can be marked as blocked and skipped

## Validation Architecture

> This section is included because workflow.nyquist_validation is not explicitly set to false in config.json.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Manual testing (no automation framework) |
| Config file | None — test documentation in Markdown |
| Quick run command | Execute test cases from `test-cases/` directory |
| Full suite command | Run all test cases in priority order |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Test Case File | Documented? |
|--------|----------|-----------|----------------|-------------|
| CAM-01 | Input camera ID | Manual validation | 01-camera-streaming-tests.md | ✅ Planned |
| CAM-02 | Start camera stream | Manual E2E | 01-camera-streaming-tests.md | ✅ Planned |
| CAM-03 | Stop camera stream | Manual E2E | 01-camera-streaming-tests.md | ✅ Planned |
| CAM-04 | Display camera feed | Visual validation | 01-camera-streaming-tests.md | ✅ Planned |
| CAM-05 | Status message update | Manual validation | 01-camera-streaming-tests.md | ✅ Planned |
| IMG-01 | Select image file | Manual validation | 02-image-detection-tests.md | ✅ Planned |
| IMG-02 | Trigger image detection | Manual E2E | 02-image-detection-tests.md | ✅ Planned |
| IMG-03 | Display annotated image | Visual validation | 02-image-detection-tests.md | ✅ Planned |
| IMG-04 | Display person count | Data assertion | 02-image-detection-tests.md | ✅ Planned |
| IMG-05 | Error handling | Edge case testing | 02-image-detection-tests.md | ✅ Planned |
| VID-01 | Select video file | Manual validation | 03-video-detection-tests.md | ✅ Planned |
| VID-02 | Trigger video detection | Manual E2E | 03-video-detection-tests.md | ✅ Planned |
| VID-03 | Processing state indication | State validation | 03-video-detection-tests.md | ✅ Planned |
| VID-04 | Display annotated video | Visual validation | 03-video-detection-tests.md | ✅ Planned |
| VID-05 | Display video statistics | Data assertion | 03-video-detection-tests.md | ✅ Planned |
| VID-06 | Button disabled state | State validation | 03-video-detection-tests.md | ✅ Planned |

### Sampling Rate
- **Per test execution:** Manual verification following test case document
- **Per feature completion:** Review all test cases in feature suite
- **Phase gate:** All P0 tests pass, P1 tests reviewed, documented gaps accepted

### Wave 0 Gaps
- [ ] `test-cases/00-test-environment-setup.md` — Environment setup and verification
- [ ] `test-cases/01-camera-streaming-tests.md` — CAM-01 to CAM-05 test cases
- [ ] `test-cases/02-image-detection-tests.md` — IMG-01 to IMG-05 test cases
- [ ] `test-cases/03-video-detection-tests.md` — VID-01 to VID-06 test cases
- [ ] `test-cases/04-edge-cases-tests.md` — D-19 to D-22 edge case tests
- [ ] `test-cases/05-interaction-state-tests.md` — D-23 to D-25 interaction tests
- [ ] Test media files — Sample images and videos for testing
- [ ] Test evidence directory — `test-evidence/` for screenshots and recordings

## Sources

### Primary (HIGH confidence)
- [Test Case Templates: 10 Free Formats and Examples for 2026](https://monday.com/blog/rnd/test-case-template/) - Comprehensive template examples and best practices
- [Test Case Template: Free Format & Examples 2026](https://testgrid.io/blog/test-case-template/) - Standard test case attributes and structure
- [How to Write Test Cases Effectively: Example and Template](https://testomat.io/blog/how-to-write-test-cases-effectively-example-and-template/) - Test case writing mastery with reusable elements
- [A simple test plan document in markdown format (GitHub Gist)](https://gist.github.com/DanElliott/33713ab1ec67b7f00d7bd21adfc51dba) - Markdown test plan structure
- [OTT Platform Testing: The Complete Guide](https://www.pcloudy.com/blogs/ott-platform-testing-guide/) - Video streaming testing patterns
- [Test Cases for Video Player in Your Application](https://www.fastpix.io/blog/test-cases-for-video-player-in-your-application) - Video player validation checklist

### Secondary (MEDIUM confidence)
- [Test Case Prioritization: Techniques and Best Practices](https://www.aiotests.com/blog/test-case-prioritization) - Prioritization framework (P0/P1/P2)
- [Edge Case Testing Explained](https://www.virtuosoqa.com/post/edge-case-testing) - Edge case identification and testing strategies
- [Web Testing Best Practices 2026](https://mechasm.ai/blog/web-testing-best-practices) - Modern QA playbook for web applications
- [UI Testing: A Complete Guide With Checklists](https://medium.com/@abhaykhs/ui-testing-a-complete-guide-with-checklists-and-example-9ed19ea244ec) - UI testing best practices
- [How to Structure and Organize Your Test Case Repository](https://www.eficode.com/blog/how-to-structure-and-organize-your-test-case-repository) - Test repository organization patterns

### Tertiary (LOW confidence)
- [Reddit - Test Case Folder Structure Discussion](https://www.reddit.com/r/QualityAssurance/comments/1hhbtur/how_do_you_structure_your_test_cases_in_a_folder/) - Community practices
- [UI Testing Checklist - BrowserStack](https://www.browserstack.com/guide/ui-testing-checklist) - General UI testing checklist

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Manual testing with Markdown documentation is well-established practice
- Architecture: HIGH - Test case template patterns verified from multiple official sources
- Pitfalls: HIGH - Common testing issues documented across multiple QA resources
- Video streaming testing: MEDIUM - Specific patterns verified but may need adjustment for Flask backend implementation

**Research date:** 2026-04-04
**Valid until:** 2026-05-04 (30 days - testing best practices are stable)
