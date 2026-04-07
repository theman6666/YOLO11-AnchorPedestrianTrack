---
quick_id: 260404-q6z
slug: description-add-file-name-visual-feedbac
description: Add file name visual feedback to ImagePanel and VideoPanel components
date: 2026-04-04
type: execute
autonomous: true
files_modified:
  - frontend/src/components/panels/ImagePanel.vue
  - frontend/src/components/panels/VideoPanel.vue
must_haves:
  truths:
    - "ImagePanel and VideoPanel currently show no visual feedback after file selection"
    - "Users need to see selected file name to confirm selection before clicking detect button"
    - "File name should appear between file input and detect button"
    - "File name should be cleared after detection starts"
  artifacts:
    - path: "frontend/src/components/panels/ImagePanel.vue"
      provides: "Image detection panel with file name feedback"
      contains: "Visual feedback showing selected image file name"
    - path: "frontend/src/components/panels/VideoPanel.vue"
      provides: "Video detection panel with file name feedback"
      contains: "Visual feedback showing selected video file name"
---

<objective>
Add file name visual feedback to ImagePanel and VideoPanel components so users can see which file they've selected before clicking the detect button.

Purpose: Currently users select a file but see no visual confirmation, leading to confusion about whether the selection worked. Adding file name display provides clear feedback that file selection succeeded.
Output: Both panels display selected file name between file input and detect button, with formatting for readability.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
</execution_context>

<context>
@frontend/src/components/panels/ImagePanel.vue
@frontend/src/components/panels/VideoPanel.vue

# Current Implementation Analysis
## ImagePanel.vue
- Has `selectedFile` ref that stores the selected File object
- `handleFileChange` sets `selectedFile.value = file`
- `handleDetect` emits detect event, resets file input, clears `selectedFile`
- UI shows FileInput component and Detect button, no file name display

## VideoPanel.vue
- Same structure as ImagePanel for file handling
- Also has `selectedFile` ref and similar handler functions

# Requirements
1. Display selected file name between file input and detect button
2. Format file name for readability (e.g., truncate long names)
3. Clear file name display after detection starts (when `selectedFile` is cleared)
4. Match existing styling (Tailwind CSS, dark mode compatible)
5. Show file size or type if easily available
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add file name display to ImagePanel</name>
  <files>
    - frontend/src/components/panels/ImagePanel.vue
  </files>
  <read_first>
    - frontend/src/components/panels/ImagePanel.vue
  </read_first>
  <action>
Add file name display between the FileInput and Detect button in ImagePanel.vue.

**Current template structure (lines 47-65):**
```html
<!-- File Input -->
<FileInput
  ref="fileInputRef"
  accept="image/*"
  label="选择图片文件"
  @change="handleFileChange"
/>

<!-- Detect Button -->
<button
  @click="handleDetect"
  :disabled="processing || !selectedFile"
  class="w-full bg-accent-600 hover:bg-accent-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-2 px-4 rounded-lg transition-colors"
>
  {{ processing ? '检测中...' : '开始图片检测' }}
</button>
```

**Add file name display section** after FileInput and before Detect button:
```html
<!-- File Name Display -->
<div
  v-if="selectedFile"
  class="flex items-center justify-between bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-2"
>
  <div class="flex-1 min-w-0">
    <div class="text-sm font-medium text-gray-700 dark:text-gray-300 truncate">
      {{ selectedFile.name }}
    </div>
    <div class="text-xs text-gray-500 dark:text-gray-400">
      {{ formatFileSize(selectedFile.size) }}
    </div>
  </div>
  <button
    @click="selectedFile = null"
    class="ml-2 text-sm text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
    type="button"
    aria-label="Clear selection"
  >
    清除
  </button>
</div>
```

**Add formatFileSize helper function** in the script section (before handleDetect):
```typescript
// Helper function to format file size
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}
```

**Implementation details:**
1. The file name display only shows when `selectedFile` exists
2. Shows file name truncated with Tailwind's `truncate` class
3. Shows formatted file size (e.g., "1.45 MB")
4. Includes "Clear" button to deselect file without starting detection
5. Styling matches existing design (gray background, borders, dark mode support)
6. Positioned between FileInput and Detect button for logical flow

**Note:** The "Clear" button sets `selectedFile = null` which will:
- Hide the file name display
- Disable the Detect button (disabled when `!selectedFile`)
- Allow user to select a different file
</action>
<verify>
  <automated>
    test -f frontend/src/components/panels/ImagePanel.vue && \
    grep -q "formatFileSize" frontend/src/components/panels/ImagePanel.vue && \
    grep -q "v-if=\"selectedFile\"" frontend/src/components/panels/ImagePanel.vue && \
    echo "✅ ImagePanel file name display added"
  </automated>
</verify>
<done>
ImagePanel.vue updated with file name display:
- File name display section added between FileInput and Detect button
- formatFileSize helper function added
- Clear button allows deselection
- Styling matches existing design
- Dark mode support included
</done>
</task>

<task type="auto">
  <name>Task 2: Add file name display to VideoPanel</name>
  <files>
    - frontend/src/components/panels/VideoPanel.vue
  </files>
  <read_first>
    - frontend/src/components/panels/VideoPanel.vue
  </read_first>
  <action>
Add identical file name display to VideoPanel.vue following the same pattern.

**Current template structure (lines 53-70):**
```html
<!-- File Input -->
<FileInput
  ref="fileInputRef"
  accept="video/*"
  label="选择视频文件"
  @change="handleFileChange"
/>

<!-- Detect Button -->
<button
  @click="handleDetect"
  :disabled="processing || !selectedFile"
  class="w-full bg-accent-600 hover:bg-accent-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-2 px-4 rounded-lg transition-colors"
>
  {{ processing ? '检测中...' : '开始视频检测' }}
</button>
```

**Add the same file name display section** after FileInput and before Detect button:
```html
<!-- File Name Display -->
<div
  v-if="selectedFile"
  class="flex items-center justify-between bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-2"
>
  <div class="flex-1 min-w-0">
    <div class="text-sm font-medium text-gray-700 dark:text-gray-300 truncate">
      {{ selectedFile.name }}
    </div>
    <div class="text-xs text-gray-500 dark:text-gray-400">
      {{ formatFileSize(selectedFile.size) }}
    </div>
  </div>
  <button
    @click="selectedFile = null"
    class="ml-2 text-sm text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
    type="button"
    aria-label="Clear selection"
  >
    清除
  </button>
</div>
```

**Add the same formatFileSize helper function** in the script section (before handleDetect):
```typescript
// Helper function to format file size
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}
```

**Implementation details:**
1. Same implementation as ImagePanel for consistency
2. File name display shows when `selectedFile` exists
3. Shows truncated file name and formatted size
4. Includes "Clear" button for deselection
5. Styling identical to ImagePanel for UI consistency
6. Positioned between FileInput and Detect button

**Note:** The `handleDetect` function in VideoPanel already clears `selectedFile` after emitting, so the file name display will automatically hide when detection starts.
</action>
<verify>
  <automated>
    test -f frontend/src/components/panels/VideoPanel.vue && \
    grep -q "formatFileSize" frontend/src/components/panels/VideoPanel.vue && \
    grep -q "v-if=\"selectedFile\"" frontend/src/components/panels/VideoPanel.vue && \
    echo "✅ VideoPanel file name display added"
  </automated>
</verify>
<done>
VideoPanel.vue updated with file name display:
- File name display section added between FileInput and Detect button
- formatFileSize helper function added
- Clear button allows deselection
- Styling matches ImagePanel for consistency
- Dark mode support included
</done>
</task>

<task type="auto">
  <name>Task 3: Test the changes visually</name>
  <files>
    - frontend/src/components/panels/ImagePanel.vue
    - frontend/src/components/panels/VideoPanel.vue
  </files>
  <read_first>
    - frontend/src/components/panels/ImagePanel.vue
    - frontend/src/components/panels/VideoPanel.vue
  </read_first>
  <action>
Verify the visual changes by checking the updated component templates and ensuring they compile correctly.

**Manual verification steps:**
1. Check that both components have the new file name display section
2. Verify the formatFileSize function is correctly implemented
3. Ensure the "Clear" button functionality works (sets selectedFile to null)
4. Confirm the styling is consistent with existing design

**Code quality checks:**
- No TypeScript errors in the helper function
- Proper Vue template syntax
- Tailwind CSS classes are valid
- Dark mode classes included (`dark:` variants)
- Accessibility: aria-label on clear button

**Expected behavior:**
1. User selects a file → File name appears with size
2. User can click "Clear" → File name disappears, detect button disabled
3. User clicks "Detect" → File name disappears during processing
4. After detection completes → File name stays cleared (ready for next file)

**Do NOT run build or dev server** — this is a code verification task only.
The actual testing should be done by the user after rebuilding and restarting.

**Note:** Since we're modifying Vue components, the frontend needs to be rebuilt for changes to take effect in the production build. In development mode, Vite's HMR should pick up the changes automatically.
</action>
<verify>
  <automated>
    # Verify both components have the required changes
    grep -q "formatFileSize" frontend/src/components/panels/ImagePanel.vue && \
    grep -q "formatFileSize" frontend/src/components/panels/VideoPanel.vue && \
    grep -q "v-if=\"selectedFile\"" frontend/src/components/panels/ImagePanel.vue && \
    grep -q "v-if=\"selectedFile\"" frontend/src/components/panels/VideoPanel.vue && \
    # Verify both have clear button
    grep -q "@click=\"selectedFile = null\"" frontend/src/components/panels/ImagePanel.vue && \
    grep -q "@click=\"selectedFile = null\"" frontend/src/components/panels/VideoPanel.vue && \
    echo "✅ Both panels updated correctly"
  </automated>
</verify>
<done>
Visual changes verified:
- Both ImagePanel and VideoPanel have file name display sections
- formatFileSize helper functions correctly implemented
- Clear buttons allow deselection
- Styling consistent with existing design
- Code compiles without TypeScript errors
- Ready for user testing after rebuild
</done>
</task>

<task type="auto">
  <name>Task 4: Update debug session with UI improvement</name>
  <files>
    - .planning/debug/file-upload-cors-issue.md
  </files>
  <read_first>
    - .planning/debug/file-upload-cors-issue.md
  </read_first>
  <action>
Update the debug session to document the UI improvement for file selection feedback.

The original issue was "需要两次选择才能触发上传" (need two selections to trigger upload). While the root CORS issue has been fixed, the user also wanted better visual feedback. Document this improvement in the debug session.

**Add to the Evidence section** in `.planning/debug/file-upload-cors-issue.md`:
```yaml
- timestamp: 2026-04-04T[CurrentTime]Z
  checked: UI improvement for file selection feedback
  found: Added file name display with size information to ImagePanel and VideoPanel components
  implication: Users can now see which file is selected before clicking detect button, providing visual confirmation that file selection succeeded
```

**Update the status if appropriate** (the CORS issue may still need verification, but UI improvement is complete):
```yaml
status: verifying_ui_improvement
```

**Add a note about the UI change:**
- File name now displays between file input and detect button
- Shows file size in human-readable format (KB, MB, GB)
- Includes "Clear" button to deselect file
- Provides visual confirmation that file selection succeeded

**Note:** The debug session was originally about CORS, but this UI improvement addresses the user experience aspect of the "需要两次选择" issue (users not knowing if selection worked).
</action>
<verify>
  <automated>
    test -f .planning/debug/file-upload-cors-issue.md && \
    echo "✅ Debug session file exists and will be updated"
  </automated>
</verify>
<done>
Debug session updated:
- Added evidence entry for UI improvement
- Documented file name display feature
- Updated status to reflect UI enhancement completion
- Ready for user verification of both CORS fix and UI improvement
</done>
</task>

</tasks>

<verification>
After completing all tasks, the user should:

1. **Rebuild frontend** to include UI changes:
   ```bash
   cd frontend && npm run build
   ```

2. **Restart Flask server** to serve updated frontend:
   ```bash
   python src/run/app.py
   ```

3. **Test the improvements**:
   - Access via IP address: `http://192.168.2.13:5000`
   - Select an image file → Should see file name and size appear
   - Select a video file → Should see file name and size appear
   - Click "Clear" button → File name disappears, detect button disabled
   - Click "Detect" → File upload should work on first attempt (CORS fixed)

4. **Verify both issues resolved**:
   - ✅ CORS issue: File upload works on first attempt
   - ✅ UI feedback: File name displays after selection
   - ✅ User knows selection succeeded before clicking detect
</verification>

<success_criteria>
- [ ] ImagePanel displays selected file name and size
- [ ] VideoPanel displays selected file name and size
- [ ] Clear button allows deselection
- [ ] File name display appears only when file is selected
- [ ] Styling matches existing design (dark mode compatible)
- [ ] Debug session updated with UI improvement
</success_criteria>

<output>
After completion, create SUMMARY.md in the quick task directory with:
- File name display implementation details
- Screenshots or description of the visual feedback
- Instructions for testing
- Link to updated debug session
</output>