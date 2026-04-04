# Test Environment Setup

**Purpose:** Instructions for preparing the testing environment before executing Phase 5 test cases

**Last Updated:** 2026-04-04

---

## Overview

This document provides step-by-step instructions for setting up the testing environment to execute Phase 5 manual E2E test cases. The environment consists of two components:

1. **Flask Backend** - Python server providing detection and streaming APIs
2. **Vite Frontend** - Vue 3 development server serving the refactored application

Both components must be running simultaneously for test execution.

---

## System Requirements

### Hardware

- **CPU:** Modern multi-core processor (Intel i5+/AMD Ryzen 5+ recommended)
- **Memory:** Minimum 8GB RAM (16GB recommended for video processing)
- **Storage:** 500MB free disk space for application files
- **Camera:** At least one camera device connected (for camera streaming tests)

### Software

- **Operating System:** Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python:** Version 3.8 or higher (required for Flask backend)
- **Node.js:** Version 16 or higher (required for Vite dev server)
- **Browser:** Chrome 90+ or Edge 90+ (recommended for DevTools support)
- **Git:** For version control (optional for testing only)

### Test Media Files

Prepare the following test media before starting test execution:

#### Required Images (minimum 2-3 files)

- **Format:** JPG or PNG
- **Content:** Images containing people (1-5 people per image)
- **Resolution:** Any (test with various resolutions)
- **Examples:**
  - `test-image-1-person.jpg` - Image with 1 person
  - `test-image-3-people.jpg` - Image with 3 people
  - `test-image-5-people.jpg` - Image with 5+ people

#### Required Videos (minimum 1-2 files)

- **Format:** MP4 or AVI
- **Duration:** 10-30 seconds (shorter videos process faster)
- **Content:** Video containing people walking/moving
- **Resolution:** 720p or 1080p recommended
- **Examples:**
  - `test-video-short.mp4` - 10-15 second video
  - `test-video-medium.mp4` - 20-30 second video

#### Invalid Format Files (for error testing)

- `test-file.txt` - Text file
- `test-file.pdf` - PDF document
- `test-file.exe` - Executable file (optional)

#### Optional: Large Files

- `test-large-image.jpg` - 10MB+ image
- `test-large-video.mp4` - 50MB+ video (for upload testing)

---

## Backend Setup

### Step 1: Navigate to Project Root

```bash
cd /path/to/YOLO11-AnchorPedestrianTrack
```

### Step 2: Verify Python Installation

```bash
python --version
# Expected output: Python 3.8.x or higher
```

If Python is not installed, download from [python.org](https://www.python.org/downloads/).

### Step 3: Verify Dependencies

```bash
# Check if required packages are installed
pip list | grep -E "(flask|opencv|torch)"
```

Expected packages (varies by environment):
- Flask
- opencv-python
- torch
- numpy

If dependencies are missing, install them:

```bash
pip install -r requirements.txt
# Or install individual packages as needed
```

### Step 4: Start Flask Backend

```bash
python src/run/app.py
```

**Expected Terminal Output:**

```
 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### Step 5: Verify Backend is Running

Open a browser or use curl:

```bash
curl http://localhost:5000
# Or browse to: http://localhost:5000
```

**Expected Result:**
- Browser shows Flask response or API documentation
- No "Connection refused" error

### Step 6: Verify Key Endpoints

```bash
# Check video feed endpoint (should return continuous stream)
curl -I http://localhost:5000/video_feed
# Expected: HTTP/1.1 200 OK

# Check detect/image endpoint (should return 405 Method Not Allowed for GET)
curl -I http://localhost:5000/detect/image
# Expected: HTTP/1.1 405 METHOD NOT ALLOWED
```

### Step 7: Stop Flask Backend

When testing is complete:

```bash
# Press Ctrl+C in the terminal running Flask
# Expected output: "Shutting down..." or similar
```

---

## Frontend Setup

### Step 1: Navigate to Frontend Directory

```bash
cd frontend-vue
```

### Step 2: Verify Node.js Installation

```bash
node --version
# Expected output: v16.x.x or higher

npm --version
# Expected output: 8.x.x or higher
```

If Node.js is not installed, download from [nodejs.org](https://nodejs.org/).

### Step 3: Install Dependencies (First Time Only)

```bash
npm install
```

**Expected Output:**
- Downloads and installs packages
- Shows "added XXX packages" message
- No critical errors

### Step 4: Start Vite Dev Server

```bash
npm run dev
```

**Expected Terminal Output:**

```
  VITE v5.x.x  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

### Step 5: Verify Frontend is Running

Open a browser and navigate to:

```
http://localhost:5173
```

**Expected Result:**
- YOLO11 pedestrian detection application loads
- Three panels visible: Camera, Image, Video
- Dark mode theme active
- No console errors (check DevTools)

### Step 6: Stop Vite Dev Server

When testing is complete:

```bash
# Press Ctrl+C in the terminal running Vite
# Expected output: "closing server..." or similar
```

---

## Pre-Test Checklist

Before executing any test cases, verify the following:

### Environment Status

- [ ] Flask backend is running (`python src/run/app.py`)
  - [ ] Terminal shows "Running on http://127.0.0.1:5000"
  - [ ] Backend responds to http://localhost:5000

- [ ] Vite dev server is running (`npm run dev`)
  - [ ] Terminal shows "Local: http://localhost:5173/"
  - [ ] Application loads at http://localhost:5173

### Browser Setup

- [ ] Browser is open to http://localhost:5173
- [ ] DevTools are open (Press F12 or right-click → Inspect)
- [ ] Console tab shows no errors
- [ ] Network tab is enabled (for API monitoring)
- [ ] Application is visible and responsive

### Test Media

- [ ] Test images are accessible (JPG/PNG files with people)
- [ ] Test videos are accessible (MP4/AVI files, 10-30 seconds)
- [ ] Invalid format files are available (.txt, .pdf, etc.)
- [ ] Large files available if testing upload limits (optional)

### Camera Device

- [ ] Camera is connected and recognized by system
- [ ] Camera ID is known (usually 0, but can be 1, 2, etc.)
- [ ] Camera works in other applications (test with Camera app)

### Optional: Advanced Setup

- [ ] Screen recording tool ready (for documenting bugs)
- [ ] Notepad/text editor open (for recording results)
- [ ] Test case documents open for reference

---

## Post-Test Cleanup

After completing test execution:

### Application State

- [ ] Stop any active camera streams
  - [ ] Click "停止摄像头" button in Camera panel
  - [ ] Verify preview area is cleared

- [ ] Clear browser cache if needed
  - [ ] Press Ctrl+Shift+Delete (Chrome/Edge)
  - [ ] Select "Cached images and files"
  - [ ] Click "Clear data"

- [ ] Close all browser tabs
- [ ] Close DevTools

### Server Shutdown

- [ ] Stop Flask backend (Ctrl+C in backend terminal)
- [ ] Stop Vite dev server (Ctrl+C in frontend terminal)
- [ ] Verify both terminals show shutdown messages

### Test Artifacts

- [ ] Record any environment issues encountered
- [ ] Note any required environment changes for next session
- [ ] Save test results in appropriate test case documents
- [ ] Archive screenshots/recordings if collected

---

## Troubleshooting

### Backend Issues

**Problem:** Backend fails to start with "Address already in use"

**Solution:**
```bash
# Find process using port 5000
netstat -ano | findstr :5000  # Windows
lsof -i :5000                  # macOS/Linux

# Kill the process or use different port
# Edit src/run/app.py: app.run(port=5001)
```

**Problem:** Backend starts but returns 404 errors

**Solution:**
- Verify project structure matches expected layout
- Check if `src/run/app.py` exists
- Verify working directory is project root

**Problem:** Import errors for missing packages

**Solution:**
```bash
pip install flask opencv-python torch numpy
# Or use: pip install -r requirements.txt
```

### Frontend Issues

**Problem:** Vite dev server fails to start

**Solution:**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json  # macOS/Linux
rmdir /s node_modules package-lock.json  # Windows
npm install
```

**Problem:** Frontend loads but shows blank page

**Solution:**
- Check browser console for errors
- Verify all dependencies installed successfully
- Check if port 5173 is accessible (try http://localhost:5173)

**Problem:** API calls fail with "Network Error"

**Solution:**
- Verify Flask backend is running on port 5000
- Check Vite proxy configuration in `vite.config.ts`
- Check browser Network tab for failed requests

### Browser Issues

**Problem:** Application loads but camera doesn't work

**Solution:**
- Verify browser has camera permissions
- Check if camera works in other applications
- Try different camera ID values (0, 1, 2)

**Problem:** Console shows CORS errors

**Solution:**
- Verify Vite proxy is configured correctly
- Check Flask CORS settings if implemented
- Ensure both servers are running locally

**Problem:** Video detection times out

**Solution:**
- Verify video file is not corrupted
- Check backend logs for processing errors
- Try shorter video file (10-15 seconds)

---

## Quick Reference Commands

### Backend

```bash
# Start backend
python src/run/app.py

# Check backend health
curl http://localhost:5000

# Check specific endpoints
curl -I http://localhost:5000/video_feed
curl -I http://localhost:5000/detect/image
curl -I http://localhost:5000/detect/video
```

### Frontend

```bash
# Start frontend
cd frontend-vue && npm run dev

# Check frontend health
curl http://localhost:5173

# Install dependencies
npm install

# Build for production (not used in testing)
npm run build
```

### Browser DevTools

```bash
# Open DevTools
F12 or Ctrl+Shift+I (Windows/Linux)
Cmd+Option+I (macOS)

# Focus Console tab
Ctrl+Shift+J (Windows/Linux)
Cmd+Option+J (macOS)

# Focus Network tab
Ctrl+Shift+E (Windows/Linux)
Cmd+Option+E (macOS)
```

---

## Notes

- **Terminal Management:** Keep backend and frontend terminals open during testing
- **Port Conflicts:** If default ports (5000, 5173) are unavailable, modify configuration files
- **Performance:** Close unnecessary applications to ensure smooth video processing
- **Camera Testing:** If camera is unavailable, camera tests can be marked as "Blocked"
- **Test Duration:** Full test suite takes approximately 2-3 hours to execute
- **Session Management:** Document test session start/end times for traceability

---

## Next Steps

After completing environment setup:

1. ✅ Verify all items in Pre-Test Checklist
2. 📖 Open test case documents in `test-cases/` directory
3. 🧪 Execute test suites in order:
   - 01-camera-streaming-tests.md
   - 02-image-detection-tests.md
   - 03-video-detection-tests.md
   - 04-edge-cases-tests.md
   - 05-interaction-state-tests.md
4. 📊 Update 05-VALIDATION.md with test results

---

**Document Status:** ✅ Ready for use

**Version:** 1.0

**Last Reviewed:** 2026-04-04
