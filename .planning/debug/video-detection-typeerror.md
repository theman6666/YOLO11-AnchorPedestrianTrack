---
status: verifying
trigger: "video-detection-typeerror"
created: 2026-04-04T00:00:00Z
updated: 2026-04-04T00:00:09Z
---

## Current Focus
hypothesis: ROOT CAUSE CONFIRMED - reset_tracking_state() sets predictors.trackers=None, then model.track() without persist=True causes on_predict_start to return early without reinitializing trackers
test: Verify that adding persist=True to model.track() call fixes the issue
expecting: Trackers will be properly initialized and video processing will succeed
next_action: Apply fix to tracker.py

## Symptoms
expected: 视频检测成功，返回带边界框的视频
actual: 服务器 500 错误，TypeError: 'NoneType' object is not subscriptable
errors: |
  TypeError: 'NoneType' object is not subscriptable
  File: D:\YOLO11-AnchorPedestrianTrack\src\run\tracker.py, line 161, in process_video_file
  File: D:\YOLO11-AnchorPedestrianTrack\lib\site-packages\ultralytics\trackers\track.py, line 86, in on_predict_postprocess_end
      tracker = predictor.trackers[i if is_stream else 0]
reproduction: |
  1. 启动 Flask 服务器: python src/run/app.py
  2. 访问 http://localhost:5000
  3. 上传视频文件并点击"检测视频"
  4. 服务器返回 500 错误
started: 之前正常，现在失败。摄像头流工作正常，只有视频检测失败

## Eliminated

## Evidence
- timestamp: 2026-04-04T00:00:01Z
  checked: ultralytics/trackers/track.py line 61
  found: Error occurs at `tracker = predictor.trackers[i if is_stream else 0]` where predictor.trackers is None
  implication: on_predict_start callback was not called to initialize trackers

- timestamp: 2026-04-04T00:00:02Z
  checked: ultralytics/trackers/track.py on_predict_start function (line 18-45)
  found: on_predict_start initializes predictor.trackers list with tracker instances
  implication: This callback must be registered and called before on_predict_postprocess_end

- timestamp: 2026-04-04T00:00:03Z
  checked: tracker.py process_video_file method (line 151-159)
  found: Uses `self.model.track(source=str(input_path), ..., stream=True)` with persist=False (default)
  implication: persist=True might be needed to maintain tracker state across iterations

- timestamp: 2026-04-04T00:00:04Z
  checked: tracker.py process_tracked_frame method (line 83-91)
  found: Uses `self.model.track(source=frame, persist=True, ...)` for camera streaming
  implication: Camera streaming uses persist=True while video file uses default persist=False

- timestamp: 2026-04-04T00:00:05Z
  checked: Difference between process_tracked_frame and process_video_file
  found: 
    * process_tracked_frame: source=frame (numpy array), persist=True, verbose=False
    * process_video_file: source=str(input_path) (file path), persist=False (default), stream=True, verbose=False
  implication: The combination of stream=True with file path AND missing persist=True may prevent proper tracker initialization

- timestamp: 2026-04-04T00:00:06Z
  checked: ultralytics/trackers/track.py register_tracker function (line 80-89)
  found: register_tracker registers callbacks using partial(persist=persist)
  implication: The persist parameter is passed to both on_predict_start and on_predict_postprocess_end callbacks

- timestamp: 2026-04-04T00:00:07Z
  checked: tracker.py reset_tracking_state method (line 23-29)
  found: Sets `predictor.trackers = None` when resetting state
  implication: After reset_tracking_state() is called in process_video_file, the trackers are explicitly set to None

## Resolution
root_cause: |
  In tracker.py process_video_file method:
  1. reset_tracking_state() is called at line 125, which sets predictor.trackers = None
  2. model.track() is called at line 151-159 with stream=True but WITHOUT persist=True
  3. In ultralytics/trackers/track.py, on_predict_start callback (line 29-30) checks if trackers exist and persist=False, then returns early without initializing
  4. This leaves predictor.trackers = None
  5. on_predict_postprocess_end tries to access predictor.trackers[0] at line 61, causing NoneType error
  
  The process_tracked_frame method works because it uses persist=True, which forces tracker initialization even after reset.
fix: |
  Add persist=True parameter to model.track() call in process_video_file method (line 151)
  
  Change:
    results = self.model.track(
        source=str(input_path),
        tracker=self.tracker_config,
        conf=self.conf_threshold,
        iou=self.iou_threshold,
        classes=[0],
        verbose=False,
        stream=True,
    )
  
  To:
    results = self.model.track(
        source=str(input_path),
        persist=True,  # ADD THIS LINE
        tracker=self.tracker_config,
        conf=self.conf_threshold,
        iou=self.iou_threshold,
        classes=[0],
        verbose=False,
        stream=True,
    )
verification: |
  After applying fix:
  1. Start Flask server
  2. Upload a video file
  3. Click "Detect Video"
  4. Verify that video processing completes successfully without 500 error
  5. Verify that output video contains bounding boxes and tracking IDs
files_changed:
  - src/run/tracker.py: Add persist=True to model.track() call in process_video_file
