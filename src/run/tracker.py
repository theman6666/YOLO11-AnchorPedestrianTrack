import time

import cv2
from ultralytics import YOLO


class PedestrianTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        self.tracker_config = "bytetrack.yaml"
        self.conf_threshold = 0.25
        self.iou_threshold = 0.5

        self.prev_time = time.time()
        self.fps_ema = 0.0

    def process_frame(self, frame):
        if frame is None:
            return None

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],
            verbose=False,
        )

        now = time.time()
        instant_fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now
        if self.fps_ema == 0.0:
            self.fps_ema = instant_fps
        else:
            self.fps_ema = 0.9 * self.fps_ema + 0.1 * instant_fps

        annotated_frame = results[0].plot()

        count = 0
        if results[0].boxes.id is not None:
            count = len(results[0].boxes.id.tolist())

        cv2.putText(
            annotated_frame,
            f"Person Count: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated_frame,
            f"FPS: {self.fps_ema:.1f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        return annotated_frame
