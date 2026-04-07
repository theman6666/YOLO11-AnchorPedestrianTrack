import time
from pathlib import Path
from threading import Lock

import cv2
import subprocess
from ultralytics import YOLO


class PedestrianTracker:
    def __init__(self, model_path: str):
        # Use separate model instances for tracking and detection to avoid
        # predictor callback state leaking between modes.
        self.track_model = YOLO(model_path)
        self.detect_model = YOLO(model_path)

        self.tracker_config = str(Path(__file__).resolve().parents[1] / "utils" / "bytetrack.yaml")
        self.conf_threshold = 0.25
        self.iou_threshold = 0.5

        self.prev_time = time.time()
        self.fps_ema = 0.0

        # Ultralytics model is not thread-safe for concurrent inference.
        self._infer_lock = Lock()

    def reset_tracking_state(self):
        self.prev_time = time.time()
        self.fps_ema = 0.0

        predictor = getattr(self.track_model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            # Remove attribute so Ultralytics can re-initialize trackers even with persist=True.
            del predictor.trackers
        if predictor is not None and hasattr(predictor, "vid_path"):
            del predictor.vid_path

    def _update_fps(self) -> float:
        now = time.time()
        instant_fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now
        if self.fps_ema == 0.0:
            self.fps_ema = instant_fps
        else:
            self.fps_ema = 0.9 * self.fps_ema + 0.1 * instant_fps
        return self.fps_ema

    @staticmethod
    def _get_count(result) -> int:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return 0

        if getattr(boxes, "id", None) is not None:
            return len(boxes.id.tolist())

        if getattr(boxes, "xyxy", None) is not None:
            return len(boxes.xyxy)

        return 0

    @staticmethod
    def _overlay_text(frame, count: int, fps: float | None = None):
        cv2.putText(
            frame,
            f"Person Count: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if fps is not None:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

    def process_tracked_frame(self, frame):
        if frame is None:
            return None

        with self._infer_lock:
            results = self.track_model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_config,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],
                verbose=False,
            )

        result = results[0]
        annotated_frame = result.plot()
        count = self._get_count(result)
        fps = self._update_fps()
        self._overlay_text(annotated_frame, count, fps=fps)
        return annotated_frame

    def process_image(self, frame):
        if frame is None:
            return None, 0

        with self._infer_lock:
            results = self.detect_model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],
                verbose=False,
            )

        result = results[0]
        annotated_frame = result.plot()
        count = self._get_count(result)
        self._overlay_text(annotated_frame, count, fps=None)
        return annotated_frame, count

    def process_video_file(self, input_path: Path, output_path: Path):
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 重置跟踪状态
        self.reset_tracking_state()

        # 获取视频属性
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        in_fps = cap.get(cv2.CAP_PROP_FPS)
        out_fps = in_fps if in_fps and in_fps > 1 else 25.0
        cap.release()

        # 创建 VideoWriter - 使用 MJPG 编码（稳定兼容）
        temp_output_path = output_path.with_suffix(".avi")
        output_path_str = str(temp_output_path)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path_str, fourcc, out_fps, (width, height))

        if not writer.isOpened():
            raise ValueError(f"Cannot open output video writer: {output_path}")

        # 对整个视频文件调用一次 track()，使用流式处理
        start = time.time()
        frames = 0

        try:
            results = self.track_model.track(
                source=str(input_path),
                tracker=self.tracker_config,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],
                persist=True,
                verbose=False,
                stream=True,
            )

            for result in results:
                annotated_frame = result.plot()
                count = self._get_count(result)
                fps = self._update_fps()
                self._overlay_text(annotated_frame, count, fps=fps)
                writer.write(annotated_frame)
                frames += 1
        finally:
            writer.release()

        elapsed = max(time.time() - start, 1e-6)

        # Transcode to H.264 MP4 so browsers can play it reliably.
        try:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_output_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            if temp_output_path.exists():
                temp_output_path.unlink()

        return {
            "frames": frames,
            "elapsed_sec": round(elapsed, 2),
            "avg_fps": round(frames / elapsed, 2),
            "output_path": str(output_path),
            "output_ext": output_path.suffix,
        }

    # Backward-compatible name used by older code.
    def process_frame(self, frame):
        return self.process_tracked_frame(frame)
