import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------
# 让 Python 找到你本地的 ByteTrack 源码
# ---------------------------------------
BYTE_DIR = r"D:\YOLO11-AnchorPedestrianTrack\ByteTrack"
sys.path.append(BYTE_DIR)

from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer

# ---------------------------------------
# 路径配置
# ---------------------------------------
MODEL_PATH = r"D:\YOLO11-AnchorPedestrianTrack\models\YOLO11s.pt"
VIDEO_DIR = r"D:\YOLO11-AnchorPedestrianTrack\video"
OUTPUT_DIR = r"D:\YOLO11-AnchorPedestrianTrack\result"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def video_tracker(input_video):

    print("🚀 Loading YOLO11 model...")
    model = YOLO(MODEL_PATH)

    # ByteTrack 默认参数
    tracker = BYTETracker(
        track_thresh=0.5,
        match_thresh=0.8,
        track_buffer=30
    )

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("❌ Cannot open:", input_video)
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join(
        OUTPUT_DIR,
        os.path.basename(input_video).replace(".mp4", "_tracked.mp4")
    )

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    frame_id = 0
    timer = Timer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timer.tic()

        # YOLO 推理
        results = model(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0].item())
            if cls != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = box.conf[0].item()
            detections.append([x1, y1, x2, y2, score])

        detections = np.array(detections)

        # ByteTrack 跟踪
        online_targets = tracker.update(detections, [h, w], [h, w])

        # 画图
        online_tlwhs = []
        online_ids = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id

            if tlwh[2] * tlwh[3] < 10:
                continue

            online_tlwhs.append(tlwh)
            online_ids.append(tid)

        timer.toc()

        fps_text = f"FPS: {1.0 / timer.average_time:.2f}"
        result_img = plot_tracking(frame, online_tlwhs, online_ids, fps_text=fps_text)

        writer.write(result_img)

        frame_id += 1
        print(f"\r处理帧：{frame_id}", end="")

    print("\n🎉 跟踪完成，输出文件：", output_path)

    cap.release()
    writer.release()


if __name__ == "__main__":
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

    if not videos:
        print("⚠ 请将视频放在：", VIDEO_DIR)
    else:
        video_path = os.path.join(VIDEO_DIR, videos[0])
        video_tracker(video_path)
