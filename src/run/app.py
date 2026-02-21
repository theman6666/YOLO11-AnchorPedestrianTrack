from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

try:
    # python -m src.run.app
    from run.tracker import PedestrianTracker
except ModuleNotFoundError:
    # python src/run/app.py
    from tracker import PedestrianTracker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = PROJECT_ROOT / "frontend"
RUNTIME_DIR = PROJECT_ROOT / "runs" / "web"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
RESULT_DIR = RUNTIME_DIR / "results"
UPLOAD_IMAGE_DIR = UPLOAD_DIR / "images"
UPLOAD_VIDEO_DIR = UPLOAD_DIR / "videos"
RESULT_IMAGE_DIR = RESULT_DIR / "images"
RESULT_VIDEO_DIR = RESULT_DIR / "videos"
PORT = 5000

ALLOWED_IMAGE_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXTS = {"mp4", "avi", "mov", "mkv", "webm"}

for folder in (
    UPLOAD_IMAGE_DIR,
    UPLOAD_VIDEO_DIR,
    RESULT_IMAGE_DIR,
    RESULT_VIDEO_DIR,
):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

model_path = "result/hybrid_weights/YOLO11m_CBAM_Hybrid_local6/weights/best.pt"
tracker = PedestrianTracker(model_path=model_path)


def _allowed_file(filename: str, allowed_exts: set[str]) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_exts


def _timestamp_name(filename: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = secure_filename(filename)
    return f"{timestamp}_{safe_name}"


def _stream_camera(camera_id: int):
    tracker.reset_tracking_state()
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        empty = 255 * cv2.UMat(360, 640, cv2.CV_8UC3).get()
        cv2.putText(
            empty,
            f"Failed to open camera {camera_id}",
            (30, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )
        ok, buffer = cv2.imencode(".jpg", empty)
        if ok:
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            processed_frame = tracker.process_tracked_frame(frame)
            ok, buffer = cv2.imencode(".jpg", processed_frame)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    camera_id = request.args.get("camera_id", default="0")
    try:
        camera_id = int(camera_id)
    except ValueError:
        return jsonify({"ok": False, "message": "camera_id must be an integer."}), 400

    return Response(
        _stream_camera(camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/detect/image", methods=["POST"])
def detect_image():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"ok": False, "message": "No image file uploaded."}), 400

    if not _allowed_file(file.filename, ALLOWED_IMAGE_EXTS):
        return jsonify({"ok": False, "message": "Unsupported image format."}), 400

    input_name = _timestamp_name(file.filename)
    input_path = UPLOAD_IMAGE_DIR / input_name
    file.save(str(input_path))

    frame = cv2.imread(str(input_path))
    if frame is None:
        return jsonify({"ok": False, "message": "Failed to decode uploaded image."}), 400

    result_frame, count = tracker.process_image(frame)

    output_name = f"det_{Path(input_name).stem}.jpg"
    output_path = RESULT_IMAGE_DIR / output_name
    cv2.imwrite(str(output_path), result_frame)

    relative = output_path.relative_to(RESULT_DIR).as_posix()
    return jsonify(
        {
            "ok": True,
            "count": count,
            "image_url": f"/results/{relative}",
            "message": "Image detection completed.",
        }
    )


@app.route("/detect/video", methods=["POST"])
def detect_video():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"ok": False, "message": "No video file uploaded."}), 400

    if not _allowed_file(file.filename, ALLOWED_VIDEO_EXTS):
        return jsonify({"ok": False, "message": "Unsupported video format."}), 400

    input_name = _timestamp_name(file.filename)
    input_path = UPLOAD_VIDEO_DIR / input_name
    file.save(str(input_path))

    output_name = f"det_{Path(input_name).stem}.mp4"
    output_path = RESULT_VIDEO_DIR / output_name

    stats = tracker.process_video_file(input_path=input_path, output_path=output_path)

    relative = output_path.relative_to(RESULT_DIR).as_posix()
    return jsonify(
        {
            "ok": True,
            "video_url": f"/results/{relative}",
            "stats": stats,
            "message": "Video detection completed.",
        }
    )


@app.route("/results/<path:filename>")
def serve_result(filename: str):
    return send_from_directory(str(RESULT_DIR), filename)


if __name__ == "__main__":
    print(f"Web service started: http://127.0.0.1:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
