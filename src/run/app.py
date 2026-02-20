from pathlib import Path

import cv2
from flask import Flask, Response, render_template

try:
    # python -m src.run.app
    from run.tracker import PedestrianTracker
except ModuleNotFoundError:
    # python src/run/app.py
    from tracker import PedestrianTracker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = PROJECT_ROOT / "frontend"
PORT = 5000

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

model_path = "result/hybrid_weights/YOLO11m_CBAM_Hybrid_local6/weights/best.pt"
tracker = PedestrianTracker(model_path=model_path)

video_source = 0
cap = cv2.VideoCapture(video_source)


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        processed_frame = tracker.process_frame(frame)
        _, buffer = cv2.imencode(".jpg", processed_frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    print(f"Web service started: http://127.0.0.1:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
