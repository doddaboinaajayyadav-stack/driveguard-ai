
from pathlib import Path
import threading

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template

from config import BASE_DIR, FLASK_HOST, FLASK_PORT
from src.model import DrowsinessCNN

app = Flask(__name__)
ROOT = Path(BASE_DIR)

CHECKPOINT = ROOT / "models" / "best_model.pt"
DEVICE = torch.device("cpu")
CLOSED_CLASS = 1
OPEN_CLASS = 0
CLOSED_FRAMES_REQUIRED = 20

camera_lock = threading.Lock()
camera = None
closed_frames = 0
model = None
model_error = None

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def load_model():
    global model, model_error

    try:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

        network = DrowsinessCNN().to(DEVICE)
        checkpoint = torch.load(
            CHECKPOINT, map_location=DEVICE, weights_only=True
        )

        # Support either a plain state dict or a wrapped checkpoint.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

        network.load_state_dict(checkpoint)
        network.eval()

        model = network
        model_error = None
        print("CNN loaded successfully.")
    except Exception as exc:
        model = None
        model_error = str(exc)
        print(f"CNN loading failed: {exc}")


def predict_eye(eye_gray):
    """Return the predicted eye state and its softmax confidence."""
    resized = cv2.resize(eye_gray, (24, 24))
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        class_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[class_id].item())

    label = "CLOSED" if class_id == CLOSED_CLASS else "OPEN"
    return label, confidence


def generate_frames():
    global camera, closed_frames

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        print("Could not open webcam.")
        return

    with camera_lock:
        camera = cap

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(80, 80),
            )

            any_closed = False
            any_prediction = False

            for (x, y, w, h) in faces:
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (50, 200, 255), 2
                )

                face_gray = gray[y:y + h, x:x + w]
                eyes = eye_detector.detectMultiScale(
                    face_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20),
                )

                for (ex, ey, ew, eh) in eyes[:2]:
                    eye_gray = face_gray[ey:ey + eh, ex:ex + ew]

                    if model is None or eye_gray.size == 0:
                        continue

                    try:
                        label, confidence = predict_eye(eye_gray)
                    except Exception as exc:
                        print(f"Prediction error: {exc}")
                        continue

                    any_prediction = True
                    any_closed = any_closed or label == "CLOSED"

                    color = (
                        (0, 0, 255) if label == "CLOSED"
                        else (0, 220, 0)
                    )
                    cv2.rectangle(
                        frame,
                        (x + ex, y + ey),
                        (x + ex + ew, y + ey + eh),
                        color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"{label} {confidence:.0%}",
                        (x + ex, max(20, y + ey - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            if any_prediction and any_closed:
                closed_frames += 1
            elif any_prediction:
                closed_frames = 0

            if model is None:
                status = "MODEL ERROR - CHECK TERMINAL"
                status_color = (0, 165, 255)
            elif closed_frames >= CLOSED_FRAMES_REQUIRED:
                status = "WARNING: SUSTAINED EYE CLOSURE"
                status_color = (0, 0, 255)
            elif any_prediction:
                status = "CNN PREDICTIONS ACTIVE"
                status_color = (0, 220, 0)
            else:
                status = "NO EYE PREDICTION"
                status_color = (0, 165, 255)

            cv2.putText(
                frame,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_color,
                2,
            )

            ok, buffer = cv2.imencode(".jpg", frame)
            if ok:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
    finally:
        cap.release()
        with camera_lock:
            if camera is cap:
                camera = None
        closed_frames = 0


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_ready=model is not None,
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    return jsonify({
        "camera_preview": "available",
        "model_checkpoint_found": CHECKPOINT.is_file(),
        "model_loaded": model is not None,
        "drowsiness_predictions": model is not None,
        "message": model_error or (
            "CNN loaded; webcam predictions require live verification."
            if model is not None else "Model is not loaded."
        ),
    })


if __name__ == "__main__":
    load_model()
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=False,
        threaded=True,
    )
