import base64
import json
import os
import tempfile
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from face_utils import crop_face_with_box
from inference import TOPK_FRACTION, UNCERTAIN_MARGIN, Detector, is_uncertain, topk_mean

CKPT = os.environ.get("CKPT", "checkpoints/best.pt")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response
_detector = None
_lock = threading.Lock()  # one prediction at a time (the model keeps Grad-CAM state)


def get_detector():
    global _detector
    if _detector is None:
        if not os.path.exists(CKPT):
            raise FileNotFoundError(f"No trained model at {CKPT}. Run train.py first.")
        _detector = Detector(CKPT)
    return _detector


def to_data_url(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def sample_frames(path, n=16):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    # np.unique: very short videos would otherwise repeat the same frame index several times
    for i in np.unique(np.linspace(0, max(total - 1, 0), n).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/metrics")
def metrics():
    path = "results/metrics.json"
    if not os.path.exists(path):
        return jsonify([])
    with open(path) as fh:
        return jsonify([entry for entry in json.load(fh).values() if entry.get("validated") is True])


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 200
    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify(error="Choose a photo or video first."), 400
    try:
        det = get_detector()
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503

    is_video = (f.mimetype or "").startswith("video") or f.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

    if is_video:
        suffix = os.path.splitext(f.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            f.save(tmp.name)
        try:
            frames = sample_frames(tmp.name)
        finally:
            os.remove(tmp.name)
        kind = "video"
    else:
        try:
            from PIL import Image, ImageOps
            import io
            file_bytes = f.read()
            pil_img = Image.open(io.BytesIO(file_bytes))
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return jsonify(error="That file could not be read as an image. Please provide a valid JPG, PNG, or WEBP."), 400
        frames, kind = [img], "image"

    # Fast pass: probability for every sampled frame (no Grad-CAM). Heatmap only for the top frame.
    frame_probs = [None] * len(frames)   # one entry per sampled frame, None = no face found
    faces = {}
    overlay = None
    branch_weights = None   # LFAM branch attribution (populated when Grad-CAM is called)
    with _lock:
        for i, frame in enumerate(frames):
            face, _ = crop_face_with_box(frame, size=det.input_size)
            if face is None:
                continue
            frame_probs[i] = det.predict_fast(face)
            faces[i] = face
        if faces:
            best = max(faces, key=lambda i: frame_probs[i])
            _, overlay, branch_weights = det.predict(faces[best])   # Grad-CAM + LFAM weights on most suspicious frame

    if not faces:
        return jsonify(error="No face found. Try a clearer, front-facing face."), 422

    probs = [p for p in frame_probs if p is not None]
    mean_prob = float(np.mean(probs))
    topk_prob = topk_mean(probs)            # score used for the decision (== the value for a single photo)
    raw_label = "fake" if topk_prob >= det.threshold else "real"
    uncertain = not det.is_validated or is_uncertain(topk_prob, det.threshold)
    label = "uncertain" if uncertain else raw_label

    # Format LFAM weights as rounded percentages for the UI (None if not available)
    lfam_weights = None
    if branch_weights:
        total = sum(branch_weights.values()) or 1.0
        lfam_weights = {k: round(v / total * 100, 1) for k, v in branch_weights.items()}

    return jsonify(
        kind=kind,
        label=label,                         # hard decision (unchanged field)
        fake_prob=topk_prob,                 # now the top-k score for videos (mean for a photo = same value)
        threshold=det.threshold,
        frames_used=len(probs),
        face=to_data_url(faces[best]),
        heatmap=to_data_url(overlay),
        # ---- new fields ----
        verdict="uncertain" if uncertain else label,
        uncertain=uncertain,
        model_verified=det.is_validated,
        verdict_reason=det.verdict_reason if not det.is_validated else None,
        uncertain_margin=UNCERTAIN_MARGIN,
        fake_prob_mean=mean_prob,
        fake_prob_topk=topk_prob,
        topk_fraction=TOPK_FRACTION,
        frames_sampled=len(frames),
        frame_probs=[None if p is None else round(p, 4) for p in frame_probs],
        # ---- LFAM attribution (new) ----
        branch_weights=lfam_weights,         # e.g. {"Spatial": 25.3, "FFT": 52.1, "SRM": 22.6}
        model_branch=det.branch,             # e.g. "fft+srm"
    )


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """Fast endpoint for the live camera: one frame in, face box + fake probability out."""
    f = request.files.get("file")
    if f is None:
        return jsonify(error="No frame."), 400
    try:
        det = get_detector()
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="Bad frame."), 400
    with _lock:
        face, box = crop_face_with_box(img, size=det.input_size)
        if face is None:
            return jsonify(face=False)
        prob = det.predict_fast(face)
        raw_label = "fake" if prob >= det.threshold else "real"
        uncertain = not det.is_validated or is_uncertain(prob, det.threshold)
        verdict = "uncertain" if uncertain else raw_label
    return jsonify(face=True, box=[int(v) for v in box], fake_prob=prob, threshold=det.threshold,
                   label=raw_label, verdict=verdict, uncertain=uncertain,
                   model_verified=det.is_validated,
                   verdict_reason=det.verdict_reason if not det.is_validated else None)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
