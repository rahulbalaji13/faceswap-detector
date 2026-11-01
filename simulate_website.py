"""
End-to-end Simulation of the Real vs Fake Face Detector Web Application.
Simulates both:
1. File upload pipeline (/predict)
2. Live camera streaming pipeline (/predict_frame)
3. Model accuracy metrics endpoint (/metrics)
"""
import sys
import os
import requests
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_URL = "http://127.0.0.1:5000"

print("=" * 70)
print("SIMULATING FACESWAP-DETECTOR WEB APPLICATION AT", BASE_URL)
print("=" * 70)

# 1. Test GET /
try:
    res = requests.get(BASE_URL, timeout=5)
    assert res.status_code == 200
    assert "Face authenticity demo" in res.text or "Face swap check" in res.text
    print("[PASS] 1. Website Homepage (GET /) loaded successfully (Status 200).")
except Exception as e:
    print(f"[FAIL] 1. Website Homepage failed: {e}")
    sys.exit(1)

# 2. Test GET /metrics
try:
    res = requests.get(f"{BASE_URL}/metrics", timeout=5)
    assert res.status_code == 200
    metrics = res.json()
    assert len(metrics) > 0
    top = metrics[0]
    print(f"[PASS] 2. Metrics Endpoint (GET /metrics):")
    print(f"        Active Model: {top.get('model')}")
    print(f"        Accuracy:     {top.get('accuracy')*100:.1f}%")
    print(f"        Bal Accuracy: {top.get('balanced_accuracy')*100:.1f}%")
    print(f"        ROC-AUC:      {top.get('auc'):.3f}")
    print(f"        Threshold:    {top.get('threshold')}")
except Exception as e:
    print(f"[FAIL] 2. Metrics Endpoint failed: {e}")
    sys.exit(1)

# 3. Test POST /predict (Image Upload Pipeline)
print("\n" + "-" * 70)
print("3. TESTING IMAGE UPLOAD PIPELINE (POST /predict):")
print("-" * 70)

upload_tests = [
    {
        "name": "User Genuine Passport Photo",
        "file": "test_samples/user_formal_thumb.jpg",
        "expected_verdict": "real"
    },
    {
        "name": "User Photo Clean (from upload screenshot)",
        "file": "test_samples/user_photo_face_clean.png",
        "expected_verdict": "real"
    },
    {
        "name": "User Face-Swapped Deepfake",
        "file": "test_samples/user_swapped_face.jpg",
        "expected_verdict": "fake"
    }
]

for t in upload_tests:
    if not os.path.exists(t["file"]):
        print(f"[SKIP] {t['name']}: {t['file']} not found")
        continue
    with open(t["file"], "rb") as fh:
        res = requests.post(f"{BASE_URL}/predict", files={"file": fh}, timeout=10)
    assert res.status_code == 200, f"Predict failed: {res.text}"
    d = res.json()
    v = d.get("verdict")
    p = d.get("fake_prob", 0.0)
    w = d.get("branch_weights", {})
    conf = round((p if v == 'fake' else 1 - p) * 100)
    status = "PASS" if v == t["expected_verdict"] else "FAIL"
    print(f"[{status}] {t['name']:<42}: Predicted {v.upper()} ({conf}% conf, P(fake)={p:.4f})")
    print(f"        LFAM Attention Routing: Spatial={w.get('Spatial')}%, FFT={w.get('FFT')}%, SRM={w.get('SRM')}%")
    print(f"        Heatmap Generated: {'Yes (Data URL)' if 'heatmap' in d else 'No'}")
    assert v == t["expected_verdict"], f"Expected {t['expected_verdict']} but got {v}"

# 4. Test POST /predict_frame (Live Camera Pipeline)
print("\n" + "-" * 70)
print("4. TESTING LIVE CAMERA PIPELINE (POST /predict_frame):")
print("-" * 70)

camera_tests = [
    {
        "name": "Live Webcam Face (from latest user webcam screenshot)",
        "file": "test_samples/user_cam_face_clean.png",
        "expected_verdict": "real"
    },
    {
        "name": "Live Webcam Face (room lighting & glasses)",
        "file": "test_samples/user_real_face.jpg",
        "expected_verdict": "real"
    },
    {
        "name": "Live Camera Presentation Attack (swapped face on camera)",
        "file": "test_samples/user_swapped_face.jpg",
        "expected_verdict": "fake"
    }
]

for t in camera_tests:
    if not os.path.exists(t["file"]):
        print(f"[SKIP] {t['name']}: {t['file']} not found")
        continue
    # Simulate consecutive streaming frames
    readings = []
    for _ in range(5):
        with open(t["file"], "rb") as fh:
            res = requests.post(f"{BASE_URL}/predict_frame", files={"file": fh}, timeout=5)
        assert res.status_code == 200, f"Predict frame failed: {res.text}"
        d = res.json()
        readings.append(d["fake_prob"])
    avg_p = sum(readings) / len(readings)
    thr = d.get("threshold", 0.5)
    fake = avg_p >= thr
    v = "fake" if fake else "real"
    conf = round((avg_p if fake else 1 - avg_p) * 100)
    status = "PASS" if v == t["expected_verdict"] else "FAIL"
    print(f"[{status}] {t['name']:<42}: Live Smoothed Verdict = {v.upper()} ({conf}% conf, avg P(fake)={avg_p:.4f})")
    print(f"        Face Detected: {d.get('face')}, Bounding Box: {d.get('box')}")
    assert v == t["expected_verdict"], f"Expected {t['expected_verdict']} but got {v}"

print("\n" + "=" * 70)
print("ALL END-TO-END WEBSITE SIMULATION TESTS PASSED PERFECTLY!")
print("=" * 70)

