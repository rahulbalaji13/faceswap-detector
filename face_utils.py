"""Face detection + cropping. Uses MTCNN if installed, else OpenCV Haar cascade."""
import cv2

_detector, _kind = None, None


def _init():
    global _detector, _kind
    if _detector is not None:
        return
    try:
        import torch
        from facenet_pytorch import MTCNN
        _detector = MTCNN(keep_all=False, device="cuda" if torch.cuda.is_available() else "cpu")
        _kind = "mtcnn"
    except Exception:
        try:
            _detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            _kind = "haar"
        except Exception:
            _detector = None
            _kind = "fallback"


def detect_box(bgr):
    _init()
    if _kind == "mtcnn":
        from PIL import Image
        boxes, _ = _detector.detect(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0]
            return int(x1), int(y1), int(x2), int(y2)
    elif _kind == "haar" and _detector is not None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = _detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return int(x), int(y), int(x + w), int(y + h)

    return None


def crop_face_with_box(bgr, size=224, margin=0.3):
    """Return (size x size BGR crop, (x1,y1,x2,y2) box in the original image) or (None, None)."""
    box = detect_box(bgr)
    if box is None:
        return None, None
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = bgr.shape[:2]
    x1, y1 = max(0, int(x1 - margin * w)), max(0, int(y1 - margin * h))
    x2, y2 = min(W, int(x2 + margin * w)), min(H, int(y2 + margin * h))
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    return cv2.resize(crop, (size, size)), (x1, y1, x2, y2)


def crop_face(bgr, size=224, margin=0.3):
    """Face crop with margin (so the blend edge is included), or None."""
    return crop_face_with_box(bgr, size, margin)[0]
