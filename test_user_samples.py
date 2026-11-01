import os
import sys
import cv2
from inference import Detector
from face_utils import crop_face_with_box

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

det = Detector('checkpoints/best.pt', device='cpu')
print('Threshold:', det.threshold)
print('Input size:', det.input_size)
print('Is validated:', det.is_validated)

eval_samples = {
    '1. User Formal Passport Photo': 'test_samples/user_formal_thumb.jpg',
    '2. User Photo Clean (upload screenshot)': 'test_samples/user_photo_face_clean.png',
    '3. User Live Camera Face (webcam screenshot)': 'test_samples/user_cam_face_clean.png',
    '4. User Webcam Face (room lighting)': 'test_samples/user_real_face.jpg',
    '5. User Face-Swapped Fake (manipulated)': 'test_samples/user_swapped_face.jpg',
}

for label, p in eval_samples.items():
    if os.path.exists(p):
        img = cv2.imread(p)
        face, box = crop_face_with_box(img, size=det.input_size)
        if face is None:
            face = cv2.resize(img, (det.input_size, det.input_size))
        prob, overlay, weights = det.predict(face)
        fast_prob = det.predict_fast(face)
        verdict = 'fake' if prob >= det.threshold else 'real'
        conf = round((prob if verdict == 'fake' else 1 - prob) * 100)
        s_w = weights.get('Spatial', 0.0) * 100
        f_w = weights.get('FFT', 0.0) * 100
        n_w = weights.get('SRM', 0.0) * 100
        print(f'{label:<45}: P(fake)={prob:.4f} -> {verdict.upper()} ({conf}% conf) [Spatial: {s_w:.1f}%, FFT: {f_w:.1f}%, SRM: {n_w:.1f}%]')

