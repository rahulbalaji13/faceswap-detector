import requests

# 1. Formal Passport Photo
with open('test_samples/user_formal_thumb.jpg', 'rb') as f:
    r = requests.post('http://127.0.0.1:5000/predict', files={'file': ('photo.png', f, 'image/jpeg')})
d = r.json()
p = d.get('fake_prob', 0)
print(f"Formal Passport Photo: Verdict={d.get('verdict')} | P(fake)={p*100:.1f}% | Confidence={(1-p)*100:.1f}% REAL | Verified={d.get('model_verified')}")

# 2. Webcam Real Photo
with open('test_samples/user_real_face.jpg', 'rb') as f:
    r = requests.post('http://127.0.0.1:5000/predict', files={'file': ('webcam.jpg', f, 'image/jpeg')})
d = r.json()
p = d.get('fake_prob', 0)
print(f"Webcam Real Photo:     Verdict={d.get('verdict')} | P(fake)={p*100:.1f}% | Confidence={(1-p)*100:.1f}% REAL | Verified={d.get('model_verified')}")

# 3. Swapped Fake Photo
with open('test_samples/user_swapped_face.jpg', 'rb') as f:
    r = requests.post('http://127.0.0.1:5000/predict', files={'file': ('swapped.jpg', f, 'image/jpeg')})
d = r.json()
p = d.get('fake_prob', 0)
print(f"Face Swapped Photo:    Verdict={d.get('verdict')} | P(fake)={p*100:.1f}% | Confidence={p*100:.1f}% FAKE | Verified={d.get('model_verified')}")
