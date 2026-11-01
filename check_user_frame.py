import urllib.request
import json
import uuid

def test_frame(filepath):
    boundary = uuid.uuid4().hex
    with open(filepath, 'rb') as f:
        b = f.read()

    part1 = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="frame.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
    part2 = f'\r\n--{boundary}--\r\n'.encode()
    body = part1 + b + part2

    req = urllib.request.Request(
        'http://127.0.0.1:5000/predict_frame',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        prob = res['fake_prob']
        thr = res['threshold']
        verdict = "Likely swapped" if prob >= thr else "Likely real"
        return prob, thr, verdict

real_prob, thr, real_verdict = test_frame('test_samples/user_real_face.jpg')
print(f"User Real Face:    P(fake) = {real_prob*100:.1f}% -> Verdict: {real_verdict}")

swapped_prob, thr, swapped_verdict = test_frame('test_samples/user_swapped_face.jpg')
print(f"User Swapped Face: P(fake) = {swapped_prob*100:.1f}% -> Verdict: {swapped_verdict}")
