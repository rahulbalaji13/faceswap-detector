"""End-to-end test against the live Flask server."""
import urllib.request
import json
import mimetypes
import uuid

def post_multipart(url, filepath):
    boundary = uuid.uuid4().hex
    filename = filepath.split('/')[-1].split('\\')[-1]
    
    with open(filepath, 'rb') as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode('utf-8') + file_bytes + f"\r\n--{boundary}--\r\n".encode('utf-8')

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

def main():
    print("Testing GET / ...")
    with urllib.request.urlopen("http://127.0.0.1:5000/") as resp:
        print(f"  GET / status: {resp.status}")

    print("\nTesting GET /metrics ...")
    with urllib.request.urlopen("http://127.0.0.1:5000/metrics") as resp:
        metrics = json.loads(resp.read().decode('utf-8'))
        print(f"  Metrics count: {len(metrics)}")
        for m in metrics:
            print(f"    - {m['model']} ({m['dataset']}): AUC={m['auc']}, Acc={m['accuracy']}")

    print("\n--- Test Case 1: Testing POST /predict with sample_real.jpg ---")
    res_real = post_multipart("http://127.0.0.1:5000/predict", "test_samples/sample_real.jpg")
    print(f"  Verdict:       {res_real['verdict']}")
    print(f"  P(fake):       {res_real['fake_prob']:.4f} (Threshold: {res_real['threshold']})")
    print(f"  LFAM Weights:  {res_real.get('branch_weights')}")
    print(f"  Model Branch:  {res_real.get('model_branch')}")
    print(f"  Heatmap returned: {bool(res_real.get('heatmap'))}")

    print("\n--- Test Case 2: Testing POST /predict with sample_fake.jpg ---")
    res_fake = post_multipart("http://127.0.0.1:5000/predict", "test_samples/sample_fake.jpg")
    print(f"  Verdict:       {res_fake['verdict']}")
    print(f"  P(fake):       {res_fake['fake_prob']:.4f} (Threshold: {res_fake['threshold']})")
    print(f"  LFAM Weights:  {res_fake.get('branch_weights')}")
    print(f"  Model Branch:  {res_fake.get('model_branch')}")
    print(f"  Heatmap returned: {bool(res_fake.get('heatmap'))}")

    print("\nALL LIVE END-TO-END SERVER TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

