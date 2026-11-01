"""Comprehensive Test Suite for Real vs. Fake Face Detection.

Evaluates the detector across:
  - Test Group A: Genuine Images (mobile photos, human faces, selfies, portraits, live camera captures)
  - Test Group B: AI-Generated Images (GAN/Diffusion synthetic faces, frequency artifacts)
  - Test Group C: Manipulated Images (Deepfakes, Face swaps, blend boundaries)
  - Test Group D: Edge Cases and Robustness (JPEG compressed, blurry, varied lighting)

Generates full metric tables, accuracy per group, and failure case analysis.
"""
import json
import os
import sys
import time
from pathlib import Path

# Fix Windows encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from face_utils import crop_face
from inference import Detector, is_uncertain


def run_group(det, image_paths, expected_label, group_name):
    """Run detection on a list of images and return metrics and per-sample logs."""
    results = []
    correct_count = 0
    uncertain_count = 0
    
    print("\n" + "-" * 65)
    print(f"Running {group_name} ({len(image_paths)} samples, Expected: {expected_label.upper()})")
    print("-" * 65)
    
    for i, p in enumerate(image_paths):
        im = cv2.imread(str(p))
        if im is None:
            continue
            
        face = crop_face(im, size=det.input_size)
        if face is None:
            face = cv2.resize(im, (det.input_size, det.input_size))
            
        prob, _, weights = det.predict(face)
        
        # Verdict logic
        uncertain = is_uncertain(prob, det.threshold)
        raw_label = "fake" if prob >= det.threshold else "real"
        verdict = "uncertain" if uncertain else raw_label
        
        is_correct = (raw_label == expected_label)
        if is_correct:
            correct_count += 1
        if uncertain:
            uncertain_count += 1
            
        weights_str = ""
        if weights:
            weights_str = f" | LFAM: {', '.join(f'{k}={v*100:.0f}%' for k, v in weights.items())}"
            
        status = "[PASS]" if is_correct else "[FAIL]"
        if uncertain:
            status += " (UNCERTAIN)"
            
        print(f"[{i+1:2d}/{len(image_paths)}] {p.name[:25]:25s} | P(fake): {prob*100:5.1f}% | Verdict: {verdict.upper():9s} | {status}{weights_str}")
        
        results.append({
            "file": p.name,
            "path": str(p),
            "prob_fake": round(prob, 4),
            "threshold": round(det.threshold, 4),
            "raw_label": raw_label,
            "verdict": verdict,
            "expected": expected_label,
            "is_correct": is_correct,
            "is_uncertain": uncertain,
            "lfam_weights": weights,
        })
        
    n = len(results)
    acc = correct_count / max(n, 1)
    unc_rate = uncertain_count / max(n, 1)
    
    print("-" * 65)
    print(f"{group_name} Summary: Accuracy: {acc*100:.1f}% ({correct_count}/{n}) | Uncertain: {unc_rate*100:.1f}% ({uncertain_count}/{n})")
    
    return {
        "group_name": group_name,
        "expected": expected_label,
        "total": n,
        "correct": correct_count,
        "accuracy": round(acc, 4),
        "uncertain": uncertain_count,
        "uncertain_rate": round(unc_rate, 4),
        "samples": results,
    }


def main():
    print("=" * 65)
    print("EXECUTING COMPREHENSIVE TEST SUITE (Groups A, B, C, D)")
    print("=" * 65)
    
    det = Detector("checkpoints/best.pt")
    print(f"Loaded Detector:")
    print(f"  Branch:       {det.branch}")
    print(f"  Threshold:    {det.threshold:.4f}")
    print(f"  Temperature:  {det.temperature:.4f}")
    print(f"  Input size:   {det.input_size}x{det.input_size}")
    print(f"  Is validated: {det.is_validated}")
    
    # -- Group A: Genuine Images --
    group_a_paths = []
    # User's genuine webcam / selfie photos
    for p in Path("test_samples").glob("*real*"):
        group_a_paths.append(p)
    user_up = Path(r"C:\Users\rahul\.gemini\antigravity\brain\07027b44-7a58-4de6-a944-e9d3913eaa41\.user_uploaded\media_1793463397767_a02d8bce.jpg")
    if user_up.exists():
        group_a_paths.append(user_up)
    # Independent held-out test real images
    test_reals = sorted(Path("data_balanced/test/real").glob("*.png"))
    group_a_paths.extend(test_reals[:20])  # sample 20 test genuine images
    
    # -- Group B: AI-Generated Images --
    group_b_paths = []
    ai_art_fakes = sorted(Path("data_balanced/test/fake").glob("ai_artifact_*.png"))
    group_b_paths.extend(ai_art_fakes[:15])
    
    # -- Group C: Manipulated Images (Deepfakes & Face-Swaps) --
    group_c_paths = []
    for p in Path("test_samples").glob("*fake*"):
        group_c_paths.append(p)
    for p in Path("test_samples").glob("*swapped*"):
        group_c_paths.append(p)
    dfdc_swaps = [p for p in sorted(Path("data_balanced/test/fake").glob("*.png")) if not p.name.startswith("ai_")]
    group_c_paths.extend(dfdc_swaps[:20])
    
    # -- Group D: Edge Cases and Robustness --
    Path("test_samples/robustness").mkdir(parents=True, exist_ok=True)
    group_d_paths = []
    if group_a_paths:
        base = cv2.imread(str(group_a_paths[0]))
        if base is not None:
            # D1: Heavy JPEG compression (Q=25)
            _, enc = cv2.imencode('.jpg', base, [cv2.IMWRITE_JPEG_QUALITY, 25])
            d1_path = Path("test_samples/robustness/edge_heavy_jpeg.jpg")
            cv2.imwrite(str(d1_path), cv2.imdecode(enc, cv2.IMREAD_COLOR))
            group_d_paths.append(d1_path)
            
            # D2: Motion / Autofocus blur
            blurred = cv2.GaussianBlur(base, (7, 7), 2.5)
            d2_path = Path("test_samples/robustness/edge_blur.jpg")
            cv2.imwrite(str(d2_path), blurred)
            group_d_paths.append(d2_path)
            
            # D3: Low lighting / underexposed
            dark = np.clip(base.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
            d3_path = Path("test_samples/robustness/edge_low_light.jpg")
            cv2.imwrite(str(d3_path), dark)
            group_d_paths.append(d3_path)
            
            # D4: High exposure / glare
            glare = np.clip(base.astype(np.float32) * 1.45, 0, 255).astype(np.uint8)
            d4_path = Path("test_samples/robustness/edge_high_exposure.jpg")
            cv2.imwrite(str(d4_path), glare)
            group_d_paths.append(d4_path)
            
            # D5: Low resolution 64x64 upsampled
            lowres = cv2.resize(cv2.resize(base, (64, 64)), (base.shape[1], base.shape[0]))
            d5_path = Path("test_samples/robustness/edge_lowres_upsampled.jpg")
            cv2.imwrite(str(d5_path), lowres)
            group_d_paths.append(d5_path)
            
    # Run tests
    res_a = run_group(det, group_a_paths, "real", "Group A: Genuine Images")
    res_b = run_group(det, group_b_paths, "fake", "Group B: AI-Generated Images")
    res_c = run_group(det, group_c_paths, "fake", "Group C: Manipulated & Face-Swapped Images")
    res_d = run_group(det, group_d_paths, "real", "Group D: Robustness & Edge Cases")
    
    # Overall summary
    all_groups = [res_a, res_b, res_c, res_d]
    total_samples = sum(g["total"] for g in all_groups)
    total_correct = sum(g["correct"] for g in all_groups)
    overall_acc = total_correct / max(total_samples, 1)
    
    print("\n" + "=" * 65)
    print("OVERALL TEST SUITE SUMMARY")
    print("=" * 65)
    print(f"Group A (Genuine Images):         {res_a['accuracy']*100:.1f}% ({res_a['correct']}/{res_a['total']})")
    print(f"Group B (AI-Generated Images):    {res_b['accuracy']*100:.1f}% ({res_b['correct']}/{res_b['total']})")
    print(f"Group C (Manipulated / Swaps):    {res_c['accuracy']*100:.1f}% ({res_c['correct']}/{res_c['total']})")
    print(f"Group D (Edge Cases / Robust):    {res_d['accuracy']*100:.1f}% ({res_d['correct']}/{res_d['total']})")
    print("-" * 65)
    print(f"Overall Test Accuracy:            {overall_acc*100:.1f}% ({total_correct}/{total_samples})")
    print("=" * 65)
    
    # Save test report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_branch": det.branch,
        "threshold": det.threshold,
        "temperature": det.temperature,
        "input_size": det.input_size,
        "overall_accuracy": round(overall_acc, 4),
        "total_samples": total_samples,
        "total_correct": total_correct,
        "groups": {
            "group_a_genuine": res_a,
            "group_b_ai_generated": res_b,
            "group_c_manipulated": res_c,
            "group_d_robustness": res_d,
        }
    }
    with open("results/comprehensive_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nFull test report written to results/comprehensive_test_report.json")


if __name__ == "__main__":
    main()
