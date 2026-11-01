"""Evaluate a checkpoint on any folder that has real/ and fake/ (use this for cross-dataset testing)."""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from dataset import FaceDataset
from inference import topk_mean
from metrics_store import equal_error_rate, update_metrics
from model import load_checkpoint, model_label, model_tag

MINIMUM_DEPLOYMENT_BALANCED_ACCURACY = 0.75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing real/ and fake/")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--name", default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--roc-suffix", default="", help="appended to the ROC file name (results/roc_<name><suffix>.png)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck, branch = load_checkpoint(args.ckpt, device)  # builds the right branch(es) automatically
    synth = ck.get("synthetic_ratio", 0.0)
    thr = ck.get("threshold", 0.5)  # chosen on the validation split, NOT tuned on this data

    dl = DataLoader(FaceDataset(args.data, size=ck.get("input_size", 224)), args.batch_size, num_workers=2)
    probs, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            probs += torch.sigmoid(model(x.to(device))).cpu().flatten().tolist()
            labels += y.tolist()

    preds = [int(p >= thr) for p in probs]
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    acc05 = accuracy_score(labels, [int(p > 0.5) for p in probs])
    # NEW metrics (precision/recall use the stored threshold, "fake" is the positive class)
    eer = equal_error_rate(labels, probs)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    grouped_probs, grouped_labels = {}, {}
    for (path, label), probability in zip(dl.dataset.items, probs):
        source = re.sub(r"_\d+$", "", path.stem)
        key = (int(label), source)
        grouped_probs.setdefault(key, []).append(probability)
        grouped_labels[key] = int(label)
    video_scores = np.asarray([topk_mean(scores) for scores in grouped_probs.values()])
    video_labels = np.asarray([grouped_labels[key] for key in grouped_probs])
    video_preds = video_scores >= thr
    video_acc = accuracy_score(video_labels, video_preds)
    video_balanced_acc = balanced_accuracy_score(video_labels, video_preds)
    video_auc = roc_auc_score(video_labels, video_scores)
    deployment_ready = min(balanced_acc, video_balanced_acc) >= MINIMUM_DEPLOYMENT_BALANCED_ACCURACY
    print(f"[{args.name}] AUC {auc:.4f} | Acc@{thr:.3f} {acc:.4f} | Balanced Acc {balanced_acc:.4f} "
          f"| Acc@0.5 {acc05:.4f} | F1 {f1:.4f}")
    print(f"[{args.name}] Source-video top-30%: clips {len(video_labels)} | AUC {video_auc:.4f} "
          f"| Acc {video_acc:.4f} | Balanced Acc {video_balanced_acc:.4f}")
    print(f"[{args.name}] EER {eer:.4f} | Precision {prec:.4f} | Recall {rec:.4f}  (at threshold {thr:.2f})")
    print("Confusion matrix (rows = true real/fake, cols = predicted real/fake):")
    print(confusion_matrix(labels, preds))

    update_metrics(f"{model_tag(branch, synth)}_{args.name}", {
        "model": model_label(branch, synth),
        "dataset": args.name,
        "accuracy": round(acc, 4), "auc": round(auc, 4), "f1": round(f1, 4), "threshold": round(thr, 2),
        "balanced_accuracy": round(balanced_acc, 4),
        "video_accuracy": round(float(video_acc), 4),
        "video_balanced_accuracy": round(float(video_balanced_acc), 4),
        "video_auc": round(float(video_auc), 4),
        "video_sample_count": len(video_labels),
        "deployment_ready": deployment_ready,
        "validated": True, "sample_count": len(labels),
        # added keys (the original ones above are unchanged)
        "eer": round(eer, 4), "precision": round(prec, 4), "recall": round(rec, 4),
    })
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    checkpoint["held_out_test_balanced_accuracy"] = float(balanced_acc)
    checkpoint["held_out_test_video_balanced_accuracy"] = float(video_balanced_acc)
    checkpoint["held_out_test_sample_count"] = len(labels)
    checkpoint["deployment_ready"] = deployment_ready
    torch.save(checkpoint, args.ckpt)
    if not deployment_ready:
        print(
            "Checkpoint is not enabled for real/fake verdicts: held-out balanced accuracy "
            f"{balanced_acc:.3f} (frames) and {video_balanced_acc:.3f} (source videos) must both reach "
            f"{MINIMUM_DEPLOYMENT_BALANCED_ACCURACY:.2f}."
        )

    # Save raw labels + probabilities so compare_models.py can draw all models on one ROC plot
    Path("results").mkdir(exist_ok=True)
    preds_file = f"results/preds_{args.name}{args.roc_suffix}.npz"
    np.savez(preds_file, labels=np.array(labels), probs=np.array(probs),
             label=np.array(model_label(branch, synth)), threshold=np.array(thr))
    print(f"Saved {preds_file}")

    fpr, tpr, _ = roc_curve(labels, probs)
    Path("results").mkdir(exist_ok=True)
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC - {args.name}")
    plt.tight_layout()
    roc_file = f"results/roc_{args.name}{args.roc_suffix}.png"
    plt.savefig(roc_file, dpi=150)
    print(f"Saved {roc_file}")


if __name__ == "__main__":
    main()
