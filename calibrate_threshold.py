"""Recalibrate a checkpoint threshold on its held-out validation split."""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from dataset import FaceDataset
from metrics_store import update_metrics
from model import load_checkpoint, model_label, model_tag
from train import best_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dataset root containing val/real and val/fake")
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint, branch = load_checkpoint(args.ckpt, device)
    validation = FaceDataset(
        Path(args.data) / "val",
        size=int(checkpoint.get("input_size", 224)),
    )
    if any(count == 0 for count in validation.counts):
        raise RuntimeError("Threshold calibration requires both real and fake validation images.")

    loader = DataLoader(validation, args.batch_size, num_workers=0)
    probabilities, labels = [], []
    with torch.no_grad():
        for images, target in loader:
            logits = model(images.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().flatten().tolist())
            labels.extend(target.tolist())

    threshold, _ = best_threshold(np.asarray(labels), np.asarray(probabilities))
    predictions = np.asarray(probabilities) >= threshold
    balanced_acc = balanced_accuracy_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    checkpoint["threshold"] = threshold
    checkpoint["validated_on_held_out"] = True
    checkpoint["deployment_ready"] = False
    checkpoint["validation_sample_count"] = len(labels)
    checkpoint["validation_class_counts"] = {
        "real": validation.counts[0],
        "fake": validation.counts[1],
    }
    torch.save(checkpoint, args.ckpt)

    synthetic_ratio = checkpoint.get("synthetic_ratio", 0.0)
    update_metrics(f"{model_tag(branch, synthetic_ratio)}_validation", {
        "model": model_label(branch, synthetic_ratio),
        "dataset": "Held-out validation split (grouped by source video)",
        "accuracy": round(float(accuracy), 4),
        "balanced_accuracy": round(float(balanced_acc), 4),
        "auc": round(float(roc_auc_score(labels, probabilities)), 4),
        "f1": round(float(f1_score(labels, predictions)), 4),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "threshold": round(float(threshold), 6),
        "validated": True,
        "sample_count": len(labels),
    })
    print(
        f"Validation threshold={threshold:.4f} | accuracy={accuracy:.4f} "
        f"| balanced accuracy={balanced_acc:.4f} | samples={len(labels)} "
        f"(real={validation.counts[0]}, fake={validation.counts[1]})"
    )


if __name__ == "__main__":
    main()
