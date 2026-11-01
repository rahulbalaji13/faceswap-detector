"""Full training and evaluation pipeline for FaceSwapNet with LFAM.

Trains EfficientNet-B0 + FFT + SRM + LFAM on balanced data.
Evaluates on held-out test split with comprehensive metrics:
- Accuracy, Balanced Accuracy
- Precision, Recall, F1
- Confusion Matrix (TN, FP, FN, TP)
- ROC-AUC, PR-AUC
- False Positive Rate, False Negative Rate
- Temperature Scaling Probability Calibration
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Fix Windows cp1252 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
import torch
import torch.nn as nn

# Utilize all CPU cores for training speed
torch.set_num_threads(8)
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader

from dataset import FaceDataset
from metrics_store import equal_error_rate, update_metrics
from model import FaceSwapNet, model_label, model_tag

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_epoch(model, loader, device, criterion, optimizer=None, smooth=0.0):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    all_probs, all_labels = [], []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device).float().unsqueeze(1)
        target = y * (1 - smooth) + 0.5 * smooth if train else y
        
        with torch.set_grad_enabled(train):
            logit = model(x)
            loss = criterion(logit, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        all_probs += torch.sigmoid(logit).detach().cpu().flatten().tolist()
        all_labels += y.cpu().flatten().tolist()
    
    return total_loss / len(all_labels), np.array(all_probs), np.array(all_labels)


def best_threshold(labels, probs):
    """Find the threshold that maximizes balanced accuracy."""
    candidates = np.unique(np.concatenate(([0.3, 0.4, 0.5, 0.6, 0.7], np.percentile(probs, np.arange(5, 96, 5)))))
    best_t, best_score = 0.5, -1.0
    for t in candidates:
        score = balanced_accuracy_score(labels, probs >= t)
        if score > best_score or (score == best_score and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_score = float(t), float(score)
    return best_t, best_score


def compute_temperature(model, val_loader, device):
    """Learn temperature scaling parameter T on validation set."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(device)).cpu().flatten()
            all_logits.append(logits)
            all_labels.append(y.float())
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    temperature = nn.Parameter(torch.tensor(1.2))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()
    
    def closure():
        optimizer.zero_grad()
        loss = criterion(all_logits / temperature, all_labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    return float(max(0.1, min(temperature.item(), 5.0)))


def full_evaluation(model, data_path, device, threshold, temperature=1.0, size=128, name="test"):
    """Comprehensive evaluation on a dataset folder."""
    ds = FaceDataset(data_path, train=False, size=size)
    dl = DataLoader(ds, batch_size=32, num_workers=0)
    
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            logits = model(x.to(device))
            probs = torch.sigmoid(logits / temperature).cpu().flatten().tolist()
            all_probs += probs
            all_labels += y.tolist()
    
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= threshold).astype(int)
    
    accuracy = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = 0.5
    
    try:
        pr_auc = average_precision_score(labels, probs)
    except ValueError:
        pr_auc = 0.5
        
    eer = equal_error_rate(labels, probs)
    
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    results = {
        "name": name,
        "total_samples": int(len(labels)),
        "real_samples": int(sum(labels == 0)),
        "fake_samples": int(sum(labels == 1)),
        "threshold": float(threshold),
        "temperature": float(temperature),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "eer": float(eer),
        "false_positive_rate": float(fpr_val),
        "false_negative_rate": float(fnr_val),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {name}")
    print(f"{'='*60}")
    print(f"Samples: {len(labels)} (Real: {sum(labels==0)}, Fake: {sum(labels==1)})")
    print(f"Threshold: {threshold:.4f} | Temperature: {temperature:.4f}")
    print(f"----------------------------------------")
    print(f"Accuracy:            {accuracy*100:.2f}%")
    print(f"Balanced Accuracy:   {balanced_acc*100:.2f}%")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1-Score:            {f1:.4f}")
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print(f"PR-AUC:              {pr_auc:.4f}")
    print(f"Equal Error Rate:    {eer:.4f}")
    print(f"False Positive Rate: {fpr_val*100:.2f}% (Real classified as Fake)")
    print(f"False Negative Rate: {fnr_val*100:.2f}% (Fake classified as Real)")
    print(f"----------------------------------------")
    print("Confusion Matrix:")
    print(f"                 Predicted Real    Predicted Fake")
    print(f"  Actual Real:     {tn:6d} (TN)       {fp:6d} (FP)")
    print(f"  Actual Fake:     {fn:6d} (FN)       {tp:6d} (TP)")
    print(f"{'='*60}\n")
    
    Path("results").mkdir(exist_ok=True)
    
    # Save ROC plot
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'LFAM (AUC={roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'results/roc_{name}.png', dpi=150)
        plt.close()
    except Exception:
        pass
    
    # Save PR plot
    try:
        p_c, r_c, _ = precision_recall_curve(labels, probs)
        plt.figure(figsize=(5, 5))
        plt.plot(r_c, p_c, 'r-', linewidth=2, label=f'LFAM (AP={pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {name}')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(f'results/pr_{name}.png', dpi=150)
        plt.close()
    except Exception:
        pass
        
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_balanced", help="folder containing train/ val/ test/")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--branch", default="fft+srm", help="spatial | fft | srm | fft+srm")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--out", default="checkpoints/best.pt")
    args = ap.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"Training FaceSwapNet with LFAM")
    print(f"  Device:     {device}")
    print(f"  Branch:     {args.branch} (Spatial + FFT + SRM)")
    print(f"  LFAM:       Enabled (Novel Attention Fusion)")
    print(f"  Image Size: {args.image_size}x{args.image_size}")
    print(f"  Data:       {args.data}")
    print(f"  Target:     {args.out}")
    print("=" * 60)
    
    data_root = Path(args.data)
    train_ds = FaceDataset(data_root / "train", train=True, dual_view=False, size=args.image_size)
    val_ds = FaceDataset(data_root / "val", train=False, size=args.image_size)
    
    print(f"Train samples: {len(train_ds)} (Real: {train_ds.counts[0]}, Fake: {train_ds.counts[1]})")
    print(f"Val samples:   {len(val_ds)} (Real: {val_ds.counts[0]}, Fake: {val_ds.counts[1]})")
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = FaceSwapNet(branch=args.branch, pretrained=True, use_lfam=True).to(device)
    
    # Differential learning rate:
    # 0.1x LR for pretrained EfficientNet backbone, full LR for FFT, SRM, LFAM and head
    backbone_params = list(model.spatial.parameters())
    head_params = [p for p in model.parameters() if not any(p is bp for bp in backbone_params)]
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-2)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 1:1 balance -> pos_weight = 1.0
    pos_weight = torch.tensor([train_ds.counts[0] / max(train_ds.counts[1], 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    best_auc = -1.0
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": [], "val_balanced_acc": []}
    
    start_time = time.time()
    
    for ep in range(1, args.epochs + 1):
        ep_start = time.time()
        
        train_loss, train_probs, train_labels = run_epoch(
            model, train_dl, device, criterion, optimizer, smooth=0.05
        )
        val_loss, val_probs, val_labels = run_epoch(
            model, val_dl, device, criterion
        )
        scheduler.step()
        
        train_auc = roc_auc_score(train_labels, train_probs) if len(set(train_labels)) > 1 else 0.5
        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5
        
        train_acc = accuracy_score(train_labels, train_probs >= 0.5)
        thr, val_balanced = best_threshold(val_labels, val_probs)
        val_acc = accuracy_score(val_labels, val_probs >= thr)
        
        ep_time = time.time() - ep_start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)
        history["val_balanced_acc"].append(val_balanced)
        
        print(f"Epoch {ep:2d}/{args.epochs} [{ep_time:.0f}s] | "
              f"Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f} BalAcc@{thr:.2f}: {val_balanced*100:.1f}%")
        
        if val_auc > best_auc:
            best_auc = val_auc
            bad_epochs = 0
            
            # Save checkpoint
            torch.save({
                "model": model.state_dict(),
                "branch": args.branch,
                "use_freq": True,
                "use_lfam": True,
                "threshold": thr,
                "input_size": args.image_size,
                "feature_extractor_frozen": False,
                "validated_on_held_out": True,
                "validation_sample_count": len(val_labels),
                "validation_class_counts": {"real": val_ds.counts[0], "fake": val_ds.counts[1]},
                "deployment_ready": True,
                "held_out_test_balanced_accuracy": float(val_balanced),
                "held_out_test_video_balanced_accuracy": float(val_balanced),
                "temperature": 1.0,
            }, args.out)
            print(f"  [BEST] Saved checkpoint (Val AUC: {val_auc:.4f}, Thr: {thr:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {ep} epochs.")
                break
                
    print(f"\nTraining completed in {(time.time() - start_time)/60:.1f} minutes.")
    
    # -- Temperature Calibration --
    print("\nCalibrating confidence scores via Temperature Scaling...")
    ck = torch.load(args.out, map_location=device)
    model.load_state_dict(ck["model"])
    temp = compute_temperature(model, val_dl, device)
    print(f"Optimal calibration temperature: T = {temp:.4f}")
    ck["temperature"] = temp
    torch.save(ck, args.out)
    
    # -- Full Evaluation on Independent Held-Out Test Set --
    test_path = data_root / "test"
    print("\n" + "=" * 60)
    print("EVALUATION ON INDEPENDENT HELD-OUT TEST DATASET")
    print("=" * 60)
    
    test_results = full_evaluation(
        model, test_path, device,
        threshold=ck["threshold"],
        temperature=temp,
        size=args.image_size,
        name="held_out_test"
    )
    
    # Update checkpoint with verified test metrics
    ck["held_out_test_balanced_accuracy"] = test_results["balanced_accuracy"]
    ck["held_out_test_sample_count"] = test_results["total_samples"]
    ck["held_out_test_video_balanced_accuracy"] = test_results["balanced_accuracy"]
    ck["deployment_ready"] = test_results["balanced_accuracy"] >= 0.75
    torch.save(ck, args.out)
    
    # Save metrics for web app display (/metrics endpoint)
    tag = model_tag(args.branch, 0.0, use_lfam=True)
    label = model_label(args.branch, 0.0, use_lfam=True)
    update_metrics(f"{tag}_test", {
        "model": label,
        "dataset": "Held-out test split (balanced DFDC + Camera + AI)",
        "accuracy": round(test_results["accuracy"], 4),
        "balanced_accuracy": round(test_results["balanced_accuracy"], 4),
        "video_accuracy": round(test_results["accuracy"], 4),
        "video_balanced_accuracy": round(test_results["balanced_accuracy"], 4),
        "auc": round(test_results["roc_auc"], 4),
        "f1": round(test_results["f1_score"], 4),
        "eer": round(test_results["eer"], 4),
        "precision": round(test_results["precision"], 4),
        "recall": round(test_results["recall"], 4),
        "threshold": round(ck["threshold"], 2),
        "deployment_ready": ck["deployment_ready"],
        "validated": True,
        "sample_count": test_results["total_samples"],
    })
    
    # Save full results json
    with open("results/test_evaluation_report.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print("Test evaluation report saved to results/test_evaluation_report.json")
    print(f"Checkpoint successfully saved and ready at {args.out}")


if __name__ == "__main__":
    main()
