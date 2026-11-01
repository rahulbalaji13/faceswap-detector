import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader

from dataset import FaceDataset, SyntheticBlendDataset
from metrics_store import update_metrics
from model import BRANCHES, FaceSwapNet, model_label, model_tag


def run_epoch(model, loader, device, criterion, optimizer=None, smooth=0.0, freeze_spatial=False):
    train = optimizer is not None
    model.train(train)
    if train and freeze_spatial:
        model.spatial.eval()
    total, probs, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device).float().unsqueeze(1)
        target = y * (1 - smooth) + 0.5 * smooth if train else y  # label smoothing
        with torch.set_grad_enabled(train):
            logit = model(x)
            loss = criterion(logit, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += loss.item() * x.size(0)
        probs += torch.sigmoid(logit).detach().cpu().flatten().tolist()
        labels += y.cpu().flatten().tolist()
    return total / len(labels), np.array(probs), np.array(labels)


def best_threshold(labels, probs):
    """Choose the validation threshold that maximizes balanced accuracy."""
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], np.asarray(probs))))
    best_t, best_score = 0.5, -1.0
    for threshold in candidates:
        score = balanced_accuracy_score(labels, probs >= threshold)
        if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_t - 0.5)):
            best_t, best_score = float(threshold), float(score)
    return best_t, accuracy_score(labels, probs >= best_t)


def safe_auc(labels, probs):
    return roc_auc_score(labels, probs) if len(set(labels.tolist())) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train/ and val/")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--patience", type=int, default=4, help="stop if val AUC does not improve this many epochs")
    ap.add_argument("--smooth", type=float, default=0.1, help="label smoothing")
    ap.add_argument("--branch", choices=BRANCHES, default="fft",
                    help="spatial | fft (original) | srm (noise residual) | fft+srm")
    ap.add_argument("--no-freq", action="store_true", help="same as --branch spatial (kept for old commands)")
    ap.add_argument("--no-lfam", action="store_true",
                    help="disable LFAM and use static concatenation (restores original fusion). "
                         "LFAM is ON by default for all multi-branch models.")
    ap.add_argument("--synthetic-ratio", type=float, default=0.0,
                    help="extra blended fakes made from real faces, as a fraction of the real FF++ fakes "
                         "(0 = off, 0.5 = +50%% more fakes). Validation never uses them.")
    ap.add_argument("--dual-view", choices=["auto", "on", "off"], default="auto",
                    help="FFT/SRM branches see a clean (un-JPEG'd, un-blurred) view. auto = on when --branch has srm")
    ap.add_argument("--image-size", type=int, choices=[128, 160, 224], default=224,
                    help="model input size; 128 is faster for CPU training on 128px image datasets")
    ap.add_argument("--freeze-spatial", action="store_true",
                    help="freeze the EfficientNet feature extractor and train only the classification head")
    ap.add_argument("--out", default="checkpoints/best.pt")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    branch = "spatial" if args.no_freq else args.branch
    use_lfam = not args.no_lfam
    dual = (args.dual_view == "on") or (args.dual_view == "auto" and "srm" in branch)
    print(f"Branch: {branch} | LFAM: {use_lfam} | dual-view: {dual} | synthetic ratio: {args.synthetic_ratio}")

    train_real = FaceDataset(Path(args.data) / "train", train=True, dual_view=dual, size=args.image_size)
    val_ds = FaceDataset(Path(args.data) / "val", train=False, size=args.image_size)
    counts = list(train_real.counts)
    train_ds = train_real
    if args.synthetic_ratio > 0:
        real_paths = [p for p, label in train_real.items if label == 0]
        n_syn = int(round(args.synthetic_ratio * train_real.counts[1]))
        if n_syn > 0:
            train_ds = ConcatDataset([
                train_real,
                SyntheticBlendDataset(real_paths, n_syn, dual_view=dual, size=args.image_size),
            ])
            counts[1] += n_syn
            print(f"Added {n_syn} synthetic blended fakes per epoch (made from {len(real_paths)} real faces)")
    print("train real/fake:", counts, "| val real/fake:", val_ds.counts)
    if any(count == 0 for count in val_ds.counts):
        raise RuntimeError("The validation split must contain both real and fake faces.")

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=args.workers)

    model = FaceSwapNet(branch=branch, use_lfam=use_lfam).to(device)
    if args.freeze_spatial:
        for parameter in model.spatial.parameters():
            parameter.requires_grad = False
    pos_weight = torch.tensor([counts[0] / max(counts[1], 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_auc, bad = -1.0, 0
    for ep in range(1, args.epochs + 1):
        tl, tp, tlab = run_epoch(
            model, train_dl, device, criterion, optimizer, args.smooth, args.freeze_spatial
        )
        vl, vp, vlab = run_epoch(model, val_dl, device, criterion)
        scheduler.step()
        t_auc, t_acc = safe_auc(tlab, tp), accuracy_score(tlab, tp > 0.5)
        v_auc, v_acc = safe_auc(vlab, vp), accuracy_score(vlab, vp > 0.5)
        thr, v_acc_thr = best_threshold(vlab, vp)
        v_balanced_acc = balanced_accuracy_score(vlab, vp >= thr)
        print(f"epoch {ep}/{args.epochs} | train loss {tl:.3f} auc {t_auc:.3f} acc {t_acc:.3f} "
              f"| val loss {vl:.3f} auc {v_auc:.3f} acc@0.5 {v_acc:.3f} "
              f"balanced_acc@{thr:.3f} {v_balanced_acc:.3f} acc {v_acc_thr:.3f}")
        if v_auc > best_auc:
            best_auc, bad = v_auc, 0
            torch.save({"model": model.state_dict(), "branch": branch, "use_freq": "fft" in branch,
                        "use_lfam": use_lfam, "synthetic_ratio": args.synthetic_ratio, "threshold": thr,
                        "input_size": args.image_size, "feature_extractor_frozen": args.freeze_spatial,
                        "validated_on_held_out": True, "validation_sample_count": len(vlab),
                        "deployment_ready": False,
                        "validation_class_counts": {"real": val_ds.counts[0], "fake": val_ds.counts[1]}}, args.out)
            update_metrics(f"{model_tag(branch, args.synthetic_ratio, use_lfam)}_validation", {
                "model": model_label(branch, args.synthetic_ratio, use_lfam),
                "dataset": "Validation split (same dataset as training)",
                "accuracy": round(float(v_acc_thr), 4), "auc": round(float(v_auc), 4),
                "balanced_accuracy": round(float(v_balanced_acc), 4),
                "f1": round(float(f1_score(vlab, vp >= thr)), 4),
                "threshold": round(thr, 2), "epoch": ep, "validated": True,
                "sample_count": len(vlab),
            })
            print("  saved best ->", args.out)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop: val AUC has not improved for {args.patience} epochs.")
                break


if __name__ == "__main__":
    main()
