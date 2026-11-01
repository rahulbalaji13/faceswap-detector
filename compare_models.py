"""
Draw ONE ROC plot with several models on the same test set.

Needs the prediction files that evaluate.py saves (results/preds_<name><roc-suffix>.npz).
Typical use after `python ablation.py` (which evaluates every model with --roc-suffix _<key>):

  python compare_models.py --name celebdf_test
  python compare_models.py --files results/preds_celebdf_test_spatial.npz results/preds_celebdf_test_srm.npz
Output: results/roc_compare.png
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from metrics_store import equal_error_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="celebdf_test", help="evaluation name used in evaluate.py --name")
    ap.add_argument("--files", nargs="+", default=None, help="explicit .npz files (default: results/preds_<name>*.npz)")
    ap.add_argument("--out", default="results/roc_compare.png")
    args = ap.parse_args()

    files = args.files or sorted(glob.glob(f"results/preds_{args.name}*.npz"))
    if not files:
        raise SystemExit(f"No prediction files found (results/preds_{args.name}*.npz). Run evaluate.py first.")

    runs = []
    for f in files:
        d = np.load(f)
        labels, probs = d["labels"], d["probs"]
        label = str(d["label"]) if "label" in d.files else Path(f).stem
        runs.append((label, Path(f).stem, labels, probs))
    # if two files share a model name (e.g. two spatial runs), add the file name to tell them apart
    seen = [r[0] for r in runs]
    runs = [(f"{l} [{stem}]" if seen.count(l) > 1 else l, labels, probs) for l, stem, labels, probs in runs]

    plt.figure(figsize=(6, 5.5))
    print(f"{'Model':45s} {'AUC':>7s} {'EER':>7s}")
    for label, labels, probs in runs:
        auc = roc_auc_score(labels, probs)
        eer = equal_error_rate(labels, probs)
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC {auc:.3f})")
        print(f"{label:45s} {auc:7.4f} {eer:7.4f}")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC comparison - {args.name}")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    Path(args.out).parent.mkdir(exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
