"""
Robustness test: how much does AUC drop when the test images are degraded?

  python robustness.py --data data/celebdf/test --ckpt checkpoints/best.pt --name celebdf_test
  python robustness.py --data data/celebdf/test --name celebdf_test \
      --ckpt checkpoints/ablation_spatial.pt checkpoints/ablation_fft.pt checkpoints/ablation_srm.pt

For every corruption the test images are degraded ONCE and fed to ALL models, so every model sees
exactly the same pixels. AUC is threshold-free, so the stored thresholds are never touched.
Outputs: results/robustness_<name>.csv and results/robustness_<name>.png
"""
import argparse
import csv
import io
import zlib
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from dataset import FaceDataset, get_transform
from model import load_checkpoint, model_label


# ----------------------------------------------------------------------------
# Corruptions. Each takes an RGB PIL image (+ a numpy RNG for the noisy ones) and returns an RGB PIL image.
# ----------------------------------------------------------------------------
def c_none(img, rng):
    return img


def make_jpeg(quality):
    def f(img, rng):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return f


def make_blur(k):
    def f(img, rng):
        return Image.fromarray(cv2.GaussianBlur(np.array(img), (k, k), 0))  # sigma derived from kernel size
    return f


def make_down_up(scale):
    """Shrink then enlarge back: simulates a low-resolution / re-scaled video."""
    def f(img, rng):
        a = np.array(img)
        h, w = a.shape[:2]
        small = cv2.resize(a, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        return Image.fromarray(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR))
    return f


def make_noise(sigma):
    def f(img, rng):
        a = np.array(img).astype(np.float32)
        a += rng.normal(0.0, sigma, a.shape)  # sigma in 0..255 pixel units
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return f


CORRUPTIONS = {
    "none": c_none,
    "jpeg90": make_jpeg(90), "jpeg70": make_jpeg(70), "jpeg50": make_jpeg(50),
    "blur3": make_blur(3), "blur5": make_blur(5),
    "down0.5": make_down_up(0.5), "down0.25": make_down_up(0.25),
    "noise5": make_noise(5), "noise10": make_noise(10),
}


class CorruptedFaces(Dataset):
    """Same images/labels as FaceDataset, but corrupted before the normal eval transform."""
    def __init__(self, items, corruption, seed=0):
        self.items, self.fn, self.seed = items, CORRUPTIONS[corruption], seed
        self.tf = get_transform(False)
        # stable per-corruption offset so noise differs between corruption types but is reproducible
        self.offset = zlib.crc32(corruption.encode()) % 100000

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        rng = np.random.default_rng(self.seed * 1_000_003 + self.offset * 1_009 + i)  # same noise for every model
        img = self.fn(Image.open(path).convert("RGB"), rng)
        return self.tf(img), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing real/ and fake/ (e.g. data/celebdf/test)")
    ap.add_argument("--ckpt", nargs="+", default=["checkpoints/best.pt"], help="one or more checkpoints to compare")
    ap.add_argument("--name", default="test", help="used in the output file names")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0, help="seed for the noise corruptions")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load every checkpoint; build readable, unique column names
    models, names = [], []
    for path in args.ckpt:
        model, ck, branch = load_checkpoint(path, device)
        label = model_label(branch, ck.get("synthetic_ratio", 0.0))
        if label in names:  # two checkpoints of the same type -> tell them apart by file name
            label = f"{label} ({Path(path).stem})"
        models.append(model)
        names.append(label)
        print(f"Loaded {path} -> {label}")

    items = FaceDataset(args.data).items
    print(f"{len(items)} test images from {args.data}\n")

    auc = {n: {} for n in names}  # auc[model][corruption]
    for cname in CORRUPTIONS:
        dl = DataLoader(CorruptedFaces(items, cname, args.seed), args.batch_size, num_workers=args.workers)
        probs, labels = [[] for _ in models], []
        with torch.no_grad():
            for x, y in dl:
                x = x.to(device)
                for k, m in enumerate(models):  # the SAME degraded batch goes through every model
                    probs[k] += torch.sigmoid(m(x)).cpu().flatten().tolist()
                labels += y.tolist()
        for k, n in enumerate(names):
            auc[n][cname] = roc_auc_score(labels, probs[k]) if len(set(labels)) > 1 else float("nan")
        print(f"  {cname:9s} " + " | ".join(f"{auc[n][cname]:.4f}" for n in names), flush=True)

    # ---- table ----
    width = max(len(n) for n in names + ["Corruption"]) + 2
    print("\n" + "Corruption".ljust(12) + "".join(n.ljust(width) for n in names))
    for cname in CORRUPTIONS:
        print(cname.ljust(12) + "".join(f"{auc[n][cname]:.4f}".ljust(width) for n in names))

    # ---- csv ----
    Path("results").mkdir(exist_ok=True)
    csv_path = f"results/robustness_{args.name}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["corruption"] + names)
        for cname in CORRUPTIONS:
            w.writerow([cname] + [f"{auc[n][cname]:.4f}" for n in names])

    # ---- plot: grouped bars, one group per corruption ----
    vals = np.array([[auc[n][c] for c in CORRUPTIONS] for n in names])
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(CORRUPTIONS) + 2), 4.5))
    bw = 0.8 / len(names)
    xs = np.arange(len(CORRUPTIONS))
    for k, n in enumerate(names):
        ax.bar(xs + (k - (len(names) - 1) / 2) * bw, vals[k], bw, label=n)
    ax.axhline(0.5, ls="--", color="grey", lw=1)  # chance level
    lo = max(0.0, min(0.5, float(np.nanmin(vals))) - 0.05)
    ax.set_ylim(lo, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(CORRUPTIONS), rotation=30, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title(f"Robustness to image degradation - {args.name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png_path = f"results/robustness_{args.name}.png"
    fig.savefig(png_path, dpi=150)
    print(f"\nSaved {csv_path}\nSaved {png_path}")


if __name__ == "__main__":
    main()
