"""
Arrange the Kaggle '140k Real and Fake Faces' image dataset (real_vs_fake/real-vs-fake)
into the layout train.py expects:   OUT/train|val|test / real|fake / *.jpg

Usage:
  python prepare_image_dataset.py --src "D:/datasets/real_vs_fake/real-vs-fake" --out data/faces140k --limit 5000
"""
import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm

EXT = {".jpg", ".jpeg", ".png"}
SPLITS = {"train": "train", "valid": "val", "test": "test"}  # source folder -> output folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing train/ valid/ test/")
    ap.add_argument("--out", default="data/faces140k")
    ap.add_argument("--limit", type=int, default=5000,
                    help="max images per class for TRAIN (val/test get limit//4). 0 = use all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    for src_name, out_name in SPLITS.items():
        for label in ["real", "fake"]:
            files = [p for p in (Path(args.src) / src_name / label).rglob("*") if p.suffix.lower() in EXT]
            if not files:
                print(f"[skip] nothing in {Path(args.src) / src_name / label}")
                continue
            random.shuffle(files)
            cap = args.limit if src_name == "train" else max(args.limit // 4, 1)
            if args.limit > 0:
                files = files[:cap]
            dst = Path(args.out) / out_name / label
            dst.mkdir(parents=True, exist_ok=True)
            for f in tqdm(files, desc=f"{out_name}/{label}"):
                shutil.copy2(f, dst / f.name)
    print("Done ->", args.out)


if __name__ == "__main__":
    main()
