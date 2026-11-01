"""
Turn videos into face crops, spread evenly across each video.

Input:   SRC/real/*.mp4  and  SRC/fake/*.mp4
Output:  OUT/train|val/real|fake   (split by VIDEO, so no video is in both train and val)
For a cross-dataset test set (e.g. Celeb-DF):  --force-split test
"""
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from face_utils import crop_face

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def pick_split(name, val_ratio):
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100
    return "val" if h < val_ratio * 100 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=20, help="face crops per video")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--png", action="store_true",
                    help="save lossless PNG crops (recommended for the SRM noise branch; uses more disk)")
    ap.add_argument("--force-split", choices=["train", "val", "test"], default=None)
    args = ap.parse_args()

    for label in ["real", "fake"]:
        videos = [p for p in (Path(args.src) / label).rglob("*") if p.suffix.lower() in VIDEO_EXT]
        for vp in tqdm(videos, desc=label):
            split = args.force_split or pick_split(vp.stem, args.val_ratio)
            out_dir = Path(args.out) / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(vp))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                continue
            # twice as many candidate positions as we need, in case some frames have no face
            positions = np.linspace(0, total - 1, args.max_frames * 2).astype(int)
            saved = 0
            for pos in positions:
                if saved >= args.max_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
                ok, frame = cap.read()
                if not ok:
                    continue
                face = crop_face(frame)
                if face is not None:
                    cv2.imwrite(str(out_dir / f"{vp.stem}_{int(pos):05d}.{'png' if args.png else 'jpg'}"), face)
                    saved += 1
            cap.release()


if __name__ == "__main__":
    main()
