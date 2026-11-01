"""Split labeled face-frame folders by source-video ID to prevent frame leakage."""
import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png"}
SPLIT_BUCKETS = (("train", 0, 70), ("val", 70, 85), ("test", 85, 100))


def source_id(path):
    return re.sub(r"_\d+$", "", path.stem)


def split_for(class_name, video_id, seed):
    digest = hashlib.sha256(f"{seed}:{class_name}:{video_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    for split, start, end in SPLIT_BUCKETS:
        if start <= bucket < end:
            return split
    raise RuntimeError(f"No split assigned to bucket {bucket}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="input_kaggle_dataset")
    parser.add_argument("--out", default="data/kaggle_faces")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    grouped = {}
    for class_name in ("real", "fake"):
        folder = src / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected labeled folder: {folder}")
        videos = defaultdict(list)
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXT:
                videos[source_id(path)].append(path)
        if not videos:
            raise RuntimeError(f"No supported images found in {folder}")
        grouped[class_name] = videos

    assignments = {}
    for class_name, videos in grouped.items():
        counts = defaultdict(int)
        for video_id, paths in videos.items():
            split = split_for(class_name, video_id, args.seed)
            counts[split] += len(paths)
            dst = out / split / class_name
            dst.mkdir(parents=True, exist_ok=True)
            for path in paths:
                target = dst / path.name
                if target.exists():
                    raise FileExistsError(f"Refusing to overwrite existing image: {target}")
                try:
                    target.hardlink_to(path)
                except OSError:
                    shutil.copy2(path, target)
        assignments[class_name] = {
            "videos": {split: sum(
                split_for(class_name, video_id, args.seed) == split
                for video_id in videos
            ) for split, _, _ in SPLIT_BUCKETS},
            "images": dict(counts),
        }

    manifest = {
        "source": str(src),
        "seed": args.seed,
        "split_method": "source-video IDs; IDs ending in _<frame-number> share one split",
        "assignments": assignments,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
