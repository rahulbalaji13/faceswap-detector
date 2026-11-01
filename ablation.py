"""
Run the ablation study and print the results table for your report.

  python ablation.py                      # train + evaluate spatial, fft, srm, srm+synthetic
  python ablation.py --only-report        # just rebuild the table from results/metrics.json
  python ablation.py --configs spatial fft srm srm_synth fft_srm fft_srm_synth

Needs data/ffpp/{train,val} and data/celebdf/test (see README steps 3).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from model import model_tag

# key -> (branch, uses --synthetic-ratio?, row label)
CONFIGS = {
    "spatial":       ("spatial", False, "Spatial only"),
    "fft":           ("fft",     False, "Spatial + FFT (original)"),
    "srm":           ("srm",     False, "Spatial + SRM"),
    "srm_synth":     ("srm",     True,  "Spatial + SRM + synthetic fakes"),
    "fft_srm":       ("fft+srm", False, "Spatial + FFT + SRM"),
    "fft_srm_synth": ("fft+srm", True,  "Spatial + FFT + SRM + synthetic fakes"),
}
DEFAULT = ["spatial", "fft", "srm", "srm_synth"]


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT, choices=list(CONFIGS))
    ap.add_argument("--train-data", default="data/ffpp", help="folder with train/ and val/")
    ap.add_argument("--val-set", default="data/ffpp/val")
    ap.add_argument("--test-set", default="data/celebdf/test", help="cross-dataset test folder")
    ap.add_argument("--synthetic-ratio", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--skip-existing", action="store_true", help="don't retrain if the checkpoint already exists")
    ap.add_argument("--only-report", action="store_true", help="skip training/evaluation, only print the table")
    args = ap.parse_args()

    for key in args.configs:
        branch, uses_syn, _ = CONFIGS[key]
        ratio = args.synthetic_ratio if uses_syn else 0.0
        ckpt = f"checkpoints/ablation_{key}.pt"
        if not args.only_report:
            if not (args.skip_existing and Path(ckpt).exists()):
                run([sys.executable, "train.py", "--data", args.train_data, "--branch", branch,
                     "--synthetic-ratio", str(ratio), "--epochs", str(args.epochs), "--out", ckpt])
            for name, folder in (("ffpp_val", args.val_set), ("celebdf_test", args.test_set)):
                run([sys.executable, "evaluate.py", "--data", folder, "--ckpt", ckpt,
                     "--name", name, "--roc-suffix", f"_{key}"])

    path = Path("results/metrics.json")
    metrics = json.loads(path.read_text()) if path.exists() else {}
    lines = ["| Model | FF++ val AUC | Celeb-DF AUC |", "|---|---|---|"]
    for key in args.configs:
        branch, uses_syn, label = CONFIGS[key]
        tag = model_tag(branch, args.synthetic_ratio if uses_syn else 0.0)
        cells = []
        for name in ("ffpp_val", "celebdf_test"):
            m = metrics.get(f"{tag}_{name}")
            cells.append(f"{m['auc']:.4f}" if m else "-")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    table = "\n".join(lines)
    print("\n" + table)
    Path("results").mkdir(exist_ok=True)
    Path("results/ablation.md").write_text(table + "\n")
    print("\nSaved results/ablation.md")


if __name__ == "__main__":
    main()
