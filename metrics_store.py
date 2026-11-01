"""Saves accuracy numbers to results/metrics.json so the web app can show them."""
import json
from pathlib import Path

PATH = Path("results/metrics.json")


def update_metrics(key, data):
    PATH.parent.mkdir(exist_ok=True)
    current = json.loads(PATH.read_text()) if PATH.exists() else {}
    current[key] = data
    PATH.write_text(json.dumps(current, indent=2))


def equal_error_rate(labels, scores):
    """EER = error rate at the point where false-positive rate == false-negative rate (lower is better)."""
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))  # ROC point closest to FPR == FNR
    return float((fpr[i] + fnr[i]) / 2.0)
