"""Prediction + Grad-CAM heatmap + LFAM branch-weight attribution."""
import cv2
import numpy as np
import torch

from model import BRANCH_NAMES, load_checkpoint

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TOPK_FRACTION = 0.3      # video score = mean of the most suspicious 30% of frames
UNCERTAIN_MARGIN = 0.1   # score within +/- 0.1 of the threshold -> "uncertain"
MINIMUM_DEPLOYMENT_BALANCED_ACCURACY = 0.60


def topk_mean(probs, frac=TOPK_FRACTION):
    """Mean of the highest `frac` share of per-frame fake probabilities (always at least 1 frame).

    A plain mean lets many clean-looking frames dilute a few clearly swapped ones; top-k does not.
    """
    p = np.sort(np.asarray(probs, dtype=np.float64))[::-1]
    k = max(1, int(np.ceil(frac * len(p) - 1e-9)))
    return float(p[:k].mean())


def is_uncertain(score, threshold, margin=UNCERTAIN_MARGIN):
    """True when the score is too close to the decision threshold to call real/fake with confidence."""
    return abs(float(score) - float(threshold)) <= margin


def to_tensor(bgr_face, size=224):
    if bgr_face.shape[:2] != (size, size):
        bgr_face = cv2.resize(bgr_face, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


class GradCAM:
    def __init__(self, layer):
        self.acts, self.grads = None, None
        layer.register_forward_hook(self._forward)

    def _forward(self, module, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def compute(self):
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.acts).sum(dim=1))[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


class Detector:
    def __init__(self, ckpt_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, ck, self.branch = load_checkpoint(ckpt_path, self.device)  # spatial | fft | srm | fft+srm
        self.threshold = float(ck.get("threshold", 0.5))
        self.temperature = float(ck.get("temperature", 1.0))
        self.input_size = int(ck.get("input_size", 224))
        self.held_out_test_balanced_accuracy = float(ck.get("held_out_test_balanced_accuracy", 0.0))
        self.held_out_test_video_balanced_accuracy = float(
            ck.get("held_out_test_video_balanced_accuracy", 0.0)
        )
        counts = ck.get("validation_class_counts", {})
        self.is_validated = (
            ck.get("validated_on_held_out") is True
            and isinstance(counts, dict)
            and counts.get("real", 0) > 0
            and counts.get("fake", 0) > 0
            and ck.get("validation_sample_count", 0) >= 2
            and ck.get("deployment_ready") is True
            and self.held_out_test_balanced_accuracy >= MINIMUM_DEPLOYMENT_BALANCED_ACCURACY
            and self.held_out_test_video_balanced_accuracy >= MINIMUM_DEPLOYMENT_BALANCED_ACCURACY
        )
        self.verdict_reason = (
            "This checkpoint has no recorded held-out validation on real and fake faces."
            if not ck.get("validated_on_held_out")
            else "This model did not meet the minimum balanced accuracy on held-out test videos."
        )
        self.cam = GradCAM(self.model.spatial.blocks[-1])
        # Human-readable names for each active branch (used in LFAM weight dicts)
        self._branch_names: list[str] = BRANCH_NAMES.get(self.branch, ["Spatial"])

    def _lfam_weights(self) -> dict[str, float] | None:
        """Return LFAM branch weights as a named dict, or None if LFAM is not active."""
        w = getattr(self.model, "_last_weights", None)
        if w is None or not getattr(self.model, "use_lfam", False):
            return None
        vals = w[0].cpu().numpy()          # shape: (n_branches,)
        return {name: float(vals[i]) for i, name in enumerate(self._branch_names)}

    def predict(self, bgr_face):
        """Returns (P(fake), overlay_image, branch_weights_or_None).

        branch_weights is a dict like {"Spatial": 0.25, "FFT": 0.52, "SRM": 0.23}
        when LFAM is active, otherwise None.
        Heatmap highlights the spatial regions that pushed the score toward 'fake'.
        """
        x = to_tensor(bgr_face, self.input_size).to(self.device)
        self.model.zero_grad()
        logit = self.model(x)
        prob = torch.sigmoid(logit / self.temperature)[0, 0].item()
        logit[0, 0].backward()
        heat = cv2.resize(self.cam.compute(), (bgr_face.shape[1], bgr_face.shape[0]))
        colored = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(bgr_face, 0.55, colored, 0.45, 0)
        weights = self._lfam_weights()
        return prob, overlay, weights

    @torch.no_grad()
    def predict_fast(self, bgr_face):
        """Probability only (no heatmap). Used for the live camera."""
        x = to_tensor(bgr_face, self.input_size).to(self.device)
        return torch.sigmoid(self.model(x) / self.temperature)[0, 0].item()
