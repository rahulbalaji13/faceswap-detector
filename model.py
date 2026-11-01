import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Branch options (see train.py --branch)
#   spatial  : EfficientNet-B0 only (baseline, no LFAM)
#   fft      : spatial + FFT log-magnitude CNN
#   srm      : spatial + fixed SRM high-pass noise-residual CNN
#   fft+srm  : spatial + both side branches  ← best LFAM configuration
# ----------------------------------------------------------------------------
BRANCHES = ("spatial", "fft", "srm", "fft+srm")

_TAGS = {"spatial": "spatial_only", "fft": "fusion", "srm": "srm", "fft+srm": "fft_srm"}
_LABELS = {
    "spatial": "Spatial only",
    "fft": "Spatial + FFT",
    "srm": "Spatial + SRM noise",
    "fft+srm": "Spatial + FFT + SRM noise",
}

# Branch display names (used by LFAM weight visualization)
BRANCH_NAMES = {
    "spatial": ["Spatial"],
    "fft":     ["Spatial", "FFT"],
    "srm":     ["Spatial", "SRM"],
    "fft+srm": ["Spatial", "FFT", "SRM"],
}


def parse_branch(branch):
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, got '{branch}'")
    return "fft" in branch, "srm" in branch


def model_tag(branch, synthetic_ratio=0.0, use_lfam=True):
    """Short key used in results/metrics.json."""
    tag = _TAGS[branch]
    if use_lfam and branch != "spatial":
        tag += "_lfam"
    return tag + ("_synth" if synthetic_ratio and synthetic_ratio > 0 else "")


def model_label(branch, synthetic_ratio=0.0, use_lfam=True):
    """Human-readable name shown in the web app's accuracy table."""
    label = _LABELS[branch]
    if use_lfam and branch != "spatial":
        label += " + LFAM"
    return label + (" + synthetic fakes" if synthetic_ratio and synthetic_ratio > 0 else "")


# ----------------------------------------------------------------------------
# LFAM — Latent Frequency Attention Module  ← NOVEL CONTRIBUTION
#
# Problem with simple concatenation:
#   concat([spatial, fft, srm]) treats every branch as equally important for
#   every image.  In reality:
#     • A heavily JPEG-compressed image destroys frequency / noise traces
#       → the model should rely mostly on spatial pixel features.
#     • A high-quality GAN-generated deepfake has no visible pixel seam but
#       leaves strong upsampling grid lines in the FFT spectrum.
#     → the model should shift weight toward the FFT branch.
#
# Solution — LFAM:
#   For each image, LFAM learns a set of attention weights w_s, w_f, w_n
#   (one per active branch) via a small gating MLP, then produces a
#   per-image adaptive weighted-sum of branch features.  This lets the model
#   route evidence through the most informative domain on a sample-by-sample
#   basis and makes the decision interpretable (we can show which branch
#   "caught" the fake).
# ----------------------------------------------------------------------------
class LFAM(nn.Module):
    """Latent Frequency Attention Module.

    Args:
        in_dims   : list of feature dimensions for each branch
                    e.g. [1280, 128] for fft or [1280, 128, 128] for fft+srm
        hidden_dim: common projection dimension (default 256)
        temperature: softmax temperature; lower → sharper (more decisive) routing
    """
    HIDDEN = 256   # common projection dim (class-level so callers can query it)

    def __init__(self, in_dims: list, hidden_dim: int = 256, temperature: float = 1.5):
        super().__init__()
        self.temperature = temperature
        n = len(in_dims)

        # Project every branch to the same hidden_dim so we can weight-sum them.
        # LayerNorm keeps the projected vectors on the same scale regardless of
        # how different the raw branch output ranges are.
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ) for d in in_dims
        ])

        # Gating MLP: takes all projected features concatenated → one logit per branch.
        # Softmax of those logits (with temperature) gives the branch weights.
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * n, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n),
        )

        # Keep track of last weights for visualization / logging
        self._last_weights: torch.Tensor | None = None

    def forward(self, *feats: torch.Tensor):
        """
        Args:
            *feats : one (B, dim_i) tensor per branch
        Returns:
            fused   : (B, hidden_dim) weighted-sum of projected branch features
            weights : (B, n_branches) attention weights (sum to 1 along dim=1)
        """
        projected = [proj(f) for proj, f in zip(self.projectors, feats)]  # list of (B, H)
        gate_input = torch.cat(projected, dim=1)                           # (B, H*n)
        raw_logits = self.gate(gate_input)                                 # (B, n)
        weights = torch.softmax(raw_logits / self.temperature, dim=1)      # (B, n)

        # Weighted sum: stack projected tensors → (B, n, H), then multiply + sum
        stacked = torch.stack(projected, dim=1)                            # (B, n, H)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)               # (B, H)

        self._last_weights = weights.detach()
        return fused, weights


# ----------------------------------------------------------------------------
# FFT branch (unchanged from the original project)
# ----------------------------------------------------------------------------
def fft_magnitude(x):
    """x: (B,3,H,W) -> (B,1,H,W) normalized log-magnitude spectrum (low freq centered)."""
    x = x.float()
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    f = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
    mag = torch.log(torch.abs(f) + 1e-6)
    mean = mag.mean(dim=(1, 2), keepdim=True)
    std = mag.std(dim=(1, 2), keepdim=True) + 1e-6
    return ((mag - mean) / std).unsqueeze(1)


def conv_block(i, o, stride=2):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


# ----------------------------------------------------------------------------
# SRM noise-residual branch (NEW)
# ----------------------------------------------------------------------------
def _srm_kernels():
    """The 3 classic SRM high-pass kernels (Fridrich & Kodovsky), 5x5, each sums to 0."""
    k1 = torch.tensor([[0, 0, 0, 0, 0],
                       [0, -1, 2, -1, 0],
                       [0, 2, -4, 2, 0],
                       [0, -1, 2, -1, 0],
                       [0, 0, 0, 0, 0]], dtype=torch.float32) / 4.0
    k2 = torch.tensor([[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
    k3 = torch.tensor([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 1, -2, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]], dtype=torch.float32) / 2.0
    return torch.stack([k1, k2, k3])  # (3,5,5)


class SRMFilter(nn.Module):
    """Fixed (non-trainable) SRM high-pass layer: (B,3,H,W) normalized RGB -> (B,9,H,W) noise residuals.

    Each of the 3 kernels is applied to each RGB channel (3x3 = 9 output maps).
    The ImageNet normalization is undone first so residuals are measured in 8-bit
    gray levels, then truncated to [-trunc, trunc] (standard SRM step: keep only the
    small, noise-like values, drop strong edges) and scaled to [-1, 1].
    """
    def __init__(self, trunc=3.0):
        super().__init__()
        self.trunc = trunc
        weight = _srm_kernels().repeat(3, 1, 1).unsqueeze(1)  # (9,1,5,5): [k1,k2,k3] per input channel
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        pix = (x.float() * self.std + self.mean) * 255.0           # back to 0..255 pixel scale
        res = F.conv2d(pix, self.weight, padding=2, groups=3)       # fixed high-pass, no gradient needed
        if self.trunc:
            return res.clamp(-self.trunc, self.trunc) / self.trunc
        return res / 8.0


class FaceSwapNet(nn.Module):
    """Spatial branch (EfficientNet-B0 on pixels) + optional FFT branch + optional SRM noise branch,
    fused via LFAM (Latent Frequency Attention Module) when multiple branches are active.

    LFAM replaces the original static concatenation with a learned, per-image attention
    mechanism that dynamically routes evidence through the most informative signal domain.

    Input is (B,3,H,W) normalized RGB, OR (B,6,H,W) = [degraded view | clean view].
    The 6-channel form is only produced by the training dataset (dual-view mode): the spatial
    branch sees the augmented view (JPEG/blur/colour), while the FFT/SRM branches see the
    clean view so augmentation does not wipe out the faint artifacts they look for.
    At inference/eval a 3-channel image is used for every branch.
    """
    def __init__(self, branch="fft", pretrained=True, use_freq=None, use_lfam=True):
        super().__init__()
        if use_freq is not None:  # backward compatibility with the old signature
            branch = "fft" if use_freq else "spatial"
        self.branch = branch
        self.use_fft, self.use_srm = parse_branch(branch)
        self.use_freq = self.use_fft or self.use_srm

        # ── Spatial branch (always present) ──────────────────────────────────
        self.spatial = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        spatial_dim = self.spatial.num_features  # 1280

        branch_dims = [spatial_dim]   # start with spatial dim; side branches append below

        # ── FFT frequency branch ──────────────────────────────────────────────
        if self.use_fft:
            self.freq = nn.Sequential(  # same name/layout as the original → old checkpoints still load
                conv_block(1, 16), conv_block(16, 32), conv_block(32, 64), conv_block(64, 128),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            branch_dims.append(128)

        # ── SRM noise-residual branch ─────────────────────────────────────────
        if self.use_srm:
            self.srm = nn.Sequential(
                SRMFilter(),
                conv_block(9, 32, stride=1),  # stride 1 first: don't throw away fine noise detail
                conv_block(32, 64), conv_block(64, 128), conv_block(128, 128),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            branch_dims.append(128)

        # ── LFAM fusion (active when 2+ branches present) ─────────────────────
        # For spatial-only (single branch), LFAM is not applicable; keep original head.
        self.use_lfam = use_lfam and len(branch_dims) > 1
        if self.use_lfam:
            self.lfam = LFAM(branch_dims)
            head_in = LFAM.HIDDEN        # weighted-sum output dimension
        else:
            head_in = sum(branch_dims)   # original concatenation fallback

        # ── Classification head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(head_in, 256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, 1)
        )

        # Stores the last LFAM weight vector for visualization (set during forward)
        self._last_weights: torch.Tensor | None = None

    @staticmethod
    def _split(x):
        if x.shape[1] == 6:
            return x[:, :3], x[:, 3:]
        return x, x

    def forward(self, x):
        x_spatial, x_clean = self._split(x)
        feats = [self.spatial(x_spatial)]
        if self.use_fft:
            feats.append(self.freq(fft_magnitude(x_clean)))
        if self.use_srm:
            feats.append(self.srm(x_clean))

        if self.use_lfam:
            # LFAM: dynamic per-image weighted fusion
            fused, weights = self.lfam(*feats)
            self._last_weights = weights.detach()
            return self.head(fused)                   # logit; sigmoid(logit) = P(fake)
        else:
            # Fallback: original static concatenation (spatial-only branch)
            self._last_weights = None
            return self.head(torch.cat(feats, dim=1))


def load_checkpoint(path, device="cpu"):
    """Rebuild the right model from a checkpoint. Returns (model, checkpoint_dict, branch).

    Supports three checkpoint generations:
      • New (post-LFAM): stores 'branch' and 'use_lfam'.
      • Mid (pre-LFAM):  stores 'branch' but not 'use_lfam'  → defaults to False.
      • Old:             stores only 'use_freq'               → infers branch, no LFAM.
    """
    ck = torch.load(path, map_location=device)
    branch = ck.get("branch") or ("fft" if ck.get("use_freq", True) else "spatial")
    use_lfam = ck.get("use_lfam", False)   # old checkpoints default to no-LFAM
    model = FaceSwapNet(branch=branch, pretrained=False, use_lfam=use_lfam).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck, branch
