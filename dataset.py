import io
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

EXT = {".jpg", ".jpeg", ".png"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIZE = 224


class RandomJPEG:
    """Simulates compression so the model doesn't just learn 'compressed = fake'."""
    def __init__(self, p=0.5, quality=(40, 95)):
        self.p, self.quality = p, quality

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class TrainTransform:
    """Training augmentation that can return TWO views of the same face.

    1. geometry  : flip + shift-crop. No resampling, so the pixel noise is untouched.
    2. 'clean'   : the geometry output as is  -> fed to the FFT / SRM branches.
    3. 'degraded': clean + rescale/rotate/blur/JPEG/colour jitter -> fed to the spatial branch.

    dual_view=False -> 3-channel tensor (degraded view only, like the original pipeline).
    dual_view=True  -> 6-channel tensor [degraded | clean]; FaceSwapNet splits it again.
    """
    def __init__(self, dual_view=False, size=SIZE):
        self.dual_view = dual_view
        self.size = size
        self.geometry = T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.RandomCrop(size, padding=max(1, round(size * 12 / SIZE)), padding_mode="reflect"),
        ])
        self.degrade = T.Compose([
            T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            T.RandomRotation(10),
            T.RandomApply([T.GaussianBlur(3, (0.1, 1.5))], p=0.2),
            RandomJPEG(),
            T.ColorJitter(0.3, 0.3, 0.3),
        ])
        self.to_tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __call__(self, img):
        clean = self.geometry(img)
        x = self.to_tensor(self.degrade(clean))
        if self.dual_view:
            x = torch.cat([x, self.to_tensor(clean)], dim=0)
        return x


def get_transform(train, dual_view=False, size=SIZE):
    if not train:
        return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    return TrainTransform(dual_view, size)


class FaceDataset(Dataset):
    """root/real/*.jpg (label 0) and root/fake/*.jpg (label 1)."""
    def __init__(self, root, train=False, dual_view=False, size=SIZE):
        self.items = []
        for label, name in enumerate(["real", "fake"]):
            for p in sorted((Path(root) / name).rglob("*")):
                if p.suffix.lower() in EXT:
                    self.items.append((p, label))
        if not self.items:
            raise RuntimeError(f"No images found in {root}/real or {root}/fake")
        self.counts = [sum(1 for _, l in self.items if l == c) for c in (0, 1)]
        self.tf = get_transform(train, dual_view, size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


# ----------------------------------------------------------------------------
# Synthetic blended fakes (Face X-ray / Self-Blended-Images style)
#
# A fake is made from ONE real face: an altered copy of the face (colour shift,
# blur/sharpen, small affine warp) is blended back onto the original through a
# soft face-shaped mask. The result has the same kind of blending boundary,
# colour mismatch and resampling traces as a real face swap, but no generator.
# The model is forced to learn those generic traces instead of one dataset's fakes.
# ----------------------------------------------------------------------------
def _color_shift(img, rng):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    hsv[..., 1] *= rng.uniform(0.8, 1.2)
    hsv[..., 2] *= rng.uniform(0.85, 1.15)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    out += rng.uniform(-12, 12, size=3)  # per-channel RGB shift
    return np.clip(out, 0, 255).astype(np.uint8)


def _blur_or_sharpen(img, rng):
    r = rng.random()
    if r < 0.35:
        k = int(rng.choice([3, 5]))
        return cv2.GaussianBlur(img, (k, k), 0)
    if r < 0.60:
        blur = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.8, 1.6)))
        return cv2.addWeighted(img, 1.5, blur, -0.5, 0)  # unsharp mask
    return img


def _random_affine(img, rng):
    """Small scale / rotate / shift. The resampling also changes the noise statistics."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-3, 3)), float(rng.uniform(0.93, 1.07)))
    m[:, 2] += rng.uniform(-0.04, 0.04, size=2) * np.array([w, h])
    interp = int(rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    return cv2.warpAffine(img, m, (w, h), flags=interp, borderMode=cv2.BORDER_REPLICATE)


def _face_mask(h, w, rng):
    """Soft face-shaped mask in [0,1]: random ellipse -> elastic deformation -> erode/dilate -> feather."""
    mask = np.zeros((h, w), np.float32)
    center = (int(w * (0.5 + rng.uniform(-0.04, 0.04))), int(h * (0.5 + rng.uniform(-0.03, 0.06))))
    axes = (int(w * rng.uniform(0.25, 0.37)), int(h * rng.uniform(0.30, 0.43)))
    cv2.ellipse(mask, center, axes, float(rng.uniform(-10, 10)), 0, 360, 1.0, -1)

    # elastic deformation so the boundary isn't a perfect ellipse (stands in for a landmark hull)
    amp = float(rng.uniform(0.01, 0.035)) * w
    dx = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 18)
    dy = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 18)
    dx *= amp / (np.abs(dx).max() + 1e-6)
    dy *= amp / (np.abs(dy).max() + 1e-6)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mask = cv2.remap(mask, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    k = int(rng.integers(0, 7))
    if k > 0:
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.erode(mask, kernel) if rng.random() < 0.5 else cv2.dilate(mask, kernel)

    ks = int(rng.choice([5, 9, 15, 21, 31]))  # feathered edge (odd sizes)
    mask = cv2.GaussianBlur(mask, (ks, ks), 0)
    return np.clip(mask, 0.0, 1.0)


def blend_face(bgr, rng=None, return_mask=False):
    """bgr uint8 REAL face -> blended 'fake' uint8 (and the blend mask if return_mask)."""
    rng = rng or np.random.default_rng()
    h, w = bgr.shape[:2]
    src = _random_affine(_blur_or_sharpen(_color_shift(bgr, rng), rng), rng)
    mask = _face_mask(h, w, rng) * float(rng.uniform(0.8, 1.0))
    m3 = mask[..., None]
    out = m3 * src.astype(np.float32) + (1.0 - m3) * bgr.astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return (out, mask) if return_mask else out


class SyntheticBlendDataset(Dataset):
    """On-the-fly blended fakes (label 1) built from REAL face crops only.

    A fresh random blend is generated every time an item is read, so each epoch sees new fakes.
    """
    def __init__(self, real_paths, n_samples, dual_view=False, size=SIZE):
        self.paths = [Path(p) for p in real_paths]
        if not self.paths:
            raise RuntimeError("SyntheticBlendDataset needs at least one real face image.")
        self.n = int(n_samples)
        self.tf = get_transform(True, dual_view, size)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        path = self.paths[i % len(self.paths)]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read {path}")
        if bgr.shape[:2] != (SIZE, SIZE):
            bgr = cv2.resize(bgr, (SIZE, SIZE))
        rng = np.random.default_rng(random.getrandbits(64))  # python `random` is seeded per DataLoader worker
        fake = blend_face(bgr, rng)
        img = Image.fromarray(cv2.cvtColor(fake, cv2.COLOR_BGR2RGB))
        return self.tf(img), 1
