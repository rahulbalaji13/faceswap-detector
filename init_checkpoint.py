"""Initialize default checkpoint, metrics, and sample test images."""
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw

from model import FaceSwapNet

def create_sample_faces():
    Path("test_samples").mkdir(exist_ok=True)
    
    # 1. Generate realistic test face 1: Real Face (smooth skin tones, natural features)
    img_real = Image.new("RGB", (320, 320), color=(240, 240, 245))
    draw = ImageDraw.Draw(img_real)
    # Head contour
    draw.ellipse([60, 40, 260, 280], fill=(235, 195, 165), outline=(210, 170, 140), width=2)
    # Eyes
    draw.ellipse([100, 120, 130, 145], fill=(255, 255, 255), outline=(50, 40, 40))
    draw.ellipse([110, 127, 122, 139], fill=(60, 40, 30))
    draw.ellipse([190, 120, 220, 145], fill=(255, 255, 255), outline=(50, 40, 40))
    draw.ellipse([198, 127, 210, 139], fill=(60, 40, 30))
    # Eyebrows
    draw.arc([95, 105, 135, 125], start=190, end=350, fill=(60, 40, 30), width=3)
    draw.arc([185, 105, 225, 125], start=190, end=350, fill=(60, 40, 30), width=3)
    # Nose
    draw.line([(160, 140), (155, 180), (168, 185)], fill=(190, 140, 110), width=3)
    # Mouth
    draw.ellipse([130, 215, 190, 235], fill=(205, 100, 100), outline=(170, 70, 70))
    # Add subtle natural texture noise
    arr_real = np.array(img_real).astype(np.float32)
    noise = np.random.normal(0, 3.0, arr_real.shape)
    arr_real = np.clip(arr_real + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr_real).save("test_samples/sample_real.jpg", quality=95)

    # 2. Generate test face 2: Swapped / Manipulated Face (has inner blend boundary & frequency noise)
    img_fake = Image.new("RGB", (320, 320), color=(235, 238, 242))
    draw_f = ImageDraw.Draw(img_fake)
    # Outer head
    draw_f.ellipse([60, 40, 260, 280], fill=(225, 185, 155), outline=(200, 160, 130), width=2)
    # Swapped donor face patch (slight color mismatch & seam boundary typical of face-swap)
    draw_f.ellipse([85, 90, 235, 250], fill=(245, 205, 175), outline=(180, 130, 110), width=2)
    # Eyes
    draw_f.ellipse([105, 125, 135, 148], fill=(255, 255, 255), outline=(40, 30, 30))
    draw_f.ellipse([115, 130, 127, 142], fill=(40, 80, 110)) # blue eyes
    draw_f.ellipse([185, 125, 215, 148], fill=(255, 255, 255), outline=(40, 30, 30))
    draw_f.ellipse([193, 130, 205, 142], fill=(40, 80, 110))
    # Eyebrows
    draw_f.arc([100, 110, 140, 130], start=190, end=350, fill=(40, 30, 30), width=4)
    draw_f.arc([180, 110, 220, 130], start=190, end=350, fill=(40, 30, 30), width=4)
    # Nose
    draw_f.line([(160, 145), (153, 185), (167, 190)], fill=(180, 130, 100), width=3)
    # Mouth
    draw_f.ellipse([128, 218, 192, 240], fill=(195, 90, 90), outline=(160, 60, 60))
    
    # Add high-frequency periodic grid artifact in the swapped patch (classic GAN upsampling trace)
    arr_fake = np.array(img_fake).astype(np.float32)
    h, w, c = arr_fake.shape
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    grid_pattern = 6.0 * np.sin(2 * np.pi * x_idx / 4.0) * np.sin(2 * np.pi * y_idx / 4.0)
    arr_fake[90:250, 85:235] += grid_pattern[90:250, 85:235, None]
    arr_fake = np.clip(arr_fake, 0, 255).astype(np.uint8)
    Image.fromarray(arr_fake).save("test_samples/sample_fake.jpg", quality=90)
    print("Created test_samples/sample_real.jpg and test_samples/sample_fake.jpg")

def create_model_checkpoint():
    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path = "checkpoints/best.pt"
    
    print("Initializing FaceSwapNet with LFAM (branch: fft+srm)...")
    model = FaceSwapNet(branch="fft+srm", pretrained=True, use_lfam=True)
    model.eval()
    
    torch.save({
        "model": model.state_dict(),
        "branch": "fft+srm",
        "use_freq": True,
        "use_lfam": True,
        "synthetic_ratio": 0.5,
        "threshold": 0.50
    }, ckpt_path)
    print(f"Saved default LFAM checkpoint to {ckpt_path}")

if __name__ == "__main__":
    create_sample_faces()
    create_model_checkpoint()
