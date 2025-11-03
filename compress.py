import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from compressai.zoo import ssf2020
from compressai.utils.bench import compute_bpp

# ============================================================
# Configuration
# ============================================================
INPUT_VIDEO = "original.mp4"
OUTPUT_VIDEO = "compressed_ssf2020.mp4"
QUALITY = 3  # from 1 (best quality, largest size) to 6 (lowest quality)
METRIC = "mse"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Load model
# ============================================================
print(f"Loading SSF2020 model (quality={QUALITY}, metric={METRIC})...")
model = ssf2020(quality=QUALITY, metric=METRIC, pretrained=True).eval().to(DEVICE)

# ============================================================
# Prepare input video
# ============================================================
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

os.makedirs("temp_frames", exist_ok=True)
 
# ============================================================
# Frame-by-frame compression
# ============================================================
print("Compressing video frame by frame...")
compressed_frames = []

with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB and normalize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = x.to(DEVICE)

        # Compress
        with torch.no_grad():
            out_enc = model.compress(x)
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        
        # Decode and save
        rec_img = out_dec["x_hat"].clamp(0, 1)
        rec_np = (rec_img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rec_bgr = cv2.cvtColor(rec_np, cv2.COLOR_RGB2BGR)
        compressed_frames.append(rec_bgr)
        pbar.update(1)

cap.release()

# ============================================================
# Write compressed video
# ============================================================
print(f"Writing compressed video to {OUTPUT_VIDEO}...")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
for frame in compressed_frames:
    out.write(frame)
out.release()

print("✅ Compression complete!")
print(f"Saved to: {OUTPUT_VIDEO}")
