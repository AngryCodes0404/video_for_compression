import os
import ffmpeg
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from compressai.zoo import bmshj2018_hyperprior

# =============================
# 1. Setup
# =============================
input_video = "original.mp4"       # Input video file
frames_dir = "frames"              # Directory for extracted frames
compressed_dir = "compressed"      # Directory for reconstructed frames
output_video = "compressed_video.mp4"
fps = 30                           # Adjust if your video has a different framerate
target_width = None                # Set to an int to downscale width, or None to keep original

os.makedirs(frames_dir, exist_ok=True)
os.makedirs(compressed_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============================
# 2. Extract frames using ffmpeg
# =============================
print("Extracting frames...")
(
    ffmpeg
    .input(input_video)
    .output(os.path.join(frames_dir, "frame_%04d.png"))
    .run(overwrite_output=True)
)
print("Frames extracted successfully.")

# =============================
# 3. Load CompressAI model
# =============================
print("Loading compression model...")
model = bmshj2018_hyperprior(quality=2, pretrained=True).eval().to(device)  # max compression
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# =============================
# 4. Compress and reconstruct frames in batches
# =============================
print("Compressing and reconstructing frames in batches...")
batch_size = 16  # Adjust depending on VRAM; RTX 5090 can likely handle 16-32+
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])

for i in range(0, len(frame_files), batch_size):
    batch_files = frame_files[i:i+batch_size]
    imgs = []
    orig_sizes = []

    # Load frames and store original sizes
    max_h, max_w = 0, 0
    for frame_file in batch_files:
        img = Image.open(os.path.join(frames_dir, frame_file)).convert("RGB")

        # Optional downscale
        if target_width is not None:
            w_percent = target_width / img.width
            new_h = int(img.height * w_percent)
            img = img.resize((target_width, new_h), Image.LANCZOS)

        x = to_tensor(img)
        h, w = x.size(1), x.size(2)
        orig_sizes.append((h, w))
        max_h = max(max_h, h)
        max_w = max(max_w, w)
        imgs.append(x)

    # Pad frames to multiple of 64
    padded_imgs = []
    for x, (h, w) in zip(imgs, orig_sizes):
        new_h = (max_h + 63) // 64 * 64
        new_w = (max_w + 63) // 64 * 64
        pad_h = new_h - h
        pad_w = new_w - w
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        padded_imgs.append(x_padded)

    batch_tensor = torch.stack(padded_imgs).to(device)

    # Compress and reconstruct batch
    with torch.no_grad():
        compressed_batch = []
        for x_padded in batch_tensor:
            out = model.compress(x_padded.unsqueeze(0))
            recon = model.decompress(out["strings"], out["shape"])
            x_hat = recon["x_hat"].clamp(0, 1)
            compressed_batch.append(x_hat.squeeze(0))

    # Crop to original size and save
    for x_hat, (h, w), frame_file in zip(compressed_batch, orig_sizes, batch_files):
        x_hat_cropped = x_hat[:, :h, :w]
        to_pil(x_hat_cropped.cpu()).save(os.path.join(compressed_dir, frame_file))

print("All frames compressed successfully in batches.")

# =============================
# 5. Recombine frames into video
# =============================
print("Reassembling compressed frames into video...")

ffmpeg_input = ffmpeg.input(os.path.join(compressed_dir, "frame_%04d.png"), framerate=fps)
ffmpeg_output_args = {
    "vcodec": "libx265",
    "crf": 28,
    "preset": "slow",
    "pix_fmt": "yuv420p"
}
(
    ffmpeg_input
    .output(output_video, **ffmpeg_output_args)
    .run(overwrite_output=True)
)
print(f"✅ Compressed video saved as {output_video}")

# =============================
# 6. Optional cleanup
# =============================
# Uncomment to remove temporary frames
# import shutil
# shutil.rmtree(frames_dir)
# shutil.rmtree(compressed_dir)

print("Done!")
