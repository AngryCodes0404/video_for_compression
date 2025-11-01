import os
import ffmpeg
import torch
from PIL import Image
from torchvision import transforms
from compressai.zoo import bmshj2018_hyperprior

# =============================
# 1. Setup
# =============================
input_video = "original.mp4"  # Input video file
frames_dir = "frames"  # Directory for extracted frames
compressed_dir = "compressed"  # Directory for reconstructed frames
output_video = "compressed_video.mp4"
fps = 30  # Change if your video has a different framerate

os.makedirs(frames_dir, exist_ok=True)
os.makedirs(compressed_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============================
# 2. Extract frames using ffmpeg
# =============================
print("Extracting frames...")
(
    ffmpeg.input(input_video)
    .output(os.path.join(frames_dir, "frame_%04d.png"))
    .run(overwrite_output=True)
)
print("Frames extracted successfully.")

# =============================
# 3. Load CompressAI model
# =============================
print("Loading compression model...")
model = bmshj2018_hyperprior(quality=3, pretrained=True).eval().to(device)
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# =============================
# 4. Compress and reconstruct each frame
# =============================
print("Compressing and reconstructing frames...")
for frame_file in sorted(os.listdir(frames_dir)):
    if not frame_file.endswith(".png"):
        continue

    input_path = os.path.join(frames_dir, frame_file)
    output_path = os.path.join(compressed_dir, frame_file)

    # Load frame
    img = Image.open(input_path).convert("RGB")
    x = to_tensor(img).unsqueeze(0).to(device)

    # Compress and reconstruct
    with torch.no_grad():
        out = model.compress(x)
        recon = model.decompress(out["strings"], out["shape"])
        x_hat = recon["x_hat"].clamp(0, 1)

    # Save reconstructed frame
    to_pil(x_hat.squeeze().cpu()).save(output_path)

print("All frames compressed successfully.")

# =============================
# 5. Recombine frames into video
# =============================
print("Reassembling compressed frames into video...")
(
    ffmpeg.input(os.path.join(compressed_dir, "frame_%04d.png"), framerate=fps)
    .output(output_video, vcodec="libx264", crf=23)
    .run(overwrite_output=True)
)
print(f"✅ Compressed video saved as {output_video}")

# =============================
# 6. Optional cleanup
# =============================
# Uncomment these lines if you want to delete temporary frames
# import shutil
# shutil.rmtree(frames_dir)
# shutil.rmtree(compressed_dir)

print("Done!")
