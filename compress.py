import os
import ffmpeg
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from compressai.zoo import cheng2020_attn
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Suppress the autoregressive warning since we're aware of it
warnings.filterwarnings('ignore', message='.*autoregressive.*')

# =============================
# 1. Setup
# =============================
input_video = "original.mp4"
frames_dir = "frames"
compressed_dir = "compressed"
output_video = "compressed_video.mp4"
target_width = None

os.makedirs(frames_dir, exist_ok=True)
os.makedirs(compressed_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Enable cuDNN autotuner for faster convolutions
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

# =============================
# 1a. Auto-detect video info
# =============================
probe = ffmpeg.probe(input_video)
video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
fps_str = video_info['r_frame_rate']
num, den = map(int, fps_str.split('/'))
fps = num / den
width = int(video_info['width'])
height = int(video_info['height'])
print(f"Video info - FPS: {fps}, Width: {width}, Height: {height}")

# =============================
# 2. Extract frames using ffmpeg (GPU-accelerated if available)
# =============================
print("Extracting frames...")
try:
    # Try NVIDIA GPU-accelerated decoding
    (
        ffmpeg.input(input_video, hwaccel='cuda', hwaccel_output_format='cuda')
        .output(os.path.join(frames_dir, "frame_%04d.png"))
        .run(overwrite_output=True, quiet=True)
    )
except:
    # Fallback to CPU decoding
    (
        ffmpeg.input(input_video)
        .output(os.path.join(frames_dir, "frame_%04d.png"))
        .run(overwrite_output=True, quiet=True)
    )
print("Frames extracted successfully.")

# =============================
# 3. Load CompressAI model
# =============================
print("Loading compression model...")
model = cheng2020_attn(quality=1, pretrained=True).eval().to(device)

# Optimize model with torch.compile (PyTorch 2.0+)
if hasattr(torch, 'compile') and device == "cuda":
    print("Compiling model with torch.compile for better performance...")
    model = torch.compile(model, mode='max-autotune')

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# =============================
# 4. Parallel frame loading function
# =============================
def load_and_preprocess_frame(frame_path, target_width, pad_h, pad_w):
    """Load and preprocess a single frame"""
    img = Image.open(frame_path).convert("RGB")
    
    if target_width is not None:
        w_percent = target_width / img.width
        new_h = int(img.height * w_percent)
        img = img.resize((target_width, new_h), Image.LANCZOS)
    
    x = to_tensor(img)
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x_padded

# =============================
# 5. Compress and reconstruct frames (MAXIMUM GPU UTILIZATION)
# =============================
print("Compressing and reconstructing frames with maximum GPU utilization...")

# Adjust batch size based on VRAM (RTX 5090 has 24GB)
# Even though entropy coding is on CPU, we can still process multiple frames in parallel
batch_size = 64  # Increase this if you have VRAM headroom

frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])

# Pre-calculate padding dimensions
sample_img = Image.open(os.path.join(frames_dir, frame_files[0])).convert("RGB")
if target_width is not None:
    w_percent = target_width / sample_img.width
    new_h = int(sample_img.height * w_percent)
    sample_img = sample_img.resize((target_width, new_h), Image.LANCZOS)

sample_tensor = to_tensor(sample_img)
h, w = sample_tensor.size(1), sample_tensor.size(2)
padded_h = ((h + 63) // 64) * 64
padded_w = ((w + 63) // 64) * 64
pad_h = padded_h - h
pad_w = padded_w - w

print(f"Processing {len(frame_files)} frames with batch size {batch_size}...")
print(f"Original size: {h}x{w}, Padded size: {padded_h}x{padded_w}")

# Use CUDA streams for parallel processing
if device == "cuda":
    num_streams = 4  # Multiple CUDA streams for parallel execution
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
else:
    streams = [None]

for batch_idx in range(0, len(frame_files), batch_size):
    batch_files = frame_files[batch_idx : batch_idx + batch_size]
    
    # Parallel frame loading using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as executor:
        frame_paths = [os.path.join(frames_dir, f) for f in batch_files]
        imgs = list(executor.map(
            lambda fp: load_and_preprocess_frame(fp, target_width, pad_h, pad_w),
            frame_paths
        ))
    
    # Stack and move to GPU with pinned memory for faster transfer
    batch_tensor = torch.stack(imgs)
    if device == "cuda":
        batch_tensor = batch_tensor.pin_memory().to(device, non_blocking=True)
    else:
        batch_tensor = batch_tensor.to(device)
    
    # Process frames with CUDA streams for parallel execution
    compressed_batch = []
    
    with torch.no_grad():
        # Use mixed precision for faster computation
        if device == "cuda":
            with torch.amp.autocast("cuda"):
                for idx, x_padded in enumerate(batch_tensor):
                    stream = streams[idx % len(streams)]
                    
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            out = model.compress(x_padded.unsqueeze(0))
                            recon = model.decompress(out["strings"], out["shape"])
                            x_hat = recon["x_hat"].clamp(0, 1).squeeze(0)
                            compressed_batch.append(x_hat)
                    else:
                        out = model.compress(x_padded.unsqueeze(0))
                        recon = model.decompress(out["strings"], out["shape"])
                        x_hat = recon["x_hat"].clamp(0, 1).squeeze(0)
                        compressed_batch.append(x_hat)
                
                # Synchronize all streams
                if device == "cuda":
                    for stream in streams:
                        stream.synchronize()
        else:
            for x_padded in batch_tensor:
                out = model.compress(x_padded.unsqueeze(0))
                recon = model.decompress(out["strings"], out["shape"])
                x_hat = recon["x_hat"].clamp(0, 1).squeeze(0)
                compressed_batch.append(x_hat)
    
    # Parallel frame saving using ThreadPoolExecutor
    def save_frame(args):
        x_hat, frame_file = args
        x_hat_cropped = x_hat[:, :h, :w]
        to_pil(x_hat_cropped.cpu()).save(os.path.join(compressed_dir, frame_file))
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(save_frame, zip(compressed_batch, batch_files))
    
    # Clear GPU cache periodically
    if device == "cuda" and (batch_idx // batch_size) % 5 == 0:
        torch.cuda.empty_cache()
    
    print(f"Processed {min(batch_idx + batch_size, len(frame_files))}/{len(frame_files)} frames "
          f"({100 * min(batch_idx + batch_size, len(frame_files)) / len(frame_files):.1f}%)")

print("All frames compressed successfully.")

# =============================
# 6. Recombine frames into video (GPU-accelerated encoding)
# =============================
print("Reassembling compressed frames into video...")

ffmpeg_input = ffmpeg.input(
    os.path.join(compressed_dir, "frame_%04d.png"), framerate=fps
)

# Use GPU-accelerated H.265 encoding if available
try:
    ffmpeg_output_args = {
        "vcodec": "hevc_nvenc",  # NVIDIA GPU encoder
        "preset": "p7",  # Highest quality preset for NVENC (p1-p7)
        "cq": 18,  # Constant quality (similar to CRF)
        "rc": "vbr",  # Variable bitrate
        "rc-lookahead": 32,  # Lookahead frames for better quality
        "spatial_aq": 1,  # Spatial adaptive quantization
        "temporal_aq": 1,  # Temporal adaptive quantization
        "pix_fmt": "yuv420p",
    }
    (ffmpeg_input.output(output_video, **ffmpeg_output_args).run(overwrite_output=True, quiet=True))
    print("✅ Used GPU-accelerated encoding (hevc_nvenc)")
except Exception as e:
    # Fallback to CPU encoding
    print(f"GPU encoding failed ({e}), falling back to CPU...")
    ffmpeg_output_args = {
        "vcodec": "libx265",
        "crf": 18,
        "preset": "slow",
        "pix_fmt": "yuv420p",
    }
    (ffmpeg_input.output(output_video, **ffmpeg_output_args).run(overwrite_output=True, quiet=True))
    print("✅ Used CPU encoding (libx265)")

print(f"✅ Compressed video saved as {output_video}")

# =============================
# 7. Print GPU utilization stats
# =============================
if device == "cuda":
    print(f"\n📊 GPU Memory Stats:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"   Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# =============================
# 8. Optional cleanup
# =============================
# import shutil
# shutil.rmtree(frames_dir)
# shutil.rmtree(compressed_dir)

print("\n✅ Done!")
