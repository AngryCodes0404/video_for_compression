import torch
import cv2
from torchvision.transforms import ToTensor, ToPILImage
from compressai.zoo import bmshj2018_factorized, ssf2020
import pickle
import numpy as np
import os

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models (adjust quality: 1-8, lower = more compression)
quality = 4
i_net = bmshj2018_factorized(quality=quality, pretrained=True).to(device).eval()
p_net = ssf2020(quality=quality, metric='mse', pretrained=True).to(device).eval()

# Input video path
input_path = 'original.mp4'
compressed_path = 'compressed.bin'
reconstructed_path = 'reconstructed.mp4'

# ====================
# Encoding Section
# ====================
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise ValueError("Unable to open video file")

# List to hold encoded data: [('I' or 'P', strings, shape)]
encoded_data = []

# Read first frame (I-frame)
ret, frame = cap.read()
if not ret:
    raise ValueError("Video has no frames")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
x = ToTensor()(frame).unsqueeze(0).to(device)

with torch.no_grad():
    compressed = i_net.compress(x)
encoded_data.append(('I', compressed['strings'], compressed['shape']))

# Get reconstructed for next frame
with torch.no_grad():
    x_hat = i_net.decompress(compressed['strings'], compressed['shape'])['x_hat'].clip_(0, 1)

# Process remaining frames (P-frames)
frame_count = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = ToTensor()(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        compressed = p_net.compress(x_hat, x)
    encoded_data.append(('P', compressed['strings'], compressed['shape']))

    # Update x_hat for next
    with torch.no_grad():
        x_hat = p_net.decompress(x_hat, compressed['strings'], compressed['shape'])['x_hat'].clip_(0, 1)

# Save compressed data
with open(compressed_path, 'wb') as f:
    pickle.dump(encoded_data, f)
print(f"Compressed bitstream saved to {compressed_path} (size: {os.path.getsize(compressed_path)} bytes)")

cap.release()

# ====================
# Decoding Section
# ====================
with open(compressed_path, 'rb') as f:
    encoded_data = pickle.load(f)

# Get video properties from original (for output)
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

out = cv2.VideoWriter(reconstructed_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

to_pil = ToPILImage()

# Decode first frame (I-frame)
frame_type, strings, shape = encoded_data[0]
assert frame_type == 'I'
with torch.no_grad():
    x_hat = i_net.decompress(strings, shape)['x_hat'].clip_(0, 1)
frame = np.array(to_pil(x_hat.squeeze(0).cpu()))
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
out.write(frame)

# Decode remaining frames (P-frames)
for frame_type, strings, shape in encoded_data[1:]:
    assert frame_type == 'P'
    with torch.no_grad():
        dec_out = p_net.decompress(x_hat, strings, shape)
    x_hat = dec_out['x_hat'].clip_(0, 1)
    frame = np.array(to_pil(x_hat.squeeze(0).cpu()))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)

out.release()
print(f"Reconstructed video saved to {reconstructed_path}")