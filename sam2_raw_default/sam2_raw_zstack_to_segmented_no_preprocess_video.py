"""in this code no preprocessing was done direct conversion of the z stacks to the video is given to the sam2 model as an input """
import os
import sys
import cv2
import numpy as np
import tifffile as tiff
import torch
from tqdm import tqdm

# MAKE SAM2 PACKAGE VISIBLE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OGD_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(OGD_ROOT)

from sam2.build_sam import build_sam2_video_predictor

# PATHS

RAW_TIF_PATH = os.path.join(BASE_DIR, "Idisc001.tif")

RAW_OUTPUT_DIR = os.path.join(BASE_DIR, "sam2_raw_outputs")
os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)

RAW_VIDEO_PATH = os.path.join(RAW_OUTPUT_DIR, "Idisc001_raw_video.mp4")

RAW_MASK_DIR = os.path.join(RAW_OUTPUT_DIR, "raw_masks")
os.makedirs(RAW_MASK_DIR, exist_ok=True)

RAW_SEGMENTED_VIDEO_PATH = os.path.join(RAW_OUTPUT_DIR, "Idisc001_raw_segmented.mp4")

# STEP 1: RAW Z-STACK → VIDEO
print("\n[1] Loading RAW Z-stack...")
stack = tiff.imread(RAW_TIF_PATH)  # (Z, H, W)
Z, H, W = stack.shape
print(f"RAW stack shape: {stack.shape}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(RAW_VIDEO_PATH, fourcc, 5, (W, H), isColor=True)

print("[2] Converting RAW Z-stack to video...")
for z in tqdm(range(Z)):
    img = stack[z].astype(np.float32)
    img -= img.min()
    img /= (img.max() + 1e-6)
    img = (img * 255).astype(np.uint8)
    frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    video_writer.write(frame)

video_writer.release()
print(f"RAW video saved at: {RAW_VIDEO_PATH}")

# STEP 2: LOAD SAM2
print("\n[3] Loading SAM2 model...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = build_sam2_video_predictor(
    config_file="configs/sam2.1/sam2.1_hiera_t.yaml",
    ckpt_path=os.path.join(OGD_ROOT, "SAM2", "checkpoints", "sam2.1_hiera_tiny.pt"),
    device=DEVICE,)

print("SAM2 loaded successfully")

# STEP 3: GENERIC INITIALIZATION
print("[4] Initializing inference state...")
inference_state = predictor.init_state(RAW_VIDEO_PATH)

cap = cv2.VideoCapture(RAW_VIDEO_PATH)
ret, frame0 = cap.read()
cap.release()

H, W = frame0.shape[:2]
point = np.array([[W // 2, H // 2]])
label = np.array([1])

predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=point,
    labels=label,
)

print("Generic center-point initialization added")

# STEP 4: SEGMENTATION

print("\n[5] Running SAM2 segmentation...")
for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
    mask = masks[0].cpu().numpy()
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    mask = (mask > 0).astype(np.uint8)

    np.save(
        os.path.join(RAW_MASK_DIR, f"raw_frame_{frame_idx:04d}.npy"),
        mask
    )

print("RAW masks saved")

# STEP 5: OVERLAY VIDEO

print("\n[6] Creating overlay video...")
cap = cv2.VideoCapture(RAW_VIDEO_PATH)

out = cv2.VideoWriter(
    RAW_SEGMENTED_VIDEO_PATH,
    fourcc,
    5,
    (W, H),
)

idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = np.load(
        os.path.join(RAW_MASK_DIR, f"raw_frame_{idx:04d}.npy")
    )

    overlay = frame.copy()
    overlay[mask == 1] = [0, 0, 255]

    blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    out.write(blended)
    idx += 1

cap.release()
out.release()

print(f"\nRAW segmented video saved at:\n{RAW_SEGMENTED_VIDEO_PATH}")
