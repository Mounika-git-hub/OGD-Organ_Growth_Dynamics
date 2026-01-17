"""Below code is the automate code for multiple samples now only giving the few samples from one directory later we can modify with multiple samples into one single directory"""

"""
Automated SAM2 pipeline:
- Reads multiple Z-stack .tif files
- Converts Z-stack -> video
- Runs SAM2 video segmentation
- Saves masks + overlay video
- Processes first N samples automatically
"""

import os
import sys
import cv2
import numpy as np
import tifffile as tiff
import torch
from tqdm import tqdm

# PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OGD_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(OGD_ROOT)

from sam2.build_sam import build_sam2_video_predictor

# USER CONFIG
INPUT_TIF_DIR = (
    "/home/ibab/Downloads/Deep_Learning_for_image_analysis/"
    "Deep learning for image analysis-20250926T061103Z-1-001/"
    "Deep learning for image analysis")

OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

NUM_SAMPLES = 5          # process first 5 samples
FPS = 5

# LOAD SAM2 ONCE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\n[INFO] Loading SAM2 model...")
predictor = build_sam2_video_predictor(
    config_file="configs/sam2.1/sam2.1_hiera_t.yaml",
    ckpt_path=os.path.join(OGD_ROOT, "SAM2", "checkpoints", "sam2.1_hiera_tiny.pt"),
    device=DEVICE,)
print("SAM2 loaded successfully\n")

# COLLECT INPUT FILES
tif_files = sorted(
    f for f in os.listdir(INPUT_TIF_DIR)
    if f.lower().endswith(".tif")
)[:NUM_SAMPLES]

print(f"Processing {len(tif_files)} samples:")
for f in tif_files:
    print("  -", f)

# MAIN LOOP
for tif_name in tif_files:
    sample_id = os.path.splitext(tif_name)[0]
    print(f"Processing: {sample_id}")

    tif_path = os.path.join(INPUT_TIF_DIR, tif_name)

    sample_out = os.path.join(OUTPUT_ROOT, sample_id)
    os.makedirs(sample_out, exist_ok=True)

    raw_video_path = os.path.join(sample_out, "raw_video.mp4")
    raw_mask_dir = os.path.join(sample_out, "raw_masks")
    os.makedirs(raw_mask_dir, exist_ok=True)

    segmented_video_path = os.path.join(sample_out, "raw_segmented.mp4")

    # STEP 1: Z-STACK → VIDEO
    print("[1] Loading Z-stack...")
    stack = tiff.imread(tif_path)  # (Z, H, W)
    Z, H, W = stack.shape
    print(f"Stack shape: {stack.shape}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        raw_video_path, fourcc, FPS, (W, H), isColor=True)

    print("[2] Converting Z-stack to video...")
    for z in tqdm(range(Z)):
        img = stack[z].astype(np.float32)
        img -= img.min()
        img /= (img.max() + 1e-6)
        img = (img * 255).astype(np.uint8)
        frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        video_writer.write(frame)

    video_writer.release()
    print(f"Saved video: {raw_video_path}")

    # STEP 2: SAM2 INIT
    print("[3] Initializing SAM2 inference...")
    inference_state = predictor.init_state(raw_video_path)

    cap = cv2.VideoCapture(raw_video_path)
    ret, frame0 = cap.read()
    cap.release()

    h, w = frame0.shape[:2]
    point = np.array([[w // 2, h // 2]])
    label = np.array([1])

    predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=point,
        labels=label,)

    print("Generic center-point initialization added")

    # STEP 3: SEGMENTATION
    print("[4] Running SAM2 segmentation...")
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(
        inference_state):
        mask = masks[0].cpu().numpy()
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        mask = (mask > 0).astype(np.uint8)

        np.save(os.path.join(raw_mask_dir, f"raw_frame_{frame_idx:04d}.npy"),mask,)

    print("Masks saved")

    # STEP 4: OVERLAY VIDEO
    print("[5] Creating overlay video...")
    cap = cv2.VideoCapture(raw_video_path)
    out = cv2.VideoWriter(segmented_video_path, fourcc, FPS, (w, h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = np.load(os.path.join(raw_mask_dir, f"raw_frame_{idx:04d}.npy"))

        overlay = frame.copy()
        overlay[mask == 1] = [0, 0, 255]
        blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        out.write(blended)
        idx += 1

    cap.release()
    out.release()
    print(f"Segmented video saved: {segmented_video_path}")
print("\nAll samples processed successfully")
