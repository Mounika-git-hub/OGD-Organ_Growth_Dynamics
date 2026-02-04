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
from sam2.build_sam import build_sam2_video_predictor

INPUT_TIF_DIR = "/path/to/tif_directory"
OUTPUT_ROOT = "outputs"
NUM_SAMPLES = 5
FPS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_sam2_model(ogd_root):
    print("\nLoading SAM2 model...")
    predictor = build_sam2_video_predictor(
        config_file="configs/sam2.1/sam2.1_hiera_t.yaml",
        ckpt_path=os.path.join(ogd_root, "SAM2", "checkpoints", "sam2.1_hiera_tiny.pt"),
        device=DEVICE,)
    print("SAM2 loaded successfully\n")
    return predictor

def zstack_to_video(tif_path, video_path, fps):
    print("Loading Z-stack...")
    stack = tiff.imread(tif_path)  # (Z, H, W)
    Z, H, W = stack.shape
    print(f"Stack shape: {stack.shape}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H), isColor=True)

    print("Converting Z-stack to video...")
    for z in tqdm(range(Z)):
        img = stack[z].astype(np.float32)
        # Normalize slice to [0,255]
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = (img * 255).astype(np.uint8)
        frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        video_writer.write(frame)
    video_writer.release()
    print(f"Saved video: {video_path}")
    return Z, H, W

def initialize_sam2(predictor, video_path):
    print("Initializing SAM2 inference...")
    inference_state = predictor.init_state(video_path)
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    cap.release()
    h, w = frame0.shape[:2]
    # Center point prompt
    point = np.array([[w // 2, h // 2]])
    label = np.array([1])

    predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=point,
        labels=label,)
    print("Center-point initialization added")
    return inference_state

def run_sam2_segmentation(predictor, inference_state, mask_dir):
    print("Running SAM2 segmentation...")
    os.makedirs(mask_dir, exist_ok=True)
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
        mask = masks[0].cpu().numpy()
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        mask = (mask > 0).astype(np.uint8)
        np.save(os.path.join(mask_dir, f"raw_frame_{frame_idx:04d}.npy"), mask)
    print("Masks saved")

def create_overlay_video(video_path, mask_dir, output_path, fps, width, height):
    print("Creating overlay video...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = np.load(os.path.join(mask_dir, f"raw_frame_{idx:04d}.npy"))
        overlay = frame.copy()
        overlay[mask == 1] = [0, 0, 255]  # red mask
        blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        out.write(blended)
        idx += 1
    cap.release()
    out.release()
    print(f"Segmented video saved: {output_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ogd_root = os.path.abspath(os.path.join(base_dir, ".."))
    sys.path.append(ogd_root)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    predictor = load_sam2_model(ogd_root)
    tif_files = sorted(f for f in os.listdir(INPUT_TIF_DIR) if f.lower().endswith(".tif"))[:NUM_SAMPLES]

    print(f"Processing {len(tif_files)} samples:")
    for f in tif_files:
        print("  -", f)

    for tif_name in tif_files:
        sample_id = os.path.splitext(tif_name)[0]
        print(f"\nProcessing: {sample_id}")

        tif_path = os.path.join(INPUT_TIF_DIR, tif_name)
        sample_out = os.path.join(OUTPUT_ROOT, sample_id)
        os.makedirs(sample_out, exist_ok=True)

        raw_video_path = os.path.join(sample_out, "raw_video.mp4")
        mask_dir = os.path.join(sample_out, "raw_masks")
        segmented_video_path = os.path.join(sample_out, "raw_segmented.mp4")

        Z, H, W = zstack_to_video(tif_path, raw_video_path, FPS)
        inference_state = initialize_sam2(predictor, raw_video_path)
        run_sam2_segmentation(predictor, inference_state, mask_dir)
        create_overlay_video(raw_video_path, mask_dir, segmented_video_path, FPS, W, H)

if __name__ == "__main__":
    main()
