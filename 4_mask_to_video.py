import os
import cv2
import numpy as np

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Original video
VIDEO_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Preprocessing_Training",
    "output1",
    "Idisc001_video.mp4"
)

# Folder where CODE-3 saved masks
MASK_DIR = os.path.join(
    BASE_DIR,
    "..",
    "Preprocessing_Training",
    "sam2_outputs"
)

# Output video with masks overlay
OUTPUT_VIDEO_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Preprocessing_Training",
    "sam2_outputs",
    "Idisc001_segmented.mp4"
)

# READ ORIGINAL VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f" Cannot open video: {VIDEO_PATH}"

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()
frames = np.array(frames)
num_frames, H, W, _ = frames.shape
print(f"Loaded {num_frames} frames from original video")

# LOAD MASKS
mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith(".npy")])
assert len(mask_files) == num_frames, " Number of masks doesn't match number of frames"

masks = [np.load(os.path.join(MASK_DIR, f)) for f in mask_files]

# CREATE VIDEO WRITER
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 5.0, (W, H))  # Adjust fps if needed

# OVERLAY MASKS ON FRAMES AND WRITE VIDEO
for i, frame in enumerate(frames):
    mask = masks[i]
    mask_colored = np.zeros_like(frame)
    mask_colored[:, :, 2] = mask * 255  # Red channel overlay

    overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)
    out.write(overlay)

out.release()
print(f" Segmented video saved at: {OUTPUT_VIDEO_PATH}")
