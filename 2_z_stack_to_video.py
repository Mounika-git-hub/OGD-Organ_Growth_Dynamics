import os
import json
import cv2
import numpy as np
import tifffile as tiff
import pandas as pd
from tqdm import tqdm


# USER INPUTS
# Sample name (must match Excel)
SAMPLE_ID = "Idisc001"

# Input Z-stack (TIFF)
ZSTACK_PATH = "Idisc001_preprocessed.tif"

# Excel file containing voxel sizes
VOXEL_EXCEL_PATH = "Voxelsize.xlsx"

# Output directory
OUTPUT_DIR = "output1"

# Video settings
FPS = 5
VIDEO_NAME = "Idisc001_video.mp4"
METADATA_NAME = "Idisc001_metadata.json"

# CREATE OUTPUT DIRECTORY
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_PATH = os.path.join(OUTPUT_DIR, VIDEO_NAME)
METADATA_PATH = os.path.join(OUTPUT_DIR, METADATA_NAME)

# LOAD VOXEL SIZE FROM EXCEL
df = pd.read_excel(VOXEL_EXCEL_PATH, header=None)

row = df[df.iloc[:, 0] == SAMPLE_ID]
if row.empty:
    raise ValueError(f"Sample {SAMPLE_ID} not found in Excel file")

voxel_size = {
    "x": float(row.iloc[0, 1]),
    "y": float(row.iloc[0, 2]),
    "z": float(row.iloc[0, 3]),}

print("Voxel size loaded:", voxel_size)

# LOAD Z-STACK
stack = tiff.imread(ZSTACK_PATH)
print("Z-stack shape (Z, H, W):", stack.shape)

# NORMALIZATION FUNCTION
def normalize(slice_img):
    slice_img = slice_img.astype(np.float32)
    slice_img -= slice_img.min()
    slice_img /= (slice_img.max() + 1e-8)
    slice_img *= 255
    return slice_img.astype(np.uint8)

# INITIALIZE VIDEO WRITER (3-CHANNEL!)
height, width = stack.shape[1], stack.shape[2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(
    VIDEO_PATH,
    fourcc,
    FPS,
    (width, height),
    isColor=True
)


# WRITE VIDEO FRAMES
print("Converting slices to video...")
for slice_img in tqdm(stack):
    gray = normalize(slice_img)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # REQUIRED
    video.write(frame)

video.release()
print(f"Video saved: {VIDEO_PATH}")


# SAVE METADATA
metadata = {
    "sample_id": SAMPLE_ID,
    "voxel_size_um": voxel_size,
    "z_slices": int(stack.shape[0]),
    "height_px": int(height),
    "width_px": int(width),
    "fps": FPS
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Metadata saved: {METADATA_PATH}")
