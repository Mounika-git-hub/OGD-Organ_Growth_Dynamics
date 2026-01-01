import numpy as np
import tifffile as tiff
from tqdm import tqdm
import os

# INPUT PATH 
raw_tif_path = "Idisc001.tif"

# OUTPUT PATH
out_dir = os.path.dirname(raw_tif_path)
out_tif_path = os.path.join(out_dir, "Idisc001_preprocessed.tif")

# LOAD STACK
stack = tiff.imread(raw_tif_path)
print("Raw stack shape:", stack.shape)  # (Z, H, W)

# NORMALIZATION FUNCTION
def normalize_slice(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99))  # for fluorescence scaling
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-6)
    img = (img * 255).astype(np.uint8)
    return img

# PROCESS STACK
processed = np.zeros_like(stack, dtype=np.uint8)

for i in tqdm(range(stack.shape[0]), desc="Preprocessing slices"):
    processed[i] = normalize_slice(stack[i])

# SAVE
tiff.imwrite(out_tif_path, processed)
print("Saved preprocessed stack:", out_tif_path)
