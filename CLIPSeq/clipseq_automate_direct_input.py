"""
Automated CLIPSeg pipeline:
- Multiple Z-stack .tif inputs
- Multiple prompts
- Sample-safe naming
- Mask + video outputs per sample
"""

import os
import cv2
import torch
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image


# os.environ["http_proxy"]  = "http://245hsbd012%40ibab.ac.in:Mounik%409201@proxy.ibab.ac.in:3128"
# os.environ["https_proxy"] = "http://245hsbd012%40ibab.ac.in:Mounik%409201@proxy.ibab.ac.in:3128"

# PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TIF_DIR = (
    "/home/ibab/Downloads/Deep_Learning_for_image_analysis/"
    "Deep learning for image analysis-20250926T061103Z-1-001/"
    "Deep learning for image analysis")
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

NUM_SAMPLES = 5
FPS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PROMPTS
PROMPTS = [
    "bright biological tissue",
    # "fluorescent biological structure",
    # "largest bright object",
    # "main biological structure",
    # "dominant object in microscopy image",
    # "salient object in grayscale image",
    # "continuous bright region",
    # "high intensity connected region",
]

# LOAD CLIPSeg
print("[1] Loading CLIPSeg model...")
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=False)
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.to(DEVICE)
model.eval()
print(f"CLIPSeg loaded on {DEVICE}\n")

# COLLECT INPUT FILES
tif_files = sorted(
    f for f in os.listdir(INPUT_TIF_DIR)
    if f.lower().endswith(".tif")
)[:NUM_SAMPLES]

print("Samples to process:")
for f in tif_files:
    print("  -", f)

for tif_name in tif_files:
    sample_id = os.path.splitext(tif_name)[0]
    print(f"Processing sample: {sample_id}")

    tif_path = os.path.join(INPUT_TIF_DIR, tif_name)

    sample_out = os.path.join(OUTPUT_ROOT, sample_id, "clipseg")
    mask_root = os.path.join(sample_out, "masks")
    video_root = os.path.join(sample_out, "videos")

    os.makedirs(mask_root, exist_ok=True)
    os.makedirs(video_root, exist_ok=True)

    # LOAD Z-STACK
    stack = tiff.imread(tif_path)
    Z, H, W = stack.shape
    print(f"Z-stack shape: {stack.shape}")

    # PROCESS PROMPTS
    for prompt in PROMPTS:
        prompt_key = prompt.replace(" ", "_").lower()
        print(f"\n[CLIPSeg] Prompt: {prompt}")

        prompt_mask_dir = os.path.join(mask_root, prompt_key)
        os.makedirs(prompt_mask_dir, exist_ok=True)

        video_path = os.path.join(
            video_root,
            f"{sample_id}_prompt-{prompt_key}.mp4",)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_path, fourcc, FPS, (W, H))

        # SLICE LOOP
        for z in tqdm(range(Z), desc=f"{sample_id} | {prompt_key}"):

            img = stack[z].astype(np.float32)
            img -= img.min()
            img /= (img.max() + 1e-8)
            img = (img * 255).astype(np.uint8)

            pil_img = Image.fromarray(img).convert("RGB")

            inputs = processor(
                text=prompt,
                images=pil_img,
                return_tensors="pt",
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits.squeeze()
            mask = (
                torch.sigmoid(logits) > 0.5
            ).cpu().numpy().astype(np.uint8)

            mask = cv2.resize(
                mask, (W, H), interpolation=cv2.INTER_NEAREST)

            # SAVE MASK (sample-safe)
            np.save(
                os.path.join(
                    prompt_mask_dir,
                    f"{sample_id}_slice_{z:03d}.npy",),
                mask,
            )

            # OVERLAY VIDEO
            frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            overlay = frame.copy()
            overlay[mask == 1] = [0, 0, 255]
            blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            video.write(blended)

        video.release()
        print(f"Saved video: {video_path}")

print("\nALL CLIPSeg SAMPLES AND PROMPTS COMPLETED")
