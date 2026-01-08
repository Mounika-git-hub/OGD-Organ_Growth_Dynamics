import os
import cv2
import torch
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image

os.environ["http_proxy"]  = "http://245hsbd012%40ibab.ac.in:Mounik%409201@proxy.ibab.ac.in:3128"
os.environ["https_proxy"] = "http://245hsbd012%40ibab.ac.in:Mounik%409201@proxy.ibab.ac.in:3128"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ZSTACK_PATH = os.path.join(BASE_DIR, "Idisc001.tif")

MASK_ROOT  = os.path.join(BASE_DIR, "clipseg_outputs")
VIDEO_ROOT = os.path.join(BASE_DIR, "clipseg_videos")

os.makedirs(MASK_ROOT, exist_ok=True)
os.makedirs(VIDEO_ROOT, exist_ok=True)

PROMPTS = [
    "bright biological tissue",
    "fluorescent biological structure",
    "largest bright object",
    "main biological structure",
    "dominant object in microscopy image",
    "salient object in grayscale image",
    "continuous bright region",
    "high intensity connected region"
]

FPS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD CLIPSeg MODEL
print("[1] Loading CLIPSeg model...")

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined",use_fast=False)
model = CLIPSegForImageSegmentation.from_pretrained( "CIDAS/clipseg-rd64-refined")
model.to(DEVICE)
model.eval()
print(f"CLIPSeg loaded on {DEVICE}")

# LOAD RAW Z-STACK
print("[2] Loading RAW Z-stack...")
stack = tiff.imread(ZSTACK_PATH)  # (Z, H, W)

Z, H, W = stack.shape
print(f"Z-stack shape: {stack.shape}")

# PROCESS EACH PROMPT
for prompt in PROMPTS:

    prompt_key = prompt.replace(" ", "_").lower()
    print(f"\n[3] Processing prompt: '{prompt}'")

    prompt_mask_dir = os.path.join(MASK_ROOT, prompt_key)
    os.makedirs(prompt_mask_dir, exist_ok=True)

    video_path = os.path.join(VIDEO_ROOT,f"Idisc001_{prompt_key}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, FPS, (W, H))

    # Z-SLICE LOOP
    for z in tqdm(range(Z), desc=f"Segmenting ({prompt_key})"):

        slice_img = stack[z].astype(np.float32)

        # Normalize slice (no preprocessing, just scaling)
        slice_img -= slice_img.min()
        slice_img /= (slice_img.max() + 1e-8)
        slice_img = (slice_img * 255).astype(np.uint8)

        pil_img = Image.fromarray(slice_img).convert("RGB")

        # CLIPSeg inference
        inputs = processor(
            text=prompt,
            images=pil_img,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        # CLIPSeg OUTPUT → RESIZE BACK TO ORIGINAL SIZE
        logits = outputs.logits.squeeze()
        mask = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)

        # IMPORTANT FIX: resize 352×352 → H×W
        mask = cv2.resize(
            mask,
            (W, H),
            interpolation=cv2.INTER_NEAREST
        )

        # Save mask
        np.save(
            os.path.join(prompt_mask_dir, f"slice_{z:03d}.npy"),
            mask
        )

        # Overlay mask for video
        frame = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
        overlay = frame.copy()
        overlay[mask == 1] = [0, 0, 255]

        blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        video.write(blended)

    video.release()
    print(f"Saved video: {video_path}")

print("\nALL PROMPTS COMPLETED SUCCESSFULLY")
