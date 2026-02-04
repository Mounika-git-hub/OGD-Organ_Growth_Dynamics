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

os.environ["http_proxy"]  = "http://245hsbd012%40ibab.ac.in:Mounik%409201@proxy.ibab.ac.in:3128"
os.environ["https_proxy"] = "http://245hsbd012%40ibab.ac.in:Mounik%409201@proxy.ibab.ac.in:3128"

INPUT_TIF_DIR = "/home/ibab/Desktop/OGD/microscopy_data/Deep_Learning_for_image_analysis/"
OUTPUT_ROOT = "outputs_idisc146"
NUM_SAMPLES = 473
START_INDEX = 0
FPS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = ["bright biological tissue",
           # "fluorescent biological structure",
           # "largest bright object",
           # "main biological structure",
           # "dominant object in microscopy image",
           # "salient object in grayscale image",
           # "continuous bright region",
           # "high intensity connected region"
]

def load_clipseg_model():
    print("Loading CLIPSeg model...")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=False)
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}\n")
    return processor, model

def prepare_stack(stack):
    print(f"Original TIFF shape: {stack.shape}")
    # to remove time dimension if present (largest extra dim > Z)
    if stack.ndim == 5:
        print("5D detected removing time dimension (taking first)")
        stack = stack[0]

    # to identify channel dimension (size <= 4 but not H/W)
    if stack.ndim == 4:
        dims = list(stack.shape)
        print(f"4D dims before channel detection: {dims}")

        # Generally Height and width are the two largest dims
        sorted_dims = sorted(dims)
        H_est, W_est = sorted_dims[-2], sorted_dims[-1]

        #if Channel dim is small (<=4) and not H/W
        channel_axis = None
        for i, d in enumerate(dims):
            if d <= 4 and d != H_est and d != W_est:
                channel_axis = i
                break

        if channel_axis is None:
            raise ValueError(f"Could not identify channel dimension in shape {stack.shape}")

        print(f"Channel dimension detected at axis {channel_axis} (size={dims[channel_axis]})  selecting channel 0")
        stack = np.take(stack, indices=0, axis=channel_axis)

    # If still 3D but last dim small, treating as channel
    if stack.ndim == 3 and stack.shape[-1] <= 4:
        print("3D with small last dim  treating as channel")
        stack = stack[..., 0]

    # ensuring final shape is (Z,H,W)
    if stack.ndim == 2:
        print("Single 2D image expanding to Z=1")
        stack = stack[np.newaxis, ...]

    if stack.ndim != 3:
        raise ValueError(f"After processing, stack is not 3D: {stack.shape}")

    print(f"Final stack shape (Z,H,W): {stack.shape}")
    return stack

def process_slice(img, prompt, processor, model):
    pil_img = Image.fromarray(img).convert("RGB")
    inputs = processor(text=prompt, images=pil_img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.squeeze()
    mask = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
    return mask


def create_video_writer(path, W, H):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, FPS, (W, H))


def process_sample(tif_path, sample_id, processor, model):
    print(f"\nProcessing {sample_id}:")

    stack = tiff.imread(tif_path)
    stack = prepare_stack(stack)

    Z, H, W = stack.shape
    print(f"Z={Z}, H={H}, W={W}")

    sample_out = os.path.join(OUTPUT_ROOT, sample_id, "clipseg")
    mask_root = os.path.join(sample_out, "masks")
    video_root = os.path.join(sample_out, "videos")
    os.makedirs(mask_root, exist_ok=True)
    os.makedirs(video_root, exist_ok=True)

    for prompt in PROMPTS:
        prompt_key = prompt.replace(" ", "_").lower()
        print(f"PROMPT: {prompt}")

        prompt_mask_dir = os.path.join(mask_root, prompt_key)
        os.makedirs(prompt_mask_dir, exist_ok=True)

        video_path = os.path.join(video_root, f"{sample_id}_prompt-{prompt_key}.mp4")
        video = create_video_writer(video_path, W, H)

        for z in tqdm(range(Z), desc=f"{sample_id} | {prompt_key}"):
            mask_file = os.path.join(prompt_mask_dir, f"{sample_id}_slice_{z:03d}.npy")
            if os.path.exists(mask_file):
                continue

            try:
                img = stack[z].astype(np.float32)
                img -= img.min()
                img /= (img.max() + 1e-8)
                img = (img * 255).astype(np.uint8)

                mask = process_slice(img, prompt, processor, model)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

                np.save(mask_file, mask)

                frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                overlay = frame.copy()
                overlay[mask == 1] = [0, 0, 255]
                blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                video.write(blended)

            except Exception as e:
                print(f"\nsample={sample_id}, slice={z}, prompt={prompt}")
                print("Reason:", e)

        video.release()
        print(f"Video saved: {video_path}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    processor, model = load_clipseg_model()

    tif_files = sorted(f for f in os.listdir(INPUT_TIF_DIR) if f.lower().endswith(".tif"))
    tif_files = tif_files[START_INDEX:NUM_SAMPLES+START_INDEX]

    print(f"Processing {len(tif_files)} samples starting from index {START_INDEX}\n")
    for tif_name in tif_files:
        sample_id = os.path.splitext(tif_name)[0]
        tif_path = os.path.join(INPUT_TIF_DIR, tif_name)
        process_sample(tif_path, sample_id, processor, model)

if __name__ == "__main__":
    main()
