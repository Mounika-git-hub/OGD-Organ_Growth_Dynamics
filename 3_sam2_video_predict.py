import os
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Preprocessing_Training",
    "output1",
    "Idisc001_video.mp4"
)

OUTPUT_MASK_DIR = os.path.join(
    BASE_DIR,
    "..",
    "Preprocessing_Training",
    "sam2_outputs"
)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# SAM2 CONFIG AND CHECKPOINT
CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "..",
    "SAM2",
    "checkpoints",
    "sam2.1_hiera_tiny.pt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD SAM2 VIDEO PREDICTOR
predictor = build_sam2_video_predictor(
    config_file=CONFIG_NAME,
    ckpt_path=CHECKPOINT_PATH,
    device=DEVICE,
)

print("SAM2 video predictor loaded successfully")

# INIT INFERENCE STATE WITH VIDEO PATH
inference_state = predictor.init_state(VIDEO_PATH)
print("Video loaded and inference state initialized")

# SINGLE-POINT PROMPT (LSO)
# Add center click on first frame
frame_idx = 0
obj_id = 1
point = np.array([[0, 0]])  # placeholder, will be interpreted by predictor
label = np.array([1])       # foreground

predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=frame_idx,
    obj_id=obj_id,
    points=point,
    labels=label
)

# RUN VIDEO PROPAGATION
for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
    mask = masks[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8)

    np.save(
        os.path.join(
            OUTPUT_MASK_DIR,
            f"frame_{frame_idx:04d}.npy"
        ),
        mask
    )

    print(f"Saved mask for frame {frame_idx}")

print(" video segmentation COMPLETED")
