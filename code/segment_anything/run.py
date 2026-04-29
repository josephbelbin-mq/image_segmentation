
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pathlib import Path

# --- SETTINGS ---
repo_root = Path.cwd()
#IMAGE_PATH = repo_root / "data" / "test_images" / "Untitled.jpg"
IMAGE_PATH = repo_root / "data" / "test_images" / "resized_test_fire_frame43.jpg"

SAM_CHECKPOINT =  repo_root / "externals" / "sam_vit_h_4b8939.pth"        # path to SAM checkpoint
DEVICE = "cuda"                                 # or "cpu"

# --- LOAD IMAGE ---
init_image = Image.open(IMAGE_PATH).convert("RGB")

# --- INITIALIZE SAM ---
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam.to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

# --- GENERATE MASKS ---
image_np = np.array(init_image)
masks = mask_generator.generate(image_np)
print(f"Generated {len(masks)} masks")

# --- CREATE COLOR OVERLAY ---
seg_color_img_array = np.zeros((init_image.height, init_image.width, 3), dtype=np.uint8)

for mask_dict in masks:
    mask = mask_dict["segmentation"]          # H x W boolean
    color = (np.random.rand(3) * 255).astype(np.uint8)
    seg_color_img_array[mask] = color

# Convert to PIL RGBA for blending
seg_color_img = Image.fromarray(seg_color_img_array).convert("RGBA")
image_rgba = init_image.convert("RGBA")
blended = Image.blend(image_rgba, seg_color_img, alpha=0.3)

# --- DISPLAY 3-PANEL PLOT ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image_rgba)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Segmentation heatmap
axes[1].imshow(seg_color_img)
axes[1].set_title("Segmentation Heatmap")
axes[1].axis("off")

# Blended overlay
axes[2].imshow(blended)
axes[2].set_title("Blended Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.show()
