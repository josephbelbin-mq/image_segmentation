
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

#from code.unet.train_unet_binary import UNet, IMG_SIZE, DEVICE
from code.unet.train_unet_segmented import UNet, IMG_SIZE, DEVICE
# ---------------- PATHS ----------------
#MODEL_PATH = "unet_binary.pth"
MODEL_PATH = "unet_segmenter_smoke.pth"


IMAGE_DIR = Path(
    "/home/josbel/Boreal-Forest-Fire/"
    "Boreal-Forest-Fire-Subset-C/images/test"
)

GT_MASK_DIR = Path(
    "/home/josbel/Boreal-Forest-Fire/"
    "Boreal-Forest-Fire-Subset-C/manual_masks/test"
)
# --------------------------------------


# ---------------- LOAD MODEL ----------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
# ------------------------------------------


# iterate over GT masks (authoritative list)
gt_files = sorted(GT_MASK_DIR.glob("*.png"))

print(f"Found {len(gt_files)} GT masks")

ious = []
skipped = 0

for gt_path in gt_files:
    stem = gt_path.stem  # e.g. evoDJI_0001_frame123

    # --- find corresponding image ---
    img_path = IMAGE_DIR / f"{stem}.jpg"

    if not img_path.exists():
        print(f"⚠️ Image missing for {stem}, skipping")
        continue

    # --- load image ---
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_t = (
        torch.from_numpy(img_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --- inference ---
    with torch.no_grad():
        prob = torch.sigmoid(model(img_t))[0, 0].cpu().numpy()

    pred = prob > 0.5

    # --- load GT mask ---
    gt = Image.open(gt_path).convert("L")
    gt = gt.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
    gt = np.array(gt) > 0  # fire pixels only


    gt_fire_pixels = gt.sum()
    if gt_fire_pixels == 0:
        continue  # nothing to evaluate

    # ---- fire-only IoU ----
    intersection = np.logical_and(pred, gt).sum()
    iou_fire = intersection / gt_fire_pixels

    ious.append(iou_fire)

    if False:
        # --- visualize ---
        plt.figure(figsize=(14, 3))
        plt.suptitle(stem)

        plt.subplot(1, 4, 1)
        plt.title("Image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Human GT (Fire)")
        plt.imshow(gt, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("U-Net Probability")
        plt.imshow(prob, cmap="inferno")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title("U-Net Prediction")
        plt.imshow(pred, cmap="gray")
        plt.axis("off")

        plt.show()

if ious:
    print(f"Validated samples : {len(ious)}")
    print(f"Skipped samples   : {skipped}")
    print(f"Fire IoU (mean)   : {np.mean(ious):.4f}")
    print(f"Fire IoU (median) : {np.median(ious):.4f}")
    print(f"Fire IoU (min)    : {np.min(ious):.4f}")
    print(f"Fire IoU (max)    : {np.max(ious):.4f}")
else:
    print("❌ No valid samples evaluated")

