import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from code.unet.train_unet_segmented_mix_mask_two_heads import UNet, IMG_SIZE, DEVICE

# ---------------- PATHS ----------------
#MODEL_PATH = "unet_segmenter_smoke.pth"
MODEL_PATH = "unet_segmenter_mask_mixed.pth"


if False:
        
    BASE_DIR = Path("/home/josbel/Corsican_Fire_DB")
    IMAGE_DIR = BASE_DIR
    OUT_DIR = Path("corsican_val_plots")
    OUT_DIR.mkdir(exist_ok=True)
    gt_files = sorted(BASE_DIR.glob("*_gt.png"))
else:


    # --------------------------------------
    IMAGE_DIR = Path(
        "/home/josbel/Boreal-Forest-Fire/"
        "Boreal-Forest-Fire-Subset-C/images/test"
    )

    GT_MASK_DIR = Path(
        "/home/josbel/Boreal-Forest-Fire/"
        "Boreal-Forest-Fire-Subset-C/manual_masks/test"
    )
    gt_files = sorted(GT_MASK_DIR.glob("*.png"))





# ---------------- LOAD MODEL ----------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
# ------------------------------------------


# authoritative GT list
print(f"Found {len(gt_files)} GT masks")

count = 0

for gt_path in gt_files:
    if False:
        stem = gt_path.stem.replace("_gt", "")  # e.g. "540"
        img_path = BASE_DIR / f"{stem}_rgb.png"
    else:
        stem = gt_path.stem
        img_path = IMAGE_DIR / f"{stem}.jpg"

    if not img_path.exists():
        continue

    # ---- load image ----
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    
    img_t = (
        torch.from_numpy(img_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # ---- inference ----
    with torch.no_grad():
        fire_logits, smoke_logits = model(img_t)  # two outputs

    fire_prob  = torch.sigmoid(fire_logits[0, 0]).cpu().numpy()
    smoke_prob = torch.sigmoid(smoke_logits[0, 0]).cpu().numpy()


    T_fire  = 0.6
    T_smoke = 0.4

    fire_mask  = fire_prob  > T_fire
    smoke_mask = smoke_prob > T_smoke

    hazard_mask = fire_mask | smoke_mask


    # ---- load GT mask ----
    gt = Image.open(gt_path).convert("L")
    gt = gt.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
    gt = np.array(gt) > 0

    
    
    print("Plotting")

    plt.figure(figsize=(20, 4))
    plt.suptitle(stem)

    # --- 1. RGB ---
    plt.subplot(1, 6, 1)
    plt.title("RGB")
    plt.imshow(img_resized)
    plt.axis("off")

    # --- 2. GT ---
    plt.subplot(1, 6, 2)
    plt.title("GT")
    plt.imshow(gt, cmap="gray")
    plt.axis("off")

    # --- 3. Fire probability ---
    plt.subplot(1, 6, 3)
    plt.title("Fire Prob")
    plt.imshow(fire_prob, cmap="hot")
    plt.colorbar(fraction=0.046)
    plt.axis("off")

    # --- 4. Smoke probability ---
    plt.subplot(1, 6, 4)
    plt.title("Smoke Prob")
    plt.imshow(smoke_prob, cmap="Blues")
    plt.colorbar(fraction=0.046)
    plt.axis("off")

    # --- 5. RGB + Fire ---
    plt.subplot(1, 6, 5)
    plt.title("RGB + Fire")
    plt.imshow(img_resized)
    plt.imshow(fire_mask, cmap="Reds", alpha=0.5)
    plt.axis("off")

    # --- 6. RGB + Smoke ---
    plt.subplot(1, 6, 6)
    plt.title("RGB + Smoke")
    plt.imshow(img_resized)
    plt.imshow(smoke_mask, cmap="Blues", alpha=0.5)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()

    count += 1

print(f"✅ Plotted {count} samples")

