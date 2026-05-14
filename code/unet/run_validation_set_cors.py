import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from code.unet.train_unet_binary import UNet, IMG_SIZE, DEVICE

# ---------------- PATHS ----------------
MODEL_PATH = "unet_binary.pth"

BASE_DIR = Path("/home/josbel/Corsican_Fire_DB")
OUT_DIR = Path("corsican_val_plots")
OUT_DIR.mkdir(exist_ok=True)
# --------------------------------------


# ---------------- LOAD MODEL ----------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
# ------------------------------------------


# authoritative GT list
gt_files = sorted(BASE_DIR.glob("*_gt.png"))
print(f"Found {len(gt_files)} GT masks")

count = 0

for gt_path in gt_files:
    stem = gt_path.stem.replace("_gt", "")  # e.g. "540"
    img_path = BASE_DIR / f"{stem}_rgb.png"

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
        prob = torch.sigmoid(model(img_t))[0, 0].cpu().numpy()

    pred = prob > 0.5  # <-- adjust threshold here if needed

    # ---- load GT mask ----
    gt = Image.open(gt_path).convert("L")
    gt = gt.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
    gt = np.array(gt) > 0

    
    print("Plotting")

    plt.figure(figsize=(16, 4))
    plt.suptitle(stem)

    # --- 1. RGB ---
    plt.subplot(1, 5, 1)
    plt.title("RGB")
    plt.imshow(img_resized)
    plt.axis("off")

    # --- 2. GT ---
    plt.subplot(1, 5, 2)
    plt.title("GT")
    plt.imshow(gt, cmap="gray")
    plt.axis("off")

    # --- 3. Probability heatmap ---
    plt.subplot(1, 5, 3)
    plt.title("U-Net Prob")
    plt.imshow(prob, cmap="inferno")
    plt.colorbar(fraction=0.046)
    plt.axis("off")

    # --- 4. RGB + probability overlay ---
    plt.subplot(1, 5, 4)
    plt.title("RGB + Prob")
    plt.imshow(img_resized)
    plt.imshow(prob, cmap="inferno", alpha=0.5)  # <--- overlay
    plt.axis("off")

    # --- 5. RGB + binary prediction overlay ---
    plt.subplot(1, 5, 5)
    plt.title("RGB + Pred")
    plt.imshow(img_resized)
    plt.imshow(pred, cmap="Reds", alpha=0.5)  # <--- overlay
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()

    count += 1

print(f"✅ Plotted {count} samples")

