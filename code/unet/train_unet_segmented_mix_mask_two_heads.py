
#!/usr/bin/env python3
"""
Minimal binary U-Net training script.

- Loads images from dataset/images/{train,val}
- Loads binary masks from dataset/masks/{train,val}
- Trains a 1-channel U-Net (hazard vs background)
- Saves model to unet_binary.pth
"""

import os
from PIL import Image
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

DATASET_DIR = Path("dataset")
IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 50
#EPOCHS = 20
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FIRE_COLOR  = np.array([255,  80,  80])
SMOKE_COLOR = np.array([ 80, 160, 255])
MIXED_COLOR = np.array([255, 215,   0])

class FireSmokeDataset(Dataset):
    def __init__(self, split: str):
        self.img_dir = DATASET_DIR / "images" / split
        self.mask_dir = DATASET_DIR / "masks_seg" / split

        # ✅ load all filenames first
        all_files = sorted(os.listdir(self.img_dir))

        MAX_FIRE = 50
        rng = np.random.default_rng(seed=42)

        fire_files = []
        other_files = []

        for f in all_files:
            if "rgb" in f.lower():
                fire_files.append(f)
            else:
                other_files.append(f)

        # ✅ cap fire examples
        if len(fire_files) > MAX_FIRE:
            fire_files = rng.choice(
                fire_files, MAX_FIRE, replace=False
            ).tolist()

        # ✅ final file list
        self.files = other_files + fire_files

        print(
            f"[{split}] using {len(fire_files)} fire + "
            f"{len(other_files)} other images "
            f"(total {len(self.files)})"
        )

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        name = self.files[idx]        
        img = Image.open(str(self.img_dir / name)).convert("RGB")
        
        img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0

        img = torch.from_numpy(img).permute(2, 0, 1)
        
        mask_path = self.mask_dir / name.replace(".jpg", ".png")
        mask = Image.open(mask_path).convert("RGB")
        mask = mask.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
        mask_np = np.array(mask)

        fire  = np.zeros(mask_np.shape[:2], dtype=np.float32)
        smoke = np.zeros(mask_np.shape[:2], dtype=np.float32)

        fire_pixels  = np.all(mask_np == FIRE_COLOR,  axis=-1)
        smoke_pixels = np.all(mask_np == SMOKE_COLOR, axis=-1)
        mixed_pixels = np.all(mask_np == MIXED_COLOR, axis=-1)
        valid_mask = (~mixed_pixels).astype(np.float32)

        fire  = fire_pixels.astype(np.float32)
        smoke = smoke_pixels.astype(np.float32)


        from scipy.ndimage import binary_dilation

        FIRE_BUFFER = 5  # pixels; start with 1–3

        fire_pixels  = np.all(mask_np == FIRE_COLOR,  axis=-1)
        smoke_pixels = np.all(mask_np == SMOKE_COLOR, axis=-1)
        mixed_pixels = np.all(mask_np == MIXED_COLOR, axis=-1)

        # fire adjacency (boundary)
        fire_buffer = binary_dilation(fire_pixels, iterations=FIRE_BUFFER)
        fire_ring   = fire_buffer & ~fire_pixels
        
        # -------- targets --------
        fire  = fire_pixels.astype(np.float32)
        smoke = smoke_pixels.astype(np.float32)
        target = torch.from_numpy(np.stack([fire, smoke], axis=0))

        # -------- validity masks --------
        valid_fire  = (~mixed_pixels & ~fire_ring).astype(np.float32)
        valid_smoke = (~mixed_pixels & ~fire_ring).astype(np.float32)

        valid_fire  = torch.from_numpy(valid_fire)
        valid_smoke = torch.from_numpy(valid_smoke)

        # IMPORTANT: zero labels where invalid
        target[0] *= valid_fire
        target[1] *= valid_smoke

        return img, target, valid_fire, valid_smoke





class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.net(x)



class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        
        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c2 = DoubleConv(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c1 = DoubleConv(128, 64)

        self.fire_head  = nn.Conv2d(64, 1, 1)
        self.smoke_head = nn.Conv2d(64, 1, 1)
    
    
    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))

        x = self.u2(x3)
        x = self.c2(torch.cat([x, x2], dim=1))

        x = self.u1(x)
        x = self.c1(torch.cat([x, x1], dim=1))
        
        fire_logits  = self.fire_head(x)
        smoke_logits = self.smoke_head(x)
        return fire_logits, smoke_logits




import torch
import torch.nn as nn

def main():
    train_ds = FireSmokeDataset("train")
    val_ds   = FireSmokeDataset("val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )


    
    model = UNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print(f"Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for imgs, targets, valid_fire, valid_smoke in train_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            valid_fire = valid_fire.to(DEVICE)
            valid_smoke = valid_smoke.to(DEVICE)

            fire_logits, smoke_logits = model(imgs)

            fire_logits  = fire_logits.squeeze(1)
            smoke_logits = smoke_logits.squeeze(1)
            fire_target  = targets[:, 0] * valid_fire
            smoke_target = targets[:, 1] * valid_smoke

            fire_loss_map  = criterion(fire_logits,  fire_target)
            smoke_loss_map = criterion(smoke_logits, smoke_target)


            # Fire loss (channel 0)
                        
            fire_loss = (fire_loss_map * valid_fire).sum() / (valid_fire.sum() + 1e-6)
            smoke_loss = (smoke_loss_map * valid_smoke).sum() / (valid_smoke.sum() + 1e-6)

            loss = 0.5 * (fire_loss + smoke_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "unet_segmenter_mask_mixed.pth")
    print("✅ Training complete. Model saved to unet_segmenter.pth")



if __name__ == "__main__":
    main()






