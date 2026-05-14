
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
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BinarySegDataset(Dataset):
    def __init__(self, split: str):
        self.img_dir = DATASET_DIR / "images" / split
        self.mask_dir = DATASET_DIR / "masks" / split
        self.files = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        name = self.files[idx]        
        img = img = Image.open(str(self.img_dir / name)).convert("RGB")
        
        img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0

        img = torch.from_numpy(img).permute(2, 0, 1)
        
        mask_path = self.mask_dir / name.replace(".jpg", ".png")
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
        mask = np.array(mask, dtype=np.uint8)

        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return img, mask


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

        self.out = nn.Conv2d(64, 1, 1)
    
    
    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))

        x = self.u2(x3)
        x = self.c2(torch.cat([x, x2], dim=1))

        x = self.u1(x)
        x = self.c1(torch.cat([x, x1], dim=1))

        return self.out(x)


def main():
    train_ds = BinarySegDataset("train")
    val_ds   = BinarySegDataset("val")

   
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )


    model = UNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print(f"Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for imgs, masks in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "unet_binary.pth")
    print("✅ Training complete. Model saved to unet_binary.pth")



if __name__ == "__main__":
    main()






