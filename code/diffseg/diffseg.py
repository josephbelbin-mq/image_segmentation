import torch
import numpy as np
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)

import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg"
import matplotlib.pyplot as plt
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
from diffusers.models.attention_processor import AttnProcessor2_0
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Defines:
set_seed(42)
#IMAGE_NAME = "resized_test_fire_frame43.jpg"
repo_root = Path.cwd()
#IMAGE_NAME = repo_root / "data" / "test_images" / "Untitled.jpg"
IMAGE_NAME = repo_root / "data" / "test_images" / "KNP-backburning-5.jpeg"

KL_THRESH_1 = 0.4
KL_THRESH = 0.8
TIMESTEP = 300   # good starting point (try 250–450)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

##### 1. Replace modules and Register hook #####
pipe = init_pipeline(pipe)

vae = pipe.vae
unet = pipe.unet
scheduler = pipe.scheduler

init_image = Image.open(IMAGE_NAME).convert("RGB")
init_image = init_image.resize((512, 512), Image.BICUBIC)
print("Loaded image size:", init_image.size)  # e.g., (512, 512)

@torch.no_grad()
def encode_image(image_tensor):
    """
    image_tensor: [1, 3, 512, 512] in [-1, 1]
    """
    latents = vae.encode(image_tensor).latent_dist.sample()
    latents = 0.18215 * latents   # SD latent scaling
    return latents

@torch.no_grad()
def add_noise(latents, timestep):
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(
        latents,
        noise,
        torch.tensor([timestep], device=latents.device)
    )
    return noisy_latents

@torch.no_grad()
def run_unet(noisy_latents, timestep):
    # Dummy text embedding (unused because we want self-attn only)
    batch_size = noisy_latents.shape[0]
    encoder_hidden_states = torch.zeros(
        (batch_size, 77, unet.config.cross_attention_dim),
        device=noisy_latents.device,
        dtype=noisy_latents.dtype,
    )
    '''
    from transformers import CLIPTokenizer, CLIPTextModel

    # Tokenize your prompt
    prompt = "fire"
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )

    # Encode the tokens to get embeddings
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder.to(noisy_latents.device)

    with torch.no_grad():
        encoder_hidden_states = text_encoder(text_inputs.input_ids.to(noisy_latents.device))[0]  # [1, 77, 768]
    '''

    out = unet(
        noisy_latents,
        torch.tensor([timestep], device=noisy_latents.device),
        encoder_hidden_states=encoder_hidden_states,
        return_dict=True
    )

    return out

def diffseg_probe(image_tensor, timestep=TIMESTEP):
    latents = encode_image(image_tensor)
    noisy_latents = add_noise(latents, timestep)
    _ = run_unet(noisy_latents, timestep)

def image_to_tensor(
    image_path,
    device="cuda",
    width=512,
    height=512,
    dtype=torch.float16,
):
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize (Stable Diffusion expects multiples of 8; 512×512 is standard)
    image = image.resize((width, height), Image.BICUBIC)

    # Convert to tensor [0, 1]
    transform = T.ToTensor()
    image = transform(image)  # [3, H, W], float32 in [0,1]

    # Normalize to [-1, 1]
    image = image * 2.0 - 1.0

    # Add batch dimension
    image = image.unsqueeze(0)  # [1, 3, H, W]

    # Move to device + dtype
    image = image.to(device=device, dtype=dtype)

    return image

def merge_regions(kl_map, anchor_maps):
    n = kl_map.size(0) #64
    N, H, W = anchor_maps.shape #64, 8, 8
    adj = (kl_map < KL_THRESH_1) #64 x 64
    adj.fill_diagonal_(False)

    merged_maps = anchor_maps.clone()             

    
    for _ in range(1):
        # Step 1: merge maps for each anchor
        new_maps = torch.zeros_like(merged_maps)
        for i in range(n):
            neighbors = adj[i].nonzero(as_tuple=True)[0]  # indices of neighbors
            if len(neighbors) > 0:
                new_maps[i] = merged_maps[neighbors].mean(dim=0)
                new_maps[i] = new_maps[i] / new_maps[i].sum()
            else:
                new_maps[i] = merged_maps[i]
        merged_maps = new_maps

    return merged_maps

def find_min_kl_pair(maps, KL_map):
    N, H, W = maps.shape
    maps_flat = maps.view(N, -1).clamp_min(1e-8)
    log_maps = maps_flat.log()

    for i in range(N):
        A = maps_flat[i]                    # [HW]
        logA = log_maps[i]
        B = maps_flat
        logB = log_maps
        # KL(A || B) for all B
        KL_mapA = (A * (logA - logB)).sum(dim=1)  # [N]
        KL_mapB = (B * (logB - logA)).sum(dim=1)  # [N]
        KL_map[i] = (KL_mapA + KL_mapB) / 2

    KL_map.fill_diagonal_(float("inf"))

    i, j = torch.unravel_index(torch.argmin(KL_map), KL_map.shape)
    minval = KL_map[i, j]
    return (i, j), minval, KL_map


def merge_maps(maps, i, j, KL_map):
    if i > j:
        i, j = j, i
    merged = (maps[i] + maps[j]) / 2
    merged = merged / merged.sum()
    merged = merged.clamp_min(1e-8)

    idx = torch.arange(maps.size(0))
    idx = idx[(idx != i) & (idx != j)]  # indices to keep
    maps_remaining = maps[idx]
    KL_map = KL_map[idx][:, idx]

    maps_new = torch.cat([maps_remaining, merged.unsqueeze(0)], dim=0)
    return maps_new, KL_map

def find_min_kl_pair_incremental(maps_new, KL_map):
    N, H, W = maps_new.shape
    maps_flat = maps_new.view(N, -1).clamp_min(1e-8)
    log_maps = maps_flat.log()

    # get new row
    A = maps_flat[-1]
    logA = log_maps[-1]
    KL_new_row = (A * (logA - log_maps)).sum(dim=1)  # [N]
    KL_new_row = KL_new_row[:-1]
    KL_new_col = (maps_flat * (log_maps - logA)).sum(dim=1)  # [N_remaining]
    KL_new_col[-1] = float("inf")
    KL_new_col = KL_new_col.unsqueeze(1)
    KL_map = torch.cat([KL_map, KL_new_row.unsqueeze(0)], dim=0)  # row
    KL_map = torch.cat([KL_map, KL_new_col], dim=1)               # column

    i, j = torch.unravel_index(torch.argmin(KL_map), KL_map.shape)
    minval = KL_map[i, j]
    return (i, j), minval, KL_map

def load_and_segment(
    image_path = IMAGE_NAME,
    device="cuda",
    width=512,
    height=512,
    kl_thresh0 = KL_THRESH_1,
    kl_thresh = KL_THRESH,
    timestep = TIMESTEP,
    dtype=torch.float16,
    method="unused"
):
    print("Diffseg probe")
    image_tensor = image_to_tensor(    image_path,
    device,
    width,
    height,
    dtype)

    diffseg_probe(image_tensor)

    combined_map = None

    BATCH_NUMBER = 0 #Can change to 1
    kl_maps = []
    for t, layer_dict in attn_maps.items():
        aggre_weights = torch.zeros((64, 64, 64, 64), dtype=torch.float32)
        weights = []
        total_weight = 0
        for layer_name, tensor in layer_dict.items():
            # attn1_tensor: [B, heads, H, W, HW]
            if "up_blocks.3" in layer_name:
                continue
            if "down_blocks.1" in layer_name:
                continue
            total_weight = total_weight + tensor.shape[2]


        for layer_name, tensor in layer_dict.items():
            if "up_blocks.3" in layer_name:
                continue
            if "down_blocks.1" in layer_name:
                continue
            B, heads, H, W, HW = tensor.shape
            scaling_ratio = H / total_weight
            #get weight ratio
            target_size = 64
            ratio = 64 // H
            attn_mean = tensor.mean(dim=1)  # [B, H, W, HW]
            attn_mean = attn_mean.view(B, H, W, H, W)
            attn_mean = attn_mean[BATCH_NUMBER] # (H, W, H, W)

            attn_mean = attn_mean / attn_mean.sum(dim=(2,3), keepdim=True)
            keys = attn_mean.clamp_min(1e-8)

            keys_upsampled = F.interpolate(
                keys.view(H*W, 1, H, W),  # merge batch+query dims for interpolation
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False
            )
            keys_upsampled = keys_upsampled.view(H, W, target_size, target_size)
            keys_tiled = keys_upsampled.repeat_interleave(ratio, dim=0).repeat_interleave(ratio, dim=1)
            # Aggrgate accroding to weight_ratio
            keys_tiled = keys_tiled / keys_tiled.sum(dim=(2,3), keepdim=True) 
            aggre_weights += keys_tiled * scaling_ratio

        print(aggre_weights.shape)

    H, W, Hk, Wk = aggre_weights.shape
    qi = torch.linspace(0, H-1, steps=8).long()
    qj = torch.linspace(0, W-1, steps=8).long()

    anchor_maps = aggre_weights[qi][:, qj]
    anchor_maps = anchor_maps.reshape(-1, Hk, Wk)

    sums = anchor_maps.sum(dim=(1, 2))

    assert anchor_maps.ndim == 3
    assert torch.allclose(
        anchor_maps.sum(dim=(1, 2)),
        torch.ones(anchor_maps.shape[0], device=anchor_maps.device, dtype=anchor_maps.dtype),
        atol=1e-4
    )

    print("Running Diffseg")
    N, H, W = anchor_maps.shape
    KL = torch.empty((N, N), device=anchor_maps.device)

    KL = generate_KL_map(anchor_maps, KL)
    anchor_maps = merge_regions(KL, anchor_maps)

    ((i, j), val, KL) = find_min_kl_pair(anchor_maps, KL)
    for _ in range(0, 10230):
        print(f"Number of maps: {anchor_maps.shape[0]}. KL size: {KL.shape}. merging({i}, {j})")
        anchor_maps, KL = merge_maps(anchor_maps, i, j, KL)
        ((i, j), val, KL) = find_min_kl_pair_incremental(anchor_maps, KL)
        print(f"Merged! Number of maps: {anchor_maps.shape[0]}. KL size: {KL.shape} minval={val}")
        if val > KL_THRESH:
            break


    segmentation = torch.argmax(anchor_maps, dim=0)
    segmentation_np = segmentation.numpy()
    num_maps = segmentation_np.max() + 1
    masks = []

    for label in range(num_maps):
        mask = segmentation_np == label         # H x W boolean mask
        if mask.sum() == 0:
            continue                            # skip empty masks
        # Bounding box
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        masks.append({
            "segmentation": mask,
            "area": int(mask.sum()),
            "bbox": bbox,
            "predicted_iou": None,        # placeholder
            "stability_score": None       # placeholder
        })

    image_rgba = init_image.convert("RGBA")
    
    # Create a blank segmentation color image
    seg_color_img_array = np.zeros((init_image.height, init_image.width, 3), dtype=np.uint8) 
    def resize_mask(mask: np.ndarray, target_size):
        """
        Upscale a boolean mask to match target_size (width, height)
        """
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize(target_size, resample=Image.NEAREST)
        return np.array(mask_img) > 0

    # Assign random color to each mask
    for mask_dict in masks:
        mask = mask_dict["segmentation"]  # boolean HxW
        mask_resized = resize_mask(mask, (init_image.width, init_image.height))
        color = (np.random.rand(3) * 255).astype(np.uint8)
        seg_color_img_array[mask_resized] = color

    # Convert to PIL RGBA directly
    seg_color_img = Image.fromarray(seg_color_img_array).convert("RGBA")

    # Blend
    blended = Image.blend(image_rgba, seg_color_img, alpha=0.3) 

    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_rgba)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(seg_color_img)
    axes[1].set_title("Segmentation Heatmap")
    axes[1].axis("off")
    
    axes[2].imshow(blended)
    axes[2].set_title("Blended Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()

    return masks

def generate_KL_map(maps, KL_map):
    N, H, W = maps.shape
    maps_flat = maps.view(N, -1).clamp_min(1e-8)
    log_maps = maps_flat.log()

    for i in range(N):
        A = maps_flat[i]                    # [HW]
        logA = log_maps[i]
        B = maps_flat
        logB = log_maps
        # KL(A || B) for all B
        KL_mapA = (A * (logA - logB)).sum(dim=1)  # [N]
        KL_mapB = (B * (logB - logA)).sum(dim=1)  # [N]
        KL_map[i] = (KL_mapA + KL_mapB) / 2

    KL_map.fill_diagonal_(float("inf"))
    return KL_map


def main():
    parser = argparse.ArgumentParser(description="Load an image and generate masks with DiffSeg or SAM.")

    # Image and device
    parser.add_argument("--image_path", type=str, default=IMAGE_NAME, help="Path to input image")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (cuda or cpu)")

    # Image resizing
    parser.add_argument("--width", type=int, default=512, help="Resize image width")
    parser.add_argument("--height", type=int, default=512, help="Resize image height")

    # Model parameters
    parser.add_argument("--kl_thresh0", type=float, default=KL_THRESH_1, help="Initial KL threshold")
    parser.add_argument("--kl_thresh", type=float, default=KL_THRESH, help="KL threshold")
    parser.add_argument("--timestep", type=int, default=TIMESTEP, help="Time step for segmentation")
    parser.add_argument("--dtype", type=str, default="float16", help="Torch dtype (float16, float32)")

    # Segmentation method
    parser.add_argument("--method", type=str, choices=["sam", "diffseg"], default="diffseg", help="Segmentation method")

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype.lower(), torch.float16)

    # Load and segment
    masks = load_and_segment(
        image_path=args.image_path,
        device=args.device,
        width=args.width,
        height=args.height,
        kl_thresh0=args.kl_thresh0,
        kl_thresh=args.kl_thresh,
        timestep=args.timestep,
        dtype=dtype,
        method=args.method
    )

    # Optional: print summary
    print(f"Generated {len(masks)} masks for {args.image_path}")
    for i, m in enumerate(masks):
        print(f"Mask {i}: area={m['area']}, bbox={m['bbox']}")


if __name__ == "__main__":
    main()
