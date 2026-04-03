import torch
import numpy as np
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)

# Reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def linear_beta_schedule(timesteps):
    """
    Generate a linear beta schedule.
    
    Args:
        timesteps (int): Number of timesteps in the schedule.
    
    Returns:
        torch.Tensor: A tensor of beta values.
    """
    beta_start = 1e-4  # Smallest beta value
    beta_end = 2e-2    # Largest beta value
    return torch.linspace(beta_start, beta_end, timesteps)




set_seed(42)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Install confirmation (optional, but helpful)
try:
    import torchvision
    import transformers
    print("All dependencies installed successfully!")
except ImportError as e:
    print(f"Missing dependency: {e}")


# Example usage
timesteps = 1000
beta_schedule = linear_beta_schedule(timesteps)
print(f"First 5 beta values: {beta_schedule[:5]}")


import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg"
import matplotlib.pyplot as plt


from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
from diffusers.models.attention_processor import AttnProcessor2_0

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

##### 1. Replace modules and Register hook #####
pipe = init_pipeline(pipe)
################################################


vae = pipe.vae
unet = pipe.unet
scheduler = pipe.scheduler

#IMAGE_NAME = "resized_test_fire_frame43.jpg"
IMAGE_NAME = "Untitled.jpg"
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


TIMESTEP = 300   # good starting point (try 250–450)

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



def diffseg_probe(image_tensor, timestep=300):
    latents = encode_image(image_tensor)
    noisy_latents = add_noise(latents, timestep)
    _ = run_unet(noisy_latents, timestep)


import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T



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

print("Diffseg probe")
image_tensor = image_to_tensor(IMAGE_NAME)
diffseg_probe(image_tensor)

import torch.nn.functional as F
import matplotlib.pyplot as plt
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
        #if "down_blocks.1" in layer_name:
        #    continue
        total_weight = total_weight + tensor.shape[2]


    for layer_name, tensor in layer_dict.items():
        # attn1_tensor: [B, heads, H, W, HW]
        print("  ", layer_name, tensor.shape)
        if "up_blocks.3" in layer_name:
            continue
        #if "down_blocks.1" in layer_name:
        #    continue
        B, heads, H, W, HW = tensor.shape
        scaling_ratio = H / total_weight
        print(scaling_ratio)
        print(total_weight)
        #get weight ratio
        target_size = 64
        ratio = 64 // H
        attn_mean = tensor.mean(dim=1)  # [B, H, W, HW]
        attn_mean = attn_mean.view(B, H, W, H, W)
        attn_mean = attn_mean[BATCH_NUMBER] # (H, W, H, W)
        import torch.nn.functional as F

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

# Print the min, max, and all sums
print("Anchor map sums:", sums)
print("Min sum:", sums.min().item())
print("Max sum:", sums.max().item())
print("Max deviation from 1:", (sums - 1).abs().max().item())

# See which maps fail the atol=1e-4 tolerance
bad_idx = (sums - 1).abs() > 1e-4
print("Indices of bad maps:", torch.nonzero(bad_idx).squeeze().tolist())
print("Their sums:", sums[bad_idx])

assert anchor_maps.ndim == 3
assert torch.allclose(
    anchor_maps.sum(dim=(1, 2)),
    torch.ones(anchor_maps.shape[0], device=anchor_maps.device, dtype=anchor_maps.dtype),
    atol=1e-4
)

KL_THRESH_1 = 0.4
KL_THRESH = 0.8
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

    i, j = torch.unravel_index(torch.argmin(KL), KL.shape)
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
print(segmentation.shape)
num_maps = segmentation.max().item() + 1
colors = np.random.rand(num_maps, 3)
seg_color = colors[segmentation.numpy()]
seg_color_img = Image.fromarray((seg_color * 255).astype(np.uint8))
seg_color_img = seg_color_img.convert("RGBA")

image_rgba = init_image.convert("RGBA")
if seg_color_img.size != image_rgba.size:
    seg_color_img = seg_color_img.resize(image_rgba.size, resample=Image.BILINEAR)

blended = Image.blend(image_rgba, seg_color_img, alpha=0.5)

fig, axes = plt.subplots(1, 3, figsize=(12, 6))  # 1 row, 2 columns

# Original image
axes[0].imshow(image_rgba)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Segmentation heatmap
axes[1].imshow(seg_color_img)
axes[1].set_title("Segmentation Heatmap")
axes[1].axis("off")

blended = Image.blend(image_rgba, seg_color_img, alpha=0.3)
axes[2].imshow(blended)
axes[2].set_title("Segmentation Heatmap")
axes[2].axis("off")
plt.tight_layout()
plt.show()


















'''

import matplotlib.pyplot as plt

attn_img = combined_map.cpu().numpy() if isinstance(combined_map, torch.Tensor) else combined_map
attn_norm = (attn_img - attn_img.min()) / (attn_img.max() - attn_img.min() + 1e-8)

# Apply colormap
colormap = plt.get_cmap("jet")
attn_color = colormap(attn_norm)[:, :, :3]  # RGB
attn_color = (attn_color * 255).astype(np.uint8)
heatmap_img = Image.fromarray(attn_color).convert("RGBA")

# Original image in RGBA
image_rgba = init_image.convert("RGBA")
if heatmap_img.size != image_rgba.size:
    heatmap_img = heatmap_img.resize(image_rgba.size, resample=Image.BILINEAR)

# Blend and display
blended = Image.blend(image_rgba, heatmap_img, alpha=0.5)
plt.imshow(blended)
plt.axis("off")
plt.show()
'''
'''
for module, attn_tensor in attn_maps.items():
    if attn_tensor is None:
        continue  # skip if nothing captured
    try:    
        attn_per_token = attn_tensor[0, :, token_pos]  # batch 0, head 0, token

        import torch
        import numpy as np
        from PIL import Image, ImageEnhance
        print("attn_tensor.shape:", attn_tensor.shape)
        print("attn_per_token.shape:", attn_per_token.shape)

        # attn_per_token: (latent_seq_len,) tensor
        attn_agg = attn_per_token.mean(dim=0)  # shape: [spatial_dim]
        latent_H = latent_W = int(attn_agg.shape[0] ** 0.5)
        attn_2d = attn_agg.reshape(latent_H, latent_W).cpu().numpy()

        # 2️⃣ normalize to [0, 255]
        attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
        attn_2d = (attn_2d * 255).astype(np.uint8)

        # 3️⃣ convert to PIL grayscale image
        attn_img = Image.fromarray(attn_2d).convert("L")  # 'L' = grayscale

        # 4️⃣ resize to original image size
        attn_img = attn_img.resize(init_image.size, resample=Image.BICUBIC)

        # 5️⃣ optionally apply a colormap using PIL (simple "jet" approximation)
        import matplotlib.pyplot as plt

        colormap = plt.get_cmap("jet")
        attn_color = colormap(np.array(attn_img)/255.0)[:, :, :3]  # RGB
        attn_color = (attn_color * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(attn_color).convert("RGBA")

        # Original image (PIL)
        image = init_image.convert("RGBA")
        heatmap_img = heatmap_img.convert("RGBA")

        # Blend the two
        blended = Image.blend(image, heatmap_img, alpha=0.5)
        plt.imshow(blended)
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(e)
        continue
#        break
'''
