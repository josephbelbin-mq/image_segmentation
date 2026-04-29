import torch
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pathlib import Path
import argparse

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from x_segment_anything import sam_model_registry, SamPredictor
from torchvision.ops import nms
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


repo_root = Path.cwd()
#FILE_NAME = "resized_test_fire_frame189.jpg"
FILE_NAME = "Fire3.jpg"
IMAGE_NAME = repo_root / "data" / "test_images" / FILE_NAME
#FILE_NAME = "KNP-backburning-5.jpeg"
#FILE_NAME = "D4935-1.jpg"
#FILE_NAME = "OIP.jpg"
#FILE_NAME = "OIP_3.jpg"

processor = None
model = None
device = None

def apply_nms(candidates, iou_thresh=0.4):
    if len(candidates) == 0:
        return []

    boxes = np.stack([c[2] for c in candidates])
    scores = np.array([c[1] for c in candidates])

    keep = nms(
        torch.tensor(boxes),
        torch.tensor(scores),
        iou_thresh
    )

    return [candidates[i] for i in keep]

def tile_box(box, H, W, tile_size=256, stride=192):
    """
    Splits a large box into overlapping tiles.
    """
    x1, y1, x2, y2 = map(int, box)
    tiles = []
    for y in range(y1, y2, stride):
        for x in range(x1, x2, stride):
            tx1 = x
            ty1 = y
            tx2 = x + tile_size
            ty2 = y + tile_size

            # clip if image shape provided
            tx2 = min(tx2, W)
            ty2 = min(ty2, H)

            # skip tiny invalid tiles
            if tx2 - tx1 < 20 or ty2 - ty1 < 20:
                continue

            tiles.append([tx1, ty1, tx2, ty2])

    return tiles

def init(
    device: str = "cuda",
    dtype: str = "float16",
):
    global processor, model
    model_id = "IDEA-Research/grounding-dino-tiny"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    model.eval()

def load_and_segment(
    image_path: str,
    device: str = "cuda",
    width: int = 512,
    height: int = 512,
    second_pass: bool = False,
    threshold: float = 0.2,
    text_threshold: float = 0.2,
    second_pass_threshold: float = 0.4,
    second_pass_text_threshold: float = 0.3,
    text_prompt: str = "fire. flame.",
    dtype: str = "float16",
):
    image_path = Path(image_path).expanduser().resolve()
    image = Image.open(image_path)

    def resize_shortest_side(image, target=512):
        w, h = image.size

        # determine scale so shortest side becomes target
        scale = target / min(w, h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        return image.resize((new_w, new_h), Image.BICUBIC)
    if (width > 0 and height > 0):
        print("Resizing")
        image = resize_shortest_side(image, 1024)

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # shape: (1, H, W)
    print(logits.shape)
    heatmap_fire = torch.sigmoid(logits)[0]
    # Post-process boxes
    #heatmap_fire  = logits[0]
    #heatmap_flame  = logits[1]
    #heatmap_smoke = logits[2]
    
    heatmap = heatmap_fire
    #Threshold
    mask = heatmap > 0.1  # e.g. 0.4–0.7

    #cluster
    from sklearn.cluster import DBSCAN
    coords = torch.nonzero(mask).cpu().numpy()
    import numpy as np
    coords = np.stack([coords[:, 1], coords[:, 0]], axis=1)
    clustering = DBSCAN(eps=5, min_samples=5).fit(coords)
    labels = clustering.labels_

    from collections import defaultdict
    import numpy as np

    clusters = defaultdict(list)

    for (x, y), label in zip(coords, labels):
        if label == -1:
            continue
        clusters[label].append((x, y))
    
    points = []

    for label, pts in clusters.items():
        pts = np.array(pts)

        xs = pts[:, 0]
        ys = pts[:, 1]

        weights = heatmap[ys, xs]
        weights = weights.detach().cpu().numpy()
        cx = np.sum(xs * weights) / np.sum(weights)
        cy = np.sum(ys * weights) / np.sum(weights)

        points.append((cx, cy))

    orig_w, orig_h = image.size
    h, w = heatmap.shape

    scale_x = orig_w / w
    scale_y = orig_h / h

    points_image_space = [
        (int(x * scale_x), int(y * scale_y))
        for (x, y) in points
    ]
    input_points = np.array(points_image_space)
    input_labels = np.ones(len(input_points))  # all positive points

    print(f"SAM")
    # -------- Load SAM --------
    sam_checkpoint = repo_root / "externals" / "mobile_sam.pt"
    sam_model = sam_model_registry["vit_t"](checkpoint=sam_checkpoint)
    sam_model.to(device)
    predictor = SamPredictor(sam_model)
    predictor.set_image(np.array(image))

    filtered = []
    filtered.append(("smoke", input_points, input_labels))

    from datetime import datetime
    today = datetime.today()

    # Format as string in YYYY-MM-DD
    date_str = today.strftime("%Y%m%d")
    output_dir = repo_root / "results" / date_str
    import os
    os.makedirs(output_dir, exist_ok=True)

    # -------- Get masks --------
    from collections import defaultdict
    label_masks = defaultdict(list)
    for i, (label, input_points, input_labels) in enumerate(filtered):
        masks, scores, logits = predictor.predict(point_coords=input_points,
            point_labels=input_labels)
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        label_masks[label].append(mask)

    merged_masks = {}
    for label, masks_list in label_masks.items():
        #print(f"{label} label added")
        merged = np.logical_or.reduce(masks_list)
        merged_masks[label] = merged

    w, h = image.size
    def safe_mask(mask):
        if mask is None:
            return np.zeros((h, w), dtype=bool)
        return mask > 0
    
    fire  = safe_mask(merged_masks.get("fire"))
    smoke  = safe_mask(merged_masks.get("smoke"))
    mixed  = safe_mask(merged_masks.get("mixed"))

    
    # count how many classes are active per pixel
    stack = np.stack([fire, smoke, mixed], axis=0)
    count = stack.sum(axis=0)

    FIRE, SMOKE, MIXED = 1, 2, 3

    label_map = np.zeros((h, w), dtype=np.uint8)

    # single-class pixels
    label_map[(fire & (count == 1))] = FIRE
    label_map[(smoke & (count == 1))] = SMOKE

    # everything else becomes mixed:
    # - overlaps
    # - explicit mixed
    # - any conflict
    label_map[count >= 2] = MIXED
    label_map[mixed] = MIXED

    COLORS = {
        1: (255, 80, 80),     # fire (warm red, not pure red)
        2: (80, 160, 255),    # smoke (cool blue)
        3: (255, 215, 0),     # mixed (gold/yellow)
    }

    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    for k, color in COLORS.items():
        overlay[label_map == k] = color

    if not isinstance(image, np.ndarray):
        image_np = np.array(image)
    else:
        image_np = image
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: original
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Right: overlay
    alpha = 0.4
    blended = image_np.copy().astype(float)

    mask = (overlay > 0)
    blended[mask] = (
        (1 - alpha) * blended[mask] +
        alpha * overlay[mask]
    )

    blended = blended.astype(np.uint8)

    axes[1].imshow(blended)
    axes[1].set_title("Segmentation")
    axes[1].axis("off")

    fig.tight_layout()
    return fig

    for label, mask in merged_masks.items():
        dpi = 100
        figsize = (w / dpi, h / dpi)
        # -------- Visualization --------
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5, cmap="Reds", extent=(0, w, h, 0))
        #x1, y1, x2, y2 = box

        #import matplotlib.patches as patches
        #rect = patches.Rectangle(
        #    (x1, y1),
        #    x2 - x1,
        #    y2 - y1,
        #    fill=False,          # no patch fill
        #    edgecolor="blue",
        #    linewidth=2
        #)
        #plt.gca().add_patch(rect)
        plt.title(f"Fire instance {label} {i+1}")
        plt.axis("off")
        plt.show()

        # Save the plot as an image (PNG format)
        outname = FILE_NAME.split('.')[0]+ f"_mask_{i}.jpg"
        output_path = output_dir / outname
        #plt.savefig(output_path, dpi=dpi, bbox_inches='tight')  # High resolution

def main():
    parser = argparse.ArgumentParser(description="Load an image and generate masks with SAM.")

    # Image and device
    parser.add_argument("--image_path", type=str, default=IMAGE_NAME, help="Path to input image")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (cuda or cpu)")

    # Image resizing
    parser.add_argument("--width", type=int, default=0, help="Resize image width - by default no resizing")
    parser.add_argument("--height", type=int, default=0, help="Resize image height - by default no resizign")

    # Model parameters
    parser.add_argument("--second_pass", action="store_true", help="enable second pass mode")
    parser.add_argument("--threshold", type=float, default=0.2, help=" threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text labelling threshold")
    parser.add_argument("--second_pass_threshold", type=float, default=0.4, help=" threshold")
    parser.add_argument("--second_pass_text_threshold", type=float, default=0.3, help="text labelling threshold")
    parser.add_argument("--text_prompt", type=str, default="fire. flame. smoke.", help="Text prompt")
    parser.add_argument("--dtype", type=str, default="float16", help="Torch dtype (float16, float32)")

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype.lower(), torch.float16)

    init(device=args.device, dtype=dtype)

    def collect_images(path_str):
        path = Path(path_str).expanduser().resolve()

        valid_exts = {
            ".jpg", ".jpeg", ".png", ".bmp",
            ".tiff", ".tif", ".webp"
        }

        if path.is_file():
            return [path]

        if path.is_dir():
            # grab all jpg/jpeg files (case-insensitive)
            return sorted(p for p in path.iterdir() if p.suffix.lower() in valid_exts)

        raise ValueError(f"Invalid path: {path}")

    images = collect_images(args.image_path)

    with PdfPages("output.pdf") as pdf:
        for image_path in images:
            import time
            t0 = time.perf_counter()

            #print(f"Processing {image_path}")
            # Load and segment
            fig = load_and_segment(
                image_path=image_path,
                device=args.device,
                width=args.width,
                height=args.height,
                second_pass=args.second_pass,
                threshold=args.threshold,
                text_threshold=args.text_threshold,
                second_pass_threshold=args.second_pass_threshold,
                second_pass_text_threshold=args.second_pass_text_threshold,
                text_prompt=args.text_prompt,
                dtype=dtype,
            )
            t1 = time.perf_counter()
            print(f"processing: {t1 - t0:.3f}s")
            plt.show()
            #pdf.savefig(fig)
            plt.close(fig)

if __name__ == "__main__":
    main()
