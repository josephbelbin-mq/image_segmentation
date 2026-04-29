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
import s3fs
import boto3
from smart_open import open as smart_open


from code.utils.image_io import (
    save_all_outputs,
    save_binary_mask,
    save_overlay,
)


from code.utils.image_viz import (
    view_image,
    view_color_map,
    view_blended,
    view_binary_mask,
    make_segmentation_figure
)

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

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

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
    text_prompt: str = "fire. flame. smoke.",
    dtype: str = "float16",
    aws_profile: str = "default",
):
    def load_image(image_path, profile="default"):
        if str(image_path).startswith("s3://"):
            session = boto3.Session(profile_name=profile)
            client = session.client("s3")

            with smart_open(image_path, "rb", transport_params={"client": client}) as f:
                return Image.open(f).convert("RGB")
        else:
            image_path = Path(image_path).expanduser().resolve()
            return Image.open(image_path).convert("RGB")
    print(f"Loading {image_path}")
    image = load_image(image_path, aws_profile)

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

    # Post-process boxes
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )

    H, W = image.size[::-1]
    
    img_area = H * W
    filtered_boxes = []
    filtered_scores = []

    boxes = [r["boxes"].cpu().numpy() for r in results]
    scores = [r["scores"].cpu().numpy() for r in results]
    labels = [r["text_labels"] for r in results]

    filtered = []
    tiled = []
    candidates = []
    tiling = True
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area < 0.4 * H * W:
            filtered.append((label, score, box))
        else: 
            if label == "smoke" and score < 0.6:
                continue
            # 2nd pass candidates 
            candidates.append((label, score, box))
    if tiling:
        filtered_candidates = apply_nms(candidates, iou_thresh=0.4)

        tile = max(H, W) // 3
        #stride = max(H, W) // 4 
        stride = tile 
        for label, score, box in filtered_candidates:
            boxes = tile_box(box, H, W, tile, stride)
            for b in boxes:
                tiled.append((label, score, np.array(b, dtype=np.float32)))
    else:
        tiled = candidates

    #print(f"Number of tiles = {len(tiled)}")
    ###### 2nd pass DINO ######
    if (second_pass):
        scd_pass_boxes = []
        all_images = []
        all_boxes = []
        for i, (label, score, box) in enumerate(tiled):
            print(f"2nd pass {i}: label={label}, score={score:.3f}, box={box}")
            x1, y1, x2, y2 = box

            w = (x2 - x1)
            h = (y2 - y1)
            #expand 10%
            w = w * 1.1
            h = h * 1.1
            a = w * h
            #print(f"2nd pass {i}: x1={x1} x2 = {x2} y1={y1} y2={y2} w={w} h={h} area={a}")
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            x1 = max(cx - (w / 2), 0)
            x2 = min(cx + (w / 2), W)
            y1 = max(cy - (h / 2), 0)
            y2 = min(cy + (h / 2), H)
            expanded = (x1, y1, x2, y2)
            cropped = image.crop(expanded)

            w, h = cropped.size
            cropped = cropped.resize((w // 2, h // 2), resample=Image.BILINEAR)
            all_images.append(cropped)
            all_boxes.append((x1, y1, x2, y2))
        batch_size = 2  # start small (1–4 typical)

        results = []
        print(f"2nd pass DINO - {len(all_images)}")
        for i in range(0, len(all_images), batch_size):
            #images = cropped
            images = all_images[i:i+batch_size]
            inputs = processor(
                images=images,
                text=[text_prompt] * len(images),
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=second_pass_threshold,
                text_threshold=second_pass_text_threshold,
                target_sizes=[img.size[::-1] for img in images],
                #target_sizes=[cropped.size[::-1]],
            )
            for j, r in enumerate(results):
                boxes = r["boxes"].cpu().numpy()
                scores = r["scores"].cpu().numpy()
                labels = r["text_labels"]
                x1, y1, x2, y2 = all_boxes[i+j]
                for box, score, label in zip(boxes, scores, labels):
                    #box *= 2
                    #a1, b1, a2, b2 = box
                    #area = (a2 - a1) * (b2 - b1)
                    #if area >= 0.5 * h * w:
                    #    print(f"TOO BIG a1={a1} a2={a2} b1={b1} b2={b2} area={area}")
                    #    continue
                    if not label:
                        continue
                    if label == "smoke" and score < 0.6:
                        continue
                    shift = np.array([x1, y1, x1, y1])
                    box = box * 2
                    box = box + shift
                    print(f"Appending: label={label}, score={score:.3f}, box={box}")
                    scd_pass_boxes.append((label, score, box))

        filtered = filtered + scd_pass_boxes


    print(f"SAM")
    # -------- Load SAM --------
    sam_checkpoint = repo_root / "externals" / "mobile_sam.pt"
    sam_model = sam_model_registry["vit_t"](checkpoint=sam_checkpoint)
    sam_model.to(device)
    predictor = SamPredictor(sam_model)
    predictor.set_image(np.array(image))

    # -------- Get masks --------
    from collections import defaultdict
    label_masks = defaultdict(list)
    for i, (label, score, box) in enumerate(filtered):
        masks, scores, logits = predictor.predict(box=box)
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        if (label == "smoke"):
            label = "smoke"
        elif ("smoke" in label):
            label = "mixed"
        else:
            label = "fire"
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

    PALETTE = {
        FIRE: ("Fire", (255, 80, 80)),     # fire (warm red, not pure red)
        SMOKE: ("Smoke",(80, 160, 255)),    # smoke (cool blue)
        MIXED: ("Mixed", (255, 215, 0)),     # mixed (gold/yellow)
    }

    overlay = np.zeros((h, w, 3), dtype=np.uint8)


    if not isinstance(image, np.ndarray):
        image_np = np.array(image)
    else:
        image_np = image

    return {
        "image": image_np,
        "label_map": label_map,
        "binary_mask": (label_map > 0),
        "palette": PALETTE,
    }

    for k, color in COLORS.items():
        overlay[label_map == k] = color

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: original
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Right: overlay
    alpha = 0.4
    blended = image_np.copy().astype(float)

    blended[mask] = (
        (1 - alpha) * blended[mask] +
        alpha * overlay[mask]
    )

    blended = blended.astype(np.uint8)

    axes[1].imshow(blended)
    axes[1].set_title("Segmentation")
    axes[1].axis("off")

    fig.tight_layout()
    outname = (image_path.split('/')[-1]).split('.')[0] + f"_mask_pred.png"
    output_path = output_dir / outname
    mask_uint8 = mask.any(axis=2).astype(np.uint8) * 255
    Image.fromarray(mask_uint8).save(output_path)
    return fig

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
    parser.add_argument("--aws_profile", type=str, default="ieee-dataport", help="AWS Profile")
    parser.add_argument("--save_images", action="store_true",
                        help="Save image outputs (mask, segmentation, overlay)")
    parser.add_argument("--save_pdf", action="store_true",
                        help="Save segmentation figures to a PDF")
    parser.add_argument("--show", action="store_true",
                        help="Show results interactively")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save image outputs")
    parser.add_argument("--pdf_path", type=str, default="output.pdf",
                        help="Path to output PDF")

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype.lower(), torch.float16)

    init(device=args.device, dtype=dtype)

    def collect_images(path_str, profile=None):
        path = Path(path_str).expanduser().resolve()

        valid_exts = {
            ".jpg", ".jpeg", ".png", ".bmp",
            ".tiff", ".tif", ".webp"
        }

        if str(path_str).startswith("s3://"):
            fs = s3fs.S3FileSystem(profile=profile) if profile else s3fs.S3FileSystem()

            # strip s3://
            path = path_str.replace("s3://", "")

            # list objects under prefix
            if fs.isfile(path):
                return [f"s3://{path}"]

            if fs.isdir(path):
                files = fs.ls(path)
                return sorted(
                    f"s3://{f}" for f in files
                    if any(f.lower().endswith(ext) for ext in valid_exts)
                )

            raise ValueError(f"Invalid S3 path: {path_str}")

        else:
            path = Path(path_str).expanduser().resolve()
            if path.is_file():
                return [str(path)]

            if path.is_dir():
                # grab all jpg/jpeg files (case-insensitive)
                return sorted(str(p) for p in path.iterdir() if p.suffix.lower() in valid_exts)

            raise ValueError(f"Invalid path: {path}")

    images = collect_images(args.image_path, args.aws_profile)


    pdf = PdfPages(args.pdf_path) if args.save_pdf else None

    from datetime import datetime
    today = datetime.today()

    # Format as string in YYYY-MM-DD
    date_str = today.strftime("%Y%m%d")
    output_dir = repo_root / args.output_dir / date_str
    import os
    os.makedirs(output_dir, exist_ok=True)

    for image_path in images:
        import time
        t0 = time.perf_counter()

        #print(f"Processing {image_path}")
        # Load and segment
        res = load_and_segment(
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
            aws_profile = args.aws_profile
        )
        t1 = time.perf_counter()
        print(f"processing: {t1 - t0:.3f}s")


        fig = make_segmentation_figure(res)

        if args.save_images:
            save_all_outputs(
                image=res["image"],
                label_map=res["label_map"],
                binary_mask=res["binary_mask"],
                palette=res["palette"],
                out_dir=output_dir,
                image_path=image_path,
            )

        if pdf is not None:
            pdf.savefig(fig)

        if args.show:
            plt.show()
            

if __name__ == "__main__":
    main()
