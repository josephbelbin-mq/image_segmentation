import torch
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pathlib import Path
import argparse

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from x_segment_anything import sam_model_registry, SamPredictor
from torchvision.ops import nms
import s3fs
import boto3
from smart_open import open as smart_open
import json


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
processor = None
model = None
device = None

def init(
    device: str = "cuda",
    dtype: str = "float16",
):
    global processor, model
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    model.eval()

def load_and_segment(
    boxes_json: str,
    device: str = "cuda",
    width: int = 512,
    height: int = 512,
    clipseg_threshold: float = 0.2,
    clipseg_text_prompt: str = "fire. flame. smoke.",
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
    
    with open(boxes_json, "r") as f:
        data = json.load(f)

    image_path = data["image"]
    results = data["results"]
    
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

    # -------- Get masks --------
    from collections import defaultdict
    label_masks = defaultdict(list)

    print(f"ClipSeg")
    clipseg_text_prompt = "fire. flame."
    inputs = processor(images=image, text="fire. flame.", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits    
    # Convert logits → probability heatmap
    heatmap = torch.sigmoid(logits)

    # Remove batch / channel dims until H×W
    while heatmap.dim() > 2:
        heatmap = heatmap[0]

    w, h = image.size
    heatmap = torch.nn.functional.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )[0, 0]



    mu = heatmap.mean()
    sigma = heatmap.std() + 1e-6
    z = (heatmap - mu) / sigma

    clipseg_mask = (z > -0.25).cpu().numpy()



    #Threshold
    #clipseg_mask = (heatmap > 0.01).cpu().numpy()
    clipseg_masks = []
    sam_masks = []
    
    def clipseg_diffuse(candidate):
        regime = candidate["regime"]
        label = candidate["label"]
        score = candidate["score"]
        return regime == "diffuse"

    for r in results:
        if clipseg_diffuse(r):    
            box = r["box"]
            label = r["label"]
            score = r["score"]

            H, W = clipseg_mask.shape
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)

            mask_boxed = np.zeros_like(clipseg_mask, dtype=bool)
            mask_boxed[y1:y2, x1:x2] = clipseg_mask[y1:y2, x1:x2]
            print(f"Diffuse: {label}")

            if "smoke" in label:
                label = "mixed" if label != "smoke" else "smoke"
            else:
                label = "fire"

            label_masks[label].append(mask_boxed)
        else:
            box = r["box"]
            label = r["label"]
            score = r["score"]
            sam_masks.append((label, score, box))

    print(f"SAM")
    # -------- Load SAM --------
    sam_checkpoint = repo_root / "externals" / "mobile_sam.pt"
    sam_model = sam_model_registry["vit_t"](checkpoint=sam_checkpoint)
    sam_model.to(device)
    predictor = SamPredictor(sam_model)
    predictor.set_image(np.array(image))


    for i, (label, score, box) in enumerate(sam_masks):
        masks, scores, logits = predictor.predict(box=np.array(box, dtype=np.float32))
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
        print(f"{label} label added")
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

    
    print("fire :", None if fire is None else fire.shape)
    print("smoke:", None if smoke is None else smoke.shape)
    print("mixed:", None if mixed is None else mixed.shape)

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
        "label_map": label_map * (clipseg_mask > 0),
        #"binary_mask": (label_map > 0),
        #"binary_mask": (clipseg_mask > 0),
        "binary_mask": (label_map > 0) & (clipseg_mask > 0),
        "palette": PALETTE,
        "image_path": image_path
    }


def main():
    parser = argparse.ArgumentParser(description="Load an image and generate masks with SAM.")

    # Image and device
    
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Path to boxes JSON (if provided, skip detection and segment from JSON)"
    )

    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (cuda or cpu)")

    # Image resizing
    parser.add_argument("--width", type=int, default=0, help="Resize image width - by default no resizing")
    parser.add_argument("--height", type=int, default=0, help="Resize image height - by default no resizign")

    # Model parameters
    parser.add_argument("--clipseg_threshold", type=float, default=0.2, help=" threshold")
    parser.add_argument("--clipseg_text_prompt", type=str, default="fire. flame. smoke.", help="Text prompt")
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

    def collect_jsons(path_str, profile=None):
        path = Path(path_str).expanduser().resolve()

        valid_exts = {".json"}

        if str(path_str).startswith("s3://"):
            fs = s3fs.S3FileSystem(profile=profile) if profile else s3fs.S3FileSystem()
            path = path_str.replace("s3://", "")

            if fs.isfile(path):
                return [f"s3://{path}"]

            if fs.isdir(path):
                files = fs.ls(path)
                return sorted(
                    f"s3://{f}" for f in files
                    if f.lower().endswith(".json")
                )

            raise ValueError(f"Invalid S3 path: {path_str}")

        else:
            if path.is_file():
                return [str(path)]

            if path.is_dir():
                return sorted(
                    str(p) for p in path.iterdir()
                    if p.suffix.lower() == ".json"
                )

            raise ValueError(f"Invalid path: {path}")

    jsons = collect_jsons(args.json_path, args.aws_profile)

    pdf = PdfPages(args.pdf_path) if args.save_pdf else None

    from datetime import datetime
    today = datetime.today()

    # Format as string in YYYY-MM-DD
    date_str = today.strftime("%Y%m%d")
    output_dir = repo_root / args.output_dir / date_str
    import os
    os.makedirs(output_dir, exist_ok=True)

    for json_path in jsons:
        import time
        t0 = time.perf_counter()

        #print(f"Processing {image_path}")
        # Load and segment
        res = load_and_segment(
            boxes_json=json_path,
            device=args.device,
            width=args.width,
            height=args.height,
            clipseg_threshold=args.clipseg_threshold,
            clipseg_text_prompt=args.clipseg_text_prompt,
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
                image_path=res["image_path"],
            )

        if pdf is not None:
            pdf.savefig(fig)

        if args.show:
            plt.show()
            

if __name__ == "__main__":
    main()

