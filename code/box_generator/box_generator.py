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

from code.utils.image_viz import (
    make_detection_figure
)

repo_root = Path.cwd()

processor = None
model = None
device = None
import json

def save_results_json(res, out_dir, image_path):
    """
    res: {
        "image": image (ignored here),
        "results": list of dicts with keys:
            - label
            - score
            - box
            - regime
            - context_box (optional)
    }
    """

    stem = Path(image_path).stem
    out_path = Path(out_dir) / f"{stem}.json"

    serializable = {
        "image": str(image_path),
        "results": []
    }

    for r in res["results"]:
        entry = {
            "label": r["label"],
            "regime": r["regime"],
            "score": None if r.get("score") is None else float(r["score"]),
            "box": [float(x) for x in r["box"]] if r.get("box") is not None else None,
        }

        # include context_box only if present
        if r.get("context_box") is not None:
            entry["context_box"] = [float(x) for x in r["context_box"]]

        serializable["results"].append(entry)

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)


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


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


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

def generate_boxes(
    image_path: str,
    device: str = "cuda",
    width: int = 512,
    height: int = 512,
    threshold: float = 0.2,
    text_threshold: float = 0.2,
    second_pass_threshold = 0.4,
    second_pass_text_threshold = 0.4,
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


    results_out = []
    small_boxes = []
    large_boxes = []
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area < 0.33 * H * W:            
            small_boxes.append((label, score, box))
        else: 
            large_boxes.append((label, score, box))
    
    small_boxes = apply_nms(small_boxes, 0.7)
    for i, (label, score, box) in enumerate(small_boxes):
        results_out.append({
                    "label": label,
                    "score": float(score) if score is not None else None,
                    "box": [float(x) for x in box],          # geometry only
                    "regime": "concentrated",    # or "concentrated"
                    "context_box": None                 # filled only if concentrated
                })

    large_boxes = apply_nms(large_boxes)

    #Rerun DINO on large_boxes
    for i, (label, score, box) in enumerate(large_boxes):
        
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
        #cropped = cropped.resize((w // 2, h // 2), resample=Image.BILINEAR)
        
        inputs = processor(images=cropped, text=text_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        sub_results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=second_pass_threshold,
            text_threshold=second_pass_text_threshold,
            target_sizes=[cropped.size[::-1]],
        )

        
        refined_boxes = []

        sub_boxes  = sub_results[0]["boxes"].cpu().numpy()
        sub_scores = sub_results[0]["scores"].cpu().numpy()
        sub_labels = sub_results[0]["text_labels"]

        for sb, ss, sl in zip(sub_boxes, sub_scores, sub_labels):
            print(f"    2nd pass detection {i}: label={sl}, score={ss:.3f}, box={sb}")
            bx1, by1, bx2, by2 = sb
            bx1 += x1; bx2 += x1
            by1 += y1; by2 += y1
            refined_boxes.append((sl, ss, np.array([bx1, by1, bx2, by2])))

        
        small_boxes = []
        large_boxes_2 = []
        for sl, ss, sb in refined_boxes:
            area_ratio = box_area(sb) / box_area(box)
            if area_ratio < 0.33:      # refinement exists
                small_boxes.append((sl, ss, sb))
            else:
                large_boxes_2.append((sl, ss, sb))

        for sl, ss, sb in small_boxes:
            # ✅ object-mode
            results_out.append({
                "label": sl,
                "score": float(ss) if ss is not None else None,
                "box": [float(x) for x in sb],
                "regime": "concentrated",    # or "concentrated"
                "context_box": [float(x) for x in box]
            })

        if not small_boxes or large_boxes_2:
            # ✅ diffuse smoke still present
            results_out.append({
                "label": label,
                "score": float(score) if score is not None else None,
                "box": [float(x) for x in box],          # geometry only
                "regime": "diffuse",    # or "concentrated"
                "context_box": None
            })

    res = {
        "image": image,
        "results": results_out
    }
    return res



def main():
    parser = argparse.ArgumentParser(description="Load an image and generate masks with SAM.")

    # Image and device
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (cuda or cpu)")

    # Image resizing
    parser.add_argument("--width", type=int, default=0, help="Resize image width - by default no resizing")
    parser.add_argument("--height", type=int, default=0, help="Resize image height - by default no resizign")

    # Model parameters
    parser.add_argument("--threshold", type=float, default=0.2, help=" threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text labelling threshold")
    parser.add_argument("--second_pass_threshold", type=float, default=0.4, help=" threshold")
    parser.add_argument("--second_pass_text_threshold", type=float, default=0.3, help="text labelling threshold")
    parser.add_argument("--text_prompt", type=str, default="fire. flame. smoke.", help="Text prompt")
    parser.add_argument("--dtype", type=str, default="float16", help="Torch dtype (float16, float32)")
    parser.add_argument("--aws_profile", type=str, default="ieee-dataport", help="AWS Profile")
    parser.add_argument("--save_images", action="store_true",
                        help="Save image outputs (mask, segmentation, overlay)")
    parser.add_argument("--save_json", action="store_true",
                        help="Save json outputs")
                        
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
        res = generate_boxes(
            image_path=image_path,
            device=args.device,
            width=args.width,
            height=args.height,
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
        
        from rich import print_json

        print_json(data=res["results"])
        fig = make_detection_figure(res["image"], res["results"])
        if args.save_images:
            fig.savefig(output_dir / f"{Path(image_path).stem}_det.png", dpi=150)
        if args.save_json:
            save_results_json(res, output_dir, image_path)

        if pdf is not None:
            pdf.savefig(fig)

        if args.show:
            plt.show()

        plt.close(fig)
            

if __name__ == "__main__":
    main()


