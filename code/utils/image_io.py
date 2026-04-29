from PIL import Image
import numpy as np
from pathlib import Path
from code.utils.image_viz import colorize_label_map, blend_overlay

def get_stem(image_path: str) -> str:
    if image_path.startswith("s3://"):
        name = image_path.rsplit("/", 1)[-1]
        return Path(name).stem
    else:
        return Path(image_path).stem


def save_image(array, path):
    """
    array: (H,W) or (H,W,3), bool / uint8 / float
    """
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    Image.fromarray(array).save(path)

    
def save_binary_mask(binary_mask, path):
    save_image(binary_mask, path)

def save_color_map(label_map, palette, path):
    color_map = colorize_label_map(label_map, palette)
    save_image(color_map, path)


def save_overlay(image, label_map, palette, path, alpha=0.4):
    color_map = colorize_label_map(label_map, palette)
    blended = blend_overlay(image, color_map, alpha)
    save_image(blended, path)

def save_all_outputs(
    image,
    label_map,
    binary_mask,
    palette,
    out_dir,
    image_path,
):
    stem = get_stem(image_path)
    save_binary_mask(binary_mask, out_dir / f"{stem}_mask.png")
    save_color_map(label_map, palette, out_dir / f"{stem}_seg.png")
    save_overlay(image, label_map, palette, out_dir / f"{stem}_overlay.png")
