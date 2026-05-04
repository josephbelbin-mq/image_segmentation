
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Patch

def colorize_label_map(label_map, palette):
    """
    label_map: (H, W) int
    palette: {id: (name, (R,G,B))}
    """
    h, w = label_map.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)

    for k, (_, color) in palette.items():
        color_map[label_map == k] = color

    return color_map


def blend_overlay(image, color_map, alpha=0.4):
    """
    image: (H, W, 3) uint8
    overlay: (H, W, 3) uint8
    """
    blended = image.astype(float).copy()
    mask = color_map.any(axis=2)

    blended[mask] = (
        (1 - alpha) * blended[mask] +
        alpha * color_map[mask]
    )

    return blended.astype(np.uint8)


def view_image(image):
    return image, "Original", {}


def view_binary_mask(binary_mask):
    return binary_mask, "Binary Mask", {"cmap": "gray"}

def view_color_map(label_map, palette):
    color_map = colorize_label_map(label_map, palette)
    return color_map, "Segmentation", {}


def view_blended(image, label_map, palette, alpha=0.4):
    color_map = colorize_label_map(label_map, palette)
    blended = blend_overlay(image, color_map, alpha)
    return blended, "Overlay", {}


def plot_views(views, figsize_per_view=(5, 5)):
    """
    views: list of (image, title, imshow_kwargs)
    """
    n = len(views)
    cols = min(n, 4)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_view[0] * cols,
                 figsize_per_view[1] * rows),
    )

    # Normalize axes to a flat list
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (img, title, kwargs) in zip(axes, views):
        ax.imshow(img, **kwargs)
        ax.set_title(title)
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(views):]:
        ax.axis("off")

    fig.tight_layout()
    return fig



def add_palette_legend(fig, palette, location="lower center"):
    """
    Add a legend showing class names and colors.

    palette: {id: (name, (R, G, B))}
    """
    handles = [
        Patch(
            facecolor=np.array(color) / 255.0,
            label=name
        )
        for _, (name, color) in palette.items()
    ]

    fig.legend(
        handles=handles,
        loc=location,
        ncol=len(handles),
        frameon=False
    )


def make_segmentation_figure(res, alpha=0.4):
    views = [
        view_image(res["image"]),
        view_color_map(res["label_map"], res["palette"]),
        view_blended(res["image"], res["label_map"], res["palette"], alpha),
        view_binary_mask(res["binary_mask"]),
    ]

    fig = plot_views(views)
    add_palette_legend(fig, res["palette"])
    return fig

def make_detection_figure(image, results, figsize=(10, 10)):

    """
    image: HxWx3 numpy array
    results: list of dicts with keys:
        - label
        - box
        - regime ('concentrated' or 'diffuse')
        - context_box (optional)
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    ax.axis("off")

    for r in results:
        box = r["box"]
        regime = r["regime"]
        label = r["label"]
        ctx = r.get("context_box", None)

        # ---- draw parent/context box (dashed) ----
        if ctx is not None:
            x1, y1, x2, y2 = ctx
            ax.add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False,
                linestyle="--",
                linewidth=1.5,
                edgecolor="yellow"
            ))

        # ---- draw main box ----
        x1, y1, x2, y2 = box
        color = "lime" if regime == "concentrated" else "orange"
        style = "-" if regime == "concentrated" else "--"

        ax.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False,
            linestyle=style,
            linewidth=2.5,
            edgecolor=color
        ))

        ax.text(
            x1, y1 - 5,
            f"{label} ({regime})",
            color=color,
            fontsize=9,
            backgroundcolor="black"
        )

    return fig





