#!/usr/bin/env python3
"""
Visualise a 10×10 mean‐filter convolution sliding over an RGB image,
saving both intermediate frames and an animated GIF.
"""

import argparse
import sys
import pathlib
import shutil
from typing import Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

# ──────────────────────────────────────────────────────────────────────────────
# Default image
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_IMAGE = pathlib.Path(
    r"q:\Projects\Variable_resolution\Presentations\23.04_Tommy\Resources\motorbike_var.jpg"
)
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

# Filter size
FILTER_SIZE = 10

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def resolve_input_path(arg: str | None) -> pathlib.Path:
    """Resolve the image path: use DEFAULT_IMAGE if none provided."""
    if arg is None:
        if DEFAULT_IMAGE.exists():
            return DEFAULT_IMAGE
        sys.exit(f"❌ Default image not found at {DEFAULT_IMAGE}")

    p = pathlib.Path(arg)
    if p.is_absolute() and p.exists():
        return p

    candidate = DEFAULT_IMAGE.parent / p
    if candidate.exists():
        return candidate

    sys.exit(f"❌ Image '{arg}' not found (checked absolute path and {DEFAULT_IMAGE.parent})")


def load_and_resize(path: pathlib.Path, max_side: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load image, keep RGB, resize so max(dim) ≤ max_side.
    Returns (image_array, (orig_width, orig_height)).
    """
    img_pil = Image.open(path).convert("RGB")
    orig_w, orig_h = img_pil.size
    scale = min(1.0, max_side / max(orig_w, orig_h))
    if scale < 1.0:
        img_pil = img_pil.resize(
            (int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS
        )
    return np.asarray(img_pil, dtype=np.uint8), (orig_w, orig_h)


def compute_conv_map(img: np.ndarray) -> np.ndarray:
    """
    Apply a FILTER_SIZE×FILTER_SIZE mean filter on each RGB channel,
    producing a (H−F+1)×(W−F+1)×3 feature‐map.
    """
    H, W, C = img.shape
    fh = H - FILTER_SIZE + 1
    fw = W - FILTER_SIZE + 1
    fmap = np.zeros((fh, fw, C), dtype=np.uint8)
    for y in range(fh):
        for x in range(fw):
            window = img[y : y + FILTER_SIZE, x : x + FILTER_SIZE, :].astype(np.float32)
            fmap[y, x] = np.mean(window, axis=(0, 1)).astype(np.uint8)
    return fmap


def figure_setup(img: np.ndarray, fmap_shape: Tuple[int, int]):
    """
    Prepare the matplotlib figure: show the original RGB image with a red FILTER_SIZE×FILTER_SIZE window
    and an empty RGB feature-map panel.
    Returns (fig, rectangle_patch, feature_map_image).
    """
    H, W, _ = img.shape
    fh, fw = fmap_shape

    fig = plt.figure(figsize=(8, 4), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[W, fw])

    # Original image panel
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img)
    ax_img.set_xticks([]), ax_img.set_yticks([])
    rect = Rectangle((0.5, 0.5), FILTER_SIZE, FILTER_SIZE,
                     linewidth=2, edgecolor="red", facecolor="none")
    ax_img.add_patch(rect)

    # Feature-map panel
    ax_feat = fig.add_subplot(gs[1])
    ax_feat.set_title(f"Mean Filter Feature Map ({FILTER_SIZE}×{FILTER_SIZE})")
    ax_feat.set_xticks([]), ax_feat.set_yticks([])
    blank = np.zeros((fh, fw, 3), dtype=np.uint8)
    feat_im = ax_feat.imshow(blank, interpolation="nearest", origin="upper")

    return fig, rect, feat_im


# ──────────────────────────────────────────────────────────────────────────────
# Main routine
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Visualise a {FILTER_SIZE}×{FILTER_SIZE} mean‐filter convolution on RGB."
    )
    parser.add_argument(
        "image", nargs="?", help="image file (default: motorbike_var.jpg)"
    )
    parser.add_argument(
        "-o", "--out", type=pathlib.Path, help="output GIF path"
    )
    parser.add_argument(
        "-s", "--size", type=int, default=256, help="max side length in px"
    )
    parser.add_argument(
        "-fps", type=int, default=12, help="frames per second"
    )
    args = parser.parse_args()

    # Resolve input image
    img_path = resolve_input_path(args.image)
    print(f"📂 Using image: {img_path}")

    # Load and optionally resize
    img, (orig_w, orig_h) = load_and_resize(img_path, args.size)
    print(
        f"   Original size: {orig_w}×{orig_h}  →  "
        f"Working size: {img.shape[1]}×{img.shape[0]}"
    )

    # Compute feature-map
    fmap = compute_conv_map(img)
    fh, fw, _ = fmap.shape
    print(f"🧮 Feature-map size: {fh}×{fw}  |  Frames: {fh * fw}")

    # Prepare figure
    fig, rect, feat_im = figure_setup(img, (fh, fw))

    # Determine output paths
    out_path = args.out if args.out else SCRIPT_DIR / f"{img_path.stem}_mean{FILTER_SIZE}.gif"
    print(f"📤 GIF will be saved to: {out_path}")
    frames_dir = SCRIPT_DIR / f"{img_path.stem}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir()
    print(f"🖼️  Intermediate frames in: {frames_dir}")

    # Animation data
    fmap_progress = np.zeros_like(fmap)
    positions = [(y, x) for y in range(fh) for x in range(fw)]
    digits = len(str(len(positions) - 1))

    # Animation callbacks
    def init():
        rect.set_xy((0.5, 0.5))
        feat_im.set_data(fmap_progress)
        return rect, feat_im

    def update(idx: int):
        y, x = positions[idx]
        rect.set_xy((x + 0.5, y + 0.5))
        fmap_progress[y, x] = fmap[y, x]
        feat_im.set_data(fmap_progress)
        frame_path = frames_dir / f"frame{idx:0{digits}d}.png"
        fig.savefig(frame_path, dpi=150)
        return rect, feat_im

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(positions),
        init_func=init,
        blit=True,
        interval=1000 / args.fps,
    )
    anim.save(out_path, writer="pillow", fps=args.fps, dpi=150)
    print("✅  Done!")


if __name__ == "__main__":
    main()
