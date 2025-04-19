#!/usr/bin/env python3
"""
process_single_images.py  –  v4 FINAL COMPLETE
============================================
Detect and segment a single coloured blob in each image of a folder
using DETR‑Panoptic. For each image, produces a folder named

    <MODEL_PREFIX>_<IMAGE_NAME>/

with sub‑folders:
    • frames_blobs           – blob overlay
    • frames_collage         – 10‑best DETR query visualisations
    • frames_masks_nonmem    – binary mask PNG
    • frames_processed       – final polygon overlay

Usage:
    python process_single_images.py   # uses built‑in defaults
    python process_single_images.py --model_path /path/to/checkpoint.pth \
                                    --images_folder /path/to/images \
                                    --output_root /path/to/output
"""

from __future__ import annotations
import os
import re
import argparse
from collections import OrderedDict
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, find_contours
from scipy.optimize import linear_sum_assignment

torch.set_grad_enabled(False)

# ----------------------------------------------------------------------------
# Logging helper
# ----------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ----------------------------------------------------------------------------
# 1. Load DETR model
# ----------------------------------------------------------------------------

def load_model(ckpt_path: str) -> torch.nn.Module:
    log("Loading DETR‑Panoptic backbone …")
    model, _ = torch.hub.load(
        "facebookresearch/detr",
        "detr_resnet101_panoptic",
        pretrained=False,
        return_postprocessor=True,
        num_classes=91,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    try:
        model.load_state_dict(ckpt["model"], strict=True)
        log("Checkpoint loaded (strict match).")
    except RuntimeError:
        log("Strict load failed – stripping 'detr.' prefix …")
        stripped = OrderedDict((k.replace("detr.", ""), v) for k, v in ckpt["model"].items())
        model.load_state_dict(stripped, strict=False)
        log("Checkpoint loaded (non‑strict). Some params defaulted.")
    model.eval()
    return model

# ----------------------------------------------------------------------------
# 2. Blob detection and utilities
# ----------------------------------------------------------------------------

def detect_blob(frame: np.ndarray, thresholds: Tuple[int, ...] = (30, 15, 5)) -> np.ndarray | None:
    gray = frame.sum(axis=2)
    for thr in thresholds:
        labeled = label(gray > thr, connectivity=2)
        regs = sorted(regionprops(labeled), key=lambda r: r.area, reverse=True)
        if regs:
            log(f"    Blob detected with threshold {thr} (area={regs[0].area})")
            return labeled == regs[0].label
        else:
            log(f"    No blob found at threshold {thr}")
    return None


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else inter / union


def best_match(blob: np.ndarray, masks: List[np.ndarray]) -> Tuple[int | None, float | None]:
    if not masks:
        return None, None
    costs = np.array([-iou(blob, m) for m in masks], dtype=np.float32)
    idx = int(costs.argmin())
    return idx, float(costs[idx])


def contour_polygon(mask: np.ndarray, cx: float, cy: float) -> List[Tuple[float, float]]:
    if mask is None or mask.sum() == 0:
        return []
    contours = find_contours(mask.astype(np.uint8), 0.5)
    if not contours:
        return []
    big = max(contours, key=len)
    return [((p[1] - cx), (p[0] - cy)) for p in big]


def poly_centroid(pts: List[Tuple[float, float]]) -> Tuple[float | None, float | None]:
    if not pts:
        return None, None
    xs, ys = zip(*pts)
    return sum(xs) / len(xs), sum(ys) / len(ys)

# ----------------------------------------------------------------------------
# 3. Process a single image
# ----------------------------------------------------------------------------

def process_image(img_path: str, model: torch.nn.Module, device: torch.device,
                  model_prefix: str, out_root: str) -> None:
    name = os.path.splitext(os.path.basename(img_path))[0]
    log(f"Processing image '{name}' …")

    # Setup output directories
    base = os.path.join(out_root, f"{model_prefix}_{name}")
    dirs = {
        "blobs": os.path.join(base, "frames_blobs"),
        "collage": os.path.join(base, "frames_collage"),
        "mask": os.path.join(base, "frames_masks_nonmem"),
        "proc": os.path.join(base, "frames_processed"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Load image and coerce to RGB
    frame = imageio.imread(img_path)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    H, W, _ = frame.shape
    log(f"    Image shape: {H}x{W} RGB")

    # Detect blob
    blob = detect_blob(frame)
    if blob is None:
        log("    ✗ No blob found — skip image.")
        return

    # Save blob overlay
    overlay = frame.copy()
    overlay[blob] = [255, 0, 0]
    blob_file = os.path.join(dirs['blobs'], f"{name}_blobs.png")
    Image.fromarray(overlay).save(blob_file)
    log(f"    Saved blob overlay → {blob_file}")

    # Run DETR
    pil = Image.fromarray(frame).convert('RGB')
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    inp = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)['pred_masks']  # (1,100,h',w')

    # Upsample & extract candidate masks
    candidates: List[np.ndarray] = []
    for i in range(out.shape[1]):
        up = F.interpolate(out[0,i][None,None], size=(H,W), mode='bilinear', align_corners=False)
        bm = torch.sigmoid(up).squeeze().cpu().numpy() > 0.5
        labs = label(bm, connectivity=2)
        candidates += [(labs == lbl) for lbl in range(1, labs.max()+1)]
    log(f"    Generated {len(candidates)} candidate masks from DETR.")

    # Choose best mask
    idx, cost = best_match(blob, candidates)
    chosen = candidates[idx] if idx is not None else None
    if chosen is not None and chosen.sum() > 0:
        mask_file = os.path.join(dirs['mask'], f"mask_{name}.png")
        Image.fromarray((chosen.astype(np.uint8)*255)).save(mask_file)
        log(f"    Chosen mask idx={idx}, cost={cost:.3f} → {mask_file}")
    else:
        log("    ✗ No suitable DETR mask; proceeding without mask file.")

    # Collage of top‑10
    if candidates:
        ious = [-iou(blob, m) for m in candidates]
        best10 = np.argsort(ious)[:10]
        fig, axes = plt.subplots(1,10,figsize=(25,3),dpi=100)
        for r, j in enumerate(best10):
            ov = frame.copy()
            ov[blob] = [0,255,0]
            ov[candidates[j]] = [255,0,0]
            axes[r].imshow(ov)
            axes[r].set_title(f"#{j}\n{ious[j]:.3f}", fontsize=6)
            axes[r].axis('off')
        coll_file = os.path.join(dirs['collage'], f"{name}_collage.png")
        fig.savefig(coll_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        log(f"    Saved collage → {coll_file}")

    # Final polygon overlay
    final = Image.fromarray(frame.copy())
    if chosen is not None and chosen.sum()>0:
        poly = contour_polygon(chosen, W/2, H/2)
        if poly:
            draw = ImageDraw.Draw(final, 'RGBA')
            pts = [(x+W/2, y+H/2) for x,y in poly]
            draw.polygon(pts, fill=(255,0,0,120))
            cx, cy = poly_centroid(pts)
            if cx is not None:
                draw.text((cx,cy), 'Blob', fill=(255,255,255,255))
    final_file = os.path.join(dirs['proc'], f"{name}_overlay.png")
    final.save(final_file)
    log(f"    Saved final overlay → {final_file}\n")

# ----------------------------------------------------------------------------
# 4. Argument parsing with defaults
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Process a folder of images with DETR")
    p.add_argument(
        "--model_path", default=(
            "/home/projects/bagon/andreyg/Projects/Variable_Resolution_DETR/"
            "Programming/detr_var/trained_models/full_resolution_resnet101/"
            "box_and_segm/checkpoint.pth"
        ), help="Path to DETR checkpoint (.pth)"
    )
    p.add_argument(
        "--images_folder", default=(
            "/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/detr/"
            "EXP_3_CHANGE/Data_processed/Stimuli/Exp3b_Images"
        ), help="Folder containing input images"
    )
    p.add_argument(
        "--output_root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_images"),
        help="Root folder for output"
    )
    return p.parse_args()

# ----------------------------------------------------------------------------
# 5. Main entrypoint
# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    model_prefix = (re.search(r"trained_models/([^/]+)", args.model_path).group(1)
                    if "trained_models/" in args.model_path else "unknownModel")
    model = load_model(args.model_path).to(device)

    # Gather image files
    exts = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
    imgs = [os.path.join(args.images_folder, f) for f in sorted(os.listdir(args.images_folder))
            if os.path.splitext(f)[1].lower() in exts]
    if not imgs:
        log("✗ No images found in folder. Exiting.")
        return
    log(f"Found {len(imgs)} images in {args.images_folder}")
    os.makedirs(args.output_root, exist_ok=True)

    for img in imgs:
        try:
            process_image(img, model, device, model_prefix, args.output_root)
        except Exception as e:
            log(f"[ERROR] Processing {img}: {e}")

    log("All images processed.")

if __name__ == '__main__':
    main()
