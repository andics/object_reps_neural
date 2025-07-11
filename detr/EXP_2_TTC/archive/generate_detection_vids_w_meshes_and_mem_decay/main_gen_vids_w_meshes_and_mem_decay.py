#!/usr/bin/env python3

import os
import re
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from collections import OrderedDict

import imageio
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from PIL import Image
from skimage.measure import label, regionprops, find_contours

from itertools import combinations

torch.set_grad_enabled(False)

DEBUG_TAG = True
DEBUG_FRAME_INDEX = 85
TOPK_DEBUG = 30

# Large top-K so that effectively all DETR queries can be considered
STITCH_TOPK = 10

def parse_args():
    parser = argparse.ArgumentParser(
        "Script that forces exactly two blobs (left=0, right=1) from heuristic detection, then refines with DETR."
    )
    parser.add_argument(
        "--model_path",
        default="/home/projects/bagon/andreyg/Projects/Variable_Resolution_DETR/Programming/detr_var/trained_models/variable_pretrained_resnet101/box_and_segm/checkpoint.pth",
        help="Path to custom DETR checkpoint. We'll parse prefix from 'trained_models/<PREFIX>/' if present."
    )
    parser.add_argument(
        "--video_path",
        default="exp2TTC_files/BConcave+AConcave 3500.mp4",
        help="Path to input .mp4."
    )
    parser.add_argument(
        "--output_video_name",
        default="my_output_collage.mp4",
        help="Name of final stitched .mp4, placed in prefix_videos_processed folder."
    )
    parser.add_argument(
        "--n_blobs",
        type=int,
        default=2,
        help="We always want EXACTLY 2 color blobs (0=left,1=right)."
    )
    # The following remain for consistency but are unused in this version
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="(UNUSED) Exponential smoothing factor.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="(UNUSED) Decay factor for missing blobs.")
    parser.add_argument("--disappearance_threshold", type=int, default=3,
                        help="(UNUSED) Missing frames => disappeared.")
    parser.add_argument("--lambda_x", type=float, default=0.01,
                        help="(UNUSED) leftover param from forced-lr approach.")

    return parser.parse_args()

def extract_prefix_from_model_path(model_path):
    pat = r"trained_models/([^/]+)/"
    m = re.search(pat, model_path)
    if m:
        return m.group(1)
    else:
        return "noprefix"

def load_model(model_path):
    print("Loading DETR panoptic model from Torch Hub (no postprocessor).")
    model, _ = torch.hub.load(
        'facebookresearch/detr',
        'detr_resnet101_panoptic',
        pretrained=False,
        return_postprocessor=True,
        num_classes=91
    )
    print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    # Convert checkpoint to the format that the plain DETR model expects
    state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        if "detr." in k:
            state_dict[k.replace("detr.", "")] = v
        else:
            state_dict[k] = v

    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def find_n_color_blobs(frame_np, black_thresh=30):
    """
    Heuristic: label non-black region, sort connected components by area, return up to 2 largest.
    """
    from skimage.measure import label, regionprops
    gray = frame_np.sum(axis=2)
    non_black = gray > black_thresh
    labeled = label(non_black, connectivity=2)
    regs = regionprops(labeled)
    sorted_regs = sorted(regs, key=lambda r: r.area, reverse=True)
    top2 = sorted_regs[:2]
    masks = [(labeled == r.label) for r in top2]
    return masks

def iou(maskA, maskB):
    inter = (maskA & maskB).sum()
    union = (maskA | maskB).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)

def mask_center_x(m):
    if m is None or m.sum() == 0:
        return None
    ys, xs = np.where(m)
    return xs.mean()

def best_subset_union_for_blob(blob_mask, pred_masks, topK=10):
    """
    Finds the best union of a subset of pred_masks that maximizes IOU with blob_mask.
    1) Rank all pred_masks by IOU with blob_mask
    2) Keep topK
    3) Brute force over those topK => pick the union with highest IOU
    Return that union mask (boolean).
    """
    # 1) rank by IOU
    ious = []
    for i, pm in enumerate(pred_masks):
        ious.append((i, iou(pm, blob_mask)))
    ious_sorted = sorted(ious, key=lambda x: x[1], reverse=True)

    # 2) keep topK
    top_indices = [t[0] for t in ious_sorted[:topK]]
    top_masks = [pred_masks[idx] for idx in top_indices]

    # 3) brute force subsets
    best_iou = 0.0
    best_union = np.zeros_like(blob_mask, dtype=bool)
    found_any = False
    from itertools import combinations
    n_top = len(top_masks)
    for subset_size in range(1, n_top + 1):
        for combo in combinations(range(n_top), subset_size):
            union_ = np.zeros_like(blob_mask, dtype=bool)
            for cidx in combo:
                union_ |= top_masks[cidx]
            iou_val = iou(union_, blob_mask)
            if iou_val > best_iou:
                best_iou = iou_val
                best_union = union_
                found_any = True

    if not found_any:
        return np.zeros_like(blob_mask, dtype=bool)
    return best_union

def find_contour_polygon(bin_mask, cx, cy):
    """
    Return largest contour of bin_mask as a list of (x,y) polygon points,
    translating by (cx, cy) for display convenience.
    """
    from skimage.measure import find_contours
    if bin_mask is None or bin_mask.sum() == 0:
        return []
    cts = find_contours(bin_mask.astype(np.uint8), 0.5)
    if not cts:
        return []
    biggest_ct = max(cts, key=lambda c: c.shape[0])
    poly = []
    for point in biggest_ct:
        r, c = point
        x = c - cx
        y = r - cy
        poly.append((x, y))
    return poly

def overlay_polygons(base_frame, polygons_dict, color_dict,
                     fill_alpha=0.4, center_x=0.0, center_y=0.0,
                     label_suffix=""):
    """
    Overlays polygons onto base_frame using matplotlib, returns an RGB numpy array.
    """
    H, W, _ = base_frame.shape
    fig_dpi = 100
    fig_w = W / fig_dpi
    fig_h = H / fig_dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
    ax.imshow(base_frame)
    ax.set_axis_off()

    for b_i, poly_pts in polygons_dict.items():
        if not poly_pts:
            continue
        c = color_dict[b_i]
        col_rgba = (c[0]/255.0, c[1]/255.0, c[2]/255.0, fill_alpha)
        xs = [p[0] + center_x for p in poly_pts]
        ys = [p[1] + center_y for p in poly_pts]
        ax.fill(xs, ys, color=col_rgba)

        # Optional: label each polygon
        cx, cy = polygon_centroid(list(zip(xs, ys)))
        if cx is not None and cy is not None:
            ax.text(cx, cy,
                    f"Blob {b_i}{label_suffix}",
                    color="white", fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.5, pad=2),
                    ha='center', va='center')

    fig.tight_layout()
    fig.canvas.draw()
    out_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    out_img = out_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return out_img

def polygon_centroid(poly_pts):
    if not poly_pts:
        return None, None
    xs = [p[0] for p in poly_pts]
    ys = [p[1] for p in poly_pts]
    return sum(xs)/len(xs), sum(ys)/len(ys)

def get_detr_masks_and_logits(img_np, model, device, W, H):
    """
    Runs DETR => returns list of upsampled masks + pred_logits.
    """
    pil_img = Image.fromarray(img_np, mode='RGB')
    transform_resize = T.Resize(800)
    resized_img = transform_resize(pil_img)

    transform_norm = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    x = transform_norm(resized_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)

    pred_logits = out["pred_logits"]  # shape(1,100,92)
    pred_masks  = out["pred_masks"]   # shape(1,100,h',w')
    n_queries = pred_masks.shape[1]

    up_masks = []
    for i in range(n_queries):
        m_small = pred_masks[0, i].unsqueeze(0).unsqueeze(0)
        m_up = F.interpolate(m_small, size=(H, W), mode='bilinear', align_corners=False)
        pm = torch.sigmoid(m_up).squeeze().cpu().numpy()
        bin_m = (pm > 0.5)
        up_masks.append(bin_m)

    return up_masks, pred_logits

def debug_show_topk_predictions(frame_np, pred_logits, up_masks, topk=10):
    """
    Debug function: Show a grid of top queries by confidence ignoring 'no-object' class.
    """
    logits = pred_logits[0]  # shape(100,92)
    valid_cl = logits[:, :91]
    max_vals, max_inds = valid_cl.max(dim=-1)
    sorted_idx = torch.argsort(max_vals, descending=True)
    topk_idx = sorted_idx[:topk].cpu().numpy()

    import math
    rows = 2
    cols = math.ceil(topk / rows)
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axs = np.array(axs).reshape(rows, cols)

    for i, qidx in enumerate(topk_idx):
        r = i // cols
        c = i % cols
        ax = axs[r, c]
        overlay = frame_np.copy().astype(np.float32)
        mask = up_masks[qidx]
        color = (255, 0, 255)
        overlay[mask, 0] = color[0]
        overlay[mask, 1] = color[1]
        overlay[mask, 2] = color[2]
        overlay = overlay.astype(np.uint8)
        conf_val = max_vals[qidx].item()
        cls_id = max_inds[qidx].item()
        ax.imshow(overlay)
        ax.set_title(f"Q#{qidx}, cls={cls_id}, conf={conf_val:.3f}")
        ax.axis("off")

    total_sp = rows*cols
    if topk < total_sp:
        for i_ in range(topk, total_sp):
            r = i_ // cols
            c = i_ % cols
            axs[r, c].axis("off")
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    prefix = extract_prefix_from_model_path(args.model_path)
    print(f"Prefix from model_path => '{prefix}'")

    # Create subfolders
    videos_dir    = f"{prefix}_videos_processed"
    blobs_dir     = f"{prefix}_frames_blobs"
    processed_dir = f"{prefix}_frames_processed"
    json_dir      = f"{prefix}_frames_json_processed"

    for d_ in [videos_dir, blobs_dir, processed_dir, json_dir]:
        os.makedirs(d_, exist_ok=True)

    output_video_path = os.path.join(videos_dir, args.output_video_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading DETR ...")
    model = load_model(args.model_path).to(device)

    print(f"Reading video => {args.video_path}")
    reader = imageio.get_reader(args.video_path, format='ffmpeg')
    meta_in = reader.get_meta_data()
    fps = meta_in.get('fps', 30)
    try:
        first_frame = reader.get_data(0)
        H, W, _ = first_frame.shape
        print(f"Video shape: W={W}, H={H}, fps={fps}")
    except:
        print("Could not read first frame => exit.")
        return
    reader.set_image_index(0)

    # We'll color the left blob in red, right blob in green
    color_list = [(255, 0, 0), (0, 255, 0)]
    frame_idx = 0

    def get_two_blobs_sorted(frame_np):
        """
        Return [left_mask, right_mask].
          - If we find 2 largest components => sort by centerX => (left, right).
          - If only 1 => put it in left or right based on centerX vs W/2.
          - If 0 => both None
        """
        found_masks = find_n_color_blobs(frame_np)
        if len(found_masks) == 0:
            return [None, None]
        elif len(found_masks) == 1:
            m_ = found_masks[0]
            cX = mask_center_x(m_)
            if cX is None:
                return [None, None]
            if cX < (frame_np.shape[1] / 2.0):
                return [m_, None]
            else:
                return [None, m_]
        else:
            mA, mB = found_masks
            cAx = mask_center_x(mA)
            cBx = mask_center_x(mB)
            if cAx is None: cAx = 1e9
            if cBx is None: cBx = 1e9
            # sort by centerX
            if cAx <= cBx:
                return [mA, mB]
            else:
                return [mB, mA]

    while True:
        try:
            frame = reader.get_next_data()
        except (EOFError, StopIteration, IndexError):
            break

        if frame.shape[0] != H or frame.shape[1] != W:
            # If frame size changes, correct it
            corrected = np.zeros((H, W, 3), dtype=frame.dtype)
            hh = min(H, frame.shape[0])
            ww = min(W, frame.shape[1])
            corrected[:hh, :ww, :] = frame[:hh, :ww, :]
            frame = corrected

        # 1) Heuristic color-blob detection => [left_blob_mask, right_blob_mask]
        color_blobs_2 = get_two_blobs_sorted(frame)

        # 2) Run DETR => get all upsampled masks
        up_masks, pred_logits = get_detr_masks_and_logits(frame, model, device, W, H)

        # (Optional) debug: show top queries on a specific frame
        if DEBUG_TAG and frame_idx == DEBUG_FRAME_INDEX:
            print(f"DEBUG frame {frame_idx} => top {TOPK_DEBUG} proposals.")
            debug_show_topk_predictions(frame, pred_logits, up_masks, topk=TOPK_DEBUG)

        # 3) Refine the left blob (index=0)
        assigned_left = None
        if color_blobs_2[0] is not None and color_blobs_2[0].sum() > 0:
            assigned_left = best_subset_union_for_blob(color_blobs_2[0], up_masks, topK=STITCH_TOPK)

        # 4) For the right blob, if it exists, exclude region assigned to the left blob
        assigned_right = None
        if color_blobs_2[1] is not None and color_blobs_2[1].sum() > 0:
            if assigned_left is not None:
                # Remove any portion assigned to the left from each DETR mask
                up_masks_for_right = []
                for m_ in up_masks:
                    # exclude the left region from this mask
                    new_m = m_ & ~assigned_left
                    up_masks_for_right.append(new_m)
            else:
                up_masks_for_right = up_masks

            assigned_right = best_subset_union_for_blob(color_blobs_2[1], up_masks_for_right, topK=STITCH_TOPK)

        # 5) For debugging: Save a color-coded version of the raw color detection (no DETR)
        debug_heur = frame.astype(np.float32).copy()
        color_h = [(255,0,0), (0,255,0)]
        for i in range(2):
            bm = color_blobs_2[i]
            if bm is not None:
                debug_heur[bm, 0] = color_h[i][0]
                debug_heur[bm, 1] = color_h[i][1]
                debug_heur[bm, 2] = color_h[i][2]
        heur_path = os.path.join(blobs_dir, f"heuristic_blobs_frame_{frame_idx:06d}.png")
        imageio.imwrite(heur_path, debug_heur.astype(np.uint8))

        # 6) Build polygons for the current frame
        center_x = W / 2.0
        center_y = H / 2.0

        polys_current = {}
        # left
        if assigned_left is not None and assigned_left.sum() > 0:
            polys_current[0] = find_contour_polygon(assigned_left, center_x, center_y)
        else:
            polys_current[0] = []
        # right
        if assigned_right is not None and assigned_right.sum() > 0:
            polys_current[1] = find_contour_polygon(assigned_right, center_x, center_y)
        else:
            polys_current[1] = []

        # 7) Store these polygons in a JSON
        cur_json = {
            "segmentation_blob_0": polys_current[0],
            "segmentation_blob_1": polys_current[1],
        }
        json_current_path = os.path.join(json_dir, f"current_segm_mask_frame_{frame_idx:06d}.json")
        with open(json_current_path, 'w') as jf:
            json.dump(cur_json, jf, indent=2)

        # 8) Render a single collage image showing the 2 polygons over the original frame
        over_cur = overlay_polygons(
            frame, polys_current,
            {0: color_list[0], 1: color_list[1]},
            fill_alpha=0.5,
            center_x=center_x, center_y=center_y,
            label_suffix="(cur)"
        )

        collage_path = os.path.join(processed_dir, f"collage_frame_{frame_idx:06d}.png")
        imageio.imwrite(collage_path, over_cur)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}")

    reader.close()
    total_frames = frame_idx
    print(f"Total frames read => {total_frames}")

    # Make a final video from the collage images
    print(f"Stitching collage => {output_video_path} at fps={fps}")
    writer = imageio.get_writer(output_video_path, fps=fps)
    for i in range(total_frames):
        cpath = os.path.join(processed_dir, f"collage_frame_{i:06d}.png")
        if not os.path.exists(cpath):
            continue
        fr_ = imageio.imread(cpath)
        writer.append_data(fr_)
    writer.close()

    print("Done! Video =>", output_video_path)


if __name__ == "__main__":
    main()
