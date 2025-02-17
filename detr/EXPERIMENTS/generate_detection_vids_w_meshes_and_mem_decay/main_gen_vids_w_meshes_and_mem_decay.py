#!/usr/bin/env python3
"""
final_gen_video_with_decay.py

Enhancements over the original final_gen_video.py to handle:
1) Exponential memory decay for each blob's average segmentation.
2) Handling of blob disappearance:
   - Stop searching for the missing blob after a grace period.
   - Keep the last average mask displayed (ghost).
3) Dual segmentation overlays: current frame's segmentation + average segmentation.
4) Collision detection for the two average masks (first overlap).
5) Logging key events: disappearance & collision.
6) Automatic arrow drawing from the remaining blob to the disappeared blob's last position.
7) All original outputs (blobs frames, processed frames, JSON) still produced,
   plus an additional 'collision_info.json' summarizing collision details.

Requires:
  - PyTorch
  - torchvision
  - imageio[ffmpeg]
  - numpy, scipy, scikit-image, matplotlib
  - opencv-python (for arrow drawing, etc.)
"""

import os
import argparse
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from collections import OrderedDict
import logging

import cv2  # for arrow drawing, etc.
from PIL import Image
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from skimage.measure import label, regionprops, find_contours
from scipy.optimize import linear_sum_assignment

torch.set_grad_enabled(False)

COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def setup_logging():
    """
    Configure a basic logger to print key events.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

def load_model(model_path):
    """
    Load DETR (panoptic variant) from torch hub, then a custom checkpoint.
    We do NOT use the postprocessor.
    """
    logging.info("Loading DETR panoptic model from torch hub (no postprocessor).")
    model, _ = torch.hub.load(
        'facebookresearch/detr',
        'detr_resnet101_panoptic',
        pretrained=False,
        return_postprocessor=True,
        num_classes=91
    )
    logging.info(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        if "detr." in k:
            # remove "detr." prefix
            state_dict[k.replace("detr.", "")] = v
        else:
            state_dict[k] = v
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def find_n_color_blobs(frame_np, n_blobs=2, black_thresh=30):
    """
    Heuristic detection of color blobs on black background.
    - We define "black" as sum_of_RGB < black_thresh.
    - We label the connected non-black region and pick the top n largest regions.
    Return a list of boolean masks in descending order of area.
    """
    gray = frame_np.sum(axis=2)  # shape (H, W)
    non_black = gray > black_thresh
    labeled = label(non_black, connectivity=2)
    regions = regionprops(labeled)
    # sort by area
    sorted_regs = sorted(regions, key=lambda r: r.area, reverse=True)
    top_regs = sorted_regs[:n_blobs]

    masks = []
    for r in top_regs:
        m = (labeled == r.label)
        masks.append(m)
    return masks

def iou(maskA, maskB):
    inter = (maskA & maskB).sum()
    union = (maskA | maskB).sum()
    if union == 0:
        return 0.0
    return inter / union

def bipartite_assign_blobs_to_masks(blob_masks, pred_masks):
    """
    Build a cost matrix of shape (num_blobs, num_preds) = -IOU,
    then solve with Hungarian to maximize total IOU.
    Returns a list 'assign' of length num_blobs where assign[b] = pred_idx or None.
    """
    nb = len(blob_masks)
    np_ = len(pred_masks)
    if np_ == 0:
        return [None]*nb

    cost = np.zeros((nb, np_), dtype=np.float32)
    for b in range(nb):
        for p in range(np_):
            cost[b,p] = -iou(blob_masks[b], pred_masks[p])

    row_idx, col_idx = linear_sum_assignment(cost)  # minimize
    # row_idx => blob index, col_idx => pred index
    assign = [None]*nb
    for i in range(len(row_idx)):
        b = row_idx[i]
        p = col_idx[i]
        assign[b] = p
    return assign

def make_masks_disjoint(masks):
    """
    We remove overlap among a list of boolean masks in-place.
    For i in range(len(masks)):
       for j in range(i+1, len(masks)):
         masks[j] = masks[j] & ~masks[i]
    Return updated list.
    """
    for i in range(len(masks)):
        if masks[i] is None:
            continue
        for j in range(i+1, len(masks)):
            if masks[j] is None:
                continue
            masks[j] = masks[j] & ~masks[i]
    return masks

def find_contour_polygon(bin_mask, center_x, center_y):
    """
    Use skimage.find_contours => pick the largest boundary. Convert from row,col
    to a polygon in center-based coords: x = col - center_x, y = row - center_y
    Return list of (x,y) floats. If mask is empty, return [].
    """
    if bin_mask is None or bin_mask.sum() == 0:
        return []

    cts = find_contours(bin_mask.astype(np.uint8), 0.5)
    if len(cts) == 0:
        return []

    # pick largest
    biggest_ct = max(cts, key=lambda c: c.shape[0])
    poly = []
    for point in biggest_ct:
        r = point[0]
        c = point[1]
        x = c - center_x
        y = r - center_y
        poly.append((x,y))
    return poly

def polygon_centroid(poly_pts):
    """
    Approximate centroid of a polygon in top-left coords.
    If empty, return (None,None).
    We'll do a simple arithmetic mean of points.
    """
    if len(poly_pts) == 0:
        return None, None
    xs = [p[0] for p in poly_pts]
    ys = [p[1] for p in poly_pts]
    return (sum(xs)/len(xs), sum(ys)/len(ys))

def get_mask_centroid(bin_mask):
    """
    Return (cx, cy) of a binary mask's centroid in top-left coords.
    If empty, return (None, None).
    """
    if bin_mask is None or bin_mask.sum() == 0:
        return None, None
    ys, xs = np.where(bin_mask)
    cx = xs.mean()
    cy = ys.mean()
    return cx, cy

def update_avg_mask(avg_mask, current_mask, alpha, beta, blob_present):
    """
    Exponential memory approach.
    - If blob_present: avg_mask = alpha*current_mask + (1-alpha)*avg_mask
    - If not present:  avg_mask = beta*avg_mask
    current_mask is boolean => convert to float for averaging.
    """
    if avg_mask is None:
        # Initialize with current mask as float
        if blob_present:
            return current_mask.astype(np.float32)
        else:
            return None

    if blob_present:
        # Weighted average
        current_f = current_mask.astype(np.float32)
        avg_mask = alpha*current_f + (1.0-alpha)*avg_mask
    else:
        # Decay
        avg_mask = beta*avg_mask
    return avg_mask

def threshold_mask(m, thresh=0.5):
    """
    Return a boolean mask from a float array m, using thresh.
    If m is None, return None.
    """
    if m is None:
        return None
    return (m > thresh)

def main():
    setup_logging()

    parser = argparse.ArgumentParser("Final script with heuristics, disjoint masks, polygons, memory decay, etc.")
    parser.add_argument("--model_path", required=True, help="Path to custom DETR checkpoint (.pth)")
    parser.add_argument("--video_path", required=True, help="Path to input .mp4")
    parser.add_argument("--output_video_path", required=True, help="Path to output .mp4 (same resolution)")
    parser.add_argument("--n_blobs", type=int, default=2, help="Initial number of color blobs to track (default=2)")
    parser.add_argument("--blobs_dir", default="frames_blobs", help="Dir to save color-blob debug images.")
    parser.add_argument("--processed_dir", default="frames_processed", help="Dir to save final overlay frames.")
    parser.add_argument("--json_dir", default="frames_json", help="Dir to save JSON for each frame.")

    # memory decay parameters
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Exponential smoothing weight for current frame (0<alpha<=1). Higher=less smoothing.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Decay factor when blob is missing (0<beta<=1). 1=no fade, <1 => fade over time.")
    parser.add_argument("--disappearance_threshold", type=int, default=3,
                        help="Number of consecutive frames a blob must be missing before confirmed disappeared.")
    parser.add_argument("--collision_json", default="collision_info.json",
                        help="Filename for collision output JSON.")
    args = parser.parse_args()

    # 1) Load the model
    model = load_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2) Prepare local folders
    for d in [args.blobs_dir, args.processed_dir, args.json_dir]:
        os.makedirs(d, exist_ok=True)

    # Logging info
    logging.info(f"Reading input video {args.video_path}")
    reader = imageio.get_reader(args.video_path, format='ffmpeg')
    meta_in = reader.get_meta_data()
    fps = meta_in.get('fps', 30)

    # try first frame to get W,H
    try:
        first_frame = reader.get_data(0)
        H, W, _ = first_frame.shape
        logging.info(f"Video shape: W={W}, H={H}, fps={fps}")
    except:
        logging.error("Could not read first frame to determine shape.")
        reader.close()
        return
    # reset
    reader.set_image_index(0)

    # For collision detection
    collision_frame = None
    collision_centroids = (None, None)  # ((cx1,cy1),(cx2,cy2))

    # We'll store frames to processed_dir, then stitch them
    frames_processed_paths = []

    # Track average masks for each blob
    # Initially, we have up to n_blobs (2) => store in a list of dict, e.g. blob_info_list
    # Each entry: { 'avg_mask': None, 'missing_count':0, 'disappeared':False }
    # We'll also keep 'color' for drawing. We'll pick from a color map:
    color_list = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
    # We'll track up to args.n_blobs
    blob_info_list = []
    for b in range(args.n_blobs):
        blob_info_list.append({
            'avg_mask': None,
            'missing_count': 0,
            'disappeared': False,
            'color': color_list[b%len(color_list)]
        })

    frame_idx = 0

    # Helper function to get DETR upsampled masks
    def get_detr_masks(img, W, H):
        pil_img = Image.fromarray(img, mode="RGB")
        transform_resize = T.Resize(800)
        resized_img = transform_resize(pil_img)
        rw, rh = resized_img.size

        transform_norm = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        model_in = transform_norm(resized_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(model_in)
        pred_logits = outputs["pred_logits"]  # (1,100,92)
        pred_masks  = outputs["pred_masks"]   # (1,100,h',w')

        # upsample masks
        n_queries = pred_logits.shape[1]
        up_masks = []
        for i in range(n_queries):
            ml = pred_masks[0,i].unsqueeze(0).unsqueeze(0) # (1,1,h',w')
            up = F.interpolate(ml, size=(H,W), mode="bilinear", align_corners=False)
            pm = torch.sigmoid(up).squeeze(0).squeeze(0).cpu().numpy()
            bin_m = (pm > 0.5)
            up_masks.append(bin_m)
        return up_masks

    # define a function to check collision among average masks
    def check_collision(bin_mask_a, bin_mask_b):
        if bin_mask_a is None or bin_mask_b is None:
            return False
        overlap = (bin_mask_a & bin_mask_b)
        return np.any(overlap)

    for frame in reader:
        # Ensure shape matches W,H
        if frame.shape[0] != H or frame.shape[1] != W:
            corrected = np.zeros((H,W,3), dtype=frame.dtype)
            h_ = min(H, frame.shape[0])
            w_ = min(W, frame.shape[1])
            corrected[0:h_, 0:w_, :] = frame[0:h_, 0:w_, :]
            frame = corrected

        # --------------------------------------
        # 1) Detect color blobs
        # For now we still try to find up to n active blobs, but we handle disappearance logic below
        # This is the heuristic. We'll do bipartite matching with DETR as in original code.
        # But if a blob is disappeared, we skip it from matching.
        # If there's fewer than n active blobs, we only match for the active ones.

        # We check how many are still "active" (not disappeared)
        active_blob_indices = [i for i,bi in enumerate(blob_info_list) if not bi['disappeared']]
        n_active = len(active_blob_indices)
        # if n_active is 0 => no tracking. But let's proceed normally

        # find color blobs heuristically
        # We'll find up to n_active color blobs. If we have 2 but one is disappeared => we only find up to 1
        if n_active > 0:
            heuristic_n = n_active
        else:
            heuristic_n = 0

        color_blob_masks = find_n_color_blobs(frame, n_blobs=heuristic_n)

        # 2) DETR predicted masks
        pred_up_masks = get_detr_masks(frame, W, H)

        # bipartite assignment
        assign = bipartite_assign_blobs_to_masks(color_blob_masks, pred_up_masks)

        # build final assigned mask list
        assigned = []
        for b_idx in range(len(color_blob_masks)):
            p_idx = assign[b_idx]
            if p_idx is not None:
                assigned.append(pred_up_masks[p_idx])
            else:
                assigned.append(None)

        # ensure disjoint
        assigned = make_masks_disjoint(assigned)

        # NOTE: assigned is in the order of color_blob_masks. That is for the active blobs only.
        # We'll map these assigned masks back to the global blob_info_list in active_blob_indices order.
        # e.g., if active_blob_indices = [0,2] => the 0-th color blob => blob_info_list[0], 1st => blob_info_list[2].
        # If there's a mismatch in count, we handle that.
        # If heuristic_n=0 => no color_blob_masks => assigned=empty
        # Let's do a safe approach:
        for idx, global_idx in enumerate(active_blob_indices):
            if idx < len(assigned):
                mask_here = assigned[idx]
                # Update or not
                if mask_here is not None and mask_here.sum() > 0:
                    # We found a blob => update avg mask
                    blob_info_list[global_idx]['avg_mask'] = update_avg_mask(
                        blob_info_list[global_idx]['avg_mask'],
                        mask_here,
                        alpha=args.alpha,
                        beta=args.beta,
                        blob_present=True
                    )
                    blob_info_list[global_idx]['missing_count'] = 0
                else:
                    # Not found => increment missing_count
                    blob_info_list[global_idx]['missing_count'] += 1
                    # Decay the avg mask
                    blob_info_list[global_idx]['avg_mask'] = update_avg_mask(
                        blob_info_list[global_idx]['avg_mask'],
                        None,
                        alpha=args.alpha,
                        beta=args.beta,
                        blob_present=False
                    )
            else:
                # no assigned mask => definitely missing
                blob_info_list[global_idx]['missing_count'] += 1
                blob_info_list[global_idx]['avg_mask'] = update_avg_mask(
                    blob_info_list[global_idx]['avg_mask'],
                    None,
                    alpha=args.alpha,
                    beta=args.beta,
                    blob_present=False
                )

        # check disappearance threshold
        for i,blob_info in enumerate(blob_info_list):
            if not blob_info['disappeared']:
                if blob_info['missing_count'] >= args.disappearance_threshold:
                    # mark disappeared
                    blob_info['disappeared'] = True
                    logging.info(f"Blob {i} disappeared at frame {frame_idx}.")

        # 3) Visual debug of color blobs => frames_blobs
        debug_blob = frame.astype(np.float32).copy()
        for i,bm in enumerate(color_blob_masks):
            c = color_list[i%len(color_list)]
            if bm is not None:
                debug_blob[bm,0] = c[0]
                debug_blob[bm,1] = c[1]
                debug_blob[bm,2] = c[2]
        debug_blob_path = os.path.join(args.blobs_dir, f"frame_{frame_idx:06d}_blobs.png")
        Image.fromarray(debug_blob.astype(np.uint8)).save(debug_blob_path)

        # 4) For each active blob, build final assigned mask (for polygons) => assigned
        # But we already have assigned for the active set. We also want the original code's polygon logic
        # We'll gather polygons for each tracked blob. Sort by horizontal centroid to label them consistently.
        # We'll also keep the newly assigned mask (immediate) for drawing.
        # But note that assigned might have fewer or no entries if the blob isn't found.

        # We'll store in a structure: poly_current[i], poly_avg[i]
        center_x = W/2.0
        center_y = H/2.0

        poly_current = {}
        poly_avg     = {}

        # Re-build current assignment in the global order
        # assigned_in_global_order[i] => None or the assigned mask
        assigned_in_global_order = [None]*len(blob_info_list)
        for idx,global_idx in enumerate(active_blob_indices):
            if idx < len(assigned):
                assigned_in_global_order[global_idx] = assigned[idx]

        # Now compute polygons (current mask) + polygons (avg mask)
        for i in range(len(blob_info_list)):
            cmask = assigned_in_global_order[i]
            amask_float = blob_info_list[i]['avg_mask']  # float
            if amask_float is not None:
                amask_bin = threshold_mask(amask_float, thresh=0.5)
            else:
                amask_bin = None

            # current polygon
            if cmask is not None and cmask.sum()>0:
                cpoly = find_contour_polygon(cmask, center_x, center_y)
            else:
                cpoly = []
            poly_current[i] = cpoly

            # average polygon
            if amask_bin is not None and amask_bin.sum()>0:
                apoly = find_contour_polygon(amask_bin, center_x, center_y)
            else:
                apoly = []
            poly_avg[i] = apoly

        # We want to rename them in left->right order as the original script does:
        # But we'll do that for final JSON. We'll gather them with their mean_col and sort.
        # We have up to len(blob_info_list)=2 if some haven't disappeared.
        # Actually let's keep it the same approach: compute mean_col for each average polygon.
        # Then reorder. We'll store polygons_json in final form.

        blob_order_data = []
        for i in range(len(blob_info_list)):
            # find an approximate mean_col from either current or avg
            # If avg polygon is valid, we use that, else current
            # or we compute from average mask centroid
            amask_float = blob_info_list[i]['avg_mask']
            if amask_float is not None:
                bin_mask = (amask_float>0.5)
                if bin_mask.sum()>0:
                    ys, xs = np.where(bin_mask)
                    mean_col = xs.mean()
                else:
                    mean_col = float('inf')
            else:
                mean_col = float('inf')
            blob_order_data.append((i, mean_col))

        blob_order_data.sort(key=lambda x: x[1])  # left->right
        polygons_json = {}
        for order_idx, (i, mc) in enumerate(blob_order_data):
            polygons_json[f"segmentation_blob_{order_idx}_current"] = poly_current[i]
            polygons_json[f"segmentation_blob_{order_idx}_average"] = poly_avg[i]

        # 5) store the JSON
        json_path = os.path.join(args.json_dir, f"frame_{frame_idx:06d}.json")
        with open(json_path,'w') as fjson:
            json.dump(polygons_json, fjson, indent=2)

        # 6) overlay polygons on the frame => frames_processed
        fig_dpi=100
        fig_w = W/fig_dpi
        fig_h = H/fig_dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
        ax.imshow(frame)  # shape is top-left origin
        ax.set_axis_off()

        # we do the overlay for each blob in the order of blob_order_data so the labeling matches
        # color maps or direct color usage
        color_maps = ['Reds','Blues','Greens','Oranges','Purples','pink','YlOrBr']

        # We'll also store centroids for arrow usage
        disappeared_blob_positions = []
        active_blob_positions = []
        for order_idx, (blob_idx, _) in enumerate(blob_order_data):
            # color approach
            cmap = color_maps[order_idx % len(color_maps)]
            base_color_rgba = cm.get_cmap(cmap)(0.6)  # RGBA

            cpoly = poly_current[blob_idx]
            apoly = poly_avg[blob_idx]

            # Convert to top-left coords
            def center_to_tl(poly):
                pts = []
                for (xC, yC) in poly:
                    x_tl = xC + center_x
                    y_tl = yC + center_y
                    pts.append((x_tl, y_tl))
                return pts
            cpoly_tl = center_to_tl(cpoly)
            apoly_tl = center_to_tl(apoly)

            # fill average with a lighter alpha
            if len(apoly_tl)>0:
                ax.fill(
                    [p[0] for p in apoly_tl],
                    [p[1] for p in apoly_tl],
                    alpha=0.3, color=base_color_rgba, linewidth=0
                )
                # label near centroid
                acx, acy = polygon_centroid(apoly_tl)
                if acx is not None and acy is not None:
                    ax.text(
                        x=acx, y=acy,
                        s=f"Blob{blob_idx} avg",
                        color="white",
                        fontsize=10,
                        fontweight="bold",
                        bbox=dict(facecolor='black', alpha=0.5, pad=2),
                        ha='center', va='center', zorder=999
                    )

            # fill current with a bit stronger alpha or different shade
            if len(cpoly_tl)>0:
                ax.fill(
                    [p[0] for p in cpoly_tl],
                    [p[1] for p in cpoly_tl],
                    alpha=0.5, color=base_color_rgba, linewidth=0
                )
                # label near centroid
                ccx, ccy = polygon_centroid(cpoly_tl)
                if ccx is not None and ccy is not None:
                    ax.text(
                        x=ccx, y=ccy,
                        s=f"Blob{blob_idx} cur",
                        color="black",
                        fontsize=8,
                        fontweight="bold",
                        bbox=dict(facecolor='white', alpha=0.7, pad=2),
                        ha='center', va='center', zorder=999
                    )

            # gather centroid for arrow usage
            # if the blob disappeared or not
            if blob_info_list[blob_idx]['disappeared']:
                # we want the last average mask centroid (if any)
                amask_float = blob_info_list[blob_idx]['avg_mask']
                if amask_float is not None:
                    bin_mask = (amask_float>0.5)
                    cx, cy = get_mask_centroid(bin_mask)
                    if cx is not None and cy is not None:
                        disappeared_blob_positions.append((cx, cy, blob_idx))
            else:
                # active => we can get the current avg mask centroid
                amask_float = blob_info_list[blob_idx]['avg_mask']
                if amask_float is not None:
                    bin_mask = (amask_float>0.5)
                    cx, cy = get_mask_centroid(bin_mask)
                    if cx is not None and cy is not None:
                        active_blob_positions.append((cx, cy, blob_idx))

        # if there's 1 or more disappeared and 1 or more active => draw arrow from active to disappeared
        # possibly only the first disappeared or so
        for (cx_a, cy_a, a_idx) in active_blob_positions:
            # draw arrow to each disappeared? Or just the first? Let's do each if multiple
            for (cx_d, cy_d, d_idx) in disappeared_blob_positions:
                pt_active = (int(cx_a), int(cy_a))
                pt_disp   = (int(cx_d), int(cy_d))
                # draw arrow with cv2
                # we have to get the numpy array behind the figure => messy.
                # We'll do a quick hack: we can skip it since we are using matplotlib.
                # But let's attempt a quick approach with arrow in matplotlib:
                ax.annotate(
                    "", xy=pt_disp, xytext=pt_active,
                    arrowprops=dict(arrowstyle="->", color='white', lw=2)
                )

        # now check for collision among average masks (only if we haven't found one yet)
        if collision_frame is None:
            # gather average masks
            # we only do collision if 2 are not disappeared or if we want to see if the active overlaps the ghost
            # user said we want to track the collision with the disappeared's ghost as well. So let's do for all pairs
            # up to 2 blobs total anyway
            bin_masks = []
            for i in range(len(blob_info_list)):
                am = blob_info_list[i]['avg_mask']
                if am is not None:
                    bin_masks.append( (i, threshold_mask(am,0.5)) )
            # check pairs
            for i in range(len(bin_masks)):
                for j in range(i+1, len(bin_masks)):
                    idx_i, mask_i = bin_masks[i]
                    idx_j, mask_j = bin_masks[j]
                    if check_collision(mask_i, mask_j):
                        collision_frame = frame_idx
                        # get centroids
                        cx_i, cy_i = get_mask_centroid(mask_i)
                        cx_j, cy_j = get_mask_centroid(mask_j)
                        collision_centroids = ((cx_i, cy_i), (cx_j, cy_j))
                        logging.info(f"Collision detected at frame {frame_idx} (Blob{idx_i} vs Blob{idx_j}).")
                        break
                if collision_frame is not None:
                    break

        # finalize and save overlay
        fig.tight_layout()
        out_path = os.path.join(args.processed_dir, f"frame_{frame_idx:06d}.png")
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        frames_processed_paths.append(out_path)
        if (frame_idx+1)%10==0:
            logging.info(f"Processed frame {frame_idx+1}")

        frame_idx += 1

    reader.close()
    total_frames = frame_idx
    logging.info(f"Total frames read: {total_frames}")

    # 7) Stitch frames_processed into final video
    logging.info(f"Writing final video to {args.output_video_path} with fps={fps}")
    writer = imageio.get_writer(args.output_video_path, fps=fps)
    for i in range(total_frames):
        fname = os.path.join(args.processed_dir, f"frame_{i:06d}.png")
        im_proc = imageio.imread(fname)
        writer.append_data(im_proc)
    writer.close()

    logging.info("Final video written.")

    # 8) If collision_frame was found, write collision info
    collision_info = {
        "collision_frame": collision_frame,
        "input_video": args.video_path,
        "output_video": args.output_video_path,
        "centroid_blob1": None,
        "centroid_blob2": None
    }
    if collision_frame is not None:
        ((cx1, cy1), (cx2, cy2)) = collision_centroids
        collision_info["centroid_blob1"] = [float(cx1), float(cy1)] if cx1 is not None else None
        collision_info["centroid_blob2"] = [float(cx2), float(cy2)] if cx2 is not None else None

    with open(args.collision_json, 'w') as f:
        json.dump(collision_info, f, indent=2)

    logging.info(f"Collision info saved to {args.collision_json}")
    logging.info("Done!")

if __name__ == "__main__":
    main()
