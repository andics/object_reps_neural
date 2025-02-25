#!/usr/bin/env python3
"""
final_gen_video.py

Modified to:
1) Split each DETR query mask into disjoint sub-masks before bipartite matching.
2) Handle the case where fewer than `n_blobs` are detected:
   - If 0 blobs => skip matching, produce empty JSON, etc.
   - If 1 blob => produce a single row in the collage, handle axis indexing properly.
3) Still produce the top-10 minimal-cost sub-masks per blob for frames >= 30.
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

from PIL import Image
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from skimage.measure import label, regionprops, find_contours
from scipy.optimize import linear_sum_assignment

torch.set_grad_enabled(False)

# -------------- DETR's 91 COCO Classes (not necessarily needed, but here for reference)
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


def load_model(model_path):
    """
    Load DETR (panoptic variant) from torch hub, then a custom checkpoint.
    We do NOT use the postprocessor.
    """
    print("Loading DETR panoptic model from torch hub (no postprocessor).")
    model, _ = torch.hub.load(
        'facebookresearch/detr',
        'detr_resnet101_panoptic',
        pretrained=False,
        return_postprocessor=True,
        num_classes=91
    )
    print(f"Loading checkpoint from {model_path}")
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
    Return a list of boolean masks in descending order of area (could be fewer than n_blobs).
    """
    gray = frame_np.sum(axis=2)  # shape (H, W)
    non_black = gray > black_thresh
    labeled = label(non_black, connectivity=2)
    regions = regionprops(labeled)
    sorted_regs = sorted(regions, key=lambda r: r.area, reverse=True)

    # If fewer regions exist than n_blobs, we'll just return all of them
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

    Returns:
      assign: a list of length num_blobs where assign[b] = pred_idx or None
      cost: the cost matrix of shape (num_blobs, num_preds)
            cost[b,p] = -iou(blob_masks[b], pred_masks[p]).
    """
    nb = len(blob_masks)
    np_ = len(pred_masks)
    if np_ == 0:
        return [None] * nb, None

    cost = np.zeros((nb, np_), dtype=np.float32)
    for b in range(nb):
        for p in range(np_):
            cost[b, p] = -iou(blob_masks[b], pred_masks[p])

    row_idx, col_idx = linear_sum_assignment(cost)  # minimize negative => maximize IOU
    assign = [None] * nb
    for i in range(len(row_idx)):
        b = row_idx[i]
        p = col_idx[i]
        assign[b] = p
    return assign, cost


def make_masks_disjoint(masks):
    """
    We remove overlap among a list of boolean masks in-place.
    For example, in order: for i in range(len(masks)):
       for j in range(i+1, len(masks)):
         masks[j] = masks[j] & ~masks[i]
    This ensures final masks do not overlap. Masks are likely assigned to different blobs.
    Return updated list.
    """
    for i in range(len(masks)):
        if masks[i] is None:
            continue
        for j in range(i + 1, len(masks)):
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
        poly.append((x, y))
    return poly


def polygon_centroid(poly_pts):
    """
    Approximate centroid of a polygon in top-left coords. If empty, return (None,None).
    We'll do a simple arithmetic mean.
    """
    if len(poly_pts) == 0:
        return None, None
    xs = [p[0] for p in poly_pts]
    ys = [p[1] for p in poly_pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def main():
    parser = argparse.ArgumentParser("Final script with sub-mask splitting, top-10 collage, etc.")
    parser.add_argument("--model_path", required=True, help="Path to custom DETR checkpoint (.pth)")
    parser.add_argument("--video_path", required=True, help="Path to input .mp4")
    parser.add_argument("--output_video_path", required=True, help="Path to output .mp4 (same resolution)")
    parser.add_argument("--n_blobs", type=int, default=2, help="Number of color blobs to track (default=2)")
    parser.add_argument("--blobs_dir", default="frames_blobs", help="Dir to save color-blob debug images.")
    parser.add_argument("--processed_dir", default="frames_processed", help="Dir to save final overlay frames.")
    parser.add_argument("--json_dir", default="frames_json", help="Dir to save JSON for each frame.")
    parser.add_argument("--collage_dir", default="frames_collage",
                        help="Dir to save top-10 mask collage (for frames > 30).")
    args = parser.parse_args()

    # 1) Load the model
    model = load_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2) Prepare local folders
    for d in [args.blobs_dir, args.processed_dir, args.json_dir, args.collage_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # 3) Read input video with imageio
    print(f"Reading input video {args.video_path}")
    reader = imageio.get_reader(args.video_path, format='ffmpeg')
    meta_in = reader.get_meta_data()
    fps = meta_in.get('fps', 30)

    # try first frame to get W,H
    try:
        first_frame = reader.get_data(0)
        H, W, _ = first_frame.shape
        print(f"Video shape: W={W}, H={H}, fps={fps}")
    except:
        print("Could not read first frame to determine shape.")
        reader.close()
        return
    # reset
    reader.set_image_index(0)

    frame_idx = 0
    frames_processed_paths = []

    for frame in reader:
        # Ensure shape matches W,H
        if frame.shape[0] != H or frame.shape[1] != W:
            # if something is off, let's fix it with a quick pad or resize
            corrected = np.zeros((H, W, 3), dtype=frame.dtype)
            h_ = min(H, frame.shape[0])
            w_ = min(W, frame.shape[1])
            corrected[0:h_, 0:w_, :] = frame[0:h_, 0:w_, :]
            frame = corrected

        ### 3a) Find color-blobs
        blob_masks = find_n_color_blobs(frame, n_blobs=args.n_blobs)
        nb = len(blob_masks)  # actual number of blobs found (can be 0,1,...,n_blobs)

        # Let's color them for debug
        debug_blob = frame.astype(np.float32).copy()
        color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for i, bm in enumerate(blob_masks):
            c = color_list[i % len(color_list)]
            debug_blob[bm, 0] = c[0]
            debug_blob[bm, 1] = c[1]
            debug_blob[bm, 2] = c[2]
        debug_blob_path = os.path.join(args.blobs_dir, f"frame_{frame_idx:06d}_blobs.png")
        Image.fromarray(debug_blob.astype(np.uint8)).save(debug_blob_path)

        ### 3b) DETR predicted masks => upsample
        pil_img = Image.fromarray(frame, mode="RGB")
        transform_resize = T.Resize(800)
        resized_img = transform_resize(pil_img)
        rw, rh = resized_img.size
        transform_norm = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model_in = transform_norm(resized_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(model_in)
        pred_logits = outputs["pred_logits"]  # (1,100,92)
        pred_masks = outputs["pred_masks"]  # (1,100,h',w')

        n_queries = pred_logits.shape[1]
        up_masks = []
        for i in range(n_queries):
            ml = pred_masks[0, i].unsqueeze(0).unsqueeze(0)  # (1,1,h',w')
            up = F.interpolate(ml, size=(H, W), mode="bilinear", align_corners=False)
            pm = torch.sigmoid(up).squeeze(0).squeeze(0).cpu().numpy()
            bin_m = (pm > 0.5)
            up_masks.append(bin_m)

        ### (NEW) Split each predicted mask into disjoint sub-masks
        split_pred_masks = []
        orig_index_for_submask = []
        for q_idx, bin_m in enumerate(up_masks):
            labeled = label(bin_m, connectivity=2)
            num_cc = labeled.max()
            for cc_label in range(1, num_cc + 1):
                submask = (labeled == cc_label)
                split_pred_masks.append(submask)
                orig_index_for_submask.append(q_idx)

        # If no blobs found => store empty JSON & skip collage
        if nb == 0:
            # Save an empty JSON
            json_path = os.path.join(args.json_dir, f"frame_{frame_idx:06d}.json")
            with open(json_path, 'w') as f:
                json.dump({}, f, indent=2)

            # Just save the original frame as processed (no polygons)
            fig_dpi = 100
            fig_w = W / fig_dpi
            fig_h = H / fig_dpi
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
            ax.imshow(frame)
            ax.set_axis_off()
            fig.tight_layout()
            out_path = os.path.join(args.processed_dir, f"frame_{frame_idx:06d}.png")
            fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            frames_processed_paths.append(out_path)
            frame_idx += 1
            if (frame_idx) % 10 == 0:
                print(f"Processed frame {frame_idx}")
            continue

        ### 3c) bipartite assignment
        assign, cost = bipartite_assign_blobs_to_masks(blob_masks, split_pred_masks)

        # 3d) If frame_idx >= 30, create collage for top-10 minimal cost sub-masks per blob
        #     but only if nb>0
        if cost is not None and frame_idx >= 30 and nb > 0:
            # cost shape => (nb, n_submasks)
            # create figure with nb rows, each row => 10 subplots
            fig, axes = plt.subplots(nb, 10, figsize=(25, 5 * nb), dpi=100)

            # If nb==1, axes is shape (10,) => make it (1,10) to index as axes[b_idx, col]
            if nb == 1:
                axes = np.array([axes])  # shape => (1, 10)

            for b_idx in range(nb):
                row_costs = cost[b_idx, :]
                idx_sorted = np.argsort(row_costs)  # ascending => best IOU => top-10
                best_10 = idx_sorted[:10]

                for rank_i, submask_idx in enumerate(best_10):
                    # This reference will work for nb>=1
                    ax = axes[b_idx, rank_i]

                    overlay = frame.copy()
                    blob_m = blob_masks[b_idx]
                    sub_m = split_pred_masks[submask_idx]
                    # Mark the blob in green
                    overlay[blob_m, 0] = 0
                    overlay[blob_m, 1] = 255
                    overlay[blob_m, 2] = 0
                    # Mark the predicted submask in red
                    overlay[sub_m, 0] = 255
                    overlay[sub_m, 1] = 0
                    overlay[sub_m, 2] = 0

                    cost_val = row_costs[submask_idx]
                    orig_q = orig_index_for_submask[submask_idx]

                    ax.imshow(overlay)
                    ax.set_title(
                        f"Blob {b_idx}, submask {submask_idx}\n(orig mask={orig_q}), cost={cost_val:.3f}",
                        fontsize=9
                    )
                    ax.set_axis_off()

            fig.suptitle(f"Frame {frame_idx} - top 10 minimal-cost sub-masks per blob", fontsize=16)
            fig.tight_layout()
            collage_path = os.path.join(args.collage_dir, f"frame_{frame_idx:06d}_collage.png")
            fig.savefig(collage_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        # build final assigned mask list in the same order as blob_masks
        assigned = []
        for b_idx in range(nb):
            submask_idx = assign[b_idx] if assign else None
            if submask_idx is not None:
                assigned.append(split_pred_masks[submask_idx])
            else:
                assigned.append(None)

        # Ensure disjoint
        assigned = make_masks_disjoint(assigned)

        # 3e) from each assigned submask, find polygon => store JSON
        center_x = W / 2.0
        center_y = H / 2.0

        # Re-sort them left->right by their centroid's column
        blob_info = []
        for b_idx, mask_b in enumerate(assigned):
            if mask_b is None or mask_b.sum() == 0:
                blob_info.append((b_idx, None, 0))
            else:
                coords = np.argwhere(mask_b)
                mean_col = coords[:, 1].mean()
                blob_info.append((b_idx, mask_b, mean_col))

        blob_info.sort(key=lambda x: x[2])
        polygons_json = {}
        for order_idx, (b_idx, m, mean_c) in enumerate(blob_info):
            poly = find_contour_polygon(m, center_x, center_y) if m is not None else []
            polygons_json[f"segmentation_blob_{order_idx}"] = poly

        # store the JSON
        json_path = os.path.join(args.json_dir, f"frame_{frame_idx:06d}.json")
        with open(json_path, 'w') as f:
            json.dump(polygons_json, f, indent=2)

        ### 3f) overlay these polygons on the frame => store in frames_processed
        fig_dpi = 100
        fig_w = W / fig_dpi
        fig_h = H / fig_dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
        ax.imshow(frame)  # shape is top-left origin
        ax.set_axis_off()

        # color maps for each polygon
        color_maps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'pink', 'YlOrBr']

        for i, (b_idx, mask_b, mean_c) in enumerate(blob_info):
            poly_pts = polygons_json[f"segmentation_blob_{i}"]
            cmap = color_maps[i % len(color_maps)]
            color = cm.get_cmap(cmap)(0.6)  # RGBA

            if len(poly_pts) == 0:
                continue

            xs_tl = []
            ys_tl = []
            for (xC, yC) in poly_pts:
                xs_tl.append(xC + center_x)
                ys_tl.append(yC + center_y)

            # fill polygon
            ax.fill(xs_tl, ys_tl, alpha=0.4, color=color, linewidth=0)

            # label near centroid
            cx, cy = polygon_centroid([(xx, yy) for (xx, yy) in zip(xs_tl, ys_tl)])
            if cx is not None and cy is not None:
                ax.text(
                    x=cx,
                    y=cy,
                    s=f"Blob {i}",
                    color="black",
                    fontsize=14,
                    fontweight="bold",
                    bbox=dict(facecolor='white', alpha=0.7, pad=2),
                    ha='center', va='center', zorder=999
                )

        fig.tight_layout()
        out_path = os.path.join(args.processed_dir, f"frame_{frame_idx:06d}.png")
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        frames_processed_paths.append(out_path)

        if (frame_idx + 1) % 10 == 0:
            print(f"Processed frame {frame_idx + 1}")
        frame_idx += 1

    reader.close()
    total_frames = frame_idx
    print(f"Total frames read: {total_frames}")

    ### 4) Stitch frames_processed into final video
    print(f"Writing final video to {args.output_video_path} with fps={fps}")
    writer = imageio.get_writer(args.output_video_path, fps=fps)
    for i in range(total_frames):
        fname = os.path.join(args.processed_dir, f"frame_{i:06d}.png")
        im_proc = imageio.imread(fname)
        writer.append_data(im_proc)
    writer.close()

    print("Done!")


if __name__ == "__main__":
    main()
