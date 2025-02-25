#!/usr/bin/env python3

import os
import argparse
import json
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

# DETR's 91 COCO Classes (for reference only)
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


def parse_model_prefix(model_path):
    """
    Given a model path containing .../trained_models/<SOMETHING>/...,
    extract <SOMETHING> as the 'model_prefix'.
    If not found, fallback to 'unknownModel'.
    """
    parts = model_path.split("trained_models/")
    if len(parts) < 2:
        return "unknownModel"
    after = parts[1]  # e.g. "myModel/some/other/..."
    # take substring up to first slash
    subparts = after.split("/")
    model_prefix = subparts[0]  # e.g. "myModel"
    if not model_prefix:
        return "unknownModel"
    return model_prefix


def parse_video_prefix(video_path):
    """
    Extract the base filename (without .mp4) and replace spaces with '+'.
    """
    base = os.path.basename(video_path)  # e.g. "BConcave+AConcave 3500.mp4"
    root, _ = os.path.splitext(base)     # e.g. "BConcave+AConcave 3500"
    # Replace spaces with '+'
    return root.replace(" ", "+")


def find_n_color_blobs(frame_np, n_blobs=2, black_thresh=30):
    """
    Heuristic detection of color blobs on black background.
    - 'black' if sum_of_RGB < black_thresh
    - pick up to n_blobs largest connected components
    """
    from skimage.measure import label, regionprops
    gray = frame_np.sum(axis=2)
    non_black = gray > black_thresh
    labeled = label(non_black, connectivity=2)
    regions = regionprops(labeled)
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
    Returns:
      assign: list of length num_blobs => submask_idx or None
      cost: cost matrix shape (nb, np_) = -IOU
    """
    nb = len(blob_masks)
    np_ = len(pred_masks)
    if np_ == 0:
        return [None]*nb, None

    cost = np.zeros((nb, np_), dtype=np.float32)
    for b in range(nb):
        for p in range(np_):
            cost[b,p] = -iou(blob_masks[b], pred_masks[p])

    row_idx, col_idx = linear_sum_assignment(cost)
    assign = [None]*nb
    for i in range(len(row_idx)):
        b = row_idx[i]
        p = col_idx[i]
        assign[b] = p
    return assign, cost


def make_masks_disjoint(masks):
    """
    Remove overlap among a list of boolean masks in-place.
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
    Use skimage.find_contours => pick largest boundary, then
    convert from (row,col) to (x-center_x, y-center_y).
    """
    from skimage.measure import find_contours
    if bin_mask is None or bin_mask.sum() == 0:
        return []
    cts = find_contours(bin_mask.astype(np.uint8), 0.5)
    if len(cts) == 0:
        return []
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
    Approx centroid (mean x, mean y).
    """
    if len(poly_pts) == 0:
        return None, None
    xs = [p[0] for p in poly_pts]
    ys = [p[1] for p in poly_pts]
    return (sum(xs)/len(xs), sum(ys)/len(ys))


def compute_memory_mask(list_of_submasks, height, width, a=1.3):
    """
    Weighted "memory" mask for a blob. More recent submasks = higher weight.
    sum_{i=0..N-1} (a^i * submask_i) / sum_{i=0..N-1} (a^i)
    then threshold at 0.5.
    """
    N = len(list_of_submasks)
    if N == 0:
        return np.zeros((height, width), dtype=bool)

    denom = 0.0
    for i in range(N):
        denom += (a**i)

    accum = np.zeros((height, width), dtype=float)
    for i, mask_i in enumerate(list_of_submasks):
        w = a**i
        accum += w * mask_i.astype(float)

    accum /= denom
    mem_bin = (accum > 0.5)
    return mem_bin


def main():
    parser = argparse.ArgumentParser("Script with sub-mask splitting + memory-based segmentations + skip frames + dynamic folder naming.")
    parser.add_argument("--model_path", required=True,
                        help="Path to custom DETR checkpoint (.pth), must contain 'trained_models/.../'")
    parser.add_argument("--video_path", default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/detr/EXPERIMENTS/generate_detection_videos_and_meshes/videos_org/BConcave+AConcave 3500.mp4",
                        help="Path to input .mp4")
    parser.add_argument("--n_blobs", type=int, default=2,
                        help="Number of color blobs to track (default=2)")
    parser.add_argument("--initial_skip_frames", type=int, default=13,
                        help="Number of initial frames to skip (copy as-is).")
    args = parser.parse_args()

    # 1) Parse prefixes
    model_prefix = parse_model_prefix(args.model_path)  # e.g. "myModel"
    video_prefix = parse_video_prefix(args.video_path)  # e.g. "BConcave+AConcave+3500"

    # Example folder naming => "myModel-BConcave+AConcave+3500-frames_blobs", etc.
    folder_root_blobs   = f"{model_prefix}-{video_prefix}-frames_blobs"
    folder_root_json    = f"{model_prefix}-{video_prefix}-frames_json"
    folder_root_memjson = f"{model_prefix}-{video_prefix}-frames_json_memory_processed"
    folder_root_collage = f"{model_prefix}-{video_prefix}-frames_collage"
    folder_root_memcollage = f"{model_prefix}-{video_prefix}-frames_memorycollage"
    folder_root_proc    = f"{model_prefix}-{video_prefix}-frames_processed"
    folder_root_videos  = f"{model_prefix}-{video_prefix}-videos_processed"

    # The final video file path => e.g. "myModel-BConcave+AConcave+3500-videos_processed/BConcave+AConcave+3500.mp4"
    final_video_path = os.path.join(folder_root_videos, f"{video_prefix}.mp4")

    # 2) Create local folders
    for d in [folder_root_blobs, folder_root_json, folder_root_memjson,
              folder_root_collage, folder_root_memcollage, folder_root_proc,
              folder_root_videos]:
        if not os.path.exists(d):
            os.makedirs(d)

    # 3) Load model
    model = load_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4) Read input video
    print(f"Reading input video {args.video_path}")
    reader = imageio.get_reader(args.video_path, format='ffmpeg')
    meta_in = reader.get_meta_data()
    fps = meta_in.get('fps', 30)

    # Attempt first frame to get shape
    try:
        first_frame = reader.get_data(0)
        H, W, _ = first_frame.shape
        print(f"Video shape: W={W}, H={H}, fps={fps}")
    except:
        print("Could not read first frame to determine shape.")
        reader.close()
        return

    # Keep a memory of submasks for each of the n_blobs
    blob_memories = [[] for _ in range(args.n_blobs)]

    reader.set_image_index(0)
    frame_idx = 0
    frames_processed_paths = []

    while True:
        try:
            frame = reader.get_data(frame_idx)
        except IndexError:
            # no more frames
            break

        # If shape doesn't match, do quick fix
        if frame.shape[0] != H or frame.shape[1] != W:
            corrected = np.zeros((H,W,3), dtype=frame.dtype)
            h_ = min(H, frame.shape[0])
            w_ = min(W, frame.shape[1])
            corrected[0:h_, 0:w_, :] = frame[0:h_, 0:w_, :]
            frame = corrected

        # Skip initial frames
        if frame_idx < args.initial_skip_frames:
            # save an empty JSON for both normal and memory
            # just to be consistent
            empty_json = {}
            with open(os.path.join(folder_root_json, f"frame_{frame_idx:06d}.json"), 'w') as f:
                json.dump(empty_json, f, indent=2)
            with open(os.path.join(folder_root_memjson, f"frame_{frame_idx:06d}.json"), 'w') as f:
                json.dump(empty_json, f, indent=2)

            # Save "processed" image as is
            out_path = os.path.join(folder_root_proc, f"frame_{frame_idx:06d}.png")
            Image.fromarray(frame).save(out_path)
            frames_processed_paths.append(out_path)

            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Skipped frame {frame_idx}")
            continue

        # ========== 1) Find color blobs
        blob_masks = find_n_color_blobs(frame, n_blobs=args.n_blobs)
        nb = len(blob_masks)

        # debug => color them
        debug_blob = frame.astype(np.float32).copy()
        color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
        for i, bm in enumerate(blob_masks):
            c = color_list[i % len(color_list)]
            debug_blob[bm,0] = c[0]
            debug_blob[bm,1] = c[1]
            debug_blob[bm,2] = c[2]
        debug_blob_path = os.path.join(folder_root_blobs, f"frame_{frame_idx:06d}_blobs.png")
        Image.fromarray(debug_blob.astype(np.uint8)).save(debug_blob_path)

        # ========== 2) Run DETR => upsample => split submasks
        pil_img = Image.fromarray(frame, mode="RGB")
        transform_resize = T.Resize(800)
        resized_img = transform_resize(pil_img)
        transform_norm = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        model_in = transform_norm(resized_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(model_in)
        pred_logits = outputs["pred_logits"]  # (1,100,92)
        pred_masks  = outputs["pred_masks"]   # (1,100,h',w')

        n_queries = pred_logits.shape[1]
        up_masks = []
        for i in range(n_queries):
            ml = pred_masks[0,i].unsqueeze(0).unsqueeze(0)
            up = F.interpolate(ml, size=(H,W), mode="bilinear", align_corners=False)
            pm = torch.sigmoid(up).squeeze(0).squeeze(0).cpu().numpy()
            bin_m = (pm > 0.5)
            up_masks.append(bin_m)

        # split disjoint submasks
        split_pred_masks = []
        orig_index_for_submask = []
        for q_idx, bin_m in enumerate(up_masks):
            labeled = label(bin_m, connectivity=2)
            num_cc = labeled.max()
            for cc in range(1, num_cc+1):
                submask = (labeled == cc)
                split_pred_masks.append(submask)
                orig_index_for_submask.append(q_idx)

        # ========== 3) If no blobs => empty JSON + copy
        if nb == 0:
            empty_json = {}
            # normal polygons
            with open(os.path.join(folder_root_json, f"frame_{frame_idx:06d}.json"), 'w') as f:
                json.dump(empty_json, f, indent=2)
            # memory polygons
            with open(os.path.join(folder_root_memjson, f"frame_{frame_idx:06d}.json"), 'w') as f:
                json.dump(empty_json, f, indent=2)

            out_path = os.path.join(folder_root_proc, f"frame_{frame_idx:06d}.png")
            Image.fromarray(frame).save(out_path)
            frames_processed_paths.append(out_path)

            frame_idx += 1
            if frame_idx%10 == 0:
                print(f"Processed frame {frame_idx}")
            continue

        # ========== 4) Bipartite assignment
        assign, cost = bipartite_assign_blobs_to_masks(blob_masks, split_pred_masks)

        # Collage of top-10 minimal cost (for frames>=30)
        if cost is not None and frame_idx >= 30 and nb>0:
            fig, axes = plt.subplots(nb, 10, figsize=(25, 5*nb), dpi=100)
            if nb == 1:
                axes = np.array([axes])  # shape => (1,10)

            for b_idx in range(nb):
                row_costs = cost[b_idx, :]
                idx_sorted = np.argsort(row_costs)
                best_10 = idx_sorted[:10]
                for rank_i, sm_idx in enumerate(best_10):
                    ax = axes[b_idx, rank_i]
                    overlay = frame.copy()
                    blob_m = blob_masks[b_idx]
                    sub_m = split_pred_masks[sm_idx]

                    # mark blob in green
                    overlay[blob_m, 0] = 0
                    overlay[blob_m, 1] = 255
                    overlay[blob_m, 2] = 0
                    # submask in red
                    overlay[sub_m, 0] = 255
                    overlay[sub_m, 1] = 0
                    overlay[sub_m, 2] = 0

                    cost_val = row_costs[sm_idx]
                    orig_q = orig_index_for_submask[sm_idx]

                    ax.imshow(overlay)
                    ax.set_title(f"Blob {b_idx}, sub={sm_idx}\n(orig={orig_q}), cost={cost_val:.3f}",
                                 fontsize=9)
                    ax.set_axis_off()

            fig.suptitle(f"Frame {frame_idx} - top 10 minimal-cost sub-masks per blob", fontsize=16)
            fig.tight_layout()
            collage_path = os.path.join(folder_root_collage, f"frame_{frame_idx:06d}_collage.png")
            fig.savefig(collage_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        # assigned submasks
        assigned_submasks = []
        for b_idx in range(nb):
            submask_idx = assign[b_idx]
            if submask_idx is not None:
                assigned_submasks.append(split_pred_masks[submask_idx])
            else:
                assigned_submasks.append(None)

        # ========== 5) Update memory
        for b_idx in range(nb):
            if assigned_submasks[b_idx] is not None:
                blob_memories[b_idx].append(assigned_submasks[b_idx])

        # ========== 6) Compute memory-based masks
        memory_masks = []
        for b_idx in range(nb):
            mem_mask = compute_memory_mask(blob_memories[b_idx], H, W, a=1.3)
            memory_masks.append(mem_mask)

        # ========== 7) Output a memory collage: current assigned vs memory
        fig, axes = plt.subplots(nb, 2, figsize=(10, 5*nb), dpi=100)
        if nb == 1:
            axes = np.array([axes])  # shape => (1,2)

        for b_idx in range(nb):
            ax_left = axes[b_idx, 0]
            ax_right = axes[b_idx, 1]

            # Left => assigned submask
            overlay_cur = frame.copy()
            if assigned_submasks[b_idx] is not None:
                overlay_cur[assigned_submasks[b_idx], 0] = 255
                overlay_cur[assigned_submasks[b_idx], 1] = 0
                overlay_cur[assigned_submasks[b_idx], 2] = 0
            ax_left.imshow(overlay_cur)
            ax_left.set_title(f"Blob {b_idx} - Current submask", fontsize=10)
            ax_left.set_axis_off()

            # Right => memory
            overlay_mem = frame.copy()
            overlay_mem[memory_masks[b_idx], 0] = 0
            overlay_mem[memory_masks[b_idx], 1] = 255
            overlay_mem[memory_masks[b_idx], 2] = 0
            ax_right.imshow(overlay_mem)
            ax_right.set_title(f"Blob {b_idx} - Memory mask", fontsize=10)
            ax_right.set_axis_off()

        fig.suptitle(f"Frame {frame_idx} - Memory Collage", fontsize=16)
        fig.tight_layout()
        mem_collage_path = os.path.join(folder_root_memcollage, f"frame_{frame_idx:06d}_memcollage.png")
        fig.savefig(mem_collage_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # ========== 8) Polygons for "assigned" submasks => store in frames_json
        center_x = W/2.0
        center_y = H/2.0
        # make disjoint, but for assigned submasks we do them as-is or we do disjoint?
        assigned_disjoint = make_masks_disjoint(assigned_submasks.copy())

        assigned_info = []
        for b_idx, msk in enumerate(assigned_disjoint):
            if msk is None or msk.sum()==0:
                assigned_info.append((b_idx, None, 0))
            else:
                coords = np.argwhere(msk)
                mean_col = coords[:,1].mean()
                assigned_info.append((b_idx, msk, mean_col))

        assigned_info.sort(key=lambda x: x[2])
        assigned_polygons_json = {}
        for order_idx, (b_idx, m, mean_c) in enumerate(assigned_info):
            poly = find_contour_polygon(m, center_x, center_y) if m is not None else []
            assigned_polygons_json[f"segmentation_blob_{order_idx}"] = poly

        # store assigned-submask polygons
        assigned_json_path = os.path.join(folder_root_json, f"frame_{frame_idx:06d}.json")
        with open(assigned_json_path, 'w') as f:
            json.dump(assigned_polygons_json, f, indent=2)

        # ========== 9) Polygons for memory-based final => store in frames_json_memory_processed
        final_mem_disjoint = make_masks_disjoint(memory_masks.copy())

        mem_info = []
        for b_idx, msk in enumerate(final_mem_disjoint):
            if msk is None or msk.sum()==0:
                mem_info.append((b_idx, None, 0))
            else:
                coords = np.argwhere(msk)
                mean_col = coords[:,1].mean()
                mem_info.append((b_idx, msk, mean_col))

        mem_info.sort(key=lambda x: x[2])
        memory_polygons_json = {}
        for order_idx, (b_idx, m, mean_c) in enumerate(mem_info):
            poly = find_contour_polygon(m, center_x, center_y) if m is not None else []
            memory_polygons_json[f"segmentation_blob_{order_idx}"] = poly

        mem_json_path = os.path.join(folder_root_memjson, f"frame_{frame_idx:06d}.json")
        with open(mem_json_path, 'w') as f:
            json.dump(memory_polygons_json, f, indent=2)

        # ========== 10) Final overlay => uses memory polygons => frames_processed
        fig_dpi=100
        fig_w = W/fig_dpi
        fig_h = H/fig_dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
        ax.imshow(frame)
        ax.set_axis_off()

        color_maps = ['Reds','Blues','Greens','Oranges','Purples','pink','YlOrBr']
        for i, (b_idx, mask_b, mean_c) in enumerate(mem_info):
            poly_pts = memory_polygons_json[f"segmentation_blob_{i}"]
            if len(poly_pts)==0:
                continue
            cmap = color_maps[i % len(color_maps)]
            color_rgba = cm.get_cmap(cmap)(0.6)

            xs_tl = []
            ys_tl = []
            for (xC, yC) in poly_pts:
                xs_tl.append(xC + center_x)
                ys_tl.append(yC + center_y)
            ax.fill(xs_tl, ys_tl, alpha=0.4, color=color_rgba, linewidth=0)

            cx, cy = polygon_centroid([(xx,yy) for (xx,yy) in zip(xs_tl, ys_tl)])
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
        out_path = os.path.join(folder_root_proc, f"frame_{frame_idx:06d}.png")
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        frames_processed_paths.append(out_path)
        frame_idx += 1

        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}")

    # done reading
    reader.close()
    total_frames = frame_idx
    print(f"Total frames read: {total_frames}")

    # ========== 11) Stitch frames_processed => final video
    print(f"Writing final video to {final_video_path} with fps={fps}")
    writer = imageio.get_writer(final_video_path, fps=fps)
    for i in range(total_frames):
        fname = os.path.join(folder_root_proc, f"frame_{i:06d}.png")
        im_proc = imageio.imread(fname)
        writer.append_data(im_proc)
    writer.close()

    print("Done!")


if __name__ == "__main__":
    main()
