#!/usr/bin/env python3
"""
main_gen_vids_w_meshes_and_mem_decay_no_cv2_collage.py

Changes:
 - We catch StopIteration or EOFError properly, so we don't crash when video ends.
 - Instead of overlaying both current and memory masks on the same image, we create
   a **collage** side by side: left half shows 'current' segmentation, right half
   shows 'memory' segmentation. This makes them clearly distinguishable.

Steps:
1) Read frames with imageio.
2) Heuristic detection of up to two color blobs, ignoring black.
3) DETR segmentation, bipartite assignment, disjoint masks, memory update with alpha/beta.
4) Output per frame:
   - heuristic_blobs_frame_XXXXXX.png
   - current_segm_mask_frame_XXXXXX.json
   - current_segm_mask_frame_XXXXXX.png
   - memory_frame_mask_frame_XXXXXX.json
   - memory_segm_mask_frame_XXXXXX.png
   - collage_frame_XXXXXX.png ( side-by-side collage: left=Current, right=Memory )
5) Finally, stitch all collage_frame_XXXXXX.png => final mp4.

No collision detection, no arrow drawing, no OpenCV usage.
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from collections import OrderedDict

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from skimage.measure import label, regionprops, find_contours
from scipy.optimize import linear_sum_assignment

torch.set_grad_enabled(False)


def parse_args():
    """
    Only model_path and video_path are truly needed. The rest have defaults.
    """
    parser = argparse.ArgumentParser(
        "Script for two-blob segmentation with memory decay and collage output, no cv2 usage."
    )

    # Minimal required arguments, but we also give them defaults for convenience:
    parser.add_argument(
        "--model_path",
        default="/home/projects/bagon/andreyg/Projects/Variable_Resolution_DETR/Programming/detr_var/trained_models/variable_pretrained_resnet101/box_and_segm/checkpoint.pth",
        help="Path to custom DETR checkpoint (.pth)."
    )
    parser.add_argument(
        "--video_path",
        default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/detr/EXPERIMENTS/generate_detection_vids_w_meshes_and_mem_decay/videos_org/BConcave+AConcave 3500.mp4",
        help="Path to the input .mp4."
    )

    # All the rest are optional, with sensible defaults:
    parser.add_argument(
        "--output_video_path",
        default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/detr/EXPERIMENTS/generate_detection_vids_w_meshes_and_mem_decay/videos_processed/BConcave+AConcave 3500.mp4",
        help="Path to output .mp4 (stitched from collage images)."
    )
    parser.add_argument(
        "--n_blobs",
        type=int,
        default=2,
        help="Number of color blobs (max) to track initially (default=2)."
    )
    parser.add_argument(
        "--blobs_dir",
        default="frames_blobs",
        help="Dir to save color-blob debug images."
    )
    parser.add_argument(
        "--processed_dir",
        default="frames_processed",
        help="Dir to save final overlay frames (current, memory, collage)."
    )
    parser.add_argument(
        "--json_dir",
        default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/detr/EXPERIMENTS/generate_detection_vids_w_meshes_and_mem_decay/frames_json_processed",
        help="Dir to save JSON for each frame (current + memory)."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Exponential smoothing weight for current frame (0<alpha<=1)."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Decay factor if blob is missing (0<beta<=1). 1=no fade."
    )
    parser.add_argument(
        "--disappearance_threshold",
        type=int,
        default=3,
        help="Consecutive frames missing => blob is disappeared."
    )

    args = parser.parse_args()
    return args


def load_model(model_path):
    """
    Load DETR (panoptic variant) from torch.hub, then apply
    a custom checkpoint. We do NOT use the postprocessor.
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
    - 'black' => sum_of_RGB < black_thresh
    - label connected non-black pixels, pick top n largest
    Returns a list of boolean masks, sorted by area desc.
    """
    gray = frame_np.sum(axis=2)  # shape (H, W)
    non_black = gray > black_thresh
    labeled = label(non_black, connectivity=2)
    regions = regionprops(labeled)
    sorted_regs = sorted(regions, key=lambda r: r.area, reverse=True)
    top_regs = sorted_regs[:n_blobs]

    masks = []
    for r in top_regs:
        mask = (labeled == r.label)
        masks.append(mask)
    return masks


def iou(maskA, maskB):
    inter = (maskA & maskB).sum()
    union = (maskA | maskB).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def bipartite_assign_blobs_to_masks(blob_masks, pred_masks):
    """
    cost matrix = -IOU => Hungarian => maximize total IOU.
    Returns list of length num_blobs: index of matched pred_mask or None.
    """
    nb = len(blob_masks)
    np_ = len(pred_masks)
    if np_ == 0:
        return [None]*nb

    import numpy as np
    from scipy.optimize import linear_sum_assignment

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
    return assign


def make_masks_disjoint(masks):
    """
    Make a list of boolean masks disjoint in-place:
      for i in range(len(masks)):
        for j in range(i+1, len(masks)):
          masks[j] = masks[j] & ~masks[i]
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
    Using skimage.find_contours => pick largest boundary => convert
    from row,col to (x=col-center_x, y=row-center_y). Returns a list
    of (x,y). If mask empty => [].
    """
    if bin_mask is None or bin_mask.sum()==0:
        return []
    cts = find_contours(bin_mask.astype(np.uint8), 0.5)
    if len(cts) == 0:
        return []
    biggest_ct = max(cts, key=lambda c: c.shape[0])
    poly = []
    for point in biggest_ct:
        r, c = point
        x = c - center_x
        y = r - center_y
        poly.append((x,y))
    return poly


def polygon_centroid(poly_pts):
    """
    Returns arithmetic mean (centroid) of (x,y) points. If empty => (None,None).
    """
    if not poly_pts:
        return None, None
    xs = [p[0] for p in poly_pts]
    ys = [p[1] for p in poly_pts]
    return (sum(xs)/len(xs), sum(ys)/len(ys))


def threshold_mask(m, thresh=0.5):
    """
    Convert float array (0..1) to boolean by threshold. Return None if m is None.
    """
    if m is None:
        return None
    return (m > thresh)


def overlay_polygons(base_frame, polygons_dict, color_dict,
                     fill_alpha=0.4, center_x=0, center_y=0, label_suffix=""):
    """
    Overlays polygons in a single panel using matplotlib, returns final RGB image array.
    polygons_dict: blob_id -> list of (x_center,y_center)
    color_dict: blob_id -> (R,G,B)
    """
    H, W, _ = base_frame.shape
    fig_dpi=100
    fig_w = W/fig_dpi
    fig_h = H/fig_dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
    ax.imshow(base_frame)
    ax.set_axis_off()

    for i, poly_pts in polygons_dict.items():
        if not poly_pts:
            continue
        c = color_dict[i]
        col_rgba = (c[0]/255.0, c[1]/255.0, c[2]/255.0, fill_alpha)
        xs = [p[0] + center_x for p in poly_pts]
        ys = [p[1] + center_y for p in poly_pts]
        ax.fill(xs, ys, color=col_rgba)
        cx, cy = polygon_centroid(list(zip(xs, ys)))
        if cx is not None and cy is not None:
            ax.text(cx, cy, f"Blob {i}{label_suffix}",
                    color="white", fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.5, pad=2),
                    ha='center', va='center')

    fig.tight_layout()
    fig.canvas.draw()
    overlay_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    overlay_img = overlay_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return overlay_img


def main():
    args = parse_args()

    # create directories
    os.makedirs(args.blobs_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.json_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading DETR model ...")
    model = load_model(args.model_path).to(device)

    print(f"Reading video: {args.video_path} ...")
    reader = imageio.get_reader(args.video_path, format='ffmpeg')
    meta_in = reader.get_meta_data()
    fps = meta_in.get('fps', 30)

    # Attempt to get shape from first frame
    try:
        first_frame = reader.get_data(0)
        H, W, _ = first_frame.shape
        print(f"Video shape: W={W}, H={H}, fps={fps}")
    except:
        print("Could not read first frame. Exiting.")
        return
    # reset
    reader.set_image_index(0)

    # track up to n_blobs => store in a list of dict
    color_list = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
    blob_info_list = []
    for i in range(args.n_blobs):
        blob_info_list.append({
            'avg_mask': None,
            'missing_count': 0,
            'disappeared': False,
            'color': color_list[i%len(color_list)]
        })

    frame_idx = 0

    def get_detr_masks(img_np, full_w, full_h):
        """
        Runs DETR on the image, upsamples masks to full size, returns list of boolean masks.
        """
        pil_img = Image.fromarray(img_np, mode="RGB")
        transform_resize = T.Resize(800)
        resized_img = transform_resize(pil_img)
        transform_norm = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        x = transform_norm(resized_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
        pred_masks = out["pred_masks"]  # shape (1,100,H',W')
        n_queries = pred_masks.shape[1]
        up_masks = []
        for i in range(n_queries):
            m_small = pred_masks[0,i].unsqueeze(0).unsqueeze(0)  # (1,1,h',w')
            m_up = F.interpolate(m_small, size=(full_h, full_w), mode="bilinear", align_corners=False)
            pm = torch.sigmoid(m_up).squeeze().cpu().numpy()
            bin_m = (pm > 0.5)
            up_masks.append(bin_m)
        return up_masks

    while True:
        try:
            frame = reader.get_next_data()
        except (EOFError, StopIteration, IndexError):
            # This handles the end-of-video gracefully
            break

        if frame.shape[0] != H or frame.shape[1] != W:
            # fix shape if needed
            corrected = np.zeros((H,W,3), dtype=frame.dtype)
            h_ = min(H, frame.shape[0])
            w_ = min(W, frame.shape[1])
            corrected[:h_, :w_, :] = frame[:h_, :w_, :]
            frame = corrected

        # how many active
        active_ids = [i for i,bi in enumerate(blob_info_list) if not bi['disappeared']]
        n_active = len(active_ids)

        if n_active > 0:
            heuristic_n = n_active
        else:
            heuristic_n = 0

        # color blobs
        color_blob_masks = find_n_color_blobs(frame, n_blobs=heuristic_n)

        # DETR
        detr_masks = get_detr_masks(frame, W, H)

        # bipartite assign
        assign = bipartite_assign_blobs_to_masks(color_blob_masks, detr_masks)
        assigned = []
        for b_idx in range(len(color_blob_masks)):
            p_idx = assign[b_idx]
            if p_idx is not None:
                assigned.append(detr_masks[p_idx])
            else:
                assigned.append(None)
        assigned = make_masks_disjoint(assigned)

        # map assigned to global
        assigned_in_global = [None]*len(blob_info_list)
        for idx,gid in enumerate(active_ids):
            if idx < len(assigned):
                assigned_in_global[gid] = assigned[idx]

        # update memory
        for i, bi in enumerate(blob_info_list):
            if bi['disappeared']:
                continue
            m_cur = assigned_in_global[i]
            if m_cur is not None and m_cur.sum()>0:
                bi['missing_count'] = 0
                if bi['avg_mask'] is None:
                    bi['avg_mask'] = m_cur.astype(np.float32)
                else:
                    new_f = m_cur.astype(np.float32)
                    old_f = bi['avg_mask']
                    # average
                    bi['avg_mask'] = args.alpha*new_f + (1-args.alpha)*old_f
            else:
                # missing
                bi['missing_count'] += 1
                if bi['missing_count'] >= args.disappearance_threshold:
                    bi['disappeared'] = True
                else:
                    if bi['avg_mask'] is not None:
                        bi['avg_mask'] = args.beta * bi['avg_mask']

        # 1) Save heuristic color-blobs image
        debug_heuristic = frame.astype(np.float32).copy()
        color_h = [(255,0,0), (0,255,0)]
        for i,bm in enumerate(color_blob_masks):
            col = color_h[i%len(color_h)]
            if bm is not None:
                debug_heuristic[bm,0] = col[0]
                debug_heuristic[bm,1] = col[1]
                debug_heuristic[bm,2] = col[2]
        heur_path = os.path.join(args.blobs_dir, f"heuristic_blobs_frame_{frame_idx:06d}.png")
        imageio.imwrite(heur_path, debug_heuristic.astype(np.uint8))

        # Build polygons
        center_x = W/2.0
        center_y = H/2.0
        poly_current = {}
        poly_memory = {}
        for i in range(len(blob_info_list)):
            c_m = assigned_in_global[i]
            if c_m is not None and c_m.sum()>0 and not blob_info_list[i]['disappeared']:
                cp = find_contour_polygon(c_m, center_x, center_y)
            else:
                cp = []
            poly_current[i] = cp

            am = blob_info_list[i]['avg_mask']
            if am is not None:
                am_bin = threshold_mask(am, 0.5)
                if am_bin is not None and am_bin.sum()>0:
                    mp = find_contour_polygon(am_bin, center_x, center_y)
                else:
                    mp = []
            else:
                mp = []
            poly_memory[i] = mp

        # 2) JSON for current masks
        cur_json_data = {
            f"segmentation_blob_{i}": poly_current[i]
            for i in range(len(blob_info_list))
        }
        json_current_path = os.path.join(args.json_dir, f"current_segm_mask_frame_{frame_idx:06d}.json")
        with open(json_current_path,'w') as jf:
            json.dump(cur_json_data, jf, indent=2)

        # 3) JSON for memory masks
        mem_json_data = {
            f"segmentation_blob_{i}": poly_memory[i]
            for i in range(len(blob_info_list))
        }
        json_memory_path = os.path.join(args.json_dir, f"memory_frame_mask_frame_{frame_idx:06d}.json")
        with open(json_memory_path,'w') as jf:
            json.dump(mem_json_data, jf, indent=2)

        # 4) Overlays: current, memory
        over_cur = overlay_polygons(
            frame, poly_current,
            {i:blob_info_list[i]['color'] for i in range(len(blob_info_list))},
            fill_alpha=0.5, center_x=center_x, center_y=center_y, label_suffix="(cur)"
        )
        out_cur_path = os.path.join(args.processed_dir, f"current_segm_mask_frame_{frame_idx:06d}.png")
        imageio.imwrite(out_cur_path, over_cur)

        over_mem = overlay_polygons(
            frame, poly_memory,
            {i:blob_info_list[i]['color'] for i in range(len(blob_info_list))},
            fill_alpha=0.4, center_x=center_x, center_y=center_y, label_suffix="(mem)"
        )
        out_mem_path = os.path.join(args.processed_dir, f"memory_segm_mask_frame_{frame_idx:06d}_mem.png")
        imageio.imwrite(out_mem_path, over_mem)

        # 5) Collage: side-by-side of "current" (left) vs "memory" (right)
        # We'll do a 1-row, 2-col figure, placing each overlay in a subplot.
        fig_dpi = 100
        # Both images have shape (H, W, 3). We'll combine them side-by-side
        # For a direct side-by-side, we can do np.concatenate along width dimension.
        # But let's do it with plt subplots for labeling clarity.
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(2*W/fig_dpi, H/fig_dpi), dpi=fig_dpi)
        axs[0].imshow(over_cur)
        axs[0].set_title("Current")
        axs[0].axis("off")
        axs[1].imshow(over_mem)
        axs[1].set_title("Memory")
        axs[1].axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        collage_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        collage_img = collage_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # Save collage
        collage_path = os.path.join(args.processed_dir, f"collage_frame_{frame_idx:06d}.png")
        imageio.imwrite(collage_path, collage_img)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}")

    reader.close()
    total_frames = frame_idx
    print(f"Total frames read: {total_frames}")

    # 6) Stitch the final collage images into a .mp4
    print(f"Writing final video => {args.output_video_path} with fps={fps}")
    writer = imageio.get_writer(args.output_video_path, fps=fps)
    for i in range(total_frames):
        path_ = os.path.join(args.processed_dir, f"collage_frame_{i:06d}.png")
        if not os.path.exists(path_):
            continue
        frame_ = imageio.imread(path_)
        writer.append_data(frame_)
    writer.close()

    print("Done! The final stitched video is at:", args.output_video_path)


if __name__ == "__main__":
    main()
