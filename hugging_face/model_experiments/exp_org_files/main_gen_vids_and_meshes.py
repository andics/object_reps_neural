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

from PIL import Image, ImageDraw, ImageFont
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops, find_contours
from scipy.optimize import linear_sum_assignment

torch.set_grad_enabled(False)

# DETR's 91 COCO Classes (for reference)
COCO_CLASSES = [
    'N/A','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','N/A','stop sign','parking meter','bench','bird','cat','dog','horse',
    'sheep','cow','elephant','bear','zebra','giraffe','N/A','backpack','umbrella','handbag','tie',
    'suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
    'skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon',
    'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',
    'chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
    'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','N/A','book','clock',
    'vase','scissors','teddy bear','hair drier','toothbrush'
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
    parts = model_path.split("trained_models/")
    if len(parts) < 2:
        return "unknownModel"
    after = parts[1]
    subparts = after.split("/")
    prefix = subparts[0]
    if not prefix:
        return "unknownModel"
    return prefix

def parse_video_prefix(video_path):
    base = os.path.basename(video_path)
    root, _ = os.path.splitext(base)
    return root.replace(" ", "+")

def video_is_flipped(video_path):
    """
    Returns True if the video filename (without extension)
    contains the substring 'flipped'. Otherwise returns False.
    """
    base = os.path.basename(video_path)
    root, _ = os.path.splitext(base)
    return "flipped" in root

def find_n_color_blobs(frame_np, n_blobs=2, black_thresh=30, flip_blobs=False):
    """
    Finds up to n_blobs colored regions in frame_np.
    Normally sorted from left → right (by mean column).
    If flip_blobs=True, sort from right → left instead.
    """
    gray = frame_np.sum(axis=2)
    non_black = (gray > black_thresh)
    labeled = label(non_black, connectivity=2)
    regs = regionprops(labeled)
    regs_sorted = sorted(regs, key=lambda r: r.area, reverse=True)
    top = regs_sorted[:n_blobs]

    # Prepare (region, mean_col) for sorting
    reg_info = []
    for r in top:
        coords = r.coords  # shape (N,2)
        mean_col = coords[:, 1].mean()
        reg_info.append((r, mean_col))

    # If not flipped, we sort ascending -> left->right
    # If flipped, we sort descending -> right->left
    reg_info.sort(key=lambda x: x[1], reverse=flip_blobs)

    # Extract final masks in sorted order
    masks = []
    for (r, _) in reg_info:
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
    nb = len(blob_masks)
    np_ = len(pred_masks)
    if np_ == 0:
        return [None]*nb, None
    cost = np.zeros((nb, np_), dtype=np.float32)
    for b in range(nb):
        for p in range(np_):
            cost[b, p] = -iou(blob_masks[b], pred_masks[p])
    row_idx, col_idx = linear_sum_assignment(cost)
    assign = [None]*nb
    for i in range(len(row_idx)):
        b = row_idx[i]
        p = col_idx[i]
        assign[b] = p
    return assign, cost

def make_masks_disjoint(masks):
    for i in range(len(masks)):
        if masks[i] is None:
            continue
        for j in range(i+1, len(masks)):
            if masks[j] is None:
                continue
            masks[j] = masks[j] & ~masks[i]
    return masks

def find_contour_polygon(bin_mask, center_x, center_y):
    if bin_mask is None or bin_mask.sum() == 0:
        return []
    cts = find_contours(bin_mask.astype(np.uint8), 0.5)
    if len(cts) == 0:
        return []
    big_ct = max(cts, key=lambda c: c.shape[0])
    poly = []
    for point in big_ct:
        r = point[0]
        c = point[1]
        x = c - center_x
        y = r - center_y
        poly.append((x, y))
    return poly

def polygon_centroid(poly_pts):
    """
    Simple centroid calculation for a list of (x, y) points.
    """
    if len(poly_pts) == 0:
        return None, None
    xs = [p[0] for p in poly_pts]
    ys = [p[1] for p in poly_pts]
    return (sum(xs)/len(xs), sum(ys)/len(ys))

def main():
    parser = argparse.ArgumentParser(
        "Process video frames with DETR, skip detection for initial frames, save binary masks, "
        "and ensure final PNG frames match video size."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--current_working_directory", required=False, default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/detr/VIDEO_PROCESSING_TOOLS/generate_detection_videos_and_meshes/exp_1_videos_processed")
    parser.add_argument("--n_blobs", type=int, default=2)
    parser.add_argument("--initial_skip_frames", type=int, default=13)
    parser.add_argument("--alpha", type=float, default=0.7)
    args = parser.parse_args()

    model_prefix = parse_model_prefix(args.model_path)
    video_prefix = parse_video_prefix(args.video_path)
    flip_blobs = video_is_flipped(args.video_path)  # True if filename ends with "_flipped"

    # Create a single root folder = "model_prefix-video_prefix"
    root_folder = os.path.join(args.current_working_directory, f"{model_prefix.split('_')[0]}_videos_processed",
                               f"{model_prefix}-{video_prefix}")

    # Now place subfolders inside that root folder
    folder_root_blobs   = os.path.join(root_folder, "frames_blobs")
    folder_root_masks   = os.path.join(root_folder, "frames_masks")
    folder_root_masks_nonmem = os.path.join(root_folder, "frames_masks_nonmem")
    folder_root_memjson = os.path.join(root_folder, "frames_json_memory_processed")
    folder_root_collage = os.path.join(root_folder, "frames_collage")
    folder_root_memcollage = os.path.join(root_folder, "frames_memorycollage")
    folder_root_proc    = os.path.join(root_folder, "frames_processed")
    folder_root_videos  = os.path.join(root_folder, "videos_processed")

    final_video_path = os.path.join(folder_root_videos, f"{video_prefix}.mp4")

    for d in [
        folder_root_blobs, folder_root_masks, folder_root_masks_nonmem,
        folder_root_memjson, folder_root_collage, folder_root_memcollage,
        folder_root_proc, folder_root_videos
    ]:
        os.makedirs(d, exist_ok=True)

    # 1) Load DETR
    model = load_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2) Read video => shape
    print(f"Reading input video {args.video_path}")
    reader = imageio.get_reader(args.video_path, format='ffmpeg')
    meta_in = reader.get_meta_data()
    fps = float(meta_in.get('fps', 30))

    try:
        first_fr = reader.get_data(0)
        H, W, _ = first_fr.shape
        print(f"Video shape: W={W}, H={H}, fps={fps}")
    except:
        print("Could not read first frame.")
        reader.close()
        return

    frame_idx = 0
    # 3) memory for rolling masks
    mem_floats = [np.zeros((H, W), dtype=np.float32) for _ in range(args.n_blobs)]

    while True:
        try:
            frame = reader.get_data(frame_idx)
        except IndexError:
            break  # no more frames

        # Ensure shape exactly (H,W,3)
        if frame.shape[0] != H or frame.shape[1] != W:
            corrected = np.zeros((H, W, 3), dtype=frame.dtype)
            hh_ = min(H, frame.shape[0])
            ww_ = min(W, frame.shape[1])
            corrected[:hh_, :ww_, :] = frame[:hh_, :ww_, :]
            frame = corrected

        # =====================
        # Skip detection for first 'initial_skip_frames' frames
        # =====================
        if frame_idx < args.initial_skip_frames:
            # Just output a copy of the frame
            outp = os.path.join(folder_root_proc, f"frame_{frame_idx:06d}.png")
            Image.fromarray(frame).save(outp)

            # Also create empty memory JSON if desired
            empty_data = {}
            mem_json_path = os.path.join(folder_root_memjson, f"frame_{frame_idx:06d}.json")
            with open(mem_json_path, 'w') as f:
                json.dump(empty_data, f, indent=2)

            if frame_idx % 10 == 0:
                print(f"Writing skipped frame {frame_idx} (no detection).")
            frame_idx += 1
            continue

        # ========== 1) color blobs
        blob_masks = find_n_color_blobs(frame, n_blobs=args.n_blobs, flip_blobs=flip_blobs)
        nb = len(blob_masks)

        dbg = frame.astype(np.float32).copy()
        c_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for i, bm in enumerate(blob_masks):
            c_ = c_list[i % len(c_list)]
            dbg[bm, 0] = c_[0]
            dbg[bm, 1] = c_[1]
            dbg[bm, 2] = c_[2]
        dbg_path = os.path.join(folder_root_blobs, f"frame_{frame_idx:06d}_blobs.png")
        Image.fromarray(dbg.astype(np.uint8)).save(dbg_path)

        # ========== 2) DETR => submasks
        pil_img = Image.fromarray(frame, 'RGB')
        transform_resize = T.Resize(800)
        rimg = transform_resize(pil_img)
        transform_norm = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        model_in = transform_norm(rimg).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs_ = model(model_in)
        pmasks = outputs_["pred_masks"]  # (1,100,h',w')
        n_queries = pmasks.shape[1]
        up_masks = []
        for i2 in range(n_queries):
            mm1 = pmasks[0, i2].unsqueeze(0).unsqueeze(0)
            up_ = F.interpolate(mm1, size=(H, W), mode='bilinear', align_corners=False)
            pm_ = torch.sigmoid(up_).squeeze().cpu().numpy()
            up_masks.append(pm_ > 0.5)

        # Split connected components in each query
        split_pred_masks = []
        for q_i, bn in enumerate(up_masks):
            labeled_ = label(bn, connectivity=2)
            max_cc = labeled_.max()
            for cc_label in range(1, max_cc+1):
                sub_ = (labeled_ == cc_label)
                split_pred_masks.append(sub_)

        # (3) if nb==0 => no color blobs => skip
        if nb == 0:
            outp = os.path.join(folder_root_proc, f"frame_{frame_idx:06d}.png")
            Image.fromarray(frame).save(outp)
            mem_json_path = os.path.join(folder_root_memjson, f"frame_{frame_idx:06d}.json")
            with open(mem_json_path, 'w') as f:
                json.dump({}, f, indent=2)

            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Processed frame {frame_idx} (no color blobs).")
            continue

        # (4) bipartite
        assign, cost = bipartite_assign_blobs_to_masks(blob_masks, split_pred_masks)

        # -- top10 collage (for debugging)
        if cost is not None and frame_idx >= 30 and nb > 0:
            fig, axes = plt.subplots(nb, 10, figsize=(25, 5*nb), dpi=100)
            # If nb==1, axes can be 1D, so reshape if needed
            if nb == 1 and len(axes.shape) == 1:
                axes = axes[np.newaxis, :]
            for b_idx in range(nb):
                row_cost = cost[b_idx, :]
                idx_sorted = np.argsort(row_cost)
                best10 = idx_sorted[:10]
                for rank_i, sm_idx in enumerate(best10):
                    ax = axes[b_idx, rank_i]
                    overlay = frame.copy()
                    # green for the blob
                    overlay[blob_masks[b_idx], 0] = 0
                    overlay[blob_masks[b_idx], 1] = 255
                    overlay[blob_masks[b_idx], 2] = 0
                    # red for the submask
                    overlay[split_pred_masks[sm_idx], 0] = 255
                    overlay[split_pred_masks[sm_idx], 1] = 0
                    overlay[split_pred_masks[sm_idx], 2] = 0
                    cost_val = row_cost[sm_idx]
                    ax.imshow(overlay)
                    ax.set_title(f"Blob {b_idx}, sub={sm_idx}\nCost={cost_val:.3f}", fontsize=8)
                    ax.set_axis_off()
            fig.suptitle(f"Frame {frame_idx} - top10 collage", fontsize=14)
            coll_path = os.path.join(folder_root_collage, f"frame_{frame_idx:06d}_collage.png")
            fig.savefig(coll_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        # assigned_submasks => immediate submasks
        assigned_submasks = []
        for b_idx in range(nb):
            sm_idx = assign[b_idx]
            if sm_idx is not None:
                assigned_submasks.append(split_pred_masks[sm_idx])
            else:
                assigned_submasks.append(None)

        # ========== Save each assigned_submask (NON-memory) as PNG ==========
        for b_i in range(nb):
            submask = assigned_submasks[b_i]
            if submask is not None and submask.sum() > 0:
                mask_255 = (submask.astype(np.uint8)) * 255
                mask_path = os.path.join(
                    folder_root_masks_nonmem,
                    f"mask_blob_{b_i}_frame_{frame_idx:06d}.png"
                )
                Image.fromarray(mask_255).save(mask_path)

        # ========== 5) update memory
        for b_i in range(args.n_blobs):
            if b_i < nb and assigned_submasks[b_i] is not None:
                sub_f = assigned_submasks[b_i].astype(np.float32)
                mem_floats[b_i] = args.alpha*mem_floats[b_i] + (1-args.alpha)*sub_f

        # ========== 6) memory => bool
        memory_masks = []
        for b_i in range(args.n_blobs):
            memory_masks.append(mem_floats[b_i] > 0.5)

        # ========== Save each memory mask ==========
        for b_i in range(args.n_blobs):
            mem_mask = memory_masks[b_i]
            if mem_mask.sum() > 0:
                mask_255 = (mem_mask.astype(np.uint8)) * 255
                mask_path = os.path.join(
                    folder_root_masks,
                    f"mask_memory_blob_{b_i}_frame_{frame_idx:06d}.png"
                )
                Image.fromarray(mask_255).save(mask_path)

        # ========== 7) memory collage => shape (nb,2) [debug only]
        fig, axes = plt.subplots(nb, 2, figsize=(10, 5*nb), dpi=100)
        if nb == 1 and len(axes.shape) == 1:
            axes = axes[np.newaxis, :]
        for b_i in range(nb):
            axL = axes[b_i, 0]
            axR = axes[b_i, 1]
            overlay_cur = frame.copy()
            if assigned_submasks[b_i] is not None:
                overlay_cur[assigned_submasks[b_i], 0] = 255
                overlay_cur[assigned_submasks[b_i], 1] = 0
                overlay_cur[assigned_submasks[b_i], 2] = 0
            axL.imshow(overlay_cur)
            axL.set_title(f"Blob {b_i} - Current", fontsize=8)
            axL.set_axis_off()

            overlay_mem = frame.copy()
            overlay_mem[memory_masks[b_i], 0] = 0
            overlay_mem[memory_masks[b_i], 1] = 255
            overlay_mem[memory_masks[b_i], 2] = 0
            axR.imshow(overlay_mem)
            axR.set_title(f"Blob {b_i} - Memory", fontsize=8)
            axR.set_axis_off()

        fig.suptitle(f"Frame {frame_idx} - memory collage", fontsize=14)
        memcoll_path = os.path.join(folder_root_memcollage, f"frame_{frame_idx:06d}_memcollage.png")
        fig.savefig(memcoll_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # ========== 8) polygons => assigned (for overlay)
        assigned_dis = make_masks_disjoint(assigned_submasks.copy())
        assigned_info = []
        for b_i in range(nb):
            mm = assigned_dis[b_i]
            if mm is None or mm.sum() == 0:
                assigned_info.append((b_i, None, 999999))
            else:
                coords = np.argwhere(mm)
                c_ = coords[:, 1].mean()
                assigned_info.append((b_i, mm, c_))
        # Sort assigned blobs left→right, or right→left if flipped
        assigned_info.sort(key=lambda x: x[2], reverse=flip_blobs)
        assigned_polys = {}
        for order_i,(b_i,msk_, col_) in enumerate(assigned_info):
            poly_ = find_contour_polygon(msk_, W/2.0, H/2.0) if msk_ is not None else []
            assigned_polys[f"segmentation_blob_{order_i}"] = poly_

        # ========== 9) polygons => memory
        mem_dis = make_masks_disjoint(memory_masks.copy())
        mem_info=[]
        for b_i in range(args.n_blobs):
            mm_ = mem_dis[b_i]
            if mm_ is None or mm_.sum()==0:
                mem_info.append((b_i,None,999999))
            else:
                coords = np.argwhere(mm_)
                c_ = coords[:,1].mean()
                mem_info.append((b_i,mm_,c_))
        # Sort memory blobs left→right, or right→left if flipped
        mem_info.sort(key=lambda x: x[2], reverse=flip_blobs)
        memory_polys={}
        for order_i,(b_i,msk_,c_) in enumerate(mem_info):
            poly_ = find_contour_polygon(msk_, W/2.0, H/2.0) if msk_ is not None else []
            memory_polys[f"segmentation_blob_{order_i}"] = poly_

        # ========== 10) final overlay (memory polygons)
        overlay_img = Image.fromarray(frame)  # 'RGB' by default
        draw = ImageDraw.Draw(overlay_img, "RGBA")

        color_list = [
            (255, 0, 0, 100),
            (0, 255, 0, 100),
            (0, 0, 255, 100),
            (255, 255, 0, 100),
            (255, 0, 255, 100),
        ]
        text_fill = (255, 255, 255, 255)  # White text

        cx_ = W / 2.0
        cy_ = H / 2.0

        # Draw memory polygons
        for i, key in enumerate(memory_polys.keys()):
            pts_ = memory_polys[key]
            if not pts_:
                continue
            # Shift from center-based coords to image coords
            shifted_pts = [(x+cx_, y+cy_) for (x, y) in pts_]
            # Draw the polygon with a semi-transparent fill
            draw.polygon(shifted_pts, fill=color_list[i % len(color_list)])
            # Optionally draw a label at centroid
            cxx, cyy = polygon_centroid(shifted_pts)
            if cxx is not None and cyy is not None:
                draw.text(
                    (cxx, cyy),
                    f"Blob {i}",
                    fill=text_fill
                )

        # Save the final overlaid image
        out_path = os.path.join(folder_root_proc, f"frame_{frame_idx:06d}.png")
        overlay_img.save(out_path)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}")

    reader.close()
    total_frames = frame_idx
    print(f"Total frames read: {total_frames}")

    # ========== 11) final video
    print(f"Writing final video to {final_video_path} with fps={fps}")
    writer = imageio.get_writer(final_video_path, fps=fps, macro_block_size=1)
    for i in range(total_frames):
        fname = os.path.join(folder_root_proc, f"frame_{i:06d}.png")
        if os.path.isfile(fname):
            im_ = imageio.v2.imread(fname)
            writer.append_data(im_)
    writer.close()
    print("Done!")

if __name__ == "__main__":
    main()
