#!/usr/bin/env python3
"""
final_gen_video.py

1) Reads an input .mp4 (with "imageio[ffmpeg]").
2) For each frame, we find N color blobs (heuristic: ignoring black).
3) We run DETR for 100 predictions. We upsample masks to full size.
4) We do a bipartite assignment so each blob is matched to exactly one DETR mask,
   maximizing IOU. Then we forcibly remove overlapping areas among assigned masks
   to ensure disjoint final masks.
5) We find polygon boundaries (with center-based coordinates) for each assigned mask.
6) We output:
   - frames_blobs/frame_XXXXXX.png => color-coded heuristic blobs
   - frames_processed/frame_XXXXXX.png => final overlay using the polygons
   - frames_json/frame_XXXXXX.json => containing N polygons (possibly empty for some)
7) Finally, we stitch frames_processed/* into the output .mp4 at the same resolution & FPS.

Set N with --n_blobs. The script uses scikit-image, scipy, matplotlib, imageio, PyTorch, etc.
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
    Return a list of boolean masks in descending order of area.
    """
    # sum of channels
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
    import math
    from scipy.optimize import linear_sum_assignment

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
    For example, in order: for i in range(len(masks)):
       for j in range(i+1, len(masks)):
         masks[j] = masks[j] & ~masks[i]
    This ensures final masks do not overlap. Masks are likely assigned to different blobs.
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
    to a polygon in center-based coords:  x = col - center_x, y = row - center_y
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
    Approximate centroid of a polygon in top-left coords. If empty, return (None,None).
    poly_pts is a list of (x_tl, y_tl).
    We'll do a simple arithmetic mean. If self-intersecting, area-based approach might be needed,
    but let's keep it simple.
    """
    if len(poly_pts) == 0:
        return None, None
    xs = [p[0] for p in poly_pts]
    ys = [p[1] for p in poly_pts]
    return (sum(xs)/len(xs), sum(ys)/len(ys))


def main():
    parser = argparse.ArgumentParser("Final script with heuristics, disjoint masks, polygons, etc.")
    parser.add_argument("--model_path", required=True, help="Path to custom DETR checkpoint (.pth)")
    parser.add_argument("--video_path", required=True, help="Path to input .mp4")
    parser.add_argument("--output_video_path", required=True, help="Path to output .mp4 (same resolution)")
    parser.add_argument("--n_blobs", type=int, default=2, help="Number of color blobs to track (default=2)")
    parser.add_argument("--blobs_dir", default="frames_blobs", help="Dir to save color-blob debug images.")
    parser.add_argument("--processed_dir", default="frames_processed", help="Dir to save final overlay frames.")
    parser.add_argument("--json_dir", default="frames_json", help="Dir to save JSON for each frame.")
    args = parser.parse_args()

    # 1) Load the model
    model = load_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2) Prepare local folders
    for d in [args.blobs_dir, args.processed_dir, args.json_dir]:
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
    # We'll store frames to processed_dir, then stitch them after
    frames_processed_paths = []

    for frame in reader:
        # Ensure shape matches W,H
        if frame.shape[0] != H or frame.shape[1] != W:
            # if something is off, let's fix it with a quick pad or resize
            # but typically this does not happen
            corrected = np.zeros((H,W,3), dtype=frame.dtype)
            h_ = min(H, frame.shape[0])
            w_ = min(W, frame.shape[1])
            corrected[0:h_, 0:w_, :] = frame[0:h_, 0:w_, :]
            frame = corrected

        ### 3a) Show color-blobs. We'll store in frames_blobs
        blob_masks = find_n_color_blobs(frame, n_blobs=args.n_blobs)
        # Let's color them for debug
        debug_blob = frame.astype(np.float32).copy()
        color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
        for i,bm in enumerate(blob_masks):
            c = color_list[i%len(color_list)]
            debug_blob[bm,0] = c[0]
            debug_blob[bm,1] = c[1]
            debug_blob[bm,2] = c[2]
        debug_blob_path = os.path.join(args.blobs_dir, f"frame_{frame_idx:06d}_blobs.png")
        Image.fromarray(debug_blob.astype(np.uint8)).save(debug_blob_path)

        ### 3b) DETR predicted masks
        pil_img = Image.fromarray(frame, mode="RGB")
        # transform short side=800
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
        # ignoring "no object" => best confidence
        # but we want all
        n_queries = pred_logits.shape[1]
        up_masks = []
        for i in range(n_queries):
            ml = pred_masks[0,i].unsqueeze(0).unsqueeze(0) # (1,1,h',w')
            up = F.interpolate(ml, size=(H,W), mode="bilinear", align_corners=False)
            pm = torch.sigmoid(up).squeeze(0).squeeze(0).cpu().numpy()
            bin_m = (pm > 0.5)
            up_masks.append(bin_m)

        ### 3c) bipartite assignment
        assign = bipartite_assign_blobs_to_masks(blob_masks, up_masks)
        # [blob_idx -> pred_idx or None]

        # build final assigned mask list in the same order as blob_masks
        assigned = []
        for b_idx in range(len(blob_masks)):
            p_idx = assign[b_idx]
            if p_idx is not None:
                assigned.append(up_masks[p_idx])
            else:
                assigned.append(None)

        # 3d) ensure disjoint => assigned = make_masks_disjoint(assigned)
        assigned = make_masks_disjoint(assigned)

        # 3e) from each assigned mask, find polygon => store JSON
        center_x = W/2.0
        center_y = H/2.0

        # We'll keep them in the same left->right order as the blob heuristics
        # or do we re-sort them? The user said "left-most is segmentation_blob_0, etc."
        # We can do that by finding each assigned mask's centroid in top-left coords
        # and sorting by that. Let's do that:
        blob_info = []
        for b_idx, mask_b in enumerate(assigned):
            if mask_b is None or mask_b.sum()==0:
                # empty
                blob_info.append((b_idx, None, 0))
            else:
                # compute centroid col
                coords = np.argwhere(mask_b)
                mean_col = coords[:,1].mean()
                blob_info.append((b_idx, mask_b, mean_col))

        # sort by mean_col
        blob_info.sort(key=lambda x: x[2])
        # now build polygons in that order => segmentation_blob_0,1,...
        polygons_json = {}
        # we also want them in a *strict* 0..(n_blobs-1) range
        for order_idx, (b_idx, m, mean_c) in enumerate(blob_info):
            poly = find_contour_polygon(m, center_x, center_y) if m is not None else []
            polygons_json[f"segmentation_blob_{order_idx}"] = poly

        # store the JSON
        json_path = os.path.join(args.json_dir, f"frame_{frame_idx:06d}.json")
        with open(json_path,'w') as f:
            json.dump(polygons_json, f, indent=2)

        ### 3f) overlay these polygons on the frame => store in frames_processed
        # We'll do a matplotlib figure sized to W,H, then fill polygons with partial alpha.
        fig_dpi=100
        fig_w = W/fig_dpi
        fig_h = H/fig_dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
        ax.imshow(frame)  # shape is top-left origin
        ax.set_axis_off()

        # color maps for each polygon
        color_maps = ['Reds','Blues','Greens','Oranges','Purples','pink','YlOrBr']

        for i, (b_idx, mask_b, mean_c) in enumerate(blob_info):
            poly_pts = polygons_json[f"segmentation_blob_{i}"]
            cmap = color_maps[i % len(color_maps)]
            color = cm.get_cmap(cmap)(0.6)  # RGBA

            if len(poly_pts)==0:
                # no polygon => skip
                continue

            # Convert center-based coords -> top-left. x_tl= x + center_x, y_tl=y + center_y
            # polygon is in order (x,y). We need separate lists for fill
            xs_tl = []
            ys_tl = []
            for (xC, yC) in poly_pts:
                xs_tl.append(xC + center_x)
                ys_tl.append(yC + center_y)

            # fill
            ax.fill(xs_tl, ys_tl, alpha=0.4, color=color, linewidth=0)

            # label near centroid
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
        out_path = os.path.join(args.processed_dir, f"frame_{frame_idx:06d}.png")
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        frames_processed_paths.append(out_path)

        if (frame_idx+1)%10==0:
            print(f"Processed frame {frame_idx+1}")
        frame_idx+=1

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
