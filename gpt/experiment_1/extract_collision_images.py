#!/usr/bin/env python3

import os
import argparse
import imageio
import numpy as np
from PIL import Image

def label_connected(bin_image):
    """
    Simple 8-connected component labeling.
    Returns (labeled_image, num_labels), where labeled_image has int labels [0..num_labels].
    """
    h, w = bin_image.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    current_label = 0

    def neighbors(r, c):
        for nr in (r-1, r, r+1):
            for nc in (c-1, c, c+1):
                if 0 <= nr < h and 0 <= nc < w:
                    yield nr, nc

    for rr in range(h):
        for cc in range(w):
            if bin_image[rr, cc] and labeled[rr, cc] == 0:
                current_label += 1
                stack = [(rr, cc)]
                labeled[rr, cc] = current_label
                while stack:
                    r_, c_ = stack.pop()
                    for nr, nc in neighbors(r_, c_):
                        if bin_image[nr, nc] and labeled[nr, nc] == 0:
                            labeled[nr, nc] = current_label
                            stack.append((nr, nc))

    return labeled, current_label

def find_n_color_blobs(frame_np, n_blobs=2, black_thresh=30):
    """
    Segment up to n_blobs by thresholding near-black areas.
    Return a list of boolean masks (largest area first).
    """
    # sum across RGB => "gray"
    gray = frame_np.sum(axis=2)
    non_black = (gray > black_thresh)

    labeled, num_labels = label_connected(non_black)
    if num_labels < 1:
        return []

    regions = []
    for lbl_id in range(1, num_labels+1):
        mask_ = (labeled == lbl_id)
        area_ = mask_.sum()
        regions.append((area_, mask_))
    regions.sort(key=lambda x: x[0], reverse=True)

    # up to n_blobs largest
    top = [r[1] for r in regions[:n_blobs]]
    return top

def compute_centroid(bin_mask):
    """
    Returns (y, x) centroid of bin_mask, or (None, None) if empty.
    """
    if bin_mask is None or bin_mask.sum() == 0:
        return None, None
    coords = np.argwhere(bin_mask)
    y_ = coords[:,0].mean()
    x_ = coords[:,1].mean()
    return (y_, x_)

def clamp_crop_coords(cx, cy, crop_w, crop_h, img_w, img_h):
    """
    Compute a rectangular region of size (crop_w x crop_h)
    centered on (cx,cy), clamped to [0..img_w-1, 0..img_h-1].
    Returns (left, top, right, bottom) for PIL Image.crop().
    """
    half_w = crop_w // 2
    half_h = crop_h // 2

    left = int(cx - half_w)
    right = left + crop_w
    top = int(cy - half_h)
    bottom = top + crop_h

    # clamp horizontally
    if left < 0:
        right -= left
        left = 0
    if right > img_w:
        diff = right - img_w
        right = img_w
        left -= diff
        if left < 0:
            left = 0

    # clamp vertically
    if top < 0:
        bottom -= top
        top = 0
    if bottom > img_h:
        diff = bottom - img_h
        bottom = img_h
        top -= diff
        if top < 0:
            top = 0

    return (left, top, right, bottom)

def masks_are_different(maskA, maskB):
    """
    Returns True if maskA != maskB in any pixel.
    If either is None, and the other is not empty, that also means 'different'.
    """
    if maskA is None and maskB is None:
        return False  # both empty => no difference
    if maskA is None and maskB is not None:
        return maskB.sum() > 0
    if maskB is None and maskA is not None:
        return maskA.sum() > 0

    # XOR => True where they differ
    diff = np.logical_xor(maskA, maskB)
    return diff.any()

def main():
    parser = argparse.ArgumentParser(
        description="Detect which blob is moving from classical CV, track until it stops, produce 3 final crops."
    )
    parser.add_argument("--video_path",
        default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/gpt/experiment_1/Exp1/Stimuli/Exp1_videos/A_concave_4.mp4",
        help="Input video path."
    )
    parser.add_argument("--start_frame", type=int, default=200,
        help="Frame index to begin the segmentation & movement check."
    )
    parser.add_argument("--blobs_dir", default="blobs",
        help="Folder to store each frame's blob masks (two per frame)."
    )
    parser.add_argument("--output_dir", default="crops_output",
        help="Folder to store final 3 crops."
    )
    parser.add_argument("--crop_width", type=int, default=500, help="Width of final crops.")
    parser.add_argument("--crop_height", type=int, default=400, help="Height of final crops.")
    args = parser.parse_args()

    # 1) Read all frames
    print(f"Reading video: {args.video_path}")
    reader = imageio.get_reader(args.video_path, format='ffmpeg')
    frames = [frm for frm in reader]
    reader.close()
    total_frames = len(frames)
    print(f"Total frames read: {total_frames}\n")

    # Guard checks
    if args.start_frame < 0 or args.start_frame >= total_frames-1:
        print(f"ERROR: start_frame={args.start_frame} is out of range [0..{total_frames-2}]. Exiting.")
        return

    # 2) Prepare folder for storing blob masks
    os.makedirs(args.blobs_dir, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    blob_subfolder = os.path.join(args.blobs_dir, video_basename)
    os.makedirs(blob_subfolder, exist_ok=True)

    def detect_and_sort_two_blobs(frame_img):
        """
        Return (mask_left, mask_right) sorted by each blob's centroid.x
        If fewer than 2 found, pad with None.
        """
        found = find_n_color_blobs(frame_img, n_blobs=2, black_thresh=30)
        while len(found) < 2:
            found.append(None)

        # We'll have up to 2 masks in 'found', each is a boolean array or None
        # Let's compute their centroids
        results = []
        for m in found:
            cy, cx = compute_centroid(m)
            results.append((m, (cy, cx)))
        # Sort by x of centroid
        results.sort(key=lambda x: (x[1][1] if x[1][1] is not None else 999999))

        # Return just the masks in left->right order
        mask_left = results[0][0]
        mask_right = results[1][0]
        return mask_left, mask_right

    def save_mask_image(bin_mask, out_path, shape_for_empty):
        """
        Save bin_mask as single-channel (0 or 255) PNG.
        If None, or empty, produce a black image of shape_for_empty (H,W).
        """
        if bin_mask is None or bin_mask.sum() == 0:
            h_, w_ = shape_for_empty
            empty_im = Image.new("L", (w_, h_), color=0)
            empty_im.save(out_path)
            return
        mask_255 = (bin_mask.astype(np.uint8))*255
        Image.fromarray(mask_255, "L").save(out_path)

    # 3) We'll keep track of:
    #    - prev_left_mask, prev_right_mask from the previous frame
    #    - which blob is "moving_blob"? ("left" or "right") or None if not determined yet
    #    - stop_frame = None => frame at which we see the mover has not moved
    #
    # We'll process from [start_frame .. total_frames-1], or until we break

    prev_left_mask = None
    prev_right_mask = None
    moving_blob = None  # "left" or "right"
    stop_frame = None

    # 4) Loop over frames from start_frame..(end)
    # We do the segmentation, store each mask, then compare with previous to see if there's movement
    for fidx in range(args.start_frame, total_frames):
        frame_img = frames[fidx]
        h, w, _ = frame_img.shape

        # detect 2 masks, sorted left->right
        left_mask, right_mask = detect_and_sort_two_blobs(frame_img)

        # store them
        out_left = os.path.join(blob_subfolder, f"blob_left_{fidx:05d}.png")
        out_right = os.path.join(blob_subfolder, f"blob_right_{fidx:05d}.png")
        save_mask_image(left_mask, out_left, (h,w))
        save_mask_image(right_mask, out_right, (h,w))

        if fidx == args.start_frame:
            # no previous frame to compare => skip
            prev_left_mask = left_mask
            prev_right_mask = right_mask
            continue

        # compare with previous to see which masks changed
        left_changed = masks_are_different(prev_left_mask, left_mask)
        right_changed = masks_are_different(prev_right_mask, right_mask)

        if moving_blob is None:
            # we haven't flagged a mover yet
            # if exactly one changed, that is the mover
            # if both changed, pick the one that changed more, or just pick left arbitrarily?
            # For simplicity: if both changed, pick whichever has bigger XOR count
            if left_changed or right_changed:
                if left_changed and right_changed:
                    # measure actual pixel difference to see which is bigger
                    diff_left = 0 if (left_mask is None or prev_left_mask is None) else np.sum(np.logical_xor(left_mask, prev_left_mask))
                    diff_right = 0 if (right_mask is None or prev_right_mask is None) else np.sum(np.logical_xor(right_mask, prev_right_mask))
                    if diff_left >= diff_right:
                        moving_blob = "left"
                    else:
                        moving_blob = "right"
                elif left_changed:
                    moving_blob = "left"
                elif right_changed:
                    moving_blob = "right"
                print(f"[Frame={fidx}] Identified moving_blob='{moving_blob}'.")
        else:
            # we do have a moving_blob => check if it still changed
            if moving_blob == "left":
                if not left_changed:
                    # that means it has stopped
                    stop_frame = fidx
                    print(f"[Frame={fidx}] Blob_0 (left) has STOPPED moving (no difference).")
                    break
            else:  # moving_blob=='right'
                if not right_changed:
                    stop_frame = fidx
                    print(f"[Frame={fidx}] Blob_0 (right) has STOPPED moving (no difference).")
                    break

        # update previous
        prev_left_mask = left_mask
        prev_right_mask = right_mask

    # end for

    if moving_blob is None:
        print("\n[WARNING] We never identified which blob was moving. No stops found => no crops.")
        return

    if stop_frame is None:
        print("\nWe reached the end of the video without seeing the mover blob STOP. No crops will be produced.")
        return

    M = stop_frame
    print(f"\nFinal STOP frame is M={M}. We'll produce 3 crops: (M-10), M, (M+10).")

    # 5) Produce the 3 crops from frames (M-10), M, (M+10)
    # center them on the UNION of the two blob masks at frame M
    frames_of_interest = [M-10, M, M+10]
    valid_frames = [f for f in frames_of_interest if 0 <= f < total_frames]

    # Re-segment frame M to get union
    frameM_img = frames[M]
    hM, wM, _ = frameM_img.shape
    # get left->right again
    leftM_mask, rightM_mask = detect_and_sort_two_blobs(frameM_img)
    union_mask = np.zeros((hM, wM), dtype=bool)
    if leftM_mask is not None:
        union_mask |= leftM_mask
    if rightM_mask is not None:
        union_mask |= rightM_mask

    if union_mask.sum() == 0:
        print(f"[ERROR] Union mask at frame M={M} is empty => no centroid => no crops.")
        return

    # centroid
    coords = np.argwhere(union_mask)
    yU = coords[:,0].mean()
    xU = coords[:,1].mean()
    print(f"Union centroid at M={M} => (y={yU:.1f}, x={xU:.1f}).")

    # produce crops
    os.makedirs(args.output_dir, exist_ok=True)
    for fidx in valid_frames:
        crop_img = frames[fidx]
        pil_img = Image.fromarray(crop_img)
        left, top, right, bottom = clamp_crop_coords(
            cx=xU, cy=yU,
            crop_w=args.crop_width, crop_h=args.crop_height,
            img_w=wM, img_h=hM
        )
        cropped = pil_img.crop((left, top, right, bottom))

        out_name = f"crop_frame_{fidx:05d}.png"
        out_path = os.path.join(args.output_dir, out_name)
        cropped.save(out_path)
        print(f"  Saved crop from frame={fidx}: {out_path}")

    print("\n=== DONE! ===")
    print(f"   - Moving blob was '{moving_blob}'.")
    print(f"   - It stopped at frame M={M}.")
    print("   - Blob masks are in:", blob_subfolder)
    print("   - Final 3 crops are in:", args.output_dir)


if __name__ == "__main__":
    main()
