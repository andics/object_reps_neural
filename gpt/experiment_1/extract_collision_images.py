#!/usr/bin/env python3

import os
import argparse
import imageio
import numpy as np
from PIL import Image

def label_connected(bin_image):
    """
    Simple connected-component labeling using 8-connectivity.
    Returns (labeled_image, num_labels).
    labeled_image has int labels in [0..num_labels].

    For heavier usage, you could replace with scikit-image's 'label()':
        from skimage.measure import label
        labeled = label(bin_image, connectivity=2)
        num_labels = labeled.max()
        return labeled, num_labels
    """
    h, w = bin_image.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    current_label = 0

    def neighbors(r, c):
        for nr in (r - 1, r, r + 1):
            for nc in (c - 1, c, r + 1):
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
    Finds up to n_blobs "colored" regions in frame_np by simple summation of RGB,
    thresholding out near-black. Sorts by area descending, returns boolean masks.
    """
    gray = frame_np.sum(axis=2)  # sum across RGB
    non_black = (gray > black_thresh)

    labeled, num_labels = label_connected(non_black)
    if num_labels < 1:
        return []

    # gather (area, mask) for each label
    regions = []
    for lbl_id in range(1, num_labels + 1):
        mask_ = (labeled == lbl_id)
        area_ = mask_.sum()
        regions.append((area_, mask_))
    regions.sort(key=lambda x: x[0], reverse=True)

    # return up to n_blobs largest
    top = [r[1] for r in regions[:n_blobs]]
    return top


def compute_centroid(bin_mask):
    """
    Returns (y, x) centroid of bin_mask, or (None, None) if empty.
    """
    if bin_mask is None or bin_mask.sum() == 0:
        return None, None
    coords = np.argwhere(bin_mask)
    y_ = coords[:, 0].mean()
    x_ = coords[:, 1].mean()
    return y_, x_


def distance(c0, c1):
    """
    Euclidean distance between two centroids (y0,x0) and (y1,x1),
    or 0 if either is None.
    """
    (y0, x0), (y1, x1) = c0, c1
    if (y0 is None) or (x0 is None) or (y1 is None) or (x1 is None):
        return 0.0
    return np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)


def clamp_crop_coords(cx, cy, crop_w, crop_h, img_w, img_h):
    """
    Ensures we can produce a crop_w x crop_h region within [0..img_w-1, 0..img_h-1].
    Returns (left, top, right, bottom) for PIL's Image.crop().
    - cx, cy: center in (x, y) coords
    - crop_w: desired width
    - crop_h: desired height
    - img_w: full image width
    - img_h: full image height
    """
    half_w = crop_w // 2
    half_h = crop_h // 2

    left = int(cx - half_w)
    right = left + crop_w
    top = int(cy - half_h)
    bottom = top + crop_h

    # clamp horizontally
    if left < 0:
        right -= left  # shift
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


def main():
    parser = argparse.ArgumentParser(
        "Track two blobs from frame 240 onward, find the last moving frame of blob_0, and produce crops."
        "Stop processing once we are 10 frames past the stopping moment."
    )
    parser.add_argument("--video_path",
                        default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/gpt/experiment_1/Exp1/Stimuli/Exp1_videos/A_concave_4.mp4",
                        required=False,
                        help="Input video path.")
    parser.add_argument("--output_dir", default="crops_output", help="Folder to place final 3 crops.")
    parser.add_argument("--blobs_dir", default="blobs", help="Folder to store each frame's blob masks.")
    parser.add_argument("--move_thresh", type=float, default=2.0, help="Velocity threshold for 'moving'.")
    parser.add_argument("--crop_width", type=int, default=500, help="Width of the final crops.")
    parser.add_argument("--crop_height", type=int, default=400, help="Height of the final crops.")
    args = parser.parse_args()

    print("==== START ====")
    print(f"Video path: {args.video_path}")
    print(f"Output directory for crops: {args.output_dir}")
    print(f"Blobs directory: {args.blobs_dir}")
    print(f"move_thresh={args.move_thresh}, crop_width={args.crop_width}, crop_height={args.crop_height}")

    # 1) Ensure output dirs
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare subfolder for the "blobs" masks
    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    blob_subfolder = os.path.join(args.blobs_dir, video_basename)
    os.makedirs(blob_subfolder, exist_ok=True)

    # 2) Read ALL frames from the video into memory
    reader = imageio.get_reader(args.video_path, format='ffmpeg')
    frames = []
    for frame_idx, frame in enumerate(reader):
        frames.append(frame)
    reader.close()

    total_frames = len(frames)
    print(f"Read total {total_frames} frames from {args.video_path}.")

    # We assume frames >= 241 exist, so we can compare frames 239 <-> 240, etc.
    if total_frames < 241:
        print("[WARNING] This script expects >= 241 frames so we can start at frame 239/240. Exiting.")
        return

    # 3) Identify which blob is "blob_0" (the first mover) by comparing frames 239->240
    def detect_and_sort_blobs(img):
        """
        Detect up to 2 blobs, sort them left->right by x-centroid.
        Returns: [ (mask, (cy,cx)), (mask, (cy,cx)) ] sorted by cx ascending
        If fewer than 2 found, pad with None.
        """
        found = find_n_color_blobs(img, n_blobs=2, black_thresh=30)
        while len(found) < 2:
            found.append(None)
        info = []
        for m in found:
            cy, cx = compute_centroid(m)
            info.append((m, (cy, cx)))
        # Sort by x
        info.sort(key=lambda x: (x[1][1] if x[1][1] is not None else 999999))
        return info

    info_239 = detect_and_sort_blobs(frames[239])
    info_240 = detect_and_sort_blobs(frames[240])

    dist_left = distance(info_239[0][1], info_240[0][1])   # displacement for left-labeled
    dist_right = distance(info_239[1][1], info_240[1][1]) # displacement for right-labeled

    if dist_left > dist_right:
        # left moves first => blob_0, right => blob_1
        blob0_side = "left"
        blob1_side = "right"
    else:
        # right moves first => blob_0, left => blob_1
        blob0_side = "right"
        blob1_side = "left"

    print(f"Between frames 239->240, the {blob0_side} blob moved more => it is blob_0. The other is blob_1.")

    def save_mask_image(bin_mask, out_path, frame_idx):
        """
        Save bin_mask as a single-channel (0 or 255) PNG.
        If bin_mask is None, creates an empty black image of the frame's size.
        """
        if bin_mask is None:
            print(f"   [Info] Frame {frame_idx}: No blob => saving empty mask.")
            # Suppose we use the shape from frames[240] if in doubt
            h_ = frames[frame_idx].shape[0]
            w_ = frames[frame_idx].shape[1]
            empty_im = Image.new("L", (w_, h_), color=0)
            empty_im.save(out_path)
            return
        mask_255 = (bin_mask.astype(np.uint8)) * 255
        Image.fromarray(mask_255, mode="L").save(out_path)

    def get_consistent_blob_labeling(frame_idx):
        """
        Detect & sort left->right => info_ = [ (mask_left,(cyL,cxL)), (mask_right,(cyR,cxR)) ]
        Then reorder so that index0 => blob_0, index1 => blob_1
        """
        info_ = detect_and_sort_blobs(frames[frame_idx])
        if blob0_side == "left":
            # index0 => blob_0, index1 => blob_1
            return info_[0], info_[1]
        else:
            # index1 => blob_0, index0 => blob_1
            return info_[1], info_[0]

    # 4) We define a loop from frame=240 onward, storing each blob mask and
    #    measuring velocity of blob_0. The last frame for which velocity>threshold
    #    will be called M. Once we pass M+10, we stop processing further frames.

    # First, let's handle frame 239->240 to initialize.
    # We'll create variables:
    #   last_moving_frame: the last frame index for which velocity_0 > threshold
    #   we start it at None for safety.
    last_moving_frame = None

    # We'll store reference centroid for blob_0 from the previous frame
    # We'll do the detection for frame=240 to get that "previous" centroid.
    b0_240, b1_240 = get_consistent_blob_labeling(240)
    prev_centroid_0 = b0_240[1]

    # Save masks for frame 240
    out0_240 = os.path.join(blob_subfolder, f"blob_0_{240:05d}.png")
    out1_240 = os.path.join(blob_subfolder, f"blob_1_{240:05d}.png")
    save_mask_image(b0_240[0], out0_240, 240)
    save_mask_image(b1_240[0], out1_240, 240)

    # We also check the velocity from frame 239->240 to see if it was > threshold
    # (the code above used that for deciding which side is which, but let's do it again for completeness)
    # We'll detect the 'actual' blob_0 at frame239
    b0_239, b1_239 = get_consistent_blob_labeling(239)
    c0_239 = b0_239[1]  # centroid
    c0_240 = b0_240[1]  # centroid
    v_239_240 = distance(c0_239, c0_240)
    if v_239_240 > args.move_thresh:
        last_moving_frame = 240

    print(f"[Initialization] Velocity(239->240)={v_239_240:.3f}, last_moving_frame={last_moving_frame}")

    # Now loop from frame=241..end. If we pass M+10, we break.
    # Because the user only wants up to 10 frames after the blob_0 stops.
    # We'll define a small helper so we know if we have definitely a 'stop' moment:
    def have_stop_moment():
        return (last_moving_frame is not None)

    # process frames
    fidx = 241
    while fidx < total_frames:
        # If we have a last_moving_frame, and we've gone 10 frames beyond that => break
        if have_stop_moment() and fidx > (last_moving_frame + 10):
            print(f"[Info] Reached frame {fidx}, which is more than 10 frames after stop (M={last_moving_frame}). Stopping.")
            break

        # detect
        b0_info, b1_info = get_consistent_blob_labeling(fidx)
        mask0, c0 = b0_info
        mask1, c1 = b1_info

        # store each mask
        out0 = os.path.join(blob_subfolder, f"blob_0_{fidx:05d}.png")
        out1 = os.path.join(blob_subfolder, f"blob_1_{fidx:05d}.png")
        save_mask_image(mask0, out0, fidx)
        save_mask_image(mask1, out1, fidx)

        # compute velocity from prev_centroid_0
        vel_0 = distance(prev_centroid_0, c0)
        if vel_0 > args.move_thresh:
            last_moving_frame = fidx

        prev_centroid_0 = c0
        fidx += 1

    # At this point, we've either processed all frames or stopped upon reaching last_moving_frame+10.

    if last_moving_frame is None:
        print("[WARNING] It appears blob_0 never exceeded the velocity threshold after frame 239. No 'stop' moment to record.")
        print("No crops will be produced.")
        return

    M = last_moving_frame
    print(f"Determined that blob_0 stops moving at frame M={M} (the last frame with velocity > threshold).")

    # 5) Produce 3 crops at frames: M-10, M, M+10
    frames_needed = [M - 10, M, M + 10]
    frames_valid = [f for f in frames_needed if (0 <= f < total_frames)]

    # Union of the two blobs at frame M => centroid => clamp/crop
    b0_M, b1_M = get_consistent_blob_labeling(M)
    mask0_M, c0_M = b0_M
    mask1_M, c1_M = b1_M

    if mask0_M is None and mask1_M is None:
        print(f"[ERROR] At frame M={M}, both blob_0 and blob_1 are empty. Can't define union centroid. No crops.")
        return

    h, w, _ = frames[M].shape
    union_mask = np.zeros((h, w), dtype=bool)
    if mask0_M is not None:
        union_mask |= mask0_M
    if mask1_M is not None:
        union_mask |= mask1_M

    yU, xU = compute_centroid(union_mask)
    if (yU is None) or (xU is None):
        print("[ERROR] Union mask is empty at frame M. No meaningful crop.")
        return

    print(f"[Info] For final crops, union centroid at M={M} => (y={yU:.1f}, x={xU:.1f}).")

    # create the crops (width=500, height=400 by default)
    for fidx in frames_valid:
        frm = frames[fidx]
        pil_im = Image.fromarray(frm)
        # clamp
        left, top, right, bottom = clamp_crop_coords(
            cx=xU,
            cy=yU,
            crop_w=args.crop_width,
            crop_h=args.crop_height,
            img_w=w,
            img_h=h
        )
        crop_im = pil_im.crop((left, top, right, bottom))

        out_name = f"crop_frame_{fidx:05d}.png"
        out_path = os.path.join(args.output_dir, out_name)
        crop_im.save(out_path)
        print(f"Saved crop for frame={fidx}: {out_path}")

    print("==== DONE ====")
    print(f"All blob segmentations are in: {blob_subfolder}")
    print(f"Final 3 crops are in: {args.output_dir}")

if __name__ == "__main__":
    main()
