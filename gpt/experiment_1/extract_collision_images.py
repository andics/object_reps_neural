#!/usr/bin/env python3

import os
import argparse
import imageio
import numpy as np
from PIL import Image
from datetime import datetime

def label_connected(bin_image):
    """
    Simple 8-connected component labeling.
    Returns (labeled_image, num_labels), where labeled_image has int labels [0..num_labels].
    """
    h, w = bin_image.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    current_label = 0

    def neighbors(r, c):
        for nr in (r - 1, r, r + 1):
            for nc in (c - 1, c, c + 1):
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
    gray = frame_np.sum(axis=2)  # sum across RGB
    non_black = (gray > black_thresh)

    labeled, num_labels = label_connected(non_black)
    if num_labels < 1:
        return []

    regions = []
    for lbl_id in range(1, num_labels + 1):
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
    y_ = coords[:, 0].mean()
    x_ = coords[:, 1].mean()
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
    If either is None, and the other is not empty, that means 'different'.
    """
    if maskA is None and maskB is None:
        return False  # both None => no difference
    if maskA is None and maskB is not None:
        return maskB.sum() > 0
    if maskB is None and maskA is not None:
        return maskA.sum() > 0

    # XOR => True where they differ
    diff = np.logical_xor(maskA, maskB)
    return diff.any()

def combine_left_right_masks(left_mask, right_mask, shape):
    """
    Create a color image (H,W,3) with black background.
    Left mask => color it RED (255,0,0)
    Right mask => color it GREEN (0,255,0)
    Returns a uint8 array.
    If both masks have overlap (the problem statement says they won't),
    they'd appear yellow, but we won't handle that explicitly.
    """
    h, w = shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    if left_mask is not None and left_mask.sum() > 0:
        rgb[left_mask, 0] = 255  # R
        rgb[left_mask, 1] = 0
        rgb[left_mask, 2] = 0

    if right_mask is not None and right_mask.sum() > 0:
        rgb[right_mask, 0] = 0
        rgb[right_mask, 1] = 255  # G
        rgb[right_mask, 2] = 0

    return rgb

def main():
    parser = argparse.ArgumentParser(
        description="Detect which blob is moving from classical CV, track until it stops, produce 3 final crops, and log everything."
    )
    parser.add_argument("--video_path",
        default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/gpt/experiment_1/Exp1/Stimuli/Exp1_videos/A_concave_4.mp4",
        help="Input video path."
    )
    parser.add_argument("--start_frame", type=int, default=200,
        help="Frame index to begin the segmentation & movement check."
    )
    parser.add_argument("--blobs_dir", default="blobs",
        help="Folder to store each frame's blob masks (three per frame: left, right, combined)."
    )
    parser.add_argument("--output_dir", default="crops_output",
        help="Folder to store final 3 crops."
    )
    parser.add_argument("--crop_width", type=int, default=500, help="Width of final crops.")
    parser.add_argument("--crop_height", type=int, default=400, help="Height of final crops.")
    args = parser.parse_args()

    # Create a time-stamped log file
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_{now_str}.log"
    log_file = open(log_filename, "w")

    def log_line(message):
        # write to console + log file
        print(message)
        log_file.write(message + "\n")

    log_line("========================================")
    log_line(f"Video path: {args.video_path}")
    log_line(f"start_frame={args.start_frame}")
    log_line(f"blobs_dir={args.blobs_dir}")
    log_line(f"output_dir={args.output_dir}")
    log_line(f"crop_width={args.crop_width}, crop_height={args.crop_height}")
    log_line("========================================\n")

    # 1) Read all frames
    log_line("Reading all frames from video...")
    reader = imageio.get_reader(args.video_path, format='ffmpeg')
    frames = [frm for frm in reader]
    reader.close()
    total_frames = len(frames)
    log_line(f"Read total {total_frames} frames.\n")

    # Guard checks
    if args.start_frame < 0 or args.start_frame >= total_frames - 1:
        log_line(f"ERROR: start_frame={args.start_frame} is out of range [0..{total_frames-2}]. Exiting.")
        log_file.close()
        return

    # 2) Prepare folder for storing blob masks
    os.makedirs(args.blobs_dir, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    blob_subfolder = os.path.join(args.blobs_dir, video_basename)
    os.makedirs(blob_subfolder, exist_ok=True)

    # Utility to segment & order two blobs left->right
    def detect_and_sort_two_blobs(frame_img):
        found = find_n_color_blobs(frame_img, n_blobs=2, black_thresh=30)
        while len(found) < 2:
            found.append(None)

        results = []
        for m in found:
            cy, cx = compute_centroid(m)
            results.append((m, (cy, cx)))
        # sort by x
        results.sort(key=lambda x: (x[1][1] if x[1][1] is not None else 9999999))

        # return masks only
        return results[0][0], results[1][0]  # left_mask, right_mask

    def save_mask_as_png(mask_, path_, shape):
        """
        If mask_ is None or empty => black image of shape
        Else => single-channel 0/255.
        """
        if mask_ is None or mask_.sum() == 0:
            h_, w_ = shape
            empty_im = Image.new("L", (w_, h_), color=0)
            empty_im.save(path_)
            return
        mask_255 = (mask_.astype(np.uint8))*255
        Image.fromarray(mask_255, "L").save(path_)

    moving_blob = None  # "left" or "right"
    stop_frame = None

    prev_left_mask = None
    prev_right_mask = None

    # 3) Main loop: from start_frame..(end)
    for fidx in range(args.start_frame, total_frames):
        frame_img = frames[fidx]
        h, w, _ = frame_img.shape

        left_mask, right_mask = detect_and_sort_two_blobs(frame_img)

        # Save separate masks
        out_left_path = os.path.join(blob_subfolder, f"blob_left_{fidx:05d}.png")
        out_right_path = os.path.join(blob_subfolder, f"blob_right_{fidx:05d}.png")
        save_mask_as_png(left_mask, out_left_path, (h, w))
        save_mask_as_png(right_mask, out_right_path, (h, w))

        # Also save combined
        combined_rgb = combine_left_right_masks(left_mask, right_mask, (h, w))
        out_combined_path = os.path.join(blob_subfolder, f"blob_01_{fidx:05d}.png")
        Image.fromarray(combined_rgb, "RGB").save(out_combined_path)

        if fidx == args.start_frame:
            # No previous frame to compare => just set prev & continue
            prev_left_mask = left_mask
            prev_right_mask = right_mask
            log_line(f"[Frame={fidx}] (start_frame). No previous => can't detect movement yet.")
            continue

        # compare with previous
        left_diff = masks_are_different(prev_left_mask, left_mask)
        right_diff = masks_are_different(prev_right_mask, right_mask)

        log_line(f"[Frame={fidx}] left_diff={left_diff}, right_diff={right_diff}, moving_blob={moving_blob}")

        if moving_blob is None:
            # we haven't flagged a mover yet
            if left_diff or right_diff:
                if left_diff and right_diff:
                    # both changed => pick whichever changed more
                    # to measure "which changed more," compute sum of XOR
                    def mask_diff_count(m1, m2):
                        if m1 is None or m2 is None:
                            if m1 is None and m2 is not None:
                                return m2.sum()
                            elif m2 is None and m1 is not None:
                                return m1.sum()
                            else:
                                return 0
                        d_ = np.logical_xor(m1, m2)
                        return d_.sum()

                    diff_left = mask_diff_count(prev_left_mask, left_mask)
                    diff_right = mask_diff_count(prev_right_mask, right_mask)
                    if diff_left >= diff_right:
                        moving_blob = "left"
                    else:
                        moving_blob = "right"
                elif left_diff:
                    moving_blob = "left"
                else:
                    moving_blob = "right"
                log_line(f"   => Identified mover as '{moving_blob}' this frame.")
        else:
            # we do have a mover. Check if it has stopped
            if moving_blob == "left":
                if not left_diff:
                    # that means it has stopped this frame
                    stop_frame = fidx
                    log_line(f"   => Blob_0 (left) STOPPED at frame={fidx}.")
                    break
            else:  # "right"
                if not right_diff:
                    stop_frame = fidx
                    log_line(f"   => Blob_0 (right) STOPPED at frame={fidx}.")
                    break

        # update prev
        prev_left_mask = left_mask
        prev_right_mask = right_mask

    # end loop

    if moving_blob is None:
        log_line("\n[WARNING] We never found any moving blob. No stop => no crops.")
        log_file.close()
        return

    if stop_frame is None:
        log_line("\nWe reached the end of the video => the moving blob never fully stopped. No crops produced.")
        log_file.close()
        return

    M = stop_frame
    log_line(f"\nFinal STOP frame is M={M}. We'll produce 3 crops => (M-10), M, (M+10).")

    frames_of_interest = [M-10, M, M+10]
    valid_frames = [f for f in frames_of_interest if 0 <= f < total_frames]

    # Re-segment frame M to get union
    frameM = frames[M]
    hM, wM, _ = frameM.shape
    leftM, rightM = detect_and_sort_two_blobs(frameM)

    union_mask = np.zeros((hM, wM), dtype=bool)
    if leftM is not None:
        union_mask |= leftM
    if rightM is not None:
        union_mask |= rightM

    if union_mask.sum() == 0:
        log_line("[ERROR] Union mask at frame M is empty => no centroid => no crops.")
        log_file.close()
        return

    coords = np.argwhere(union_mask)
    yU = coords[:,0].mean()
    xU = coords[:,1].mean()
    log_line(f"Union centroid at M={M} => (y={yU:.1f}, x={xU:.1f}). Creating crops...")

    os.makedirs(args.output_dir, exist_ok=True)
    for fidx in valid_frames:
        frm = frames[fidx]
        pil_img = Image.fromarray(frm)
        left, top, right, bottom = clamp_crop_coords(
            cx=xU, cy=yU,
            crop_w=args.crop_width, crop_h=args.crop_height,
            img_w=wM, img_h=hM
        )
        cropped = pil_img.crop((left, top, right, bottom))

        out_name = f"crop_frame_{fidx:05d}.png"
        out_path = os.path.join(args.output_dir, out_name)
        cropped.save(out_path)
        log_line(f"   => Saved crop for frame={fidx}: {out_path}")

    log_line("\n=== DONE! ===")
    log_line(f"   - Moving blob was '{moving_blob}'.")
    log_line(f"   - It stopped at frame M={M}.")
    log_line(f"   - Blob masks are in: {blob_subfolder}")
    log_line(f"   - Final 3 crops are in: {args.output_dir}")
    log_line(f"   - Logs saved to: {log_filename}")

    log_file.close()

if __name__ == "__main__":
    main()
