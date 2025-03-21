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
    Returns (labeled_image, num_labels).
    labeled_image has int labels [0..num_labels].
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

    top = [r[1] for r in regions[:n_blobs]]
    return top

def compute_centroid(bin_mask):
    """
    Returns (y, x) centroid of bin_mask, or (None, None) if empty.
    """
    if bin_mask is None or bin_mask.sum() == 0:
        return (None, None)
    coords = np.argwhere(bin_mask)
    y_ = coords[:, 0].mean()
    x_ = coords[:, 1].mean()
    return (y_, x_)

def masks_are_different(maskA, maskB):
    """
    Returns True if maskA != maskB in any pixel.
    If one is None while the other has content, that's different as well.
    """
    if maskA is None and maskB is None:
        return False
    if maskA is None and maskB is not None:
        return (maskB.sum() > 0)
    if maskB is None and maskA is not None:
        return (maskA.sum() > 0)

    diff = np.logical_xor(maskA, maskB)
    return diff.any()

def combine_left_right_masks(left_mask, right_mask, shape):
    """
    Create a color image (H,W,3) with black background.
    - left_mask => color: RED (255,0,0)
    - right_mask => color: GREEN (0,255,0)
    """
    h, w = shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    if left_mask is not None and left_mask.sum() > 0:
        rgb[left_mask, 0] = 255  # R
        rgb[left_mask, 1] = 0
        rgb[left_mask, 2] = 0

    if right_mask is not None and right_mask.sum() > 0:
        rgb[right_mask, 0] = 0
        rgb[right_mask, 1] = 255
        rgb[right_mask, 2] = 0

    return rgb

def clamp_crop_coords(cx, cy, crop_w, crop_h, img_w, img_h):
    """
    Compute a rectangular region (left, top, right, bottom)
    of size crop_w x crop_h centered on (cx,cy),
    clamped within [0..img_w-1, 0..img_h-1].
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

def process_one_video(
    video_path: str,
    start_frame: int,
    crop_width: int,
    crop_height: int,
    output_root: str
):
    """
    Process a single video using the "movement until stop" logic.
    Creates an output subfolder named after the video base name under 'output_root',
    with:
      - blobs/ (left, right, combined)
      - crops_output/ (final 3 crops)
      - log_<timestamp>.log
    """

    # Prepare base output folder for this video
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_root, video_basename)
    os.makedirs(video_output_folder, exist_ok=True)

    # Subfolders
    blobs_folder = os.path.join(video_output_folder, "blobs")
    crops_folder = os.path.join(video_output_folder, "crops_output")
    os.makedirs(blobs_folder, exist_ok=True)
    os.makedirs(crops_folder, exist_ok=True)

    # Per-video log file
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(video_output_folder, f"log_{now_str}.log")
    log_file = open(log_filename, "w")

    def log_line(msg):
        print(msg)
        log_file.write(msg + "\n")

    log_line("========================================")
    log_line(f"Processing video: {video_path}")
    log_line(f"start_frame={start_frame}")
    log_line(f"crop_width={crop_width}, crop_height={crop_height}")
    log_line("========================================\n")

    # Read all frames
    log_line("Reading frames...")
    reader = imageio.get_reader(video_path, format='ffmpeg')
    frames = [frm for frm in reader]
    reader.close()
    total_frames = len(frames)
    log_line(f"  - Total frames: {total_frames}")

    if start_frame < 0 or start_frame >= total_frames - 1:
        log_line(f"ERROR: start_frame={start_frame} out of range [0..{total_frames-2}]. Skipping.")
        log_file.close()
        return

    # Utility to detect and sort two blobs
    def detect_and_sort_two_blobs(frame_img):
        found = find_n_color_blobs(frame_img, n_blobs=2, black_thresh=30)
        while len(found) < 2:
            found.append(None)

        results = []
        for m in found:
            cy, cx = compute_centroid(m)
            results.append((m, (cy, cx)))
        # sort by centroid.x
        results.sort(key=lambda x: (x[1][1] if x[1][1] is not None else 9999999))

        return results[0][0], results[1][0]  # left_mask, right_mask

    def save_mask_as_png(mask_, path_, shape):
        """
        If mask_ is None or empty => black image of shape (H,W).
        Else => single-channel 0/255.
        """
        if mask_ is None or mask_.sum() == 0:
            h_, w_ = shape
            empty_im = Image.new("L", (w_, h_), color=0)
            empty_im.save(path_)
            return
        mask_255 = (mask_.astype(np.uint8)) * 255
        Image.fromarray(mask_255, "L").save(path_)

    moving_blob = None  # "left" or "right"
    stop_frame = None
    prev_left_mask = None
    prev_right_mask = None

    # Main loop
    for fidx in range(start_frame, total_frames):
        frame_img = frames[fidx]
        h, w, _ = frame_img.shape

        left_mask, right_mask = detect_and_sort_two_blobs(frame_img)

        # Save left, right
        out_left_path = os.path.join(blobs_folder, f"blob_left_{fidx:05d}.png")
        out_right_path = os.path.join(blobs_folder, f"blob_right_{fidx:05d}.png")
        save_mask_as_png(left_mask, out_left_path, (h, w))
        save_mask_as_png(right_mask, out_right_path, (h, w))

        # Combined (colored)
        combined_rgb = combine_left_right_masks(left_mask, right_mask, (h, w))
        out_combined_path = os.path.join(blobs_folder, f"blob_01_{fidx:05d}.png")
        Image.fromarray(combined_rgb, "RGB").save(out_combined_path)

        if fidx == start_frame:
            log_line(f"[Frame={fidx}] => first frame in sequence => no previous => cannot detect movement yet.")
            prev_left_mask = left_mask
            prev_right_mask = right_mask
            continue

        left_diff = masks_are_different(prev_left_mask, left_mask)
        right_diff = masks_are_different(prev_right_mask, right_mask)

        log_line(f"[Frame={fidx}] left_diff={left_diff}, right_diff={right_diff}, moving_blob={moving_blob}")

        if moving_blob is None:
            # Identify which side is moving
            if left_diff or right_diff:
                # If both changed, pick the side with bigger difference
                if left_diff and right_diff:
                    def mask_diff_count(m1, m2):
                        if m1 is None and m2 is None:
                            return 0
                        if m1 is None and m2 is not None:
                            return m2.sum()
                        if m2 is None and m1 is not None:
                            return m1.sum()
                        d_ = np.logical_xor(m1, m2)
                        return d_.sum()

                    dl = mask_diff_count(prev_left_mask, left_mask)
                    dr = mask_diff_count(prev_right_mask, right_mask)
                    if dl >= dr:
                        moving_blob = "left"
                    else:
                        moving_blob = "right"
                elif left_diff:
                    moving_blob = "left"
                else:
                    moving_blob = "right"
                log_line(f"   => Identified mover as '{moving_blob}'.")
        else:
            # We already know who is moving => check if it has stopped
            if moving_blob == "left":
                if not left_diff:
                    # Stopped here
                    stop_frame = fidx
                    log_line(f"   => Blob_0 (left) STOPPED at frame={fidx}")
                    break
            else:  # "right"
                if not right_diff:
                    stop_frame = fidx
                    log_line(f"   => Blob_0 (right) STOPPED at frame={fidx}")
                    break

        prev_left_mask = left_mask
        prev_right_mask = right_mask

    # Done loop
    if moving_blob is None:
        log_line("\n[WARNING] No moving blob found => no stop => no crops.")
        log_file.close()
        return

    if stop_frame is None:
        log_line("\nReached end of video => mover never fully stopped => no crops.")
        log_file.close()
        return

    M = stop_frame
    log_line(f"\nStop frame = M={M}. We'll produce crops (M-10, M, M+10).")

    # Generate final 3 crops
    frames_of_interest = [M-10, M, M+10]
    valid_frames = [f for f in frames_of_interest if 0 <= f < total_frames]

    # union at M
    frameM = frames[M]
    hM, wM, _ = frameM.shape
    leftM, rightM = detect_and_sort_two_blobs(frameM)
    union_mask = np.zeros((hM, wM), dtype=bool)
    if leftM is not None:
        union_mask |= leftM
    if rightM is not None:
        union_mask |= rightM

    if union_mask.sum() == 0:
        log_line("[ERROR] Union at M is empty => no centroid => no crops.")
        log_file.close()
        return

    coords = np.argwhere(union_mask)
    yU = coords[:, 0].mean()
    xU = coords[:, 1].mean()
    log_line(f"Union centroid => (y={yU:.1f}, x={xU:.1f}).")

    for fidx in valid_frames:
        frm = frames[fidx]
        pil_img = Image.fromarray(frm)

        (left, top, right, bottom) = clamp_crop_coords(
            cx=xU, cy=yU,
            crop_w=crop_width, crop_h=crop_height,
            img_w=wM, img_h=hM
        )
        crop_im = pil_img.crop((left, top, right, bottom))

        out_name = f"crop_frame_{fidx:05d}.png"
        out_path = os.path.join(crops_folder, out_name)
        crop_im.save(out_path)
        log_line(f"   => Saved crop for frame={fidx}: {out_path}")

    log_line("\n=== DONE WITH THIS VIDEO! ===")
    log_line(f"   - Mover side = '{moving_blob}'")
    log_line(f"   - Stopped at frame M={M}")
    log_line(f"   - Blob masks => {blobs_folder}")
    log_line(f"   - 3 crops => {crops_folder}")
    log_line(f"   - Log file => {log_filename}")
    log_file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Process a folder of .mp4 videos, for each do the 'moving blob until stop' logic."
    )
    parser.add_argument("--videos_folder",
        default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/gpt/experiment_1/Exp1/Stimuli/Exp1_videos", type=str, required=False,
        help="Path to folder containing .mp4 videos.")
    parser.add_argument("--start_frame", type=int, default=200,
        help="Frame index to begin checking movement."
    )
    parser.add_argument("--output_folder", default="videos_processed",
        help="Root output folder. For each video we'll create a subfolder named after that video."
    )
    parser.add_argument("--crop_width", type=int, default=500, help="Width of final crops.")
    parser.add_argument("--crop_height", type=int, default=400, help="Height of final crops.")
    args = parser.parse_args()

    print(f"Looking for .mp4 files in: {args.videos_folder}")
    files = sorted(os.listdir(args.videos_folder))
    mp4_files = [f for f in files if f.lower().endswith(".mp4")]

    if not mp4_files:
        print("No .mp4 files found in folder. Exiting.")
        return

    os.makedirs(args.output_folder, exist_ok=True)

    for video_file in mp4_files:
        video_path = os.path.join(args.videos_folder, video_file)
        print(f"\n===== PROCESSING VIDEO: {video_file} =====")
        process_one_video(
            video_path=video_path,
            start_frame=args.start_frame,
            crop_width=args.crop_width,
            crop_height=args.crop_height,
            output_root=args.output_folder
        )

    print("\nAll videos processed. Exiting.")

if __name__ == "__main__":
    main()
