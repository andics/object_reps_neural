#!/usr/bin/env python3
"""
detect_collisions_memory_from_png.py

Detect collisions between blob_0 and a "frozen" blob_1 polygon,
*without* using Shapely, but this time using PNG masks directly.

HOW IT WORKS:
-------------
1) We read a directory of PNG masks named:
   - mask_blob_0_frame_000018.png
   - mask_blob_1_frame_000018.png
   ...
   Each is a binary mask over the *entire* frame, with black background
   for empty pixels (0) and white (255) for the blob.

2) For blob_1, we keep track of the "last valid" mask. Once we reach a frame
   where blob_1's mask is missing, we consider blob_1 "frozen" at that last
   known mask. From that point on, we compare all subsequent blob_0 masks
   against the frozen blob_1 mask to see if they overlap.

3) Overlap is defined as any pixel where both masks are True/1.

4) We stop at the first frame where they overlap and report that frame (and time in ms)
   in 'collision_info.json'. If there's no collision, we output None for both time/frame.

USAGE:
------
    python detect_collisions_memory_from_png.py \
      --json_dir /some/dir/with/png_masks \
      --output_dir outputs \
      --fps 60

By default, it will look for all 'mask_blob_0_frame_*.png' and 'mask_blob_1_frame_*.png',
sorted by the frame index. The naming pattern should include the frame number with
leading zeros if desired, but it must be parseable by the regex:  r"mask_blob_([01])_frame_(\d+)\.png".
"""

import os
import re
import json
import argparse
import numpy as np
from PIL import Image

def load_mask(png_path):
    """
    Loads a PNG file (binary mask) using Pillow and returns
    a boolean NumPy array of shape (H, W), where True indicates
    the blob is present in that pixel.
    """
    img = Image.open(png_path).convert('L')  # grayscale
    arr = np.array(img, dtype=np.uint8)
    # Assume 0 or 255; consider >128 as True
    return arr > 128

def masks_intersect(maskA, maskB):
    """
    Returns True if there's any pixel overlap between two boolean
    mask arrays of the same shape.
    """
    if maskA.shape != maskB.shape:
        # We expect same shape if the original frames are the same size.
        # If not, you might need to resize or handle differently.
        raise ValueError("Mask shapes differ, cannot check overlap.")
    return np.any(maskA & maskB)

def main():
    parser = argparse.ArgumentParser(
        "Detect collisions between a 'frozen' blob-1 memory and updated blob-0 PNG masks."
    )
    parser.add_argument("--json_dir", default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/detr/EXP_2_TTC/generate_detection_videos_and_meshes/variable_pretrained_resnet101-BConcave+AConcave+3500-frames_masks",
                        required=False,
                        help="Folder containing mask_blob_0_frame_XXXXXX.png and mask_blob_1_frame_XXXXXX.png.")
    parser.add_argument("--output_dir", default="outputs",
                        required=False,
                        help="Folder to write collision_info.json (and logs).")
    parser.add_argument("--fps", type=float, default=60.0,
                        help="Frames per second. Default=60 => ~16.67 ms/frame.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Regex to match the naming pattern:
    # mask_blob_0_frame_000018.png or mask_blob_1_frame_000018.png
    pattern = r"mask_blob_([01])_frame_(\d+)\.png"

    # We'll gather all frames for blob 0 and blob 1
    blob0_frames = {}
    blob1_frames = {}

    # Scan the directory
    for fname in os.listdir(args.json_dir):
        m = re.match(pattern, fname)
        if m:
            blob_idx = int(m.group(1))   # 0 or 1
            frame_idx = int(m.group(2)) # e.g. 18
            full_path = os.path.join(args.json_dir, fname)

            if blob_idx == 0:
                blob0_frames[frame_idx] = full_path
            else:
                blob1_frames[frame_idx] = full_path

    # We'll consider all frame indices that appear in either blob_0 or blob_1
    all_frames = sorted(set(blob0_frames.keys()) | set(blob1_frames.keys()))

    last_blob1_mask = None
    blob1_frozen = False

    collision_found = False
    collision_frame_idx = None

    # Iterate frames in ascending order
    for frame_idx in all_frames:
        # Attempt to load blob_0 mask if it exists
        mask0 = None
        if frame_idx in blob0_frames:
            mask0 = load_mask(blob0_frames[frame_idx])

        # If blob_1 not yet frozen => check if there's a new mask
        if not blob1_frozen:
            if frame_idx in blob1_frames:
                # We have a fresh mask for blob_1
                mask1 = load_mask(blob1_frames[frame_idx])
                last_blob1_mask = mask1
            else:
                # No mask for blob_1 => freeze the last known
                blob1_frozen = True
                # If we never had a valid mask => no collisions possible
                if last_blob1_mask is None:
                    break
        # If blob_1 is already frozen, we keep last_blob1_mask as is

        # Now, if we have a valid last_blob1_mask and a mask0 => check collision
        if last_blob1_mask is not None and mask0 is not None:
            if masks_intersect(mask0, last_blob1_mask):
                collision_found = True
                collision_frame_idx = frame_idx
                break

    # Prepare output
    result = {}
    if collision_found:
        # compute time in ms => frame_index * (1000/fps)
        time_ms = collision_frame_idx * (1000.0 / args.fps)
        result["collision_time_ms"] = time_ms
        result["collision_frame_idx"] = collision_frame_idx
    else:
        result["collision_time_ms"] = None
        result["collision_frame_idx"] = None

    # Write to JSON
    out_json = os.path.join(args.output_dir, "collision_info.json")
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)

    print("Done! Collision info saved at:", out_json)
    if collision_found:
        print(f"First collision at frame={collision_frame_idx}, time={result['collision_time_ms']:.2f} ms")
    else:
        print("No collision found.")


if __name__ == "__main__":
    main()
