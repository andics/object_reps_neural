#!/usr/bin/env python3
"""
detect_collisions_memory_no_shapely_fixed.py

Detect collisions between blob_0 and a "frozen" blob_1 polygon,
*without* using Shapely. Instead we:

1) Rasterize each polygon in a minimal bounding box using matplotlib.path.Path.
2) Check if there's any overlap of "True" pixels in their boolean masks.

We read a folder of JSON files (frame_XXXXXX.json) with:
{
  "segmentation_blob_0": [[x0,y0], [x1,y1], ...],
  "segmentation_blob_1": [[x0,y0], [x1,y1], ...],
  ...
}
All coordinates are in "center-based" format.

We track the last valid polygon for blob 1. Once we see an empty one,
we "freeze" that last polygon for blob 1. For each subsequent frame,
we see if blob_0's polygon intersects the frozen blob_1 polygon.
If so, we report the frame/time in ms, given fps=60 by default.

Usage:
    python detect_collisions_memory_no_shapely_fixed.py \
      --json_dir /some/dir/with/frame_jsons \
      --output_dir outputs \
      --fps 60
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib.path as mpath  # for polygon fill


def polygons_intersect(ptsA, ptsB):
    """
    Return True if the polygons described by ptsA and ptsB (each a list of [x, y])
    intersect. We'll do a rasterization approach:
      1) If either polygon has fewer than 3 points => no intersection
      2) Compute bounding box (integer-floored/ceiled) of all points
      3) Shift them so bounding box min_x => 0, min_y => 0
      4) Rasterize each polygon in that bounding box => 2D boolean arrays
      5) If bitwise AND of the two masks has any 'True' => polygons intersect
    """
    if len(ptsA) < 3 or len(ptsB) < 3:
        return False

    # 1) bounding box for all points
    all_x = [p[0] for p in ptsA] + [p[0] for p in ptsB]
    all_y = [p[1] for p in ptsA] + [p[1] for p in ptsB]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Convert to integer bounding box using floor/ceil
    min_xi = int(np.floor(min_x))
    max_xi = int(np.ceil(max_x))
    min_yi = int(np.floor(min_y))
    max_yi = int(np.ceil(max_y))

    # 2) determine bounding box size
    width = max_xi - min_xi + 1
    height = max_yi - min_yi + 1
    if width <= 0 or height <= 0:
        return False

    # Shift polygons so that min_xi => 0, min_yi => 0
    shift_x = -min_xi
    shift_y = -min_yi
    shiftedA = [(p[0] + shift_x, p[1] + shift_y) for p in ptsA]
    shiftedB = [(p[0] + shift_x, p[1] + shift_y) for p in ptsB]

    # 3) Rasterize
    maskA = polygon_to_mask(shiftedA, (height, width))
    maskB = polygon_to_mask(shiftedB, (height, width))

    # 4) Check overlap
    overlap = maskA & maskB
    return overlap.any()


def polygon_to_mask(polygon_points, mask_shape):
    """
    Convert the polygon (list of (x,y)) into a 2D boolean array of shape=mask_shape=(H,W),
    using a fill operation via matplotlib.path.Path.

    polygon_points are in the same "pixel" space => we fill them.
    """
    (H, W) = mask_shape
    if not polygon_points:
        return np.zeros(mask_shape, dtype=bool)

    # Ensure it is treated as a closed polygon
    poly_path = mpath.Path(
        np.array(polygon_points, dtype=np.float32),
        closed=True
    )

    # We'll sample each pixel center => (col+0.5, row+0.5)
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    flat_coords = np.vstack((grid_x.ravel() + 0.5,
                             grid_y.ravel() + 0.5)).T  # (H*W, 2)

    inside = poly_path.contains_points(flat_coords)
    inside_mask = inside.reshape((H, W))
    return inside_mask


def main():
    parser = argparse.ArgumentParser(
        "Detect collisions between a 'frozen' blob-1 memory and updated blob-0 polygons (no Shapely)."
    )
    parser.add_argument("--json_dir", default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/detr/EXPERIMENTS/generate_detection_videos_and_meshes/variable_pretrained_resnet101-BConcave+AConcave+3500-frames_json_memory_processed",
                        required=False,
                        help="Folder containing frame_XXXXXX.json files.")
    parser.add_argument("--output_dir", default="outputs",
                        required=False,
                        help="Folder to write collision_info.json.")
    parser.add_argument("--fps", type=float, default=60.0,
                        help="Frames per second. Default=60 => ~16.67 ms/frame.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # gather JSON files in ascending order of frame index
    pattern = r"frame_(\d+)\.json"
    frames = []
    for fname in os.listdir(args.json_dir):
        m = re.match(pattern, fname)
        if m:
            idx = int(m.group(1))
            frames.append((idx, os.path.join(args.json_dir, fname)))
    frames.sort(key=lambda x: x[0])

    last_poly_blob1 = None
    blob1_frozen = False

    collision_found = False
    collision_frame_idx = None

    for frame_idx, json_path in frames:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract polygons
        poly_blob0 = data.get("segmentation_blob_0", [])
        poly_blob1 = data.get("segmentation_blob_1", [])

        # If blob1 not frozen => update or freeze
        if not blob1_frozen:
            if len(poly_blob1) >= 3:
                last_poly_blob1 = poly_blob1
            else:
                # freeze now
                blob1_frozen = True
                # If we never had a valid poly => done (no collisions possible)
                if last_poly_blob1 is None:
                    break

        # If blob1 is frozen & we have a last_poly_blob1 => check intersection
        if blob1_frozen and last_poly_blob1 is not None:
            if polygons_intersect(poly_blob0, last_poly_blob1):
                collision_found = True
                collision_frame_idx = frame_idx
                break

    result = {}
    if collision_found:
        # compute time in ms => frame_index * (1000/fps)
        time_ms = collision_frame_idx * (1000.0 / args.fps)
        result["collision_time_ms"] = time_ms
        result["collision_frame_idx"] = collision_frame_idx
    else:
        result["collision_time_ms"] = None
        result["collision_frame_idx"] = None

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
