#!/usr/bin/env python3

import os
import re
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import logging
import datetime

##############################################################################
# 0) LOGGING SETUP
##############################################################################

def setup_logger():
    """
    Creates a logger that writes DEBUG-level messages to a timestamped file
    in 'logs/' (relative to this script) and INFO-level messages to the console.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(logs_dir, f"run_log_{timestamp}.txt")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # File handler (DEBUG messages)
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    f_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(f_formatter)
    logger.addHandler(fh)

    # Console handler (INFO messages)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    c_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(c_formatter)
    logger.addHandler(ch)

    logger.info("Logger initialized. Writing detailed log to %s", log_file_path)
    return logger

##############################################################################
# 1) HELPER FUNCTIONS
##############################################################################

def parse_name_and_distance(folder_name):
    """
    Extract the folder name exactly (short_name) plus an integer at the very end if present.
    e.g. 'flipped_A_concave_5' -> short_name='flipped_A_concave_5', distance=5
         'B_something_-3'      -> short_name='B_something_-3', distance=-3
    If no integer is found, distance defaults to 0.
    """
    short = folder_name
    match = re.search(r'_(-?\d+)$', short)
    if match:
        gt_dist = int(match.group(1))
    else:
        gt_dist = 0
    return short, gt_dist

def find_smallest_frame(crops_output_dir, logger):
    """
    Given a directory containing exactly 3 files named like crop_frame_XXXXX.png,
    parse the 3 numeric parts, sort them, and return the *smallest* one.
    """
    files = [
        f for f in os.listdir(crops_output_dir)
        if f.startswith("crop_frame_") and f.endswith(".png")
    ]
    if len(files) != 3:
        msg = (f"Expected exactly 3 crop_frame_*.png in {crops_output_dir}, "
               f"found {len(files)}")
        logger.error(msg)
        raise ValueError(msg)

    frame_nums = []
    for f in files:
        match = re.search(r'crop_frame_(\d+)\.png', f)
        if not match:
            msg = f"Cannot parse frame number from '{f}'"
            logger.error(msg)
            raise ValueError(msg)
        frame_idx = int(match.group(1))
        frame_nums.append(frame_idx)
    frame_nums.sort()
    smallest = frame_nums[0]
    logger.debug("Frames in '%s': %s -> smallest is %d", crops_output_dir, frame_nums, smallest)
    return smallest

def load_mask(mask_path, logger):
    """
    Load a mask image (PNG) as a boolean numpy array (True = foreground).
    """
    logger.debug("Loading mask from %s", mask_path)
    img = Image.open(mask_path).convert('L')  # grayscale
    arr = np.array(img, dtype=np.uint8)
    return (arr > 0)

def get_centroid(mask):
    """
    Returns the (cy, cx) centroid of a binary mask.
    If no pixels are set, returns (0,0).
    """
    coords = np.argwhere(mask)  # list of (y, x)
    if len(coords) == 0:
        return (0.0, 0.0)
    cy, cx = coords.mean(axis=0)
    return (cy, cx)

def place_mask_in_array(mask, arr, offset_x, offset_y):
    """
    Place mask into arr at the given offset. True => foreground pixel.
    """
    coords = np.argwhere(mask)
    for (y, x) in coords:
        ry = y + offset_y
        rx = x + offset_x
        arr[ry, rx] = True

def shift_and_compute_overlap(mask_left, mask_right, shift):
    """
    Returns:
      overlap_pixels = number of overlapping pixels
      union_pixels   = total union of pixels
    We'll shift mask_right horizontally by shift (positive => move right).
    """
    coords_left = np.argwhere(mask_left)
    coords_right = np.argwhere(mask_right)

    if len(coords_left) == 0 and len(coords_right) == 0:
        return 0, 0
    if len(coords_left) == 0:
        return 0, len(coords_right)
    if len(coords_right) == 0:
        return 0, len(coords_left)

    yL_min, xL_min = coords_left.min(axis=0)
    yL_max, xL_max = coords_left.max(axis=0)
    yR_min, xR_min = coords_right.min(axis=0)
    yR_max, xR_max = coords_right.max(axis=0)

    # shift bounding box for right
    xR_min_shifted = xR_min + shift
    xR_max_shifted = xR_max + shift

    x_min = min(xL_min, xR_min_shifted)
    x_max = max(xL_max, xR_max_shifted)
    y_min = min(yL_min, yR_min)
    y_max = max(yL_max, yR_max)

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    if width <= 0 or height <= 0:
        # no overlap or invalid bounding region
        overlap = 0
        union = mask_left.sum() + mask_right.sum()
        return overlap, union

    arr_left = np.zeros((height, width), dtype=bool)
    arr_right = np.zeros((height, width), dtype=bool)

    offset_left_x = -x_min
    offset_right_x = -x_min + shift
    offset_left_y = -y_min
    offset_right_y = -y_min

    place_mask_in_array(mask_left, arr_left, offset_left_x, offset_left_y)
    place_mask_in_array(mask_right, arr_right, offset_right_x, offset_right_y)

    overlap_array = np.logical_and(arr_left, arr_right)
    union_array = np.logical_or(arr_left, arr_right)

    overlap = overlap_array.sum()
    union = union_array.sum()
    return overlap, union

def measure_shift_needed(mask0, mask1, threshold, logger, max_shift=500):
    """
    We measure how many horizontal pixels of shift are needed for the two blobs
    to meet or exceed some 'threshold' condition:
      - If threshold >= 1, it means the # overlapping pixels >= threshold
      - If threshold < 1, it means IoU >= threshold
    The logic:
      1) Determine which blob is "left" vs. "right" by comparing centroid Xs.
      2) Evaluate if threshold is met at shift=0
         - If yes, we shift "right" outward until the threshold is no longer met,
           and return negative of that shift.
         - If no, we shift "right" inward until we do meet threshold,
           and return that shift as a positive number.
      3) If none found up to max_shift, we return ±(max_shift+1).
    """
    c0y, c0x = get_centroid(mask0)
    c1y, c1x = get_centroid(mask1)
    if c0x <= c1x:
        mask_left, mask_right = mask0, mask1
    else:
        mask_left, mask_right = mask1, mask0

    # function to check if threshold is met at a certain shift
    def threshold_met(shift):
        overlap, union = shift_and_compute_overlap(mask_left, mask_right, shift)
        if threshold >= 1:
            # interpret threshold as "pixel overlap >= threshold"
            return overlap >= threshold
        else:
            # interpret threshold as IoU
            if union == 0:
                return False
            iou = overlap / union
            return iou >= threshold

    # see if threshold is met at shift=0
    meets = threshold_met(0)
    if meets:
        # SHIFT OUTWARD: shift from 0..max_shift until it no longer meets threshold
        for d in range(max_shift + 1):
            if not threshold_met(d):
                # the threshold is no longer met => distance is negative of that shift
                return float(-d)
        return float(-(max_shift + 1))
    else:
        # SHIFT INWARD: shift from 0..-max_shift until threshold is met
        for d in range(max_shift + 1):
            test_shift = -d
            if threshold_met(test_shift):
                return float(d)
        return float(max_shift + 1)

##############################################################################
# 2) MAIN LOGIC
##############################################################################

def main(parent_full, parent_crops, output_csv_base, logger):
    """
    1) List all subfolders in parent_crops
    2) Match each subfolder to 'full_resolution_resnet101-{subfolder}' in parent_full
    3) For each match:
       - find the *smallest* frame
       - load mask_blob_0, mask_blob_1
       - compute centroid distance (once)
       - for each threshold in [1 pixel, IoU=0.05..0.95],
           measure shift needed -> 'distance_to_boundary'
           append row to the corresponding CSV file in real time
    """
    logger.info("Starting main with parent_full=%s, parent_crops=%s, output_csv_base=%s",
                parent_full, parent_crops, output_csv_base)

    if not os.path.isdir(parent_full):
        logger.error("Parent_full directory does not exist: %s", parent_full)
        return
    if not os.path.isdir(parent_crops):
        logger.error("Parent_crops directory does not exist: %s", parent_crops)
        return

    # We'll produce multiple CSV files, one for each threshold.
    thresholds = [1]  # "touch by 1 pixel"
    # Then IoU thresholds from 0.05..0.95 in steps of 0.05
    iou_list = [round(x, 2) for x in np.arange(0.05, 0.45, 0.05)]
    thresholds.extend(iou_list)

    # We'll build a map from threshold -> CSV path
    csv_paths = {}
    # For each threshold, define a suffix, e.g. "_1px", "_0.05", etc.
    for thr in thresholds:
        if thr == 1:
            suffix = "_1px"
        else:
            suffix = f"_{thr:.2f}"
        csv_file = os.path.splitext(output_csv_base)[0] + suffix + ".csv"
        csv_paths[thr] = csv_file

    # We'll also define the columns
    columns = ["folder_name", "gt_distance", "distance_to_boundary", "distance_to_centroid", "frame_used"]

    # Initialize each CSV (write header if file doesn't exist)
    for thr, path in csv_paths.items():
        if not os.path.exists(path):
            df_init = pd.DataFrame(columns=columns)
            df_init.to_csv(path, index=False)

    # Gather subfolders in parent_crops
    crop_folders = [
        d for d in os.listdir(parent_crops)
        if os.path.isdir(os.path.join(parent_crops, d))
    ]
    logger.info("Found %d subfolders in %s", len(crop_folders), parent_crops)

    for cfold in crop_folders:
        short_name, gt_dist = parse_name_and_distance(cfold)
        logger.info("Processing subfolder: %s => short_name=%s, gt_distance=%d", cfold, short_name, gt_dist)

        # The matching folder in parent_full
        expected_full_folder = f"full_resolution_resnet101-{short_name}"
        full_dir = os.path.join(parent_full, expected_full_folder)
        if not os.path.isdir(full_dir):
            logger.warning("No matching folder for '%s' -> expected '%s' in '%s'. Skipping.",
                           cfold, expected_full_folder, parent_full)
            continue

        mask_dir = os.path.join(full_dir, "frames_masks_nonmem")
        if not os.path.isdir(mask_dir):
            logger.warning("frames_masks_nonmem not found in %s. Skipping %s.", full_dir, cfold)
            continue

        # The 'crops_output' subfolder
        crops_out_dir = os.path.join(parent_crops, cfold, "crops_output")
        if not os.path.isdir(crops_out_dir):
            logger.warning("Missing crops_output in %s, skipping.", crops_out_dir)
            continue

        # Step 1: find the smallest frame among the 3
        try:
            #---MODIFICATION--
            #Subtract 100 from the frame used in order to deal with spilling out segmentations
            frame_used = find_smallest_frame(crops_out_dir, logger) - 100
        except ValueError as e:
            logger.error("Could not find smallest frame in %s: %s", crops_out_dir, str(e))
            continue

        frame_str = f"{frame_used:06d}"
        mask0_path = os.path.join(mask_dir, f"mask_blob_0_frame_{frame_str}.png")
        mask1_path = os.path.join(mask_dir, f"mask_blob_1_frame_{frame_str}.png")

        if not (os.path.exists(mask0_path) and os.path.exists(mask1_path)):
            logger.warning("Missing mask_blob_0 or mask_blob_1 for frame=%s in %s. Skipping folder=%s.",
                           frame_str, mask_dir, cfold)
            continue

        # Load the masks
        m0 = load_mask(mask0_path, logger)
        m1 = load_mask(mask1_path, logger)

        # Compute centroid distance once
        cy0, cx0 = get_centroid(m0)
        cy1, cx1 = get_centroid(m1)
        dist_centroids = float(np.sqrt((cx1 - cx0)**2 + (cy1 - cy0)**2))

        # For each threshold, measure distance and append a row
        for thr in thresholds:
            dist_boundary = measure_shift_needed(m0, m1, thr, logger, max_shift=500)

            row_dict = {
                "folder_name": short_name,
                "gt_distance": gt_dist,
                "distance_to_boundary": dist_boundary,
                "distance_to_centroid": dist_centroids,
                "frame_used": frame_used
            }
            csv_path = csv_paths[thr]
            df_row = pd.DataFrame([row_dict], columns=columns)
            df_row.to_csv(csv_path, mode='a', header=False, index=False)

            logger.info("Appended row for threshold=%s to %s: %s", thr, csv_path, row_dict)

    logger.info("Processing complete for all thresholds.")

##############################################################################
# 3) ARGPARSE ENTRY POINT
##############################################################################

def default_output_csv(parent_full):
    """
    Example logic:
      1) Take the base name of 'parent_full' (e.g. "full_resolution_resnet101")
      2) Append "_csv"
      3) Put that directory in the same folder as this script
      4) Use "results_final.csv" as the *base* name.
    Then for each threshold, we produce e.g. "results_final_1px.csv", "results_final_0.05.csv", ...
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parent_full_basename = os.path.basename(parent_full.rstrip('/'))
    out_dir_name = parent_full_basename + "_csv"
    full_out_dir = os.path.join(script_dir, out_dir_name)
    os.makedirs(full_out_dir, exist_ok=True)

    return os.path.join(full_out_dir, "results_final.csv")

if __name__ == "__main__":
    logger = setup_logger()

    parser = argparse.ArgumentParser(
        description="Compute collision distances (for 1-pixel overlap, or IoU=0.05..0.95) "
                    "between matched subfolders {folder} and full_resolution_resnet101-{folder}, "
                    "appending results to separate CSV files in real time."
    )
    parser.add_argument("--parent_full",
                        default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/detr/VIDEO_PROCESSING_TOOLS/generate_detection_videos_and_meshes/exp_1_videos_processed/full_videos_processed",
                        help="Directory containing full_resolution_resnet101-<folder> subdirs.")
    parser.add_argument("--parent_crops",
                        default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/gpt/experiment_1/videos_processed",
                        help="Directory containing subfolders (each with 'crops_output').")
    parser.add_argument("--output_csv_base",
                        default=None,
                        help="If given, it's the base CSV path for final outputs. E.g. 'myout.csv' "
                             "will produce 'myout_1px.csv', 'myout_0.05.csv', etc. "
                             "If not given, we place them in <script_dir>/<basename_of_parent_full>_csv/results_final.csv")

    args = parser.parse_args()

    if args.output_csv_base is None:
        output_csv_path = default_output_csv(args.parent_full)
        logger.info("No --output_csv_base given. Defaulting to %s", output_csv_path)
    else:
        output_csv_path = args.output_csv_base

    main(args.parent_full, args.parent_crops, output_csv_path, logger)