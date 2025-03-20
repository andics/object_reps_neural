#!/usr/bin/env python3

import argparse
import os
import re
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Compute memory-based blob masks.")
    parser.add_argument(
        "--parent_dir",
        type=str,
        default=r"Q:\Projects\Object_reps_neural\Programming\detr\EXPERIMENTS\generate_detection_videos_and_meshes\videos_processed",
        required=False,
        help="Path to the parent directory containing subfolders with frames_masks_nonmem directories."
    )
    parser.add_argument(
        "--memory_function",
        type=str,
        default="sigmoid",
        required=False,
        choices=["sigmoid", "parabola", "linear"],
        help="Which memory weighting function to use."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="Scale parameter for the chosen memory function (e.g. controlling steepness for sigmoid)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold in [0,1] to binarize the accumulated memory mask."
    )
    return parser.parse_args()


def get_frame_and_blob_indices(filename):
    """
    Given a filename like mask_blob_0_frame_000023.png
    return (blob_id, frame_id) as integers.
    """
    # A simple regex approach:
    # mask_memory_blob_(\d+)_frame_(\d+).png
    match = re.search(r"mask_blob_(\d+)_frame_(\d+)\.png", filename)
    if not match:
        return None, None
    blob_id = int(match.group(1))
    frame_id = int(match.group(2))
    return blob_id, frame_id


def list_video_subfolders(parent_dir):
    """
    Return all immediate subfolders of parent_dir
    that contain a 'frames_masks_nonmem' directory.
    """
    video_subfolders = []
    for entry in os.scandir(parent_dir):
        if entry.is_dir():
            frames_dir = os.path.join(entry.path, "frames_masks_nonmem")
            if os.path.isdir(frames_dir):
                video_subfolders.append(entry.name)
    return video_subfolders


def sigmoid_weight(i, j, scale=5.0):
    """
    Sigmoid weighting: frames close to i have higher weight.
    i = current frame index, j = historical frame index.
    scale controls the steepness.
    """
    # We want large weight for j close to i, smaller for j far from i
    # A simple variant: weight = 1 / (1 + exp((j - i)/scale))
    return 1.0 / (1.0 + np.exp((j - i)/scale))

def parabola_weight(i, j, scale=5.0):
    """
    Parabolic weighting: emphasize first and last frames among [0..i].
    This is just one of many ways to define a 'parabola' shape.
    We'll do something that peaks near j=0 and j=i, lower in the middle.
    
    For example:
       dist_to_ends = min(j, i-j)
       weight = 1 - (dist_to_ends / scale)^2, clipped at >= 0
    You can be more creative as you like.
    """
    dist_to_start = j
    dist_to_end = i - j
    dist_to_closest_end = min(dist_to_start, dist_to_end)
    w = 1.0 - (dist_to_closest_end / scale)**2
    return max(0.0, w)

def linear_weight(i, j, scale=5.0):
    """
    Linear weighting that decreases as we go back in time from i.
    If i-j >= scale, weight is 0. Otherwise it is 1 - (i-j)/scale.
    """
    dist = i - j
    if dist < 0:
        return 0.0
    if dist >= scale:
        return 0.0
    return 1.0 - dist/scale


def compute_memory_mask_for_frame(masks_dict, frame_idx, memory_func, scale=5.0, threshold=0.5):
    """
    Compute the memory-based mask for 'frame_idx' by accumulating
    all frames 0..frame_idx for which we have a mask, weighted
    by the chosen memory function. Then threshold to get a binary mask.

    masks_dict: dict of {frame_id: np.ndarray (binary mask, 0/255) }
    frame_idx: current frame index
    memory_func: a function w(i,j,scale) that returns a float weight
    scale: weighting scale parameter
    threshold: final threshold in [0,1] for binarizing

    Return: np.ndarray (same shape as the input masks) in {0, 255}.
    """
    # Gather all frames up to frame_idx that exist in masks_dict
    # If there are none, return an all-zero mask.
    valid_frames = [f for f in masks_dict.keys() if f <= frame_idx]
    if not valid_frames:
        # no frames at all => blank mask
        example_size = None
        return None

    # We assume all masks have the same size. We'll take the shape from the first.
    h, w = next(iter(masks_dict.values())).shape
    acc = np.zeros((h, w), dtype=np.float32)
    total_weight = 0.0

    for f_id in valid_frames:
        mask_f = masks_dict[f_id] / 255.0  # convert from {0,255} to {0,1} for weighting
        w_f = memory_func(frame_idx, f_id, scale=scale)
        acc += w_f * mask_f
        total_weight += w_f

    if total_weight > 1e-7:
        acc = acc / total_weight

    # Binarize
    # acc is now in [0,1], so threshold
    mem_mask = (acc >= threshold).astype(np.uint8) * 255
    return mem_mask


def process_single_subfolder(
    parent_dir,
    subfolder_name,
    memory_function,
    scale=5.0,
    threshold=0.5,
    script_root_output="memory_modelling"
):
    """
    Process one video subfolder (e.g. "variable_pretrained_resnet101-BConcave+AConvex+3700")
    that contains 'frames_masks_nonmem'. Build memory masks for each blob
    across all frames, according to the chosen memory function.

    Output the results to:
      root_dir_of_script / memory_modelling / subfolder_name / memory_function / mask_memory_blob_?_frame_??????.png
    """
    frames_dir = os.path.join(parent_dir, subfolder_name, "frames_masks_nonmem")
    if not os.path.isdir(frames_dir):
        print(f"[WARNING] No frames_masks_nonmem directory in {subfolder_name}, skipping.")
        return

    # 1. Gather all .pngs
    all_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not all_files:
        print(f"[WARNING] No .png files found in {frames_dir}, skipping.")
        return

    # 2. Group them by blob_id
    blob_to_frames = {}  # blob_id -> { frame_id: mask }
    max_frame_index = -1

    for fname in all_files:
        blob_id, frame_id = get_frame_and_blob_indices(fname)
        if blob_id is None or frame_id is None:
            continue

        full_path = os.path.join(frames_dir, fname)
        mask_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue

        if blob_id not in blob_to_frames:
            blob_to_frames[blob_id] = {}
        blob_to_frames[blob_id][frame_id] = mask_img

        if frame_id > max_frame_index:
            max_frame_index = frame_id

    if max_frame_index < 0:
        print(f"[WARNING] Could not parse any valid mask files in {frames_dir}, skipping.")
        return

    # 3. Prepare output directory: root_dir_of_script/memory_modelling/<subfolder_name>/<memory_function>/
    #    (We assume this script is being run from somewhere. We'll just create it relative to the script's CWD
    #     or you might want an absolute path. Adjust as necessary.)
    output_dir = os.path.join(
        os.getcwd(),                     # root_dir_of_script can be your script's directory or wherever you want
        script_root_output,
        subfolder_name,
        memory_function
    )
    os.makedirs(output_dir, exist_ok=True)

    # 4. Decide which weighting function to use
    if memory_function == "sigmoid":
        weight_func = sigmoid_weight
    elif memory_function == "parabola":
        weight_func = parabola_weight
    elif memory_function == "linear":
        weight_func = linear_weight
    else:
        raise ValueError(f"Unknown memory function: {memory_function}")

    # 5. For each blob, compute memory masks for frames 0..max_frame_index
    #    BUT remember: if a blob disappears after last_appearance, we keep memory constant.
    #    If it hasn't appeared yet, memory is blank.

    for blob_id, frames_dict in blob_to_frames.items():
        # Find first_appearance and last_appearance
        sorted_frames = sorted(frames_dict.keys())
        first_appearance = sorted_frames[0]
        last_appearance = sorted_frames[-1]

        # Precompute memory masks for each frame in [0..max_frame_index]
        memory_masks = {}
        for i in range(max_frame_index + 1):
            if i < first_appearance:
                # Blob not appeared yet => blank
                # We'll output an all-0 mask with the same size as the first_appearance's mask
                h, w = frames_dict[first_appearance].shape
                mem_mask = np.zeros((h, w), dtype=np.uint8)
            elif i > last_appearance:
                # Blob has disappeared => keep memory constant
                mem_mask = memory_masks[last_appearance]
            else:
                # Blob is active in range [first_appearance..last_appearance]
                mem_mask = compute_memory_mask_for_frame(
                    frames_dict,
                    i,
                    weight_func,
                    scale=scale,
                    threshold=threshold
                )
                if mem_mask is None:
                    # This means no frames up to i exist, fallback to 0 or previous
                    # but in principle we do have frames if i >= first_appearance
                    h, w = frames_dict[first_appearance].shape
                    mem_mask = np.zeros((h, w), dtype=np.uint8)

            memory_masks[i] = mem_mask

        # 6. Write memory masks to disk
        for i in range(max_frame_index + 1):
            out_mask = memory_masks[i]
            # Ensure directory exists
            out_fname = f"mask_memory_blob_{blob_id}_frame_{i:06d}.png"
            out_path = os.path.join(output_dir, out_fname)
            cv2.imwrite(out_path, out_mask)

    print(f"[INFO] Finished processing {subfolder_name} with memory function '{memory_function}'.")


def main():
    args = parse_args()

    parent_dir = args.parent_dir
    memory_function = args.memory_function
    scale = args.scale
    threshold = args.threshold

    # 1. List all subfolders that contain frames_masks_nonmem
    video_subfolders = list_video_subfolders(parent_dir)
    if not video_subfolders:
        print("[WARNING] No subfolders with frames_masks_nonmem found. Exiting.")
        return

    # 2. Process each subfolder
    for sub_name in video_subfolders:
        process_single_subfolder(
            parent_dir=parent_dir,
            subfolder_name=sub_name,
            memory_function=memory_function,
            scale=scale,
            threshold=threshold,
            script_root_output="memory_modelling"  # top-level directory for outputs
        )


if __name__ == "__main__":
    main()
