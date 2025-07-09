#!/usr/bin/env python3
"""
process_collisions.py

Example usage:
--------------
python process_collisions.py \
    --zip_path path/to/input.zip \
    --name_mapping path/to/name_mapping.json \
    --csv_path path/to/participant_data.csv \
    --iou_start 0.05 \
    --iou_end 0.95 \
    --iou_step 0.05
"""
# Taken from: detr\EXP_2_TTC\data_analysis_v2.py

import argparse
import os
import sys
import json
import zipfile
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Any


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute collision times for video frames under varying IoU thresholds "
                    "and correlate with participant responses."
    )
    parser.add_argument(
        "--zip_path",
        required=True,
        help="Path to the input .zip file containing the extracted frames."
    )
    parser.add_argument(
        "--name_mapping",
        required=True,
        help="Path to the name_mapping.json file."
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        help="Path to the CSV file containing participant data."
    )
    parser.add_argument(
        "--iou_start",
        type=float,
        default=0.05,
        help="Starting IoU threshold (float). Default is 0.05."
    )
    parser.add_argument(
        "--iou_end",
        type=float,
        default=0.95,
        help="Ending IoU threshold (float). Default is 0.95."
    )
    parser.add_argument(
        "--iou_step",
        type=float,
        default=0.05,
        help="IoU increment (float). Default is 0.05."
    )

    return parser.parse_args()


def setup_logging() -> logging.Logger:
    """Configure logging to a file and return the logger."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"log_{timestamp}.txt"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Also print to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    logger.info("Logger initialized.")
    return logger


def extract_zip_if_needed(zip_path: str, logger: logging.Logger):
    """Extract zip file into `temp_zip_extract/` if it doesn't already exist."""
    extract_dir = Path("temp_zip_extract")
    if not extract_dir.exists():
        logger.info(f"Extracting {zip_path} into {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info("Extraction complete.")
    else:
        logger.info("Extraction directory already exists; skipping extraction.")


def read_name_mapping(name_mapping_path: str, logger: logging.Logger) -> Dict[str, str]:
    """Read the JSON name mapping (key -> subfolder name)."""
    logger.info(f"Reading name mapping from {name_mapping_path}...")
    with open(name_mapping_path, 'r') as f:
        mapping = json.load(f)
    logger.info(f"Loaded name mapping with {len(mapping)} entries.")
    return mapping


def read_participant_csv(csv_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Read the participant CSV into a pandas DataFrame."""
    logger.info(f"Reading participant CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns.")
    return df


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute the IoU of two boolean masks (0 or 1).
    mask1, mask2: shape (H, W).
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def find_first_collision_time(
    folder_path: Path,
    iou_threshold: float,
    logger: logging.Logger,
    fps=60
) -> float:
    """
    For a given subfolder's frames_masks/, compute the earliest frame
    at which IoU >= iou_threshold. Return that collision time in ms.
    If no collision found, return NaN.
    """
    frames_masks_dir = folder_path / "frames_masks"
    blob0_files = sorted(frames_masks_dir.glob("mask_memory_blob_0_frame_*.png"))
    blob1_files = sorted(frames_masks_dir.glob("mask_memory_blob_1_frame_*.png"))

    # Simple approach: rely on parallel indexing.
    # Usually, frame numbers match, but let's use dict for quick access by frame number.
    def extract_frame_idx(fpath: Path) -> int:
        # Example: mask_memory_blob_0_frame_000013.png -> 13
        name = fpath.stem  # e.g. "mask_memory_blob_0_frame_000013"
        parts = name.split("_")
        return int(parts[-1])  # "000013" -> 13

    blob0_dict = {extract_frame_idx(f): f for f in blob0_files}
    blob1_dict = {extract_frame_idx(f): f for f in blob1_files}

    # Start checking from around frame ~13 if desired.
    # We'll just iterate from the min frame onward.
    min_frame = max(13, min(blob0_dict.keys(), default=0), min(blob1_dict.keys(), default=0))
    max_frame = min(max(blob0_dict.keys(), default=0), max(blob1_dict.keys(), default=0))

    for frame_idx in range(min_frame, max_frame + 1):
        if frame_idx in blob0_dict and frame_idx in blob1_dict:
            mask0 = plt.imread(blob0_dict[frame_idx])
            mask1 = plt.imread(blob1_dict[frame_idx])

            # Convert to boolean: If pixel > 0, consider it inside the mask
            mask0_bool = mask0 > 0.5
            mask1_bool = mask1 > 0.5

            iou_val = compute_iou(mask0_bool, mask1_bool)
            if iou_val >= iou_threshold:
                # Convert frame to ms
                collision_time_ms = (frame_idx / fps) * 1000
                return collision_time_ms

    return float('nan')


def parse_subfolder_name(subfolder_name: str) -> Dict[str, Any]:
    """
    Parse the subfolder name of format:
    variable_pretrained_resnet101-BConcave+AConcave+3500(_flipped)

    Return a dict with:
    {
      'model_name': 'variable_pretrained_resnet101',
      'tokens': ['BConcave', 'AConcave'],
      'ground_truth': 3500
    }

    This is a *basic* parser and may need adjustments depending on your naming.
    """
    # Example: variable_pretrained_resnet101-BConcave+AConcave+3500
    # Split at the first dash
    parts = subfolder_name.split('-', 1)
    model_name = parts[0]
    remainder = parts[1] if len(parts) > 1 else ""

    # Now split remainder by '+'
    tokens = remainder.split('+')

    # The last token presumably ends with the GT time, e.g. "3500" or "3500_flipped"
    last_token = tokens[-1]
    # If it has "_flipped", remove that
    last_token_clean = last_token.replace("_flipped", "")
    try:
        ground_truth = int(last_token_clean)
    except ValueError:
        ground_truth = None  # If, for some reason, not parseable.

    # Remove the last token from the tokens list so we only keep the shape tokens
    tokens = tokens[:-1]  # e.g. ["BConcave", "AConcave"]

    return {
        "model_name": model_name,
        "tokens": tokens,
        "ground_truth": ground_truth
    }


def main():
    args = parse_arguments()
    logger = setup_logging()

    # Step 1: Zip Extraction
    extract_zip_if_needed(args.zip_path, logger)

    # Step 2: Load name mapping, participant CSV
    name_mapping = read_name_mapping(args.name_mapping, logger)
    df = read_participant_csv(args.csv_path, logger)

    # Build a dictionary to quickly map from 'Stimulus/XX.mp4' to the subfolder name
    # name_mapping is something like: { "1": "BConcave+AConcave+3500.mp4", "2": "..." }
    # We want a mapping from e.g. "Stimulus/1.mp4" -> "variable_pretrained_resnet101-BConcave+AConcave+3500"
    # But in your example, "BConcave+AConcave+3500.mp4" might be preceded by "variable_pretrained_resnet101-" in the folder.
    # We'll have to do a second step to identify the real folder name in `videos_processed_copy/`.
    #
    # For simplicity, let's store the "base name" of the folder from the JSON directly,
    # and we will prepend "variable_pretrained_resnet101-" if needed.
    # If your JSON already includes the entire folder name, you can skip some logic.

    # We assume that the JSON values are partial folder names, e.g. "BConcave+AConcave+3500.mp4".
    # Then the actual extracted folder might be:
    #   "variable_pretrained_resnet101-BConcave+AConcave+3500" (or ..._flipped).
    # We can handle that by listing all subfolders, matching patterns, etc.

    # For quick matching, let's read all subfolders in videos_processed_copy.
    videos_processed_dir = Path("temp_zip_extract") / "videos_processed_copy"
    if not videos_processed_dir.exists():
        logger.error(f"{videos_processed_dir} does not exist. Aborting.")
        sys.exit(1)

    subfolders = [f for f in videos_processed_dir.iterdir() if f.is_dir()]

    # Create a dictionary: "BConcave+AConcave+3500" -> actual subfolder path
    # We'll do naive matching (the left part is in the subfolder name).
    # If there's a _flipped variant, that also might match.
    # We'll store them in a dict for easy retrieval.
    folder_map = {}
    for sub in subfolders:
        # remove leading "variable_pretrained_resnet101-"
        # or any prefix up to the first '-'
        sub_name = sub.name
        if '-' in sub_name:
            after_dash = sub_name.split('-', 1)[1]  # e.g. "BConcave+AConcave+3500(_flipped)"
        else:
            after_dash = sub_name

        # Also remove the trailing .mp4 from the JSON if needed
        # We'll remove the ".mp4" part from the JSON mapping to match subfolder.
        # Because name_mapping values are e.g. "BConcave+AConcave+3500.mp4".
        # We will store "BConcave+AConcave+3500" or "BConcave+AConcave+3500_flipped".
        folder_map[sub_name] = after_dash

    # Next, let's create a function that, given "Stimulus/X.mp4", returns
    # the actual subfolder path or None if not found.
    def get_subfolder_for_stimulus(stimulus: str) -> Path:
        # Stimulus is e.g. "Stimulus/1.mp4".
        # We want to find name_mapping["1"] -> "BConcave+AConcave+3500.mp4"
        # Then we remove ".mp4" => "BConcave+AConcave+3500"
        # Then find subfolder that contains "BConcave+AConcave+3500" in its (after-dash) name.

        # Extract "1" from "Stimulus/1.mp4"
        base_stim = os.path.basename(stimulus)  # "1.mp4"
        stim_id = os.path.splitext(base_stim)[0]  # "1"

        if stim_id not in name_mapping:
            return None

        raw_folder_name = name_mapping[stim_id]  # e.g. "BConcave+AConcave+3500.mp4"
        folder_key = os.path.splitext(raw_folder_name)[0]  # "BConcave+AConcave+3500"

        # Now find the actual subfolder that ends with or contains this folder_key
        for sub in subfolders:
            if folder_key in folder_map[sub.name]:
                return sub
        return None

    # Step 2 continued: Collision detection for each IoU
    iou_values = np.arange(args.iou_start, args.iou_end + args.iou_step, args.iou_step)
    iou_values = np.round(iou_values, decimals=3)  # round for safer file/folder naming

    # We'll store collision times in a data structure:
    # collisions[(subfolder_name, iou_threshold)] = collision_time_ms
    collisions = {}

    logger.info("Starting collision detection across subfolders and IoU thresholds...")

    for sub in subfolders:
        sub_info = parse_subfolder_name(sub.name)
        for iou_thr in iou_values:
            iou_key = (sub.name, iou_thr)
            collision_time = find_first_collision_time(sub, iou_thr, logger)
            collisions[iou_key] = collision_time

            # Write the iou_{THRESHOLD}.json in the same subfolder
            # e.g. iou_0.05.json
            output_json_path = sub / f"iou_{iou_thr}.json"
            with open(output_json_path, 'w') as fjson:
                json.dump({"collision_time": collision_time}, fjson, indent=2)

    logger.info("Collision detection complete.")

    # Step 3: Create output directories per IoU threshold for each model
    # Example: variable_pretrained_resnet101_IoU_0.05/
    # We'll create these as needed, then do analysis inside them.

    # Let's group subfolders by model_name
    model_to_subfolders = {}
    for sub in subfolders:
        sub_info = parse_subfolder_name(sub.name)
        model_name = sub_info["model_name"]
        model_to_subfolders.setdefault(model_name, []).append(sub)

    # Step 4: Participant Analysis
    # For each model and each IoU threshold, we want:
    #   model_name_IoU_{thr}/
    #       ID/
    #         ID.png, ID.json
    #       Average_person/
    #         average_person.png, average_person.json
    #         concave_vs_convex.png  (Step 5)
    #

    # We'll define a function to get the collision time from collisions dict
    def get_collision_time(subfolder_name: str, iou_thr: float) -> float:
        return collisions.get((subfolder_name, iou_thr), float('nan'))

    # Helper: compute correlation
    def compute_correlation(xvals: List[float], yvals: List[float]) -> float:
        """
        Return Pearson correlation. If insufficient data, return NaN.
        """
        if len(xvals) < 2:
            return float('nan')
        # We can use numpy.corrcoef or e.g. from scipy.stats import pearsonr
        r = np.corrcoef(xvals, yvals)[0, 1]
        return float(r)

    # We'll do all IoU thresholds. For each model_name, we'll create subfolders for results.
    for model_name, subs in model_to_subfolders.items():
        for iou_thr in iou_values:
            # Create output directory for this IoU
            out_dir_name = f"{model_name}_IoU_{iou_thr}"
            out_dir = Path(out_dir_name)
            out_dir.mkdir(exist_ok=True)

            # Also subfolders: ID/ and Average_person/ and concave_vs_convex/
            out_dir_id = out_dir / "ID"
            out_dir_id.mkdir(exist_ok=True)
            out_dir_avg = out_dir / "Average_person"
            out_dir_avg.mkdir(exist_ok=True)
            out_dir_cc = out_dir / "concave_vs_convex"
            out_dir_cc.mkdir(exist_ok=True)

            # -- 4a: Individual Analysis
            # We'll group df by participant ID
            participant_groups = df.groupby("ID")

            for pid, group in participant_groups:
                # group is the subset of df for that participant
                # Identify videos for which we have subfolders
                # We'll build xvals (model predicted) vs yvals (human RT).
                predicted_times = []
                human_times = []
                used_videos = []

                for idx, row in group.iterrows():
                    stim = row["stimulus"]  # e.g. "Stimulus/1.mp4"
                    # find subfolder
                    subpath = get_subfolder_for_stimulus(stim)
                    if subpath is not None:
                        # get predicted collision time
                        ctime = get_collision_time(subpath.name, iou_thr)
                        if not np.isnan(ctime):
                            predicted_times.append(ctime)
                            human_times.append(row["rt"])
                            used_videos.append(subpath.name)

                # If we have data, compute correlation and scatter plot
                if len(predicted_times) > 1:
                    r_val = compute_correlation(predicted_times, human_times)
                else:
                    r_val = float('nan')

                # Plot scatter
                fig, ax = plt.subplots()
                ax.scatter(human_times, predicted_times, c='blue', alpha=0.6)
                ax.set_xlabel("Human RT (ms)")
                ax.set_ylabel("Model Collision Time (ms)")
                ax.set_title(f"Participant {pid}, IoU={iou_thr}, r={r_val:.3f}")

                # Save figure
                fig_out = out_dir_id / f"{pid}.png"
                plt.savefig(fig_out)
                plt.close(fig)

                # Save JSON
                json_out = out_dir_id / f"{pid}.json"
                with open(json_out, 'w') as jf:
                    json.dump({
                        "correlation": r_val,
                        "videos_used": used_videos
                    }, jf, indent=2)

            # -- 4b: Average Participant Analysis
            # For the "average person," first figure out which subfolders exist in this model
            # 'subs' is the list of subfolders for this model.
            # We'll map subfolder -> collision_time
            subfolder_to_ctime = {}
            subfolder_names = [s.name for s in subs]
            for sname in subfolder_names:
                subfolder_to_ctime[sname] = get_collision_time(sname, iou_thr)

            # Now, for each subfolder, find all participants who responded to that video
            # and compute the average RT.
            # We need to map subfolder -> "Stimulus/X.mp4" (via name_mapping).
            # But we actually do the reverse: for each row in df, get subfolder.
            # We'll store them in {subfolder: [rts]}
            subfolder_rts = {sname: [] for sname in subfolder_names}

            for idx, row in df.iterrows():
                stim = row["stimulus"]
                subpath = get_subfolder_for_stimulus(stim)
                if subpath is not None:
                    if subpath.name in subfolder_rts:
                        subfolder_rts[subpath.name].append(row["rt"])

            # compute average RT
            avg_human_times = []
            model_times = []
            used_subfolders = []

            for sname in subfolder_names:
                rts = subfolder_rts[sname]
                if len(rts) > 0:
                    avg_rt = np.mean(rts)
                    pred_ctime = subfolder_to_ctime[sname]
                    if not np.isnan(pred_ctime):
                        avg_human_times.append(avg_rt)
                        model_times.append(pred_ctime)
                        used_subfolders.append(sname)

            # correlation
            if len(avg_human_times) > 1:
                r_val_avg = compute_correlation(avg_human_times, model_times)
            else:
                r_val_avg = float('nan')

            # scatter plot for average
            fig, ax = plt.subplots()
            ax.scatter(avg_human_times, model_times, c='red', alpha=0.7)
            ax.set_xlabel("Average Human RT (ms)")
            ax.set_ylabel("Model Collision Time (ms)")
            ax.set_title(f"Average Person, {model_name}, IoU={iou_thr}, r={r_val_avg:.3f}")
            fig_out = out_dir_avg / "average_person.png"
            plt.savefig(fig_out)
            plt.close(fig)

            # save correlation JSON
            json_out = out_dir_avg / "average_person.json"
            with open(json_out, 'w') as jf:
                json.dump({
                    "correlation": r_val_avg,
                    "videos_used": used_subfolders
                }, jf, indent=2)

            # Step 5: Concave vs Convex Reaction-Time Difference
            # We'll produce a single figure 'concave_vs_convex.png' in out_dir_cc
            #
            # Model analysis: group subfolders by ground_truth => separate concave vs convex => average predicted
            # We'll store differences in a dict: ground_truth -> (concave_mean, convex_mean)
            # We'll define a helper to see if a token is concave or convex (by your convention)
            def is_concave_token(token: str) -> bool:
                # e.g. "AConcave" or "AConvex"
                return "Concave" in token

            # Group subfolders by ground_truth
            gt_to_concave_vals = {}
            gt_to_convex_vals = {}

            # We'll do the same for human data
            # We'll group the CSV by groundTruth, then is_concave
            # Then compute average RT in each group.
            # But we only consider groundTruth that we actually have in subfolders for this model.

            # Model data first
            for s in subs:
                info = parse_subfolder_name(s.name)
                gt = info["ground_truth"]
                tokens = info["tokens"]  # e.g. ["BConcave", "AConcave"]
                # We'll check the *second* token or "the token after the first +",
                # but the specification might vary. Adjust as needed.
                # This example says: "Identify if the subfolder is concave or convex by checking
                # the token after the first '+' in the subfolder name"
                # In the parse_subfolder_name, tokens is everything after the dash, minus the ground_truth token.
                # e.g. "BConcave", "AConcave"
                # Typically, you might define that the second token is the one that matters.
                if len(tokens) >= 2:
                    shape_token = tokens[1]  # e.g. "AConcave" or "AConvex"
                else:
                    shape_token = tokens[0]

                ctime = collisions.get((s.name, iou_thr), float('nan'))
                if not np.isnan(ctime) and gt is not None:
                    if is_concave_token(shape_token):
                        gt_to_concave_vals.setdefault(gt, []).append(ctime)
                    else:
                        gt_to_convex_vals.setdefault(gt, []).append(ctime)

            # For each GT, compute mean concave vs mean convex
            gt_sorted = sorted(set(list(gt_to_concave_vals.keys()) + list(gt_to_convex_vals.keys())))
            model_diffs = []
            for gt in gt_sorted:
                cvals = gt_to_concave_vals.get(gt, [])
                xvals = gt_to_convex_vals.get(gt, [])
                if len(cvals) > 0:
                    mean_c = np.mean(cvals)
                else:
                    mean_c = float('nan')
                if len(xvals) > 0:
                    mean_x = np.mean(xvals)
                else:
                    mean_x = float('nan')
                diff = abs(mean_c - mean_x)
                model_diffs.append(diff)

            # Human data: group by groundTruth => separate is_concave = 1 vs 0
            # We'll store differences similarly
            df_grouped = df.groupby("groundTruth")
            human_diffs = []
            for gt in gt_sorted:
                if gt in df_grouped.groups:
                    subdf = df_grouped.get_group(gt)
                    # separate concave vs convex
                    cdf = subdf[subdf["is_concave"] == 1]
                    xdf = subdf[subdf["is_concave"] == 0]
                    if len(cdf) > 0:
                        mean_c = cdf["rt"].mean()
                    else:
                        mean_c = float('nan')
                    if len(xdf) > 0:
                        mean_x = xdf["rt"].mean()
                    else:
                        mean_x = float('nan')
                    diff = abs(mean_c - mean_x)
                else:
                    diff = float('nan')
                human_diffs.append(diff)

            # Plot side-by-side box or bar for each groundTruth
            # Here we'll just do a bar for model vs human difference on the same axis
            # x-axis enumerates GT times
            x_indices = np.arange(len(gt_sorted))
            width = 0.3

            fig, ax = plt.subplots(figsize=(8, 5))
            b1 = ax.bar(x_indices - width/2, model_diffs, width, label='Model')
            b2 = ax.bar(x_indices + width/2, human_diffs, width, label='Human')
            ax.set_xticks(x_indices)
            ax.set_xticklabels([str(g) for g in gt_sorted])
            ax.set_ylabel("Concave vs Convex (Absolute Difference in ms)")
            ax.set_title(f"Concave vs Convex Differences, IoU={iou_thr}")
            ax.legend()

            fig_out = out_dir_cc / "concave_vs_convex.png"
            plt.tight_layout()
            plt.savefig(fig_out)
            plt.close(fig)

            logger.info(f"Done creating results for {model_name}, IoU={iou_thr}")

    logger.info("All analyses complete. Exiting.")


if __name__ == "__main__":
    main()
