#!/usr/bin/env python3
"""
Script to:
1) For each IoU threshold from iou_start..iou_end in steps of iou_step:
   - If an output folder for that IoU already exists (e.g. modelName_0.05), skip it.
   - Otherwise:
     a) For each subfolder in temp_zip_extract/videos_processed_copy/:
        - Find frames in frames_masks/ from frame >=14 onwards.
        - Compute the first frame where IoU >= threshold.
        - Convert to ms and store in iou_{threshold}.json in that same subfolder.
     b) Using those collision times, generate:
        - Per-participant plots (scatter x=rt, y=predicted-rt).
        - "Average person" scatter (all videos in the .zip).
        - Concave vs. Convex absolute difference box-plot (human vs. model).
"""

import argparse
import logging
import datetime
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image
from scipy.stats import pearsonr
from collections import defaultdict

# -------------------------------------------------------------
# Utility: create directory if it doesn't exist
# -------------------------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# -------------------------------------------------------------
# Compute IoU from two binary mask arrays
# -------------------------------------------------------------
def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def main():
    parser = argparse.ArgumentParser(description="Compute collision times and generate plots.")
    parser.add_argument(
        "--extracted_dir",
        type=str,
        default="temp_zip_extract/videos_processed_copy",
        help="Path to the extracted 'videos_processed_copy' directory."
    )
    parser.add_argument(
        "--name_mapping_path",
        type=str,
        default="name_mapping.json",
        help="Path to the name_mapping.json file."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="responses.csv",
        help="Path to the CSV with participant data."
    )
    parser.add_argument(
        "--iou_start",
        type=float,
        default=0.05,
        help="Starting IoU threshold (inclusive)."
    )
    parser.add_argument(
        "--iou_end",
        type=float,
        default=0.95,
        help="Ending IoU threshold (inclusive)."
    )
    parser.add_argument(
        "--iou_step",
        type=float,
        default=0.05,
        help="Step size for IoU thresholds."
    )
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Setup logging
    # ---------------------------------------------------------
    ensure_dir("logs")
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join("logs", f"log_{time_str}.txt")
    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logging.info("Script started.")
    logging.info("Reading command-line arguments...")

    extracted_dir = args.extracted_dir
    name_mapping_path = args.name_mapping_path
    csv_path = args.csv_path
    iou_thresholds = np.arange(args.iou_start, args.iou_end + 1e-9, args.iou_step)

    logging.info(f"extracted_dir: {extracted_dir}")
    logging.info(f"name_mapping_path: {name_mapping_path}")
    logging.info(f"csv_path: {csv_path}")
    logging.info(f"IoU thresholds: {iou_thresholds}")

    # ---------------------------------------------------------
    # Load CSV
    # ---------------------------------------------------------
    if not os.path.isfile(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    required_cols = ["rt", "groundTruth", "stimulus", "ID", "is_concave"]
    for col in required_cols:
        if col not in df.columns:
            logging.error(f"CSV missing required column: {col}")
            return
    logging.info(f"CSV loaded. Shape = {df.shape}")

    # ---------------------------------------------------------
    # Load name_mapping.json
    # ---------------------------------------------------------
    if not os.path.isfile(name_mapping_path):
        logging.error(f"name_mapping.json not found: {name_mapping_path}")
        return
    with open(name_mapping_path, "r") as f:
        name_map = json.load(f)

    # Invert name_map: e.g. "BConcave+AConcave+3500.mp4" -> "1"
    inverted_map = {}
    for k, v in name_map.items():
        inverted_map[v] = k

    # ---------------------------------------------------------
    # Identify all subfolders in extracted_dir
    # e.g. variable_pretrained_resnet101-BConcave+AConcave+3500
    # We'll parse the model_name and shapes_str from each.
    # ---------------------------------------------------------
    if not os.path.isdir(extracted_dir):
        logging.error(f"Extracted directory not found: {extracted_dir}")
        return

    all_subfolders = []
    for entry in os.scandir(extracted_dir):
        if entry.is_dir():
            # e.g. entry.name = "variable_pretrained_resnet101-BConcave+AConcave+3500"
            all_subfolders.append(entry.path)

    logging.info(f"Found {len(all_subfolders)} subfolders in {extracted_dir}.")

    # Helper to parse model_name, shapes_str from folder name
    def parse_folder(folder_name: str):
        # e.g. folder_name = "variable_pretrained_resnet101-BConcave+AConcave+3500"
        if "-" not in folder_name:
            return None, None
        idx = folder_name.index("-")
        model_name = folder_name[:idx]
        shapes_str = folder_name[idx+1:]
        return model_name, shapes_str

    # For each subfolder, also check if frames_masks/ exists
    # We'll store a list of dict with:
    # {
    #   "path": full_path_to_subfolder,
    #   "model_name": ...,
    #   "shapes_str": ...,
    #   "frames_masks_path": ...
    # }
    folder_info_list = []
    for sf in all_subfolders:
        subfolder_name = os.path.basename(sf)  # e.g. "variable_pretrained_resnet101-BConcave+AConcave+3500"
        model_name, shapes_str = parse_folder(subfolder_name)
        if not model_name or not shapes_str:
            logging.warning(f"Skipping folder without '-' in name: {subfolder_name}")
            continue

        frames_masks_path = os.path.join(sf, "frames_masks")
        if not os.path.isdir(frames_masks_path):
            logging.warning(f"Skipping folder without frames_masks/: {sf}")
            continue

        folder_info_list.append({
            "path": sf,
            "model_name": model_name,
            "shapes_str": shapes_str,
            "frames_masks_path": frames_masks_path
        })

    logging.info(f"Final relevant subfolders with frames_masks: {len(folder_info_list)}")

    # Precompile regex for mask filenames
    # e.g. mask_memory_blob_0_frame_000014.png
    filename_pattern = re.compile(r"mask_memory_blob_(\d+)_frame_(\d+)\.png")

    # ---------------------------------------------------------
    # For each IoU threshold, skip if an output folder "model_name_iou" exists
    # Otherwise, compute + store collision times, then do all plots
    # ---------------------------------------------------------
    for iou_thr in iou_thresholds:
        iou_str = round(iou_thr, 2)
        logging.info(f"\n===== IoU threshold = {iou_str} =====")

        # We want to group subfolders by model_name to produce an output folder
        # named e.g. model_name_{iou_thr}. We'll process them model by model.
        # Let's find all distinct model_names in folder_info_list
        model_names = set([fi["model_name"] for fi in folder_info_list])

        for mname in model_names:
            out_dir = f"{mname}_{iou_str}"
            if os.path.isdir(out_dir):
                # If this directory already exists, skip as requested
                logging.info(f"Output folder {out_dir} already exists -> Skipping IoU {iou_str} for model {mname}")
                continue

            logging.info(f"Processing IoU {iou_str} for model {mname} -> output to {out_dir}")
            ensure_dir(out_dir)

            # 1) For each subfolder that has model_name == mname:
            #    - compute or load collision_time for IoU = iou_thr
            for folder_info in folder_info_list:
                if folder_info["model_name"] != mname:
                    continue
                subf_path = folder_info["path"]
                shapes_str = folder_info["shapes_str"]
                frames_dir = folder_info["frames_masks_path"]

                # We'll write collision time to e.g. iou_0.05.json in subfolder
                collision_json_path = os.path.join(subf_path, f"iou_{iou_str}.json")
                if os.path.isfile(collision_json_path):
                    # Already computed
                    logging.info(f"Skipping collision-time compute: already exists {collision_json_path}")
                    continue

                # Otherwise, compute
                # We'll gather blob0_masks[frame_idx], blob1_masks[frame_idx]
                blob0_masks = {}
                blob1_masks = {}

                # List PNG files in frames_dir
                possible_files = os.listdir(frames_dir)
                for fname in possible_files:
                    match = filename_pattern.match(fname)
                    if not match:
                        continue
                    blob_idx_str, frame_idx_str = match.groups()
                    blob_idx = int(blob_idx_str)
                    frame_idx = int(frame_idx_str)
                    # Only consider frames >= 14
                    if frame_idx < 14:
                        continue

                    full_path = os.path.join(frames_dir, fname)
                    im = Image.open(full_path).convert("L")
                    mask_arr = (np.array(im) > 0)
                    if blob_idx == 0:
                        blob0_masks[frame_idx] = mask_arr
                    else:
                        blob1_masks[frame_idx] = mask_arr

                common_frames = sorted(set(blob0_masks.keys()) & set(blob1_masks.keys()))
                collision_time = None
                for frame_idx in common_frames:
                    iou_val = compute_iou(blob0_masks[frame_idx], blob1_masks[frame_idx])
                    if iou_val >= iou_thr:
                        # first collision
                        collision_time = (frame_idx / 60.0) * 1000.0
                        break

                # Save JSON
                with open(collision_json_path, "w") as jf:
                    json.dump({"collision_time": collision_time}, jf, indent=4)
                logging.info(f"Saved collision_time for {os.path.basename(subf_path)} @ IoU={iou_str}: {collision_time}")

            # 2) Having all collision times in place, produce the usual plots:
            #    - Per-participant scatter
            #    - Average-person scatter (all videos in the .zip)
            #    - Concave vs. Convex difference box-plot

            # Gather list of subfolders for this model_name
            model_subfolders = [fi for fi in folder_info_list if fi["model_name"] == mname]

            # Map shapes_str -> collision_time (for the current iou_thr)
            # We'll parse each subfolder for iou_{iou_str}.json
            collisions = {}
            shapes_to_stimulus_map = {}
            for fi in model_subfolders:
                subf_name = os.path.basename(fi["path"])  # e.g. "variable_pretrained_resnet101-BConcave+AConcave+3500"
                shapes_str = fi["shapes_str"]
                # collision JSON
                collision_json_path = os.path.join(fi["path"], f"iou_{iou_str}.json")
                if not os.path.isfile(collision_json_path):
                    # means no collision_time found or subfolder was skipped
                    collisions[shapes_str] = None
                else:
                    with open(collision_json_path, "r") as jf:
                        dct = json.load(jf)
                    collisions[shapes_str] = dct.get("collision_time", None)

                # Also build shapes_str -> Stimulus mapping
                # shapes_str e.g. "BConcave+AConcave+3500" => "BConcave+AConcave+3500.mp4"
                mp4_name = shapes_str + ".mp4"
                if mp4_name in inverted_map:
                    # e.g. "BConcave+AConcave+3500.mp4" -> "1"
                    # => "Stimulus/1.mp4"
                    shapes_to_stimulus_map[shapes_str] = f"Stimulus/{inverted_map[mp4_name]}.mp4"
                else:
                    shapes_to_stimulus_map[shapes_str] = None

            # == Per-participant scatter plots ==
            participants = df["ID"].unique()
            for pid in participants:
                dfp = df[df["ID"] == pid].copy()
                valid_rows = []
                pred_times = []
                rts = []
                used_videos = []

                for fi in model_subfolders:
                    sstr = fi["shapes_str"]
                    stim_key = shapes_to_stimulus_map[sstr]
                    if not stim_key:
                        continue
                    df_match = dfp[dfp["stimulus"] == stim_key]
                    if len(df_match) > 0:
                        # predicted collision time
                        ctime = collisions[sstr]
                        if ctime is not None:
                            for _, row in df_match.iterrows():
                                valid_rows.append(row)
                                pred_times.append(ctime)
                                rts.append(row["rt"])
                                used_videos.append(stim_key)

                if len(valid_rows) == 0:
                    continue

                # Plot
                fig = plt.figure(figsize=(6, 6))
                plt.scatter(rts, pred_times, c='blue', alpha=0.7)
                plt.xlabel("Human RT (ms)")
                plt.ylabel("Predicted Collision (ms)")
                plt.title(f"Participant {pid} - IoU {iou_str}")
                plt.tight_layout()

                pid_dir = os.path.join(out_dir, str(pid))
                ensure_dir(pid_dir)
                out_plot_path = os.path.join(pid_dir, f"{pid}.png")
                plt.savefig(out_plot_path)
                plt.close(fig)

                # Correlation
                if len(rts) > 1:
                    corr_val, _ = pearsonr(rts, pred_times)
                else:
                    corr_val = float('nan')

                # Save ID.json
                out_json_path = os.path.join(pid_dir, f"{pid}.json")
                data_to_save = {
                    "correlation": float(corr_val),
                    "videos_used": list(set(used_videos))
                }
                with open(out_json_path, "w") as jf:
                    json.dump(data_to_save, jf, indent=4)

            # == Average-person scatter (all .zip videos) ==
            # For each shapes_str in collisions, if collisions[sstr] is not None,
            # we gather all participants who saw that video and compute average RT
            shape_to_rts = defaultdict(list)
            for sstr in collisions.keys():
                stim_key = shapes_to_stimulus_map[sstr]
                if stim_key is None:
                    continue
                df_stim = df[df["stimulus"] == stim_key]
                if len(df_stim) == 0:
                    continue
                shape_to_rts[sstr] = list(df_stim["rt"].values)

            shape_to_avg_rt = {}
            for sstr, rts_ in shape_to_rts.items():
                if len(rts_) > 0:
                    shape_to_avg_rt[sstr] = float(np.mean(rts_))
                else:
                    shape_to_avg_rt[sstr] = None

            # Build final arrays: (human average, predicted collision)
            final_human = []
            final_model = []
            final_sstrs = []
            for sstr in collisions.keys():
                ctime = collisions[sstr]
                if ctime is not None and (sstr in shape_to_avg_rt) and (shape_to_avg_rt[sstr] is not None):
                    final_human.append(shape_to_avg_rt[sstr])
                    final_model.append(ctime)
                    final_sstrs.append(sstr)

            if len(final_sstrs) > 0:
                fig = plt.figure(figsize=(6, 6))
                plt.scatter(final_human, final_model, c='green', alpha=0.7)
                plt.xlabel("Average Human RT (ms)")
                plt.ylabel("Predicted Collision (ms)")
                plt.title(f"Average Person - IoU {iou_str} [{mname}]")
                plt.tight_layout()

                avg_dir = os.path.join(out_dir, "Average_person")
                ensure_dir(avg_dir)
                avg_plot_path = os.path.join(avg_dir, "average_person.png")
                plt.savefig(avg_plot_path)
                plt.close(fig)

                if len(final_human) > 1:
                    corr_val, _ = pearsonr(final_human, final_model)
                else:
                    corr_val = float('nan')

                out_json_path = os.path.join(avg_dir, "average_person.json")
                data_to_save = {
                    "correlation": float(corr_val),
                    "num_videos_used": len(final_sstrs)
                }
                with open(out_json_path, "w") as jf:
                    json.dump(data_to_save, jf, indent=4)

            # == Concave vs. Convex difference box-plot ==
            #
            #   1) Extract groundTruth from subfolder name: last token after '+', e.g. 3500
            #   2) Identify if subfolder is concave or convex by the token after the first '+'
            #      e.g. "BConcave+AConcave+3500" => second token is "AConcave" => concave.
            #   3) For each groundTruth, gather all concave subfolders' predicted times
            #      and all convex subfolders' predicted times.
            #      Then compute the average of each group => take absolute difference.
            #   4) For humans, do the same: separate by is_concave=1 or 0, average => abs diff.
            #   5) Make box-plot side by side for each groundTruth.
            #

            # Build subfolder->(groundTruth, isConcave) map
            # from shapes_str = e.g. "BConcave+AConcave+3500"
            folder_meta = {}
            for fi in model_subfolders:
                sstr = fi["shapes_str"]
                parts = sstr.split("+")
                if len(parts) != 3:
                    continue
                gt_str = parts[2]  # e.g. "3500"
                try:
                    gt_val = int(gt_str)
                except:
                    gt_val = None
                second_shape_str = parts[1]  # e.g. "AConcave" or "AConvex"
                is_second_concave = second_shape_str.endswith("Concave")
                folder_meta[sstr] = {
                    "groundTruth": gt_val,
                    "is_second_concave": is_second_concave
                }

            # For the model: groundTruth -> {True: [...], False: [...]} for (concave/convex)
            # We'll fill it with predicted collision times. Then we average + do abs diff.
            model_dict = defaultdict(lambda: defaultdict(list))

            for sstr, col_time in collisions.items():
                if sstr not in folder_meta:
                    continue
                if col_time is None:
                    continue  # no collision or skip
                gt_val = folder_meta[sstr]["groundTruth"]
                is_conc = folder_meta[sstr]["is_second_concave"]
                model_dict[gt_val][is_conc].append(col_time)

            # For humans: groundTruth -> {True: [...], False: [...]} for (concave/convex)
            # We'll gather from the CSV's is_concave column, but only for stimuli that are in shapes_to_stimulus_map.
            human_dict = defaultdict(lambda: defaultdict(list))

            # Build a reverse map: Stimulus -> (gt_val, is_second_concave)
            stim_to_gt_and_conc = {}
            for sstr in folder_meta:
                stim_key = shapes_to_stimulus_map[sstr]
                if stim_key is not None:
                    gt_val = folder_meta[sstr]["groundTruth"]
                    is_conc = folder_meta[sstr]["is_second_concave"]
                    stim_to_gt_and_conc[stim_key] = (gt_val, is_conc)

            df_in_zip = df[df["stimulus"].isin(stim_to_gt_and_conc.keys())]
            for idx, row in df_in_zip.iterrows():
                stim_key = row["stimulus"]
                gt_val, _ = stim_to_gt_and_conc[stim_key]
                # For humans, we rely on row["is_concave"]: 1 => concave, 0 => convex
                is_c = (row["is_concave"] == 1)
                human_dict[gt_val][is_c].append(row["rt"])

            # Now compute absolute difference for each groundTruth
            model_diff_by_gt = defaultdict(list)
            human_diff_by_gt = defaultdict(list)

            for gt_val, cc_dict in model_dict.items():
                conc_list = cc_dict[True]
                conv_list = cc_dict[False]
                if len(conc_list) > 0 and len(conv_list) > 0:
                    mean_conc = np.mean(conc_list)
                    mean_conv = np.mean(conv_list)
                    diff_abs = abs(mean_conc - mean_conv)
                    model_diff_by_gt[gt_val].append(diff_abs)

            for gt_val, cc_dict in human_dict.items():
                conc_list = cc_dict[True]
                conv_list = cc_dict[False]
                if len(conc_list) > 0 and len(conv_list) > 0:
                    mean_conc = np.mean(conc_list)
                    mean_conv = np.mean(conv_list)
                    diff_abs = abs(mean_conc - mean_conv)
                    human_diff_by_gt[gt_val].append(diff_abs)

            all_gts = set(list(model_diff_by_gt.keys()) + list(human_diff_by_gt.keys()))
            sorted_gts = sorted(all_gts)

            if len(sorted_gts) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                positions_humans = []
                positions_model = []
                data_humans = []
                data_model = []

                for i, gt in enumerate(sorted_gts):
                    hd = human_diff_by_gt[gt] if gt in human_diff_by_gt else []
                    md = model_diff_by_gt[gt] if gt in model_diff_by_gt else []
                    base_x = i * 2.0
                    positions_humans.append(base_x)
                    positions_model.append(base_x + 0.8)
                    data_humans.append(hd)
                    data_model.append(md)

                bph = ax.boxplot(
                    data_humans,
                    positions=positions_humans,
                    widths=0.6,
                    patch_artist=True
                )
                bpm = ax.boxplot(
                    data_model,
                    positions=positions_model,
                    widths=0.6,
                    patch_artist=True
                )

                for patch in bph['boxes']:
                    patch.set_facecolor("lightblue")
                for patch in bpm['boxes']:
                    patch.set_facecolor("lightgreen")

                ax.set_xticks(positions_humans)
                ax.set_xticklabels([str(gt) for gt in sorted_gts])
                ax.set_xlabel("Ground Truth (ms)")
                ax.set_ylabel("Absolute Difference (Concave vs. Convex) (ms)")
                ax.set_title(f"Concave vs. Convex Differences (Abs) - IoU {iou_str}")

                h1, = plt.plot([], [], color="lightblue", label="Humans", marker='s', linestyle='')
                h2, = plt.plot([], [], color="lightgreen", label="Model", marker='s', linestyle='')
                ax.legend(handles=[h1, h2], loc="best")

                plt.tight_layout()
                cvc_dir = os.path.join(out_dir, "Average_person", "concave_vs_convex")
                ensure_dir(cvc_dir)
                cvc_plot_path = os.path.join(cvc_dir, "concave_vs_convex.png")
                plt.savefig(cvc_plot_path)
                plt.close(fig)

    logging.info("All processing complete. Exiting script.")

if __name__ == "__main__":
    main()
