import argparse
import os
import sys
import json
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Any


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Reuse pre-existing iou_{X}.json collision times from 'videos_processed_copy' to perform the same analysis and also produce a second bar plot of concave vs convex average times."
    )
    parser.add_argument(
        "--zip_path",
        required=True,
        help="(Not used, but kept for argument-compatibility) Path to the input .zip file."
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
    parser.add_argument(
        "--tick_size_factor",
        type=float,
        default=1.5,
        help="Factor to adjust tick font size (default is 1.3, meaning 30% larger)."
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


def parse_subfolder_name(subfolder_name: str) -> Dict[str, Any]:
    """
    Parse the subfolder name of format:
    variable_pretrained_resnet101-BConcave+AConcave+3500(_flipped)

    Return:
    {
      'model_name': 'variable_pretrained_resnet101',
      'tokens': ['BConcave', 'AConcave'],
      'ground_truth': 3500
    }
    """
    parts = subfolder_name.split('-', 1)
    model_name = parts[0]
    remainder = parts[1] if len(parts) > 1 else ""

    tokens = remainder.split('+')
    last_token = tokens[-1]
    last_token_clean = last_token.replace("_flipped", "")
    try:
        ground_truth = int(last_token_clean)
    except ValueError:
        ground_truth = None
    tokens = tokens[:-1]

    return {
        "model_name": model_name,
        "tokens": tokens,
        "ground_truth": ground_truth
    }


def main():
    args = parse_arguments()
    logger = setup_logging()

    logger.info("Skipping collision detection; reusing existing iou_{X}.json data only.")

    # Configure tick size scaling factor for all plots
    # Fix for the error: Convert the font size to a number before multiplying
    tick_size_factor = args.tick_size_factor

    # Get current font sizes as numeric values
    xtick_size = plt.rcParams['xtick.labelsize']
    ytick_size = plt.rcParams['ytick.labelsize']

    # Convert to float if they're strings
    if isinstance(xtick_size, str):
        try:
            xtick_size = float(xtick_size)
        except ValueError:
            xtick_size = 10.0  # Default if conversion fails

    if isinstance(ytick_size, str):
        try:
            ytick_size = float(ytick_size)
        except ValueError:
            ytick_size = 10.0  # Default if conversion fails

    # Set the new font sizes
    plt.rc('xtick', labelsize=xtick_size * tick_size_factor)
    plt.rc('ytick', labelsize=ytick_size * tick_size_factor)

    # Step 1: Read name mapping, CSV
    name_mapping = read_name_mapping(args.name_mapping, logger)
    df = read_participant_csv(args.csv_path, logger)

    # Step 2: We assume videos_processed_copy is in temp_zip_extract
    videos_processed_dir = Path("temp_zip_extract") / "videos_processed_copy"
    if not videos_processed_dir.exists():
        logger.error(f"{videos_processed_dir} does not exist. Aborting.")
        sys.exit(1)

    subfolders = [f for f in videos_processed_dir.iterdir() if f.is_dir()]
    if not subfolders:
        logger.error("No subfolders found in videos_processed_copy; aborting.")
        sys.exit(1)

    # Build naive matching
    folder_map = {}
    for sub in subfolders:
        sub_name = sub.name
        if '-' in sub_name:
            after_dash = sub_name.split('-', 1)[1]
        else:
            after_dash = sub_name
        folder_map[sub_name] = after_dash

    def get_subfolder_for_stimulus(stimulus: str) -> Path:
        # "Stimulus/1.mp4" => name_mapping["1"] => "BConcave+AConcave+3500.mp4" => "BConcave+AConcave+3500"
        base_stim = os.path.basename(stimulus)
        stim_id = os.path.splitext(base_stim)[0]
        if stim_id not in name_mapping:
            return None
        raw_folder_name = name_mapping[stim_id]
        folder_key = os.path.splitext(raw_folder_name)[0]
        for sub_ in subfolders:
            if folder_key in folder_map[sub_.name]:
                return sub_
        return None

    # Step 3: Load collision times from existing iou_{thr}.json
    iou_values = np.arange(args.iou_start, args.iou_end + args.iou_step, args.iou_step)
    iou_values = np.round(iou_values, 3)

    collisions = {}
    logger.info("Reading precomputed collision times from iou_{thr}.json files...")

    for sub in subfolders:
        sname = sub.name
        for iou_thr in iou_values:
            iou_json_path = sub / f"iou_{iou_thr}.json"
            if iou_json_path.exists():
                try:
                    with open(iou_json_path, 'r') as jf:
                        data = json.load(jf)
                    collisions[(sname, iou_thr)] = data.get("collision_time", float('nan'))
                except:
                    collisions[(sname, iou_thr)] = float('nan')
            else:
                collisions[(sname, iou_thr)] = float('nan')

    logger.info("Collision time loading complete.")

    # Step 4: Group subfolders by model_name
    model_to_subfolders = {}
    for sub in subfolders:
        info = parse_subfolder_name(sub.name)
        model_name = info["model_name"]
        model_to_subfolders.setdefault(model_name, []).append(sub)

    # Helper to fetch collisions
    def get_collision_time(subfolder_name: str, iou_thr: float) -> float:
        return collisions.get((subfolder_name, iou_thr), float('nan'))

    def compute_correlation(xvals: list, yvals: list) -> float:
        if len(xvals) < 2:
            return float('nan')
        return np.corrcoef(xvals, yvals)[0, 1]

    # Step 4 & 5 & new Step 6: Analysis
    for model_name, subs in model_to_subfolders.items():
        for iou_thr in iou_values:
            out_dir_name = f"{model_name}_IoU_{iou_thr}"
            out_dir = Path(out_dir_name)
            out_dir.mkdir(exist_ok=True)

            out_dir_id = out_dir / "ID"
            out_dir_id.mkdir(exist_ok=True)

            out_dir_avg = out_dir / "Average_person"
            out_dir_avg.mkdir(exist_ok=True)

            out_dir_cc = out_dir / "concave_vs_convex"
            out_dir_cc.mkdir(exist_ok=True)

            # 4a: Individual (Participant) Analysis
            participant_groups = df.groupby("ID")
            for pid, group in participant_groups:
                predicted_times = []
                human_times = []
                used_videos = []

                for idx, row in group.iterrows():
                    stim = row["stimulus"]
                    subpath = get_subfolder_for_stimulus(stim)
                    if subpath is not None:
                        ctime = get_collision_time(subpath.name, iou_thr)
                        if not np.isnan(ctime):
                            predicted_times.append(ctime)
                            human_times.append(row["rt"])
                            used_videos.append(subpath.name)

                if len(predicted_times) > 1:
                    r_val = compute_correlation(predicted_times, human_times)
                else:
                    r_val = float('nan')

                fig, ax = plt.subplots()
                ax.scatter(human_times, predicted_times, c='blue', alpha=0.6)
                ax.set_xlabel("Human RT (ms)")
                ax.set_ylabel("Model Collision Time (ms)")
                ax.set_title(f"Participant {pid}, IoU={iou_thr}, r={r_val:.3f}")

                fig_out = out_dir_id / f"{pid}.png"
                plt.savefig(fig_out)
                plt.close(fig)

                json_out = out_dir_id / f"{pid}.json"
                with open(json_out, 'w') as jf:
                    json.dump({
                        "correlation": r_val,
                        "videos_used": used_videos
                    }, jf, indent=2)

            # 4b: Average Participant Analysis
            subfolder_names = [s.name for s in subs]
            subfolder_to_ctime = {
                sname: get_collision_time(sname, iou_thr)
                for sname in subfolder_names
            }

            subfolder_rts = {sname: [] for sname in subfolder_names}
            for idx, row in df.iterrows():
                stim = row["stimulus"]
                subpath = get_subfolder_for_stimulus(stim)
                if subpath is not None and subpath.name in subfolder_rts:
                    subfolder_rts[subpath.name].append(row["rt"])

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

            if len(avg_human_times) > 1:
                r_val_avg = compute_correlation(avg_human_times, model_times)
            else:
                r_val_avg = float('nan')

            fig, ax = plt.subplots()
            ax.scatter(avg_human_times, model_times, c='red', alpha=0.7)
            ax.set_xlabel("Average Human RT (ms)")
            ax.set_ylabel("Model Collision Time (ms)")
            ax.set_title(f"Average Person, {model_name}, IoU={iou_thr}, r={r_val_avg:.3f}")
            fig_out = out_dir_avg / "average_person.png"
            plt.savefig(fig_out)
            plt.close(fig)

            json_out = out_dir_avg / "average_person.json"
            with open(json_out, 'w') as jf:
                json.dump({
                    "correlation": r_val_avg,
                    "videos_used": used_subfolders
                }, jf, indent=2)

            # Step 5: Concave vs Convex Reaction-Time Difference
            def is_concave_token(token: str) -> bool:
                return "Concave" in token

            gt_to_concave_vals = {}
            gt_to_convex_vals = {}

            for s in subs:
                info = parse_subfolder_name(s.name)
                gt = info["ground_truth"]
                tokens = info["tokens"]
                if len(tokens) >= 2:
                    shape_token = tokens[1]
                else:
                    shape_token = tokens[0]

                ctime = get_collision_time(s.name, iou_thr)
                if not np.isnan(ctime) and gt is not None:
                    if is_concave_token(shape_token):
                        gt_to_concave_vals.setdefault(gt, []).append(ctime)
                    else:
                        gt_to_convex_vals.setdefault(gt, []).append(ctime)

            gt_sorted = sorted(
                set(gt_to_concave_vals.keys()) | set(gt_to_convex_vals.keys())
            )
            # Model diffs
            model_diffs = []
            for gt in gt_sorted:
                cvals = gt_to_concave_vals.get(gt, [])
                xvals = gt_to_convex_vals.get(gt, [])
                mc = np.mean(cvals) if cvals else float('nan')
                mx = np.mean(xvals) if xvals else float('nan')
                model_diffs.append(abs(mc - mx))

            # Human diffs
            df_grouped = df.groupby("groundTruth")
            human_diffs = []
            for gt in gt_sorted:
                if gt in df_grouped.groups:
                    subdf = df_grouped.get_group(gt)
                    cdf = subdf[subdf["is_concave"] == 1]
                    xdf = subdf[subdf["is_concave"] == 0]
                    hc = cdf["rt"].mean() if len(cdf) > 0 else float('nan')
                    hx = xdf["rt"].mean() if len(xdf) > 0 else float('nan')
                    human_diffs.append(abs(hc - hx))
                else:
                    human_diffs.append(float('nan'))

            x_indices = np.arange(len(gt_sorted))
            width = 0.22  # Make bars even narrower

            # Step 5: First plot (differences)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x_indices - width / 2, model_diffs, width, label='Model')
            ax.bar(x_indices + width / 2, human_diffs, width, label='Human')
            ax.set_xticks(x_indices)
            ax.set_xticklabels([str(g) for g in gt_sorted])
            ax.set_ylabel("Concave vs Convex (Absolute Difference in ms)")
            ax.set_title(f"Concave vs Convex Differences, IoU={iou_thr}")
            ax.legend()

            fig_out = out_dir_cc / "concave_vs_convex.png"
            plt.tight_layout()
            plt.savefig(fig_out)
            plt.close(fig)

            # Step 6: Modified two-box plot with bars right next to each other and SEM error bars
            concave_means = []
            concave_sems = []  # Standard error of the mean
            convex_means = []
            convex_sems = []  # Standard error of the mean

            for gt in gt_sorted:
                cvals = gt_to_concave_vals.get(gt, [])
                xvals = gt_to_convex_vals.get(gt, [])

                # Calculate means
                mc = np.mean(cvals) if cvals else float('nan')
                mx = np.mean(xvals) if xvals else float('nan')
                concave_means.append(mc)
                convex_means.append(mx)

                # Calculate standard errors
                c_sem = np.std(cvals, ddof=1) / np.sqrt(len(cvals)) if len(cvals) > 1 else 0
                x_sem = np.std(xvals, ddof=1) / np.sqrt(len(xvals)) if len(xvals) > 1 else 0
                concave_sems.append(c_sem)
                convex_sems.append(x_sem)

            # Create the two-box plot with requested changes
            fig2, ax2 = plt.subplots(figsize=(10, 6))

            # Define parameters for the grouped bar chart
            bar_width = 0.40  # Modified: Increased from 0.25 to make bars wider
            # Adjust spacing between groups by modifying the x_positions calculation
            x_positions = np.arange(
                len(gt_sorted)) * 1.0  # Modified: Reduced spacing factor from 1.1 to 1.0 (~10% reduction)

            # Place bars directly next to each other (no gap)
            concave_bars = ax2.bar(x_positions - bar_width / 2, concave_means, bar_width,
                                   color='#FFBE48', label='Concave')
            convex_bars = ax2.bar(x_positions + bar_width / 2, convex_means, bar_width,
                                  color='#56A036', label='Convex')

            # Add error bars on TOP of each bar
            ax2.errorbar(x_positions - bar_width / 2, concave_means, yerr=concave_sems,
                         fmt='none', ecolor='black', capsize=3)
            ax2.errorbar(x_positions + bar_width / 2, convex_means, yerr=convex_sems,
                         fmt='none', ecolor='black', capsize=3)

            # Set x and y axis labels as requested
            ax2.set_xlabel("Ground Truth Time-to-Collision (ms)")
            ax2.set_ylabel("Model Time-to-Collision (ms)")
            ax2.set_title(f"Concave vs Convex Collision Times, IoU={iou_thr}")

            # Customize x-axis to reduce space between units
            ax2.set_xticks(x_positions)
            ax2.set_xticklabels([str(g) for g in gt_sorted])

            # MODIFICATION 2: Reduce space between x-axis entries by making figure more compact horizontally
            # Set a more compact figure width (60% of the original width)
            fig2.set_size_inches(6, fig2.get_figheight())

            # MODIFICATION 1: Set a better y-axis range - calculate based on actual data
            # Find maximum data point value (including error bars)
            max_data_value = max([
                max([c + e for c, e in zip(concave_means, concave_sems) if not np.isnan(c)], default=0),
                max([c + e for c, e in zip(convex_means, convex_sems) if not np.isnan(c)], default=0)
            ])
            # Set y-axis min/max with a reasonable buffer (200ms above highest point)
            y_min = 3400  # A bit lower than 3500
            y_max = min(5500, max_data_value + 200)  # Cap at 5500 unless data exceeds it
            ax2.set_ylim(y_min, y_max)

            # Reduce space between y-axis ticks
            ax2.yaxis.set_major_locator(plt.MaxNLocator(10))  # Increase number of ticks to reduce space

            # Add a grid for better readability
            ax2.grid(axis='y', linestyle='--', alpha=0.3)

            # Add legend
            ax2.legend()

            # Adjust figure to reduce spacing
            plt.tight_layout()

            # Save figure
            fig_out2 = out_dir_cc / "concave_vs_convex_two_box.png"
            plt.savefig(fig_out2)
            plt.close(fig2)

            # Create a third visualization (line plot) with modified axis labels
            fig3, ax3 = plt.subplots(figsize=(10, 6))

            # Create line plots
            ax3.plot(gt_sorted, concave_means, 'o-', color='gold', label='Concave', linewidth=2, markersize=8)
            ax3.plot(gt_sorted, convex_means, 's-', color='green', label='Convex', linewidth=2, markersize=8)

            # Add gap area shading between the lines
            for i in range(len(gt_sorted)):
                if not np.isnan(concave_means[i]) and not np.isnan(convex_means[i]):
                    lower = min(concave_means[i], convex_means[i])
                    upper = max(concave_means[i], convex_means[i])
                    # Draw connecting line and add gap label
                    ax3.plot([gt_sorted[i], gt_sorted[i]], [lower, upper], 'k--', alpha=0.3)
                    ax3.text(gt_sorted[i] + 0.1, (lower + upper) / 2, f'Δ={abs(upper - lower):.0f}',
                             fontsize=8, va='center')

            # Update axis labels as requested
            ax3.set_xlabel("Ground Truth Time-to-Collision (ms)")
            ax3.set_ylabel("Model Time-to-Collision (ms)")
            ax3.set_title(f"Concave vs Convex Collision Times (Line Plot), IoU={iou_thr}")

            # Set x-axis ticks
            ax3.set_xticks(gt_sorted)
            ax3.set_xticklabels([str(g) for g in gt_sorted])

            # Add a grid for better readability
            ax3.grid(True, linestyle='--', alpha=0.3)

            ax3.legend()

            fig_out3 = out_dir_cc / "concave_vs_convex_line_plot.png"
            plt.tight_layout()
            plt.savefig(fig_out3)
            plt.close(fig3)

            logger.info(f"Done creating results for {model_name}, IoU={iou_thr}")

    logger.info("All analyses complete. Exiting.")


if __name__ == "__main__":
    main()