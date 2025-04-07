#!/usr/bin/env python3
"""
Script: plot_causality.py
=========================
Reads a CSV file with columns:
    folder_name, gt_distance, distance_to_boundary, distance_to_centroid, frame_used

Steps performed:
1) Reads data from CSV (default path is specified; can be overridden via --csv_path).
2) Rows are kept only if folder_name contains "concave" or "convex" (case-insensitive). Others ignored.
3) Negative distance_to_boundary is set to 0 (instead of discarding).
4) Two "causality" measures are computed:
   - boundary_causality = a * exp(-distance_to_boundary / b) + c
   - centroid_causality = a * exp(-distance_to_centroid   / B) + C
   (each with min≈7, max≈2 in a simplified fit)
5) The script then plots, side-by-side:
   - Subplot #1: GT Distance (x-axis) vs. boundary_causality (y-axis)
   - Subplot #2: GT Distance (x-axis) vs. centroid_causality (y-axis)
   with lines for concave (orange) vs. convex (green). The figure is displayed and saved.
6) A t-test (Welch's) is performed on distance_to_boundary and distance_to_centroid
   for each GT distance, comparing concave vs. convex. Results are written to a .log file
   in a 'logs' subfolder, along with descriptive statistics (mean, std, sample sizes).

Usage:
    python plot_causality.py --csv_path <path_to_csv>

If --csv_path is omitted, it defaults to:
    Q:\Projects\Object_reps_neural\Programming\detr\EXP_1_CAUS\gen_collision_dist_csv_from_frames\full_videos_processed_csv\results_final_1px.csv

Outputs:
    - A PNG figure saved in a 'plots' subfolder (created if not exist) in the same folder as this script.
    - A log file with t-test results + descriptive statistics saved in a 'logs' subfolder in the same folder.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.stats import ttest_ind

def main():
    parser = argparse.ArgumentParser(
        description="Plot concave vs convex data from CSV and compute t-tests."
    )
    parser.add_argument("--csv_path", type=str,
                        required=False,
                        default=r"Q:\Projects\Object_reps_neural\Programming\detr\EXP_1_CAUS\gen_collision_dist_csv_from_frames\full_videos_processed_csv\results_final_1px.csv",
                        help="Path to the input CSV file.")
    args = parser.parse_args()

    # Identify directory of this script to store logs & plots in subfolders there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir   = os.path.join(script_dir, "logs")
    plots_dir  = os.path.join(script_dir, "plots")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Step 1: Read the CSV
    print(f"[INFO] Reading CSV from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    # Step 2: Keep rows where folder_name has "concave" or "convex"
    print("[INFO] Filtering rows for folder_name containing 'concave' or 'convex'.")
    mask_concave_convex = df["folder_name"].str.contains("concave|convex", case=False, na=False)
    df = df[mask_concave_convex].copy()

    # Step 3: For distance_to_boundary < 0, set to 0
    print("[INFO] Setting negative distance_to_boundary to 0 (instead of discarding).")
    df.loc[df["distance_to_boundary"] < 0, "distance_to_boundary"] = 0

    # -------------------------------------------
    # Prepare to compute "causality" from distance
    # using the form:  C(D) = a * exp(-D / b) + c
    #
    # We'll pick a, b, c so that min distance -> ~7
    # and max distance -> ~2 (rough approximation).
    # We'll do so separately for boundary & centroid.
    # -------------------------------------------
    def build_exponential_mapper(distances):
        """
        Returns a function C(D) = a * exp(-D / b) + c
        such that C(min(D)) ~ 7, and C(max(D)) ~ 2 (roughly).
        """
        d_min = distances.min()
        d_max = distances.max()

        # We'll use c=2, a=5 => a + c=7 => that covers the top value
        c_val = 2.0
        a_val = 5.0
        # Solve for b so that at d_max => a * exp(-d_max/b) is ~ a*0.01 => near c
        # => -d_max/b = ln(0.01) => b = d_max/4.605
        if d_max > 0:
            b_val = d_max / 4.605
        else:
            # If d_max=0, fallback
            b_val = 1.0

        def causality_func(D):
            return a_val * np.exp(-D / b_val) + c_val

        return causality_func

    # Build separate causality mappers
    print("[INFO] Building exponential mappers for boundary and centroid.")
    boundary_mapper = build_exponential_mapper(df["distance_to_boundary"])
    centroid_mapper = build_exponential_mapper(df["distance_to_centroid"])

    # Apply them
    df["boundary_causality"] = df["distance_to_boundary"].apply(boundary_mapper)
    df["centroid_causality"] = df["distance_to_centroid"].apply(centroid_mapper)

    # Separate concave / convex subsets
    concave_df = df[df["folder_name"].str.contains("concave", case=False)]
    convex_df  = df[df["folder_name"].str.contains("convex", case=False)]

    # Group by gt_distance for boundary causality
    concave_bc = concave_df.groupby("gt_distance")["boundary_causality"].agg(["mean","std"])
    convex_bc  = convex_df.groupby("gt_distance")["boundary_causality"].agg(["mean","std"])

    # Group by gt_distance for centroid causality
    concave_cc = concave_df.groupby("gt_distance")["centroid_causality"].agg(["mean","std"])
    convex_cc  = convex_df.groupby("gt_distance")["centroid_causality"].agg(["mean","std"])

    # Ensure sorted order by gt_distance
    concave_bc.sort_index(inplace=True)
    convex_bc.sort_index(inplace=True)
    concave_cc.sort_index(inplace=True)
    convex_cc.sort_index(inplace=True)

    # Step 4: Plot side-by-side subplots
    print("[INFO] Generating subplots: boundary on left, centroid on right.")
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

    # --- Left subplot: boundary causality ---
    ax1.errorbar(
        x=concave_bc.index,
        y=concave_bc["mean"],
        yerr=concave_bc["std"],
        label="Concave",
        color="orange",
        fmt='o-',
        capsize=3
    )
    ax1.errorbar(
        x=convex_bc.index,
        y=convex_bc["mean"],
        yerr=convex_bc["std"],
        label="Convex",
        color="green",
        fmt='o-',
        capsize=3
    )
    ax1.set_xlabel("GT Distance")
    ax1.set_ylabel("Boundary Causality")
    ax1.set_title("Boundary Causality vs. GT Distance")
    ax1.set_ylim([1, 8])
    ax1.legend()

    # --- Right subplot: centroid causality ---
    ax2.errorbar(
        x=concave_cc.index,
        y=concave_cc["mean"],
        yerr=concave_cc["std"],
        label="Concave",
        color="orange",
        fmt='o-',
        capsize=3
    )
    ax2.errorbar(
        x=convex_cc.index,
        y=convex_cc["mean"],
        yerr=convex_cc["std"],
        label="Convex",
        color="green",
        fmt='o-',
        capsize=3
    )
    ax2.set_xlabel("GT Distance")
    ax2.set_ylabel("Centroid Causality")
    ax2.set_title("Centroid Causality vs. GT Distance")
    ax2.set_ylim([1, 8])
    ax2.legend()

    fig.suptitle("Causality Metrics: Boundary vs. Centroid", fontsize=14)

    # Save the figure in the 'plots' subfolder
    plot_filename = f"causality_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    print(f"[INFO] Plot saved to {plot_path}")

    # Also display the plot
    plt.show()

    # Step 5: T-test on distance_to_boundary AND distance_to_centroid
    #         for concave vs. convex by gt_distance
    print("[INFO] Performing t-tests and writing descriptive stats to log file.")
    unique_gt_distances = sorted(df["gt_distance"].unique())

    log_filename = f"t_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(logs_dir, log_filename)

    with open(log_path, "w") as f:
        # Header
        f.write("===== Causality + Statistical Metrics Log =====\n")
        f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for gtd in unique_gt_distances:
            # Subset: concave / convex, for the current GT distance
            concave_sub = concave_df[concave_df["gt_distance"] == gtd]
            convex_sub  = convex_df[convex_df["gt_distance"] == gtd]

            # Distances
            c_b_vals = concave_sub["distance_to_boundary"]
            v_b_vals = convex_sub["distance_to_boundary"]
            c_c_vals = concave_sub["distance_to_centroid"]
            v_c_vals = convex_sub["distance_to_centroid"]

            # Means, STDs, Ns
            c_b_mean, c_b_std, c_b_n = c_b_vals.mean(), c_b_vals.std(), len(c_b_vals)
            v_b_mean, v_b_std, v_b_n = v_b_vals.mean(), v_b_vals.std(), len(v_b_vals)
            c_c_mean, c_c_std, c_c_n = c_c_vals.mean(), c_c_vals.std(), len(c_c_vals)
            v_c_mean, v_c_std, v_c_n = v_c_vals.mean(), v_c_vals.std(), len(v_c_vals)

            f.write(f"--- GT distance = {gtd} ---\n")
            f.write(f"Concave Distance-to-Boundary: mean={c_b_mean:.3f}, std={c_b_std:.3f}, n={c_b_n}\n")
            f.write(f"Concave Distance-to-Centroid:  mean={c_c_mean:.3f}, std={c_c_std:.3f}, n={c_c_n}\n")
            f.write(f"Convex  Distance-to-Boundary: mean={v_b_mean:.3f}, std={v_b_std:.3f}, n={v_b_n}\n")
            f.write(f"Convex  Distance-to-Centroid:  mean={v_c_mean:.3f}, std={v_c_std:.3f}, n={v_c_n}\n")

            # T-tests for boundary
            if c_b_n > 1 and v_b_n > 1:
                t_stat_b, p_val_b = ttest_ind(c_b_vals, v_b_vals, equal_var=False)
                f.write(f"  T-test (boundary): t={t_stat_b:.4f}, p={p_val_b:.6f}\n")
            else:
                f.write("  T-test (boundary): Not enough data (need >1 in each group)\n")

            # T-tests for centroid
            if c_c_n > 1 and v_c_n > 1:
                t_stat_c, p_val_c = ttest_ind(c_c_vals, v_c_vals, equal_var=False)
                f.write(f"  T-test (centroid): t={t_stat_c:.4f}, p={p_val_c:.6f}\n")
            else:
                f.write("  T-test (centroid): Not enough data (need >1 in each group)\n")

            f.write("\n")  # blank line between GT-dist blocks

    print(f"[INFO] T-test results & metrics saved to {log_path}")
    print("[INFO] Script completed successfully.")

if __name__ == "__main__":
    main()
