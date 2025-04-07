#!/usr/bin/env python3
import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict


###############################################################################
# Exponential decay function to map distance -> causality
# We will figure out alpha so that distance=0 => 7, distance=d_max => 2:
#   C(d) = 7 * exp( -alpha * d )
#   with alpha = ln(7/2) / d_max
###############################################################################
def distance_to_causality(d, d_max):
    """
    Exponential mapping from distance [0..d_max] to causality [7..2].
    - If d_max == 0 (edge case, e.g. if all distances are 0), we default
      causality to 7.
    """
    if d_max <= 0:
        return 7.0
    alpha = math.log(7.0 / 2.0) / d_max
    cval = 7.0 * math.exp(-alpha * d)
    return cval


###############################################################################
# Read the CSV (produced by the first script for threshold=1 pixel).
# It has columns: folder_name, gt_distance, distance_to_boundary, ...
# We parse them, find shape from "folder_name", and compute a causality measure.
# Return:
#   convex_list => [ (distance, causality), ...]
#   concave_list => [ (distance, causality), ...]
###############################################################################
def read_and_convert(csv_path):
    """
    Reads the _1px.csv file, IGNORES (skips) any row where distance_to_boundary < 0,
    and converts each (non-negative) distance to a [2..7] causality by exponential decay.
    Returns two lists of (distance, causality): one for convex, one for concave.
    """
    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV file '{csv_path}' not found.")
        return [], []

    # We must gather all distances first to find d_max
    all_distances = []
    rows_cache = []

    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                dval = float(row.get("distance_to_boundary", "0"))
            except ValueError:
                continue
            # Skip rows that have distance_to_boundary below zero
            if dval < 0:
                continue

            rows_cache.append(row)
            all_distances.append(dval)

    # find d_max among the non-negative distances we kept
    if len(all_distances) == 0:
        d_max = 0
    else:
        d_max = max(all_distances)

    convex_data = []
    concave_data = []

    for row in rows_cache:
        fname = row.get("folder_name", "").lower()
        try:
            dval = float(row["distance_to_boundary"])
        except:
            continue
        # (We already know dval >= 0 since we didn't skip it.)

        caus = distance_to_causality(dval, d_max)
        data_point = (dval, caus)

        if "convex" in fname:
            convex_data.append(data_point)
        elif "concave" in fname:
            concave_data.append(data_point)
        else:
            # If neither "convex" nor "concave" is in folder_name,
            # we skip or store in another category.
            pass

    return convex_data, concave_data


###############################################################################
# Utility: compute average and standard errors for each distinct distance bucket
###############################################################################
def compute_stats(data_list):
    """
    data_list is a list of (distance, causality).
    We'll group by distance, compute the mean/SEM for each distinct distance.
    Return arrays xvals, means, sems
    """
    from collections import defaultdict
    bucket = defaultdict(list)
    for (dist_, c_) in data_list:
        bucket[dist_].append(c_)

    xvals = []
    means = []
    sems = []
    for dist_ in sorted(bucket.keys()):
        arr = np.array(bucket[dist_], dtype=float)
        m_ = np.mean(arr)
        # std error of the mean
        sem_ = np.std(arr, ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else 0
        xvals.append(dist_)
        means.append(m_)
        sems.append(sem_)
    return np.array(xvals), np.array(means), np.array(sems)


###############################################################################
# We'll do a t-test for all convex vs concave causality (pooled).
###############################################################################
def perform_statistical_tests(convex_data, concave_data):
    """
    Just lumps all convex_data, concave_data together and does a Welch's t-test.
    Also prints out means, difference, p-value, etc.
    """
    cvx_vals = np.array([c for (_, c) in convex_data])
    ccv_vals = np.array([c for (_, c) in concave_data])

    if len(cvx_vals) < 2 or len(ccv_vals) < 2:
        print("Not enough data points for a meaningful t-test.")
        return

    t_stat, p_val = stats.ttest_ind(cvx_vals, ccv_vals, equal_var=False)

    cvx_mean = np.mean(cvx_vals)
    ccv_mean = np.mean(ccv_vals)
    diff = cvx_mean - ccv_mean

    # Compute effect size (Cohen's d)
    n1, n2 = len(cvx_vals), len(ccv_vals)
    var1, var2 = np.var(cvx_vals, ddof=1), np.var(ccv_vals, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std != 0:
        cohens_d = diff / pooled_std
    else:
        cohens_d = float('inf')

    print("\n===== STATISTICAL ANALYSIS (Overall) =====")
    print(f"Convex mean: {cvx_mean:.3f}  (n={n1})")
    print(f"Concave mean: {ccv_mean:.3f} (n={n2})")
    print(f"Difference: {diff:.3f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_val:.6f}")
    print(f"Cohen's d: {cohens_d:.3f}")
    if p_val < 0.05:
        if diff < 0:
            print("=> Convex is significantly LOWER in causality than Concave (p<0.05).")
        else:
            print("=> Convex is significantly HIGHER in causality than Concave (p<0.05).")
    else:
        print("=> No statistically significant difference (p ≥ 0.05).")


###############################################################################
# Simple function to plot raw data => scatter
###############################################################################
def plot_raw(ax, data_list, color_, label_):
    """
    data_list => list of (distance, causality)
    Just scatter them, no fancy fitting for now.
    """
    if not data_list:
        return
    data_list = sorted(data_list, key=lambda x: x[0])
    xvals = [d[0] for d in data_list]
    yvals = [d[1] for d in data_list]

    ax.scatter(xvals, yvals, s=60, alpha=0.7, color=color_, label=label_)


###############################################################################
# Plot average => error bar
###############################################################################
def plot_avg(ax, xvals, means, sems, color_, label_):
    """
    xvals, means, sems => arrays from compute_stats
    We just do an errorbar plot.
    """
    if len(xvals) == 0:
        return
    ax.errorbar(xvals, means, yerr=sems, fmt='o-', color=color_,
                ecolor=color_, capsize=4, alpha=0.9, label=label_)


###############################################################################
# Main plotting routine
###############################################################################
def main():
    # Update this path to point to the CSV that was produced by the first script for "1 pixel" overlap:
    # e.g. "results_final_1px.csv"
    CSV_1PX_PATH = "/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/detr/EXP_1_CAUS/gen_collision_dist_csv_from_frames/full_videos_processed_csv/results_final_1px.csv"

    # Load data (skipping negative distances)
    convex_data, concave_data = read_and_convert(CSV_1PX_PATH)
    print(f"Loaded {len(convex_data)} convex points, {len(concave_data)} concave points.")

    # Perform a simple overall t-test
    perform_statistical_tests(convex_data, concave_data)

    # Prepare data for average plots
    cvx_x, cvx_mean, cvx_sem = compute_stats(convex_data)
    ccv_x, ccv_mean, ccv_sem = compute_stats(concave_data)

    # Create figure with 2×3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ax_crv_raw = axes[0, 0]
    ax_ccv_raw = axes[0, 1]
    ax_over_raw = axes[0, 2]
    ax_crv_avg = axes[1, 0]
    ax_ccv_avg = axes[1, 1]
    ax_over_avg = axes[1, 2]

    # Some aesthetic adjustments
    label_fontsize = 16
    tick_fontsize = 12
    for ax in axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.set_ylim(1, 8)  # causality ~ [2..7], so a bit of margin

    # Colors
    convex_color = "#39A039"  # green
    concave_color = "#FEB02F"  # yellowish/orange

    # (0,0) Convex RAW
    ax_crv_raw.set_title("Convex RAW", fontsize=label_fontsize)
    plot_raw(ax_crv_raw, convex_data, convex_color, "Convex")
    ax_crv_raw.set_xlabel("Distance", fontsize=label_fontsize)
    ax_crv_raw.set_ylabel("Causality", fontsize=label_fontsize)
    ax_crv_raw.legend()

    # (0,1) Concave RAW
    ax_ccv_raw.set_title("Concave RAW", fontsize=label_fontsize)
    plot_raw(ax_ccv_raw, concave_data, concave_color, "Concave")
    ax_ccv_raw.set_xlabel("Distance", fontsize=label_fontsize)
    ax_ccv_raw.set_ylabel("Causality", fontsize=label_fontsize)
    ax_ccv_raw.legend()

    # (0,2) Overlay RAW
    ax_over_raw.set_title("Overlay RAW", fontsize=label_fontsize)
    plot_raw(ax_over_raw, convex_data, convex_color, "Convex")
    plot_raw(ax_over_raw, concave_data, concave_color, "Concave")
    ax_over_raw.set_xlabel("Distance", fontsize=label_fontsize)
    ax_over_raw.set_ylabel("Causality", fontsize=label_fontsize)
    ax_over_raw.legend()

    # (1,0) Convex AVERAGE
    ax_crv_avg.set_title("Convex AVERAGE", fontsize=label_fontsize)
    plot_avg(ax_crv_avg, cvx_x, cvx_mean, cvx_sem, convex_color, "Convex AVG")
    ax_crv_avg.set_xlabel("Distance", fontsize=label_fontsize)
    ax_crv_avg.set_ylabel("Causality", fontsize=label_fontsize)
    ax_crv_avg.legend()

    # (1,1) Concave AVERAGE
    ax_ccv_avg.set_title("Concave AVERAGE", fontsize=label_fontsize)
    plot_avg(ax_ccv_avg, ccv_x, ccv_mean, ccv_sem, concave_color, "Concave AVG")
    ax_ccv_avg.set_xlabel("Distance", fontsize=label_fontsize)
    ax_ccv_avg.set_ylabel("Causality", fontsize=label_fontsize)
    ax_ccv_avg.legend()

    # (1,2) Overlay AVERAGE
    ax_over_avg.set_title("Overlay AVERAGE", fontsize=label_fontsize)
    plot_avg(ax_over_avg, cvx_x, cvx_mean, cvx_sem, convex_color, "Convex AVG")
    plot_avg(ax_over_avg, ccv_x, ccv_mean, ccv_sem, concave_color, "Concave AVG")
    ax_over_avg.set_xlabel("Distance", fontsize=label_fontsize)
    ax_over_avg.set_ylabel("Causality", fontsize=label_fontsize)
    ax_over_avg.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
