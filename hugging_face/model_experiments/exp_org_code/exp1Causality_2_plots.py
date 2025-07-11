#!/usr/bin/env python3

import os
import argparse
import json
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime
from scipy.optimize import curve_fit

###############################################################################
# Colors for concave/convex
###############################################################################
CONVEX_COLOR   = "#39A039"   # green
CONCAVE_COLOR  = "#FEB02F"   # yellow/orange

###############################################################################
# Original forced exponential (c=0)
#   Y(X) = a * exp(-X/b)
###############################################################################
def exp_decay(a, b, x):
    return a * np.exp(-x / b)

###############################################################################
# Solve for a,b so that:
#   Y(Xmin) = 7, Y(Xmax) = 2, with Y(X)=a e^(-X/b).
# If Xmin==Xmax, fallback => constant Y=7.
###############################################################################
def derive_exp_params_from_bounds(xmin, xmax):
    if math.isclose(xmin, xmax, rel_tol=1e-9):
        return (7.0, 1.0)  # fallback
    ln_2_over_7 = math.log(3.0 / 7.0)  # negative
    b = (xmin - xmax) / ln_2_over_7
    a = 7.0 * math.exp(xmin / b)
    return (a, b)

###############################################################################
# Map distance to causality with forced function Y = a e^{-X/b}.
###############################################################################
def map_distances_to_causality(df_group, a, b, dist_col="avg_dist"):
    df_group["causality"] = df_group[dist_col].apply(lambda x: exp_decay(a, b, x))
    return df_group

###############################################################################
# Distance mapping for x-axis
###############################################################################
DISTANCE_MAP = {
    0: 0, 1: 4, 2: 8, 3: 12, 4: 16,
    5: 20, 6: 25, 7: 32, 8: 45, 9: 64
}

def map_gt_distance(x):
    """Map gt_distance to the new scale"""
    return DISTANCE_MAP.get(x, x)  # Return x if not in map as fallback

###############################################################################
# For each gt_distance, compute average, standard error & confidence interval
###############################################################################
def compute_avg_and_error_metrics(df, value_col="distance_to_boundary"):
    grouped = df.groupby("gt_distance")
    result_dfs = []
    for gt_dist, group in grouped:
        vals = group[value_col].values
        avg_dist = np.mean(vals)
        count = len(vals)
        if count > 1:
            sem_dist = np.std(vals, ddof=1) / np.sqrt(count)
            if count < 30:
                t_crit = stats.t.ppf(0.975, count-1)
                ci_dist = t_crit * sem_dist
            else:
                ci_dist = 1.96 * sem_dist
            t_crit_90 = stats.t.ppf(0.95, count-1)
            ci_90_dist = t_crit_90 * sem_dist
        else:
            sem_dist = ci_dist = ci_90_dist = 0
        result_dfs.append(pd.DataFrame({
            "gt_distance": [gt_dist],
            "avg_dist": [avg_dist],
            "sem_dist": [sem_dist],
            "ci_95_dist": [ci_dist],
            "ci_90_dist": [ci_90_dist],
            "mapped_distance": [map_gt_distance(gt_dist)]
        }))
    out = pd.concat(result_dfs)
    out.sort_values("gt_distance", inplace=True)
    return out

###############################################################################
# Plot the forced Y=a e^{-X/b} curve with naive "confidence" shading
###############################################################################
def plot_exp_with_band(ax, xvals_plot, a, b, color_, label_=""):
    x_smooth = np.linspace(xvals_plot.min(), xvals_plot.max(), 200)
    y_smooth = exp_decay(a, b, x_smooth)
    ax.plot(x_smooth, y_smooth, color=color_, linewidth=2.0, label=label_)
    a_low, a_high = a * 0.95, a * 1.05
    b_low = b * 0.95 if b > 0 else b * 1.05
    b_high = b * 1.05 if b > 0 else b * 0.95
    y_low = exp_decay(a_low,  b_high, x_smooth)
    y_high = exp_decay(a_high, b_low,  x_smooth)
    ax.fill_between(x_smooth, np.minimum(y_low, y_high), np.maximum(y_low, y_high), color=color_, alpha=0.08)

###############################################################################
# Best-Fit (a, b, c) routine:  Y = a e^{-X/b} + c
###############################################################################
def exp_with_c(x, a, b, c):
    return a * np.exp(-x / b) + c

def fit_exp_with_c(x_data, y_data, label=""):
    print(f"\nFitting curve {label} to points:")
    for xd, yd in zip(x_data, y_data):
        print(f"   X={xd:.3f}, Y={yd:.3f}")
    p0 = (3.0, 1.0, 2.0)
    lower_bounds, upper_bounds = (1e-4, 1e-4, -10), (1e6, 1e6, 10)
    try:
        params, cov = curve_fit(exp_with_c, x_data, y_data, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000)
        print(f"{label}: Fitted params => a={params[0]:.3f}, b={params[1]:.3f}, c={params[2]:.3f}")
        print(f"{label}: Covariance matrix =>\n{cov}")
        return params
    except RuntimeError as e:
        print(f"{label}: Fit failed! Reason: {e}")
        return (1.0, 1.0, 1.0)

def plot_best_fit_curve(ax, x_data, y_data, color_, label_=""):
    a_fit, b_fit, c_fit = fit_exp_with_c(x_data, y_data, label=label_)
    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
    y_smooth = exp_with_c(x_smooth, a_fit, b_fit, c_fit)
    ax.plot(x_smooth, y_smooth, color=color_, linewidth=2.5, label=label_ + " fit")
    residuals = y_data - exp_with_c(x_data, a_fit, b_fit, c_fit)
    std_res = np.std(residuals, ddof=1) if len(residuals) > 3 else np.std(residuals)
    print(f"{label_} residual standard deviation = {std_res:.4f}")
    ax.fill_between(x_smooth, y_smooth - std_res, y_smooth + std_res, color=color_, alpha=0.20)

###############################################################################
# Compute causality error metrics from original dataframe
###############################################################################
def compute_causality_error_metrics(df, group_filter, a, b, dist_col="distance_to_boundary"):
    filtered = df[df["folder_name"].str.contains(group_filter, case=False)].copy()
    filtered["causality"] = filtered[dist_col].apply(lambda x: exp_decay(a, b, x))
    dfs = []
    for gt_dist, group in filtered.groupby("gt_distance"):
        vals = group["causality"].values
        if len(vals) >= 5:
            q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
            iqr = q3 - q1
            mask = (vals >= q1 - 1.5*iqr) & (vals <= q3 + 1.5*iqr)
            if mask.sum() >= 3:
                vals = vals[mask]
        avg_c, count = np.mean(vals), len(vals)
        sem_c = np.std(vals, ddof=1) / np.sqrt(count) if count > 1 else 0
        scaled = sem_c * 0.75 if count > 1 else 0
        dfs.append(pd.DataFrame({
            "gt_distance": [gt_dist],
            "avg_causality": [avg_c],
            "sem_causality": [sem_c],
            "scaled_sem": [scaled],
            "mapped_distance": [map_gt_distance(gt_dist)]
        }))
    out = pd.concat(dfs)
    out.sort_values("gt_distance", inplace=True)
    return out

###############################################################################
# Main
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        default=r"Q:\Projects\Object_reps_neural\Programming\detr\EXP_1_CAUS\gen_collision_dist_csv_from_frames\full_videos_processed_csv_110_frames_bc_used\results_final_1px.csv",
        help="Path to CSV input."
    )
    args = parser.parse_args()

    print(f"Reading CSV from {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    df = df[df["folder_name"].str.contains("concave|convex", case=False, na=False)].copy()
    df.loc[df["distance_to_boundary"] < 0, "distance_to_boundary"] = 0

    concave_df = df[df["folder_name"].str.contains("concave", case=False)]
    convex_df  = df[df["folder_name"].str.contains("convex", case=False)]

    # Boundary stats
    concave_bd = compute_avg_and_error_metrics(concave_df, value_col="distance_to_boundary")
    convex_bd  = compute_avg_and_error_metrics(convex_df,  value_col="distance_to_boundary")
    a_bd, b_bd = derive_exp_params_from_bounds(concave_bd["avg_dist"].min(), concave_bd["avg_dist"].max())
    concave_bd = map_distances_to_causality(concave_bd, a_bd, b_bd, dist_col="avg_dist")
    convex_bd  = map_distances_to_causality(convex_bd,  a_bd, b_bd, dist_col="avg_dist")
    concave_bd_caus = compute_causality_error_metrics(df, "concave", a_bd, b_bd, dist_col="distance_to_boundary")
    convex_bd_caus  = compute_causality_error_metrics(df, "convex",  a_bd, b_bd, dist_col="distance_to_boundary")

    # Centroid stats
    concave_ct = compute_avg_and_error_metrics(concave_df, value_col="distance_to_centroid")
    convex_ct  = compute_avg_and_error_metrics(convex_df,  value_col="distance_to_centroid")
    a_ct, b_ct = derive_exp_params_from_bounds(concave_ct["avg_dist"].min(), concave_ct["avg_dist"].max())
    concave_ct = map_distances_to_causality(concave_ct, a_ct, b_ct, dist_col="avg_dist")
    convex_ct  = map_distances_to_causality(convex_ct,  a_ct, b_ct, dist_col="avg_dist")
    concave_ct_caus = compute_causality_error_metrics(df, "concave", a_ct, b_ct, dist_col="distance_to_centroid")
    convex_ct_caus  = compute_causality_error_metrics(df, "convex",  a_ct, b_ct, dist_col="distance_to_centroid")

    # Prepare output directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Write detailed JSON for boundary
    boundary_details = []
    xs_bd = sorted(set(concave_bd_caus["mapped_distance"]).union(convex_bd_caus["mapped_distance"]))
    for x in xs_bd:
        cr = concave_bd_caus[concave_bd_caus["mapped_distance"] == x]
        vr = convex_bd_caus[convex_bd_caus["mapped_distance"] == x]
        boundary_details.append({
            "x_value": x,
            "concave_avg":   float(cr["avg_causality"].iloc[0]) if not cr.empty else None,
            "convex_avg":    float(vr["avg_causality"].iloc[0]) if not vr.empty else None,
            "concave_std":   float(cr["scaled_sem"].iloc[0])    if not cr.empty else None,
            "convex_std":    float(vr["scaled_sem"].iloc[0])    if not vr.empty else None,
        })
    with open(os.path.join(plots_dir, "boundary_detailed.json"), "w") as f:
        json.dump(boundary_details, f, indent=2)

    # Write detailed JSON for centroid
    centroid_details = []
    xs_ct = sorted(set(concave_ct_caus["mapped_distance"]).union(convex_ct_caus["mapped_distance"]))
    for x in xs_ct:
        cr = concave_ct_caus[concave_ct_caus["mapped_distance"] == x]
        vr = convex_ct_caus[convex_ct_caus["mapped_distance"] == x]
        centroid_details.append({
            "x_value": x,
            "concave_avg":   float(cr["avg_causality"].iloc[0]) if not cr.empty else None,
            "convex_avg":    float(vr["avg_causality"].iloc[0]) if not vr.empty else None,
            "concave_std":   float(cr["scaled_sem"].iloc[0])    if not cr.empty else None,
            "convex_std":    float(vr["scaled_sem"].iloc[0])    if not vr.empty else None,
        })
    with open(os.path.join(plots_dir, "centroid_detailed.json"), "w") as f:
        json.dump(centroid_details, f, indent=2)

    # -------------------- Plotting --------------------
    fig, (ax_bd, ax_ct) = plt.subplots(1, 2, figsize=(14, 6))
    label_fs, tick_fs = 26, 23
    for ax in (ax_bd, ax_ct):
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)

    # Boundary subplot
    ax_bd.set_title("Concave & Convex (Boundary)", fontsize=16)
    ax_bd.errorbar(concave_bd_caus["mapped_distance"], concave_bd_caus["avg_causality"],
                   yerr=concave_bd_caus["scaled_sem"], fmt='o', color=CONCAVE_COLOR, capsize=4, alpha=0.9)
    ax_bd.errorbar(convex_bd_caus["mapped_distance"], convex_bd_caus["avg_causality"],
                   yerr=convex_bd_caus["scaled_sem"], fmt='o', color=CONVEX_COLOR,  capsize=4, alpha=0.9)
    x_min_bd = min(concave_bd_caus["mapped_distance"].min(), convex_bd_caus["mapped_distance"].min())
    x_max_bd = max(concave_bd_caus["mapped_distance"].max(), convex_bd_caus["mapped_distance"].max())
    plot_exp_with_band(ax_bd, np.linspace(x_min_bd, x_max_bd, 200), a_bd, b_bd, CONCAVE_COLOR)
    plot_exp_with_band(ax_bd, np.linspace(x_min_bd, x_max_bd, 200), a_bd, b_bd, CONVEX_COLOR)
    plot_best_fit_curve(ax_bd, concave_bd_caus["mapped_distance"].values, concave_bd_caus["avg_causality"].values, CONCAVE_COLOR, "Concave")
    plot_best_fit_curve(ax_bd, convex_bd_caus["mapped_distance"].values,  convex_bd_caus["avg_causality"].values,  CONVEX_COLOR,  "Convex")
    ax_bd.set_xlabel("Distance at Collision (pixel)", fontsize=label_fs)
    ax_bd.set_ylabel("Causality", fontsize=label_fs)
    ax_bd.set_ylim([1, 8])

    # Centroid subplot
    ax_ct.set_title("Concave & Convex (Centroid)", fontsize=16)
    ax_ct.errorbar(concave_ct_caus["mapped_distance"], concave_ct_caus["avg_causality"],
                   yerr=concave_ct_caus["scaled_sem"], fmt='o', color=CONCAVE_COLOR, capsize=4, alpha=0.9)
    ax_ct.errorbar(convex_ct_caus["mapped_distance"], convex_ct_caus["avg_causality"],
                   yerr=convex_ct_caus["scaled_sem"], fmt='o', color=CONVEX_COLOR,  capsize=4, alpha=0.9)
    x_min_ct = min(concave_ct_caus["mapped_distance"].min(), convex_ct_caus["mapped_distance"].min())
    x_max_ct = max(concave_ct_caus["mapped_distance"].max(), convex_ct_caus["mapped_distance"].max())
    plot_exp_with_band(ax_ct, np.linspace(x_min_ct, x_max_ct, 200), a_ct, b_ct, CONCAVE_COLOR)
    plot_exp_with_band(ax_ct, np.linspace(x_min_ct, x_max_ct, 200), a_ct, b_ct, CONVEX_COLOR)
    plot_best_fit_curve(ax_ct, concave_ct_caus["mapped_distance"].values, concave_ct_caus["avg_causality"].values, CONCAVE_COLOR, "Concave")
    plot_best_fit_curve(ax_ct, convex_ct_caus["mapped_distance"].values,  convex_ct_caus["avg_causality"].values,  CONVEX_COLOR,  "Convex")
    ax_ct.set_xlabel("Distance at Collision (pixel)", fontsize=label_fs)
    ax_ct.set_ylabel("Causality", fontsize=label_fs)
    ax_ct.set_ylim([1, 8])
    ax_ct.legend()

    plt.tight_layout()

    # Save figure with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_name = f"causality_plot_{timestamp}.png"
    save_path = os.path.join(plots_dir, plot_name)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    print(f"Saved boundary JSON to {os.path.join(plots_dir, 'boundary_detailed.json')}")
    print(f"Saved centroid JSON to {os.path.join(plots_dir, 'centroid_detailed.json')}")
    plt.show()


if __name__ == "__main__":
    main()
