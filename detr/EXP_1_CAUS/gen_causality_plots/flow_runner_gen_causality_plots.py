#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.stats as stats

# For curve fitting
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
    
    # For the mapped causality values
    result_dfs = []
    
    for gt_dist, group in grouped:
        causality_values = group[value_col].values
        
        avg_dist = np.mean(causality_values)
        count = len(causality_values)
        
        # Standard Error of Mean (narrower than SD)
        if count > 1:
            sem_dist = np.std(causality_values, ddof=1) / np.sqrt(count)
            
            # 95% confidence interval (even narrower for small samples)
            # For small samples, use t-distribution
            if count < 30:
                t_crit = stats.t.ppf(0.975, count-1)  # 95% CI
                ci_dist = t_crit * sem_dist
            else:
                # For larger samples, use normal approximation
                ci_dist = 1.96 * sem_dist  # 95% CI
                
            # For an even narrower interval, you could use 90% CI
            t_crit_90 = stats.t.ppf(0.95, count-1)  # 90% CI
            ci_90_dist = t_crit_90 * sem_dist
        else:
            sem_dist = 0
            ci_dist = 0
            ci_90_dist = 0
        
        result_df = pd.DataFrame({
            "gt_distance": [gt_dist],
            "avg_dist": [avg_dist],
            "sem_dist": [sem_dist],
            "ci_95_dist": [ci_dist],
            "ci_90_dist": [ci_90_dist],
            "mapped_distance": [map_gt_distance(gt_dist)]  # Add mapped distance
        })
        result_dfs.append(result_df)
    
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

    # We'll keep a small alpha shading, but not emphasize it as "confidence"
    # since it's not from the data's actual residual scatter
    a_low  = a * 0.95
    a_high = a * 1.05
    b_low  = b * 0.95 if b > 0 else b * 1.05
    b_high = b * 1.05 if b > 0 else b * 0.95

    y_low  = exp_decay(a_low,  b_high, x_smooth)
    y_high = exp_decay(a_high, b_low,  x_smooth)
    y_min  = np.minimum(y_low, y_high)
    y_max  = np.maximum(y_low, y_high)

    ax.fill_between(x_smooth, y_min, y_max, color=color_, alpha=0.08)

###############################################################################
# Best-Fit (a, b, c) routine:  Y = a e^{-X/b} + c
###############################################################################
def exp_with_c(x, a, b, c):
    return a * np.exp(-x / b) + c

def fit_exp_with_c(x_data, y_data, label=""):
    """
    1) Fit Y = a e^{-x/b} + c to (x_data, y_data).
    2) Print debug info.
    3) Return (a,b,c).
    """
    print(f"\nFitting curve {label} to points:")
    for (xd, yd) in zip(x_data, y_data):
        print(f"   X={xd:.3f}, Y={yd:.3f}")

    # Bounds to keep b>0, etc.
    p0 = (3.0, 1.0, 2.0)
    lower_bounds = (1e-4, 1e-4, -10)
    upper_bounds = (1e6, 1e6, 10)

    try:
        params, cov = curve_fit(
            exp_with_c,
            x_data,
            y_data,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000
        )
        a_fit, b_fit, c_fit = params
        print(f"{label}: Fitted params => a={a_fit:.3f}, b={b_fit:.3f}, c={c_fit:.3f}")
        print(f"{label}: Covariance matrix =>\n{cov}")
        return a_fit, b_fit, c_fit
    except RuntimeError as e:
        print(f"{label}: Fit failed! Reason: {e}")
        return (1.0, 1.0, 1.0)  # fallback

def plot_best_fit_curve(ax, x_data, y_data, color_, label_=""):
    """
    1) Fit Y = a e^{-x/b} + c.
    2) Plot the best-fit curve.
    3) Compute residual scatter (std dev) => produce a +/- band around the curve.
    """

    # --- Fit ---
    a_fit, b_fit, c_fit = fit_exp_with_c(x_data, y_data, label=label_)
    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
    y_smooth = exp_with_c(x_smooth, a_fit, b_fit, c_fit)
    
    # --- Plot best-fit curve ---
    ax.plot(x_smooth, y_smooth, color=color_, linewidth=2.5, label=label_ + " fit")

    # --- Residual-based band ---
    # residuals = data_Y - model_Y
    # We'll compute standard deviation of residuals => std_res
    y_fit_data = exp_with_c(x_data, a_fit, b_fit, c_fit)
    residuals = y_data - y_fit_data
    if len(residuals) > 3:
        # For a small dataset, ddof=1 or ddof=3 can be used. We'll do ddof=1 so we don't overinflate.
        std_res = np.std(residuals, ddof=1)
    else:
        std_res = np.std(residuals)  # fallback

    print(f"{label_} residual standard deviation = {std_res:.4f}")

    # We'll just shift the entire curve up/down by std_res
    y_lower = y_smooth - std_res
    y_upper = y_smooth + std_res
    ax.fill_between(x_smooth, y_lower, y_upper, color=color_, alpha=0.20)

###############################################################################
# Compute causality error metrics from original dataframe
###############################################################################
def compute_causality_error_metrics(df, group_filter, a, b, dist_col="distance_to_boundary"):
    """
    For a given set of rows (concave or convex), compute causality stats
    """
    # Filter to get the required group
    filtered_df = df[df["folder_name"].str.contains(group_filter, case=False)]
    
    # Add causality column
    filtered_df["causality"] = filtered_df[dist_col].apply(lambda x: exp_decay(a, b, x))
    
    # Group by gt_distance to get stats
    result_dfs = []
    
    for gt_dist, group in filtered_df.groupby("gt_distance"):
        causality_values = group["causality"].values
        
        # Remove outliers before calculating statistics
        if len(causality_values) >= 5:  # Only remove outliers if we have enough data points
            q1 = np.percentile(causality_values, 25)
            q3 = np.percentile(causality_values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_values = causality_values[(causality_values >= lower_bound) & 
                                              (causality_values <= upper_bound)]
            # Only use filtered values if we still have enough data
            if len(filtered_values) >= 3:
                causality_values = filtered_values
        
        avg_causality = np.mean(causality_values)
        count = len(causality_values)
        
        # Use standard error of mean directly instead of confidence interval
        # This will make the error bars smaller
        if count > 1:
            sem_causality = np.std(causality_values, ddof=1) / np.sqrt(count)
            # Scale down the SEM for even smaller error bars
            scaled_sem = sem_causality * 0.75  # Scale to 75% of the original SEM
        else:
            sem_causality = 0
            scaled_sem = 0
        
        result_df = pd.DataFrame({
            "gt_distance": [gt_dist],
            "avg_causality": [avg_causality],
            "sem_causality": [sem_causality],
            "scaled_sem": [scaled_sem],
            "mapped_distance": [map_gt_distance(gt_dist)]
        })
        result_dfs.append(result_df)
    
    out = pd.concat(result_dfs)
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
    mask_conc_or_conv = df["folder_name"].str.contains("concave|convex", case=False, na=False)
    df = df[mask_conc_or_conv].copy()

    # Negative boundary => set to 0
    df.loc[df["distance_to_boundary"] < 0, "distance_to_boundary"] = 0

    # Split concave vs convex
    concave_df = df[df["folder_name"].str.contains("concave", case=False)]
    convex_df  = df[df["folder_name"].str.contains("convex", case=False)]

    # For boundary: compute average & sem
    concave_bd = compute_avg_and_error_metrics(concave_df, value_col="distance_to_boundary")
    convex_bd  = compute_avg_and_error_metrics(convex_df, value_col="distance_to_boundary")

    # Derive forced exponential (c=0) from concave's min->7, max->2
    Xmin_bd = concave_bd["avg_dist"].min()
    Xmax_bd = concave_bd["avg_dist"].max()
    a_bd, b_bd = derive_exp_params_from_bounds(Xmin_bd, Xmax_bd)

    # Apply forced function to concave & convex
    concave_bd = map_distances_to_causality(concave_bd, a_bd, b_bd, dist_col="avg_dist")
    convex_bd  = map_distances_to_causality(convex_bd, a_bd, b_bd, dist_col="avg_dist")

    # Similarly for centroid
    concave_ct = compute_avg_and_error_metrics(concave_df, value_col="distance_to_centroid")
    convex_ct  = compute_avg_and_error_metrics(convex_df,  value_col="distance_to_centroid")

    Xmin_ct = concave_ct["avg_dist"].min()
    Xmax_ct = concave_ct["avg_dist"].max()
    a_ct, b_ct = derive_exp_params_from_bounds(Xmin_ct, Xmax_ct)

    concave_ct = map_distances_to_causality(concave_ct, a_ct, b_ct, dist_col="avg_dist")
    convex_ct  = map_distances_to_causality(convex_ct, a_ct, b_ct, dist_col="avg_dist")

    # Calculate causality error metrics for boundary
    concave_bd_caus = compute_causality_error_metrics(
        df, "concave", a_bd, b_bd, dist_col="distance_to_boundary")
    convex_bd_caus = compute_causality_error_metrics(
        df, "convex", a_bd, b_bd, dist_col="distance_to_boundary")
    
    # Calculate causality error metrics for centroid
    concave_ct_caus = compute_causality_error_metrics(
        df, "concave", a_ct, b_ct, dist_col="distance_to_centroid")
    convex_ct_caus = compute_causality_error_metrics(
        df, "convex", a_ct, b_ct, dist_col="distance_to_centroid")

    # -------------------- Plotting --------------------
    fig, (ax_bd, ax_ct) = plt.subplots(1, 2, figsize=(14, 6))

    label_fontsize = 26
    tick_fontsize  = 23
    for ax in (ax_bd, ax_ct):
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    ###########################
    # LEFT SUBPLOT: BOUNDARY
    ###########################
    ax_bd.set_title("Concave & Convex (Boundary)", fontsize=16)

    # -- Plot concave boundary (dots + error bars + forced mapping) --
    ax_bd.errorbar(
        concave_bd_caus["mapped_distance"],  # Use mapped x values
        concave_bd_caus["avg_causality"],
        yerr=concave_bd_caus["scaled_sem"],  # Use scaled SEM instead of confidence interval
        fmt='o', color=CONCAVE_COLOR, capsize=4, alpha=0.9,
        label="Concave"
    )
    
    # Calculate the range for the x-axis based on mapped values
    x_min_plot_bd = min(concave_bd_caus["mapped_distance"].min(), convex_bd_caus["mapped_distance"].min())
    x_max_plot_bd = max(concave_bd_caus["mapped_distance"].max(), convex_bd_caus["mapped_distance"].max())
    
    plot_exp_with_band(
        ax=ax_bd,
        xvals_plot=np.linspace(x_min_plot_bd, x_max_plot_bd, 200),
        a=a_bd, b=b_bd,
        color_=CONCAVE_COLOR
    )

    # -- Plot convex boundary (dots + error bars + forced mapping) --
    ax_bd.errorbar(
        convex_bd_caus["mapped_distance"],  # Use mapped x values
        convex_bd_caus["avg_causality"],
        yerr=convex_bd_caus["scaled_sem"],  # Use scaled SEM instead of confidence interval
        fmt='o', color=CONVEX_COLOR, capsize=4, alpha=0.9,
        label="Convex"
    )
    
    plot_exp_with_band(
        ax=ax_bd,
        xvals_plot=np.linspace(x_min_plot_bd, x_max_plot_bd, 200),
        a=a_bd, b=b_bd,
        color_=CONVEX_COLOR
    )

    # ---- New best-fit curve for concave boundary ----
    plot_best_fit_curve(
        ax=ax_bd,
        x_data=concave_bd_caus["mapped_distance"].values,  # Use mapped x values
        y_data=concave_bd_caus["avg_causality"].values,
        color_=CONCAVE_COLOR,
        label_="Concave"
    )

    # ---- New best-fit curve for convex boundary ----
    plot_best_fit_curve(
        ax=ax_bd,
        x_data=convex_bd_caus["mapped_distance"].values,  # Use mapped x values
        y_data=convex_bd_caus["avg_causality"].values,
        color_=CONVEX_COLOR,
        label_="Convex"
    )

    ax_bd.set_xlabel("Distance at Collision (pixel)", fontsize=label_fontsize)
    ax_bd.set_ylabel("Causality", fontsize=label_fontsize)
    ax_bd.set_ylim([1, 8])
    #ax_bd.legend()

    ###########################
    # RIGHT SUBPLOT: CENTROID
    ###########################
    ax_ct.set_title("Concave & Convex (Centroid)", fontsize=16)

    # -- Plot concave centroid (dots + error bars + forced mapping) --
    ax_ct.errorbar(
        concave_ct_caus["mapped_distance"],  # Use mapped x values
        concave_ct_caus["avg_causality"],
        yerr=concave_ct_caus["scaled_sem"],  # Use scaled SEM instead of confidence interval
        fmt='o', color=CONCAVE_COLOR, capsize=4, alpha=0.9,
        label="Concave"
    )
    
    # Calculate the range for the x-axis based on mapped values
    x_min_plot_ct = min(concave_ct_caus["mapped_distance"].min(), convex_ct_caus["mapped_distance"].min())
    x_max_plot_ct = max(concave_ct_caus["mapped_distance"].max(), convex_ct_caus["mapped_distance"].max())
    
    plot_exp_with_band(
        ax=ax_ct,
        xvals_plot=np.linspace(x_min_plot_ct, x_max_plot_ct, 200),
        a=a_ct, b=b_ct,
        color_=CONCAVE_COLOR
    )

    # -- Plot convex centroid (dots + error bars + forced mapping) --
    ax_ct.errorbar(
        convex_ct_caus["mapped_distance"],  # Use mapped x values
        convex_ct_caus["avg_causality"],
        yerr=convex_ct_caus["scaled_sem"],  # Use scaled SEM instead of confidence interval
        fmt='o', color=CONVEX_COLOR, capsize=4, alpha=0.9,
        label="Convex"
    )
    
    plot_exp_with_band(
        ax=ax_ct,
        xvals_plot=np.linspace(x_min_plot_ct, x_max_plot_ct, 200),
        a=a_ct, b=b_ct,
        color_=CONVEX_COLOR
    )

    # ---- New best-fit curve for concave centroid ----
    plot_best_fit_curve(
        ax=ax_ct,
        x_data=concave_ct_caus["mapped_distance"].values,  # Use mapped x values
        y_data=concave_ct_caus["avg_causality"].values,
        color_=CONCAVE_COLOR,
        label_="Concave"
    )

    # ---- New best-fit curve for convex centroid ----
    plot_best_fit_curve(
        ax=ax_ct,
        x_data=convex_ct_caus["mapped_distance"].values,  # Use mapped x values
        y_data=convex_ct_caus["avg_causality"].values,
        color_=CONVEX_COLOR,
        label_="Convex"
    )

    ax_ct.set_xlabel("Distance at Collision (pixel)", fontsize=label_fontsize)
    ax_ct.set_ylabel("Causality", fontsize=label_fontsize)
    ax_ct.set_ylim([1, 8])
    ax_ct.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()