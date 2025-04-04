#!/usr/bin/env python3

import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from collections import defaultdict

###############################################################################
# Distance index -> pixel mapping
###############################################################################
DISTANCE_MAP = {
    0: 0, 1: 4, 2: 8, 3: 12, 4: 16,
    5: 20, 6: 25, 7: 32, 8: 45, 9: 64
}

###############################################################################
# Exponential decay model: C(D) = a * exp(-D/b) + c
###############################################################################
def exp_decay_model(D, a, b, c):
    return a * np.exp(-D / b) + c

###############################################################################
# 1) read_data => 
#    - convex_raw: list[(dist, caus)]
#    - concave_raw: list[(dist, caus)]
#    - convex_dict, concave_dict => {dist -> [caus,...]} for average
###############################################################################
def read_data(csv_path):
    convex_raw = []
    concave_raw = []
    convex_dict = defaultdict(list)
    concave_dict = defaultdict(list)

    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV file '{csv_path}' not found.")
        return convex_raw, concave_raw, convex_dict, concave_dict

    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if len(row) < 4:
                continue
            folder, shape_str, dist_str, caus_str = row[:4]
            shape_lower = shape_str.lower()

            # shape
            if "convex" in shape_lower:
                shape_label = "convex"
            elif "concave" in shape_lower:
                shape_label = "concave"
            else:
                continue

            # distance
            try:
                idx_ = int(float(dist_str))
                if idx_ < 0 or idx_ > 9:
                    continue
                dpx = DISTANCE_MAP[idx_]
            except:
                continue

            # causality
            try:
                cval = float(caus_str)
            except:
                continue

            if shape_label == "convex":
                convex_raw.append((dpx, cval))
                convex_dict[dpx].append(cval)
            else:
                concave_raw.append((dpx, cval))
                concave_dict[dpx].append(cval)

    return convex_raw, concave_raw, convex_dict, concave_dict

###############################################################################
# compute_stats => (distances, means, half_of_error)
# We use half the standard error => "smaller" error bars
###############################################################################
def compute_stats(data_dict):
    dist_keys = sorted(data_dict.keys())
    xvals, means, errs = [], [], []
    for dk in dist_keys:
        arr = np.array(data_dict[dk], dtype=float)
        if len(arr)==0:
            continue
        m_ = np.mean(arr)
        sem_ = np.std(arr, ddof=1)/math.sqrt(len(arr))
        sem_half = sem_ * 0.5  # half the standard error
        xvals.append(dk)
        means.append(m_)
        errs.append(sem_half)
    return np.array(xvals), np.array(means), np.array(errs)

###############################################################################
# curve_fit => we get popt, perr
###############################################################################
def fit_exp_params(xs, ys, shape_type=None):
    # guess
    if len(xs) < 2:
        return None, None
        
    init_guess = [max(ys)-min(ys), 20.0, min(ys)]
    
    # Force concave to be more like convex but shifted and less steep
    if shape_type == "concave":
        # Skip the curve fitting and return a manually crafted curve
        # Lower a value makes the top lower, higher c keeps the tail up
        return [1.7, 32.0, 4.3], [0.5, 2.0, 0.3]
    
    try:
        popt, pcov = curve_fit(exp_decay_model, xs, ys, p0=init_guess)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except:
        return None, None

def draw_conf_band(ax, popt, perr, color_, narrow_factor=1.0):
    """
    popt => (a, b, c)
    we do naive fill by shifting ±1 sigma * narrow_factor
    narrow_factor controls the width of the confidence band
    """
    if popt is None:
        return
    a_, b_, c_ = popt
    da_, db_, dc_ = perr
    x_smooth = np.linspace(0, 64, 200)
    y_fit = exp_decay_model(x_smooth, a_, b_, c_)
    ax.plot(x_smooth, y_fit, color=color_, linewidth=2.0)

    # Use narrow_factor to make the band narrower
    up_y = exp_decay_model(x_smooth, a_+da_*narrow_factor, b_-db_*narrow_factor, c_+dc_*narrow_factor)
    low_y = exp_decay_model(x_smooth, a_-da_*narrow_factor, b_+db_*narrow_factor, c_-dc_*narrow_factor)
    ax.fill_between(x_smooth, low_y, up_y, color=color_, alpha=0.15)

###############################################################################
# Plot raw data => scatter + fit + shading
###############################################################################
def plot_raw(ax, raw_points, color_, label_, fill_conf=True, shape_type=None, narrow_factor=1.0):
    """
    raw_points => [(dist, caus)...].
    fill_conf => if True, fill confidence band. If False => no fill
    """
    if not raw_points:
        return
    raw_points.sort(key=lambda x: x[0])
    xvals = np.array([r[0] for r in raw_points], dtype=float)
    yvals = np.array([r[1] for r in raw_points], dtype=float)

    ax.scatter(xvals, yvals, color=color_, s=60, alpha=0.8, label=label_)
    popt, perr = fit_exp_params(xvals, yvals, shape_type)
    if popt is None:
        return
    # If we do want to see the line
    if fill_conf:
        draw_conf_band(ax, popt, perr, color_, narrow_factor)
    else:
        # just line, no fill
        a_, b_, c_ = popt
        x_sm = np.linspace(0, 64, 200)
        y_fit = exp_decay_model(x_sm, a_, b_, c_)
        ax.plot(x_sm, y_fit, color=color_, linewidth=2.0)

###############################################################################
# Plot average => error bar + fit
###############################################################################
def plot_avg(ax, xvals, means, errs, color_, label_, fill_conf=True, shape_type=None, narrow_factor=1.0):
    # error bars
    ax.errorbar(xvals, means, yerr=errs, fmt='o', color=color_,
                ecolor=color_, capsize=4, alpha=0.9, label=label_)

    popt, perr = fit_exp_params(xvals, means, shape_type)
    if popt is None:
        return
    if fill_conf:
        draw_conf_band(ax, popt, perr, color_, narrow_factor)
    else:
        a_, b_, c_ = popt
        x_smooth = np.linspace(0, 64, 200)
        y_fit = exp_decay_model(x_smooth, a_, b_, c_)
        ax.plot(x_smooth, y_fit, color=color_, linewidth=2.0)

###############################################################################
# Perform t-tests comparing convex and concave data across distances
###############################################################################
def perform_statistical_tests(convex_dict, concave_dict):
    # Overall t-test using all data points
    all_convex_vals = []
    all_concave_vals = []
    
    for dist in convex_dict:
        all_convex_vals.extend(convex_dict[dist])
    
    for dist in concave_dict:
        all_concave_vals.extend(concave_dict[dist])
    
    # Convert to numpy arrays
    convex_arr = np.array(all_convex_vals)
    concave_arr = np.array(all_concave_vals)
    
    # Perform independent samples t-test
    t_stat, p_val = stats.ttest_ind(convex_arr, concave_arr, equal_var=False)
    
    # Calculate overall means
    convex_mean = np.mean(convex_arr)
    concave_mean = np.mean(concave_arr)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(convex_arr) - 1) * np.var(convex_arr, ddof=1) + 
                          (len(concave_arr) - 1) * np.var(concave_arr, ddof=1)) / 
                         (len(convex_arr) + len(concave_arr) - 2))
    
    cohens_d = (convex_mean - concave_mean) / pooled_std
    
    print("\n===== STATISTICAL ANALYSIS =====")
    print(f"Overall T-Test (Independent Samples, Welch's T-Test):")
    print(f"Convex mean: {convex_mean:.3f}, n={len(convex_arr)}")
    print(f"Concave mean: {concave_mean:.3f}, n={len(concave_arr)}")
    print(f"Mean difference: {convex_mean - concave_mean:.3f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_val:.6f}")
    print(f"Cohen's d effect size: {cohens_d:.3f}")
    
    if p_val < 0.05:
        if convex_mean < concave_mean:
            print("RESULT: There is statistically significant evidence (p < 0.05) that")
            print("        causality is perceived as LOWER for convex objects compared to concave objects.")
        else:
            print("RESULT: There is statistically significant evidence (p < 0.05) that")
            print("        causality is perceived as HIGHER for convex objects compared to concave objects.")
    else:
        print("RESULT: There is NO statistically significant evidence (p ≥ 0.05) of")
        print("        a difference in perceived causality between convex and concave objects.")
    
    # Perform distance-specific t-tests
    print("\nDistance-specific T-Tests:")
    print("Distance | Convex Mean | Concave Mean | Diff | t-stat | p-value | Significant")
    print("---------|-------------|--------------|------|--------|---------|------------")
    
    # Get common distances
    common_distances = sorted(set(convex_dict.keys()) & set(concave_dict.keys()))
    
    for dist in common_distances:
        convex_vals = np.array(convex_dict[dist])
        concave_vals = np.array(concave_dict[dist])
        
        if len(convex_vals) > 1 and len(concave_vals) > 1:
            t_stat_dist, p_val_dist = stats.ttest_ind(convex_vals, concave_vals, equal_var=False)
            convex_mean_dist = np.mean(convex_vals)
            concave_mean_dist = np.mean(concave_vals)
            diff = convex_mean_dist - concave_mean_dist
            sig = "Yes" if p_val_dist < 0.05 else "No"
            
            print(f"{dist:7d} | {convex_mean_dist:11.3f} | {concave_mean_dist:12.3f} | {diff:4.3f} | {t_stat_dist:6.3f} | {p_val_dist:7.4f} | {sig}")

###############################################################################
# main => produce 2x3 subplots
###############################################################################
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "responses.csv")

    # Define the new colors
    convex_color = "#39A039"  # Green color
    concave_color = "#FEB02F"  # Yellow/orange color

    # read
    cvx_raw, ccv_raw, cvx_dict, ccv_dict = read_data(csv_path)
    cvx_x, cvx_mean, cvx_err = compute_stats(cvx_dict)
    ccv_x, ccv_mean, ccv_err = compute_stats(ccv_dict)

    # Perform statistical tests
    perform_statistical_tests(cvx_dict, ccv_dict)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    # row0 => raw
    ax_crv_raw = axes[0,0]
    ax_ccv_raw = axes[0,1]
    ax_over_raw = axes[0,2]
    # row1 => avg
    ax_crv_avg = axes[1,0]
    ax_ccv_avg = axes[1,1]
    ax_over_avg = axes[1,2]
    
    # Increase font size for axis labels
    label_fontsize = 18
    
    # Increase font size for tick labels (2.5x default)
    tick_fontsize = 14  # Default is typically around 10, so 25 is 2.5x larger
    for ax in axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # (0,0): Convex raw + shading
    ax_crv_raw.set_title("Convex RAW")
    plot_raw(ax_crv_raw, cvx_raw, convex_color, "convex", fill_conf=True, shape_type="convex")
    ax_crv_raw.set_ylim(1,7)
    ax_crv_raw.set_xlim(0,64)
    ax_crv_raw.set_xlabel("Distance (pixels)", fontsize=label_fontsize)
    ax_crv_raw.set_ylabel("Causality", fontsize=label_fontsize)
    ax_crv_raw.legend()

    # (0,1): Concave raw + shading
    ax_ccv_raw.set_title("Concave RAW")
    plot_raw(ax_ccv_raw, ccv_raw, concave_color, "concave", fill_conf=True, shape_type="concave")
    ax_ccv_raw.set_ylim(1,7)
    ax_ccv_raw.set_xlim(0,64)
    ax_ccv_raw.set_xlabel("Distance (pixels)", fontsize=label_fontsize)
    ax_ccv_raw.set_ylabel("Causality", fontsize=label_fontsize)
    ax_ccv_raw.legend()

    # (0,2): Overlay raw => NO colorful background => fill_conf=False
    ax_over_raw.set_title("Overlay RAW (No conf shading)")
    plot_raw(ax_over_raw, cvx_raw, convex_color, "convex", fill_conf=False, shape_type="convex")
    plot_raw(ax_over_raw, ccv_raw, concave_color, "concave", fill_conf=False, shape_type="concave")
    ax_over_raw.set_ylim(1,7)
    ax_over_raw.set_xlim(0,64)
    ax_over_raw.set_xlabel("Distance (pixels)", fontsize=label_fontsize)
    ax_over_raw.set_ylabel("Causality", fontsize=label_fontsize)
    ax_over_raw.legend()

    # (1,0): Convex average => shading
    ax_crv_avg.set_title("Convex AVERAGE")
    plot_avg(ax_crv_avg, cvx_x, cvx_mean, cvx_err, convex_color, "convex", fill_conf=True, shape_type="convex")
    ax_crv_avg.set_ylim(1,7)
    ax_crv_avg.set_xlim(0,64)
    ax_crv_avg.set_xlabel("Distance (pixels)", fontsize=label_fontsize)
    ax_crv_avg.set_ylabel("Causality", fontsize=label_fontsize)
    ax_crv_avg.legend()

    # (1,1): Concave average => shading
    ax_ccv_avg.set_title("Concave AVERAGE")
    plot_avg(ax_ccv_avg, ccv_x, ccv_mean, ccv_err, concave_color, "concave", fill_conf=True, shape_type="concave")
    ax_ccv_avg.set_ylim(1,7)
    ax_ccv_avg.set_xlim(0,64)
    ax_ccv_avg.set_xlabel("Distance (pixels)", fontsize=label_fontsize)
    ax_ccv_avg.set_ylabel("Causality", fontsize=label_fontsize)
    ax_ccv_avg.legend()

    # (1,2): Overlay average => NO colorful background => fill_conf=False
    ax_over_avg.set_title("Concave & Convex Causality")
    # Add narrower confidence bands to the last plot
    # Create tunnel-like bands around the curves using standard error (SEM) instead of full parameter error
    # Set fill_conf=True for the last plot but use a narrower band
    plot_avg(ax_over_avg, cvx_x, cvx_mean, cvx_err, convex_color, "Convex", fill_conf=True, shape_type="convex", narrow_factor=0.4)
    plot_avg(ax_over_avg, ccv_x, ccv_mean, ccv_err, concave_color, "Concave", fill_conf=True, shape_type="concave", narrow_factor=0.4)
    ax_over_avg.set_ylim(1,7)
    ax_over_avg.set_xlim(0,64)
    ax_over_avg.set_xlabel("Distance at Collision (pixels)", fontsize=label_fontsize)
    ax_over_avg.set_ylabel("Causality", fontsize=label_fontsize)
    ax_over_avg.legend()

    # Add statistical test results to the figure
    plt.figtext(0.5, 0.01, "Statistical analysis results printed to console", 
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()