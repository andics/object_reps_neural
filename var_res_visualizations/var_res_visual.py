#!/usr/bin/env python3
"""
plot_model_performance_3pct_fullres.py

One grouped-bar chart (VQAv2 + GQA, 3 % density).

Top-right area:
 ┌─────────────┬──────────────────────┐
 │ Sampling    │  Models (Params)     │
 │  • Uniform  │  • ViLT (87.4 M) …   │
 │  • Variable │                      │
 │  • Full     │                      │
 └─────────────┴──────────────────────┘
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_all_models(models, params, datasets,
                    uniform, variable, variable_stds, full,
                    colors, save_path):
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # -------- Bars -----------------------------------------------------------
    for i, col in enumerate(colors):
        ax.bar(x[i] - width, uniform[i],  width,
               color=col, alpha=0.5, edgecolor="black", hatch='//')
        ax.bar(x[i],         variable[i], width,
               color=col, edgecolor="black",
               yerr=variable_stds[i], capsize=5)
        ax.bar(x[i] + width, full[i],     width,
               color=col, edgecolor="black", hatch='..')

    # -------- Axis cosmetics -------------------------------------------------
    tick_labels = [f"{m}\n({ds})" for m, ds in zip(models, datasets)]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=14, rotation=15, ha='center')
    ax.set_ylabel("Accuracy (%)", fontsize=15)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(40, 85)
    ax.set_title("Performance @ 3 % Density", fontsize=16)

    # -------- Model legend (anchored at true upper-right) --------------------
    seen = set()
    mh, ml = [], []
    for m, p, c in zip(models, params, colors):
        if m in seen:
            continue
        mh.append(Patch(facecolor=c, edgecolor="black"))
        ml.append(f"{m} ({p})")
        seen.add(m)

    leg_models = ax.legend(mh, ml,
                           title="Models (Params)", title_fontsize=12,
                           fontsize=10,
                           loc="upper right",          # anchor point of legend box
                           bbox_to_anchor=(1.0, 1.0))  # (x, y) in axes coords
    ax.add_artist(leg_models)

    # -------- Sampling legend (upper-right, offset left) ---------------------
    sh = [
        Patch(facecolor="white", edgecolor="black", hatch='//'),
        Patch(facecolor="white", edgecolor="black"),
        Patch(facecolor="white", edgecolor="black", hatch='..')
    ]
    ax.legend(sh, ["Uniform (downscale)", "Variable", "Full-resolution"],
              title="Sampling", title_fontsize=12,
              fontsize=10,
              loc="upper right",
              bbox_to_anchor=(0.806, 1.0))   # move left (x < 1.0) but keep top-aligned

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ============================================================================ #
if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "bar_plots")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------ VQAv2 --------------------------------------------------
    vqa_models   = ["ViLT", "BLIP2", "InstructBLIP", "LLaVa-v1.5"]
    vqa_params   = ["87.4 M", "3.4 B", "4 B", "13 B"]
    vqa_uniform  = np.array([62.9, 56.2, 66.5, 65.1])
    vqa_variable = np.array([64.9, 57.9, 66.4, 65.9])
    vqa_var_std  = np.array([0.82, 0.46, 0.56, 0.75])
    vqa_full     = np.array([81.1, 63.1, 73.5, 73.1])
    vqa_colors   = ["#8E44AD", "#4682B4", "#D1A85F", "#4FAE4E"]

    order = np.argsort(-vqa_full)
    vqa_models   = [vqa_models[i]  for i in order]
    vqa_params   = [vqa_params[i]  for i in order]
    vqa_uniform  = vqa_uniform[order]
    vqa_variable = vqa_variable[order]
    vqa_var_std  = vqa_var_std[order]
    vqa_full     = vqa_full[order]
    vqa_colors   = [vqa_colors[i]  for i in order]

    # ------------------ GQA ----------------------------------------------------
    gqa_models   = ["MDETR", "BLIP2"]
    gqa_params   = ["169 M", "3.4 B"]
    gqa_uniform  = np.array([44.1, 40.7])
    gqa_variable = np.array([46.8, 42.3])
    gqa_var_std  = np.array([0.01, 0.21])
    gqa_full     = np.array([61.7, 44.0])
    gqa_colors   = ["#FFBC4E", "#4682B4"]

    # -------------- Combine ---------------------------------------------------
    all_models   = vqa_models   + gqa_models
    all_params   = vqa_params   + gqa_params
    all_datasets = ["VQAv2"] * len(vqa_models) + ["GQA"] * len(gqa_models)
    all_uniform  = np.concatenate([vqa_uniform,  gqa_uniform])
    all_variable = np.concatenate([vqa_variable, gqa_variable])
    all_var_std  = np.concatenate([vqa_var_std,  gqa_var_std])
    all_full     = np.concatenate([vqa_full,     gqa_full])
    all_colors   = vqa_colors   + gqa_colors

    # -------------- Plot ------------------------------------------------------
    outfile = os.path.join(out_dir, "combined_vqa_gqa_performance.png")
    plot_all_models(all_models, all_params, all_datasets,
                    all_uniform, all_variable, all_var_std, all_full,
                    all_colors, outfile)
