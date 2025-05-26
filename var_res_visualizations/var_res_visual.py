#!/usr/bin/env python3
"""
plot_model_performance_3pct_fullres.py

Generate bar plots of Uniform, Variable, and Full performance at 3% density,
separately for GQA and VQAv2. “Full” bars use a dotted hatch texture to
distinguish them. The GQA plot includes both sampling and model legends; the
VQAv2 plot only includes the model legend. GQA shows both, VQAv2 only model legend.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_group(models, params,
               uniform, variable, variable_stds, full,
               colors, title, save_path,
               show_sampling_legend=False,
               show_model_legend=False):
    """
    Draws and saves a grouped bar chart for one dataset.

    - show_sampling_legend: include the Uniform/Variable/Full legend if True
    - show_model_legend   : include the Models (Params) legend if True
    """
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Uniform bars (half-opaque + hatch)
    for i, col in enumerate(colors):
        ax.bar(
            x[i] - width,
            uniform[i],
            width,
            color=col,
            alpha=0.5,
            edgecolor="black",
            hatch="//"
        )

    # Variable bars (solid + error bars)
    for i, col in enumerate(colors):
        ax.bar(
            x[i],
            variable[i],
            width,
            color=col,
            edgecolor="black",
            yerr=variable_stds[i],
            capsize=5
        )

    # Full bars (solid + dotted hatch)
    for i, col in enumerate(colors):
        ax.bar(
            x[i] + width,
            full[i],
            width,
            color=col,
            edgecolor="black",
            hatch=".."
        )

    # Configure axes
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, rotation=15)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(40, 85)

    # Sampling legend
    if show_sampling_legend:
        sampling_handles = [
            Patch(facecolor="white", edgecolor="black", hatch="//"),
            Patch(facecolor="white", edgecolor="black"),
            Patch(facecolor="white", edgecolor="black", hatch=".."),
        ]
        sampling_labels = ["Uniform", "Variable", "Full"]
        leg1 = ax.legend(
            sampling_handles,
            sampling_labels,
            title="Sampling",
            title_fontsize=12,
            fontsize=10,
            loc="upper left"
        )
        ax.add_artist(leg1)

    # Model legend
    if show_model_legend:
        model_handles = [Patch(facecolor=c, edgecolor="black") for c in colors]
        model_labels = [f"{m} ({p})" for m, p in zip(models, params)]
        ax.legend(
            model_handles,
            model_labels,
            title="Models (Params)",
            title_fontsize=12,
            fontsize=10,
            loc="upper right"
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "bar_plots")
    os.makedirs(out_dir, exist_ok=True)

    # --- GQA plot: include both legends ---
    gqa_models = ["MDETR", "BLIP2"]
    gqa_params = ["169 M", "3.4 B"]
    gqa_uniform = np.array([44.1, 40.7])
    gqa_variable = np.array([46.8, 42.3])
    gqa_variable_stds = np.array([0.01, 0.21])
    gqa_full = np.array([61.7, 44.0])
    gqa_colors = ["#FFBC4E", "#4682B4"]

    plot_group(
        gqa_models, gqa_params,
        gqa_uniform, gqa_variable, gqa_variable_stds, gqa_full,
        gqa_colors,
        title="Performance @ 3% Density (GQA)",
        save_path=os.path.join(out_dir, "gqa_performance.png"),
        show_sampling_legend=True,
        show_model_legend=True
    )

    # --- VQAv2 plot: only model legend ---
    vqa_models = ["ViLT", "BLIP2", "InstructBLIP", "LLaVa-v1.5"]
    vqa_params = ["87.4 M", "3.4 B", "4 B", "13 B"]
    vqa_uniform = np.array([62.9, 56.2, 66.5, 65.1])
    vqa_variable = np.array([64.9, 57.9, 66.4, 65.9])
    vqa_variable_stds = np.array([0.82, 0.46, 0.56, 0.75])
    vqa_full = np.array([81.1, 63.1, 73.5, 73.1])
    vqa_colors = ["#8E44AD", "#4682B4", "#D1A85F", "#4FAE4E"]

    # Order VQAv2 by descending Full performance
    order = np.argsort(-vqa_full)
    vqa_models = [vqa_models[i] for i in order]
    vqa_params = [vqa_params[i] for i in order]
    vqa_uniform = vqa_uniform[order]
    vqa_variable = vqa_variable[order]
    vqa_variable_stds = vqa_variable_stds[order]
    vqa_full = vqa_full[order]
    vqa_colors = [vqa_colors[i] for i in order]

    plot_group(
        vqa_models, vqa_params,
        vqa_uniform, vqa_variable, vqa_variable_stds, vqa_full,
        vqa_colors,
        title="Performance @ 3% Density (VQAv2)",
        save_path=os.path.join(out_dir, "vqav2_performance.png"),
        show_sampling_legend=False,
        show_model_legend=True
    )
