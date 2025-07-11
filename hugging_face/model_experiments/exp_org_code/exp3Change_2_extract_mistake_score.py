#!/usr/bin/env python3
"""
threshold_analysis.py

Analyze area-change thresholds across image pairs, producing per-image details,
overall detection rates, and bar charts with diagonal hatches and downscaled SEM error bars,
without outlier filtering, with an increased left margin before the first bar (3× default),
bars filling space (no gaps), and no x-axis ticks or labels.

For each threshold (<pct>_comparison/):
  • per_image_detailed.json
  • overall_comparison.json
  • overall_comparison.png   # detection for all four types
  • three_comparison.png     # detection for concave, concave_nofill, convex

Usage:
    python threshold_analysis.py \
        --input_root /path/to/processed_images \
        --output_root /path/to/threshold_results

Requires: numpy, Pillow, matplotlib
"""
import os
import argparse
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

label_fontsize = 21
ticks_fontsize = 19

def parse_args():
    p = argparse.ArgumentParser("Analyze area-change thresholds across image pairs")
    p.add_argument(
        "--input_root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_images"),
        help="Folder containing processed image subfolders"
    )
    p.add_argument(
        "--output_root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold_results"),
        help="Where to write threshold analyses"
    )
    return p.parse_args()


def load_mask(mask_dir):
    """
    Returns the first PNG in mask_dir as a boolean mask.
    """
    for fn in os.listdir(mask_dir):
        if fn.lower().endswith('.png'):
            img = Image.open(os.path.join(mask_dir, fn)).convert('L')
            return np.array(img) > 0
    raise FileNotFoundError(f"No PNG mask in {mask_dir}")


def img_type_from_name(base: str) -> str:
    if 'concave_nofill' in base or ('nofill' in base and 'concave' not in base): return 'concave_nofill'
    if 'concave' in base: return 'concave'
    if 'convex' in base:   return 'convex'
    if 'no_change' in base: return 'no_change'
    return 'unknown'


def compute_sem_binary(dets: np.ndarray) -> float:
    """
    Compute standard error of the mean (SEM) as percentage, then downscale by half.
    SEM = sqrt(p * (1 - p) / N) * 100
    """
    n = dets.size
    if n == 0:
        return 0.0
    p = dets.mean()
    sem = np.sqrt(p * (1 - p) / n) * 100
    return sem * 0.5


def main():
    args = parse_args()
    # collect before/after pairs
    pairs = []
    for name in sorted(os.listdir(args.input_root)):
        if not name.endswith('_init'):
            continue
        if 'catch_shape' in name:
            continue
        base = name[:-5]
        init_dir = os.path.join(args.input_root, name)
        out_dir  = os.path.join(args.input_root, base + '_out')
        if os.path.isdir(init_dir) and os.path.isdir(out_dir):
            pairs.append((base, init_dir, out_dir))
    if not pairs:
        print(f"No valid pairs in {args.input_root}")
        return

    # thresholds including 2%
    thresholds = [0.01,0.02,0.03,0.04,0.05,0.06] + [i/100 for i in range(8,21,2)]
    os.makedirs(args.output_root, exist_ok=True)

    # compute per-image changes
    details = []
    for base, init_dir, out_dir in pairs:
        try:
            m0 = load_mask(os.path.join(init_dir, 'frames_masks_nonmem'))
            m1 = load_mask(os.path.join(out_dir,  'frames_masks_nonmem'))
        except Exception as e:
            print(f"[WARN] {base}: {e}")
            continue
        a0, a1 = m0.sum(), m1.sum()
        ratio = None if a0 == 0 else abs(a1 - a0) / a0
        details.append({
            'base': base,
            'type': img_type_from_name(base),
            'before_mask': init_dir  + '/frames_masks_nonmem',
            'after_mask':  out_dir   + '/frames_masks_nonmem',
            'area_before': int(a0),
            'area_after':  int(a1),
            'area_change': ratio
        })

    types = ['concave','concave_nofill','convex','no_change']

    # plotting parameters
    width = 1.0
    left_margin = 0.5
    high_dpi = 200

    for thr in thresholds:
        pct = int(round(thr*100))
        dir_out = os.path.join(args.output_root, f"{pct}_comparison")
        os.makedirs(dir_out, exist_ok=True)

        # save per-image details
        with open(os.path.join(dir_out,'per_image_detailed.json'),'w') as f:
            json.dump(details, f, indent=2)

        # compute detections
        dets = {t: [] for t in types}
        for d in details:
            t = d['type']
            if t in dets:
                dets[t].append(1 if (d['area_change'] is not None and d['area_change'] > thr) else 0)

        # overall summary
        summary = {t: {'detected': int(sum(dets[t])), 'total': len(dets[t])} for t in types}
        with open(os.path.join(dir_out,'overall_comparison.json'),'w') as f:
            json.dump(summary, f, indent=2)

        # compute rates & SEMs
        rates = np.array([(np.mean(dets[t]) * 100 if dets[t] else 0.0) for t in types])
        sems  = np.array([compute_sem_binary(np.array(dets[t])) for t in types])

        # overall plot
        x = np.arange(len(types)) + left_margin
        fig, ax = plt.subplots(figsize=(4.8, 4), dpi=high_dpi)
        for i, t in enumerate(types):
            ax.bar(x[i], rates[i], width,
                   color='lightgray', edgecolor='black', hatch='//',
                   yerr=sems[i], capsize=5)
        ax.set_xticks([])
        ax.set_ylabel('% Detection Rate')
        ax.set_ylim(0, 100)
        ax.set_xlim(left_margin - 0.8*width, left_margin + len(types) - 0.2*width)
        ax.set_title(f'Threshold = {pct}%')
        plt.tight_layout()
        fig.savefig(os.path.join(dir_out,'overall_comparison.png'), dpi=high_dpi)
        plt.close(fig)

        # three-condition plot with additional 10% widening
        three = ['concave','concave_nofill','convex']
        colors = [
            (255/255, 188/255, 78/255),
            (209/255, 168/255, 95/255),
            (79/255, 168/255, 78/255)
        ]
        # base 15% + extra 10% => 1.15 * 1.10 = 1.265
        width_three = width * 2.1
        # maintain original data-space margin
        margin_data = width * 0.8
        # bar centers spaced by new width_three
        x2 = np.arange(len(three)) * width_three + left_margin
        fig, ax = plt.subplots(figsize=(4.5, 6), dpi=high_dpi)
        for i, t in enumerate(three):
            idx = types.index(t)
            ax.bar(x2[i], rates[idx], width_three,
                   color=colors[i], edgecolor='black', hatch='//',
                   yerr=sems[idx], capsize=5)
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=ticks_fontsize)
        ax.set_ylabel('% Noticing Change', fontsize=label_fontsize)
        ax.set_ylim(0, 100)
        left_lim = x2[0] - width_three/2 - margin_data
        right_lim = x2[-1] + width_three/2 + margin_data
        ax.set_xlim(left_lim, right_lim)
        ax.set_title(f'Threshold = {pct}%')
        plt.tight_layout()
        fig.savefig(os.path.join(dir_out, 'three_comparison.png'), dpi=high_dpi)
        plt.close(fig)

    print(f"Completed thresholds: {[int(t*100) for t in thresholds]}%.")

if __name__ == '__main__':
    main()
