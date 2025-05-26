import numpy as np
import matplotlib.pyplot as plt

# Data
fractions = np.array([1.0, 0.9])
eds_values = np.array([20, 21])         # EDS values
correlations = np.array([0.793, 0.850])

# Reference EDS and normalised precision
eds_ref = eds_values[fractions.argmax()]   # 20
norm_precision = eds_ref / eds_values       # 1.0 and ~0.952

# Font sizes
label_fs = 16
tick_fs = 12
cbar_label_fs = 16
title_fs = 18

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(
    fractions,
    norm_precision,
    c=correlations,
    cmap='viridis',
    s=120,
    edgecolors='black',
    linewidths=0.8
)

# Colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Correlation', rotation=270, labelpad=20, fontsize=cbar_label_fs)
cbar.ax.tick_params(labelsize=tick_fs)

# Labels
ax.set_xlabel('Fraction of Neurons', fontsize=label_fs)
ax.set_ylabel(f'Normalized Segmentation Precision (Wasserstein)', fontsize=label_fs)

# Axis limits
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0, 1.1)

# Tick marks
ax.set_xticks(np.linspace(0, 1, 6))            # 0 … 1.0
ax.set_yticks(np.arange(0.0, 1.01, 0.1))        # 0.0, 0.1, …, 1.0
ax.tick_params(axis='both', which='major', labelsize=tick_fs)

# Grid & title
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_title('Normalized Segmentation Precision vs Fraction of Neurons', fontsize=title_fs)

plt.tight_layout()
plt.show()
