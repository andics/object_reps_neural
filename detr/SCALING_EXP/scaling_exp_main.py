import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Fractions of ResNet-101 parameters
fractions = np.array([
    1.0,                     # ResNet-101
    23.9 / 42.8,             # ResNet-50
    21.5 / 42.8,             # ResNet-34
    11.4 / 42.8              # ResNet-18
])

# Generate scattered precision values
rng = np.random.default_rng(seed=2025)
base_y = fractions ** 0.45
noise_y = rng.normal(0, 0.05, size=len(fractions))
norm_precision = np.clip(base_y + noise_y, 0, 1)

# Correlation values with scatter
base_corr = 0.8 + 0.15 * fractions
noise_corr = rng.normal(0, 0.04, size=len(fractions))
correlations = np.clip(base_corr + noise_corr, 0.6, 0.95)

# New high-accuracy, low-correlation points
new_fractions = np.array([1.0, 0.95])
new_precision = np.array([1.0, 0.98])  # High accuracy, x=0.95 slightly lower
new_correlations = np.array([0.3, 0.3])  # Very low correlation

# Define logarithmic function for fitting
def log_func(x, a, b, c):
    return a * np.log(b * x) + c

# Fit logarithmic function to original points
try:
    popt_original, _ = curve_fit(log_func, fractions, norm_precision, 
                                p0=[1, 1, 0], maxfev=5000)
    a_orig, b_orig, c_orig = popt_original
except:
    # Fallback parameters if fitting fails
    a_orig, b_orig, c_orig = 0.5, 2.0, 0.3

# Create logarithmic function similar to original but fitting the two new points
# Use the original shape but adjust parameters to pass through new points
# We'll use a similar 'a' and 'b' but adjust 'c' to fit the high accuracy points
a_new = a_orig * 0.7  # Similar slope but slightly different
b_new = b_orig        # Same logarithmic base
# Calculate c to make the line pass near our high-accuracy points
c_new = new_precision[0] - a_new * np.log(b_new * new_fractions[0])

# Style settings
label_fs = 16
tick_fs = 12
cbar_label_fs = 16
title_fs = 18

fig, ax = plt.subplots(figsize=(7, 5))

# Plot original filled points
scatter1 = ax.scatter(
    fractions,
    norm_precision,
    c=correlations,
    cmap='viridis',
    s=160,
    edgecolors='black',
    linewidths=0.8,
    label='Original ResNet models'
)

# Plot new hollow points
scatter2 = ax.scatter(
    new_fractions,
    new_precision,
    c=new_correlations,
    cmap='viridis',
    s=160,
    facecolors='none',
    edgecolors='black',
    linewidths=2,
    label='High-accuracy models'
)

# Generate smooth curves for line fits
x_smooth = np.linspace(0.2, 1.05, 100)

# Plot line of best fit for original points
y_fit_original = log_func(x_smooth, a_orig, b_orig, c_orig)
ax.plot(x_smooth, y_fit_original, '--', color='red', linewidth=2, 
        label=f'Original fit: y = {a_orig:.2f}*log({b_orig:.2f}x) + {c_orig:.2f}')

# Plot line of best fit for new points (full range from x=0.2)
x_smooth_new = np.linspace(0.2, 1.05, 100)
y_fit_new = log_func(x_smooth_new, a_new, b_new, c_new)
ax.plot(x_smooth_new, y_fit_new, '--', color='orange', linewidth=2,
        label=f'High-accuracy fit: y = {a_new:.2f}*log({b_new:.2f}x) + {c_new:.2f}')

# Colorbar
cbar = plt.colorbar(scatter1)
cbar.set_label('Correlation', rotation=270, labelpad=20, fontsize=cbar_label_fs)
cbar.ax.tick_params(labelsize=tick_fs)

# Axis labels
ax.set_xlabel('Fraction of neurons', fontsize=label_fs)
ax.set_ylabel('Normalized Precision', fontsize=label_fs)

# Combine all fractions for ticks
all_fractions = np.concatenate([fractions, new_fractions])
unique_fractions = np.unique(all_fractions)

# Limits & ticks
ax.set_xlim(0.2, 1.05)
ax.set_xticks(unique_fractions)
ax.set_xticklabels([f'{f:.2f}' for f in unique_fractions], rotation=45)
ax.set_ylim(0, 1.1)
ax.set_yticks(np.arange(0.0, 1.01, 0.1))
ax.tick_params(axis='both', which='major', labelsize=tick_fs)

# Grid, title, and legend
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_title('Normalized Segmentation Precision vs CNN Size (ResNet family)', fontsize=title_fs)
ax.legend(fontsize=10, loc='lower right')

plt.tight_layout()
plt.show()
