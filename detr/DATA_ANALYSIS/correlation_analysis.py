import numpy as np
from scipy.stats import pearsonr, spearmanr
# Here we'll perform correlation analysis on the data from the 3 experiments
# and plot the results

# Experiment 1: causality
# Model
causality_concave_model_avg = [
    6.515471743689135,
    6.423219111305542,
    5.330302635680974,
    5.9056307637786745,
    5.439935246060781,
    4.934269424422913,
    4.729851708282164,
    4.446808385178003,
    3.873785490359845,
    2.9129584232487984
]

causality_convex_model_avg = [
    5.276223029689829,
    5.055682761628078,
    4.838863595211646,
    4.565358823094536,
    4.392061623582376,
    4.235454375994574,
    3.942476537769437,
    3.6769700151494797,
    3.1525236807444523,
    2.5515846792933834
]

# Humans
causality_concave_human_avg = [
    6.684523809523809,
    6.489795918367347,
    6.129251700680272,
    5.641156462585034,
    5.08843537414966,
    4.72108843537415,
    4.211734693877551,
    3.687925170068027,
    3.0314625850340136,
    2.4719387755102042
]

causality_convex_human_avg = [
    6.6726190476190474,
    6.318027210884353,
    5.6020408163265305,
    4.686224489795919,
    4.071428571428571,
    3.6624149659863945,
    3.360544217687075,
    2.9073129251700682,
    2.517857142857143,
    2.179421768707483
]

# Experiment 2: TTC
ttc_correlation = 0.865
ttc_n = 45

# Experiment 3: Object change
# Model: concave, concave_nofill, convex
detected_changes_model_ratios = [11/16, 14/16, 14/16]

# Humans: concave, concave_nofill, convex
detected_changes_human_ratios = [72.25/100, 89.125/100, 88/100]

alpha = 0.05

def report_corr(name, x, y):
    r, p = pearsonr(x, y)
    rho, p_s = spearmanr(x, y)
    sig_pearson = "significant" if p < alpha else "not significant"
    sig_spearman = "significant" if p_s < alpha else "not significant"
    print(f"{name}:")
    print(f"  Pearson r = {r:.3f}, p = {p:.3f} → {sig_pearson} at α = {alpha}")
    print(f"  Spearman ρ = {rho:.3f}, p = {p_s:.3f} → {sig_spearman} at α = {alpha}")
    print()

print("Experiment 1 Correlations:")
report_corr("Concave model vs. human", causality_concave_model_avg, causality_concave_human_avg)
report_corr("Convex model vs. human", causality_convex_model_avg, causality_convex_human_avg)

# Overall (combined concave+convex)
model_all = np.concatenate([causality_concave_model_avg, causality_convex_model_avg])
human_all = np.concatenate([causality_concave_human_avg, causality_convex_human_avg])
report_corr("Overall model vs. human", model_all, human_all)

# Experiment 3: detected changes
detected_changes_model = np.array([11/16, 14/16, 14/16])
detected_changes_human = np.array([72.25/100, 89.125/100, 88/100])

print("Experiment 3 Correlation:")
report_corr("Model vs. human detected-changes", detected_changes_model, detected_changes_human)