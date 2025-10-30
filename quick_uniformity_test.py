"""Quick uniformity test for new 7D dataset"""
import pickle
import numpy as np
from scipy import stats
from model_common import PARAM_NAMES

# Load new 7D dataset
DATASET_PATH = "datasets/dataset_10000_dl100_7d_20251030_124511.pkl"

with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

X = data['X']
n_samples = X.shape[0]

print("="*80)
print("⚖️  UNIFORMITY TESTING (Chi-Square Test)")
print("="*80)
print(f"Dataset: {DATASET_PATH}")
print(f"Samples: {n_samples:,}\n")
print("Null hypothesis: Distribution is uniform")
print("If p-value > 0.05: Accept (uniform) ✅")
print("If p-value < 0.05: Reject (non-uniform) ❌\n")

uniformity_results = []

for i, param in enumerate(PARAM_NAMES):
    # Create bins and observed frequencies
    n_bins = 20
    observed, bin_edges = np.histogram(X[:, i], bins=n_bins)

    # Expected frequency (uniform)
    expected = np.full(n_bins, n_samples / n_bins)

    # Chi-square test
    chi2_stat, p_value = stats.chisquare(observed, expected)

    # Interpret
    is_uniform = p_value > 0.05

    # Unique values
    unique_count = len(np.unique(X[:, i]))

    uniformity_results.append({
        'Parameter': param,
        'Unique': unique_count,
        'Chi-Square': f"{chi2_stat:.2f}",
        'p-value': f"{p_value:.4f}",
        'Uniform?': '✅ Yes' if is_uniform else '❌ No',
    })

    print(f"{param:8s}: χ²={chi2_stat:8.2f}, p={p_value:.4f}, unique={unique_count:3d}  {'✅ UNIFORM' if is_uniform else '❌ NON-UNIFORM'}")

# Overall verdict
n_uniform = sum(1 for r in uniformity_results if '✅' in r['Uniform?'])
print("\n" + "="*80)
print(f"Uniform parameters: {n_uniform}/{len(PARAM_NAMES)}")
if n_uniform == len(PARAM_NAMES):
    print("✅ ALL DISTRIBUTIONS UNIFORM - Excellent for ML training!")
elif n_uniform >= len(PARAM_NAMES) * 0.8:
    print("⚠️  MOSTLY UNIFORM - Acceptable for training")
else:
    print("❌ NON-UNIFORM - May cause training bias!")
print("="*80)
