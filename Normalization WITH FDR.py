import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

#THIS IS FOR UN-NORMALIZED DATA WITH SAMPLES AS COLUMNS

# USER INPUT + OUTPUT FILES
input_file = "/Users/elizabethflammer/Desktop/Metformin POS 26W Grouping.csv"

output_normalized_csv = "/Users/elizabethflammer/Desktop/26W POS Norm.csv"
output_plot = "/Users/elizabethflammer/Desktop/26W POS Norm.png"
output_significant_excel = "/Users/elizabethflammer/Desktop/26W POS Norm FDR.xlsx"

# 1. Load CSV
df = pd.read_csv(input_file, header=None)

metabolite_ids = df.iloc[0, 1:]
groups = df.iloc[1, 1:]
data = df.iloc[2:, :]

data.columns = df.iloc[0, :]
data.index = df.iloc[2:, 0]
data = data.iloc[:, 1:]
data = data.apply(pd.to_numeric, errors='coerce')

# 2. Sample Normalization (Normalize by sum)
normalized = data.div(data.sum(axis=0), axis=1)

# 3. Log10 Transformation
offset = 1e-9
log_transformed = np.log10(normalized + offset)

# 4. Autoscaling (mean-center + divide by STD per metabolite)
means = log_transformed.mean(axis=1)
stds = log_transformed.std(axis=1)
autoscaled = log_transformed.sub(means, axis=0).div(stds, axis=0)

# 5. Reassemble output table
out_df = pd.DataFrame(index=data.index, columns=data.columns, data=autoscaled)
out_df.loc["Group"] = groups.values
out_df.loc["ID"] = metabolite_ids.values
out_df = out_df.loc[["ID", "Group"] + [idx for idx in data.index]]

# 6. Save to CSV
out_df.to_csv(output_normalized_csv)
print("Saved normalized file to:", output_normalized_csv)

# 7. Plots
before_vals = data.values.flatten()
after_vals = autoscaled.values.flatten()

max_features = 50
subset_before = data.iloc[:max_features, :]
subset_after = autoscaled.iloc[:max_features, :]
short_labels = [str(met)[:15] for met in subset_before.index]

fig, axes = plt.subplots(2, 2, figsize=(20, 16), gridspec_kw={"height_ratios": [1, 3]})

pd.Series(before_vals).plot(kind="kde", ax=axes[0, 0])
axes[0, 0].set_title("Density Plot – Before Normalization")
axes[0, 0].set_xlabel("Intensity")
axes[0, 0].set_ylabel("Density")
axes[0, 0].set_xlim(left=0)

axes[1, 0].boxplot(subset_before.values.T, vert=False, tick_labels=short_labels,
                   patch_artist=True, boxprops=dict(facecolor="#99cc99"))
axes[1, 0].set_title("Before Normalization (up to 50 metabolites)")
axes[1, 0].set_xlabel("Intensity")

pd.Series(after_vals).plot(kind="kde", ax=axes[0, 1])
axes[0, 1].set_title("Density Plot – After Normalization")
axes[0, 1].set_xlabel("Intensity")
axes[0, 1].set_ylabel("Density")

axes[1, 1].boxplot(subset_after.values.T, vert=False, tick_labels=short_labels,
                   patch_artist=True, boxprops=dict(facecolor="#99cc99"))
axes[1, 1].set_title("After Normalization (up to 50 metabolites)")
axes[1, 1].set_xlabel("Normalized Intensity")

plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print("Saved normalization plot as:", output_plot)

# 8. T-tests across ALL metabolites first, then apply FDR correction
group_labels = groups.values
unique_groups = np.unique(group_labels)
if len(unique_groups) != 2:
    raise ValueError("Exactly two groups are required for t-test analysis.")

groupA, groupB = unique_groups
groupA_cols = [col for col, grp in zip(data.columns, group_labels) if grp == groupA]
groupB_cols = [col for col, grp in zip(data.columns, group_labels) if grp == groupB]

# Run t-test on ALL metabolites (required for valid FDR correction)
all_results = []
for metabolite in autoscaled.index:
    group1 = autoscaled.loc[metabolite, groupA_cols].astype(float)
    group2 = autoscaled.loc[metabolite, groupB_cols].astype(float)
    t_stat, p_val = ttest_ind(group1, group2, equal_var=True, nan_policy='omit')
    all_results.append([metabolite, p_val])

all_results_df = pd.DataFrame(all_results, columns=["Metabolite", "P Value"])

# Apply Benjamini-Hochberg FDR correction across all metabolites
_, fdr_corrected, _, _ = multipletests(all_results_df["P Value"], method='fdr_bh')
all_results_df["FDR Corrected P Value"] = fdr_corrected

# Filter to significant by raw p-value threshold
sig_df = all_results_df[all_results_df["P Value"] < 0.104].copy()
sig_df = sig_df[["Metabolite", "P Value", "FDR Corrected P Value"]]

# Save as Excel
sig_df.to_excel(output_significant_excel, index=False)
print("Saved significant metabolite list to:", output_significant_excel)
