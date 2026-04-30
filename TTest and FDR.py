import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# THIS IS FOR NORMALIZED DATA OR META DATA: COLUMN 1 - GROUP COLUMN 2 - METABOLITE

# Config
CSV_PATH    = "Demo Table.csv"   # Path to your CSV file
P_THRESHOLD = 0.05              # Raw p-value filter threshold

# Data
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")

# Column 1 = groups, Column 2+ = metabolite values
group_col = df.columns[0]
groups    = df[group_col]
data      = df.iloc[:, 1:]  # all columns after the first

print(f"Group column : '{group_col}'")
print(f"Metabolites  : {data.shape[1]}")
print(f"Groups found : {groups.value_counts().to_dict()}\n")

# Transpose so metabolites are rows, samples are columns
autoscaled = data.T
autoscaled.columns = range(len(autoscaled.columns))  # numeric index aligned with groups

# T Test
group_labels = groups.values
unique_groups = np.unique(group_labels)
if len(unique_groups) != 2:
    raise ValueError(f"Exactly 2 groups required, found: {unique_groups}")

groupA, groupB = unique_groups
groupA_cols = [i for i, grp in enumerate(group_labels) if grp == groupA]
groupB_cols = [i for i, grp in enumerate(group_labels) if grp == groupB]

print(f"Group '{groupA}': n={len(groupA_cols)}")
print(f"Group '{groupB}': n={len(groupB_cols)}\n")

# Run t-test on ALL metabolites first (required for valid FDR correction)
all_results = []
for metabolite in autoscaled.index:
    group1 = autoscaled.loc[metabolite, groupA_cols].astype(float)
    group2 = autoscaled.loc[metabolite, groupB_cols].astype(float)
    t_stat, p_val = ttest_ind(group1, group2, equal_var=True, nan_policy='omit')
    mean_diff = group1.mean() - group2.mean()
    all_results.append([metabolite, t_stat, p_val, mean_diff])

all_results_df = pd.DataFrame(
    all_results,
    columns=["Metabolite", "T Statistic", "P Value", f"Mean Diff ({groupA} - {groupB})"]
)

# FDR
_, fdr_corrected, _, _ = multipletests(all_results_df["P Value"], method='fdr_bh')
all_results_df["FDR Corrected P Value"] = fdr_corrected

sig_df = all_results_df[all_results_df["P Value"] < P_THRESHOLD].copy()
sig_df = sig_df[
    ["Metabolite", "T Statistic", "P Value", "FDR Corrected P Value", f"Mean Diff ({groupA} - {groupB})"]
].sort_values("P Value")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.6f}".format)

print("=" * 80)
print(f"All metabolites tested (n={len(all_results_df)}), FDR correction applied across all.")
print(f"Showing {len(sig_df)} metabolites with raw p < {P_THRESHOLD}")
print("=" * 80)
print(sig_df.to_string(index=False))

out_path = "FDR and TTest.csv"
all_results_df.sort_values("P Value").to_csv(out_path, index=False)
print(f"\nFull results (all metabolites) saved to: {out_path}")
