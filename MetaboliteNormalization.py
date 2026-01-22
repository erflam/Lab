import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


# USER INPUT + OUTPUT FILES
input_file = "Grouping.csv"

output_normalized_csv = "Normalized.csv"
output_plot = "Norm.png"
output_significant_excel = "TTest.xlsx"


# 1. Load CSV
df = pd.read_csv(input_file, header=None)

# Structure:
# Row0 = metabolite ID label row
# Row1 = group row
# Row2+ = metabolite values

metabolite_ids = df.iloc[0, 1:]      # sample names start column 1
groups = df.iloc[1, 1:]
data = df.iloc[2:, :]                # actual numeric data

# Rename columns (first column is metabolite name)
data.columns = df.iloc[0, :]         # use the "ID" row for column names

# Set metabolite names as index
data.index = df.iloc[2:, 0]

# Drop the first column (ID column) and keep only sample columns
data = data.iloc[:, 1:]

# Convert to numeric
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

# 5. Reassemble output table (add back ID + Group rows)
out_df = pd.DataFrame(index=data.index, columns=data.columns, data=autoscaled)

out_df.loc["Group"] = groups.values
out_df.loc["ID"] = metabolite_ids.values

out_df = out_df.loc[["ID", "Group"] + [idx for idx in data.index]]

# 6. Save to CSV
out_df.to_csv(output_normalized_csv)
print("Saved normalized file to:", output_normalized_csv)

# 7. Plots
# Flatten values for density plots
before_vals = data.values.flatten()
after_vals = autoscaled.values.flatten()

# Limit boxplots to first 50 metabolites
max_features = 50
subset_before = data.iloc[:max_features, :]
subset_after = autoscaled.iloc[:max_features, :]

# Truncate metabolite labels
short_labels = [str(met)[:15] for met in subset_before.index]

fig, axes = plt.subplots(
    2, 2,
    figsize=(20, 16),
    gridspec_kw={"height_ratios": [1, 3]}
)

# BEFORE — Density Plot (your KDE) with x-axis starting at 0
pd.Series(before_vals).plot(kind="kde", ax=axes[0, 0])
axes[0, 0].set_title("Density Plot – Before Normalization")
axes[0, 0].set_xlabel("Intensity")
axes[0, 0].set_ylabel("Density")
axes[0, 0].set_xlim(left=0)     # Force axis start at 0

# BEFORE — Vertical Boxplots
axes[1, 0].boxplot(
    subset_before.values.T,
    vert=False,
    tick_labels=short_labels,
    patch_artist=True,
    boxprops=dict(facecolor="#99cc99")
)
axes[1, 0].set_title("Before Normalization (up to 50 metabolites)")
axes[1, 0].set_xlabel("Intensity")

# AFTER — Density Plot
pd.Series(after_vals).plot(kind="kde", ax=axes[0, 1])
axes[0, 1].set_title("Density Plot – After Normalization")
axes[0, 1].set_xlabel("Intensity")
axes[0, 1].set_ylabel("Density")

# AFTER — Vertical Boxplots
axes[1, 1].boxplot(
    subset_after.values.T,
    vert=False,
    tick_labels=short_labels,
    patch_artist=True,
    boxprops=dict(facecolor="#99cc99")
)
axes[1, 1].set_title("After Normalization (up to 50 metabolites)")
axes[1, 1].set_xlabel("Normalized Intensity")

plt.tight_layout()

# SAVE PLOT
plt.savefig(output_plot, dpi=300)
print("Saved normalization plot as:", output_plot)

# Run t-tests on normalized data and export significant metabolites

# Extract groups from the original file
group_labels = groups.values     # row 1 in your CSV

# Unique group names
unique_groups = np.unique(group_labels)
if len(unique_groups) != 2:
    raise ValueError("Exactly two groups are required for t-test analysis.")

groupA, groupB = unique_groups

# Identify sample columns belonging to each group
groupA_cols = [col for col, grp in zip(data.columns, group_labels) if grp == groupA]
groupB_cols = [col for col, grp in zip(data.columns, group_labels) if grp == groupB]

# Prepare output list
results = []

# Loop through each metabolite and run t-test
for metabolite in autoscaled.index:
    group1 = autoscaled.loc[metabolite, groupA_cols].astype(float)
    group2 = autoscaled.loc[metabolite, groupB_cols].astype(float)

    t_stat, p_val = ttest_ind(group1, group2, equal_var=True, nan_policy='omit')

    # Keep only significant metabolites
    if p_val < 0.104:
        results.append([metabolite, p_val])

# Convert to DataFrame
ttest_df = pd.DataFrame(results, columns=["Metabolite", "P Value"])

# Save as Excel file
ttest_df.to_excel(output_significant_excel, index=False)
print("Saved significant metabolite list to:", output_significant_excel)
