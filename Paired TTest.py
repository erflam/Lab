# This is for data that has already been NORMALIZED

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

# USER SETTINGS
input_csv = "INPUT DATA.csv"
output_xlsx = "OUTPUT TTEST.xlsx"
p_threshold = 0.0505

# LOAD CSV
df = pd.read_csv(input_csv, header=None, low_memory=False)

# Row 0 = IDs
sample_ids = df.iloc[0, 1:].values

# Row 1 = group labels (e.g., Baseline or Treatment)
group_labels = df.iloc[1, 1:].astype(str).str.strip().values

# Row 2 = subject IDs (to match pairs)
subject_ids = df.iloc[2, 1:].astype(str).str.strip().values

# Identify unique groups
unique_groups = np.unique(group_labels)
print(f"Unique groups found: {unique_groups}")
print(f"Unique subjects: {np.unique(subject_ids)}")

# Validate that we have exactly 2 groups for paired t-test
if len(unique_groups) != 2:
    raise ValueError(f"Paired t-test requires exactly 2 groups, but found {len(unique_groups)}: {unique_groups}")

# Assign group names dynamically
group1, group2 = unique_groups[0], unique_groups[1]
print(f"\nComparing: {group1} vs {group2}")

# Extract metabolite block (row 3 onward)
metabolite_df = df.iloc[3:, :].copy()

# Set metabolite names as index (column 0)
metabolite_df.index = metabolite_df.iloc[:, 0]

# Drop the first column (metabolite name column)
metabolite_df = metabolite_df.iloc[:, 1:]

# Rename columns to sample IDs for tracking
metabolite_df.columns = sample_ids

# Convert all data to numeric (coerce errors to NaN)
metabolite_df = metabolite_df.apply(pd.to_numeric, errors='coerce')

# Create a mapping of each sample to its subject and group
sample_info = pd.DataFrame({
    'Sample': sample_ids,
    'Subject': subject_ids,
    'Group': group_labels
})

print(f"\nTotal samples: {len(sample_info)}")
print(f"{group1} samples: {sum(sample_info['Group'] == group1)}")
print(f"{group2} samples: {sum(sample_info['Group'] == group2)}")

# Run Paired TTest
results = []

for metabolite in metabolite_df.index:
    # Get all values for this metabolite
    metabolite_values = metabolite_df.loc[metabolite]

    # Create a dataframe with metabolite values, subjects, and groups
    temp_df = pd.DataFrame({
        'Sample': sample_ids,
        'Subject': subject_ids,
        'Group': group_labels,
        'Value': metabolite_values.values
    })

    # Pivot to get both groups side by side for each subject
    pivot_df = temp_df.pivot(index='Subject', columns='Group', values='Value')

    # Check if both groups exist
    if group1 not in pivot_df.columns or group2 not in pivot_df.columns:
        continue

    # Drop subjects with missing data in either condition
    complete_pairs = pivot_df.dropna()

    if len(complete_pairs) < 2:  # Need at least 2 pairs
        continue

    group1_vals = complete_pairs[group1].values
    group2_vals = complete_pairs[group2].values

    # Perform paired t-test (group2 - group1)
    t_stat, p_val = ttest_rel(group2_vals, group1_vals)

    if p_val < p_threshold:
        results.append({
            'Metabolite': metabolite,
            'P_Value': p_val,
            'T_Statistic': t_stat,
            'N_Pairs': len(complete_pairs),
            'Comparison': f'{group2} vs {group1}'
        })

# Save Results
if len(results) > 0:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('P_Value')
    results_df.to_excel(output_xlsx, index=False)

    print(f"\n✓ Analysis complete!")
    print(f"✓ Found {len(results_df)} metabolites with p-value < {p_threshold}")
    print(f"✓ Comparison: {group2} vs {group1}")
    print(f"✓ Results saved to: {output_xlsx}")
