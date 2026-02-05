import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pycombat import Combat
import warnings
import os

warnings.filterwarnings('ignore')

# Load data CSV FILE
file_path = "PreComBat.csv"

# Read the full file without headers
df = pd.read_csv(file_path, header=None)

# Extract metadata
sample_ids = df.iloc[0, 1:].values
batch_labels = df.iloc[1, 1:].values
qc_labels = df.iloc[2, 1:].values

# Convert QC labels to int
try:
    qc_labels = qc_labels.astype(int)
except ValueError:
    print("Warning: QC labels contain non-integer values. Converting to int where possible.")
    qc_labels = pd.to_numeric(qc_labels, errors='coerce').fillna(0).astype(int)

# Extract metabolite data (rows 3 onwards)
metabolite_names = df.iloc[3:, 0].values
data = df.iloc[3:, 1:].apply(pd.to_numeric, errors='coerce')
data.index = metabolite_names
data.columns = sample_ids

print(f"Data shape: {data.shape}")
print(f"Batches: {np.unique(batch_labels)}")
print(f"Number of QC samples: {np.sum(qc_labels == 1)}")
print(f"Number of patient samples: {np.sum(qc_labels == 0)}")

# Check for any NaN values
if data.isna().any().any():
    print(f"\nWarning: Found {data.isna().sum().sum()} NaN values in data. Filling with 0.")
    data = data.fillna(0)

# STEP 1: Normalize and Transform Each Batch Separately
print("\n" + "=" * 80)
print("STEP 1: Normalize and Transform Each Batch Separately")
print("=" * 80)

normalized_batches = []
log_transformed_batches = []

for batch in np.unique(batch_labels):
    batch_mask = batch_labels == batch
    batch_data = data.loc[:, batch_mask]

    print(f"\nProcessing {batch}: {batch_data.shape[1]} samples")

    # Sample Normalization (normalize by sum for each sample/column)
    batch_normalized = batch_data.div(batch_data.sum(axis=0), axis=1)

    # Log10 Transformation
    offset = 1e-9
    batch_log = np.log10(batch_normalized + offset)

    normalized_batches.append(batch_normalized)
    log_transformed_batches.append(batch_log)

# Combine all batches back together in original order
log_transformed = pd.concat(log_transformed_batches, axis=1)
log_transformed = log_transformed[sample_ids]  # Ensure original column order

print(f"\nCombined log-transformed data shape: {log_transformed.shape}")

# STEP 2: Autoscale BEFORE ComBat (per metabolite across ALL samples)
print("\n" + "=" * 80)
print("STEP 2: Autoscale BEFORE ComBat")
print("=" * 80)

# Autoscaling (mean-center + divide by STD per metabolite) - BEFORE batch correction
means = log_transformed.mean(axis=1)
stds = log_transformed.std(axis=1)
autoscaled_before = log_transformed.sub(means, axis=0).div(stds, axis=0)

print(f"Autoscaled data shape: {autoscaled_before.shape}")

# STEP 3: Apply ComBat Batch Correction on AUTOSCALED data
print("\n" + "=" * 80)
print("STEP 3: Apply ComBat Batch Correction")
print("=" * 80)

# Initialize Combat
combat = Combat()

# ComBat expects: samples x features (TRANSPOSED!)
autoscaled_before_T = autoscaled_before.T  # Now shape is (99 samples, 12686 metabolites)

print(f"Transposed data shape for ComBat: {autoscaled_before_T.shape}")
print(f"Batch labels shape: {batch_labels.shape}")

# Apply ComBat correction
autoscaled_after_T = combat.fit_transform(autoscaled_before_T.values, batch_labels)

# Transpose back to metabolites x samples
autoscaled_after = autoscaled_after_T.T

# Convert back to DataFrame
autoscaled_after = pd.DataFrame(
    autoscaled_after,
    index=autoscaled_before.index,
    columns=autoscaled_before.columns
)

print("ComBat correction completed!")
print(f"Corrected data shape: {autoscaled_after.shape}")

# STEP 4: PCA Analysis and Visualization
print("\n" + "=" * 80)
print("STEP 4: PCA Analysis and Visualization")
print("=" * 80)

# PCA expects samples x features, so transpose
pca = PCA(n_components=2)

# Before correction
pca_before = pca.fit_transform(autoscaled_before.T)
explained_var_before = pca.explained_variance_ratio_

# After correction
pca_after = pca.fit_transform(autoscaled_after.T)
explained_var_after = pca.explained_variance_ratio_

# Create PCA plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Define colors and markers
unique_batches = np.unique(batch_labels)
batch_colors = {unique_batches[0]: '#1f77b4',
                unique_batches[1]: '#ff7f0e',
                unique_batches[2]: '#2ca02c'}
markers = {0: 'o', 1: '^'}  # circles for patients, triangles for QC
marker_labels = {0: 'Patient', 1: 'QC'}

# Plot Before Correction
ax = axes[0]
for batch in unique_batches:
    for qc in [0, 1]:
        mask = (batch_labels == batch) & (qc_labels == qc)
        if np.any(mask):
            ax.scatter(
                pca_before[mask, 0],
                pca_before[mask, 1],
                c=batch_colors[batch],
                marker=markers[qc],
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidth=1,
                label=f'{batch} - {marker_labels[qc]}'
            )

ax.set_xlabel(f'PC1 ({explained_var_before[0] * 100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({explained_var_before[1] * 100:.1f}%)', fontsize=12)
ax.set_title('(+) Before Batch Correction', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot After Correction
ax = axes[1]
for batch in unique_batches:
    for qc in [0, 1]:
        mask = (batch_labels == batch) & (qc_labels == qc)
        if np.any(mask):
            ax.scatter(
                pca_after[mask, 0],
                pca_after[mask, 1],
                c=batch_colors[batch],
                marker=markers[qc],
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidth=1,
                label=f'{batch} - {marker_labels[qc]}'
            )

ax.set_xlabel(f'PC1 ({explained_var_after[0] * 100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({explained_var_after[1] * 100:.1f}%)', fontsize=12)
ax.set_title('(+) After ComBat Batch Correction', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.splitext(file_path)[0] + "_PCA_BaselineComBat.png", dpi=300, bbox_inches='tight')
print("\nPCA plot saved")
plt.show()

# STEP 5: Save Corrected Data
print("\n" + "=" * 80)
print("STEP 5: Saving Corrected Data")
print("=" * 80)

# Create output in the same format as input
# Row 0: ID, sample_id1, sample_id2, ...
# Row 1: Batch, B1, B2, ...
# Row 2: QC, 0, 1, ...
# Row 3+: metabolite_name, value1, value2, ...

output_rows = []

# Row 0: ID header
row0 = ['ID'] + list(sample_ids)
output_rows.append(row0)

# Row 1: Batch labels
row1 = ['Batch'] + list(batch_labels)
output_rows.append(row1)

# Row 2: QC labels
row2 = ['QC'] + list(qc_labels.astype(int))
output_rows.append(row2)

# Rows 3+: Metabolite data
for metabolite in autoscaled_after.index:
    row = [metabolite] + list(autoscaled_after.loc[metabolite, :].values)
    output_rows.append(row)

# Convert to DataFrame and save
output_df = pd.DataFrame(output_rows)

output_path = file_path.replace('.csv', 'ComBat Corrected.csv')
output_df.to_csv(output_path, index=False, header=False)
print(f"\nBatch-corrected data saved to: {output_path}")

# STEP 6: Summary Statistics
print("\n" + "=" * 80)
print("STEP 6: Summary Statistics - QC Sample Quality")
print("=" * 80)

qc_mask = qc_labels == 1
if np.sum(qc_mask) > 0:
    # Calculate RSD (Relative Standard Deviation) for QC samples
    # Using the non-autoscaled log-transformed data for more interpretable RSD
    qc_before = log_transformed.loc[:, qc_mask]
    qc_after_log = log_transformed.copy()

    # Need to "reverse" the autoscaling on the corrected data to get back to log scale
    # Actually, let's recalculate: apply combat to log-transformed data directly
    print("\nRecalculating with log-transformed data for RSD metrics...")

    combat2 = Combat()
    log_corrected_T = combat2.fit_transform(log_transformed.T.values, batch_labels)
    log_corrected = pd.DataFrame(log_corrected_T.T,
                                 index=log_transformed.index,
                                 columns=log_transformed.columns)

    qc_after = log_corrected.loc[:, qc_mask]

    # Calculate mean and std for QC samples
    qc_mean_before = qc_before.mean(axis=1)
    qc_std_before = qc_before.std(axis=1)
    rsd_before = (qc_std_before / qc_mean_before.abs() * 100)

    qc_mean_after = qc_after.mean(axis=1)
    qc_std_after = qc_after.std(axis=1)
    rsd_after = (qc_std_after / qc_mean_after.abs() * 100)

    # Replace inf values with NaN
    rsd_before = rsd_before.replace([np.inf, -np.inf], np.nan)
    rsd_after = rsd_after.replace([np.inf, -np.inf], np.nan)

    rsd_comparison = pd.DataFrame({
        'Metabolite': metabolite_names,
        'RSD_Before': rsd_before.values,
        'RSD_After': rsd_after.values,
        'Improvement': rsd_before.values - rsd_after.values
    })

    # Remove rows with NaN values
    rsd_comparison = rsd_comparison.dropna()

    # Show statistics
    if len(rsd_comparison) > 0:
        print("\nTop 10 Metabolites with Most Improved QC RSD:")
        print(rsd_comparison.nlargest(10, 'Improvement').to_string(index=False))

        print("\nTop 10 Metabolites with Worst Change in QC RSD:")
        print(rsd_comparison.nsmallest(10, 'Improvement').to_string(index=False))

        print(f"\n{'=' * 80}")
        print(f"Mean QC RSD before correction: {rsd_comparison['RSD_Before'].mean():.2f}%")
        print(f"Mean QC RSD after correction: {rsd_comparison['RSD_After'].mean():.2f}%")
        print(f"Overall improvement: {rsd_comparison['Improvement'].mean():.2f}%")
        print(f"Median QC RSD before correction: {rsd_comparison['RSD_Before'].median():.2f}%")
        print(f"Median QC RSD after correction: {rsd_comparison['RSD_After'].median():.2f}%")

        # Additional statistics
        improved = rsd_comparison[rsd_comparison['Improvement'] > 0]
        worsened = rsd_comparison[rsd_comparison['Improvement'] < 0]
        print(f"\nMetabolites improved: {len(improved)} ({len(improved) / len(rsd_comparison) * 100:.1f}%)")
        print(f"Metabolites worsened: {len(worsened)} ({len(worsened) / len(rsd_comparison) * 100:.1f}%)")

        # Show RSD distribution
        print(f"\nRSD Distribution Before:")
        print(f"  <10%: {np.sum(rsd_comparison['RSD_Before'] < 10)} metabolites")
        print(
            f"  10-20%: {np.sum((rsd_comparison['RSD_Before'] >= 10) & (rsd_comparison['RSD_Before'] < 20))} metabolites")
        print(
            f"  20-30%: {np.sum((rsd_comparison['RSD_Before'] >= 20) & (rsd_comparison['RSD_Before'] < 30))} metabolites")
        print(f"  >30%: {np.sum(rsd_comparison['RSD_Before'] >= 30)} metabolites")

        print(f"\nRSD Distribution After:")
        print(f"  <10%: {np.sum(rsd_comparison['RSD_After'] < 10)} metabolites")
        print(
            f"  10-20%: {np.sum((rsd_comparison['RSD_After'] >= 10) & (rsd_comparison['RSD_After'] < 20))} metabolites")
        print(
            f"  20-30%: {np.sum((rsd_comparison['RSD_After'] >= 20) & (rsd_comparison['RSD_After'] < 30))} metabolites")
        print(f"  >30%: {np.sum(rsd_comparison['RSD_After'] >= 30)} metabolites")
    else:
        print("Not enough valid data for RSD calculation.")
else:
    print("No QC samples found for RSD calculation.")

print("Analysis complete!")
