import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.decomposition import PCA

# THIS IS FOR NORMALIZED DATA WITH SAMPLES AS COLUMNS

# Close all plots and clear any cached state
plt.close('all')

def run_pca_analysis(file_path):
    """Run complete PCA analysis without additional scaling"""

    raw = pd.read_csv(file_path, header=None)

    sample_ids = raw.iloc[0, 1:]
    groups = raw.iloc[1, 1:]

    metabolites = raw.iloc[2:, :]
    metabolites.columns = raw.iloc[0, :]
    metabolites.rename(columns={metabolites.columns[0]: "Metabolite"}, inplace=True)
    metabolites = metabolites.set_index("Metabolite").T
    metabolites = metabolites.apply(pd.to_numeric, errors="coerce")

    # Handle missing values
    for col in metabolites.columns:
        if metabolites[col].isnull().any():
            min_val = metabolites[col].min()
            if pd.notna(min_val) and min_val > 0:
                metabolites[col].fillna(min_val / 2, inplace=True)
            else:
                metabolites[col].fillna(1e-10, inplace=True)

    # Remove metabolites with near-zero variance
    variance_threshold = 1e-10
    variances = metabolites.var(axis=0)
    metabolites = metabolites.loc[:, variances > variance_threshold]
    print(f"After removing near-zero variance: {metabolites.shape}")

    # SD filtering: Keep top 8000 metabolites
    sd_values = metabolites.std(axis=0)
    sd_sorted = sd_values.sort_values(ascending=False)
    n_keep = min(8000, len(sd_sorted))
    top_metabolites = sd_sorted.head(n_keep).index
    metabolites_filtered = metabolites[top_metabolites]
    print(f"After SD filtering (top {n_keep}): {metabolites_filtered.shape}")

    print(f"Using pre-normalized data (no additional scaling)")

    data_for_pca = metabolites_filtered.values

    # Create fresh PCA instance
    pca_model = PCA(n_components=2)
    pca_result = pca_model.fit_transform(data_for_pca)

    pca_df = pd.DataFrame({
        "ID": sample_ids.values,
        "Group": groups.values,
        "PCA1": pca_result[:, 0],
        "PCA2": pca_result[:, 1]
    })

    def plot_confidence_ellipse(x, y, ax, n_std=1.96, edgecolor='black',
                                facecolor='none', alpha=0.2):
        if len(x) < 2:
            return
        cov = np.cov(x, y)
        mean_x, mean_y = np.mean(x), np.mean(y)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        width, height = 2 * n_std * np.sqrt(vals)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        ellipse = Ellipse((0, 0), width, height, angle=angle,
                          edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)
        transform = transforms.Affine2D().translate(mean_x, mean_y) + ax.transData
        ellipse.set_transform(transform)
        ax.add_patch(ellipse)

    plt.rcParams['font.family'] = 'Arial'

    # Make figure wider to accommodate legend while keeping plot square
    fig, ax = plt.subplots(figsize=(7, 6))

    unique_groups = pca_df["Group"].unique()
    colors = dict(zip(unique_groups, ["red", "blue", "green", "purple", "orange"]))

    for group in unique_groups:
        subset = pca_df[pca_df["Group"] == group]
        ax.scatter(subset["PCA1"], subset["PCA2"],
                   color=colors[group], alpha=0.6, label=group)
        plot_confidence_ellipse(
            subset["PCA1"], subset["PCA2"], ax,
            edgecolor=colors[group], facecolor=colors[group], alpha=0.2
        )

    ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("PCA Plot", fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.relim()
    ax.autoscale_view()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Add 10% padding
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    x_padding = x_range * 0.1
    y_padding = y_range * 0.1

    xlim_padded = [xlim[0] - x_padding, xlim[1] + x_padding]
    ylim_padded = [ylim[0] - y_padding, ylim[1] + y_padding]

    x_center = np.mean(xlim_padded)
    y_center = np.mean(ylim_padded)
    max_range = max(xlim_padded[1] - xlim_padded[0], ylim_padded[1] - ylim_padded[0])

    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False)

    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(os.path.dirname(file_path), f"{base_name}_PCA.png")

    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"\nPCA figure saved as:\n{output_path}")
    print(f"{'=' * 60}\n")

    return pca_model, pca_result, pca_df

# Run the analysis
if __name__ == "__main__":
    file_path = "Normalized Data.csv"
    pca_model, pca_result, pca_df = run_pca_analysis(file_path)
