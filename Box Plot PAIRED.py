import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import os

#THIS IS FOR A PAIRED ANALYSIS

plt.rcParams['font.family'] = 'Arial'

# Load the data
file_path = "BP Paired.csv"
df = pd.read_csv(file_path)

# Column setup
subject_col = "Subject"
group_col = "Group"
group_names = df[group_col].unique()

# All other columns are metabolites
metabolite_cols = [c for c in df.columns if c not in [subject_col, group_col]]

# Colors
color_group1 = "#1E40AF"
color_group2 = "#2ca02c"

# Directory to save individual plots
out_dir = 'Desktop/Box Plots'
os.makedirs(out_dir, exist_ok=True)

for metabolite in metabolite_cols:
    data = df[[subject_col, group_col, metabolite]].copy()
    data[metabolite] = pd.to_numeric(data[metabolite], errors='coerce')
    data = data.dropna()

    group1_df = data[data[group_col] == group_names[0]].set_index(subject_col)
    group2_df = data[data[group_col] == group_names[1]].set_index(subject_col)

    # Keep only subjects present in BOTH groups (valid pairs)
    common_subjects = group1_df.index.intersection(group2_df.index)

    if len(common_subjects) < 2:
        print(f"Skipping {metabolite}: fewer than 2 paired subjects.")
        continue

    group1 = group1_df.loc[common_subjects, metabolite]
    group2 = group2_df.loc[common_subjects, metabolite]

    # Paired t-test
    t_stat, p_val = ttest_rel(group1, group2)

    # Plotting
    plot_data = data[data[subject_col].isin(common_subjects)]
    fig, ax = plt.subplots(figsize=(3.5, 5))

    sns.boxplot(
        x=group_col,
        y=metabolite,
        hue=group_col,
        data=plot_data,
        ax=ax,
        palette=[color_group1, color_group2],
        dodge=False,
        legend=False,
        showfliers=False,
        linewidth=1,
        saturation=1.0
    )

    # Title and axis labels
    ax.set_title(metabolite, fontsize=12, fontname='Arial', fontweight='bold')
    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('Concentration (uM)', fontsize=8, fontname='Arial', fontweight='bold')

    # Tick formatting
    ax.tick_params(axis='both', which='both', width=1, length=3)
    ax.tick_params(axis='y', which='both', width=1, length=3, labelsize=7)
    ax.tick_params(axis='x', which='both', width=1, length=3, labelsize=10)

    for label in ax.get_xticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(10)
        label.set_fontweight('bold')

    for label in ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(7)

    # Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1)

    # Padding and p-value label
    ymin, ymax = ax.get_ylim()
    pad = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin, ymax + pad)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))

    p_label = f"p = {p_val:.1g}"
    ax.text(
        0.5, 0.97, p_label,
        transform=ax.transAxes,
        ha='center', va='top',
        fontsize=10, fontname='Arial', color='black',
    )

    fig.set_size_inches(2.5, 3)
    plt.tight_layout(pad=0.6)

    # Save PNG and SVG
    safe_name = metabolite.replace(' ', '_').replace('/', '_')
    png_path = os.path.join(out_dir, f'{safe_name}_Base_boxplot.png')
    svg_path = os.path.join(out_dir, f'{safe_name}_Base_boxplot.svg')

    fig.savefig(png_path, dpi=1200, pad_inches=0.05, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', pad_inches=0.05, bbox_inches='tight')
    plt.close(fig)

print(f"Saved {len(metabolite_cols)} individual boxplots (PNG + SVG) to\n{out_dir}")
