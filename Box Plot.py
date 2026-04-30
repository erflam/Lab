import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os

plt.rcParams['font.family'] = 'Arial'

# Load the data
file_path = "Box Plot.csv"
df = pd.read_csv(file_path)

# First column is the group column
group_col = df.columns[0]
group_names = df[group_col].unique()

# All other columns are metabolites
metabolite_cols = df.columns[1:]

# Colors
color_group1    = "#8E44AD"     # color for metabolites elevated in group1
color_group2    = "#FFD700"     # color for metabolites elevated in group2

# Directory to save individual plots
out_dir = 'Desktop/Box Plots'
os.makedirs(out_dir, exist_ok=True)

for metabolite in metabolite_cols:
    data = df[[group_col, metabolite]].copy()
    data[metabolite] = pd.to_numeric(data[metabolite], errors='coerce')
    data = data.dropna()

    group1 = data[data[group_col] == group_names[0]][metabolite]
    group2 = data[data[group_col] == group_names[1]][metabolite]

    # Skip if not enough data
    if len(group1) < 2 or len(group2) < 2:
        print(f"Skipping {metabolite} due to insufficient data.")
        continue

    # t-test
    t_stat, p_val = ttest_ind(group1, group2, equal_var=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(3.5, 5))

    sns.boxplot(
        x=group_col,
        y=metabolite,
        hue=group_col,
        data=data,
        ax=ax,
        palette=[color_group1, color_group2],
        dodge=False,
        legend=False,
        showfliers=False,
        linewidth=1,
        saturation=1.0
    )

    # Title and axis labels — bold, Arial, matching volcano style
    ax.set_title(metabolite, fontsize=12, fontname='Arial', fontweight='bold')
    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('Concentration (ng/mL)', fontsize=8, fontname='Arial', fontweight='bold')

    # Tick formatting to match volcano plot
    # Tick formatting to match volcano plot
    ax.tick_params(axis='both', which='both', width=1, length=3)
    ax.tick_params(axis='y', which='both', width=1, length=3, labelsize=7)
    ax.tick_params(axis='x', which='both', width=1, length=3, labelsize=10)

    # Set font properties
    for label in ax.get_xticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(10)  # group label size
        label.set_fontweight('bold')  # optional

    for label in ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(7)

    # Spines — match volcano (no top/right, bottom/left linewidth=1)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1)

    # Add vertical padding for p-value label
    ymin, ymax = ax.get_ylim()
    pad = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin, ymax + pad)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))

    # p-value label
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
    png_path = os.path.join(out_dir, f'{safe_name}_boxplot.png')
    svg_path = os.path.join(out_dir, f'{safe_name}_boxplot.svg')

    fig.savefig(png_path, dpi=1200, pad_inches=0.05, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', pad_inches=0.05, bbox_inches='tight')
    plt.close(fig)

print(f"Saved {len(metabolite_cols)} individual boxplots (PNG + SVG) to\n{out_dir}")
