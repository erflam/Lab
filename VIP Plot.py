import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import matplotlib.patches as mpatches

# --- Input/Output Paths ---
INPUT_CSV       = "VIP.csv"
OUTPUT_PNG      = "VIP.png"
OUTPUT_XLSX     = "VIP Output.xlsx"

# --- Plot Options ---
SHOW_PVALS      = True          # True = show p-value labels on bars

# --- Figure Size ---
FIG_WIDTH           = 11        # inches
ROW_HEIGHT          = 0.4       # inches per metabolite — adjust to match your 10-metabolite look
FIG_HEIGHT_MINIMUM  = 3         # inches — prevents tiny figures for very few metabolites
TIGHT_PAD       = 0.6
SAVE_DPI        = 600
SAVE_PAD_INCHES = 0.05

# --- Colors ---
COLOR_GROUP1    = "#1e40af"     # color for metabolites elevated in group1
COLOR_GROUP2    = "#d62728"     # color for metabolites elevated in group2

# --- Font Sizes ---
FONTSIZE_AXIS_LABEL   = 12
FONTSIZE_X_TICK       = 10
FONTSIZE_Y_TICK       = 10
FONTSIZE_PVAL_LABEL   = 8
FONTSIZE_LEGEND       = 10
FONTSIZE_LEGEND_TITLE = 12

# --- Fonts ---
FONT_MAIN       = "Arial"
FONT_ALPHA      = "Arial Unicode MS"

# --- Legend ---
LEGEND_TITLE    = "Lean T1D Vs Obese T1D"
LEGEND_LOC      = "lower right"

# --- Axis / Layout ---
YLABEL_X_COORD  = -0.28         # horizontal position of y-axis label
PVAL_OFFSET     = 0.075         # horizontal offset for p-value text from bar end
TICK_WIDTH      = 1
TICK_LENGTH     = 3

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

print("\nUnique groups found in file:")
print(df['Group'].unique())
print("\nGroup counts (including NaN if any):")
print(df['Group'].value_counts(dropna=False))

groups = df['Group'].unique()
if len(groups) != 2:
    raise ValueError(f"Expected exactly 2 groups, but found {len(groups)}: {groups}")

group1, group2 = groups
print(f"\nGroups used for comparison: {group1} and {group2}")

metabolites = df.columns[1:]

results = []
for met in metabolites:
    vals1 = df[df['Group'] == group1][met].dropna()
    vals2 = df[df['Group'] == group2][met].dropna()
    mean1, mean2 = vals1.mean(), vals2.mean()

    _, pval = ttest_ind(vals1, vals2, equal_var=True, nan_policy="omit")

    signed_sig = -np.log10(pval)
    if mean2 > mean1:
        signed_sig = -signed_sig

    results.append([met, signed_sig, pval, mean1, mean2])

vip_df = pd.DataFrame(results, columns=["Metabolite", "signed_log10p", "p-value", f"{group1}_mean", f"{group2}_mean"])
vip_df = vip_df.sort_values("signed_log10p", ascending=True)
vip_df["color"] = vip_df["signed_log10p"].apply(lambda val: COLOR_GROUP1 if val > 0 else COLOR_GROUP2)

def make_plot(show_pvals=SHOW_PVALS, filename=OUTPUT_PNG):
    fig_height = max(FIG_HEIGHT_MINIMUM, len(vip_df) * ROW_HEIGHT)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_height))

    bars = ax.barh(vip_df["Metabolite"], vip_df["signed_log10p"], color=vip_df["color"])

    ax.axvline(0, color="black", lw=0)

    ax.set_xlabel("± -log10(p-value)", fontsize=FONTSIZE_AXIS_LABEL, fontname=FONT_MAIN, fontweight="bold")
    ax.set_ylabel("Metabolite", fontsize=FONTSIZE_AXIS_LABEL, fontname=FONT_MAIN, fontweight="bold", labelpad=8)
    ax.yaxis.set_label_coords(YLABEL_X_COORD, 0.5)

    ax.tick_params(axis="both", which="both", width=TICK_WIDTH, length=TICK_LENGTH)
    plt.xticks(fontsize=FONTSIZE_X_TICK, fontname=FONT_MAIN, fontweight="bold")

    ax.set_yticks(range(len(vip_df)))
    for i, met in enumerate(vip_df["Metabolite"]):
        font = FONT_ALPHA if '⍺' in met else FONT_MAIN
        ax.text(-0.01, i, met,
                transform=ax.get_yaxis_transform(),
                fontsize=FONTSIZE_Y_TICK, fontname=font,
                fontweight="bold", va="center", ha="right")
    ax.set_yticklabels([])  # hide the original tick labels

    if show_pvals:
        for bar, pval in zip(bars, vip_df["p-value"]):
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            label = f"p = {pval:.1g}"
            if width > 0:
                x, ha = -PVAL_OFFSET, "right"
            else:
                x, ha = +PVAL_OFFSET, "left"
            ax.text(x=x, y=y, s=label, va="center", ha=ha,
                    fontsize=FONTSIZE_PVAL_LABEL, fontname=FONT_MAIN, color="black")

    up_patch   = mpatches.Patch(color=COLOR_GROUP1, label=f"Elevated in {group1}")
    down_patch = mpatches.Patch(color=COLOR_GROUP2, label=f"Elevated in {group2}")
    legend = ax.legend(handles=[up_patch, down_patch],
                       title=LEGEND_TITLE, loc=LEGEND_LOC,
                       frameon=False, fontsize=FONTSIZE_LEGEND, title_fontsize=FONTSIZE_LEGEND_TITLE)

    for text in legend.get_texts():
        text.set_fontname(FONT_MAIN)
    legend.get_title().set_fontname(FONT_MAIN)
    legend.get_title().set_fontweight("bold")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(1)

    plt.tight_layout(pad=TIGHT_PAD)
    ax.set_ylim(-0.5, len(vip_df) - 0.5)

    plt.savefig(filename, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    print(f"Saved {filename}")

    svg_filename = filename.rsplit(".", 1)[0] + ".svg"
    plt.savefig(svg_filename, format="svg", pad_inches=SAVE_PAD_INCHES)
    print(f"Saved {svg_filename}")

    plt.close(fig)

make_plot()

export_df = vip_df[["Metabolite", "p-value", "signed_log10p"]].copy()
export_df["Elevated in"] = export_df["signed_log10p"].apply(lambda val: group1 if val > 0 else group2)
export_df = export_df.drop(columns=["signed_log10p"])
export_df = export_df.sort_values("p-value", ascending=True)
export_df.to_excel(OUTPUT_XLSX, index=False)
print(f"Saved {OUTPUT_XLSX}")
