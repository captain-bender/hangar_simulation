import json
import csv
import statistics
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# =============================================================================
# USER PATHS — update these to match your local directory structure
# =============================================================================
OBB_JSON      = r"OBB two-stage.json"
PCA_JSON      = r"PCA-Refined KPP(bias-corrected).json"
RAW_KPP_JSON  = r"raw KPP(v1_err_deg_value).json"
OA_CSV        = r"orient-anything.csv"
OUTPUT_FIG    = r"heading_comparison_boxplot.png"   # change to .png if preferred

# =============================================================================
# STYLE CONFIGURATION — adjust independently from data logic
# =============================================================================
FIGSIZE        = (7.5, 4.8)          # width x height in inches (single-col journal)
DPI            = 300
COLORS         = ["#2E86AB",         # steel-blue  → OBB Two-Stage
                  "#3BB273",         # green       → PCA-Refined KPP
                  "#F4A261"]         # amber       → Orient-Anything v1
MEDIAN_COLOR   = "black"
WHISKER_PCT    = (5, 95)             # whisker extents (percentile, not 1.5×IQR)
FLIER_STYLE    = dict(marker="o", markersize=4, linestyle="none", alpha=0.7)
GRID_ALPHA     = 0.35
FONTSIZE_LABEL = 11
FONTSIZE_TICK  = 10
FONTSIZE_ANNOT = 8.5

# =============================================================================
# LOAD DATA
# =============================================================================
with open(OBB_JSON) as f:
    obb_data = json.load(f)
obb_errors = [p["angular_error_deg"] for p in obb_data["predictions"]]   # 39 values, absolute

with open(PCA_JSON) as f:
    pca_data = json.load(f)
pca_errors = [p["angular_error_deg"] for p in pca_data["predictions"]]   # 40 values, absolute, sim GT

with open(OA_CSV) as f:
    oa_rows = list(csv.DictReader(f))
oa_errors = [float(r["axis_error"]) for r in oa_rows]                    # 40 values, axis error

# =============================================================================
# ASSEMBLE GROUPS
# =============================================================================
groups = [
    ("OBB\nTwo-Stage",           obb_errors),
    ("KPP PCA-Refined\n(bias-corr.)", pca_errors),
    ("Orient-Anything\nv1",      oa_errors),
]
labels = [g[0] for g in groups]
data   = [g[1] for g in groups]

# =============================================================================
# CUSTOM WHISKER COMPUTATION (5th–95th percentile)
# =============================================================================
def box_stats(arr, lo_pct=5, hi_pct=95):
    a = sorted(arr)
    n = len(a)
    q1  = np.percentile(a, 25)
    q3  = np.percentile(a, 75)
    med = np.median(a)
    lo  = np.percentile(a, lo_pct)
    hi  = np.percentile(a, hi_pct)
    fliers = [v for v in a if v < lo or v > hi]
    return dict(med=med, q1=q1, q3=q3, whislo=lo, whishi=hi, fliers=fliers, mean=np.mean(a))

stats = [box_stats(d) for d in data]

# =============================================================================
# PLOT
# =============================================================================
fig, ax = plt.subplots(figsize=FIGSIZE)

positions = np.arange(1, len(groups) + 1)

bp = ax.bxp(
    stats,
    positions=positions,
    widths=0.45,
    patch_artist=True,
    showfliers=True,
    showmeans=True,
    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black",
                   markeredgewidth=1.2, markersize=5, zorder=5),
    medianprops=dict(color=MEDIAN_COLOR, linewidth=1.8),
    whiskerprops=dict(linewidth=1.2, linestyle="--"),
    capprops=dict(linewidth=1.5),
    flierprops=dict(**FLIER_STYLE),
)

# Apply per-box colours and set flier colours to match
for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.72)

for flier, color in zip(bp["fliers"], COLORS):
    flier.set(markerfacecolor=color, markeredgecolor=color)

# Annotate median and mean values above each box
for i, s in enumerate(stats):
    x = positions[i]
    yhi = s["whishi"]
    ax.text(x, yhi + 0.18,
            f"med={s['med']:.2f}°\nmean={s['mean']:.2f}°",
            ha="center", va="bottom", fontsize=FONTSIZE_ANNOT,
            color="dimgray", linespacing=1.35)

# 3° and 5° reference lines
for ref, ls, lbl in [(3.0, "--", "3°"), (5.0, ":", "5°")]:
    ax.axhline(ref, color="gray", linewidth=0.9, linestyle=ls, zorder=0)
    ax.text(len(groups) + 0.62, ref, lbl, va="center", fontsize=FONTSIZE_ANNOT - 0.5,
            color="gray")

# Axes formatting
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=FONTSIZE_TICK)
ax.set_ylabel("Absolute heading error (°)", fontsize=FONTSIZE_LABEL)
ax.set_xlim(0.4, len(groups) + 0.9)
ax.set_ylim(-0.3, max(max(d) for d in data) + 1.6)
ax.yaxis.grid(True, alpha=GRID_ALPHA)
ax.set_axisbelow(True)

# Legend: median bar + mean diamond
legend_elements = [
    mpatches.Patch(facecolor="lightgray", edgecolor="black", label="IQR (box)"),
    plt.Line2D([0], [0], color="black", linewidth=1.8, label="Median"),
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=6, label="Mean"),
    plt.Line2D([0], [0], color="gray", linewidth=0.9, linestyle="--",
               label="Whiskers: 5th–95th pct."),
]
ax.legend(handles=legend_elements, fontsize=FONTSIZE_ANNOT, loc="upper right",
          framealpha=0.85, edgecolor="lightgray")

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=DPI, bbox_inches="tight")
print(f"Saved → {OUTPUT_FIG}")
plt.show()
