"""
Figure for Section 5.1 – KPP Chassis Centroid Localisation Accuracy
Panel (a): Histogram + CDF of pixel-level centroid errors (pred kp1 vs annotated kp1)
Panel (b): Quiver plot of world-frame errors (pred kp1 vs Blender GT)

Adjust the STYLE CONFIG block to match paper aesthetics.
All statistics are hardcoded from the evaluation outputs.
"""

import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# =============================================================================
# STYLE CONFIG — adjust here without touching plot logic
# =============================================================================
FONT_FAMILY     = "serif"
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK  = 9
FONT_SIZE_ANNOT = 8.5
FONT_SIZE_LEGEND= 9

BAR_COLOR       = "#4878CF"   # histogram bars
CDF_COLOR       = "#E8800A"   # CDF line
QUIVER_COLOR    = "#4878CF"   # error arrows
GT_COLOR        = "#333333"   # GT marker colour
PERCENTILE_COLOR= "red"       # percentile marker lines

FIG_WIDTH       = 10.0        # inches
FIG_HEIGHT      = 4.2         # inches
DPI             = 300

# Paths to data files — update if needed
PIXELS_JSON = Path("./evaluation/v1-pose/centers_by_image.json")
GT_STATS_JSON = Path("./evaluation/v1-pose/errors_in_centres_with_stats_gt.json")

# =============================================================================
# DATA LOADING
# =============================================================================

# --- Panel (a): pixel errors computed from centers_by_image.json ---
raw = json.loads(PIXELS_JSON.read_text())
pixel_errors = []
for row in raw:
    pred = row["pred_centers"]
    gt   = row["gt_centers"]
    if pred and gt:
        dx = pred[0]["x"] - gt[0]["x"]
        dy = pred[0]["y"] - gt[0]["y"]
        pixel_errors.append(math.sqrt(dx**2 + dy**2))

pixel_errors = np.array(sorted(pixel_errors))
n = len(pixel_errors)

# Hardcoded summary stats (from earlier computation)
px_mean   = 2.45
px_median = 2.24
px_std    = 1.14
px_max    = 5.29
px_p90    = 4.43
px_p95    = 5.15

# --- Panel (b): world-frame quiver data from GT stats json ---
gt_data = json.loads(GT_STATS_JSON.read_text())
per_image = gt_data["per_image"]

gt_x, gt_y, dx_list, dy_list, err_list = [], [], [], [], []
for row in per_image:
    gx = row["gt_xy"]["x"]
    gy = row["gt_xy"]["y"]
    px = row["pred_xy_used"]["x"]
    py = row["pred_xy_used"]["y"]
    err = row["error_m"]
    gt_x.append(gx)
    gt_y.append(gy)
    dx_list.append(px - gx)
    dy_list.append(py - gy)
    err_list.append(err * 100)  # convert to cm

gt_x    = np.array(gt_x)
gt_y    = np.array(gt_y)
dx_arr  = np.array(dx_list)          # metres
dy_arr  = np.array(dy_list)          # metres
err_arr = np.array(err_list)         # cm (only used for colouring)

# =============================================================================
# FIGURE
# =============================================================================
plt.rcParams.update({
    "font.family":  FONT_FAMILY,
    "font.size":    FONT_SIZE_LABEL,
    "axes.labelsize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "legend.fontsize": FONT_SIZE_LEGEND,
})

# ---------------------------------------------------------------------------
# Panel (a) — Histogram + CDF
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(5.5, 4.2))
ax1r = ax1.twinx()

bins = np.linspace(0, math.ceil(px_max) + 0.5, 14)
counts, edges = np.histogram(pixel_errors, bins=bins)
ax1.bar(edges[:-1], counts, width=np.diff(edges), color="steelblue",
        edgecolor="white", alpha=0.85, align="edge", label="Error frequency")

# CDF
cdf = np.arange(1, n + 1) / n * 100
ax1r.plot(pixel_errors, cdf, color=CDF_COLOR, linewidth=1.8, label="CDF")
ax1r.set_ylim(0, 110)
ax1r.set_ylabel("Cumulative Distribution (%)", color=CDF_COLOR,
                fontsize=FONT_SIZE_LABEL)
ax1r.tick_params(axis="y", labelcolor=CDF_COLOR, labelsize=FONT_SIZE_TICK)

# Percentile markers
for pval, plabel in [(px_median, f"p50={px_median:.2f}px"),
                     (px_p90,   f"p90={px_p90:.2f}px")]:
    ax1.axvline(pval, color=PERCENTILE_COLOR, linestyle="--", linewidth=1.0)
    ax1.text(pval + 0.05, ax1.get_ylim()[1] * 0.02, plabel,
             color=PERCENTILE_COLOR, fontsize=12, rotation=90,
             va="bottom")

# Stats annotation box
stats_txt = (f"Mean:   {px_mean:.2f} px\n"
             f"Median: {px_median:.2f} px\n"
             f"Std:      {px_std:.2f} px\n"
             f"Max:    {px_max:.2f} px")
ax1.text(0.97, 0.84, stats_txt, transform=ax1.transAxes,
         fontsize=FONT_SIZE_ANNOT, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

ax1.set_xlabel("Centroid Pixel Error (px)")
ax1.set_ylabel("Count")
ax1.set_xlim(left=0)
ax1.grid(True, linestyle="--", alpha=0.4)
# ax1.set_title("Centroid Pixel Error — Annotated vs Predicted", fontsize=12, fontweight="bold")

# Combined legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax1r.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=FONT_SIZE_LEGEND,
           framealpha=0.8)

# ---------------------------------------------------------------------------
# Panel (b) — Scatter plot of world-frame errors coloured by magnitude
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(5.5, 4.2))

cmap = plt.cm.YlOrRd
norm = plt.Normalize(vmin=0, vmax=err_arr.max())

sc = ax2.scatter(gt_x, gt_y, c=err_arr, cmap=cmap, norm=norm,
                 s=40, zorder=5, edgecolors="grey", linewidths=0.4)

# GT positions as cross markers underneath
ax2.scatter(gt_x, gt_y, s=18, color=GT_COLOR, zorder=4,
            marker="+", linewidths=0.8)

cbar = fig2.colorbar(sc, ax=ax2, pad=0.02)
cbar.set_label("World Error (cm)", fontsize=FONT_SIZE_LABEL)
cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

# Annotate mean and max
stats_txt2 = (f"Mean:   {0.27:.2f} cm\n"
              f"Median: {0.22:.2f} cm\n"
              f"Max:    {0.70:.2f} cm")
ax2.text(0.5, 0.97, stats_txt2, transform=ax2.transAxes,
         fontsize=FONT_SIZE_ANNOT, va="top", ha="center",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

ax2.set_xlabel("X Position (m)")
ax2.set_ylabel("Y Position (m)")
ax2.set_xlim(0, 9)
ax2.set_ylim(0, 14.5)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path_1 = Path("./figure_51a_centroid_histogram.png")
fig1.savefig(out_path_1, dpi=DPI, bbox_inches="tight")
print(f"Saved: {out_path_1}")

out_path_2 = Path("./figure_51b_centroid_scatter.png")
fig2.savefig(out_path_2, dpi=DPI, bbox_inches="tight")
print(f"Saved: {out_path_2}")

plt.close("all")