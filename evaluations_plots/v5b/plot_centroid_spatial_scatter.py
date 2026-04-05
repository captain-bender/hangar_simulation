import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

# ── USER PATHS ────────────────────────────────────────────────────────────────
INPUT_JSON  = "./errors_in_centres_with_stats_pred.json"  # <-- update
OUTPUT_PNG  = "./centroid_spatial_scatter.png"            # <-- update
OUTPUT_PDF  = "./centroid_spatial_scatter.pdf"            # <-- update
# ─────────────────────────────────────────────────────────────────────────────

# ── STYLE CONFIGURATION ───────────────────────────────────────────────────────
FIGURE_SIZE          = (7, 6)

# Scatter points
POINT_SIZE           = 80         # marker size
POINT_ALPHA          = 0.85       # transparency
POINT_MARKER         = 'o'
POINT_EDGECOLOR      = 'white'
POINT_LINEWIDTH      = 0.5

# Colourmap — sequential, perceptually uniform
CMAP                 = 'YlOrRd'   # yellow->orange->red; low error=yellow, high=red

# GT point style
GT_MARKER            = '+'
GT_COLOR             = 'steelblue'
GT_SIZE              = 60
GT_LINEWIDTH         = 1.2
GT_ALPHA             = 0.5
SHOW_GT              = True       # set False to hide GT points

# Connecting lines between pred and GT
SHOW_LINES           = False      # set True to draw a line from pred to GT per point

# Workspace boundary (metres) — update to match your hangar FOV
WORKSPACE_X          = (0, 9)     # x axis limits in metres
WORKSPACE_Y          = (0, 14.2)  # y axis limits in metres

# Colourbar
CBAR_LABEL           = 'Centroid World Error (cm)'

# Axis labels and title
XLABEL               = 'X Position (m)'
YLABEL               = 'Y Position (m)'
TITLE                = 'Predicted Centroid World Error — Spatial Distribution'
TITLE_FONTSIZE       = 12
LABEL_FONTSIZE       = 11
TICK_FONTSIZE        = 9
LEGEND_FONTSIZE      = 9

# Grid
GRID_ALPHA           = 0.3
GRID_LINESTYLE       = '--'
# ─────────────────────────────────────────────────────────────────────────────

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with open(INPUT_JSON) as f:
    data = json.load(f)

entries = data['per_image']

pred_x  = np.array([e['pred_xy_used']['x'] for e in entries])
pred_y  = np.array([e['pred_xy_used']['y'] for e in entries])
gt_x    = np.array([e['gt_xy']['x'] for e in entries])
gt_y    = np.array([e['gt_xy']['y'] for e in entries])
errors  = np.array([e['error_m'] for e in entries]) * 100  # convert to cm

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGURE_SIZE)

# Optional connecting lines
if SHOW_LINES:
    for px, py, gx, gy in zip(pred_x, pred_y, gt_x, gt_y):
        ax.plot([gx, px], [gy, py], color='gray', linewidth=0.5, alpha=0.4, zorder=1)

# GT points
if SHOW_GT:
    ax.scatter(gt_x, gt_y,
               marker=GT_MARKER, s=GT_SIZE, color=GT_COLOR,
               linewidths=GT_LINEWIDTH, alpha=GT_ALPHA,
               label='Ground truth', zorder=2)

# Predicted points coloured by error
norm = mcolors.Normalize(vmin=errors.min(), vmax=errors.max())
cmap = plt.get_cmap(CMAP)

sc = ax.scatter(pred_x, pred_y,
                c=errors, cmap=CMAP, norm=norm,
                s=POINT_SIZE, alpha=POINT_ALPHA,
                marker=POINT_MARKER,
                edgecolors=POINT_EDGECOLOR,
                linewidths=POINT_LINEWIDTH,
                label='Predicted centroid',
                zorder=3)

# Colourbar
cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label(CBAR_LABEL, fontsize=LABEL_FONTSIZE)
cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

# Axes
ax.set_xlim(WORKSPACE_X)
ax.set_ylim(WORKSPACE_Y)
ax.set_xlabel(XLABEL, fontsize=LABEL_FONTSIZE)
ax.set_ylabel(YLABEL, fontsize=LABEL_FONTSIZE)
ax.set_title(TITLE, fontsize=TITLE_FONTSIZE, fontweight='bold')
ax.tick_params(labelsize=TICK_FONTSIZE)
ax.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
ax.set_aspect('equal')

# Stats annotation
stats      = data['summary_extended']['error_m']
stats_text = (f"Mean:   {stats['mean']*100:.2f} cm\n"
              f"Median: {stats['median']*100:.2f} cm\n"
              f"Max:    {stats['max']*100:.2f} cm\n"
              f"All within 5 cm: 100%")
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=8.5, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='lightgray', alpha=0.9))

if SHOW_GT:
    ax.legend(fontsize=LEGEND_FONTSIZE, loc='lower right')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=300)
print("Done — spatial scatter plot saved.")
