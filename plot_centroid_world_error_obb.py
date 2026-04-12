import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── USER PATHS ──────────────────────────────────────────────────────────────
INPUT_JSON  = "./errors_in_centres_with_stats_pred.json" 
OUTPUT_PNG  = "./centroid_world_error.png"                
OUTPUT_PDF  = "./centroid_world_error.pdf"               
# ────────────────────────────────────────────────────────────────────────────

with open(INPUT_JSON) as f:
    data = json.load(f)

errors        = np.array([entry['error_m'] for entry in data['per_image']])
sorted_errors = np.sort(errors)
cdf           = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

fig, ax1 = plt.subplots(figsize=(6, 4.5))

ax1.hist(errors * 100, bins=10, color='steelblue', edgecolor='white', alpha=0.85, label='Error frequency')
ax1.set_xlabel('Centroid World Error (cm)', fontsize=11)
ax1.set_ylabel('Count', fontsize=11, color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_xlim(0, errors.max() * 100 * 1.1)

ax2 = ax1.twinx()
ax2.plot(sorted_errors * 100, cdf * 100, color='darkorange', linewidth=2.5, label='CDF')
ax2.set_ylabel('Cumulative Distribution (%)', fontsize=11, color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')
ax2.set_ylim(0, 105)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d%%'))

p50_val = np.median(errors) * 100
p90_val = np.percentile(errors, 90) * 100

ax2.axvline(p50_val, color='red', linestyle='--', linewidth=1.8)
ax2.text(p50_val + 0.05, 5,  f'p50={p50_val:.2f}cm', fontsize=12, color='red')
ax2.axvline(p90_val, color='red', linestyle=':',  linewidth=1.8)
ax2.text(p90_val + 0.05, 12, f'p90={p90_val:.2f}cm', fontsize=12, color='red')

stats = data['summary_extended']['error_m']
stats_text = (f"Mean: {stats['mean']*100:.2f} cm\n"
              f"Median: {stats['median']*100:.2f} cm\n"
              f"Std: {stats['std']*100:.2f} cm\n"
              f"Max: {stats['max']*100:.2f} cm")
ax1.text(0.97, 0.84, stats_text, transform=ax1.transAxes,
         fontsize=8.5, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='lightgray', alpha=0.9))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

ax1.set_title('Centroid World Error — Blender GT', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_PDF, bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=300)
print("Done — centroid world error plot saved.")
