import json
import math
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
IN_JSON = Path("./evaluation/v1-pose/errors_in_centres_gt.json")
OUT_JSON = Path("./evaluation/v1-pose/errors_in_centres_with_stats_gt.json")

# thresholds in meters for "success rate"
THRESHOLDS_M = [0.01, 0.02, 0.05, 0.10]  # 1cm, 2cm, 5cm, 10cm

# trimmed mean settings
TRIM_FRACTION = 0.10  # 10% trimmed mean (drop lowest/highest 10%)

# inlier RMSE settings
INLIER_MAX_M = 0.05   # RMSE computed only over errors <= 5cm


# -----------------------------
# Helpers
# -----------------------------
def percentile(sorted_vals, p):
    """Linear interpolation percentile. sorted_vals must be sorted."""
    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))

def mean(xs):
    return (sum(xs) / len(xs)) if xs else None

def std(xs):
    if not xs:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)

def rmse(xs):
    if not xs:
        return None
    return math.sqrt(sum(x * x for x in xs) / len(xs))

def trimmed_mean(sorted_vals, trim_fraction):
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    k = int(math.floor(n * trim_fraction))
    if 2 * k >= n:
        return None
    trimmed = sorted_vals[k:n - k]
    return float(sum(trimmed) / len(trimmed))

def safe_get(d, path, default=None):
    """
    path like ["local_point_xy","x"]
    """
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

# -----------------------------
# Main
# -----------------------------
def main():
    payload = json.loads(IN_JSON.read_text(encoding="utf-8"))
    per_image = payload.get("per_image", [])

    errors = []
    dxs = []
    dys = []
    chosen_counts = {"local_point": 0, "dataset_xy_m": 0, "other": 0}

    for row in per_image:
        err = row.get("error_m", None)
        if err is None:
            continue
        err = float(err)
        errors.append(err)

        chosen = row.get("chosen", "other")
        if chosen in chosen_counts:
            chosen_counts[chosen] += 1
        else:
            chosen_counts["other"] += 1

        # signed dx/dy from the actually used prediction
        gt = row.get("gt_xy", {})
        pred_used = row.get("pred_xy_used", {})
        if isinstance(gt, dict) and isinstance(pred_used, dict):
            if "x" in gt and "y" in gt and "x" in pred_used and "y" in pred_used:
                dxs.append(float(pred_used["x"]) - float(gt["x"]))
                dys.append(float(pred_used["y"]) - float(gt["y"]))

    errors_sorted = sorted(errors)
    n = len(errors_sorted)

    # success rates
    success_rates = {}
    for t in THRESHOLDS_M:
        if n == 0:
            success_rates[f"<= {t} m"] = None
        else:
            success_rates[f"<= {t} m"] = float(sum(1 for e in errors_sorted if e <= t) / n)

    # inlier set
    inliers = [e for e in errors_sorted if e <= INLIER_MAX_M]
    inlier_rmse = rmse(inliers)
    inlier_mean = mean(inliers)
    inlier_frac = (len(inliers) / n) if n else None

    # main summary
    summary = {
        "matched": n,
        "chosen_counts": chosen_counts,

        "error_m": {
            "mean": mean(errors_sorted),
            "median": percentile(errors_sorted, 50),
            "std": std(errors_sorted),
            "rmse": rmse(errors_sorted),
            "min": float(errors_sorted[0]) if n else None,
            "max": float(errors_sorted[-1]) if n else None,
            "p90": percentile(errors_sorted, 90),
            "p95": percentile(errors_sorted, 95),
            "p99": percentile(errors_sorted, 99),
            "trimmed_mean_10pct": trimmed_mean(errors_sorted, TRIM_FRACTION),
        },

        "success_rate": success_rates,

        "inliers_(<=5cm)": {
            "threshold_m": INLIER_MAX_M,
            "fraction": inlier_frac,
            "mean_error_m": inlier_mean,
            "rmse_m": inlier_rmse,
            "max_error_m": float(max(inliers)) if inliers else None,
        },

        "signed_error_used_pred_minus_gt_m": {
            "mean_dx": mean(dxs),
            "mean_dy": mean(dys),
            "std_dx": std(dxs),
            "std_dy": std(dys),
        }
    }

    out = {
        "meta": payload.get("meta", {}),
        "summary_original": payload.get("summary", {}),
        "summary_extended": summary,
        "per_image": per_image,  # keep as-is
    }

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote:", OUT_JSON)
    print("Extended summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
