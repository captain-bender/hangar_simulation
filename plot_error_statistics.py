import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_series(payload: dict):
    """
    Returns:
      errors_m: list[float]
      dx_m: list[float]
      dy_m: list[float]
      chosen: list[str]  ("local_point" or "dataset_xy_m" or other)
      position_ids: list[str]
    """
    per = payload.get("per_image", [])
    errors_m, dx_m, dy_m, chosen, position_ids = [], [], [], [], []

    for row in per:
        if "error_m" not in row:
            continue
        e = row["error_m"]
        if e is None:
            continue

        # Use the prediction actually used and gt to compute signed error
        gt = row.get("gt_xy", {})
        pred = row.get("pred_xy_used", {})

        if not (isinstance(gt, dict) and isinstance(pred, dict)):
            continue
        if not all(k in gt for k in ("x", "y")):
            continue
        if not all(k in pred for k in ("x", "y")):
            continue

        ex = float(pred["x"]) - float(gt["x"])
        ey = float(pred["y"]) - float(gt["y"])

        errors_m.append(float(e))
        dx_m.append(ex)
        dy_m.append(ey)
        chosen.append(str(row.get("chosen", "other")))
        position_ids.append(str(row.get("position_id", "")))

    return errors_m, dx_m, dy_m, chosen, position_ids


def to_cm(xs_m):
    return [x * 100.0 for x in xs_m]


def make_histogram(errors_cm_a, label_a, errors_cm_b=None, label_b=None, outpath=None, bins=15):
    plt.figure()
    plt.hist(errors_cm_a, bins=bins, alpha=0.6, label=label_a)
    if errors_cm_b is not None:
        plt.hist(errors_cm_b, bins=bins, alpha=0.6, label=label_b)
    plt.xlabel("Euclidean error [cm]")
    plt.ylabel("Count")
    plt.title("Error histogram")
    plt.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.close()


def make_cdf(errors_cm_a, label_a, errors_cm_b=None, label_b=None, outpath=None):
    def cdf_xy(vals):
        vals = sorted(vals)
        n = len(vals)
        if n == 0:
            return [], []
        ys = [(i + 1) / n for i in range(n)]
        return vals, ys

    xa, ya = cdf_xy(errors_cm_a)

    plt.figure()
    plt.plot(xa, ya, marker=".", linestyle="-", label=label_a)
    if errors_cm_b is not None:
        xb, yb = cdf_xy(errors_cm_b)
        plt.plot(xb, yb, marker=".", linestyle="-", label=label_b)

    plt.xlabel("Euclidean error [cm]")
    plt.ylabel("Fraction ≤ error")
    plt.title("Error CDF")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.close()


def make_dxdy_scatter(dx_cm_a, dy_cm_a, label_a, dx_cm_b=None, dy_cm_b=None, label_b=None, outpath=None):
    plt.figure()
    plt.scatter(dx_cm_a, dy_cm_a, s=18, alpha=0.7, label=label_a)
    if dx_cm_b is not None and dy_cm_b is not None:
        plt.scatter(dx_cm_b, dy_cm_b, s=18, alpha=0.7, label=label_b)

    # axes lines at 0 (default black)
    plt.axhline(0.0, linewidth=1)
    plt.axvline(0.0, linewidth=1)

    plt.xlabel("dx = (pred_x - gt_x) [cm]")
    plt.ylabel("dy = (pred_y - gt_y) [cm]")
    plt.title("Signed error scatter (dx vs dy)")
    plt.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.close()


def make_error_by_sample(errors_cm, chosen_list, label, outpath=None):
    """
    Scatter error vs sample index, split by chosen frame using marker shape
    (no explicit colors so matplotlib defaults handle it).
    """
    idx_local = [i for i, c in enumerate(chosen_list) if c == "local_point"]
    idx_ds = [i for i, c in enumerate(chosen_list) if c == "dataset_xy_m"]
    idx_other = [i for i, c in enumerate(chosen_list) if c not in ("local_point", "dataset_xy_m")]

    plt.figure()
    if idx_local:
        plt.plot(idx_local, [errors_cm[i] for i in idx_local], linestyle="None", marker="o",
                 label=f"{label}: local_point")
    if idx_ds:
        plt.plot(idx_ds, [errors_cm[i] for i in idx_ds], linestyle="None", marker="^",
                 label=f"{label}: dataset_xy_m")
    if idx_other:
        plt.plot(idx_other, [errors_cm[i] for i in idx_other], linestyle="None", marker="x",
                 label=f"{label}: other")

    plt.xlabel("Sample index")
    plt.ylabel("Euclidean error [cm]")
    plt.title("Per-sample error (with chosen frame)")
    plt.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot localisation error statistics from *_with_stats.json files.")
    ap.add_argument("--yolo", type=str, default="./evaluation/errors_in_centres_with_stats_pred.json",
                    help="YOLO centres stats JSON (with per_image).")
    ap.add_argument("--gt", type=str, default="./evaluation/errors_in_centres_with_stats_gt.json",
                    help="GT/annotated centres stats JSON (with per_image).")
    ap.add_argument("--outdir", type=str, default="./evaluation/plots",
                    help="Output directory for figures.")
    ap.add_argument("--prefix", type=str, default="centres",
                    help="Filename prefix for output plots.")
    ap.add_argument("--bins", type=int, default=15,
                    help="Histogram bins.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    yolo_path = Path(args.yolo)
    gt_path = Path(args.gt)

    yolo_payload = load_payload(yolo_path)
    gt_payload = load_payload(gt_path)

    yolo_errors_m, yolo_dx_m, yolo_dy_m, yolo_chosen, _ = extract_series(yolo_payload)
    gt_errors_m, gt_dx_m, gt_dy_m, gt_chosen, _ = extract_series(gt_payload)

    yolo_errors_cm = to_cm(yolo_errors_m)
    gt_errors_cm = to_cm(gt_errors_m)

    yolo_dx_cm = to_cm(yolo_dx_m)
    yolo_dy_cm = to_cm(yolo_dy_m)
    gt_dx_cm = to_cm(gt_dx_m)
    gt_dy_cm = to_cm(gt_dy_m)

    # 1) Histogram overlay
    make_histogram(
        yolo_errors_cm, "YOLO centres",
        gt_errors_cm, "Annotated centres",
        outpath=outdir / f"{args.prefix}_hist.png",
        bins=args.bins
    )

    # 2) CDF overlay
    make_cdf(
        yolo_errors_cm, "YOLO centres",
        gt_errors_cm, "Annotated centres",
        outpath=outdir / f"{args.prefix}_cdf.png"
    )

    # 3) dx vs dy overlay
    make_dxdy_scatter(
        yolo_dx_cm, yolo_dy_cm, "YOLO centres",
        gt_dx_cm, gt_dy_cm, "Annotated centres",
        outpath=outdir / f"{args.prefix}_dxdy.png"
    )

    # 4) Per-sample error vs index (separate plots for clarity)
    make_error_by_sample(
        yolo_errors_cm, yolo_chosen, "YOLO centres",
        outpath=outdir / f"{args.prefix}_per_sample_yolo.png"
    )
    make_error_by_sample(
        gt_errors_cm, gt_chosen, "Annotated centres",
        outpath=outdir / f"{args.prefix}_per_sample_gt.png"
    )

    print("Wrote plots to:", outdir.resolve())
    print("Files:")
    for p in sorted(outdir.glob(f"{args.prefix}_*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
