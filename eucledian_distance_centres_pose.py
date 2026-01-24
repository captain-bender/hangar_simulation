import argparse
import json, math, re, sys
from pathlib import Path

PRED_JSON = Path("./evaluation/v1-pose/centers_by_image_with_meters_pred.json")
GT_JSON   = Path("./configs/rover.json")                             
OUT_JSON  = Path("./evaluation/v1-pose/errors_in_centres_pred.json")
OUT_SWAPPED = Path("./evaluation/v1-pose/swapped_positions.json")

POS_ID_RE = re.compile(r"(position_\d+)", re.IGNORECASE)

def pid_from_file(fn):
    m = POS_ID_RE.search(fn or "")
    return m.group(1).lower() if m else None

def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

def f(x, default=0.0):
    try: return float(x)
    except: return default

def main():
    parser = argparse.ArgumentParser(description="Compute errors or list positions where dataset axes appear swapped")
    parser.add_argument("--list-swapped", "-l", action="store_true", help="Write a JSON list of positions where dataset_xy_m is a better match (suggests swapped axes)")
    parser.add_argument("--swapped-out", default=str(OUT_SWAPPED), help="Output path for swapped positions JSON")
    parser.add_argument("--out-json", default=str(OUT_JSON), help="Output path for errors summary JSON (default behavior)")
    args = parser.parse_args()

    try:
        pred = json.loads(PRED_JSON.read_text(encoding="utf-8"))
        gt = json.loads(GT_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        print("Error reading input JSONs:", e)
        sys.exit(1)

    gt_xy = {}
    for r in gt.get("rovers", []):
        pid = str(r.get("id","")).lower()
        loc = r.get("location", None)
        if pid and isinstance(loc, list) and len(loc) >= 2:
            gt_xy[pid] = (float(loc[0]), float(loc[1]))

    rows = []
    flips = 0

    # If the user requested a swapped-list, collect entries where dataset_xy_m
    # interpreted directly as (x,y) is a better match than the reported local_point.
    swapped_list = []

    for item in pred:
        pid = pid_from_file(item.get("file",""))
        if not pid or pid not in gt_xy:
            continue

        centers = item.get("yolo_pose_centers", []) or []
        if not centers:
            continue

        best = max(centers, key=lambda c: f(c.get("conf", 0.0), 0.0))

        lp = best.get("local_point", {})
        ds = best.get("dataset_xy_m", {})

        if "x" not in lp or "y" not in lp or "x" not in ds or "y" not in ds:
            continue

        gt_x, gt_y = gt_xy[pid]

        # Candidate 1: local_point
        p1x, p1y = float(lp["x"]), float(lp["y"])
        e1 = dist(p1x, p1y, gt_x, gt_y)

        # Candidate 2: dataset_xy_m interpreted as (x,y)
        p2x, p2y = float(ds["x"]), float(ds["y"])
        e2 = dist(p2x, p2y, gt_x, gt_y)

        if e2 < e1:
            chosen = "dataset_xy_m"
            flips += 1
            pred_x, pred_y, err = p2x, p2y, e2
            swapped_list.append({
                "position_id": pid,
                "file": item.get("file", ""),
                "gt_xy": {"x": gt_x, "y": gt_y},
                "local_point_xy": {"x": p1x, "y": p1y, "err_to_gt": e1},
                "dataset_xy_m_xy": {"x": p2x, "y": p2y, "err_to_gt": e2},
                "err_diff": e1 - e2,
            })
        else:
            chosen = "local_point"
            pred_x, pred_y, err = p1x, p1y, e1

        rows.append({
            "position_id": pid,
            "file": item.get("file",""),
            "gt_xy": {"x": gt_x, "y": gt_y},
            "local_point_xy": {"x": p1x, "y": p1y, "err_to_gt": e1},
            "dataset_xy_m_xy": {"x": p2x, "y": p2y, "err_to_gt": e2},
            "chosen": chosen,
            "pred_xy_used": {"x": pred_x, "y": pred_y},
            "error_m": err,
        })

    if args.list_swapped:
        out_path = Path(args.swapped_out)
        out_path.write_text(json.dumps({"count": len(swapped_list), "swapped_positions": swapped_list}, indent=2), encoding="utf-8")
        print("Wrote swapped positions list to", out_path)
        print("Found", len(swapped_list), "positions where dataset_xy_m gave smaller error than local_point")
        return

    errors = [r["error_m"] for r in rows]
    errors_sorted = sorted(errors)
    n = len(errors_sorted)
    summary = {
        "matched": n,
        "chosen_dataset_xy_m_count": flips,
        "chosen_local_point_count": n - flips,
        "mean_error_m": sum(errors)/n if n else None,
        "median_error_m": (errors_sorted[n//2] if n%2==1 else 0.5*(errors_sorted[n//2-1]+errors_sorted[n//2])) if n else None,
        "max_error_m": max(errors) if errors else None,
    }

    OUT_JSON.write_text(json.dumps({"summary": summary, "per_image": rows}, indent=2), encoding="utf-8")
    print("Wrote", OUT_JSON)
    print("Summary:", summary)

if __name__ == "__main__":
    main()
