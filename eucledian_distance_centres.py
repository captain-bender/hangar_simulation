import json, math, re
from pathlib import Path

PRED_JSON = Path("./evaluation/centers_by_image_with_meters_gt.json")
GT_JSON   = Path("./configs/rover.json")                             
OUT_JSON  = Path("./evaluation/errors_in_centres_gt.json")

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
    pred = json.loads(PRED_JSON.read_text(encoding="utf-8"))
    gt = json.loads(GT_JSON.read_text(encoding="utf-8"))

    gt_xy = {}
    for r in gt.get("rovers", []):
        pid = str(r.get("id","")).lower()
        loc = r.get("location", None)
        if pid and isinstance(loc, list) and len(loc) >= 2:
            gt_xy[pid] = (float(loc[0]), float(loc[1]))

    rows = []
    flips = 0

    for item in pred:
        pid = pid_from_file(item.get("file",""))
        if not pid or pid not in gt_xy:
            continue

        centers = item.get("yolo_obb_centers", []) or []
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

        # Candidate 2: swapped (dataset_xy_m interpreted back as (x,y) in rover.json frame)
        # dataset_xy_m = (local.y, local.x), so swap back -> (ds.y, ds.x) == local
        # BUT your evidence shows rover.json sometimes matches ds directly,
        # so we test ds-as-(x,y) too:
        p2x, p2y = float(ds["x"]), float(ds["y"])
        e2 = dist(p2x, p2y, gt_x, gt_y)

        if e2 < e1:
            chosen = "dataset_xy_m"
            flips += 1
            pred_x, pred_y, err = p2x, p2y, e2
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
