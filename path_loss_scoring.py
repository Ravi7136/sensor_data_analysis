"""
Score & rank WAPs by how likely the package sits NEAR them, combining two
signals per WAP:

  * median path loss   -> lower = physically closer  (better)
  * reading count      -> more  = more trustworthy    (better)

Scoring:
    proximity = clip((PL_FAR - path_loss) / (PL_FAR - PL_NEAR), 0, 1)   # 0..1
    support   = readings / (readings + SUPPORT_K)                       # 0..1
    score     = proximity * support * 100                               # 0..100

Multiplication acts as a GATE: a WAP must be BOTH near AND well-heard to score
high. So a close-but-single-reading WAP (e.g. one 70 dB hit) is correctly
demoted, while a near WAP heard hundreds of times rises to the top.

Input  : CSV with columns  WAP, MedianPathLoss, Readings
         (default: wap_path_loss.csv)
Output : same rows + two new columns  Score  and  Rank  (1 = best),
         sorted best-first  (default: wap_path_loss_scored.csv)

Run:  python path_loss_scoring.py [input_csv] [output_csv]
"""

import csv
import sys

INPUT_CSV = "wap_path_loss.csv"
OUTPUT_CSV = "wap_path_loss_scored.csv"

# --------------------------------------------------------------------------- #
# TUNABLES
# --------------------------------------------------------------------------- #
PL_NEAR = 65.0      # path loss (dB) treated as "closest"  -> proximity 1.0
PL_FAR = 95.0       # path loss (dB) treated as "farthest" -> proximity 0.0
SUPPORT_K = 20.0    # readings needed for "half confidence" (support = 0.5)


# --------------------------------------------------------------------------- #
# SCORING
# --------------------------------------------------------------------------- #
def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def proximity_score(path_loss):
    """Lower path loss -> higher proximity (0..1), using fixed near/far anchors."""
    return _clip((PL_FAR - path_loss) / (PL_FAR - PL_NEAR), 0.0, 1.0)


def support_score(readings):
    """More readings -> higher confidence (0..1), saturating (dataset-independent)."""
    return readings / (readings + SUPPORT_K)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def _find_col(fieldnames, *keywords):
    for name in fieldnames:
        low = name.lower().replace(" ", "").replace("_", "")
        if all(k in low for k in keywords):
            return name
    return None


def load_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        wap_c = _find_col(fields, "wap") or (fields[0] if fields else None)
        pl_c = _find_col(fields, "path", "loss") or _find_col(fields, "loss") or _find_col(fields, "path")
        rd_c = _find_col(fields, "read") or _find_col(fields, "count")
        if not (wap_c and pl_c and rd_c):
            raise SystemExit(
                "Could not find WAP / path-loss / readings columns in " + path)
        rows = []
        for r in reader:
            if not (r.get(wap_c) or "").strip():
                continue
            rows.append({
                "raw": r,
                "wap": r[wap_c].strip(),
                "pl": float(r[pl_c]),
                "readings": int(float(r[rd_c])),
            })
        return rows, fields


def score_rows(rows):
    for r in rows:
        r["prox"] = proximity_score(r["pl"])
        r["supp"] = support_score(r["readings"])
        r["score"] = round(r["prox"] * r["supp"] * 100.0, 1)
    # rank best-first; ties broken by lower path loss, then more readings
    rows.sort(key=lambda r: (-r["score"], r["pl"], -r["readings"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def write_output(rows, fields, path):
    out_fields = list(fields) + ["Score", "Rank"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            row = dict(r["raw"])
            row["Score"] = r["score"]
            row["Rank"] = r["rank"]
            w.writerow(row)


def print_table(rows):
    print("=" * 82)
    print(f"WAP PROXIMITY SCORING   (PL_NEAR={PL_NEAR:.0f} dB, PL_FAR={PL_FAR:.0f} dB, "
          f"k={SUPPORT_K:.0f})")
    print("=" * 82)
    print(f"{'Rank':>4}  {'WAP':<24}{'PathLoss':>10}{'Reads':>7}"
          f"{'Prox':>7}{'Supp':>7}{'Score':>8}")
    print("-" * 82)
    for r in rows:
        print(f"{r['rank']:>4}  {r['wap']:<24}{r['pl']:>8.1f}dB{r['readings']:>7}"
              f"{r['prox']:>7.2f}{r['supp']:>7.2f}{r['score']:>8.1f}")
    print("=" * 82)


def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else INPUT_CSV
    out_csv = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_CSV
    rows, fields = load_rows(in_csv)
    if not rows:
        raise SystemExit("No data rows found in " + in_csv)
    score_rows(rows)
    write_output(rows, fields, out_csv)
    print_table(rows)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
