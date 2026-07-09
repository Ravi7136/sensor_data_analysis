"""
Cold-room WAP-cluster detector.

Combines the two upstream outputs to decide WHICH spatial cluster of WAPs is the
cold room:

  1. Cluster info  (from analyze_wap_clusters.py)  -> which WAPs form each
     cluster + the cluster's MEDIAN package temperature.
  2. WAP proximity (from the tx-power / RSSI path-loss scoring)  -> per-WAP
     Score / Rank (how close + how often the package was heard).

Why BOTH signals are needed
---------------------------
Temperature alone is ambiguous: two clusters can share the same cold median
temperature (e.g. a real cold room AND a nearby cold corridor). Proximity breaks
the tie - the true cold room is the cold cluster the package is physically
NEAREST to and heard MOST within.

Decision logic (per cluster)
----------------------------
  temp_score  = clip((WARM_REF - median_temp) / (WARM_REF - COLD_REF), 0, 1)
                # colder cluster -> higher (0..1)
  prox_sum    = sum of member-WAP proximity Scores
  prox_share  = prox_sum / (total prox_sum across all clusters)
                # fraction of the package's total 'nearness' this cluster owns
  cold_score  = temp_score * prox_share * 100
                # multiplicative GATE: must be BOTH cold AND near

The cluster with the highest cold_score is flagged as the COLD ROOM.

Inputs  (CSV):
  * wap_clusters.csv            Cluster, WAP, ClusterMedianTemp, ClusterReadings
  * wap_path_loss_scored.csv    WAP, MedianPathLoss, Readings, Score, Rank
Output  (CSV):
  * coldroom_cluster_detection.csv   per-cluster scores + IsColdRoom flag

Run:  python coldroomwap_detect.py [clusters_csv] [proximity_csv] [output_csv]
"""

import csv
import sys

CLUSTERS_CSV = "wap_clusters.csv"
PROXIMITY_CSV = "wap_path_loss_scored.csv"
OUTPUT_CSV = "coldroom_cluster_detection.csv"

# --------------------------------------------------------------------------- #
# TUNABLES
# --------------------------------------------------------------------------- #
COLD_REF = 5.0     # median temp (C) treated as fully "cold"  -> temp_score 1.0
WARM_REF = 25.0    # median temp (C) treated as fully "warm"  -> temp_score 0.0
# A cluster whose leading cold_score exceeds the runner-up by this ratio is
# reported as a HIGH-confidence detection.
CONFIDENCE_RATIO = 3.0


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _find_col(fieldnames, *keywords):
    for name in fieldnames:
        low = name.lower().replace(" ", "").replace("_", "")
        if all(k in low for k in keywords):
            return name
    return None


def temp_score(median_temp):
    return _clip((WARM_REF - median_temp) / (WARM_REF - COLD_REF), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# LOAD
# --------------------------------------------------------------------------- #
def load_clusters(path):
    """Return {cluster_id: {"members": [...], "median_temp": t, "readings": n}}."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        c_col = _find_col(fields, "cluster")
        w_col = _find_col(fields, "wap")
        t_col = _find_col(fields, "median", "temp") or _find_col(fields, "temp")
        n_col = _find_col(fields, "cluster", "read") or _find_col(fields, "read")
        if not (c_col and w_col and t_col):
            raise SystemExit("clusters CSV needs Cluster, WAP, MedianTemp columns")
        clusters = {}
        for r in reader:
            cid = r[c_col].strip()
            if not cid:
                continue
            c = clusters.setdefault(cid, {"members": [], "median_temp": None,
                                          "readings": None})
            c["members"].append(r[w_col].strip())
            c["median_temp"] = float(r[t_col])
            if n_col and (r.get(n_col) or "").strip():
                c["readings"] = int(float(r[n_col]))
        return clusters


def load_proximity(path):
    """Return {wap: {"score": s, "rank": r, "pl": pl, "readings": n}}."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        w_col = _find_col(fields, "wap")
        s_col = _find_col(fields, "score")
        r_col = _find_col(fields, "rank")
        pl_col = _find_col(fields, "path", "loss") or _find_col(fields, "loss")
        n_col = _find_col(fields, "read")
        if not (w_col and s_col):
            raise SystemExit("proximity CSV needs WAP and Score columns")
        prox = {}
        for r in reader:
            wap = (r.get(w_col) or "").strip()
            if not wap:
                continue
            prox[wap] = {
                "score": float(r[s_col]) if (r.get(s_col) or "").strip() else 0.0,
                "rank": int(float(r[r_col])) if r_col and (r.get(r_col) or "").strip() else None,
                "pl": float(r[pl_col]) if pl_col and (r.get(pl_col) or "").strip() else None,
                "readings": int(float(r[n_col])) if n_col and (r.get(n_col) or "").strip() else None,
            }
        return prox


# --------------------------------------------------------------------------- #
# DETECT
# --------------------------------------------------------------------------- #
def detect(clusters, prox):
    # 1) per-cluster proximity sum + temperature score
    for cid, c in clusters.items():
        c["prox_sum"] = sum(prox.get(w, {}).get("score", 0.0) for w in c["members"])
        c["best_wap"] = min(
            (w for w in c["members"] if w in prox),
            key=lambda w: prox[w].get("rank") or 1e9, default=None)
        c["temp_score"] = temp_score(c["median_temp"])

    # 2) proximity share across clusters
    total_prox = sum(c["prox_sum"] for c in clusters.values()) or 1.0
    for c in clusters.values():
        c["prox_share"] = c["prox_sum"] / total_prox
        c["cold_score"] = round(c["temp_score"] * c["prox_share"] * 100.0, 2)

    # 3) rank clusters (best first) + flag the winner
    ordered = sorted(clusters.items(), key=lambda kv: kv[1]["cold_score"], reverse=True)
    for rank, (cid, c) in enumerate(ordered, 1):
        c["rank"] = rank
        c["is_cold_room"] = (rank == 1 and c["cold_score"] > 0)
    return ordered


def confidence(ordered):
    if len(ordered) < 2:
        return "SINGLE CLUSTER"
    top = ordered[0][1]["cold_score"]
    second = ordered[1][1]["cold_score"]
    if top <= 0:
        return "NONE (no cold+near cluster)"
    if second <= 0 or top / second >= CONFIDENCE_RATIO:
        return "HIGH"
    return "LOW (top two clusters are close - review)"


# --------------------------------------------------------------------------- #
# REPORT
# --------------------------------------------------------------------------- #
def print_report(ordered, prox):
    print("=" * 84)
    print(f"COLD-ROOM CLUSTER DETECTION   (COLD_REF={COLD_REF:.0f}C, WARM_REF={WARM_REF:.0f}C)")
    print("=" * 84)
    print(f"{'Rank':>4}  {'Cluster':<8}{'MedTemp':>8}{'TempScr':>8}"
          f"{'ProxSum':>9}{'ProxShr':>9}{'ColdScore':>11}  Cold?")
    print("-" * 84)
    for cid, c in ordered:
        print(f"{c['rank']:>4}  {cid:<8}{c['median_temp']:>7.1f}C{c['temp_score']:>8.2f}"
              f"{c['prox_sum']:>9.1f}{c['prox_share']:>9.2f}{c['cold_score']:>11.2f}"
              f"  {'<== COLD ROOM' if c['is_cold_room'] else ''}")
    print("-" * 84)

    winner_id, winner = ordered[0]
    print(f"Detected cold-room cluster : {winner_id}  (confidence: {confidence(ordered)})")
    print(f"Cold-room WAPs             : {', '.join(winner['members'])}")
    if winner.get("best_wap"):
        bw = winner["best_wap"]
        print(f"Closest WAP (entry beacon) : {bw}  "
              f"(rank {prox[bw].get('rank')}, score {prox[bw].get('score')}, "
              f"path loss {prox[bw].get('pl')} dB)")
    print("=" * 84)


def write_output(ordered, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Rank", "Cluster", "Members", "MedianTemp", "TempScore",
                    "ProxSum", "ProxShare", "ColdScore", "IsColdRoom"])
        for cid, c in ordered:
            w.writerow([c["rank"], cid, "; ".join(c["members"]),
                        c["median_temp"], round(c["temp_score"], 3),
                        round(c["prox_sum"], 2), round(c["prox_share"], 3),
                        c["cold_score"], "YES" if c["is_cold_room"] else "NO"])


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    clusters_csv = sys.argv[1] if len(sys.argv) > 1 else CLUSTERS_CSV
    proximity_csv = sys.argv[2] if len(sys.argv) > 2 else PROXIMITY_CSV
    out_csv = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_CSV

    clusters = load_clusters(clusters_csv)
    prox = load_proximity(proximity_csv)
    if not clusters:
        raise SystemExit("No clusters loaded from " + clusters_csv)

    ordered = detect(clusters, prox)
    print_report(ordered, prox)
    write_output(ordered, out_csv)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
