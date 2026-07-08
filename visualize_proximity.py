"""
Proximity visualization using TX-POWER STEPPING.

Every minute the package advertises the same packet at 5 transmit powers
(+4, 0, -4, -8, -12 dBm) and each is heard with its own RSSI. Because the
package is at ONE position during that minute, all 5 obey the radio model:

    RSSI = TX_power - PATH_LOSS(distance) + noise

So subtracting the (known) TX power removes the deliberate 16 dB power swing and
leaves PATH LOSS - a clean, distance-faithful proximity signal:

    path_loss = TX_power - RSSI          (bigger = farther, smaller = closer)

This tells us HOW CLOSE the package is to its closest WAP - and, aggregated per
WAP, WHICH WAP it sits nearest to (the cold-room cluster once parked).

Path loss is reported in dB only (measured, no distance model / assumptions):
lower path loss = physically closer to that WAP.

Outputs:
  * package_proximity.png  - (1) path loss to closest WAP over time,
                             (2) the RSSI-vs-TXpower model check, and
                             (3) per-WAP proximity ranking (nearest first).
  * printed proximity summary.

Requires matplotlib + numpy.
Run:  python visualize_proximity.py
"""

import csv
import statistics
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import detect_cold_room as det
import analyze_wap_clusters as awc

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
INPUT_CSV = det.INPUT_CSV
TX_COLS = [f"txPower{i}" for i in range(1, 6)]
RSSI_COLS = [f"RSSI{i}" for i in range(1, 6)]

C_NEAR = "#2e7d32"   # green = cold-room WAP (near, once parked)
C_FAR = "#c62828"    # red   = transit / outer WAP


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def _to_int(value):
    value = (value or "").strip()
    if value == "":
        return None
    try:
        v = int(float(value))
    except ValueError:
        return None
    return None if v == 0 else v      # 0 = "not heard"


def cold_room_waps():
    """Names of the WAPs in the largest (cold-room) spatial cluster."""
    waps = awc.load_waps(INPUT_CSV)
    clusters = awc.cluster_waps(waps, awc.CLUSTER_DISTANCE_M)
    return set(clusters[0]) if clusters else set()


def load_rows(path):
    """Load rows with per-minute (tx, rssi) pairs and their median path loss."""
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            txs = [_to_int(raw.get(c)) for c in TX_COLS]
            rssis = [_to_int(raw.get(c)) for c in RSSI_COLS]
            pairs = [(t, r) for t, r in zip(txs, rssis)
                     if t is not None and r is not None]
            if not pairs:
                continue
            pls = [t - r for t, r in pairs]      # path loss per packet (dB)
            rows.append({
                "dt": datetime.strptime(raw["EVENTTIME"].strip(), det.TIME_FMT),
                "wap": raw["HARDWARENAME"].strip(),
                "temp": float(raw["IDNODECHIPTEMPARATURE"]) if (raw.get("IDNODECHIPTEMPARATURE") or "").strip() else None,
                "pairs": pairs,
                "pl": statistics.median(pls),
                "pl_spread": (max(pls) - min(pls)),
            })
    rows.sort(key=lambda r: r["dt"])
    return rows


# --------------------------------------------------------------------------- #
# PLOT
# --------------------------------------------------------------------------- #
def plot_proximity(rows, cold_set, path="package_proximity.png"):
    xs = [mdates.date2num(r["dt"]) for r in rows]
    pl = [r["pl"] for r in rows]
    colors = [C_NEAR if r["wap"] in cold_set else C_FAR for r in rows]

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.8, 1.8], hspace=0.32)
    fig.suptitle("Package Proximity from TX-Power Stepping "
                 "(path loss = TX power - RSSI)", fontsize=15, fontweight="bold")

    # --- Panel 1: path loss (and est. distance) to the closest WAP over time
    ax = fig.add_subplot(gs[0])
    ax.scatter(xs, pl, c=colors, s=16, zorder=3)
    ax.invert_yaxis()                    # smaller path loss (closer) at the top
    near_med = statistics.median([r["pl"] for r in rows if r["wap"] in cold_set] or [0])
    ax.axhline(near_med, color=C_NEAR, ls=":", lw=1.4,
               label=f"cold-room median ~{near_med:.0f} dB")
    ax.set_ylabel("Path loss to closest WAP (dB)\n<- closer            farther ->")
    ax.set_title("1) How CLOSE is the package (lower = nearer). "
                 "Green = cold-room WAP, red = outer/transit WAP.",
                 fontsize=11, loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # --- Panel 2: the model check - RSSI vs TX power for sample minutes
    ax2 = fig.add_subplot(gs[1])
    n = len(rows)
    sample_idx = sorted(set([0, 1, 2] + [int(n * k / 6) for k in range(1, 6)] + [n - 1]))
    sample_idx = [i for i in sample_idx if 0 <= i < n]
    tx_line = np.array([-13, 5])
    for i in sample_idx:
        r = rows[i]
        near = r["wap"] in cold_set
        col = C_NEAR if near else C_FAR
        tx = np.array([p[0] for p in r["pairs"]])
        rssi = np.array([p[1] for p in r["pairs"]])
        ax2.scatter(tx, rssi, color=col, s=40, zorder=3)
        ax2.plot(tx_line, tx_line - r["pl"], color=col, lw=1.2, alpha=0.7, zorder=2)
    ax2.set_xlabel("TX power (dBm)")
    ax2.set_ylabel("RSSI (dBm)")
    ax2.set_title("2) Why it works: for one minute RSSI rises ~1 dB per 1 dB of "
                  "TX power.\nThe line's offset below TX = path loss (distance). "
                  "Lower lines = farther.", fontsize=11, loc="left")
    ax2.grid(True, alpha=0.3)
    ax2.plot([], [], color=C_NEAR, marker="o", ls="-", label="a minute inside cold room")
    ax2.plot([], [], color=C_FAR, marker="o", ls="-", label="a minute in transit")
    ax2.legend(loc="upper left", fontsize=9)

    # --- Panel 3: per-WAP proximity ranking (nearest first)
    ax3 = fig.add_subplot(gs[2])
    by_wap = {}
    for r in rows:
        by_wap.setdefault(r["wap"], []).append(r["pl"])
    order = sorted(by_wap, key=lambda w: statistics.median(by_wap[w]))
    data = [by_wap[w] for w in order]
    positions = range(1, len(order) + 1)
    bp = ax3.boxplot(data, positions=list(positions), vert=True, patch_artist=True,
                     widths=0.6, showfliers=False)
    for patch, w in zip(bp["boxes"], order):
        patch.set_facecolor(C_NEAR if w in cold_set else C_FAR)
        patch.set_alpha(0.65)
    for med in bp["medians"]:
        med.set_color("black")
    ax3.set_xticks(list(positions))
    ax3.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax3.invert_yaxis()
    ax3.set_ylabel("Path loss when this WAP\nis closest (dB)  <- nearer")
    ax3.set_title("3) Which WAP is the package nearest to? "
                  "Lowest path loss = physically closest (the cold-room cluster).",
                  fontsize=11, loc="left")
    ax3.grid(True, axis="y", alpha=0.3)

    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    rows = load_rows(INPUT_CSV)
    if not rows:
        raise SystemExit("No usable tx/rssi data in " + INPUT_CSV)
    cold_set = cold_room_waps()

    near = [r["pl"] for r in rows if r["wap"] in cold_set]
    far = [r["pl"] for r in rows if r["wap"] not in cold_set]
    by_wap = {}
    for r in rows:
        by_wap.setdefault(r["wap"], []).append(r["pl"])
    nearest = min(by_wap, key=lambda w: statistics.median(by_wap[w]))

    print("=" * 72)
    print("PROXIMITY (TX-POWER) SUMMARY")
    print("=" * 72)
    print(f"Readings with tx/rssi : {len(rows)}")
    print(f"Cold-room WAPs        : {', '.join(sorted(cold_set)) or '(none found)'}")
    if near:
        print(f"Path loss inside      : median {statistics.median(near):.1f} dB")
    if far:
        print(f"Path loss transit/out : median {statistics.median(far):.1f} dB")
    print(f"Nearest WAP overall   : {nearest} "
          f"(median path loss {statistics.median(by_wap[nearest]):.1f} dB)")
    print("=" * 72)

    plot_proximity(rows, cold_set)


if __name__ == "__main__":
    main()
