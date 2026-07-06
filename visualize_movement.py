"""
Package MOVEMENT visualization: "transit then settle".

Shows how the package device moved through the station and the exact point in
time after which it stopped moving and stayed parked (inside the cold room).

Position source: the device's own trilaterated position (IDNODELATITUDE /
IDNODELONGITUDE) when present, else a fallback to the closest-WAP coordinates
(LATITUDE / LONGITUDE). A light smoothing tames trilateration jitter.

The 'settle' point is derived from the distance to the final resting spot: it is
the last moment the device was farther than SETTLE_RADIUS_M from where it ends up.

Outputs:
  * package_movement.png  - (1) trajectory map coloured by time with the parked
                            zone circled + direction arrows, (2) plain-English
                            transit/parked timeline bar, (3) distance-to-resting
                            -spot vs time (falls to ~0 and stays).
  * printed movement summary.

Requires matplotlib:  pip install matplotlib
Run:                  python visualize_movement.py
"""

import csv
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import detect_cold_room as det

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
INPUT_CSV = det.INPUT_CSV
IDNODE_LAT_COL = "IDNODELATITUDE"     # package device position (trilateration)
IDNODE_LON_COL = "IDNODELONGITUDE"

SMOOTH_WINDOW_MIN = 5     # light smoothing to tame trilateration noise
REST_TAIL_MIN = 60        # window at the end used to define the resting spot
SETTLE_RADIUS_M = 12      # within this of the resting spot = parked

C_MOVE = "#c62828"        # red   = in transit
C_STATIC = "#2e7d32"      # green = parked
C_LINE = "#5c6bc0"


# --------------------------------------------------------------------------- #
# LOADING + GEOMETRY
# --------------------------------------------------------------------------- #
def _to_float(value):
    value = (value or "").strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(path):
    """Load timestamp-sorted rows using the device position, with WAP fallback."""
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            ilat = _to_float(raw.get(IDNODE_LAT_COL))
            ilon = _to_float(raw.get(IDNODE_LON_COL))
            have = ilat is not None and ilon is not None
            rows.append({
                "dt": datetime.strptime(raw["EVENTTIME"].strip(), det.TIME_FMT),
                "lat": ilat if have else float(raw["LATITUDE"]),
                "lon": ilon if have else float(raw["LONGITUDE"]),
                "pos_src": "device" if have else "wap",
            })
    rows.sort(key=lambda r: r["dt"])
    return rows


def math_hypot(a, b):
    return (a * a + b * b) ** 0.5


def to_local_metres(rows):
    """Attach (east_m, north_m) to each row, relative to the SW-most point."""
    ref_lat = min(r["lat"] for r in rows)
    ref_lon = min(r["lon"] for r in rows)
    for r in rows:
        r["east"] = det.dist_m(ref_lat, ref_lon, ref_lat, r["lon"]) * (1 if r["lon"] >= ref_lon else -1)
        r["north"] = det.dist_m(ref_lat, ref_lon, r["lat"], ref_lon) * (1 if r["lat"] >= ref_lat else -1)
    return rows


def smooth_positions(rows):
    """Light trailing-window average of the device position (se, sn)."""
    n = len(rows)
    j = 0
    for i in range(n):
        t_i = rows[i]["dt"]
        while det._minutes(t_i, rows[j]["dt"]) > SMOOTH_WINDOW_MIN and rows[j]["dt"] < t_i:
            j += 1
        cnt = i - j + 1
        rows[i]["se"] = sum(rows[k]["east"] for k in range(j, i + 1)) / cnt
        rows[i]["sn"] = sum(rows[k]["north"] for k in range(j, i + 1)) / cnt
    return rows


def resting_spot(rows):
    """Centroid of the smoothed positions over the final REST_TAIL_MIN."""
    t_end = rows[-1]["dt"]
    tail = [r for r in rows if det._minutes(t_end, r["dt"]) <= REST_TAIL_MIN] or rows[-1:]
    return (sum(r["se"] for r in tail) / len(tail),
            sum(r["sn"] for r in tail) / len(tail))


def distance_to_rest(rows, rest):
    """Per-reading distance from the final resting spot (transit->settle story)."""
    rx, ry = rest
    return [math_hypot(r["se"] - rx, r["sn"] - ry) for r in rows]


def find_settle_index(dist_rest):
    """First reading after which the device stays within SETTLE_RADIUS_M forever."""
    last_out = -1
    for i, d in enumerate(dist_rest):
        if d > SETTLE_RADIUS_M:
            last_out = i
    if last_out == -1:
        return 0                      # already at rest the whole time
    if last_out >= len(dist_rest) - 1:
        return None                   # still moving at the end
    return last_out + 1


# --------------------------------------------------------------------------- #
# PLOT
# --------------------------------------------------------------------------- #
def plot_movement(rows, dist_rest, settle_i, path="package_movement.png"):
    times = [r["dt"] for r in rows]
    se = [r["se"] for r in rows]
    sn = [r["sn"] for r in rows]
    hours = [det._minutes(t, times[0]) / 60.0 for t in times]
    end_h = hours[-1]
    settle_t = times[settle_i] if settle_i is not None else None
    settle_h = det._minutes(settle_t, times[0]) / 60.0 if settle_t else None
    src = rows[0]["pos_src"]

    fig = plt.figure(figsize=(12, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 0.8, 2], hspace=0.42)
    fig.suptitle("Package Journey - travelled through the station, then parked",
                 fontsize=16, fontweight="bold")

    # --- Panel 1: trajectory map coloured by time, with direction arrows ---
    ax = fig.add_subplot(gs[0])
    ax.plot(se, sn, color="#cfd8dc", lw=1.5, zorder=1)
    sc = ax.scatter(se, sn, c=hours, cmap="viridis", s=26, zorder=2)
    fig.colorbar(sc, ax=ax, label="hours since first reading")

    # a few arrowheads along the transit path to show direction of travel
    end_arrow = settle_i if settle_i not in (None, 0) else len(rows) - 1
    if end_arrow >= 2:
        for k in range(1, 6):
            idx = max(1, int(end_arrow * k / 6))
            ax.annotate("", xy=(se[idx], sn[idx]), xytext=(se[idx - 1], sn[idx - 1]),
                        arrowprops=dict(arrowstyle="-|>", color="#78909c", lw=1.6), zorder=3)

    ax.scatter([se[0]], [sn[0]], s=240, marker="o", color=C_MOVE,
               edgecolor="black", zorder=5, label="START (in transit)")
    if settle_i is not None:
        cx = sum(se[settle_i:]) / len(se[settle_i:])
        cy = sum(sn[settle_i:]) / len(sn[settle_i:])
        ax.add_patch(plt.Circle((cx, cy), max(SETTLE_RADIUS_M, 5), color=C_STATIC, alpha=0.18, zorder=1))
        ax.scatter([cx], [cy], s=340, marker="*", color=C_STATIC, edgecolor="black",
                   zorder=6, label=f"PARKED from {settle_t:%d-%b %H:%M}")
    ax.set_xlabel("East (metres)")
    ax.set_ylabel("North (metres)")
    ax.set_title(f"1) Where the package went (colour = time; position source: {src}). "
                 "Arrows = direction; tight dot = parked.",
                 fontsize=11, loc="left")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # --- Panel 2: plain-English red/green timeline bar ---
    ax_tl = fig.add_subplot(gs[1])
    if settle_h is None:
        ax_tl.axvspan(0, end_h, color=C_MOVE, alpha=0.85)
        ax_tl.text(end_h / 2, 0.5, "IN TRANSIT (whole time)", ha="center", va="center",
                   color="white", fontsize=13, fontweight="bold")
    else:
        ax_tl.axvspan(0, settle_h, color=C_MOVE, alpha=0.85)
        ax_tl.axvspan(settle_h, end_h, color=C_STATIC, alpha=0.85)
        ax_tl.axvline(settle_h, color="black", lw=2)
        ax_tl.text(max(settle_h / 2, end_h * 0.04), 0.5,
                   f"IN TRANSIT\n{det._fmt_duration(settle_h * 60)}",
                   ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax_tl.text((settle_h + end_h) / 2, 0.5,
                   f"PARKED IN COLD ROOM\n{det._fmt_duration((end_h - settle_h) * 60)}",
                   ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax_tl.annotate(f"parked at {settle_t:%d-%b %H:%M}", xy=(settle_h, 1.0),
                       xytext=(settle_h, 1.35), ha="center", fontsize=10, fontweight="bold",
                       arrowprops=dict(arrowstyle="->"))
    ax_tl.set_xlim(0, end_h)
    ax_tl.set_ylim(0, 1)
    ax_tl.set_yticks([])
    ax_tl.set_xlabel("Hours since first reading")
    ax_tl.set_title("2) Simple timeline: red = moving, green = parked",
                    fontsize=11, loc="left")

    # --- Panel 3: distance to the final resting spot vs time ---
    ax2 = fig.add_subplot(gs[2])
    ax2.fill_between(hours, dist_rest, color=C_LINE, alpha=0.3)
    ax2.plot(hours, dist_rest, color=C_LINE, lw=1.8, label="distance from final resting spot")
    ax2.axhline(SETTLE_RADIUS_M, color=C_STATIC, ls=":", lw=1.5,
                label=f"'parked' radius {SETTLE_RADIUS_M} m")
    if settle_h is not None:
        ax2.axvline(settle_h, color=C_STATIC, ls="--", lw=2)
        ax2.axvspan(settle_h, end_h, color=C_STATIC, alpha=0.12)
        ax2.annotate("SETTLED", xy=(settle_h, SETTLE_RADIUS_M),
                     xytext=(settle_h + end_h * 0.05, max(dist_rest) * 0.6 if max(dist_rest) else 1),
                     color=C_STATIC, fontweight="bold", fontsize=11,
                     arrowprops=dict(arrowstyle="->", color=C_STATIC))
    ax2.set_ylabel("Distance to resting spot (m)")
    ax2.set_xlabel("Hours since first reading")
    ax2.set_xlim(0, end_h)
    ax2.set_title("3) How far from its final spot - falls to ~0 and stays (settled)",
                  fontsize=11, loc="left")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    rows = load_rows(INPUT_CSV)
    if not rows:
        raise SystemExit("No data loaded from " + INPUT_CSV)
    to_local_metres(rows)
    smooth_positions(rows)
    rest = resting_spot(rows)
    dist_rest = distance_to_rest(rows, rest)
    settle_i = find_settle_index(dist_rest)

    times = [r["dt"] for r in rows]
    total_dist = sum(math_hypot(rows[i]["se"] - rows[i - 1]["se"],
                                rows[i]["sn"] - rows[i - 1]["sn"])
                     for i in range(1, len(rows)))
    n_device = sum(1 for r in rows if r["pos_src"] == "device")

    print("=" * 70)
    print("PACKAGE MOVEMENT SUMMARY")
    print("=" * 70)
    print(f"Readings              : {len(rows)}  "
          f"(device pos: {n_device}, WAP fallback: {len(rows) - n_device})")
    print(f"Distance travelled    : {total_dist:.0f} m (approx)")
    if settle_i is None:
        print("Movement              : still moving at end of trace")
    elif settle_i == 0:
        print("Movement              : parked for the entire trace")
    else:
        st = times[settle_i]
        print(f"Moved until           : {st:%d-%b %H:%M} "
              f"(after {det._fmt_duration(det._minutes(st, times[0]))} of movement)")
        print(f"Parked afterwards for : {det._fmt_duration(det._minutes(times[-1], st))} "
              f"(until end of trace)")
    print("=" * 70)

    plot_movement(rows, dist_rest, settle_i)


if __name__ == "__main__":
    main()
