"""
Visualization + EDA for the cold-room ENTRY / EXIT detection.

Purpose: give the business team a clear, visual story of HOW and WHEN a package
enters / stays in / exits the cold room, plus data-fact insights.

It reuses the exact detection pipeline from detect_cold_room.py, so the charts
always match what the detector decides.

Outputs:
  * cold_room_timeline.png  - 3-panel timeline (temperature / RSSI / WAP),
                              cold-room stay shaded, ENTRY & EXIT marked.
  * cold_room_insights.png  - a one-page "data facts" summary card.
  * printed EDA insights in the console.

Requires matplotlib:   pip install matplotlib
Run:                    python visualize_cold_room.py
"""

import statistics
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")  # write files without needing a display
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import detect_cold_room as det

# --------------------------------------------------------------------------- #
# Colours (business-friendly, high contrast)
# --------------------------------------------------------------------------- #
C_INSIDE = "#2e7d32"     # green  = inside cold room
C_OUTSIDE = "#c62828"    # red    = outside / transit
C_TEMP = "#1565c0"       # blue   = temperature
C_SHADE = "#a5d6a7"      # light green shade for the stay
C_ENTRY = "#1b5e20"
C_EXIT = "#b71c1c"
C_THRESH = "#757575"


# --------------------------------------------------------------------------- #
# Pipeline (mirror detect_cold_room.main, but keep the row-level data)
# --------------------------------------------------------------------------- #
def run_pipeline():
    rows = det.load_rows(det.INPUT_CSV)
    if not rows:
        raise SystemExit("No data loaded from " + det.INPUT_CSV)
    if det.COLD_ROOM_WAPS:
        cold_waps, info = set(det.COLD_ROOM_WAPS), None
    else:
        cold_waps, info = det.infer_cold_room_waps(rows)
    det.mark_inside(rows, cold_waps)
    det.smooth_inside(rows)
    visits = det.detect_visits(rows)
    return rows, cold_waps, info, visits


# --------------------------------------------------------------------------- #
# EDA insights
# --------------------------------------------------------------------------- #
def compute_insights(rows, cold_waps, visits):
    times = [r["dt"] for r in rows]
    span_min = det._minutes(times[-1], times[0])
    gaps = [det._minutes(times[i], times[i - 1]) for i in range(1, len(times))]

    inside_rows = [r for r in rows if r["inside_raw"]]
    outside_rows = [r for r in rows if not r["inside_raw"]]

    def temps(rs):
        return [r["temp"] for r in rs if r["temp"] is not None]

    in_temps, out_temps = temps(inside_rows), temps(outside_rows)

    # cooldown rate over the first visit's approach (entry -> +30 min)
    cooldown_rate = None
    if visits:
        v = visits[0]
        window = [r for r in rows if v["entry"] <= r["dt"] <= v["entry"] + timedelta(minutes=30)
                  and r["temp"] is not None]
        if len(window) >= 2:
            dt_min = det._minutes(window[-1]["dt"], window[0]["dt"])
            if dt_min > 0:
                cooldown_rate = (window[0]["temp"] - window[-1]["temp"]) / dt_min

    wap_counts = {}
    for r in rows:
        wap_counts[r["wap"]] = wap_counts.get(r["wap"], 0) + 1

    inside_min = sum(v["duration_min"] for v in visits)

    return {
        "n_readings": len(rows),
        "first": times[0], "last": times[-1],
        "span_min": span_min,
        "avg_gap": statistics.mean(gaps) if gaps else 0,
        "max_gap": max(gaps) if gaps else 0,
        "pct_inside_time": 100.0 * inside_min / span_min if span_min else 0,
        "n_inside_rows": len(inside_rows),
        "in_temp_mean": statistics.mean(in_temps) if in_temps else None,
        "in_temp_min": min(in_temps) if in_temps else None,
        "out_temp_mean": statistics.mean(out_temps) if out_temps else None,
        "out_temp_max": max(out_temps) if out_temps else None,
        "cooldown_rate": cooldown_rate,
        "wap_counts": wap_counts,
        "cold_waps": sorted(cold_waps),
        "n_visits": len(visits),
        "inside_min": inside_min,
    }


def print_insights(ins, visits):
    print("=" * 78)
    print("EDA INSIGHTS - COLD-ROOM PACKAGE MOVEMENT")
    print("=" * 78)
    print(f"Readings              : {ins['n_readings']}")
    print(f"Time span             : {ins['first']:%d-%b %H:%M} -> {ins['last']:%d-%b %H:%M} "
          f"({det._fmt_duration(ins['span_min'])})")
    print(f"Reading cadence       : avg {ins['avg_gap']:.1f} min, longest gap {ins['max_gap']:.0f} min")
    print(f"Cold-room WAP cluster : {ins['cold_waps']}")
    print(f"Time inside cold room : {det._fmt_duration(ins['inside_min'])} "
          f"({ins['pct_inside_time']:.1f}% of tracked time), {ins['n_visits']} visit(s)")
    if ins["in_temp_mean"] is not None:
        print(f"Temp INSIDE           : mean {ins['in_temp_mean']:.2f} C, min {ins['in_temp_min']:.2f} C")
    if ins["out_temp_mean"] is not None:
        print(f"Temp OUTSIDE          : mean {ins['out_temp_mean']:.2f} C, max {ins['out_temp_max']:.2f} C")
    if ins["cooldown_rate"] is not None:
        print(f"Cool-down rate        : {ins['cooldown_rate']:.2f} C/min just after entry")
    print("-" * 78)
    for i, v in enumerate(visits, 1):
        exit_s = "ONGOING" if v["ongoing"] else f"{v['exit']:%d-%b %H:%M}"
        print(f"Visit #{i}: ENTRY {v['entry']:%d-%b %H:%M}  ->  EXIT {exit_s}  "
              f"({det._fmt_duration(v['duration_min'])})")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Timeline figure
# --------------------------------------------------------------------------- #
def plot_timeline(rows, cold_waps, visits, ins, path="cold_room_timeline.png"):
    times = [r["dt"] for r in rows]
    temps = [r["temp"] for r in rows]
    rssi = [r["rssi"] for r in rows]
    inside = [r["inside_raw"] for r in rows]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(15, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 2], "hspace": 0.12})
    fig.suptitle("Cold-Room Package Movement - Entry / Stay / Exit",
                 fontsize=16, fontweight="bold")

    def shade_visits(ax):
        for v in visits:
            end = v["exit"] if not v["ongoing"] else times[-1]
            ax.axvspan(v["entry"], end, color=C_SHADE, alpha=0.45, zorder=0)
            ax.axvline(v["entry"], color=C_ENTRY, ls="--", lw=1.6)
            if not v["ongoing"]:
                ax.axvline(v["exit"], color=C_EXIT, ls="--", lw=1.6)

    # --- Panel 1: temperature ---
    shade_visits(ax1)
    ax1.plot(times, temps, color=C_TEMP, lw=1.8)
    ax1.set_ylabel("Package temp (C)", fontsize=11)
    ax1.set_title("1) Temperature - drops on entry, stays cold, rises on exit "
                  "(supporting evidence)", fontsize=11, loc="left")
    ax1.grid(True, alpha=0.3)
    for v in visits:
        ax1.annotate("ENTRY", xy=(v["entry"], max(t for t in temps if t is not None)),
                     xytext=(6, -4), textcoords="offset points",
                     color=C_ENTRY, fontweight="bold", fontsize=10)
        if not v["ongoing"]:
            ax1.annotate("EXIT", xy=(v["exit"], max(t for t in temps if t is not None)),
                         xytext=(6, -4), textcoords="offset points",
                         color=C_EXIT, fontweight="bold", fontsize=10)

    # --- Panel 2: RSSI (primary signal) ---
    shade_visits(ax2)
    in_t = [t for t, f in zip(times, inside) if f]
    in_r = [r for r, f in zip(rssi, inside) if f]
    out_t = [t for t, f, r in zip(times, inside, rssi) if not f and r is not None]
    out_r = [r for f, r in zip(inside, rssi) if not f and r is not None]
    ax2.scatter(out_t, out_r, s=12, color=C_OUTSIDE, alpha=0.6, label="outside / transit")
    ax2.scatter(in_t, in_r, s=14, color=C_INSIDE, alpha=0.8, label="inside cold room")
    ax2.axhline(det.STRONG_RSSI_DBM, color=C_THRESH, ls=":", lw=1.4,
                label=f"'near' threshold {det.STRONG_RSSI_DBM} dBm")
    ax2.set_ylabel("Strongest RSSI (dBm)", fontsize=11)
    ax2.set_title("2) RSSI to closest WAP - the PRIMARY signal (works even in winter)",
                  fontsize=11, loc="left")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: which WAP heard the package ---
    shade_visits(ax3)
    wap_order = ins["cold_waps"] + sorted(w for w in ins["wap_counts"] if w not in cold_waps)
    ypos = {w: i for i, w in enumerate(wap_order)}
    cw_t = [t for t, r in zip(times, rows) if r["wap"] in cold_waps]
    cw_y = [ypos[r["wap"]] for r in rows if r["wap"] in cold_waps]
    ow_t = [t for t, r in zip(times, rows) if r["wap"] not in cold_waps]
    ow_y = [ypos[r["wap"]] for r in rows if r["wap"] not in cold_waps]
    ax3.scatter(ow_t, ow_y, s=10, color=C_OUTSIDE, alpha=0.6)
    ax3.scatter(cw_t, cw_y, s=10, color=C_INSIDE, alpha=0.8)
    ax3.set_yticks(range(len(wap_order)))
    ax3.set_yticklabels(wap_order, fontsize=8)
    ax3.set_ylabel("Closest WAP", fontsize=11)
    ax3.set_title("3) WAP attribution - inside = only cold-room WAPs (green)",
                  fontsize=11, loc="left")
    ax3.grid(True, alpha=0.3, axis="x")

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.legend(handles=[Patch(color=C_SHADE, label="Inside cold room (detected)")],
               loc="upper right", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# --------------------------------------------------------------------------- #
# Insights "card" figure
# --------------------------------------------------------------------------- #
def plot_insights_card(ins, visits, path="cold_room_insights.png"):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    ax.set_title("Cold-Room Detection - Data Facts & How It Works",
                 fontsize=15, fontweight="bold", pad=16)

    lines = []
    lines.append(("HOW WE DECIDE ENTRY / EXIT", True))
    lines.append(("- PRIMARY: package is 'inside' when its signal is consistently heard by the", False))
    lines.append(("  cold-room WAPs with strong RSSI (>= {} dBm).".format(det.STRONG_RSSI_DBM), False))
    lines.append(("- SUPPORTING: temperature drop confirms it (skipped in winter, small gradient).", False))
    lines.append(("- ENTRY = start of a sustained inside period; EXIT = sustained return outside.", False))
    lines.append(("", False))
    lines.append(("DATA FACTS", True))
    lines.append((f"- Readings tracked        : {ins['n_readings']}", False))
    lines.append((f"- Tracking window         : {ins['first']:%d-%b %H:%M} -> {ins['last']:%d-%b %H:%M} "
                  f"({det._fmt_duration(ins['span_min'])})", False))
    lines.append((f"- Reading cadence         : avg {ins['avg_gap']:.1f} min (longest gap {ins['max_gap']:.0f} min)", False))
    lines.append((f"- Time inside cold room   : {det._fmt_duration(ins['inside_min'])} "
                  f"({ins['pct_inside_time']:.0f}% of tracked time)", False))
    if ins["in_temp_mean"] is not None and ins["out_temp_mean"] is not None:
        lines.append((f"- Temp inside vs outside  : {ins['in_temp_mean']:.1f} C  vs  {ins['out_temp_mean']:.1f} C", False))
    if ins["cooldown_rate"] is not None:
        lines.append((f"- Cool-down after entry   : {ins['cooldown_rate']:.2f} C/min", False))
    lines.append((f"- Cold-room WAP cluster   : {', '.join(ins['cold_waps'])}", False))
    lines.append(("", False))
    lines.append(("DETECTED VISITS", True))
    for i, v in enumerate(visits, 1):
        exit_s = "ONGOING (still inside at trace end)" if v["ongoing"] else f"{v['exit']:%d-%b %H:%M}"
        lines.append((f"- Visit {i}: ENTRY {v['entry']:%d-%b %H:%M}  ->  EXIT {exit_s}   "
                      f"[{det._fmt_duration(v['duration_min'])}]", False))

    y = 0.93
    for text, header in lines:
        ax.text(0.02, y, text, fontsize=12 if header else 10.5,
                fontweight="bold" if header else "normal",
                color="#0d47a1" if header else "#212121",
                family="monospace", transform=ax.transAxes)
        y -= 0.052 if header else 0.045

    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    rows, cold_waps, info, visits = run_pipeline()
    ins = compute_insights(rows, cold_waps, visits)
    print_insights(ins, visits)
    plot_timeline(rows, cold_waps, visits, ins)
    plot_insights_card(ins, visits)


if __name__ == "__main__":
    main()
