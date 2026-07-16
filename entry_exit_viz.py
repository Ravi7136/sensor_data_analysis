"""
Cold-room ENTRY-TIME visualiser.

Draws an annotated, business-readable chart that JUSTIFIES the entry timestamp
produced by entry_exit.py. It reuses that module's detector directly, so the
picture can never disagree with the algorithm.

The figure is a single annotated panel:

  Temperature vs time
         * raw readings (dots, spacing shows the irregular sampling)
         * smoothed uniform-grid curve (line)
         * warm-baseline and cold-median reference lines
         * WARM / COOLING / COLD phases shaded
         * ENTRY point and STABILISED-COLD point marked + annotated

Inputs  (CSV):
  * entry_exit.csv   EVENTTIME, IDNODECHIPTEMPERATURE
Output  (PNG):
  * entry_exit_viz.png

Run:  python entry_exit_viz.py [input_csv] [output_png] [cold_median]

Requires: seaborn, matplotlib  (pip install seaborn matplotlib)
"""

import sys

import matplotlib
matplotlib.use("Agg")                       # file output; no display needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns

from entry_exit import (
    INPUT_CSV,
    load_series, detect_entry,
)

OUTPUT_PNG = "entry_exit_viz.png"

# Colour palette (colour-blind friendly).
C_RAW = "#4C72B0"       # raw readings
C_SMOOTH = "#DD8452"     # smoothed grid curve
C_BASE = "#8C8C8C"       # warm baseline
C_COLD = "#55A868"       # cold median
C_ENTRY = "#C44E52"      # entry marker
C_STABLE = "#8172B3"     # stabilised-cold marker
SHADE_WARM = "#F6D6C2"
SHADE_COOL = "#FCE8B2"
SHADE_COLD = "#CDE7D0"


def _fmt_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())


def plot(rows, res, out_png):
    if not res.get("ok"):
        raise SystemExit("Detection failed (%s) - nothing to plot"
                         % res.get("reason"))

    times = [r[0] for r in rows]
    temps = [r[1] for r in rows]
    gtimes = res["gtimes"]
    gsm = res["gsm"]
    baseline = res["baseline"]
    cold_median = res["cold_median"]
    span = res["span"]
    onset_i = res["onset"]
    cold_i = res["cold_idx"]
    onset_time = res["onset_time"]
    cold_time = res["cold_time"]
    entry_temp = temps[onset_i]

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # ---------------- temperature ---------------- #
    # Phase shading
    ax1.axvspan(times[0], onset_time, color=SHADE_WARM, alpha=0.6,
                label="_nolegend_")
    if cold_time is not None:
        ax1.axvspan(onset_time, cold_time, color=SHADE_COOL, alpha=0.6)
        ax1.axvspan(cold_time, times[-1], color=SHADE_COLD, alpha=0.6)
    else:
        ax1.axvspan(onset_time, times[-1], color=SHADE_COOL, alpha=0.6)

    # Curves
    ax1.plot(gtimes, gsm, color=C_SMOOTH, lw=2.5, zorder=3,
             label="Smoothed (uniform grid)")
    ax1.scatter(times, temps, s=28, color=C_RAW, zorder=4, alpha=0.85,
                label="Raw readings")

    # Reference lines
    ax1.axhline(baseline, color=C_BASE, ls="--", lw=1.5,
                label="Warm baseline (%.1f C)" % baseline)
    ax1.axhline(cold_median, color=C_COLD, ls="--", lw=1.5,
                label="Cold-room median (%.1f C)" % cold_median)

    # Entry marker + annotation
    ax1.axvline(onset_time, color=C_ENTRY, lw=2.2, zorder=5)
    ax1.scatter([times[onset_i]], [entry_temp], s=170, color=C_ENTRY,
                edgecolor="white", zorder=6)
    ax1.annotate(
        "ENTRY: %s\n%.1f C  (cooling begins)" % (rows[onset_i][2], entry_temp),
        xy=(times[onset_i], entry_temp),
        xytext=(20, 30), textcoords="offset points",
        fontsize=13, fontweight="bold", color=C_ENTRY,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_ENTRY, lw=1.5),
        arrowprops=dict(arrowstyle="->", color=C_ENTRY, lw=1.8))

    # Stabilised-cold marker + annotation
    if cold_i is not None:
        ax1.axvline(cold_time, color=C_STABLE, lw=2.0, ls="-.", zorder=5)
        ax1.scatter([times[cold_i]], [temps[cold_i]], s=150, color=C_STABLE,
                    edgecolor="white", zorder=6)
        ax1.annotate(
            "STABILISED COLD: %s\n%.1f C" % (rows[cold_i][2], temps[cold_i]),
            xy=(times[cold_i], temps[cold_i]),
            xytext=(15, -55), textcoords="offset points",
            fontsize=12, color=C_STABLE,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_STABLE, lw=1.3),
            arrowprops=dict(arrowstyle="->", color=C_STABLE, lw=1.6))

    # Phase text labels near the top
    ymax = baseline + span * 0.12
    ax1.set_ylim(min(temps) - span * 0.1, ymax + span * 0.05)
    ax1.text(times[0], ymax, "  WARM", va="bottom", ha="left",
             fontsize=12, color="#9c5a3c", fontweight="bold")
    ax1.text(onset_time, ymax, "  COOLING", va="bottom", ha="left",
             fontsize=12, color="#b58a1b", fontweight="bold")
    if cold_time is not None:
        ax1.text(cold_time, ymax, "  COLD (stable)", va="bottom", ha="left",
                 fontsize=12, color="#3f7a47", fontweight="bold")

    ax1.set_ylabel("Chip temperature (C)")
    ax1.set_xlabel("Time (HH:MM)")
    ax1.set_title("Cold-room entry detection - temperature cooling curve",
                  fontsize=17, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=11, framealpha=0.95)

    _fmt_time_axis(ax1)
    fig.autofmt_xdate()

    # Caption tying it together for business readers.
    fig.text(
        0.5, 0.005,
        "Entry = onset of the sustained decline (temperature leaves the warm "
        "baseline and keeps falling toward the cold median), not where it "
        "finally stabilises.",
        ha="center", fontsize=11, style="italic", color="#444444")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print("Saved %s" % out_png)


def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else INPUT_CSV
    out_png = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PNG
    cold_median = float(sys.argv[3]) if len(sys.argv) > 3 else None

    rows = load_series(in_csv)
    if not rows:
        raise SystemExit("No usable rows loaded from " + in_csv)
    res = detect_entry(rows, cold_median)
    plot(rows, res, out_png)


if __name__ == "__main__":
    main()
