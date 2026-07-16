"""
Cold-room EXIT-TIME visualiser.

Mirror of entry_exit_viz.py for the WARMING side. Draws an annotated,
business-readable chart that JUSTIFIES the exit timestamp produced by
entry_exit.py, reusing that module's detector directly so the picture can never
disagree with the algorithm.

The figure is a single annotated panel:

  Temperature vs time
         * raw readings (dots, spacing shows the irregular sampling)
         * smoothed uniform-grid curve (line)
         * cold-median and warm-baseline (ambient) reference lines
         * COLD / WARMING / AMBIENT phases shaded
         * EXIT point and REACHED-AMBIENT point marked + annotated

Inputs  (CSV):
  * entry_exit.csv   EVENTTIME, IDNODECHIPTEMPERATURE
Output  (PNG):
  * exit_viz.png

Run:  python exit_viz.py [input_csv] [output_png] [cold_median]

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

OUTPUT_PNG = "exit_viz.png"

# Colour palette (colour-blind friendly) - shared with the entry visual.
C_RAW = "#4C72B0"       # raw readings
C_SMOOTH = "#DD8452"     # smoothed grid curve
C_BASE = "#8C8C8C"       # warm baseline (ambient)
C_COLD = "#55A868"       # cold median
C_EXIT = "#C44E52"      # exit marker
C_AMBIENT = "#8172B3"    # reached-ambient marker
SHADE_COLD = "#CDE7D0"
SHADE_WARMING = "#FCE8B2"
SHADE_AMBIENT = "#F6D6C2"


def _fmt_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())


def plot(rows, res, out_png):
    if not res.get("ok"):
        raise SystemExit("Detection failed (%s) - nothing to plot"
                         % res.get("reason"))
    if res.get("exit_onset") is None:
        raise SystemExit("No cold-room exit detected - nothing to plot")

    times = [r[0] for r in rows]
    temps = [r[1] for r in rows]
    gtimes = res["gtimes"]
    gsm = res["gsm"]
    baseline = res["baseline"]
    cold_median = res["cold_median"]
    span = res["span"]
    exit_i = res["exit_onset"]
    warm_i = res["exit_warm"]
    exit_time = res["exit_time"]
    warm_time = res["warm_time"]
    exit_temp = temps[exit_i]

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # ---------------- temperature ---------------- #
    # Phase shading: COLD -> WARMING -> AMBIENT
    ax1.axvspan(times[0], exit_time, color=SHADE_COLD, alpha=0.6,
                label="_nolegend_")
    if warm_time is not None:
        ax1.axvspan(exit_time, warm_time, color=SHADE_WARMING, alpha=0.6)
        ax1.axvspan(warm_time, times[-1], color=SHADE_AMBIENT, alpha=0.6)
    else:
        ax1.axvspan(exit_time, times[-1], color=SHADE_WARMING, alpha=0.6)

    # Curves
    ax1.plot(gtimes, gsm, color=C_SMOOTH, lw=2.5, zorder=3,
             label="_nolegend_")
    ax1.scatter(times, temps, s=28, color=C_RAW, zorder=4, alpha=0.85,
                label="_nolegend_")

    # Reference lines
    ax1.axhline(cold_median, color=C_COLD, ls="--", lw=1.5, label="_nolegend_")
    ax1.axhline(baseline, color=C_BASE, ls="--", lw=1.5, label="_nolegend_")

    # Exit marker + annotation
    ax1.axvline(exit_time, color=C_EXIT, lw=2.2, zorder=5)
    ax1.scatter([times[exit_i]], [exit_temp], s=170, color=C_EXIT,
                edgecolor="white", zorder=6)
    ax1.annotate(
        "EXIT: %s\n%.1f C  (warming begins)" % (rows[exit_i][2], exit_temp),
        xy=(times[exit_i], exit_temp),
        xytext=(20, 40), textcoords="offset points",
        fontsize=13, fontweight="bold", color=C_EXIT,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_EXIT, lw=1.5),
        arrowprops=dict(arrowstyle="->", color=C_EXIT, lw=1.8))

    # Reached-ambient marker + annotation
    if warm_i is not None:
        ax1.axvline(warm_time, color=C_AMBIENT, lw=2.0, ls="-.", zorder=5)
        ax1.scatter([times[warm_i]], [temps[warm_i]], s=150, color=C_AMBIENT,
                    edgecolor="white", zorder=6)
        ax1.annotate(
            "REACHED AMBIENT: %s\n%.1f C" % (rows[warm_i][2], temps[warm_i]),
            xy=(times[warm_i], temps[warm_i]),
            xytext=(-40, -60), textcoords="offset points",
            fontsize=12, color=C_AMBIENT,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_AMBIENT, lw=1.3),
            arrowprops=dict(arrowstyle="->", color=C_AMBIENT, lw=1.6))

    # Phase text labels near the top
    ymax = baseline + span * 0.12
    ax1.set_ylim(min(temps) - span * 0.1, ymax + span * 0.05)
    ax1.text(times[0], ymax, "  COLD (stable)", va="bottom", ha="left",
             fontsize=12, color="#3f7a47", fontweight="bold")
    ax1.text(exit_time, ymax, "  WARMING", va="bottom", ha="left",
             fontsize=12, color="#b58a1b", fontweight="bold")
    if warm_time is not None:
        ax1.text(warm_time, ymax, "  AMBIENT", va="bottom", ha="left",
                 fontsize=12, color="#9c5a3c", fontweight="bold")

    ax1.set_ylabel("Chip temperature (C)")
    ax1.set_xlabel("Time (HH:MM)")
    ax1.set_title("Cold-room exit detection - temperature warming curve",
                  fontsize=17, fontweight="bold")

    _fmt_time_axis(ax1)
    fig.autofmt_xdate()

    fig.tight_layout()
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
