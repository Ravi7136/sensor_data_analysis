"""
Cold-room ENTRY / EXIT detection for the supply-chain package-movement use case.

Input : the device export CSV (schema produced by generate_synthetic_data.py):
    EVENTTIME, HARDWARENAME, IDNODESERIALNUMBER, IDNODECHIPTEMPARATURE,
    LATITUDE, LONGITUDE, RSSI1..RSSI5
Output: a table (and CSV) of cold-room visits with entry time, exit time,
    duration and supporting temperature stats.

DETECTION PHILOSOPHY
--------------------
The robust discriminator is LOCATION, not temperature:

  * Each row is already resolved by the server to the *closest* WAP. A package
    that is physically inside the cold room is therefore consistently attributed
    to the small cluster of WAPs that live inside the cold room, and heard by
    them with strong RSSI.
  * Temperature is only CORROBORATING evidence. In winter the outside can be
    ~4 C and the cold room ~3 C, so the temperature gradient is tiny and cannot
    be used to tell "inside" from "outside". The location signal still can.

So the pipeline is:
  1. Load + clean the readings (fold RSSI1..5 into one "closest hearing" value).
  2. Identify which WAPs form the cold-room cluster (auto-inferred from the
     longest strong-RSSI dwell; if several long dwells exist AND a clear
     temperature gradient is present, the coldest dwell is chosen). Can be
     overridden with COLD_ROOM_WAPS.
  3. Mark each reading inside/outside (attributed to cold-room cluster + strong
     RSSI), smooth with a time window, and run a hysteresis state machine to
     emit ENTRY / EXIT events.

All tunables live in the CONFIG block.  Run:  python detect_cold_room.py
"""

import csv
import math
import statistics
from datetime import datetime, timedelta

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
INPUT_CSV = "synthetic_cold_room_data.csv"
OUTPUT_EVENTS_CSV = "cold_room_events.csv"
TIME_FMT = "%d-%m-%Y %H:%M"

# Cold-room WAP cluster. Leave as None to auto-infer from the data, or set an
# explicit list of HARDWARENAMEs if the facility knows which WAPs are inside.
COLD_ROOM_WAPS = None  # e.g. ["XX-4A-ap03", "XX-2B-ap02", "XX-2A-ap01", "XX-7B-ap04"]

# --- RSSI / location gating ---
# RSSI quality bands (dBm), from the standard reference table:
#   -30 excellent (very close) | -40 excellent (same room) | -50 very good
#   -60 good (reliable) | -70 fair (edge of coverage) | -80 weak (unreliable)
#   -90 very weak (almost disconnected). Less negative = stronger = closer.
# A package physically inside the cold room should be heard by the closest
# cold-room WAP at "fair or better". Cold rooms are metal-walled (extra
# attenuation), so -75 is a tolerant default; raise toward -65/-70 for a
# tighter room boundary.
STRONG_RSSI_DBM = -75        # strongest hearing >= this (closer to 0) => "near"

# --- state machine (time based, because the cadence is irregular) ---
SMOOTH_WINDOW_MIN = 15       # trailing window used to smooth the inside signal
INSIDE_FRACTION = 0.5        # >= this fraction of the window inside -> inside
MIN_DWELL_MIN = 10           # ignore cold-room visits shorter than this
MIN_OUT_MIN = 10             # ignore momentary drop-outs shorter than this

# --- cold-room auto-inference ---
CLUSTER_RADIUS_M = 60        # WAPs within this radius are the "same location"
SEG_MAX_GAP_MIN = 30         # a location dwell may tolerate gaps up to this
INFER_MIN_DWELL_MIN = 15     # a candidate room dwell must last at least this
TEMP_GRADIENT_C = 3.0        # gradient needed to prefer the *coldest* dwell


# --------------------------------------------------------------------------- #
# LOADING / CLEANING
# --------------------------------------------------------------------------- #
def _to_rssi(value):
    """Parse one RSSI cell. Blank or 0 -> missing (None). RSSI is negative dBm."""
    value = (value or "").strip()
    if value == "":
        return None
    try:
        num = int(float(value))
    except ValueError:
        return None
    if num == 0:            # 0 in the export means "not heard that time"
        return None
    return num


def load_rows(path):
    """Load and clean the CSV into a list of timestamp-sorted reading dicts."""
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            rssis = [_to_rssi(raw.get(f"RSSI{i}")) for i in range(1, 6)]
            valid = [r for r in rssis if r is not None]
            temp_raw = (raw.get("IDNODECHIPTEMPARATURE") or "").strip()
            rows.append({
                "dt": datetime.strptime(raw["EVENTTIME"].strip(), TIME_FMT),
                "wap": raw["HARDWARENAME"].strip(),
                "serial": raw["IDNODESERIALNUMBER"].strip(),
                "temp": float(temp_raw) if temp_raw != "" else None,
                "lat": float(raw["LATITUDE"]),
                "lon": float(raw["LONGITUDE"]),
                # strongest hearing = closest; None if the WAP heard nothing valid
                "rssi": max(valid) if valid else None,
                "n_hearings": len(valid),
            })
    rows.sort(key=lambda r: r["dt"])
    return rows


# --------------------------------------------------------------------------- #
# GEO HELPERS
# --------------------------------------------------------------------------- #
def dist_m(lat1, lon1, lat2, lon2):
    """Approximate planar distance in metres (fine for a single station)."""
    R = 6371000.0
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    x = math.radians(lon2 - lon1) * math.cos(mean_lat)
    y = math.radians(lat2 - lat1)
    return R * math.hypot(x, y)


def _minutes(a, b):
    return abs((a - b).total_seconds()) / 60.0


# --------------------------------------------------------------------------- #
# COLD-ROOM CLUSTER INFERENCE
# --------------------------------------------------------------------------- #
def _segment_by_location(rows):
    """Group consecutive readings into dwell segments at one physical location.

    A new reading joins the current segment if it is within CLUSTER_RADIUS_M of
    the segment centroid and the time gap is <= SEG_MAX_GAP_MIN; otherwise a new
    segment starts. This turns the trace into 'moved here and stayed a while'
    blocks.
    """
    segments = []
    seg = None
    for r in rows:
        if seg is None:
            seg = _new_segment(r)
            continue
        near = dist_m(seg["clat"], seg["clon"], r["lat"], r["lon"]) <= CLUSTER_RADIUS_M
        in_time = _minutes(r["dt"], seg["end"]) <= SEG_MAX_GAP_MIN
        if near and in_time:
            _extend_segment(seg, r)
        else:
            segments.append(seg)
            seg = _new_segment(r)
    if seg is not None:
        segments.append(seg)
    return segments


def _new_segment(r):
    return {
        "start": r["dt"], "end": r["dt"],
        "clat": r["lat"], "clon": r["lon"], "n": 1,
        "waps": {r["wap"]}, "temps": ([r["temp"]] if r["temp"] is not None else []),
    }


def _extend_segment(seg, r):
    seg["n"] += 1
    # running-mean centroid keeps the dwell anchored to its location
    seg["clat"] += (r["lat"] - seg["clat"]) / seg["n"]
    seg["clon"] += (r["lon"] - seg["clon"]) / seg["n"]
    seg["end"] = r["dt"]
    seg["waps"].add(r["wap"])
    if r["temp"] is not None:
        seg["temps"].append(r["temp"])


def infer_cold_room_waps(rows):
    """Return (set_of_wap_names, info_dict) for the inferred cold-room cluster."""
    segments = _segment_by_location(rows)
    dwells = [s for s in segments if _minutes(s["end"], s["start"]) >= INFER_MIN_DWELL_MIN]
    if not dwells:
        raise RuntimeError("No sustained dwell found; cannot infer the cold room.")

    def mean_temp(s):
        return statistics.mean(s["temps"]) if s["temps"] else float("nan")

    longest = max(dwells, key=lambda s: _minutes(s["end"], s["start"]))
    temps = [mean_temp(s) for s in dwells if s["temps"]]
    gradient = (max(temps) - min(temps)) if len(temps) >= 2 else 0.0

    if gradient >= TEMP_GRADIENT_C:
        # Clear temperature gradient (e.g. summer): the coldest dwell is the room.
        chosen = min((s for s in dwells if s["temps"]), key=mean_temp)
        reason = f"coldest dwell (temp gradient {gradient:.1f} C across dwells)"
    else:
        # Small/no gradient (e.g. winter): fall back to the dominant dwell.
        chosen = longest
        reason = "longest strong-RSSI dwell (temperature gradient too small)"

    info = {
        "reason": reason,
        "mean_temp": mean_temp(chosen),
        "duration_min": _minutes(chosen["end"], chosen["start"]),
        "n_dwells": len(dwells),
    }
    return set(chosen["waps"]), info


# --------------------------------------------------------------------------- #
# INSIDE / OUTSIDE + STATE MACHINE
# --------------------------------------------------------------------------- #
def mark_inside(rows, cold_waps):
    """Per-row raw 'inside' flag: attributed to a cold-room WAP with strong RSSI."""
    for r in rows:
        r["inside_raw"] = (
            r["wap"] in cold_waps
            and r["rssi"] is not None
            and r["rssi"] >= STRONG_RSSI_DBM
        )
    return rows


def smooth_inside(rows):
    """Trailing-window fraction of inside_raw -> debounced boolean per row."""
    n = len(rows)
    j = 0
    for i in range(n):
        t_i = rows[i]["dt"]
        while _minutes(t_i, rows[j]["dt"]) > SMOOTH_WINDOW_MIN and rows[j]["dt"] < t_i:
            j += 1
        window = rows[j:i + 1]
        frac = sum(1 for w in window if w["inside_raw"]) / len(window)
        rows[i]["inside"] = frac >= INSIDE_FRACTION
    return rows


def detect_visits(rows):
    """Run the hysteresis state machine and return a list of cold-room visits."""
    # 1) contiguous smoothed-inside runs
    runs = []
    start = None
    for i, r in enumerate(rows):
        if r["inside"] and start is None:
            start = i
        elif not r["inside"] and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(rows) - 1))

    # 2) merge runs separated by a short drop-out (< MIN_OUT_MIN)
    merged = []
    for run in runs:
        if merged and _minutes(rows[run[0]]["dt"], rows[merged[-1][1]]["dt"]) < MIN_OUT_MIN:
            merged[-1] = (merged[-1][0], run[1])
        else:
            merged.append(list(run))

    # 3) build visit records, snapping entry/exit to the raw inside edges
    visits = []
    last_idx = len(rows) - 1
    for a, b in merged:
        inside_idx = [k for k in range(a, b + 1) if rows[k]["inside_raw"]]
        if not inside_idx:
            continue
        entry_i, exit_i = inside_idx[0], inside_idx[-1]
        if _minutes(rows[exit_i]["dt"], rows[entry_i]["dt"]) < MIN_DWELL_MIN:
            continue
        temps = [rows[k]["temp"] for k in range(entry_i, exit_i + 1) if rows[k]["temp"] is not None]
        visits.append({
            "entry": rows[entry_i]["dt"],
            "exit": rows[exit_i]["dt"],
            "ongoing": exit_i >= last_idx,
            "duration_min": _minutes(rows[exit_i]["dt"], rows[entry_i]["dt"]),
            "mean_temp": statistics.mean(temps) if temps else None,
            "min_temp": min(temps) if temps else None,
            "n_readings": exit_i - entry_i + 1,
        })
    return visits


# --------------------------------------------------------------------------- #
# REPORTING
# --------------------------------------------------------------------------- #
def _fmt_duration(minutes):
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}h {m:02d}m"


def report(visits, cold_waps, info):
    print("=" * 78)
    print("COLD-ROOM ENTRY / EXIT DETECTION")
    print("=" * 78)
    print(f"Cold-room WAP cluster : {sorted(cold_waps)}")
    if info:
        print(f"  inferred by         : {info['reason']}")
        print(f"  dwell mean temp     : {info['mean_temp']:.2f} C over "
              f"{_fmt_duration(info['duration_min'])} ({info['n_dwells']} dwell(s) seen)")
    print("-" * 78)
    if not visits:
        print("No cold-room visits detected.")
        return
    for i, v in enumerate(visits, 1):
        exit_str = "ONGOING (trace ended while inside)" if v["ongoing"] else v["exit"].strftime(TIME_FMT)
        print(f"Visit #{i}")
        print(f"  ENTRY    : {v['entry'].strftime(TIME_FMT)}")
        print(f"  EXIT     : {exit_str}")
        print(f"  DURATION : {_fmt_duration(v['duration_min'])}")
        temp_txt = "n/a" if v["mean_temp"] is None else f"mean {v['mean_temp']:.2f} C, min {v['min_temp']:.2f} C"
        print(f"  TEMP     : {temp_txt}   ({v['n_readings']} readings)")
        print("-" * 78)


def save_events(visits, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["VISIT", "ENTRY_TIME", "EXIT_TIME", "ONGOING",
                    "DURATION_MIN", "MEAN_TEMP_C", "MIN_TEMP_C", "N_READINGS"])
        for i, v in enumerate(visits, 1):
            w.writerow([
                i, v["entry"].strftime(TIME_FMT),
                "" if v["ongoing"] else v["exit"].strftime(TIME_FMT),
                v["ongoing"], round(v["duration_min"], 1),
                "" if v["mean_temp"] is None else round(v["mean_temp"], 2),
                "" if v["min_temp"] is None else round(v["min_temp"], 2),
                v["n_readings"],
            ])


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    rows = load_rows(INPUT_CSV)
    if not rows:
        print("No data loaded.")
        return

    if COLD_ROOM_WAPS:
        cold_waps, info = set(COLD_ROOM_WAPS), None
    else:
        cold_waps, info = infer_cold_room_waps(rows)

    mark_inside(rows, cold_waps)
    smooth_inside(rows)
    visits = detect_visits(rows)

    report(visits, cold_waps, info)
    save_events(visits, OUTPUT_EVENTS_CSV)
    print(f"Events written to {OUTPUT_EVENTS_CSV}")


if __name__ == "__main__":
    main()
