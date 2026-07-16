"""
Cold-room ENTRY-TIME detector.

Given the temperature time-series of a single package (already filtered to the
cold-room WAPs), decide WHEN the package physically entered the cold room.

The problem
-----------
A package's chip temperature does NOT drop the instant it enters the cold room.
It follows a gradual cooling curve: a warm plateau, then a sustained decline,
then a flat tail that stabilises around the cold-room MEDIAN temperature.

The entry time is the ONSET of the sustained decline - the "knee" where the
warm plateau turns into a downward trend - NOT the point where the temperature
finally reaches / stabilises at the cold median.

Method
------
Signals are combined so no single noisy reading can move the answer, and the
IRREGULAR sampling interval is neutralised up front:

  0. Resample        - the raw readings (2, 3, 5... minute gaps) are linearly
                        interpolated onto a UNIFORM time grid, so every slope /
                        trend test sees equal time steps and a long gap can no
                        longer distort the measured cooling rate. The detected
                        onset is snapped back to the nearest real EVENTTIME.
  1. Smoothing        - short centred moving average kills single-sample noise.
  2. Cold anchor      - locate where the curve first REACHES and STAYS at the
                        cold-room median (median passed in / estimated from tail).
  3. Backward walk    - from that stabilisation point, walk UP the cooling curve
                        (backwards in time) while the per-minute rise stays steep,
                        stopping at the plateau. That stop = onset of decline.
  4. Segment check    - the onset->cold segment must be a large, predominantly
                        monotonic drop (a real cooling curve, not a wobble).
  5. Fallback         - if the curve never stabilises cold, use the first steep,
                        sustained negative slope (rate-of-change) instead.

Cold-room EXIT is detected by the MIRROR of the same idea: anchor on the stable
WARM (ambient) region that FOLLOWS the cold hold, then walk backwards DOWN the
warming curve until the rise flattens into the cold plateau. That knee is the
exit time (the moment the package leaves the cold room and starts warming).

Thresholds are expressed as FRACTIONS of the total cooling span
(baseline_warm - cold_median) so the detector generalises across packages that
cool by different absolute amounts / at different rates.

Inputs  (CSV):
  * entry_exit.csv   EVENTTIME, IDNODECHIPTEMPERATURE
Output  (CSV):
  * entry_exit_detection.csv   per-reading Temp/Slope/GapMin/Phase + ENTRY flag

Run:  python entry_exit.py [input_csv] [output_csv] [cold_median]
"""

import csv
import sys
from datetime import datetime, timedelta

INPUT_CSV = "entry_exit.csv"
OUTPUT_CSV = "entry_exit_detection.csv"

# --------------------------------------------------------------------------- #
# TUNABLES  (fractions are of the cooling span = baseline_warm - cold_median)
# --------------------------------------------------------------------------- #
RESAMPLE_MINUTES = 1.0   # uniform grid step the series is interpolated onto
SMOOTH_MINUTES = 3.0     # moving-average width in MINUTES (grid-based); keeps knee
REACH_FRAC = 0.10        # "reached cold" = within 10% of span above the median
STABLE_MINUTES = 10.0    # must stay cold this long to count as stabilised
FLAT_FRAC = 0.02         # onset stop: rise-rate below this*span (C/min) = plateau
MIN_SPAN = 2.0           # C; below this there is no meaningful cooling to detect
MIN_SEG_DROP_FRAC = 0.30 # onset->cold must drop at least this fraction of span
MONO_FRAC = 0.60         # >= this fraction of onset->cold steps must be downward
FALLBACK_SLOPE_WIN = 8.0 # minutes; look-ahead window for the slope fallback

TIME_FORMATS = (
    "%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S",
    "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M",
)


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def _find_col(fieldnames, *keywords):
    for name in fieldnames:
        low = name.lower().replace(" ", "").replace("_", "")
        if all(k in low for k in keywords):
            return name
    return None


def _parse_time(text):
    text = (text or "").strip()
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError("Unrecognised EVENTTIME format: %r" % text)


def _minutes(a, b):
    """Absolute minutes between two datetimes (>= a tiny epsilon)."""
    return max(abs((b - a).total_seconds()) / 60.0, 1e-9)


def moving_average(vals, window):
    """Centred moving average; edges shrink the window symmetrically."""
    n = len(vals)
    if window <= 1 or n == 0:
        return list(vals)
    half = window // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = vals[lo:hi]
        out.append(sum(seg) / len(seg))
    return out


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _interp_temp(times, temps, target):
    """Linear interpolation of temperature at datetime `target`."""
    if target <= times[0]:
        return temps[0]
    if target >= times[-1]:
        return temps[-1]
    lo, hi = 0, len(times) - 1
    while hi - lo > 1:                       # binary search for bracketing pair
        mid = (lo + hi) // 2
        if times[mid] <= target:
            lo = mid
        else:
            hi = mid
    t0, t1 = times[lo], times[hi]
    span = (t1 - t0).total_seconds()
    if span <= 0:
        return temps[lo]
    frac = (target - t0).total_seconds() / span
    return temps[lo] + (temps[hi] - temps[lo]) * frac


def resample_uniform(times, temps, step_min):
    """Interpolate an irregularly-sampled series onto a uniform time grid so that
    every consecutive step is exactly `step_min` apart. Returns (times, temps)."""
    total_min = (times[-1] - times[0]).total_seconds() / 60.0
    n_steps = max(1, int(round(total_min / step_min)))
    step = timedelta(minutes=step_min)
    gtimes, gtemps = [], []
    for k in range(n_steps + 1):
        target = times[0] + step * k
        gtimes.append(target)
        gtemps.append(_interp_temp(times, temps, target))
    return gtimes, gtemps


def _nearest_row(times, target):
    """Index of the actual reading whose timestamp is closest to `target`."""
    if target is None:
        return None
    best_i, best_d = 0, None
    for i, t in enumerate(times):
        d = abs((t - target).total_seconds())
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return best_i


# --------------------------------------------------------------------------- #
# LOAD
# --------------------------------------------------------------------------- #
def load_series(path):
    """Return a time-sorted list of (datetime, temp, raw_time_string)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        t_col = _find_col(fields, "event", "time") or _find_col(fields, "time")
        v_col = _find_col(fields, "chip", "temp") or _find_col(fields, "temp")
        if not (t_col and v_col):
            raise SystemExit("input CSV needs EVENTTIME and TEMPERATURE columns")
        rows = []
        for r in reader:
            raw_t = (r.get(t_col) or "").strip()
            raw_v = (r.get(v_col) or "").strip()
            if not raw_t or not raw_v:
                continue
            try:
                rows.append((_parse_time(raw_t), float(raw_v), raw_t))
            except ValueError:
                continue
        rows.sort(key=lambda x: x[0])
        return rows


# --------------------------------------------------------------------------- #
# DETECT
# --------------------------------------------------------------------------- #
def _slopes_per_min(times, sm):
    """Backward slope (C/min) into each index; index 0 is 0."""
    slopes = [0.0]
    for i in range(1, len(sm)):
        slopes.append((sm[i] - sm[i - 1]) / _minutes(times[i - 1], times[i]))
    return slopes


def _first_stable_cold(times, sm, band):
    """First index whose smoothed value is <= band AND stays cold-ish for
    STABLE_MINUTES afterwards (or until the series ends)."""
    n = len(sm)
    tol = band  # already includes the reach margin
    for i in range(n):
        if sm[i] > band:
            continue
        ok = True
        for j in range(i + 1, n):
            if _minutes(times[i], times[j]) > STABLE_MINUTES:
                break
            if sm[j] > tol:
                ok = False
                break
        if ok:
            return i
    return None


def _walk_back_to_onset(times, sm, cold_idx, flat_rate, warm_level):
    """From the stabilisation index, climb backwards up the cooling slope until
    the WARM plateau is reached and the rise flattens (the knee). Local flat
    spots on the way down do not stop the walk - only genuine arrival back at
    the warm plateau does."""
    k = cold_idx
    while k - 1 >= 0:
        rise_rate = (sm[k - 1] - sm[k]) / _minutes(times[k - 1], times[k])
        reached_warm = sm[k] >= warm_level
        if rise_rate >= flat_rate or not reached_warm:
            k -= 1
        else:
            break
    return k


def _segment_is_real(sm, onset, cold_idx, min_drop, noise):
    """Confirm onset->cold is a big, predominantly monotonic decline."""
    if cold_idx <= onset:
        return False
    drop = sm[onset] - sm[cold_idx]
    if drop < min_drop:
        return False
    down = sum(1 for k in range(onset + 1, cold_idx + 1)
               if sm[k] <= sm[k - 1] + noise)
    steps = cold_idx - onset
    return steps > 0 and (down / steps) >= MONO_FRAC


def _fallback_slope_onset(times, sm, warm_gate, flat_rate):
    """First index that begins a steep, sustained downward slope."""
    n = len(sm)
    for i in range(n - 1):
        if sm[i] < warm_gate:
            continue
        # net slope over the look-ahead window
        j = i
        while j + 1 < n and _minutes(times[i], times[j + 1]) <= FALLBACK_SLOPE_WIN:
            j += 1
        if j == i:
            continue
        rate = (sm[i] - sm[j]) / _minutes(times[i], times[j])  # positive = cooling
        if rate >= flat_rate and sm[j] < sm[i]:
            return i
    return None


# --- EXIT-side mirrors of the entry helpers ------------------------------- #
def _last_at_or_below(sm, band):
    """Last index whose smoothed value is <= band (i.e. end of the cold hold)."""
    idx = None
    for i, v in enumerate(sm):
        if v <= band:
            idx = i
    return idx


def _first_stable_warm(times, sm, band, start=0):
    """First index at/after `start` whose smoothed value is >= band AND stays
    warm for STABLE_MINUTES afterwards (or until the series ends)."""
    n = len(sm)
    for i in range(start, n):
        if sm[i] < band:
            continue
        ok = True
        for j in range(i + 1, n):
            if _minutes(times[i], times[j]) > STABLE_MINUTES:
                break
            if sm[j] < band:
                ok = False
                break
        if ok:
            return i
    return None


def _walk_back_to_exit(times, sm, warm_idx, flat_rate, cold_level):
    """From the stable-warm index, descend backwards down the warming slope until
    the COLD plateau is reached and the rise flattens (the knee). Local flat
    spots on the warming shoulder do not stop the walk - only genuine arrival
    back at the cold plateau does."""
    k = warm_idx
    while k - 1 >= 0:
        rise_rate = (sm[k] - sm[k - 1]) / _minutes(times[k - 1], times[k])
        reached_cold = sm[k] <= cold_level
        if rise_rate >= flat_rate or not reached_cold:
            k -= 1
        else:
            break
    return k


def _segment_is_rise(sm, onset, warm_idx, min_rise, noise):
    """Confirm onset->warm is a big, predominantly monotonic incline."""
    if warm_idx <= onset:
        return False
    rise = sm[warm_idx] - sm[onset]
    if rise < min_rise:
        return False
    up = sum(1 for k in range(onset + 1, warm_idx + 1)
             if sm[k] >= sm[k - 1] - noise)
    steps = warm_idx - onset
    return steps > 0 and (up / steps) >= MONO_FRAC


def detect_entry(rows, cold_median=None):
    """Return a result dict describing the detected cold-room entry point.

    Irregular sampling is handled by interpolating onto a uniform time grid
    before any trend/slope analysis, then snapping the answer back to the
    nearest actual reading."""
    times = [r[0] for r in rows]
    temps = [r[1] for r in rows]
    n = len(temps)
    if n < 3:
        return {"ok": False, "reason": "need at least 3 readings"}

    # --- neutralise irregular sampling: uniform-grid interpolation ---
    gtimes, gtemps = resample_uniform(times, temps, RESAMPLE_MINUTES)
    win_pts = max(1, int(round(SMOOTH_MINUTES / RESAMPLE_MINUTES)))
    gsm = moving_average(gtemps, win_pts)
    slopes = _slopes_per_min(gtimes, gsm)

    baseline = max(gsm)                      # warm plateau level
    if cold_median is None:                  # robust: median of the coldest quartile
        ssorted = sorted(temps)              # (the tail may be warm after an exit)
        k = max(3, len(ssorted) // 4)
        cold_median = _median(ssorted[:k])

    span = baseline - cold_median
    result = {
        "ok": False, "gtimes": gtimes, "gsm": gsm, "slopes": slopes,
        "baseline": baseline, "cold_median": cold_median, "span": span,
        "step": RESAMPLE_MINUTES, "onset_time": None, "cold_time": None,
        "onset": None, "cold_idx": None, "method": None, "reason": None,
        "exit_onset": None, "exit_warm": None, "exit_time": None,
        "warm_time": None, "exit_method": None,
    }
    if span < MIN_SPAN:
        result["reason"] = "cooling span %.2f C below MIN_SPAN" % span
        return result

    flat_rate = span * FLAT_FRAC             # C/min plateau threshold
    noise = span * 0.03
    band = cold_median + span * REACH_FRAC
    warm_band = baseline - span * REACH_FRAC
    warm_gate = cold_median + span * 0.5     # onset must sit in the warm half

    # All searches run on the uniform grid, so equal time steps everywhere.
    g_cold = _first_stable_cold(gtimes, gsm, band)
    onset_g, method = None, None
    if g_cold is not None:
        cand = _walk_back_to_onset(gtimes, gsm, g_cold, flat_rate, warm_band)
        if _segment_is_real(gsm, cand, g_cold, span * MIN_SEG_DROP_FRAC, noise):
            onset_g, method = cand, "cold-anchored backward walk (resampled)"

    if onset_g is None:                      # fallback: first steep sustained slope
        cand = _fallback_slope_onset(gtimes, gsm, warm_gate, flat_rate)
        if cand is not None:
            onset_g, method = cand, "rate-of-change fallback (resampled)"

    if onset_g is None:
        result["reason"] = "no sustained cooling trend found"
        return result

    onset_time = gtimes[onset_g]
    cold_time = gtimes[g_cold] if g_cold is not None else None
    result.update(
        ok=True, method=method, onset_time=onset_time, cold_time=cold_time,
        onset=_nearest_row(times, onset_time),
        cold_idx=_nearest_row(times, cold_time),
    )

    # ----------------------------------------------------------------- #
    # EXIT: mirror of the entry logic on the WARMING side. Anchor on the
    # stable warm (ambient) region that FOLLOWS the cold hold, then walk
    # backwards down the warming curve to the knee = exit time.
    # ----------------------------------------------------------------- #
    cold_end = _last_at_or_below(gsm, band)          # end of the cold hold
    if cold_end is not None:
        g_warm = _first_stable_warm(gtimes, gsm, warm_band, start=cold_end + 1)
        if g_warm is not None:
            cand = _walk_back_to_exit(gtimes, gsm, g_warm, flat_rate, band)
            if _segment_is_rise(gsm, cand, g_warm,
                                 span * MIN_SEG_DROP_FRAC, noise):
                exit_time = gtimes[cand]
                warm_time = gtimes[g_warm]
                result.update(
                    exit_onset=_nearest_row(times, exit_time),
                    exit_warm=_nearest_row(times, warm_time),
                    exit_time=exit_time, warm_time=warm_time,
                    exit_method="warm-anchored backward walk (resampled)",
                )
    return result


# --------------------------------------------------------------------------- #
# REPORT
# --------------------------------------------------------------------------- #
def _phase(t, onset_time, cold_time, exit_time=None, warm_time=None):
    if onset_time is None or t < onset_time:
        return "WARM"
    if exit_time is not None and t >= exit_time:
        if warm_time is not None and t >= warm_time:
            return "AMBIENT"
        return "WARMING"
    if cold_time is not None and t >= cold_time:
        return "COLD"
    return "COOLING"


def print_report(rows, res):
    print("=" * 72)
    print("COLD-ROOM ENTRY / EXIT DETECTION")
    print("=" * 72)
    print("Readings            : %d" % len(rows))
    print("Warm baseline       : %.2f C" % res.get("baseline", 0.0))
    print("Cold-room median    : %.2f C" % res.get("cold_median", 0.0))
    print("Cooling span        : %.2f C" % res.get("span", 0.0))
    if not res["ok"]:
        print("Result              : NOT DETECTED (%s)" % res.get("reason"))
        print("=" * 72)
        return
    onset = res["onset"]
    print("Detection method    : %s" % res["method"])
    print("-" * 72)
    print("ENTRY TIME          : %s   (nearest actual reading)" % rows[onset][2])
    print("Temp at entry       : %.2f C" % rows[onset][1])
    if res.get("onset_time") is not None:
        print("Cooling onset (grid): %s"
              % res["onset_time"].strftime("%d-%m-%Y %H:%M"))
    if res["cold_idx"] is not None:
        c = res["cold_idx"]
        mins = _minutes(rows[onset][0], rows[c][0])
        print("Stabilised cold at  : %s  (%.2f C)" % (rows[c][2], rows[c][1]))
        print("Cool-down duration  : %.0f min  (drop %.2f C)"
              % (mins, rows[onset][1] - rows[c][1]))
    if res.get("exit_onset") is not None:
        xo = res["exit_onset"]
        print("-" * 72)
        print("EXIT TIME           : %s   (nearest actual reading)"
              % rows[xo][2])
        print("Temp at exit        : %.2f C" % rows[xo][1])
        print("Warming onset (grid): %s"
              % res["exit_time"].strftime("%d-%m-%Y %H:%M"))
        if res.get("exit_warm") is not None:
            xw = res["exit_warm"]
            mins2 = _minutes(rows[xo][0], rows[xw][0])
            print("Reached ambient at  : %s  (%.2f C)"
                  % (rows[xw][2], rows[xw][1]))
            print("Warm-up duration    : %.0f min  (rise %.2f C)"
                  % (mins2, rows[xw][1] - rows[xo][1]))
    else:
        print("-" * 72)
        print("EXIT TIME           : not detected "
              "(no warming trend after cold hold)")
    print("=" * 72)


def write_output(rows, res, path):
    onset = res.get("onset")
    cold_idx = res.get("cold_idx")
    exit_onset = res.get("exit_onset")
    exit_warm = res.get("exit_warm")
    onset_time = res.get("onset_time")
    cold_time = res.get("cold_time")
    exit_time = res.get("exit_time")
    warm_time = res.get("warm_time")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EVENTTIME", "Temperature", "SlopePerMin",
                    "GapMin", "Phase", "Marker"])
        for i, (t, temp, raw) in enumerate(rows):
            if i == 0:
                slope, gap = 0.0, 0.0
            else:
                gap = _minutes(rows[i - 1][0], t)
                slope = (temp - rows[i - 1][1]) / gap
            marker = ""
            if i == onset:
                marker = "<== ENTRY"
            elif i == cold_idx:
                marker = "<== STABLE COLD"
            elif i == exit_onset:
                marker = "<== EXIT"
            elif i == exit_warm:
                marker = "<== REACHED AMBIENT"
            w.writerow([raw, temp, round(slope, 4), round(gap, 1),
                        _phase(t, onset_time, cold_time, exit_time, warm_time),
                        marker])


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else INPUT_CSV
    out_csv = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_CSV
    cold_median = float(sys.argv[3]) if len(sys.argv) > 3 else None

    rows = load_series(in_csv)
    if not rows:
        raise SystemExit("No usable rows loaded from " + in_csv)

    res = detect_entry(rows, cold_median)
    print_report(rows, res)
    write_output(rows, res, out_csv)
    print("Saved %s" % out_csv)


if __name__ == "__main__":
    main()
