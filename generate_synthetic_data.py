"""
Generate synthetic supply-chain package-movement data for the
'cold room ENTRY / EXIT detection' use case.

The output CSV mirrors the real device-export schema (one row per
timestamp, already resolved by the server to the closest WAP), i.e. columns:

    EVENTTIME, HARDWARENAME, IDNODESERIALNUMBER, IDNODECHIPTEMPARATURE,
    LATITUDE, LONGITUDE, RSSI1, RSSI2, RSSI3, RSSI4, RSSI5

Physical model (learned from the sample sheets seq-1 .. seq-11):
  * Package starts at ambient (~23.5 C) and is carried through the station
    (heard by mixed / outer WAPs).
  * On entering the cold room the chip temperature decays exponentially to a
    cold set-point (~6 C) and then stays stable; only the cold-room WAP
    cluster hears it, with strong RSSI.
  * The provided trace ends while still cold (no full exit).

Everything tunable lives in the CONFIG block below.
Run:  python generate_synthetic_data.py
"""

import csv
import math
import random
from datetime import datetime, timedelta

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
SEED = 42

START_TIME = datetime(2026, 4, 30, 9, 31)      # 30-04-2026 09:31
END_TIME = datetime(2026, 5, 1, 14, 30)        # 01-05-2026 14:30
TIME_FMT = "%d-%m-%Y %H:%M"

IDNODE_SERIAL = "123"

# Phase boundaries (minutes are derived from these timestamps).
COOLDOWN_START = datetime(2026, 4, 30, 9, 44)  # temperature starts dropping
COLD_REACHED = datetime(2026, 4, 30, 10, 9)    # cold set-point reached
END_WARM_START = datetime(2026, 5, 1, 14, 10)  # slight warming near the end

AMBIENT_TEMP = 23.5
COLD_SETPOINT = 6.0
COOLDOWN_TAU_MIN = 6.0                          # exp-decay time constant (min)
END_WARM_TEMP = 7.5

# Inter-row interval (minutes) and their weights -> irregular cadence.
# Mostly 1-2 min apart, sometimes 3-5, occasionally a longer 6-8 min gap.
INTERVAL_CHOICES = [1, 2, 3, 4, 5, 6, 7, 8]
INTERVAL_WEIGHTS = [34, 26, 14, 9, 7, 5, 3, 2]

# WAPs: (name, latitude, longitude)
COLD_ROOM_WAPS = [
    ("XX-4A-ap03", 35.06809, -89.977458),
    ("XX-2B-ap02", 35.067973, -89.977737),
    ("XX-2A-ap01", 35.068087, -89.977752),
    ("XX-7B-ap04", 35.067998, -89.977108),
]
TRANSIT_WAPS = [
    ("XX-RM150-ap12", 35.06831, -89.977074),
    ("XX-6C-ap10", 35.067826, -89.977146),
    ("XX-12C-ap09", 35.067846, -89.97634),
    ("XX-10B-ap05", 35.068087, -89.976599),
]

OUTPUT_CSV = "synthetic_cold_room_data.csv"

# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def quantize_half(value):
    """Round to nearest 0.5 (the real chip temperature is reported in 0.5 steps)."""
    return round(value * 2) / 2.0


def temperature_at(ts):
    """Return the package chip temperature for a given timestamp."""
    if ts < COOLDOWN_START:
        # Ambient / transit: ~23.5 with tiny jitter.
        return quantize_half(AMBIENT_TEMP + random.choice([-1.0, -0.5, 0.0, 0.0]))
    if ts < COLD_REACHED:
        # Exponential cool-down toward the cold set-point.
        minutes = (ts - COOLDOWN_START).total_seconds() / 60.0
        temp = COLD_SETPOINT + (AMBIENT_TEMP - COLD_SETPOINT) * math.exp(-minutes / COOLDOWN_TAU_MIN)
        return quantize_half(temp)
    if ts >= END_WARM_START:
        # Slight warming near the end of the trace (as seen in the real data).
        minutes = (ts - END_WARM_START).total_seconds() / 60.0
        span = max((END_TIME - END_WARM_START).total_seconds() / 60.0, 1.0)
        temp = COLD_SETPOINT + (END_WARM_TEMP - COLD_SETPOINT) * min(minutes / span, 1.0)
        return quantize_half(temp + random.choice([-0.5, 0.0, 0.0]))
    # Cold-stable: hovers around the set-point.
    return quantize_half(COLD_SETPOINT + random.choice([-1.0, -0.5, 0.0, 0.0, 0.5]))


def pick_wap(ts):
    """Choose the 'closest' WAP for this row based on the phase."""
    if ts < COOLDOWN_START:
        # Transit: mixed, outer WAPs weighted higher, cold cluster possible.
        pool = TRANSIT_WAPS * 2 + COLD_ROOM_WAPS
        return random.choice(pool)
    # Inside the cold room: only the cold-room cluster hears it.
    return random.choice(COLD_ROOM_WAPS)


def make_rssi_values(ts):
    """Generate the 5 per-minute RSSI hearings for a row.

    Returns a list of 5 items; each is an int (dBm), 0, or '' (blank),
    mirroring the sparse pattern in the real export. RSSI1 is always present.
    """
    inside = ts >= COOLDOWN_START
    if inside:
        strong_lo, strong_hi = -80, -56   # closest cold-room WAP -> strong
        other_lo, other_hi = -95, -60
        p_blank, p_zero = 0.18, 0.08
    else:
        strong_lo, strong_hi = -90, -66   # transit -> generally weaker
        other_lo, other_hi = -96, -70
        p_blank, p_zero = 0.24, 0.10

    values = []
    for i in range(5):
        if i == 0:
            values.append(random.randint(strong_lo, strong_hi))
            continue
        r = random.random()
        if r < p_blank:
            values.append("")
        elif r < p_blank + p_zero:
            values.append(0)
        else:
            values.append(random.randint(other_lo, other_hi))
    return values


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    random.seed(SEED)

    rows = []
    ts = START_TIME
    while ts <= END_TIME:
        name, lat, lon = pick_wap(ts)
        temp = temperature_at(ts)
        rssi = make_rssi_values(ts)
        rows.append([
            ts.strftime(TIME_FMT),
            name,
            IDNODE_SERIAL,
            temp,
            lat,
            lon,
            *rssi,
        ])
        ts += timedelta(minutes=random.choices(INTERVAL_CHOICES, INTERVAL_WEIGHTS)[0])

    header = [
        "EVENTTIME", "HARDWARENAME", "IDNODESERIALNUMBER", "IDNODECHIPTEMPARATURE",
        "LATITUDE", "LONGITUDE", "RSSI1", "RSSI2", "RSSI3", "RSSI4", "RSSI5",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Ground-truth summary (for later logic validation - NOT written to the CSV).
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    print("Ground truth (embedded scenario):")
    print(f"  Ambient/transit : {START_TIME:%d-%m-%Y %H:%M} -> {COOLDOWN_START:%H:%M}")
    print(f"  ENTRY (cooldown): {COOLDOWN_START:%d-%m-%Y %H:%M} -> cold by {COLD_REACHED:%H:%M}")
    print(f"  Cold-stable stay: {COLD_REACHED:%d-%m-%Y %H:%M} -> {END_TIME:%d-%m-%Y %H:%M}")
    print("  EXIT            : none (trace ends while still cold)")


if __name__ == "__main__":
    main()
