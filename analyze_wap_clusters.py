"""
WAP layout analysis for the business team.

Takes every WAP (HARDWARENAME, LATITUDE, LONGITUDE) from the device export,
projects them into local metres, and groups WAPs that are physically close
into clusters. This shows the station layout: which WAPs sit together (e.g. the
cold-room cluster) and how far apart the different clusters are.

Method (no heavy deps): two WAPs are 'connected' if they are within
CLUSTER_DISTANCE_M of each other; connected WAPs form one cluster
(single-linkage / connected components via union-find).

Outputs:
  * wap_clusters_map.png       - metric map of WAPs coloured by cluster.
  * wap_distance_matrix.png    - heatmap of pairwise distances (metres).
  * printed cluster membership + distance table.

Requires matplotlib:  pip install matplotlib
Run:                  python analyze_wap_clusters.py
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import detect_cold_room as det  # reuse dist_m + INPUT_CSV

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
INPUT_CSV = det.INPUT_CSV
CLUSTER_DISTANCE_M = 40.0    # WAPs within this distance are grouped together


# --------------------------------------------------------------------------- #
# LOAD UNIQUE WAPS
# --------------------------------------------------------------------------- #
def load_waps(path):
    """Return {name: (lat, lon)} for every distinct WAP in the export."""
    waps = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["HARDWARENAME"].strip()
            if name not in waps:
                waps[name] = (float(row["LATITUDE"]), float(row["LONGITUDE"]))
    return waps


def to_local_metres(waps):
    """Project lat/lon to local (east_m, north_m) from the SW-most reference."""
    ref_lat = min(lat for lat, _ in waps.values())
    ref_lon = min(lon for _, lon in waps.values())
    xy = {}
    for name, (lat, lon) in waps.items():
        east = det.dist_m(ref_lat, ref_lon, ref_lat, lon) * (1 if lon >= ref_lon else -1)
        north = det.dist_m(ref_lat, ref_lon, lat, ref_lon) * (1 if lat >= ref_lat else -1)
        xy[name] = (east, north)
    return xy


# --------------------------------------------------------------------------- #
# CLUSTERING (union-find / connected components)
# --------------------------------------------------------------------------- #
def cluster_waps(waps, threshold):
    names = list(waps)
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d = det.dist_m(*waps[a], *waps[b])
            if d <= threshold:
                union(a, b)

    clusters = {}
    for n in names:
        clusters.setdefault(find(n), []).append(n)
    # return as a sorted list of member-lists (largest first)
    return sorted(clusters.values(), key=len, reverse=True)


def distance_matrix(waps):
    names = list(waps)
    mat = [[det.dist_m(*waps[a], *waps[b]) for b in names] for a in names]
    return names, mat


# --------------------------------------------------------------------------- #
# REPORTING
# --------------------------------------------------------------------------- #
def print_report(waps, clusters, names, mat):
    print("=" * 78)
    print(f"WAP LAYOUT ANALYSIS  ({len(waps)} WAPs, cluster threshold "
          f"{CLUSTER_DISTANCE_M:.0f} m)")
    print("=" * 78)
    for i, members in enumerate(clusters, 1):
        clat = sum(waps[m][0] for m in members) / len(members)
        clon = sum(waps[m][1] for m in members) / len(members)
        print(f"Cluster {i} ({len(members)} WAP(s)) centroid ~({clat:.6f}, {clon:.6f}):")
        for m in sorted(members):
            print(f"    - {m}  ({waps[m][0]:.6f}, {waps[m][1]:.6f})")
    print("-" * 78)

    # inter-cluster nearest distances
    if len(clusters) > 1:
        print("Nearest distance between clusters:")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = min(det.dist_m(*waps[a], *waps[b])
                        for a in clusters[i] for b in clusters[j])
                print(f"    Cluster {i+1} <-> Cluster {j+1}: {d:6.1f} m")
        print("-" * 78)

    print("Pairwise distance matrix (metres):")
    header = "".join(f"{n.split('-')[1]:>8}" for n in names)  # short labels
    print(f"{'':>10}{header}")
    for i, a in enumerate(names):
        row = "".join(f"{mat[i][j]:8.0f}" for j in range(len(names)))
        print(f"{a.split('-')[1]:>10}{row}")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# PLOTS
# --------------------------------------------------------------------------- #
def plot_map(waps, clusters, path="wap_clusters_map.png"):
    xy = to_local_metres(waps)
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10, 8))
    for ci, members in enumerate(clusters):
        color = cmap(ci % 10)
        xs = [xy[m][0] for m in members]
        ys = [xy[m][1] for m in members]
        ax.scatter(xs, ys, s=160, color=color, edgecolor="black",
                   zorder=3, label=f"Cluster {ci+1} ({len(members)})")
        # connect members to make the cluster obvious
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        for x, y in zip(xs, ys):
            ax.plot([cx, x], [cy, y], color=color, lw=1, alpha=0.5, zorder=1)
        for m in members:
            ax.annotate(m, xy[m], xytext=(6, 6), textcoords="offset points",
                        fontsize=9, fontweight="bold")

    ax.set_xlabel("East (metres)", fontsize=11)
    ax.set_ylabel("North (metres)", fontsize=11)
    ax.set_title(f"Station WAP Layout - Spatial Clusters "
                 f"(threshold {CLUSTER_DISTANCE_M:.0f} m)",
                 fontsize=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_distance_matrix(waps, clusters, path="wap_distance_matrix.png"):
    """Lower-triangle heatmap, WAPs ordered by cluster, green=near / red=far.

    Grouping by cluster makes the cold-room block appear as a green corner;
    showing only the lower triangle removes redundant / self distances.
    """
    order = [m for cl in clusters for m in sorted(cl)]
    n = len(order)
    d = np.array([[det.dist_m(*waps[order[i]], *waps[order[j]]) for j in range(n)]
                  for i in range(n)])

    # keep only the lower triangle (i > j); mask the rest
    masked = np.ma.masked_where(np.triu(np.ones_like(d, dtype=bool)), d)
    cmap = plt.get_cmap("RdYlGn_r").copy()   # low dist -> green, high -> red
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(masked, cmap=cmap)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(order, fontsize=9)

    for i in range(n):
        for j in range(i):  # lower triangle only
            ax.text(j, i, f"{d[i][j]:.0f} m", ha="center", va="center",
                    color="black", fontsize=10, fontweight="bold")

    # thin white grid lines between cells for a clean, table-like look
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    ax.set_title("Pairwise WAP Distance (metres) - grouped by cluster\n"
                 "green = close together   |   red = far apart",
                 fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="distance (m)")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    waps = load_waps(INPUT_CSV)
    clusters = cluster_waps(waps, CLUSTER_DISTANCE_M)
    names, mat = distance_matrix(waps)
    print_report(waps, clusters, names, mat)
    plot_map(waps, clusters)
    plot_distance_matrix(waps, clusters)


if __name__ == "__main__":
    main()
