import os
import glob
import math

import numpy as np
import pandas as pd
import csv

# Groups and paths
GROUPS = ["Shallow", "Oblique", "Pocket"]
BASE_OUTPUT = "outputs"

# Camera frame rate
FPS = 240.0

# Spikeball geometry
BALL_CIRCUMFERENCE_M = 0.3048          # 12 in
BALL_RADIUS_M = BALL_CIRCUMFERENCE_M / (2.0 * math.pi)

# Global scale from shallow1
GLOBAL_M_PER_PX = 0.0005627249736020139


def load_centers_csv(path):
    """
    Load frame_idx, cx_px, cy_px, radius_px from a centers CSV using pandas.
    Any non numeric or blank entries are converted to NaN.
    """
    df = pd.read_csv(path)

    frame_idx = pd.to_numeric(df["frame_idx"], errors="coerce").to_numpy()
    cx        = pd.to_numeric(df["cx_px"],     errors="coerce").to_numpy()
    cy        = pd.to_numeric(df["cy_px"],     errors="coerce").to_numpy()
    radius    = pd.to_numeric(df["radius_px"], errors="coerce").to_numpy()

    # Drop rows with NaN frame index
    valid = ~np.isnan(frame_idx)
    frame_idx = frame_idx[valid].astype(int)
    cx        = cx[valid]
    cy        = cy[valid]
    radius    = radius[valid]

    return frame_idx, cx, cy, radius


def filter_by_radius_band(frame_idx, cx, cy, radius, lower_ratio=0.5, upper_ratio=1.5):
    """
    Remove frames whose detected radius is way too small/large compared to
    the median radius in that shot.

    Keeps frames with:
        lower_ratio * median_r <= radius <= upper_ratio * median_r

    If there are too few valid radii to estimate a median, returns data unchanged.
    """
    valid_r = radius[~np.isnan(radius)]
    if valid_r.size < 3:
        # not enough info to define a band; do nothing
        return frame_idx, cx, cy, radius

    med_r = np.median(valid_r)
    lower = lower_ratio * med_r
    upper = upper_ratio * med_r

    rad_ok = (radius >= lower) & (radius <= upper)
    if rad_ok.sum() < 3:
        # avoid nuking everything
        return frame_idx, cx, cy, radius

    frame_idx_f = frame_idx[rad_ok]
    cx_f        = cx[rad_ok]
    cy_f        = cy[rad_ok]
    radius_f    = radius[rad_ok]

    return frame_idx_f, cx_f, cy_f, radius_f


def estimate_scale_m_per_px(radius_px):
    """
    Hybrid scaling:
      - If per-shot radius is plausible and consistent, use per-shot scale.
      - Otherwise, fall back to global scale.
    """

    valid = radius_px[~np.isnan(radius_px)]
    if valid.size < 4:
        # too few detections → fallback
        return GLOBAL_M_PER_PX

    med = np.median(valid)
    std = np.std(valid)

    print(med)
    # Check plausible radius range
    if med < 55 or med > 140:
        print("Using global scale here. OUT OF RANGE")
        return GLOBAL_M_PER_PX

    # Check consistency
    if std > 20:
        print("Using global scale here. INCONSISTENT")
        return GLOBAL_M_PER_PX

    # If all good: use per-shot scale
    return BALL_RADIUS_M / med


def find_impact_pos(cy_px):
    """
    Return the index (0..N-1) where impact occurs.

    Heuristic:
      1) Work only on finite cy values.
      2) Compute gradient of cy.
      3) Look for the first sign change in the gradient from positive
         (moving downward in the image) to negative (moving upward).
         That is taken as the impact index.
      4) If no sign change exists, fall back to the index where
         |gradient| is minimal.
    """
    cy_px = np.asarray(cy_px, dtype=float)
    valid = ~np.isnan(cy_px)
    if valid.sum() < 3:
        return None

    cy_valid = cy_px[valid]
    grad = np.gradient(cy_valid)

    full_indices = np.nonzero(valid)[0]

    # primary rule: first positive to negative gradient sign change
    for i in range(1, len(grad)):
        if grad[i - 1] > 0 and grad[i] < 0:
            return int(full_indices[i])

    # fallback: same as before, smallest |gradient|
    local_idx = int(np.argmin(np.abs(grad)))
    return int(full_indices[local_idx])


def fit_velocity_mask(frame_idx, cx_px, cy_px, m_per_px, fps, mask):
    """
    Fit vx, vy in m/s using only entries where mask == True.
    Returns vx, vy, speed, elev_deg or None if fit fails.
    """
    mask = mask & ~np.isnan(cx_px) & ~np.isnan(cy_px)

    # be a bit forgiving: allow as low as 2 points to define a line
    if mask.sum() < 2:
        return None

    t = frame_idx[mask] / fps
    x_m = cx_px[mask] * m_per_px
    y_m = -cy_px[mask] * m_per_px  # minus because image y increases downward

    px = np.polyfit(t, x_m, 1)
    py = np.polyfit(t, y_m, 1)

    vx = px[0]
    vy = py[0]
    speed = math.sqrt(vx * vx + vy * vy)
    elev_deg = math.degrees(math.atan2(vy, vx))

    return vx, vy, speed, elev_deg


def process_one_centers_file(group, centers_path):
    frame_idx, cx, cy, radius = load_centers_csv(centers_path)

    # radius-based cleaning to throw away tiny/huge blobs
    frame_idx, cx, cy, radius = filter_by_radius_band(frame_idx, cx, cy, radius)

    # if filtering nuked almost everything, skip this shot
    if frame_idx.size < 3:
        return None

    # scale (per shot if possible, otherwise global)
    m_per_px = estimate_scale_m_per_px(radius)

    # find impact position (index in arrays)
    impact_pos = find_impact_pos(cy)
    if impact_pos is None:
        return None

    N = frame_idx.size
    indices = np.arange(N)

    # buffer in index space (how many data points to exclude around impact)
    buffer_idx = 1

    in_mask = indices <= max(0, impact_pos - buffer_idx)
    out_mask = indices >= min(N - 1, impact_pos + buffer_idx)

    inbound  = fit_velocity_mask(frame_idx, cx, cy, m_per_px, FPS, in_mask)
    outbound = fit_velocity_mask(frame_idx, cx, cy, m_per_px, FPS, out_mask)

    if inbound is None or outbound is None:
        return None

    vx_in, vy_in, speed_in, elev_in = inbound
    vx_out, vy_out, speed_out, elev_out = outbound

    shot_name = os.path.splitext(os.path.basename(centers_path))[0]

    return {
        "group": group,
        "shot": shot_name,
        "m_per_px": m_per_px,
        "impact_frame": int(frame_idx[impact_pos]),
        "vx_in_mps": vx_in,
        "vy_in_mps": vy_in,
        "speed_in_mps": speed_in,
        "elev_in_deg": elev_in,
        "vx_out_mps": vx_out,
        "vy_out_mps": vy_out,
        "speed_out_mps": speed_out,
        "elev_out_deg": elev_out,
    }


def main():
    results = []

    for group in GROUPS:
        centers_dir = os.path.join(BASE_OUTPUT, group, "centers")
        pattern = os.path.join(centers_dir, "*_centers.csv")
        for path in sorted(glob.glob(pattern)):
            print(f"Processing {path} ...")
            try:
                info = process_one_centers_file(group, path)
                if info is not None:
                    results.append(info)
                else:
                    print(f"  Skipped {path}: insufficient data or failed fit")
            except Exception as e:
                print(f"  Error on {path}: {e}")

    out_path = os.path.join(BASE_OUTPUT, "spikeball_velocity_summary.csv")
    fieldnames = [
        "group",
        "shot",
        "m_per_px",
        "impact_frame",
        "vx_in_mps",
        "vy_in_mps",
        "speed_in_mps",
        "elev_in_deg",
        "vx_out_mps",
        "vy_out_mps",
        "speed_out_mps",
        "elev_out_deg",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nWrote summary to {out_path}")
    print(f"Rows: {len(results)}")


if __name__ == "__main__":
    main()