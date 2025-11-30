#!/usr/bin/env python
import argparse
import os
import math

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt


# ---------- Helpers for reading centers ----------

def load_centers_csv(path):
    """
    Load cx, cy, radius from a centers CSV.
    Returns three 1D numpy arrays (cx, cy, r) with NaNs for missing values.
    """
    df = pd.read_csv(path)
    cx = pd.to_numeric(df["cx_px"], errors="coerce").to_numpy()
    cy = pd.to_numeric(df["cy_px"], errors="coerce").to_numpy()
    r = pd.to_numeric(df["radius_px"], errors="coerce").to_numpy()
    return cx, cy, r


# ---------- Geometry helpers ----------

def angle_diff_mod_pi(a2, a1):
    """
    Difference between two orientation angles defined modulo pi.
    Both a1 and a2 should already be mapped to [-pi/2, pi/2].
    Result is also in [-pi/2, pi/2].
    """
    d = a2 - a1
    while d > math.pi / 2:
        d -= math.pi
    while d < -math.pi / 2:
        d += math.pi
    return d


def pca_orientation(points):
    """
    Given Nx2 array of points, return principal direction angle in radians.
    Angle is in [-pi/2, pi/2].
    """
    if points.shape[0] < 2:
        return None

    mean = points.mean(axis=0)
    X = points - mean

    # PCA via SVD
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    direction = vh[0]
    angle = math.atan2(direction[1], direction[0])

    if angle > math.pi / 2:
        angle -= math.pi
    elif angle < -math.pi / 2:
        angle += math.pi
    return angle


# ---------- Stripe orientation per frame ----------

def estimate_stripe_angle_for_frame(gray, cx, cy, r):
    """
    Estimate the orientation of the dark stripe on the ball in one frame.

    Returns:
        angle (radians in [-pi/2, pi/2]) or None if no good stripe found.
    """
    h, w = gray.shape

    R = int(max(r * 1.4, 10))
    cx_i = int(round(cx))
    cy_i = int(round(cy))

    x0 = max(cx_i - R, 0)
    x1 = min(cx_i + R, w)
    y0 = max(cy_i - R, 0)
    y1 = min(cy_i + R, h)

    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    ph, pw = patch.shape
    yy, xx = np.mgrid[0:ph, 0:pw]
    cx_p = pw / 2.0
    cy_p = ph / 2.0

    rr2 = (xx - cx_p) ** 2 + (yy - cy_p) ** 2
    mask_inside = rr2 <= (0.9 * r) ** 2
    if mask_inside.sum() < 30:
        return None

    blurred = cv2.GaussianBlur(patch, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges[~mask_inside] = 0

    ys, xs = np.nonzero(edges)
    if len(xs) < 20:
        return None

    pts = np.column_stack((xs, ys))

    eps = max(r * 0.15, 3.0)
    db = DBSCAN(eps=eps, min_samples=10)
    labels = db.fit_predict(pts)

    unique_labels = [lab for lab in np.unique(labels) if lab != -1]
    if not unique_labels:
        return None

    best_angle = None
    best_score = -np.inf

    for lab in unique_labels:
        cl_pts = pts[labels == lab]
        if cl_pts.shape[0] < 10:
            continue

        angle = pca_orientation(cl_pts)
        if angle is None:
            continue

        mean = cl_pts.mean(axis=0)
        X = cl_pts - mean
        direction = np.array([math.cos(angle), math.sin(angle)])
        proj = X @ direction
        spread = proj.max() - proj.min()

        if spread > best_score:
            best_score = spread
            best_angle = angle

    return best_angle


# ---------- Spin estimation over time ----------

def estimate_spin_from_stripes(
    video_path,
    centers_csv_path,
    fps=240.0,
    angle_smooth_window=5,
    spin_smooth_window=5,
    max_angle_jump=0.20,
):
    """
    Compute spin time series for a single shot.

    Returns:
        frame_indices : array of frame indices where spin is defined
        omega         : array of smoothed spin values (rad/s)
        theta_series  : corresponding smoothed orientation angles (rad)
    """
    cx_arr, cy_arr, r_arr = load_centers_csv(centers_csv_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    n_frames_csv = len(cx_arr)
    angles = []
    frame_idx = 0

    # Pass 1: orientation per frame
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= n_frames_csv:
            break

        cx = cx_arr[frame_idx]
        cy = cy_arr[frame_idx]
        r = r_arr[frame_idx]

        if np.isnan(cx) or np.isnan(cy) or np.isnan(r):
            angles.append(np.nan)
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        angle = estimate_stripe_angle_for_frame(gray, cx, cy, r)
        if angle is None:
            angles.append(np.nan)
        else:
            angles.append(angle)

        frame_idx += 1

    cap.release()

    angles = np.array(angles, dtype=float)

    if np.isfinite(angles).sum() < 3:
        return np.array([], dtype=int), np.array([]), np.array([])

    # Smooth angles with median filter
    angles_smooth = angles.copy()
    finite_mask = np.isfinite(angles)
    if finite_mask.sum() >= angle_smooth_window:
        tmp = angles[finite_mask]
        tmp_filt = median_filter(tmp, size=angle_smooth_window)
        angles_smooth[finite_mask] = tmp_filt

    dt = 1.0 / fps
    frame_indices = []
    omega = []
    theta_series = []

    prev_angle = None
    prev_idx = None

    # Pass 2: finite differences to get spin
    for i in range(len(angles_smooth)):
        a = angles_smooth[i]
        if not np.isfinite(a):
            prev_angle = None
            prev_idx = None
            continue

        if prev_angle is None:
            prev_angle = a
            prev_idx = i
            continue

        if i == prev_idx:
            continue

        dtheta = angle_diff_mod_pi(a, prev_angle)

        if abs(dtheta) > max_angle_jump:
            prev_angle = a
            prev_idx = i
            continue

        dt_eff = (i - prev_idx) * dt
        if dt_eff <= 0:
            prev_angle = a
            prev_idx = i
            continue

        w = dtheta / dt_eff

        frame_indices.append(i)
        omega.append(w)
        theta_series.append(a)

        prev_angle = a
        prev_idx = i

    if not frame_indices:
        return np.array([], dtype=int), np.array([]), np.array([])

    frame_indices = np.array(frame_indices, dtype=int)
    omega = np.array(omega, dtype=float)
    theta_series = np.array(theta_series, dtype=float)

    if len(omega) >= spin_smooth_window:
        omega = median_filter(omega, size=spin_smooth_window)

    return frame_indices, omega, theta_series


# ---------- CLI + plotting + summary merge ----------

def main():
    parser = argparse.ArgumentParser(
        description="Estimate spikeball spin from striped ball video."
    )
    parser.add_argument(
        "video",
        help="Path to input video (e.g. Shallow/shallow1.mp4)",
    )
    parser.add_argument(
        "centers_csv",
        help="Path to centers CSV for this shot",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=240.0,
        help="Video frame rate (default: 240)",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional path to save per-frame spin CSV",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default=None,
        help="Optional path to save spin vs frame plot (PNG, etc.)",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Optional existing velocity summary CSV to update with spin_in/out",
    )
    parser.add_argument(
        "--shot-name",
        type=str,
        default=None,
        help="Shot identifier in summary CSV (defaults to centers CSV basename)",
    )

    args = parser.parse_args()

    video_path = args.video
    centers_path = args.centers_csv

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(centers_path):
        raise FileNotFoundError(f"Centers CSV not found: {centers_path}")

    frame_idx, omega, theta = estimate_spin_from_stripes(
        video_path, centers_path, fps=args.fps
    )

    if omega.size == 0:
        print("No reliable spin estimates found.")
        return

    mean_spin = float(np.nanmean(omega))
    mean_abs_spin = float(np.nanmean(np.abs(omega)))
    min_spin = float(np.nanmin(omega))
    max_spin = float(np.nanmax(omega))

    print(f"Got spin estimates on {len(omega)} frames")
    print(f"  mean spin (rad/s): {mean_spin:.3f}")
    print(f"  mean abs spin (rad/s): {mean_abs_spin:.3f}")
    print(f"  min spin: {min_spin:.3f}")
    print(f"  max spin: {max_spin:.3f}")

    # Save per-frame CSV
    if args.out_csv is not None:
        df_out = pd.DataFrame(
            {
                "frame_idx": frame_idx,
                "theta_rad": theta,
                "omega_rad_s": omega,
            }
        )
        df_out.to_csv(args.out_csv, index=False)
        print(f"Saved spin series to {args.out_csv}")

    # Save plot
    if args.out_plot is not None:
        shot_name = args.shot_name or os.path.splitext(os.path.basename(centers_path))[0]
        pretty_title = shot_name.replace("_centers", "").replace("_", " ").title()

        plt.figure(figsize=(7, 4))
        plt.plot(frame_idx, omega, label="Spin (rad/s)")

        # try to read impact_frame from summary CSV if available
        impact_frame_for_plot = None
        if args.summary_csv is not None and os.path.isfile(args.summary_csv):
            df_tmp = pd.read_csv(args.summary_csv)
            if "shot" in df_tmp.columns and (df_tmp["shot"] == shot_name).any():
                impact_frame_for_plot = int(
                    df_tmp.loc[df_tmp["shot"] == shot_name, "impact_frame"].iloc[0]
                )

        # draw vertical line and label if impact known
        if impact_frame_for_plot is not None:
            plt.axvline(
                x=impact_frame_for_plot,
                color="red",
                linestyle="--",
                linewidth=2,
            )
            # text label slightly above plotted data range
            ymax = np.nanmax(omega) if np.isfinite(omega).any() else 0
            plt.text(
                impact_frame_for_plot,
                ymax + 0.05 * abs(ymax),
                "Impact",
                color="red",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.xlabel("Frame index")
        plt.ylabel("Spin (rad/s)")
        plt.title(f"{pretty_title} Spin vs Frame")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(args.out_plot)
        plt.close()
        print(f"Saved spin plot to {args.out_plot}")


    # Merge spin_in and spin_out into velocity summary
    if args.summary_csv is not None:
        if not os.path.isfile(args.summary_csv):
            print(f"Warning: summary CSV {args.summary_csv} not found; skipping merge")
        else:
            shot_name = args.shot_name
            if shot_name is None:
                # centers file is e.g. shallow1_centers.csv
                shot_name = os.path.splitext(os.path.basename(centers_path))[0]

            df_sum = pd.read_csv(args.summary_csv)
            if "shot" not in df_sum.columns:
                print("Warning: summary CSV has no 'shot' column; cannot merge spin.")
            else:
                mask = df_sum["shot"] == shot_name
                if not mask.any():
                    print(
                        f"Warning: shot '{shot_name}' not found in summary CSV; "
                        "spin stats not merged."
                    )
                else:
                    impact_frame = int(df_sum.loc[mask, "impact_frame"].iloc[0])

                    # last spin frame before impact
                    before = frame_idx[frame_idx < impact_frame]
                    spin_in = float("nan")
                    if before.size > 0:
                        last_before = before.max()
                        idx_in = np.where(frame_idx == last_before)[0][0]
                        spin_in = float(omega[idx_in])

                    # first spin frame after impact
                    after = frame_idx[frame_idx > impact_frame]
                    spin_out = float("nan")
                    if after.size > 0:
                        first_after = after.min()
                        idx_out = np.where(frame_idx == first_after)[0][0]
                        spin_out = float(omega[idx_out])

                    print(f"  spin_in (one frame before impact): {spin_in}")
                    print(f"  spin_out (one frame after impact): {spin_out}")

                    df_sum.loc[mask, "spin_in_rad_s"] = spin_in
                    df_sum.loc[mask, "spin_out_rad_s"] = spin_out
                    df_sum.to_csv(args.summary_csv, index=False)
                    print(
                        f"Updated {args.summary_csv} with spin_in and spin_out "
                        f"for shot {shot_name}"
                    )


if __name__ == "__main__":
    main()