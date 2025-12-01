#!/usr/bin/env python3
"""
Speed summary plots for Spikeball shots.

Reads:
    outputs/spikeball_velocity_summary.csv

Produces:
    - Figure 1: grouped bar chart of mean inbound/outbound speeds
      saved to outputs/speed_summary_mean.png
    - Figure 2: scatter of inbound vs outbound speeds
      saved to outputs/speed_summary_scatter.png
    - Per-shot speed vs frame plots with red impact line:
        outputs/<Group>/speed_plots/<shot>_speed.png
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_OUTPUT = Path("outputs")
SUMMARY_CSV = BASE_OUTPUT / "spikeball_velocity_summary.csv"
FPS = 240.0


def load_velocity_summary(path: Path) -> pd.DataFrame:
    """Load the summary CSV and drop any rows missing speeds."""
    df = pd.read_csv(path)

    for col in ["speed_in_mps", "speed_out_mps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["speed_in_mps", "speed_out_mps"])
    return df


def plot_group_means(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart with error bars, saved to out_path."""
    groups = sorted(df["group"].unique())

    means_in = []
    means_out = []
    std_in = []
    std_out = []

    for g in groups:
        sub = df[df["group"] == g]
        means_in.append(sub["speed_in_mps"].mean())
        means_out.append(sub["speed_out_mps"].mean())
        std_in.append(sub["speed_in_mps"].std())
        std_out.append(sub["speed_out_mps"].std())

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - width / 2, means_in, width, yerr=std_in, capsize=5, label="Inbound")
    ax.bar(x + width / 2, means_out, width, yerr=std_out, capsize=5, label="Outbound")

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Mean inbound vs outbound speed by condition")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_in_vs_out_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter plot, saved to out_path."""
    fig, ax = plt.subplots(figsize=(6, 6))

    groups = sorted(df["group"].unique())
    cmap = plt.get_cmap("tab10")

    for i, g in enumerate(groups):
        sub = df[df["group"] == g]
        ax.scatter(
            sub["speed_in_mps"],
            sub["speed_out_mps"],
            label=g,
            alpha=0.8,
            s=40,
            color=cmap(i),
        )

    min_val = min(df["speed_in_mps"].min(), df["speed_out_mps"].min())
    max_val = max(df["speed_in_mps"].max(), df["speed_out_mps"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)

    ax.set_xlabel("Inbound speed (m/s)")
    ax.set_ylabel("Outbound speed (m/s)")
    ax.set_title("Outbound vs inbound speed per shot")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def load_centers_csv(path: Path):
    df = pd.read_csv(path)

    frame_idx = pd.to_numeric(df["frame_idx"], errors="coerce").to_numpy()
    cx = pd.to_numeric(df["cx_px"], errors="coerce").to_numpy()
    cy = pd.to_numeric(df["cy_px"], errors="coerce").to_numpy()
    radius = pd.to_numeric(df["radius_px"], errors="coerce").to_numpy()

    valid = ~np.isnan(frame_idx)
    frame_idx = frame_idx[valid].astype(int)
    cx = cx[valid]
    cy = cy[valid]
    radius = radius[valid]

    return frame_idx, cx, cy, radius


def plot_speed_timeseries(
    centers_path: Path,
    impact_frame: int,
    m_per_px: float,
    fps: float,
    title: str,
    out_path: Path,
) -> None:
    frame_idx, cx, cy, _radius = load_centers_csv(centers_path)

    x_m = cx * m_per_px
    y_m = -cy * m_per_px

    speed = np.zeros_like(x_m, dtype=float)
    for i in range(1, len(frame_idx)):
        dt = (frame_idx[i] - frame_idx[i - 1]) / fps
        dx = x_m[i] - x_m[i - 1]
        dy = y_m[i] - y_m[i - 1]
        if dt > 0:
            speed[i] = math.sqrt(dx * dx + dy * dy) / dt
        else:
            speed[i] = np.nan

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(frame_idx, speed, label="Speed (m/s)", color="steelblue")

    ax.axvline(impact_frame, color="red", linestyle="--", linewidth=2, label="Impact")

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Summary CSV not found at {SUMMARY_CSV}")

    df = load_velocity_summary(SUMMARY_CSV)
    print(f"Loaded {len(df)} shots from {SUMMARY_CSV}")

    # Save global summary plots
    plot_group_means(df, BASE_OUTPUT / "speed_summary_mean.png")
    plot_in_vs_out_scatter(df, BASE_OUTPUT / "speed_summary_scatter.png")

    # Per-shot plots
    for _, row in df.iterrows():
        group = row["group"]
        shot_name = row["shot"]
        impact_frame = int(row["impact_frame"])
        m_per_px = float(row["m_per_px"])
        base_name = shot_name.replace("_centers", "")

        centers_path = BASE_OUTPUT / group / "centers" / f"{base_name}_centers.csv"
        if not centers_path.exists():
            print(f"Skipping {base_name}: centers CSV missing at {centers_path}")
            continue

        out_path = BASE_OUTPUT / group / "speed_plots" / f"{base_name}_speed.png"
        title = f"{group} — {base_name}"

        print(f"Saving {title} to {out_path}")
        plot_speed_timeseries(
            centers_path,
            impact_frame,
            m_per_px,
            FPS,
            title,
            out_path,
        )

    print("\nDone generating all speed plots.")


if __name__ == "__main__":
    main()
