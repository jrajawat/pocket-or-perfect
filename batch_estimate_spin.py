import os
import subprocess

# -----------------------------------------
# Video groups for each category
# -----------------------------------------
VIDEO_GROUPS = {
    "Shallow": ["shallow1.mp4", "shallow2.mp4", "shallow3.mp4", "shallow4.mp4", "shallow5.mp4"],
    "Oblique": ["oblique1.mp4", "oblique2.mp4", "oblique3.mp4", "oblique4.mp4", "oblique5.mp4"],
    "Pocket":  ["pocket1.mp4",  "pocket2.mp4",  "pocket3.mp4",  "pocket4.mp4",  "pocket5.mp4"],
}

BASE_OUTPUT = "outputs"
FPS = 240.0
SUMMARY_CSV = os.path.join(BASE_OUTPUT, "spikeball_velocity_summary.csv")

# -----------------------------------------
# Ensure directory structure exists
# -----------------------------------------
for group in VIDEO_GROUPS:
    base = os.path.join(BASE_OUTPUT, group)
    os.makedirs(os.path.join(base, "spin_csv"), exist_ok=True)
    os.makedirs(os.path.join(base, "spin_plots"), exist_ok=True)

# -----------------------------------------
# Process each video for spin
# -----------------------------------------
for group, videos in VIDEO_GROUPS.items():
    for vid in videos:
        video_path = os.path.join(group, vid)
        base_name = os.path.splitext(vid)[0]

        centers_csv = os.path.join(
            BASE_OUTPUT, group, "centers", f"{base_name}_centers.csv"
        )
        spin_csv_out = os.path.join(
            BASE_OUTPUT, group, "spin_csv", f"{base_name}_spin.csv"
        )
        spin_plot_out = os.path.join(
            BASE_OUTPUT, group, "spin_plots", f"{base_name}_spin.png"
        )

        print("\n======================================================")
        print(f"Estimating spin for {video_path}")
        print(f"Using centers: {centers_csv}")
        print("Saving:")
        print(f"  spin CSV   -> {spin_csv_out}")
        print(f"  spin plot  -> {spin_plot_out}")
        print("Also updating summary (if present):")
        print(f"  {SUMMARY_CSV}")
        print("======================================================")

        if not os.path.isfile(centers_csv):
            print(f"  Skipping {video_path}: centers file not found.")
            continue

        cmd = [
            "python",
            "estimate_spin.py",
            video_path,
            centers_csv,
            "--fps",
            str(FPS),
            "--out-csv",
            spin_csv_out,
            "--out-plot",
            spin_plot_out,
        ]

        # Only pass summary_csv if it exists so you do not get warnings early on
        if os.path.isfile(SUMMARY_CSV):
            cmd.extend(["--summary-csv", SUMMARY_CSV])
            # shot-name is optional because estimate_spin.py already infers it
            # from centers_csv basename, which matches your "shot" column.

        subprocess.run(cmd)