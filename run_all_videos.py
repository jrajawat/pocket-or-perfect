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

BALL_RADIUS = 80
RADIUS_TOL  = 20

BASE_OUTPUT = "outputs"

# -----------------------------------------
# Ensure directory structure exists
# -----------------------------------------
for group in VIDEO_GROUPS:
    base = os.path.join(BASE_OUTPUT, group)
    os.makedirs(os.path.join(base, "centers"), exist_ok=True)
    os.makedirs(os.path.join(base, "mask"), exist_ok=True)
    os.makedirs(os.path.join(base, "debug"), exist_ok=True)

# -----------------------------------------
# Process each video
# -----------------------------------------
for group, videos in VIDEO_GROUPS.items():
    for vid in videos:
        video_path = f"{group}/{vid}"
        base_name = os.path.splitext(vid)[0]

        # Output paths
        centers_out = os.path.join(BASE_OUTPUT, group, "centers", f"{base_name}_centers.csv")
        mask_out    = os.path.join(BASE_OUTPUT, group, "mask",    f"{base_name}_mask.mp4")
        debug_out   = os.path.join(BASE_OUTPUT, group, "debug",   f"{base_name}_debug.mp4")

        print("\n======================================================")
        print(f"Processing {video_path}")
        print(f"Saving:")
        print(f"  CSV   -> {centers_out}")
        print(f"  MASK  -> {mask_out}")
        print(f"  DEBUG -> {debug_out}")
        print("======================================================")

        cmd = [
            "python", "detect_spikeball.py",
            video_path,
            "--ball-radius-px", str(BALL_RADIUS),
            "--radius-tol-px", str(RADIUS_TOL),
            "--output", centers_out,
            "--save-mask-video", mask_out,
            "--save-debug-video", debug_out,
        ]

        subprocess.run(cmd)
