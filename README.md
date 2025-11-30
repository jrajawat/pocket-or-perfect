Pocket or Perfect? The Mechanics of a Spikeball Rebound
Spikeball Trajectory + Spin Analysis Pipeline

This project provides an end-to-end pipeline for analyzing Spikeball shots from high-speed video.
It extracts:

Ball centers per frame

Inbound and outbound velocity (magnitude + components + elevation angle)

Spin rate (rad/s)

Mask videos showing the yellow-range detection

Debug videos showing detected circles

Spin-vs-time plots with impact markers

Summary CSVs with velocity + spin per shot

The workflow is fully automated through batch scripts and modular processing tools.

1. Project Structure
bme-spikeball-550/
│
├── detect_spikeball.py          # Detect ball centers in each video
├── analyze_spikeball_velocities.py
├── estimate_spin.py             # Extract stripe orientation + spin
├── batch_detect_centers.py      # Process all videos for centers
├── batch_estimate_spin.py       # Process all videos for spin
│
├── Shallow/                     # Raw videos for Shallow shots
├── Oblique/                     # Raw videos for Oblique shots
├── Pocket/                      # Raw videos for Pocket shots
│
└── outputs/
    ├── Shallow/
    │   ├── centers/             # per-frame center CSVs
    │   ├── mask/                # mask MP4s (yellow segmentation)
    │   ├── debug/               # debug MP4s (circle overlays)
    │   └── spin/                # spin CSVs + PNGs
    │
    ├── Oblique/
    └── Pocket/


2. Libraries Used
Purpose	Library
Video I/O, image processing	OpenCV (cv2)
Math, numeric operations	NumPy, SciPy
Data tables	Pandas
Clustering edges for stripe detection	sklearn (DBSCAN)
Smoothing (median filters)	SciPy ndimage
Plotting spin curves	Matplotlib

You must use a SciPy/NumPy/Sklearn combination that matches versions.
This project currently uses:

numpy 1.26.x

scipy 1.10.x

scikit-learn 1.3.x

3. Step 1 — Detecting Ball Centers
Command (single shot)
python detect_spikeball.py Shallow/shallow1.mp4 \
    --ball-radius-px 80 \
    --radius-tol-px 20 \
    --output outputs/Shallow/centers/shallow1_centers.csv \
    --save-mask-video outputs/Shallow/mask/shallow1_mask.mp4 \
    --save-debug-video outputs/Shallow/debug/shallow1_debug.mp4

How it works

Convert each frame to HSV

Mask pixels within a yellow hue range

Apply contour segmentation

Filter contours by:

area

circularity

radius tolerance

Use continuity tracking (nearest center to previous frame)

Save:

cx_px, cy_px — ball center in pixels

radius_px — estimated pixel radius

Output Files
Mask video (mask/*.mp4)

Shows the binary yellow segmentation used to see if detection is clean.

Debug video (debug/*.mp4)

Shows the ball outline every frame — useful for validating tracking.

4. Step 2 — Converting Pixel Motion to Velocities

Once center CSVs exist, run:

python analyze_spikeball_velocities.py


This script processes all center CSVs to generate velocities and impact frames.

How velocities are computed

Convert pixel centers → meters

Compute a global meters-per-pixel scale using the measured circumference of a Spikeball.

This ensures radius errors do not destroy velocity values.

Fit a linear model to center position vs time for:

Frames before impact → inbound velocity

Frames after impact → outbound velocity

Compute:

vx_in_mps, vy_in_mps

speed_in_mps

elev_in_deg (elevation angle)

same for outbound

Save one row per shot in:

outputs/spikeball_velocity_summary.csv

5. Step 3 — Extracting Stripe Orientation & Spin

To compute spin, the ball must have dark stripes drawn on it (like the golf lab).

Command (single shot)
python estimate_spin.py Shallow/shallow1.mp4 \
    outputs/Shallow/centers/shallow1_centers.csv \
    --fps 240 \
    --out-csv outputs/Shallow/spin/shallow1_spin.csv \
    --out-plot outputs/Shallow/spin/shallow1_spin.png \
    --summary-csv outputs/spikeball_velocity_summary.csv \
    --shot-name shallow1_centers

How spin estimation works

For each frame where the ball is detected:

Crop a patch around the ball

Detect edge points inside the ball using Canny

Cluster the edge points using DBSCAN

Run PCA on the largest consistent cluster

Extract the principal orientation angle

Unwrap orientation modulo π (stripes repeat every 180 degrees)

Compute dθ/dt → spin rate (rad/s)

Smooth both:

angle trace

spin trace

Output Files
Spin CSV

frame_idx, theta_rad, omega_rad_s

Spin Plot PNG

Shows:

Spin vs frame

Red dashed line marking impact

Text label “Impact” above the line

Plot title showing shot name (“Shallow1 Spin vs Frame”)

6. Step 4 — Batch Processing

You can run center detection for all 15 videos with:

python batch_estimate_speed.py


This script:

Iterates through all Shallow / Oblique / Pocket folders

Generates centers CSVs

Saves mask and debug videos in the right folders

Then generate spin for all shots:

python batch_estimate_spin.py


This script:

Reads all center CSVs

Creates spin CSVs + PNGs

Updates the velocity summary CSV with spin-in and spin-out values

7. What “Spin In” and “Spin Out” Mean

We mimic the golf ball trackman convention:

Spin In: average spin measured on frames before impact

Spin Out: average spin measured on frames after impact

Each shot now contains:

spin_in_rad_s
spin_out_rad_s


8. Troubleshooting
Mask shows many noisy blobs

→ Adjust HSV yellow range.

Debug shows wrong circle radius

→ Adjust --ball-radius-px and --radius-tol-px.

Spin = zero for a shot

→ Stripe wasn’t visible or DBSCAN didn’t find a dominant cluster.

Hyper-spiky spin plot

→ Increase smoothing windows in estimate_spin.py:

angle_smooth_window

spin_smooth_window

9. Summary

This pipeline allows fully automated Spikeball biomechanics analysis:

Ball tracking (mask + debug video validation)

Accurate inbound/outbound velocity measurement

Spin estimation from stripe orientation

Automated batch processing for full datasets

Consistent structured output for analysis and reporting