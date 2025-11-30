# Pocket or Perfect? The Mechanics of a Spikeball Rebound
## Spikeball Trajectory + Spin Analysis Pipeline

This project provides an end to end system for analyzing Spikeball shots from high speed video.  
It automatically measures:

- Ball centers per frame  
- Inbound and outbound velocity (components, magnitude, elevation angle)  
- Spin rate (rad/s)  
- Mask videos showing yellow detection  
- Debug videos showing circle overlays  
- Spin vs time plots with impact markers  
- Summary CSVs combining all velocity and spin data  

---

# 1. Project Structure

```
bme-spikeball-550/
│
├── detect_spikeball.py              # Detect ball centers per frame
├── analyze_spikeball_velocities.py  # Compute inbound/outbound velocities
├── estimate_spin.py                 # Stripe tracking + spin computation
│
├── batch_estimate_speed.py          # Batch: centers, mask, debug
├── batch_estimate_spin.py           # Batch: spin CSVs + plots
│
├── Shallow/                         # Raw videos
├── Oblique/
├── Pocket/
│
└── outputs/
    ├── Shallow/
    │   ├── centers/                 # cx_px, cy_px, radius_px
    │   ├── mask/                    # yellow mask videos
    │   ├── debug/                   # circle overlay videos
    │   └── spin/                    # spin CSVs + spin plots
    │
    ├── Oblique/
    └── Pocket/
```

---

# 2. Libraries Used

| Purpose | Library |
|--------|---------|
| Video and image processing | OpenCV (cv2) |
| Numeric computation | NumPy |
| Scientific functions | SciPy |
| Data tables | Pandas |
| Stripe clustering | scikit-learn (DBSCAN) |
| Smoothing filters | SciPy ndimage |
| Plotting | Matplotlib |

### Recommended versions  
These versions are compatible with one another:

- numpy 1.26.x  
- scipy 1.10.x  
- scikit-learn 1.3.x  

---

# 3. Step 1 — Detecting Ball Centers

### Single-shot example:
```bash
python detect_spikeball.py Shallow/shallow1.mp4 \
    --ball-radius-px 80 \
    --radius-tol-px 20 \
    --output outputs/Shallow/centers/shallow1_centers.csv \
    --save-mask-video outputs/Shallow/mask/shallow1_mask.mp4 \
    --save-debug-video outputs/Shallow/debug/shallow1_debug.mp4
```

### How center detection works
1. Convert frame to HSV  
2. Mask pixels within a calibrated yellow hue range  
3. Extract contours  
4. Filter by area, circularity, and radius tolerance  
5. Track the nearest center between frames  
6. Save per-frame output:
   - `cx_px`
   - `cy_px`
   - `radius_px`

### Output files
- **Mask video** — shows the binary segmentation used for detection  
- **Debug video** — shows detected circle outlines  

---

# 4. Step 2 — Computing Velocities

Run:
```bash
python analyze_spikeball_velocities.py
```

### How velocities are computed
1. Convert pixel motion to meters using a global `m_per_px`
2. Compute that scale using the known circumference of a Spikeball  
3. Fit a linear model to:
   - pre impact frames → inbound velocity  
   - post impact frames → outbound velocity  
4. Compute:
   - vx_in_mps, vy_in_mps  
   - speed_in_mps  
   - elev_in_deg  
   - and the same for outbound

### Output
`outputs/spikeball_velocity_summary.csv`  
(one row per shot)

---

# 5. Step 3 — Estimating Spin From Stripes

The ball must have black stripes drawn on it, similar to the golf lab spin experiment.

### Single-shot example:
```bash
python estimate_spin.py Shallow/shallow1.mp4 \
    outputs/Shallow/centers/shallow1_centers.csv \
    --fps 240 \
    --out-csv outputs/Shallow/spin/shallow1_spin.csv \
    --out-plot outputs/Shallow/spin/shallow1_spin.png \
    --summary-csv outputs/spikeball_velocity_summary.csv \
    --shot-name shallow1_centers
```

### How spin is computed
For each frame:
1. Crop a patch around the ball  
2. Detect edges using Canny  
3. Cluster edges using DBSCAN  
4. Run PCA on the dominant cluster to estimate stripe orientation  
5. Account for the 180° symmetry (orientation modulo π)  
6. Compute spin:
   ```
   spin = dθ/dt
   ```
7. Smooth both the angle trace and spin trace  
8. Identify spin before and after impact

### Outputs
- **Spin CSV**: frame_idx, theta_rad, omega_rad_s  
- **Spin Plot**: includes a red dashed impact marker + label

---

# 6. Batch Processing

### Detect centers for all shots:
```bash
python batch_estimate_speed.py
```

Produces:
```
outputs/<Group>/centers/
outputs/<Group>/mask/
outputs/<Group>/debug/
```

### Compute spin for all shots:
```bash
python batch_estimate_spin.py
```

Produces:
```
outputs/<Group>/spin/*.csv
outputs/<Group>/spin/*.png
```

Also updates:
```
spikeball_velocity_summary.csv
```

with:
- spin_in_rad_s  
- spin_out_rad_s  

---

# 7. Spin In and Spin Out

Matches golf-ball tracking convention:

| Name | Meaning |
|------|---------|
| **Spin In** | average spin across frames before impact |
| **Spin Out** | average spin across frames after impact |

Impact frame comes from the velocity script.

---

# 8. Troubleshooting

### Mask has too many blobs  
Adjust the HSV yellow thresholds.

### Circle detection looks wrong  
Tune:
- `--ball-radius-px`
- `--radius-tol-px`

### Spin = zero  
Stripe was not visible or DBSCAN didn’t find enough edge points.

### Spin plot is noisy  
Increase:
- angle_smooth_window  
- spin_smooth_window  

---

# 9. Summary

This system provides:

- Spikeball tracking  
- Inbound and outbound velocity  
- Stripe based spin measurement using PCA + clustering  
- Automated processing across all shots  
- Final combined velocity + spin dataset