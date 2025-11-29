import argparse
import os

import cv2
import numpy as np
import pandas as pd


def detect_ball_in_frame_color(
    frame_bgr,
    ballR_px_est,
    radius_tol_px,
    prev_center=None,
    max_jump_px=None,
    min_circularity=0.4,
):
    """
    Detect ball using HSV yellow color mask and contour filtering.
    Returns (cx, cy, r) or NaNs.

    Strategy:
      1) Build a list of "good" yellow blobs based on area and circularity.
      2) Among those, prefer ones whose radius is near ballR_px_est, but
         if none match the radius window, still fall back to best blob.
      3) If we have a previous center, pick the nearest good blob to keep
         the track continuous even if radius is a bit off.
    """

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Yellow spikeball hue range (tune if needed)
    lower_yellow = np.array([15, 40, 40], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.medianBlur(mask, 5)

    if mask.sum() == 0:
        # nothing in the yellow range in this frame
        return np.nan, np.nan, np.nan

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    strict_candidates = []   # pass radius window + circularity
    loose_candidates = []    # pass area + circularity only

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:        # slightly lower than 50 to allow small partial views
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        r_est = np.sqrt(area / np.pi)

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        loose_candidates.append((cx, cy, r_est, circularity))

        # radius based filter is now advisory, not fatal
        if ballR_px_est - radius_tol_px <= r_est <= ballR_px_est + radius_tol_px:
            strict_candidates.append((cx, cy, r_est, circularity))

    if not loose_candidates:
        return np.nan, np.nan, np.nan

    # choose which candidate set to use: strict if available, otherwise loose
    candidates = strict_candidates if strict_candidates else loose_candidates

    # continuity filter if we have a previous center
    if prev_center is not None and max_jump_px is not None:
        cx_prev, cy_prev = prev_center
        best = None
        best_dist = None
        for cx, cy, r_est, circ in candidates:
            d = np.hypot(cx - cx_prev, cy - cy_prev)
            if best is None or d < best_dist:
                best = (cx, cy, r_est)
                best_dist = d

        # if the nearest candidate is way too far, we declare no detection
        if best_dist is not None and best_dist <= max_jump_px:
            return float(best[0]), float(best[1]), float(best[2])
        else:
            return np.nan, np.nan, np.nan

    # otherwise just pick the most circular blob in the chosen set
    best = max(candidates, key=lambda t: t[3])
    return float(best[0]), float(best[1]), float(best[2])


def track_ball_centers(
    video_path,
    ballR_px_est,
    radius_tol_px,
    debug_vis=False,
    debug_stride=10,
    save_mask_video_path=None,
    save_debug_video_path=None,
):
    """
    Read a video and detect the ball center and radius in each frame.

    Optionally:
      - show live debug windows
      - save a mask video
      - save a debug overlay video with the circle drawn
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    centers = []
    radii = []

    frame_idx = 0
    prev_center = None

    mask_writer = None
    debug_writer = None

    # get fps and frame size once
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 240.0  # fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # run detection
        cx, cy, r = detect_ball_in_frame_color(
            frame_bgr,
            ballR_px_est,
            radius_tol_px,
            prev_center=prev_center,
            max_jump_px=ballR_px_est * 5,
        )

        centers.append([cx, cy])
        radii.append(r)

        if not np.isnan(cx) and not np.isnan(cy):
            prev_center = (cx, cy)

        # build yellow mask once per frame
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 40, 40], dtype=np.uint8)
        upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # prepare debug overlay frame
        debug_frame = frame_bgr.copy()
        if not np.isnan(cx):
            cv2.circle(
                debug_frame,
                (int(cx), int(cy)),
                int(r),
                (0, 255, 0),
                2,
            )
        cv2.putText(
            debug_frame,
            f"frame {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # save mask video if requested
        if save_mask_video_path is not None:
            if mask_writer is None:
                mask_writer = cv2.VideoWriter(
                    save_mask_video_path, fourcc, fps, frame_size
                )
            # convert mask to BGR so it matches writer expectations
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_writer.write(mask_bgr)

        # save debug overlay video if requested
        if save_debug_video_path is not None:
            if debug_writer is None:
                debug_writer = cv2.VideoWriter(
                    save_debug_video_path, fourcc, fps, frame_size
                )
            debug_writer.write(debug_frame)

        # on screen debug every N frames
        if debug_vis and (frame_idx % debug_stride == 0):
            cv2.imshow("ball detection debug", debug_frame)
            cv2.imshow("yellow mask", mask)
            key = cv2.waitKey(1)
            if key == 27:
                debug_vis = False

        frame_idx += 1

    cap.release()
    if mask_writer is not None:
        mask_writer.release()
    if debug_writer is not None:
        debug_writer.release()
    cv2.destroyAllWindows()

    centers = np.array(centers, dtype=float)
    radii = np.array(radii, dtype=float)
    return centers, radii


def save_centers_to_csv(video_path, centers, radii, output_path=None):
    """
    Save per-frame centers and radii to CSV.

    Columns: frame_idx, cx_px, cy_px, radius_px
    """
    n_frames = centers.shape[0]
    frame_indices = np.arange(n_frames, dtype=int)

    df = pd.DataFrame(
        {
            "frame_idx": frame_indices,
            "cx_px": centers[:, 0],
            "cy_px": centers[:, 1],
            "radius_px": radii,
        }
    )

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base}_centers.csv"

    df.to_csv(output_path, index=False)
    print(f"Saved centers and radii to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect spikeball center and radius in a single video."
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--ball-radius-px",
        type=float,
        default=80.0,
        help="Estimated spikeball radius in pixels (default: 80)",
    )
    parser.add_argument(
        "--radius-tol-px",
        type=float,
        default=20.0,
        help="Tolerance on radius in pixels (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <video_basename>_centers.csv)",
    )
    parser.add_argument(
        "--debug-vis",
        action="store_true",
        help="Show debug windows with detected circle overlay",
    )
    parser.add_argument(
        "--debug-stride",
        type=int,
        default=10,
        help="Show one debug frame every N frames (default: 10)",
    )
    parser.add_argument(
        "--save-mask-video",
        type=str,
        default=None,
        help="If set, save the yellow mask video to this path (e.g. mask.mp4)",
    )
    parser.add_argument(
        "--save-debug-video",
        type=str,
        default=None,
        help="If set, save the debug overlay video to this path (e.g. debug.mp4)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"Processing video: {video_path}")
    print(
        f"Using ball radius estimate {args.ball_radius_px} px "
        f"with tolerance {args.radius_tol_px} px"
    )

    centers, radii = track_ball_centers(
        video_path=video_path,
        ballR_px_est=args.ball_radius_px,
        radius_tol_px=args.radius_tol_px,
        debug_vis=args.debug_vis,
        debug_stride=args.debug_stride,
        save_mask_video_path=args.save_mask_video,
        save_debug_video_path=args.save_debug_video,
    )

    save_centers_to_csv(
        video_path=video_path,
        centers=centers,
        radii=radii,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
