import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from common_m3ot import (
    M3OTLoader,
    make_cv_model,
    KalmanFilterCV,
    simulate_detector_from_gt,
    run_tracker_on_sequence,
    tracker_output_to_mot_like,
    anonymous_output_to_mot_like,
)
from run_jpda import JPDATracker
from run_phd import GMPHDFilter, GaussianComponent
from run_pmbm_lite import PMBMLite


# =========================
# SETTINGS
# =========================
JSON_PATH = r"D:\kbtu\6 semester\information theory\M3OT\M3OT\Annotations\1\ir\test_cocoformat.json"
VIDEO_ID = 9

ALGO = "PHD"   # "JPDA", "PHD", "PMBM"

START_FRAME = 0
NUM_FRAMES = 5

IMAGE_ROOT = r"D:\kbtu\6 semester\information theory\M3OT\M3OT\1\ir\test\1-03T\img1"
IMAGE_SIZE = (640, 512)

ZOOM_X = (420, 630)
ZOOM_Y = (512, 350)

OUTPUT_DIR = "visual_outputs"


# =========================
# HELPERS
# =========================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_background_image(frame):
    filename = Path(frame["file_name"]).name
    full_path = Path(IMAGE_ROOT) / filename
    if full_path.exists():
        return mpimg.imread(full_path)
    return np.zeros((512, 640))


def setup_axis(ax, title, bg):
    ax.imshow(bg, cmap="gray")
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlim(*ZOOM_X)
    ax.set_ylim(*ZOOM_Y)
    ax.grid(alpha=0.15)


def draw_gt(ax, frame):
    centers = []
    for det in frame["detections"]:
        x, y, w, h = det["bbox_xywh"]
        rect = patches.Rectangle(
            (x, y), w, h,
            fill=False,
            edgecolor="lime",
            linewidth=2.2
        )
        ax.add_patch(rect)
        centers.append((x + w / 2, y + h / 2))
    return centers


def draw_noisy(ax, detections):
    pts = []
    for det in detections:
        x, y = det
        ax.scatter(
            x, y,
            s=90,
            color="red",
            edgecolors="black",
            linewidths=1.2,
            zorder=5
        )
        pts.append((float(x), float(y)))
    return pts


def algo_style(algo_name):
    if algo_name == "JPDA":
        return "orange", "JPDA"
    if algo_name == "PHD":
        return "dodgerblue", "PHD"
    return "cyan", "PMBM"


def draw_algo(ax, pred, algo_name):
    color, short_name = algo_style(algo_name)
    pts = []
    for obj in pred:
        x, y = obj["xy"]
        ax.scatter(
            x, y,
            s=100,
            color=color,
            edgecolors="black",
            linewidths=1.2,
            zorder=6
        )
        pts.append((float(x), float(y)))
    return pts


def nearest_point(target_xy, points):
    if not points:
        return None
    tx, ty = target_xy
    best = None
    best_d = float("inf")
    for px, py in points:
        d = (px - tx) ** 2 + (py - ty) ** 2
        if d < best_d:
            best_d = d
            best = (px, py)
    return best, best_d


def annotate_overlay(ax, gt_centers, noisy_pts, pred_pts, algo_name):
    if not gt_centers:
        return

    _, algo_label = algo_style(algo_name)
    gt0 = gt_centers[0]

    nearest_noisy = nearest_point(gt0, noisy_pts)
    if nearest_noisy is not None:
        (nx, ny), _ = nearest_noisy
        ax.annotate(
            "noisy detection",
            xy=(nx, ny),
            xytext=(nx - 75, ny - 20),
            color="red",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="red", lw=1.8),
        )

    nearest_pred = nearest_point(gt0, pred_pts)
    if nearest_pred is not None:
        (px, py), d2 = nearest_pred
        if d2 < 900:
            ax.annotate(
                f"{algo_label} tracking",
                xy=(px, py),
                xytext=(px + 20, py - 25),
                color=algo_style(algo_name)[0],
                fontsize=9,
                arrowprops=dict(
                    arrowstyle="->",
                    color=algo_style(algo_name)[0],
                    lw=1.8
                ),
            )
        else:
            ax.annotate(
                "missed / shifted",
                xy=gt0,
                xytext=(gt0[0] + 20, gt0[1] + 20),
                color="magenta",
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="magenta", lw=1.8),
            )
    else:
        ax.annotate(
            "missed object",
            xy=gt0,
            xytext=(gt0[0] + 20, gt0[1] + 20),
            color="magenta",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="magenta", lw=1.8),
        )

    for fp in noisy_pts:
        dmin = min((fp[0] - gx) ** 2 + (fp[1] - gy) ** 2 for gx, gy in gt_centers)
        if dmin > 5000:
            ax.annotate(
                "false detection",
                xy=fp,
                xytext=(fp[0] + 15, fp[1] + 20),
                color="yellow",
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.8),
            )
            break


def build_tracker_and_predictions(algo_name, kf, detections):
    if algo_name == "JPDA":
        tracker = JPDATracker(kf)
        outputs = run_tracker_on_sequence(tracker, detections)
        pred = tracker_output_to_mot_like(outputs)
        return pred

    if algo_name == "PHD":
        births = [
            GaussianComponent(
                weight=0.05,
                mean=np.array([320.0, 256.0, 0.0, 0.0]),
                cov=np.diag([20000.0, 20000.0, 25.0, 25.0])
            )
        ]
        tracker = GMPHDFilter(kf, birth_components=births)
        outputs = run_tracker_on_sequence(tracker, detections)
        pred = anonymous_output_to_mot_like(outputs)
        return pred

    if algo_name == "PMBM":
        tracker = PMBMLite(kf)
        outputs = run_tracker_on_sequence(tracker, detections)
        pred = tracker_output_to_mot_like(outputs)
        return pred

    raise ValueError(f"Unknown algorithm: {algo_name}")


# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    loader = M3OTLoader(JSON_PATH)
    sequence = loader.get_video_sequence(VIDEO_ID)

    detections, _ = simulate_detector_from_gt(
        sequence,
        meas_noise_std=3.0,
        miss_prob=0.2,
        false_per_frame=2,
        image_size=IMAGE_SIZE,
        seed=42
    )

    F, Q, H, R = make_cv_model(1.0, 1.0, 9.0)
    kf = KalmanFilterCV(F, Q, H, R)

    pred = build_tracker_and_predictions(ALGO, kf, detections)

    frame_indices = list(range(START_FRAME, min(START_FRAME + NUM_FRAMES, len(sequence))))
    n = len(frame_indices)

    fig, axes = plt.subplots(n, 4, figsize=(18, 4 * n))
    if n == 1:
        axes = np.array([axes])

    for row, idx in enumerate(frame_indices):
        frame = sequence[idx]
        bg = get_background_image(frame)

        setup_axis(axes[row, 0], f"Frame {idx+1} - GT", bg)
        gt_centers = draw_gt(axes[row, 0], frame)

        setup_axis(axes[row, 1], f"Frame {idx+1} - Noisy", bg)
        noisy_pts = draw_noisy(axes[row, 1], detections[idx])

        setup_axis(axes[row, 2], f"Frame {idx+1} - {ALGO}", bg)
        pred_pts = draw_algo(axes[row, 2], pred[idx], ALGO)

        setup_axis(axes[row, 3], f"Frame {idx+1} - Overlay", bg)
        draw_gt(axes[row, 3], frame)
        noisy_pts = draw_noisy(axes[row, 3], detections[idx])
        pred_pts = draw_algo(axes[row, 3], pred[idx], ALGO)
        annotate_overlay(axes[row, 3], gt_centers, noisy_pts, pred_pts, ALGO)

    plt.suptitle(
        f"{ALGO} Tracking Across Multiple Frames: GT vs Noisy Detections vs {ALGO}",
        fontsize=16,
        weight="bold"
    )

    plt.subplots_adjust(
        left=0.05,
        right=0.97,
        top=0.93,
        bottom=0.04,
        wspace=0.22,
        hspace=0.35
    )

    out_path = Path(OUTPUT_DIR) / f"{ALGO.lower()}_multi_frame_visual.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    summary = []
    for idx in frame_indices:
        summary.append({
            "frame": idx + 1,
            "file_name": sequence[idx]["file_name"],
            "num_gt": len(sequence[idx]["detections"]),
            "num_noisy": len(detections[idx]),
            f"num_{ALGO.lower()}": len(pred[idx]),
        })

    with open(Path(OUTPUT_DIR) / f"{ALGO.lower()}_multi_frame_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()