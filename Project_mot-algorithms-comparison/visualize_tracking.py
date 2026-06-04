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
FRAME_INDEX = 0

IMAGE_ROOT = r"D:\kbtu\6 semester\information theory\M3OT\M3OT\1\ir\test\1-03T\img1"

IMAGE_SIZE = (640, 512)

# zoom (очень важно)
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
    else:
        return np.zeros((512, 640))


def setup_axis(ax, title, bg):
    ax.imshow(bg, cmap="gray")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlim(*ZOOM_X)
    ax.set_ylim(*ZOOM_Y)
    ax.grid(alpha=0.2)


def draw_gt(ax, frame):
    for det in frame["detections"]:
        x, y, w, h = det["bbox_xywh"]

        rect = patches.Rectangle(
            (x, y), w, h,
            fill=False,
            edgecolor="lime",
            linewidth=2.5
        )
        ax.add_patch(rect)


def draw_noisy(ax, detections):
    for det in detections:
        x, y = det
        ax.scatter(x, y, s=120, color="red", edgecolors="black")


def draw_pred(ax, pred, color, label):
    for obj in pred:
        x, y = obj["xy"]
        ax.scatter(x, y, s=120, color=color, edgecolors="black")
        ax.text(x+3, y+3, label, fontsize=8)


def annotate_overlay(ax, frame, noisy, pred):
    # GT center
    gt = frame["detections"][0]
    x, y, w, h = gt["bbox_xywh"]
    gt_center = (x + w/2, y + h/2)

    # nearest pred
    if pred:
        px, py = pred[0]["xy"]
        ax.annotate(
            "correct tracking",
            xy=(px, py),
            xytext=(px+30, py-40),
            color="cyan",
            arrowprops=dict(arrowstyle="->", color="cyan")
        )

    # false detection
    if len(noisy) > 1:
        fx, fy = noisy[1]
        ax.annotate(
            "false detection",
            xy=(fx, fy),
            xytext=(fx+20, fy+30),
            color="yellow",
            arrowprops=dict(arrowstyle="->", color="yellow")
        )

    # noisy
    nx, ny = noisy[0]
    ax.annotate(
        "noisy detection",
        xy=(nx, ny),
        xytext=(nx-80, ny-30),
        color="red",
        arrowprops=dict(arrowstyle="->", color="red")
    )


# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    loader = M3OTLoader(JSON_PATH)
    sequence = loader.get_video_sequence(VIDEO_ID)

    detections, _ = simulate_detector_from_gt(
        sequence,
        meas_noise_std=3,
        miss_prob=0.2,
        false_per_frame=2,
        image_size=IMAGE_SIZE,
        seed=42
    )

    F, Q, H, R = make_cv_model(1.0, 1.0, 9.0)
    kf = KalmanFilterCV(F, Q, H, R)

    # JPDA
    jpda = JPDATracker(kf)
    jpda_pred = tracker_output_to_mot_like(
        run_tracker_on_sequence(jpda, detections)
    )

    # PHD
    births = [
        GaussianComponent(
            0.05,
            np.array([320,256,0,0]),
            np.diag([20000,20000,25,25])
        )
    ]
    phd = GMPHDFilter(kf, births)
    phd_pred = anonymous_output_to_mot_like(
        run_tracker_on_sequence(phd, detections)
    )

    # PMBM
    pmbm = PMBMLite(kf)
    pmbm_pred = tracker_output_to_mot_like(
        run_tracker_on_sequence(pmbm, detections)
    )

    frame = sequence[FRAME_INDEX]
    bg = get_background_image(frame)

    fig, axes = plt.subplots(2,3, figsize=(18,12))

    # GT
    setup_axis(axes[0,0], "GT", bg)
    draw_gt(axes[0,0], frame)

    # noisy
    setup_axis(axes[0,1], "Noisy", bg)
    draw_noisy(axes[0,1], detections[FRAME_INDEX])

    # JPDA
    setup_axis(axes[0,2], "JPDA", bg)
    draw_pred(axes[0,2], jpda_pred[FRAME_INDEX], "orange", "J")

    # PHD
    setup_axis(axes[1,0], "PHD", bg)
    draw_pred(axes[1,0], phd_pred[FRAME_INDEX], "blue", "P")

    # PMBM
    setup_axis(axes[1,1], "PMBM", bg)
    draw_pred(axes[1,1], pmbm_pred[FRAME_INDEX], "cyan", "M")

    # overlay
    setup_axis(axes[1,2], "Overlay", bg)
    draw_gt(axes[1,2], frame)
    draw_noisy(axes[1,2], detections[FRAME_INDEX])
    draw_pred(axes[1,2], pmbm_pred[FRAME_INDEX], "cyan", "")
    annotate_overlay(axes[1,2], frame, detections[FRAME_INDEX], pmbm_pred[FRAME_INDEX])

    #plt.tight_layout()
    plt.subplots_adjust(
    left=0.05,
    right=0.95,
    top=0.92,
    bottom=0.05,
    wspace=0.25,   # расстояние по горизонтали
    hspace=0.3     # расстояние по вертикали
)
    plt.suptitle(
        "Tracking Comparison: GT vs Noisy Detections vs Algorithms",
        fontsize=16,
        weight='bold'
    )
    plt.savefig(f"{OUTPUT_DIR}/final_visual.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()