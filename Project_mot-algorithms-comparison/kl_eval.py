import numpy as np
import json
import csv
from pathlib import Path

from common_m3ot import (
    M3OTLoader,
    make_cv_model,
    KalmanFilterCV,
    simulate_detector_from_gt,
    sequence_to_measurements_xy,
    run_tracker_on_sequence,
    tracker_output_to_mot_like,
    anonymous_output_to_mot_like,
    evaluate_simple,
)

from run_jpda import JPDATracker
from run_phd import GMPHDFilter, GaussianComponent
from run_pmbm_lite import PMBMLite


# =========================================================
# Density / KL utilities
# =========================================================

def gaussian_2d_grid(height_cells, width_cells, cx, cy, sigma):
    """
    Create 2D Gaussian over grid coordinates.
    cx, cy are in grid coordinates, not pixel coordinates.
    """
    ys = np.arange(height_cells, dtype=float)
    xs = np.arange(width_cells, dtype=float)
    X, Y = np.meshgrid(xs, ys)
    g = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2))
    return g


def frame_to_density(frame_objects,
                     image_width,
                     image_height,
                     cell_size=10,
                     sigma_cells=1.5,
                     eps=1e-12):
    """
    Converts a frame with objects into a normalized 2D density map.

    frame_objects format:
    [
        {"xy": np.array([x, y]), ...},
        ...
    ]
    """
    width_cells = int(np.ceil(image_width / cell_size))
    height_cells = int(np.ceil(image_height / cell_size))

    density = np.zeros((height_cells, width_cells), dtype=float)

    for obj in frame_objects:
        x, y = obj["xy"]
        cx = x / cell_size
        cy = y / cell_size
        density += gaussian_2d_grid(height_cells, width_cells, cx, cy, sigma_cells)

    total = density.sum()
    if total <= eps:
        density[:] = 1.0 / density.size
    else:
        density /= total

    return density


def kl_divergence(p, q, eps=1e-12):
    """
    KL(P || Q) = sum P log(P / Q)
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    p /= p.sum()
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))


def symmetric_kl(p, q, eps=1e-12):
    return 0.5 * (kl_divergence(p, q, eps) + kl_divergence(q, p, eps))


def evaluate_kl(gt_per_frame,
                pred_per_frame,
                image_width=640,
                image_height=512,
                cell_size=10,
                sigma_cells=1.5):
    """
    Computes frame-wise KL and symmetric KL, then averages them.
    """
    kl_gt_pred_all = []
    kl_pred_gt_all = []
    skl_all = []

    for gt_frame, pred_frame in zip(gt_per_frame, pred_per_frame):
        p_gt = frame_to_density(
            gt_frame,
            image_width=image_width,
            image_height=image_height,
            cell_size=cell_size,
            sigma_cells=sigma_cells,
        )
        p_pred = frame_to_density(
            pred_frame,
            image_width=image_width,
            image_height=image_height,
            cell_size=cell_size,
            sigma_cells=sigma_cells,
        )

        kl1 = kl_divergence(p_gt, p_pred)
        kl2 = kl_divergence(p_pred, p_gt)
        skl = 0.5 * (kl1 + kl2)

        kl_gt_pred_all.append(kl1)
        kl_pred_gt_all.append(kl2)
        skl_all.append(skl)

    return {
        "mean_KL_GT_to_PRED": float(np.mean(kl_gt_pred_all)),
        "mean_KL_PRED_to_GT": float(np.mean(kl_pred_gt_all)),
        "mean_symmetric_KL": float(np.mean(skl_all)),
    }

def save_predictions(pred_per_frame, name):
    data = []

    for frame_id, frame in enumerate(pred_per_frame):
        for obj in frame:
            x, y = obj["xy"]
            data.append({
                "frame": frame_id + 1,
                "x": float(x),
                "y": float(y)
            })

    with open(f"{name}_predictions.json", "w") as f:
        json.dump(data, f, indent=4)
# =========================================================
# Experiment runner
# =========================================================

def run_all_algorithms(json_path,
                       video_id=None,
                       use_simulated_detector=True,
                       meas_noise_std=2.0,
                       miss_prob=0.05,
                       false_per_frame=1,
                       cell_size=10,
                       sigma_cells=1.5):
    loader = M3OTLoader(json_path)
    video_ids = loader.get_video_ids()

    if video_id is None:
        video_id = video_ids[0]

    sequence = loader.get_video_sequence(video_id)

    image_width = 640
    image_height = 512
    if len(sequence) > 0:
        image_width = 640
        image_height = 512

    if use_simulated_detector:
        detections_per_frame, gt_per_frame = simulate_detector_from_gt(
            sequence,
            meas_noise_std=meas_noise_std,
            miss_prob=miss_prob,
            false_per_frame=false_per_frame,
            image_size=(image_width, image_height),
            seed=42,
        )
    else:
        detections_per_frame, gt_per_frame = sequence_to_measurements_xy(sequence)

    F, Q, H, R = make_cv_model(dt=1.0, process_var=1.0, meas_var=9.0)
    kf = KalmanFilterCV(F, Q, H, R)

    # ---------------- JPDA ----------------
    jpda = JPDATracker(kf)
    jpda_out = run_tracker_on_sequence(jpda, detections_per_frame)
    jpda_pred = tracker_output_to_mot_like(jpda_out)
    save_predictions(jpda_pred, "jpda")
    jpda_basic = evaluate_simple(gt_per_frame, jpda_pred)
    jpda_kl = evaluate_kl(
        gt_per_frame,
        jpda_pred,
        image_width=image_width,
        image_height=image_height,
        cell_size=cell_size,
        sigma_cells=sigma_cells,
    )

    # ---------------- GM-PHD ----------------
    births = [
        GaussianComponent(
            weight=0.05,
            mean=np.array([320.0, 256.0, 0.0, 0.0]),
            cov=np.diag([20000.0, 20000.0, 25.0, 25.0])
        )
    ]
    phd = GMPHDFilter(kf, birth_components=births)
    phd_out = run_tracker_on_sequence(phd, detections_per_frame)
    phd_pred = anonymous_output_to_mot_like(phd_out)
    save_predictions(phd_pred, "phd")
    phd_basic = evaluate_simple(gt_per_frame, phd_pred)
    phd_kl = evaluate_kl(
        gt_per_frame,
        phd_pred,
        image_width=image_width,
        image_height=image_height,
        cell_size=cell_size,
        sigma_cells=sigma_cells,
    )

    # ---------------- PMBM-lite ----------------
    pmbm = PMBMLite(kf)
    pmbm_out = run_tracker_on_sequence(pmbm, detections_per_frame)
    pmbm_pred = tracker_output_to_mot_like(pmbm_out)
    save_predictions(pmbm_pred, "pmbm")
    pmbm_basic = evaluate_simple(gt_per_frame, pmbm_pred)
    pmbm_kl = evaluate_kl(
        gt_per_frame,
        pmbm_pred,
        image_width=image_width,
        image_height=image_height,
        cell_size=cell_size,
        sigma_cells=sigma_cells,
    )

    return {
        "video_id": video_id,
        "JPDA": {**jpda_basic, **jpda_kl},
        "GM-PHD": {**phd_basic, **phd_kl},
        "PMBM-lite": {**pmbm_basic, **pmbm_kl},
    }


def print_results_table(results):
    print("\n================ FINAL COMPARISON ================\n")
    print(f"VIDEO_ID: {results['video_id']}\n")

    headers = [
        "Algorithm",
        "Precision",
        "Recall",
        "F1",
        "KL(GT||Pred)",
        "KL(Pred||GT)",
        "SymKL",
    ]

    row_fmt = "{:<12} {:>10} {:>10} {:>10} {:>14} {:>14} {:>12}"
    print(row_fmt.format(*headers))
    print("-" * 88)

    for algo in ["JPDA", "GM-PHD", "PMBM-lite"]:
        r = results[algo]
        print(row_fmt.format(
            algo,
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['mean_KL_GT_to_PRED']:.4f}",
            f"{r['mean_KL_PRED_to_GT']:.4f}",
            f"{r['mean_symmetric_KL']:.4f}",
        ))

    print("\nLower KL is better.")
    print("Higher Precision / Recall / F1 is better.")

    best_by_symkl = min(
        ["JPDA", "GM-PHD", "PMBM-lite"],
        key=lambda a: results[a]["mean_symmetric_KL"]
    )
    best_by_f1 = max(
        ["JPDA", "GM-PHD", "PMBM-lite"],
        key=lambda a: results[a]["f1"]
    )

    print(f"\nBest by symmetric KL: {best_by_symkl}")
    print(f"Best by F1: {best_by_f1}")

def save_results_json(results, output_path="results.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def save_results_csv(results, output_path="results.csv"):
    rows = []
    for algo in ["JPDA", "GM-PHD", "PMBM-lite"]:
        r = results[algo]
        rows.append({
            "video_id": results["video_id"],
            "algorithm": algo,
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "tp": r["tp"],
            "fp": r["fp"],
            "fn": r["fn"],
            "mean_distance": r["mean_distance"],
            "mean_KL_GT_to_PRED": r["mean_KL_GT_to_PRED"],
            "mean_KL_PRED_to_GT": r["mean_KL_PRED_to_GT"],
            "mean_symmetric_KL": r["mean_symmetric_KL"],
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_human_readable_report(results, output_path="results_report.txt"):
    lines = []
    lines.append("FINAL COMPARISON REPORT")
    lines.append("=" * 50)
    lines.append(f"Video ID: {results['video_id']}")
    lines.append("")

    for algo in ["JPDA", "GM-PHD", "PMBM-lite"]:
        r = results[algo]
        lines.append(f"{algo}")
        lines.append("-" * 30)
        lines.append(f"Precision: {r['precision']:.4f}")
        lines.append(f"Recall: {r['recall']:.4f}")
        lines.append(f"F1-score: {r['f1']:.4f}")
        lines.append(f"True Positives: {r['tp']}")
        lines.append(f"False Positives: {r['fp']}")
        lines.append(f"False Negatives: {r['fn']}")
        lines.append(f"Mean Distance: {r['mean_distance']:.4f}")
        lines.append(f"KL(GT||Pred): {r['mean_KL_GT_to_PRED']:.4f}")
        lines.append(f"KL(Pred||GT): {r['mean_KL_PRED_to_GT']:.4f}")
        lines.append(f"Symmetric KL: {r['mean_symmetric_KL']:.4f}")
        lines.append("")

    best_by_symkl = min(
        ["JPDA", "GM-PHD", "PMBM-lite"],
        key=lambda a: results[a]["mean_symmetric_KL"]
    )
    best_by_f1 = max(
        ["JPDA", "GM-PHD", "PMBM-lite"],
        key=lambda a: results[a]["f1"]
    )

    lines.append("SUMMARY")
    lines.append("-" * 30)
    lines.append(f"Best by symmetric KL: {best_by_symkl}")
    lines.append(f"Best by F1-score: {best_by_f1}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("Lower KL means the predicted spatial distribution is closer to the ground truth.")
    lines.append("Higher F1-score means a better balance between precision and recall.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    JSON_PATH = r"D:\kbtu\6 semester\information theory\M3OT\M3OT\Annotations\1\ir\test_cocoformat.json"
    VIDEO_ID = 9
    USE_SIMULATED_DETECTOR = True

    results = run_all_algorithms(
        json_path=JSON_PATH,
        video_id=VIDEO_ID,
        use_simulated_detector=USE_SIMULATED_DETECTOR,
        meas_noise_std=2.0,
        miss_prob=0.05,
        false_per_frame=1,
        cell_size=10,       # larger = faster, smaller = more detailed
        sigma_cells=1.5,    # Gaussian blur width on the density grid
    )

    print_results_table(results)
    save_results_json(results, "results.json")
    save_results_csv(results, "results.csv")
    save_human_readable_report(results, "results_report.txt")

    print("\nSaved:")
    print("- results.json")
    print("- results.csv")
    print("- results_report.txt")