import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment


# =========================
# Dataset loader
# =========================

class M3OTLoader:
    """
    Loads M3OT COCO+MOT style annotations.
    Works for both IR and RGB JSON files.
    """

    def __init__(self, json_path: str):
        self.json_path = Path(json_path)

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.images = self.data["images"]
        self.annotations = self.data["annotations"]
        self.categories = self.data.get("categories", [])

        self.images_by_id = {img["id"]: img for img in self.images}

        self.video_to_images = defaultdict(list)
        self.video_to_annotations = defaultdict(list)

        for img in self.images:
            self.video_to_images[img["video_id"]].append(img)

        for ann in self.annotations:
            img = self.images_by_id[ann["image_id"]]
            self.video_to_annotations[img["video_id"]].append(ann)

        for vid in self.video_to_images:
            self.video_to_images[vid].sort(key=lambda x: x["frame_id"])

    def get_video_ids(self) -> List[int]:
        return sorted(self.video_to_images.keys())

    def get_video_sequence(self, video_id: int) -> List[Dict[str, Any]]:
        anns_by_image = defaultdict(list)
        for ann in self.video_to_annotations[video_id]:
            anns_by_image[ann["image_id"]].append(ann)

        sequence = []
        for img in self.video_to_images[video_id]:
            dets = []
            for ann in anns_by_image[img["id"]]:
                dets.append({
                    "bbox_xywh": ann["bbox"],
                    "gt_track_id": ann["mot_instance_id"],
                    "category_id": ann["category_id"],
                    "visibility": ann.get("visibility", 1.0),
                    "area": ann.get("area", None),
                })

            sequence.append({
                "image_id": img["id"],
                "video_id": img["video_id"],
                "frame_id": img["frame_id"],
                "mot_frame_id": img["mot_frame_id"],
                "file_name": img["file_name"],
                "detections": dets,
            })

        return sequence


# =========================
# Geometry helpers
# =========================

def bbox_xywh_to_xy_center(bbox):
    x, y, w, h = bbox
    return np.array([x + w / 2.0, y + h / 2.0], dtype=float)


def sequence_to_measurements_xy(sequence):
    """
    Uses GT boxes directly as detections.
    Good baseline for tracking-only experiments.
    """
    detections_per_frame = []
    gt_per_frame = []

    for frame in sequence:
        Z = []
        GT = []

        for det in frame["detections"]:
            z = bbox_xywh_to_xy_center(det["bbox_xywh"])
            Z.append(z)

            GT.append({
                "track_id": det["gt_track_id"],
                "xy": z,
                "bbox_xywh": det["bbox_xywh"],
            })

        Z = np.asarray(Z, dtype=float) if len(Z) else np.empty((0, 2), dtype=float)

        detections_per_frame.append(Z)
        gt_per_frame.append(GT)

    return detections_per_frame, gt_per_frame


def simulate_detector_from_gt(sequence,
                              meas_noise_std=2.0,
                              miss_prob=0.05,
                              false_per_frame=1,
                              image_size=(640, 512),
                              seed=42):
    """
    Simulates imperfect detector from GT:
    - noise
    - missed detections
    - false positives
    """
    rng = np.random.default_rng(seed)
    W, H = image_size

    detections_per_frame = []
    gt_per_frame = []

    for frame in sequence:
        Z = []
        GT = []

        for det in frame["detections"]:
            gt_xy = bbox_xywh_to_xy_center(det["bbox_xywh"])
            GT.append({
                "track_id": det["gt_track_id"],
                "xy": gt_xy,
                "bbox_xywh": det["bbox_xywh"],
            })

            if rng.random() < miss_prob:
                continue

            noisy = gt_xy + rng.normal(0.0, meas_noise_std, size=2)
            noisy[0] = np.clip(noisy[0], 0, W - 1)
            noisy[1] = np.clip(noisy[1], 0, H - 1)
            Z.append(noisy)

        for _ in range(false_per_frame):
            fp = np.array([
                rng.uniform(0, W - 1),
                rng.uniform(0, H - 1),
            ], dtype=float)
            Z.append(fp)

        Z = np.asarray(Z, dtype=float) if len(Z) else np.empty((0, 2), dtype=float)
        detections_per_frame.append(Z)
        gt_per_frame.append(GT)

    return detections_per_frame, gt_per_frame


# =========================
# Motion model + Kalman
# =========================

def make_cv_model(dt=1.0, process_var=1.0, meas_var=9.0):
    """
    Constant velocity model:
    state = [x, y, vx, vy]
    meas  = [x, y]
    """
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ], dtype=float)

    q = process_var
    Q = q * np.array([
        [dt**4 / 4, 0,         dt**3 / 2, 0],
        [0,         dt**4 / 4, 0,         dt**3 / 2],
        [dt**3 / 2, 0,         dt**2,     0],
        [0,         dt**3 / 2, 0,         dt**2]
    ], dtype=float)

    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=float)

    R = meas_var * np.eye(2)
    return F, Q, H, R


@dataclass
class GaussianState:
    mean: np.ndarray
    cov: np.ndarray


class KalmanFilterCV:
    def __init__(self, F, Q, H, R):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.nx = F.shape[0]
        self.nz = H.shape[0]

    def predict(self, state: GaussianState) -> GaussianState:
        m = self.F @ state.mean
        P = self.F @ state.cov @ self.F.T + self.Q
        return GaussianState(m, P)

    def innovation(self, state: GaussianState, z: np.ndarray):
        z_pred = self.H @ state.mean
        S = self.H @ state.cov @ self.H.T + self.R
        v = z - z_pred
        return v, S, z_pred

    def update(self, state: GaussianState, z: np.ndarray) -> GaussianState:
        v, S, _ = self.innovation(state, z)
        K = state.cov @ self.H.T @ np.linalg.inv(S)
        m = state.mean + K @ v
        I = np.eye(self.nx)
        P = (I - K @ self.H) @ state.cov
        return GaussianState(m, P)

    def predict_measurement(self, state: GaussianState):
        z_pred = self.H @ state.mean
        S = self.H @ state.cov @ self.H.T + self.R
        return z_pred, S

    def mahalanobis2(self, state: GaussianState, z: np.ndarray) -> float:
        v, S, _ = self.innovation(state, z)
        return float(v.T @ np.linalg.inv(S) @ v)

    def likelihood(self, state: GaussianState, z: np.ndarray) -> float:
        z_pred, S = self.predict_measurement(state)
        return float(multivariate_normal.pdf(z, mean=z_pred, cov=S))


def gate_measurements(kf: KalmanFilterCV,
                      state: GaussianState,
                      measurements: np.ndarray,
                      gate_threshold: float = 9.21):
    valid = []
    for i, z in enumerate(measurements):
        d2 = kf.mahalanobis2(state, z)
        if d2 <= gate_threshold:
            valid.append(i)
    return valid


# =========================
# Evaluation
# =========================

def pairwise_dist(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)))
    D = np.zeros((len(a), len(b)), dtype=float)
    for i in range(len(a)):
        for j in range(len(b)):
            D[i, j] = np.linalg.norm(a[i] - b[j])
    return D


def simple_frame_metrics(gt_frame, pred_frame, dist_threshold=30.0):
    gt_xy = [g["xy"] for g in gt_frame]
    pr_xy = [p["xy"] for p in pred_frame]

    if len(gt_xy) == 0 and len(pr_xy) == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "mean_dist": 0.0}

    if len(gt_xy) == 0:
        return {"tp": 0, "fp": len(pr_xy), "fn": 0, "mean_dist": 0.0}

    if len(pr_xy) == 0:
        return {"tp": 0, "fp": 0, "fn": len(gt_xy), "mean_dist": 0.0}

    D = pairwise_dist(gt_xy, pr_xy)
    r, c = linear_sum_assignment(D)

    matched_dists = []
    for i, j in zip(r, c):
        if D[i, j] <= dist_threshold:
            matched_dists.append(D[i, j])

    tp = len(matched_dists)
    fp = len(pr_xy) - tp
    fn = len(gt_xy) - tp
    mean_dist = float(np.mean(matched_dists)) if matched_dists else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "mean_dist": mean_dist}


def evaluate_simple(gt_per_frame, pred_per_frame, dist_threshold=30.0):
    total_tp, total_fp, total_fn = 0, 0, 0
    dists = []

    for gt_f, pr_f in zip(gt_per_frame, pred_per_frame):
        m = simple_frame_metrics(gt_f, pr_f, dist_threshold)
        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]
        if m["mean_dist"] > 0:
            dists.append(m["mean_dist"])

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    mean_dist = float(np.mean(dists)) if dists else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_distance": mean_dist,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def tracker_output_to_mot_like(outputs):
    result = []
    for frame_item in outputs:
        cur = []
        for st in frame_item["states"]:
            if "track_id" not in st:
                continue
            cur.append({
                "track_id": st["track_id"],
                "xy": np.array([st["x"], st["y"]], dtype=float)
            })
        result.append(cur)
    return result


def anonymous_output_to_mot_like(outputs):
    """
    For PHD outputs that do not naturally maintain track IDs.
    """
    result = []
    for frame_item in outputs:
        cur = []
        for i, st in enumerate(frame_item["states"]):
            cur.append({
                "track_id": i + 1,
                "xy": np.array([st["x"], st["y"]], dtype=float)
            })
        result.append(cur)
    return result


def run_tracker_on_sequence(tracker, detections_per_frame):
    outputs = []
    for frame_idx, Z in enumerate(detections_per_frame):
        tracker.step(Z)

        if hasattr(tracker, "get_states"):
            states = tracker.get_states()
        elif hasattr(tracker, "extract_states"):
            states = tracker.extract_states()
        else:
            states = []

        outputs.append({
            "frame": frame_idx,
            "states": states
        })
    return outputs