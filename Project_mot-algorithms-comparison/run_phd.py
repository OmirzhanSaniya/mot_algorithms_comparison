import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from common_m3ot import (
    M3OTLoader,
    make_cv_model,
    KalmanFilterCV,
    GaussianState,
    simulate_detector_from_gt,
    sequence_to_measurements_xy,
    run_tracker_on_sequence,
    anonymous_output_to_mot_like,
    evaluate_simple,
)



@dataclass
class GaussianComponent:
    weight: float
    mean: np.ndarray
    cov: np.ndarray


class GMPHDFilter:
    """
    Gaussian Mixture PHD filter.
    Good practical baseline for multi-target tracking.
    """

    def __init__(self,
                 kf: KalmanFilterCV,
                 p_s: float = 0.99,
                 p_d: float = 0.9,
                 clutter_intensity: float = 1e-4,
                 birth_components: Optional[List[GaussianComponent]] = None,
                 prune_threshold: float = 1e-4,
                 merge_threshold: float = 4.0,
                 max_components: int = 100):
        self.kf = kf
        self.p_s = p_s
        self.p_d = p_d
        self.clutter_intensity = clutter_intensity
        self.birth_components = birth_components or []
        self.prune_threshold = prune_threshold
        self.merge_threshold = merge_threshold
        self.max_components = max_components

        self.components: List[GaussianComponent] = []

    def predict(self):
        predicted = []

        for comp in self.components:
            pred_state = self.kf.predict(GaussianState(comp.mean, comp.cov))
            predicted.append(
                GaussianComponent(
                    weight=self.p_s * comp.weight,
                    mean=pred_state.mean,
                    cov=pred_state.cov
                )
            )

        predicted.extend(self.birth_components)
        self.components = predicted

    def update(self, measurements: np.ndarray):
        updated = []

        # missed detections
        for comp in self.components:
            updated.append(
                GaussianComponent(
                    weight=(1.0 - self.p_d) * comp.weight,
                    mean=comp.mean,
                    cov=comp.cov
                )
            )

        # detection updates
        for z in measurements:
            numerators = []
            for comp in self.components:
                state = GaussianState(comp.mean, comp.cov)
                lh = self.kf.likelihood(state, z)
                numerators.append(self.p_d * comp.weight * lh)

            denom = self.clutter_intensity + sum(numerators)

            for comp, num in zip(self.components, numerators):
                if num <= 0:
                    continue
                state = GaussianState(comp.mean, comp.cov)
                upd = self.kf.update(state, z)
                w = num / denom
                updated.append(
                    GaussianComponent(weight=w, mean=upd.mean, cov=upd.cov)
                )

        self.components = updated
        self.prune_and_merge()

    def prune_and_merge(self):
        comps = [c for c in self.components if c.weight > self.prune_threshold]
        if not comps:
            self.components = []
            return

        comps.sort(key=lambda c: c.weight, reverse=True)

        merged = []
        used = [False] * len(comps)

        for i, ci in enumerate(comps):
            if used[i]:
                continue

            cluster = [i]
            used[i] = True

            for j in range(i + 1, len(comps)):
                if used[j]:
                    continue
                cj = comps[j]
                diff = cj.mean - ci.mean
                try:
                    d2 = float(diff.T @ np.linalg.inv(ci.cov) @ diff)
                except np.linalg.LinAlgError:
                    d2 = np.inf
                if d2 < self.merge_threshold:
                    cluster.append(j)
                    used[j] = True

            w_sum = sum(comps[k].weight for k in cluster)
            m = sum(comps[k].weight * comps[k].mean for k in cluster) / w_sum

            P = np.zeros_like(ci.cov)
            for k in cluster:
                dk = comps[k].mean - m
                P += comps[k].weight * (comps[k].cov + np.outer(dk, dk))
            P /= w_sum

            merged.append(GaussianComponent(weight=w_sum, mean=m, cov=P))

        merged.sort(key=lambda c: c.weight, reverse=True)
        self.components = merged[:self.max_components]

    def step(self, measurements: np.ndarray):
        self.predict()
        self.update(measurements)

    def extract_states(self, weight_threshold: float = 0.5):
        states = []
        for c in self.components:
            if c.weight >= weight_threshold:
                states.append({
                    "x": float(c.mean[0]),
                    "y": float(c.mean[1]),
                    "vx": float(c.mean[2]),
                    "vy": float(c.mean[3]),
                    "weight": float(c.weight),
                })
        return states


if __name__ == "__main__":
    JSON_PATH = r"D:\kbtu\6 semester\information theory\M3OT\M3OT\Annotations\1\ir\test_cocoformat.json"
    VIDEO_ID = None
    USE_SIMULATED_DETECTOR = True

    loader = M3OTLoader(JSON_PATH)
    video_ids = loader.get_video_ids()
    if VIDEO_ID is None:
        VIDEO_ID = video_ids[0]

    sequence = loader.get_video_sequence(VIDEO_ID)

    if USE_SIMULATED_DETECTOR:
        detections_per_frame, gt_per_frame = simulate_detector_from_gt(
            sequence,
            meas_noise_std=2.0,
            miss_prob=0.05,
            false_per_frame=1,
            image_size=(640, 512),
            seed=42,
        )
    else:
        detections_per_frame, gt_per_frame = sequence_to_measurements_xy(sequence)

    F, Q, H, R = make_cv_model(dt=1.0, process_var=1.0, meas_var=9.0)
    kf = KalmanFilterCV(F, Q, H, R)

    births = [
        GaussianComponent(
            weight=0.05,
            mean=np.array([320.0, 256.0, 0.0, 0.0]),
            cov=np.diag([20000.0, 20000.0, 25.0, 25.0])
        )
    ]

    tracker = GMPHDFilter(kf, birth_components=births)
    outputs = run_tracker_on_sequence(tracker, detections_per_frame)
    pred = anonymous_output_to_mot_like(outputs)
    metrics = evaluate_simple(gt_per_frame, pred)

    print("\n=== GM-PHD RESULTS ===")
    print(f"JSON_PATH: {JSON_PATH}")
    print(f"VIDEO_ID: {VIDEO_ID}")
    for k, v in metrics.items():
        print(f"{k}: {v}")