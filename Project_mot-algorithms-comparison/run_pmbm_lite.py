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
    tracker_output_to_mot_like,
    evaluate_simple,
)


@dataclass
class BernoulliTrack:
    existence_prob: float
    state: GaussianState
    track_id: int
    miss_count: int = 0


@dataclass
class PoissonComponent:
    weight: float
    state: GaussianState


class PMBMLite:
    """
    Practical PMBM-inspired skeleton.
    Not full academic PMBM, but useful for experiments.
    """

    def __init__(self,
                 kf: KalmanFilterCV,
                 p_s: float = 0.99,
                 p_d: float = 0.9,
                 clutter_intensity: float = 1e-4,
                 existence_confirm: float = 0.7,
                 existence_delete: float = 0.05,
                 max_misses: int = 10,
                 init_cov: Optional[np.ndarray] = None):
        self.kf = kf
        self.p_s = p_s
        self.p_d = p_d
        self.clutter_intensity = clutter_intensity
        self.existence_confirm = existence_confirm
        self.existence_delete = existence_delete
        self.max_misses = max_misses
        self.init_cov = init_cov if init_cov is not None else np.diag([25, 25, 25, 25])

        self.poisson: List[PoissonComponent] = []
        self.bernoullis: List[BernoulliTrack] = []
        self.next_track_id = 1

    def predict(self):
        new_poisson = []
        for pc in self.poisson:
            pred = self.kf.predict(pc.state)
            new_poisson.append(
                PoissonComponent(weight=self.p_s * pc.weight, state=pred)
            )
        self.poisson = new_poisson

        for b in self.bernoullis:
            b.state = self.kf.predict(b.state)
            b.existence_prob *= self.p_s

    def update(self, measurements: np.ndarray):
        used = set()

        # update existing Bernoulli tracks
        for b in self.bernoullis:
            best_j = -1
            best_lh = 0.0

            for j, z in enumerate(measurements):
                d2 = self.kf.mahalanobis2(b.state, z)
                if d2 <= 9.21:
                    lh = self.kf.likelihood(b.state, z)
                    if lh > best_lh:
                        best_lh = lh
                        best_j = j

            if best_j >= 0:
                z = measurements[best_j]
                b.state = self.kf.update(b.state, z)
                b.existence_prob = min(0.99, b.existence_prob + 0.2)
                b.miss_count = 0
                used.add(best_j)
            else:
                b.existence_prob *= (1.0 - self.p_d)
                b.miss_count += 1

        # spawn/reinforce tracks from unused measurements
        for j, z in enumerate(measurements):
            if j in used:
                continue

            best_score = 0.0
            best_pc = None

            for pc in self.poisson:
                lh = self.kf.likelihood(pc.state, z)
                score = pc.weight * lh
                if score > best_score:
                    best_score = score
                    best_pc = pc

            if best_pc is not None and best_score > 1e-6:
                upd_state = self.kf.update(best_pc.state, z)
                self.bernoullis.append(
                    BernoulliTrack(
                        existence_prob=0.6,
                        state=upd_state,
                        track_id=self.next_track_id
                    )
                )
                self.next_track_id += 1
            else:
                mean = np.array([z[0], z[1], 0.0, 0.0], dtype=float)
                cov = self.init_cov.copy()
                self.bernoullis.append(
                    BernoulliTrack(
                        existence_prob=0.5,
                        state=GaussianState(mean, cov),
                        track_id=self.next_track_id
                    )
                )
                self.next_track_id += 1

        keep = []
        for b in self.bernoullis:
            if b.existence_prob < self.existence_delete:
                continue
            if b.miss_count > self.max_misses:
                continue
            keep.append(b)
        self.bernoullis = keep

    def step(self, measurements: np.ndarray):
        self.predict()
        self.update(measurements)

    def get_states(self):
        out = []
        for b in self.bernoullis:
            if b.existence_prob >= self.existence_confirm:
                out.append({
                    "track_id": b.track_id,
                    "x": float(b.state.mean[0]),
                    "y": float(b.state.mean[1]),
                    "vx": float(b.state.mean[2]),
                    "vy": float(b.state.mean[3]),
                    "existence_prob": float(b.existence_prob),
                })
        return out


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

    tracker = PMBMLite(kf)
    outputs = run_tracker_on_sequence(tracker, detections_per_frame)
    pred = tracker_output_to_mot_like(outputs)
    metrics = evaluate_simple(gt_per_frame, pred)

    print("\n=== PMBM-LITE RESULTS ===")
    print(f"JSON_PATH: {JSON_PATH}")
    print(f"VIDEO_ID: {VIDEO_ID}")
    for k, v in metrics.items():
        print(f"{k}: {v}")