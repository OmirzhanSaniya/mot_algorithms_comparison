import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from common_m3ot import (
    M3OTLoader,
    make_cv_model,
    KalmanFilterCV,
    GaussianState,
    gate_measurements,
    simulate_detector_from_gt,
    sequence_to_measurements_xy,
    run_tracker_on_sequence,
    tracker_output_to_mot_like,
    evaluate_simple,
)


@dataclass
class Track:
    track_id: int
    state: GaussianState
    existence_prob: float = 1.0
    miss_count: int = 0
    hit_count: int = 0


class JPDATracker:
    """
    Simplified JPDA-style tracker.
    Practical baseline, not full exact hypothesis JPDA.
    """

    def __init__(self,
                 kf: KalmanFilterCV,
                 p_d: float = 0.9,
                 clutter_intensity: float = 1e-4,
                 gate_threshold: float = 5.99,
                 init_cov: Optional[np.ndarray] = None,
                 max_misses: int = 5):
        self.kf = kf
        self.p_d = p_d
        self.clutter_intensity = clutter_intensity
        self.gate_threshold = gate_threshold
        self.max_misses = max_misses
        self.init_cov = init_cov if init_cov is not None else np.diag([25, 25, 25, 25])

        self.tracks: List[Track] = []
        self.next_track_id = 1

    def _init_track(self, z):
        mean = np.array([z[0], z[1], 0.0, 0.0], dtype=float)
        cov = self.init_cov.copy()
        self.tracks.append(
            Track(
                track_id=self.next_track_id,
                state=GaussianState(mean, cov),
                existence_prob=1.0,
                miss_count=0,
                hit_count=1,
            )
        )
        self.next_track_id += 1

    def predict(self):
        for trk in self.tracks:
            trk.state = self.kf.predict(trk.state)

    def update(self, measurements: np.ndarray):
        if len(self.tracks) == 0:
            for z in measurements:
                self._init_track(z)
            return

        M = len(measurements)
        T = len(self.tracks)

        L = np.zeros((T, M), dtype=float)

        for i, trk in enumerate(self.tracks):
            valid_idx = gate_measurements(self.kf, trk.state, measurements, self.gate_threshold)
            for j in valid_idx:
                L[i, j] = self.p_d * self.kf.likelihood(trk.state, measurements[j])

        beta = np.zeros((T, M), dtype=float)
        beta0 = np.zeros(T, dtype=float)

        for i in range(T):
            assoc_terms = L[i].copy()
            miss_term = (1.0 - self.p_d)
            denom = assoc_terms.sum() + miss_term + 1e-12
            beta[i] = assoc_terms / denom
            beta0[i] = miss_term / denom

        used_meas = set()
        new_tracks = []

        for i, trk in enumerate(self.tracks):
            if beta[i].sum() < 1e-9:
                trk.miss_count += 1
                continue

            z_bar = np.zeros(self.kf.nz, dtype=float)
            total_beta = beta[i].sum()

            for j in range(M):
                if beta[i, j] > 0:
                    z_bar += beta[i, j] * measurements[j]

            z_bar /= max(total_beta, 1e-12)
            trk.state = self.kf.update(trk.state, z_bar)

            if total_beta > beta0[i]:
                trk.hit_count += 1
                trk.miss_count = 0
                best_j = int(np.argmax(beta[i]))
                used_meas.add(best_j)
            else:
                trk.miss_count += 1

        for j in range(M):
            if j not in used_meas:
                new_tracks.append(measurements[j])

        self.tracks = [t for t in self.tracks if t.miss_count <= self.max_misses]

        for z in new_tracks:
            self._init_track(z)

    def step(self, measurements: np.ndarray):
        self.predict()
        self.update(measurements)

    def get_states(self):
        return [
            {
                "track_id": t.track_id,
                "x": float(t.state.mean[0]),
                "y": float(t.state.mean[1]),
                "vx": float(t.state.mean[2]),
                "vy": float(t.state.mean[3]),
                "misses": t.miss_count,
                "hits": t.hit_count,
            }
            for t in self.tracks
        ]


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

    tracker = JPDATracker(kf)
    outputs = run_tracker_on_sequence(tracker, detections_per_frame)
    pred = tracker_output_to_mot_like(outputs)
    metrics = evaluate_simple(gt_per_frame, pred)

    print("\n=== JPDA RESULTS ===")
    print(f"JSON_PATH: {JSON_PATH}")
    print(f"VIDEO_ID: {VIDEO_ID}")
    for k, v in metrics.items():
        print(f"{k}: {v}")