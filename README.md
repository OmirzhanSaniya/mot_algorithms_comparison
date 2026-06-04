# Multi-Object Tracking Algorithms Comparison

Comparison of three multi-object tracking algorithms — JPDA, GM-PHD, and PMBM-lite — evaluated using classical metrics and KL divergence.

## Project Overview

This project was done as part of the Information Theory course at KBTU. The goal was to compare tracking algorithms under realistic conditions using the M3OT aerial imagery dataset.

## Algorithms

- **JPDA** (Joint Probabilistic Data Association) — high recall, many false positives
- **GM-PHD** (Gaussian Mixture Probability Hypothesis Density) — perfect precision, low recall
- **PMBM-lite** (Poisson Multi-Bernoulli Mixture) — best balance between precision and recall

## Results

| Algorithm | Precision | Recall | F1 | KL Divergence |
|-----------|-----------|--------|----|---------------|
| JPDA | 0.127 | 0.998 | 0.226 | 3.23 |
| GM-PHD | 1.000 | 0.451 | 0.622 | 6.21 |
| PMBM-lite | 0.831 | 0.783 | 0.806 | 1.75 |

PMBM-lite achieved the best F1-score and lowest KL divergence.

## Tech Stack

- Python
- Kalman Filter for motion modeling
- M3OT Dataset (UAV aerial imagery)

## Files

- `common_m3ot.py` — data loading and detection simulation
- `kl_eval.py` — KL divergence evaluation
- `multi-frame visualizer.py` — visualization across frames
- `plot_current_results.py` — results plotting
