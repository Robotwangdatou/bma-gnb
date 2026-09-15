# BMA-GNB: From Heuristic Fusion to Bayesian Model Averaging — Pointwise-Optimal Global–Local Model Mixing

This repository contains the code for the paper **"From Heuristic Fusion to Bayesian Model Averaging: Pointwise-Optimal Global–Local Model Mixing"** (anonymous submission). It implements a theoretically grounded, CPU-only framework for fusing a global and a local generative classifier under class imbalance.

## 1. Overview

The proposed framework combines two Gaussian Naive Bayes (GNB) classifiers with a learnable pointwise mixing weight:

- **Global model G** — standard GNB estimated on the full training set.
- **Local model L** — NLD-IGNB (neighbor-driven local parameters + prior correction), reused here as the local generative model; a vectorized, mathematically equivalent implementation (`FastIntegratedGNB`) is used for speed.
- **BMA gate** — the pointwise mixing weight
  `λ(x) = σ(φᵀ f(x))`, with `f(x) = [disagreement (TV distance), local density, neighborhood class entropy]`,
  trained by weighted logistic regression on the **validation disagreement set** (samples where G and L disagree).
- **Mixture** — `P(c|x) = (1 − λ(x))·P_G(c|x) + λ(x)·P_L(c|x)`.

The mixing weight is derived in closed form from Bayesian model averaging (Theorem 1 / Theorem 4), approximated by the learnable gate under model misspecification (Algorithm 1), accompanied by a PAC-Bayes generalization bound (Theorem 2), and characterized by the error-orthogonality coefficient γ and separation level τ as a post-hoc diagnosable condition (Theorem 3, Corollary 2).

## 2. Repository layout

| File | Role |
|---|---|
| `bma_gnb.py` | **Main model** — `BMA_GateGNB`: G + L + logistic gate + BMA mixture, with class2x weighting, feature ablation, and the degenerate-data fallback of Remark 3. |
| `nld_ignb.py` | NLD-IGNB standard implementation (`IntegratedGNB`), reused as the local model L. |
| `nld_fast.py` | `FastIntegratedGNB` — vectorized drop-in for `IntegratedGNB` (identical math, batched KNN; used in the experiments). |
| `run_experiments.py` | **Main entry** — batch run over the 39 benchmark datasets: GNB vs NLD-IGNB vs BMA-GNB under 5-fold CV. |
| `dataset/` | 39 KEEL imbalanced datasets as CSV (self-contained; see Section 4). |
| `README.md` | This file. |

## 3. Requirements

- Python 3.8+ (developed with 3.10)
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- **No GPU required** — all experiments run on CPU.

No additional packages beyond the standard scientific stack are needed.

## 4. Data

The 39 binary imbalanced datasets are the standard KEEL imbalanced benchmark used in NLD-IGNB (same protocol: stratified 5-fold CV, minority class as positive).

- **Expected format** (per dataset): a CSV with feature columns followed by the label column named `class`; binary labels. E.g. `x1,x2,...,xd,class`.
- **Location**: the repository is self-contained — the 39 CSV files ship under `dataset/`, which `run_experiments.py` reads by default. To use another location, set the `DATA_DIR` environment variable (e.g. `$env:DATA_DIR = "C:\path\to\data"` on Windows, or `DATA_DIR=/path/to/data` on Linux/macOS).
- **Large datasets**: `covtype_4vs_2` and `creditcard` are evaluated on a stratified 60k subsample (`MAX_SAMPLES = 60000`, `APPROX_DATASETS` in `run_experiments.py`) to keep CPU runtime feasible; their results are marked as approximate in the paper. Full-size validation of creditcard is reported separately in the paper (Section 4.8, Table 9).

## 5. Reproducing the main experiment

```bash
# Windows (PowerShell)
$env:GATE_MODE = "class2x"     # final paper protocol: discrete supervision + class2x weighting
$env:OUT_DIR  = "results_v4b"  # any output directory
python run_experiments.py
```

```bash
# Linux / macOS
GATE_MODE=class2x OUT_DIR=results_v4b python run_experiments.py
```

- `GATE_MODE` selects the gate supervision signal:
  - `class2x` — **paper protocol**: discrete target (L-wins indicator) with minority disagreement samples up-weighted by 2.
  - `binary` — discrete target without class re-weighting.
  - `conf_weighted` / `conf_gate2x` / `both` — continuous (confidence-weighted) supervision variants used in the paper's negative-result analysis (continuous supervision collapses on the extreme-imbalance endpoint).
- Output: `per_fold_all.csv` (per-fold metrics for G/N/B) and `summary_per_dataset.csv` (dataset-level aggregates, win/tie/loss counts, average ranks).
- Evaluation protocol: 5-fold stratified CV; per fold, 25% of the training data is held out as a stratified validation split for the gate; `StandardScaler` is fit on the (post-split) training fold only; metrics are binary with the minority class as positive; global seed 42.

## 6. Additional experiments

The paper additionally reports strong-baseline comparisons (SMOTEBoost, CS-RF, CS-LR, Focal-LGBM), PR-AUC, a synthetic γ–τ controlled validation, an MLP gate capacity study, deep-learner scope grounding (LDAM-DRW, BBN), IA-BMA, and several ablations. All of them follow the same evaluation protocol (seed 42, identical folds, same scaler convention), and their full per-dataset results are reported in the paper and the supplementary material. The scripts for these optional experiments are not shipped here to keep the repository minimal; the main experiment above is fully self-contained and reproduces the primary comparison (Tables 1–2).

## 7. Notes

- **Reproducibility**: all random operations use seed 42; no dataset-specific tuning is performed — the gate uses `C = 1.0` (scikit-learn default), `k = 10`, and class2x weighting fixed across all 39 datasets.
- **CPU-only**: experiments were run on a desktop CPU (Intel Core i7-12700H, 16 GB RAM, Windows 11 / Python 3.10).
- This is an **anonymous** submission; please keep author identity out of the repository until the review process is complete.
