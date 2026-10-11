"""Sample-size dependence of the ES-BMA feature posterior (Fig 8, Fig 9).

Code of notebook/experiment3/esbma_nanosheet.py (yield; namiuchi) and of
katsuta/notebook/experiment3/{size,size_distribution}/esbma_nanosheet_*.py (identical apart from
dataset/target). For a data size n and trial i = 0..29:
    random.seed(i); np.random.seed(i)
    idx = random.choices(range(N_total), k=n)          # resampling WITH replacement
    X, y = X_all[idx], y_all[idx]; exhbma standardisation;
    ES-BMA -> feature posteriors. The original runs used the grid inverse(logspace(-2.5, 2.5, 20)); the unified
    recomputation uses the Fig 4 grid inverse(logspace(-2.5, 2.5, 30)) (default here) and the fast evaluator
    (fast_evidence.py, identical evidences to exhbma).
Fig 9 plots, for each target, the mean over trials 0..29 of the posterior of two descriptors.
"""
import random

import numpy as np

from .fast_evidence import fit_fast

GRID = (-2.5, 2.5, 30)


def trial_indices(n_total, n_sample, trial):
    random.seed(trial)
    np.random.seed(trial)
    return random.choices(list(range(n_total)), k=n_sample)


def run_trial(X, y, n_sample, trial, grid=GRID, use_exhbma=False, n_jobs=-1):
    idx = trial_indices(len(X), n_sample, trial)
    if use_exhbma:
        from .esbma import fit_exhaustive
        r = fit_exhaustive(X[idx], y[idx], *grid, n_jobs=n_jobs)
    else:
        r = fit_fast(X[idx], y[idx], *grid)
    return idx, r["feature_posteriors"]
