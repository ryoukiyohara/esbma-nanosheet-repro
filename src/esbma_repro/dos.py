"""Density of states (DoS) of the 5-fold cross-validation error -- code of run_dos_cve.py.

For every non-empty subset (2^16 - 1 = 65,535 states) of the 16 descriptors:
  folds: KFold(n_splits=5, shuffle=True, random_state=43)
  X standardised with train-fold mean/std (ddof=0), y centred with train-fold mean,
  OLS without intercept on the selected columns, held-out RMSE; CVE = mean over the 5 folds.
Bit i of `mask` = feature i (x_{i+1}).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def build_folds(X, y, n_splits=5, random_state=43):
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    folds = []
    for tr, va in KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X):
        mean = X[tr].mean(axis=0); std = X[tr].std(axis=0)
        ym = y[tr].mean()
        folds.append(((X[tr] - mean) / std, (X[va] - mean) / std, y[tr] - ym, y[va] - ym))
    return folds


def cve_for_mask(mask, folds, n_feat=16):
    cols = [i for i in range(n_feat) if (mask >> i) & 1]
    errs = []
    for Xtr, Xva, ytr, yva in folds:
        reg = LinearRegression(fit_intercept=False).fit(Xtr[:, cols], ytr)
        errs.append(np.sqrt(np.mean((yva - reg.predict(Xva[:, cols])) ** 2)))
    return float(np.mean(errs))


def _chunk(masks, folds, n_feat):
    return [cve_for_mask(int(m), folds, n_feat) for m in masks]


def all_states(X, y, names, n_jobs=-1, max_states=None):
    """CVE of all non-empty subsets, sorted ascending (rank 1 = lowest CVE).
    max_states: evaluate only masks 1..max_states (for quick tests)."""
    n_feat = len(names)
    folds = build_folds(X, y)
    n_states = (1 << n_feat) - 1 if max_states is None else max_states
    masks = np.arange(1, n_states + 1, dtype=np.int64)
    if n_jobs == 1:
        cve = np.array(_chunk(masks, folds, n_feat))
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_chunk)(c.tolist(), folds, n_feat) for c in np.array_split(masks, 256))
        cve = np.concatenate([np.asarray(r, float) for r in res])
    df = pd.DataFrame({
        "indicator": ["".join("1" if (int(m) >> i) & 1 else "0" for i in range(n_feat)) for m in masks],
        "mask": masks.astype(int),
        "features": ["|".join(names[i] for i in range(n_feat) if (int(m) >> i) & 1) for m in masks],
        "size": [bin(int(m)).count("1") for m in masks],
        "cve": cve})
    df = df.sort_values("cve", ascending=True).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df
